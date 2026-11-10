"""Backend parameter handling and retry policy: no invented defaults, no
cross-cell leakage, a fatal error never retried.  Provider-specific shapes
are in test_http.py, HF generation in test_hf_backend.py."""

import pytest

from evalcore.core.backends import (
    DECODE_KEYS,
    Cache,
    Fatal,
    HFBackend,
    Request,
    Response,
    decode_signature,
    merge_params,
    require,
    with_retries,
)


def test_merge_params_request_wins():
    assert merge_params({"max_tokens": 8}, {"max_tokens": 64})["max_tokens"] == 64
    assert merge_params({"max_tokens": 8}, {})["max_tokens"] == 8
    assert merge_params(None, {"temperature": 0.0})["temperature"] == 0.0


def test_require_refuses_to_invent():
    with pytest.raises(Fatal, match="requires 'max_tokens'"):
        require({}, "max_tokens", "anthropic")
    with pytest.raises(Fatal):
        require({"max_tokens": None}, "max_tokens", "anthropic")
    assert require({"max_tokens": 4}, "max_tokens", "anthropic") == 4


def test_a37_fatal_is_called_exactly_once():
    """A37. A Fatal is not worth retrying: with attempts=5 the function must
    be called once (counted), and the row says fatal."""
    calls = []

    def fn(req):
        calls.append(1)
        raise Fatal("bad request")

    r = with_retries(fn, Request("u", [{"role": "user", "content": "q"}], {}),
                     attempts=5)
    assert len(calls) == 1
    assert r.error == "fatal: bad request"


@pytest.mark.parametrize("key,a,b", [("max_tokens", 8, 9),
                                     ("temperature", 0.0, 0.7),
                                     ("repetition_penalty", 1.0, 1.1)])
def test_decode_signature_tracks_decode_keys_only(key, a, b):
    """Each of these changes what the model emits (a checkpoint-supplied
    repetition penalty included), so cells differing in it must not share a
    batch.  A non-decode key ("model") must not split one."""
    assert key in DECODE_KEYS
    assert decode_signature({key: a}) != decode_signature({key: b})
    assert (decode_signature({key: a, "model": "x"})
            == decode_signature({key: a, "model": "y"}))


@pytest.mark.parametrize("budgets,groups", [([8, 64, 8, 64], [[8, 8], [64, 64]]),
                                            ([8, 8, 8, 8], [[8, 8, 8, 8]])])
def test_hf_splits_a_batch_by_decode_params_only(budgets, groups):
    """The bug this guards: one generate() call per batch means a mixed batch
    silently ran every request at reqs[0]'s settings.  A mixed batch runs
    as one call per setting, and a homogeneous one stays one call."""
    seen = []

    class Spy(HFBackend):
        def _run_batch(self, reqs):
            seen.append([r.params["max_tokens"] for r in reqs])
            return [Response(r.unit_id, text=str(r.params["max_tokens"]))
                    for r in reqs]

    be = Spy(model=None, tokenizer=None, generate=True)
    out = be.complete_batch([Request(str(i), [{"role": "user", "content": "x"}],
                                     {"max_tokens": mt})
                             for i, mt in enumerate(budgets)])
    assert [r.unit_id for r in out] == [str(i) for i in range(4)]  # order kept
    assert [r.text for r in out] == [str(b) for b in budgets]      # own params
    assert seen == groups


def test_hf_run_batch_refuses_mixed_decode_params():
    """_run_batch decodes the whole batch at reqs[0]'s params, so called
    directly with a mixed batch it must fail rather than run it."""
    pytest.importorskip("torch")
    be = HFBackend(model=None, tokenizer=None, generate=True)
    reqs = [Request(str(i), [{"role": "user", "content": "x"}],
                    {"max_tokens": mt}) for i, mt in enumerate([8, 64])]
    with pytest.raises(AssertionError, match="2 decode signatures"):
        be._run_batch(reqs)


class _StubTok:
    """Minimal stand-in: the two attributes HFBackend pins, and nothing else."""

    def __init__(self, eos="</s>"):
        self.padding_side = "right"
        self.pad_token = None
        self.eos_token = eos

    @property
    def pad_token_id(self):
        return None if self.pad_token is None else 2


def test_hf_pins_left_padding_and_a_pad_token():
    tok = _StubTok()
    HFBackend(model=None, tokenizer=tok)
    assert tok.padding_side == "left"      # gen[:, prompt_len:] assumes it
    assert tok.pad_token == "</s>"


def test_hf_refuses_a_tokenizer_with_no_pad_and_no_eos():
    with pytest.raises(ValueError):
        HFBackend(model=None, tokenizer=_StubTok(eos=None))


def test_cache_put_is_concurrent_safe(tmp_path):
    """Two workers can hold the same cache key at once.

    Any grid with repeats > 1 at a non-zero temperature builds identical
    requests by construction -- the repeat index is not part of the request --
    so the collision is the normal case, not a rare race. With a temp file
    named per key rather than per writer, the second os.replace raised
    FileNotFoundError and killed the batch.
    """
    import concurrent.futures as cf

    cache = Cache(tmp_path / "cache")
    key = "a" * 32
    with cf.ThreadPoolExecutor(16) as pool:
        errs = [e for e in pool.map(
            lambda i: _put_or_err(cache, key, {"text": f"x{i}"}), range(200))
            if e is not None]

    assert errs == []
    assert cache.get(key)["text"].startswith("x")
    assert list((tmp_path / "cache").rglob("*.tmp")) == []


def _put_or_err(cache, key, payload):
    try:
        cache.put(key, payload)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - collects the bug
        return f"{type(exc).__name__}: {exc}"
    return None
