"""Backend parameter handling: no invented defaults, no cross-cell leakage,
truncation visible in the record."""

import pytest

from evalcore.core.backends import (
    DECODE_KEYS,
    Fatal,
    HFBackend,
    Request,
    Response,
    decode_signature,
    merge_params,
    require,
)
from evalcore.core.backends.http import (
    _anthropic_adapter,
    _anthropic_parse,
    _gemini_parse,
    _openai_parse,
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


def test_anthropic_adapter_needs_an_explicit_budget():
    msgs = [{"role": "user", "content": "hi"}]
    with pytest.raises(Fatal):
        _anthropic_adapter("m", msgs, {})
    _, body = _anthropic_adapter("m", msgs, {"max_tokens": 32})
    assert body["max_tokens"] == 32


@pytest.mark.parametrize("parse,js,want", [
    (_anthropic_parse, {"content": [], "stop_reason": "max_tokens"}, True),
    (_anthropic_parse, {"content": [], "stop_reason": "end_turn"}, False),
    (_openai_parse, {"choices": [{"message": {}, "finish_reason": "length"}]}, True),
    (_openai_parse, {"choices": [{"message": {}, "finish_reason": "stop"}]}, False),
    (_gemini_parse, {"candidates": [{"finishReason": "MAX_TOKENS"}]}, True),
    (_gemini_parse, {"candidates": [{"finishReason": "STOP"}]}, False),
])
def test_truncation_is_normalised_across_providers(parse, js, want):
    assert parse(js)[1]["truncated"] is want


def test_decode_signature_ignores_non_decode_keys():
    a = decode_signature({"max_tokens": 8, "model": "x"})
    b = decode_signature({"max_tokens": 8, "model": "y"})
    c = decode_signature({"max_tokens": 9, "model": "x"})
    assert a == b and a != c
    assert "max_tokens" in DECODE_KEYS


def test_hf_splits_mixed_decode_params_instead_of_taking_the_first():
    """The bug this guards: one generate() call per batch means a mixed batch
    silently ran every request at reqs[0]'s settings."""
    seen = []

    class Spy(HFBackend):
        def _run_batch(self, reqs):
            seen.append([r.params["max_tokens"] for r in reqs])
            return [Response(r.unit_id, text=str(r.params["max_tokens"]))
                    for r in reqs]

    be = Spy(model=None, tokenizer=None, generate=True)
    reqs = [Request(str(i), [{"role": "user", "content": "x"}],
                    {"max_tokens": mt})
            for i, mt in enumerate([8, 64, 8, 64])]
    out = be.complete_batch(reqs)

    assert [r.unit_id for r in out] == ["0", "1", "2", "3"]   # order preserved
    assert [r.text for r in out] == ["8", "64", "8", "64"]    # own params used
    assert sorted(len(g) for g in seen) == [2, 2]             # two groups, not one


def test_hf_homogeneous_batch_stays_one_call():
    seen = []

    class Spy(HFBackend):
        def _run_batch(self, reqs):
            seen.append(len(reqs))
            return [Response(r.unit_id, text="ok") for r in reqs]

    be = Spy(model=None, tokenizer=None, generate=True)
    reqs = [Request(str(i), [{"role": "user", "content": "x"}],
                    {"max_tokens": 8}) for i in range(4)]
    be.complete_batch(reqs)
    assert seen == [4]


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


def test_repetition_penalty_is_a_decode_parameter():
    # a checkpoint-supplied penalty changes what the model emits, so cells that
    # differ in it are not comparable and must not share a batch
    assert "repetition_penalty" in DECODE_KEYS
    assert (decode_signature({"repetition_penalty": 1.0})
            != decode_signature({"repetition_penalty": 1.1}))
