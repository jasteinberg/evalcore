"""Teacher-forced and candidate scoring in HFBackend, with no download.

A toy causal model and a toy tokenizer stand in for a checkpoint.  The
tokenizer is one token per character except that "ab" is a single token,
so a join between "a" and "b" is a merge the boundary check must catch.
The reference values come from running the model on each prefix of one
unpadded sequence separately -- no batching, no padding, no index
arithmetic shared with the code under test -- so agreement checks the
position bookkeeping, not a restatement of it.
"""

from __future__ import annotations

import pytest

from evalcore import FunctionProbe, HFBackend, Request

torch = pytest.importorskip("torch")

from support.toy_hf import CharTok, IgnoreKeepLM, KeepLM, SwallowLM, ToyLM  # noqa: E402

MERGES = ("ab",)


@torch.no_grad()
def reference(model, tok, prompt, target):
    """logp and argmax_ok of each target token, one prefix at a time."""
    p_ids = tok(prompt)["input_ids"]
    ids = tok(prompt + target)["input_ids"]
    logp, ok = [], []
    for t in range(len(p_ids), len(ids)):
        z = model(torch.tensor([ids[:t]])).logits[0, -1].double()
        lsm = z.log_softmax(-1)
        logp.append(float(lsm[ids[t]]))
        ok.append(int(z.argmax()) == ids[t])
    return logp, ok


def backend():
    tok = CharTok(MERGES)
    return HFBackend(ToyLM(len(tok.vocab)), tok, device="cpu"), tok


def req(uid, prompt, target, **params):
    return Request(uid, [{"role": "user", "content": prompt}], params,
                   target=target)


CASES = [("12 + 34 =", " 46"), ("7 + 5 =", " 12"),
         ("123 + 456 + 789 =", " 1368"), ("x =", " y")]


def test_batched_scores_match_one_prefix_at_a_time():
    be, tok = backend()
    resps = be.complete_batch([req(str(k), p, t)
                               for k, (p, t) in enumerate(CASES)])
    for (p, t), r in zip(CASES, resps, strict=True):
        assert r.ok, r.error
        logp, ok = reference(be.model, tok, p, t)
        assert r.meta["n_target"] == len(t)          # char tokens
        assert r.meta["logp"] == pytest.approx(logp, abs=1e-5)
        assert r.meta["argmax_ok"] == ok
        assert r.meta["em_tf"] is all(ok)
        assert r.meta["sum_logp"] == pytest.approx(sum(logp), abs=1e-5)
        assert "".join(r.meta["target_tokens"]) == t
        assert r.meta["truncated"] is False and r.text is None
        # padding to the longest sequence changes no score: the same request
        # alone, through the same backend, to the tighter 1e-6
        alone = be.complete_batch([req(r.unit_id, p, t)])[0]
        assert alone.meta["logp"] == pytest.approx(r.meta["logp"], abs=1e-6)


def test_a_merge_across_the_join_is_refused_and_only_that_request_fails():
    be, _ = backend()
    resps = be.complete_batch([req("ok", "12 =", " 3"),
                               req("bad", "xa", "b")])  # "ab" spans the join
    assert resps[0].ok
    assert not resps[1].ok and "token boundary" in resps[1].error


def test_an_empty_prompt_is_refused():
    be, _ = backend()
    r = be.complete_batch([Request("u", [{"role": "user", "content": ""}],
                                   {}, target="5")])[0]
    assert not r.ok and "non-empty prompt" in r.error


def test_pythia_tokenizer_boundary(monkeypatch):
    """The real convention on the real tokenizer, if it is cached locally:
    answers with a leading space pass, a mid-word split is refused."""
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    transformers = pytest.importorskip("transformers")
    try:
        tok = transformers.AutoTokenizer.from_pretrained("EleutherAI/pythia-70m")
    except OSError:
        pytest.skip("pythia-70m tokenizer not cached")
    be = HFBackend(None, tok, device="cpu")
    ids, n_prompt = be._target_ids(req("u", "12 + 34 =", " 1046"))
    assert [tok.decode([i]) for i in ids[n_prompt:]] == [" 10", "46"]
    from evalcore.core.backends import Fatal
    with pytest.raises(Fatal, match="token boundary"):
        be._target_ids(req("u", "hello wor", "ld"))


# --- review findings (2 Oct) -------------------------------------------------

def test_hf_backend_runs_at_batch_size_one(tmp_path):
    """execute() calls backend.complete when batch_size == 1.  HFBackend only
    implemented complete_batch, so every unit came back as a
    NotImplementedError row."""
    import io

    from evalcore import execute, grid, units
    from evalcore.core.spec import Item

    be, _ = backend()
    its = [Item(f"q{i}", {"p": p, "t": t}) for i, (p, t) in enumerate(CASES)]

    def render(u):
        return Request(u.id, [{"role": "user", "content": u.item.payload["p"]}],
                       {}, target=u.item.payload["t"])

    df = execute(units(grid({"m": ["toy"]}), its), be, render,
             lambda u, r: {"em": float(r.meta["em_tf"])}, tmp_path / "r.jsonl",
             workers=1, batch_size=1, stream=io.StringIO())
    assert (df["status"] == "ok").all(), df["error"].tolist()


def test_a_probe_in_scoring_mode_is_told_the_layout():
    seen = {}

    def extract(out, enc, reqs):
        seen.update(enc)
        return [{} for _ in reqs]

    tok = CharTok(MERGES)
    be = HFBackend(ToyLM(len(tok.vocab)), tok, device="cpu",
                   probe=FunctionProbe(extract))
    be.complete_batch([req("a", "1 =", " 1"), req("b", "123 + 4 =", " 127")])
    assert seen["padding_side"] == "right"
    assert seen["n_prompt"] == [3, 9]
    assert seen["attention_mask"].sum(1).tolist() == [5, 13]


def test_backend_decode_defaults_do_not_block_scoring():
    """Defaults set for generation are not part of a scoring request.  Only
    decode keys the REQUEST carries are refused."""
    tok = CharTok(MERGES)
    be = HFBackend(ToyLM(len(tok.vocab)), tok, device="cpu", generate=True,
                   default_params={"max_tokens": 4, "temperature": 0.0})
    assert be.complete_batch([req("u", "1 =", " 1")])[0].ok
    bad = be.complete_batch([req("u", "1 =", " 1", temperature=0.7)])[0]
    assert not bad.ok and "temperature" in bad.error


def test_device_defaults_to_where_the_model_is():
    """No device given: inputs follow the model.  A guessed accelerator
    would differ from an unmoved model's device and fail."""
    tok = CharTok(MERGES)
    model = ToyLM(len(tok.vocab))
    be = HFBackend(model, tok)
    assert str(be._device()) == "cpu"
    assert be.complete_batch([req("u", "1 =", " 1")])[0].ok

    class OnDevice(ToyLM):
        device = "meta-device-label"            # HF models expose .device

    assert HFBackend(OnDevice(len(tok.vocab)), tok)._device() == \
        "meta-device-label"
    assert HFBackend(model, tok, device="cpu")._device() == "cpu"


# --- candidate scoring and token-id prompts ------------------------------------

@torch.no_grad()
def next_logp(model, ids):
    return model(torch.tensor([ids])).logits[0, -1].double().log_softmax(-1)


def choice(uid, ids, cands):
    return Request(uid, [], {}, input_ids=ids, candidates=cands)


@pytest.mark.parametrize("cls", [ToyLM, KeepLM, SwallowLM],
                         ids=["full", "keep", "kwargs"])
def test_candidate_logp_matches_an_unpadded_single_run(cls):
    tok = CharTok(MERGES)
    model = cls(len(tok.vocab))
    be = HFBackend(model, tok, device="cpu")
    prompts = [[3, 4, 5, 6, 7], [8, 9], [10, 11, 12, 13, 14, 15, 16, 17]]
    cands = [[20, 21, 22], [5, 30], [2, 3, 4, 25]]
    out = be.complete_batch([choice(str(k), p, c)
                             for k, (p, c) in enumerate(zip(prompts, cands,
                                                            strict=True))])
    for p, c, r in zip(prompts, cands, out, strict=True):
        assert r.ok, r.error
        ref = next_logp(model, p)
        assert r.meta["cand_logp"] == pytest.approx([float(ref[i]) for i in c],
                                                    abs=1e-5)
        assert r.meta["leakage"] == pytest.approx(1 - r.meta["cand_mass"])
        assert r.meta["cand_mass"] == pytest.approx(
            float(ref[c].exp().sum()), abs=1e-5)
        top = int(ref.argmax())
        assert r.meta["top1_id"] == top
        assert r.meta["top1_cand_index"] == (c.index(top) if top in c else -1)
        assert r.meta["n_prompt"] == len(p)


def test_candidates_over_a_rendered_prompt_and_bad_sets_refused():
    tok = CharTok(MERGES)
    be = HFBackend(KeepLM(len(tok.vocab)), tok, device="cpu")
    good = Request("g", [{"role": "user", "content": "12 + 34 ="}], {},
                   candidates=[tok.index["4"], tok.index["5"]])
    out = be.complete_batch([good, choice("e", [3, 4], []),
                             choice("d", [3, 4], [5, 5]),
                             choice("v", [3, 4], [10_000])])
    assert out[0].ok
    assert "empty" in out[1].error and "duplicates" in out[2].error
    assert "outside the vocabulary" in out[3].error


def test_a_mixed_batch_is_split_by_kind():
    """Candidates, a target and a plain prompt in one batch: each goes to
    its own path, and the responses come back in request order."""
    class Spy(HFBackend):
        def _run_batch(self, reqs):
            from evalcore import Response
            return [Response(r.unit_id, text="generated") for r in reqs]

    tok = CharTok(MERGES)
    be = Spy(KeepLM(len(tok.vocab)), tok, device="cpu")
    plain = Request("p", [{"role": "user", "content": "1 ="}], {})
    out = be.complete_batch([choice("c", [3, 4, 5], [6, 7]),
                             req("t", "1 =", " 1"), plain,
                             choice("c2", [8], [9])])
    assert [r.unit_id for r in out] == ["c", "t", "p", "c2"]
    assert "cand_logp" in out[0].meta and "cand_logp" in out[3].meta
    assert out[1].meta["n_target"] == 2 and out[2].text == "generated"


def test_a_target_with_input_ids_is_refused():
    tok = CharTok(MERGES)
    be = HFBackend(KeepLM(len(tok.vocab)), tok, device="cpu")
    r = be.complete_batch([Request("u", [], {}, input_ids=[3, 4],
                                   target=" 5")])[0]
    assert not r.ok and "token boundary" in r.error


def test_a_forward_that_ignores_a_declared_logits_to_keep_fails_loudly():
    """Full logits read as kept ones would score the wrong positions."""
    tok = CharTok(MERGES)
    be = HFBackend(IgnoreKeepLM(len(tok.vocab)), tok, device="cpu")
    with pytest.raises(RuntimeError, match="does not honour logits_to_keep"):
        be.complete_batch([choice("u", [3, 4, 5, 6], [7, 8])])


def test_a_type_error_inside_the_model_is_not_retried_without_keep():
    """The forward's signature, not a caught TypeError, decides the call."""
    calls = []

    class Broken(KeepLM):
        def forward(self, input_ids, attention_mask=None,
                    output_attentions=False, logits_to_keep=None):
            calls.append(logits_to_keep is not None)
            raise TypeError("a bug inside the model")

    tok = CharTok(MERGES)
    be = HFBackend(Broken(len(tok.vocab)), tok, device="cpu")
    with pytest.raises(TypeError, match="a bug inside the model"):
        be.complete_batch([choice("u", [3, 4, 5], [6, 7])])
    assert calls == [True]                      # once, and with keep
