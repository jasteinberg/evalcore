"""Teacher-forced scoring in HFBackend, with no download.

A toy causal model and a toy tokenizer stand in for a checkpoint.  The
reference values come from running the model on each prefix of one
unpadded sequence separately -- no batching, no padding, no index
arithmetic shared with the code under test -- so agreement checks the
position bookkeeping, not a restatement of it.
"""

from __future__ import annotations

import pytest

from evalcore import FunctionProbe, HFBackend, Request

torch = pytest.importorskip("torch")


class ToyTok:
    """Character tokenizer, except that "ab" is a single token, so a join
    between "a" and "b" is a merge the boundary check must catch."""

    def __init__(self):
        chars = "abcdefghijklmnopqrstuvwxyz0123456789 +=\n"
        self.vocab = ["<pad>", "ab", *chars]
        self.index = {t: i for i, t in enumerate(self.vocab)}
        self.pad_token = self.eos_token = "<pad>"
        self.pad_token_id = 0
        self.padding_side = "right"
        self.chat_template = None

    def __call__(self, text, add_special_tokens=True):
        ids, i = [], 0
        while i < len(text):
            if text.startswith("ab", i):
                ids.append(self.index["ab"])
                i += 2
            else:
                ids.append(self.index[text[i]])
                i += 1
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(self.vocab[i] for i in ids)


class ToyLM(torch.nn.Module):
    """h_t = mean of the embeddings of x_0..x_t, logits = W h_t.  Causal by
    construction, so right padding cannot reach an earlier position."""

    def __init__(self, vocab, d=16, seed=0):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.emb = torch.nn.Parameter(torch.randn(vocab, d, generator=g))
        self.out = torch.nn.Parameter(torch.randn(d, vocab, generator=g))

    def forward(self, input_ids, attention_mask=None, output_attentions=False):
        e = self.emb[input_ids]
        h = e.cumsum(1) / torch.arange(1, e.shape[1] + 1)[None, :, None]

        class Out:
            logits = h @ self.out
        return Out()


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
    tok = ToyTok()
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


def test_batch_equals_single():
    """Padding to the longest sequence must not change any score."""
    be, _ = backend()
    batch = be.complete_batch([req(str(k), p, t)
                               for k, (p, t) in enumerate(CASES)])
    for k, (p, t) in enumerate(CASES):
        alone = be.complete_batch([req(str(k), p, t)])[0]
        assert alone.meta["logp"] == pytest.approx(batch[k].meta["logp"],
                                                   abs=1e-6)


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


def test_decode_params_are_refused_when_scoring():
    """Teacher forcing reads logits; temperature or max_tokens would move the
    cell id and the cache key while changing nothing that is computed."""
    be, _ = backend()
    r = be.complete_batch([req("u", "1 =", " 1", temperature=0.7)])[0]
    assert not r.ok and "temperature" in r.error


def test_mixed_batch_scores_targets_and_routes_the_rest():
    class Spy(HFBackend):
        def _run_batch(self, reqs):
            from evalcore import Response
            return [Response(r.unit_id, text="generated") for r in reqs]

    tok = ToyTok()
    be = Spy(ToyLM(len(tok.vocab)), tok, device="cpu")
    plain = Request("p", [{"role": "user", "content": "1 ="}], {})
    out = be.complete_batch([req("s", "1 =", " 1"), plain])
    assert [r.unit_id for r in out] == ["s", "p"]
    assert out[0].meta["n_target"] == 2 and out[1].text == "generated"


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

    tok = ToyTok()
    be = HFBackend(ToyLM(len(tok.vocab)), tok, device="cpu",
                   probe=FunctionProbe(extract))
    be.complete_batch([req("a", "1 =", " 1"), req("b", "123 + 4 =", " 127")])
    assert seen["padding_side"] == "right"
    assert seen["n_prompt"] == [3, 9]
    assert seen["attention_mask"].sum(1).tolist() == [5, 13]


class BosTok(ToyTok):
    """Adds a BOS ("^") unless told not to, and has a chat template that
    also emits one -- the Llama arrangement."""

    def __init__(self):
        super().__init__()
        self.vocab.append("^")
        self.index["^"] = len(self.vocab) - 1
        self.chat_template = "{{ bos_token }}..."

    def __call__(self, text, add_special_tokens=True):
        ids = super().__call__(text)["input_ids"]
        return {"input_ids": ([self.index["^"]] if add_special_tokens else [])
                + ids}

    def apply_chat_template(self, msgs, tokenize=False,
                            add_generation_prompt=True):
        return "^" + "".join(m["content"] for m in msgs)


def test_a_templated_prompt_gets_exactly_one_bos():
    tok = BosTok()
    be = HFBackend(ToyLM(len(tok.vocab)), tok, device="cpu")
    ids, _ = be._target_ids(req("u", "1 =", " 1"))
    bos = tok.index["^"]
    assert ids[0] == bos and ids[1] != bos


def test_backend_decode_defaults_do_not_block_scoring():
    """Defaults set for generation are not part of a scoring request; only
    decode keys the REQUEST carries are refused."""
    tok = ToyTok()
    be = HFBackend(ToyLM(len(tok.vocab)), tok, device="cpu", generate=True,
                   default_params={"max_tokens": 4, "temperature": 0.0})
    assert be.complete_batch([req("u", "1 =", " 1")])[0].ok
    bad = be.complete_batch([req("u", "1 =", " 1", temperature=0.7)])[0]
    assert not bad.ok and "temperature" in bad.error


def test_device_defaults_to_where_the_model_is():
    """No device given: inputs follow the model.  A guessed accelerator
    would differ from an unmoved model's device and fail."""
    tok = ToyTok()
    model = ToyLM(len(tok.vocab))
    be = HFBackend(model, tok)
    assert str(be._device()) == "cpu"
    assert be.complete_batch([req("u", "1 =", " 1")])[0].ok

    class OnDevice(ToyLM):
        device = "meta-device-label"            # HF models expose .device

    assert HFBackend(OnDevice(len(tok.vocab)), tok)._device() == \
        "meta-device-label"
    assert HFBackend(model, tok, device="cpu")._device() == "cpu"


def test_a_probe_returning_the_wrong_count_fails_by_name():
    tok = ToyTok()

    def short(out, enc, reqs):
        return [{}]                              # one dict for two requests

    be = HFBackend(ToyLM(len(tok.vocab)), tok, device="cpu",
                   probe=FunctionProbe(short))
    with pytest.raises(ValueError, match=r"FunctionProbe\(short\) returned 1 "
                       r"dicts for 2"):
        be.complete_batch([req("a", "1 =", " 1"), req("b", "2 =", " 2")])


# --- candidate scoring and token-id prompts ------------------------------------

class KeepLM(ToyLM):
    """ToyLM whose forward takes `logits_to_keep` (a tensor of positions),
    as current HF causal LMs do."""

    def forward(self, input_ids, attention_mask=None, output_attentions=False,
                logits_to_keep=None):
        full = super().forward(input_ids).logits
        keep = full if logits_to_keep is None else full[:, logits_to_keep, :]

        class Out:
            logits = keep
        return Out()


@torch.no_grad()
def next_logp(model, ids):
    return model(torch.tensor([ids])).logits[0, -1].double().log_softmax(-1)


def choice(uid, ids, cands):
    return Request(uid, [], {}, input_ids=ids, candidates=cands)


@pytest.mark.parametrize("cls", [ToyLM, KeepLM], ids=["full", "keep"])
def test_candidate_logp_matches_an_unpadded_single_run(cls):
    tok = ToyTok()
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
    tok = ToyTok()
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
    tok = ToyTok()
    be = HFBackend(KeepLM(len(tok.vocab)), tok, device="cpu")
    out = be.complete_batch([choice("c", [3, 4, 5], [6, 7]),
                             req("t", "1 =", " 1"),
                             choice("c2", [8], [9])])
    assert [r.unit_id for r in out] == ["c", "t", "c2"]
    assert "cand_logp" in out[0].meta and "logp" in out[1].meta
    assert "cand_logp" in out[2].meta


def test_a_target_with_input_ids_is_refused():
    tok = ToyTok()
    be = HFBackend(KeepLM(len(tok.vocab)), tok, device="cpu")
    r = be.complete_batch([Request("u", [], {}, input_ids=[3, 4],
                                   target=" 5")])[0]
    assert not r.ok and "token boundary" in r.error


def test_two_revisions_of_one_checkpoint_are_different_backends():
    """from_pretrained(id, revision=r) keeps name_or_path == id for every r;
    the snapshot loaded is config._commit_hash.  Pythia's training-step axis
    is exactly this case: one cache dir, revisions step1000 ... step143000,
    and each must get its own request keys."""
    from types import SimpleNamespace
    tok = ToyTok()
    keys = set()
    for commit in ("a39f36b", "de3e4e2"):
        m = ToyLM(len(tok.vocab))
        m.name_or_path = "EleutherAI/pythia-70m"
        m.config = SimpleNamespace(_commit_hash=commit)
        keys.add(str(HFBackend(m, tok).identity()))
    assert len(keys) == 2
