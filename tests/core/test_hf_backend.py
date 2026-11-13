"""HFBackend: what its identity covers, which decode params reach the model,
generation bookkeeping, and the BOS under a chat template.

Expected values never come from running the code under test: they are
hand-written token ids and kwargs, a direct `model.generate` call on one
unpadded prompt, or an invariance (two configurations that change outputs
must have two identities).  Models are random-init and tiny (a 2-layer
Qwen2 of width 32, the mean-pool toy LM).  Nothing is downloaded.

Teacher-forced and candidate scoring are in test_hf_scoring.py, prefix
reuse in test_hf_prefix_cache.py."""

from __future__ import annotations

import pytest

from evalcore import FunctionProbe, HFBackend, Request
from evalcore.core.backends.base import with_retries
from evalcore.testing import assert_same

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from support.toy_hf import (  # noqa: E402
    CharTok,
    KeepLM,
    Recorder,
    ToyLM,
    tiny_qwen,
    user,
)


@pytest.fixture(scope="module")
def qwen():
    return tiny_qwen(len(CharTok().vocab), n_layers=2, width=32)


# --- identity --------------------------------------------------------------------

def test_a1_hf_identity_changes_with_dtype():
    """A1. Invariance: the same weights in float32 and float64 give
    different logits, so the two backends must have different identities."""
    tok = CharTok()
    lo = HFBackend(KeepLM(len(tok.vocab)).float(), tok, device="cpu")
    hi = HFBackend(KeepLM(len(tok.vocab)).double(), tok, device="cpu")
    assert lo.identity() != hi.identity()


def test_a1_hf_identity_changes_with_chat_template():
    """A1. Invariance: two chat templates render the same messages to
    different prompts, so the two backends must have different identities.
    The templates here are hand-written strings that differ."""
    model = KeepLM(len(CharTok().vocab))
    a = HFBackend(model, CharTok(chat_template="A {{ m }}"), device="cpu")
    b = HFBackend(model, CharTok(chat_template="B {{ m }}"), device="cpu")
    assert a.identity() != b.identity()


def test_a24_hf_identity_covers_defaults_generate_and_attentions():
    """A24. Invariance: each of max_tokens 8 vs 64, generate True vs False,
    output_attentions True vs False changes what a row holds, so each pair
    must give two identities."""
    tok = CharTok()
    model = KeepLM(len(tok.vocab))

    def ident(**kw):
        return HFBackend(model, tok, device="cpu", **kw).identity()

    assert ident(default_params={"max_tokens": 8}) != \
        ident(default_params={"max_tokens": 64})
    assert ident(generate=True) != ident(generate=False)
    assert ident(output_attentions=True) != ident(output_attentions=False)


def test_two_revisions_of_one_checkpoint_are_different_backends():
    """from_pretrained(id, revision=r) keeps name_or_path == id for every r.
    The snapshot loaded is config._commit_hash.  Pythia's training-step axis
    is exactly this case: one cache dir, revisions step1000 ... step143000,
    and each must get its own request keys."""
    from types import SimpleNamespace
    tok = CharTok()
    keys = set()
    for commit in ("a39f36b", "de3e4e2"):
        m = ToyLM(len(tok.vocab))
        m.name_or_path = "EleutherAI/pythia-70m"
        m.config = SimpleNamespace(_commit_hash=commit)
        keys.add(str(HFBackend(m, tok).identity()))
    assert len(keys) == 2


# --- which decode params reach the model ----------------------------------------

@pytest.mark.parametrize("path", ["target", "candidates", "forward"])
def test_decode_params_are_refused_off_the_generate_path(path):
    """A3, A10.  Teacher forcing, candidate scoring and a plain forward
    (generate=False) each read logits once.  A temperature would move the
    cell id and the cache key while changing nothing that is computed, so
    a request carrying one must come back as an error naming it.  The same
    request without it must succeed (the fixture works)."""
    tok = CharTok()
    be = HFBackend(Recorder(len(tok.vocab)), tok, device="cpu")

    def make(params):
        if path == "candidates":
            return Request("c", [], params, input_ids=[3, 4], candidates=[5])
        return user("u", "1 =", params, **(
            {"target": " 1"} if path == "target" else {}))

    plain = with_retries(be.complete, make({}), attempts=1)
    assert plain.ok, plain.error
    r = with_retries(be.complete, make({"temperature": 0.7}), attempts=1)
    assert not r.ok
    assert "temperature" in r.error


@pytest.mark.parametrize("generate", [True, False])
def test_a_refused_decode_param_fails_only_its_own_request(generate):
    """In a batch of two, the request carrying a decode param this path
    never sends (top_p for generate(), any decode param for a plain
    forward) comes back as an error naming it, and its batch-mate still
    runs: the model is called once, on the good prompt alone."""
    tok = CharTok()
    model = Recorder(len(tok.vocab))
    be = HFBackend(model, tok, device="cpu", generate=generate,
                   default_params={"max_tokens": 2} if generate else None)
    good, bad = user("g", "ab"), user("b", "cd", {"top_p": 0.9})
    r_good, r_bad = be.complete_batch([good, bad])
    assert r_good.ok, r_good.error
    assert not r_bad.ok and "top_p" in r_bad.error
    fed = ([c["input_ids"] for c in model.generate_calls] if generate
           else model.forward_ids)
    assert fed == [[tok("ab")["input_ids"]]]


def test_a9_generate_keys_are_what_generate_receives():
    """A9. Expected kwargs are hand-written: max_tokens and
    repetition_penalty map to the generate() kwargs below and arrive
    unchanged.  temperature reaches the per-row sampler, not generate():
    generate() runs greedy, and the sampler's scores for logits z are
    z / T + noise.  The same request twice gets the same seed, so the noise
    cancels in proc_1(z) - proc_2(0), which must be z / 0.7 (hand-computed
    below), whatever the noise is."""
    to_kwarg = {"max_tokens": "max_new_tokens",
                "repetition_penalty": "repetition_penalty"}
    assert set(to_kwarg) | {"temperature"} == HFBackend.GENERATE_KEYS
    tok = CharTok()
    model = Recorder(len(tok.vocab))
    be = HFBackend(model, tok, device="cpu", generate=True)
    p = {"max_tokens": 3, "temperature": 0.7, "repetition_penalty": 1.3}
    for _ in range(2):
        r = be.complete(user("u", "ab", p))
        assert r.ok, r.error
    kw1, kw2 = model.generate_calls
    for key, kwarg in to_kwarg.items():
        assert kw1[kwarg] == p[key], key
    assert kw1["do_sample"] is False and kw1["temperature"] == 1.0
    z = torch.tensor([[0.0, 0.7, -1.4, 2.1]])
    ids = torch.zeros((1, 2), dtype=torch.long)
    diff = (kw1["logits_processor"](ids, z)
            - kw2["logits_processor"](ids, torch.zeros_like(z)))
    assert torch.allclose(diff, torch.tensor([[0.0, 1.0, -2.0, 3.0]]),
                          atol=1e-6)

    greedy = HFBackend(Recorder(len(tok.vocab)), tok, device="cpu",
                       generate=True)
    assert greedy.complete(user("g", "ab", {**p, "temperature": 0.0})).ok
    (kw,) = greedy.model.generate_calls
    assert kw["do_sample"] is False and "logits_processor" not in kw


# --- generation bookkeeping ------------------------------------------------------

def test_a15_left_padded_batch_generation_equals_per_prompt_generate(qwen):
    """A15. Expected text: transformers' own model.generate on each prompt
    alone, unpadded, greedy.  Prompts of lengths 5, 9 and 14 make the batch
    pad by 9 and 5 positions.  Compared as decoded strings: greedy ids are
    discrete, so any padding leak that moves an argmax shows."""
    tok = CharTok()
    texts = ["abcde", "the river", "where is it, q"]
    assert [len(t) for t in texts] == [5, 9, 14]
    be = HFBackend(qwen, tok, device="cpu", generate=True,
                   default_params={"max_tokens": 8, "temperature": 0.0})
    got = be.complete_batch([user(str(k), t) for k, t in enumerate(texts)])
    want = []
    for t in texts:
        ids = torch.tensor([tok(t)["input_ids"]])
        with torch.no_grad():
            gen = qwen.generate(ids, attention_mask=torch.ones_like(ids),
                                max_new_tokens=8, do_sample=False,
                                repetition_penalty=1.0, pad_token_id=0)
        want.append(tok.batch_decode(gen[:, ids.shape[1]:])[0])
    assert all(r.ok for r in got), [r.error for r in got]
    assert_same([r.text for r in got], want, what="batched vs per-prompt text")


def test_a16_truncated_iff_the_budget_was_hit():
    """A16. Hand-written generated rows, budget 4, pad = eos = 0:
    four real tokens -> truncated, two real tokens then eos and pad ->
    finished, not truncated."""
    tok = CharTok()
    model = Recorder(len(tok.vocab), new=[[5, 6, 7, 8], [5, 6, 0, 0]])
    be = HFBackend(model, tok, device="cpu", generate=True,
                   default_params={"max_tokens": 4, "temperature": 0.0})
    out = be.complete_batch([user("a", "ab"), user("b", "cd")])
    assert [r.meta["truncated"] for r in out] == [True, False]
    assert [r.text for r in out] == ["efgh", "ef"]


def test_a16_eos_in_the_last_slot_is_not_truncation_when_pad_differs():
    """A16. Hand-written row, budget 4, pad 0, eos "." (a different id):
    three real tokens then eos is a finished generation, not a cut one."""
    tok = CharTok()
    tok.eos_token = "."
    eos = tok.index["."]
    model = Recorder(len(tok.vocab), new=[[5, 6, 7, eos]])
    be = HFBackend(model, tok, device="cpu", generate=True,
                   default_params={"max_tokens": 4, "temperature": 0.0})
    (r,) = be.complete_batch([user("a", "ab")])
    assert r.meta["truncated"] is False


def test_a16_a_generation_config_eos_ends_a_generation_too():
    """A16.  generate() also stops on generation_config.eos_token_id (Llama 3
    instruct: <|eot_id|> besides the tokenizer's eos).  Hand-written rows,
    budget 4, extra stop ids "." and ":": a full row ending in ":" is
    finished.  A full row of letters is cut."""
    from types import SimpleNamespace
    tok = CharTok()
    stops = [tok.index["."], tok.index[":"]]
    model = Recorder(len(tok.vocab), new=[[5, 6, 7, stops[1]], [5, 6, 7, 8]])
    model.generation_config = SimpleNamespace(eos_token_id=stops)
    be = HFBackend(model, tok, device="cpu", generate=True,
                   default_params={"max_tokens": 4, "temperature": 0.0})
    out = be.complete_batch([user("a", "ab"), user("b", "cd")])
    assert [r.meta["truncated"] for r in out] == [False, True]


# --- encoding --------------------------------------------------------------------

@pytest.mark.parametrize("path", ["generate", "forward", "candidates", "target"])
def test_a20_one_bos_under_a_chat_template_on_every_encode_path(path):
    """A20. Expected: each row's first real token is the BOS and it is the
    only one.  The template writes one BOS, and the tokenizer would add a
    second only if asked for special tokens.  The model records the ids it
    actually receives, pad columns excluded."""
    tok = CharTok(bos="^")
    bos = tok.index["^"]
    model = Recorder(len(tok.vocab))
    be = HFBackend(model, tok, device="cpu", generate=path == "generate",
                   default_params=({"max_tokens": 2, "temperature": 0.0}
                                   if path == "generate" else None))
    kw = {"candidates": {"candidates": [5]}, "target": {"target": " c"}}.get(
        path, {})
    out = be.complete_batch([user("a", "ab", **kw), user("b", "abcdef", **kw)])
    assert all(r.ok for r in out), [r.error for r in out]
    rows = (model.generate_calls[0]["input_ids"] if path == "generate"
            else model.forward_ids[0])
    assert len(rows) == 2
    real = [[t for t in row if t != 0] for row in rows]
    assert [r[0] for r in real] == [bos, bos]
    assert [r.count(bos) for r in real] == [1, 1]


def test_a_param_nothing_reads_is_refused_unless_a_probe_may_read_it():
    """Without a probe HFBackend reads only DECODE_KEYS, so a label in the
    params ("model") would move the cell id while the forward pass stays the
    same: refused, naming the key.  A probe receives the requests and may
    read the key itself, so with one it is allowed and reaches the probe."""
    tok = CharTok()
    labelled = user("b", "1+1=", {"model": "x"})
    bare = HFBackend(KeepLM(len(tok.vocab)), tok, device="cpu")
    assert bare.refusal(user("a", "1+1=")) is None
    assert "'model'" in str(bare.refusal(labelled))

    seen: list[dict] = []

    def read_params(out, enc, reqs):
        seen.extend(dict(r.params) for r in reqs)
        return [{} for _ in reqs]

    probed = HFBackend(KeepLM(len(tok.vocab)), tok, device="cpu",
                       probe=FunctionProbe(read_params))
    assert probed.refusal(labelled) is None
    assert all(r.ok for r in probed.complete_batch([labelled]))
    assert seen == [{"model": "x"}]
