"""Prefix KV-cache reuse in HFBackend, on a random-init Qwen2 in float32.

Reference values come from the model itself run on each FULL prompt with no
cache (`model(ids)` and `model.generate(ids)` called directly, hidden states
from `output_hidden_states`), so nothing is shared with the reuse
bookkeeping under test.  The expected reuse counts come from the character
tokenizer: one token per character, so the shared prefix of two prompts is
the shared prefix of the strings.

Tolerance.  Reuse changes only the order of floating-point sums (attention
over cached keys, matmuls over a shorter sequence), not the arithmetic.
Under round-to-nearest the error of a length-k sum grows like
sqrt(k) * eps * (scale) for random-sign rounding.  Here float32
eps = 2^-23, the longest inner sum is k <= 2 * WIDTH = 128 (the MLP's
inner width; attention sums run over T <= 2 * WIDTH positions, checked
below), and there is one such error per layer plus the head.  A quantity of
magnitude <= S read after the model's layers may then move by at most about

    (N_LAYERS + 1) * sqrt(2 * WIDTH) * eps * S,

and log-softmax adds O(eps * ZMAX).  The factor 2 in TOL is margin.  The
tests check |z| <= ZMAX on the reference logits, and take S = max |x| of the
reference vector for residual reads, so the bound applies.
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from evalcore import FunctionProbe, HFBackend
from evalcore.core.backends import Fatal
from evalcore.core.backends.probe import Probe
from evalcore.testing import assert_same

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from support.toy_hf import CharTok, tiny_qwen, user  # noqa: E402

N_LAYERS, WIDTH, ZMAX = 4, 64, 10.0
EPS = float(torch.finfo(torch.float32).eps)
REL = 2 * (N_LAYERS + 1) * math.sqrt(2 * WIDTH) * EPS
TOL = REL * ZMAX
BLOCK = 1                                  # the hooked decoder layer


@pytest.fixture(scope="module")
def model():
    return tiny_qwen(len(CharTok().vocab), n_layers=N_LAYERS, width=WIDTH)


@torch.no_grad()
def ref_pass(model, ids) -> dict[str, Any]:
    """A full uncached forward on `ids`: last-position log probs, the output
    of layer BLOCK and lm_head's input there.  hidden_states[i] is the input
    to layer i (so layer BLOCK's output is [BLOCK + 1]) and the last entry is
    the final-normed state lm_head reads."""
    assert len(ids) <= 2 * WIDTH            # the bound in the docstring
    out = model(torch.tensor([ids]), output_hidden_states=True)
    z = out.logits[0, -1]
    assert float(z.abs().max()) <= ZMAX
    return {"logp": z.log_softmax(-1),
            "resid": out.hidden_states[BLOCK + 1][0, -1],
            "head_in": out.hidden_states[-1][0, -1]}


def full_generate(model, ids, n=6) -> str:
    """Greedy text from a direct uncached generate() on `ids`."""
    x = torch.tensor([ids])
    with torch.no_grad():
        g = model.generate(x, attention_mask=torch.ones_like(x),
                           max_new_tokens=n, do_sample=False,
                           repetition_penalty=1.0, pad_token_id=0)
    return CharTok().batch_decode(g[:, x.shape[1]:])[0]


def close(got, ref):
    """Elementwise within REL * max|ref| (the docstring's S)."""
    ref = torch.as_tensor(ref, dtype=torch.float32)
    err = float((torch.as_tensor(got) - ref).abs().max())
    assert err <= REL * float(ref.abs().max()), err


class ReadPatch(Probe):
    """A generic read-and-patch probe written against the documented `enc`
    contract.  On the first forward of each call it reads layer BLOCK's
    output and lm_head's input at the read position, then (if the request
    has params["patch"] = alpha) adds alpha * DIRECTION to layer BLOCK's
    output there; with params["patch_every"] also at every generated token.
    The read happens before the patch is added."""

    calls_per_forward = None                 # generate runs several forwards

    def __init__(self) -> None:
        g = torch.Generator().manual_seed(1)
        d = torch.randn(WIDTH, generator=g)
        self.direction = d / d.norm()

    def identity(self) -> dict[str, Any]:
        return {"probe": "ReadPatch", "block": BLOCK}

    def hooks(self):
        return {f"model.layers.{BLOCK}": self._block, "lm_head": self._head}

    def start(self, enc, reqs) -> None:
        width = enc["attention_mask"].shape[1]
        self.read = [(enc["last"][k] if enc["padding_side"] == "right"
                      else width - 1) - enc["n_prefix_reused"][k]
                     for k in range(len(reqs))]
        self.patch = [(r.params.get("patch"), r.params.get("patch_every"))
                      for r in reqs]
        self.n_block = self.n_head = 0
        self.rows: list[dict[str, Any]] = [{} for _ in reqs]
        self.first_seq: int | None = None

    def _block(self, name, module, args, out):
        h = out[0] if isinstance(out, tuple) else out
        first = self.n_block == 0
        self.n_block += 1
        if first:
            self.first_seq = h.shape[1]
        h = h.clone()
        for k, (alpha, every) in enumerate(self.patch):
            if first:
                self.rows[k]["resid"] = h[k, self.read[k]].tolist()
            if alpha is not None and (first or every):
                h[k, self.read[k] if first else -1] += alpha * self.direction
        return (h, *out[1:]) if isinstance(out, tuple) else h

    def _head(self, name, module, args, out):
        if self.n_head == 0:
            x = args[0]
            for k in range(x.shape[0]):
                col = -1 if x.shape[1] == 1 else self.read[k]
                self.rows[k]["head_in"] = x[k, col].tolist()
        self.n_head += 1

    def collect(self, out, enc, reqs):
        return [{**r, "first_seq": self.first_seq} for r in self.rows]


# Each prompt extends the last, then one diverges at token 0, then a repeat:
# (text, expected reused positions).  A stored prefix is the prompt minus
# its last token, so an extension reuses len(previous) - 1.
CTX = "log:\n1 abc.\n"
SEQ = [(CTX + "you:", 0),
       (CTX + "2 xyz.\nyou:", len(CTX)),
       (CTX + "2 xyz.\n3 qq.\nyou:", len(CTX + "2 xyz.\n")),
       ("zzz unrelated", 0),                          # diverges at token 0
       ("zzz unrelated", len("zzz unrelated") - 1)]   # a repeat feeds 1 token


def cands_of(tok):
    return [tok.index[c] for c in "abc12y"]


@pytest.mark.parametrize("with_probe", [False, True])
def test_candidates_with_reuse_equal_a_full_pass(model, with_probe):
    tok = CharTok()
    be = HFBackend(model, tok, device="cpu", prefix_cache=True, prefix_slots=1,
                   probe=ReadPatch() if with_probe else None)
    cands = cands_of(tok)
    for k, (text, want_n) in enumerate(SEQ):
        r = be.complete_batch([user(str(k), text, candidates=cands)])[0]
        assert r.ok, r.error
        assert r.meta["n_prefix_reused"] == want_n, text
        assert r.meta["prefix_slot"] == (None if want_n == 0 else 0)
        ref = ref_pass(model, tok(text)["input_ids"])
        assert r.meta["cand_logp"] == pytest.approx(
            [float(ref["logp"][c]) for c in cands], abs=TOL)
        assert r.meta["top1_id"] == int(ref["logp"].argmax())
        if with_probe:
            assert r.meta["first_seq"] == len(text) - want_n   # only the tail
            close(r.meta["resid"], ref["resid"])
            close(r.meta["head_in"], ref["head_in"])


@pytest.mark.parametrize("with_probe", [False, True])
def test_greedy_generation_with_reuse_equals_a_full_pass(model, with_probe):
    tok = CharTok()
    be = HFBackend(model, tok, device="cpu", generate=True, prefix_cache=True,
                   prefix_slots=1, probe=ReadPatch() if with_probe else None,
                   default_params={"max_tokens": 6, "temperature": 0.0})
    for k, (text, want_n) in enumerate(SEQ):
        r = be.complete_batch([user(str(k), text)])[0]
        assert r.ok, r.error
        assert r.meta["n_prefix_reused"] == want_n, text
        assert r.text == full_generate(model, tok(text)["input_ids"]), text
        if with_probe:
            assert r.meta["first_seq"] == len(text) - want_n
            ref = ref_pass(model, tok(text)["input_ids"])
            close(r.meta["resid"], ref["resid"])
            close(r.meta["head_in"], ref["head_in"])


@pytest.mark.parametrize("path", ["candidates", "generate"])
@pytest.mark.parametrize("follow", ["same", "extension"])
def test_a_patched_call_leaves_no_trace_in_the_stored_prefix(model, path,
                                                              follow):
    """A patched call on prompt P (read position and every generated token),
    then an unpatched call on P itself or on an extension of P.  The second
    reuses len(P) - 1 positions stored by the first, and must give the
    unpatched full-pass result.  The extension is the case that needs the
    T - 1 crop: it would otherwise reuse P's patched last position.  The
    patched call itself must differ from the reference by far more than the
    tolerance, or the test would pass with a patch that did nothing."""
    tok = CharTok()
    gen = path == "generate"
    be = HFBackend(model, tok, device="cpu", generate=gen, prefix_cache=True,
                   prefix_slots=1, probe=ReadPatch(),
                   default_params=({"max_tokens": 6, "temperature": 0.0}
                                   if gen else None))
    text = CTX + "2 xyz.\nyou:"
    later = text if follow == "same" else text + " ok.\nyou:"
    cands = cands_of(tok)
    kw = {} if gen else {"candidates": cands}
    patched = be.complete_batch([user("p", text, {"patch": 50.0,
                                                  "patch_every": True}, **kw)])[0]
    plain = be.complete_batch([user("u", later, **kw)])[0]
    assert patched.ok and plain.ok, (patched.error, plain.error)
    assert plain.meta["n_prefix_reused"] == len(text) - 1
    ref = ref_pass(model, tok(later)["input_ids"])
    if gen:
        assert patched.text != full_generate(model, tok(text)["input_ids"])
        assert plain.text == full_generate(model, tok(later)["input_ids"])
    else:
        unpatched = ref_pass(model, tok(text)["input_ids"])["logp"]
        moved = max(abs(a - float(unpatched[c])) for a, c in
                    zip(patched.meta["cand_logp"], cands, strict=True))
        assert moved > 1000 * TOL                 # the patch took effect
        assert plain.meta["cand_logp"] == pytest.approx(
            [float(ref["logp"][c]) for c in cands], abs=TOL)
    close(plain.meta["resid"], ref["resid"])
    close(plain.meta["head_in"], ref["head_in"])


def test_k_slots_keep_interleaved_prefixes_apart(model):
    """Prompts A, B, A', B' where A' extends A, B' extends B, and A and B
    share a header H.  With two slots each extension reuses its own prompt
    (len - 1 positions); with one slot B evicts A, so A' reuses only H.
    Every result also equals a full pass."""
    tok = CharTok()
    H = "rules: be brief.\n"
    A, B = H + "log 1 abc.\nyou:", H + "mode: choose.\nyou:"
    A2, B2 = A + " ok.\n2 xy.\nyou:", B + " a.\nyou:"
    cands = cands_of(tok)

    def run(slots):
        be = HFBackend(model, tok, device="cpu", prefix_cache=True,
                       prefix_slots=slots, probe=ReadPatch())
        rows = []
        for k, text in enumerate([A, B, A2, B2]):
            r = be.complete_batch([user(str(k), text, candidates=cands)])[0]
            assert r.ok, r.error
            ref = ref_pass(model, tok(text)["input_ids"])
            assert r.meta["cand_logp"] == pytest.approx(
                [float(ref["logp"][c]) for c in cands], abs=TOL)
            close(r.meta["resid"], ref["resid"])
            rows.append((r.meta["n_prefix_reused"], r.meta["prefix_slot"]))
        return rows

    assert run(2) == [(0, None), (len(H), 0), (len(A) - 1, 0), (len(B) - 1, 1)]
    assert run(1) == [(0, None), (len(H), 0), (len(H), 0), (len(H), 0)]


def test_a_prompt_sharing_no_first_token_runs_a_full_pass(model):
    tok = CharTok()
    be = HFBackend(model, tok, device="cpu", prefix_cache=True, prefix_slots=3,
                   probe=ReadPatch())
    cands = cands_of(tok)
    for k, text in enumerate(["abc log", "bcd log", "cde log"]):
        be.complete_batch([user(str(k), text, candidates=cands)])
    text = "zzz log"                    # first token matches no slot
    r = be.complete_batch([user("z", text, candidates=cands)])[0]
    assert (r.meta["n_prefix_reused"], r.meta["prefix_slot"]) == (0, None)
    assert r.meta["first_seq"] == len(text)
    ref = ref_pass(model, tok(text)["input_ids"])
    assert r.meta["cand_logp"] == pytest.approx(
        [float(ref["logp"][c]) for c in cands], abs=TOL)


def test_enc_describes_the_full_prompt_under_reuse(model):
    """The documented contract: input_ids and attention_mask cover cached +
    fed positions, n_prefix_reused says how many were not fed, and the
    key is present (as zeros) without reuse."""
    tok = CharTok()
    seen: list[dict[str, Any]] = []

    def grab(out, enc, reqs):
        seen.append({"ids": enc["input_ids"][0].tolist(),
                     "mask": int(enc["attention_mask"].sum()),
                     "n": list(enc["n_prefix_reused"]),
                     "side": enc["padding_side"]})
        return [{}]

    for gen in (False, True):
        for on in (False, True):
            seen.clear()
            be = HFBackend(model, tok, device="cpu", generate=gen,
                           probe=FunctionProbe(grab),
                           default_params=({"max_tokens": 2} if gen else None),
                           **({"prefix_cache": True, "prefix_slots": 1}
                              if on else {}))
            for k, text in enumerate([CTX, CTX + "q"]):
                be.complete_batch([user(str(k), text)])
                assert seen[-1]["ids"] == tok(text)["input_ids"]
                assert seen[-1]["mask"] == len(text)
                assert seen[-1]["side"] == "left"
            assert [s["n"] for s in seen] == [[0], [len(CTX) - 1 if on else 0]]


def test_a_generated_reply_does_not_stay_in_the_stored_prefix(model):
    """generate() appends its new tokens to the cache.  The slot must hold
    exactly the prompt's first T - 1 positions afterwards."""
    tok = CharTok()
    be = HFBackend(model, tok, device="cpu", generate=True, prefix_cache=True,
                   prefix_slots=1,
                   default_params={"max_tokens": 8, "temperature": 0.0})
    be.complete_batch([user("0", CTX)])
    stored = be._slots[0]
    assert stored is not None
    stored_ids, cache = stored
    assert stored_ids == tok(CTX)["input_ids"][:-1]
    assert cache.get_seq_length() == len(CTX) - 1


def test_the_flag_and_slots_enter_identity_only_when_on(model):
    off = HFBackend(model, CharTok(), device="cpu").identity()
    on = HFBackend(model, CharTok(), device="cpu", prefix_cache=True,
                   prefix_slots=2).identity()
    assert "prefix_cache" not in off and "prefix_slots" not in off
    assert on == {**off, "prefix_cache": True, "prefix_slots": 2}


def test_unsupported_combinations_are_refused(model):
    with pytest.raises(ValueError, match="needs prefix_slots"):
        HFBackend(model, CharTok(), prefix_cache=True)
    with pytest.raises(ValueError, match="prefix_cache is off"):
        HFBackend(model, CharTok(), prefix_slots=2)
    with pytest.raises(ValueError, match=">= 1"):
        HFBackend(model, CharTok(), prefix_cache=True, prefix_slots=0)
    be = HFBackend(model, CharTok(), device="cpu", prefix_cache=True,
                   prefix_slots=2, probe=ReadPatch())
    two = [user(str(k), "ab", candidates=[1]) for k in range(2)]
    with pytest.raises(Fatal, match="batch_size=1"):
        be.complete_batch(two)
    tgt = user("t", "ab", target="c")
    assert "teacher-force" in be.refusal(tgt)
    with pytest.raises(Fatal, match="teacher-force"):
        be.complete_batch([tgt])


def test_a23_a_raising_forward_leaves_the_slot_empty():
    """A23. The model raises on its 2nd call, which consumed the only slot.
    Expected: the slot is empty afterwards, the next call reuses nothing
    (n_prefix_reused 0), and its candidate log-probs equal a direct full
    pass of the model on the same ids (no backend involved).  Tolerance 0:
    with nothing reused the backend makes the same single forward call on
    the same ids."""

    class FailsOnce(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.inner, self.n = inner, 0

        def forward(self, **kw):
            self.n += 1
            if self.n == 2:
                raise RuntimeError("device lost")
            return self.inner(**kw)

    tok = CharTok()
    model = tiny_qwen(len(tok.vocab), n_layers=2, width=32)
    be = HFBackend(FailsOnce(model), tok, device="cpu", prefix_cache=True,
                   prefix_slots=1)
    cands = [tok.index[c] for c in "xyz"]
    first = be.complete(user("0", "log: abc", candidates=cands))
    assert first.ok and be._slots[0] is not None
    with pytest.raises(RuntimeError, match="device lost"):
        be.complete_batch([user("1", "log: abc q", candidates=cands)])
    assert be._slots == [None]
    text = "log: abc q"
    r = be.complete(user("2", text, candidates=cands))
    assert r.ok, r.error
    assert r.meta["n_prefix_reused"] == 0
    with torch.no_grad():
        ref = model(torch.tensor([tok(text)["input_ids"]])).logits[0, -1]
    ref = ref.float().log_softmax(-1)
    assert_same(r.meta["cand_logp"], [float(ref[c]) for c in cands])
