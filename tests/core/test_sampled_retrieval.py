"""Sampled (Plackett-Luce) retrieval: draw frequencies against the exact
law, the two limits beta = 0 and beta = inf, the seeding rule, and the
full-score interface it rests on.

Statistical tests use a fixed seed, so each is deterministic.  Their
thresholds are set in advance at a false-alarm rate ALPHA = 1e-3 under the
stated law, and are not tuned to the observed values."""

from __future__ import annotations

import io
import itertools
import json
import math
import subprocess
import sys
from collections import Counter

import numpy as np
import pytest
from scipy import stats

from evalcore.core.backends import Request
from evalcore.core.embed import FunctionEmbedder
from evalcore.core.runner import execute, rendered
from evalcore.core.search import (
    BM25,
    DenseRetriever,
    ExactIndex,
    FunctionScorer,
    Reranker,
    RetrieverBackend,
    SampledRetriever,
    search_tool,
)
from evalcore.core.spec import Item, Unit, digest, grid, units

ALPHA = 1e-3
SCORES = {"a": 0.0, "b": 0.5, "c": 1.0, "d": 2.0}


class Fixed:
    """A base retriever whose scores do not depend on the query."""

    def __init__(self, scores: dict[str, float]) -> None:
        self.ids = sorted(scores)
        self.s = np.array([scores[d] for d in self.ids], float)

    def identity(self):
        return {"retriever": "fixed", "scores": dict(zip(self.ids,
                                                         self.s.tolist(),
                                                         strict=True))}

    def score_all(self, queries):
        return list(self.ids), np.tile(self.s, (len(queries), 1))

    def retrieve(self, queries, k, repeat=0):
        order = sorted(range(len(self.ids)),
                       key=lambda j: (-self.s[j], self.ids[j]))[:k]
        return [[(self.ids[j], float(self.s[j])) for j in order]
                for _ in queries]


def _chi2_pvalue(counts: Counter, probs: dict, n: int) -> float:
    obs = np.array([counts[c] for c in probs], float)
    exp = n * np.array(list(probs.values()))
    assert sum(counts.values()) == n and set(counts) <= set(probs)
    assert exp.min() >= 5                 # chi-square approximation holds
    return float(stats.chisquare(obs, exp).pvalue)


def test_ordered_pairs_follow_plackett_luce():
    """k = 2 on four documents: P(i then j) = w_i / W * w_j / (W - w_i),
    w = exp(beta s).  Each draw is a distinct query, so independent."""
    beta, n = 1.3, 20000
    w = {d: math.exp(beta * s) for d, s in SCORES.items()}
    W = sum(w.values())
    probs = {(i, j): w[i] / W * w[j] / (W - w[i])
             for i, j in itertools.permutations(SCORES, 2)}
    assert sum(probs.values()) == pytest.approx(1.0)
    r = SampledRetriever(Fixed(SCORES), beta, seed=0)
    hits = r.retrieve([f"q{t}" for t in range(n)], k=2)
    counts = Counter(tuple(d for d, _ in h) for h in hits)
    assert _chi2_pvalue(counts, probs, n) > ALPHA


def test_beta_zero_draws_uniform_subsets():
    """beta = 0: each of the C(4, 2) = 6 subsets has probability 1/6."""
    n = 12000
    r = SampledRetriever(Fixed(SCORES), 0.0, seed=1)
    hits = r.retrieve([f"q{t}" for t in range(n)], k=2)
    counts = Counter(frozenset(d for d, _ in h) for h in hits)
    probs = {frozenset(p): 1 / math.comb(4, 2)
             for p in itertools.combinations(SCORES, 2)}
    assert _chi2_pvalue(counts, probs, n) > ALPHA


def test_noise_is_standard_gumbel():
    """Pooled g_i against the Gumbel(0, 1) CDF (KS), plus its mean, the
    Euler-Mascheroni constant, within 4 standard errors (sd pi / sqrt 6)."""
    r = SampledRetriever(Fixed(SCORES), 1.0, seed=2)
    g = np.concatenate([d.noise for d in r.draw([f"q{t}" for t in range(2500)],
                                                k=1)])
    assert stats.kstest(g, stats.gumbel_r.cdf).pvalue > ALPHA
    se = math.pi / math.sqrt(6) / math.sqrt(len(g))
    assert abs(g.mean() - np.euler_gamma) < 4 * se


def test_beta_inf_is_the_base_ranking_with_its_tie_rule():
    """Equal scores a = b = 1 order by id; written out by hand."""
    base = Fixed({"a": 1.0, "b": 1.0, "c": 0.5, "d": 2.0})
    r = SampledRetriever(base, math.inf, seed=0)
    want = [("d", 2.0), ("a", 1.0), ("b", 1.0)]
    for rep in range(3):
        assert r.retrieve(["x", "y"], k=3, repeat=rep) == [want, want]
    (dr,) = r.draw(["x"], k=3)
    assert dr.noise is None and dr.perturbed is None
    assert dr.rank.tolist() == [1, 2, 3, 0]          # a, b, c, d


def test_large_beta_on_separated_scores_is_the_base_order():
    """Gaps of 1 at beta = 50: a swap needs a Gumbel difference above 50,
    probability about e^-50 per pair."""
    r = SampledRetriever(Fixed(SCORES), 50.0, seed=3)
    hits = r.retrieve([f"q{t}" for t in range(200)], k=4)
    assert all([d for d, _ in h] == ["d", "c", "b", "a"] for h in hits)


def test_hit_scores_are_base_scores_not_perturbed():
    r = SampledRetriever(Fixed(SCORES), 1.0, seed=4)
    for dr in r.draw([f"q{t}" for t in range(50)], k=3):
        for d, s in dr.hits:
            assert s == SCORES[d]
        assert dr.perturbed is not None and dr.noise is not None
        np.testing.assert_array_equal(dr.perturbed,
                                      1.0 * dr.scores + dr.noise)
        # hits are the documents ranked 0..k-1 by the perturbed score
        top = sorted(range(4), key=lambda j: -dr.perturbed[j])[:3]
        assert [dr.ids[j] for j in top] == [d for d, _ in dr.hits]
        assert [dr.rank[j] for j in top] == [0, 1, 2]


def test_k_larger_than_corpus_returns_every_document():
    r = SampledRetriever(Fixed(SCORES), 1.0, seed=5)
    (h,) = r.retrieve(["q"], k=10)
    assert sorted(d for d, _ in h) == sorted(SCORES)


def test_draw_is_a_function_of_its_key():
    flat = Fixed({f"d{i:02d}": 0.0 for i in range(20)})
    r = SampledRetriever(flat, 1.0, seed=6)
    a = r.retrieve(["x", "y"], k=5, repeat=0)
    # batch and call order do not matter
    assert r.retrieve(["y"], k=5, repeat=0)[0] == a[1]
    assert r.retrieve(["x"], k=5, repeat=0)[0] == a[0]
    # a fresh retriever with the same identity draws the same
    assert SampledRetriever(Fixed({f"d{i:02d}": 0.0 for i in range(20)}),
                            1.0, seed=6).retrieve(["x"], k=5) == [a[0]]
    # repeat, seed and query each change it: P(same ordered 5 of 20)
    # = 1 / (20 19 18 17 16) ~ 5e-7
    assert r.retrieve(["x"], k=5, repeat=1)[0] != a[0]
    assert SampledRetriever(flat, 1.0, seed=7).retrieve(["x"], k=5)[0] != a[0]
    assert a[0] != a[1]


_DOCS = {"d1": "the cat sat", "d2": "the dog sat on the dog mat",
         "d3": "a bird", "d4": "fish swim"}
_FRESH = (
    "import json; from evalcore.core.search import BM25, SampledRetriever; "
    f"docs = {_DOCS!r}; "
    "r = SampledRetriever(BM25(docs), 1.0, seed=11); "
    "print(json.dumps(r.retrieve(['cat sat'], 3, repeat=2)))"
)


def test_same_draw_in_a_fresh_process():
    p = subprocess.run([sys.executable, "-c", _FRESH], capture_output=True,
                       text=True, timeout=120, check=True)
    there = [[tuple(h) for h in hs] for hs in json.loads(p.stdout)]
    here = SampledRetriever(BM25(_DOCS), 1.0, seed=11).retrieve(
        ["cat sat"], 3, repeat=2)
    assert there == here


def test_bm25_zero_score_documents_are_in_the_population():
    """"cat" matches only d1, so d2..d4 score 0.  At beta = 1 and k = 1,
    P(first is not d1) = 3 / (e^{s_1} + 3) > 0.3 (s_1 < 2 here), so in 100
    draws a non-match appears with probability > 1 - 0.7^100.  At
    beta = inf the result is BM25's own: d1 alone."""
    bm = BM25(_DOCS)
    assert 0 < bm.scores("cat")[0] < 2
    firsts = {h[0][0] for h in SampledRetriever(bm, 1.0, seed=12).retrieve(
        [f"cat {'x' * t}" for t in range(100)], k=1)}
    assert firsts & {"d2", "d3", "d4"}
    inf = SampledRetriever(bm, math.inf, seed=12).retrieve(["cat"], k=3)
    assert inf == bm.retrieve(["cat"], k=3)
    assert [d for d, _ in inf[0]] == ["d1"]


def test_bm25_score_all_keeps_zeros_and_matches_by_hand():
    """Columns in id order; only d1 contains "cat", so the rest are 0."""
    ids, S = BM25(_DOCS).score_all(["cat", "zzz"])
    assert ids == ["d1", "d2", "d3", "d4"] and S.shape == (2, 4)
    assert S[0, 0] > 0 and S[0, 1:].tolist() == [0, 0, 0]
    assert S[1].tolist() == [0, 0, 0, 0]


def _onehot_embedder():
    vocab = ["cat", "dog", "bird", "fish", "sat", "mat"]

    def fn(texts, role):
        return np.array([[t.split().count(v) for v in vocab]
                         for t in texts], float)
    return FunctionEmbedder(fn, {"model": "counts"})


def test_dense_score_all_is_cosine_against_every_document():
    """Against a direct cosine of the embedder's own vectors (float32 inputs,
    so a float32 tolerance)."""
    emb = _onehot_embedder()
    dense = DenseRetriever(_DOCS, emb)
    ids, S = dense.score_all(["dog mat", "fish"])
    V = emb.fn([_DOCS[d] for d in sorted(_DOCS)], "document")
    Q = emb.fn(["dog mat", "fish"], "query")
    cos = (Q / np.linalg.norm(Q, axis=1, keepdims=True)) @ (
        V / np.linalg.norm(V, axis=1, keepdims=True)).T
    assert ids == sorted(_DOCS)
    np.testing.assert_allclose(S, cos, rtol=0, atol=1e-6)


def test_dense_score_all_refuses_an_approximate_index():
    class Approx(ExactIndex):
        exact = False

    dense = DenseRetriever(_DOCS, _onehot_embedder(), index=Approx(),
                           ann_sample=4, ann_k=2)
    with pytest.raises(TypeError, match="ExactIndex"):
        dense.score_all(["cat"])


def test_dense_beta_inf_matches_dense_retrieve():
    dense = DenseRetriever(_DOCS, _onehot_embedder())
    r = SampledRetriever(dense, math.inf, seed=0)
    assert r.retrieve(["dog mat"], k=3) == dense.retrieve(["dog mat"], k=3)


@pytest.mark.parametrize("beta", [-0.1, math.nan])
def test_bad_beta_is_refused(beta):
    with pytest.raises(ValueError, match="beta"):
        SampledRetriever(Fixed(SCORES), beta, seed=0)


def test_bad_k_and_seed_are_refused():
    r = SampledRetriever(Fixed(SCORES), 1.0, seed=0)
    with pytest.raises(ValueError, match="k must"):
        r.retrieve(["q"], k=0)
    with pytest.raises(TypeError, match="seed"):
        SampledRetriever(Fixed(SCORES), 1.0, seed=1.5)   # type: ignore[arg-type]


def test_identity_names_every_input_and_is_digestable_at_inf():
    base = Fixed(SCORES)
    ident = SampledRetriever(base, math.inf, seed=3).identity()
    assert ident == {"retriever": "sampled", "base": base.identity(),
                     "beta": "inf", "seed": 3}
    keys = {digest(SampledRetriever(base, b, seed=s).identity())
            for b in (0.0, 1.0, math.inf) for s in (0, 1)}
    assert len(keys) == 6
    tool = search_tool(SampledRetriever(BM25(_DOCS), math.inf, seed=0), _DOCS)
    assert tool.version.startswith("sampled-")


# --- the repeat index, from the runner to the draw -----------------------------

FLAT = {f"d{i:02d}": 0.0 for i in range(20)}


def _render(u: Unit) -> Request:
    return Request(u.id, [{"role": "user", "content": u.item.payload["q"]}])


def test_runner_stamps_the_unit_repeat_and_refuses_a_conflict():
    (u0, u1, u2) = units(grid({"x": [1]}), [Item("i", {"q": "dog"})], repeats=3)
    assert [rendered(_render, u).repeat for u in (u0, u1, u2)] == [0, 1, 2]

    def wrong(u: Unit) -> Request:
        return Request(u.id, [{"role": "user", "content": "q"}], repeat=5)
    with pytest.raises(ValueError, match="repeat=5"):
        rendered(wrong, u1)


@pytest.mark.parametrize("batch_size", [1, 6])
def test_each_repeat_of_a_unit_gets_its_own_draw(tmp_path, batch_size):
    """Six repeats of one query through the runner.  Row r must hold the
    draw keyed by repeat r (the direct call is the specification), whether
    the six share one complete_batch call or not.  With 20 equally scored
    documents and k = 3, two repeats coincide with probability 1/6840 per
    pair, so the six are not all equal."""
    r = SampledRetriever(Fixed(FLAT), 1.0, seed=0)
    us = list(units(grid({"x": [1]}), [Item("i", {"q": "dog"})], repeats=6))
    df = execute(us, RetrieverBackend(r, k=3), _render,
                 lambda u, resp: {"retrieved": resp.meta["retrieved"]},
                 tmp_path / "r.jsonl", workers=1, batch_size=batch_size,
                 stream=io.StringIO())
    got = {int(row.repeat): list(row.retrieved) for row in df.itertuples()}
    want = {rep: [d for d, _ in r.retrieve(["dog"], 3, repeat=rep)[0]]
            for rep in range(6)}
    assert got == want
    assert len({tuple(v) for v in got.values()}) > 1


def test_backend_keeps_request_order_across_mixed_repeats():
    r = SampledRetriever(Fixed(FLAT), 1.0, seed=0)
    reqs = [Request(f"u{i}", [{"role": "user", "content": q}], repeat=rep)
            for i, (q, rep) in enumerate([("a", 1), ("b", 0), ("a", 0),
                                          ("b", 1)])]
    out = RetrieverBackend(r, k=3).complete_batch(reqs)
    for req, resp in zip(reqs, out, strict=True):
        assert resp.unit_id == req.unit_id
        assert resp.meta["retrieved"] == [d for d, _ in r.retrieve(
            [req.messages[0]["content"]], 3, repeat=req.repeat)[0]]


def test_reranker_passes_repeat_to_its_first_stage():
    """A constant scorer keeps the first stage's candidates; the reranker
    then orders them by id, so its output is the sorted first-stage set."""
    first = SampledRetriever(Fixed(FLAT), 1.0, seed=0)
    rr = Reranker(first, FunctionScorer(lambda q, ds: [0.0] * len(ds),
                                        {"fn": "zero"}), FLAT, depth=4)
    for rep in range(3):
        cands = {d for d, _ in first.retrieve(["q"], 4, repeat=rep)[0]}
        got = rr.retrieve(["q"], 4, repeat=rep)[0]
        assert [d for d, _ in got] == sorted(cands)
