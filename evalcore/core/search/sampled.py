r"""Sampled retrieval: a random k-subset drawn from a base retriever's
scores.

Sampled retrieval.  With base scores s_i over the n documents of the
corpus and inverse temperature beta >= 0, put w_i = exp(beta s_i) and
W = sum_i w_i.  SampledRetriever draws k documents without replacement,
each step choosing among those left with probability proportional to w
(the Plackett-Luce law):

    P(i_1, ..., i_k) = prod_{m=1..k} w_{i_m} / (W - sum_{l<m} w_{i_l}) .

It does so by the Gumbel top-k trick: z_i = beta s_i + g_i with g_i iid
standard Gumbel (CDF F(x) = exp(-e^{-x})), and the draw is the k largest
z in descending order.  The first step is the Gumbel-max identity,

    P(argmax z = i) = int dx F'(x - beta s_i) prod_{j != i} F(x - beta s_j)
                    = int dx e^{-(x - beta s_i)} exp(-W e^{-x})  =  w_i / W ,

and the ordered top k of z has exactly the sequential law above (Yellott
1977; Kool, van Hoof and Welling 2019).  Limits: beta = 0 makes every
ordering equally likely, so every k-subset has probability 1 / C(n, k),
and beta -> inf orders by s.  At beta = inf the base retriever's own
ranking is returned unchanged and no noise is drawn.  That includes its
tie rule and anything it leaves out (BM25 drops documents sharing no term
with the query), whereas at finite beta every document is in the
population.  The returned score is the base score s_i, not z_i.

Seeding.  The noise for one query is a pure function of
digest(identity, query, repeat): a fresh generator is seeded from that
digest for every query, and no stream advances across calls.  A draw is
therefore fixed by its key, whatever the call order or batch, which is what
makes it cacheable.  The same (query, repeat) twice gives the same draw.
Independent draws need distinct `repeat`, or a distinct `seed`.  Through
the runner, `repeat` is the unit's: the runner sets Request.repeat and
RetrieverBackend passes it on (a Reranker forwards it to its first stage).
In an agent episode, `search_tool` over a retriever that declares
`uses_repeat` is a context tool (tools.py), so each repeat of a unit
searches with its own repeat index.  The noise comes from numpy's
Generator, so it is bitwise reproducible within a numpy version but not
promised across versions.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np

from ..spec import digest
from .base import Hit, ScoredRetriever, _all_scores, _check_k, _top


@dataclass(frozen=True)
class SampledDraw:
    """One query's draw, for logging.  Arrays run over the whole corpus,
    columns in `ids` order.  `rank[j]` is document j's position in the full
    descending order of `perturbed` (0 first), and the first k positions
    are `hits`.  At beta = inf, `noise` and `perturbed` are None (none is
    drawn), `rank` is the order by (score desc, id asc), and `hits` are
    the base retriever's own."""

    ids: list[str]
    scores: np.ndarray                 # (n,) base scores s_i
    noise: np.ndarray | None           # (n,) g_i ~ Gumbel(0, 1)
    perturbed: np.ndarray | None       # (n,) beta s_i + g_i
    rank: np.ndarray                   # (n,) int
    hits: list[Hit]                    # (doc id, s_i), in draw order


class SampledRetriever:
    """k documents drawn without replacement with probability proportional
    to exp(beta s_i), s_i the base retriever's score (law, limits and
    seeding rule in the module docstring).  `retrieve` satisfies Retriever,
    and `draw` returns the same hits with the noise behind them."""

    uses_repeat = True

    def __init__(self, base: ScoredRetriever, beta: float, seed: int) -> None:
        beta = float(beta)
        if math.isnan(beta) or beta < 0:
            raise ValueError(f"beta must be >= 0 (inf allowed), got {beta}")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError(f"seed must be an int, got {seed!r}")
        self.base, self.beta, self.seed = base, beta, seed
        # canonical JSON refuses inf, so an infinite beta is spelled out
        self._identity = {"retriever": "sampled", "base": base.identity(),
                          "beta": "inf" if math.isinf(beta) else beta,
                          "seed": seed}

    def identity(self) -> dict[str, Any]:
        return self._identity

    def _noise(self, query: str, repeat: int, n: int) -> np.ndarray:
        key = digest({"sampled": self._identity, "query": query,
                      "repeat": repeat})
        return np.random.default_rng(int(key, 16)).gumbel(size=n)

    def draw(self, queries: Sequence[str], k: int,
             repeat: int = 0) -> list[SampledDraw]:
        _check_k(k)
        queries = list(queries)
        if not queries:
            return []
        ids, S = _all_scores(self.base, queries)
        n = len(ids)
        if math.isinf(self.beta):
            order, _ = _top(S, n)
            return [SampledDraw(ids, s, None, None, _ranks(o), h)
                    for s, o, h in zip(S, order, self.base.retrieve(queries, k),
                                       strict=True)]
        out = []
        for q, s in zip(queries, S, strict=True):
            g = self._noise(q, repeat, n)
            z = self.beta * s + g
            o = _top(z[None, :], n)[0][0]
            out.append(SampledDraw(ids, s, g, z, _ranks(o),
                                   [(ids[j], float(s[j])) for j in o[:k]]))
        return out

    def retrieve(self, queries: Sequence[str], k: int,
                 repeat: int = 0) -> list[list[Hit]]:
        return [d.hits for d in self.draw(queries, k, repeat)]


def _ranks(order: np.ndarray) -> np.ndarray:
    """The inverse permutation: rank[order[m]] = m."""
    rank = np.empty(len(order), np.int64)
    rank[order] = np.arange(len(order))
    return rank
