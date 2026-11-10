r"""Okapi BM25: lexical retrieval, exact, with no model.

BM25.  For query terms t and document d of length |d| (average length
avgdl, N documents, n_t containing t, f_{t,d} occurrences),

    score(q, d) = sum_{t in q} idf_t f_{t,d} (k1 + 1)
                                / (f_{t,d} + k1 (1 - b + b |d| / avgdl)),
    idf_t = ln(1 + (N - n_t + 0.5) / (n_t + 0.5)),

the non-negative idf (Lucene's), so a term in most documents adds little
rather than subtracting.  Terms are lowercase alphanumeric runs, with
no stemming.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np

from .base import _WORD, Hit, _corpus, _top


class BM25:
    """Okapi BM25 (formula in the module docstring), exact."""

    uses_repeat = False

    def __init__(self, docs: Mapping[str, str], k1: float = 1.2,
                 b: float = 0.75) -> None:
        self.docs = dict(docs)
        self.ids, self.corpus = _corpus(self.docs)
        self.k1, self.b = k1, b
        tfs = [Counter(_WORD.findall(self.docs[d].lower())) for d in self.ids]
        lens = np.array([sum(t.values()) for t in tfs], float)
        avgdl = lens.mean() if len(lens) and lens.mean() > 0 else 1.0
        norm = k1 * (1 - b + b * lens / avgdl)
        post: dict[str, tuple[list[int], list[int]]] = {}
        for j, tf in enumerate(tfs):
            for t, f in tf.items():
                post.setdefault(t, ([], []))[0].append(j)
                post[t][1].append(f)
        n = len(self.ids)
        # term -> (doc indices, per-doc weight). The weight is the whole
        # summand, so a query is a sum of precomputed vectors
        self.postings: dict[str, tuple[np.ndarray, np.ndarray]] = {}
        for t, (js, fs) in post.items():
            ja, fa = np.array(js), np.array(fs, float)
            idf = math.log(1 + (n - len(ja) + 0.5) / (len(ja) + 0.5))
            self.postings[t] = (ja, idf * fa * (k1 + 1) / (fa + norm[ja]))

    def identity(self) -> dict[str, Any]:
        return {"retriever": "bm25", "corpus": self.corpus, "k1": self.k1,
                "b": self.b}

    def scores(self, query: str) -> np.ndarray:
        s = np.zeros(len(self.ids))
        for t in _WORD.findall(query.lower()):     # a repeated term counts again
            if t in self.postings:
                js, w = self.postings[t]
                s[js] += w
        return s

    def score_all(self, queries: Sequence[str]
                  ) -> tuple[list[str], np.ndarray]:
        """Every document's score, zeros included (columns in `ids` order)."""
        return list(self.ids), np.array([self.scores(q) for q in queries],
                                        float).reshape(len(queries), len(self.ids))

    def retrieve(self, queries: Sequence[str], k: int,
                 repeat: int = 0) -> list[list[Hit]]:
        out = []
        for q in queries:
            s = self.scores(q)
            idx, sc = _top(s[None, :], k)
            out.append([(self.ids[j], float(v)) for j, v in zip(idx[0], sc[0],
                                                                strict=True)
                        if v > 0])                # no shared term: no result
        return out
