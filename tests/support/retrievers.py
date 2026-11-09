"""A base retriever with fixed, query-independent scores, so a test's
expected rankings can be written down by hand."""

from __future__ import annotations

import numpy as np


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
