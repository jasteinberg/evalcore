r"""Shared by every retriever here: the Retriever and ScoredRetriever
protocols, the hit type, and the ranking helpers.

Ties are broken by document id everywhere, so a ranking is a function of
the corpus and the query alone.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from ..spec import digest

Hit = tuple[str, float]
_WORD = re.compile(r"[a-z0-9]+")


class Retriever(Protocol):
    def identity(self) -> dict[str, Any]: ...
    def retrieve(self, queries: Sequence[str], k: int,
                 repeat: int = 0) -> list[list[Hit]]:
        """Top k per query.  `repeat` is the unit's repeat index: a sampled
        retriever seeds its draw with it, a deterministic one ignores it."""
        ...


class ScoredRetriever(Retriever, Protocol):
    def score_all(self, queries: Sequence[str]
                  ) -> tuple[list[str], np.ndarray]:
        """Doc ids in tie-break order, and the (n_queries, n_docs) matrix of
        every document's score against every query, columns in that order."""
        ...


def _corpus(docs: Mapping[str, str]) -> tuple[list[str], str]:
    """Ids in sorted order (the tie-break) and a digest of the corpus, so an
    identity never has to carry the documents themselves."""
    ids = sorted(docs)
    return ids, digest({d: docs[d] for d in ids})


def _top(scores: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
    """Per row, the k best columns by (score desc, column asc)."""
    k = min(k, scores.shape[1])
    part = np.argpartition(-scores, k - 1, axis=1)[:, :k]
    # argpartition may cut a tie at the boundary arbitrarily: widen to every
    # column scoring at least the k-th score, then order exactly
    kth = np.take_along_axis(scores, part, 1).min(1, keepdims=True)
    idx = np.empty((len(scores), k), np.int64)
    for r in range(len(scores)):
        cand = np.flatnonzero(scores[r] >= kth[r])
        order = np.lexsort((cand, -scores[r, cand]))[:k]
        idx[r] = cand[order]
    return idx, np.take_along_axis(scores, idx, 1)


def _check_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"k must be >= 1, got {k}")


def _uses_repeat(retriever: Any) -> bool:
    """Whether a retriever's draw depends on the repeat index.  A retriever
    that does not say is deterministic, by the Retriever contract."""
    return bool(getattr(retriever, "uses_repeat", False))


def _all_scores(base: ScoredRetriever, queries: Sequence[str]
                ) -> tuple[list[str], np.ndarray]:
    """`base.score_all`, checked: a non-empty corpus and a finite float64
    (n_queries, n_docs) matrix, the shape its ids promise."""
    ids, S = base.score_all(queries)
    S = np.asarray(S, np.float64)
    if not ids:
        raise ValueError("the base retriever's corpus is empty")
    if S.shape != (len(queries), len(ids)):
        raise ValueError(f"score_all gave shape {S.shape}, expected "
                         f"{(len(queries), len(ids))}")
    if not np.isfinite(S).all():
        raise ValueError("base scores must be finite")
    return list(ids), S
