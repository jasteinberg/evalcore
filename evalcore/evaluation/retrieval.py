r"""Ranking metrics for retrieval, per query, with cluster-bootstrap intervals.

A query q has a judged relevance g_q(d) >= 0 for each document d (binary:
g in {0, 1}; the relevant set R_q = {d : g_q(d) > 0}).  A retriever returns
a ranked list d_1, d_2, ... .  Per query:

    recall@k = |R_q n {d_1..d_k}| / |R_q|
    hit@k    = 1[R_q n {d_1..d_k} != {}]
    RR       = 1 / (rank of the first relevant document), 0 if none is
               returned -- so MRR depends on how deep the list goes; it is
               MRR at the returned depth, and rows must share that depth to
               be compared.
    nDCG@k   = DCG@k / IDCG@k,  DCG@k = sum_{i <= k} g(d_i) / log2(i + 1),

with IDCG@k the DCG@k of the ideal order (judged gains sorted descending),
so nDCG@k = 1 exactly when the top k are the best k available.  The gain is
linear in g; for binary judgments that is the same as the 2^g - 1 form.

recall@k cannot reach 1 when |R_q| > k (a two-hop question with both
documents relevant has recall@1 <= 1/2); hit@k can.  Report the one that
asks the question you mean.

Every metric is a mean over queries, and queries derived from one source
(paraphrases, hops of one question) are not independent: the intervals
come from `stats.summarize`, which resamples clusters (stats.py, eq. 1).
A query with no relevant document has no defined value and is returned as
NaN, which `summarize` counts in `n_excluded` rather than averaging in.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Collection, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from ..core.backends import Response
from ..core.spec import Unit
from .stats import summarize

__all__ = ["rank_metrics", "retrieval_summary", "score_ranked"]


def _gains(relevant: Mapping[str, float] | Collection[str]) -> dict[str, float]:
    if isinstance(relevant, Mapping):
        return {d: float(g) for d, g in relevant.items() if g > 0}
    return dict.fromkeys(relevant, 1.0)


def rank_metrics(ranked: Sequence[str],
                 relevant: Mapping[str, float] | Collection[str],
                 ks: Sequence[int] = (1, 5, 10)) -> dict[str, float]:
    """recall@k, hit@k and nDCG@k for each k, and RR, for one query.
    `relevant` is a set of ids (binary) or {id: gain}.  A repeated id in
    `ranked` counts at its first position only."""
    gains = _gains(relevant)
    names = [f"{m}@{k}" for k in ks for m in ("recall", "hit", "ndcg")]
    if not gains:
        return dict.fromkeys([*names, "rr"], math.nan)
    order = list(dict.fromkeys(ranked))        # first occurrence of each id
    g = np.array([gains.get(d, 0.0) for d in order])
    disc = 1.0 / np.log2(np.arange(2, len(g) + 2))
    ideal = np.sort(np.fromiter(gains.values(), float))[::-1]
    idisc = 1.0 / np.log2(np.arange(2, len(ideal) + 2))
    out: dict[str, float] = {}
    for k in ks:
        n_rel = int((g[:k] > 0).sum())
        out[f"recall@{k}"] = n_rel / len(gains)
        out[f"hit@{k}"] = float(n_rel > 0)
        out[f"ndcg@{k}"] = float((g[:k] * disc[:k]).sum()
                                 / (ideal[:k] * idisc[:k]).sum())
    first = np.flatnonzero(g > 0)
    out["rr"] = 1.0 / (first[0] + 1) if first.size else 0.0
    return out


def score_ranked(ks: Sequence[int] = (1, 5, 10), gold_field: str = "gold_docs",
                 ranked_meta: str = "retrieved"
                 ) -> Callable[[Unit, Response], dict[str, Any]]:
    """A runner `score` function for retrieval rows: the ranked ids from
    `response.meta[ranked_meta]` (what RetrieverBackend returns) against
    the item's `payload[gold_field]`, a list of ids or {id: gain}."""
    def score(u: Unit, r: Response) -> dict[str, Any]:
        return {**rank_metrics(r.meta[ranked_meta], u.item.payload[gold_field],
                               ks),
                "depth": len(r.meta[ranked_meta])}
    return score


def retrieval_summary(df: pd.DataFrame, by: Sequence[str],
                      metrics: Sequence[str] | None = None,
                      **kw: Any) -> pd.DataFrame:
    """`stats.summarize` for each metric column (default: every recall@,
    hit@, ndcg@ column and rr), stacked with a `metric` column.  Keyword
    arguments pass through (cluster, n_boot, method, ...).  Pass every row:
    only `ok` rows carry scores, and the rest are counted in n_excluded."""
    if metrics is None:
        metrics = [c for c in df.columns
                   if c == "rr" or c.split("@")[0] in ("recall", "hit", "ndcg")]
    if ("rr" in metrics and "depth" in df.columns and len(df)
            and df.groupby(list(by))["depth"].nunique().max() > 1):
        raise ValueError("rows of one cell retrieved to different depths; "
                         "RR (and so MRR) is only comparable at one depth")
    return pd.concat([summarize(df, m, by, **kw).assign(metric=m)
                      for m in metrics], ignore_index=True)
