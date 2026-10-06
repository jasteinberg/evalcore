"""Ranking metrics against hand arithmetic (values in the docstrings)."""

from __future__ import annotations

import math

import pandas as pd
import pytest

from evalcore.evaluation.retrieval import rank_metrics, retrieval_summary

L2, L3 = 1.0, math.log2(3)


def test_binary_relevance_by_hand():
    """R = {a, b, c, z}, z never returned; ranked a x b y c.
    recall@1,3,5 = 1/4, 2/4, 3/4.  nDCG@3: DCG = 1/log2 2 + 1/log2 4 = 1.5,
    IDCG = 1 + 1/log2 3 + 1/2, so 1.5 / (1.5 + 1/log2 3)."""
    m = rank_metrics(list("axbyc"), {"a", "b", "c", "z"}, ks=(1, 3, 5))
    assert [m["recall@1"], m["recall@3"], m["recall@5"]] == [0.25, 0.5, 0.75]
    assert m["hit@1"] == 1.0 and m["rr"] == 1.0
    assert m["ndcg@3"] == pytest.approx(1.5 / (1.5 + 1 / L3))


def test_reciprocal_rank_and_a_late_hit():
    """ranked x y a, R = {a}: RR = 1/3, hit@1 = 0, nDCG@5 = (1/log2 4) / 1."""
    m = rank_metrics(list("xya"), {"a"}, ks=(1, 5))
    assert m["rr"] == pytest.approx(1 / 3)
    assert m["hit@1"] == 0.0 and m["hit@5"] == 1.0
    assert m["ndcg@5"] == pytest.approx(0.5)


def test_graded_relevance_by_hand():
    """g(a) = 2, g(b) = 1, ranked b a: DCG = 1 + 2/log2 3, IDCG = 2 + 1/log2 3."""
    m = rank_metrics(["b", "a"], {"a": 2, "b": 1}, ks=(2,))
    assert m["ndcg@2"] == pytest.approx((1 + 2 / L3) / (2 + 1 / L3))


def test_the_ideal_order_scores_one_and_recall_is_capped_by_k():
    m = rank_metrics(["a", "b", "x"], {"a", "b"}, ks=(1, 2))
    assert m["ndcg@1"] == m["ndcg@2"] == 1.0
    assert m["recall@1"] == 0.5                 # |R| = 2 > k = 1
    assert m["hit@1"] == 1.0


def test_a_repeated_id_counts_once():
    assert rank_metrics(["a", "a", "b"], {"b"}, ks=(2,))["rr"] == 0.5


def test_no_relevant_documents_is_undefined_not_zero():
    m = rank_metrics(["a"], set(), ks=(1,))
    assert all(math.isnan(v) for v in m.values())


def test_summary_refuses_mrr_across_depths_within_a_cell():
    df = pd.DataFrame({"cell": ["c", "c"], "cluster": ["q1", "q2"],
                       "status": "ok", "rr": [1.0, 0.5], "depth": [5, 10]})
    with pytest.raises(ValueError, match="different depths"):
        retrieval_summary(df, ["cell"], metrics=["rr"])
