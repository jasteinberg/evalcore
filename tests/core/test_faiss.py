"""Approximate indexes, measured against the exact one.  Run in a process
without torch: see test_search.test_faiss_tests_pass_in_a_process_without_torch."""

from __future__ import annotations

import sys

import numpy as np
import pytest

from evalcore.core.embed import FunctionEmbedder
from evalcore.core.search import DenseRetriever, ExactIndex, FaissIndex, ann_recall

if "torch" in sys.modules:
    pytest.skip("torch is loaded; run in a fresh process", allow_module_level=True)
faiss = pytest.importorskip("faiss")


def clustered(n=2000, d=32, seed=0):
    rng = np.random.default_rng(seed)
    centres = rng.normal(size=(40, d))
    return centres[rng.integers(0, 40, n)] + 0.3 * rng.normal(size=(n, d))


def test_hnsw_recall_is_measured_against_exact():
    V = clustered()
    exact, hnsw = ExactIndex(), FaissIndex("hnsw")
    exact.build(V)
    hnsw.build(V)
    r = ann_recall(hnsw, exact, clustered(100, seed=1), 10)
    assert r.shape == (100,) and r.mean() > 0.9


def test_a_starved_ivf_index_is_measured_and_warned_about():
    """One probe out of ~180 lists: most true neighbours sit in other lists."""
    V = clustered()
    docs = {f"d{i:04d}": f"doc {i}" for i in range(len(V))}
    emb = FunctionEmbedder(lambda texts, role: V[[int(t.split()[1])
                                                  for t in texts]],
                           {"model": "fixed"})
    with pytest.warns(RuntimeWarning, match="approximate index recall@10"):
        r = DenseRetriever(docs, emb, FaissIndex("ivf", nlist=180, nprobe=1),
                           ann_sample=100)
    assert r.ann_recall["n"] == 100 and r.ann_recall["mean"] < 0.95
    assert r.ann_recall["queries"] == "documents"


