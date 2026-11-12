r"""Dense retrieval: an Embedder plus an exact or approximate Index.

Indexes.  `ExactIndex` is brute-force cosine (or dot product) in numpy:
exact, dependency-free, and fast enough to roughly 1e5-1e6 passages.
`FaissIndex` (HNSW or IVF, `pip install faiss-cpu`) is approximate, so a
DenseRetriever over it MEASURES what the approximation costs: recall@k of
the ANN results against the exact ones on a sample of queries,

    ann_recall@k = < |ANN_k(q) n EXACT_k(q)| / k >_q ,

stored in `ann_recall` and warned about below `min_ann_recall`.  A weaker
RAG result then shows up as a number, not as a silently worse retriever.
With documents as the sample queries (the default when no queries are
given) each query finds itself, which flatters the figure by up to 1/k.
Pass real queries for the honest one.
"""

from __future__ import annotations

import importlib.util
import math
import sys
import warnings
from abc import ABC, abstractmethod
from collections.abc import Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from ..embed import Embedder, EmbeddingCache
from .base import Hit, _corpus, _top


class Index(ABC):
    exact: bool = True
    metric: str = "cosine"

    @abstractmethod
    def identity(self) -> dict[str, Any]: ...

    @abstractmethod
    def build(self, vectors: np.ndarray) -> None: ...

    @abstractmethod
    def search(self, queries: np.ndarray, k: int
               ) -> tuple[np.ndarray, np.ndarray]:
        """(n, k) row indices and scores, with -1 where fewer than k exist."""

    @staticmethod
    def _check_metric(metric: str) -> None:
        if metric not in ("cosine", "dot"):
            raise ValueError("metric must be cosine or dot")

    def _prep(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, np.float32)
        if self.metric == "cosine":
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return x


class ExactIndex(Index):
    """Brute force: every query against every vector, in query chunks."""

    def __init__(self, metric: str = "cosine", chunk: int = 1024) -> None:
        self._check_metric(metric)
        self.metric, self.chunk = metric, chunk
        self.V = np.zeros((0, 0), np.float32)

    def identity(self) -> dict[str, Any]:
        return {"index": "exact", "metric": self.metric}

    def build(self, vectors: np.ndarray) -> None:
        self.V = self._prep(vectors)

    def _blocks(self, queries: np.ndarray) -> Iterator[np.ndarray]:
        """The full score matrix, one chunk of query rows at a time."""
        Q = self._prep(queries)
        for a in range(0, len(Q), self.chunk):
            yield Q[a:a + self.chunk] @ self.V.T

    def search(self, queries: np.ndarray, k: int
               ) -> tuple[np.ndarray, np.ndarray]:
        parts = [_top(s, k) for s in self._blocks(queries)]
        return (np.concatenate([p[0] for p in parts]),
                np.concatenate([p[1] for p in parts]))

    def scores(self, queries: np.ndarray) -> np.ndarray:
        """(n_queries, n_vectors): every vector's score, as `search` ranks."""
        return np.concatenate(list(self._blocks(queries)))


class FaissIndex(Index):
    """FAISS, imported here only.  kind "hnsw" (m, ef_construction,
    ef_search) or "ivf" (nlist, nprobe), which is trained on the corpus
    itself.  Cosine is inner product on normalised vectors.  An HNSW index
    built on several threads is not guaranteed to be bit-reproducible, and
    the measured ann_recall reports any recall that costs.  On macOS with
    pip wheels, faiss and torch cannot share a process (`omp_clash`), so
    `build` refuses with the reason rather than let the process abort."""

    exact = False

    def __init__(self, kind: str = "hnsw", metric: str = "cosine",
                 m: int = 32, ef_construction: int = 200, ef_search: int = 128,
                 nlist: int | None = None, nprobe: int = 16) -> None:
        if kind not in ("hnsw", "ivf"):
            raise ValueError("kind must be hnsw or ivf")
        self._check_metric(metric)
        self.kind, self.metric = kind, metric
        self.params: dict[str, Any] = (
            {"m": m, "ef_construction": ef_construction, "ef_search": ef_search}
            if kind == "hnsw" else {"nlist": nlist, "nprobe": nprobe})
        self.index: Any = None

    def identity(self) -> dict[str, Any]:
        return {"index": f"faiss-{self.kind}", "metric": self.metric,
                **self.params}

    def build(self, vectors: np.ndarray) -> None:
        clash = omp_clash()
        if clash:
            raise ImportError(clash)
        try:
            import faiss
        except ImportError as exc:
            raise ImportError("FaissIndex needs faiss: pip install faiss-cpu "
                              "(or evalcore[retrieval])") from exc
        V = self._prep(vectors)
        d, ip = V.shape[1], faiss.METRIC_INNER_PRODUCT
        if self.kind == "hnsw":
            p = self.params
            self.index = faiss.IndexHNSWFlat(d, p["m"], ip)
            self.index.hnsw.efConstruction = p["ef_construction"]
            self.index.add(V)
            self.index.hnsw.efSearch = p["ef_search"]
        else:
            nlist = self.params["nlist"] or max(1, int(4 * math.sqrt(len(V))))
            self.index = faiss.IndexIVFFlat(faiss.IndexFlatIP(d), d, nlist, ip)
            self.index.train(V)
            self.index.add(V)
            self.index.nprobe = self.params["nprobe"]

    def search(self, queries: np.ndarray, k: int
               ) -> tuple[np.ndarray, np.ndarray]:
        D, idx = self.index.search(self._prep(queries), k)
        return idx.astype(np.int64), D


def omp_clash() -> str | None:
    """Why importing faiss here would abort the process, or None.

    On macOS the pip wheels of torch and faiss-cpu each bundle their own
    libomp, and a process that loads both is killed by the OpenMP runtime
    (OMP Error #15).  That is an abort, not an exception, so it cannot be
    caught afterwards.  This checks for exactly that pair before faiss is
    imported: torch already loaded, faiss not yet, both carrying a bundled
    libomp."""
    if sys.platform != "darwin" or "faiss" in sys.modules:
        return None
    torch = sys.modules.get("torch")
    spec = importlib.util.find_spec("faiss")
    if torch is None or spec is None or spec.origin is None:
        return None
    torch_omp = Path(str(torch.__file__)).parent / "lib" / "libomp.dylib"
    faiss_omp = Path(spec.origin).parent / ".dylibs" / "libomp.dylib"
    if torch_omp.exists() and faiss_omp.exists():
        return ("torch is loaded, and the pip wheels of torch and faiss-cpu "
                "each bundle libomp: importing faiss now would abort the "
                "process (OMP Error #15).  Build and query the FAISS index in "
                "a process that does not import torch (embed first, cache the "
                "vectors), or install both from conda-forge, which share one "
                "OpenMP runtime.")
    return None


def ann_recall(approx: Index, exact: Index, queries: np.ndarray,
               k: int) -> np.ndarray:
    """Per query, |approx top k n exact top k| / k (module docstring)."""
    a, _ = approx.search(queries, k)
    e, _ = exact.search(queries, k)
    kk = min(k, e.shape[1])
    return np.array([len(set(ar[ar >= 0]) & set(er[:kk])) / kk
                     for ar, er in zip(a, e, strict=True)])


class DenseRetriever:
    """Embed the corpus (cached), index it, retrieve by query embedding.

    With an approximate index, `ann_recall` is measured at build on
    `ann_queries` (or `ann_sample` documents) at depth `ann_k`."""

    uses_repeat = False

    def __init__(self, docs: Mapping[str, str], embedder: Embedder,
                 index: Index | None = None, cache: EmbeddingCache | None = None,
                 ann_queries: Sequence[str] | None = None, ann_sample: int = 200,
                 ann_k: int = 10, min_ann_recall: float = 0.95,
                 seed: int = 0) -> None:
        self.docs = dict(docs)
        self.ids, self.corpus = _corpus(self.docs)
        self.embedder, self.cache = embedder, cache
        self.index = index or ExactIndex()
        V = embedder.embed([self.docs[d] for d in self.ids], "document", cache)
        self.index.build(V)
        self.ann_recall: dict[str, Any] | None = None
        if not self.index.exact:
            exact = ExactIndex(metric=self.index.metric)
            exact.build(V)
            if ann_queries:
                Q, source = embedder.embed(list(ann_queries), "query", cache), "queries"
            else:
                rng = np.random.default_rng(seed)
                pick = rng.choice(len(V), min(ann_sample, len(V)), replace=False)
                Q, source = V[np.sort(pick)], "documents"
            r = ann_recall(self.index, exact, Q, ann_k)
            self.ann_recall = {"k": ann_k, "mean": float(r.mean()),
                               "min": float(r.min()), "n": len(r),
                               "queries": source}
            if r.mean() < min_ann_recall:
                warnings.warn(f"approximate index recall@{ann_k} vs exact is "
                              f"{r.mean():.3f} (< {min_ann_recall}) on {len(r)} "
                              f"{source}: retrieval is measurably weaker than "
                              f"the exact search", RuntimeWarning, stacklevel=2)

    def identity(self) -> dict[str, Any]:
        return {"retriever": "dense", "corpus": self.corpus,
                "embedder": self.embedder.identity(),
                "index": self.index.identity()}

    def retrieve(self, queries: Sequence[str], k: int,
                 repeat: int = 0) -> list[list[Hit]]:
        Q = self.embedder.embed(list(queries), "query", self.cache)
        idx, sc = self.index.search(Q, k)
        return [[(self.ids[j], float(v)) for j, v in zip(ir, sr, strict=True)
                 if j >= 0] for ir, sr in zip(idx, sc, strict=True)]

    def score_all(self, queries: Sequence[str]
                  ) -> tuple[list[str], np.ndarray]:
        """Every document's score.  Exact indexes only: an approximate one
        never computes most of them, and a substitute would change the
        population being sampled."""
        if not (isinstance(self.index, ExactIndex) and self.index.exact):
            raise TypeError(f"score_all needs an ExactIndex, and "
                            f"{type(self.index).__name__} cannot give every "
                            f"document's score exactly")
        Q = self.embedder.embed(list(queries), "query", self.cache)
        return list(self.ids), self.index.scores(Q)
