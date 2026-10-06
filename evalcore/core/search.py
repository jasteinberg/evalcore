r"""Retrieval over a local corpus: lexical, dense, exact and approximate.

A `Retriever` maps queries to ranked (doc id, score) lists and has an
`identity()`.  Three are here, all offline:

    BM25             lexical, exact, no model -- the default search tool
    DenseRetriever   an Embedder plus an Index
    Reranker         any retriever's top `depth`, rescored by a PairScorer
                     (a function, or an HF cross-encoder)

Indexes.  `ExactIndex` is brute-force cosine (or dot product) in numpy:
exact, dependency-free, and fast enough to roughly 1e5-1e6 passages.
`FaissIndex` (HNSW or IVF; `pip install faiss-cpu`) is approximate, so a
DenseRetriever over it MEASURES what the approximation costs: recall@k of
the ANN results against the exact ones on a sample of queries,

    ann_recall@k = < |ANN_k(q) n EXACT_k(q)| / k >_q ,

stored in `ann_recall` and warned about below `min_ann_recall`.  A weaker
RAG result then shows up as a number, not as a silently worse retriever.
With documents as the sample queries (the default when no queries are
given) each query finds itself, which flatters the figure by up to 1/k;
pass real queries for the honest one.

Ties are broken by document id everywhere, so a ranking is a function of
the corpus and the query alone.

BM25.  For query terms t and document d of length |d| (average length
avgdl, N documents, n_t containing t, f_{t,d} occurrences),

    score(q, d) = sum_{t in q} idf_t f_{t,d} (k1 + 1)
                                / (f_{t,d} + k1 (1 - b + b |d| / avgdl)),
    idf_t = ln(1 + (N - n_t + 0.5) / (n_t + 0.5)),

the non-negative idf (Lucene's), so a term in most documents adds little
rather than subtracting.  Terms are lowercase alphanumeric runs; there is
no stemming.

`search_tool(retriever, docs)` turns any retriever into the agent loop's
`search` tool, and `RetrieverBackend` runs a retriever through the runner
(cache, resume, rows), so evaluation/retrieval.py can score it like any
other model.
"""

from __future__ import annotations

import importlib.util
import math
import re
import sys
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .backends.base import Backend, Request, Response
from .embed import Embedder, EmbeddingCache
from .spec import digest
from .tools import Tool, ToolInputError

__all__ = ["BM25", "CorpusSearch", "DenseRetriever", "ExactIndex", "FaissIndex",
           "FunctionScorer", "HFCrossEncoder", "Hit", "Index", "PairScorer",
           "Reranker", "Retriever", "RetrieverBackend", "ann_recall", "omp_clash",
           "search_tool"]

Hit = tuple[str, float]
_WORD = re.compile(r"[a-z0-9]+")


class Retriever(Protocol):
    def identity(self) -> dict[str, Any]: ...
    def retrieve(self, queries: Sequence[str], k: int) -> list[list[Hit]]: ...


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


# --- lexical --------------------------------------------------------------------

class BM25:
    """Okapi BM25 (formula in the module docstring), exact."""

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
        # term -> (doc indices, per-doc weight); the weight is the whole
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

    def retrieve(self, queries: Sequence[str], k: int) -> list[list[Hit]]:
        out = []
        for q in queries:
            s = self.scores(q)
            idx, sc = _top(s[None, :], k)
            out.append([(self.ids[j], float(v)) for j, v in zip(idx[0], sc[0],
                                                                strict=True)
                        if v > 0])                # no shared term: no result
        return out


# --- dense ------------------------------------------------------------------------

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
        """(n, k) row indices and scores; -1 where fewer than k exist."""

    def _prep(self, x: np.ndarray) -> np.ndarray:
        x = np.ascontiguousarray(x, np.float32)
        if self.metric == "cosine":
            x = x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)
        return x


class ExactIndex(Index):
    """Brute force: every query against every vector, in query chunks."""

    def __init__(self, metric: str = "cosine", chunk: int = 1024) -> None:
        if metric not in ("cosine", "dot"):
            raise ValueError("metric must be cosine or dot")
        self.metric, self.chunk = metric, chunk
        self.V = np.zeros((0, 0), np.float32)

    def identity(self) -> dict[str, Any]:
        return {"index": "exact", "metric": self.metric}

    def build(self, vectors: np.ndarray) -> None:
        self.V = self._prep(vectors)

    def search(self, queries: np.ndarray, k: int
               ) -> tuple[np.ndarray, np.ndarray]:
        Q = self._prep(queries)
        parts = [_top(Q[a:a + self.chunk] @ self.V.T, k)
                 for a in range(0, len(Q), self.chunk)]
        return (np.concatenate([p[0] for p in parts]),
                np.concatenate([p[1] for p in parts]))


class FaissIndex(Index):
    """FAISS, imported here only.  kind "hnsw" (m, ef_construction,
    ef_search) or "ivf" (nlist, nprobe; trained on the corpus itself).
    Cosine is inner product on normalised vectors.  HNSW built on several
    threads need not be bit-reproducible; the measured ann_recall is the
    guard.  On macOS with pip wheels, faiss and torch cannot share a
    process (`omp_clash`); `build` refuses with the reason rather than let
    the process abort."""

    exact = False

    def __init__(self, kind: str = "hnsw", metric: str = "cosine",
                 m: int = 32, ef_construction: int = 200, ef_search: int = 128,
                 nlist: int | None = None, nprobe: int = 16) -> None:
        if kind not in ("hnsw", "ivf"):
            raise ValueError("kind must be hnsw or ivf")
        if metric not in ("cosine", "dot"):
            raise ValueError("metric must be cosine or dot")
        self.kind, self.metric = kind, metric
        self.params: dict[str, Any] = ({"m": m, "ef_construction": ef_construction,
                        "ef_search": ef_search} if kind == "hnsw"
                       else {"nlist": nlist, "nprobe": nprobe})
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
    (OMP Error #15) -- an abort, not an exception, so it cannot be caught
    afterwards.  This checks for exactly that pair before faiss is imported:
    torch already loaded, faiss not yet, both carrying a bundled libomp."""
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

    def retrieve(self, queries: Sequence[str], k: int) -> list[list[Hit]]:
        Q = self.embedder.embed(list(queries), "query", self.cache)
        idx, sc = self.index.search(Q, k)
        return [[(self.ids[j], float(v)) for j, v in zip(ir, sr, strict=True)
                 if j >= 0] for ir, sr in zip(idx, sc, strict=True)]


# --- reranking ------------------------------------------------------------------

class PairScorer(Protocol):
    """Relevance of each document to one query, higher is better."""

    def identity(self) -> dict[str, Any]: ...
    def score(self, query: str, docs: Sequence[str]) -> np.ndarray: ...


class FunctionScorer:
    """`fn(query, docs) -> scores`, with a required identity (as for
    FunctionEmbedder)."""

    def __init__(self, fn: Any, identity: Mapping[str, Any]) -> None:
        self.fn, self._identity = fn, dict(identity)

    def identity(self) -> dict[str, Any]:
        return {"scorer": "function", **self._identity}

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        return np.asarray(self.fn(query, list(docs)), float)


class HFCrossEncoder:
    """A sequence-classification model reading (query, document) as one
    pair, with a single relevance logit (`num_labels == 1`, as MS MARCO
    cross-encoders and bge-rerankers are).  A model with more labels is
    refused rather than guessing which logit means "relevant".  Pairs
    longer than `max_length` tokens lose the end of the document."""

    def __init__(self, model: Any, tokenizer: Any, max_length: int = 512,
                 batch_size: int = 32, device: Any = None) -> None:
        n = getattr(getattr(model, "config", None), "num_labels", 1)
        if n != 1:
            raise ValueError(f"cross-encoder must have one output logit, "
                             f"this one has num_labels={n}")
        self.model, self.tok = model, tokenizer
        self.max_length, self.batch_size, self.device = max_length, batch_size, device

    def identity(self) -> dict[str, Any]:
        cfg = getattr(self.model, "config", None)
        return {"scorer": "hf-cross-encoder",
                "model": getattr(self.model, "name_or_path", None),
                "commit": getattr(cfg, "_commit_hash", None),
                "max_length": self.max_length}

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        import torch
        dev = self.device or next(self.model.parameters()).device
        out = []
        for a in range(0, len(docs), self.batch_size):
            part = list(docs[a:a + self.batch_size])
            enc = self.tok([query] * len(part), part, padding=True,
                           truncation="only_second", max_length=self.max_length,
                           return_tensors="pt")
            enc = {k: v.to(dev) for k, v in enc.items()}
            with torch.no_grad():
                logits = self.model(**enc).logits
            out.append(logits.reshape(len(part)).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


class Reranker:
    """Two stages: `first` retrieves `depth` candidates, `scorer` rescores
    them against the query, and the top k by (score desc, id asc) are
    returned.  A document the first stage misses at `depth` cannot be
    recovered, so recall@depth of the first stage bounds the reranker's
    recall at any k -- evaluate the first stage at `depth` to see that
    ceiling."""

    def __init__(self, first: Retriever, scorer: PairScorer,
                 docs: Mapping[str, str], depth: int) -> None:
        self.first, self.scorer, self.depth = first, scorer, depth
        self.docs = dict(docs)

    def identity(self) -> dict[str, Any]:
        return {"retriever": "rerank", "first": self.first.identity(),
                "scorer": self.scorer.identity(), "depth": self.depth}

    def retrieve(self, queries: Sequence[str], k: int) -> list[list[Hit]]:
        out: list[list[Hit]] = []
        for q, cands in zip(queries, self.first.retrieve(queries, self.depth),
                            strict=True):
            ids = [d for d, _ in cands]
            if not ids:
                out.append([])
                continue
            sc = self.scorer.score(q, [self.docs[d] for d in ids])
            order = sorted(range(len(ids)), key=lambda i: (-sc[i], ids[i]))[:k]
            out.append([(ids[i], float(sc[i])) for i in order])
        return out


# --- as a tool, and as a backend ------------------------------------------------------

def search_tool(retriever: Retriever, docs: Mapping[str, str], k: int = 3,
                name: str = "search") -> Tool:
    """The retriever as the agent loop's search tool: the top k documents,
    each as "[id] text".  The tool version is a digest of the retriever's
    identity and k, so a changed corpus or retriever re-runs its rows."""
    def run(args: Mapping[str, Any]) -> str:
        q = str(args["query"])
        if not _WORD.search(q.lower()):
            raise ToolInputError(f"{name}: empty query")
        hits = retriever.retrieve([q], k)[0]
        if not hits:
            return "No results."
        return "\n\n".join(f"[{d}] {docs[d]}" for d, _ in hits)
    ident = retriever.identity()
    return Tool(name, "Search the document collection. Returns the best "
                "matching documents with their ids.",
                {"type": "object",
                 "properties": {"query": {"type": "string",
                                          "description": "search terms"}},
                 "required": ["query"]},
                run, version=f"{ident.get('retriever', 'search')}-"
                             f"{digest({'retriever': ident, 'k': k})}")


class CorpusSearch:
    """A retriever over a corpus with its agent tool: `.docs`, `.retriever`
    (BM25 unless given), `.tool`."""

    def __init__(self, docs: Mapping[str, str], retriever: Retriever | None = None,
                 k: int = 3, name: str = "search") -> None:
        self.docs, self.k = dict(docs), k
        self.retriever = retriever if retriever is not None else BM25(self.docs)
        self.tool = search_tool(self.retriever, self.docs, k, name)


class RetrieverBackend(Backend):
    """A retriever behind the Backend contract: the query is the last user
    message, the response text the ranked ids one per line, and meta holds
    `retrieved` (ids) and `scores`.  Score rows with
    evaluation.retrieval.score_ranked."""

    name = "retriever"

    def __init__(self, retriever: Retriever, k: int = 10) -> None:
        self.retriever, self.k = retriever, k
        self._identity = {**self._base_identity(),
                          "retriever": retriever.identity(), "k": k}

    def identity(self) -> dict[str, Any]:
        return self._identity                       # computed once: a corpus digest

    @staticmethod
    def _query(req: Request) -> str:
        return [m for m in req.messages if m.get("role") == "user"][-1]["content"]

    def refusal(self, req: Request) -> str | None:
        users = [m for m in req.messages if m.get("role") == "user"]
        if not users or not isinstance(users[-1].get("content"), str):
            return ("RetrieverBackend needs a last user message with text "
                    "content to use as the query")
        return super().refusal(req)

    def complete(self, req: Request) -> Response:
        return self.complete_batch([req])[0]

    def complete_batch(self, reqs: Sequence[Request]) -> list[Response]:
        hits = self.retriever.retrieve([self._query(r) for r in reqs], self.k)
        return [Response(r.unit_id, text="\n".join(d for d, _ in h),
                         meta={"retrieved": [d for d, _ in h],
                               "scores": [s for _, s in h]})
                for r, h in zip(reqs, hits, strict=True)]
