"""Retrieval: BM25 against hand arithmetic, the exact index against a naive
sort, approximate indexes measured against exact, and a retriever run and
scored through the ordinary runner."""

from __future__ import annotations

import math
import os
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from evalcore import Request, grid
from evalcore.core.backends import request_key
from evalcore.core.embed import FunctionEmbedder
from evalcore.core.search import (
    BM25,
    DenseRetriever,
    ExactIndex,
    FaissIndex,
    FunctionScorer,
    HFCrossEncoder,
    Index,
    Reranker,
    RetrieverBackend,
    _top,
    ann_recall,
    omp_clash,
    search_tool,
)
from evalcore.core.spec import digest
from evalcore.evaluation.retrieval import retrieval_summary, score_ranked
from evalcore.tasks.base import Task
from evalcore.tasks.search_qa import make_world, search_qa_items
from evalcore.testing import assert_same

DOCS = {"d1": "the cat sat", "d2": "the dog sat on the dog mat",
        "d3": "a bird"}


def test_bm25_by_hand():
    """N = 3, lengths 3, 7, 2, avgdl = 4.  "dog": n = 1, f = 2 in d2,
    idf = ln(1 + 2.5/1.5); norm(d2) = k1 (1 - b + b 7/4).
    score = idf * 2 (k1 + 1) / (2 + norm)."""
    k1, b = 1.2, 0.75
    s = BM25(DOCS, k1, b).scores("dog")
    norm = k1 * (1 - b + b * 7 / 4)
    want = math.log(1 + 2.5 / 1.5) * 2 * (k1 + 1) / (2 + norm)
    assert s.tolist() == pytest.approx([0.0, want, 0.0])


def test_bm25_ranks_by_score_then_id_and_drops_non_matches():
    r = BM25({"b": "x y", "a": "x y", "c": "z"})
    hits = r.retrieve(["x", "q"], k=3)
    assert [d for d, _ in hits[0]] == ["a", "b"]       # tie -> id order
    assert hits[1] == []


def test_top_k_breaks_ties_at_the_boundary_by_column():
    idx, sc = _top(np.array([[1.0, 2.0, 2.0, 2.0, 0.0]]), 2)
    assert idx.tolist() == [[1, 2]] and sc.tolist() == [[2.0, 2.0]]


def test_exact_index_matches_a_naive_sort():
    rng = np.random.default_rng(0)
    V, Q = rng.normal(size=(300, 16)), rng.normal(size=(20, 16))
    ix = ExactIndex(chunk=7)
    ix.build(V)
    idx, sc = ix.search(Q, 5)
    Vn = V / np.linalg.norm(V, axis=1, keepdims=True)
    S = (Q / np.linalg.norm(Q, axis=1, keepdims=True)) @ Vn.T
    assert idx.tolist() == np.argsort(-S, axis=1)[:, :5].tolist()
    np.testing.assert_allclose(sc, np.sort(S, axis=1)[:, ::-1][:, :5], rtol=1e-5)


def bow(dim=64):
    """A deterministic bag-of-words embedding (hashed words)."""
    def fn(texts, role):
        out = np.zeros((len(texts), dim))
        for i, t in enumerate(texts):
            for w in t.lower().replace("?", " ").replace(".", " ").split():
                out[i, int(digest(w)[:8], 16) % dim] += 1.0
        return out
    return FunctionEmbedder(fn, {"model": "bow", "dim": dim})


def test_dense_retrieval_finds_the_document():
    r = DenseRetriever(DOCS, bow())
    assert r.retrieve(["a bird"], 1)[0][0][0] == "d3"
    assert r.ann_recall is None                          # exact: nothing to measure


VOCAB = {"a": 0, "bird": 1, "the": 2, "cat": 3}


def onehot() -> FunctionEmbedder:
    """Bag of words over a four-word vocabulary, by hand."""
    def fn(texts, role):
        out = np.zeros((len(texts), len(VOCAB)))
        for i, t in enumerate(texts):
            for w in t.split():
                out[i, VOCAB[w]] += 1.0
        return out
    return FunctionEmbedder(fn, {"model": "onehot"})


def test_dense_row_j_is_document_ids_j():
    """B31.  Insertion order (z, a) is the reverse of id order (a, z), so a
    retriever that builds rows in one order and names them in the other
    swaps the two.  "a bird" is the document "z" exactly: cosine 1 with z
    and 0 with "the cat" (no shared word)."""
    r = DenseRetriever({"z": "a bird", "a": "the cat"}, onehot())
    hits = r.retrieve(["a bird", "the cat"], 1)
    assert [h[0][0] for h in hits] == ["z", "a"]
    assert hits[0][0][1] == pytest.approx(1.0, abs=1e-6)   # float32 rounding


def test_retriever_identity_covers_corpus_text_k_and_bm25_params():
    """B33.  Each pair below differs in one thing that changes what is
    retrieved (the depth k, a document's text, BM25's k1 or b), so each
    pair's identities -- and with them the request keys -- must differ."""
    docs = {"d1": "x", "d2": "the cat"}
    req = Request("u", [{"role": "user", "content": "x"}], {})
    pairs = [(RetrieverBackend(BM25(docs), k=5), RetrieverBackend(BM25(docs), k=3)),
             (RetrieverBackend(BM25(docs), k=5),
              RetrieverBackend(BM25({**docs, "d1": "y"}), k=5)),
             (RetrieverBackend(BM25(docs, k1=1.2), k=5),
              RetrieverBackend(BM25(docs, k1=2.0), k=5)),
             (RetrieverBackend(BM25(docs, b=0.75), k=5),
              RetrieverBackend(BM25(docs, b=0.5), k=5))]
    for one, two in pairs:
        assert one.identity() != two.identity()
        assert request_key(req, 0, one) != request_key(req, 0, two)
    dense = DenseRetriever(docs, FunctionEmbedder(
        lambda t, r: np.ones((len(t), 2)), {"model": "const"}))
    other = DenseRetriever({**docs, "d1": "y"}, FunctionEmbedder(
        lambda t, r: np.ones((len(t), 2)), {"model": "const"}))
    assert dense.identity() != other.identity()


# --- approximate indexes ---------------------------------------------------------

class Fixed(Index):
    """An index that returns hand-set row ids, whatever the query."""

    def __init__(self, idx: list[list[int]], exact: bool) -> None:
        self.idx, self.exact = np.array(idx, np.int64), exact

    def identity(self) -> dict:
        return {"index": "fixed"}

    def build(self, vectors: np.ndarray) -> None:
        pass

    def search(self, queries: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray]:
        return self.idx[:, :k], np.zeros(self.idx[:, :k].shape)


def test_ann_recall_by_hand():
    """B32.  recall@3 = |A n E| / 3 per query.  Query 1: {0,1,2} n {0,3,4} =
    {0}, 1/3.  Query 2: {0} (the rest is -1 padding, i.e. no result) n
    {0,1,2} = {0}, 1/3 -- the padding counts as a miss, not a shrunken
    denominator.  Query 3: {5,6,7} both, 1."""
    approx = Fixed([[0, 1, 2], [0, -1, -1], [7, 5, 6]], exact=False)
    exact = Fixed([[0, 3, 4], [0, 1, 2], [5, 6, 7]], exact=True)
    got = ann_recall(approx, exact, np.zeros((3, 2)), k=3)
    assert_same(got, np.array([1 / 3, 1 / 3, 1.0]), what="ann_recall")


def test_faiss_tests_pass_in_a_process_without_torch():
    """FAISS and torch cannot share a process on macOS with pip wheels
    (search.omp_clash), and this suite imports torch.  When that applies,
    the FAISS tests run in a fresh interpreter instead of being skipped."""
    if omp_clash() is None:
        pytest.skip("no clash in this process: test_faiss.py runs directly")
    r = subprocess.run([sys.executable, "-m", "pytest", "-q", "-p",
                        "no:cacheprovider",
                        str(Path(__file__).with_name("test_faiss.py"))],
                       capture_output=True, text=True, timeout=300,
                       check=False,
                       env={**os.environ, "EVALCORE_FAISS_ALONE": "1"})
    assert r.returncode == 0, r.stdout[-3000:]
    assert " passed" in r.stdout and " skipped" not in r.stdout


def test_faiss_refuses_to_load_beside_a_bundled_torch_runtime():
    if omp_clash() is None:
        pytest.skip(f"no bundled-libomp pair here ({sys.platform})")
    with pytest.raises(ImportError, match="OMP Error #15"):
        FaissIndex().build(np.zeros((4, 2)))


# --- tool and backend -------------------------------------------------------------

def test_search_tool_formats_hits_and_versions_by_retriever():
    t = search_tool(BM25(DOCS), DOCS, k=1)
    assert t({"query": "dog"}) == "[d2] the dog sat on the dog mat"
    assert t({"query": "zebra"}) == "No results."
    assert t.version != search_tool(BM25(DOCS, k1=2.0), DOCS, k=1).version


def test_a_retriever_runs_and_scores_through_the_runner(tmp_path):
    """A one-hop question names its town, and at most three documents
    contain that name (the town's, the river rising there, the person born
    there; the near-name twin is another token), each scoring far above any
    document without it.  So recall@3 = 1 on every question.  recall@1 is
    not: BM25's length normalisation ranks the shorter river and person
    documents above the town's (seen on the first run of this test)."""
    world = make_world(10, 0)
    items = [it for it in search_qa_items(world, 0)
             if it.payload["meta"]["type"] == "pop"]
    task = Task("pop_retrieval", items, score=score_ranked(ks=(1, 3)))
    df = task.run(grid({"retriever": ["bm25"]}),
                  RetrieverBackend(BM25(world.docs), k=5), tmp_path / "r.jsonl",
                  workers=1, batch_size=4)
    assert (df["status"] == "ok").all() and len(df) == len(items)
    assert df["recall@3"].tolist() == [1.0] * len(items)
    assert df["recall@1"].mean() < 1.0
    with pytest.warns(RuntimeWarning, match="carry caveats"):   # 10 queries
        s = retrieval_summary(df, ["retriever"], n_boot=200)
    assert set(s["metric"]) == {"recall@1", "hit@1", "ndcg@1", "recall@3",
                                "hit@3", "ndcg@3", "rr"}


# --- reranking ---------------------------------------------------------------------

def test_a_reranker_reorders_only_what_the_first_stage_found():
    """First stage BM25 at depth 2 for "sat": d1 and d2 (d3 lacks the word).
    The scorer prefers shorter documents, so d1 (3 words) leads d2 (7).
    d3, the shortest, is outside the first stage and never appears."""
    shorter = FunctionScorer(lambda q, docs: [-len(d.split()) for d in docs],
                             {"rule": "shorter"})
    r = Reranker(BM25(DOCS), shorter, DOCS, depth=2)
    hits = r.retrieve(["sat"], k=3)[0]
    assert [d for d, _ in hits] == ["d1", "d2"]
    assert [s for _, s in hits] == [-3.0, -7.0]
    assert r.identity()["first"] == BM25(DOCS).identity()


def test_reranker_ties_break_by_id():
    flat = FunctionScorer(lambda q, docs: [0.0] * len(docs), {"rule": "flat"})
    hits = Reranker(BM25(DOCS), flat, DOCS, depth=3).retrieve(["the"], 2)[0]
    assert [d for d, _ in hits] == ["d1", "d2"]


def test_cross_encoder_scores_do_not_depend_on_the_batch():
    torch = pytest.importorskip("torch")

    class PairTok:
        def __call__(self, qs, ds, padding, truncation, max_length,
                     return_tensors):
            ids = [[ord(c) % 40 + 1 for c in q + "|" + d][:max_length]
                   for q, d in zip(qs, ds, strict=True)]
            T = max(map(len, ids))
            return {"input_ids": torch.tensor([x + [0] * (T - len(x))
                                               for x in ids]),
                    "attention_mask": torch.tensor([[1] * len(x) + [0] * (T - len(x))
                                                    for x in ids])}

    class Pair(torch.nn.Module):
        def __init__(self, labels=1):
            super().__init__()
            self.config = SimpleNamespace(num_labels=labels)
            self.emb = torch.nn.Parameter(
                torch.randn(64, labels, generator=torch.Generator().manual_seed(0)))

        def forward(self, input_ids, attention_mask):
            m = attention_mask.unsqueeze(-1).float()
            return SimpleNamespace(logits=(self.emb[input_ids] * m).sum(1)
                                   / m.sum(1))

    docs = ["short", "a much longer document", "mid one"]
    one = HFCrossEncoder(Pair(), PairTok(), batch_size=1, device="cpu")
    three = HFCrossEncoder(Pair(), PairTok(), batch_size=3, device="cpu")
    np.testing.assert_allclose(one.score("q", docs), three.score("q", docs),
                               atol=1e-6)
    with pytest.raises(ValueError, match="num_labels=2"):
        HFCrossEncoder(Pair(labels=2), PairTok())
