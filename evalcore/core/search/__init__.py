r"""Retrieval over a local corpus: lexical, dense, exact and approximate.

A `Retriever` maps queries to ranked (doc id, score) lists and has an
`identity()`.  Four are here, all offline:

    BM25             lexical, exact, no model (the default search tool)
    DenseRetriever   an Embedder plus an Index
    Reranker         any retriever's top `depth`, rescored by a PairScorer
                     (a function, or an HF cross-encoder)
    SampledRetriever a random k-subset drawn from a base retriever's scores
                     (opt-in: the three above are deterministic)

and one control, for evaluating a reader that consumes retrieval:

    GoldControl      the base ranking with each query's gold documents
                     forced in ("oracle"), taken out ("removed"), or left
                     alone ("none", the matched base condition), at the same k

A `ScoredRetriever` also gives `score_all`: every document's score, not
only the top k.  BM25 and a DenseRetriever over an ExactIndex do, but an
approximate index cannot, and refuses.

Ties are broken by document id everywhere, so a ranking is a function of
the corpus and the query alone.

`search_tool(retriever, docs)` turns any retriever into the agent loop's
`search` tool (`gold_search_tool` is the controls' version), and
`RetrieverBackend` runs a retriever through the runner (cache, resume,
rows), so evaluation/retrieval.py can score it like any other model.

The package, one job per module:

    base       the protocols, Hit, ranking helpers
    lexical    BM25
    dense      Index, ExactIndex, FaissIndex, DenseRetriever, ann_recall
    rerank     PairScorer, FunctionScorer, HFCrossEncoder, Reranker
    sampled    SampledRetriever (the law and the seeding rule)
    controls   GoldControl, gold_by_query, gold_by_item
    adapters   search_tool, gold_search_tool, CorpusSearch, RetrieverBackend

Only `adapters` imports the harness (tools, backends).  Everything is
re-exported here, so `from evalcore.core.search import X` works for all of
it.
"""

from .adapters import CorpusSearch, RetrieverBackend, gold_search_tool, search_tool
from .base import Hit, Retriever, ScoredRetriever
from .base import _top as _top  # a test checks the tie rule on it directly
from .controls import GoldControl, gold_by_item, gold_by_query
from .dense import DenseRetriever, ExactIndex, FaissIndex, Index, ann_recall, omp_clash
from .lexical import BM25
from .rerank import FunctionScorer, HFCrossEncoder, PairScorer, Reranker
from .sampled import SampledDraw, SampledRetriever

__all__ = [
    "BM25",
    "CorpusSearch",
    "DenseRetriever",
    "ExactIndex",
    "FaissIndex",
    "FunctionScorer",
    "GoldControl",
    "HFCrossEncoder",
    "Hit",
    "Index",
    "PairScorer",
    "Reranker",
    "Retriever",
    "RetrieverBackend",
    "SampledDraw",
    "SampledRetriever",
    "ScoredRetriever",
    "ann_recall",
    "gold_by_item",
    "gold_by_query",
    "gold_search_tool",
    "omp_clash",
    "search_tool",
]
