r"""Retrievers in the harness: as an agent's search tool (search_tool, and
gold_search_tool for the controls), as a corpus with its tool
(CorpusSearch), and as a backend the runner can call (RetrieverBackend).
The algorithm modules beside this one import none of the harness; this
module is where retrieval meets tools and backends.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from ..backends.base import Backend, Request, Response
from ..spec import digest
from ..tools import Tool, ToolContext, ToolInputError
from .base import (
    _WORD,
    Hit,
    Retriever,
    ScoredRetriever,
    _all_scores,
    _check_k,
    _uses_repeat,
)
from .controls import _check_mode, _control_hits, _gold_map
from .lexical import BM25

_SEARCH_DESCRIPTION = ("Search the document collection. Returns the best "
                       "matching documents with their ids.")
_SEARCH_PARAMETERS = {"type": "object",
                      "properties": {"query": {"type": "string",
                                               "description": "search terms"}},
                      "required": ["query"]}


def _query(args: Mapping[str, Any], name: str) -> str:
    q = str(args["query"])
    if not _WORD.search(q.lower()):
        raise ToolInputError(f"{name}: empty query")
    return q


def _render_hits(hits: list[Hit], docs: Mapping[str, str]) -> str:
    """A search tool's output: each hit as "[id] text"."""
    if not hits:
        return "No results."
    return "\n\n".join(f"[{d}] {docs[d]}" for d, _ in hits)


def search_tool(retriever: Retriever, docs: Mapping[str, str], k: int = 3,
                name: str = "search") -> Tool:
    """The retriever as the agent loop's search tool: the top k documents,
    each as "[id] text".  The tool version is a digest of the retriever's
    identity and k, so a changed corpus or retriever re-runs its rows.

    A retriever whose draw depends on the repeat index (`uses_repeat`, as
    SampledRetriever and a Reranker over one declare) makes this a context
    tool: each repeat of a unit then searches with its own repeat index.  A
    retriever that does not declare it is deterministic by the Retriever
    contract, and its searches keep sharing cached results across units."""
    ident = retriever.identity()
    version = (f"{ident.get('retriever', 'search')}-"
               f"{digest({'retriever': ident, 'k': k})}")
    if _uses_repeat(retriever):
        def run_ctx(args: Mapping[str, Any], ctx: ToolContext) -> str:
            return _render_hits(retriever.retrieve([_query(args, name)], k,
                                                   repeat=ctx.repeat)[0], docs)
        return Tool(name, _SEARCH_DESCRIPTION, _SEARCH_PARAMETERS, run_ctx,
                    version=version, uses_context=True)

    def run(args: Mapping[str, Any]) -> str:
        return _render_hits(retriever.retrieve([_query(args, name)], k)[0], docs)
    return Tool(name, _SEARCH_DESCRIPTION, _SEARCH_PARAMETERS, run,
                version=version)


def gold_search_tool(base: ScoredRetriever, gold: Mapping[str, Sequence[str]],
                     docs: Mapping[str, str], mode: str, k: int = 3,
                     name: str = "search") -> Tool:
    """The gold controls (module docstring) as an agent's search tool.  The
    ranking is the base's for the agent's own query, and the gold set is the
    item's, looked up by the unit's item id in `gold` (gold_by_item builds
    it).  An item with no entry is an error, never a plain search.  The
    tool presents itself exactly as search_tool does, so the agent cannot
    tell the conditions apart from the tool's name or description."""
    _check_mode(mode)
    _check_k(k)
    gold_ids = _gold_map(gold)
    ident = {"retriever": f"gold-{mode}", "base": base.identity(),
             "gold": digest(gold_ids), "by": "item"}

    def run(args: Mapping[str, Any], ctx: ToolContext) -> str:
        q = _query(args, name)
        if ctx.item_id not in gold_ids:
            raise ValueError(f"{name}: no gold entry for item {ctx.item_id!r}")
        ids, S = _all_scores(base, [q])
        return _render_hits(_control_hits(ids, S[0], gold_ids[ctx.item_id], mode,
                                          k, f"item {ctx.item_id!r}"), docs)
    return Tool(name, _SEARCH_DESCRIPTION, _SEARCH_PARAMETERS, run,
                version=f"gold-{mode}-{digest({'retriever': ident, 'k': k})}",
                uses_context=True)


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
        hits: list[list[Hit]] = [[] for _ in reqs]
        for rep in sorted({r.repeat for r in reqs}):       # one call per repeat
            at = [i for i, r in enumerate(reqs) if r.repeat == rep]
            for i, h in zip(at, self.retriever.retrieve(
                    [self._query(reqs[i]) for i in at], self.k, repeat=rep),
                    strict=True):
                hits[i] = h
        return [Response(r.unit_id, text="\n".join(d for d, _ in h),
                         meta={"retrieved": [d for d, _ in h],
                               "scores": [s for _, s in h]})
                for r, h in zip(reqs, hits, strict=True)]
