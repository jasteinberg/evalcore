r"""Gold controls: the base ranking with each query's gold documents
forced in, taken out, or left alone, at the same k.

Controls.  Let pi be the base's full ranking of the corpus by (score desc,
id asc), G the query's gold set, and |G| <= k.  Then

    none(q)    = pi, first k
    oracle(q)  = [G in pi order] + [pi minus G], first k
    removed(q) = [pi minus G], first k .

All three return k documents (removed returns n - |G| if that is fewer),
filled by the same rule, so a reader's accuracy differs between them only
through whether gold is present, not through how much context it reads.
The difference oracle - none measures what better retrieval could still
buy.  The difference none - removed measures how much of the score depends
on the gold text at all, and accuracy under removed is what the reader
answers without it (parametric memory, or contamination).

"none" is the base condition to compare the controls against, rather than
the base retriever itself.  pi is the FULL ranking, from `score_all`, so a
filler can be a document BM25.retrieve would drop (sharing no term with
the query).  For a query that shares terms with fewer than k documents,
BM25 alone returns fewer than k, and comparing it with the controls would
mix the gold effect with a difference in context length.  For a dense base
over an exact index, none(q) is exactly base.retrieve(q, k).

GoldControl looks G up by the query string, so it fits a reader whose
retrieval query is the item's own question.  An agent writes its own
searches, and `gold_search_tool` applies the same three modes to each of
them: pi is the base's ranking for the agent's query, and G is the gold set
of the item the unit serves (`gold_by_item`), read from the tool's context.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import numpy as np

from ..spec import digest
from .base import Hit, ScoredRetriever, _all_scores, _check_k, _top

_CONTROL_MODES = ("none", "oracle", "removed")


def _check_mode(mode: str) -> None:
    if mode not in _CONTROL_MODES:
        raise ValueError(f"mode must be one of {_CONTROL_MODES}, got {mode!r}")


def _gold_set(ids: Iterable[str]) -> list[str]:
    """Gold ids sorted and deduplicated.  The controls use this form and
    their identities digest it, so the order in which a gold list was given
    does not change an identity."""
    return sorted(set(ids))


def _gold_map(gold: Mapping[str, Iterable[str]]) -> dict[str, list[str]]:
    return {key: _gold_set(ids) for key, ids in gold.items()}


class GoldControl:
    """A control condition for a reader that consumes retrieval: the base
    ranking with the query's gold documents forced in ("oracle"), taken
    out ("removed"), or left alone ("none", the condition to compare those two
    against), at the same k (definitions in the module docstring).

    `gold` maps each query, as the retriever receives it, to its gold doc
    ids, and `gold_by_query` builds it from items.  A query not in it is
    refused, never answered from the base alone, in every mode, so the
    three conditions run on the same queries."""

    uses_repeat = False

    def __init__(self, base: ScoredRetriever, gold: Mapping[str, Sequence[str]],
                 mode: str) -> None:
        _check_mode(mode)
        self.base, self.mode = base, mode
        self.gold = _gold_map(gold)
        self._identity = {"retriever": f"gold-{mode}", "base": base.identity(),
                          "gold": digest(self.gold)}

    def identity(self) -> dict[str, Any]:
        return self._identity

    def retrieve(self, queries: Sequence[str], k: int,
                 repeat: int = 0) -> list[list[Hit]]:
        """`repeat` is ignored: the control is deterministic."""
        _check_k(k)
        queries = list(queries)
        missing = [q for q in queries if q not in self.gold]
        if missing:
            raise KeyError(f"{len(missing)} query(ies) have no gold entry, "
                           f"e.g. {missing[0]!r}")
        if not queries:
            return []
        ids, S = _all_scores(self.base, queries)
        return [_control_hits(ids, s, self.gold[q], self.mode, k, repr(q))
                for q, s in zip(queries, S, strict=True)]


def _control_hits(ids: list[str], s: np.ndarray, gold: Sequence[str],
                  mode: str, k: int, label: str) -> list[Hit]:
    """none / oracle / removed (module docstring) for one ranking: `s` is
    every document's score, columns in `ids` order.  GoldControl and
    gold_search_tool both use this one implementation."""
    col = {d: j for j, d in enumerate(ids)}
    unknown = [d for d in gold if d not in col]
    if unknown:
        raise ValueError(f"gold ids not in the corpus for {label}: {unknown}")
    g = {col[d] for d in gold}
    if mode == "oracle" and len(g) > k:
        raise ValueError(f"{len(g)} gold documents for {label} do not fit "
                         f"in k = {k}")
    order = _top(s[None, :], len(ids))[0][0]      # score desc, id asc
    rest = [j for j in order if j not in g]
    if mode == "none":
        pick = list(order[:k])
    elif mode == "oracle":
        pick = ([j for j in order if j in g] + rest)[:k]
    else:
        pick = rest[:k]
    return [(ids[j], float(s[j])) for j in pick]


def _gold_ids(item: Any, gold_field: str) -> list[str]:
    """An item's gold doc ids, sorted: `payload[gold_field]` is a list of
    ids or {id: gain}."""
    return _gold_set(item.payload[gold_field])


def gold_by_query(items: Sequence[Any], gold_field: str = "gold_docs"
                  ) -> dict[str, list[str]]:
    """{query: gold ids} from items in the canonical format: the query is
    the last user message, as RetrieverBackend reads it, and the gold is
    `payload[gold_field]` (a list of ids, or {id: gain}).  Two items with
    the same query and different gold are refused."""
    out: dict[str, list[str]] = {}
    for it in items:
        users = [m for m in it.payload["messages"] if m.get("role") == "user"]
        if not users:
            raise ValueError(f"item {it.item_id} has no user message")
        q, g = users[-1]["content"], _gold_ids(it, gold_field)
        if q in out and out[q] != g:
            raise ValueError(f"query {q!r} has two different gold sets: "
                             f"{out[q]} and {g}")
        out[q] = g
    return out


def gold_by_item(items: Sequence[Any], gold_field: str = "gold_docs"
                 ) -> dict[str, list[str]]:
    """{item_id: gold ids}, for gold_search_tool.  Duplicate item ids are
    refused (the item id is the cluster label, so it must be unique)."""
    out: dict[str, list[str]] = {}
    for it in items:
        if it.item_id in out:
            raise ValueError(f"duplicate item id {it.item_id!r}")
        out[it.item_id] = _gold_ids(it, gold_field)
    return out
