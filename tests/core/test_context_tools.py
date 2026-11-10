"""Context tools: an agent's tool told which unit it serves.  Plain tools and
plain agents keep their identities and keys; a sampled search draws per
repeat inside agent episodes; the gold controls work for an agent's own
queries; and an item-dependent episode is never served from another item's
cache entry."""

from __future__ import annotations

import io

import pytest
from support.retrievers import Fixed

from evalcore import (
    Cache,
    FunctionBackend,
    Item,
    Request,
    Response,
    execute,
    grid,
    units,
)
from evalcore.core.agent import AgentBackend
from evalcore.core.backends import request_key
from evalcore.core.search import (
    BM25,
    FunctionScorer,
    Reranker,
    SampledRetriever,
    gold_by_item,
    gold_search_tool,
    search_tool,
)
from evalcore.core.tools import Tool, ToolContext

FLAT = {f"d{i:02d}": 0.0 for i in range(20)}
SCORES = {"a": 0.0, "b": 0.5, "c": 1.0, "d": 2.0}     # pi = d, c, b, a


def _docs(ids):
    return {d: f"text of {d}" for d in ids}


def search_once(req: Request) -> Response:
    """Search once with a fixed query, then answer with the ids returned."""
    seen = [m for m in req.messages if m.get("role") == "tool"]
    if not seen:
        return Response("x", meta={"tool_calls": [
            {"id": "c1", "name": "search", "arguments": {"query": "anything"}}]})
    ids = [line[1:line.index("]")] for line in seen[0]["content"].split("\n\n")
           if line.startswith("[")]
    return Response("x", text=" ".join(ids), meta={"truncated": False})


def agent(tools, **kw) -> AgentBackend:
    return AgentBackend(FunctionBackend(search_once, identity={"policy": "once"},
                                        supports_tools=True), tools, **kw)


def run(tmp_path, be, its, repeats=1, cache=None, name="r"):
    us = list(units(grid({"x": [1]}), its, repeats))
    return execute(us, be, lambda u: Request(u.id, [{"role": "user",
                                                     "content": "q"}]),
                   lambda u, r: {}, tmp_path / f"{name}.jsonl", workers=1,
                   cache=cache, stream=io.StringIO())


# --- the contract --------------------------------------------------------------

def test_plain_tool_identity_and_call_are_unchanged():
    t = Tool("t", "d", {}, lambda a: "x")
    assert t.identity() == {"name": "t", "description": "d", "parameters": {},
                            "version": "1"}
    assert t({}) == "x" and t({}, ToolContext("u", "i", 0)) == "x"


def test_context_tool_gets_the_context_and_refuses_to_run_without_it():
    t = Tool("t", "d", {}, lambda a, ctx: f"{ctx.item_id}/{ctx.repeat}",
             uses_context=True)
    assert t.identity()["uses_context"] is True
    assert t({}, ToolContext("u", "i7", 3)) == "i7/3"
    with pytest.raises(TypeError, match="without a ToolContext"):
        t({})


def test_keys_move_with_the_item_only_for_an_agent_with_context_tools():
    plain = agent([search_tool(BM25(_docs(SCORES)), _docs(SCORES))])
    ctx = agent([gold_search_tool(Fixed(SCORES), {"i0": ["a"]},
                                  _docs(SCORES), "oracle")])
    req = Request("u", [{"role": "user", "content": "q"}])
    a, b = (Request("u", req.messages, item_id=i) for i in ("i0", "i1"))
    assert request_key(a, 0, plain) == request_key(b, 0, plain) \
        == request_key(req, 0, plain)
    assert request_key(a, 0, ctx) != request_key(b, 0, ctx)
    assert plain.refusal(req) is None
    assert "item_id" in ctx.refusal(req)


def test_only_retrievers_that_use_repeat_make_a_context_search_tool():
    docs = _docs(FLAT)
    sampled = SampledRetriever(Fixed(FLAT), 1.0, seed=0)
    zero = FunctionScorer(lambda q, ds: [0.0] * len(ds), {"fn": "zero"})
    assert search_tool(sampled, docs).uses_context
    assert search_tool(Reranker(sampled, zero, docs, 5), docs).uses_context
    assert not search_tool(BM25(docs), docs).uses_context
    assert not search_tool(Reranker(BM25(docs), zero, docs, 5), docs).uses_context


# --- sampled search inside agent episodes ----------------------------------------

def test_each_repeat_of_an_agent_unit_draws_its_own_search(tmp_path):
    """Four repeats of one item.  The search results row r saw must be the
    draw keyed by repeat r (the direct call is the specification), also with
    a tool cache, which would replay repeat 0 for every repeat if its key
    left the repeat out.  Twenty equally scored documents, k = 3: the four
    draws are not all equal (probability of any two coinciding ~ 1/6840)."""
    r = SampledRetriever(Fixed(FLAT), 1.0, seed=0)
    be = agent([search_tool(r, _docs(FLAT), k=3)],
               tool_cache=Cache(tmp_path / "tools"))
    df = run(tmp_path, be, [Item("i0", {})], repeats=4)
    got = {int(x.repeat): x.text.split() for x in df.itertuples()}
    want = {rep: [d for d, _ in r.retrieve(["anything"], 3, repeat=rep)[0]]
            for rep in range(4)}
    assert got == want
    assert len({tuple(v) for v in got.values()}) > 1


def test_a_plain_search_tool_still_shares_cached_results_across_units(tmp_path):
    calls = []

    class Counted(Fixed):
        def retrieve(self, queries, k, repeat=0):
            calls.append(list(queries))
            return super().retrieve(queries, k, repeat)
    be = agent([search_tool(Counted(SCORES), _docs(SCORES), k=2)],
               tool_cache=Cache(tmp_path / "tools"))
    df = run(tmp_path, be, [Item("i0", {}), Item("i1", {})], repeats=2)
    assert (df["status"] == "ok").all() and len(df) == 4
    assert calls == [["anything"]]            # one search served all four


# --- gold controls for an agent's own queries ------------------------------------

GOLD = {"i0": ["a"], "i1": ["c", "d"]}
WANT = {   # by hand from pi = d, c, b, a and k = 3
    "none": {"i0": ["d", "c", "b"], "i1": ["d", "c", "b"]},
    "oracle": {"i0": ["a", "d", "c"], "i1": ["d", "c", "b"]},
    "removed": {"i0": ["d", "c", "b"], "i1": ["b", "a"]},
}


@pytest.mark.parametrize("mode", ["none", "oracle", "removed"])
def test_gold_search_tool_reads_the_units_item(tmp_path, mode):
    """Both items render the SAME prompt, and the run uses a response
    cache: the item must be in the request key, or i1 would be served i0's
    cached episode."""
    its = [Item(i, {"gold_docs": g}) for i, g in GOLD.items()]
    tool = gold_search_tool(Fixed(SCORES), gold_by_item(its), _docs(SCORES),
                            mode, k=3)
    df = run(tmp_path, agent([tool]), its, cache=Cache(tmp_path / "resp"),
             name=mode)
    got = {x.item_id: x.text.split() for x in df.itertuples()}
    assert got == WANT[mode]


def test_gold_search_tool_refuses_an_item_without_gold(tmp_path):
    tool = gold_search_tool(Fixed(SCORES), {"i0": ["a"]}, _docs(SCORES), "oracle")
    with pytest.raises(ValueError, match="no gold entry"):
        tool({"query": "q"}, ToolContext("u", "i9", 0))
    df = run(tmp_path, agent([tool]), [Item("i9", {})])
    assert df.iloc[0]["status"] == "error"


def test_gold_search_tool_looks_like_search_tool_to_the_model():
    plain = search_tool(BM25(_docs(SCORES)), _docs(SCORES))
    gold = gold_search_tool(Fixed(SCORES), GOLD, _docs(SCORES), "removed")
    assert gold.spec() == plain.spec()


def test_gold_by_item_refuses_duplicate_ids():
    with pytest.raises(ValueError, match="duplicate item id"):
        gold_by_item([Item("i", {"gold_docs": ["a"]}),
                      Item("i", {"gold_docs": ["b"]})])
