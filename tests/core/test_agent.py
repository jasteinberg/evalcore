"""The tool-use loop: episodes end the way they should, every failure lands
in the right category, and a rerun replays the same tool results."""

from __future__ import annotations

import io
from collections.abc import Callable

import pandas as pd
import pytest

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
from evalcore.core.search import CorpusSearch
from evalcore.core.tools import Tool, ToolInputError, ToolUnavailable
from evalcore.evaluation.attrition import attrition

DOCS = {"d1": "The Velor river rises in the Askern hills.",
        "d2": "Askern is a town of 4,200 people.",
        "d3": "Brindle is a port on the Velor."}


def scripted(policy: Callable[[Request], Response | str],
             name: str = "scripted") -> FunctionBackend:
    """A tool-capable model whose turns are a function of the conversation."""
    return FunctionBackend(policy, identity={"policy": name},
                           supports_tools=True)


def call(name, args, cid="c1"):
    return Response("x", meta={"tool_calls": [{"id": cid, "name": name,
                                               "arguments": args}]})


def two_hop(req: Request):
    """Search for the river's source, then the town's population, then
    answer from what came back."""
    seen = [m for m in req.messages if m.get("role") == "tool"]
    assert req.tools and req.tools[0]["name"] == "search"
    if not seen:
        return call("search", {"query": "Velor river rises"})
    if len(seen) == 1:
        assert "Askern" in seen[0]["content"]
        return call("search", {"query": "Askern town people"}, "c2")
    pop = seen[1]["content"].split("of ")[1].split(" people")[0]
    return Response("x", text=f"About {pop} people.",
                    meta={"truncated": False})


def episode(policy, tools=None, **kw) -> Response:
    be = AgentBackend(scripted(policy), tools or [CorpusSearch(DOCS).tool], **kw)
    return be.complete(Request("u", [{"role": "user", "content": "q"}], {}))


def one_row(tmp_path, be, label="a", tools=None,
            score=lambda u, r: {}) -> pd.Series:
    """`be` run through the runner on one item (a request carrying `tools`
    specs if given); the row."""
    df = execute(units(grid({"condition": [label]}), [Item("q0", {})]), be,
                 lambda u: Request(u.id, [{"role": "user", "content": "q"}],
                                   {}, tools=tools),
                 score, tmp_path / f"{label}.jsonl", workers=1,
                 stream=io.StringIO())
    return df.iloc[0]


def test_a_two_hop_episode_records_its_trajectory():
    r = episode(two_hop)
    assert r.ok and r.text == "About 4,200 people."
    assert r.meta["turns"] == 3 and r.meta["n_tool_calls"] == 2
    assert [t["tool_calls"][0]["arguments"]["query"]
            for t in r.meta["trajectory"][:2]] == ["Velor river rises",
                                                   "Askern town people"]
    assert r.meta["trajectory"][0]["results"][0]["content"].startswith("[d1]")
    assert r.meta["truncated"] is False


def test_caps_end_an_episode_as_truncated():
    loop = lambda req: call("search", {"query": "Velor"})   # noqa: E731
    r = episode(loop, max_turns=3)
    assert r.meta["truncated"] and r.meta["truncated_by"] == "max_turns"
    assert r.meta["turns"] == 3
    r = episode(loop, max_turns=10, max_tool_calls=2)
    assert r.meta["truncated_by"] == "max_tool_calls"


def test_model_mistakes_go_back_to_the_model():
    def clumsy(req):
        tool_msgs = [m for m in req.messages if m.get("role") == "tool"]
        if len(tool_msgs) == 0:
            return call("browse", {"url": "x"})              # no such tool
        if len(tool_msgs) == 1:
            assert tool_msgs[0]["content"].startswith("error: unknown tool")
            return call("search", {"q": "Velor"}, "c2")      # wrong argument
        assert "missing ['query']; unknown argument(s) ['q']" in \
            tool_msgs[1]["content"]
        return Response("x", text="gave up", meta={"truncated": False})

    r = episode(clumsy)
    assert r.ok and r.meta["tool_input_errors"] == 2
    kinds = [res["kind"] for t in r.meta["trajectory"] for res in t["results"]]
    assert kinds == ["input_error", "input_error"]


def search_tool(fn: Callable[[dict], str] = lambda a: "r",
                version: str = "1") -> Tool:
    return Tool("search", "s", {"type": "object", "properties":
                                {"query": {"type": "string"}}}, fn,
                version=version)


def flaky_tool(exc: Exception) -> Tool:
    def fn(args):
        raise exc
    return search_tool(fn)


def test_infrastructure_and_bugs_are_told_apart(tmp_path):
    """Through the runner: a search outage is `tool_error`, a bug in a
    tool is `error`, a model failure is `error` -- and attrition keeps
    the first apart from the others."""
    first_call = lambda req: call("search", {"query": "x"})   # noqa: E731
    outs = {}
    for label, be in [
        ("outage", AgentBackend(scripted(first_call, "o"),
                                [flaky_tool(ToolUnavailable("503"))])),
        ("bug", AgentBackend(scripted(first_call, "b"),
                             [flaky_tool(KeyError("oops"))])),
        ("model", AgentBackend(FunctionBackend(
            lambda r: Response("x", error="fatal: refused"),
            identity={"m": "down"}, supports_tools=True),
            [CorpusSearch(DOCS).tool])),
    ]:
        outs[label] = one_row(tmp_path, be, label)
    assert outs["outage"]["status"] == "tool_error"
    assert outs["outage"]["error"].startswith("tool unavailable")
    assert outs["bug"]["status"] == "error" and "KeyError" in outs["bug"]["error"]
    assert outs["model"]["status"] == "error"
    assert outs["model"]["error"].startswith("model:")
    a = attrition(outs["outage"].to_frame().T, ["condition"]).iloc[0]
    assert (a["n_tool_error"], a["n_err"]) == (1, 0)


def test_a33_tool_results_are_cached_except_failures_and_keyed_on_version(
        tmp_path):
    """A33.  The tool's result drifts per call, so only the cache can make a
    rerun replay it.  Expected call counts are hand-counted: episode 1's
    tool call raises ToolUnavailable (1 call); episode 2 must call the tool
    again (2), then succeed and cache; episode 3, same tool, is served from
    the cache (still 2); episode 4 with version "2" must miss (3)."""
    calls = []

    def fn(args):
        calls.append(args["query"])
        if len(calls) == 1:
            raise ToolUnavailable("503")
        return f"result {len(calls)}"

    def once(req):
        if not any(m.get("role") == "tool" for m in req.messages):
            return call("search", {"query": "V"})
        return Response("x", text=req.messages[-1]["content"],
                        meta={"truncated": False})

    cache = Cache(tmp_path / "tools")
    r1 = episode(once, [search_tool(fn)], tool_cache=cache)
    assert not r1.ok and r1.meta["tool_error"] and len(calls) == 1
    r2 = episode(once, [search_tool(fn)], tool_cache=cache)
    assert r2.ok and r2.text == "result 2" and len(calls) == 2
    r3 = episode(once, [search_tool(fn)], tool_cache=cache)
    assert r3.text == "result 2" and len(calls) == 2
    r4 = episode(once, [search_tool(fn, version="2")], tool_cache=cache)
    assert r4.text == "result 3" and len(calls) == 3


def test_a32_identity_covers_inner_model_tools_and_caps():
    """A32. Invariance: a different inner model, a different corpus behind
    the tool, a bumped tool version or a different cap each changes what an
    episode can return, so each must give a different identity from the
    base; the same construction twice must give the same one."""

    def agent(policy="p", tools=None, **kw):
        return str(AgentBackend(scripted(two_hop, policy),
                                tools or [CorpusSearch(DOCS).tool],
                                **kw).identity())

    base = agent()
    assert agent() == base                             # deterministic
    variants = [agent(policy="other"),
                agent(tools=[CorpusSearch({**DOCS, "d4": "x"}).tool]),
                agent(tools=[search_tool()]),
                agent(tools=[search_tool(version="2")]),
                agent(max_turns=2), agent(max_tool_calls=3)]
    assert len({base, *variants}) == 1 + len(variants)


def test_a_tool_request_to_a_backend_without_tools_is_refused(tmp_path):
    with pytest.raises(TypeError, match="cannot send tools"):
        AgentBackend(FunctionBackend(lambda r: "x", identity={"m": 1}), [])
    row = one_row(tmp_path, FunctionBackend(lambda r: "x", identity={"m": 1}),
                  tools=[CorpusSearch(DOCS).tool.spec()])
    assert row["error"].startswith("fatal: FunctionBackend cannot send tools")


def test_a35_a_truncated_final_turn_gives_a_truncated_row(tmp_path):
    """A35. The model's final answer carries truncated=True (it hit its
    budget).  Expected row status: truncated, as for any backend."""

    def cut(req):
        return Response("x", text="The answer is", meta={"truncated": True})

    row = one_row(tmp_path, AgentBackend(scripted(cut, "cut"), [search_tool()]),
                  score=lambda u, r: {"s": 1.0})
    assert row["status"] == "truncated"


def test_a38_duplicate_tool_names_are_refused():
    """A38. Two tools named "search": the model's call would be ambiguous,
    so construction must raise naming the duplicate."""
    with pytest.raises(ValueError, match="search"):
        AgentBackend(scripted(two_hop), [search_tool(),
                                         search_tool(version="2")])


def test_corpus_search_refuses_an_empty_query():
    with pytest.raises(ToolInputError):
        CorpusSearch(DOCS, k=2).tool({"query": "  "})


def test_an_episode_reports_tokens_summed_over_its_turns():
    """Cost is per episode: every turn's input is billed, and an agent's
    input grows each turn.  Turns here report 100/10, 150/12, 210/30."""
    usage = iter([(100, 10), (150, 12), (210, 30)])

    def policy(req):
        i, o = next(usage)
        meta = {"in_tokens": i, "out_tokens": o, "truncated": False}
        if sum(m.get("role") == "tool" for m in req.messages) < 2:
            return Response("x", meta={**meta, "tool_calls": [
                {"id": "c", "name": "search", "arguments": {"query": "Velor"}}]})
        return Response("x", text="done", meta=meta)

    r = episode(policy)
    assert (r.meta["in_tokens"], r.meta["out_tokens"]) == (460, 52)
    assert [t["in_tokens"] for t in r.meta["trajectory"]] == [100, 150, 210]
