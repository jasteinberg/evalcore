"""Multi-turn tool use as a backend.

`AgentBackend(inner, tools)` is itself a Backend: its `complete` runs

    model turn -> zero or more tool calls -> results appended -> repeat

until the model answers without calling a tool, or a cap is hit.  Because
the loop is a backend, the runner, cache, resume and attrition need nothing
new: one unit is one episode and one row, and the row's meta holds the
whole trajectory.

Outcomes, and the row status each becomes:

    final answer                         ok (then graded as usual)
    max_turns or max_tool_calls reached  truncated (truncated_by says which)
    model call failed (after retries)    error
    ToolUnavailable from a tool          tool_error (infrastructure)
    other exception inside a tool        error (a tool bug)
    ToolInputError / unknown tool name   returned to the model as the
                                         tool's output; counted in
                                         meta["tool_input_errors"]

Tool results can be cached content-addressed on (tool identity,
arguments).  With live tools -- a web search whose results drift -- that
cache is what makes a rerun see the same results, so pass one.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .backends.base import Backend, Request, Response, with_retries
from .backends.cache import Cache
from .spec import digest
from .tools import Tool, ToolInputError, ToolUnavailable, tool_index

__all__ = ["AgentBackend"]


class AgentBackend(Backend):
    name = "agent"

    def __init__(self, inner: Backend, tools: Sequence[Tool],
                 max_turns: int = 8, max_tool_calls: int = 16,
                 tool_cache: Cache | None = None, attempts: int = 5,
                 result_chars: int = 20_000) -> None:
        if not inner.supports_tools:
            raise TypeError(f"{type(inner).__name__} cannot send tools "
                            f"(supports_tools is False)")
        self.inner, self.tools = inner, tool_index(tools)
        self.max_turns, self.max_tool_calls = max_turns, max_tool_calls
        self.tool_cache, self.attempts = tool_cache, attempts
        self.result_chars = result_chars

    def identity(self) -> dict[str, Any]:
        return {"backend": self.name, "inner": self.inner.identity(),
                "tools": [t.identity() for t in self.tools.values()],
                "max_turns": self.max_turns,
                "max_tool_calls": self.max_tool_calls}

    def close(self) -> None:
        self.inner.close()

    def _call_tool(self, tool: Tool, args: Any) -> str:
        key = digest({"tool": tool.identity(), "args": args})
        hit = self.tool_cache.get(key) if self.tool_cache else None
        if hit is not None:
            return hit["content"]
        content = tool(args)
        if self.tool_cache:
            self.tool_cache.put(key, {"content": content})
        return content

    def complete(self, req: Request) -> Response:
        specs = [t.spec() for t in self.tools.values()]
        msgs = [dict(m) for m in req.messages]
        trajectory: list[dict[str, Any]] = []
        n_calls = input_errors = 0
        meta: dict[str, Any] = {"backend": self.name}

        def total(key: str) -> int | None:
            # cost is per episode: every turn is billed, and the input grows
            # each turn, so the last turn's count alone under-reports it
            vals = [t.get(key) for t in trajectory]
            return sum(v for v in vals if v is not None) if any(
                v is not None for v in vals) else None

        def done(**kw: Any) -> Response:
            return Response(req.unit_id, text=kw.pop("text", None),
                            error=kw.pop("error", None),
                            meta={**meta, "trajectory": trajectory,
                                  "turns": len(trajectory),
                                  "n_tool_calls": n_calls,
                                  "tool_input_errors": input_errors,
                                  **kw, "in_tokens": total("in_tokens"),
                                  "out_tokens": total("out_tokens")})

        for _ in range(self.max_turns):
            r = with_retries(self.inner.complete,
                             Request(req.unit_id, msgs, req.params,
                                     tools=specs), self.attempts)
            if not r.ok:
                return done(error=f"model: {r.error}", truncated=False)
            calls = list((r.meta or {}).get("tool_calls") or [])
            turn: dict[str, Any] = {"text": r.text, "tool_calls": calls,
                                    "results": [],
                                    "in_tokens": (r.meta or {}).get("in_tokens"),
                                    "out_tokens": (r.meta or {}).get("out_tokens")}
            trajectory.append(turn)
            if not calls:
                return done(text=r.text,
                            truncated=bool((r.meta or {}).get("truncated")),
                            **{k: v for k, v in (r.meta or {}).items()
                               if k in ("stop_reason", "model")})
            asst = {"role": "assistant", "content": r.text,
                    "tool_calls": calls}
            if (r.meta or {}).get("raw_turn"):
                asst["raw_turn"] = r.meta["raw_turn"]   # echoed verbatim
            msgs.append(asst)
            for c in calls:
                n_calls += 1
                if n_calls > self.max_tool_calls:
                    return done(truncated=True,
                                truncated_by="max_tool_calls")
                tool = self.tools.get(c.get("name"))
                try:
                    if tool is None:
                        raise ToolInputError(
                            f"unknown tool {c.get('name')!r}; available: "
                            f"{sorted(self.tools)}")
                    content, kind = self._call_tool(tool, c.get("arguments")), "ok"
                except ToolInputError as exc:
                    input_errors += 1
                    content, kind = f"error: {exc}", "input_error"
                except ToolUnavailable as exc:
                    return done(error=f"tool unavailable: {c.get('name')}: "
                                      f"{exc}", tool_error=True,
                                truncated=False)
                turn["results"].append({
                    "name": c.get("name"), "arguments": c.get("arguments"),
                    "kind": kind, "content": content[: self.result_chars]})
                msgs.append({"role": "tool", "tool_call_id": c.get("id"),
                             "name": c.get("name"), "content": content,
                             "is_error": kind == "input_error"})
        return done(truncated=True, truncated_by="max_turns")

