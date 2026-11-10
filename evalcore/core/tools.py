"""Tools a model can call, in a provider-neutral form.

A tool is a name, a description, a JSON-schema for its arguments, and a
function from arguments to text.  Providers spell tool calls differently,
so the adapters translate, and everything above them sees only:

    request:   Request.tools = [tool.spec(), ...]
    response:  meta["tool_calls"] = [{"id", "name", "arguments"}, ...]
    result:    {"role": "tool", "tool_call_id", "name", "content"}

Failures are split by whose fault they are, because they mean different
things in a results table:

    ToolInputError    the MODEL asked for something invalid (bad or missing
                      arguments, an unknown tool).  The message goes back to
                      the model as the tool's output, and recovering is
                      part of the behaviour being measured.
    ToolUnavailable   the TOOL's infrastructure failed (a search API is
                      down, rate-limited).  The episode ends with status
                      `tool_error`: neither a model failure nor gradable.
    anything else     a bug in the tool's code.  An error row, never hidden
                      as model misbehaviour.

Context tools.  Most tools are a function of their arguments alone.  A tool
whose output also depends on WHICH unit it serves sets `uses_context=True`
and is called as fn(arguments, ctx), with ctx a ToolContext (unit_id,
item_id, repeat).  Two uses: a sampled search that must draw independently
in every repeat of a unit, and a control that reads the item's gold
documents.  Its output must be a function of (arguments, item_id, repeat),
because those, and only those, key its cached results: the agent's tool
cache adds them, and the agent's request key adds the item, so item B is
never served item A's results.  A tool that does not set the flag keeps its
identity and its cache keys, and shares cached results across units.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Tool", "ToolContext", "ToolInputError", "ToolUnavailable"]


class ToolInputError(ValueError):
    """The model's call was invalid, and the message is returned to it."""


class ToolUnavailable(RuntimeError):
    """The tool's infrastructure failed, so the episode is a `tool_error`."""


@dataclass(frozen=True)
class ToolContext:
    """The unit a context tool is serving (module docstring)."""

    unit_id: str
    item_id: str
    repeat: int

    def key(self) -> dict[str, Any]:
        """What keys a context tool's cached result: not the unit id, which
        also names the cell, so cells sharing an item share results."""
        return {"item_id": self.item_id, "repeat": self.repeat}


@dataclass(frozen=True)
class Tool:
    """`fn(arguments) -> str`, or `fn(arguments, ctx) -> str` with
    `uses_context=True`.  `version` belongs in the identity: bump it when
    the tool's behaviour changes, so cached results and finished rows made
    with the old one are not reused."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    fn: Callable[..., str] = field(compare=False)
    version: str = "1"
    uses_context: bool = False

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": dict(self.parameters)}

    def identity(self) -> dict[str, Any]:
        ident = {**self.spec(), "version": self.version}
        if self.uses_context:            # absent, so existing identities hold
            ident["uses_context"] = True
        return ident

    def __call__(self, arguments: Mapping[str, Any],
                 ctx: ToolContext | None = None) -> str:
        if not isinstance(arguments, Mapping):
            raise ToolInputError(f"{self.name}: arguments must be an object")
        props = self.parameters.get("properties", {})
        missing = [k for k in self.parameters.get("required", ())
                   if k not in arguments]
        unknown = [k for k in arguments if props and k not in props]
        if missing or unknown:
            # both, so the model can correct the call in one turn
            problems = ([f"missing {missing}"] if missing else []) + (
                [f"unknown argument(s) {unknown}"] if unknown else [])
            raise ToolInputError(f"{self.name}: " + "; ".join(problems))
        if self.uses_context:
            if ctx is None:
                raise TypeError(f"tool {self.name!r} reads the unit it serves "
                                f"and was called without a ToolContext")
            return self.fn(arguments, ctx)
        return self.fn(arguments)


def tool_index(tools: Sequence[Tool]) -> dict[str, Tool]:
    names = [t.name for t in tools]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate tool names: {names}")
    return {t.name: t for t in tools}
