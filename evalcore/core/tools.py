"""Tools a model can call, in a provider-neutral form.

A tool is a name, a description, a JSON-schema for its arguments, and a
function from arguments to text.  Providers spell tool calls differently;
the adapters translate, and everything above them sees only:

    request:   Request.tools = [tool.spec(), ...]
    response:  meta["tool_calls"] = [{"id", "name", "arguments"}, ...]
    result:    {"role": "tool", "tool_call_id", "name", "content"}

Failures are split by whose fault they are, because they mean different
things in a results table:

    ToolInputError    the MODEL asked for something invalid (bad or missing
                      arguments, an unknown tool).  The message goes back to
                      the model as the tool's output; recovering is part of
                      the behaviour being measured.
    ToolUnavailable   the TOOL's infrastructure failed (a search API is
                      down, rate-limited).  The episode ends with status
                      `tool_error`: neither a model failure nor gradable.
    anything else     a bug in the tool's code.  An error row, never hidden
                      as model misbehaviour.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

__all__ = ["Tool", "ToolInputError", "ToolUnavailable"]


class ToolInputError(ValueError):
    """The model's call was invalid; the message is returned to the model."""


class ToolUnavailable(RuntimeError):
    """The tool's infrastructure failed; the episode is a `tool_error`."""


@dataclass(frozen=True)
class Tool:
    """`fn(arguments) -> str`.  `version` belongs in the identity: bump it
    when the tool's behaviour changes, so cached results and finished rows
    made with the old one are not reused."""

    name: str
    description: str
    parameters: Mapping[str, Any]
    fn: Callable[[Mapping[str, Any]], str] = field(compare=False)
    version: str = "1"

    def spec(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description,
                "parameters": dict(self.parameters)}

    def identity(self) -> dict[str, Any]:
        return {**self.spec(), "version": self.version}

    def __call__(self, arguments: Mapping[str, Any]) -> str:
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
        return self.fn(arguments)


def tool_index(tools: Sequence[Tool]) -> dict[str, Tool]:
    names = [t.name for t in tools]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate tool names: {names}")
    return {t.name: t for t in tools}
