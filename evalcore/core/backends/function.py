"""Wrap a plain function as a backend: the quickest way to bring a model the
harness has no adapter for."""

from __future__ import annotations

import dataclasses
import time
from collections.abc import Callable, Mapping
from typing import Any

from ..spec import canonical
from .base import Backend, Request, Response


class FunctionBackend(Backend):
    """`fn(request) -> str` (the text) or `-> Response` (anything richer).

    `identity` is required: it is what the request key, and so the cache
    and resume, know about the model behind `fn` -- a checkpoint path, a
    revision, a server URL, sampling settings held outside the request.
    Two FunctionBackends with different models and the same identity would
    share results, which is the failure this argument exists to prevent.

    `fn` may raise `Transient` (retried) or `Fatal` (not); anything else is
    turned into an error row by the runner.  A returned Response is bound
    to the request it answers, so fn need not copy the unit_id."""

    name = "function"

    def __init__(self, fn: Callable[[Request], str | Response],
                 identity: Mapping[str, Any], name: str | None = None,
                 supports_tools: bool = False) -> None:
        if not identity:
            raise ValueError("identity must name what determines the outputs")
        canonical(dict(identity))         # JSON-serialisable, no NaN, or raise
        self.fn = fn
        self._identity = dict(identity)
        if name is not None:
            self.name = name
        # set when fn reads Request.tools and returns meta["tool_calls"]
        self.supports_tools = supports_tools

    def identity(self) -> dict[str, Any]:
        return {**self._base_identity(), **self._identity}

    def complete(self, req: Request) -> Response:
        t0 = time.perf_counter()
        out = self.fn(req)
        if isinstance(out, Response):
            # fn answers exactly this request, so its response belongs to
            # it whatever unit_id fn wrote (often none, or a copy)
            return dataclasses.replace(out, unit_id=req.unit_id)
        return Response(req.unit_id, text=out,
                        meta={"backend": self.name,
                              "latency_s": time.perf_counter() - t0})
