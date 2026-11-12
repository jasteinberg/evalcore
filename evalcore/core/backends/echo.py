"""Deterministic offline backend, for dry runs and tests."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .base import Backend, Request, Response, qualified_name


class EchoBackend(Backend):
    """Deterministic offline stand-in.  It lets the whole pipeline (grid,
    runner, cache, frame, CIs, plots) run end to end with no API spend and
    no network, so that a dry run tests the harness rather than the
    vendor."""

    name = "echo"

    def __init__(self, fn: Callable[[Request], str] | None = None,
                 latency: float = 0.0) -> None:
        self.fn = fn or (lambda r: r.messages[-1]["content"][::-1])
        self.latency = latency

    def identity(self) -> dict[str, Any]:
        # two echo backends with different functions are different models
        return {**self._base_identity(), "fn": qualified_name(self.fn)}

    def complete(self, req: Request) -> Response:
        if self.latency:
            time.sleep(self.latency)
        return Response(req.unit_id, text=self.fn(req),
                        meta={"backend": self.name})
