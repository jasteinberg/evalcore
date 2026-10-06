"""Deterministic offline backend, for dry runs and tests."""

from __future__ import annotations

import time
from collections.abc import Callable
from typing import Any

from .base import (
    Backend,
    Request,
    Response,
)


class EchoBackend(Backend):
    """Deterministic offline stand-in.  Exists so the whole pipeline --
    grid, runner, cache, frame, CIs, plots -- can be exercised end to end
    with zero API spend and zero network, which is the only way a dry run
    actually tests the harness rather than the vendor."""

    name = "echo"

    def __init__(self, fn: Callable[[Request], str] | None = None,
                 latency: float = 0.0) -> None:
        self.fn = fn or (lambda r: r.messages[-1]["content"][::-1])
        self.latency = latency

    def identity(self) -> dict[str, Any]:
        # Two echo backends with different functions are different models.
        # Identified by name, so editing a function's body is not detected.
        fn = self.fn
        return {**self._base_identity(),
                "fn": f"{getattr(fn, '__module__', '?')}."
                      f"{getattr(fn, '__qualname__', type(fn).__qualname__)}"}

    def complete(self, req: Request) -> Response:
        if self.latency:
            time.sleep(self.latency)
        return Response(req.unit_id, text=self.fn(req),
                        meta={"backend": self.name})
