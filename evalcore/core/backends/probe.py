"""What to capture from inside a forward pass of a local model.

A `Probe` is handed to `HFBackend(probe=...)`.  Around EVERY forward pass
the backend

    1. calls `start(enc, reqs)`        -- the batch layout, before the pass
    2. installs `hooks()`              -- via core.hooks.hooked: patterns
                                          must match, removal guaranteed
    3. runs the forward
    4. checks each hooked module ran `calls_per_forward` times
    5. calls `collect(out, enc, reqs)` -- one dict per request

so the hooks see the whole padded batch, and `collect` splits it per row
using `enc`, which gives the padding side and, per row, either `last` (the
final real position) or `n_prompt`.  This makes capture correct at any
batch size: no probe has to assume that its request is row 0, or that the
last column is a real token.

Under `HFBackend(prefix_cache=True)` the first forward is fed only the
uncached tail of the prompt.  `enc` still describes the FULL prompt
(input_ids and attention_mask cover cached + fed positions), and
`enc["n_prefix_reused"][k]` says how many leading positions were not fed.
On the first forward, full position p is hook index p - n_prefix_reused[k],
and the read position (the last prompt token) is always the last fed
index.  `enc["n_prefix_reused"]` is present on every path (zeros
without reuse), so one probe works either way.  A probe may patch the
read position and generated tokens, never an earlier fed position: those
are stored for reuse by later calls.  HFBackend._probing has the full
key-by-key contract.

`identity()` is abstract because what a probe captures is part of what a
cached response contains: a changed probe must not be served old arrays.
Arrays returned by `collect` go to the run's sidecar store
(core/records.py), not into the JSONL row.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from typing import Any

from .base import qualified_name

__all__ = ["FunctionProbe", "Probe"]


class Probe(ABC):
    pre: bool = False                # pre-forward hooks instead of forward hooks
    with_kwargs: bool = False        # hooks receive the module's kwargs
    # How many times each hooked module must run per forward pass.  None
    # disables the check (e.g. a module reused a variable number of times).
    calls_per_forward: int | None = 1

    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """What determines the captured values (layers, positions, reductions)."""

    def hooks(self) -> Mapping[str, Callable[..., Any]]:
        """{module name or glob pattern: fn(name, module, *hook_args)}."""
        return {}

    def start(self, enc: Mapping[str, Any], reqs: Sequence[Any]) -> None:  # noqa: B027 - optional no-op
        """Called before each forward pass.  Reset per-batch state here."""

    @abstractmethod
    def collect(self, out: Any, enc: Mapping[str, Any],
                reqs: Sequence[Any]) -> list[dict[str, Any]]:
        """One dict per request, after the forward pass."""


class FunctionProbe(Probe):
    """A hook-less probe from one function `fn(out, enc, reqs)`, which reads
    the model output (hidden states, attentions, logits) after the pass.
    Identified by the function's qualified name: rename it, or add a
    version to the name, when what it returns changes."""

    def __init__(self, fn: Callable[..., list[dict[str, Any]]]) -> None:
        self.fn = fn

    def identity(self) -> dict[str, Any]:
        return {"function": qualified_name(self.fn)}

    def collect(self, out: Any, enc: Mapping[str, Any],
                reqs: Sequence[Any]) -> list[dict[str, Any]]:
        return list(self.fn(out, enc, reqs))
