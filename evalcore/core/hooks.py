"""Forward hooks with the three guarantees a measurement needs.

1. A module name or pattern that matches nothing raises.  A hook on a
   misspelled module never runs, and an intervention that never runs reads
   as a clean null result.
2. Every hook is removed when the block exits, including on an exception
   mid-forward.  A hook left registered contaminates every later forward
   pass in the session.
3. Calls are counted per module, so "each hooked layer ran exactly once per
   forward pass" is an assertion (`check_counts`), not an assumption.

Model-specific capture code -- which tensors to read, which positions --
stays with the caller (typically a Probe handed to HFBackend);
torch is never imported here.
"""

from __future__ import annotations

import fnmatch
from collections import Counter
from collections.abc import Callable, Iterator, Mapping
from contextlib import contextmanager
from typing import Any

__all__ = ["check_counts", "hooked", "match_modules"]


def match_modules(model: Any, pattern: str) -> list[str]:
    """Names of `model.named_modules()` matching `pattern` exactly or as a
    glob ("model.layers.*.self_attn").  Raises if none match."""
    names = [n for n, _ in model.named_modules()]
    hits = [n for n in names if n == pattern or fnmatch.fnmatchcase(n, pattern)]
    if not hits:
        close = [n for n in names if pattern.rsplit(".", maxsplit=1)[-1] in n][:5]
        raise KeyError(f"no module matches {pattern!r}"
                       + (f"; similar names: {close}" if close else ""))
    return hits


@contextmanager
def hooked(model: Any, hooks: Mapping[str, Callable[..., Any]],
           pre: bool = False, with_kwargs: bool = False
           ) -> Iterator[Counter[str]]:
    """Register `fn(name, module, *hook_args)` on every module matching each
    pattern in `hooks`, for the duration of the block.

    Forward hooks get (module, args, output) -- or (module, args, kwargs,
    output) with `with_kwargs` -- and pre-hooks (`pre=True`) get
    (module, args) or (module, args, kwargs); `fn` receives the module's
    name first.  A non-None return replaces the output (or the inputs, for
    pre-hooks), as torch defines.  Yields the per-module call counter."""
    counts: Counter[str] = Counter()
    targets = [(n, fn) for pat, fn in hooks.items()
               for n in match_modules(model, pat)]   # raise before registering
    modules = dict(model.named_modules())
    handles = []
    try:
        for name, fn in targets:
            def wrapped(module: Any, *a: Any, _name: str = name,
                        _fn: Callable[..., Any] = fn) -> Any:
                counts[_name] += 1
                return _fn(_name, module, *a)
            m = modules[name]
            reg = m.register_forward_pre_hook if pre else m.register_forward_hook
            handles.append(reg(wrapped, with_kwargs=with_kwargs))
        yield counts
    finally:
        for h in handles:
            h.remove()


def check_counts(counts: Counter[str], expected: int,
                 names: list[str] | None = None) -> None:
    """Assert every hooked module (or every name in `names`) ran exactly
    `expected` times -- e.g. once per forward pass."""
    names = list(counts) if names is None else names
    bad = {n: counts.get(n, 0) for n in names if counts.get(n, 0) != expected}
    if not names or bad:
        raise AssertionError(f"hook call counts differ from {expected}: "
                             f"{bad or 'no hooks ran'}")
