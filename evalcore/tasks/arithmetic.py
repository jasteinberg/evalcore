r"""Few-shot integer arithmetic, scored by teacher forcing.

Each item is n_shots worked examples and a query,

    "a1 + b1 = c1\n ... a + b =",       target " c",

with d-digit operands drawn uniformly (the leading digit non-zero for
d > 1).  The target begins with a space and the prompt ends without one, so
the answer starts on a token boundary for byte-level BPE tokenisers; the
backend verifies that per item rather than assuming it.

Few-shot leakage.  Shots drawn independently of the query sometimes carry
the query's answer verbatim (for d = 2, about 2.7% of items: four shots,
each matching one of ~180 possible sums), and small models copy it.  At the
foot of an emergence curve the copied answers can be a large share of the
correct ones.  `exclude_leaks=True` (the default) redraws any shot whose
answer equals the query's, or which is the query itself; every item records
whether it leaked in `meta["leak"]`, so `leak_rate` can audit any dataset.
`exclude_leaks=False` reproduces the earlier generator's draws exactly
(shots first, then the query, from one `random.Random(seed)`), for
comparison with results produced from it.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from typing import Any

from ..core.spec import Item
from .base import Task
from .formats import render_canonical
from .graders import score_teacher_forced

__all__ = ["arithmetic_items", "arithmetic_task", "leak_rate"]

_OPS = {"+": lambda a, b: (a, b, a + b),
        "-": lambda a, b: (max(a, b), min(a, b), max(a, b) - min(a, b)),
        "*": lambda a, b: (a, b, a * b)}


def _draw(rng: random.Random, digits: int, op: str) -> tuple[int, int, int]:
    lo = 10 ** (digits - 1) if digits > 1 else 0
    hi = 10 ** digits - 1
    a = rng.randint(lo, hi)
    b = rng.randint(lo, hi)
    return _OPS[op](a, b)


def _answer(c: int, spaced: bool) -> str:
    return " ".join(str(c)) if spaced else str(c)


def _line(a: int, b: int, op: str) -> str:
    return f"{a} {op} {b} ="


def arithmetic_items(digits: int = 2, n_items: int = 1024, n_shots: int = 4,
                     op: str = "+", seed: int = 0, spaced: bool = False,
                     exclude_leaks: bool = True) -> list[Item]:
    """Deterministic items.  Queries are unique while the question space
    allows (d = 1 has only 100 pairs; then duplicates are admitted rather
    than looping forever).  `spaced` renders answers digit by digit, which
    makes the answer length equal its digit count."""
    if op not in _OPS:
        raise ValueError(f"op must be one of {sorted(_OPS)}")
    lo = 10 ** (digits - 1) if digits > 1 else 0
    space = (10 ** digits - lo) ** 2
    rng = random.Random(seed)
    items: list[Item] = []
    seen: set[tuple[int, int]] = set()
    while len(items) < n_items:
        shots = [_draw(rng, digits, op) for _ in range(n_shots)]
        a, b, c = _draw(rng, digits, op)
        if (a, b) in seen and len(seen) < space:
            continue
        seen.add((a, b))
        if exclude_leaks:
            for k in range(len(shots)):
                while shots[k][2] == c or shots[k][:2] == (a, b):
                    shots[k] = _draw(rng, digits, op)
        leak = any(s[2] == c or s[:2] == (a, b) for s in shots)
        prompt = "".join(f"{_line(*s[:2], op)} {_answer(s[2], spaced)}\n"
                         for s in shots) + _line(a, b, op)
        meta: dict[str, Any] = {"a": a, "b": b, "answer": c, "op": op,
                                "digits": digits, "n_shots": n_shots,
                                "spaced": spaced, "leak": leak}
        items.append(Item(f"{op}d{digits}s{n_shots}-{seed}-{len(items):05d}",
                          {"messages": [{"role": "user", "content": prompt}],
                           "target": f" {_answer(c, spaced)}",
                           "gold": _answer(c, spaced), "meta": meta}))
    return items


def leak_rate(items: Sequence[Item]) -> float:
    """Fraction of items whose answer appears verbatim among their shots."""
    return sum(bool(i.payload["meta"]["leak"]) for i in items) / len(items)


def arithmetic_task(digits: int = 2, n_items: int = 1024, n_shots: int = 4,
                    op: str = "+", seed: int = 0, spaced: bool = False,
                    exclude_leaks: bool = True) -> Task:
    """The items as a Task, scored by teacher forcing (needs a backend with
    supports_target, i.e. HFBackend)."""
    items = arithmetic_items(digits, n_items, n_shots, op, seed, spaced,
                             exclude_leaks)
    name = {"+": "add", "-": "sub", "*": "mul"}[op]
    return Task(f"{name}_d{digits}", items, render=render_canonical,
                score=score_teacher_forced)
