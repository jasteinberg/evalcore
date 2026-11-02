"""Fakes for the runner tests: items, a render and a scorer, and scripted
backends whose behaviour is fixed by construction (which unit asked what,
how many calls arrived), so a test's expected values never come from the
code under test."""

from __future__ import annotations

import io
import json
import threading
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

import evalcore.core.backends.base as backends_mod
from evalcore import Backend, Item, Request, Response, Unit, execute

# Two values per decode key, chosen to differ; nested values included because
# `stop` is a list and a key that only looked at its first element would pass
# a scalar-only test.
DECODE_VALUES: dict[str, tuple[Any, Any]] = {
    "temperature": (0.0, 0.7), "top_p": (1.0, 0.9), "top_k": (1, 40),
    "max_tokens": (8, 256), "seed": (0, 1), "stop": (["\n"], ["\n", "##"]),
    "stop_sequences": (["</a>"], ["</b>"]), "repetition_penalty": (1.0, 1.1),
}


def items(n: int, aug: int = 0) -> list[Item]:
    """`n` source items q0..q{n-1} asking "question i", each followed by
    `aug` augmentations q{i}-a{a} in its cluster (parent_id = q{i})."""
    out = []
    for i in range(n):
        out.append(Item(f"q{i}", {"q": f"question {i}"}))
        out += [Item(f"q{i}-a{a}", {"q": f"QUESTION {i} v{a}"}, parent_id=f"q{i}")
                for a in range(aug)]
    return out


def render(u: Unit) -> Request:
    return Request(u.id, [{"role": "user", "content": u.item.payload["q"]}],
                   dict(u.cell.params))


def echo_score(u: Unit, r: Response) -> dict[str, Any]:
    return {"answer": r.text}


def raw_rows(path: Path) -> list[dict]:
    """The sink as written.  `to_frame` deduplicates on unit_id, so a frame
    cannot show a duplicated row even when the sink holds one."""
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def cache_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.json"))


def run(us: list[Unit], backend: Backend, out: Path, **kw: Any) -> Any:
    """`execute` with one worker, a silent log, and `echo_score` unless a
    `score=` is given."""
    return execute(us, backend, render, kw.pop("score", echo_score), out,
                   workers=1, stream=kw.pop("stream", io.StringIO()), **kw)


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries sleep U(0, 2^k) s; the schedule is not what is under test."""
    monkeypatch.setattr(backends_mod.time, "sleep", lambda s: None)


class Answers(Backend):
    """Answers each request with its own prompt, so a mis-join is visible as
    a row whose `answer` names another item.  Counts calls per prompt."""

    name = "answers"

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.lock = threading.Lock()

    def identity(self) -> dict:
        return {"backend": "answers"}

    def complete(self, req: Request) -> Response:
        q = req.messages[-1]["content"]
        with self.lock:
            self.calls[q] += 1
        return Response(req.unit_id, text=q)


class Truncates(Backend):
    """Cut off at the budget: every odd-numbered item, or every item when
    `odd_only` is False.  Counts calls in `n`."""

    def __init__(self, odd_only: bool = True) -> None:
        self.odd_only, self.n = odd_only, 0

    def identity(self) -> dict:
        return {"backend": "truncates"}

    def complete(self, req: Request) -> Response:
        self.n += 1
        i = int(req.messages[-1]["content"].split()[-1])
        return Response(req.unit_id, text="The answer is",
                        meta={"truncated": i % 2 == 1 or not self.odd_only})


class Dies(Backend):
    """Answers `alive` requests, then dies -- a kill, not an error row.

    It raises KeyboardInterrupt rather than a plain exception on purpose:
    `with_retries` turns any Exception into an error ROW, which is the design
    commitment this package is built on, so the never-attempted category can
    only ever be produced by the process itself going away.  Simulating the
    kill with anything catchable would test the wrong path.

    `fail_on` names item_ids that come back as error RESPONSES instead --
    the other, already-visible category; the two must not be conflated.
    `on_call(k)` runs before the k-th call.
    """

    name = "dies"

    def __init__(self, alive: int, fail_on: set[str] = frozenset(),
                 on_call: Callable[[int], None] | None = None) -> None:
        self.alive, self.fail_on, self.on_call = alive, set(fail_on), on_call
        self.calls = 0

    def identity(self) -> dict:
        return {"backend": "dies"}

    def complete(self, req: Request) -> Response:
        if self.on_call is not None:
            self.on_call(self.calls)
        if self.calls >= self.alive:
            raise KeyboardInterrupt("process killed mid-sweep")
        self.calls += 1
        if req.messages[-1]["content"] in {f"question {int(t[1:])}"
                                           for t in self.fail_on}:
            return Response(req.unit_id, error="fatal: refused")
        return Response(req.unit_id, text=req.messages[-1]["content"][::-1])
