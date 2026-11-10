"""A scripted model and the toy Experiment built on it, for pipeline tests.

The model reads the integer at the end of the prompt, answers it on even
items and -1 on odd ones (so accuracy is exactly 1/2 on 0..n-1, n even),
reports 100 in / 5 out tokens per call, and counts every call by prompt."""

from __future__ import annotations

from collections import Counter
from collections.abc import Sequence
from pathlib import Path

from evalcore import Cell, Fatal, FunctionBackend, Item, Request, Response, grid
from evalcore.pipeline import Experiment
from evalcore.tasks.base import Task


class Counting:
    """Correct on even items, counts calls, and refuses the items in `fail`.
    `tag` enters the backend identity, so two tags are two backends."""

    def __init__(self, fail: Sequence[int] = (), tag: str = "v1") -> None:
        self.calls: Counter[str] = Counter()
        self.fail, self.tag = set(fail), tag

    def __call__(self, req: Request) -> Response:
        q = req.messages[-1]["content"]
        self.calls[q] += 1
        i = int(q.split()[-1])
        if i in self.fail:
            raise Fatal("refused")
        return Response(req.unit_id, text=str(i if i % 2 == 0 else -1),
                        meta={"truncated": False, "in_tokens": 100,
                              "out_tokens": 5})


def items(n: int = 6, gold_shift: int = 0,
          template: str = "what is {i}") -> list[Item]:
    """Items q0..q{n-1}, numeric grader.  `gold_shift` moves the gold
    without touching the prompt."""
    return [Item(f"q{i}", {"messages": [{"role": "user",
                                         "content": template.format(i=i)}],
                           "gold": i + gold_shift, "grader": "numeric"})
            for i in range(n)]


def make(tmp_path: Path, fn: Counting | None = None,
         its: list[Item] | None = None, model: str = "scripted",
         cells: list[Cell] | None = None, **kw) -> tuple[Experiment, Counting]:
    """A one-cell Experiment over `its` (default: six items) writing
    tmp_path/r.jsonl, scored by `correct`.  Returns it with its model."""
    fn = fn or Counting()
    exp = Experiment(Task("toy", its or items()),
                     FunctionBackend(fn, identity={"model": model,
                                                   "tag": fn.tag}),
                     cells or grid({"model": ["toy"]}), tmp_path / "r.jsonl",
                     metric="correct", **kw)
    return exp, fn
