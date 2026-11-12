"""A task: the items, how each becomes a request, and how a response is
scored, held in one object instead of passed to `execute` as three
separate arguments."""

from __future__ import annotations

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.backends import Backend, Request, Response
from ..core.runner import execute
from ..core.spec import Cell, Item, Unit, units
from .formats import load_tasks, render_canonical
from .graders import make_score

__all__ = ["Task"]


@dataclass(frozen=True)
class Task:
    """`render` and `score` default to the canonical ones (an item's own
    messages and grader), so a task loaded from a file needs no code.

    `name` and `version` are written into every row and the plan manifest
    as cell TAGS.  Tags are not part of the request key, so renaming or
    re-versioning a task never invalidates the cache.  Changing what a task
    asks changes the rendered requests, and resume re-runs exactly those
    units on its own (the request key covers them).  Bump `version` when
    the scoring changes: `run` (and the pipeline) pass it to `execute` as
    `score_version`, so rows scored under the old version are re-scored
    from their stored responses, with no model call."""

    name: str
    items: Sequence[Item]
    render: Callable[[Unit], Request] = render_canonical
    score: Callable[[Unit, Response], dict[str, Any]] = field(
        default_factory=make_score)
    version: str = "1"

    @classmethod
    def from_file(cls, path: str | Path, name: str | None = None,
                  mapping: Mapping[str, Any] | Callable | None = None,
                  **kw: Any) -> Task:
        return cls(name or Path(path).stem, load_tasks(path, mapping), **kw)

    def tagged(self, cells: Iterable[Cell]) -> list[Cell]:
        return [Cell(c.params, {**c.tags, "task": self.name,
                                "task_version": self.version})
                for c in cells]

    def units(self, cells: Iterable[Cell], repeats: int = 1) -> list[Unit]:
        return list(units(self.tagged(cells), list(self.items), repeats))

    def run(self, cells: Iterable[Cell], backend: Backend, out: str | Path,
            repeats: int = 1, **kw: Any) -> pd.DataFrame:
        """`runner.execute` over this task's units.  Keyword arguments pass
        through (cache, workers, batch_size, resume, ...)."""
        kw.setdefault("score_version", self.version)
        return execute(self.units(cells, repeats), backend, self.render,
                       self.score, out, **kw)
