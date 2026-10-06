"""One experiment, start to finish.

    exp = Experiment(task, backend, cells, out="runs/sweep.jsonl")
    res = exp.run()        # calibrate -> dry_run -> execute -> analyse -> report

An `Experiment` holds what the experiment is (task, backend, cells,
repeats) and how to run it (`RunOptions`).  Each stage is a plain function
field with one signature, `stage(exp, res) -> None`: it reads what earlier
stages put in `res` and adds its own.  Replace a stage by passing a
function; switch it off with None (`execute=None` re-analyses an existing
sink without calling the model).  A stage may halt the run by setting
`res.halted` -- the dry run does, over budget -- and the report is still
written, so a halted run is documented too.

What the analyse stage computes is itself pluggable: `analyses` maps a
name to any `analysis(frame, ctx) -> table` (analyses.py has the usual
ones).  `metric="correct"` is shorthand for attrition plus the bootstrap
summary of that column.
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from .analyses import Analysis, AnalysisContext, attrition_table, summary
from .core.backends import Backend, Cache
from .core.calibrate import Calibration, calibrate, derive
from .core.events import events_path, summarize_events
from .core.records import manifest_path, to_frame
from .core.runner import Plan, execute, plan
from .core.spec import Cell, Unit, digest
from .tasks.base import Task

__all__ = ["Experiment", "Results", "RunOptions", "Stage"]


class Stage(Protocol):
    def __call__(self, exp: Experiment, res: Results) -> None: ...


@dataclass
class RunOptions:
    """How to run.  None for workers / batch_size means: from calibration
    if it ran, else 8 workers and batch 1."""

    workers: int | None = None
    batch_size: int | None = None
    attempts: int = 5
    resume: bool = True
    retry_errors: bool = True
    limit: int | None = None
    cache_dir: str | Path | None = None
    events: bool = True
    calibrate_n: int = 5                    # real calls spent on calibration
    recalibrate: bool = False               # ignore a saved calibration
    calibration_max_age_h: float = 24.0
    prices: dict[str, float] | None = None  # $ per 1M tokens: {"in", "out"}
    max_cost: float | None = None           # dry run halts above this ...
    confirm: bool = False                   # ... unless confirmed
    dry_run_only: bool = False


@dataclass
class Results:
    """What each stage produced, in stage order."""

    out: Path
    calibration: Calibration | None = None
    settings: dict[str, Any] = field(default_factory=dict)
    plan: Plan | None = None
    dry_run: dict[str, Any] | None = None
    frame: pd.DataFrame | None = None
    analysis: dict[str, pd.DataFrame] = field(default_factory=dict)
    halted: str | None = None
    report_path: Path | None = None


# --- default stages -----------------------------------------------------------

def calibrate_stage(exp: Experiment, res: Results) -> None:
    """Measure the backend on the experiment's own requests -- interleaved
    across cells -- or reuse a saved calibration of the same backend
    identity that is fresh enough; then derive run settings.  Explicit
    RunOptions win over derived ones, and the record says which applied."""
    path = exp.path("calibration.json")
    cal = None if exp.options.recalibrate else _saved_calibration(exp, path)
    if cal is None:
        # interleaved across cells, so every cell's prompts are sampled; a
        # local model needs enough requests for its batch-size ladder
        by_cell: dict[str, list[Unit]] = {}
        for u in exp.units():
            by_cell.setdefault(u.cell.id, []).append(u)
        interleaved = [u for group in zip_longest(*by_cell.values())
                       for u in group if u is not None]
        k = 32 if exp.backend.supports_target else exp.options.calibrate_n
        sample = [exp.task.render(u) for u in interleaved[:k]]
        cal = calibrate(exp.backend, sample, n=exp.options.calibrate_n)
        path.write_text(json.dumps(cal.to_dict(), indent=1))
    res.calibration = cal
    derived = derive(cal)
    o = exp.options
    res.settings = {
        "workers": o.workers if o.workers is not None else derived["workers"],
        "batch_size": (o.batch_size if o.batch_size is not None
                       else derived["batch_size"]),
        "derived": derived}


def _saved_calibration(exp: Experiment, path: Path) -> Calibration | None:
    if not path.exists():
        return None
    d = json.loads(path.read_text())
    age_h = (datetime.now(timezone.utc)
             - datetime.fromisoformat(d["measured_at"])).total_seconds() / 3600
    if (d["identity"] != digest(exp.backend.identity())
            or age_h > exp.options.calibration_max_age_h):
        return None
    d["throughput"] = {int(k): v for k, v in d.get("throughput", {}).items()}
    return Calibration(**d)


def dry_run_stage(exp: Experiment, res: Results) -> None:
    """The resume decisions and the spend, without calling the model:
    units to call, how many are already cached, and estimated tokens and
    cost from the calibration (output tokens are an estimate, input tokens
    are measured per call).  Halts over `max_cost` unless confirmed."""
    o = exp.options
    p = plan(exp.units(), exp.backend, exp.task.render, exp.out, o.resume,
             o.retry_errors, o.limit)
    cache = Cache(o.cache_dir) if o.cache_dir else None
    cached = sum(1 for u in p.todo if cache and cache.has(p.keys[u.id]))
    to_call = len(p.todo) - cached
    d: dict[str, Any] = {"units": len(p.units), "done": p.done,
                         "to_go": len(p.todo), "cached": cached,
                         "to_call": to_call, "stale": p.stale,
                         "retried": p.retried}
    cal = res.calibration
    if cal and cal.in_tokens is not None:
        d["est_in_tokens"] = cal.in_tokens * to_call
        d["est_out_tokens"] = (cal.out_tokens or 0.0) * to_call
        if o.prices:
            d["est_cost"] = (d["est_in_tokens"] * o.prices["in"]
                             + d["est_out_tokens"] * o.prices["out"]) / 1e6
    res.plan, res.dry_run = p, d
    if o.max_cost is not None and d.get("est_cost", 0.0) > o.max_cost \
            and not o.confirm:
        res.halted = (f"estimated cost ${d['est_cost']:.2f} exceeds max_cost "
                      f"${o.max_cost:.2f}; rerun with confirm=True to proceed")
    elif o.dry_run_only:
        res.halted = "dry run only"


def execute_stage(exp: Experiment, res: Results) -> None:
    o, s = exp.options, res.settings
    res.frame = execute(
        exp.units(), exp.backend, exp.task.render, exp.task.score, exp.out,
        cache=Cache(o.cache_dir) if o.cache_dir else None,
        workers=s.get("workers") or o.workers or 8,
        batch_size=s.get("batch_size") or o.batch_size or 1,
        attempts=o.attempts, resume=o.resume, retry_errors=o.retry_errors,
        limit=o.limit, events=o.events)


def analyse_stage(exp: Experiment, res: Results) -> None:
    """Every analysis in `exp.all_analyses()` over the results frame,
    written as CSV beside the sink (a dict result as `<name>.<key>.csv`)."""
    assert isinstance(exp.out, Path)
    frame = res.frame if res.frame is not None else to_frame(exp.out)
    ctx = AnalysisContext(exp.axes(), manifest_path(exp.out), exp.out)
    for name, analysis in exp.all_analyses().items():
        out = analysis(frame, ctx)
        if isinstance(out, pd.DataFrame):
            res.analysis[name] = out
        else:
            res.analysis.update({f"{name}.{k}": t for k, t in out.items()})
    adir = exp.path("analysis")
    adir.mkdir(exist_ok=True)
    for name, table in res.analysis.items():
        table.to_csv(adir / f"{name}.csv", index=False)


def report_stage(exp: Experiment, res: Results) -> None:
    """run.json: what was run, how, what each stage found, where the outputs
    are, and the provenance needed to reproduce it."""
    ev = events_path(exp.out)
    report = {
        "experiment": exp.describe(),
        "options": asdict(exp.options),
        "calibration": res.calibration.to_dict() if res.calibration else None,
        "settings": res.settings,
        "dry_run": res.dry_run,
        "halted": res.halted,
        "events": summarize_events(ev) if ev.exists() else None,
        "outputs": {"sink": str(exp.out), "events": str(ev),
                    "manifest": str(manifest_path(exp.out)),
                    "analysis": sorted(str(exp.path("analysis") / f"{k}.csv")
                                       for k in res.analysis)},
        "provenance": provenance(),
    }
    res.report_path = exp.path("run.json")
    res.report_path.write_text(json.dumps(report, indent=1, default=str))


# --- the experiment -------------------------------------------------------------

@dataclass
class Experiment:
    task: Task
    backend: Backend
    cells: Sequence[Cell]
    out: str | Path
    repeats: int = 1
    metric: str | None = None              # shorthand: attrition + summary
    analyses: Mapping[str, Analysis] | None = None
    options: RunOptions = field(default_factory=RunOptions)
    calibrate: Stage | None = calibrate_stage
    dry_run: Stage | None = dry_run_stage
    execute: Stage | None = execute_stage
    analyse: Stage | None = analyse_stage
    report: Stage | None = report_stage

    STAGES = ("calibrate", "dry_run", "execute", "analyse", "report")

    def __post_init__(self) -> None:
        self.out = Path(self.out)
        self.out.parent.mkdir(parents=True, exist_ok=True)
        self.backend.identity()             # refuse an unidentified backend now
        if not self.cells or not self.task.items:
            raise ValueError("an experiment needs at least one cell and one item")
        if self.metric is not None and self.analyses is not None:
            raise ValueError("give metric (shorthand) or analyses, not both; "
                             "add summary(metric) to analyses instead")
        # a task the backend cannot serve at all fails here, before any spend
        problem = self.backend.refusal(self.task.render(self.units()[0]))
        if problem:
            raise ValueError(problem)

    def units(self) -> list[Unit]:
        return self.task.units(self.cells, self.repeats)

    def axes(self) -> list[str]:
        """The grid axes, which every analysis groups by (["task"] for a
        grid of one empty cell)."""
        return sorted({k for c in self.cells for k in c.params}) or ["task"]

    def all_analyses(self) -> dict[str, Analysis]:
        """`analyses` as given, or the default: attrition, plus the summary
        of `metric` if one is named."""
        if self.analyses is not None:
            return dict(self.analyses)
        out: dict[str, Analysis] = {"attrition": attrition_table()}
        if self.metric:
            out["summary"] = summary(self.metric)
        return out

    def path(self, name: str) -> Path:
        """A file beside the sink: `runs/x.jsonl` -> `runs/x.jsonl.<name>`."""
        assert isinstance(self.out, Path)
        return self.out.with_name(f"{self.out.name}.{name}")

    def describe(self) -> dict[str, Any]:
        return {"task": self.task.name, "task_version": self.task.version,
                "n_items": len(self.task.items), "repeats": self.repeats,
                "cells": [dict(c.params) for c in self.cells],
                "backend": self.backend.identity(),
                "analyses": sorted(self.all_analyses())}

    def run(self) -> Results:
        """Run every enabled stage in order; after a halt, only the report."""
        assert isinstance(self.out, Path)
        res = Results(self.out)
        for name in self.STAGES:
            stage = getattr(self, name)
            if stage is None or (res.halted and name != "report"):
                continue
            stage(self, res)
        return res


def provenance() -> dict[str, Any]:
    """Enough to say what produced a result: when, on what, with which
    versions -- including the evalcore commit when run from a checkout."""
    versions = {}
    for mod in ("numpy", "pandas", "scipy", "torch", "transformers", "httpx"):
        m = sys.modules.get(mod)
        if m is not None:
            versions[mod] = getattr(m, "__version__", "?")
    pkg = Path(__file__).resolve().parent
    try:
        commit = subprocess.run(["git", "-C", str(pkg), "rev-parse", "HEAD"],
                                capture_output=True, text=True, timeout=5,
                                check=True).stdout.strip()
        dirty = bool(subprocess.run(
            ["git", "-C", str(pkg), "status", "--porcelain", "--", "."],
            capture_output=True, text=True, timeout=5,
            check=True).stdout.strip())
    except (OSError, subprocess.SubprocessError):
        commit, dirty = None, None
    from . import __version__
    return {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "evalcore": __version__, "commit": commit, "dirty": dirty,
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": versions}
