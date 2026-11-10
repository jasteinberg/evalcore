"""One experiment, start to finish.

    exp = Experiment(task, backend, cells, out="runs/sweep.jsonl")
    res = exp.run()        # calibrate -> dry_run -> execute -> analyse -> report

An `Experiment` holds what the experiment is (task, backend, cells,
repeats) and how to run it (`RunOptions`).  Each stage is a plain function
field with one signature, `stage(exp, res) -> None`: it reads what earlier
stages put in `res` and adds its own.  Replace a stage by passing a
function, or switch it off with None (`execute=None` re-analyses an
existing sink without calling the model).  A stage may halt the run by
setting `res.halted` -- the dry run does, over budget -- and the report is
still written, so a halted run is documented too.

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
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd

from .analyses import Analysis, AnalysisContext, attrition_table, summary
from .core.backends import Backend, Cache, request_key
from .core.calibrate import Calibration, calibrate, derive
from .core.events import events_path, summarize_events
from .core.records import manifest_path, to_frame, utc_now, write_atomic
from .core.runner import Plan, execute, plan, rendered
from .core.spec import Cell, Unit, digest
from .tasks.base import Task

__all__ = ["Experiment", "Results", "RunOptions", "Stage", "job_status",
           "read_marker", "write_marker"]


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
    max_cost: float | None = None           # dry run halts above this, or
    confirm: bool = False                   # when it cannot estimate, unless
                                            # confirmed.  Needs prices.
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
    job_hash: str | None = None
    skipped: str | None = None            # why the model stages did not run
    marker: dict[str, Any] | None = None  # what <sink>.done now says


# --- default stages -----------------------------------------------------------

def calibrate_stage(exp: Experiment, res: Results) -> None:
    """Measure the backend on the experiment's own requests -- interleaved
    across cells -- or reuse a saved calibration of the same backend
    identity that is fresh enough, then derive run settings.  Explicit
    RunOptions win over derived ones, and the record says which applied."""
    path = exp.path("calibration.json")
    cal = None if exp.options.recalibrate else _saved_calibration(exp, path)
    if cal is None:
        # interleaved across cells, so every cell's prompts are sampled.  A
        # local model needs enough requests for its batch-size ladder
        by_cell: dict[str, list[Unit]] = {}
        for u in exp.units():
            by_cell.setdefault(u.cell.id, []).append(u)
        interleaved = [u for group in zip_longest(*by_cell.values())
                       for u in group if u is not None]
        k = 32 if exp.backend.supports_target else exp.options.calibrate_n
        sample = [rendered(exp.task.render, u) for u in interleaved[:k]]
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
    are measured per call).  Halts over `max_cost` unless confirmed, and
    also when the calibration measured no token counts, so the cost of the
    calls still to make cannot be estimated: an unchecked budget is not a
    budget."""
    o = exp.options
    p = plan(exp.units(), exp.backend, exp.task.render, exp.out, o.resume,
             o.retry_errors, o.limit)
    cache = exp.cache()
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
    budgeted = o.max_cost is not None and not o.confirm
    if budgeted and to_call and "est_cost" not in d:
        res.halted = (f"max_cost ${o.max_cost:.2f} cannot be checked: the "
                      f"calibration measured no token counts for "
                      f"{exp.backend.name}. Rerun with confirm=True to proceed "
                      f"unbudgeted")
    elif budgeted and d.get("est_cost", 0.0) > o.max_cost:
        res.halted = (f"estimated cost ${d['est_cost']:.2f} exceeds max_cost "
                      f"${o.max_cost:.2f}. Rerun with confirm=True to proceed")
    elif o.dry_run_only:
        res.halted = "dry run only"


def execute_stage(exp: Experiment, res: Results) -> None:
    o, s = exp.options, res.settings
    res.frame = execute(
        exp.units(), exp.backend, exp.task.render, exp.task.score, exp.out,
        cache=exp.cache(),
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
        "job_hash": res.job_hash, "skipped": res.skipped,
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
            raise ValueError("give metric (shorthand) or analyses, not both. "
                             "Add summary(metric) to analyses instead")
        o = self.options
        # with execute off nothing is spent, so there is no budget to check
        budgeted = o.max_cost is not None and self.execute is not None
        if budgeted and not o.prices:
            raise ValueError("max_cost needs prices ({'in', 'out'}: $ per 1M "
                             "tokens). Without them the budget cannot be "
                             "checked")
        if budgeted and (self.calibrate is None or self.dry_run is None):
            raise ValueError("max_cost is checked by the dry run against the "
                             "calibration. With the calibrate or dry_run "
                             "stage off the budget cannot be checked")
        # a request the backend cannot serve fails here, before any spend:
        # every unit's, since one item with a target among plain ones
        # would otherwise surface only as error rows after the spend
        refused: dict[str, list[str]] = {}
        all_units = self.units()
        for u in all_units:
            problem = self.backend.refusal(rendered(self.task.render, u))
            if problem:
                refused.setdefault(problem, []).append(u.id)
        if refused:
            n = sum(map(len, refused.values()))
            problem, ids = next(iter(refused.items()))
            more = (f" ({len(refused)} distinct reasons)" if len(refused) > 1
                    else "")
            raise ValueError(f"{n} of {len(all_units)} requests cannot be "
                             f"served{more}, e.g. unit {ids[0]}: {problem}")

    def units(self) -> list[Unit]:
        return self.task.units(self.cells, self.repeats)

    def planned(self) -> list[Unit]:
        """The units a run plans: all of them, or the first `limit`."""
        return self.units()[:self.options.limit]

    def cache(self) -> Cache | None:
        return Cache(self.options.cache_dir) if self.options.cache_dir else None

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

    def job_hash(self) -> str:
        """Content address of the job: what the run is, not how it is run.

            job_hash = digest(task name and version, items, cells, repeats,
                              backend identity, batch_size, limit,
                              digest(sorted request_keys))

        Items enter in full, so a gold answer that never reaches the prompt
        still changes the job.  The request keys cover each rendered prompt
        (a template edit with no version bump), the backend's identity and
        defaults, and the repeat index.  batch_size enters because padding
        can change a local model's numerics, and limit because it changes
        which units are planned.  Workers, retries, caching and the code
        commit do not: they change how a result is obtained, not what it is
        (the commit is recorded in the marker)."""
        o = self.options
        keys = sorted(request_key(rendered(self.task.render, u), u.repeat,
                                  self.backend)
                      for u in self.planned())
        return digest({
            "task": self.task.name, "task_version": self.task.version,
            "items": [[i.item_id, dict(i.payload), i.group, i.parent_id]
                      for i in self.task.items],
            "cells": [dict(c.params) for c in self.cells],
            "repeats": self.repeats, "backend": self.backend.identity(),
            "batch_size": o.batch_size, "limit": o.limit,
            "requests": digest(keys)})

    def run(self) -> Results:
        """Run every enabled stage in order, and after a halt only the report.

        Completion marking: `<sink>.done` records the job hash and a status.
        If it already holds this job's hash with status "complete" and
        resume is on, the model-facing stages (calibrate, dry run, execute)
        are skipped and only analyse and report run.  Otherwise the marker
        is set to "running" before the first model-facing stage, so a crash
        leaves "running", never a stale "complete", and is rewritten with
        the final status at the end (see `job_status`)."""
        assert isinstance(self.out, Path)
        res = Results(self.out)
        res.job_hash = self.job_hash()
        done = self.path("done")
        prior = read_marker(done)
        # the sink is re-read rather than trusted: one deleted or edited by
        # hand since the marker was written must not be reported as done
        if (self.options.resume and prior is not None
                and prior["job_hash"] == res.job_hash
                and prior["status"] == "complete"
                and job_status(self.out, self.planned())[0] == "complete"):
            res.skipped = (f"job {res.job_hash} already complete "
                           f"({prior['finished_at']})")
        model_stages = ("calibrate", "dry_run", "execute")
        for name in self.STAGES:
            stage = getattr(self, name)
            if (stage is None or (res.halted and name != "report")
                    or (res.skipped and name in model_stages)):
                continue
            if name in model_stages:
                write_marker(done, {"job_hash": res.job_hash,
                                    "status": "running",
                                    "started_at": utc_now()})
            stage(self, res)
        if res.skipped:
            res.marker = prior
            return res
        status, counts = job_status(self.out, self.planned())
        prov = provenance()
        res.marker = {"job_hash": res.job_hash,
                      "status": "halted" if res.halted else status,
                      "halted": res.halted, "counts": counts,
                      "finished_at": utc_now(), "commit": prov["commit"],
                      "dirty": prov["dirty"]}
        write_marker(done, res.marker)
        return res


# --- completion marking ------------------------------------------------------------

RESULT_STATUSES = frozenset({"ok", "truncated", "unparsed"})


def job_status(out: Path, planned: Sequence[Unit]
               ) -> tuple[str, dict[str, int]]:
    """Status of the planned units from the sink, and the counts behind it.

    Each planned unit takes the status of its last row (to_frame is
    last-wins), or "missing" if it has none.  "complete": every unit has a
    result (ok, truncated or unparsed -- a result the scorer saw or
    deliberately did not score).  "complete_with_errors": every unit has a
    row, and some are error or tool_error.  "incomplete": some unit has no
    row at all."""
    last: dict[str, str] = {}
    if out.exists():
        frame = to_frame(out)
        if len(frame):
            last = dict(zip(frame["unit_id"], frame["status"], strict=True))
    counts = Counter(last.get(u.id, "missing") for u in planned)
    if counts.get("missing"):
        status = "incomplete"
    elif set(counts) <= RESULT_STATUSES:
        status = "complete"
    else:
        status = "complete_with_errors"
    return status, {"planned": len(planned), **dict(sorted(counts.items()))}


def read_marker(path: Path) -> dict[str, Any] | None:
    """The marker's record, or None if there is none."""
    if not path.exists():
        return None
    return cast(dict[str, Any], json.loads(path.read_text()))


def write_marker(path: Path, record: Mapping[str, Any]) -> None:
    """Replace the marker atomically: write a sibling, then rename over it,
    so a reader sees the old record or the new one, never half of one."""
    write_atomic(path, lambda tmp: tmp.write_text(
        json.dumps(dict(record), indent=1, default=str)))


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
    return {"time": utc_now(),
            "evalcore": __version__, "commit": commit, "dirty": dirty,
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": versions}
