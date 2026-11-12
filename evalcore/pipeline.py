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

import functools
import hashlib
import importlib.metadata
import json
import platform
import subprocess
import sys
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from itertools import zip_longest
from pathlib import Path
from typing import Any, Protocol, cast

import pandas as pd

from .analyses import Analysis, AnalysisContext, attrition_table, summary
from .core.backends import Backend, Cache, request_key
from .core.calibrate import Calibration, calibrate, derive
from .core.events import events_path, summarize_events
from .core.records import (
    SinkState,
    manifest_path,
    to_frame,
    utc_now,
    write_atomic,
)
from .core.runner import Plan, execute, plan, rendered
from .core.shards import check_shard, combine_shards, shard_of, shard_path
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
    path = exp.calibration_path()
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
        # shards may share this file: replace it whole, never half-written
        text = json.dumps(cal.to_dict(), indent=1)
        write_atomic(path, lambda tmp: tmp.write_text(text))
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
    budget.

    Under a shard, `max_cost` is the budget of the whole sweep, not of each
    shard: the check is against every shard's calls still to make (read
    from all the shard sinks), so n shards cannot spend n times it."""
    o = exp.options
    p = plan(exp.planned(), exp.backend, exp.task.render, exp.out, o.resume,
             o.retry_errors, score_version=exp.task.version,
             batch_size=batch_size_for(exp, res))
    cache = exp.cache()

    def calls(q: Plan) -> int:
        return sum(1 for u in q.todo if not (cache and cache.has(q.keys[u.id])))

    to_call = calls(p)
    cached = len(p.todo) - to_call
    d: dict[str, Any] = {"units": len(p.units), "done": p.done,
                         "to_go": len(p.todo), "cached": cached,
                         "to_call": to_call, "stale": p.stale,
                         "rescore": len(p.rescore),
                         "retried": p.retried}
    budget_calls, scope = to_call, ""
    if exp.shard is not None:
        assert exp.whole is not None
        i, n = exp.shard
        whole = exp.units()[:o.limit]
        budget_calls = to_call + sum(
            calls(plan([u for u in whole if shard_of(u.id, n) == j],
                       exp.backend, exp.task.render,
                       shard_path(exp.whole, j, n), o.resume, o.retry_errors))
            for j in range(n) if j != i)
        d["sweep_to_call"], scope = budget_calls, "sweep_"
    cal = res.calibration
    if cal and cal.in_tokens is not None:
        for pre, k in [("", to_call)] + ([(scope, budget_calls)] if scope
                                         else []):
            d[f"{pre}est_in_tokens"] = cal.in_tokens * k
            d[f"{pre}est_out_tokens"] = (cal.out_tokens or 0.0) * k
            if o.prices:
                d[f"{pre}est_cost"] = (d[f"{pre}est_in_tokens"] * o.prices["in"]
                                       + d[f"{pre}est_out_tokens"]
                                       * o.prices["out"]) / 1e6
    res.plan, res.dry_run = p, d
    budgeted = o.max_cost is not None and not o.confirm
    cost_key = f"{scope}est_cost"
    whose = "the whole sweep's " if scope else ""
    if budgeted and budget_calls and cost_key not in d:
        res.halted = (f"max_cost ${o.max_cost:.2f} cannot be checked: the "
                      f"calibration measured no token counts for "
                      f"{exp.backend.name}. Rerun with confirm=True to proceed "
                      f"unbudgeted")
    elif budgeted and d.get(cost_key, 0.0) > o.max_cost:
        res.halted = (f"{whose}estimated cost ${d[cost_key]:.2f} exceeds "
                      f"max_cost ${o.max_cost:.2f}. Rerun with confirm=True "
                      f"to proceed")
    elif o.dry_run_only:
        res.halted = "dry run only"


def batch_size_for(exp: Experiment, res: Results) -> int:
    """The batch size this invocation runs at: the explicit option, else,
    for a local model (whose numerics padding can change) the one the sink's
    rows were already made at, else calibration's, else 1.  Reusing the
    sink's keeps a resume from re-deriving a different size after the
    calibration expires and redoing every row (plan treats a row of another
    batch size as stale for such a backend)."""
    o = exp.options
    if o.batch_size is not None:
        return o.batch_size
    if exp.backend.supports_target:
        made = Counter(b for b in SinkState(exp.out).batch.values()
                       if b is not None)
        if made:
            return made.most_common(1)[0][0]
    return res.settings.get("batch_size") or 1


def execute_stage(exp: Experiment, res: Results) -> None:
    o, s = exp.options, res.settings
    res.frame = execute(
        exp.units(), exp.backend, exp.task.render, exp.task.score, exp.out,
        cache=exp.cache(),
        workers=s.get("workers") or o.workers or 8,
        batch_size=batch_size_for(exp, res),
        attempts=o.attempts, resume=o.resume, retry_errors=o.retry_errors,
        limit=o.limit, events=o.events, shard=exp.shard,
        score_version=exp.task.version)


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
    are, and the provenance needed to reproduce it.  Beside it, the files
    that make the run reproducible from this directory alone:

        <out>.config.json   the resolved config (from_config runs)
        <out>.script.py     the script that built the experiment (Python
                            runs started as `python script.py`)
        <out>.repo.patch    the package's uncommitted changes, when the
                            checkout is dirty: commit + patch = the code

    `provenance.source_digest` hashes the package's own files, so it
    identifies the code even in a vendored copy with no git."""
    ev = events_path(exp.out)
    prov = provenance()
    files: dict[str, str] = {}
    if exp.config is not None:
        files["config"] = _write_beside(exp, "config.json", json.dumps(
            exp.config, indent=1, default=str))
    script = prov["script"]
    if script and script["path"] and exp.config is None:
        files["script"] = _write_beside(exp, "script.py",
                                        Path(script["path"]).read_text())
    diff = prov.pop("diff")
    if diff:
        files["repo_patch"] = _write_beside(exp, "repo.patch", diff)
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
                                       for k in res.analysis),
                    **files},
        "provenance": prov,
    }
    res.report_path = exp.path("run.json")
    res.report_path.write_text(json.dumps(report, indent=1, default=str))


# --- the experiment -------------------------------------------------------------

@dataclass
class Experiment:
    """What to run (task, backend, cells, repeats, out) and how
    (`options`), plus one field per stage.  `run()` calls the stages in
    the order of `STAGES`, each as `stage(self, res)`.  Each stage field
    defaults to the function of the same name above:

        calibrate  calibrate_stage   core.calibrate.calibrate, then derive
        dry_run    dry_run_stage     core.runner.plan, priced, no calls
        execute    execute_stage     core.runner.execute(self.units(),
                                     backend, task.render, task.score, out)
        analyse    analyse_stage     every analysis in all_analyses()
        report     report_stage      run.json beside the sink

    So `exp.execute` is the stage, not `core.runner.execute`: the stage
    is what calls it, with the options and the calibrated settings.  Pass
    a function to replace a stage, or None to skip it."""

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
    # set by `sharded` only, so not __init__ arguments: (i, n), and the
    # unsharded sink, beside which the shards share one calibration
    shard: tuple[int, int] | None = field(default=None, init=False)
    whole: Path | None = field(default=None, init=False)
    # the resolved config, set by `from_config`; None when built in Python
    config: dict[str, Any] | None = field(default=None, init=False)

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
        """The units a run plans: all of them, or the first `limit`, and
        under a shard only that shard's share of those."""
        us = self.units()[:self.options.limit]
        if self.shard is None:
            return us
        i, n = self.shard
        return [u for u in us if shard_of(u.id, n) == i]

    def sharded(self, i: int, n: int) -> Experiment:
        """This experiment as shard i of n: its own sink at
        `shard_path(out, i, n)` (report and marker beside it; the
        calibration is shared, beside `out`),
        only its share of the units, and no analysis, since one shard's
        tables would describe a fraction of the sweep.  `merge(n)` analyses
        the whole once the shards are done."""
        if self.shard is not None:
            raise ValueError(f"already shard {self.shard}")
        shard = check_shard((i, n))
        assert isinstance(self.out, Path)
        exp = replace(self, out=shard_path(self.out, i, n))
        exp.shard, exp.whole, exp.config = shard, self.out, self.config
        return exp

    def merge(self, n: int) -> Results:
        """Combine the n shards into one sink at `out` (core.shards.
        combine_shards), then analyse and report it as an unsharded run
        would.  The marker records the merged sink's status, so a later
        unsharded `run` of the same job skips straight to analysis."""
        if self.shard is not None:
            raise ValueError("merge the unsharded experiment, not a shard")
        assert isinstance(self.out, Path)
        shard_status = self._check_shard_jobs(n)
        combine_shards(self.out, n)
        res = Results(self.out, job_hash=self.job_hash())
        res.frame = to_frame(self.out)
        for stage in (self.analyse, self.report):
            if stage is not None:
                stage(self, res)
        status, counts = job_status(self.out, self.planned())
        prov = checkout()
        res.marker = {"job_hash": res.job_hash, "status": status,
                      "halted": None, "counts": counts,
                      "merged_from": n, "shards": shard_status,
                      "finished_at": utc_now(),
                      "commit": prov["commit"], "dirty": prov["dirty"]}
        write_marker(self.path("done"), res.marker)
        return res

    def _check_shard_jobs(self, n: int) -> dict[str, str]:
        """Every shard that wrote anything ran THIS job: its marker holds
        the job hash this experiment gives that shard.  Otherwise (a model
        or default changed between the shards and the merge) the merge
        would mark one job's rows complete under another job's hash.
        Returns each shard's marker status ("never started" without one),
        so a merge says which shard is still running, halted or died: a
        crashed shard's marker stays "running"."""
        status: dict[str, str] = {}
        for i in range(n):
            shard = self.sharded(i, n)
            assert isinstance(shard.out, Path)
            marker = read_marker(shard.path("done"))
            if marker is None:
                if shard.out.exists():
                    raise ValueError(
                        f"shard {i}: {shard.out.name} has rows but no "
                        f"marker, so which job it ran is unknown")
                status[str(i)] = "never started"   # its units: missing
                continue
            if marker["job_hash"] != shard.job_hash():
                raise ValueError(
                    f"shard {i} ran job {marker['job_hash']}, but this config "
                    f"gives that shard job {shard.job_hash()}: the config "
                    f"changed since the shards ran. Rerun them, or merge with "
                    f"the config they ran")
            status[str(i)] = marker["status"]
        return status

    def calibration_path(self) -> Path:
        """`<out>.calibration.json`, the unsharded `out` under a shard, so a
        sweep calibrates once, not once per shard."""
        out = self.whole if self.whole is not None else self.out
        assert isinstance(out, Path)
        return out.with_name(f"{out.name}.calibration.json")

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
        defaults, and the repeat index.  An explicit batch_size enters
        because padding can change a local model's numerics; a calibrated
        one is not known yet when the hash is taken, so every row records
        the batch size it was made at and `plan` redoes a local model's rows
        made at another (see `batch_size_for`).  limit enters because it
        changes which units are planned.  Workers, retries, caching and the code
        commit do not: they change how a result is obtained, not what it is
        (the commit is recorded in the marker)."""
        o = self.options
        keys = sorted(request_key(rendered(self.task.render, u), u.repeat,
                                  self.backend)
                      for u in self.planned())
        job = {
            "task": self.task.name, "task_version": self.task.version,
            "items": [[i.item_id, dict(i.payload), i.group, i.parent_id]
                      for i in self.task.items],
            "cells": [dict(c.params) for c in self.cells],
            "repeats": self.repeats, "backend": self.backend.identity(),
            "batch_size": o.batch_size, "limit": o.limit,
            "requests": digest(keys)}
        if self.shard is not None:        # unsharded hashes stay as they were
            job["shard"] = list(self.shard)
        return digest(job)

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
            stage = getattr(self, name)     # e.g. self.execute: execute_stage
            if (stage is None or (res.halted and name != "report")
                    or (res.skipped and name in model_stages)
                    or (self.shard is not None and name == "analyse")):
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
        prov = checkout()
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


def _write_beside(exp: Experiment, name: str, text: str) -> str:
    path = exp.path(name)
    write_atomic(path, lambda tmp: tmp.write_text(text))
    return str(path)


def source_digest(pkg: Path) -> str:
    """sha256 over the package's .py files (relative path and bytes, in
    sorted order).  Equal digests mean the same code, with or without git,
    and an edit anywhere in the package changes it."""
    h = hashlib.sha256()
    for f in sorted(pkg.rglob("*.py")):
        h.update(f.relative_to(pkg).as_posix().encode() + b"\0")
        h.update(f.read_bytes() + b"\0")
    return h.hexdigest()


def _git(pkg: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(pkg), *args], capture_output=True,
                          text=True, timeout=10, check=True).stdout


def _script() -> dict[str, Any] | None:
    """The script `python script.py` ran, with its digest, or None under
    `python -m`, an interactive session or a test runner."""
    main = sys.modules.get("__main__")
    path = getattr(main, "__file__", None)
    if (not path or not path.endswith(".py") or getattr(main, "__spec__", None)
            is not None or not Path(path).is_file()):
        return {"path": None, "argv": list(sys.argv)}
    return {"path": str(Path(path).resolve()), "argv": list(sys.argv),
            "sha256": hashlib.sha256(Path(path).read_bytes()).hexdigest()}


PACKAGE = Path(__file__).resolve().parent


def checkout(pkg: Path = PACKAGE) -> dict[str, Any]:
    """The commit and dirty flag alone (None, None outside a git checkout),
    for the completion marker, which is written often.

    The package counts as a checkout only if the repository git finds
    tracks the package's own `__init__.py`.  git searches upward from
    `pkg`, so a copy installed into a virtualenv that sits inside another
    project's working tree (gitignored) would otherwise report that
    project's HEAD as evalcore's commit."""
    try:
        if not _git(pkg, "ls-files", "--", "__init__.py").strip():
            return {"commit": None, "dirty": None}   # not this repo's file
        return {"commit": _git(pkg, "rev-parse", "HEAD").strip(),
                "dirty": bool(_git(pkg, "status", "--porcelain", "--",
                                   ".").strip())}
    except (OSError, subprocess.SubprocessError):  # no git, or not a checkout
        return {"commit": None, "dirty": None}


@functools.cache
def _distributions() -> dict[str, str]:
    """Every installed distribution's version (fixed for a process)."""
    return dict(sorted((d.metadata["Name"], d.version)
                       for d in importlib.metadata.distributions()
                       if d.metadata["Name"]))


def provenance(pkg: Path = PACKAGE) -> dict[str, Any]:
    """Enough to say what produced a result: when, on what, with which
    versions.  The code is identified three ways: `source_digest` (always),
    the commit (from a checkout), and `diff`, the package's uncommitted
    changes against that commit with untracked files listed (report_stage
    writes it to a .patch file).  `distributions` lists every installed
    package's version."""
    versions = {}
    for mod in ("numpy", "pandas", "scipy", "torch", "transformers", "httpx"):
        m = sys.modules.get(mod)
        if m is not None:
            versions[mod] = getattr(m, "__version__", "?")
    co = checkout(pkg)
    commit, dirty = co["commit"], co["dirty"]
    untracked: list[str] = []
    diff, prefix = "", None
    try:
        if commit is not None:          # where the package sits in the repo
            prefix = _git(pkg, "rev-parse", "--show-prefix").strip() or "."
        if dirty:
            # paths relative to the package directory, binary-safe:
            # `git apply --directory=<pkg path in repo>` from the repo root
            diff = _git(pkg, "diff", "--binary", "--relative", "HEAD", "--", ".")
            # NUL-separated: a file name may hold spaces or newlines
            untracked = [u for u in _git(pkg, "ls-files", "-z", "--others",
                                         "--exclude-standard", "--",
                                         ".").split("\0") if u]
            for u in untracked:     # new files, as patches against nothing
                p = subprocess.run(
                    ["git", "-C", str(pkg), "diff", "--binary", "--no-index",
                     "--", "/dev/null", u], capture_output=True, text=True,
                    timeout=10, check=False)
                if p.returncode not in (0, 1):     # 1 = "differs", the norm
                    raise RuntimeError(f"git diff of untracked {u!r} failed "
                                       f"({p.returncode}): {p.stderr.strip()}")
                diff += p.stdout
    except (OSError, subprocess.SubprocessError) as exc:
        # the checkout answered rev-parse and status a moment ago
        raise RuntimeError(f"git failed in a checkout that answered "
                           f"rev-parse a moment ago: {exc}") from exc
    from . import __version__
    return {"time": utc_now(),
            "evalcore": __version__, "commit": commit, "dirty": dirty,
            "source_digest": source_digest(pkg), "untracked": untracked,
            # the code again: git checkout <commit>, then from the repo root
            # git apply --directory=<package_dir> <out>.repo.patch
            "package_dir": prefix,
            "diff": diff, "script": _script(),
            "python": platform.python_version(), "platform": platform.platform(),
            "versions": versions,
            "distributions": _distributions()}
