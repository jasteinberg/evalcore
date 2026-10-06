"""Batched, cached, resumable runner; JSONL sink; tidy frame.

Design commitments, each of which is a lesson rather than a preference:

* Append-only JSONL, flushed per result.  A sweep that dies eight hours
  in must lose nothing.  Parquet at the end, JSONL during.
* Resume by unit_id, not by position.  Reordering the grid, adding an arm,
  or widening the item set must not invalidate finished work.  But a
  finished row counts only if it was made from the request the unit renders
  to NOW: every row records its `request_key` (backends.request_key), and a
  unit whose prompt, item payload, backend or backend defaults changed is
  re-run rather than silently kept.
* Errors become ROWS, not exceptions.  A cell that fails on 3 of 200 items
  is a result with n=197 and a documented attrition, not a crashed run --
  but the attrition is written down, because silent dropping is how an
  eval acquires a bias nobody can see afterwards.
* Scoring is separated from generation.  `score` runs on the response, so
  changing a metric costs a re-score over cached responses, not a re-run.
* Every row has a `status`: ok, error, truncated or unparsed.  Only `ok`
  rows carry metrics.  A truncated response (the generation hit its token
  budget) is not scored, because a cut-off answer is not an answer; a
  scorer that cannot read an answer raises `ParseFailure`, because a parse
  failure is a fault of the eval, not a wrong answer by the model.  Both
  are excluded from the metric and counted in `attrition`, never folded
  into it as zeros.

  Excluding truncated rows is a choice with a cost: truncation is not
  missing at random (long answers come disproportionately from hard
  items), so the surviving mean is biased upward by an amount that grows
  with the truncation rate.  Report `n_truncated` beside the metric.  If
  the token budget is part of what the cell claims to measure, the other
  defensible reading is "failed under the budget": set the metric to 0
  for `status == "truncated"` before summarizing, and say so.
* The PLAN is written down before the first call.  Without it, a sweep that
  dies partway leaves no evidence it was ever going to do more work: the
  frame contains what came back, and units never attempted are indis-
  tinguishable from units never planned.  The manifest is a record of
  intent, not a second source of truth -- resume reads the JSONL, and a
  stale or absent manifest costs information, never correctness.
"""

from __future__ import annotations

import os
import sys
import threading
import time
from collections.abc import Callable, Iterable, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, TextIO

import pandas as pd

from .backends import (
    Backend,
    Cache,
    Request,
    Response,
    batch_with_retries,
    filled,
    request_key,
    with_retries,
)
from .events import EventLog, events_path
from .records import (
    ArrayStore,
    JsonlSink,
    SinkState,
    _read_records,
    manifest_path,
    split_arrays,
    to_frame,
    write_manifest,
)
from .spec import Unit, digest

Render = Callable[[Unit], Request]
Score = Callable[[Unit, Response], dict[str, Any]]


class ParseFailure(ValueError):
    """Raised by a scorer that cannot read an answer out of a response.

    The row gets `status="unparsed"` and no metrics, rather than an error
    (the call worked) or a zero (the model was not shown to be wrong).  Any
    other exception from a scorer is a scorer bug and becomes an error row."""


# --- plan: what will run, and why ----------------------------------------------

@dataclass
class Plan:
    """The resume decisions for one invocation, made without calling the
    model.  `execute` runs `todo`; a dry run reports it."""

    units: list[Unit]
    keys: dict[str, str]              # unit_id -> request_key
    todo: list[Unit]
    stale: int = 0        # finished, but from a different request: redo
    retried: int = 0      # last row was an error: try again
    unreadable: int = 0   # sink lines a crash cut short

    @property
    def done(self) -> int:
        return len(self.units) - len(self.todo)


def plan(units: Iterable[Unit], backend: Backend, render: Render,
         out: str | Path, resume: bool = True, retry_errors: bool = True,
         limit: int | None = None) -> Plan:
    """Decide what an invocation must run.

    A unit counts as done only if its last row is a result: one whose
    request_key matches the request it renders to now and which is not an
    error.  An error row records a failed attempt -- a 429 storm that
    outlasted the retries, an outage, a harness bug -- so it is attempted
    again unless `retry_errors=False` (for a deterministic refusal that
    would only fail again at a cost).  A changed prompt, item, backend or
    backend default changes the request_key, so the unit is redone and the
    new row supersedes the old."""
    units = list(units)[:limit]
    keys = {u.id: request_key(render(u), u.repeat, backend) for u in units}
    state = SinkState(out)
    p = Plan(units, keys, [], unreadable=state.unreadable)
    for u in units:
        if not resume or u.id not in state.seen:
            p.todo.append(u)
        elif retry_errors and u.id in state.errored:
            p.retried += 1
            p.todo.append(u)
        elif state.seen[u.id] != keys[u.id]:    # incl. a row with no key
            p.stale += 1
            p.todo.append(u)
    return p


def _report_plan(p: Plan, out: str | Path, stream: TextIO) -> None:
    def say(msg: str) -> None:
        print(f"[run] {msg}", file=stream, flush=True)
    if p.unreadable:
        say(f"WARNING {p.unreadable} unreadable line(s) in {str(out)!r} (a "
            f"write interrupted by a crash?); their units will be re-run")
    if p.stale:
        say(f"{p.stale} finished units were made from a different request "
            f"(prompt, item, backend or its defaults changed); re-running "
            f"them, and the new rows supersede the old")
    if p.retried:
        say(f"{p.retried} units errored last time; retrying them "
            f"(retry_errors=False keeps error rows as final)")
    say(f"{len(p.units)} units, {p.done} already done, {len(p.todo)} to go")


# --- execute: one invocation's shared state ---------------------------------------

@dataclass
class _Context:
    """Everything the worker threads share during one invocation."""

    backend: Backend
    render: Render
    score: Score
    keys: dict[str, str]
    out: Path
    sink: JsonlSink
    store: ArrayStore
    cache: Cache | None
    events: EventLog | None
    attempts: int
    batch_size: int
    total: int
    log_every: int
    stream: TextIO
    t0: float = field(default_factory=time.perf_counter)
    n: int = 0
    errors: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)

    def emit(self, event: str, **fields: Any) -> None:
        if self.events is not None:
            self.events.emit(event, **fields)

    def count(self, errored: bool) -> None:
        # += from several worker threads needs the lock to be exact
        with self.lock:
            self.n += 1
            self.errors += errored
            n, errs = self.n, self.errors
        if n % self.log_every == 0:
            rate = n / max(time.perf_counter() - self.t0, 1e-9)
            eta = (self.total - n) / max(rate, 1e-9)
            print(f"[run] {n}/{self.total}  {rate:.1f}/s  eta {eta / 60:.1f}m"
                  f"  err {errs}", file=self.stream, flush=True)


def _join(reqs: Sequence[Request],
          resps: Sequence[Response | None]) -> list[Response]:
    """Responses matched to requests by unit_id, never by position.

    The backend contract is "same length and order", but a batched backend
    that collects results as they complete, or loses one, breaks it without
    raising -- and a positional zip then writes one unit's answer into
    another unit's row, which no statistic downstream can detect.  A request
    left without a response becomes an error row, not a neighbour's answer.
    """
    by_id: dict[str, Response] = {}
    for r in resps or ():
        if r is not None:
            by_id.setdefault(r.unit_id, r)
    return [by_id.get(q.unit_id) or Response(
        q.unit_id, error="missing: backend returned no response for this unit")
        for q in reqs]


def _call(ctx: _Context, reqs: list[Request], rkeys: list[str]
          ) -> list[Response]:
    """One response per request: refused, cached, or freshly called (with
    retries), arrays moved to sidecars before anything serialises them."""
    resps: list[Response | None] = [None] * len(reqs)
    pending = []
    for i, rq in enumerate(reqs):
        refusal = ctx.backend.refusal(rq)
        hit = None if refusal or ctx.cache is None else ctx.cache.get(rkeys[i])
        if refusal:
            resps[i] = Response(rq.unit_id, error=f"fatal: {refusal}")
        elif hit is not None:
            resps[i] = Response(rq.unit_id, hit.get("text"), hit.get("meta", {}),
                                hit.get("error"), cached=True)
        else:
            pending.append(i)
    if pending:
        sub = [reqs[i] for i in pending]

        def on_retry(attempt: int, wait_s: float, reason: str,
                     server_wait_s: float | None) -> None:
            ctx.emit("retry", unit_ids=[q.unit_id for q in sub],
                     attempt=attempt, wait_s=wait_s, reason=str(reason)[:300],
                     server_wait_s=server_wait_s)

        fresh = (batch_with_retries(ctx.backend.complete_batch, sub,
                                    ctx.attempts, on_retry=on_retry)
                 if len(sub) > 1 or ctx.batch_size > 1 else
                 [with_retries(ctx.backend.complete, sub[0], ctx.attempts,
                               on_retry=on_retry)])
        for i, rp in zip(pending, _join(sub, fresh), strict=True):
            meta, arrays = split_arrays(rp.meta or {})
            if arrays:
                f = ctx.store.put(rkeys[i], arrays)
                meta["arrays"] = os.path.relpath(f, ctx.out.parent)
            rp.meta = meta
            resps[i] = rp
            if ctx.cache and rp.ok:
                ctx.cache.put(rkeys[i], {"text": rp.text, "meta": rp.meta})
    return filled(resps)


def _row(u: Unit, rkey: str, rp: Response, score: Score) -> dict[str, Any]:
    """The result row: identity columns, the response, a status, and the
    scores -- which only an `ok` response gets."""
    meta = rp.meta or {}
    row = {**u.row(), "text": rp.text, "error": rp.error, "cached": rp.cached,
           "request_key": rkey, **{f"meta_{k}": v for k, v in meta.items()}}
    if not rp.ok:
        # a tool's infrastructure failing is not the model failing
        row["status"] = "tool_error" if meta.get("tool_error") else "error"
    elif meta.get("truncated"):
        row["status"] = "truncated"           # cut off: not an answer
    else:
        try:
            row.update(score(u, rp))
            row["status"] = "ok"
        except ParseFailure as exc:
            row.update(status="unparsed", parse_error=str(exc))
        except Exception as exc:  # noqa: BLE001 - a scorer bug is a row, too
            row.update(status="error", error=f"score: {type(exc).__name__}: {exc}")
    return row


def _batch(ctx: _Context, batch: Sequence[Unit]) -> None:
    rkeys = [ctx.keys[u.id] for u in batch]
    resps = _call(ctx, [ctx.render(u) for u in batch], rkeys)
    for u, rk, rp in zip(batch, rkeys, resps, strict=True):
        row = _row(u, rk, rp, ctx.score)
        ctx.sink.write(row)
        meta = rp.meta or {}
        ctx.emit("unit", unit_id=u.id, status=row["status"], cached=rp.cached,
                 latency_s=meta.get("latency_s"), in_tokens=meta.get("in_tokens"),
                 out_tokens=meta.get("out_tokens"),
                 error=str(row["error"])[:300] if row.get("error") else None)
        ctx.count(errored=bool(row.get("error")))


# --- run ------------------------------------------------------------------------

def execute(units: Iterable[Unit], backend: Backend, render: Render, score: Score,
        out: str | Path, *, cache: Cache | None = None, workers: int = 8,
        batch_size: int = 1, attempts: int = 5, resume: bool = True,
        retry_errors: bool = True, limit: int | None = None,
        log_every: int = 25, events: bool = True,
        manifest: str | Path | bool = True,
        stream: TextIO = sys.stderr) -> pd.DataFrame:
    """Execute `units`, writing one row per unit to `out` (JSONL), and
    return the frame.

    workers x batch_size is the concurrency knob.  HTTP: workers=8..16,
    batch_size=1.  Local HF: workers=1, batch_size=8..32 (one process owns
    the device; threads would only contend for it).

    The plan is written to `manifest` (default: a sibling of `out`) BEFORE
    the first call, so a sweep killed midway is still accountable -- pass
    that file to `attrition` to see the units nobody ever attempted.  Every
    unit, retry and run boundary is appended to `<out>.events.jsonl`
    (`events=False` disables it).  Resume decisions: see `plan`.
    """
    backend.identity()                    # refuse before touching any file
    out = Path(out)
    p = plan(units, backend, render, out, resume, retry_errors, limit)
    mpath = (None if manifest is False or manifest is None
             else manifest_path(out) if manifest is True else Path(manifest))
    if mpath is not None:
        write_manifest(p.units, mpath)        # intent, before any execution
    _report_plan(p, out, stream)
    ctx = _Context(
        backend, render, score, p.keys, out, JsonlSink(out),
        # sidecars live with the cache when there is one, so a cache hit in
        # another run finds them; otherwise beside the sink
        ArrayStore(cache.root / "arrays" if cache and cache.enabled
                   else Path(str(out) + ".arrays")),
        cache, EventLog(events_path(out)) if events else None,
        attempts, batch_size, len(p.todo), log_every, stream)
    ctx.emit("run_start", n_units=len(p.units), to_go=len(p.todo), done=p.done,
             stale=p.stale, retried=p.retried, workers=workers,
             batch_size=batch_size, backend=type(backend).__name__,
             identity=digest(backend.identity()), out=str(out))
    batches = [p.todo[i:i + max(1, batch_size)]
               for i in range(0, len(p.todo), max(1, batch_size))]
    try:
        if workers <= 1:
            for b in batches:
                _batch(ctx, b)
        else:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                list(pool.map(lambda b: _batch(ctx, b), batches))
    except BaseException:
        if ctx.events is not None:        # no run_end: the log says "killed"
            ctx.events.close()
        raise
    finally:
        ctx.sink.close()
        backend.close()
    _finish(ctx, mpath)
    return to_frame(out)


def _finish(ctx: _Context, mpath: Path | None) -> None:
    el = time.perf_counter() - ctx.t0
    c = ctx.cache
    ctx.emit("run_end", units_done=ctx.n, errors=ctx.errors, elapsed_s=el,
             cache_hits=c.hits if c else None,
             cache_misses=c.misses if c else None)
    if ctx.events is not None:
        ctx.events.close()
    cache_msg = f", cache {c.hits} hit / {c.misses} miss" if c else ""
    print(f"[run] done: {ctx.n} units in {el / 60:.1f}m, {ctx.errors} "
          f"errors{cache_msg}", file=ctx.stream, flush=True)
    if mpath is not None:
        # Against the manifest, not this invocation's units: a narrowed
        # resume (a `limit=`, one arm re-run) is exactly the case where the
        # shortfall is invisible from the call site.
        never = len({r["unit_id"] for r in _read_records(mpath)}
                    - ctx.sink.seen.keys())
        if never:
            print(f"[run] WARNING {never} planned units were never attempted; "
                  f"pass manifest={str(mpath)!r} to attrition()",
                  file=ctx.stream, flush=True)
