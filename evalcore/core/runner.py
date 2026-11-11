"""Plan and execute a sweep: resume decisions, batching, retries, and one
row per unit, with a status.  The sink, the manifest and the tidy frame
the rows become are in records.py.

Design commitments, each of which is a lesson rather than a preference:

* Append-only JSONL, flushed per result.  A sweep that dies eight hours
  in must lose nothing.  Parquet at the end, JSONL during.
* Resume by unit_id, not by position.  Reordering the grid, adding a condition,
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
  changing a metric costs a re-score, not a re-run.  Pass `score_version`
  (the pipeline and `Task.run` pass `Task.version`): every row records the
  version that scored it, and a finished row scored under another version
  is re-scored from its stored text and meta, with no model call and no
  cache needed.  The new row supersedes the old.
* Every row has a `status`: ok, error, truncated or unparsed.  Only `ok`
  rows carry metrics.  A truncated response (the generation hit its token
  budget) is not scored, because a cut-off answer is not an answer.  A
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

import json
import os
import sys
import threading
import time
from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field, replace
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
    _MANIFEST_LOCK,
    ArrayStore,
    SinkLog,
    SinkState,
    _read_records,
    append_plan,
    json_default,
    manifest_path,
    split_arrays,
    to_frame,
)
from .shards import check_shard, shard_of
from .spec import Unit, digest

Render = Callable[[Unit], Request]
Score = Callable[[Unit, Response], dict[str, Any]]

# Columns the runner writes into every row.  A cell param or tag with one of
# these names, or with the `meta_` prefix the response's meta is spread
# under, would overwrite the runner's value or be overwritten by it, and
# the row would then describe a different unit or response than the one it
# came from.  Refused by `plan`, before any call.
RUNNER_COLUMNS = frozenset({
    "cell_id", "item_id", "cluster", "group", "repeat", "unit_id", "text",
    "error", "cached", "request_key", "status", "parse_error",
    "score_version", "batch_size"})


class ParseFailure(ValueError):
    """Raised by a scorer that cannot read an answer out of a response.

    The row gets `status="unparsed"` and no metrics, rather than an error
    (the call worked) or a zero (the model was not shown to be wrong).  Any
    other exception from a scorer is a scorer bug and becomes an error row."""


# --- plan: what will run, and why ----------------------------------------------

@dataclass
class Plan:
    """The resume decisions for one invocation, made without calling the
    model.  `execute` runs `todo`, and a dry run reports it."""

    units: list[Unit]
    keys: dict[str, str]              # unit_id -> request_key
    todo: list[Unit]
    stale: int = 0        # finished, but from a different request: redo
    retried: int = 0      # last row was an error: try again
    unreadable: int = 0   # sink lines a crash cut short
    # finished under another score_version: re-scored from the row, no call
    rescore: list[Unit] = field(default_factory=list)
    rows: dict[str, Mapping[str, Any]] = field(default_factory=dict,
                                               repr=False)

    @property
    def done(self) -> int:
        """Units needing no model call (rescores included)."""
        return len(self.units) - len(self.todo)


def rendered(render: Render, u: Unit) -> Request:
    """`render(u)` carrying the unit's repeat index and item id.  The runner
    owns both: a render that sets different ones is refused."""
    req = render(u)
    if req.repeat not in (0, u.repeat):
        raise ValueError(f"render set repeat={req.repeat} on unit {u.id} "
                         f"with repeat {u.repeat}")
    if req.item_id not in (None, u.item.item_id):
        raise ValueError(f"render set item_id={req.item_id!r} on unit {u.id} "
                         f"of item {u.item.item_id!r}")
    if req.repeat == u.repeat and req.item_id == u.item.item_id:
        return req
    return replace(req, repeat=u.repeat, item_id=u.item.item_id)


def plan(units: Iterable[Unit], backend: Backend, render: Render,
         out: str | Path, resume: bool = True, retry_errors: bool = True,
         limit: int | None = None, score_version: str | None = None,
         batch_size: int | None = None) -> Plan:
    """Decide what an invocation must run.

    A unit counts as done only if its last row is a result: one whose
    request_key matches the request it renders to now and which is not an
    error.  An error row records a failed attempt -- a 429 storm that
    outlasted the retries, an outage, a harness bug -- so it is attempted
    again unless `retry_errors=False` (for a deterministic refusal that
    would only fail again at a cost).  A changed prompt, item, backend or
    backend default changes the request_key, so the unit is redone and the
    new row supersedes the old.

    With `score_version` given, a unit that is done but whose last row was
    scored under another version (or none) goes to `rescore`: its response
    is read back from the row and scored again, without a call.

    With `batch_size` given and a backend whose numerics padding can change
    (`supports_target`: a local model), a row made at another batch size
    (or an unrecorded one) is stale, so one sink never mixes batch sizes."""
    units = list(units)[:limit]
    return _plan(units, backend, render,
                 SinkState(out, keep_rows=score_version is not None),
                 resume, retry_errors, score_version, batch_size)


def _plan(units: list[Unit], backend: Backend, render: Render,
          state: SinkState, resume: bool, retry_errors: bool,
          score_version: str | None, batch_size: int | None,
          content: dict[str, str] | None = None) -> Plan:
    """`plan` against a sink state already in memory.  `content` carries
    the clash check across calls (see `_refuse_clashes`).  It learns the
    units' items only once every unit has rendered, so a unit whose render
    raised (and which therefore wrote no row) is not remembered: corrected
    and run again, it is planned as if it had never been seen."""
    new = _refuse_clashes(units, content)
    keys = {u.id: request_key(rendered(render, u), u.repeat, backend)
            for u in units}
    if content is not None:
        content.update(new)
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
        elif (batch_size is not None and backend.supports_target
              and state.batch.get(u.id) != batch_size):
            p.stale += 1
            p.todo.append(u)
        elif (score_version is not None and u.id not in state.errored
              and state.rows[u.id].get("score_version") != score_version):
            p.rescore.append(u)
            p.rows[u.id] = state.rows[u.id]
    return p


def _refuse_clashes(units: Sequence[Unit],
                    seen: Mapping[str, str] | None = None) -> dict[str, str]:
    """Refuse a plan whose rows could not each describe their own unit.

    Two units with one unit_id but different items (an item_id reused for a
    different question) would share a row: resume and `to_frame` keep the
    last, so one answer is silently lost.  A cell param or tag named like a
    runner column (RUNNER_COLUMNS, or `meta_*`) would be overwritten in the
    row, or overwrite it.

    `seen` (unit_id -> item content digest) is the memory of earlier calls:
    a Runner passes the same dict every time, so the check spans every unit
    it has planned, as one `execute` call's does.  Returns the digests of
    `units`, for the caller to merge into `seen` once the plan is made."""
    seen = {} if seen is None else seen
    new: dict[str, str] = {}
    names: set[str] = set()
    for u in units:
        it = u.item
        content = digest({"payload": dict(it.payload), "group": it.group,
                          "parent_id": it.parent_id})
        if (seen.get(u.id) or new.setdefault(u.id, content)) != content:
            raise ValueError(
                f"item_id {it.item_id!r} names two different items (payload, "
                f"group or parent differ). Item ids must be unique, or one "
                f"item's rows would silently replace the other's")
        names.update(u.cell.params, u.cell.tags)
    bad = sorted(n for n in names
                 if n in RUNNER_COLUMNS or str(n).startswith("meta_"))
    if bad:
        raise ValueError(
            f"cell param/tag name(s) {bad} clash with columns the runner "
            f"writes ({sorted(RUNNER_COLUMNS)} and meta_*). Rename them")
    return new


def _report_plan(p: Plan, out: str | Path, stream: TextIO) -> None:
    def say(msg: str) -> None:
        print(f"[run] {msg}", file=stream, flush=True)
    if p.unreadable:
        say(f"WARNING {p.unreadable} unreadable line(s) in {str(out)!r} (a "
            f"write interrupted by a crash?). Their units will be re-run")
    if p.stale:
        say(f"{p.stale} finished units were made from a different request "
            f"(prompt, item, backend or its defaults changed, or a local "
            f"model's batch size). Re-running them, and the new rows "
            f"supersede the old")
    if p.retried:
        say(f"{p.retried} units errored last time. Retrying them "
            f"(retry_errors=False keeps error rows as final)")
    if p.rescore:
        say(f"{len(p.rescore)} finished units were scored under another "
            f"score_version. Re-scoring them from their rows, no calls")
    say(f"{len(p.units)} units, {p.done} already done, {len(p.todo)} to go")


# --- execute: one invocation's shared state ---------------------------------------

@dataclass
class _Context:
    """Everything the worker threads share while a Runner is open."""

    backend: Backend
    render: Render
    score: Score
    keys: dict[str, str]
    out: Path
    sink: SinkLog
    store: ArrayStore
    cache: Cache | None
    events: EventLog | None
    attempts: int
    batch_size: int
    total: int | None         # units to call, when known in advance
    log_every: int
    stream: TextIO
    t0: float = field(default_factory=time.perf_counter)
    n: int = 0
    errors: int = 0
    lock: threading.Lock = field(default_factory=threading.Lock)
    score_version: str | None = None

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
            if self.total is None:
                msg = f"{n}  {rate:.1f}/s"
            else:
                eta = (self.total - n) / max(rate, 1e-9)
                msg = f"{n}/{self.total}  {rate:.1f}/s  eta {eta / 60:.1f}m"
            print(f"[run] {msg}  err {errs}", file=self.stream, flush=True)


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
            meta = dict(hit.get("meta", {}))
            if "arrays" in meta:
                meta["arrays"] = _sidecar(ctx, rkeys[i])
            resps[i] = Response(rq.unit_id, hit.get("text"), meta,
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
                ctx.store.put(rkeys[i], arrays)
                meta["arrays"] = _sidecar(ctx, rkeys[i])
            rp.meta = meta
            resps[i] = rp
            if ctx.cache and rp.ok:
                ctx.cache.put(rkeys[i], {"text": rp.text, "meta": rp.meta})
    return filled(resps)


def _sidecar(ctx: _Context, rkey: str) -> str:
    """A row's `meta_arrays`: the sidecar's path relative to THIS sink.

    Computed per sink, never taken from the cache: the cached meta carries
    the path relative to whichever sink first wrote it, which from a sink
    in another directory points at nothing (or at another file).  The
    sidecar itself is keyed on the request key, so it is the same file."""
    f = ctx.store.path(rkey)
    if not f.exists():
        raise FileNotFoundError(
            f"array sidecar {str(f)!r} for request {rkey} is missing. The "
            f"cache entry refers to it, so the cache directory was edited")
    return os.path.relpath(f, ctx.out.parent)


def _row(u: Unit, rkey: str, rp: Response, score: Score,
         score_version: str | None = None,
         batch_size: int | None = None) -> dict[str, Any]:
    """The result row: identity columns, the response, a status, and the
    scores -- which only an `ok` response gets."""
    meta = rp.meta or {}
    row = {**u.row(), "text": rp.text, "error": rp.error, "cached": rp.cached,
           "request_key": rkey, **{f"meta_{k}": v for k, v in meta.items()}}
    if score_version is not None:
        row["score_version"] = score_version
    if batch_size is not None:
        row["batch_size"] = batch_size
    if not rp.ok:
        # a tool's infrastructure failing is not the model failing
        row["status"] = "tool_error" if meta.get("tool_error") else "error"
    elif meta.get("truncated"):
        row["status"] = "truncated"           # cut off: not an answer
    else:
        try:
            scores = score(u, rp)
            clash = sorted(set(scores) & set(row))
            if clash:
                # a score named like an identity, param or response column
                # would re-key or rewrite the row: a scorer bug, so an
                # error row that keeps the unit's own columns
                row.update(status="error", error=f"score: returned {clash}, "
                           f"which name columns the row already has")
            else:
                row.update(scores)
                row["status"] = "ok"
        except ParseFailure as exc:
            row.update(status="unparsed", parse_error=str(exc))
        except Exception as exc:  # noqa: BLE001 - a scorer bug is a row, too
            row.update(status="error", error=f"score: {type(exc).__name__}: {exc}")
    return row


def _response_of(row: Mapping[str, Any]) -> Response:
    """The response a row was made from, read back for a re-score: its
    text, error, cached flag and meta (the `meta_` columns)."""
    return Response(row["unit_id"], row.get("text"),
                    {k[5:]: v for k, v in row.items() if k.startswith("meta_")},
                    row.get("error"), cached=bool(row.get("cached")))


def _batch(ctx: _Context, batch: Sequence[Unit]) -> None:
    rkeys = [ctx.keys[u.id] for u in batch]
    resps = _call(ctx, [rendered(ctx.render, u) for u in batch], rkeys)
    for u, rk, rp in zip(batch, rkeys, resps, strict=True):
        row = _row(u, rk, rp, ctx.score, ctx.score_version, ctx.batch_size)
        ctx.sink.write(row)
        meta = rp.meta or {}
        ctx.emit("unit", unit_id=u.id, status=row["status"], cached=rp.cached,
                 latency_s=meta.get("latency_s"), in_tokens=meta.get("in_tokens"),
                 out_tokens=meta.get("out_tokens"),
                 error=str(row["error"])[:300] if row.get("error") else None)
        ctx.count(errored=bool(row.get("error")))


# --- run ------------------------------------------------------------------------

class Runner:
    """Execute units against one sink, a call (or a batch) at a time, with
    the sink's state and the manifest held in memory.

    For a sequential caller, whose next unit depends on the previous
    unit's output and so cannot be handed over in a batch.  N single-unit
    `execute` calls into one sink read the whole sink and manifest N times
    (O(N^2) lines parsed).  A Runner reads each once, when constructed.
    Each `run` then appends the unit's row, its manifest record (if not
    already recorded) and its events, and reads nothing:

        with Runner(backend, render, score, out, cache=cache) as r:
            row = r.run(unit)          # the unit's row, as written

    `execute` is built on this class, so the resume, retry, cache, sidecar
    and manifest semantics are one implementation.  A unit already done is
    returned from the sink with no call.  A stale or errored one is
    redone, and one finished under another `score_version` is re-scored
    (see `plan`).  The event log gets one run_start when the Runner opens
    and one run_end when it closes cleanly, with each unit's events
    between.  A Runner left by an exception closes without a run_end, so
    the log says the run was killed, as for `execute`.  A `run` refused
    before any call (an item_id reused for another item, a cell name that
    clashes with a runner column) writes no row and logs a `refused` event
    with the reason, then raises.  Caught, the Runner carries on.  Not
    caught, it is that exception that ends the run.

    The clash check spans every unit the Runner has planned, so an item_id
    reused with another payload is refused.  N single-unit `execute` calls
    did not check across calls: the later row silently superseded the
    earlier one.  That is the one way a Runner's rows differ from theirs.

    Nothing between the call to `run` and the backend's call draws from a
    global random generator (Python's, numpy's or torch's), so a caller
    that seeds before each `run` gives the backend the state it seeded.

    The sink's rows (each unit's last) are held in memory, so that a done
    unit's row is returned without reading the file: memory grows with the
    sink.  (`execute`'s Runner holds them only while planning a re-score.)
    One caller thread: `workers` parallelises within a `run_batch`, not
    across concurrent calls.

    One writer per sink.  The state is read once, so rows another process
    appends to the same sink while this Runner is open are not seen: both
    could call the same unit, and resume answers from this Runner's view.
    Shards write separate sinks (see `execute`).  An incremental caller is
    not sharded: `run` is given units one at a time, so the manifest could
    not record the whole plan before the first call, which is what lets
    `merge_shards` count a shard that never started.
    """

    def __init__(self, backend: Backend, render: Render, score: Score,
                 out: str | Path, *, cache: Cache | None = None,
                 workers: int = 1, batch_size: int = 1, attempts: int = 5,
                 resume: bool = True, retry_errors: bool = True,
                 log_every: int = 25, events: bool = True,
                 manifest: str | Path | bool = True,
                 score_version: str | None = None,
                 stream: TextIO = sys.stderr,
                 _shard: tuple[int, int] | None = None,
                 _keep_rows: bool = True) -> None:
        backend.identity()                # refuse before touching any file
        self.backend, self.render, self.score = backend, render, score
        self.out = Path(out)
        self.cache, self.workers, self.batch_size = cache, workers, batch_size
        self.attempts, self.resume = attempts, resume
        self.retry_errors, self.log_every = retry_errors, log_every
        self.events, self.score_version, self.stream = (events, score_version,
                                                        stream)
        self.mine: Callable[[str], bool] = lambda uid: True
        if _shard is not None:              # execute's only
            si, sn = check_shard(_shard)
            self.mine = lambda uid: shard_of(uid, sn) == si
        self.mpath = (None if manifest is False or manifest is None
                      else manifest_path(self.out) if manifest is True
                      else Path(manifest))
        # the only reads: the sink and the manifest, once each
        self.state = SinkState(self.out, keep_rows=_keep_rows)
        self.planned: set[str] = ({r["unit_id"] for r in
                                   _read_records(self.mpath)}
                                  if self.mpath is not None else set())
        self.keys: dict[str, str] = {}
        self._content: dict[str, str] = {}
        self.tally: Counter[str] = Counter()
        self.ctx: _Context | None = None
        self.closed = False

    # -- the steps, in the order execute takes them --------------------------

    def _plan_units(self, units: Sequence[Unit]) -> Plan:
        """The resume decisions for `units` against the sink as it stands
        in memory (`plan`), without a call or a file read."""
        p = _plan(list(units), self.backend, self.render, self.state,
                  self.resume, self.retry_errors, self.score_version,
                  self.batch_size, self._content)
        self.keys.update(p.keys)
        return p

    def _record(self, units: Iterable[Unit]) -> None:
        """Append to the manifest the units it does not yet record (the
        plan, before any call: see `write_manifest`)."""
        if self.mpath is None:
            return
        units = list(units)
        if self.mpath.exists() and all(u.id in self.planned for u in units):
            return
        self.mpath.parent.mkdir(parents=True, exist_ok=True)
        with _MANIFEST_LOCK:
            append_plan(self.mpath, units, self.planned)

    def open(self, start: Mapping[str, Any] | None = None,
             total: int | None = None) -> Runner:
        """Open the sink and the event log for appending and emit
        run_start.  `start` gives run_start's plan fields (`execute`
        passes its plan's counts).  Without it the run_start says
        `incremental=True`, since the units are not known in advance."""
        if self.ctx is not None:
            raise RuntimeError("Runner already open")
        if self.closed:
            raise RuntimeError("Runner closed: open a new one on the sink")
        b, c, out = self.backend, self.cache, self.out
        self.ctx = _Context(
            b, self.render, self.score, self.keys, out,
            SinkLog(out, self.state),
            # sidecars live with the cache when there is one, so a cache hit
            # in another run finds them, and otherwise beside the sink
            ArrayStore(c.root / "arrays" if c and c.enabled
                       else Path(str(out) + ".arrays")),
            c, EventLog(events_path(out)) if self.events else None,
            self.attempts, self.batch_size, total, self.log_every,
            self.stream, score_version=self.score_version)
        if start is None:
            start = {"incremental": True}
            if self.state.unreadable:
                print(f"[run] WARNING {self.state.unreadable} unreadable "
                      f"line(s) in {str(out)!r} (a write interrupted by a "
                      f"crash?). Their units will be re-run", file=self.stream,
                      flush=True)
        self.ctx.emit("run_start", **start, workers=self.workers,
                      batch_size=self.batch_size, backend=type(b).__name__,
                      identity=digest(b.identity()), out=str(out))
        return self

    def _rescore(self, p: Plan) -> None:
        """Write a re-scored row for each unit of `p.rescore`: the stored
        response scored again under this Runner's `score_version`, with
        no call."""
        ctx = self._open_ctx()
        for u in p.rescore:               # no call: score the stored response
            row = _row(u, p.keys[u.id], _response_of(p.rows[u.id]), self.score,
                       self.score_version, p.rows[u.id].get("batch_size"))
            ctx.sink.write(row)
            ctx.emit("rescore", unit_id=u.id, status=row["status"])

    def _call(self, todo: Sequence[Unit]) -> None:
        """Call the backend for `todo`, in batches of `batch_size` over
        `workers` threads, writing each unit's row as it comes back."""
        ctx, bs = self._open_ctx(), max(1, self.batch_size)
        batches = [todo[i:i + bs] for i in range(0, len(todo), bs)]
        if self.workers <= 1 or len(batches) <= 1:
            for b in batches:
                _batch(ctx, b)
        else:
            with ThreadPoolExecutor(max_workers=self.workers) as pool:
                list(pool.map(lambda b: _batch(ctx, b), batches))

    def close(self, killed: bool = False) -> None:
        """Close the sink and the backend, then emit run_end and close the
        event log -- unless `killed`, when the log is left without a
        run_end, which is how it says the run did not finish."""
        if self.closed:
            return
        self.closed = True
        ctx = self.ctx
        if ctx is None:
            return
        if killed and ctx.events is not None:
            ctx.events.close()
        try:
            ctx.sink.close()
        finally:
            self.backend.close()
        if not killed:
            self._finish(ctx)

    # -- the incremental interface ------------------------------------------

    def run(self, unit: Unit) -> dict[str, Any]:
        """The unit's row: from the sink if the unit is done, otherwise
        made (by a call, or a re-score) and appended."""
        return self.run_batch([unit])[0]

    def run_batch(self, units: Iterable[Unit]) -> list[dict[str, Any]]:
        """The rows of `units`, in order, as `run` makes each one, with
        the calls batched by `batch_size` over `workers` threads."""
        ctx = self._open_ctx()
        if not self.state.keep_rows:
            raise RuntimeError("this Runner does not hold the sink's rows")
        units = list(units)
        try:
            p = self._plan_units(units)
        except ValueError as exc:         # a refusal: nothing was written
            ctx.emit("refused", unit_ids=[u.id for u in units][:10],
                     error=str(exc)[:300])
            raise
        self._record(p.units)
        self.tally.update(stale=p.stale, retried=p.retried,
                          rescored=len(p.rescore),
                          from_sink=p.done - len(p.rescore))
        self._rescore(p)
        self._call(p.todo)
        # as the sink holds it (JSON), so the row equals what to_frame reads
        return [json.loads(json.dumps(self.state.rows[u.id],
                                      default=json_default))
                for u in p.units]

    def __enter__(self) -> Runner:
        if self.ctx is None:
            self.open()
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: object) -> None:
        self.close(killed=exc_type is not None)

    # -- internals -----------------------------------------------------------

    def _open_ctx(self) -> _Context:
        if self.ctx is None or self.closed:
            raise RuntimeError("Runner not open: use `with Runner(...)` or "
                               "call open()")
        return self.ctx

    def _finish(self, ctx: _Context) -> None:
        el = time.perf_counter() - ctx.t0
        c = ctx.cache
        ctx.emit("run_end", units_done=ctx.n, errors=ctx.errors, elapsed_s=el,
                 cache_hits=c.hits if c else None,
                 cache_misses=c.misses if c else None)
        if ctx.events is not None:
            ctx.events.close()
        cache_msg = f", cache {c.hits} hit / {c.misses} miss" if c else ""
        print(f"[run] done: {ctx.n} units in {el / 60:.1f}m, {ctx.errors} "
              f"errors{cache_msg}", file=self.stream, flush=True)
        t = self.tally
        if any(t.values()):
            print(f"[run] also: {t['from_sink']} returned from the sink, "
                  f"{t['rescored']} re-scored, {t['stale']} stale and "
                  f"{t['retried']} errored units redone", file=self.stream,
                  flush=True)
        if self.mpath is not None:
            # Against the manifest, not this invocation's units: a narrowed
            # resume (a `limit=`, one condition re-run) is exactly the case
            # where the shortfall is invisible from the call site.  Under a
            # shard, only this shard's units: the others are other
            # processes' work, not this one's shortfall.
            never = len({u for u in self.planned if self.mine(u)}
                        - self.state.seen.keys())
            if never:
                print(f"[run] WARNING {never} planned units were never "
                      f"attempted. Pass manifest={str(self.mpath)!r} to "
                      f"attrition()", file=self.stream, flush=True)


def execute(units: Iterable[Unit], backend: Backend, render: Render, score: Score,
        out: str | Path, *, cache: Cache | None = None, workers: int = 8,
        batch_size: int = 1, attempts: int = 5, resume: bool = True,
        retry_errors: bool = True, limit: int | None = None,
        log_every: int = 25, events: bool = True,
        manifest: str | Path | bool = True,
        shard: tuple[int, int] | None = None,
        score_version: str | None = None,
        stream: TextIO = sys.stderr) -> pd.DataFrame:
    """Execute `units`, writing one row per unit to `out` (JSONL), and
    return the frame.

    workers x batch_size is the concurrency knob.  HTTP: workers=8..16,
    batch_size=1.  Local HF: workers=1, batch_size=8..32 (one process owns
    the device, and threads would only contend for it).

    The plan is written to `manifest` (default: a sibling of `out`) BEFORE
    the first call, so a sweep killed midway is still accountable -- pass
    that file to `attrition` to see the units nobody ever attempted.  Every
    unit, retry and run boundary is appended to `<out>.events.jsonl`
    (`events=False` disables it).  Resume decisions: see `plan`.

    `shard=(i, n)` runs only the units of shard i of n (see core.shards),
    for one task of a SLURM array or one of several machines.  Give each
    shard its own `out` (`shard_path`), and merge with `merge_shards`.  The
    manifest still records the whole plan, so a shard that never started
    is counted missing.

    `score_version` names the scorer: rows record it, and a finished row
    scored under another one is re-scored without a call (see `plan`).

    One call reads the whole sink and manifest.  A caller that executes
    one unit at a time should hold a `Runner` open instead.
    """
    everything = list(units)[:limit]
    r = Runner(backend, render, score, out, cache=cache, workers=workers,
               batch_size=batch_size, attempts=attempts, resume=resume,
               retry_errors=retry_errors, log_every=log_every, events=events,
               manifest=manifest, score_version=score_version,
               stream=stream, _shard=shard,
               _keep_rows=score_version is not None)
    p = r._plan_units([u for u in everything if r.mine(u.id)])
    # p.rows holds the rows a re-score reads.  Nothing else in execute
    # reads the state's rows, so it stops keeping them (and every row
    # written from here on): memory stays at the plan's, as before Runner.
    r.state.keep_rows = False
    r.state.rows.clear()
    r._record(everything)                 # intent, before any execution
    _report_plan(p, r.out, stream)
    r.open({"n_units": len(p.units), "to_go": len(p.todo), "done": p.done,
            "stale": p.stale, "retried": p.retried}, total=len(p.todo))
    try:
        r._rescore(p)
        r._call(p.todo)
    except BaseException:
        r.close(killed=True)
        raise
    r.close()
    return to_frame(r.out)
