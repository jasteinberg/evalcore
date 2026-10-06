"""A machine-readable record of what a run did, and the tools to query it.

Every `execute()` appends events to `<sink>.events.jsonl`, one JSON object per
line, each with a wall-clock time `t` (ISO 8601, UTC) and `event`:

    run_start   plan size, what resume decided (done / stale / retried),
                settings, backend identity digest
    unit        one per finished unit: status, latency, tokens, cached,
                error (first 300 chars)
    retry       one per retried attempt: unit, attempt, wait, reason
    run_end     units done, errors, elapsed, cache hits / misses

A run killed partway has a run_start and no run_end, which `summarize`
reports.  The human-readable progress lines are unchanged; this file is
for asking questions afterwards ("why did the sweep take nine hours",
"which errors, how many, since when") without reading them.

`read_events` returns the log as a frame for arbitrary queries;
`summarize_events` answers the usual questions in one call; `parse_log`
turns a foreign text log into the same kind of frame with named-group
regexes, so another tool's output can be queried the same way.
"""

from __future__ import annotations

import json
import re
import threading
from collections.abc import Mapping
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .records import json_default

__all__ = ["EventLog", "events_path", "format_summary", "parse_log", "read_events",
           "summarize_events"]

EVENTS_SUFFIX = ".events.jsonl"


def events_path(out: str | Path) -> Path:
    """The event log belonging to a sink (or the path itself if it already
    is one)."""
    p = Path(out)
    return p if p.name.endswith(EVENTS_SUFFIX) else p.with_name(
        p.name + EVENTS_SUFFIX)


class EventLog:
    """Thread-safe append-only event writer, flushed per event."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.fh = self.path.open("a")

    def emit(self, event: str, **fields: Any) -> None:
        rec = {"t": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
               "event": event, **fields}
        with self.lock:
            self.fh.write(json.dumps(rec, default=json_default) + "\n")
            self.fh.flush()

    def close(self) -> None:
        self.fh.close()


def read_events(path: str | Path) -> pd.DataFrame:
    """The event log as a frame (`t` parsed to datetimes).  Unreadable
    lines -- a write cut off by a kill -- are skipped and counted in
    `attrs["unreadable"]`."""
    rows, bad = [], 0
    with events_path(path).open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    df = pd.DataFrame(rows)
    if not df.empty:
        df["t"] = pd.to_datetime(df["t"], utc=True)
    df.attrs["unreadable"] = bad
    return df


_VOLATILE = re.compile(r"\b[0-9a-f]{12,}\b|\d+(?:\.\d+)?")


def error_kind(msg: str) -> str:
    """An error message with ids and numbers masked, so "HTTP 429 after 3
    tries (req abc123...)" and "... 7 tries ..." group together."""
    return _VOLATILE.sub("#", str(msg))[:160]


def summarize_events(path: str | Path, prices: Mapping[str, float] | None = None,
                     stall_s: float = 300.0) -> dict[str, Any]:
    """The usual questions about a run, answered from its event log.

    `prices` = {"in": $ per 1M input tokens, "out": $ per 1M output tokens}
    adds a cost line; nothing is priced by default.  A gap of more than
    `stall_s` seconds between consecutive unit events is reported as a
    stall."""
    ev = read_events(path)
    out: dict[str, Any] = {"unreadable_lines": ev.attrs.get("unreadable", 0)}
    if ev.empty:
        return {**out, "runs": []}
    starts = ev[ev["event"] == "run_start"]
    ends = ev[ev["event"] == "run_end"]
    units = ev[ev["event"] == "unit"]
    retries = ev[ev["event"] == "retry"]

    runs = []
    bounds = [*starts["t"], pd.Timestamp.max.tz_localize("UTC")]
    for k, (_, s) in enumerate(starts.iterrows()):
        lo, hi = bounds[k], bounds[k + 1]
        u = units[(units["t"] >= lo) & (units["t"] < hi)]
        e = ends[(ends["t"] >= lo) & (ends["t"] < hi)]
        last = (e["t"].iloc[0] if len(e) else
                u["t"].max() if len(u) else lo)
        runs.append({"started": lo.isoformat(), "ended": bool(len(e)),
                     "last_event": last.isoformat(),
                     "elapsed_s": float((last - lo).total_seconds()),
                     "planned": int(s.get("n_units", 0) or 0),
                     "to_go": int(s.get("to_go", 0) or 0),
                     "units_done": len(u)})
    out["runs"] = runs
    out["killed_runs"] = sum(not r["ended"] for r in runs)
    if not units.empty:
        out["status_counts"] = units["status"].value_counts().to_dict()
        errs = (units[units["error"].notna()] if "error" in units
                else units.iloc[:0])
        out["errors"] = []
        if len(errs):
            kinds = (errs.assign(kind=errs["error"].map(error_kind))
                     .groupby("kind").agg(n=("error", "size"),
                                          example=("error", "first"),
                                          first=("t", "min"),
                                          last=("t", "max"))
                     .sort_values("n", ascending=False))
            # item access, not attributes: on pandas < 3 `r.first` is the
            # Series.first METHOD, not this column
            out["errors"] = [{"kind": k, "n": int(r["n"]), "example": r["example"],
                              "first": r["first"].isoformat(),
                              "last": r["last"].isoformat()}
                             for k, r in kinds.iterrows()]
        lat = units["latency_s"].dropna() if "latency_s" in units else []
        if len(lat):
            out["latency_s"] = {"median": float(np.median(lat)),
                                "p95": float(np.quantile(lat, 0.95))}
        tok_in = float(units.get("in_tokens", pd.Series(dtype=float)).fillna(0).sum())
        tok_out = float(units.get("out_tokens", pd.Series(dtype=float)).fillna(0).sum())
        out["tokens"] = {"in": tok_in, "out": tok_out}
        if prices:
            out["cost"] = (tok_in * prices["in"] + tok_out * prices["out"]) / 1e6
        ts = units["t"].sort_values()
        gaps = ts.diff().dt.total_seconds()
        out["stalls"] = [{"after": ts.iloc[i - 1].isoformat(),
                          "gap_s": float(g)}
                         for i, g in enumerate(gaps) if g > stall_s]
        per_min = ts.dt.floor("min").value_counts().sort_index()
        out["units_per_minute"] = {"median": float(per_min.median()),
                                   "min": int(per_min.min()),
                                   "max": int(per_min.max())}
    if not retries.empty:
        out["retries"] = {"n": len(retries),
                          "wait_s": float(retries["wait_s"].sum()),
                          "server_requested": int(
                              retries.get("server_wait_s",
                                          pd.Series(dtype=float)).notna().sum()),
                          "by_kind": retries["reason"].map(error_kind)
                          .value_counts().head(5).to_dict()}
    return out


def parse_log(path: str | Path, patterns: Mapping[str, str],
              time_format: str | None = None) -> pd.DataFrame:
    """A foreign text log as an events frame.

    `patterns` maps an event name to a regex with named groups; a line
    matching it becomes {"event": name, **groups}, numeric-looking groups
    are converted, and a group named `t` is parsed as the time (with
    `time_format`, or pandas' parser).  Lines matching no pattern are
    counted in `attrs["unmatched"]` -- never silently dropped -- with up to
    five examples in `attrs["unmatched_examples"]`."""
    compiled = {k: re.compile(v) for k, v in patterns.items()}
    rows: list[dict[str, Any]] = []
    unmatched = 0
    examples: list[str] = []
    with Path(path).open(errors="replace") as fh:
        for raw in fh:
            line = raw.rstrip("\n")
            if not line.strip():
                continue
            for name, rx in compiled.items():
                m = rx.search(line)
                if m:
                    rec = {"event": name}
                    for g, v in m.groupdict().items():
                        rec[g] = _number(v)
                    rows.append(rec)
                    break
            else:
                unmatched += 1
                if len(examples) < 5:
                    examples.append(line[:200])
    df = pd.DataFrame(rows)
    if "t" in df:
        df["t"] = pd.to_datetime(df["t"], format=time_format, utc=True)
    df.attrs["unmatched"], df.attrs["unmatched_examples"] = unmatched, examples
    return df


def _number(v: str | None) -> Any:
    if v is None:
        return None
    try:
        return int(v)
    except ValueError:
        try:
            return float(v)
        except ValueError:
            return v


def format_summary(s: Mapping[str, Any]) -> str:
    """`summarize_events` output as a few readable lines."""
    lines = []
    for k, r in enumerate(s.get("runs", []), 1):
        state = "finished" if r["ended"] else "NO run_end (killed or still running)"
        lines.append(f"run {k}: {r['started']}  {r['units_done']}/{r['to_go']} "
                     f"units in {r['elapsed_s'] / 60:.1f} min  [{state}]")
    if "status_counts" in s:
        lines.append("status: " + ", ".join(f"{k} {v}" for k, v in
                                            s["status_counts"].items()))
    for e in s.get("errors", [])[:5]:
        lines.append(f"  {e['n']:>5} x {e['example'][:100]}")
    if "retries" in s:
        r = s["retries"]
        lines.append(f"retries: {r['n']} ({r['server_requested']} server-requested),"
                     f" {r['wait_s']:.0f} s waiting")
    if "latency_s" in s:
        lines.append(f"latency: median {s['latency_s']['median']:.2f} s, "
                     f"p95 {s['latency_s']['p95']:.2f} s")
    if "tokens" in s:
        cost = f", cost ${s['cost']:.4f}" if "cost" in s else ""
        lines.append(f"tokens: {s['tokens']['in']:.0f} in, "
                     f"{s['tokens']['out']:.0f} out{cost}")
    for st in s.get("stalls", [])[:5]:
        lines.append(f"stall: {st['gap_s'] / 60:.1f} min after {st['after']}")
    if s.get("unreadable_lines"):
        lines.append(f"unreadable lines: {s['unreadable_lines']}")
    return "\n".join(lines)
