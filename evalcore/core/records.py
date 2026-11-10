"""The storage layer: an append-only JSONL sink, the plan manifest, and
the reader that turns a sink into the tidy frame.

* Append-only, flushed per result.  A sweep that dies eight hours in must
  lose nothing, and a write cut off by a crash is repaired by terminating
  the fragment, never by rewriting what is already on disk.
* The manifest is a record of intent written before the first call: what a
  sweep was going to do, so units never attempted can be counted.  It is
  never a second source of truth -- resume reads the sink.
"""

from __future__ import annotations

import json
import os
import threading
import uuid
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .spec import Unit

MANIFEST_SUFFIX = ".manifest.jsonl"
STATUSES = ("ok", "error", "tool_error", "truncated", "unparsed")

_MANIFEST_LOCK = threading.Lock()


def utc_now(timespec: str = "seconds") -> str:
    """The current UTC time in ISO 8601, the one timestamp format every file
    written here uses."""
    return datetime.now(timezone.utc).isoformat(timespec=timespec)


def write_atomic(path: Path, write: Callable[[Path], object],
                 suffix: str = ".tmp") -> None:
    """Write `path` through a sibling temp file, then rename it over `path`,
    so a reader sees the old file or the new one, never half of one.

    The temp name is unique per call, not per path.  Two writers can hold
    one path at once (repeats of an identical request at a non-zero
    temperature are identical by construction), and with a shared temp
    the second os.replace raises FileNotFoundError because the first
    already moved the file away.  Unique temp, last writer wins, still
    atomic.  `suffix` ends the temp name: np.savez appends ".npz" to a
    name not already ending so, and readers skip names ending ".tmp.npz"."""
    tmp = path.with_name(f"{path.name}.{uuid.uuid4().hex}{suffix}")
    write(tmp)
    os.replace(tmp, path)


class SinkState:
    """What a sink already holds, read without opening it for writing:
    for each unit_id its LAST row's request_key (None for rows written
    before request keys existed) and whether that row errored, last-wins
    as in `to_frame`; plus how many lines could not be read."""

    def __init__(self, path: str | Path) -> None:
        self.seen: dict[str, str | None] = {}
        self.errored: set[str] = set()
        rows, self.unreadable = (read_jsonl(path) if Path(path).exists()
                                 else ([], 0))
        for row in rows:
            try:
                self.note(row)
            except (KeyError, TypeError, AttributeError):   # not a row
                self.unreadable += 1

    def note(self, row: Mapping[str, Any]) -> None:
        uid = row["unit_id"]
        self.seen[uid] = row.get("request_key")
        if row.get("error"):
            self.errored.add(uid)
        else:
            self.errored.discard(uid)


def _open_append(path: Path) -> int:
    return os.open(path, os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o644)


def _write_line(fd: int, rec: Mapping[str, Any], path: Path,
                default: Callable[[Any], Any]) -> None:
    """Append one record as ONE write(2) on an O_APPEND descriptor.

    O_APPEND makes each write land at the current end of the file, and a
    single write to a local regular file is not interleaved with another
    process's.  A buffered file object gives neither guarantee per record.
    It flushes in fixed-size chunks, so a record can straddle two writes and
    a concurrent writer's record can land between its halves (a torn line,
    dropped on read).  Between two of these writes the file always ends in a
    newline, so `_terminate_partial_line` in another writer cannot mistake a
    live writer's record for a crash fragment.  The bytes reach the OS when
    the call returns, as a flush would.  A short write raises."""
    data = (json.dumps(rec, default=default) + "\n").encode()
    n = os.write(fd, data)
    if n != len(data):
        raise OSError(f"short write to {path}: {n} of {len(data)} bytes")


class AppendLog:
    """Thread-safe append-only JSONL writer, one write(2) per record
    (`_write_line`).  A fragment a kill left mid-write is terminated before
    the first append, so the next record is not glued onto it and lost."""

    def __init__(self, path: str | Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        _terminate_partial_line(self.path)
        self.fd = _open_append(self.path)

    def write(self, rec: Mapping[str, Any]) -> None:
        with self.lock:
            _write_line(self.fd, rec, self.path, json_default)
            self._written(rec)

    def _written(self, rec: Mapping[str, Any]) -> None:
        """Called under the lock after each record is on disk."""

    def close(self) -> None:
        """Idempotent: a raw descriptor closed twice could close whatever
        file has since been given the same number."""
        if self.fd >= 0:
            os.close(self.fd)
            self.fd = -1


class JsonlSink(SinkState, AppendLog):
    """The run's sink: an AppendLog that keeps its SinkState current as rows
    are written."""

    def __init__(self, path: str | Path) -> None:
        SinkState.__init__(self, path)      # read before the fragment is closed
        AppendLog.__init__(self, path)

    def _written(self, rec: Mapping[str, Any]) -> None:
        self.note(rec)


def _terminate_partial_line(path: Path) -> bool:
    """Close an unterminated final line before anything is appended.

    A process killed inside `write` leaves a fragment with no newline.
    Appending after it glues the next record onto the fragment, and the
    reader then loses both: the interrupted unit is re-run on resume, and
    its fresh row is unreadable too, which on the sink also made every later
    `to_frame` raise.  Terminating the fragment confines the damage to the
    one record the crash interrupted, and rewrites nothing already on disk.
    """
    if not path.exists() or path.stat().st_size == 0:
        return False
    with path.open("rb") as fh:
        fh.seek(-1, os.SEEK_END)
        if fh.read(1) == b"\n":
            return False
    with path.open("a") as fh:
        fh.write("\n")
    return True


# --- the plan manifest ------------------------------------------------------
# One record per PLANNED unit, written before the first call.  The record is
# `Unit.row()` -- the same identity columns the tidy frame carries -- so the
# manifest joins to the frame on unit_id and can be grouped by any cell
# param or tag without a second lookup table.
#
# Why the plan has to be on disk rather than inferred: units are dispatched
# cell-major (`spec.units` loops cells, then items, then repeats), so a
# sweep killed at fraction f of the plan does not lose a uniform 1-f of every
# cell -- it loses the LAST cells entirely.  The missingness is maximally
# confounded with the grid axis, which is the one thing a per-cell table is
# meant to contrast.  Inferring "we probably meant to run more" from the
# frame is impossible for exactly the cells where it matters: a cell with
# zero rows has no rows to notice.

def manifest_path(out: str | Path) -> Path:
    """Manifest belonging to a JSONL sink: `runs.jsonl` ->
    `runs.jsonl.manifest.jsonl`.

    Built from the sink's full name rather than its stem, so the map is
    injective.  On the stem, `runs.jsonl` and `runs.json` in one directory
    share a manifest, and each sweep then reports the other's units as
    never attempted -- inventing precisely the number this file exists to
    report.  An ugly name is cheaper than a fabricated one.
    """
    q = Path(out)
    return q.with_name(q.name + MANIFEST_SUFFIX)


def write_manifest(units: Iterable[Unit], path: str | Path) -> Path:
    """Append the planned units to `path`, skipping those already recorded.

    Append rather than rewrite, for the same reason the sink is append-only.
    A read-modify-write loses the plan of any concurrent writer: two
    processes sharding one sweep both read the file, both write their own
    union, and the second erases the first -- a silent loss of exactly the
    evidence this file exists to keep.  Appends never lose a record; at
    worst two writers both append the same unit, and `read_manifest`
    deduplicates on the way in.

    The union is over the sink's whole history, not one invocation.  A sweep
    that planned 200, died at 50 and is restarted with `limit=60` must still
    report the 140 nobody has ever attempted.

    First record of a unit_id wins, since re-appending an already-planned
    unit to refresh a renamed tag would grow the file on every resume for a
    field that bears no identity.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with _MANIFEST_LOCK:                  # serialises this process's threads
        have = {r["unit_id"] for r in _read_records(path)}
        _terminate_partial_line(path)
        fresh = []
        for u in units:
            if u.id not in have:
                have.add(u.id)
                fresh.append(u.row())
        _append_records(path, fresh)
    return path


def _append_records(path: Path, recs: Sequence[Mapping[str, Any]]) -> None:
    """The manifest's records, each one `_write_line`, then fsync: the plan
    must be on disk before the first call."""
    fd = _open_append(path)
    try:
        for rec in recs:
            _write_line(fd, rec, path, str)
        os.fsync(fd)
    finally:
        os.close(fd)


def read_jsonl(path: str | Path) -> tuple[list[Any], int]:
    """(the parsed values of every non-blank line, the number of lines that
    did not parse).  A missing file raises: callers that allow one check."""
    rows, bad = [], 0
    with Path(path).open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                bad += 1
    return rows, bad


def _read_records(path: str | Path) -> list[dict]:
    """Tolerant line reader: a truncated or corrupt manifest degrades to the
    records that do parse.  It is a record of intent, and a partial record of
    intent is still better than none.  Lines that do not parse are counted
    in a RuntimeWarning, not dropped silently: each is a planned unit the
    attrition table can no longer report as missing."""
    if not Path(path).exists():
        return []
    rows, bad = read_jsonl(path)
    if bad:
        warnings.warn(f"{path}: {bad} manifest line(s) did not parse and are "
                      f"left out of the plan (usually a write cut off by a "
                      f"crash)", RuntimeWarning, stacklevel=2)
    return [r for r in rows if isinstance(r, dict) and "unit_id" in r]


def read_manifest(path: str | Path) -> pd.DataFrame:
    """Manifest -> frame, one row per planned unit.  A missing file is an
    empty frame, not an error."""
    recs = _read_records(path)
    if not recs:
        return pd.DataFrame()
    return (pd.DataFrame(recs)
            .drop_duplicates(subset="unit_id", keep="last")
            .reset_index(drop=True))


def to_frame(path: str | Path, drop_errors: bool = False) -> pd.DataFrame:
    """JSONL -> tidy frame, one row per (cell, item, repeat).

    Deduplicates on unit_id keeping the LAST occurrence, so re-running with
    a fixed scorer over the same sink supersedes the earlier rows without
    anyone having to delete a file by hand.

    An unreadable line (the fragment a crash leaves mid-write) is skipped
    with a RuntimeWarning giving the count, never silently.  Its unit is
    then absent from the frame, which `attrition(..., manifest=...)` reports
    as missing, and resume re-runs it.
    """
    rows, bad = read_jsonl(path)
    if bad:
        warnings.warn(
            f"{str(path)!r}: skipped {bad} unreadable line(s); a write "
            f"interrupted by a crash leaves one, and its unit is re-run on "
            f"resume", RuntimeWarning, stacklevel=2)
    if not rows:
        return pd.DataFrame()
    df = pd.DataFrame(rows)
    df = df.drop_duplicates(subset="unit_id", keep="last").reset_index(drop=True)
    # rows written before `status` existed: derived from `error` alone, the
    # only distinction those rows recorded
    df["status"] = row_status(df)
    if "error" in df and drop_errors:
        df = df[df["error"].isna()].reset_index(drop=True)
    return df


def row_status(df: pd.DataFrame) -> pd.Series:
    """Row status, derived for frames that predate the column.  A row with an
    error is never "ok": it keeps a failure status it was given
    (`tool_error`) and is "error" otherwise."""
    err = (df["error"].notna() if "error" in df
           else pd.Series(False, index=df.index))
    st = (df["status"] if "status" in df
          else pd.Series(None, index=df.index, dtype=object))
    st = st.where(st.notna(), "ok").astype(object)
    return st.where(~err | st.isin(["error", "tool_error"]), "error")


# --- numpy values and array sidecars ------------------------------------------
# A response's meta may carry numpy or torch values from a probe.  JSON
# would stringify a scalar silently (a float32 0.25 came back as the string
# "0.25", a torch scalar as "tensor(0.2500)") and refuse an array outright,
# so: scalars become Python numbers at any depth, nested arrays become
# lists, and top-level arrays -- per-head or per-layer tensors, megabytes per
# unit -- go to an .npz sidecar named by the unit's request key, with only
# its path left in the row.  Torch is duck-typed (`detach`): no torch import.

def _is_tensor(o: object) -> bool:
    return hasattr(o, "detach") and hasattr(o, "shape")


def json_default(o: object) -> object:
    """`default=` for json.dumps: numpy and torch scalars to Python, arrays
    to lists, anything else to its string (the previous behaviour)."""
    if _is_tensor(o):
        o = o.detach().cpu().numpy()          # type: ignore[attr-defined]
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _plain(v: object) -> object:
    """`v` with every numpy/torch scalar, at any depth of dicts, lists and
    tuples, made a Python number, and nested arrays made lists, so the
    cache (plain `json.dumps`) and the sink store the same thing."""
    if isinstance(v, dict):
        return {k: _plain(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_plain(x) for x in v]
    if _is_tensor(v) or isinstance(v, (np.generic, np.ndarray)):
        return json_default(v)
    return v


def _as_array(v: object) -> np.ndarray | None:
    if isinstance(v, np.ndarray) and v.ndim > 0:
        return v
    if _is_tensor(v) and len(v.shape) > 0:       # type: ignore[attr-defined]
        return v.detach().cpu().numpy()          # type: ignore[attr-defined]
    return None


def split_arrays(meta: dict) -> tuple[dict, dict]:
    """(meta with top-level arrays removed and every scalar made Python,
    arrays)."""
    plain, arrays = {}, {}
    for k, v in meta.items():
        a = _as_array(v)
        if a is not None:
            arrays[k] = a
        else:
            plain[k] = _plain(v)
    return plain, arrays


class ArrayStore:
    """Content-addressed .npz files, one per request key."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)

    def path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.npz"

    def put(self, key: str, arrays: dict) -> Path:
        p = self.path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        write_atomic(p, lambda tmp: np.savez_compressed(tmp, **arrays),
                     suffix=".tmp.npz")
        return p


def load_arrays(out: str | Path, rel: str) -> dict:
    """The arrays of one row: `rel` is its `meta_arrays` value, a path
    relative to the sink's directory."""
    with np.load(Path(out).parent / rel) as z:
        return {k: z[k] for k in z.files}
