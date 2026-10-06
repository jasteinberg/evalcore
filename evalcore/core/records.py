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
import warnings
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .spec import Unit

MANIFEST_SUFFIX = ".manifest.jsonl"
STATUSES = ("ok", "error", "tool_error", "truncated", "unparsed")

_MANIFEST_LOCK = threading.Lock()


class SinkState:
    """What a sink already holds, read without opening it for writing:
    for each unit_id its LAST row's request_key (None for rows written
    before request keys existed) and whether that row errored, last-wins
    as in `to_frame`; plus how many lines could not be read."""

    def __init__(self, path: str | Path) -> None:
        self.seen: dict[str, str | None] = {}
        self.errored: set[str] = set()
        self.unreadable = 0
        p = Path(path)
        if p.exists():
            with p.open() as fh:
                for line in fh:
                    if not line.strip():
                        continue
                    try:
                        self.note(json.loads(line))
                    except (json.JSONDecodeError, KeyError, TypeError,
                            AttributeError):
                        self.unreadable += 1

    def note(self, row: Mapping[str, Any]) -> None:
        uid = row["unit_id"]
        self.seen[uid] = row.get("request_key")
        if row.get("error"):
            self.errored.add(uid)
        else:
            self.errored.discard(uid)


class JsonlSink(SinkState):
    """Thread-safe append-only sink, flushed per row; keeps its SinkState
    current as rows are written."""

    def __init__(self, path: str | Path) -> None:
        super().__init__(path)
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        _terminate_partial_line(self.path)
        self.fh = self.path.open("a")

    def write(self, row: dict) -> None:
        with self.lock:
            self.fh.write(json.dumps(row, default=json_default) + "\n")
            self.fh.flush()
            self.note(row)

    def close(self) -> None:
        self.fh.close()


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
        with path.open("a") as fh:        # O_APPEND: additive across processes
            for rec in fresh:
                fh.write(json.dumps(rec, default=str) + "\n")
            fh.flush()
            os.fsync(fh.fileno())
    return path


def _read_records(path: str | Path) -> list[dict]:
    """Tolerant line reader: a truncated or corrupt manifest degrades to the
    records that do parse.  It is a record of intent, and a partial record of
    intent is still better than none."""
    path = Path(path)
    if not path.exists():
        return []
    out = []
    with path.open() as fh:
        for raw in fh:
            line = raw.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict) and "unit_id" in rec:
                out.append(rec)
    return out


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
    df["status"] = _status(df)
    if "error" in df and drop_errors:
        df = df[df["error"].isna()].reset_index(drop=True)
    return df


def _status(df: pd.DataFrame) -> pd.Series:
    """Row status, derived for frames that predate the column.  A row with an
    error is never "ok": it keeps a failure status it was given
    (`tool_error`) and is "error" otherwise."""
    err = (df["error"].notna() if "error" in df
           else pd.Series(False, index=df.index))
    st = (df["status"] if "status" in df
          else pd.Series(None, index=df.index, dtype=object))
    st = st.where(st.notna(), "ok").astype(object)
    return st.where(~err | st.isin(["error", "tool_error"]), "error")


# A unit with several rows (a judge panel) takes its worst row's status.


# --- numpy values and array sidecars ------------------------------------------
# A response's meta may carry numpy values from a probe.  JSON
# would stringify a numpy scalar silently (a float32 0.25 came back as the
# string "0.25") and refuse an array outright, so: scalars become Python
# numbers, nested arrays become lists, and top-level arrays -- per-head or
# per-layer tensors, megabytes per unit -- go to an .npz sidecar named by the
# unit's request key, with only its path left in the row.

def json_default(o: object) -> object:
    """`default=` for json.dumps: numpy scalars to Python, arrays to lists,
    anything else to its string (the previous behaviour)."""
    if isinstance(o, np.generic):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    return str(o)


def _as_array(v: object) -> np.ndarray | None:
    if isinstance(v, np.ndarray) and v.ndim > 0:
        return v
    if hasattr(v, "detach") and hasattr(v, "shape") and len(v.shape) > 0:
        return v.detach().cpu().numpy()          # a torch tensor
    return None


def split_arrays(meta: dict) -> tuple[dict, dict]:
    """(meta with arrays removed and numpy scalars made Python, arrays)."""
    plain, arrays = {}, {}
    for k, v in meta.items():
        a = _as_array(v)
        if a is not None:
            arrays[k] = a
        elif isinstance(v, np.generic):
            plain[k] = v.item()
        else:
            plain[k] = v
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
        tmp = p.with_suffix(f".{os.getpid()}.{threading.get_ident()}.tmp.npz")
        np.savez_compressed(tmp, **arrays)
        os.replace(tmp, p)                      # atomic, like the cache
        return p


def load_arrays(out: str | Path, rel: str) -> dict:
    """The arrays of one row: `rel` is its `meta_arrays` value, a path
    relative to the sink's directory."""
    with np.load(Path(out).parent / rel) as z:
        return {k: z[k] for k in z.files}
