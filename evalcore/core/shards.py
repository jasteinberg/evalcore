"""Split one sweep across independent processes (a SLURM array, several
machines) and merge the results.

    #SBATCH --array=0-15
    i, n = shard_from_env()                  # 3, 16 inside array task 3
    task.run(cells, backend, shard_path(out, i, n), shard=(i, n), cache=cache)

    # afterwards, anywhere
    df, plan = merge_shards(out, 16)
    attrition(df, by, manifest=plan)

A unit belongs to shard `sha256(unit_id) mod n`.  The assignment depends on
the unit alone, never on its position in the grid or on how many units
there are, so every process computes the same partition without talking to
the others, and a resumed array task picks up exactly the units it had.
Python's `hash` would not do: it is salted per process.

Each shard writes its own sink, events log and manifest.  Shared files
would rely on O_APPEND being atomic across machines, which network file
systems (NFS, Lustre) do not promise.  The response cache can be shared:
its writes are whole-file renames.

Each shard's manifest records the WHOLE plan, not just its own units.  A
shard that dies before its first write leaves no file at all, and its units
would vanish from the accounting if only it could vouch for them.  With the
whole plan in every manifest, `merge_shards` reports them as `n_missing` as
long as any shard started.

Changing `n` changes the partition, and `shard_path` puts `n` in the file
name, so a resharded sweep writes fresh sinks rather than mixing two
partitions.  The cache still holds every response already paid for.
"""

from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

from .records import (
    manifest_path,
    read_jsonl,
    read_manifest,
    to_frame,
    write_atomic,
)


def check_shard(shard: tuple[int, int]) -> tuple[int, int]:
    """`(i, n)` with n >= 1 and 0 <= i < n, or ValueError."""
    i, n = shard
    if not (isinstance(i, int) and isinstance(n, int)) or n < 1 \
            or not 0 <= i < n:
        raise ValueError(f"shard must be (i, n) with integers 0 <= i < n, "
                         f"got {shard!r}")
    return i, n


def shard_of(unit_id: str, n: int) -> int:
    """The shard `unit_id` belongs to, of `n`."""
    return int(hashlib.sha256(unit_id.encode()).hexdigest()[:16], 16) % n


def shard_path(out: str | Path, i: int, n: int) -> Path:
    """`runs.jsonl` -> `runs.shard03-of-16.jsonl`."""
    check_shard((i, n))
    q = Path(out)
    return q.with_name(f"{q.stem}.shard{i:0{len(str(n - 1))}d}-of-{n}"
                       f"{q.suffix}")


def shard_from_env(env: Mapping[str, str] | None = None) -> tuple[int, int]:
    """`(i, n)` for the current SLURM array task, counted from 0 whatever
    the array's first index.  Raises outside an array job, and for a strided
    array (`--array=0-15:2`), whose task ids do not number the shards."""
    env = os.environ if env is None else env
    need = ("SLURM_ARRAY_TASK_ID", "SLURM_ARRAY_TASK_MIN",
            "SLURM_ARRAY_TASK_COUNT")
    absent = [k for k in need if k not in env]
    if absent:
        raise RuntimeError(f"not inside a SLURM array task ({absent} unset). "
                           f"Pass shard=(i, n) explicitly instead")
    if env.get("SLURM_ARRAY_TASK_STEP", "1") != "1":
        raise RuntimeError(
            f"strided array (step {env['SLURM_ARRAY_TASK_STEP']}): task ids "
            f"are not 0..n-1. Use a contiguous --array range")
    i = int(env["SLURM_ARRAY_TASK_ID"]) - int(env["SLURM_ARRAY_TASK_MIN"])
    return check_shard((i, int(env["SLURM_ARRAY_TASK_COUNT"])))


def merge_shards(out: str | Path, n: int) -> tuple[pd.DataFrame, pd.DataFrame]:
    """(the frame of every shard's rows, the plan) for a sweep run as `n`
    shards of `out`.  Pass the plan to `attrition(..., manifest=)`.

    A shard with no sink is not an error: its units are in the other
    shards' manifests and are counted missing.  No sink at all is, and so
    is a unit with rows in two shards, which means two partitions were
    mixed (n changed under one file name, or a file was copied)."""
    paths = [shard_path(out, i, n) for i in range(n)]
    sinks = [p for p in paths if p.exists()]
    if not sinks:
        raise FileNotFoundError(f"no shard of {str(out)!r} exists (looked for "
                                f"{paths[0].name} ... {paths[-1].name})")
    frames: list[pd.DataFrame] = []
    owner: dict[str, int] = {}
    for i, p in enumerate(paths):
        f = to_frame(p) if p.exists() else pd.DataFrame()
        if f.empty:
            continue
        for uid in f["unit_id"]:
            if owner.setdefault(uid, i) != i:
                raise ValueError(
                    f"unit {uid!r} has rows in shards {owner[uid]} and {i}. "
                    f"Two partitions were mixed")
        frames.append(f)
    df = (pd.concat(frames, ignore_index=True) if frames else pd.DataFrame())
    plans = [m for p in paths if manifest_path(p).exists()
             for m in [read_manifest(manifest_path(p))] if not m.empty]
    plan = (pd.concat(plans, ignore_index=True)
            .drop_duplicates(subset="unit_id", keep="first")
            .reset_index(drop=True) if plans else pd.DataFrame())
    return df, plan


MERGED_SUFFIX = ".shards.json"


def combine_shards(out: str | Path, n: int) -> Path:
    """Write the `n` shards of `out` as one sink at `out`, with one manifest
    beside it, so everything that reads a sink (`to_frame`, `attrition`, the
    analyses, a later unsharded resume) reads the sweep as if one process
    had run it.  Returns `out`.

    The rows are the shards' lines verbatim, after `merge_shards` has
    checked that no unit is in two shards.  The manifest keeps each planned
    unit's first record.  `<out>.shards.json` records that `out` is a merge,
    and a merge overwrites only a previous merge: an `out` written any other
    way holds rows that exist nowhere else, so it is refused."""
    out = Path(out)
    mark = out.with_name(out.name + MERGED_SUFFIX)
    if out.exists() and not mark.exists():
        raise FileExistsError(
            f"{str(out)!r} exists and is not a previous merge. Move it away "
            f"first: merging would overwrite rows that exist only there")
    merge_shards(out, n)                              # refuses mixed partitions
    paths = [shard_path(out, i, n) for i in range(n)]
    # a crash can leave a shard's last line unterminated, and the next
    # shard's first row must not be glued onto it
    texts = [p.read_text() for p in paths if p.exists()]
    rows = "".join(t if t.endswith("\n") or not t else t + "\n"
                   for t in texts)
    seen: set[str] = set()
    plan: list[str] = []
    for p in paths:
        for rec in (read_jsonl(manifest_path(p))[0]
                    if manifest_path(p).exists() else []):
            if isinstance(rec, dict) and rec.get("unit_id") not in seen:
                seen.add(rec["unit_id"])
                plan.append(json.dumps(rec) + "\n")
    write_atomic(out, lambda tmp: tmp.write_text(rows))
    write_atomic(manifest_path(out), lambda tmp: tmp.write_text("".join(plan)))
    write_atomic(mark, lambda tmp: tmp.write_text(json.dumps(
        {"n": n, "shards": [p.name for p in paths if p.exists()]})))
    return out
