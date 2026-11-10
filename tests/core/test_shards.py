"""Sharding one sweep across processes: the partition, the merge, and the
accounting when a shard never runs.  Expected values come from invariances
(a sharded sweep must reproduce the single-process one) and hand-built
environments, not from the shard assignment itself."""

from __future__ import annotations

import io
import shutil

import pytest
from support.runner import Answers, items, run

from evalcore import (
    attrition,
    grid,
    merge_shards,
    shard_from_env,
    shard_path,
    units,
)
from evalcore.core.shards import check_shard, shard_of

N_ITEMS, N_SHARDS = 30, 3


def sweep() -> list:
    """Two cells x 30 items: 60 units."""
    return list(units(grid({"model": ["a", "b"]}), items(N_ITEMS)))


def comparable(df) -> list[tuple]:
    return sorted(zip(df["unit_id"], df["model"], df["item_id"],
                      df["answer"], df["status"], strict=True))


def test_every_unit_lands_in_exactly_one_shard():
    us = sweep()
    got = [[u.id for u in us if shard_of(u.id, N_SHARDS) == i]
           for i in range(N_SHARDS)]
    flat = [uid for g in got for uid in g]
    assert sorted(flat) == sorted(u.id for u in us)      # cover, no repeats
    assert all(got)                                       # none empty here


def test_assignment_depends_on_the_unit_alone():
    """Not on order or on the other units: a reversed or halved sweep puts
    each unit in the same shard."""
    us = sweep()
    a = {u.id: shard_of(u.id, 7) for u in us}
    b = {u.id: shard_of(u.id, 7) for u in reversed(us[: len(us) // 2])}
    assert all(a[k] == v for k, v in b.items())


def test_sharded_sweep_reproduces_the_single_process_sweep(tmp_path):
    us = sweep()
    whole = run(us, Answers(), tmp_path / "whole.jsonl")
    for i in range(N_SHARDS):
        run(us, Answers(), shard_path(tmp_path / "r.jsonl", i, N_SHARDS),
            shard=(i, N_SHARDS))
    df, plan = merge_shards(tmp_path / "r.jsonl", N_SHARDS)
    assert comparable(df) == comparable(whole)
    assert sorted(plan["unit_id"]) == sorted(u.id for u in us)
    a = attrition(df, ["model"], manifest=plan)
    assert a["n_planned"].tolist() == [N_ITEMS, N_ITEMS]
    assert a["n_missing"].tolist() == [0, 0]


def test_a_shard_that_never_ran_is_counted_missing(tmp_path):
    """Shard 1 is never started, so it leaves no file at all.  Its units
    must still appear in the plan, as missing, and the shards that did run
    must not warn about units that were never theirs."""
    us = sweep()
    logs = []
    for i in (0, 2):
        log = io.StringIO()
        run(us, Answers(), shard_path(tmp_path / "r.jsonl", i, N_SHARDS),
            shard=(i, N_SHARDS), stream=log)
        logs.append(log.getvalue())
    assert not shard_path(tmp_path / "r.jsonl", 1, N_SHARDS).exists()
    assert not any("never attempted" in s for s in logs)

    df, plan = merge_shards(tmp_path / "r.jsonl", N_SHARDS)
    a = attrition(df, ["model"], manifest=plan)
    assert a["n_planned"].sum() == len(us)
    assert a["n_completed"].sum() == len(df)
    assert a["n_missing"].sum() == len(us) - len(df) > 0


def test_resuming_a_shard_calls_nothing_it_already_has(tmp_path):
    us, out = sweep(), shard_path(tmp_path / "r.jsonl", 1, N_SHARDS)
    run(us, Answers(), out, shard=(1, N_SHARDS))
    again = Answers()
    run(us, again, out, shard=(1, N_SHARDS))
    assert sum(again.calls.values()) == 0


def test_rows_of_one_unit_in_two_shards_are_refused(tmp_path):
    us = sweep()
    for i in range(N_SHARDS):
        run(us, Answers(), shard_path(tmp_path / "r.jsonl", i, N_SHARDS),
            shard=(i, N_SHARDS))
    shutil.copy(shard_path(tmp_path / "r.jsonl", 0, N_SHARDS),
                shard_path(tmp_path / "r.jsonl", 2, N_SHARDS))
    with pytest.raises(ValueError, match="Two partitions were mixed"):
        merge_shards(tmp_path / "r.jsonl", N_SHARDS)


def test_merging_with_no_shard_on_disk_is_refused(tmp_path):
    with pytest.raises(FileNotFoundError, match="no shard"):
        merge_shards(tmp_path / "r.jsonl", 4)


def test_shard_path_names():
    assert shard_path("d/runs.jsonl", 3, 16).name == "runs.shard03-of-16.jsonl"
    assert shard_path("runs.jsonl", 0, 1).name == "runs.shard0-of-1.jsonl"
    assert shard_path("runs.jsonl", 9, 10).name == "runs.shard9-of-10.jsonl"


@pytest.mark.parametrize("bad", [(3, 3), (-1, 3), (0, 0), (1.0, 3), (0, "3")])
def test_bad_shard_is_refused(bad, tmp_path):
    with pytest.raises(ValueError, match="shard must be"):
        check_shard(bad)
    with pytest.raises(ValueError, match="shard must be"):
        run(sweep(), Answers(), tmp_path / "r.jsonl", shard=bad)


def test_shard_from_a_slurm_array_counts_from_zero():
    env = {"SLURM_ARRAY_TASK_ID": "5", "SLURM_ARRAY_TASK_MIN": "1",
           "SLURM_ARRAY_TASK_COUNT": "16", "SLURM_ARRAY_TASK_STEP": "1"}
    assert shard_from_env(env) == (4, 16)


def test_shard_from_env_refuses_outside_an_array_and_strided_arrays():
    with pytest.raises(RuntimeError, match="not inside a SLURM array"):
        shard_from_env({"SLURM_JOB_ID": "7"})
    with pytest.raises(RuntimeError, match="strided"):
        shard_from_env({"SLURM_ARRAY_TASK_ID": "2", "SLURM_ARRAY_TASK_MIN": "0",
                        "SLURM_ARRAY_TASK_COUNT": "8",
                        "SLURM_ARRAY_TASK_STEP": "2"})
