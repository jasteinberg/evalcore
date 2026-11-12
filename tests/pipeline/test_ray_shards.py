"""run_on_ray: a sharded run over Ray actors merges to the rows of the same
run unsharded, and each actor builds its experiment (its model) once,
however many shards it runs.  Skipped where Ray is not installed (it is
not a dependency of evalcore)."""

from __future__ import annotations

import functools
import json
import sys
from pathlib import Path

import pytest

ray = pytest.importorskip("ray")

from evalcore import from_config, to_frame  # noqa: E402
from evalcore.ray_shards import run_on_ray  # noqa: E402

COLS = ["unit_id", "item_id", "text", "status", "correct"]


def cfg(tmp_path: Path, out: str) -> Path:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps(
        {"id": f"t{i}", "prompt": f"say {i}", "gold": i, "grader": "numeric"})
        for i in range(24)))
    p = tmp_path / f"{out}.json"
    p.write_text(json.dumps({
        "out": str(tmp_path / "runs" / f"{out}.jsonl"),
        "task": {"type": "file", "path": str(tasks)},
        "backend": {"type": "echo"}, "grid": {"model": ["echo"]},
        "metric": "correct", "options": {"workers": 1}}))
    return p


def build(path: Path, count: Path):
    """What an actor runs once: one line per build in `count`."""
    with count.open("a") as fh:
        fh.write("built\n")
    return from_config(path)


@pytest.fixture(scope="module")
def local_ray():
    ray.cloudpickle.register_pickle_by_value(sys.modules[__name__])
    ray.init(num_cpus=2, include_dashboard=False, log_to_driver=False)
    yield
    ray.shutdown()


def test_ray_shards_merge_to_the_unsharded_rows_with_one_build_per_actor(
        tmp_path, local_ray):
    ref = from_config(cfg(tmp_path, "ref")).run()
    want = to_frame(ref.out)[COLS].sort_values("unit_id").reset_index(drop=True)

    count = tmp_path / "builds.txt"
    res = run_on_ray(functools.partial(build, cfg(tmp_path, "ray"), count),
                     n_shards=6, actors=2)
    assert res["halted"] == {}
    got = to_frame(res["out"])[COLS].sort_values("unit_id").reset_index(
        drop=True)
    assert got.equals(want) and len(got) == 24
    assert count.read_text().count("built") == 2       # not 6
    assert sorted(p.name for p in (tmp_path / "runs").glob(
        "ray.shard*-of-6.jsonl")) == [f"ray.shard{i}-of-6.jsonl"
                                      for i in range(6)]
