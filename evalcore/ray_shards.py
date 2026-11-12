"""Run a sharded experiment on a Ray cluster, then merge it.

    from evalcore.ray_shards import run_on_ray
    run_on_ray(partial(from_config, "config.json"), n_shards=256, actors=16,
               num_gpus=1)

What Ray adds to `--shard` under SLURM or a Kubernetes Indexed Job:

  * Each actor builds the experiment once (`build()`, which loads a local
    model) and then runs shard after shard, so a model is loaded once per
    actor, not once per shard.
  * Shards go to whichever actor is free (`ActorPool`), so with many more
    shards than actors (256 over 16) a slow shard delays only itself, and
    the wall time approaches total work / actors instead of the slowest
    of 16 fixed shares.
  * An actor that dies is restarted and its shard retried, and the retry
    resumes from the shard's sink, so no finished unit is called again.

The partition, the sinks, the manifests and the merge are the ones of
core/shards.py: nothing about a unit's result depends on Ray.  Every actor
and the merge must see one file system (the sinks are appended to and the
cache writes by rename).  Run `--dry-run` unsharded first so that the
shards share one calibration instead of each calibrating.

Ray is imported here only, so it is not a dependency of evalcore.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from .core.shards import check_shard
from .pipeline import Experiment

__all__ = ["run_on_ray"]


class _ShardWorker:
    """Holds one built experiment and runs the shards it is given."""

    def __init__(self, build: Callable[[], Experiment]) -> None:
        self.exp = build()

    def run(self, i: int, n: int) -> tuple[int, str | None]:
        res = self.exp.sharded(i, n).run()
        return i, res.halted

    def merge(self, n: int) -> dict[str, Any]:
        res = self.exp.merge(n)
        return {"out": str(res.out), "report": str(res.report_path),
                "marker": res.marker}


def run_on_ray(build: Callable[[], Experiment], n_shards: int, *,
               actors: int, num_gpus: float = 0.0, num_cpus: float = 1.0,
               max_restarts: int = 2) -> dict[str, Any]:
    """Run `build()`'s experiment as `n_shards` shards over `actors` Ray
    actors, then merge it on one of them.  Returns the merge's output
    paths and marker, and each shard's halt reason (None for a shard that
    ran to the end).  `build` must be picklable (a module-level function,
    or `functools.partial(from_config, path)`).  Connects to the cluster
    `ray.init()` finds, or starts a local one.  A shard that halted (a
    cost ceiling) is reported, not merged over: the merge runs only when
    every shard finished."""
    import ray
    from ray.util import ActorPool

    check_shard((0, n_shards))
    if actors < 1:
        raise ValueError(f"actors must be >= 1, got {actors}")
    if not ray.is_initialized():
        ray.init()
    worker = ray.remote(num_gpus=num_gpus, num_cpus=num_cpus,
                        max_restarts=max_restarts,
                        max_task_retries=max_restarts)(_ShardWorker)
    pool_actors = [worker.remote(build) for _ in range(min(actors, n_shards))]
    pool = ActorPool(pool_actors)
    halted = dict(pool.map_unordered(lambda a, i: a.run.remote(i, n_shards),
                                     range(n_shards)))
    stopped = {i: h for i, h in halted.items() if h is not None}
    out: dict[str, Any] = {"shards": n_shards,
                           "halted": dict(sorted(stopped.items()))}
    if not stopped:
        out |= ray.get(pool_actors[0].merge.remote(n_shards))
    for a in pool_actors:
        ray.kill(a)
    return out
