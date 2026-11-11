"""A scenario run one unit at a time, by successive single-unit `execute`
calls or by one `Runner`, and the files it leaves, read into one comparable
value.

The golden file tests/golden/incremental/execute_per_unit.json is this
scenario's `files("execute", ...)` as written by `execute` at commit
9c95d2f, before `execute` was built on `Runner`.  It is the independent
reference for "a Runner writes what N single-unit execute calls wrote".
Regenerate it only from that commit's code:

    git archive 9c95d2f | tar -x -C <dir>
    PYTHONPATH=<dir>:tests python -m support.incremental <out.json>

The scenario covers every path a row can take: a fresh call with an array
sidecar and a numpy scalar in meta, an error response, a truncated
response, a parse failure, cache hits into a second sink, and a re-score
under a new score_version with an errored unit retried.
"""

from __future__ import annotations

import io
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from evalcore import Backend, Cache, ParseFailure, Request, Response, Unit, grid, units
from evalcore.core.records import load_arrays

from .runner import items, render

# Event fields that differ between two runs of the same units by design:
# the wall-clock time, and run_start/run_end, which N execute calls write
# N times and a Runner once.
VOLATILE_EVENTS = {"run_start", "run_end"}


class Scripted(Backend):
    """Item i answers "a{i}" with an array and a numpy scalar in meta,
    except item 3 (refused: an error response) and item 4 (truncated)."""

    name = "scripted"

    def __init__(self) -> None:
        self.calls: list[str] = []

    def identity(self) -> dict:
        return {"backend": "scripted"}

    def complete(self, req: Request) -> Response:
        i = int(req.messages[-1]["content"].split()[-1])
        self.calls.append(req.unit_id)
        if i == 3:
            return Response(req.unit_id, error="fatal: refused")
        if i == 4:
            return Response(req.unit_id, text="cut", meta={"truncated": True})
        return Response(req.unit_id, text=f"a{i}",
                        meta={"truncated": False, "n": np.int64(i),
                              "acts": np.arange(3, dtype=float) * i})


def score(u: Unit, r: Response) -> dict[str, Any]:
    """Item 5's answer cannot be read.  Otherwise correct on even items."""
    i = int(u.item.payload["q"].split()[-1])
    if i == 5:
        raise ParseFailure("no answer in a5")
    return {"correct": int(i % 2 == 0), "length": len(r.text or "")}


def scenario_units() -> list[Unit]:
    return list(units(grid({"model": ["m"]}), items(6)))


def run_scenario(mode: str, root: Path) -> None:
    """Three passes, one unit at a time: into rows.jsonl (fresh calls),
    into rows2.jsonl (every success a cache hit), and into rows.jsonl again
    under score_version v2 (re-scores, and the errored unit retried)."""
    us = scenario_units()
    cache = Cache(root / "cache")
    for out, sv in (("rows.jsonl", "v1"), ("rows2.jsonl", "v1"),
                    ("rows.jsonl", "v2")):
        backend = Scripted()
        if mode == "execute":
            from evalcore import execute
            for u in us:
                execute([u], backend, render, score, root / out, cache=cache,
                        workers=1, batch_size=1, score_version=sv,
                        stream=io.StringIO())
        elif mode == "runner":
            from evalcore import Runner
            with Runner(backend, render, score, root / out, cache=cache,
                        score_version=sv, stream=io.StringIO()) as r:
                for u in us:
                    r.run(u)
        else:
            raise ValueError(mode)


def _lines(p: Path) -> list[Any]:
    return [json.loads(x) for x in p.read_text().splitlines() if x.strip()]


def files(mode: str, root: Path) -> dict[str, Any]:
    """Run the scenario under `root` and read back everything it wrote.
    Sidecars are compared by content: the .npz bytes carry zip timestamps."""
    run_scenario(mode, root)
    out: dict[str, Any] = {}
    for name in ("rows.jsonl", "rows2.jsonl"):
        out[name] = _lines(root / name)
        out[name + ".manifest.jsonl"] = _lines(root / (name + ".manifest.jsonl"))
        out[name + ".events"] = [
            {k: v for k, v in e.items() if k != "t"}
            for e in _lines(root / (name + ".events.jsonl"))
            if e["event"] not in VOLATILE_EVENTS]
    cache = root / "cache"
    out["cache"] = {str(p.relative_to(cache)): json.loads(p.read_text())
                    for p in sorted(cache.rglob("*.json"))}
    out["sidecars"] = {
        rel: {k: v.tolist() for k, v in load_arrays(root / "rows.jsonl",
                                                    rel).items()}
        for rel in sorted({r["meta_arrays"] for r in out["rows.jsonl"]
                           if r.get("meta_arrays")})}
    return out


if __name__ == "__main__":
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        result = files("execute", Path(d))
    Path(sys.argv[1]).write_text(json.dumps(result, indent=1, sort_keys=True))
