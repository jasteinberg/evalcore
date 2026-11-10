"""A scorer change with a version bump re-scores the stored responses:
no model call, the new scores in the frame, and the marker "complete" for
rows that carry the new version.  Expected values are the two scorers'
constant outputs, by construction."""

from __future__ import annotations

import json
from dataclasses import replace

import pytest

from evalcore import FunctionBackend, Item, grid, to_frame
from evalcore.pipeline import Experiment, RunOptions
from evalcore.tasks.base import Task

pytestmark = pytest.mark.filterwarnings("ignore:.*carry caveats:RuntimeWarning")


def experiment(tmp_path, task: Task, calls: list[str]) -> Experiment:
    def model(req):
        calls.append(req.unit_id)
        return "an answer"
    be = FunctionBackend(model, identity={"m": "const"})
    return Experiment(task, be, grid({"model": ["m"]}), tmp_path / "r.jsonl",
                      metric="correct", calibrate=None,
                      options=RunOptions(workers=1))


def v1() -> Task:
    items = [Item(f"i{k}", {"messages": [{"role": "user", "content": f"q{k}"}]})
             for k in range(4)]
    return Task("t", items, score=lambda u, r: {"correct": 0.0}, version="1")


def test_a_version_bump_rescores_without_calling_the_model(tmp_path):
    calls: list[str] = []
    experiment(tmp_path, v1(), calls).run()
    assert len(calls) == 4
    t2 = replace(v1(), score=lambda u, r: {"correct": 1.0}, version="2")
    res = experiment(tmp_path, t2, calls).run()
    assert len(calls) == 4                                  # no new call
    df = to_frame(tmp_path / "r.jsonl")
    assert df["correct"].tolist() == [1.0] * 4
    assert set(df["score_version"]) == {"2"}
    assert res.marker["status"] == "complete"
    assert res.dry_run["rescore"] == 4 and res.dry_run["to_call"] == 0
    events = [json.loads(x)["event"] for x in
              (tmp_path / "r.jsonl.events.jsonl").read_text().splitlines()]
    assert events.count("rescore") == 4


def test_the_same_version_rescores_nothing(tmp_path):
    calls: list[str] = []
    experiment(tmp_path, v1(), calls).run()
    n_lines = len((tmp_path / "r.jsonl").read_text().splitlines())
    res = experiment(tmp_path, v1(), calls).run()
    assert res.skipped                       # complete under the same job
    assert len((tmp_path / "r.jsonl").read_text().splitlines()) == n_lines
    assert len(calls) == 4


def test_a_bare_execute_without_a_version_keeps_its_rows(tmp_path):
    """No score_version: nothing is recorded or re-scored (the old
    behaviour, for callers of `execute` that name no scorer version)."""
    from evalcore import execute
    calls: list[str] = []
    t = v1()
    be = FunctionBackend(lambda r: calls.append(r.unit_id) or "a",
                         identity={"m": "c"})
    out = tmp_path / "b.jsonl"
    execute(t.units(grid({"m": ["x"]})), be, t.render, t.score, out, workers=1)
    execute(t.units(grid({"m": ["x"]})), be, t.render,
            lambda u, r: {"correct": 1.0}, out, workers=1)
    df = to_frame(out)
    assert df["correct"].tolist() == [0.0] * 4 and "score_version" not in df
