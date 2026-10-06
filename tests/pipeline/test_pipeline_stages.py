"""The Experiment pipeline: each stage does its job, can be replaced or
switched off, and a halted run is still reported."""

from __future__ import annotations

import json
from collections import Counter

import pytest

from evalcore import FunctionBackend, Item, Response, grid
from evalcore.pipeline import Experiment, RunOptions
from evalcore.tasks.base import Task

pytestmark = pytest.mark.filterwarnings(
    # plumbing fixtures are deliberately tiny; caveats are tested in test_stats
    "ignore:.*carry caveats:RuntimeWarning")



class Counting:
    """A scripted model: correct on even items; counts every call."""

    def __init__(self, tag="v1"):
        self.calls, self.tag = Counter(), tag

    def __call__(self, req):
        q = req.messages[-1]["content"]
        self.calls[q] += 1
        i = int(q.split()[-1])
        return Response(req.unit_id, text=str(i if i % 2 == 0 else -1),
                        meta={"truncated": False, "in_tokens": 100,
                              "out_tokens": 5})


def make(tmp_path, n=6, fn=None, **kw):
    fn = fn or Counting()
    items = [Item(f"q{i}", {"messages": [{"role": "user",
                                          "content": f"what is {i}"}],
                            "gold": i, "grader": "numeric"}) for i in range(n)]
    exp = Experiment(Task("toy", items), FunctionBackend(
        fn, identity={"model": "scripted", "tag": fn.tag}),
        grid({"model": ["toy"]}), tmp_path / "r.jsonl", metric="correct", **kw)
    return exp, fn


def test_a_full_run_produces_every_stage_output(tmp_path):
    exp, _ = make(tmp_path)
    res = exp.run()
    assert res.calibration is not None and res.calibration.n_calls == 5
    assert res.settings["workers"] == 2          # no rate-limit headers
    assert res.dry_run["to_call"] == 6 and res.dry_run["est_in_tokens"] == 600
    assert len(res.frame) == 6 and (res.frame["status"] == "ok").all()
    s = res.analysis["summary"].iloc[0]
    assert s["point"] == pytest.approx(0.5) and s["n_rows"] == 6
    assert res.analysis["attrition"]["n_completed"].sum() == 6
    report = json.loads(res.report_path.read_text())
    assert report["experiment"]["task"] == "toy" and report["halted"] is None
    assert report["events"]["status_counts"] == {"ok": 6}
    assert report["provenance"]["python"]
    assert (tmp_path / "r.jsonl.analysis" / "summary.csv").exists()


def test_stages_switch_off_and_can_be_replaced(tmp_path):
    seen = []
    exp, fn = make(tmp_path, calibrate=None,
                   analyse=lambda e, r: seen.append(len(r.frame)))
    res = exp.run()
    assert res.calibration is None and fn.calls.total() == 6   # no probe calls
    assert seen == [6] and res.analysis == {}


def test_execute_none_reanalyses_an_existing_sink(tmp_path):
    make(tmp_path)[0].run()
    exp, fn = make(tmp_path, calibrate=None, dry_run=None, execute=None)
    res = exp.run()
    assert fn.calls.total() == 0
    assert res.analysis["summary"].iloc[0]["point"] == pytest.approx(0.5)


def test_over_budget_halts_before_any_unit_and_still_reports(tmp_path):
    exp, fn = make(tmp_path, options=RunOptions(
        prices={"in": 1000.0, "out": 1000.0}, max_cost=0.01))
    res = exp.run()
    assert res.halted.startswith("estimated cost $0.63 exceeds max_cost")
    assert fn.calls.total() == 5                 # calibration calls only
    assert not (tmp_path / "r.jsonl").exists() and res.frame is None
    assert json.loads(res.report_path.read_text())["halted"] == res.halted
    exp2, _ = make(tmp_path, options=RunOptions(
        prices={"in": 1000.0, "out": 1000.0}, max_cost=0.01, confirm=True))
    assert len(exp2.run().frame) == 6


def test_dry_run_only_reports_what_would_run(tmp_path):
    exp, _ = make(tmp_path, options=RunOptions(limit=4))
    exp.run()                                    # 4 of 6 done
    exp2, fn2 = make(tmp_path, options=RunOptions(dry_run_only=True))
    res = exp2.run()
    assert res.halted == "dry run only" and fn2.calls.total() == 0  # saved cal
    assert (res.dry_run["done"], res.dry_run["to_go"]) == (4, 2)


def test_a_saved_calibration_is_reused_until_the_backend_changes(tmp_path):
    only_calibrate = {"dry_run": None, "execute": None, "analyse": None,
                      "report": None}
    first, fn = make(tmp_path, **only_calibrate)
    first.run()
    assert fn.calls.total() == 5                 # measured and saved
    again, fn = make(tmp_path, **only_calibrate)
    again.run()
    assert fn.calls.total() == 0                 # same identity, fresh: reused
    changed, fn = make(tmp_path, fn=Counting(tag="v2"), **only_calibrate)
    changed.run()
    assert fn.calls.total() == 5                 # another backend: measured
    forced, fn = make(tmp_path, fn=Counting(tag="v2"), **only_calibrate,
                      options=RunOptions(recalibrate=True))
    forced.run()
    assert fn.calls.total() == 5


def test_explicit_options_win_over_derived_settings(tmp_path):
    exp, _ = make(tmp_path, options=RunOptions(workers=1, batch_size=3))
    res = exp.run()
    assert (res.settings["workers"], res.settings["batch_size"]) == (1, 3)
    assert res.settings["derived"]["workers"] == 2


def test_an_experiment_refuses_what_cannot_run(tmp_path):
    with pytest.raises(ValueError, match="at least one cell"):
        Experiment(Task("t", []), FunctionBackend(lambda r: "x",
                   identity={"m": 1}), grid({"m": [1]}), tmp_path / "r.jsonl")
