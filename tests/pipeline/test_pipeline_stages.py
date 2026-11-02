"""The Experiment pipeline: each stage does its job, can be replaced or
switched off, and a halted run is still reported."""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest
from support.scripted import Counting, items, make

import evalcore
from evalcore import FunctionBackend, Item, grid
from evalcore.pipeline import Experiment, RunOptions
from evalcore.tasks.base import Task

pytestmark = pytest.mark.filterwarnings(
    # plumbing fixtures are deliberately tiny; caveats are tested in test_stats
    "ignore:.*carry caveats:RuntimeWarning")


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


@pytest.mark.parametrize("kw,why", [
    ({"options": RunOptions(max_cost=0.01)}, "max_cost needs prices"),
    ({"options": RunOptions(max_cost=0.01, prices={"in": 1e3, "out": 1e3}),
      "calibrate": None}, "calibrate or dry_run stage off"),
    ({"options": RunOptions(max_cost=0.01, prices={"in": 1e3, "out": 1e3}),
      "dry_run": None}, "calibrate or dry_run stage off"),
])
def test_d7_max_cost_cannot_be_bypassed_silently(tmp_path, kw, why):
    """D7.  Contract: a budget that cannot be checked (no prices, or the
    stage that checks it switched off) is refused when the experiment is
    built, before any call, never run unbudgeted."""
    fn = Counting()
    with pytest.raises(ValueError, match=why):
        make(tmp_path, fn=fn, **kw)
    assert fn.calls.total() == 0
    assert not (tmp_path / "r.jsonl").exists()


def test_execute_none_reanalyses_and_needs_no_budget(tmp_path):
    """execute=None re-analyses an existing sink without a model call.  It
    spends nothing, so a max_cost left in shared options (without prices
    or calibration) is not a budget that goes unchecked: no refusal."""
    make(tmp_path)[0].run()
    exp, fn = make(tmp_path, execute=None, calibrate=None,
                   options=RunOptions(max_cost=0.01))
    res = exp.run()
    assert fn.calls.total() == 0
    assert res.analysis["summary"].iloc[0]["point"] == pytest.approx(0.5)


def test_d23_a_saved_calibration_expires_after_its_max_age(tmp_path):
    """D23.  Contract (RunOptions.calibration_max_age_h = 24): a saved
    calibration measured 25 h ago is not reused (5 new calls); one measured
    23 h ago is (0 calls).  The ages are written by hand into the file."""
    only = {"dry_run": None, "execute": None, "analyse": None, "report": None}
    exp, fn = make(tmp_path, **only)        # six units >= calibrate_n
    exp.run()
    assert fn.calls.total() == 5
    path = tmp_path / "r.jsonl.calibration.json"
    for hours, calls in ((23, 0), (25, 5)):
        d = json.loads(path.read_text())
        d["measured_at"] = (datetime.now(timezone.utc)
                            - timedelta(hours=hours)).isoformat()
        path.write_text(json.dumps(d))
        exp, fn = make(tmp_path, **only)
        exp.run()
        assert fn.calls.total() == calls, hours


def test_d25_run_json_records_the_checkout_commit(tmp_path):
    """D25.  Independent subprocess: the commit in run.json equals
    `git rev-parse HEAD` of the checkout evalcore was imported from."""
    pkg = Path(evalcore.__file__).resolve().parent
    head = subprocess.run(["git", "-C", str(pkg), "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=False)
    if head.returncode != 0:
        pytest.skip("evalcore is not imported from a git checkout")
    res = make(tmp_path, calibrate=None)[0].run()
    report = json.loads(res.report_path.read_text())
    assert report["provenance"]["commit"] == head.stdout.strip()


@pytest.mark.parametrize("confirm", [False, True])
def test_d7_a_backend_reporting_no_tokens_halts_a_budgeted_run(tmp_path,
                                                               confirm):
    """D7, the case only the dry run can see: prices and calibration are
    there, but the backend reports no token counts, so the calibration has
    no per-call tokens and the cost cannot be estimated.  The run halts
    before execute unless confirmed; confirmed, all six units run."""
    calls = []

    def silent(req):                     # text only: no in/out token counts
        calls.append(req)
        return "0"

    exp = Experiment(Task("toy", items()),
                     FunctionBackend(silent, identity={"model": "silent"}),
                     grid({"model": ["toy"]}), tmp_path / "r.jsonl",
                     metric="correct", options=RunOptions(
                         max_cost=0.01, prices={"in": 1.0, "out": 1.0},
                         confirm=confirm))
    res = exp.run()
    if confirm:
        assert res.halted is None and len(res.frame) == 6
    else:
        assert res.halted.startswith("max_cost $0.01 cannot be checked")
        assert res.frame is None and not (tmp_path / "r.jsonl").exists()


def test_d26_every_unservable_request_is_refused_before_spend(tmp_path):
    """D26.  FunctionBackend has supports_target False, so an item with a
    target cannot be served.  Two LATER items of six carry one: the
    Experiment refuses at construction, before any call, counting 2 of 6
    and saying why, so the user can find them."""
    its = items()
    for k in (2, 5):
        its[k] = Item(f"q{k}", {**its[k].payload, "target": " 1"})
    fn = Counting()
    with pytest.raises(ValueError, match=r"^2 of 6 requests cannot be served"
                                         r".*cannot teacher-force"):
        make(tmp_path, fn=fn, its=its)
    assert fn.calls.total() == 0
