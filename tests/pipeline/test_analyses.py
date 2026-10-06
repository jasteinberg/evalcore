"""The analyse stage runs whatever analyses it is given: the built-in
factories, or any function of (frame, ctx)."""

from __future__ import annotations

import math

import pytest

from evalcore import FunctionBackend, Item, grid
from evalcore.analyses import summary, value_curve
from evalcore.pipeline import Experiment
from evalcore.tasks.base import Task

pytestmark = pytest.mark.filterwarnings(
    # four identical items per point by design: the caveats fire, correctly
    "ignore:.*carry caveats:RuntimeWarning")


def curve_experiment(tmp_path, **kw):
    """v(x, arm) = log10(x) / 2 + offset(arm), the same for every item."""
    offset = {"a": 0.0, "b": 0.25}

    def model(req):
        p = req.params
        return f"{math.log10(p['x']) / 2 + offset[p['arm']]}"

    items = [Item(f"i{k}", {"messages": [{"role": "user", "content": "v?"}]})
             for k in range(4)]
    task = Task("curve", items,
                score=lambda u, r: {"v": float(r.text)})
    return Experiment(task, FunctionBackend(model, identity={"m": "log"}),
                      grid({"x": [1, 10, 100], "arm": ["a", "b"]}),
                      tmp_path / "r.jsonl", calibrate=None, dry_run=None, **kw)


def test_curves_are_fitted_per_combination_of_the_other_axes(tmp_path):
    """Crossing of 0.5, linear in log x: arm a at log10 x = 1 (x = 10);
    arm b (values 0.25, 0.75, 1.25) at log10 x = 1/2 (x = sqrt 10)."""
    res = curve_experiment(tmp_path, analyses={
        "curve": value_curve("x", "v", levels=[0.5], n_boot=50)}).run()
    table, peak = res.analysis["curve.table"], res.analysis["curve.peak"]
    assert len(table) == 6 and set(table["arm"]) == {"a", "b"}
    cross = peak[peak["quantity"] == "crossing_0.5"].set_index("arm")["point"]
    assert cross["a"] == pytest.approx(10.0)
    assert cross["b"] == pytest.approx(math.sqrt(10))
    assert (tmp_path / "r.jsonl.analysis" / "curve.peak.csv").exists()


def test_any_function_of_the_frame_is_an_analysis(tmp_path):
    def per_arm_max(frame, ctx):
        assert ctx.by == ["arm", "x"]
        ok = frame[frame["status"] == "ok"]
        return ok.groupby("arm", as_index=False)["v"].max()

    res = curve_experiment(tmp_path, analyses={"max": per_arm_max}).run()
    assert res.analysis["max"].set_index("arm")["v"].to_dict() == {
        "a": 1.0, "b": 1.25}
    assert "attrition" not in res.analysis       # analyses replace the default


def test_metric_is_shorthand_and_cannot_be_mixed_with_analyses(tmp_path):
    res = curve_experiment(tmp_path, metric="v").run()
    assert set(res.analysis) == {"attrition", "summary"}
    with pytest.raises(ValueError, match=r"metric \(shorthand\) or analyses"):
        curve_experiment(tmp_path, metric="v", analyses={"s": summary("v")})


def test_a_summary_of_a_missing_column_names_the_columns(tmp_path):
    exp = curve_experiment(tmp_path, analyses={"s": summary("vv")})
    with pytest.raises(KeyError, match="no column 'vv'"):
        exp.run()


def test_a_curve_along_a_non_axis_is_refused(tmp_path):
    exp = curve_experiment(tmp_path, analyses={"c": value_curve("y", "v")})
    with pytest.raises(ValueError, match="'y' is not a grid axis"):
        exp.run()



def test_rows_lost_before_scoring_are_counted_as_excluded(tmp_path):
    """Live run of 6 Oct: 9 of 49 rows truncated or unparsed, yet the
    summary said n_excluded = 0 and the >10% caveat stayed silent.  Here 2
    of 6 items are truncated: n_rows 4, n_excluded 2, and the caveat fires."""
    from evalcore import Response

    def model(req):
        k = int(req.messages[-1]["content"].split()[-1])
        return Response(req.unit_id, text=str(k),
                        meta={"truncated": k < 2})

    items = [Item(f"q{k}", {"messages": [{"role": "user", "content": f"say {k}"}],
                            "gold": k, "grader": "numeric"}) for k in range(6)]
    exp = Experiment(Task("t", items), FunctionBackend(model, identity={"m": 1}),
                     grid({"model": ["m"]}), tmp_path / "r.jsonl",
                     metric="correct", calibrate=None, dry_run=None)
    s = exp.run().analysis["summary"].iloc[0]
    assert (s["n_rows"], s["n_excluded"]) == (4, 2)
    assert "33% of rows have no value" in s["caveats"]
