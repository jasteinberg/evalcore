"""Job hashing and completion marking: `<sink>.done` says which job the sink
holds and whether it finished, and a rerun of a finished job calls nothing.

Expected values are hand counts on a six-item scripted task (one cell, one
repeat, so six planned units), and call counts from a counting backend."""

from __future__ import annotations

import json

import pytest
from support.scripted import Counting, items, make

from evalcore import Item, grid
from evalcore.pipeline import RunOptions, read_marker

pytestmark = pytest.mark.filterwarnings(
    # plumbing fixtures are deliberately tiny; caveats are tested in test_stats
    "ignore:.*carry caveats:RuntimeWarning")


def marker(tmp_path):
    return read_marker(tmp_path / "r.jsonl.done")


def test_a_finished_job_is_marked_complete_and_a_rerun_calls_nothing(tmp_path):
    exp, fn = make(tmp_path)
    res = exp.run()
    m = marker(tmp_path)
    assert m["status"] == "complete" and m["job_hash"] == res.job_hash
    assert m["counts"] == {"planned": 6, "ok": 6}
    assert fn.calls.total() == 5 + 6             # calibrate_n, then 6 units

    again, fn2 = make(tmp_path)
    res2 = again.run()
    assert fn2.calls.total() == 0                # not even calibration
    assert res2.skipped and res2.job_hash == res.job_hash
    assert res2.analysis["summary"]["n_rows"].tolist() == [6]   # still analysed
    report = json.loads(res2.report_path.read_text())
    assert report["skipped"] == res2.skipped


@pytest.mark.parametrize("change", [
    "item_gold", "prompt_template", "backend_identity", "cells", "repeats",
    "batch_size", "limit"])
def test_anything_that_changes_the_result_changes_the_job(tmp_path, change):
    base, _ = make(tmp_path)
    kw = {
        "item_gold": {"its": items(gold_shift=1)},     # never in the prompt
        "prompt_template": {"its": items(template="compute {i}")},
        "backend_identity": {"model": "other"},
        "cells": {"cells": grid({"model": ["toy", "toy2"]})},
        "repeats": {"repeats": 2},
        "batch_size": {"options": RunOptions(batch_size=2)},
        "limit": {"options": RunOptions(limit=3)},
    }[change]
    other, _ = make(tmp_path, **kw)
    assert other.job_hash() != base.job_hash()


@pytest.mark.parametrize("opts", [
    RunOptions(workers=1), RunOptions(attempts=2), RunOptions(events=False),
    RunOptions(calibrate_n=3)])
def test_how_a_job_is_run_does_not_change_it(tmp_path, opts):
    base, _ = make(tmp_path)
    same, _ = make(tmp_path, options=opts)
    assert same.job_hash() == base.job_hash()


def test_a_changed_job_reruns_only_what_changed(tmp_path):
    make(tmp_path)[0].run()
    edited = items()
    edited[3] = Item("q3", {"messages": [{"role": "user", "content":
                                          "what is, again, 3"}],
                            "gold": 3, "grader": "numeric"})
    exp, fn = make(tmp_path, its=edited, calibrate=None)
    res = exp.run()
    assert not res.skipped
    assert fn.calls == {"what is, again, 3": 1}  # resume does the rest
    assert marker(tmp_path)["status"] == "complete"


def test_errors_mark_the_job_complete_with_errors_and_a_rerun_retries(tmp_path):
    exp, _ = make(tmp_path, fn=Counting(fail={2}), calibrate=None)
    exp.run()
    m = marker(tmp_path)
    assert m["status"] == "complete_with_errors"
    assert m["counts"] == {"planned": 6, "error": 1, "ok": 5}
    again, fn = make(tmp_path, calibrate=None)
    res = again.run()
    assert not res.skipped and fn.calls == {"what is 2": 1}
    assert marker(tmp_path)["status"] == "complete"


def test_a_halted_job_is_marked_halted_and_not_skipped(tmp_path):
    exp, _ = make(tmp_path, calibrate=None,
                  options=RunOptions(dry_run_only=True))
    exp.run()
    m = marker(tmp_path)
    assert m["status"] == "halted" and m["halted"] == "dry run only"
    assert m["counts"] == {"planned": 6, "missing": 6}
    again, fn = make(tmp_path, calibrate=None)
    assert not again.run().skipped and fn.calls.total() == 6


def test_a_crash_leaves_running_never_complete(tmp_path):
    make(tmp_path)[0].run()                       # an older complete marker

    def crash(exp, res):
        raise KeyboardInterrupt

    exp, _ = make(tmp_path, its=items(gold_shift=1), execute=crash)
    with pytest.raises(KeyboardInterrupt):
        exp.run()
    m = marker(tmp_path)
    assert m["status"] == "running" and m["job_hash"] == exp.job_hash()
    assert list(tmp_path.glob("*.tmp")) == []     # the rename left no sibling


def test_resume_off_or_a_lost_sink_is_never_skipped(tmp_path):
    make(tmp_path)[0].run()
    exp, fn = make(tmp_path, calibrate=None, options=RunOptions(resume=False))
    assert not exp.run().skipped and fn.calls.total() == 6

    (tmp_path / "r.jsonl").unlink()               # marker says complete
    exp, fn = make(tmp_path, calibrate=None)
    assert not exp.run().skipped and fn.calls.total() == 6
