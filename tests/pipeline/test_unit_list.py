"""Experiment.from_units: an experiment over a given list of units, not
cells x items x range(repeats).

Expected values are hand-written: the list below has three units with
repeats 0 and 5 on item a and 7 on item b, so three rows, those repeats,
and a backend that answers with the request's repeat must return them."""

from __future__ import annotations

import pytest

from evalcore import Cell, Experiment, FunctionBackend, Task, Unit, load_tasks
from evalcore.core.records import manifest_path, read_manifest
from evalcore.core.shards import shard_of

pytestmark = pytest.mark.filterwarnings(
    # plumbing fixtures are deliberately tiny. Caveats are tested in test_stats
    "ignore:.*carry caveats:RuntimeWarning")

TASK = Task("t", load_tasks([
    {"id": "a", "prompt": "say a", "gold": 0, "grader": "numeric"},
    {"id": "b", "prompt": "say b", "gold": 0, "grader": "numeric"},
    {"id": "c", "prompt": "say c", "gold": 0, "grader": "numeric"}]))
A, B, C = TASK.items
CELL = Cell({"model": "m"})
UNITS = [Unit(CELL, A, 0), Unit(CELL, A, 5), Unit(CELL, B, 7)]


def backend() -> FunctionBackend:
    return FunctionBackend(lambda req: str(req.repeat), identity={"f": "rep"})


def exp(tmp_path, units=UNITS, name="r", **kw) -> Experiment:
    return Experiment.from_units(TASK, backend(), units,
                                 tmp_path / f"{name}.jsonl", calibrate=None,
                                 **kw)


def test_exactly_the_given_units_are_planned_run_and_tagged(tmp_path):
    e = exp(tmp_path)
    assert [c.id for c in e.cells] == [CELL.id]
    res = e.run()
    df = res.frame
    assert sorted(zip(df["item_id"], df["repeat"], strict=True)) == [("a", 0), ("a", 5),
                                                        ("b", 7)]
    assert sorted(df["text"]) == ["0", "5", "7"]      # each unit's own repeat
    assert set(df["unit_id"]) == {u.id for u in UNITS}
    assert set(df["task"]) == {"t"}                   # tagged as Task.units
    assert len(read_manifest(manifest_path(e.out))) == 3


def test_the_shards_partition_the_list(tmp_path):
    many = [Unit(CELL, it, r) for it in (A, B, C) for r in (0, 3, 11, 12)]
    whole = exp(tmp_path, many)
    got: list[str] = []
    for i in range(2):
        s = whole.sharded(i, 2)
        want = [u.id for u in many if shard_of(u.id, 2) == i]
        assert [u.id for u in s.planned()] == want
        got += list(s.run().frame["unit_id"])
    assert sorted(got) == sorted(u.id for u in many)


def test_the_job_hash_covers_the_list_not_its_order(tmp_path):
    h = exp(tmp_path).job_hash()
    assert exp(tmp_path, UNITS[::-1]).job_hash() == h
    assert exp(tmp_path, UNITS[:2]).job_hash() != h
    assert exp(tmp_path, [*UNITS[:2], Unit(CELL, B, 8)]).job_hash() != h


@pytest.mark.parametrize("units,kw,match", [
    ([], {}, "unit_list is empty"),
    ([UNITS[0], UNITS[0]], {}, "appears twice"),
    ([Unit(CELL, load_tasks([{"id": "z", "prompt": "q", "gold": 0,
                               "grader": "numeric"}])[0], 0)], {},
     "not one of task 't'"),
    (UNITS, {"repeats": 2}, "each unit carries its own repeat"),
])
def test_bad_lists_are_refused_when_built(tmp_path, units, kw, match):
    with pytest.raises(ValueError, match=match):
        exp(tmp_path, units, **kw)


def test_cells_that_disagree_with_the_list_are_refused(tmp_path):
    with pytest.raises(ValueError, match="cells must be the units' distinct"):
        Experiment(TASK, backend(), [Cell({"model": "other"})],
                   tmp_path / "r.jsonl", unit_list=UNITS, calibrate=None)
