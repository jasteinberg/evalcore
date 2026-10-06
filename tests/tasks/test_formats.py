"""The canonical format, translation into it, and the Task object."""

from __future__ import annotations

import io
import json

import pytest

from evalcore import FunctionBackend, grid
from evalcore.tasks.base import Task
from evalcore.tasks.formats import TaskFormatError, load_tasks, save_tasks

CANON = [
    {"id": "q1", "prompt": "2+2?", "gold": 4, "grader": "numeric"},
    {"id": "q2", "messages": [{"role": "system", "content": "terse"},
                              {"role": "user", "content": "Capital of France?"}],
     "gold": ["Paris"], "grader": {"name": "exact"}, "group": "geo",
     "cluster": "src-7", "meta": {"source": "hand"}},
    {"id": "q3", "prompt": "12 + 34 =", "target": " 46"},
]


def test_canonical_jsonl_and_json_load_the_same(tmp_path):
    (tmp_path / "t.jsonl").write_text("\n".join(json.dumps(r) for r in CANON))
    (tmp_path / "t.json").write_text(json.dumps({"items": CANON}))
    a, b = load_tasks(tmp_path / "t.jsonl"), load_tasks(tmp_path / "t.json")
    assert a == b and len(a) == 3
    q2 = a[1]
    assert q2.payload["messages"][0]["role"] == "system"
    assert q2.group == "geo" and q2.cluster == "src-7"
    assert a[0].payload["messages"] == [{"role": "user", "content": "2+2?"}]
    assert a[2].payload["target"] == " 46" and a[2].cluster == "q3"


def test_save_then_load_is_the_identity(tmp_path):
    items = load_tasks(CANON)
    assert load_tasks(save_tasks(items, tmp_path / "out.jsonl")) == items


FOREIGN = [{"question_id": 7, "input": {"text": "2+3?"},
            "answers": ["5", "five"], "kind": "math"},
           {"question_id": 8, "input": {"text": "3+3?"},
            "answers": ["6"], "kind": "math"}]


def test_a_foreign_format_translates_by_dotted_paths():
    items = load_tasks(FOREIGN, mapping={
        "id": "question_id", "prompt": "input.text", "gold": "answers.0",
        "group": "kind", "grader": lambda r: "numeric"})
    assert [i.item_id for i in items] == ["7", "8"]
    assert items[0].payload["gold"] == "5" and items[0].group == "math"


def test_a_foreign_format_translates_by_function():
    items = load_tasks(FOREIGN, mapping=lambda r: {
        "id": f"f{r['question_id']}", "prompt": r["input"]["text"],
        "gold": r["answers"], "grader": "exact"})
    assert items[1].item_id == "f8" and items[1].payload["gold"] == ["6"]


@pytest.mark.parametrize("records,mapping,msg", [
    ([{"prompt": "x"}], None, "record 0: no id"),
    ([{"id": "a", "prompt": "x"}, {"id": "a", "prompt": "y"}], None,
     "record 1: duplicate id 'a'"),
    ([{"id": "a", "prompt": "x", "messages": [{"role": "user",
                                               "content": "x"}]}], None,
     "exactly one of"),
    ([{"id": "a"}], None, "exactly one of"),
    ([{"id": "a", "prompt": "x", "glod": 3}], None, "unknown field"),
    ([{"id": "a", "prompt": "x", "gold": 3, "grader": "fuzzy"}], None,
     "unknown grader 'fuzzy'"),
    ([{"id": "a", "prompt": "x", "grader": "numeric"}], None, "needs a 'gold'"),
    ([{"id": "a", "messages": [{"content": "x"}]}], None, "role, content"),
    (FOREIGN, {"id": "question_id", "prompt": "input.txt"}, "record 0: no value "
     "at path 'input.txt'"),
    (FOREIGN, {"id": "question_id", "promt": "input.text"}, "unknown field "
     "'promt'"),
])
def test_bad_records_are_refused_by_name(records, mapping, msg):
    with pytest.raises(TaskFormatError, match=msg):
        load_tasks(records, mapping)


def test_a_task_from_a_file_runs_and_tags_every_row(tmp_path):
    (tmp_path / "arith.jsonl").write_text("\n".join(json.dumps(r) for r in [
        {"id": "a", "prompt": "2+2?", "gold": 4, "grader": "numeric"},
        {"id": "b", "prompt": "3+3?", "gold": 6, "grader": "numeric"}]))
    task = Task.from_file(tmp_path / "arith.jsonl", version="2")
    be = FunctionBackend(lambda r: "4", identity={"model": "always-four"})
    df = task.run(grid({"model": ["toy"]}), be, tmp_path / "r.jsonl",
                  workers=1, stream=io.StringIO())
    got = df.set_index("item_id")
    assert got["correct"].to_dict() == {"a": 1.0, "b": 0.0}
    assert set(df["task"]) == {"arith"} and set(df["task_version"]) == {"2"}
    assert set(df["model"]) == {"toy"}             # cell params reach the row
