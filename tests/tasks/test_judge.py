"""Judging as a task: answers of a finished run become judge items, each
judge's verdict is a forced tool call read strictly, and the long frame
feeds the judge statistics.  Scripted judges with known behaviour give the
expected error rates by construction."""

from __future__ import annotations

import io
import re

import pytest

from evalcore import Cell, FunctionBackend, Item, Request, Response, Task
from evalcore.evaluation.judges import judge_error_rates, majority
from evalcore.tasks.judge import (
    TOOL,
    Criterion,
    judge_items,
    judge_task,
    run_panel,
    verdict_tool,
    verdicts_long,
)

CORRECT = Criterion("correct", "Does the response give the reference answer?")
GOLD = {"i0": 3, "i1": 5, "i2": 7, "i3": 9}
ANSWER = {"i0": "3", "i1": "4", "i2": "I could not find it.", "i3": "9"}


def source_run(tmp_path):
    items = [Item(i, {"messages": [{"role": "user", "content": f"q {i}?"}],
                      "gold": g, "grader": {"name": "numeric", "tol": 0}})
             for i, g in GOLD.items()]
    task = Task("src", items)

    def model(req: Request) -> str:
        return ANSWER[req.messages[-1]["content"][2:4]]
    be = FunctionBackend(model, {"model": "scripted"})
    df = task.run([Cell({"model": "m"})], be, tmp_path / "src.jsonl",
                  workers=1, stream=io.StringIO())
    return task, df


def call(args, name="verdict") -> Response:
    return Response("x", text="", meta={"tool_calls": [
        {"id": "c1", "name": name, "arguments": args}]})


def judge_backend(decide, name, forced=True):
    """decide(answer, reference) -> bool, returned as a verdict call."""
    def fn(req: Request) -> Response:
        assert req.params.get("tool_choice") == (TOOL if forced else None)
        assert [t["name"] for t in req.tools] == ["verdict"]
        body = req.messages[-1]["content"]
        ref = re.search(r"Reference answer:\n(.*)\n\nResponse:", body).group(1)
        ans = body.split("Response:\n", 1)[1]
        return call({"correct": decide(ans, ref), "reason": "r"})
    return FunctionBackend(fn, {"judge": name}, supports_tools=True)


HONEST = lambda a, r: a.strip() == r                       # noqa: E731
LENIENT = lambda a, r: True                                # noqa: E731
CONTRARY = lambda a, r: a.strip() != r                     # noqa: E731


# --- items and requests -----------------------------------------------------

def test_verdict_tool_schema():
    t = verdict_tool([CORRECT])
    p = t["parameters"]
    assert t["name"] == "verdict" and p["required"] == ["correct", "reason"]
    assert p["properties"]["correct"]["type"] == "boolean"


@pytest.mark.parametrize("bad", ["Correct", "reason", "1x", ""])
def test_criterion_names_are_checked(bad):
    with pytest.raises(ValueError):
        Criterion(bad, "q")


def test_judge_items_keep_the_source_unit_and_cluster(tmp_path):
    task, df = source_run(tmp_path)
    assert sorted(df["status"]) == ["ok", "ok", "ok", "unparsed"]
    items = judge_items(df, task)
    src = df.set_index("unit_id")
    assert sorted(it.item_id for it in items) == sorted(df["unit_id"])
    for it in items:
        assert it.cluster == src.at[it.item_id, "cluster"]
        assert it.payload["reference"] == str(GOLD[src.at[it.item_id, "item_id"]])
        assert it.payload["answer"] == src.at[it.item_id, "text"]
    only_ok = judge_items(df, task, statuses=("ok",))
    assert len(only_ok) == 3


def test_judge_items_refuse_a_row_without_text(tmp_path):
    task, df = source_run(tmp_path)
    df.loc[0, "text"] = None
    with pytest.raises(ValueError, match="no text to judge"):
        judge_items(df, task)


def test_judge_task_mixing_reference_and_none_is_refused(tmp_path):
    task, df = source_run(tmp_path)
    a = judge_items(df, task)[:2]
    b = judge_items(df, task, reference=False)[2:]
    with pytest.raises(ValueError, match="reference"):
        judge_task(a + b, [CORRECT])


# --- strict reading of the verdict ------------------------------------------

@pytest.mark.parametrize("reply", [
    Response("x", text="Correct: yes"),                        # prose, no call
    call({"correct": True, "reason": "r"}, name="search"),     # wrong tool
    call({"correct": True}),                                   # missing key
    call({"correct": True, "reason": "r", "extra": 1}),        # extra key
    call({"correct": "true", "reason": "r"}),                  # not a bool
    call("not an object"),
])
def test_a_malformed_verdict_is_unparsed_never_a_fail(tmp_path, reply):
    task, df = source_run(tmp_path)
    jt = judge_task(judge_items(df, task)[:1], [CORRECT])
    be = FunctionBackend(lambda req: reply, {"j": "bad"}, supports_tools=True)
    out = jt.run([Cell({"model": "j"})], be, tmp_path / "j.jsonl", workers=1,
                 stream=io.StringIO())
    assert list(out["status"]) == ["unparsed"]
    assert "v_correct" not in out or out["v_correct"].isna().all()


# --- end to end -------------------------------------------------------------

def test_panel_end_to_end_recovers_the_judges_error_rates(tmp_path):
    """Source answers: i0 right, i1 wrong, i2 an abstention (unparsed, no
    exact grade), i3 right.  Over the three exactly graded answers (two
    positives, one negative): HONEST has alpha = beta = 0; LENIENT passes
    everything, alpha = 1, beta = 0; CONTRARY inverts, alpha = beta = 1.
    The majority passes every answer: LENIENT votes 1 and CONTRARY votes
    1 - h where HONEST votes h, so the pass votes are 1 + h + (1 - h) = 2
    of 3 whatever the answer.  Two biased judges outvote a perfect one."""
    task, df = source_run(tmp_path)
    jt = judge_task(judge_items(df, task), [CORRECT])
    judges = [("honest", judge_backend(HONEST, "h"),
               {"model": "h", "tool_choice": TOOL}),
              ("lenient", judge_backend(LENIENT, "l", forced=False),
               {"model": "l"}),
              ("contrary", judge_backend(CONTRARY, "c"),
               {"model": "c", "tool_choice": TOOL})]
    judged = run_panel(jt, judges, tmp_path / "judges", workers=1,
                       stream=io.StringIO())
    assert len(judged) == 12 and (judged["status"] == "ok").all()
    for n in ("honest", "lenient", "contrary"):
        assert (tmp_path / "judges" / f"{n}.jsonl").is_file()
    long = verdicts_long(judged, [CORRECT], source=df,
                         reference={"correct": "correct"})
    assert len(long) == 12 and long["reference"].isna().sum() == 3
    cal = judge_error_rates(long.dropna(subset=["reference"])).set_index("judge")
    assert (cal.loc["honest", "alpha"], cal.loc["honest", "beta"]) == (0, 0)
    assert (cal.loc["lenient", "alpha"], cal.loc["lenient", "beta"]) == (1, 0)
    assert (cal.loc["contrary", "alpha"], cal.loc["contrary", "beta"]) == (1, 1)
    assert (cal["n_pos"] == 2).all() and (cal["n_neg"] == 1).all()

    panel = majority(long, ["honest", "lenient", "contrary"]).set_index("answer")
    assert (panel["panel"] == 1.0).all() and len(panel) == 4
    assert (panel["status"] == "ok").all()


def test_run_panel_refuses_duplicate_judge_names(tmp_path):
    task, df = source_run(tmp_path)
    jt = judge_task(judge_items(df, task), [CORRECT])
    be = judge_backend(HONEST, "h")
    with pytest.raises(ValueError, match="distinct"):
        run_panel(jt, [("a", be, {"model": "h"}), ("a", be, {"model": "h"})],
                  tmp_path / "j")
    with pytest.raises(ValueError, match="own backend instance"):
        run_panel(jt, [("a", be, {"model": "h"}), ("b", be, {"model": "i"})],
                  tmp_path / "j")


def test_forcing_is_the_judges_param_and_is_recorded_in_its_rows(tmp_path):
    task, df = source_run(tmp_path)
    jt = judge_task(judge_items(df, task)[:1], [CORRECT])
    seen = []

    def fn(req: Request) -> Response:
        seen.append(dict(req.params))
        return call({"correct": True, "reason": "r"})
    def be():
        return FunctionBackend(fn, {"j": "x"}, supports_tools=True)
    out = run_panel(jt, [("forced", be(), {"model": "a", "tool_choice": TOOL}),
                         ("free", be(), {"model": "b"})], tmp_path / "j",
                    workers=1, stream=io.StringIO())
    assert seen == [{"model": "a", "tool_choice": "verdict"}, {"model": "b"}]
    rows = out.set_index("judge")
    assert rows.loc["forced", "tool_choice"] == "verdict"
    assert rows.loc["free", "tool_choice"] != "verdict"


def test_stacked_runs_with_coinciding_unit_ids_need_an_id_column(tmp_path):
    """Two conditions differing only in tags share unit ids, so stacking
    them is refused until an id column tells their answers apart."""
    import pandas as pd
    task, df = source_run(tmp_path)
    both = pd.concat([df.assign(condition="a"), df.assign(condition="b")],
                     ignore_index=True)
    with pytest.raises(ValueError, match="duplicate unit_id"):
        judge_items(both, task)
    both["answer_id"] = both["condition"] + ":" + both["unit_id"]
    items = judge_items(both, task, id_col="answer_id")
    assert len(items) == 8 and len({it.item_id for it in items}) == 8
    jt = judge_task(items, [CORRECT])
    judged = run_panel(jt, [("h", judge_backend(HONEST, "h"),
                             {"model": "h", "tool_choice": TOOL})],
                       tmp_path / "j", workers=1, stream=io.StringIO())
    long = verdicts_long(judged, [CORRECT], source=both,
                         reference={"correct": "correct"}, id_col="answer_id")
    assert long["answer"].str[:2].value_counts().to_dict() == {"a:": 4, "b:": 4}
    assert long["reference"].notna().sum() == 6
