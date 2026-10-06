"""Graders: right, wrong, and unreadable are three different outcomes."""

from __future__ import annotations

import io

import pytest

from evalcore import FunctionBackend, Item, ParseFailure, Request, execute, grid, units
from evalcore.tasks.graders import grade, make_score


@pytest.mark.parametrize("spec,text,gold,correct", [
    ("exact", "  Paris. ", "paris", 1.0),
    ("exact", "Lyon", ["Paris", "City of Light"], 0.0),
    ({"name": "exact", "extract": r"Answer:\s*(\w+)"}, "So... Answer: Paris",
     "paris", 1.0),
    ("contains", "I think it is Paris, France", "paris", 1.0),
    ("numeric", "about 1,234 units", 1234, 1.0),
    ({"name": "numeric", "rel": 0.01}, "3.14", 3.1416, 1.0),
    ({"name": "numeric", "pick": "first"}, "7 then 8", 8, 0.0),
    ("numeric", "it is -0.5", -0.5, 1.0),
    ("choice", "Let me think. Answer: (C)", "C", 1.0),
    ("choice", "A good question; the answer is B", "B", 1.0),
    ("choice", "(D)", "C", 0.0),
    ("yes_no", "Yes, it failed twice.", "yes", 1.0),
    ({"name": "set_f1", "pattern": r"s\d{3}"}, "see s001 and s004", ["s001",
                                                                    "s004"], 1.0),
    ("abstain", "That is not mentioned in the log.", None, 1.0),
    ("abstain", "It was 42.", None, 0.0),
])
def test_right_and_wrong(spec, text, gold, correct):
    assert grade(spec, text, gold)["correct"] == correct


@pytest.mark.parametrize("spec,text,gold", [
    ("exact", "   ", "x"),
    ({"name": "exact", "extract": r"Answer:\s*(\w+)"}, "Paris", "paris"),
    ("numeric", "no digits here", 3),
    ("choice", "Either A or B could work", "A"),        # two letters: ambiguous
    ("choice", "I am not sure", "A"),
    ("yes_no", "yes and no", "yes"),
    ({"name": "set_f1", "pattern": r"s\d{3}"}, "no ids", ["s001"]),
])
def test_unreadable_is_a_parse_failure_not_a_wrong_answer(spec, text, gold):
    with pytest.raises(ParseFailure):
        grade(spec, text, gold)


def test_set_f1_partial_credit_by_hand():
    """pred {s1, s2, s9}, gold {s1, s2, s3, s4}: p = 2/3, r = 1/2,
    F1 = 2pr/(p+r) = 4/7."""
    out = grade({"name": "set_f1", "pattern": r"s\d"}, "s1 s2 s9",
                ["s1", "s2", "s3", "s4"])
    assert out["precision"] == pytest.approx(2 / 3)
    assert out["recall"] == pytest.approx(1 / 2)
    assert out["f1"] == pytest.approx(4 / 7) and out["correct"] == 0.0


def test_unknown_grader_is_named():
    with pytest.raises(KeyError, match="unknown grader 'fuzzy'"):
        grade("fuzzy", "x", "x")


def test_through_the_runner_unreadable_rows_are_unparsed(tmp_path):
    """The property end to end: an unreadable response is an `unparsed` row
    with no `correct`, never a 0 averaged into accuracy."""
    its = [Item("q0", {"q": "2+2", "gold": 4, "grader": "numeric"}),
           Item("q1", {"q": "3+3", "gold": 6, "grader": "numeric"}),
           Item("q2", {"q": "pick", "gold": "B", "grader": "choice"})]
    replies = {"2+2": "4", "3+3": "six", "pick": "A or B"}
    be = FunctionBackend(lambda r: replies[r.messages[-1]["content"]],
                         identity={"model": "scripted"})
    df = execute(units(grid({"m": ["x"]}), its), be,
             lambda u: Request(u.id, [{"role": "user",
                                       "content": u.item.payload["q"]}], {}),
             make_score(), tmp_path / "r.jsonl", workers=1,
             stream=io.StringIO())
    got = df.set_index("item_id")
    assert got["status"].to_dict() == {"q0": "ok", "q1": "unparsed",
                                       "q2": "unparsed"}
    assert got["correct"].count() == 1 and got.loc["q0", "correct"] == 1.0


LABELS = {"name": "label", "options": ["fetch_page", "parse_rows", "job"]}


def test_label_reads_one_named_option_inside_free_text():
    assert grade(LABELS, "The slowest was fetch_page, by far.", "fetch_page") \
        == {"correct": 1.0, "pred": "fetch_page"}
    assert grade(LABELS, "parse_rows", "fetch_page")["correct"] == 0.0


@pytest.mark.parametrize("text", ["no idea", "fetch_page or parse_rows"])
def test_label_with_none_or_several_options_is_unparsed(text):
    with pytest.raises(ParseFailure):
        grade(LABELS, text, "fetch_page")
