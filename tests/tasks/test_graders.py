"""Graders: right, wrong, and unreadable are three different outcomes."""

from __future__ import annotations

import io

import pytest

from evalcore import (
    FunctionBackend,
    Item,
    ParseFailure,
    Request,
    Response,
    execute,
    grid,
    units,
)
from evalcore.tasks.graders import (
    GRADERS,
    check_spec,
    grade,
    grader,
    make_score,
    score_teacher_forced,
)


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


@pytest.mark.parametrize("spec,text,gold,match", [
    ({"name": "numeric", "pick": "frist"}, "1 then 2", 2, "pick must be"),
    ({"name": "label", "options": ["fetch", "parse"]}, "fetch", "zeta",
     "not one of the options"),
    ("choice", "Answer: A", "E", "'E' is not one of the options"),
])
def test_d4_grader_options_are_validated(spec, text, gold, match):
    """D4.  Contract: pick is "first" or "last", and a label or choice gold
    must be one of the options (else no response can be right).  Each case
    must raise, not grade."""
    with pytest.raises(ValueError, match=match):
        grade(spec, text, gold)


def test_d5_a_taken_grader_name_needs_replace_true(monkeypatch):
    """D5.  Contract: one implementation per grader name, as the component
    registry already enforces (Registry.add refuses a taken name).  The
    override is possible, but only by saying so."""
    monkeypatch.setitem(GRADERS, "exact", GRADERS["exact"])   # restored after
    with pytest.raises(ValueError):
        grader("exact")(lambda text, gold: {"correct": 1.0})
    grader("exact", replace=True)(lambda text, gold: {"correct": 1.0})
    assert grade("exact", "anything", "x") == {"correct": 1.0}


def test_d14_score_teacher_forced_by_hand():
    """D14.  Hand-computed: argmax_ok [1, 0, 1, 1] gives tok_acc 3/4.  One
    miss means no exact match (em_tf 0).  sum_logp and n_target pass through."""
    meta = {"argmax_ok": [1, 0, 1, 1], "em_tf": False, "sum_logp": -3.5,
            "n_target": 4}
    out = score_teacher_forced(None, Response("u", text=None, meta=meta))
    assert out == {"em_tf": 0.0, "tok_acc": 0.75, "sum_logp": -3.5,
                   "n_target": 4}


@pytest.mark.parametrize("text,pred", [
    ("answer: (c)", "C"),                 # lower case in parentheses
    ("The answer is c.", "C"),            # lower case before punctuation
    ("the answer is c", "C"),             # lower case at the end
    ("ANSWER IS D", "D"),                 # the keyword in any case
    ("The answer is a tricky one; maybe C", "C"),   # article, then a letter
    ("The answer is a tricky one, but B", "B"),
])
def test_d1_the_letter_rule_by_hand(text, pred):
    """D1.  The rule in `choice`'s docstring, case by case."""
    assert grade("choice", text, pred) == {"correct": 1.0, "pred": pred}


@pytest.mark.parametrize("spec,gold,match", [
    ({"name": "numeric", "rell": 0.01}, 1, r"no option\(s\) \['rell'\]"),
    ({"name": "set_f1"}, ["s001"], r"needs option\(s\) \['pattern'\]"),
    ({"name": "numeric", "pick": "frist"}, 1, "pick must be"),
    ({"name": "label", "options": ["a", "b"]}, "c", "not one of the options"),
    ("contains", " ... ", "normalises to an empty string"),
    ("numeric", [1, 2], "grader 'numeric'"),        # float() of a list
])
def test_check_spec_refuses_without_a_response(spec, gold, match):
    with pytest.raises(ValueError, match=match):
        check_spec(spec, gold)


@pytest.mark.parametrize("spec,gold", [
    ("numeric", 3), ({"name": "set_f1", "pattern": r"s\d"}, ["s1"]),
    ("abstain", None), ({"name": "label", "options": ["a", "b"]}, "a"),
    ("choice", "b"), ({"name": "exact", "case": True}, ["Paris", "x"]),
])
def test_check_spec_passes_a_sound_spec(spec, gold):
    check_spec(spec, gold)



# --- three outcomes: correct, wrong, abstained -------------------------------

NUM = {"name": "numeric", "tol": 0, "abstain": True}


@pytest.mark.parametrize("text,want", [
    ("42", (1, 0, 0)),
    ("It is 41.", (0, 1, 0)),
    ("cannot answer", (0, 0, 1)),
    ("The documents do not contain this. Cannot answer.", (0, 0, 1)),
    # phrasings Haiku used in the controls pilot
    ("I'm unable to find any information about Vubarn Thiavaen.", (0, 0, 1)),
    ("I could not find the town.", (0, 0, 1)),
    ("I couldn't find it.", (0, 0, 1)),
    ("I apologize, but I cannot find any information about him.", (0, 0, 1)),
    ("I'm not finding any information about Vubarn Thiavaen.", (0, 0, 1)),
    ("The search results don't contain information about him.", (0, 0, 1)),
    # declines and also offers a value: counted as abstained, by definition
    ("The closest match has 42. If you meant another place, I cannot "
     "answer.", (0, 0, 1)),
])
def test_abstain_option_gives_three_outcomes_that_sum_to_one(text, want):
    out = grade(NUM, text, 42)
    assert (out["correct"], out["wrong"], out["abstained"]) == want


def test_abstain_option_works_for_any_grader():
    spec = {"name": "label", "options": ["Velor", "Askern"], "abstain": True}
    assert grade(spec, "the Velor", "Velor")["correct"] == 1.0
    assert grade(spec, "the Askern", "Velor")["wrong"] == 1.0
    assert grade(spec, "no information", "Velor")["abstained"] == 1.0


def test_abstain_option_still_leaves_unreadable_answers_unparsed():
    with pytest.raises(ParseFailure):
        grade(NUM, "", 42)
    with pytest.raises(ParseFailure):
        grade(NUM, "many people", 42)


def test_abstain_option_is_checked():
    check_spec(NUM, 42)
    with pytest.raises(ValueError, match="true or false"):
        grade({"name": "numeric", "abstain": "yes"}, "42", 42)
    with pytest.raises(ValueError, match="abstain grader itself"):
        grade({"name": "abstain", "abstain": True}, "x", None)


def test_without_the_option_nothing_changes():
    out = grade({"name": "numeric", "tol": 0}, "42", 42)
    assert "wrong" not in out and "abstained" not in out


@pytest.mark.parametrize("text, gold, correct", [
    ("The roots are 2,3", 3, 1.0),        # a list: last number is 3
    ("candidates 7,8", 78, 0.0),          # not seventy-eight
    ("about 1,234 people", 1234, 1.0),    # a thousands separator
    ("12,345,678 total", 12345678, 1.0),
    ("pick 1,2345", 2345, 1.0),           # four digits after: two numbers
])
def test_a_comma_groups_thousands_only_before_three_digits(text, gold, correct):
    assert grade("numeric", text, gold)["correct"] == correct
