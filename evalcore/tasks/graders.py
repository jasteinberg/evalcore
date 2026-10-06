"""Deterministic graders: response text and a gold answer in, metrics out.

Every grader returns at least `correct` (0.0 or 1.0), and raises
`ParseFailure` when it cannot read an answer out of the text at all.  The
distinction is the point: "the model said 7 and the answer is 8" is a wrong
answer, "the model said nothing a parser can read" is a fault of the eval or
of the format, and the runner records it as an `unparsed` row that is
counted in attrition instead of being averaged in as a zero.

A grader is named in an item's payload as a string or a dict,
`"numeric"` or `{"name": "numeric", "rel": 0.01}`, so a task file can say how
it is graded without code.  `grade(spec, text, gold)` dispatches;
`make_score()` builds a runner `score` function that reads the spec and the
gold answer from each item.
"""

from __future__ import annotations

import re
import string
from collections.abc import Callable, Mapping, Sequence
from typing import Any, TypeVar

from ..core.backends import Response
from ..core.runner import ParseFailure
from ..core.spec import Unit

__all__ = ["GRADERS", "grade", "grader", "make_score", "normalize",
           "score_teacher_forced"]

GRADERS: dict[str, Callable[..., dict[str, Any]]] = {}


GraderFn = TypeVar("GraderFn", bound=Callable[..., dict[str, Any]])


def grader(name: str) -> Callable[[GraderFn], GraderFn]:
    """Register a grader under `name`."""
    def deco(fn: GraderFn) -> GraderFn:
        GRADERS[name] = fn
        return fn
    return deco


_PUNCT = str.maketrans("", "", string.punctuation)


def normalize(s: str, case: bool = False, punct: bool = False) -> str:
    """Collapse whitespace; lower-case unless `case`; drop ASCII punctuation
    unless `punct`."""
    s = s if case else s.lower()
    s = s if punct else s.translate(_PUNCT)
    return " ".join(s.split())


def _text(text: str | None) -> str:
    if text is None or not text.strip():
        raise ParseFailure("empty response")
    return text


def _aliases(gold: Any) -> list[str]:
    return [str(g) for g in gold] if isinstance(gold, (list, tuple)) else [str(gold)]


@grader("exact")
def exact(text: str, gold: Any, case: bool = False, punct: bool = False,
          extract: str | None = None) -> dict[str, Any]:
    """Normalised string equality against the gold or any of its aliases.
    `extract` is a regex whose first group is the answer; no match is a
    parse failure."""
    t = _text(text)
    if extract is not None:
        m = re.search(extract, t, re.S)
        if m is None:
            raise ParseFailure(f"no match for {extract!r}")
        t = m.group(1)
    pred = normalize(t, case, punct)
    return {"correct": float(any(pred == normalize(g, case, punct)
                                 for g in _aliases(gold))),
            "pred": pred}


@grader("contains")
def contains(text: str, gold: Any, case: bool = False,
             punct: bool = False) -> dict[str, Any]:
    """The normalised gold (or an alias) occurs in the normalised text.  It
    rewards verbosity -- a response listing every candidate contains the
    right one -- so prefer `exact` with `extract` where the format allows."""
    t = normalize(_text(text), case, punct)
    return {"correct": float(any(normalize(g, case, punct) in t
                                 for g in _aliases(gold)))}


_NUM = re.compile(r"-?\d[\d,]*(?:\.\d+)?|-?\.\d+")


@grader("numeric")
def numeric(text: str, gold: Any, tol: float = 0.0, rel: float = 0.0,
            pick: str = "last") -> dict[str, Any]:
    """Compare the first or last number in the text with the gold:
    correct iff |x - g| <= max(tol, rel |g|).  Thousands separators are
    removed ("1,234" is 1234).  No number at all is a parse failure."""
    nums = _NUM.findall(_text(text))
    if not nums:
        raise ParseFailure("no number in the response")
    raw = nums[-1] if pick == "last" else nums[0]
    x, g = float(raw.replace(",", "")), float(gold)
    return {"correct": float(abs(x - g) <= max(tol, rel * abs(g))),
            "pred": x, "abs_err": abs(x - g)}


@grader("choice")
def choice(text: str, gold: str, options: str = "ABCD") -> dict[str, Any]:
    """A multiple-choice letter.  An explicit statement wins ("Answer: C",
    "answer is (C)"); otherwise a standalone option letter, but only if
    exactly ONE distinct option letter stands alone in the text.  Taking the
    first letter instead would read "A good question..." as answer A, and
    several distinct letters are ambiguous, which is a parse failure."""
    t = _text(text)
    opts = re.escape(options)
    m = re.findall(rf"answer\s*(?:is|:)?\s*\(?([{opts}])\)?\b", t, re.I)
    if m:
        pred = m[-1].upper()
    else:
        alone = set(re.findall(rf"(?<![A-Za-z])\(?([{opts}])\)?(?![A-Za-z])",
                               t))
        if len(alone) != 1:
            raise ParseFailure("no unambiguous option letter"
                               if alone else "no option letter")
        pred = alone.pop()
    return {"correct": float(pred == str(gold).upper()), "pred": pred}


@grader("yes_no")
def yes_no(text: str, gold: str) -> dict[str, Any]:
    """A yes/no verdict.  Both words, or neither, is a parse failure."""
    words = set(re.findall(r"\b(yes|no)\b", _text(text).lower()))
    if len(words) != 1:
        raise ParseFailure("no unambiguous yes/no" if words else "no yes/no")
    pred = words.pop()
    return {"correct": float(pred == str(gold).lower()), "pred": pred}


@grader("set_f1")
def set_f1(text: str, gold: Sequence[str], pattern: str) -> dict[str, Any]:
    """Identifiers matching `pattern` (cited span ids, entities) against the
    gold set.  `correct` is exact set equality; precision, recall and F1 are
    reported beside it.  No identifier at all is a parse failure."""
    pred = set(re.findall(pattern, _text(text)))
    if not pred:
        raise ParseFailure(f"no identifier matching {pattern!r}")
    truth = set(gold)
    tp = len(pred & truth)
    p = tp / len(pred)
    r = tp / len(truth) if truth else float(not pred)
    f1 = 0.0 if p + r == 0 else 2 * p * r / (p + r)
    return {"correct": float(pred == truth), "precision": p, "recall": r,
            "f1": f1, "n_pred": len(pred)}


ABSTAIN_CUES = ("not mentioned", "no mention", "not in the", "does not contain",
                "doesn't contain", "no information", "cannot answer",
                "can't answer", "cannot be determined", "not present",
                "does not appear", "no evidence", "unknown")


@grader("abstain")
def abstain(text: str, gold: Any = None,
            cues: Sequence[str] = ABSTAIN_CUES) -> dict[str, Any]:
    """For a question whose answer is NOT in the context: correct iff the
    response declines.  An answer here is a hallucination, a third outcome
    that "incorrect" would hide, so it is reported separately."""
    t = normalize(_text(text), punct=True)
    said = any(c in t for c in cues)
    return {"correct": float(said), "abstained": float(said),
            "hallucinated": float(not said)}


def grade(spec: str | Mapping[str, Any], text: str | None,
          gold: Any) -> dict[str, Any]:
    """Dispatch on a grader spec: a name, or {"name": ..., **options}."""
    if isinstance(spec, str):
        name, opts = spec, {}
    else:
        opts = dict(spec)
        name = opts.pop("name")
    if name not in GRADERS:
        raise KeyError(f"unknown grader {name!r}; known: {sorted(GRADERS)}")
    return GRADERS[name](text, gold, **opts)


def make_score(grader_field: str = "grader", gold_field: str = "gold",
               default: str | Mapping[str, Any] | None = None
               ) -> Callable[[Unit, Response], dict[str, Any]]:
    """A runner `score` function that grades each response by the spec and
    gold in its item's payload (`default` when the item names none).  A
    ParseFailure propagates, so the row is `unparsed`."""
    def score(u: Unit, r: Response) -> dict[str, Any]:
        p = u.item.payload
        spec = p.get(grader_field, default)
        if spec is None:
            raise KeyError(f"item {u.item.item_id!r} names no grader and no "
                           f"default was given")
        out = grade(spec, r.text, p[gold_field])
        return {**out, "grader": spec if isinstance(spec, str)
                else spec["name"]}
    return score


def score_teacher_forced(u: Unit, r: Response) -> dict[str, Any]:
    """Score a teacher-forced response (Request.target) from its per-token
    meta: exact match `em_tf` (every target token the argmax), the mean
    per-token argmax accuracy, and the total target log probability.  The
    per-token lists stay in the row's meta columns for `axis_curves`."""
    m = r.meta
    ok = m["argmax_ok"]
    return {"em_tf": float(m["em_tf"]), "tok_acc": sum(ok) / len(ok),
            "sum_logp": m["sum_logp"], "n_target": m["n_target"]}


@grader("label")
def label(text: str, gold: str, options: Sequence[str]) -> dict[str, Any]:
    """A closed-set answer: the response must name exactly one of `options`
    (whole-word, case-insensitive), and it is compared with the gold.  Free
    text around it is fine ("the slowest was fetch_page").  Naming none is a
    parse failure; naming several is ambiguous, also a parse failure --
    otherwise listing every option would score as right."""
    t = _text(text).lower()
    named = {o for o in options
             if re.search(rf"(?<![\w]){re.escape(o.lower())}(?![\w])", t)}
    if len(named) != 1:
        raise ParseFailure("names several options" if named
                           else "names none of the options")
    pred = named.pop()
    return {"correct": float(pred == gold), "pred": pred}
