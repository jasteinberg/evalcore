"""Generated log QA: the traces are well formed, every gold answer is
re-derived here from the serialised trace by separate code, and graders
separate right, wrong, hallucinated and unreadable."""

from __future__ import annotations

import hashlib
import io
import json
import random
import re
from itertools import pairwise

import pytest

from evalcore import FunctionBackend, grid, summarize
from evalcore.tasks.log_qa import (
    FORMATS,
    generate_trace,
    log_qa_items,
    log_qa_task,
    serialise,
)


@pytest.mark.parametrize("seed", range(20))
def test_traces_are_well_formed(seed):
    spans = generate_trace(random.Random(seed), n_spans=25, max_depth=4)
    by_id = {s.id: s for s in spans}
    assert len(by_id) == len(spans) == 25
    assert spans[0].parent is None and all(s.parent in by_id
                                           for s in spans[1:])
    kids: dict = {}
    for s in spans:
        assert s.end_ms > s.start_ms
        if s.parent:
            p = by_id[s.parent]
            assert p.start_ms <= s.start_ms and s.end_ms <= p.end_ms
            kids.setdefault(s.parent, []).append(s)
    for group in kids.values():                   # siblings run in sequence
        sib = sorted(group, key=lambda s: s.start_ms)
        assert all(a.end_ms < b.start_ms for a, b in pairwise(sib))
        for a, b in pairwise(sib):                # a retry follows a failure
            if b.attempt > 1:
                assert (a.name, a.status, a.attempt) == (b.name, "error",
                                                         b.attempt - 1)


def test_every_format_carries_every_span():
    spans = generate_trace(random.Random(0), n_spans=15)
    for fmt in FORMATS:
        text = serialise(spans, fmt)
        for s in spans:
            n = len(re.findall(rf"\b{s.id}\b", text))
            # log: START and END lines, plus mentions as a parent
            assert n >= (2 if fmt == "log" else 1), (fmt, s.id)


def recompute(trace: list[dict], qtype: str, question: str):  # noqa: PLR0911 - one return per question type
    """The gold answer, derived from the JSON serialisation alone."""
    body = trace[1:]
    span_ref = re.search(r"span (s\d{3})", question)
    op_ref = re.search(r"(?:operation|did|spent in) ([a-z_]+_[a-z_]+)", question)
    if qtype == "count":
        return sum(s["name"] == op_ref.group(1) for s in body)
    if qtype == "slowest":
        return max(body, key=lambda s: s["duration_ms"])["name"]
    if qtype == "first_fail":
        return min((s for s in body if s["status"] == "error"),
                   key=lambda s: (s["end_ms"], s["id"]))["name"]
    if qtype == "any_fail":
        return "yes" if any(s["status"] == "error" for s in body) else "no"
    if qtype == "parent":
        return next(s["parent"] for s in trace if s["id"] == span_ref.group(1))
    if qtype == "retries":
        return max(s["attempt"] for s in body if s["name"] == op_ref.group(1)) - 1
    if qtype == "total_ms":
        return sum(s["duration_ms"] for s in body if s["name"] == op_ref.group(1))
    if qtype == "failed_ids":
        return sorted(s["id"] for s in body if s["status"] == "error")
    if qtype == "attr":
        attr = re.search(r"value of (\w+)", question).group(1)
        return next(s for s in trace if s["id"] == span_ref.group(1))["attrs"][attr]
    if qtype == "absent":
        attr = re.search(r"value of (\w+)", question).group(1)
        assert attr not in next(s for s in trace
                                if s["id"] == span_ref.group(1))["attrs"]
        return None
    raise AssertionError(qtype)


def test_every_gold_answer_is_rederived_from_the_trace():
    items = log_qa_items(n_traces=60, fmt="json", n_spans=18, seed=7)
    seen = set()
    for it in items:
        user = it.payload["messages"][1]["content"]
        trace = json.loads(user.split("Trace:\n", 1)[1].split("\n\nQuestion:")[0])
        question = user.rsplit("Question: ", 1)[1]
        qtype = it.payload["meta"]["type"]
        assert recompute(trace, qtype, question) == it.payload["gold"], it.item_id
        seen.add(qtype)
    assert len(seen) == 10                       # every type was exercised


def test_position_control_moves_the_target_span():
    early = log_qa_items(n_traces=40, types=["attr", "parent"], position=0.0,
                         n_spans=30, seed=1)
    late = log_qa_items(n_traces=40, types=["attr", "parent"], position=1.0,
                        n_spans=30, seed=1)
    def mean(its):
        return sum(i.payload["meta"]["position"] for i in its) / len(its)

    assert mean(early) < 0.25
    assert mean(late) > 0.75


def _answer(item, mode):
    g, t = item.payload["gold"], item.payload["meta"]["type"]
    if mode == "garble":
        return "I would need to think about that."
    if t == "absent":
        return "The trace does not contain that." if mode == "oracle" else "42"
    if mode == "wrong" and t in ("count", "total_ms", "retries", "attr"):
        return str(g + 1)
    if isinstance(g, list):
        return "Failed: " + ", ".join(g)
    return f"The answer is {g}."


@pytest.mark.filterwarnings("ignore:.*carry caveats:RuntimeWarning")  # tiny fixture
@pytest.mark.parametrize("mode", ["oracle", "wrong", "garble"])
def test_graded_end_to_end(tmp_path, mode):
    task = log_qa_task(n_traces=15, fmt="tree", seed=2)
    answers = {it.item_id: _answer(it, mode) for it in task.items}
    by_prompt = {it.payload["messages"][1]["content"]: it.item_id
                 for it in task.items}
    be = FunctionBackend(lambda r: answers[by_prompt[r.messages[1]["content"]]],
                         identity={"model": f"scripted-{mode}"})
    df = task.run(grid({"model": ["toy"]}), be, tmp_path / "r.jsonl",
                  workers=1, stream=io.StringIO())
    t = df.set_index("item_id")
    if mode == "oracle":
        assert (t["status"] == "ok").all() and (t["correct"] == 1.0).all()
        s = summarize(df, "correct", by=["task"], n_boot=200).iloc[0]
        assert s["n_clusters"] == 15                # traces, not questions
    elif mode == "wrong":
        absent = t[t["group"] == "absent"]
        assert (absent["hallucinated"] == 1.0).all()
        numeric = t[t["group"].isin(["count", "total_ms", "attr"])]
        assert (numeric["correct"] == 0.0).all()
    else:
        unparsed = t[t["status"] == "unparsed"]
        # every closed-form type is unreadable; abstain cannot be "unreadable"
        assert set(unparsed["group"]) >= {"count", "slowest", "parent",
                                          "any_fail", "failed_ids"}
        assert unparsed["correct"].isna().all()


def test_d11_regeneration_and_prefix_stability():
    """D11.  Invariance: the same seed reproduces the items exactly, and
    asking for more traces extends the list without changing its prefix
    (so a larger run contains the smaller one)."""
    def make(n):
        return log_qa_items(n_traces=n, fmt="log", n_spans=8, seed=5)
    small = make(6)
    assert make(6) == small
    assert make(12)[:len(small)] == small


_START = re.compile(r"t=(\d+)ms START (\S+) id=(\S+)")
_END = re.compile(r"t=(\d+)ms END \S+ id=(\S+) status=(\S+)")


def _spans(item):
    """{span id: (name, start_ms, end_ms, status)}, read from the context
    text by a parser written here: json.loads for json, regexes for log."""
    text = item.payload["messages"][1]["content"].split("Trace:\n", 1)[1] \
        .split("\n\nQuestion:", 1)[0]
    if item.payload["meta"]["fmt"] == "json":
        return {s["id"]: (s["name"], s["start_ms"], s["end_ms"], s["status"])
                for s in json.loads(text)}
    start = {i: (n, int(t)) for t, n, i in _START.findall(text)}
    end = {i: (int(t), st) for t, i, st in _END.findall(text)}
    assert set(start) == set(end) and start
    return {i: (start[i][0], start[i][1], *end[i]) for i in start}


@pytest.mark.parametrize("position", [None, 0.5])
def test_d9_fmt_axis_is_paired(position):
    """D9.  Invariance: fmt only re-serialises a trace, so json and log
    under one seed hold the same 20 traces (every span's name, times and
    status, parsed from each text independently); with position unset the
    item ids and golds are equal too."""
    a = log_qa_items(n_traces=20, fmt="json", n_spans=10, seed=3,
                     position=position)
    b = log_qa_items(n_traces=20, fmt="log", n_spans=10, seed=3,
                     position=position)
    ta = {i.payload["meta"]["trace"]: _spans(i) for i in a}
    tb = {i.payload["meta"]["trace"]: _spans(i) for i in b}
    assert len(ta) == 20 and ta == tb
    if position is None:
        assert [i.item_id for i in a] == [i.item_id for i in b]
        assert [i.payload["gold"] for i in a] == [i.payload["gold"] for i in b]


@pytest.mark.parametrize("fmt,sha", [
    # Recorded 31 Oct from the generator as it was before the D9 fix, which
    # must leave the position=None stream untouched.
    ("json", "a7f4283301ef669e"),
    ("log", "9ac5437f95022a2e"),
    ("tree", "4e195c4f43745fd0"),
])
def test_unpositioned_items_are_pinned(fmt, sha):
    its = log_qa_items(n_traces=20, fmt=fmt, n_spans=10, seed=3)
    rows = [[i.item_id, i.payload, i.group, i.parent_id] for i in its]
    got = hashlib.sha256(json.dumps(rows, sort_keys=True).encode())
    assert len(its) == 174 and got.hexdigest()[:16] == sha
