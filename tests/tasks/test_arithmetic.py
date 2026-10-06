"""Few-shot arithmetic: exact reproduction of the earlier generator, and
leak-free shots by default."""

from __future__ import annotations

import hashlib
import re

import pytest

from evalcore import Cell
from evalcore.tasks.arithmetic import arithmetic_items, arithmetic_task, leak_rate


def digest(items) -> str:
    return hashlib.sha256("".join(i.payload["messages"][0]["content"]
                                  + i.payload["target"] for i in items)
                          .encode()).hexdigest()[:16]


@pytest.mark.parametrize("digits,n,sha,leaked", [
    # Pinned 2 Oct by comparing, item by item, against the generator in the
    # scaling-experiments repo (utils/arithmetic_tasks.make_dataset, seed 0):
    # all prompts and targets identical.  The leak counts were measured on
    # that generator's output independently, before this module existed.
    (2, 4096, "df525635c4072dd3", 109),
    (3, 4096, "73e958044929043e", 6),
])
def test_compat_mode_reproduces_the_published_datasets(digits, n, sha, leaked):
    items = arithmetic_items(digits=digits, n_items=n, exclude_leaks=False)
    assert digest(items) == sha
    assert sum(i.payload["meta"]["leak"] for i in items) == leaked


@pytest.mark.parametrize("digits", [1, 2, 3])
def test_default_items_never_show_the_answer_in_context(digits):
    items = arithmetic_items(digits=digits, n_items=500)
    assert leak_rate(items) == 0.0
    for it in items:
        prompt, m = it.payload["messages"][0]["content"], it.payload["meta"]
        shots = re.findall(r"(\d+) \+ (\d+) = (\d+)\n", prompt)
        assert len(shots) == m["n_shots"]
        assert all(int(c) != m["answer"] for _, _, c in shots)
        assert all((int(a), int(b)) != (m["a"], m["b"]) for a, b, _ in shots)


def test_items_are_well_formed():
    for op in "+-*":
        for it in arithmetic_items(digits=2, n_items=200, op=op):
            m = it.payload["meta"]
            prompt = it.payload["messages"][0]["content"]
            assert prompt.endswith(f"{m['a']} {op} {m['b']} =")   # no space
            assert it.payload["target"] == f" {m['answer']}"       # leading space
            assert {"+": m["a"] + m["b"], "-": m["a"] - m["b"],
                    "*": m["a"] * m["b"]}[op] == m["answer"]
            assert 10 <= m["a"] <= 99 and 10 <= m["b"] <= 99
    sp = arithmetic_items(digits=3, n_items=5, spaced=True)[0]
    assert sp.payload["target"] == " " + " ".join(str(sp.payload["meta"]["answer"]))


def test_queries_are_unique_until_the_space_runs_out():
    two = arithmetic_items(digits=2, n_items=2000)
    assert len({(i.payload["meta"]["a"], i.payload["meta"]["b"])
                for i in two}) == 2000
    one = arithmetic_items(digits=1, n_items=150)                 # 100 pairs
    assert len(one) == 150


def test_the_task_scores_by_teacher_forcing():
    t = arithmetic_task(digits=2, n_items=10)
    assert t.name == "add_d2" and len(t.items) == 10
    req = t.render(t.units([Cell({"model": "m"})])[0])
    assert req.target.startswith(" ") and req.messages[0]["role"] == "user"
