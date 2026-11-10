"""The judge calibration set: labels follow from the committed value alone,
negatives never commit to gold, the hard transformations are right on hand
cases, and the set is a deterministic function of (world, seed)."""

from __future__ import annotations

import random
import re

import pytest

from evalcore.tasks.judge_calibration import (
    NUMERIC_KINDS,
    RIVER_KINDS,
    _transposed,
    answer_variants,
    calibration_set,
)
from evalcore.tasks.search_qa import make_world, search_qa_items

POSITIVE = {"bare", "sentence", "buried", "self_correct_right"}


def last_value(text, rivers=None):
    """The value mentioned last: the answer a reader takes as final."""
    if rivers is None:
        return int(re.findall(r"\d[\d,]*", text)[-1].replace(",", ""))
    return max(rivers, key=lambda r: text.rfind(r) if r in text else -1)


def test_transposed_hand_case():
    # 43220: swappable adjacent pairs at 0 (4,3), 1 (3,2), 3 (2,0)
    seen = {_transposed(43220, random.Random(s)) for s in range(50)}
    assert seen == {34220, 42320, 43202}


def test_transposed_refuses_when_only_a_leading_zero_would_do():
    with pytest.raises(ValueError, match="leading zero"):
        _transposed(800, random.Random(0))


def test_near_name_is_the_closest_other_river():
    vs = answer_variants("Murath", ["near_name"], random.Random(0),
                         rivers=["Murath", "Murah", "Sithand", "Zukiath"])
    assert vs[0].final == "Murah" and vs[0].label == 0
    assert vs[0].text == "The Murah flows through it."


@pytest.mark.parametrize("gold,kinds,kw", [
    (43220, NUMERIC_KINDS, {"twin": 84560}),
    ("Murath", RIVER_KINDS, {"rivers": ["Murath", "Murah", "Sithand",
                                        "Zukiath", "Rastiarn"]}),
])
def test_labels_follow_from_the_committed_value(gold, kinds, kw):
    for seed in range(20):
        vs = answer_variants(gold, kinds, random.Random(seed), **kw)
        assert [v.kind for v in vs] == list(kinds)
        for v in vs:
            assert v.label == int(v.final == gold)
            assert v.label == (v.kind in POSITIVE)
            if v.final is not None:
                assert last_value(v.text, kw.get("rivers")) == v.final


def test_the_hard_negatives_mention_gold():
    """A judge matching strings would pass these, which is the point."""
    vs = {v.kind: v for v in answer_variants(43220, NUMERIC_KINDS,
                                             random.Random(3), twin=84560)}
    assert "43,220" in vs["self_correct_wrong"].text
    assert "43,220" in vs["hedge"].text
    assert vs["twin"].final == 84560


def test_calibration_set_size_ids_clusters_and_determinism():
    world = make_world(40)
    items = search_qa_items(world)
    judge, frame = calibration_set(world, items, seed=0)
    n_num = sum(it.payload["meta"]["type"] != "person_river" for it in items)
    n_riv = len(items) - n_num
    assert len(judge) == len(frame) == 10 * n_num + 9 * n_riv == 960
    assert frame["answer_id"].is_unique
    assert [it.item_id for it in judge] == list(frame["answer_id"])
    assert all(it.cluster == c for it, c in zip(judge, frame["cluster"],
                                                strict=True))
    assert frame["correct"].sum() == 4 * len(items)
    for it, row in zip(judge, frame.itertuples(), strict=True):
        assert it.payload["reference"] == str(
            next(i for i in items if i.item_id == row.cluster).payload["gold"])
    again, _ = calibration_set(world, list(reversed(items)), seed=0)
    by_id = {it.item_id: it for it in again}
    assert all(by_id[it.item_id] == it for it in judge)
    other, _ = calibration_set(world, items, seed=1)
    assert [it.payload["answer"] for it in other] != \
        [it.payload["answer"] for it in judge]
