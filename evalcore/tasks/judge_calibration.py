"""A calibration set for judges: answers whose correctness is known by
construction, written to be hard to grade.

For every search_qa question, `answer_variants` writes one answer per
variant kind.  Each variant records the value it finally commits to
(`final`, None for a hedge or an abstention), and its label is computed
from that alone,

    label = 1  iff  final == gold,

in the same function that writes the text, so the text and the label cannot
disagree (the code-audit rule for generated data).  The criterion judged is
"does the response commit to the reference answer as its final answer?".

Variant kinds, with the label each gets:

    bare, sentence          the gold value, plainly                      1
    buried                  gold after two distractor values             1
    self_correct_right      a wrong value, then corrected to gold        1
    self_correct_wrong      gold, then "corrected" to a wrong value      0
    hedge                   "either gold or a wrong value"               0
    abstain                 no answer                                    0
    twin                    the near-name twin town's population         0
    transposed              gold with two adjacent digits swapped        0
    near                    gold +- 10..50                               0
    near_name               the river whose name is closest to gold      0
    other                   another river                                0

Numeric questions (pop, river_pop) get every kind but near_name and other;
river questions (person_river) get those two instead of twin, transposed
and near.  Negatives use the same sentence templates as positives, so the
surface form carries no label.  The judge sees the gold value as
`str(gold)` and numeric answers formatted with thousands separators, as a
model would write them.
"""

from __future__ import annotations

import random
from collections.abc import Sequence
from dataclasses import dataclass

import pandas as pd

from ..core.spec import Item, digest
from .search_qa import World

__all__ = ["NUMERIC_KINDS", "RIVER_KINDS", "Variant", "answer_variants",
           "calibration_set"]

NUMERIC_KINDS = ("bare", "sentence", "buried", "self_correct_right",
                 "self_correct_wrong", "hedge", "abstain", "twin",
                 "transposed", "near")
RIVER_KINDS = ("bare", "sentence", "buried", "self_correct_right",
               "self_correct_wrong", "hedge", "abstain", "near_name", "other")


@dataclass(frozen=True)
class Variant:
    kind: str
    text: str
    final: int | str | None
    label: int


def _edit(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _transposed(g: int, rng: random.Random) -> int:
    s = str(g)
    pairs = [i for i in range(len(s) - 1) if s[i] != s[i + 1]
             and not (i == 0 and s[1] == "0")]
    if not pairs:
        raise ValueError(f"{g} has no adjacent digit swap that changes it "
                         f"without a leading zero")
    i = rng.choice(pairs)
    return int(s[:i] + s[i + 1] + s[i] + s[i + 2:])


def _fmt(v: int | str) -> str:
    return f"{v:,}" if isinstance(v, int) else str(v)


def answer_variants(gold: int | str, kinds: Sequence[str],
                    rng: random.Random, twin: int | None = None,
                    rivers: Sequence[str] = ()) -> list[Variant]:
    """One Variant per kind.  `twin` is the near-name town's population
    (numeric kinds), `rivers` every river name (river kinds)."""
    numeric = isinstance(gold, int)
    distract: list[int | str]
    if numeric:
        def wrong() -> int | str:
            return int(gold) + 10 * rng.choice([-5, -4, -3, -2, -1,
                                                1, 2, 3, 4, 5])

        def sent(v: int | str) -> str:
            return f"The population is {_fmt(v)}."
        distract = [int(gold) + 10 * rng.randrange(200, 2000) for _ in range(2)]
        lead = "The search results mention {} for one town and {} for another."
        tail = "For the town asked about, the figure is {}."
    else:
        others = [r for r in rivers if r != gold]
        if len(others) < 3:
            raise ValueError("river variants need at least three other rivers")

        def wrong() -> int | str:
            return rng.choice(others)

        def sent(v: int | str) -> str:
            return f"The {v} flows through it."
        distract = list(rng.sample(others, 2))
        lead = "The {} and the {} also appear in the results."
        tail = "The river through that town is the {}."

    out = []
    for kind in kinds:
        final: int | str | None
        if kind == "bare":
            final, text = gold, _fmt(gold)
        elif kind == "sentence":
            final, text = gold, sent(gold)
        elif kind == "buried":
            final = gold
            text = (lead.format(*(_fmt(d) for d in distract)) + " "
                    + tail.format(_fmt(gold)))
        elif kind in ("self_correct_right", "self_correct_wrong"):
            w = wrong()
            first, last = (w, gold) if kind == "self_correct_right" else (gold, w)
            final = last
            text = (f"{sent(first)} Correction: I misread the document, and "
                    f"the right answer is {_fmt(last)}.")
        elif kind == "hedge":
            a, b = gold, wrong()
            if rng.random() < 0.5:
                a, b = b, a
            final = None
            text = (f"The documents are unclear: it is either {_fmt(a)} or "
                    f"{_fmt(b)}.")
        elif kind == "abstain":
            final, text = None, "I could not find this in the documents."
        elif kind == "twin" and numeric:
            if twin is None or twin == gold:
                raise ValueError("twin variant needs the twin's population")
            final, text = twin, sent(twin)
        elif kind == "transposed" and numeric:
            final = _transposed(int(gold), rng)
            text = sent(final)
        elif kind == "near" and numeric:
            final = wrong()
            text = sent(final)
        elif kind == "near_name" and not numeric:
            final = min(others, key=lambda r: (_edit(r, str(gold)), r))
            text = sent(final)
        elif kind == "other" and not numeric:
            final = wrong()
            text = sent(final)
        else:
            raise ValueError(f"unknown variant kind {kind!r} for "
                             f"{'numeric' if numeric else 'river'} gold")
        out.append(Variant(kind, text, final, int(final == gold)))
    return out


def _twin_name(t: str) -> str:
    return t[:-4] if t.endswith("ford") else t + "ford"


def calibration_set(world: World, items: Sequence[Item], seed: int = 0
                    ) -> tuple[list[Item], pd.DataFrame]:
    """Judge items (question, reference, answer) and the frame of their
    labels: one row per answer with answer_id, cluster (the question),
    kind, final and `correct` (the label).  Each question's answers are
    drawn from an rng seeded by (seed, item id), so the set is the same
    whatever the order of `items`."""
    entity = {d: e for e, d in world.doc_of.items()}
    rivers = sorted(world.rivers)
    out_items, rows = [], []
    for it in items:
        p = it.payload
        q = [m["content"] for m in p["messages"] if m["role"] == "user"][-1]
        rng = random.Random(digest({"seed": seed, "item": it.item_id}))
        if p["meta"]["type"] in ("pop", "river_pop"):
            town = entity[p["gold_docs"][-1]]
            twin = world.towns.get(_twin_name(town), {}).get("population")
            # an odd n_towns leaves the last town without a twin, and a twin
            # with the same population is not a wrong answer: no twin variant
            kinds = (NUMERIC_KINDS if twin is not None and twin != int(p["gold"])
                     else tuple(k for k in NUMERIC_KINDS if k != "twin"))
            vs = answer_variants(int(p["gold"]), kinds, rng, twin=twin)
        else:
            vs = answer_variants(str(p["gold"]), RIVER_KINDS, rng,
                                 rivers=rivers)
        for v in vs:
            aid = f"{it.item_id}:{v.kind}"
            out_items.append(Item(aid, {"question": q, "answer": v.text,
                                        "reference": str(p["gold"])},
                                  group=v.kind, parent_id=it.item_id))
            rows.append({"answer_id": aid, "cluster": it.item_id,
                         "kind": v.kind, "qtype": p["meta"]["type"],
                         "final": None if v.final is None else str(v.final),
                         "correct": float(v.label)})
    return out_items, pd.DataFrame(rows)
