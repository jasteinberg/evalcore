"""Spec: content-addressed ids, grid expansion, clusters.

An id is an identity only if it is injective on what changes a result and
blind to what does not.  Each expected value is an invariance fixed by the
construction (two inputs that change the output must get two ids; two that
cannot must get one), never a digest computed by the code under test.

A newly found violation of a catalogue row lands as a strict xfail whose
reason names the defect: it turns into an XPASS -- a failure under
strict -- the day the defect is fixed, so the marker cannot outlive it.
"""

from __future__ import annotations

import numpy as np
import pytest
from support.runner import DECODE_VALUES, Answers, items, raw_rows, run

from evalcore import Cell, Item, Unit, grid, units
from evalcore.core.backends import DECODE_KEYS


def test_ids_ignore_key_order_and_tags():
    """Tags bear no identity: renaming an arm must not rerun it."""
    a = Cell({"model": "m", "temperature": 0.0}, {"arm": "baseline"})
    b = Cell({"temperature": 0.0, "model": "m"}, {"arm": "renamed"})
    assert a.id == b.id
    i = Item("q0", {"q": "?"})
    assert Unit(a, i, 0).id == Unit(b, i, 0).id


@pytest.mark.parametrize("key", [*sorted(DECODE_KEYS), "model", "template"])
def test_every_param_moves_the_cell_and_unit_id(key):
    a, b = DECODE_VALUES.get(key, ("x", "y"))
    item = Item("q0", {"q": "?"})
    ca, cb = Cell({"model": "m", key: a}), Cell({"model": "m", key: b})
    assert ca.id != cb.id
    assert Unit(ca, item).id != Unit(cb, item).id


def test_unit_id_covers_cell_item_and_repeat():
    c, i = Cell({"model": "m"}), Item("q0", {"q": "?"})
    base = Unit(c, i, 0).id
    assert Unit(Cell({"model": "n"}), i, 0).id != base
    assert Unit(c, Item("q1", {"q": "?"}), 0).id != base
    assert Unit(c, i, 1).id != base


def test_cell_ids_of_different_large_arrays_differ():
    """B1.  zeros(2000) and the unit vector e_500 are different parameter
    values, so their cells must have different ids (injectivity).  Both
    print as "[0. 0. 0. ... 0. 0. 0.]" once numpy summarises (> 1000
    elements), which is what a str-based digest sees."""
    e = np.zeros(2000)
    e[500] = 1.0
    assert Cell({"v": np.zeros(2000)}).id != Cell({"v": e}).id


def test_ids_without_arrays_did_not_move_with_the_array_fix():
    """B1's fix may change only the ids of array-valued params: every
    existing cache key and sink row of a JSON-only sweep must still match.
    The literals were recorded from the code BEFORE the fix (31 Oct), so
    they are independent of it.  np.float64 is a float subclass and was
    always hashed as its value; np.int64 always went through `str`."""
    p = {"model": "claude-x", "temperature": 0.7, "max_tokens": 256,
         "stop": ["\n", "END"], "nested": {"b": [1, 2.5, None, True], "a": "é"}}
    assert Cell(p).id == "5001073a30fbad2e14c76e8eca5ac6f7"
    assert Unit(Cell(p), Item("q0", {"q": "?"}), 2).id == (
        "bd41ff3404d5c9d236fbcf7f74ae2915")
    assert Cell({"x": np.float64(0.5)}).id == "7f42d42b99c63894a27aebc3cbc43af2"
    assert Cell({"x": np.int64(3)}).id == "4edf899483b007ef64a238eae59572c2"


def test_tags_never_shadow_params():
    """B35.  A tag and a param share the key `model`; by construction the
    param is "m" and the tag "x".  Tags bear no identity, so the frame's
    `model` column must be the param, and the cell id must equal that of the
    same params with no tags."""
    c = Cell(params={"model": "m"}, tags={"model": "x", "arm": "A"})
    got = c.merged()
    assert got["model"] == "m"
    assert got["arm"] == "A"
    assert got["cell_id"] == Cell(params={"model": "m"}).id


# --- grid ------------------------------------------------------------------------

def test_grid_exclusions():
    g = grid({"model": ["a", "b"], "temperature": [0.0, 1.0]},
             exclude=[{"model": "a", "temperature": 1.0}])
    assert len(g) == 3
    assert all(not (c.get("model") == "a" and c.get("temperature") == 1.0)
               for c in g)


def test_an_exclude_pattern_on_an_unknown_key_is_refused():
    """B35.  `modle` is not an axis of the grid, so the pattern can never
    match on its own terms: a typo that would otherwise silently keep the
    cell it meant to drop (4 cells instead of 3).  Expected: refused."""
    with pytest.raises(ValueError, match="modle"):
        grid({"model": ["a", "b"], "temp": [0.0, 1.0]},
             exclude=[{"modle": "a", "temp": 1.0}])


def test_augmentations_cluster_with_their_parent():
    """The resampling unit: an augmentation's cluster is its parent's id,
    an original's its own."""
    assert Item("q0", {}).cluster == "q0"
    assert Item("q0-a1", {}, parent_id="q0").cluster == "q0"


# --- names the runner owns ---------------------------------------------------------

def test_a_duplicate_item_id_with_different_payloads_is_refused(tmp_path):
    """B2.  Two items share id "q0" but ask different questions, so their
    units share a unit_id while being different calls: one of the two
    answers can only be kept by dropping the other.  The harness must
    refuse the dataset (at `units` or at `execute`) rather than run it."""
    dup = [Item("q0", {"q": "first question"}),
           Item("q0", {"q": "second question"})]
    with pytest.raises(ValueError, match="q0"):
        run(list(units(grid({"model": ["m"]}), dup)), Answers(),
            tmp_path / "r.jsonl")


def test_a_cell_param_cannot_take_an_identity_column_name(tmp_path):
    """B3.  `text` is a column the runner writes (the response text), so a
    grid axis named `text` either shadows it or is shadowed.  Construction:
    the param is "PARAM" and the backend answers "question 0", so a row
    holds at most one of them.  Expected: refused before any call."""
    with pytest.raises(ValueError, match="text"):
        run(list(units(grid({"text": ["PARAM"]}), items(1))), Answers(),
            tmp_path / "r.jsonl")


def test_a_scorer_cannot_overwrite_the_unit_id(tmp_path):
    """B3.  The unit's id is fixed by its (cell, item, repeat); a scorer that
    returns a `unit_id` key would re-key the row onto another unit (or none).
    Expected: the row keeps the unit's own id and is an error row."""
    us = list(units(grid({"model": ["m"]}), items(1)))
    out = tmp_path / "r.jsonl"
    run(us, Answers(), out, score=lambda u, r: {"unit_id": "forged"})
    row = raw_rows(out)[-1]
    assert row["unit_id"] == us[0].id
    assert row["status"] == "error"
