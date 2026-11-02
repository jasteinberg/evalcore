"""evalcore.testing: the comparison every regression test leans on, so it
is checked against hand-built pairs whose verdict is known by inspection."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from evalcore.testing import assert_same, mismatches, to_plain


def test_equal_structures_have_no_mismatch():
    a = {"x": [1, 2.5, None, "s", True], "y": np.arange(3.0),
         "f": pd.DataFrame({"a": [1, 2], "b": ["u", "v"]})}
    b = {"x": [1, 2.5, None, "s", True], "y": np.arange(3.0),
         "f": pd.DataFrame({"a": [1, 2], "b": ["u", "v"]})}
    assert mismatches(a, b) == []
    assert_same(a, b)


def test_every_mismatch_is_reported_with_its_path():
    a = {"m": {"logp": [0.0, -1.0, -2.0]}, "s": "ok", "k": 1}
    b = {"m": {"logp": [0.0, -1.5, -2.5]}, "s": "error", "j": 1}
    found = mismatches(a, b)
    assert len(found) == 5        # two numbers, one string, k only in a, j only in b
    assert any(f.startswith("root['m']['logp'][1]") for f in found)
    assert any(f.startswith("root['m']['logp'][2]") for f in found)
    assert any(f.startswith("root['s']") for f in found)
    with pytest.raises(AssertionError, match="5 place"):
        assert_same(a, b)


@pytest.mark.parametrize("x,y,rtol,atol,same", [
    (1.0, 1.0 + 1e-9, 0.0, 1e-8, True),      # |d| = 1e-9 <= atol
    (1.0, 1.0 + 1e-7, 0.0, 1e-8, False),
    (100.0, 101.0, 0.01, 0.0, True),         # 1 <= 0.01 * 101
    (100.0, 102.0, 0.01, 0.0, False),        # 2 > 1.02
    (math.nan, math.nan, 0.0, 0.0, True),    # NaN matches NaN in place
    (math.nan, 0.0, 0.0, 1.0, False),
    (math.inf, math.inf, 0.0, 0.0, True),
    (math.inf, 1e308, 1.0, 0.0, False),
])
def test_numbers_use_the_stated_tolerance(x, y, rtol, atol, same):
    assert (mismatches(x, y, rtol, atol) == []) is same


def test_shapes_lengths_columns_and_types_must_match():
    assert mismatches(np.zeros(3), np.zeros(4))
    assert mismatches(np.zeros((2, 3)), np.zeros((3, 2)))
    assert mismatches([1, 2], [1, 2, 3])
    assert mismatches(pd.DataFrame({"a": [1]}), pd.DataFrame({"b": [1]}))
    assert mismatches(pd.DataFrame({"a": [1]}), pd.DataFrame({"a": [1, 2]}))
    assert mismatches("1", 1)                 # a string never equals a number
    assert mismatches(True, 1)                # nor a bool
    assert mismatches(None, 0.0)


def test_a_tensor_is_compared_as_its_values():
    torch = pytest.importorskip("torch")
    assert mismatches(torch.tensor([1.0, 2.0]), np.array([1.0, 2.0])) == []
    assert mismatches(torch.tensor([1.0, 2.0]), np.array([1.0, 3.0]))


def test_to_plain_round_trips_through_json_and_compares_equal():
    import json
    obj = {"f": pd.DataFrame({"a": [1, 2], "b": [0.5, math.nan]}),
           "arr": np.arange(4).reshape(2, 2), "n": np.float32(0.25),
           "i": np.int64(3), "flag": np.bool_(True)}
    plain = to_plain(obj)
    back = json.loads(json.dumps(plain))
    assert back["arr"] == [[0, 1], [2, 3]]
    assert back["n"] == 0.25 and back["i"] == 3 and back["flag"] is True
    assert back["f"]["columns"] == ["a", "b"]
    assert_same(back, plain)
