"""Contamination measures: correctness, and the properties the docstrings claim."""

import numpy as np
import pytest

from evalcore.evaluation.contamination import (
    grams,
    longest_match,
    ngram_overlap,
    report,
)


def test_grams_works_on_strings_and_token_lists():
    assert grams("abcd", 2) == ["ab", "bc", "cd"]
    assert grams(["a", "b", "c"], 2) == [("a", "b"), ("b", "c")]
    assert grams([1, 2, 3], 4) == []          # shorter than n
    with pytest.raises(ValueError):
        grams("abc", 0)


def test_overlap_is_type_level_not_token_level():
    """Repeating a memorised phrase must not inflate the score.

    "abababzq" against "abcd" at n=2 has bigram tokens ab ba ab ba ab bz zq
    and types {ab, ba, bz, zq}; only "ab" is in the reference.  Type-level
    overlap is 1/4; token-level would be 3/7, inflated by the repeats."""
    assert ngram_overlap("abababzq", "abcd", 2) == pytest.approx(1 / 4)
    assert ngram_overlap("abzq", "abcd", 2) == pytest.approx(1 / 3)
    ref = "the cat sat on the mat"
    assert ngram_overlap("the cat sat", ref, 4) == pytest.approx(1.0)
    assert ngram_overlap("zzzz qqqq", ref, 4) == 0.0


def test_overlap_nan_when_candidate_too_short():
    assert np.isnan(ngram_overlap("ab", "abcdef", 4))


def test_longest_match_exact_and_monotone():
    ref = "alpha beta gamma delta"
    assert longest_match("beta gamma", ref) == len("beta gamma")
    assert longest_match("xyz", ref) == 0
    assert longest_match("", ref) == 0
    # embedded shared run, novel surroundings
    assert longest_match("QQQ beta gamma QQQ", ref) == len(" beta gamma ")


def test_longest_match_cap_is_a_lower_bound_not_a_value():
    shared = "abcdefghij" * 5
    assert longest_match(shared, shared, max_n=12) == 12    # "at least 12"
    assert longest_match(shared, shared, max_n=100) == 50


def test_longest_match_on_token_lists():
    ref = ["the", "cat", "sat", "on", "the", "mat"]
    assert longest_match(["cat", "sat", "on"], ref) == 3


def test_report_excess_is_relative_to_the_control():
    """The headline property: a candidate is judged against coincidental
    overlap, not against zero."""
    reference = "the quick brown fox jumps over the lazy dog " * 20
    memorised = "the quick brown fox jumps over the lazy dog"
    control = "a sluggish grey wolf ambles past the busy cat"
    novel = "a sluggish grey wolf ambles past the busy cat and yawns"

    df = report({"memorised": memorised, "novel": novel},
                reference, control=control, ns=(4, 8))

    assert set(df.index) == {"memorised", "novel", "control"}
    assert df.loc["control", "is_control"]
    assert df.loc["control", "longest_match_excess"] == 0
    assert df.loc["memorised", "longest_match_excess"] > 0
    assert (df.loc["novel", "longest_match_excess"]
            <= df.loc["memorised", "longest_match_excess"])
    assert df.loc["memorised", "overlap_8"] > df.loc["novel", "overlap_8"]


def test_report_without_control_omits_excess_columns():
    df = report({"a": "hello there"}, "hello there world", ns=(2,))
    assert "overlap_2" in df.columns
    assert not any(c.endswith("_excess") for c in df.columns)
    assert not df.loc["a", "is_control"]
