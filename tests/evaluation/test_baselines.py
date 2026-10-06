"""Floors: the arithmetic, and the ranking behaviour that is the point."""

import pandas as pd
import pytest

from evalcore.evaluation.baselines import (
    floors,
    headroom,
    majority_rate,
    prior_matched_rate,
    uniform_rate,
)


def test_rates_on_a_known_marginal():
    labels = ["a"] * 60 + ["b"] * 30 + ["c"] * 10
    assert majority_rate(labels) == pytest.approx(0.60)
    assert prior_matched_rate(labels) == pytest.approx(0.36 + 0.09 + 0.01)
    assert uniform_rate(labels) == pytest.approx(1 / 3)


def test_majority_dominates_prior_matched_dominates_nothing():
    """sum p_i^2 <= max p_i always, with equality only when degenerate."""
    for labels in (["a"] * 5 + ["b"] * 5,
                   ["a"] * 9 + ["b"],
                   list("abcdefgh")):
        assert prior_matched_rate(labels) <= majority_rate(labels) + 1e-12
    assert prior_matched_rate(["a"] * 4) == pytest.approx(1.0)
    assert majority_rate(["a"] * 4) == pytest.approx(1.0)


def test_empty_labels_raise():
    with pytest.raises(ValueError):
        majority_rate([])


def test_floors_sorted_strongest_first_and_custom_is_ranked_with_them():
    labels = ["a"] * 70 + ["b"] * 30
    df = floors(labels, custom={"stated_reference": 0.55})
    assert list(df["value"]) == sorted(df["value"], reverse=True)
    assert df.iloc[0]["floor"] == "majority"          # 0.70 beats the rest
    assert set(df["floor"]) == {"majority", "prior_matched", "uniform",
                                "stated_reference"}
    assert df.set_index("floor").loc["stated_reference", "derived_from"] == "supplied"


def test_headroom_binds_to_the_strongest_floor_not_the_stated_one():
    """The failure this guards: a stated reference that sits below a trivial
    predictor, so clearing it demonstrates nothing."""
    labels = ["a"] * 70 + ["b"] * 30
    df = floors(labels, custom={"stated_reference": 0.55})
    got = headroom(0.62, df)
    assert got["binding_floor"] == "majority"
    assert got["floor_value"] == pytest.approx(0.70)
    assert got["headroom"] == pytest.approx(-0.08)
    assert got["above_floor"] is False       # beats the stated bar, fails the real one


def test_headroom_lower_is_better_for_losses():
    df = floors(custom={"trigram": 1.784, "stated": 1.90}, custom_units="nats")
    got = headroom(1.692, df, higher_is_better=False)
    assert got["binding_floor"] == "trigram"
    assert got["headroom"] == pytest.approx(0.092)
    assert got["above_floor"] is True
    assert got["units"] == "nats"


def test_headroom_refuses_to_rank_across_scales():
    """An accuracy and a loss in one table have no meaningful ordering; the
    'strongest floor' would just be the smallest number present."""
    mixed = pd.concat([floors(["a", "b"]),
                       floors(custom={"trigram": 1.784}, custom_units="nats")])
    with pytest.raises(ValueError, match="more than one scale"):
        headroom(1.692, mixed, higher_is_better=False)


def test_floors_needs_something_to_work_from():
    with pytest.raises(ValueError, match="needs labels"):
        floors()


def test_headroom_needs_floors():
    with pytest.raises(ValueError):
        headroom(0.5, pd.DataFrame(columns=["floor", "value"]))
