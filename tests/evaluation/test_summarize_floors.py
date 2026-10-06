"""summarize() attaching floors: the columns, and the CI-not-point verdict."""

import numpy as np
import pandas as pd
import pytest

from evalcore.evaluation.stats import summarize


def _frame(rate: float, n_items: int = 120, seed: int = 0) -> pd.DataFrame:
    """One arm, `rate` correct, labels 70/30 so majority = 0.70."""
    rng = np.random.default_rng(seed)
    labels = ["a"] * int(0.7 * n_items) + ["b"] * (n_items - int(0.7 * n_items))
    return pd.DataFrame({
        "arm": "m",
        "cluster": [f"i{i}" for i in range(n_items)],
        "label": labels,
        "score": rng.binomial(1, rate, n_items).astype(float),
    })


def test_no_floor_args_leaves_the_table_unchanged():
    df = _frame(0.8)
    out = summarize(df, "score", ["arm"], n_boot=200)
    assert not any(c.startswith(("binding_floor", "above_floor"))
                   for c in out.columns)


def test_floor_columns_appear_and_bind_to_the_strongest():
    df = _frame(0.9)
    out = summarize(df, "score", ["arm"], n_boot=500, label="label",
                    floor={"stated_reference": 0.55})
    row = out.iloc[0]
    assert row["binding_floor"] == "majority"        # 0.70 beats the stated 0.55
    assert row["floor_value"] == pytest.approx(0.70)
    assert row["headroom"] == pytest.approx(row["point"] - 0.70)


def test_verdict_uses_the_interval_not_the_point():
    """A cell just above its floor by the point estimate, with an interval
    that straddles it, must not be called above the floor."""
    df = _frame(0.72, n_items=40, seed=3)
    out = summarize(df, "score", ["arm"], n_boot=800, label="label")
    row = out.iloc[0]
    assert row["point"] > row["floor_value"]         # point clears it
    assert row["lo"] < row["floor_value"]            # interval does not
    assert bool(row["above_floor_ci"]) is False   # numpy bool_ in a frame


def test_clearly_above_floor_is_reported_as_such():
    df = _frame(0.97, n_items=200, seed=1)
    out = summarize(df, "score", ["arm"], n_boot=800, label="label")
    assert bool(out.iloc[0]["above_floor_ci"]) is True


@pytest.mark.filterwarnings("ignore:.*carry caveats:RuntimeWarning")  # tiny fixture
def test_supplied_only_floors_on_a_loss_scale():
    df = _frame(0.5)
    df["loss"] = 1.692
    out = summarize(df, "loss", ["arm"], n_boot=200,
                    floor={"trigram": 1.784, "stated": 1.90},
                    floor_units="nats", higher_is_better=False)
    row = out.iloc[0]
    assert row["binding_floor"] == "trigram"
    assert row["headroom"] == pytest.approx(0.092, abs=1e-6)
