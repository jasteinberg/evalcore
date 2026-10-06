"""Trivial-predictor floors, so a headline number has a scale.

A metric alone is not interpretable.  "1.69 nats" or "68% accurate" is a
number, not a result; what makes it a result is the distance to what you get
for free.  This module computes the free numbers.

The failure it guards against is specific and common: a stated baseline that
is *below* a trivial predictor, so clearing it demonstrates nothing.  It is
easy to miss because the comparison is never made -- the reference number
arrives from outside, is treated as the bar, and nobody checks the bar.

Three floors, all computed from the label marginal alone:

    majority        always predict the modal class
    prior_matched   sample a label from the marginal, sum_i p_i^2
    uniform         guess uniformly among observed classes, 1/k

`headroom` compares an observed metric to the *strongest* of these, because
the honest bar is the best thing you get for free, not the most flattering
one.  A negative headroom means the model has not yet earned its complexity.

Floors are deterministic given the labels: they have no sampling error of
their own, so the uncertainty in any comparison lives entirely in the observed
metric and belongs to the bootstrap in `stats`.  For metrics with no label
structure (perplexity, free-form judge scores), a floor has to come from a
cheap model rather than from the marginal -- `custom` exists to carry that
number into the same table so the comparison is made in one place.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Mapping, Sequence

import numpy as np
import pandas as pd

__all__ = [
    "floors",
    "floors_from_frame",
    "headroom",
    "majority_rate",
    "prior_matched_rate",
    "uniform_rate",
]


def _marginal(labels: Sequence) -> tuple[np.ndarray, int]:
    counts = np.array(sorted(Counter(labels).values(), reverse=True),
                      dtype=float)
    if counts.size == 0:
        raise ValueError("no labels")
    return counts / counts.sum(), counts.size


def majority_rate(labels: Sequence) -> float:
    """Accuracy of always predicting the modal class."""
    p, _ = _marginal(labels)
    return float(p[0])


def prior_matched_rate(labels: Sequence) -> float:
    """Accuracy of sampling a prediction from the label marginal, sum p_i^2.

    Also the collision probability of the marginal, and 1 minus the Gini
    impurity -- the same quantity that appears as an inverse participation
    ratio elsewhere in this package."""
    p, _ = _marginal(labels)
    return float(np.sum(p ** 2))


def uniform_rate(labels: Sequence) -> float:
    """Accuracy of guessing uniformly among the classes actually observed."""
    _, k = _marginal(labels)
    return 1.0 / k


def floors(labels: Sequence | None = None,
           custom: Mapping[str, float] | None = None,
           custom_units: str = "rate") -> pd.DataFrame:
    """The free numbers for this label distribution, strongest first.

    `custom` carries floors that cannot be derived from the marginal -- an
    n-gram model's loss, a retrieval baseline, a published reference figure.
    Passing a stated reference here is the point: it gets ranked against the
    trivial predictors instead of being assumed to sit above them.

    Every row carries `units`, and the marginal-derived floors are on the
    `rate` scale (an accuracy in [0, 1]).  A loss-scale floor therefore needs
    `custom_units` set and `labels` left out: a table holding both a 0.5
    accuracy and a 1.78-nat loss has no meaningful ordering, and `headroom`
    refuses it rather than silently comparing across scales.
    """
    rows: list[dict] = []
    if labels is not None:
        rows += [{"floor": "majority", "value": majority_rate(labels),
                  "units": "rate", "derived_from": "label marginal"},
                 {"floor": "prior_matched", "value": prior_matched_rate(labels),
                  "units": "rate", "derived_from": "label marginal"},
                 {"floor": "uniform", "value": uniform_rate(labels),
                  "units": "rate", "derived_from": "class count"}]
    for name, value in (custom or {}).items():
        rows.append({"floor": name, "value": float(value),
                     "units": custom_units, "derived_from": "supplied"})
    if not rows:
        raise ValueError("floors() needs labels, custom floors, or both")
    return (pd.DataFrame(rows)
            .sort_values("value", ascending=False, kind="stable")
            .reset_index(drop=True))


def floors_from_frame(df: pd.DataFrame, label: str,
                      custom: Mapping[str, float] | None = None,
                      custom_units: str = "rate") -> pd.DataFrame:
    """Floors for the label column of a tidy frame.

    The frame-shaped entry point, so a floor can be computed from the same
    object every other analysis function consumes rather than from a bare
    sequence assembled by hand at the call site."""
    if label not in df.columns:
        raise KeyError(f"no label column {label!r} in frame")
    return floors(df[label].tolist(), custom=custom, custom_units=custom_units)


def headroom(observed: float, floors_df: pd.DataFrame,
             higher_is_better: bool = True) -> dict:
    """Distance from an observed metric to the strongest floor.

    Returns the binding floor by name, so a result can be reported as "beats
    the trigram floor by 0.09" rather than "beats the stated reference",
    which may be the weaker claim.

    Refuses a table with mixed units: ranking an accuracy against a loss
    produces a binding floor that is merely the smallest number present.
    """
    if floors_df.empty:
        raise ValueError("no floors to compare against")
    units = set(floors_df.get("units", pd.Series(["?"] * len(floors_df))))
    if len(units) > 1:
        raise ValueError(
            f"floors span more than one scale ({sorted(units)}); compare "
            f"within a scale, since the ordering across them is meaningless")
    idx = (floors_df["value"].idxmax() if higher_is_better
           else floors_df["value"].idxmin())
    binding = floors_df.loc[idx]
    gap = (observed - binding["value"] if higher_is_better
           else binding["value"] - observed)
    return {"observed": float(observed),
            "binding_floor": str(binding["floor"]),
            "floor_value": float(binding["value"]),
            "units": str(binding.get("units", "?")),
            "headroom": float(gap),
            "above_floor": bool(gap > 0)}
