"""Plots draw what the tables say: values read back from the Axes."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import pandas as pd
from matplotlib.figure import Figure

from evalcore.evaluation.plots import ci_plot, forest, paired_scatter
from evalcore.testing import assert_same


def _bar_segments(ax):
    """The error-bar segments of the first errorbar container on ax."""
    (barcol,) = ax.containers[0].lines[2]
    return [tuple(map(tuple, s)) for s in barcol.get_segments()]


def test_c36_plots_draw_the_interval_and_the_cluster_means():
    """C36.  An asymmetric interval point 0.5, [0.2, 0.9] must be drawn
    from 0.2 to 0.9 (a symmetric bar, or swapped half-widths, would give
    0.1..0.8).  ci_plot draws it vertically at x = 0, forest horizontally
    at y = 0.  paired_scatter plots one point per cluster at (mean B,
    mean A): cluster c0 has A rows 0, 1 and B rows 1, 1, 0, so (2/3, 1/2).
    Cluster c1 has A 1 and B 0, 1, so (1/2, 1).  Values read back from the
    Axes (Agg backend, no pyplot figure).  Tolerance 1e-12: float
    arithmetic on the half-widths 0.3 and 0.4."""
    summ = pd.DataFrame({"condition": ["m"], "point": [0.5], "lo": [0.2], "hi": [0.9]})
    ax = Figure().add_subplot()
    ci_plot(summ, "condition", ax=ax)
    assert_same(_bar_segments(ax), [((0.0, 0.2), (0.0, 0.9))], atol=1e-12)

    ax = Figure().add_subplot()
    forest(summ.rename(columns={"condition": "contrast"}), ax=ax)
    assert_same(_bar_segments(ax), [((0.2, 0.0), (0.9, 0.0))], atol=1e-12)

    df = pd.DataFrame({"cluster": ["c0"] * 5 + ["c1"] * 3,
                       "condition": ["a", "a", "b", "b", "b", "a", "b", "b"],
                       "y": [0.0, 1, 1, 1, 0, 1, 0, 1]})
    ax = Figure().add_subplot()
    paired_scatter(df, "y", "condition", "a", "b", ax=ax)
    pts = sorted(map(tuple, ax.collections[0].get_offsets().tolist()))
    assert_same(pts, [(0.5, 1.0), (2 / 3, 0.5)], atol=1e-12)
