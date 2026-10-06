"""Four plots.  Each one exists to make a specific error visible.

* ci_plot        -- a metric across the grid, with the interval, never a bar.
* forest         -- paired differences; the zero line is the whole figure.
* paired_scatter -- per-item A vs B against y = x; shows whether a mean
                    difference is a uniform shift or a few items moving,
                    which the mean and its CI cannot distinguish.
* deff_curve     -- n_eff against rows per cluster; shows augmentation
                    saturating.
"""

from __future__ import annotations

from collections.abc import Sequence

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.axes import Axes

from .stats import effective_n


def ci_plot(summary: pd.DataFrame, x: str, hue: str | None = None,
            ax: Axes | None = None, ylabel: str = "metric",
            jitter: float = 0.06) -> Axes:
    ax = ax or plt.subplots(figsize=(7, 4))[1]
    groups = [(None, summary)] if hue is None else list(
        summary.groupby(hue, observed=True))
    xs = list(dict.fromkeys(summary[x]))
    pos = {v: i for i, v in enumerate(xs)}
    for j, (name, sub) in enumerate(groups):
        off = (j - (len(groups) - 1) / 2) * jitter
        p = np.array([pos[v] for v in sub[x]], dtype=float) + off
        lo = sub["point"] - sub["lo"]
        hi = sub["hi"] - sub["point"]
        ax.errorbar(p, sub["point"], yerr=np.vstack([lo, hi]), fmt="o",
                    capsize=3, lw=1.2, label=str(name) if hue else None)
    ax.set_xticks(range(len(xs)), [str(v) for v in xs])
    ax.set_xlabel(x)
    ax.set_ylabel(ylabel)
    if hue:
        ax.legend(title=hue, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def forest(diffs: pd.DataFrame, label: str = "contrast",
           ax: Axes | None = None,
           xlabel: str = "paired difference") -> Axes:
    """`diffs` needs columns [label, point, lo, hi]."""
    ax = ax or plt.subplots(figsize=(7, 0.5 * len(diffs) + 1.5))[1]
    y = np.arange(len(diffs))
    err = np.vstack([diffs["point"] - diffs["lo"], diffs["hi"] - diffs["point"]])
    crosses = (diffs["lo"] <= 0) & (diffs["hi"] >= 0)
    ax.errorbar(diffs["point"], y, xerr=err, fmt="o", capsize=3, lw=1.2,
                color="0.25")
    ax.scatter(diffs["point"][~crosses], y[~crosses], zorder=3, color="C3")
    ax.axvline(0, color="0.6", lw=1, ls="--")
    ax.set_yticks(y, diffs[label].astype(str))
    ax.invert_yaxis()
    ax.set_xlabel(xlabel)
    ax.spines[["top", "right", "left"]].set_visible(False)
    return ax


def paired_scatter(df: pd.DataFrame, value: str, arm: str, a: str, b: str,
                   cluster: str = "cluster", ax: Axes | None = None) -> Axes:
    ax = ax or plt.subplots(figsize=(4.6, 4.6))[1]
    piv = (df[df[arm].isin([a, b])]
           .groupby([cluster, arm], observed=True)[value].mean()
           .unstack(arm)[[a, b]].dropna())
    ax.scatter(piv[b], piv[a], s=18, alpha=0.6, edgecolor="none")
    lim = [min(piv.min()), max(piv.max())]
    ax.plot(lim, lim, color="0.6", lw=1, ls="--")
    r = float(np.corrcoef(piv[a], piv[b])[0, 1])
    ax.set_xlabel(b), ax.set_ylabel(a)
    ax.set_title(f"per-item, r = {r:.2f}, k = {len(piv)}", fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    ax.spines[["top", "right"]].set_visible(False)
    return ax


def deff_curve(df: pd.DataFrame, value: str, cluster: str = "cluster",
               ms: Sequence[int] = (1, 2, 4, 8, 16), seed: int = 0,
               ax: Axes | None = None) -> Axes:
    """Subsample m rows per cluster and plot n_eff(m) against the naive N = k m.
    The gap between the two curves is exactly the DEFF of eq. (3)."""
    ax = ax or plt.subplots(figsize=(5.2, 4))[1]
    rng = np.random.default_rng(seed)
    naive, eff = [], []
    for m in ms:
        sub = (df.groupby(cluster, observed=True, group_keys=False)
                 .apply(lambda t, m=m: t.sample(
                     min(m, len(t)), random_state=int(rng.integers(1 << 31)))))
        o = effective_n(sub, value, cluster)
        naive.append(o["N"])
        eff.append(o["n_eff"])
    ax.plot(ms, naive, "o--", color="0.6", label="rows (naive N)")
    ax.plot(ms, eff, "o-", color="C0", label="$n_{\\mathrm{eff}}$")
    ax.set_xlabel("rows per cluster $m$"), ax.set_ylabel("sample size")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    return ax
