r"""Uncertainty quantification for clustered evaluation data.

The unit of resampling is the CLUSTER (a source item and every repeat or
augmentation derived from it), never the row.  Motivation, in one
calculation.  Take the one-way random-effects model for a per-row score

    X_{ij} = mu + a_i + e_{ij},   i = 1..k clusters,  j = 1..m rows each,
    Var(a_i) = s2_a,  Var(e_ij) = s2_e,  all independent.

The estimator is the grand mean, theta = (1/k) sum_i Xbar_{i.}.  Since
Var(Xbar_{i.}) = s2_a + s2_e/m,

    Var(theta) = (s2_a + s2_e/m) / k.                                   (1)

A bootstrap that resamples the N = km ROWS as if exchangeable instead
reports

    Var_naive = (s2_a + s2_e) / (km).                                   (2)

The ratio of (1) to (2) is the design effect

    DEFF = [m s2_a + s2_e] / [s2_a + s2_e] = 1 + (m - 1) rho,
    rho  = s2_a / (s2_a + s2_e),                                        (3)

so the honest sample size is n_eff = N / DEFF.  At rho = 1 (an augmentation
that changes nothing the model is sensitive to) n_eff = k: ten paraphrases
of the same question buy you exactly one question.  At rho = 0 they buy ten.
Reporting n_eff alongside any augmented-data result is the difference
between claiming a 10x dataset and having one.
"""

from __future__ import annotations

import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from .baselines import floors, floors_from_frame, headroom

__all__ = [
    "BootResult",
    "benjamini_hochberg",
    "cluster_bootstrap",
    "effective_n",
    "holm",
    "icc1",
    "mcnemar_exact",
    "paired_bootstrap",
    "summarize",
    "wilson_interval",
]


@dataclass
class BootResult:
    point: float
    lo: float
    hi: float
    se: float
    n_clusters: int
    n_rows: int
    alpha: float = 0.05
    method: str = "bca"
    replicates: np.ndarray | None = field(default=None, repr=False)

    def as_row(self) -> dict:
        return {"point": self.point, "lo": self.lo, "hi": self.hi,
                "se": self.se, "n_clusters": self.n_clusters,
                "n_rows": self.n_rows, "ci_method": self.method}

    def __str__(self) -> str:
        return (f"{self.point:.4f}  [{self.lo:.4f}, {self.hi:.4f}]  "
                f"(se {self.se:.4f}, k={self.n_clusters}, N={self.n_rows})")


def _cluster_index(clusters: np.ndarray) -> list[np.ndarray]:
    """Row indices grouped by cluster label, order stable."""
    order = np.argsort(clusters, kind="stable")
    sorted_c = clusters[order]
    edges = np.flatnonzero(np.r_[True, sorted_c[1:] != sorted_c[:-1], True])
    return [order[edges[i]:edges[i + 1]] for i in range(len(edges) - 1)]


def _bca_bounds(theta: float, reps: np.ndarray, jack: np.ndarray,
                alpha: float) -> tuple[float, float]:
    """Bias-corrected and accelerated percentile bounds (Efron 1987).

    z0 corrects median bias:      z0 = Phi^{-1}( #{theta* < theta} / B ).
    a  corrects skew, from the leave-one-CLUSTER-out jackknife values
    theta_(i):   a = sum (jbar - theta_(i))^3 / (6 [sum (jbar - theta_(i))^2]^{3/2}).
    """
    B = reps.size
    prop = float(np.mean(reps < theta))
    prop = min(max(prop, 1.0 / (2 * B)), 1.0 - 1.0 / (2 * B))
    z0 = sps.norm.ppf(prop)
    d = jack.mean() - jack
    denom = 6.0 * (np.sum(d ** 2) ** 1.5)
    a = 0.0 if denom == 0 else float(np.sum(d ** 3) / denom)
    out = []
    for q in (alpha / 2, 1 - alpha / 2):
        z = sps.norm.ppf(q)
        adj = z0 + (z0 + z) / (1 - a * (z0 + z))
        out.append(float(np.clip(sps.norm.cdf(adj), 0.0, 1.0)))
    return tuple(np.quantile(reps, out))


def cluster_bootstrap(df: pd.DataFrame, value: str, cluster: str = "cluster",
                      stat: Callable[[np.ndarray], float] | None = None,
                      n_boot: int = 10_000, seed: int = 0, alpha: float = 0.05,
                      method: str = "bca") -> BootResult:
    """Nonparametric cluster bootstrap: resample CLUSTER LABELS with
    replacement, keep every row of each drawn cluster, recompute `stat`.

    `stat` defaults to the unweighted mean over clusters of the within-
    cluster mean -- i.e. every source item counts once regardless of how
    many augmentations it spawned.  This is almost always what you want:
    the row-weighted mean silently up-weights whichever items happened to
    augment most, which is a property of your generator, not the model.
    """
    d = df[[value, cluster]].dropna()
    vals = d[value].to_numpy(dtype=float)
    idx = _cluster_index(d[cluster].to_numpy())
    k = len(idx)
    if k < 2:
        raise ValueError(f"need >=2 clusters for a CI, got {k}")

    if stat is None:
        # fast path: mean of within-cluster means, via per-cluster means
        cmeans = np.array([vals[ix].mean() for ix in idx])
        theta = float(cmeans.mean())
        rng = np.random.default_rng(seed)
        draws = rng.integers(0, k, size=(n_boot, k))
        reps = cmeans[draws].mean(axis=1)
        jack = np.array([np.delete(cmeans, i).mean() for i in range(k)])
    else:
        theta = float(stat(vals))
        rng = np.random.default_rng(seed)
        reps = np.empty(n_boot)
        for b in range(n_boot):
            draw = rng.integers(0, k, size=k)
            reps[b] = stat(np.concatenate([vals[idx[i]] for i in draw]))
        jack = np.array([
            stat(np.concatenate([vals[idx[j]] for j in range(k) if j != i]))
            for i in range(k)])

    if method == "bca":
        lo, hi = _bca_bounds(theta, reps, jack, alpha)
    elif method == "percentile":
        lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    else:
        raise ValueError(f"unknown method {method!r}")
    return BootResult(theta, float(lo), float(hi), float(reps.std(ddof=1)),
                      k, len(vals), alpha, method, reps)


def paired_bootstrap(df: pd.DataFrame, value: str, arm: str, a: str, b: str,
                     cluster: str = "cluster", n_boot: int = 10_000,
                     seed: int = 0, alpha: float = 0.05,
                     method: str = "bca") -> BootResult:
    r"""CI for the paired difference E[A] - E[B] on the clusters both arms
    saw.  Reduce each cluster to D_i = Abar_i - Bbar_i, then bootstrap D.

        Var(Dbar) = (s2_A + s2_B - 2 r s_A s_B) / k                     (4)

    with r the across-item correlation of the two arms' scores.  Item
    difficulty is common to both arms, so r is typically 0.6-0.9 on eval
    data and (4) is several times smaller than the unpaired variance.
    Geometrically: differencing projects out the shared item-difficulty
    direction in R^k, and the CI shrinks by the length of what was removed.

    This is also why "the two CIs overlap" is NOT a test.  For independent
    arms with equal SE, 95% intervals stop overlapping only around
    p ~ 0.006; for paired arms the overlap criterion ignores r entirely and
    can be arbitrarily conservative.  Always report the difference's own CI.
    """
    d = df[df[arm].isin([a, b])][[value, arm, cluster]].dropna()
    piv = (d.groupby([cluster, arm], observed=True)[value]
             .mean().unstack(arm))
    if a not in piv or b not in piv:
        raise ValueError(f"arms {a!r}/{b!r} not both present")
    piv = piv[[a, b]].dropna()
    if len(piv) < 2:
        raise ValueError("need >=2 clusters seen by both arms")
    diff = (piv[a] - piv[b]).to_numpy(dtype=float)
    k = diff.size
    theta = float(diff.mean())
    rng = np.random.default_rng(seed)
    reps = diff[rng.integers(0, k, size=(n_boot, k))].mean(axis=1)
    jack = np.array([np.delete(diff, i).mean() for i in range(k)])
    if method == "bca":
        lo, hi = _bca_bounds(theta, reps, jack, alpha)
    else:
        lo, hi = np.quantile(reps, [alpha / 2, 1 - alpha / 2])
    res = BootResult(theta, float(lo), float(hi), float(reps.std(ddof=1)),
                     k, int(d.shape[0]), alpha, method, reps)
    return res


def paired_pvalue(res: BootResult) -> float:
    """Two-sided bootstrap p for H0: difference = 0, by inversion."""
    reps = res.replicates
    if reps is None:
        raise ValueError("this BootResult carries no replicates")
    p = 2 * min(float(np.mean(reps <= 0)), float(np.mean(reps >= 0)))
    return float(min(1.0, max(p, 1.0 / reps.size)))


def icc1(df: pd.DataFrame, value: str, cluster: str = "cluster") -> dict:
    r"""One-way random-effects ICC(1) for unbalanced clusters.

        MS_b = sum_i m_i (xbar_i - xbar)^2 / (k - 1)
        MS_w = sum_i sum_j (x_ij - xbar_i)^2 / (N - k)
        m_0  = (N - sum_i m_i^2 / N) / (k - 1)          [balanced: m_0 = m]
        s2_a = (MS_b - MS_w) / m_0,     rho = s2_a / (s2_a + MS_w)

    m_0 is the variance-matched cluster size; it reduces to m when the
    design is balanced and is what belongs in DEFF when it is not.
    rho is clipped at 0 -- a negative moment estimate means the between-
    cluster variance is not resolved by this much data, not that it is
    negative.
    """
    d = df[[value, cluster]].dropna()
    g = d.groupby(cluster, observed=True)[value]
    m = g.size().to_numpy(dtype=float)
    means = g.mean().to_numpy(dtype=float)
    k, N = m.size, float(m.sum())
    if k < 2 or N <= k:
        return {"k": k, "N": int(N), "m0": float("nan"), "rho": float("nan"),
                "ms_between": float("nan"), "ms_within": float("nan")}
    grand = float((m * means).sum() / N)
    ms_b = float((m * (means - grand) ** 2).sum() / (k - 1))
    ss_w = float(((d[value].to_numpy(dtype=float)
                   - d[cluster].map(g.mean()).to_numpy(dtype=float)) ** 2).sum())
    ms_w = ss_w / (N - k)
    m0 = float((N - (m ** 2).sum() / N) / (k - 1))
    s2_a = (ms_b - ms_w) / m0 if m0 > 0 else 0.0
    rho = 0.0 if (s2_a <= 0 or (s2_a + ms_w) <= 0) else s2_a / (s2_a + ms_w)
    return {"k": k, "N": int(N), "m0": m0, "rho": float(rho),
            "ms_between": ms_b, "ms_within": ms_w}


def effective_n(df: pd.DataFrame, value: str,
                cluster: str = "cluster") -> dict:
    """DEFF = 1 + (m0 - 1) rho and n_eff = N / DEFF; see eq. (3)."""
    o = icc1(df, value, cluster)
    if o["N"] == o["k"]:
        # No within-cluster replication: rho is not identifiable, but the
        # design effect is exactly 1 and n_eff = N. Returning NaN here would
        # blank the column on the most common design of all -- one generation
        # per item, no augmentation.
        return {**o, "rho": float("nan"), "deff": 1.0, "n_eff": float(o["N"])}
    if not np.isfinite(o["rho"]):
        return {**o, "deff": float("nan"), "n_eff": float("nan")}
    deff = 1.0 + (o["m0"] - 1.0) * o["rho"]
    return {**o, "deff": float(deff), "n_eff": float(o["N"] / deff)}


def wilson_interval(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    r"""Score interval for a binomial proportion.  Invert the score test
    (phat - p)/sqrt(p(1-p)/n) = +-z rather than substituting phat for p in
    the variance, which is what Wald does and why Wald collapses to zero
    width at phat = 0 or 1:

        centre = (phat + z^2/2n) / (1 + z^2/n),
        half   = z sqrt(phat(1-phat)/n + z^2/4n^2) / (1 + z^2/n).

    Use for a single accuracy with INDEPENDENT items.  With repeats or
    augmentations, use cluster_bootstrap instead -- Wilson has no way to
    know that n rows are not n items.
    """
    if n == 0:
        return (float("nan"), float("nan"))
    z = sps.norm.ppf(1 - alpha / 2)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (float(max(0.0, centre - half)), float(min(1.0, centre + half)))


def mcnemar_exact(b: int, c: int) -> float:
    """Exact two-sided McNemar for paired binary outcomes: b = A right /
    B wrong, c = A wrong / B right.  Concordant pairs carry no information
    about the difference, which is the whole point of pairing."""
    n = b + c
    if n == 0:
        return 1.0
    return float(min(1.0, 2 * sps.binom.cdf(min(b, c), n, 0.5)))


def benjamini_hochberg(p: Sequence[float], q: float = 0.05) -> np.ndarray:
    """BH step-up: reject the largest i with p_(i) <= i q / m."""
    pv = np.asarray(p, dtype=float)
    m = pv.size
    order = np.argsort(pv)
    thresh = (np.arange(1, m + 1) / m) * q
    passed = pv[order] <= thresh
    out = np.zeros(m, dtype=bool)
    if passed.any():
        out[order[:int(np.max(np.flatnonzero(passed))) + 1]] = True
    return out


def holm(p: Sequence[float], alpha: float = 0.05) -> np.ndarray:
    """Holm-Bonferroni step-down; controls FWER, no independence assumed."""
    pv = np.asarray(p, dtype=float)
    m = pv.size
    order = np.argsort(pv)
    out = np.zeros(m, dtype=bool)
    for rank, i in enumerate(order):
        if pv[i] > alpha / (m - rank):
            break
        out[i] = True
    return out


def summarize(df: pd.DataFrame, value: str, by: Sequence[str],
              cluster: str = "cluster", n_boot: int = 5_000, seed: int = 0,
              alpha: float = 0.05, method: str = "bca",
              label: str | None = None,
              floor: Mapping[str, float] | None = None,
              floor_units: str = "rate",
              higher_is_better: bool = True) -> pd.DataFrame:
    """One row per grid cell: point estimate, cluster-bootstrap CI, and the
    honest sample size.  `n_eff` is reported next to `n_rows` deliberately;
    a table that shows only n_rows for augmented data is misleading even
    when every number in it is correct.

    Optionally attaches the floor the cell has to beat.  `label` names a
    column of item labels, from which the trivial-predictor floors are
    computed per cell; `floor` supplies floors that cannot be derived from a
    marginal (an n-gram model, a published reference).  Either or both.

    `n_rows` counts the rows that carry `value`; `n_excluded` counts the
    rows of the cell that do not -- error, truncated and unparsed rows, or
    a NaN from the scorer.  The bootstrap drops those, and before this
    column it did so silently, so a cell could lose a third of its rows
    and show a tight interval on the survivors.  `attrition` says which
    category each one fell in.

    The verdict column is `above_floor_ci`, not `above_floor`: a point
    estimate above a floor is not evidence of being above it, and the CI is
    already computed here.  A cell whose interval straddles its floor gets
    False, which is the honest answer and the one a reader of the table
    would otherwise have to reconstruct.

    `caveats` says, per cell, when a number is computed correctly but
    cannot be read at face value (see `cell_caveats`); cells with any
    caveat are also named in one RuntimeWarning.
    """
    want_floor = label is not None or floor is not None
    rows = []
    for key, sub in df.groupby(list(by), observed=True, dropna=False):
        keys = key if isinstance(key, tuple) else (key,)
        rec = dict(zip(by, keys, strict=True))
        try:
            res = cluster_bootstrap(sub, value, cluster, n_boot=n_boot,
                                    seed=seed, alpha=alpha, method=method)
            rec.update(res.as_row())
        except ValueError as exc:
            rec.update({"point": float(sub[value].mean()), "lo": float("nan"),
                        "hi": float("nan"), "se": float("nan"),
                        "n_clusters": sub.dropna(subset=[value])[cluster]
                        .nunique(),
                        "n_rows": int(sub[value].notna().sum()),
                        "ci_method": f"skipped: {exc}"})
        rec["n_excluded"] = int(sub[value].isna().sum())
        rec.update({k: v for k, v in effective_n(sub, value, cluster).items()
                    if k in ("rho", "m0", "deff", "n_eff")})
        if want_floor:
            fl = (floors_from_frame(sub, label, custom=floor,
                                    custom_units=floor_units)
                  if label is not None
                  else floors(custom=floor, custom_units=floor_units))
            h = headroom(rec["point"], fl, higher_is_better=higher_is_better)
            bound = rec["lo"] if higher_is_better else rec["hi"]
            rec.update({"binding_floor": h["binding_floor"],
                        "floor_value": h["floor_value"],
                        "headroom": h["headroom"],
                        "above_floor_ci": bool(
                            bound > h["floor_value"] if higher_is_better
                            else bound < h["floor_value"])
                        if not np.isnan(bound) else False})
        rec["caveats"] = "; ".join(cell_caveats(rec))
        rows.append(rec)
    out = pd.DataFrame(rows)
    flagged = out[out["caveats"] != ""] if len(out) else out
    if len(flagged):
        warnings.warn(
            f"{len(flagged)} of {len(out)} cells carry caveats (see the "
            f"'caveats' column), e.g. {flagged['caveats'].iloc[0]}",
            RuntimeWarning, stacklevel=2)
    return out


# Coverage of a nominal 95% cluster-bootstrap interval against the true
# mean, by number of clusters k (BCa; 400 simulated datasets each, k
# clusters of 6 rows, rho = 0.6; binomial SE of each figure ~0.011).
# Measured 5 Oct 2026; the percentile method is within 0.03 throughout.
SMALL_K_COVERAGE = {3: 0.68, 5: 0.83, 10: 0.88, 20: 0.93, 30: 0.95}
MIN_CLUSTERS = 20
MAX_EXCLUDED_SHARE = 0.10


def few_clusters(k: int) -> str | None:
    """The few-clusters caveat for k clusters, or None at k >= MIN_CLUSTERS."""
    if not 0 < k < MIN_CLUSTERS:
        return None
    near = max((c for c in SMALL_K_COVERAGE if c <= k), default=3)
    return (f"only {k} clusters: the interval is too narrow (a nominal 95% "
            f"interval covers ~{SMALL_K_COVERAGE[near]:.0%} at k={near})")


def cell_caveats(rec: Mapping[str, Any]) -> list[str]:
    """Reasons a summary row is correct but not to be read at face value.

    * Few clusters: the bootstrap estimates the spread of a mean from the
      clusters themselves, and with few of them it underestimates it, so
      the interval is too narrow (SMALL_K_COVERAGE: a nominal 95% interval
      covers ~68% of the time at k = 3).
    * No spread: every value identical, so the interval has zero width --
      a statement about the sample, not certainty about the population.
    * Many excluded: rows without a value (errors, truncated, unparsed)
      are left out, so the estimate describes the survivors; `attrition`
      says which rows are missing and why.
    """
    out = []
    few = few_clusters(int(rec.get("n_clusters") or 0))
    if few:
        out.append(few)
    if rec.get("se") == 0.0:
        out.append("no spread in the values: a zero-width interval, not "
                   "certainty")
    n_rows, n_ex = int(rec.get("n_rows") or 0), int(rec.get("n_excluded") or 0)
    if n_rows + n_ex and n_ex / (n_rows + n_ex) > MAX_EXCLUDED_SHARE:
        out.append(f"{n_ex / (n_rows + n_ex):.0%} of rows have no value: the "
                   f"estimate describes the survivors (see attrition)")
    return out
