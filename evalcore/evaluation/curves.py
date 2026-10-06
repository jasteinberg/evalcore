r"""Curves along a model axis: accuracy, susceptibility and composition, with
uncertainty paired across the axis.

The setting.  One fixed item set is scored at every point x_1 < ... < x_n of
an axis -- parameter count N, training tokens D, a checkpoint index.  For
item i at point x, teacher forcing gives per-token correctness
c_{ij}(x) in {0, 1} for target positions j < l_i, and the sequence is
right iff every token is:

    E_i(x) = prod_{j < l_i} c_{ij}(x),        A(x) = < E_i(x) >_i.

Exact match is computed HERE, from the same c_{ij} the composition uses,
never read from a separately stored column.  A_obs and A_pred are then the
same functional of the same data, and a comparison between them cannot mix
two definitions of "exact match" (greedy against teacher-forced).

Susceptibility.  chi(x) = dA / dlog x, by centred differences on the
non-uniform grid u = log x (numpy.gradient: second order in the interior,
first order at the ends).  The peak location argmax chi and the peak value
max chi (the sharpness s) are functionals of the whole curve.

Composition.  With position marginals m_j = P(c_j = 1 | l > j), pooled over
lengths, independence across positions predicts

    A_pred = < prod_{j < l_i} m_j >_i.

If the length classes differ in their marginals, A_pred is biased even when
tokens are independent WITHIN every class (a pooling, Simpson-type effect).
The length-conditioned null removes it:

    A_pred^lc = sum_l f_l prod_{j < l} m_{j|l},   f_l = P(l_i = l),

and the residual phi_lc = A_obs - A_pred^lc = sum_l f_l phi_l is the genuine
within-class dependence, phi_l = P(all correct | l) - prod_j m_{j|l}.  For
lengths {1, 2} this is f_2 phi_2, with phi_2 = P(c_1 c_2 = 1 | l = 2) - b_1 b_2,
and the pooled residual decomposes exactly as

    phi = f_1 f_2 (1 - b_2)(a_S - b_1) + f_2 phi_2,

with a_S the single-token accuracy and b_j the two-token class's marginals
(the pooling term, plus the within-class term).

Uncertainty.  Every quantity above is a functional of the item-by-axis
table, and the items are the SAME at every x.  So the bootstrap resamples
items once per replicate and reuses that draw at every x: each replicate is
a whole curve, and chi, the peak and the residuals are computed from it.
Item difficulty is common to all points on the axis, so differences along
x (which is what chi is) shed it, exactly as a paired difference does
(stats.paired_bootstrap, eq. 4).  Resampling each x independently would
treat that shared variance as noise in chi and inflate its interval.
Intervals are percentile intervals of the replicates.  If items carry
augmentations, whole clusters are resampled.
"""

from __future__ import annotations

import functools
import warnings
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import ArrayLike

from .stats import few_clusters

__all__ = ["AxisData", "axis_bootstrap", "axis_curves", "axis_data",
           "composition", "crossing", "curve_caveats", "explode_tokens",
           "subsampled_sharpness", "susceptibility", "value_curves"]

OK, NT = "meta_argmax_ok", "meta_n_target"
MAX_CENSORED_SHARE = 0.10


def susceptibility(x: ArrayLike, A: ArrayLike,
                   log: bool = True) -> np.ndarray:
    """chi = dA/dlog x (or dA/dx) by centred differences on a non-uniform
    grid.  `A` may be (n_x,) or (n_reps, n_x)."""
    u = np.log(np.asarray(x, float)) if log else np.asarray(x, float)
    return np.gradient(np.asarray(A, float), u, axis=-1)


def explode_tokens(df: pd.DataFrame,
                   cols: Sequence[str] = ("meta_logp", "meta_argmax_ok",
                                          "meta_target_tokens",
                                          "meta_target_ids")
                   ) -> pd.DataFrame:
    """One row per (unit, target position) from a frame of teacher-forced
    rows.  Columns holding per-token lists are exploded together; `position`
    counts from 0 within each unit.  Rows without token lists (errors,
    generations) are dropped, so pass `status == "ok"` rows."""
    cols = [c for c in cols if c in df.columns]
    if not cols:
        raise ValueError("no per-token list columns in the frame")
    # arrays as well as lists: a parquet round trip returns ndarray cells
    keep = df[df[cols[0]].map(lambda v: isinstance(v, (list, tuple,
                                                         np.ndarray)))]
    # positions are counted per ROW, so the index must be unique first: a
    # concat of frames (one per checkpoint) repeats labels
    keep = keep.reset_index(drop=True)
    out = keep.explode(cols, ignore_index=False)
    out["position"] = out.groupby(level=0).cumcount()
    return out.reset_index(drop=True)


@dataclass
class AxisData:
    """The item-by-axis table, aligned: row i is the same item at every x.

    ok[x] is (k, L) float with zeros beyond each item's length, valid[x]
    marks the positions j < l_i, nt[x] is (k,).  `cluster` maps items to
    resampling units (k,) as integer codes."""
    x: np.ndarray
    items: np.ndarray
    cluster: np.ndarray
    ok: list[np.ndarray]
    valid: list[np.ndarray]
    nt: list[np.ndarray]
    n_dropped: int = 0


def _align(d: pd.DataFrame, x: str, item: str, cluster: str
           ) -> tuple[np.ndarray, dict, np.ndarray, np.ndarray, int]:
    """Validate a frame for a paired axis analysis and align it: the axis
    values, rows per x indexed by item, the items present at EVERY x, their
    cluster codes, and how many items were dropped for missing a point.

    Refused, never guessed: a non-numeric axis (strings sort as text), a
    missing cluster label, two rows for one (item, x)."""
    if not pd.api.types.is_numeric_dtype(d[x]):
        raise ValueError(
            f"axis column {x!r} must be numeric (got {d[x].dtype}); strings "
            f"would sort as text, so convert it first")
    if cluster in d.columns and d[cluster].isna().any():
        raise ValueError(
            f"{int(d[cluster].isna().sum())} rows have no {cluster!r} label; "
            f"every item needs a cluster to be resampled with")
    dup = d.duplicated([x, item])
    if dup.any():
        raise ValueError(
            f"{int(dup.sum())} (item, {x}) pairs have more than one row: two "
            f"arms, two tasks or several repeats share item ids.  Filter the "
            f"frame to one row per item per axis point first.")
    xs = np.sort(d[x].unique())
    if len(xs) < 2:
        raise ValueError(f"need >= 2 distinct values of {x!r}, got {len(xs)}")
    per_x = {v: g.set_index(item) for v, g in d.groupby(x)}
    common = set.intersection(*(set(g.index) for g in per_x.values()))
    every = set().union(*(set(g.index) for g in per_x.values()))
    items = np.array(sorted(common))
    if len(items) < 2:
        raise ValueError("fewer than 2 items are scored at every axis point")
    n_dropped = len(every) - len(common)
    if n_dropped:
        warnings.warn(f"{n_dropped} items are not scored at every value of "
                      f"{x!r} and are excluded from the paired analysis",
                      RuntimeWarning, stacklevel=3)
    first = per_x[xs[0]].loc[items]
    cl = (first[cluster] if cluster in first.columns
          else pd.Series(items, index=items))
    return xs, per_x, items, pd.factorize(cl.to_numpy())[0], n_dropped


def axis_data(df: pd.DataFrame, x: str, item: str = "item_id",
              cluster: str = "cluster", ok: str = OK,
              n_target: str = NT) -> AxisData:
    """Build the aligned table from a run frame (one row per item per x).

    Only items scored at EVERY x are kept: pairing needs the same items
    throughout, and an item missing at one point (an error row, a killed
    run) would otherwise make that point's mean a different population.
    The number dropped is recorded and warned about, never silent."""
    d = df
    if "status" in d.columns:
        d = d[d["status"] == "ok"]
    d = d[d[ok].map(lambda v: isinstance(v, (list, tuple, np.ndarray)))]
    xs, per_x, items, codes, n_dropped = _align(d, x, item, cluster)
    oks, valids, nts = [], [], []
    for v in xs:
        g = per_x[v].loc[items]
        nt = g[n_target].to_numpy(int)
        L = int(nt.max())
        o = np.zeros((len(items), L))
        for r, seq in enumerate(g[ok]):
            o[r, :len(seq)] = np.asarray(seq, float)
        if not (np.array([len(s) for s in g[ok]]) == nt).all():
            raise ValueError(f"{ok} lengths disagree with {n_target} at "
                             f"{x}={v}")
        oks.append(o)
        valids.append(np.arange(L)[None, :] < nt[:, None])
        nts.append(nt)
    return AxisData(xs.astype(float), items, codes, oks, valids, nts,
                    n_dropped)


def _wsum(w: np.ndarray, a: np.ndarray) -> np.ndarray:
    """sum_k w[b, k] a[k, ...].  einsum rather than `@`: the inner shapes are
    (replicates x items) times (items x a few positions), for which a
    threaded BLAS spends far longer starting threads than multiplying --
    measured at ~8x slower than this on a laptop."""
    return np.einsum("bk,k...->b...", w, np.asarray(a, float))


def composition(w: np.ndarray, ok: np.ndarray, valid: np.ndarray,
                nt: np.ndarray) -> dict[str, np.ndarray]:
    """A_obs, p, A_pred (pooled), A_pred_lc and the residuals at one axis
    point, as WEIGHTED means over items.  See the module docstring.

    `w` is (k,) -- unit weights give the plain estimates -- or (B, k), one
    row per bootstrap replicate, in which case every output is (B,)."""
    one = np.ndim(w) == 1
    w = np.atleast_2d(np.asarray(w, float))                # (B, k)
    W = w.sum(1)
    okv = ok * valid                                       # (k, L)
    em = np.where(valid, ok, 1.0).prod(1)                  # E_i, (k,)
    A_obs = _wsum(w, em) / W
    p = _wsum(w, okv).sum(1) / _wsum(w, valid).sum(1)

    def marg(ww: np.ndarray) -> np.ndarray:                # (B, L)
        den = _wsum(ww, valid)
        with np.errstate(invalid="ignore", divide="ignore"):
            return np.where(den > 0, _wsum(ww, okv) / den, np.nan)

    m = marg(w)
    cm = np.concatenate([np.ones((len(w), 1)), np.cumprod(m, 1)], 1)
    # A length no drawn item has leaves its marginal NaN; 0 * NaN would
    # poison the sum for items that are not in the replicate, so mask them.
    A_pred = np.where(w > 0, w * cm[:, nt], 0.0).sum(1) / W

    A_lc = np.zeros(len(w))
    out: dict[str, np.ndarray] = {}
    for L in np.unique(nt):
        sel = (nt == L).astype(float)
        wL = w * sel
        nL = wL.sum(1)
        with np.errstate(invalid="ignore", divide="ignore"):
            prodL = np.prod(marg(wL)[:, :L], 1)
            all_L = _wsum(wL, em) / nL
        A_lc = A_lc + np.where(nL > 0, nL / W * prodL, 0.0)
        out[f"phi_l{L}"] = np.where(nL > 0, all_L - prodL, np.nan)
    res = {"p": p, "A_obs": A_obs, "A_pred": A_pred, "A_pred_lc": A_lc,
           "phi": A_obs - A_pred, "phi_lc": A_obs - A_lc, **out}
    return {key: (v[0] if one else v) for key, v in res.items()}


def _weights(rng: np.random.Generator, k: int, n: int,
             m: int | None) -> np.ndarray:
    """(n, k) cluster weights: multinomial counts for the bootstrap, or an
    m-hot draw without replacement for subsampling."""
    if m is None:
        return rng.multinomial(k, np.full(k, 1.0 / k), size=n).astype(float)
    if not 1 <= m <= k:
        raise ValueError(f"subsample size {m} outside [1, {k}]")
    idx = np.argsort(rng.random((n, k)), axis=1)[:, :m]
    w = np.zeros((n, k))
    np.put_along_axis(w, idx, 1.0, axis=1)
    return w


def axis_bootstrap(data: AxisData,
                   stat: Callable[[np.ndarray, AxisData], Mapping[str, Any]],
                   n_boot: int = 1000, seed: int = 0,
                   m: int | None = None, chunk: int = 256
                   ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Point value and replicates of any functional of the aligned table.

    `stat(W, data)` receives a (B, k_items) weight matrix -- each row one
    replicate, the same row applied at every x -- and returns a dict of
    arrays with leading dimension B.  The point value is `stat` at unit
    weights.  Replicates draw CLUSTERS (with replacement, or `m` without
    replacement) and expand them to their items.  Computed in chunks of
    `chunk` replicates to bound memory."""
    k = int(data.cluster.max()) + 1
    point = {key: np.asarray(v)[0] for key, v in
             stat(np.ones((1, len(data.items))), data).items()}
    rng = np.random.default_rng(seed)
    wc = _weights(rng, k, n_boot, m)
    parts = [stat(wc[a:a + chunk][:, data.cluster], data)
             for a in range(0, n_boot, chunk)]
    reps = {key: np.concatenate([np.asarray(pt[key]) for pt in parts])
            for key in point}
    return point, reps


def _curves(w: np.ndarray, data: AxisData,
            log_x: bool = True) -> dict[str, np.ndarray]:
    rows = [composition(w, o, v, n)
            for o, v, n in zip(data.ok, data.valid, data.nt, strict=True)]
    keys = sorted(set().union(*rows))
    nan = np.full(len(w), np.nan)
    out = {key: np.stack([r.get(key, nan) for r in rows], 1)   # (B, n_x)
           for key in keys}
    for a, c in (("A_obs", "chi_obs"), ("A_pred", "chi_pred"),
                 ("A_pred_lc", "chi_pred_lc")):
        out[c] = susceptibility(data.x, out[a], log=log_x)
    i = np.argmax(out["chi_obs"], 1)
    b = np.arange(len(w))
    out["peak_x"] = data.x[i]
    # chi at an end point is a one-sided difference -- for a single jump,
    # twice the centred value -- so a peak there is a grid artefact
    out["peak_at_edge"] = ((i == 0) | (i == len(data.x) - 1)).astype(float)
    out["peak_chi"] = out["chi_obs"][b, i]
    out["peak_chi_gap"] = out["chi_obs"][b, i] - out["chi_pred"][b, i]
    out["peak_chi_gap_lc"] = out["chi_obs"][b, i] - out["chi_pred_lc"][b, i]
    return out


@dataclass
class AxisCurves:
    """`table`: one row per axis point, each quantity with a percentile
    interval.  `peak`: the scalar functionals with intervals.  `reps`: the
    raw replicates, for anything not tabulated.  `caveats`: reasons the
    numbers are not to be read at face value (`curve_caveats`), also raised
    as one RuntimeWarning."""
    table: pd.DataFrame
    peak: pd.DataFrame
    reps: dict[str, np.ndarray] = field(repr=False)
    n_items: int = 0
    n_dropped: int = 0
    caveats: list[str] = field(default_factory=list)


def curve_caveats(k: int, peak: pd.DataFrame,
                  reps: Mapping[str, np.ndarray]) -> list[str]:
    """Reasons a curve's numbers are correct but not to be read at face value.

    * Few clusters (k items or item clusters): as for a mean
      (stats.few_clusters), the percentile interval is too narrow.
    * A censored crossing: replicates whose curve never reaches the level
      are left out of the crossing's interval, so above MAX_CENSORED_SHARE
      it describes only the replicates that crossed; if the observed curve
      itself never crosses, there is no point estimate.
    * A chi peak at an end of the grid: chi there is a one-sided difference
      and the true peak may lie outside the axis.
    """
    out = []
    few = few_clusters(k)
    if few:
        out.append(few)
    pk = peak.set_index("quantity")["point"]
    for name in pk.index:
        if not (name.startswith("crossing_") and not name.endswith("_censored")):
            continue
        if not np.isfinite(pk[name]):
            out.append(f"{name}: the observed curve never reaches the level")
            continue
        share = float(np.mean(~np.isfinite(reps[name])))
        if share > MAX_CENSORED_SHARE:
            out.append(f"{name}: {share:.0%} of replicates never cross; the "
                       f"interval describes only those that did")
    if pk.get("peak_at_edge", 0.0) == 1.0:
        out.append("chi peaks at an end of the grid: a one-sided difference, "
                   "and the true peak may lie outside the axis")
    return out


def _with_caveats(res: AxisCurves, k: int) -> AxisCurves:
    res.caveats = curve_caveats(k, res.peak, res.reps)
    if res.caveats:
        warnings.warn("curves carry caveats (see .caveats): "
                      + "; ".join(res.caveats), RuntimeWarning, stacklevel=3)
    return res


def axis_curves(df: pd.DataFrame, x: str, item: str = "item_id",
                cluster: str = "cluster", ok: str = OK, n_target: str = NT,
                n_boot: int = 1000, seed: int = 0,
                alpha: float = 0.05, log_x: bool = True) -> AxisCurves:
    """The whole analysis of one model axis in one call: A_obs, p, A_pred,
    A_pred_lc, phi, phi_lc, phi_l for each length, chi of each curve, and
    the peak functionals -- every one with a paired percentile interval.

    chi is taken against log x unless `log_x=False` (an axis that includes
    0, such as a checkpoint step, needs the linear form).  `peak_at_edge`
    is 1 when the chi peak sits on an end of the grid, where chi is a
    one-sided difference; its interval says how often that happens across
    replicates."""
    data = axis_data(df, x, item, cluster, ok, n_target)
    _require_positive(data.x, x, log_x)
    point, reps = axis_bootstrap(
        data, functools.partial(_curves, log_x=log_x), n_boot, seed)
    q = [alpha / 2, 1 - alpha / 2]
    table = pd.DataFrame({x: data.x})
    peak_rows = []
    for key in point:
        val = np.asarray(point[key], float)
        lo, hi = np.nanquantile(reps[key], q, axis=0)
        if val.ndim == 0:
            peak_rows.append({"quantity": key, "point": float(val),
                              "lo": float(lo), "hi": float(hi)})
        else:
            table[key], table[f"{key}_lo"], table[f"{key}_hi"] = val, lo, hi
    return _with_caveats(AxisCurves(table, pd.DataFrame(peak_rows), reps,
                                    len(data.items), data.n_dropped),
                         int(data.cluster.max()) + 1)


def _require_positive(xs: np.ndarray, name: str, log_x: bool) -> None:
    if log_x and (xs <= 0).any():
        raise ValueError(
            f"{name} has values <= 0, so log {name} is undefined; pass "
            f"log_x=False for a linear axis (e.g. one that includes step 0)")


def subsampled_sharpness(df: pd.DataFrame, x: str, ms: Sequence[int],
                         n_draw: int = 400, seed: int = 1,
                         item: str = "item_id", cluster: str = "cluster",
                         ok: str = OK, n_target: str = NT,
                         log_x: bool = True) -> pd.DataFrame:
    """Finite-size scaling of the apparent sharpness s_m = max_x chi(x) when
    only m CLUSTERS are scored (m items when every item is its own
    cluster).  Each draw takes m clusters without replacement, the SAME
    clusters at every x.  Returns mean and sd of s_m over draws for each m;
    fitting s_m ~ m^theta is left to the caller."""
    data = axis_data(df, x, item, cluster, ok, n_target)
    _require_positive(data.x, x, log_x)

    def sharp(w: np.ndarray, d: AxisData) -> dict[str, np.ndarray]:
        A = np.stack([composition(w, o, v, n)["A_obs"]
                      for o, v, n in zip(d.ok, d.valid, d.nt, strict=True)],
                     1)
        return {"s": susceptibility(d.x, A, log=log_x).max(1)}

    rows = []
    for m in ms:
        _, reps = axis_bootstrap(data, sharp, n_draw, seed, m=m)
        rows.append({"m": m, "s_mean": float(reps["s"].mean()),
                     "s_sd": float(reps["s"].std(ddof=1)), "n_draw": n_draw})
    return pd.DataFrame(rows)


# --- curves of any per-item value ---------------------------------------------

def crossing(x: ArrayLike, y: ArrayLike, level: float = 0.0,
             log: bool = True) -> np.ndarray:
    """First x at which the curve y(x) crosses `level`, by linear
    interpolation in u = log x (or x).  `y` is (n_x,) or (B, n_x); a curve
    that never reaches the level returns NaN (censored), never an end
    point."""
    u = np.log(np.asarray(x, float)) if log else np.asarray(x, float)
    y2 = np.atleast_2d(np.asarray(y, float)) - level
    out = np.full(len(y2), np.nan)
    s = np.sign(y2)
    for b in range(len(y2)):
        exact = np.flatnonzero(s[b] == 0)
        flip = np.flatnonzero(s[b, :-1] * s[b, 1:] < 0)
        j = min(float(np.min(exact)) if exact.size else np.inf,
                float(np.min(flip)) if flip.size else np.inf)
        if not np.isfinite(j):
            continue
        j = int(j)
        if s[b, j] == 0:
            ub = u[j]
        else:
            y0, y1 = y2[b, j], y2[b, j + 1]
            ub = u[j] + (u[j + 1] - u[j]) * (-y0) / (y1 - y0)
        out[b] = np.exp(ub) if log else ub
    return out if np.ndim(y) == 2 else out[0]


@dataclass
class ValueData:
    x: np.ndarray
    items: np.ndarray
    cluster: np.ndarray
    V: np.ndarray                       # (k items, n_x)
    n_dropped: int = 0


def value_data(df: pd.DataFrame, x: str, value: str, item: str = "item_id",
               cluster: str = "cluster") -> ValueData:
    """Items x axis matrix of one per-item value, under the same alignment
    rules as `axis_data` (every item at every x, or dropped and reported).
    Rows whose value is missing (errors, unparsed, truncated) are left out
    first, so an item missing its value anywhere is dropped and counted."""
    d = df
    if "status" in d.columns:
        d = d[d["status"] == "ok"]
    d = d[d[value].notna()]
    xs, per_x, items, codes, n_dropped = _align(d, x, item, cluster)
    V = np.stack([per_x[v].loc[items, value].to_numpy(float) for v in xs], 1)
    return ValueData(xs.astype(float), items, codes, V, n_dropped)


def value_curves(df: pd.DataFrame, x: str, value: str,
                 item: str = "item_id", cluster: str = "cluster",
                 levels: Sequence[float] = (), n_boot: int = 1000,
                 seed: int = 0, alpha: float = 0.05,
                 log_x: bool = True) -> AxisCurves:
    """The mean of `value` at each x, its chi = d mean / d log x, the peak
    functionals, and the first crossing of each level in `levels` -- every
    one with an item-paired percentile interval (see the module docstring).

    For a crossing, `crossing_<level>_censored` reports the share of
    replicates whose curve never reaches the level; a large share means
    the interval describes only the replicates that did cross."""
    data = value_data(df, x, value, item, cluster)
    _require_positive(data.x, x, log_x)
    k = int(data.cluster.max()) + 1
    rng = np.random.default_rng(seed)
    W = np.vstack([np.ones((1, k)), _weights(rng, k, n_boot, None)])
    Wi = W[:, data.cluster]                                   # (1+B, items)
    mean = _wsum(Wi, data.V) / Wi.sum(1, keepdims=True)       # (1+B, n_x)
    chi = susceptibility(data.x, mean, log=log_x)
    i = np.argmax(chi, 1)
    b = np.arange(len(W))
    scal = {"peak_x": data.x[i], "peak_chi": chi[b, i],
            "peak_at_edge": ((i == 0) | (i == len(data.x) - 1)).astype(float)}
    for lv in levels:
        scal[f"crossing_{lv:g}"] = crossing(data.x, mean, lv, log=log_x)
    q = [alpha / 2, 1 - alpha / 2]
    table = pd.DataFrame({x: data.x})
    for name, arr in (("mean", mean), ("chi", chi)):
        lo, hi = np.nanquantile(arr[1:], q, axis=0)
        table[name], table[f"{name}_lo"], table[f"{name}_hi"] = arr[0], lo, hi
    rows = []
    for name, arr in scal.items():
        reps = arr[1:]
        ok = np.isfinite(reps)
        lo, hi = (np.quantile(reps[ok], q) if ok.any() else (np.nan, np.nan))
        rows.append({"quantity": name, "point": float(arr[0]),
                     "lo": float(lo), "hi": float(hi)})
        if name.startswith("crossing_"):
            rows.append({"quantity": f"{name}_censored",
                         "point": float(~np.isfinite(arr[0])),
                         "lo": float(1 - ok.mean()), "hi": float(1 - ok.mean())})
    reps = {"mean": mean[1:], "chi": chi[1:],
            **{n: a[1:] for n, a in scal.items()}}
    return _with_caveats(AxisCurves(table, pd.DataFrame(rows), reps,
                                    len(data.items), data.n_dropped), k)
