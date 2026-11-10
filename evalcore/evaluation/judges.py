r"""What a panel of judges' verdicts mean: error rates, debiasing, agreement,
correlation between judges, and the majority verdict.

Every function reads a LONG frame of verdicts, one row per (answer, judge,
criterion):

    answer     the judged answer (the source run's unit_id)
    cluster    the source item, the resampling unit as everywhere else
    judge      the judge's name
    criterion  the rubric criterion
    verdict    1 = pass, 0 = fail (a parse failure is not a row here: it
               stays in the judge run's attrition, never a fail)
    reference  1 / 0 from an exact grader, where one exists (calibration)

1. Error rates.  Against a reference y in {0, 1}, a judge's verdict v has a
false-positive rate and a false-negative rate

    alpha = P(v = 1 | y = 0),   beta = P(v = 0 | y = 1).

Each is a ratio of sums over clusters c, e.g. alpha = sum_c FP_c / sum_c N_c
(N_c the cluster's reference negatives), so its variance is the
linearised (cluster-robust) ratio variance

    Var(alpha) = k/(k-1) sum_c (FP_c - alpha N_c)^2 / (sum_c N_c)^2,     (1)

which for one row per cluster is alpha(1-alpha)/(n-1): the binomial
variance with the n-1 of an estimated mean.

2. Debiasing.  The judge's pass rate on answers with true pass rate theta is

    m = E[v] = alpha (1 - theta) + (1 - beta) theta = alpha + lambda theta,
    lambda = 1 - alpha - beta,                                           (2)

so theta = (m - alpha) / lambda.  lambda is the judge's discrimination:
lambda = 1 is a perfect judge, lambda = 0 a coin that ignores the answer
(nothing about theta can be recovered), lambda < 0 a judge that inverts.
With m, alpha, beta estimated on DISJOINT samples (independent errors), the
delta method gives, from d theta/dm = 1/lambda, d theta/d alpha =
(theta - 1)/lambda and d theta/d beta = theta/lambda,

    Var(theta) = [Var(m) + (1 - theta)^2 Var(alpha) + theta^2 Var(beta)]
                 / lambda^2.                                             (3)

The first term alone is the attenuation: a judge with discrimination lambda
turns n answers into n lambda^2 effective ones.  The other two are the
price of not knowing alpha and beta exactly.  The assumption that carries
the weight is transfer: alpha and beta measured on the calibration set hold
on the answers being scored.  That is a claim about the data, not checked
here; the disjointness of the two sets is.

3. Agreement between two judges on the same answers.  Raw agreement p_o
hides base rates, so Cohen's kappa sits next to it,

    kappa = (p_o - p_e) / (1 - p_e),   p_e = p_1 p_2 + (1 - p_1)(1 - p_2),

with p_j judge j's pass rate.  Kappa does not say which judge is more
lenient.  McNemar on the discordant pairs (b: 1 passes, 2 fails; c: the
reverse) does, so both are reported.

4. How many judges a panel is worth.  With error indicators e_j = 1[v_j !=
y] and pairwise correlation rho between judges' errors, the mean of J
verdicts has the design effect of eq. (3) in stats.py with m -> J:

    n_eff = J / (1 + (J - 1) rho_bar).                                   (4)

Two judges from one model family share most of their errors (rho near 1,
n_eff near 1).  rho is measured against the reference, not assumed.  Eq.
(4) is for the mean of the verdicts.  The majority vote's own error rates
are measured directly by calibrating the panel column as one more judge.
"""

from __future__ import annotations

import itertools
from collections.abc import Collection, Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sps

from .stats import cluster_bootstrap, mcnemar_exact

__all__ = ["agreement", "debias", "debiased_mean",
           "error_correlation", "judge_error_rates", "majority", "ratio_se"]

KEYS = ("answer", "criterion")


def _require(df: pd.DataFrame, cols: Sequence[str], where: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{where}: missing column(s) {missing}")
    for c in cols:
        if df[c].isna().any():
            raise ValueError(f"{where}: column {c!r} has missing values; "
                             f"drop or account for those rows first")


def _binary(df: pd.DataFrame, col: str, where: str) -> np.ndarray:
    x = df[col].to_numpy(dtype=float)
    if not np.isin(x, (0.0, 1.0)).all():
        raise ValueError(f"{where}: {col!r} must be 0 or 1 in every row")
    return x


def ratio_se(num: np.ndarray, den: np.ndarray) -> float:
    """Cluster-robust standard error of sum(num)/sum(den), eq. (1).  One
    entry per cluster.  NaN with fewer than two clusters."""
    num, den = np.asarray(num, float), np.asarray(den, float)
    k, total = len(num), den.sum()
    if k < 2 or total == 0:
        return float("nan")
    r = num.sum() / total
    return float(np.sqrt(k / (k - 1) * ((num - r * den) ** 2).sum()) / total)


def judge_error_rates(df: pd.DataFrame, by: Sequence[str] = ("judge", "criterion"),
              verdict: str = "verdict", reference: str = "reference",
              cluster: str = "cluster") -> pd.DataFrame:
    """The 2x2 confusion matrix against the reference, alpha, beta and
    lambda = 1 - alpha - beta, with cluster-robust standard errors (eq. 1),
    one row per group in `by`."""
    where = "judge_error_rates"
    _require(df, [*by, verdict, reference, cluster], where)
    v, y = _binary(df, verdict, where), _binary(df, reference, where)
    d = df.assign(_tp=v * y, _fn=(1 - v) * y, _fp=v * (1 - y),
                  _tn=(1 - v) * (1 - y), _pos=y, _neg=1 - y)
    out = []
    for key, g in d.groupby(list(by), sort=True):
        per = g.groupby(cluster)[["_tp", "_fn", "_fp", "_tn", "_pos",
                                  "_neg"]].sum()
        tp, fn, fp, tn = (int(per[c].sum()) for c in ("_tp", "_fn", "_fp",
                                                      "_tn"))
        alpha = fp / (fp + tn) if fp + tn else float("nan")
        beta = fn / (fn + tp) if fn + tp else float("nan")
        out.append({**dict(zip(by, key if isinstance(key, tuple) else (key,),
                               strict=True)),
                    "tp": tp, "fn": fn, "fp": fp, "tn": tn,
                    "n_pos": tp + fn, "n_neg": fp + tn, "k": len(per),
                    "alpha": alpha, "beta": beta,
                    "lam": 1 - alpha - beta,
                    "se_alpha": ratio_se(per["_fp"], per["_neg"]),
                    "se_beta": ratio_se(per["_fn"], per["_pos"])})
    return pd.DataFrame(out)


def debias(m: float, se_m: float, alpha: float, se_alpha: float,
           beta: float, se_beta: float, ci: float = 0.95) -> dict[str, float]:
    """theta = (m - alpha)/lambda with the variance of eq. (3) and a Wald
    interval.  Refuses lambda <= 0: such a judge carries no information
    about theta, or inverts it.  `attenuation` is lambda^2, the factor by
    which the judge shrinks the effective sample size."""
    lam = 1.0 - alpha - beta
    if not np.isfinite(lam) or lam <= 0:
        raise ValueError(f"lambda = 1 - alpha - beta = {lam:.3g}: the judge "
                         f"does not discriminate, so theta is not "
                         f"recoverable from its verdicts")
    theta = (m - alpha) / lam
    var = (se_m ** 2 + (1 - theta) ** 2 * se_alpha ** 2
           + theta ** 2 * se_beta ** 2) / lam ** 2
    se = float(np.sqrt(var))
    z = float(sps.norm.ppf(0.5 + ci / 2))
    return {"theta": float(theta), "se": se, "lo": float(theta - z * se),
            "hi": float(theta + z * se), "lam": float(lam),
            "attenuation": float(lam ** 2), "m": float(m)}


def debiased_mean(df: pd.DataFrame, cal: Mapping[str, Any],
                  cal_clusters: Collection[str], verdict: str = "verdict",
                  cluster: str = "cluster", n_boot: int = 10_000,
                  seed: int = 0) -> dict[str, float]:
    """The debiased pass rate of one judge on one criterion over `df`, with
    `cal` a row of `judge_error_rates` for that judge and criterion.  m and its se
    come from the cluster bootstrap.  Refuses any cluster shared with the
    calibration set: eq. (3) assumes independent errors in m and in
    (alpha, beta), and a shared item also means the reference was there."""
    _require(df, [verdict, cluster], "debiased_mean")
    shared = set(df[cluster]) & set(cal_clusters)
    if shared:
        raise ValueError(f"debiased_mean: {len(shared)} cluster(s) are in "
                         f"both the calibration and the scored set, e.g. "
                         f"{sorted(shared)[:3]}")
    for col in ("judge", "criterion"):
        if col in df.columns and col in cal and set(df[col]) != {cal[col]}:
            raise ValueError(f"debiased_mean: df has {col} "
                             f"{sorted(set(df[col]))}, calibration is for "
                             f"{cal[col]!r}")
    b = cluster_bootstrap(df, verdict, cluster, n_boot=n_boot, seed=seed)
    return {**debias(b.point, b.se, cal["alpha"], cal["se_alpha"],
                     cal["beta"], cal["se_beta"]),
            "n_clusters": b.n_clusters, "n_rows": b.n_rows}


def _wide(df: pd.DataFrame, value: str, where: str) -> pd.DataFrame:
    """One row per (answer, criterion), one column per judge.  Answers that
    not every judge scored are dropped from pairwise statistics only after
    being counted (n in each output row)."""
    _require(df, [*KEYS, "judge", value], where)
    dup = df.duplicated([*KEYS, "judge"])
    if dup.any():
        raise ValueError(f"{where}: {int(dup.sum())} duplicate (answer, "
                         f"criterion, judge) rows")
    return df.pivot(index=list(KEYS), columns="judge", values=value)


def agreement(df: pd.DataFrame, verdict: str = "verdict",
              by: Sequence[str] = ("criterion",)) -> pd.DataFrame:
    """For every pair of judges, per group in `by`: the answers both
    scored (n), raw agreement, Cohen's kappa, the discordant counts b and c
    and the exact McNemar p-value (section 3)."""
    _binary(df, verdict, "agreement")
    out = []
    for key, g in df.groupby(list(by), sort=True):
        w = _wide(g, verdict, "agreement")
        for j1, j2 in itertools.combinations(sorted(w.columns), 2):
            pair = w[[j1, j2]].dropna()
            a, b_ = pair[j1].to_numpy(), pair[j2].to_numpy()
            n = len(pair)
            p_o = float((a == b_).mean()) if n else float("nan")
            p1, p2 = (float(a.mean()), float(b_.mean())) if n else (
                float("nan"), float("nan"))
            p_e = p1 * p2 + (1 - p1) * (1 - p2)
            kappa = (p_o - p_e) / (1 - p_e) if n and p_e < 1 else float("nan")
            b = int(((a == 1) & (b_ == 0)).sum())
            c = int(((a == 0) & (b_ == 1)).sum())
            out.append({**dict(zip(by, key if isinstance(key, tuple)
                                   else (key,), strict=True)),
                        "judge_1": j1, "judge_2": j2, "n": n,
                        "agreement": p_o, "kappa": kappa,
                        "pass_1": p1, "pass_2": p2, "b": b, "c": c,
                        "mcnemar_p": mcnemar_exact(b, c)})
    return pd.DataFrame(out)


def error_correlation(df: pd.DataFrame, verdict: str = "verdict",
                      reference: str = "reference") -> dict[str, Any]:
    """Pairwise correlation of the judges' errors e_j = 1[v_j != y] over
    the answers every judge scored, their mean rho_bar, and the panel's
    effective number of judges, eq. (4).  A judge with no errors (or only
    errors) has an undefined correlation, reported as NaN."""
    _binary(df, verdict, "error_correlation")
    _binary(df, reference, "error_correlation")
    err = df.assign(_e=(df[verdict] != df[reference]).astype(float))
    w = _wide(err, "_e", "error_correlation").dropna()
    judges = sorted(w.columns)
    if len(judges) < 2:
        raise ValueError("error_correlation needs at least two judges")
    pairs = {}
    for j1, j2 in itertools.combinations(judges, 2):
        x, y = w[j1].to_numpy(), w[j2].to_numpy()
        ok = x.std() > 0 and y.std() > 0
        pairs[(j1, j2)] = float(np.corrcoef(x, y)[0, 1]) if ok else float("nan")
    vals = np.array(list(pairs.values()))
    rho = float(vals.mean()) if np.isfinite(vals).all() else float("nan")
    n_j = len(judges)
    return {"judges": judges, "n": len(w), "pairs": pairs,
            "error_rates": {j: float(w[j].mean()) for j in judges},
            "rho_bar": rho, "n_eff": n_j / (1 + (n_j - 1) * rho)}


def majority(df: pd.DataFrame, judges: Sequence[str],
             verdict: str = "verdict") -> pd.DataFrame:
    """The panel verdict per (answer, criterion): the majority of `judges`,
    an odd number, only where every one of them returned a verdict.  An
    answer some judge left unscored gets NaN and status "incomplete",
    counted rather than decided by the judges that remain (with two left, a
    tie has no majority)."""
    if len(judges) % 2 == 0 or len(set(judges)) != len(judges):
        raise ValueError(f"majority needs an odd number of distinct judges, "
                         f"got {list(judges)}")
    _binary(df, verdict, "majority")
    unknown = sorted(set(df["judge"]) - set(judges))
    if unknown:
        raise ValueError(f"majority: verdicts from judges not on the panel "
                         f"{unknown}")
    w = _wide(df, verdict, "majority").reindex(columns=list(judges))
    complete = w.notna().all(axis=1)
    votes = w.sum(axis=1, min_count=len(judges))
    out = pd.DataFrame({"n_judges": w.notna().sum(axis=1),
                        "panel": np.where(complete,
                                          (votes > len(judges) / 2).astype(float),
                                          np.nan),
                        "status": np.where(complete, "ok", "incomplete")},
                       index=w.index).reset_index()
    extra = [c for c in ("cluster", "reference") if c in df.columns]
    if extra:
        first = df.drop_duplicates(list(KEYS)).set_index(list(KEYS))[extra]
        out = out.join(first, on=list(KEYS))
    return out
