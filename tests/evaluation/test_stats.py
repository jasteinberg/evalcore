"""Calibration tests, hand-computed cases and invariances of the cluster
statistics.  A CI routine that is never checked against a known generating
process is decoration, and the coverage tests check that.  The rest take
their expected values from closed forms (exact fractions where possible),
tiny hand-computed tables, or two paths that must agree.  A row of the
invariant catalogue (C1, C2, ...) is named first in its docstring, and a
newly found violation lands as a strict xfail that must fail, for the
stated reason, until the defect is fixed."""

import warnings
from fractions import Fraction

import numpy as np
import pandas as pd
import pytest

from evalcore import (
    attrition,
    benjamini_hochberg,
    cluster_bootstrap,
    effective_n,
    holm,
    icc1,
    paired_bootstrap,
    wilson_interval,
)
from evalcore.evaluation.stats import (
    SMALL_K_COVERAGE,
    BootResult,
    cell_caveats,
    mcnemar_exact,
    paired_pvalue,
    summarize,
)
from evalcore.testing import assert_same

# small fixtures by design, and the caveats themselves are tested below
tiny = pytest.mark.filterwarnings("ignore:.*carry caveats:RuntimeWarning")


def clustered(k=60, m=5, s2_a=0.6, s2_e=0.4, mu=0.0, rng=None):
    """X_ij = mu + a_i + e_ij, one row per (cluster, repeat)."""
    rng = rng or np.random.default_rng(0)
    a = rng.normal(0, np.sqrt(s2_a), size=k)
    e = rng.normal(0, np.sqrt(s2_e), size=(k, m))
    x = mu + a[:, None] + e
    return pd.DataFrame({"cluster": np.repeat(np.arange(k), m),
                         "y": x.ravel()})


def unbalanced(sizes, seed=0):
    """Rows y_ij = a_i + e_ij, sd(e) = 0.5, for clusters of the given
    sizes, condition "a".  (Draws a_i then that cluster's e_ij, cluster by
    cluster, which `clustered` cannot do for unequal sizes.)"""
    rng = np.random.default_rng(seed)
    rows = []
    for i, m in enumerate(sizes):
        a = rng.normal()
        rows += [{"condition": "a", "cluster": f"c{i:03d}", "y": a + rng.normal(0, .5)}
                 for _ in range(m)]
    return pd.DataFrame(rows)


def _ci(res):
    return {"point": res.point, "lo": res.lo, "hi": res.hi, "se": res.se}


def test_icc_recovers_rho():
    s2_a, s2_e, m = 0.6, 0.4, 8
    rho_true = s2_a / (s2_a + s2_e)
    est = [icc1(clustered(k=400, m=m, s2_a=s2_a, s2_e=s2_e,
                          rng=np.random.default_rng(s)), "y")["rho"]
           for s in range(8)]
    # One estimate at k=400, m=8 has sd ~0.027 over seeds (Var(rho_hat) ~
    # 2(1-rho)^2 (1+(m-1)rho)^2 / (m(m-1)(k-1)) gives 0.020, but the
    # empirical spread is larger).  The mean of 8 then has SE ~0.0096: 3 SE.
    assert abs(np.mean(est) - rho_true) < 0.03
    o = effective_n(clustered(k=400, m=m, s2_a=s2_a, s2_e=s2_e), "y")
    assert abs(o["m0"] - m) < 1e-9                       # balanced design
    assert abs(o["deff"] - (1 + (m - 1) * rho_true)) < 0.25
    assert o["n_eff"] < o["N"] / 3                       # eq. (3) bites


def test_deff_endpoints():
    """rho -> 1: augmentation buys nothing (n_eff ~ k).
       rho -> 0: augmentation buys everything (n_eff ~ N)."""
    hi = effective_n(clustered(k=200, m=6, s2_a=1.0, s2_e=1e-4), "y")
    lo = effective_n(clustered(k=200, m=6, s2_a=1e-4, s2_e=1.0), "y")
    # rho = 1/(1 + 1e-4): n_eff = N/(1 + 5 rho) = 200.02, and rho_hat has
    # error ~1e-4 here, so 1e-3 is loose by an order of magnitude already.
    assert hi["n_eff"] == pytest.approx(200, rel=1e-3)
    # rho ~ 1e-4, but rho_hat has SE ~0.018 at k=200, m=6 and is clipped at
    # 0, so it sits about one SE high.  rel=0.25 admits rho_hat up to
    # 0.067, i.e. ~3.7 SE.
    assert lo["n_eff"] == pytest.approx(1200, rel=0.25)


def _naive_row_ci(df, value, n_boot=800, seed=0, alpha=0.05):
    """The wrong thing, kept as a control: resample ROWS."""
    v = df[value].to_numpy(float)
    rng = np.random.default_rng(seed)
    reps = v[rng.integers(0, v.size, size=(n_boot, v.size))].mean(axis=1)
    return np.quantile(reps, [alpha / 2, 1 - alpha / 2])


def _coverage(R, draw, estimate, truth):
    hits = 0
    for s in range(R):
        r = estimate(draw(s), s)
        hits += r.lo <= truth <= r.hi
    return hits / R


def test_bca_cluster_bootstrap_covers():
    """`summarize` defaults to BCa, so BCa is the estimator that has to be
    calibrated, not only the percentile method.  R=200: binomial SE of the
    coverage at 0.95 is 0.015, so the band is ~ -3/+2.7 SE."""
    cov = _coverage(
        200, lambda s: clustered(k=40, m=6, mu=0.3,
                                 rng=np.random.default_rng(1000 + s)),
        lambda df, s: cluster_bootstrap(df, "y", n_boot=800, seed=s,
                                        method="bca"), 0.3)
    assert 0.905 <= cov <= 0.99, f"BCa coverage {cov:.3f}"


def test_paired_bootstrap_covers_the_true_difference():
    """A_ij = a_i + delta + e, B_ij = a_i + e': shared difficulty a_i, three
    repeats per cluster.  Same band and R as the BCa test above."""
    k, delta = 40, 0.25

    def draw(s):
        rng = np.random.default_rng(5000 + s)
        a = rng.normal(0, 1, k)[:, None]
        ya = a + delta + rng.normal(0, 0.5, (k, 3))
        yb = a + rng.normal(0, 0.5, (k, 3))
        cl = np.repeat(np.arange(k), 3)
        return pd.DataFrame({"cluster": np.r_[cl, cl],
                             "condition": ["a"] * 3 * k + ["b"] * 3 * k,
                             "y": np.r_[ya.ravel(), yb.ravel()]})

    cov = _coverage(200, draw, lambda df, s: paired_bootstrap(
        df, "y", "condition", "a", "b", n_boot=800, seed=s), delta)
    assert 0.905 <= cov <= 0.99, f"paired coverage {cov:.3f}"


def test_cluster_bootstrap_covers_and_row_bootstrap_does_not():
    """95% nominal.  Cluster bootstrap should land near 0.95.  The row
    bootstrap should under-cover badly, by the factor DEFF in eq. (3)."""
    R, mu = 120, 0.3
    cov_c = cov_r = 0
    for s in range(R):
        df = clustered(k=40, m=6, s2_a=0.6, s2_e=0.4, mu=mu,
                       rng=np.random.default_rng(1000 + s))
        res = cluster_bootstrap(df, "y", n_boot=800, seed=s,
                                method="percentile")
        cov_c += res.lo <= mu <= res.hi
        lo, hi = _naive_row_ci(df, "y", seed=s)
        cov_r += lo <= mu <= hi
    assert 0.90 <= cov_c / R <= 0.99, f"cluster coverage {cov_c/R:.3f}"
    assert cov_r / R < 0.80, f"row coverage {cov_r/R:.3f} -- too good, check test"


def test_paired_beats_unpaired_when_items_correlate():
    rng = np.random.default_rng(7)
    k = 80
    difficulty = rng.normal(0, 1.0, size=k)          # shared across conditions
    a = difficulty + rng.normal(0.25, 0.2, size=k)
    b = difficulty + rng.normal(0.00, 0.2, size=k)
    df = pd.DataFrame({"cluster": np.r_[np.arange(k), np.arange(k)],
                       "condition": ["a"] * k + ["b"] * k, "y": np.r_[a, b]})
    pair = paired_bootstrap(df, "y", "condition", "a", "b", n_boot=2000)
    ua = cluster_bootstrap(df[df.condition == "a"], "y", n_boot=2000)
    ub = cluster_bootstrap(df[df.condition == "b"], "y", n_boot=2000)
    unpaired_se = np.hypot(ua.se, ub.se)
    assert pair.se < unpaired_se / 3                  # eq. (4), r ~ 0.98
    assert pair.lo > 0                                # and it is resolvable
    assert ua.lo < ub.hi                              # while the CIs overlap


def test_wilson_does_not_collapse_at_the_boundary():
    z2 = 1.959963984540054 ** 2
    # at phat = 0 the score equation p^2 = z^2 p(1-p)/n has roots 0 and
    # z^2/(n + z^2) = 0.1135, where Wald would give [0, 0]
    lo, hi = wilson_interval(0, 30)
    assert lo == 0.0 and hi == pytest.approx(z2 / (30 + z2), rel=1e-9)
    lo, hi = wilson_interval(30, 30)
    assert hi == pytest.approx(1.0) and lo == pytest.approx(30 / (30 + z2))


@pytest.mark.parametrize("k,n", [(0, 30), (30, 30), (7, 20), (15, 30)])
def test_wilson_endpoints_solve_the_score_equation(k, n):
    """The interval is defined as the set of p the score test accepts, so
    each endpoint must satisfy (phat - p)^2 = z^2 p(1-p)/n.  This checks the
    definition, not the closed form the implementation uses."""
    z2, ph = 1.959963984540054 ** 2, k / n
    for p in wilson_interval(k, n):
        assert (ph - p) ** 2 == pytest.approx(z2 * p * (1 - p) / n, abs=1e-12)


def test_multiple_comparisons_exact_rejection_sets():
    """By hand, m = 5, level 0.05.
    BH thresholds i q/m = .01 .02 .03 .04 .05: p_(1..3) pass, so reject 3.
    Holm thresholds a/(m-r) = .01 .0125 ...: .001 passes, .02 > .0125 stops."""
    p = [0.03, 0.001, 0.9, 0.02, 0.5]                # unsorted on purpose
    assert benjamini_hochberg(p, 0.05).tolist() == [True, True, False,
                                                    True, False]
    assert holm(p, 0.05).tolist() == [False, True, False, False, False]
    assert holm([0.9] * 5).sum() == 0
    with pytest.raises(ValueError, match="NaN"):     # was [True, True]
        holm([0.001, float("nan")])


def test_bh_is_step_up_not_step_down():
    """p_(1) = .040 > q/m = .01, so a step-DOWN procedure rejects nothing.
    BH takes the LARGEST i with p_(i) <= iq/m, which is i = 5 (.044 <= .05),
    and rejects all five."""
    p = [0.044, 0.040, 0.043, 0.041, 0.042]
    assert benjamini_hochberg(p, 0.05).all()
    assert not holm(p, 0.05).any()


def test_bca_matches_percentile_for_a_symmetric_mean():
    df = clustered(k=150, m=3, rng=np.random.default_rng(3))
    p = cluster_bootstrap(df, "y", n_boot=4000, seed=1, method="percentile")
    b = cluster_bootstrap(df, "y", n_boot=4000, seed=1, method="bca")
    # Same replicates, so the bounds differ only through z0 and a.  For a
    # symmetric mean a ~ 0, and z0 is Monte Carlo noise of sd
    # sqrt(.25/B)/phi(0) ~ 0.02 at B = 4000, which shifts each bound by
    # ~2 z0 se ~ 0.003 (se ~ 0.073).  0.01 is ~3 sd of that shift.
    assert abs(p.lo - b.lo) < 0.01 and abs(p.hi - b.hi) < 0.01


def test_cluster_bootstrap_needs_two_clusters():
    df = pd.DataFrame({"cluster": ["a"] * 5, "y": [1.0, 2, 3, 4, 5]})
    with pytest.raises(ValueError):
        cluster_bootstrap(df, "y")


def test_no_replication_gives_deff_one():
    """m = 1 per cluster: rho is unidentifiable, but DEFF is exactly 1 and
    n_eff must be N -- the most common design of all must not report NaN."""
    df = pd.DataFrame({"cluster": [f"i{j}" for j in range(30)],
                       "y": np.random.default_rng(0).normal(size=30)})
    o = effective_n(df, "y")
    assert o["deff"] == 1.0 and o["n_eff"] == 30.0
    assert not np.isfinite(o["rho"])


# --- caveats: correct numbers that are not to be read at face value ------------

def frame_of(k, m=4, const=False, missing=0, seed=0):
    rng = np.random.default_rng(seed)
    df = clustered(k=k, m=m, rng=rng)
    df["condition"] = "a"
    if const:
        df["y"] = 1.0
    if missing:
        df.loc[df.index[:missing], "y"] = np.nan
    return df


def test_few_clusters_are_flagged_with_the_measured_undercoverage():
    with pytest.warns(RuntimeWarning, match="1 of 1 cells carry caveats"):
        s = summarize(frame_of(3), "y", ["condition"], n_boot=200)
    assert "only 3 clusters" in s.iloc[0]["caveats"]
    assert "covers ~69% at k=3" in s.iloc[0]["caveats"]


def test_a_well_resolved_cell_has_no_caveats_and_no_warning(recwarn):
    s = summarize(frame_of(40), "y", ["condition"], n_boot=200)
    assert s.iloc[0]["caveats"] == ""
    assert not [w for w in recwarn if "caveats" in str(w.message)]


def test_no_spread_and_many_excluded_are_flagged():
    with pytest.warns(RuntimeWarning):
        s = summarize(frame_of(40, const=True), "y", ["condition"], n_boot=100)
    assert "no spread" in s.iloc[0]["caveats"]
    with pytest.warns(RuntimeWarning):
        s = summarize(frame_of(40, missing=40), "y", ["condition"], n_boot=100)
    assert "25% of rows have no value" in s.iloc[0]["caveats"]   # 40 of 160


def test_only_the_affected_cells_are_flagged():
    df = pd.concat([frame_of(40).assign(condition="big"),
                    frame_of(5, seed=1).assign(condition="small")])
    with pytest.warns(RuntimeWarning, match="1 of 2 cells"):
        s = summarize(df, "y", ["condition"], n_boot=100).set_index("condition")
    assert s.loc["big", "caveats"] == ""
    assert "only 5 clusters" in s.loc["small", "caveats"]


def test_the_quoted_undercoverage_is_what_the_bootstrap_does():
    """The caveat quotes SMALL_K_COVERAGE.  Check its k=5 entry against a
    fresh simulation (R=200: binomial SE ~0.027 near 0.83)."""
    R, mu, hits = 200, 0.3, 0
    for s in range(R):
        df = clustered(k=5, m=6, mu=mu, rng=np.random.default_rng(50_000 + s))
        r = cluster_bootstrap(df, "y", n_boot=400, seed=s)
        hits += r.lo <= mu <= r.hi
    assert abs(hits / R - SMALL_K_COVERAGE[5]) < 0.07
    assert cell_caveats({"n_clusters": 30, "se": 0.1, "n_rows": 10,
                         "n_excluded": 0}) == []


# --- accounting: every row and every cluster is counted somewhere -------------

def test_c2_unpaired_clusters_are_counted_and_named_in_a_warning():
    """C2.  10 shared clusters (one row per condition each: 20 rows) plus 2 seen
    only by condition a and 1 seen only by condition b.  The difference is taken on the
    10 shared clusters, so n_clusters = 10, n_rows = 20, n_unpaired = 3 and
    the warning names 3.  A balanced frame has n_unpaired = 0 and no
    warning."""
    rows = [{"cluster": f"c{i}", "condition": c, "y": float(i + (c == "a"))}
            for i in range(10) for c in ("a", "b")]
    balanced = pd.DataFrame(rows)
    rows += [{"cluster": f"u{i}", "condition": "a", "y": 0.0} for i in range(2)]
    rows += [{"cluster": "v0", "condition": "b", "y": 0.0}]
    with pytest.warns(RuntimeWarning, match=r"^3 cluster\(s\) seen by only one"):
        res = paired_bootstrap(pd.DataFrame(rows), "y", "condition", "a", "b",
                               n_boot=200)
    assert (res.n_unpaired, res.n_rows, res.n_clusters) == (3, 20, 10)
    assert res.point == 1.0                      # every D_i = (i + 1) - i
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        ok = paired_bootstrap(balanced, "y", "condition", "a", "b", n_boot=200)
    assert ok.n_unpaired == 0


def test_c1_a_row_without_a_cluster_is_refused_by_every_bootstrap():
    """C1.  A row with a value and no cluster label cannot enter the
    bootstrap, and dropping it would leave it in neither n_rows nor
    n_excluded.  3 of 20 rows carry a value and no cluster:
    cluster_bootstrap, paired_bootstrap and summarize all refuse, naming
    the count.  A row with neither value nor cluster is only an excluded
    row and passes."""
    df = pd.DataFrame({"cluster": [f"c{i // 2}" for i in range(20)],
                       "condition": ["a", "b"] * 10, "y": np.arange(20.0)})
    bad = df.copy()
    bad.loc[[0, 5, 9], "cluster"] = None
    for call in (lambda d: cluster_bootstrap(d, "y", n_boot=50),
                 lambda d: paired_bootstrap(d, "y", "condition", "a", "b", n_boot=50),
                 lambda d: summarize(d, "y", ["condition"], n_boot=50)):
        with pytest.raises(ValueError, match=r"^3 row"):
            call(bad)
    fine = df.copy()
    fine.loc[[0], ["cluster", "y"]] = None
    assert cluster_bootstrap(fine, "y", n_boot=50).n_rows == 19


def test_c3_attrition_refuses_an_unknown_status():
    """C3.  Statuses are ok / unparsed / truncated / tool_error / error.  A
    row with status "refused" belongs to none of the six states, so
    attrition must refuse it, with and without a manifest."""
    df = pd.DataFrame({"unit_id": ["u0", "u1"], "condition": ["a", "a"],
                       "status": ["ok", "refused"]})
    plan = df[["unit_id", "condition"]]
    with pytest.raises(ValueError):
        attrition(df, ["condition"])
    with pytest.raises(ValueError):
        attrition(df, ["condition"], manifest=plan)


# --- the estimator: unweighted mean of cluster means --------------------------

@tiny
def test_c4_duplicating_rows_within_a_cluster_changes_nothing():
    """C4.  theta = (1/k) sum_i xbar_i depends on each cluster only through
    its mean, and the bootstrap resamples cluster labels, so repeating a
    cluster's rows leaves the point, the replicates and so the interval
    unchanged.  Tiny case: {c0: [0], c1: [1]} and {c0: [0, 0, 0], c1: [1]}
    both give theta = (0 + 1)/2 = 1/2 (the row mean would give 1/4).
    Larger case: dyadic values (multiples of 1/8), so every sum is exact and
    the two paths must agree to the bit (tolerance 0)."""
    tiny = pd.DataFrame({"cluster": ["c0", "c1"], "y": [0.0, 1.0]})
    dup = pd.DataFrame({"cluster": ["c0"] * 3 + ["c1"], "y": [0.0] * 3 + [1.0]})
    assert cluster_bootstrap(tiny, "y", n_boot=200).point == 0.5
    assert cluster_bootstrap(dup, "y", n_boot=200).point == 0.5

    rng = np.random.default_rng(4)
    base = pd.DataFrame({"condition": "a",
                         "cluster": np.repeat([f"c{i:02d}" for i in range(24)], 2),
                         "y": rng.integers(0, 8, size=48) / 8})
    heavy = base[base.cluster.isin(["c00", "c03", "c07", "c11", "c19"])]
    more = pd.concat([base, heavy, heavy], ignore_index=True)   # x3 each
    theta = sum(Fraction(float(g.mean())) for _, g in base.groupby("cluster").y) / 24
    a = cluster_bootstrap(base, "y", n_boot=2000, seed=3)
    b = cluster_bootstrap(more, "y", n_boot=2000, seed=3)
    assert a.point == float(theta)
    assert_same(_ci(b), _ci(a), what="duplicated vs original")
    sa = summarize(base, "y", ["condition"], n_boot=2000, seed=3)
    sb = summarize(more, "y", ["condition"], n_boot=2000, seed=3)
    cols = ["point", "lo", "hi", "se", "n_clusters"]
    assert_same(sb[cols], sa[cols], what="summarize duplicated vs original")


def test_c7_paired_bootstrap_refuses_an_unknown_method():
    """C7.  cluster_bootstrap refuses method="bogus", and the paired form
    must too, or a typo silently changes the interval."""
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"cluster": np.repeat(np.arange(10), 2),
                       "condition": ["a", "b"] * 10, "y": rng.normal(size=20)})
    with pytest.raises(ValueError, match="bogus"):
        paired_bootstrap(df, "y", "condition", "a", "b", n_boot=100, method="bogus")


# --- ICC(1), m0, DEFF, n_eff ----------------------------------------------------

def test_c12_unbalanced_design_effect_by_hand():
    """C12.  Clusters [1], [0, 2], [4, 4, 7]: m = (1, 2, 3), k = 3, N = 6.

        means 1, 1, 5;  grand mean (1 + 2 + 15)/6 = 3
        MS_b = [1(1-3)^2 + 2(1-3)^2 + 3(5-3)^2] / (k-1) = (4 + 8 + 12)/2 = 12
        SS_w = (0-1)^2 + (2-1)^2 + 2(4-5)^2 + (7-5)^2 = 8,  MS_w = 8/3
        m0   = (N - sum m^2 / N)/(k-1) = (6 - 14/6)/2 = 11/6
        s2_a = (12 - 8/3)/(11/6) = 56/11
        rho  = (56/11)/(56/11 + 8/3) = 21/32
        DEFF = 1 + (11/6 - 1)(21/32) = 99/64,   n_eff = 6/(99/64) = 128/33.

    The naive m = N/k = 2 would give DEFF = 1 + 21/32 = 53/32 instead.
    Tolerance 1e-12 relative: a few float operations on O(10) numbers."""
    df = pd.DataFrame({"cluster": ["a", "b", "b", "c", "c", "c"],
                       "y": [1.0, 0, 2, 4, 4, 7]})
    o = effective_n(df, "y")
    expect = {"k": 3, "N": 6, "ms_between": 12.0, "ms_within": float(Fraction(8, 3)),
              "m0": float(Fraction(11, 6)), "rho": float(Fraction(21, 32)),
              "deff": float(Fraction(99, 64)), "n_eff": float(Fraction(128, 33))}
    assert_same({key: o[key] for key in expect}, expect, rtol=1e-12)


def test_c13_negative_between_variance_is_clipped():
    """C13.  Clusters [0, 2], [0, 2]: means 1, 1, so MS_b = 0.  Then
    MS_w = 4 (1^2)/(N - k) = 4/2 = 2, m0 = (4 - 8/4)/1 = 2 and
    s2_a = (0 - 2)/2 = -1 < 0.  The moment estimate is negative, so rho is
    clipped to 0, DEFF = 1 and n_eff = N = 4 (unclipped: rho = -1,
    DEFF = 0, n_eff infinite).  Exact: every quantity is a small integer."""
    df = pd.DataFrame({"cluster": ["a", "a", "b", "b"], "y": [0.0, 2, 0, 2]})
    o = effective_n(df, "y")
    assert_same({key: o[key] for key in ("ms_between", "ms_within", "m0", "rho",
                                         "deff", "n_eff")},
                {"ms_between": 0.0, "ms_within": 2.0, "m0": 2.0, "rho": 0.0,
                 "deff": 1.0, "n_eff": 4.0})


# --- BCa ----------------------------------------------------------------------

def test_c15_bca_shifts_up_for_right_skewed_cluster_means():
    """C15.  For theta = mean, the leave-one-out values are
    theta_(i) = (k xbar - x_i)/(k - 1), so jbar - theta_(i) = (x_i - xbar)/(k-1)
    and the acceleration is the sample skewness over 6 sqrt(k):

        a = sum (x_i - xbar)^3 / (6 [sum (x_i - xbar)^2]^{3/2}).

    Lognormal(0, 1.5) cluster means are right-skewed, so a > 0 (asserted
    here from the closed form).  With a > 0 the adjusted level
    z0 + (z0 + z)/(1 - a(z0 + z)) exceeds z0 + z at both z = -1.96 and
    z = +1.96, and z0 > 0 for a right-skewed mean pushes the same way, so
    both BCa bounds lie above the percentile bounds (same replicates).  With
    the sign of a flipped, the 1/(1 - a(.)) factor shrinks the upper level
    and stretches the lower one, putting both bounds below.  Margins at this
    seed are ~0.1 and ~0.2 against a Monte Carlo jitter of the bounds of
    ~0.01 at B = 4000."""
    x = np.random.default_rng(11).lognormal(0, 1.5, 30)
    d = x - x.mean()
    assert (d ** 3).sum() / (6 * (d ** 2).sum() ** 1.5) > 0.05
    df = pd.DataFrame({"cluster": np.arange(30), "y": x})
    p = cluster_bootstrap(df, "y", n_boot=4000, seed=0, method="percentile")
    b = cluster_bootstrap(df, "y", n_boot=4000, seed=0, method="bca")
    assert b.lo > p.lo and b.hi > p.hi


# --- floors: summarize attaches them and the verdict uses the interval ----------

def _frame(rate: float, n_items: int = 120, seed: int = 0) -> pd.DataFrame:
    """One condition, `rate` correct, labels 70/30 so majority = 0.70."""
    rng = np.random.default_rng(seed)
    labels = ["a"] * int(0.7 * n_items) + ["b"] * (n_items - int(0.7 * n_items))
    return pd.DataFrame({
        "condition": "m",
        "cluster": [f"i{i}" for i in range(n_items)],
        "label": labels,
        "score": rng.binomial(1, rate, n_items).astype(float),
    })


def test_no_floor_args_leaves_the_table_unchanged():
    df = _frame(0.8)
    out = summarize(df, "score", ["condition"], n_boot=200)
    assert not any(c.startswith(("binding_floor", "above_floor"))
                   for c in out.columns)


def test_floor_columns_appear_and_bind_to_the_strongest():
    df = _frame(0.9)
    out = summarize(df, "score", ["condition"], n_boot=500, label="label",
                    floor={"stated_reference": 0.55})
    row = out.iloc[0]
    assert row["binding_floor"] == "majority"        # 0.70 beats the stated 0.55
    assert row["floor_value"] == pytest.approx(0.70)
    assert row["headroom"] == pytest.approx(row["point"] - 0.70)


def test_verdict_uses_the_interval_not_the_point():
    """A cell just above its floor by the point estimate, with an interval
    that straddles it, must not be called above the floor."""
    df = _frame(0.72, n_items=40, seed=3)
    out = summarize(df, "score", ["condition"], n_boot=800, label="label")
    row = out.iloc[0]
    assert row["point"] > row["floor_value"]         # point clears it
    assert row["lo"] < row["floor_value"]            # interval does not
    assert bool(row["above_floor_ci"]) is False   # numpy bool_ in a frame


def test_clearly_above_floor_is_reported_as_such():
    df = _frame(0.97, n_items=200, seed=1)
    out = summarize(df, "score", ["condition"], n_boot=800, label="label")
    assert bool(out.iloc[0]["above_floor_ci"]) is True


@tiny
def test_supplied_only_floors_on_a_loss_scale():
    df = _frame(0.5)
    df["loss"] = 1.692
    out = summarize(df, "loss", ["condition"], n_boot=200,
                    floor={"trigram": 1.784, "stated": 1.90},
                    floor_units="nats", higher_is_better=False)
    row = out.iloc[0]
    assert row["binding_floor"] == "trigram"
    assert row["headroom"] == pytest.approx(0.092, abs=1e-6)


@tiny
def test_c16_loss_floor_verdict_uses_the_upper_bound():
    """C16.  For a loss (lower is better) the cell beats a floor F only if
    its whole interval is below F, i.e. hi < F.  Losses 0..29 / 10 over 30
    clusters: point = 1.45 exactly, interval width ~0.6 (se = sd/sqrt(30)
    ~ 0.16).  F = point + 1e-6 sits inside (point, hi) because hi - point
    is ~0.3, far beyond 1e-6: the point beats F, the interval does not, so
    the verdict is False.  F = 10 is above every value, hence above hi:
    True."""
    df = pd.DataFrame({"condition": "a", "cluster": [f"c{i}" for i in range(30)],
                       "loss": np.arange(30) / 10})
    kw = {"n_boot": 2000, "floor_units": "nats", "higher_is_better": False}
    straddle = summarize(df, "loss", ["condition"], floor={"ref": 1.45 + 1e-6},
                         **kw).iloc[0]
    assert straddle["point"] == pytest.approx(1.45, abs=1e-12)
    assert straddle["point"] < 1.45 + 1e-6 < straddle["hi"]
    assert bool(straddle["above_floor_ci"]) is False
    clear = summarize(df, "loss", ["condition"], floor={"ref": 10.0}, **kw).iloc[0]
    assert bool(clear["above_floor_ci"]) is True


# --- p-values -------------------------------------------------------------------

def test_c32_two_sided_p_values_by_hand():
    """C32.  Bootstrap inversion: replicates 5 x (-1) and 95 x (+1) give
    P(D* <= 0) = 0.05 and P(D* >= 0) = 0.95, so p = 2 min = 0.10.  The
    mirror image gives the same.  Exact McNemar: (b, c) = (0, 6) gives
    2 P(X <= 0 | n = 6, 1/2) = 2/64 = 0.03125, and (3, 3) gives
    2 P(X <= 3) = 2 (1 + 6 + 15 + 20)/64 = 84/64, capped at 1."""
    def boot(reps):
        return BootResult(0.0, 0.0, 0.0, 0.0, 1, 1, replicates=np.array(reps))
    assert paired_pvalue(boot([-1.0] * 5 + [1.0] * 95)) == pytest.approx(0.10)
    assert paired_pvalue(boot([1.0] * 5 + [-1.0] * 95)) == pytest.approx(0.10)
    assert mcnemar_exact(0, 6) == pytest.approx(float(Fraction(1, 32)), rel=1e-12)
    assert mcnemar_exact(6, 0) == pytest.approx(float(Fraction(1, 32)), rel=1e-12)
    assert mcnemar_exact(3, 3) == 1.0


# --- invariances of the tables ------------------------------------------------------

@tiny
def test_c34_row_order_does_not_change_any_estimate():
    """C34.  Every estimator is a function of the multiset of (cluster,
    value) rows, so the row order carries no information.  Shuffled vs
    original through cluster_bootstrap, paired_bootstrap, effective_n and
    summarize, same seed.  Tolerance 1e-12 relative: within-cluster sums
    run in a different order (rounding ~1e-16 per term)."""
    rng = np.random.default_rng(2)
    df = unbalanced(rng.integers(1, 5, size=30), seed=2)
    df = pd.concat([df, df.assign(condition="b",
                                  y=df.y + rng.normal(0.2, .3, len(df)))],
                   ignore_index=True)
    shuf = df.sample(frac=1.0, random_state=9).reset_index(drop=True)
    a = df[df.condition == "a"]
    sa = shuf[shuf.condition == "a"]
    assert_same(_ci(cluster_bootstrap(sa, "y", n_boot=1000, seed=1)),
                _ci(cluster_bootstrap(a, "y", n_boot=1000, seed=1)), rtol=1e-12)
    assert_same(_ci(paired_bootstrap(shuf, "y", "condition", "a", "b", n_boot=1000)),
                _ci(paired_bootstrap(df, "y", "condition", "a", "b", n_boot=1000)),
                rtol=1e-12)
    assert_same(effective_n(sa, "y"), effective_n(a, "y"), rtol=1e-12)
    assert_same(summarize(shuf, "y", ["condition"], n_boot=1000),
                summarize(df, "y", ["condition"], n_boot=1000), rtol=1e-12)


@tiny
def test_c35_summarize_keeps_a_cell_with_a_missing_key():
    """C35.  Two cells, condition "a" and condition missing (None).  The missing-key
    cell is a cell of the grid, not noise: 2 output rows, and the
    missing-condition row's point is the mean of its cluster means, here
    (1 + 3)/2 = 2 exactly."""
    df = pd.DataFrame({"condition": ["a", "a", None, None, None],
                       "cluster": ["c0", "c1", "c2", "c2", "c3"],
                       "y": [0.0, 1.0, 0.0, 2.0, 3.0]})
    out = summarize(df, "y", ["condition"], n_boot=200)
    assert len(out) == 2
    miss = out[out["condition"].isna()]
    assert len(miss) == 1 and miss["point"].iloc[0] == 2.0


# --- BCa with ties: a 0/1 metric ----------------------------------------------

def test_bca_on_symmetric_binary_clusters_is_symmetric():
    """[1]*5 + [0]*5, one row per cluster.  A replicate mean is X/10 with
    X ~ Bin(10, 1/2): P(X <= 1) = 11/1024 = 1.1% and P(X <= 2) = 56/1024 =
    5.5%, so the 2.5% quantile is 0.2 and, by symmetry, the 97.5% one 0.8.
    With ties counted half z0 ~ 0 and a = 0 (the jackknife values are
    symmetric), so BCa is that interval.  Counting only the strict "<"
    gave [0.1, 0.7]."""
    df = pd.DataFrame({"cluster": range(10), "y": [1.0] * 5 + [0.0] * 5})
    r = cluster_bootstrap(df, "y", n_boot=20_000, seed=0, method="bca")
    assert (r.lo, r.hi) == pytest.approx((0.2, 0.8))


def test_bca_misses_a_binary_mean_equally_often_on_each_side():
    """Bernoulli(1/2), one 0/1 value per cluster, k = 20, R = 400.  By the
    symmetry of the problem the interval lies wholly below 1/2 as often as
    wholly above it.  Each rate is ~1.3% (SE ~0.6%), so the SE of their
    difference is ~0.8%; the band is 3 SE.  The strict-"<" z0 gave 4.5%
    below against 0.17% above."""
    R, below, above = 400, 0, 0
    for s in range(R):
        rng = np.random.default_rng(80_000 + s)
        df = pd.DataFrame({"cluster": range(20),
                           "y": (rng.random(20) < 0.5).astype(float)})
        r = cluster_bootstrap(df, "y", n_boot=1000, seed=s, method="bca")
        below += r.hi < 0.5
        above += r.lo > 0.5
    assert abs(below - above) / R < 0.025, (below, above)
