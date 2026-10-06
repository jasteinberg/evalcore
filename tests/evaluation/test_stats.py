"""Calibration tests.  A CI routine that is never checked against a known
generating process is decoration; these check coverage."""

import numpy as np
import pandas as pd
import pytest

from evalcore import (
    benjamini_hochberg,
    cluster_bootstrap,
    effective_n,
    holm,
    icc1,
    paired_bootstrap,
    wilson_interval,
)


def clustered(k=60, m=5, s2_a=0.6, s2_e=0.4, mu=0.0, rng=None):
    """X_ij = mu + a_i + e_ij, one row per (cluster, repeat)."""
    rng = rng or np.random.default_rng(0)
    a = rng.normal(0, np.sqrt(s2_a), size=k)
    e = rng.normal(0, np.sqrt(s2_e), size=(k, m))
    x = mu + a[:, None] + e
    return pd.DataFrame({"cluster": np.repeat(np.arange(k), m),
                         "y": x.ravel()})


def test_icc_recovers_rho():
    s2_a, s2_e, m = 0.6, 0.4, 8
    rho_true = s2_a / (s2_a + s2_e)
    est = [icc1(clustered(k=400, m=m, s2_a=s2_a, s2_e=s2_e,
                          rng=np.random.default_rng(s)), "y")["rho"]
           for s in range(8)]
    # One estimate at k=400, m=8 has sd ~0.027 over seeds (Var(rho_hat) ~
    # 2(1-rho)^2 (1+(m-1)rho)^2 / (m(m-1)(k-1)) gives 0.020; the empirical
    # spread is larger).  The mean of 8 then has SE ~0.0096: 3 SE.
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
    # 0, so it sits about one SE high; rel=0.25 admits rho_hat up to 0.067,
    # i.e. ~3.7 SE.
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
    coverage at 0.95 is 0.015; the band is ~ -3/+2.7 SE."""
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
                             "arm": ["a"] * 3 * k + ["b"] * 3 * k,
                             "y": np.r_[ya.ravel(), yb.ravel()]})

    cov = _coverage(200, draw, lambda df, s: paired_bootstrap(
        df, "y", "arm", "a", "b", n_boot=800, seed=s), delta)
    assert 0.905 <= cov <= 0.99, f"paired coverage {cov:.3f}"


def test_cluster_bootstrap_covers_and_row_bootstrap_does_not():
    """95% nominal.  Cluster bootstrap should land near 0.95; the row
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
    difficulty = rng.normal(0, 1.0, size=k)          # shared across arms
    a = difficulty + rng.normal(0.25, 0.2, size=k)
    b = difficulty + rng.normal(0.00, 0.2, size=k)
    df = pd.DataFrame({"cluster": np.r_[np.arange(k), np.arange(k)],
                       "arm": ["a"] * k + ["b"] * k, "y": np.r_[a, b]})
    pair = paired_bootstrap(df, "y", "arm", "a", "b", n_boot=2000)
    ua = cluster_bootstrap(df[df.arm == "a"], "y", n_boot=2000)
    ub = cluster_bootstrap(df[df.arm == "b"], "y", n_boot=2000)
    unpaired_se = np.hypot(ua.se, ub.se)
    assert pair.se < unpaired_se / 3                  # eq. (4), r ~ 0.98
    assert pair.lo > 0                                # and it is resolvable
    assert ua.lo < ub.hi                              # while the CIs overlap


def test_wilson_does_not_collapse_at_the_boundary():
    z2 = 1.959963984540054 ** 2
    # at phat = 0 the score equation p^2 = z^2 p(1-p)/n has roots 0 and
    # z^2/(n + z^2) = 0.1135; Wald would give [0, 0]
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


def test_bh_is_step_up_not_step_down():
    """p_(1) = .040 > q/m = .01, so a step-DOWN procedure rejects nothing;
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

from evalcore.evaluation.stats import (  # noqa: E402
    SMALL_K_COVERAGE,
    cell_caveats,
    summarize,
)


def frame_of(k, m=4, const=False, missing=0, seed=0):
    rng = np.random.default_rng(seed)
    df = clustered(k=k, m=m, rng=rng)
    df["arm"] = "a"
    if const:
        df["y"] = 1.0
    if missing:
        df.loc[df.index[:missing], "y"] = np.nan
    return df


def test_few_clusters_are_flagged_with_the_measured_undercoverage():
    with pytest.warns(RuntimeWarning, match="1 of 1 cells carry caveats"):
        s = summarize(frame_of(3), "y", ["arm"], n_boot=200)
    assert "only 3 clusters" in s.iloc[0]["caveats"]
    assert "covers ~68% at k=3" in s.iloc[0]["caveats"]


def test_a_well_resolved_cell_has_no_caveats_and_no_warning(recwarn):
    s = summarize(frame_of(40), "y", ["arm"], n_boot=200)
    assert s.iloc[0]["caveats"] == ""
    assert not [w for w in recwarn if "caveats" in str(w.message)]


def test_no_spread_and_many_excluded_are_flagged():
    with pytest.warns(RuntimeWarning):
        s = summarize(frame_of(40, const=True), "y", ["arm"], n_boot=100)
    assert "no spread" in s.iloc[0]["caveats"]
    with pytest.warns(RuntimeWarning):
        s = summarize(frame_of(40, missing=40), "y", ["arm"], n_boot=100)
    assert "25% of rows have no value" in s.iloc[0]["caveats"]   # 40 of 160


def test_only_the_affected_cells_are_flagged():
    df = pd.concat([frame_of(40).assign(arm="big"),
                    frame_of(5, seed=1).assign(arm="small")])
    with pytest.warns(RuntimeWarning, match="1 of 2 cells"):
        s = summarize(df, "y", ["arm"], n_boot=100).set_index("arm")
    assert s.loc["big", "caveats"] == ""
    assert "only 5 clusters" in s.loc["small", "caveats"]


def test_the_quoted_undercoverage_is_what_the_bootstrap_does():
    """The caveat quotes SMALL_K_COVERAGE; check its k=5 entry against a
    fresh simulation (R=200: binomial SE ~0.027 near 0.83)."""
    R, mu, hits = 200, 0.3, 0
    for s in range(R):
        df = clustered(k=5, m=6, mu=mu, rng=np.random.default_rng(50_000 + s))
        r = cluster_bootstrap(df, "y", n_boot=400, seed=s)
        hits += r.lo <= mu <= r.hi
    assert abs(hits / R - SMALL_K_COVERAGE[5]) < 0.07
    assert cell_caveats({"n_clusters": 30, "se": 0.1, "n_rows": 10,
                         "n_excluded": 0}) == []
