"""Judge statistics: error rates, debiasing, agreement, error correlation
and the majority verdict.  Expected values are hand-computed tiny cases,
closed forms, or a coverage simulation against a known generating process
(the debiased interval)."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evalcore.evaluation.judges import (
    agreement,
    debias,
    debiased_mean,
    error_correlation,
    judge_error_rates,
    majority,
    ratio_se,
)


def long(verdicts: dict[str, list[int]], reference: list[int] | None = None,
         criterion: str = "c") -> pd.DataFrame:
    """One row per (answer, judge); answer i is cluster i."""
    rows = []
    for judge, vs in verdicts.items():
        for i, v in enumerate(vs):
            if v is None:
                continue
            r = {"answer": f"a{i}", "cluster": f"k{i}", "judge": judge,
                 "criterion": criterion, "verdict": v}
            if reference is not None:
                r["reference"] = reference[i]
            rows.append(r)
    return pd.DataFrame(rows)


# --- error rates ------------------------------------------------------------

def test_ratio_se_hand_case():
    # r = 3/5; residuals -0.2, -0.6, 0.8; sum of squares 1.04;
    # sqrt(3/2 * 1.04) / 5 = 0.249799...
    assert ratio_se(np.array([1, 0, 2]), np.array([2, 1, 2])) == \
        pytest.approx(np.sqrt(1.5 * 1.04) / 5, rel=1e-12)


def test_ratio_se_with_one_row_per_cluster_is_the_binomial_se():
    x = np.array([1, 0, 0, 1, 1, 0, 0, 0])            # alpha = 3/8, n = 8
    assert ratio_se(x, np.ones(8)) == pytest.approx(
        np.sqrt(3 / 8 * 5 / 8 / 7), rel=1e-12)


def test_calibrate_hand_case():
    df = long({"A": [1, 1, 0, 1, 0, 0]}, reference=[1, 1, 1, 0, 0, 0])
    row = judge_error_rates(df).iloc[0]
    assert (row.tp, row.fn, row.fp, row.tn) == (2, 1, 1, 2)
    assert row.alpha == pytest.approx(1 / 3) and row.beta == pytest.approx(1 / 3)
    assert row.lam == pytest.approx(1 / 3)
    assert (row.n_pos, row.n_neg, row.k) == (3, 3, 6)


def test_calibrate_refuses_non_binary_and_missing_reference():
    df = long({"A": [1, 0]}, reference=[1, 0])
    with pytest.raises(ValueError, match="0 or 1"):
        judge_error_rates(df.assign(verdict=[1, 2]))
    with pytest.raises(ValueError, match="missing values"):
        judge_error_rates(df.assign(reference=[1.0, np.nan]))


# --- debiasing --------------------------------------------------------------

def test_debias_point_hand_case():
    # theta = (0.6 - 0.1) / (1 - 0.1 - 0.2) = 5/7
    d = debias(0.6, 0.0, 0.1, 0.0, 0.2, 0.0)
    assert d["theta"] == pytest.approx(5 / 7) and d["se"] == 0.0
    assert d["attenuation"] == pytest.approx(0.49)


def test_debias_variance_matches_finite_differences():
    """Eq. (3) is the delta method: the gradient of theta(m, alpha, beta),
    taken numerically here, must reproduce it."""
    def theta(m, a, b):
        return (m - a) / (1 - a - b)
    p, ses, h = (0.6, 0.1, 0.2), (0.03, 0.02, 0.04), 1e-6
    grad = [(theta(*[x + h * (i == j) for j, x in enumerate(p)])
             - theta(*[x - h * (i == j) for j, x in enumerate(p)])) / (2 * h)
            for i in range(3)]
    want = np.sqrt(sum((g * s) ** 2 for g, s in zip(grad, ses, strict=True)))
    assert debias(p[0], ses[0], p[1], ses[1], p[2], ses[2])["se"] == \
        pytest.approx(want, rel=1e-6)


@pytest.mark.parametrize("a,b", [(0.5, 0.5), (0.6, 0.5)])
def test_debias_refuses_a_judge_that_does_not_discriminate(a, b):
    with pytest.raises(ValueError, match="does not discriminate"):
        debias(0.5, 0.01, a, 0.01, b, 0.01)


def test_debiased_interval_covers_the_true_rate():
    """Calibration test.  A judge with alpha = 0.15, beta = 0.25 scores
    n = 300 answers with true pass rate theta = 0.6, after being calibrated
    on 150 reference positives and 150 negatives drawn separately.  The 95%
    debiased interval must cover theta at the nominal rate (MC se of the
    coverage over 2000 reps is 0.005).  The naive interval on m is centred
    at alpha + lambda theta = 0.51 with half-width 1.96 sqrt(0.25/300) =
    0.0566, so it covers theta = 0.6 only when m > 0.5434, with probability
    1 - Phi((0.5434 - 0.51)/0.0289) = Phi(-1.157) = 0.124."""
    rng = np.random.default_rng(0)
    theta, alpha, beta, n, n_cal, reps = 0.6, 0.15, 0.25, 300, 150, 2000
    hit = naive_hit = 0
    est = []
    for _ in range(reps):
        y = rng.random(n) < theta
        v = np.where(y, rng.random(n) >= beta, rng.random(n) < alpha)
        m = v.mean()
        fp = (rng.random(n_cal) < alpha).astype(float)
        fn = (rng.random(n_cal) < beta).astype(float)
        a_hat, b_hat = fp.mean(), fn.mean()
        d = debias(m, np.sqrt(m * (1 - m) / (n - 1)),
                   a_hat, ratio_se(fp, np.ones(n_cal)),
                   b_hat, ratio_se(fn, np.ones(n_cal)))
        hit += d["lo"] <= theta <= d["hi"]
        half = 1.96 * np.sqrt(m * (1 - m) / (n - 1))
        naive_hit += m - half <= theta <= m + half
        est.append(d["theta"])
    assert abs(hit / reps - 0.95) < 0.02
    assert abs(naive_hit / reps - 0.124) < 0.025      # MC se 0.007
    assert abs(np.mean(est) - theta) < 0.01


def test_debiased_mean_refuses_overlap_with_the_calibration_set():
    df = long({"A": [1, 0, 1]})
    cal = {"judge": "A", "criterion": "c", "alpha": 0.1, "se_alpha": 0.01,
           "beta": 0.1, "se_beta": 0.01}
    with pytest.raises(ValueError, match="both the calibration"):
        debiased_mean(df, cal, cal_clusters={"k1"}, n_boot=200)
    with pytest.raises(ValueError, match="calibration is for"):
        debiased_mean(df, {**cal, "judge": "B"}, cal_clusters=set(), n_boot=200)


def test_debiased_mean_of_a_perfect_judge_is_the_pass_rate():
    df = long({"A": [1, 0, 1, 1]})
    cal = {"judge": "A", "criterion": "c", "alpha": 0.0, "se_alpha": 0.0,
           "beta": 0.0, "se_beta": 0.0}
    d = debiased_mean(df, cal, cal_clusters=set(), n_boot=500)
    assert d["theta"] == pytest.approx(0.75) and d["n_clusters"] == 4


# --- agreement and correlation ----------------------------------------------

def test_agreement_hand_case():
    # p_o = 3/4, p_A = 1/2, p_B = 1/4, p_e = 1/8 + 3/8 = 1/2, kappa = 1/2;
    # b = 1 (A passes, B fails), c = 0, exact McNemar p = 1
    row = agreement(long({"A": [1, 1, 0, 0], "B": [1, 0, 0, 0]})).iloc[0]
    assert (row.judge_1, row.judge_2, row.n) == ("A", "B", 4)
    assert row.agreement == pytest.approx(0.75)
    assert row.kappa == pytest.approx(0.5)
    assert (row.b, row.c, row.mcnemar_p) == (1, 0, 1.0)


def test_agreement_counts_only_answers_both_judges_scored():
    row = agreement(long({"A": [1, 1, 0], "B": [1, None, 0]})).iloc[0]
    assert row.n == 2 and row.agreement == 1.0


def test_identical_errors_make_two_judges_worth_one():
    ref = [1, 1, 0, 0, 1, 0]
    v = [1, 0, 0, 1, 1, 0]
    out = error_correlation(long({"A": v, "B": v}, reference=ref))
    assert out["pairs"][("A", "B")] == pytest.approx(1.0)
    assert out["n_eff"] == pytest.approx(1.0)


def test_error_correlation_matches_the_common_shock_closed_form():
    """Errors e_j = C or I_j with a shared shock C ~ Bern(c) and
    independent I_j ~ Bern(q).  Then P(e_j) = p = c + (1 - c) q,
    P(e_i e_j) = c + (1 - c) q^2, and
    rho = [c + (1 - c) q^2 - p^2] / [p (1 - p)].  With c = 0.1, q = 0.1:
    p = 0.19, rho = (0.109 - 0.0361) / 0.1539 = 0.47368...  N = 100k, so
    the sampling se of each rho is about 0.003."""
    rng = np.random.default_rng(1)
    n, c, q = 100_000, 0.1, 0.1
    y = (rng.random(n) < 0.5).astype(int)
    shock = rng.random(n) < c
    verdicts = {}
    for j in "ABC":
        err = shock | (rng.random(n) < q)
        verdicts[j] = list(np.where(err, 1 - y, y))
    out = error_correlation(long(verdicts, reference=list(y)))
    p = c + (1 - c) * q
    rho = (c + (1 - c) * q * q - p * p) / (p * (1 - p))
    for r in out["pairs"].values():
        assert r == pytest.approx(rho, abs=0.015)
    assert out["n_eff"] == pytest.approx(3 / (1 + 2 * rho), abs=0.05)


def test_error_correlation_of_an_errorless_judge_is_nan():
    out = error_correlation(long({"A": [1, 0, 1], "B": [0, 0, 1]},
                                 reference=[1, 0, 1]))
    assert np.isnan(out["pairs"][("A", "B")]) and np.isnan(out["n_eff"])


# --- majority ---------------------------------------------------------------

def test_majority_hand_case_and_incomplete_panels():
    df = long({"A": [1, 0, 1], "B": [1, 0, None], "C": [0, 1, 1]})
    out = majority(df, ["A", "B", "C"]).set_index("answer")
    assert out.loc["a0", "panel"] == 1.0 and out.loc["a1", "panel"] == 0.0
    assert np.isnan(out.loc["a2", "panel"])
    assert out.loc["a2", "status"] == "incomplete" and out.loc["a2", "n_judges"] == 2
    assert out.loc["a0", "cluster"] == "k0"


def test_majority_refuses_even_panels_and_unknown_judges():
    df = long({"A": [1], "B": [0]})
    with pytest.raises(ValueError, match="odd number"):
        majority(df, ["A", "B"])
    with pytest.raises(ValueError, match="not on the panel"):
        majority(df, ["A"])


def test_duplicate_verdicts_are_refused():
    df = long({"A": [1, 0], "B": [1, 0]})
    with pytest.raises(ValueError, match="duplicate"):
        agreement(pd.concat([df, df.iloc[[0]]]))


def test_ratio_se_with_the_whole_domain_in_one_cluster_is_nan():
    """All negatives in cluster 0: the one residual is 1 - (1/3)*3 = 0, so
    eq. (1) gives 0.  That is no estimate of the spread: NaN."""
    assert np.isnan(ratio_se(np.array([1, 0, 0]), np.array([3, 0, 0])))


def test_ratio_se_counts_clusters_outside_the_domain_in_k():
    """By hand: r = 3/5, residuals -0.2, -0.6, 0.8 and 0 for the den = 0
    cluster, sum of squares 1.04, k = 4: sqrt(4/3 * 1.04) / 5."""
    got = ratio_se(np.array([1, 0, 2, 0]), np.array([2, 1, 2, 0]))
    assert got == pytest.approx(np.sqrt(4 / 3 * 1.04) / 5)


def test_error_correlation_at_the_lower_bound_gives_infinite_n_eff():
    """Two judges whose errors never coincide and cover every answer:
    e_A = 1 - e_B, rho = -1 = -1/(J-1).  The mean error is constant, so
    J / (1 + (J-1) rho) = 2/0 -> +inf."""
    y = [1, 1, 1, 1]
    out = error_correlation(long({"A": [0, 0, 1, 1], "B": [1, 1, 0, 0]},
                                 reference=y))
    assert out["rho_bar"] == pytest.approx(-1.0)
    assert out["n_eff"] == float("inf")
