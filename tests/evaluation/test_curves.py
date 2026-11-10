"""Curves along a model axis.  Expected values come from hand arithmetic,
an identity stated independently of the code, or a generating process
with a known curve -- never from running the estimator."""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

from evalcore import (
    axis_curves,
    axis_data,
    composition,
    explode_tokens,
    subsampled_sharpness,
    susceptibility,
)
from evalcore.evaluation.curves import (
    MAX_CENSORED_SHARE,
    crossing,
    value_curves,
    value_data,
)
from evalcore.testing import assert_same

pytestmark = pytest.mark.filterwarnings(
    # small fixtures by design. The caveats themselves are tested at the end
    "ignore:curves carry caveats:RuntimeWarning")


def frame(ok_by_x: dict[float, list[list[int]]], cluster=None,
          status=None) -> pd.DataFrame:
    """Run-frame rows: one per (item, x), per-token correctness as lists.
    `status` maps an item index to a non-ok status at every x."""
    rows = []
    for x, oks in ok_by_x.items():
        for i, ok in enumerate(oks):
            st = (status or {}).get(i, "ok")
            rows.append({"x": x, "item_id": f"i{i}",
                         "cluster": cluster[i] if cluster else f"i{i}",
                         "status": st,
                         "meta_argmax_ok": list(map(bool, ok)) if st == "ok"
                         else None,
                         "meta_n_target": len(ok)})
    return pd.DataFrame(rows)


# six items: two of length 1, four of length 2
HAND = [[1], [0], [1, 1], [1, 0], [0, 1], [1, 1]]


def test_composition_by_hand():
    """A_obs = 3/6.  Pooled marginals m0 = 4/6, m1 = 3/4, so
    A_pred = (2 (2/3) + 4 (2/3)(3/4)) / 6 = 5/9 and phi = -1/18.
    Length classes: f1 = 1/3 with a_S = 1/2, and f2 = 2/3 with b1 = b2 = 3/4,
    so A_pred_lc = 1/6 + (2/3)(9/16) = 13/24, phi_lc = -1/24,
    phi_2 = 2/4 - 9/16 = -1/16.  Tokens: 7 of 10 correct."""
    d = axis_data(frame({1.0: HAND, 2.0: HAND}), "x")
    c = composition(np.ones(6), d.ok[0], d.valid[0], d.nt[0])
    assert c["A_obs"] == pytest.approx(1 / 2)
    assert c["A_pred"] == pytest.approx(5 / 9)
    assert c["phi"] == pytest.approx(-1 / 18)
    assert c["A_pred_lc"] == pytest.approx(13 / 24)
    assert c["phi_lc"] == pytest.approx(-1 / 24)
    assert c["phi_l2"] == pytest.approx(-1 / 16)
    assert c["phi_l1"] == pytest.approx(0.0)       # one token: A = p exactly
    assert c["p"] == pytest.approx(0.7)


def test_pooled_residual_decomposes_as_stated():
    """phi = f1 f2 (1 - b2)(a_S - b1) + f2 phi_2: pooling term plus the
    within-class term.  An identity of the definitions, checked on random
    tables rather than on the hand case it was derived from."""
    rng = np.random.default_rng(0)
    for _ in range(20):
        oks = [list(rng.integers(0, 2, size=rng.integers(1, 3)))
               for _ in range(50)]
        d = axis_data(frame({1.0: oks, 2.0: oks}), "x")
        c = composition(np.ones(50), d.ok[0], d.valid[0], d.nt[0])
        one = [o for o in oks if len(o) == 1]
        two = np.array([o for o in oks if len(o) == 2])
        f1, f2 = len(one) / 50, len(two) / 50
        a_S = np.mean([o[0] for o in one])
        b1, b2 = two[:, 0].mean(), two[:, 1].mean()
        phi2 = (two[:, 0] * two[:, 1]).mean() - b1 * b2
        assert c["phi"] == pytest.approx(
            f1 * f2 * (1 - b2) * (a_S - b1) + f2 * phi2, abs=1e-12)
        assert c["phi_lc"] == pytest.approx(f2 * phi2, abs=1e-12)


def test_weights_are_resampling_not_reweighting():
    """Integer weights must equal duplicating rows: w = (2,0,1,...) is the
    table with item 0 twice and item 1 absent."""
    d = axis_data(frame({1.0: HAND, 2.0: HAND}), "x")
    w = np.array([2, 0, 1, 3, 0, 1.0])
    dup = [o for o, k in zip(HAND, w.astype(int), strict=True) for _ in range(k)]
    e = axis_data(frame({1.0: dup, 2.0: dup}), "x")
    a = composition(w, d.ok[0], d.valid[0], d.nt[0])
    b = composition(np.ones(len(dup)), e.ok[0], e.valid[0], e.nt[0])
    for key in ("A_obs", "A_pred", "A_pred_lc", "phi", "p"):
        assert a[key] == pytest.approx(b[key]), key


def test_susceptibility_is_exact_for_a_curve_linear_in_log_x():
    x = np.array([1e7, 3e7, 2e8, 5e8, 4e9])           # non-uniform in log x
    A = 0.05 * np.log(x) - 0.3
    assert susceptibility(x, A) == pytest.approx(np.full(5, 0.05))
    assert susceptibility(x, np.vstack([A, 2 * A])) == pytest.approx(
        np.repeat([[0.05], [0.10]], 5, axis=1))          # row by row


CENTRE = np.log(1e9)


def logistic_items(k, xs, rng, slope=3.0, centre=CENTRE):
    """Single-token items, comonotone along the axis: item i is right at x
    iff u_i < A(x) = logistic(slope (log x - centre)).  The population
    curve, and so its finite-difference chi on the grid, is known."""
    A = 1 / (1 + np.exp(-slope * (np.log(xs) - centre)))
    u = rng.random(k)
    return {x: [[int(ui < a)] for ui in u] for x, a in zip(xs, A, strict=True)}, A


def test_chi_interval_covers_the_true_finite_difference():
    """Coverage at the true peak, 95% nominal.  R = 120 datasets: binomial
    SE of the coverage is 0.02.  The band is -2.5/+2 SE."""
    xs = np.geomspace(1e8, 1e10, 7)
    R, hits = 120, 0
    for s in range(R):
        rng = np.random.default_rng(100 + s)
        ok_by_x, A = logistic_items(300, xs, rng)
        chi_true = susceptibility(xs, A)
        j = int(np.argmax(chi_true))
        res = axis_curves(frame(ok_by_x), "x", n_boot=400, seed=s)
        row = res.table.iloc[j]
        hits += row["chi_obs_lo"] <= chi_true[j] <= row["chi_obs_hi"]
    assert 0.90 <= hits / R <= 0.99, f"chi coverage {hits / R:.3f}"


def test_chi_standard_error_matches_the_paired_closed_form():
    """On a uniform log grid with step h, chi_j = (A_{j+1} - A_{j-1}) / 2h.
    For comonotone items Cov(A_hat_1, A_hat_2) = A_1 (1 - A_2), so

        paired:    Var = d (1 - d) / k,             d = A_{j+1} - A_{j-1},
        unpaired:  Var = [A_{j+1}(1 - A_{j+1}) + A_{j-1}(1 - A_{j-1})] / k,

    each divided by (2h)^2.  The bootstrap SE must match the first.  With
    the pairing destroyed (items reshuffled at each x) it must match the
    second.  Dense grid, so the gain is large, as for checkpoints."""
    k = 2000
    xs = np.geomspace(1e8, 1e10, 21)
    h = np.log(xs[1] / xs[0])
    ok_by_x, A = logistic_items(k, xs, np.random.default_rng(0), slope=1.0)
    j = 10                                              # the centre point
    d = A[j + 1] - A[j - 1]
    se_paired = np.sqrt(d * (1 - d) / k) / (2 * h)
    se_unpaired = np.sqrt((A[j + 1] * (1 - A[j + 1])
                           + A[j - 1] * (1 - A[j - 1])) / k) / (2 * h)

    paired = axis_curves(frame(ok_by_x), "x", n_boot=1000)
    rng = np.random.default_rng(1)
    shuffled = {x: [ok[i] for i in rng.permutation(len(ok))]
                for x, ok in ok_by_x.items()}
    unpaired = axis_curves(frame(shuffled), "x", n_boot=1000)
    # sd of a bootstrap sd from 1000 replicates is ~2.2%, so 10% is ~4.5 sd
    assert paired.reps["chi_obs"][:, j].std() == pytest.approx(se_paired,
                                                               rel=0.10)
    assert unpaired.reps["chi_obs"][:, j].std() == pytest.approx(
        se_unpaired, rel=0.10)
    # closed forms give 0.45 here: what pairing buys on this grid
    assert se_paired < se_unpaired / 2


def test_independent_tokens_give_phi_lc_near_zero_while_pooled_is_biased():
    """Tokens independent within each length class, but the classes have
    different marginals: the pooled null is biased (a pooling artifact) and
    the length-conditioned one is not.  Coverage of 0 by the phi_lc
    interval over R datasets."""
    xs = [1.0, 2.0, 4.0]
    R, cover, pooled = 80, 0, []
    for s in range(R):
        rng = np.random.default_rng(500 + s)
        ok_by_x = {}
        for x in xs:
            q1 = 0.3 * x / 4                               # length-1 items
            q2 = (0.9, 0.5 + 0.1 * x)                      # length-2 items
            oks = [[int(rng.random() < q1)] for _ in range(200)]
            oks += [[int(rng.random() < q2[0]), int(rng.random() < q2[1])]
                    for _ in range(200)]
            ok_by_x[x] = oks
        t = axis_curves(frame(ok_by_x), "x", n_boot=300, seed=s).table
        cover += (t.phi_lc_lo.iloc[1] <= 0 <= t.phi_lc_hi.iloc[1])
        pooled.append(t.phi.iloc[1])
    assert 0.88 <= cover / R <= 1.0, f"phi_lc covers 0 in {cover / R:.3f}"
    # pooling term at x=2: f1 f2 (1 - b2)(a_S - b1) = .25 (.3)(.15 - .9)
    assert np.mean(pooled) == pytest.approx(0.25 * 0.3 * (0.15 - 0.9),
                                            abs=0.01)


def test_items_missing_at_one_point_are_dropped_and_reported():
    oks = {1.0: HAND, 2.0: HAND[:5]}
    with pytest.warns(RuntimeWarning, match="1 items are not scored"):
        res = axis_curves(frame(oks), "x", n_boot=50)
    assert res.n_items == 5 and res.n_dropped == 1


def test_clusters_are_resampled_whole():
    """Two items per cluster that always agree: resampling items would see
    twice the independent information.  With clusters, the replicate spread
    must match the single-item table's."""
    rng = np.random.default_rng(3)
    xs = np.geomspace(1, 100, 5)
    ok_by_x, _ = logistic_items(150, xs, rng, slope=1.0, centre=np.log(10))
    twin = {x: [o for o in oks for _ in (0, 1)] for x, oks in ok_by_x.items()}
    clus = [f"c{i // 2}" for i in range(300)]
    single = axis_curves(frame(ok_by_x), "x", n_boot=800, seed=0)
    paired = axis_curves(frame(twin, cluster=clus), "x", n_boot=800, seed=0)
    s1 = single.reps["A_obs"].std(0)
    s2 = paired.reps["A_obs"].std(0)
    assert s2 == pytest.approx(s1, rel=0.15)


def test_subsampled_sharpness_shrinks_its_spread_with_m():
    xs = np.geomspace(1e8, 1e10, 7)
    ok_by_x, _ = logistic_items(2000, xs, np.random.default_rng(0))
    out = subsampled_sharpness(frame(ok_by_x), "x", ms=[100, 400, 1600],
                               n_draw=200)
    assert list(out.m) == [100, 400, 1600]
    assert out.s_sd.is_monotonic_decreasing
    # sd ~ 1/sqrt(m) up to the finite-population correction (m/k <= 0.8)
    assert out.s_sd.iloc[0] / out.s_sd.iloc[1] == pytest.approx(2.0, rel=0.3)


def test_explode_tokens():
    """Lists explode together with a per-unit position, and ndarray cells (a
    parquet round trip) and a repeated index (a concat of frames) give the
    same positions."""
    df = pd.DataFrame({"unit_id": ["a", "b"], "meta_logp": [[-0.1, -2.0], [-0.5]],
                       "meta_argmax_ok": [[True, False], [True]]})
    t = explode_tokens(df)
    assert t[["unit_id", "position"]].values.tolist() == [["a", 0], ["a", 1],
                                                          ["b", 0]]
    assert t["meta_logp"].tolist() == [-0.1, -2.0, -0.5]
    arr = pd.DataFrame({"unit_id": ["a", "b"],
                        "meta_argmax_ok": [np.array([True, False]),
                                           np.array([True])]})
    both = pd.concat([arr, arr.assign(unit_id=["c", "d"])])  # repeated index
    assert explode_tokens(both)["position"].tolist() == [0, 1, 0, 0, 1, 0]


# --- input validation -----------------------------------------------------------

def test_axis_values_must_be_numeric():
    f = frame({1.0: HAND, 2.0: HAND})
    f["x"] = f["x"].map({1.0: "70000000", 2.0: "160000000"})
    with pytest.raises(ValueError, match="numeric"):
        axis_data(f, "x")


def test_nonpositive_x_needs_a_linear_axis():
    f = frame({0.0: HAND, 1000.0: HAND, 2000.0: HAND})
    with pytest.raises(ValueError, match="log_x=False"):
        axis_curves(f, "x", n_boot=20)
    t = axis_curves(f, "x", n_boot=20, log_x=False).table
    assert np.isfinite(t["chi_obs"]).all()


def test_duplicate_item_rows_at_one_x_are_refused():
    f = pd.concat([frame({1.0: HAND, 2.0: HAND}),
                   frame({1.0: HAND, 2.0: HAND})])     # e.g. two conditions
    with pytest.raises(ValueError, match="more than one row"):
        axis_data(f, "x")


def test_missing_cluster_labels_are_refused():
    f = frame({1.0: HAND, 2.0: HAND})
    f.loc[f.item_id == "i3", "cluster"] = None
    with pytest.raises(ValueError, match="cluster"):
        axis_data(f, "x")


def test_a_peak_on_the_grid_edge_is_flagged():
    """chi at an end point is a one-sided difference, twice the centred one
    for a single jump, so an edge peak is an artefact of the grid."""
    jump = {1.0: [[0]] * 4 + [[1]] * 4, 10.0: [[1]] * 8, 100.0: [[1]] * 8}
    peak = axis_curves(frame(jump), "x", n_boot=20).peak.set_index("quantity")
    assert peak.loc["peak_at_edge", "point"] == 1.0


# --- curves of any per-item value ----------------------------------------------

def test_crossing_is_exact_on_a_line_in_log_x():
    x = np.array([1.0, 10.0, 100.0, 1000.0])
    y = np.log10(x) - 1.5                       # crosses 0 at x = 10^1.5
    assert crossing(x, y) == pytest.approx(10 ** 1.5)
    assert crossing(x, -y) == pytest.approx(10 ** 1.5)       # decreasing
    assert crossing(x, np.log10(x) - 2) == pytest.approx(100.0)   # on a point
    assert np.isnan(crossing(x, y + 10))                      # never: censored
    assert crossing(x, y, level=0.5) == pytest.approx(100.0)
    both = crossing(x, np.vstack([y, y + 10]))
    assert both[0] == pytest.approx(10 ** 1.5) and np.isnan(both[1])


def vframe(V, xs, items=None, cluster=None, status=None):
    """Per-item value rows: V is (items, n_x).  `status` maps an item
    index to a non-ok status at every x."""
    items = items or [f"i{i}" for i in range(V.shape[0])]
    return pd.DataFrame([{"x": x, "item_id": items[i],
                          "cluster": cluster[i] if cluster else items[i],
                          "status": (status or {}).get(i, "ok"),
                          "v": V[i, j]}
                         for i in range(V.shape[0]) for j, x in enumerate(xs)])


def test_value_curves_by_hand():
    xs = [1.0, 10.0, 100.0]
    V = np.array([[-1.0, 0.0, 2.0], [-3.0, 0.0, 1.0], [-2.0, 3.0, 3.0]])
    res = value_curves(vframe(V, xs), "x", "v", levels=[0.0], n_boot=50)
    assert res.table["mean"].tolist() == pytest.approx([-2.0, 1.0, 2.0])
    # mean crosses 0 between x=1 (-2) and x=10 (+1): u = 0 + ln10 * 2/3
    peak = res.peak.set_index("quantity")
    assert peak.loc["crossing_0", "point"] == pytest.approx(10 ** (2 / 3))
    assert peak.loc["crossing_0_censored", "point"] == 0.0


def test_crossing_interval_covers_the_population_crossing():
    """y_ix = (ln x - c) + a_i + e_ix with E[a] = 0: the population mean
    curve crosses 0 at x = e^c.  R = 100: binomial SE of the coverage is
    0.022.  The band is about -2.3/+1.8 SE."""
    c, xs = np.log(50.0), np.geomspace(5, 500, 7)
    R, hits = 100, 0
    for s in range(R):
        rng = np.random.default_rng(900 + s)
        a = rng.normal(0, 0.4, size=60)
        V = (np.log(xs)[None, :] - c) + a[:, None] + rng.normal(0, 0.3,
                                                                (60, 7))
        pk = value_curves(vframe(V, xs), "x", "v", levels=[0.0], n_boot=400,
                          seed=s).peak.set_index("quantity")
        hits += pk.loc["crossing_0", "lo"] <= 50.0 <= pk.loc["crossing_0", "hi"]
    assert 0.90 <= hits / R <= 0.99, f"crossing coverage {hits / R:.3f}"


def test_missing_values_drop_the_item_everywhere_and_say_so():
    xs = [1.0, 10.0]
    V = np.array([[1.0, 2.0], [3.0, np.nan], [5.0, 6.0]])
    with pytest.warns(RuntimeWarning, match="1 items are not scored"):
        res = value_curves(vframe(V, xs), "x", "v", n_boot=20)
    assert res.n_items == 2 and res.table["mean"].tolist() == [3.0, 4.0]


# --- caveats --------------------------------------------------------------------

def sigmoid_frame(k: int, seed: int = 0, noise: float = 0.1) -> pd.DataFrame:
    """k items around a logistic in log x centred mid-grid: chi peaks inside
    the grid and the mean crosses 0.5 at x = 30."""
    xs = np.geomspace(1, 900, 7)
    rng = np.random.default_rng(seed)
    mean = 1 / (1 + np.exp(-(np.log(xs) - np.log(30.0)) * 2))
    return vframe(mean[None, :] + rng.normal(0, noise, (k, 7)), xs)


def test_a_well_resolved_curve_carries_no_caveats():
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        res = value_curves(sigmoid_frame(60), "x", "v", levels=[0.5],
                           n_boot=300)
    assert res.caveats == []


def test_few_items_behind_a_curve_are_flagged():
    with pytest.warns(RuntimeWarning, match="curves carry caveats"):
        res = value_curves(sigmoid_frame(8), "x", "v", levels=[0.5],
                           n_boot=300)
    assert any(c.startswith("only 8 clusters") for c in res.caveats)


def test_a_censored_crossing_is_counted_and_flagged_with_its_share():
    """Replicates that never cross are counted, not hidden: the share is
    reported, and above MAX_CENSORED_SHARE it becomes a caveat."""
    xs = [1.0, 10.0, 100.0]
    rng = np.random.default_rng(0)
    V = np.array([-0.05, 0.0, 0.05])[None, :] + rng.normal(0, 0.3, (30, 3))
    with pytest.warns(RuntimeWarning):
        res = value_curves(vframe(V, xs), "x", "v", levels=[0.0], n_boot=400)
    share = res.peak.set_index("quantity").loc["crossing_0_censored", "lo"]
    assert MAX_CENSORED_SHARE < share < 1.0     # some replicates do cross
    assert f"crossing_0: {share:.0%} of replicates never cross" in \
        "; ".join(res.caveats)


def test_a_curve_that_never_crosses_says_so():
    with pytest.warns(RuntimeWarning):
        res = value_curves(sigmoid_frame(60), "x", "v", levels=[2.0],
                           n_boot=100)
    assert "crossing_2: the observed curve never reaches the level" \
        in res.caveats


def test_a_peak_at_the_grid_edge_is_flagged():
    """mean = x on a log grid: chi = dA/dlog x = x grows to the last point."""
    xs = np.geomspace(1, 100, 5)
    V = xs[None, :] + np.random.default_rng(0).normal(0, 0.1, (40, 5))
    with pytest.warns(RuntimeWarning):
        res = value_curves(vframe(V, xs), "x", "v", n_boot=100)
    assert any(c.startswith("chi peaks at an end") for c in res.caveats)


# --- invariants: accounting, cluster resampling, exact cases ----------------------

@pytest.mark.filterwarnings("ignore:.*items are not scored:RuntimeWarning")
def test_c6_an_item_failing_everywhere_is_counted():
    """C6.  Four items at x = 1, 2.  Item i3 is an error row at both points.
    It is planned and absent from the paired analysis, exactly like an item
    missing at one point, so n_dropped = 1 (three items analysed)."""
    oks = [[1], [0], [1, 1], [1, 0]]
    tf = frame({1.0: oks, 2.0: oks}, status={3: "error"})
    assert len(axis_data(tf, "x").items) == 3
    assert axis_data(tf, "x").n_dropped == 1
    V = np.array([[0.0, 1], [1, 2], [2, 3], [3, 4]])
    vd = value_data(vframe(V, [1.0, 2.0], status={3: "error"}), "x", "v")
    assert vd.n_dropped == 1


def test_c22_value_curves_resample_whole_clusters():
    """C22.  Duplicate every item into two rows of one cluster (ids c000a,
    c000b in cluster c000).  The cluster means, and so the mean curve, are
    unchanged.  A bootstrap over the 150 clusters draws the same cluster
    weights from the same seed, and doubling every weight cancels in
    sum w V / sum w.  So every replicate of the twin table equals the
    single table's.  Resampling items instead would draw 300 independent
    weights and shrink the spread by ~1/sqrt(2).  Tolerance 1e-12
    relative: the sums run over 300 terms instead of 150."""
    xs = np.geomspace(1, 100, 5)
    rng = np.random.default_rng(3)
    V = np.log(xs)[None, :] + rng.normal(0, 1, (150, 1)) + rng.normal(0, .3, (150, 5))
    ids = [f"c{i:03d}" for i in range(150)]
    single = value_curves(vframe(V, xs, items=ids), "x", "v", n_boot=400, seed=0)
    twin_ids = [f"{c}{s}" for c in ids for s in "ab"]
    twin = value_curves(vframe(np.repeat(V, 2, axis=0), xs, items=twin_ids,
                               cluster=[c for c in ids for _ in "ab"]),
                        "x", "v", n_boot=400, seed=0)
    assert_same(twin.reps["mean"], single.reps["mean"], rtol=1e-12)
    assert_same(twin.table, single.table, rtol=1e-12)


def test_c25_phi_lc_interval_width_matches_the_sampling_spread():
    """C25.  The bootstrap sd of phi_lc must estimate its true sampling sd.
    Truth by simulation: R = 200 independent datasets of 300 items from one
    process (odd items length 1 with p = 0.6, even items length 2 with
    c2 = c1 half the time, so phi_2 = Cov(c1, c2) = 0.5 Var(c1) = 0.105).
    The across-dataset sd of the point phi_lc has relative SE
    1/sqrt(2(R - 1)) = 0.050, while the mean over datasets of the replicate
    sd (B = 200 each) has a much smaller one.  15% is 3 SE."""
    R, k, pts, sds = 200, 300, [], []
    for s in range(R):
        rng = np.random.default_rng(7000 + s)
        oks = []
        for i in range(k):
            if i % 2:
                oks.append([int(rng.random() < 0.6)])
            else:
                c1 = int(rng.random() < 0.7)
                c2 = c1 if rng.random() < 0.5 else int(rng.random() < 0.5)
                oks.append([c1, c2])
        res = axis_curves(frame({1.0: oks, 2.0: oks}), "x", n_boot=200, seed=s)
        pts.append(res.table["phi_lc"].iloc[0])
        sds.append(res.reps["phi_lc"][:, 0].std(ddof=1))
    assert np.mean(sds) == pytest.approx(np.std(pts, ddof=1), rel=0.15)


def test_c26_linear_axis_chi_is_exact():
    """C26.  On a linear axis chi = dA/dx.  numpy's non-uniform centred
    difference is exact for a linear function (second order in the
    interior, and the one-sided ends are exact for a line), so A = 0.01 x
    gives chi = 0.01 at every point, on a grid including x = 0 where the log
    form is undefined.  Through the full analyses: 100 single-token items,
    item i right at x iff i < x, so A(x) = x/100 exactly on x in
    {0, 10, 30, 70}, plus a per-item value 0.01 x.  Tolerance 1e-12
    absolute: rounding in differences of O(1) numbers."""
    x = np.array([0.0, 1, 3, 7, 20])
    assert_same(susceptibility(x, 0.01 * x, log=False), np.full(5, 0.01),
                atol=1e-12)
    xs = [0.0, 10.0, 30.0, 70.0]
    oks = {xv: [[int(i < xv)] for i in range(100)] for xv in xs}
    t = axis_curves(frame(oks), "x", n_boot=50, log_x=False).table
    assert_same(t["A_obs"].to_numpy(), np.array(xs) / 100, atol=1e-12)
    assert_same(t["chi_obs"].to_numpy(), np.full(4, 0.01), atol=1e-12)
    V = np.tile(0.01 * np.array(xs), (25, 1))
    vt = value_curves(vframe(V, xs), "x", "v", n_boot=50, log_x=False).table
    assert_same(vt["chi"].to_numpy(), np.full(4, 0.01), atol=1e-12)


def test_c28_axis_curves_reports_the_nan_replicate_share():
    """C28.  40 items, one of length 3 and the rest length 1.  phi_l3 is
    undefined (NaN) in every replicate that does not draw that item:
    probability (1 - 1/40)^40 = 0.3632.  The interval of phi_l3 then
    describes only 64% of replicates, so the share must be reported, as
    value_curves does for a crossing (`<name>_censored`).  B = 1000:
    binomial SE of the share is 0.015, so tolerance 0.05 is 3.3 SE."""
    rng = np.random.default_rng(0)
    oks = [[int(rng.random() < 0.5)] for _ in range(39)] + [[1, 0, 1]]
    res = axis_curves(frame({1.0: oks, 2.0: oks}), "x", n_boot=1000)
    expect = (1 - 1 / 40) ** 40
    assert "phi_l3_censored" in res.table.columns, "NaN share not reported"
    assert res.table["phi_l3_censored"].iloc[0] == pytest.approx(expect, abs=0.05)
