"""Coverage of a nominal 95% interval against the number of clusters k.

Simulates X_ij = mu + a_i + e_ij (k clusters of m Gaussian rows, intraclass
correlation rho = s2_a / (s2_a + s2_e)), builds three intervals per dataset
and counts how often each contains mu:

  cluster bootstrap, BCa         (what `summarize` reports)
  cluster bootstrap, percentile
  row bootstrap, percentile      (ignores the clustering; covers less by
                                  the factor DEFF = 1 + (m-1) rho in variance)

Prints the table and writes a figure with every measured value labelled
and binomial 1-SE bars, sqrt(c(1-c)/R).  These numbers are the source of
SMALL_K_COVERAGE in evaluation/stats.py and of the README's tables.

    python scripts/coverage_figure.py --out docs/coverage.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

from evalcore.evaluation.stats import cluster_bootstrap

METHODS = ("cluster BCa", "cluster percentile", "row percentile")


def draw(k: int, m: int, rho: float, rng: np.random.Generator) -> pd.DataFrame:
    """k clusters of m rows, unit total variance, mean 0."""
    a = rng.normal(0, np.sqrt(rho), size=k)
    e = rng.normal(0, np.sqrt(1 - rho), size=(k, m))
    return pd.DataFrame({"cluster": np.repeat(np.arange(k), m),
                         "row": np.arange(k * m),
                         "y": (a[:, None] + e).ravel()})


def covers(df: pd.DataFrame, n_boot: int, seed: int) -> dict[str, bool]:
    out = {}
    for name, cl, method in (("cluster BCa", "cluster", "bca"),
                             ("cluster percentile", "cluster", "percentile"),
                             ("row percentile", "row", "percentile")):
        r = cluster_bootstrap(df, "y", cluster=cl, n_boot=n_boot, seed=seed,
                              method=method)
        out[name] = bool(r.lo <= 0.0 <= r.hi)
    return out


def measure(ks: list[int], m: int, rho: float, R: int, n_boot: int,
            seed: int) -> pd.DataFrame:
    rows = []
    for k in ks:
        hits = dict.fromkeys(METHODS, 0)
        for s in range(R):
            rng = np.random.default_rng([seed, k, s])
            for name, ok in covers(draw(k, m, rho, rng), n_boot, s).items():
                hits[name] += ok
        for name in METHODS:
            c = hits[name] / R
            rows.append({"k": k, "method": name, "coverage": c,
                         "se": np.sqrt(c * (1 - c) / R)})
        print(f"k={k:3d}  " + "  ".join(
            f"{n} {hits[n] / R:.3f}" for n in METHODS), flush=True)
    return pd.DataFrame(rows)


def row_limit(m: int, rho: float) -> float:
    """Large-k coverage of the row bootstrap: its variance is Var/DEFF, so
    the nominal +-1.96 sd interval spans +-1.96/sqrt(DEFF) true sd."""
    return float(2 * norm.cdf(1.959964 / np.sqrt(1 + (m - 1) * rho)) - 1)


def figure(res: pd.DataFrame, m: int, rho: float, R: int, n_boot: int,
           path: Path) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.4))
    style = {"cluster BCa": ("#2a6fdb", "o"),
             "cluster percentile": ("#8a8a8a", "s"),
             "row percentile": ("#d1495b", "^")}
    ax.axhline(0.95, color="black", lw=0.8, ls="--")
    ax.text(res.k.min(), 0.957, "nominal 0.95", ha="left", va="bottom",
            fontsize=8)
    lim = row_limit(m, rho)
    ax.axhline(lim, color=style["row percentile"][0], lw=0.8, ls=":")
    ax.text(4.0, lim + 0.007,
            r"row bootstrap as $k\to\infty$: "
            rf"$2\Phi(1.96/\sqrt{{\mathrm{{DEFF}}}}) - 1 = {lim:.3f}$",
            ha="left", va="bottom", fontsize=8,
            color=style["row percentile"][0])
    for name in METHODS:
        d = res[res.method == name]
        color, marker = style[name]
        ax.errorbar(d.k, d.coverage, yerr=d.se, color=color, marker=marker,
                    lw=1.6, ms=6, capsize=2, label=name)
        if name != "cluster percentile":
            for k, c in zip(d.k, d.coverage, strict=True):
                ax.annotate(f"{c:.3f}", (k, c), textcoords="offset points",
                            xytext=(0, -15), ha="center", fontsize=8,
                            color=color)
    ax.set_xscale("log")
    ax.set_xticks(sorted(res.k.unique()))
    ax.set_xticklabels([str(k) for k in sorted(res.k.unique())])
    ax.minorticks_off()
    ax.set_xlabel("clusters k")
    ax.set_ylabel("coverage of a nominal 95% interval")
    deff = 1 + (m - 1) * rho
    ax.set_title(f"m={m} rows per cluster, rho={rho}, DEFF={deff:.1f}; "
                 f"R={R} datasets per k, B={n_boot}; bars 1 binomial SE",
                 fontsize=9)
    ax.set_ylim(0.45, 1.0)
    ax.grid(alpha=0.25)
    ax.legend(loc="lower right", fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--out", type=Path, required=True,
                   help="figure path; the table goes beside it as .csv")
    p.add_argument("--ks", type=int, nargs="+",
                   default=[3, 5, 10, 20, 30, 40])
    p.add_argument("--m", type=int, default=6)
    p.add_argument("--rho", type=float, default=0.6)
    p.add_argument("--R", type=int, default=1000)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--replot", action="store_true",
                   help="redraw from the .csv beside --out, no simulation "
                        "(pass the same --m --rho --R --n-boot it was run with)")
    a = p.parse_args()
    a.out.parent.mkdir(parents=True, exist_ok=True)
    if a.replot:
        res = pd.read_csv(a.out.with_suffix(".csv"))
    else:
        res = measure(a.ks, a.m, a.rho, a.R, a.n_boot, a.seed)
        res.to_csv(a.out.with_suffix(".csv"), index=False)
    figure(res, a.m, a.rho, a.R, a.n_boot, a.out)
    print(f"wrote {a.out} and {a.out.with_suffix('.csv')}")


if __name__ == "__main__":
    main()
