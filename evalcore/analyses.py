"""Analyses: what the `analyse` stage computes from the results frame.

An analysis is any function

    analysis(frame, ctx) -> DataFrame | {name: DataFrame}

where `frame` holds every row of the run (all statuses) and `ctx` says the
grid axes, the manifest and the sink.  An Experiment takes a mapping
{name: analysis}; each table is written to `<sink>.analysis/<name>.csv`
(a dict result to `<name>.<key>.csv`).  The factories below build the
usual ones; write your own for anything else, or pass `None` for the
analyse stage to skip it.

Only `ok` rows carry scores.  A summary is still handed every row, so
that the rows lost before scoring (error, truncated, unparsed) are counted
in its `n_excluded` and trip the excluded-rows caveat; handing it the ok
rows alone would make a cell that lost a third of its rows look whole.
`attrition` says which state each lost row is in.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TypeAlias

import pandas as pd

from .evaluation.attrition import attrition
from .evaluation.curves import AxisCurves, axis_curves, value_curves
from .evaluation.retrieval import retrieval_summary
from .evaluation.stats import summarize

__all__ = ["Analysis", "AnalysisContext", "attrition_table", "axis_curve",
           "retrieval_table", "summary", "value_curve"]

Tables: TypeAlias = "pd.DataFrame | Mapping[str, pd.DataFrame]"


@dataclass(frozen=True)
class AnalysisContext:
    by: list[str]           # the grid axes (["task"] for a one-cell grid)
    manifest: Path          # every planned unit, for attrition
    out: Path               # the sink


Analysis: TypeAlias = Callable[[pd.DataFrame, AnalysisContext], Tables]


def _ok(frame: pd.DataFrame) -> pd.DataFrame:
    return frame[frame["status"] == "ok"] if "status" in frame else frame


def attrition_table() -> Analysis:
    """Per cell: planned, ok, error, tool_error, truncated, unparsed, missing."""
    def run(frame: pd.DataFrame, ctx: AnalysisContext) -> Tables:
        return attrition(frame, ctx.by, manifest=ctx.manifest)
    return run


def summary(metric: str, by: Sequence[str] | None = None,
            **summarize_kw: Any) -> Analysis:
    """The cluster-bootstrap summary of `metric` per cell (stats.summarize;
    keyword arguments pass through: n_boot, method, label, floor, ...)."""
    def run(frame: pd.DataFrame, ctx: AnalysisContext) -> Tables:
        if metric not in frame:
            raise KeyError(f"summary: no column {metric!r} in the results; "
                           f"columns are {sorted(map(str, frame.columns))}")
        return summarize(frame, metric, list(by or ctx.by), **summarize_kw)
    return run


def retrieval_table(metrics: Sequence[str] | None = None,
                    by: Sequence[str] | None = None,
                    **summarize_kw: Any) -> Analysis:
    """Every ranking metric per cell (evaluation.retrieval_summary)."""
    def run(frame: pd.DataFrame, ctx: AnalysisContext) -> Tables:
        return retrieval_summary(frame, list(by or ctx.by), metrics,
                                 **summarize_kw)
    return run


def _per_group(frame: pd.DataFrame, ctx: AnalysisContext, x: str,
               fit: Callable[[pd.DataFrame], AxisCurves]) -> Tables:
    """Curves along grid axis `x`, one set per combination of the other
    axes, stacked with those axes as columns."""
    if x not in ctx.by:
        raise ValueError(f"curves: {x!r} is not a grid axis ({ctx.by})")
    rest = [a for a in ctx.by if a != x]
    groups = frame.groupby(rest, sort=True) if rest else [((), frame)]
    tables: list[pd.DataFrame] = []
    peaks: list[pd.DataFrame] = []
    for key, sub in groups:
        res = fit(sub)
        labels = dict(zip(rest, key if isinstance(key, tuple) else (key,),
                          strict=True))
        tables.append(res.table.assign(**labels))
        peaks.append(res.peak.assign(**labels,
                                     caveats="; ".join(res.caveats)))
    return {"table": pd.concat(tables, ignore_index=True),
            "peak": pd.concat(peaks, ignore_index=True)}


def value_curve(x: str, value: str, levels: Sequence[float] = (),
                **curve_kw: Any) -> Analysis:
    """`value` along grid axis `x` with chi, its peak and level crossings
    (evaluation.value_curves), per combination of the other axes."""
    def run(frame: pd.DataFrame, ctx: AnalysisContext) -> Tables:
        return _per_group(_ok(frame), ctx, x, lambda d: value_curves(
            d, x, value, levels=levels, **curve_kw))
    return run


def axis_curve(x: str, **curve_kw: Any) -> Analysis:
    """Teacher-forced accuracy along grid axis `x` with its composition
    (evaluation.axis_curves), per combination of the other axes."""
    def run(frame: pd.DataFrame, ctx: AnalysisContext) -> Tables:
        return _per_group(_ok(frame), ctx, x, lambda d: axis_curves(
            d, x, **curve_kw))
    return run

