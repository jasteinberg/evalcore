"""Per-cell accounting of a plan: what was completed, errored, cut off,
unparsed, or never attempted."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import pandas as pd

from ..core.records import _status, read_manifest

_RANK = {"ok": 0, "unparsed": 1, "truncated": 2, "tool_error": 3,
         "error": 4}


def attrition(df: pd.DataFrame, by: Sequence[str],
              manifest: str | Path | pd.DataFrame | None = None
              ) -> pd.DataFrame:
    """Per-cell accounting of the plan.  Report this next to every metric
    table; a cell whose failures correlate with item difficulty has a biased
    mean even if the surviving rows are scored perfectly.

    Without `manifest` this tabulates what came back -- `n_rows`, `n_err`,
    `n_truncated`, `n_unparsed`, `err_rate` -- which is the older and weaker
    question, because the denominator is the survivors.  A cell that died
    before its first call contributes no rows and therefore does not appear
    at all: the table is silent about it, and silence reads as "nothing to
    report".

    With `manifest` (the file `execute` wrote, or the frame `read_manifest`
    returns) every planned unit falls in exactly one of six states, and
    the identity holds per cell and in total:

        n_planned = n_completed + n_err + n_tool_error + n_truncated
                    + n_unparsed + n_missing

    `n_completed` counts units with a scored row.  `n_truncated` and
    `n_unparsed` were answered but carry no metric (see the module
    docstring), so they are attrition from the metric's point of view while
    being distinct from both a failed call and a unit nobody attempted.
    `n_missing` -- planned, never attempted -- is the category that a kill,
    an out-of-quota, or a `limit=` left over from a debugging session all
    land in, and the only one the frame cannot show.

    Every count except `n_rows` is a count of UNITS, and each planned unit is
    assigned its state once, from the plan side: absent from the frame is
    missing; otherwise the worst status among its rows, ranked
    error > tool_error > truncated > unparsed > ok.  Counting rows instead
    breaks the identity on any frame carrying more than one row per unit --
    a judge
    panel with three verdicts per response is the case that will arrive --
    where it reports err_rate = 2.0 and a negative `n_completed`.  `n_rows`
    is kept as the frame's own row count precisely so that multiplicity
    stays visible: for the one-row-per-unit frames `to_frame` produces, and
    only for those,
    `n_rows = n_completed + n_err + n_tool_error + n_truncated + n_unparsed
    + n_unplanned`.

    Two staleness cases, both non-fatal by construction.  A manifest that is
    missing, empty, or unreadable degrades to the no-manifest columns.  A
    manifest that is older than the sink (rows in the frame it never
    planned) is reported as `n_unplanned` rather than being treated as
    negative attrition -- the counts stay non-negative and the discrepancy
    stays visible.

    Pass the UNFILTERED frame: rows dropped by `to_frame(drop_errors=True)`
    are indistinguishable from units never attempted, and would be counted
    as missing.

    `by` must name columns the manifest also carries, i.e. cell params, tags,
    or item identity -- never a scored column.  A never-attempted unit has no
    score by definition, so grouping the plan by one is not a table that can
    be filled in.
    """
    by = list(by)
    st = _status(df)
    df = df.assign(_err=st.eq("error"), _tool=st.eq("tool_error"),
                   _trunc=st.eq("truncated"), _unp=st.eq("unparsed"),
                   _rank=st.map(_RANK))

    plan = (manifest if isinstance(manifest, pd.DataFrame)
            else read_manifest(manifest) if manifest is not None
            else pd.DataFrame())
    if plan.empty or "unit_id" not in plan:
        return (df.groupby(by, observed=True, dropna=False)
                  .agg(n_rows=("_err", "size"), n_err=("_err", "sum"),
                       n_tool_error=("_tool", "sum"),
                       n_truncated=("_trunc", "sum"),
                       n_unparsed=("_unp", "sum"))
                  .assign(err_rate=lambda t: t.n_err / t.n_rows)
                  .reset_index())

    absent = [c for c in by if c not in plan.columns]
    if absent:
        raise ValueError(
            f"manifest has no column(s) {absent}; group attrition by cell "
            f"params, tags or item identity -- a unit that was never "
            f"attempted has no scored columns to group by.")

    if df.empty or "unit_id" not in df.columns:
        # The limiting case, and the one worth getting right: a sweep that
        # died before its first call has no frame at all.  Every planned
        # unit is missing, which is a full accounting -- whereas the frame
        # alone cannot even name the columns to group by.
        df = pd.DataFrame(columns=[*by, "unit_id", "_rank"], dtype=object)

    # State is decided per planned unit, on the plan side, so the five
    # categories partition it by construction whatever the frame's row
    # multiplicity.  NaN = never seen; otherwise the worst row's rank.
    state = df.groupby("unit_id", observed=True)["_rank"].max()
    p = plan.assign(_st=plan["unit_id"].map(state))
    p = p.assign(_missing=p["_st"].isna(), _done=p["_st"].eq(0),
                 _unp=p["_st"].eq(1), _trunc=p["_st"].eq(2),
                 _tool=p["_st"].eq(3), _err=p["_st"].eq(4))
    gp = p.groupby(by, observed=True, dropna=False)
    unplanned = df[~df["unit_id"].isin(set(plan["unit_id"]))]

    parts = [gp.size().rename("n_planned"),
             gp["_done"].sum().rename("n_completed"),
             gp["_err"].sum().rename("n_err"),
             gp["_tool"].sum().rename("n_tool_error"),
             gp["_trunc"].sum().rename("n_truncated"),
             gp["_unp"].sum().rename("n_unparsed"),
             gp["_missing"].sum().rename("n_missing"),
             df.groupby(by, observed=True, dropna=False).size().rename(
                 "n_rows"),
             unplanned.groupby(by, observed=True, dropna=False)["unit_id"]
             .nunique().rename("n_unplanned")]
    idx = parts[0].index
    for s in parts[1:]:
        idx = idx.union(s.index)
    out = pd.concat([s.reindex(idx) for s in parts],
                    axis=1).fillna(0).astype(int)
    # Rates take the plan as denominator wherever there is one; a cell known
    # only to the frame (stale manifest) gets NaN rather than a made-up 0.
    plan_n = out.n_planned.where(out.n_planned > 0)
    att_n = (out.n_planned - out.n_missing).where(
        out.n_planned - out.n_missing > 0)
    out["err_rate"] = out.n_err / att_n
    out["missing_rate"] = out.n_missing / plan_n
    out["attrition_rate"] = 1.0 - out.n_completed / plan_n
    cols = ["n_planned", "n_rows", "n_completed", "n_err", "n_tool_error",
            "n_truncated",
            "n_unparsed", "n_missing", "n_unplanned", "err_rate",
            "missing_rate", "attrition_rate"]
    return out[cols].reset_index()
