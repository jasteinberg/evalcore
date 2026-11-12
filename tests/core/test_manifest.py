"""The plan manifest: what a sweep intended, not just what it returned.

The failure this guards against is silent by construction.  `attrition`
without a manifest counts errors among the rows that came back, so a sweep
killed partway reports a clean table: every row it has is fine.  The units it
never attempted contribute nothing, and a cell that died before its first
call does not appear in the table at all.

That loss is not missing-at-random.  Units are dispatched cell-major, so a
kill at fraction f of the plan removes the LAST cells entirely rather than a
uniform 1-f of each -- the missingness is perfectly confounded with the grid
axis the table is meant to contrast.  These tests check the accounting
identity `n_planned = n_completed + n_err + n_missing` holds cell by cell,
and that the blind table is the one that cannot see it.
"""

import io
import os

import pandas as pd
import pytest
from support.runner import Dies, items, run

from evalcore import (
    EchoBackend,
    attrition,
    grid,
    manifest_path,
    read_manifest,
    to_frame,
    units,
    write_manifest,
)


def sweep(tmp_path, models=("m1", "m2", "m3"), n_items=8, alive=10,
          fail_on=frozenset(), on_call=None, **kw):
    """Run a cell-major sweep that dies after `alive` calls."""
    cells = grid({"model": list(models)}, tags={"condition": "base"})
    out = tmp_path / "runs.jsonl"
    be = Dies(alive, fail_on, on_call)
    with pytest.raises(KeyboardInterrupt):
        run(list(units(cells, items(n_items))), be, out, **kw)
    return out, be, len(cells) * n_items


def test_manifest_exists_in_full_before_the_first_call(tmp_path):
    """Written before execution, not as execution proceeds: a kill on call
    zero must still leave the whole plan on disk."""
    seen = {}

    def check(call_idx):
        if call_idx == 0:
            seen["plan"] = read_manifest(
                manifest_path(tmp_path / "runs.jsonl"))

    out, be, planned = sweep(tmp_path, alive=0, on_call=check)
    assert be.calls == 0
    assert len(seen["plan"]) == planned            # all 24, before call one
    assert {"unit_id", "cell_id", "item_id"} <= set(seen["plan"].columns)
    assert not out.read_text().strip()             # and no rows written yet


def test_a_sweep_that_died_before_its_first_call_is_still_accountable(
        tmp_path):
    """The limiting case: no rows at all.  The frame cannot even name the
    cells, so the blind table is not merely incomplete -- it does not
    exist.  The manifest still gives a complete accounting."""
    out, _be, planned = sweep(tmp_path, alive=0)
    df = to_frame(out)
    assert df.empty

    full = attrition(df, ["model"], manifest=manifest_path(out))
    assert set(full["model"]) == {"m1", "m2", "m3"}
    assert full["n_planned"].sum() == planned
    assert full["n_missing"].sum() == planned
    assert full["n_completed"].sum() == 0 and full["n_rows"].sum() == 0
    assert (full["missing_rate"] == 1.0).all()


def test_manifest_joins_to_the_frame_on_unit_id(tmp_path):
    cells = grid({"model": ["m1", "m2"]}, tags={"condition": "base"})
    out = tmp_path / "runs.jsonl"
    df = run(list(units(cells, items(6))), EchoBackend(), out)
    plan = read_manifest(manifest_path(out))

    assert len(plan) == len(df) == 12
    j = plan.merge(df, on="unit_id", suffixes=("_plan", ""))
    assert len(j) == 12
    assert (j["cell_id_plan"] == j["cell_id"]).all()
    assert (j["item_id_plan"] == j["item_id"]).all()
    assert (j["model_plan"] == j["model"]).all()


def test_kill_partway_is_invisible_without_the_manifest(tmp_path):
    """The headline case.  Nine of twenty-four units came back, all clean.

    Blind, that is a table with no errors and no third model.  With the
    manifest it is a table that says fifteen units were never attempted and
    one whole cell never started.
    """
    out, _be, planned = sweep(tmp_path, alive=9)
    df = to_frame(out)
    assert len(df) == 9 and df["error"].isna().all()

    blind = attrition(df, ["model"])
    assert list(blind.columns) == ["model", "n_rows", "n_err", "n_tool_error",
                                   "n_truncated", "n_unparsed", "err_rate"]
    assert blind["n_err"].sum() == 0                 # nothing looks wrong
    assert set(blind["model"]) == {"m1", "m2"}       # m3 is simply absent

    full = attrition(df, ["model"], manifest=manifest_path(out))
    assert set(full["model"]) == {"m1", "m2", "m3"}
    assert full["n_planned"].sum() == planned == 24
    assert full["n_missing"].sum() == 15
    assert full["n_completed"].sum() == 9
    m3 = full.set_index("model").loc["m3"]
    assert m3["n_planned"] == 8 and m3["n_missing"] == 8
    assert m3["n_completed"] == 0 and m3["missing_rate"] == 1.0
    assert pd.isna(m3["err_rate"])                   # nothing was attempted


def test_accounting_identity_holds_cell_by_cell(tmp_path):
    """planned = completed + errored + never attempted, exactly, per cell.

    Error rows and never-attempted units are different failures with
    different remedies -- a rerun fixes one, only a longer budget fixes the
    other -- so they get separate columns and must not sum into each other.
    """
    out, _be, planned = sweep(tmp_path, alive=14,
                             fail_on={"q1", "q3", "q5"})
    df, plan = to_frame(out), read_manifest(manifest_path(out))
    full = attrition(df, ["model"], manifest=plan)

    assert (full["n_planned"] ==
            full["n_completed"] + full["n_err"] + full["n_truncated"]
            + full["n_unparsed"] + full["n_missing"]).all()
    assert (full["n_rows"] ==
            full["n_completed"] + full["n_err"] + full["n_truncated"]
            + full["n_unparsed"] + full["n_unplanned"]).all()
    assert full["n_planned"].sum() == planned
    assert full["n_err"].sum() == 6                  # 3 items x 2 live cells
    assert full["n_missing"].sum() == planned - 14
    assert full["n_unplanned"].sum() == 0
    got = full.set_index("model")
    assert got.loc["m1", "err_rate"] == pytest.approx(3 / 8)
    assert got.loc["m1", "attrition_rate"] == pytest.approx(3 / 8)
    assert got.loc["m2", "attrition_rate"] == pytest.approx(1 - 3 / 8)


def test_missing_or_corrupt_manifest_degrades_to_prior_behaviour(tmp_path):
    """A record of intent, not a second source of truth: losing it costs
    information and never correctness."""
    out, _be, _planned = sweep(tmp_path, alive=9)
    df = to_frame(out)
    base = attrition(df, ["model"])

    absent = attrition(df, ["model"], manifest=tmp_path / "gone.jsonl")
    pd.testing.assert_frame_equal(base, absent)

    empty = attrition(df, ["model"], manifest=pd.DataFrame())
    pd.testing.assert_frame_equal(base, empty)

    # a half-written last line survives as the records that do parse
    mp = manifest_path(out)
    keep = mp.read_text().splitlines()[:6]
    mp.write_text("\n".join(keep) + '\n{"unit_id": "trunc')
    with pytest.warns(RuntimeWarning, match="1 manifest line"):
        assert len(read_manifest(mp)) == 6
    with pytest.warns(RuntimeWarning, match="1 manifest line"):
        partial = attrition(df, ["model"], manifest=mp)
    assert partial["n_planned"].sum() == 6


def test_stale_manifest_reports_unplanned_rows_not_negative_attrition(
        tmp_path):
    """A manifest older than the sink names fewer units than the frame has.
    Those rows are surfaced as `n_unplanned`.  Nothing goes negative and no
    missingness is invented."""
    cells = grid({"model": ["m1", "m2"]})
    out = tmp_path / "runs.jsonl"
    all_units = list(units(cells, items(6)))
    run(all_units, EchoBackend(), out)

    stale = write_manifest(all_units[:4], tmp_path / "stale.manifest.jsonl")
    full = attrition(to_frame(out), ["model"], manifest=stale)

    assert full["n_planned"].sum() == 4
    assert full["n_missing"].sum() == 0
    assert full["n_unplanned"].sum() == 8
    assert (full[["n_planned", "n_rows", "n_completed", "n_err",
                  "n_missing", "n_unplanned"]] >= 0).all().all()


def test_manifest_unions_across_resumes_so_a_narrowed_rerun_still_reports(
        tmp_path, capfd):
    """The plan is per sink, not per invocation.

    Sweep one plans both conditions and dies inside the first.  Sweep two is
    narrowed to the condition someone was debugging, finishes clean, and would
    otherwise report a perfect table -- the second condition having quietly left
    the plan.  The union keeps it, and `run` says so on the way out.
    """
    cells = grid({"model": ["m1", "m2"]})
    its, out = items(6), tmp_path / "runs.jsonl"
    with pytest.raises(KeyboardInterrupt):
        run(list(units(cells, its)), Dies(alive=4), out)

    narrow = list(units(cells[:1], its))             # only the debugged condition
    log = io.StringIO()
    df = run(narrow, EchoBackend(), out, stream=log)
    assert len(df) == 6                              # condition one, complete
    assert "6 planned units were never attempted" in log.getvalue()

    # stream=None: no progress lines, but the warning still reaches stderr
    out2 = tmp_path / "quiet.jsonl"
    with pytest.raises(KeyboardInterrupt):
        run(list(units(cells, its)), Dies(alive=4), out2)
    capfd.readouterr()
    run(narrow, EchoBackend(), out2, stream=None)
    err = capfd.readouterr()
    assert "6 planned units were never attempted" in err.err
    assert "[run] done:" not in err.err + err.out
    assert "units, " not in err.err + err.out        # the plan line

    full = attrition(to_frame(out), ["model"], manifest=manifest_path(out))
    assert full["n_planned"].sum() == 12
    got = full.set_index("model")
    assert got.loc["m1", "n_completed"] == 6
    assert got.loc["m1", "n_missing"] == 0
    assert got.loc["m2", "n_completed"] == 0
    assert got.loc["m2", "n_missing"] == 6      # the condition nobody re-ran


def test_resume_reads_the_jsonl_not_the_manifest(tmp_path):
    """Deleting the manifest must not cost work.  Resume is by unit_id in the
    sink.  The manifest is rebuilt and no request is re-dispatched."""
    cells = grid({"model": ["m1"]})
    its, out = items(6), tmp_path / "runs.jsonl"
    run(list(units(cells, its)), EchoBackend(), out)
    manifest_path(out).unlink()

    class Tripwire(EchoBackend):
        """Same identity as the backend that wrote the rows (so resume has
        no reason to redo them), but any call is a kill."""
        calls = 0

        def complete(self, req):
            self.calls += 1
            raise KeyboardInterrupt("resume dispatched a finished unit")

    be = Tripwire()
    df = run(list(units(cells, its)), be, out)
    assert be.calls == 0 and len(df) == 6
    assert len(read_manifest(manifest_path(out))) == 6


def test_manifest_append_after_a_truncated_line_keeps_every_unit(tmp_path):
    us = list(units(grid({"model": ["m"]}), items(6)))
    mp = tmp_path / "plan.manifest.jsonl"
    write_manifest(us[:3], mp)
    keep = mp.read_text().splitlines()
    mp.write_text("\n".join(keep[:2]) + "\n" + keep[2][:30])     # crash
    with pytest.warns(RuntimeWarning, match="1 manifest line"):
        write_manifest(us, mp)
    with pytest.warns(RuntimeWarning, match="1 manifest line"):
        assert set(read_manifest(mp)["unit_id"]) == {u.id for u in us}


def test_each_manifest_record_is_one_write(tmp_path, monkeypatch):
    """The concurrency guarantee in mechanism form, so it does not depend on
    a race showing up: every record reaches the OS as exactly one write of
    one whole, newline-terminated line.  A buffered writer (the previous
    implementation) flushes in chunks that need not align with records."""
    import json

    import evalcore.core.records as records_mod
    payloads: list[bytes] = []
    real = os.write

    def spy(fd, data):
        payloads.append(bytes(data))
        return real(fd, data)
    monkeypatch.setattr(records_mod.os, "write", spy)
    us = list(units(grid({"model": ["m1", "m2"]}), items(60)))
    write_manifest(us, tmp_path / "p.manifest.jsonl")
    assert len(payloads) == len(us) == 120
    for data in payloads:
        assert data.endswith(b"\n") and data.count(b"\n") == 1
        assert json.loads(data)["unit_id"]


def test_manifest_can_be_disabled_and_run_is_unchanged(tmp_path):
    cells = grid({"model": ["m1"]})
    out = tmp_path / "runs.jsonl"
    df = run(list(units(cells, items(4))), EchoBackend(), out, manifest=False)
    assert len(df) == 4
    assert not manifest_path(out).exists()


def test_grouping_the_plan_by_a_scored_column_is_refused(tmp_path):
    """A never-attempted unit has no score, so there is no honest cell to put
    it in.  Better a named error than a table with a silently dropped
    category."""
    out, _be, _planned = sweep(tmp_path, alive=9)
    with pytest.raises(ValueError, match="answer"):
        attrition(to_frame(out), ["answer"], manifest=manifest_path(out))


def test_manifest_writes_are_append_only_and_concurrent_safe(tmp_path,
                                                             monkeypatch):
    """Two writers must not be able to erase each other's plan.

    The cache learned this once already (`test_cache_put_is_concurrent_safe`):
    a read-modify-write under `os.replace` both raced on the temp file and
    lost the loser's records outright.  For a manifest the loss is worse than
    a crash -- the units the erased plan intended are then reported as never
    planned rather than never attempted, which is the one error this file
    exists to prevent.  Appends are additive, and duplicates are deduplicated
    on read rather than being written twice.

    The in-process lock is replaced by a no-op: separate processes do not
    share it, so with it in place a read-modify-write implementation would
    pass this test too.  Without it, writers may append the same unit twice
    (allowed, deduplicated on read) but must never lose one.
    """
    import concurrent.futures as cf
    import contextlib
    import json

    import evalcore.core.records as records_mod
    monkeypatch.setattr(records_mod, "_MANIFEST_LOCK", contextlib.nullcontext())

    its = items(50)
    plans = [list(units(grid({"model": [f"m{k}"]}), its)) for k in range(8)]
    mp, errs = tmp_path / "race.manifest.jsonl", []

    def write(plan):
        try:
            write_manifest(plan, mp)
        except Exception as exc:  # noqa: BLE001  # pragma: no cover - collects the bug
            errs.append(f"{type(exc).__name__}: {exc}")

    with cf.ThreadPoolExecutor(16) as pool:
        list(pool.map(write, plans + plans))       # every plan written twice

    got = read_manifest(mp)
    assert errs == []
    assert len(got) == 400                         # nobody's plan was erased
    assert sorted(got["model"].unique()) == [f"m{k}" for k in range(8)]
    # none torn: every record line parses.  A blank line can appear (a
    # writer's crash-fragment terminator seeing another's write in
    # progress, see _write_line), and readers skip it.
    lines = mp.read_text().splitlines()
    assert all(json.loads(x)["unit_id"] for x in lines if x.strip())
    # and re-planning does not grow the file
    write_manifest(plans[0], mp)
    assert len(mp.read_text().splitlines()) == len(lines)


def test_a_unit_is_counted_once_however_many_rows_it_has(tmp_path):
    """The partition is over units, not rows.

    A judge panel puts three verdicts against one response, so the frame
    carries three rows per unit.  Counting error ROWS against a plan counted
    in units gives err_rate above 1 and a negative n_completed -- a table
    that is not merely imprecise but arithmetically impossible.  State is
    decided per planned unit: any error row makes the unit errored.
    """
    cells = grid({"model": ["m1", "m2"]})
    its = items(3)
    plan = write_manifest(units(cells, its), tmp_path / "m.jsonl")

    rows = []
    for k, u in enumerate(list(units(cells, its))[:3]):     # condition m1 only
        r = u.row()
        r["error"] = "judge: refused" if k < 2 else None
        rows += [dict(r, judge="a"), dict(r, judge="b")]
    full = attrition(pd.DataFrame(rows), ["model"], manifest=plan)

    assert (full["n_planned"] ==
            full["n_completed"] + full["n_err"] + full["n_missing"]).all()
    assert (full[["n_completed", "n_err", "n_missing"]] >= 0).all().all()
    got = full.set_index("model")
    assert got.loc["m1", "n_rows"] == 6          # multiplicity stays visible
    assert got.loc["m1", "n_planned"] == 3
    assert got.loc["m1", "n_err"] == 2 and got.loc["m1", "n_completed"] == 1
    assert got.loc["m1", "err_rate"] == pytest.approx(2 / 3)
    assert got.loc["m2", "n_missing"] == 3


def test_manifest_path_does_not_collide_across_sinks(tmp_path):
    """Two sinks in one directory must not share a manifest: each sweep
    would then report the other's units as never attempted."""
    a, b = manifest_path(tmp_path / "runs.jsonl"), manifest_path(
        tmp_path / "runs.json")
    assert a != b
    assert a.parent == (tmp_path / "runs.jsonl").parent
    assert a.name.startswith("runs.jsonl")


def test_sink_records_are_one_write_each_and_close_is_idempotent(tmp_path,
                                                                  monkeypatch):
    """The run sink and the event log append through the same one-write path
    as the manifest (see test_each_manifest_record_is_one_write)."""
    import json

    import evalcore.core.records as records_mod
    payloads: list[bytes] = []
    real = os.write

    def spy(fd, data):
        payloads.append(bytes(data))
        return real(fd, data)
    monkeypatch.setattr(records_mod.os, "write", spy)
    log = records_mod.AppendLog(tmp_path / "log.jsonl")
    recs = [{"unit_id": f"u{i}", "x": "y" * (i * 500)} for i in range(30)]
    for r in recs:
        log.write(r)
    log.close()
    log.close()                                   # second close is a no-op
    assert [json.loads(p) for p in payloads] == recs
    assert all(p.count(b"\n") == 1 and p.endswith(b"\n") for p in payloads)


def test_read_manifest_keeps_a_units_first_record(tmp_path):
    """write_manifest says the first record of a unit_id wins; two appenders
    can both write one unit, and the reader must agree with the writer
    (and with merge_shards, which keeps the first too)."""
    import json as _json

    from evalcore import read_manifest
    p = tmp_path / "m.jsonl"
    p.write_text(_json.dumps({"unit_id": "u1", "tag": "first"}) + "\n"
                 + _json.dumps({"unit_id": "u1", "tag": "second"}) + "\n")
    assert read_manifest(p)["tag"].tolist() == ["first"]
