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

import pandas as pd
import pytest

from evalcore import (
    Backend,
    EchoBackend,
    Item,
    Request,
    Response,
    Unit,
    attrition,
    execute,
    grid,
    manifest_path,
    read_manifest,
    to_frame,
    units,
    write_manifest,
)


def items(n=8):
    return [Item(f"q{i:03d}", {"q": f"question {i}"}) for i in range(n)]


def render(u: Unit) -> Request:
    return Request(u.id, [{"role": "user", "content": u.item.payload["q"]}],
                   {"model": u.cell.get("model")})


def score(u: Unit, r: Response) -> dict:
    return {"n_chars": len(r.text)}


class DyingBackend(Backend):
    """Answers `alive` requests, then dies -- a kill, not an error row.

    It raises KeyboardInterrupt rather than a plain exception on purpose:
    `with_retries` turns any Exception into an error ROW, which is the design
    commitment this package is built on, so the never-attempted category can
    only ever be produced by the process itself going away.  Simulating the
    kill with anything catchable would test the wrong path.

    `fail_on` names item_ids that come back as error RESPONSES instead --
    the other, already-visible category; the two must not be conflated.
    """

    def identity(self) -> dict:
        return {"backend": "dying"}

    name = "dying"

    def __init__(self, alive: int, fail_on: set[str] = frozenset(),
                 on_call=None):
        self.alive, self.fail_on, self.on_call = alive, set(fail_on), on_call
        self.calls = 0

    def complete(self, req: Request) -> Response:
        if self.on_call is not None:
            self.on_call(self.calls)
        if self.calls >= self.alive:
            raise KeyboardInterrupt("process killed mid-sweep")
        self.calls += 1
        if req.messages[-1]["content"] in {f"question {int(t[1:])}"
                                           for t in self.fail_on}:
            return Response(req.unit_id, error="fatal: refused")
        return Response(req.unit_id, text=req.messages[-1]["content"][::-1])


def sweep(tmp_path, models=("m1", "m2", "m3"), n_items=8, alive=10,
          fail_on=frozenset(), on_call=None, **kw):
    """Run a cell-major sweep that dies after `alive` calls."""
    cells = grid({"model": list(models)}, tags={"arm": "base"})
    out = tmp_path / "runs.jsonl"
    be = DyingBackend(alive, fail_on, on_call)
    with pytest.raises(KeyboardInterrupt):
        execute(units(cells, items(n_items)), be, render, score, out,
            workers=1, log_every=1000, **kw)
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
    cells = grid({"model": ["m1", "m2"]}, tags={"arm": "base"})
    out = tmp_path / "runs.jsonl"
    df = execute(units(cells, items(6)), EchoBackend(), render, score, out,
             workers=2, log_every=1000)
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
                             fail_on={"q001", "q003", "q005"})
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
    assert len(read_manifest(mp)) == 6
    partial = attrition(df, ["model"], manifest=mp)
    assert partial["n_planned"].sum() == 6


def test_stale_manifest_reports_unplanned_rows_not_negative_attrition(
        tmp_path):
    """A manifest older than the sink names fewer units than the frame has.
    Those rows are surfaced as `n_unplanned`; nothing goes negative and no
    missingness is invented."""
    cells = grid({"model": ["m1", "m2"]})
    out = tmp_path / "runs.jsonl"
    all_units = list(units(cells, items(6)))
    execute(all_units, EchoBackend(), render, score, out, workers=1,
        log_every=1000)

    stale = write_manifest(all_units[:4], tmp_path / "stale.manifest.jsonl")
    full = attrition(to_frame(out), ["model"], manifest=stale)

    assert full["n_planned"].sum() == 4
    assert full["n_missing"].sum() == 0
    assert full["n_unplanned"].sum() == 8
    assert (full[["n_planned", "n_rows", "n_completed", "n_err",
                  "n_missing", "n_unplanned"]] >= 0).all().all()


def test_manifest_unions_across_resumes_so_a_narrowed_rerun_still_reports(
        tmp_path):
    """The plan is per sink, not per invocation.

    Sweep one plans both arms and dies inside the first.  Sweep two is
    narrowed to the arm someone was debugging, finishes clean, and would
    otherwise report a perfect table -- the second arm having quietly left
    the plan.  The union keeps it, and `run` says so on the way out.
    """
    cells = grid({"model": ["m1", "m2"]})
    its, out = items(6), tmp_path / "runs.jsonl"
    with pytest.raises(KeyboardInterrupt):
        execute(units(cells, its), DyingBackend(alive=4), render, score, out,
            workers=1, log_every=1000)

    narrow = list(units(cells[:1], its))             # only the debugged arm
    log = io.StringIO()
    df = execute(narrow, EchoBackend(), render, score, out, workers=1,
             log_every=1000, stream=log)
    assert len(df) == 6                              # arm one, complete
    assert "6 planned units were never attempted" in log.getvalue()

    full = attrition(to_frame(out), ["model"], manifest=manifest_path(out))
    assert full["n_planned"].sum() == 12
    got = full.set_index("model")
    assert got.loc["m1", "n_completed"] == 6
    assert got.loc["m1", "n_missing"] == 0
    assert got.loc["m2", "n_completed"] == 0
    assert got.loc["m2", "n_missing"] == 6      # the arm nobody re-ran


def test_resume_reads_the_jsonl_not_the_manifest(tmp_path):
    """Deleting the manifest must not cost work.  Resume is by unit_id in the
    sink; the manifest is rebuilt and no request is re-dispatched."""
    cells = grid({"model": ["m1"]})
    its, out = items(6), tmp_path / "runs.jsonl"
    execute(units(cells, its), EchoBackend(), render, score, out, workers=1,
        log_every=1000)
    manifest_path(out).unlink()

    class Tripwire(EchoBackend):
        """Same identity as the backend that wrote the rows (so resume has
        no reason to redo them), but any call is a kill."""
        calls = 0

        def complete(self, req):
            self.calls += 1
            raise KeyboardInterrupt("resume dispatched a finished unit")

    be = Tripwire()
    df = execute(units(cells, its), be, render, score, out, workers=1,
             log_every=1000)
    assert be.calls == 0 and len(df) == 6
    assert len(read_manifest(manifest_path(out))) == 6


def test_manifest_can_be_disabled_and_run_is_unchanged(tmp_path):
    cells = grid({"model": ["m1"]})
    out = tmp_path / "runs.jsonl"
    df = execute(units(cells, items(4)), EchoBackend(), render, score, out,
             workers=1, log_every=1000, manifest=False)
    assert len(df) == 4
    assert not manifest_path(out).exists()


def test_grouping_the_plan_by_a_scored_column_is_refused(tmp_path):
    """A never-attempted unit has no score, so there is no honest cell to put
    it in.  Better a named error than a table with a silently dropped
    category."""
    out, _be, _planned = sweep(tmp_path, alive=9)
    with pytest.raises(ValueError, match="n_chars"):
        attrition(to_frame(out), ["n_chars"], manifest=manifest_path(out))


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
    lines = mp.read_text().splitlines()
    assert all(json.loads(x)["unit_id"] for x in lines)    # none torn
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
    for k, u in enumerate(list(units(cells, its))[:3]):     # arm m1 only
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
