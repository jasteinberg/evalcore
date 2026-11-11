"""`Runner`: one unit at a time into one sink, with O(1) file I/O per unit.

For a sequential caller whose next unit depends on the previous output.
Each test pins one invariant against an independent reference: the files
the old `execute` wrote (a golden file from the commit before `execute`
was built on `Runner`), a sink built by hand, or a random draw made
directly by the test.  None takes its expected value from the code under
test.
"""

from __future__ import annotations

import builtins
import io
import json
import os
import random
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from support.incremental import files
from support.runner import Answers, Dies, echo_score, items, raw_rows, render

import evalcore.core.records as records_mod
from evalcore import (
    Backend,
    Item,
    Request,
    Response,
    Runner,
    Unit,
    attrition,
    execute,
    grid,
    manifest_path,
    read_manifest,
    to_frame,
    units,
)
from evalcore.core.backends import request_key
from evalcore.core.events import format_summary, read_events, summarize_events

GOLDEN = (Path(__file__).resolve().parents[1] / "golden" / "incremental"
          / "execute_per_unit.json")


def one_cell(n: int) -> list[Unit]:
    return list(units(grid({"model": ["m"]}), items(n)))


def runner(backend: Backend, out: Path, **kw: Any) -> Runner:
    return Runner(backend, render, kw.pop("score", echo_score), out,
                  stream=kw.pop("stream", io.StringIO()), **kw)


# --- 1. the same files as N single-unit execute calls ------------------------

@pytest.mark.parametrize("mode", ["runner", "execute"])
def test_files_match_successive_single_unit_execute_calls(tmp_path, mode):
    """Invariant 1.  The reference is what N successive execute([unit])
    calls wrote at commit 9c95d2f, before execute was built on Runner
    (support/incremental.py says how it was made).  Rows, request keys,
    manifests, cache entries, sidecar arrays and the unit/rescore events
    must be identical.  Only the event times and the run_start/run_end
    pairs differ (N pairs then, one per Runner now), and the comparison
    drops exactly those.  Both today's Runner and today's per-unit execute
    are held to it."""
    got = files(mode, tmp_path)
    want = json.loads(GOLDEN.read_text())
    assert sorted(got) == sorted(want)
    for k in want:
        assert got[k] == want[k], k


# --- 2. resume decisions on an existing sink -------------------------------

class Counting(Answers):
    """Answers and records the unit_id of every call."""

    def __init__(self) -> None:
        super().__init__()
        self.ids: list[str] = []

    def complete(self, req: Request) -> Response:
        self.ids.append(req.unit_id)
        return super().complete(req)


def hand_sink(path: Path, us: list[Unit], backend: Backend) -> dict[str, dict]:
    """A sink written by hand, one unit per resume case:

    q0  done (its current key, ok, score_version v1)
    q1  stale (a key from another request)
    q2  errored last time (current key)
    q3  never run
    q4  done under score_version v0
    q5  errored, then done: the last row wins
    """
    def key(u: Unit) -> str:
        return request_key(render(u), u.repeat, backend)

    def row(u: Unit, **kw: Any) -> dict:
        return {**u.row(), "text": "stored", "error": None, "cached": False,
                "request_key": key(u), "status": "ok", "answer": "stored",
                "score_version": "v1", "batch_size": 1, **kw}

    q = {u.item.item_id: u for u in us}
    rows = [row(q["q0"]), row(q["q1"], request_key="0" * 32),
            row(q["q2"], error="HTTP 500", status="error", text=None),
            row(q["q4"], score_version="v0"),
            row(q["q5"], error="HTTP 500", status="error", text=None),
            row(q["q5"])]
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return {r["item_id"]: r for r in rows}


def test_resume_on_an_existing_sink_redoes_exactly_what_plan_redoes(tmp_path):
    """Invariant 2, on the sink above.  Calls go to q1 (stale), q2 (errored)
    and q3 (new) only.  q0 and q5 come back as their stored rows, unchanged.
    q4 is re-scored from its stored text, with no call."""
    us, b, out = one_cell(6), Counting(), tmp_path / "rows.jsonl"
    stored = hand_sink(out, us, b)
    with runner(b, out, score_version="v1") as r:
        got = {u.item.item_id: r.run(u) for u in us}
    by_id = {u.id: u.item.item_id for u in us}
    assert sorted(by_id[i] for i in b.ids) == ["q1", "q2", "q3"]
    assert got["q0"] == stored["q0"]
    assert got["q5"] == stored["q5"]
    assert got["q4"]["score_version"] == "v1"
    assert got["q4"]["answer"] == "stored"        # the stored text, re-scored
    assert got["q1"]["answer"] == "question 1"    # a fresh call
    # each returned row is the row the sink now holds for that unit
    last = {row["item_id"]: row for row in raw_rows(out)}
    assert got == last


@pytest.mark.parametrize("kw,called", [
    ({"retry_errors": False}, ["q1", "q3"]),
    ({"resume": False}, ["q0", "q1", "q2", "q3", "q4", "q5"])])
def test_resume_switches_act_as_in_plan(tmp_path, kw, called):
    """retry_errors=False keeps an error row as final (q2 not called).
    resume=False calls every unit."""
    us, b, out = one_cell(6), Counting(), tmp_path / "rows.jsonl"
    stored = hand_sink(out, us, b)
    with runner(b, out, score_version="v1", **kw) as r:
        got = {u.item.item_id: r.run(u) for u in us}
    by_id = {u.id: u.item.item_id for u in us}
    assert sorted(by_id[i] for i in b.ids) == called
    if "retry_errors" in kw:
        assert got["q2"] == stored["q2"]


def test_an_item_id_reused_for_another_item_is_refused_across_calls(tmp_path):
    """The clash check spans every unit the Runner has planned, as one
    execute call's does over its batch.  A refused unit leaves no row."""
    a = Item("q0", {"q": "question 0"})
    b = Item("q0", {"q": "question 9"})
    cell = grid({"model": ["m"]})
    ua, ub = list(units(cell, [a])), list(units(cell, [b]))
    out = tmp_path / "rows.jsonl"
    with runner(Answers(), out) as r:
        r.run(ua[0])
        with pytest.raises(ValueError, match="names two different items"):
            r.run(ub[0])
        r.run(ua[0])                      # the first item is still fine
    assert len(raw_rows(out)) == 1


# --- 3. the manifest ---------------------------------------------------------

def test_manifest_lists_every_unit_once_and_only_grows(tmp_path):
    """Invariant 3.  Two Runners on one sink, the second with more units:
    the manifest is append-only (the first file is a byte prefix of the
    second), lists each unit once, and attrition counts against it."""
    out = tmp_path / "rows.jsonl"
    us = one_cell(5)
    with runner(Answers(), out) as r:
        for u in us[:3]:
            r.run(u)
        r.run(us[0])                      # done: no second manifest record
    first = manifest_path(out).read_bytes()
    with runner(Answers(), out) as r:
        for u in us:
            r.run(u)
    second = manifest_path(out).read_bytes()
    assert second.startswith(first)
    assert len(first.splitlines()) == 3 and len(second.splitlines()) == 5
    m = read_manifest(manifest_path(out))
    assert sorted(m["unit_id"]) == sorted(u.id for u in us)
    a = attrition(to_frame(out), ["model"], manifest=manifest_path(out))
    assert a.iloc[0]["n_planned"] == 5 and a.iloc[0]["n_missing"] == 0


def test_units_planned_by_execute_and_never_run_are_reported_at_close(tmp_path):
    """A manifest written by an earlier execute that died: the Runner's
    close warns about the units still never attempted."""
    out, us = tmp_path / "rows.jsonl", one_cell(4)
    with pytest.raises(KeyboardInterrupt):
        execute(us, Dies(alive=1), render, echo_score, out, workers=1,
                stream=io.StringIO())
    log = io.StringIO()
    with runner(Answers(), out, stream=log) as r:
        r.run(us[1])
    assert "2 planned units were never attempted" in log.getvalue()


# --- 4. a kill midway --------------------------------------------------------

def test_a_killed_runner_leaves_files_the_readers_read(tmp_path):
    """Invariant 4.  The backend dies on the third unit.  The sink holds two
    rows, the manifest three units (the third recorded before its call),
    the event log a run_start and two unit events with no run_end.  A
    second Runner on the same backend returns the two done units from the
    sink and calls only the other two."""
    out, us = tmp_path / "rows.jsonl", one_cell(4)
    with pytest.raises(KeyboardInterrupt):
        with runner(Dies(alive=2), out) as r:
            for u in us:
                r.run(u)
    df = to_frame(out)
    assert sorted(df["unit_id"]) == sorted(u.id for u in us[:2])
    ev = read_events(out)
    assert list(ev["event"]) == ["run_start", "unit", "unit"]
    s = summarize_events(out)
    assert s["killed_runs"] == 1
    assert s["runs"][0]["units_done"] == 2 and s["runs"][0]["to_go"] is None
    a = attrition(df, ["model"], manifest=manifest_path(out))
    assert (a.iloc[0]["n_planned"], a.iloc[0]["n_completed"],
            a.iloc[0]["n_missing"]) == (3, 2, 1)
    b = Dies(alive=10)                    # the same identity, so the same keys
    with runner(b, out) as r:
        rows = [r.run(u) for u in us]
    assert b.calls == 2
    assert rows[:2] == raw_rows(out)[:2]


def test_an_event_log_mixing_execute_and_runner_runs_summarises(tmp_path):
    """execute's run_start carries plan counts and a Runner's does not.  In
    one frame the missing counts are NaN, which `int(x or 0)` turned into a
    ValueError.  They are reported as unknown (None)."""
    out, us = tmp_path / "rows.jsonl", one_cell(3)
    execute(us[:1], Answers(), render, echo_score, out, workers=1,
            stream=io.StringIO())
    with runner(Answers(), out) as r:
        r.run(us[1])
        r.run(us[2])
    s = summarize_events(out)
    assert [(x["to_go"], x["units_done"], x["ended"]) for x in s["runs"]] == [
        (1, 1, True), (None, 2, True)]
    assert "2 units in" in format_summary(s)


# --- 5. global random state --------------------------------------------------

class Draws(Backend):
    """Returns, in meta, one draw from each global generator, made inside
    the call."""

    name = "draws"

    def __init__(self, torch: Any = None) -> None:
        self.torch = torch

    def identity(self) -> dict:
        return {"backend": "draws"}

    def complete(self, req: Request) -> Response:
        meta = {"py": random.random(), "np": float(np.random.random())}
        if self.torch is not None:
            meta["torch"] = float(self.torch.rand(1).item())
        return Response(req.unit_id, text="x", meta=meta)


def _check_draws(tmp_path: Path, torch: Any = None) -> None:
    out = tmp_path / "rows.jsonl"
    us = one_cell(3)
    with runner(Draws(torch), out, cache=None) as r:
        for s, u in zip((11, 12, 13), us, strict=True):
            random.seed(s)
            np.random.seed(s)
            if torch is not None:
                torch.manual_seed(s)
            row = r.run(u)
            random.seed(s)
            np.random.seed(s)
            assert row["meta_py"] == random.random()
            assert row["meta_np"] == float(np.random.random())
            if torch is not None:
                torch.manual_seed(s)
                assert row["meta_torch"] == float(torch.rand(1).item())


def test_run_does_not_touch_global_random_state(tmp_path):
    """Invariant 5.  The caller seeds, then calls run.  The backend's first
    draw from each generator must be the seed's first draw, which the test
    computes by seeding again and drawing directly."""
    _check_draws(tmp_path)


def test_run_does_not_touch_torch_random_state(tmp_path):
    torch = pytest.importorskip("torch")
    _check_draws(tmp_path, torch)


# --- O(1) file I/O per unit ------------------------------------------------------

class IOCount:
    """Counts JSONL reads (records.read_jsonl, which every sink and manifest
    read goes through) and opens of each path, through builtins.open,
    io.open (which pathlib's Path.open calls) and os.open."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.reads: list[str] = []
        self.opens: list[str] = []
        read, bopen, oopen = (records_mod.read_jsonl, builtins.open, os.open)
        assert io.open is bopen

        def counted_read(path, *a, **k):
            self.reads.append(Path(path).name)
            return read(path, *a, **k)

        def counted_bopen(file, *a, **k):
            if isinstance(file, (str, Path)):
                self.opens.append(Path(file).name)
            return bopen(file, *a, **k)

        def counted_oopen(path, *a, **k):
            self.opens.append(Path(path).name)
            return oopen(path, *a, **k)

        monkeypatch.setattr(records_mod, "read_jsonl", counted_read)
        monkeypatch.setattr(builtins, "open", counted_bopen)
        monkeypatch.setattr(io, "open", counted_bopen)
        monkeypatch.setattr(os, "open", counted_oopen)

    def reset(self) -> None:
        self.reads.clear()
        self.opens.clear()


@pytest.mark.parametrize("n", [5, 40])
def test_file_io_per_unit_does_not_grow_with_n(tmp_path, monkeypatch, n):
    """Open a Runner on a sink that already holds n rows, then run n more
    units.  Opening reads the sink and the manifest once each.  Running
    reads nothing and never reopens the sink or the event log.  The
    manifest gets two opens per new unit (the crash-fragment check and the
    append), the same for every n."""
    out, us = tmp_path / "rows.jsonl", one_cell(2 * n)
    with runner(Answers(), out) as r:
        for u in us[:n]:
            r.run(u)
    io_ = IOCount(monkeypatch)
    r = runner(Answers(), out)
    r.open()
    assert sorted(io_.reads) == ["rows.jsonl", "rows.jsonl.manifest.jsonl"]
    io_.reset()
    for u in us[n:]:
        r.run(u)
    reads, opens = list(io_.reads), list(io_.opens)
    r.close()
    assert reads == []
    assert opens.count("rows.jsonl") == 0
    assert opens.count("rows.jsonl.events.jsonl") == 0
    assert opens.count("rows.jsonl.manifest.jsonl") == 2 * n
    assert len(raw_rows(out)) == 2 * n


def test_the_io_counter_sees_what_per_unit_execute_reads(tmp_path, monkeypatch):
    """The counter's own check: single-unit execute calls read the sink
    and manifest on every call (the sink once by the Runner inside, except
    on the first call when it does not exist yet, and once by to_frame), so
    the count grows with the calls.  A counter that intercepted nothing
    would pass the test above."""
    out, us = tmp_path / "rows.jsonl", one_cell(6)
    io_ = IOCount(monkeypatch)
    for u in us:
        execute([u], Answers(), render, echo_score, out, workers=1,
                stream=io.StringIO())
    assert io_.reads.count("rows.jsonl") == 2 * len(us) - 1
    assert io_.reads.count("rows.jsonl.manifest.jsonl") == len(us) - 1
    assert io_.opens.count("rows.jsonl") >= len(us)


# --- lifecycle ---------------------------------------------------------------

def test_run_outside_open_is_refused(tmp_path):
    r = runner(Answers(), tmp_path / "rows.jsonl")
    with pytest.raises(RuntimeError, match="not open"):
        r.run(one_cell(1)[0])
    with r:
        r.run(one_cell(1)[0])
    with pytest.raises(RuntimeError, match="not open"):
        r.run(one_cell(1)[0])


def test_a_refused_construction_touches_no_file(tmp_path):
    """The backend's identity is checked before any file is opened."""
    class Broken(Answers):
        def identity(self) -> dict:
            raise ValueError("no identity")

    with pytest.raises(ValueError, match="no identity"):
        runner(Broken(), tmp_path / "rows.jsonl")
    assert list(tmp_path.iterdir()) == []
