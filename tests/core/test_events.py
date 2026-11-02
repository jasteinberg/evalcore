"""The event log: every unit, retry and run boundary recorded, and the
questions about a long run answerable from it."""

from __future__ import annotations

import io
import json

import pandas as pd
import pytest
from support.runner import (
    Dies,
    items,
    no_backoff,  # noqa: F401 - a fixture, requested by name
    render,
    run,
)

from evalcore import (
    FunctionBackend,
    Response,
    Transient,
    execute,
    grid,
    units,
)
from evalcore.core.events import (
    error_kind,
    events_path,
    parse_log,
    read_events,
    summarize_events,
)


@pytest.mark.usefixtures("no_backoff")
def test_every_unit_and_the_run_boundaries_are_recorded(tmp_path):
    calls = {"n": 0}

    def fn(req):
        calls["n"] += 1
        q = req.messages[-1]["content"]
        if calls["n"] == 1:
            raise Transient("HTTP 429: slow down", retry_after=2.0)
        if q.endswith("3"):
            return Response(req.unit_id, error="fatal: HTTP 400 req 9f3a2c1b7d8e")
        return Response(req.unit_id, text="ok",
                        meta={"truncated": q.endswith("2"), "in_tokens": 10,
                              "out_tokens": 3, "latency_s": 0.5})

    out = tmp_path / "r.jsonl"
    df = run(list(units(grid({"m": ["x"]}), items(5))),
             FunctionBackend(fn, identity={"m": "scripted"}), out)
    ev = read_events(out)
    assert list(ev["event"].iloc[[0, -1]]) == ["run_start", "run_end"]
    u = ev[ev["event"] == "unit"].set_index("unit_id")
    assert len(u) == 5
    assert u["status"].to_dict() == df.set_index("unit_id")["status"].to_dict()
    r = ev[ev["event"] == "retry"].iloc[0]
    assert r["attempt"] == 1 and r["server_wait_s"] == 2.0 and r["wait_s"] >= 2
    s = summarize_events(out, prices={"in": 1.0, "out": 5.0})
    assert s["killed_runs"] == 0 and s["runs"][0]["units_done"] == 5
    assert s["status_counts"] == {"ok": 3, "truncated": 1, "error": 1}
    assert s["errors"][0]["n"] == 1 and "HTTP 400" in s["errors"][0]["example"]
    assert s["retries"]["n"] == 1 and s["retries"]["server_requested"] == 1
    assert s["tokens"] == {"in": 40.0, "out": 12.0}
    assert s["cost"] == pytest.approx((40 * 1 + 12 * 5) / 1e6)


def test_a_killed_run_has_no_end_and_a_resume_is_a_second_run(tmp_path):
    out, us = tmp_path / "r.jsonl", list(units(grid({"m": ["x"]}), items(6)))
    with pytest.raises(KeyboardInterrupt):
        run(us, Dies(alive=2), out)
    s = summarize_events(out)
    assert s["killed_runs"] == 1 and s["runs"][0]["units_done"] == 2
    run(us, Dies(alive=100), out)
    s = summarize_events(out)
    assert [r["ended"] for r in s["runs"]] == [False, True]
    assert [r["units_done"] for r in s["runs"]] == [2, 4]
    assert s["runs"][1]["to_go"] == 4                 # resume skipped 2


def test_run_boundaries_survive_a_crash_fragment_in_the_event_log(tmp_path):
    """B24.  As above, but run 1's last event line is cut in half (a kill
    inside a write).  By construction there were still 2 runs, the first
    without a run_end."""
    out, us = tmp_path / "r.jsonl", list(units(grid({"m": ["x"]}), items(6)))
    with pytest.raises(KeyboardInterrupt):
        run(us, Dies(alive=2), out)
    ev = tmp_path / "r.jsonl.events.jsonl"
    text = ev.read_text()
    ev.write_text(text[:len(text) - len(text.splitlines()[-1]) // 2 - 1])
    run(us, Dies(alive=100), out)
    s = summarize_events(out)
    assert [r["ended"] for r in s["runs"]] == [False, True]


def test_errors_differing_only_in_numbers_and_ids_group_together():
    a = "transient x5: HTTP 429 after 37s (req 4f9a8b7c6d5e4f3a)"
    b = "transient x5: HTTP 429 after 12s (req 0011223344556677)"
    assert error_kind(a) == error_kind(b)
    assert error_kind(a) != error_kind("fatal: HTTP 400: bad field")


def test_stalls_are_found(tmp_path):
    p = tmp_path / "x.jsonl.events.jsonl"
    t = pd.Timestamp("2026-10-02T10:00:00Z")
    lines = [{"t": t.isoformat(), "event": "run_start", "n_units": 3}]
    for k, dt in enumerate([0, 5, 900]):                 # 15 minutes of nothing
        lines.append({"t": (t + pd.Timedelta(seconds=dt)).isoformat(),
                      "event": "unit", "unit_id": f"u{k}", "status": "ok"})
    p.write_text("\n".join(json.dumps(x) for x in lines) + "\n")
    s = summarize_events(p, stall_s=300)
    assert len(s["stalls"]) == 1 and s["stalls"][0]["gap_s"] == 895
    assert s["killed_runs"] == 1                         # no run_end


def test_a_foreign_text_log_is_parsed_and_unmatched_lines_counted(tmp_path):
    log = tmp_path / "sweep.log"
    log.write_text("2026-10-02 10:00:01 loaded model pythia-70m step=143000\n"
                   "2026-10-02 10:00:09 scored 512 items em=0.0071\n"
                   "some stray warning\n"
                   "2026-10-02 10:01:40 scored 512 items em=0.0127\n")
    df = parse_log(log, {
        "load": r"^(?P<t>\S+ \S+) loaded model (?P<model>\S+) step=(?P<step>\d+)",
        "score": r"^(?P<t>\S+ \S+) scored (?P<n>\d+) items em=(?P<em>[\d.]+)"})
    assert list(df["event"]) == ["load", "score", "score"]
    assert df["step"].iloc[0] == 143000 and df["em"].iloc[2] == 0.0127
    assert df.attrs["unmatched"] == 1
    assert df.attrs["unmatched_examples"] == ["some stray warning"]
    assert (df["t"].diff().dt.total_seconds().iloc[1:] > 0).all()


def test_concurrent_writers_leave_no_torn_lines(tmp_path):
    out = tmp_path / "r.jsonl"
    be = FunctionBackend(lambda r: "ok", identity={"m": "fast"})
    execute(units(grid({"m": ["x"]}), items(200)), be, render, lambda u, r: {}, out,
        workers=8, stream=io.StringIO())
    lines = events_path(out).read_text().splitlines()
    assert all(json.loads(x) for x in lines)
    ev = read_events(out)
    assert ev.attrs["unreadable"] == 0 and (ev["event"] == "unit").sum() == 200


def test_events_can_be_turned_off(tmp_path):
    out = tmp_path / "r.jsonl"
    run(list(units(grid({"m": ["x"]}), items(2))),
        FunctionBackend(lambda r: "ok", identity={"m": "x"}), out, events=False)
    assert not events_path(out).exists()
