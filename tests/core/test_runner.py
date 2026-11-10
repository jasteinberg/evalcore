"""Runner failure modes that yield a plausible number instead of a crash.

Each test pins one way the runner can lose, duplicate, mis-join or mis-score
a unit while every statistic downstream still computes.  Expected values
come from the construction of the fixture -- which unit asked what, how many
calls a backend received -- never from running the code under test.

Row counts are taken from the raw JSONL wherever duplication is the
question: `to_frame` deduplicates on unit_id, so a frame cannot show a
duplicated row even when the sink holds one.

Rows of the invariant catalogue (B16b, B18, ...) are named in the docstring
of the test that holds them; a newly found violation lands as a strict
xfail whose reason names the defect.
"""

from __future__ import annotations

import importlib.util
import io
import json
import random
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest
from support.runner import (
    Answers,
    Truncates,
    cache_files,
    echo_score,
    items,
    no_backoff,  # noqa: F401 - a fixture, requested by name
    raw_rows,
    render,
    run,
)

from evalcore import (
    Backend,
    Cache,
    Fatal,
    FunctionBackend,
    Item,
    ParseFailure,
    Request,
    Response,
    Transient,
    Unit,
    attrition,
    execute,
    grid,
    manifest_path,
    summarize,
    to_frame,
    units,
    write_manifest,
)
from evalcore.core.records import load_arrays
from evalcore.testing import assert_same

REPO = Path(__file__).resolve().parents[2]


def one_cell(n: int) -> list[Unit]:
    """Items q0..q{n-1} in a single cell, model "m"."""
    return list(units(grid({"model": ["m"]}), items(n)))


# --- retry: fail once, then succeed ----------------------------------------

class FailsOnce(Answers):
    """The first call of the whole run raises Transient; every later call
    answers.  One 429 in a sweep, the most common failure there is."""

    def __init__(self) -> None:
        super().__init__()
        self.failed = False

    def complete(self, req: Request) -> Response:
        with self.lock:
            first, self.failed = not self.failed, True
        if first:
            raise Transient("HTTP 429: slow down")
        return super().complete(req)


@pytest.mark.usefixtures("no_backoff")
@pytest.mark.parametrize("batch_size", [1, 4])
def test_fail_once_then_succeed_writes_exactly_one_row(tmp_path, batch_size):
    n = 8
    out, cache = tmp_path / "r.jsonl", Cache(tmp_path / "cache")
    be = FailsOnce()
    run(one_cell(n), be, out, cache=cache, batch_size=batch_size)

    rows = raw_rows(out)
    assert len(rows) == n                                  # not n + 1, not 0
    assert len({r["unit_id"] for r in rows}) == n
    assert all(r["error"] is None for r in rows)
    assert {r["item_id"]: r["answer"] for r in rows} == {
        f"q{i}": f"question {i}" for i in range(n)}
    assert be.failed
    # every unit answered exactly once by a SUCCESSFUL call
    assert sum(be.calls.values()) == n
    assert len(cache_files(tmp_path / "cache")) == n
    assert all(json.loads(p.read_text()).get("error") is None
               for p in cache_files(tmp_path / "cache"))


# --- an error response is never cached -------------------------------------

def _raise_fatal(req: Request) -> Response:
    raise Fatal("HTTP 400: refused")


def _raise_transient(req: Request) -> Response:
    raise Transient("HTTP 503")


def _return_error(req: Request) -> Response:
    return Response(req.unit_id, error="upstream: content filtered")


def _raise_unknown(req: Request) -> Response:
    raise ValueError("unparseable body")


@pytest.mark.usefixtures("no_backoff")
@pytest.mark.parametrize("fail", [_raise_fatal, _raise_transient,
                                  _return_error, _raise_unknown],
                         ids=["fatal", "transient-exhausted", "returned-error",
                              "unknown-exception"])
def test_an_error_is_never_cached(tmp_path, fail):
    """The failed unit must reach the backend again on the next pass, and
    only it: the other three are cache hits."""
    bad = "question 1"

    class Flaky(Answers):
        def __init__(self, failing: bool) -> None:
            super().__init__()
            self.failing = failing

        def complete(self, req: Request) -> Response:
            if self.failing and req.messages[-1]["content"] == bad:
                return fail(req)
            return super().complete(req)

    us = one_cell(4)
    out, cache = tmp_path / "r.jsonl", Cache(tmp_path / "cache")
    df1 = run(us, Flaky(failing=True), out, cache=cache, attempts=3)
    assert df1.set_index("item_id").loc["q1", "error"]     # it did fail
    assert len(cache_files(tmp_path / "cache")) == 3       # error not stored

    be = Flaky(failing=False)
    df2 = run(us, be, out, cache=Cache(tmp_path / "cache"), resume=False)
    assert be.calls == Counter({bad: 1})                   # only the failure
    assert df2["error"].isna().all()
    assert df2.set_index("item_id").loc["q1", "answer"] == bad


@pytest.mark.usefixtures("no_backoff")
def test_resume_retries_units_whose_last_row_is_an_error(tmp_path):
    """B16b.  An error row records a failed attempt, not a result.  A rerun
    must try those units again (and say so), unless asked not to; and once
    a unit's LAST row is ok, made from the same request, a further rerun
    has nothing to do.  Expected calls are fixed by the four backends."""
    us = one_cell(4)
    out = tmp_path / "r.jsonl"

    class Down(Answers):
        """Same identity as Answers (so its rows are not stale against it),
        but every call fails."""

        def complete(self, req):
            raise Transient("HTTP 503")

    run(us, Down(), out, attempts=2)
    kept = run(us, Answers(), out, retry_errors=False)
    assert (kept["status"] == "error").all()

    log, be = io.StringIO(), Answers()
    df = run(us, be, out, stream=log)
    assert sum(be.calls.values()) == 4
    assert (df["status"] == "ok").all()
    assert "4 units errored last time" in log.getvalue()

    tripwire = Answers()                       # ok rows now follow the errors
    run(us, tripwire, out)
    assert sum(tripwire.calls.values()) == 0


# --- completion order ------------------------------------------------------

def test_thread_pool_completion_order_does_not_move_answers(tmp_path):
    """Unit i may finish only after every unit j > i has finished, so the
    pool completes in exactly reverse dispatch order.  The fixture asserts
    that it did, or the test would prove nothing."""
    n = 8
    cv, finished = threading.Condition(), []

    class Reverse(Answers):
        def complete(self, req: Request) -> Response:
            i = int(req.messages[-1]["content"].split()[-1])
            with cv:
                ok = cv.wait_for(lambda: set(range(i + 1, n)) <= set(finished),
                                 timeout=10)
                assert ok, "reverse-order barrier timed out"
            resp = super().complete(req)
            with cv:
                finished.append(i)
                cv.notify_all()
            return resp

    df = execute(one_cell(n), Reverse(), render,
             echo_score, tmp_path / "r.jsonl", workers=n, stream=io.StringIO())
    assert finished == list(range(n - 1, -1, -1))          # the shuffle happened
    assert len(df) == n
    assert (df["answer"] == "question " + df["item_id"].str[1:]).all()


class Permuted(Answers):
    """A batched backend that returns responses in completion order, here a
    fixed permutation, rather than request order."""

    def __init__(self, drop: int | None = None) -> None:
        super().__init__()
        self.drop = drop

    def complete_batch(self, reqs):
        out = [self.complete(r) for r in reqs]
        random.Random(0).shuffle(out)
        if self.drop is not None:
            out = [r for r in out if not r.text.endswith(f" {self.drop}")]
        return out


def test_batch_returned_out_of_order_is_joined_by_unit_id(tmp_path):
    n = 6
    df = run(one_cell(n), Permuted(), tmp_path / "r.jsonl", batch_size=n)
    assert len(df) == n and df["error"].isna().all()
    assert (df["answer"] == "question " + df["item_id"].str[1:]).all()


def test_batch_that_drops_a_response_gives_an_error_row(tmp_path):
    """A short batch must cost exactly the unit it lost: an error row for it,
    the right answer for every other unit, no crash."""
    n = 6
    df = run(one_cell(n), Permuted(drop=2), tmp_path / "r.jsonl",
             batch_size=n)
    got = df.set_index("item_id")
    assert len(df) == n
    assert got.loc["q2", "error"].startswith("missing")
    ok = got.drop(index="q2")
    assert ok["error"].isna().all()
    assert (ok["answer"] == "question " + ok.index.str[1:]).all()


# --- a truncated final line after a crash ----------------------------------

def test_truncated_final_line_is_reported_and_loses_nothing(tmp_path):
    """A kill during `write` leaves a fragment with no newline.  The reader
    must survive it and say so; resume must re-run that unit and write its
    row intact, not glued onto the fragment."""
    us = one_cell(4)
    out = tmp_path / "r.jsonl"
    run(us, Answers(), out)
    lines = out.read_text().splitlines()
    out.write_text("\n".join(lines[:3]) + "\n" + lines[3][:40])   # the crash

    with pytest.warns(RuntimeWarning, match="1 unreadable line"):
        df = to_frame(out)
    assert sorted(df["item_id"]) == ["q0", "q1", "q2"]

    log, be = io.StringIO(), Answers()
    with pytest.warns(RuntimeWarning, match="1 unreadable line"):
        df = run(us, be, out, stream=log)
    assert "unreadable" in log.getvalue()
    assert be.calls == Counter({"question 3": 1})          # only the lost unit
    assert sorted(df["item_id"]) == ["q0", "q1", "q2", "q3"]
    assert df.set_index("item_id").loc["q3", "answer"] == "question 3"
    json.loads(out.read_text().splitlines()[-1])           # last row intact


# --- row status: five states, one partition ----------------------------------

def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_worked_example_parse_failure_is_an_unparsed_row(tmp_path):
    wex = _load(REPO / "examples" / "worked_example.py")

    class Says(Backend):
        def identity(self) -> dict:
            return {"backend": "says"}

        def complete(self, req: Request) -> Response:
            i = int(req.messages[-1]["content"].split()[-1])
            return Response(req.unit_id,
                            text="Answer: A" if i == 0 else "Hmm, maybe.")

    its = [Item(f"q{i}", {"q": f"question {i}", "gold": "A"})
           for i in range(2)]
    df = execute(units(grid({"model": ["m"]}), its), Says(), render, wex.score,
             tmp_path / "r.jsonl", workers=1, stream=io.StringIO())
    got = df.set_index("item_id")
    assert got["status"].to_dict() == {"q0": "ok", "q1": "unparsed"}
    assert got.loc["q0", "correct"] == 1.0
    assert pd.isna(got.loc["q1", "correct"])                # not a zero


def test_a_truncated_generation_is_not_scored(tmp_path):
    """Odd items come back cut off: the flag reaches the frame, the scorer
    never sees them, and their metric is missing, not a zero."""
    scored: list[str] = []

    def score(u: Unit, r: Response) -> dict[str, Any]:
        scored.append(u.item.item_id)
        return {"correct": 0.0}

    df = run(one_cell(4), Truncates(), tmp_path / "r.jsonl", score=score)
    assert sorted(scored) == ["q0", "q2"]
    got = df.set_index("item_id")
    assert got["meta_truncated"].to_dict() == {"q0": False, "q1": True,
                                               "q2": False, "q3": True}
    assert got.loc[["q1", "q3"], "correct"].isna().all()
    assert got["status"].to_dict() == {"q0": "ok", "q1": "truncated",
                                       "q2": "ok", "q3": "truncated"}


@pytest.mark.filterwarnings("ignore:.*carry caveats:RuntimeWarning")  # tiny fixture
def test_every_planned_unit_lands_in_exactly_one_state(tmp_path):
    """One unit per state, fixed by construction:
    q0, q5 ok; q1 truncated; q2 unparsed; q3 call refused; q4 scorer bug;
    q6 planned but never attempted (limit=).  The metric must see only the
    two ok rows -- a scorer that raises leaves the metric MISSING, never a
    0 averaged into accuracy -- and every other unit must be accounted for,
    by name."""
    def scorer(u: Unit, r: Response) -> dict[str, Any]:
        if r.text == "dunno":
            raise ParseFailure("no letter A-D")
        if r.text == "boom":
            raise KeyError("gold")                          # a scorer bug
        return {"correct": 1.0}

    class Mixed(Backend):
        def identity(self) -> dict:
            return {"backend": "mixed"}

        def complete(self, req: Request) -> Response:
            i = int(req.messages[-1]["content"].split()[-1])
            if i == 3:
                raise Fatal("HTTP 400")
            return Response(req.unit_id,
                            text={2: "dunno", 4: "boom"}.get(i, "A"),
                            meta={"truncated": i == 1})

    us = one_cell(7)
    out = tmp_path / "r.jsonl"
    write_manifest(us, manifest_path(out))                  # plan all seven
    df = run(us, Mixed(), out, score=scorer, limit=6)

    got = df.set_index("item_id")
    assert got["status"].to_dict() == {
        "q0": "ok", "q1": "truncated", "q2": "unparsed", "q3": "error",
        "q4": "error", "q5": "ok"}
    assert got["correct"].count() == 2                      # only ok rows
    assert got.loc["q2", "parse_error"] == "no letter A-D"
    assert pd.isna(got.loc["q2", "error"])                  # not a failed call
    assert got.loc["q4", "error"].startswith("score: KeyError")

    a = attrition(df, ["model"], manifest=manifest_path(out)).iloc[0]
    assert (a["n_planned"], a["n_completed"], a["n_err"], a["n_truncated"],
            a["n_unparsed"], a["n_missing"]) == (7, 2, 2, 1, 1, 1)
    assert a["n_rows"] == 6
    assert a["attrition_rate"] == pytest.approx(5 / 7)
    blind = attrition(df, ["model"]).iloc[0]
    assert (blind["n_err"], blind["n_truncated"], blind["n_unparsed"]) == (
        2, 1, 1)

    s = summarize(df, "correct", ["model"], n_boot=200).iloc[0]
    assert s["n_rows"] == 2 and s["n_excluded"] == 4


def test_a_unit_takes_its_worst_rows_status():
    """Several rows per unit (a judge panel): error > truncated > unparsed
    > ok, so one bad verdict is never hidden by two good ones."""
    us = one_cell(3)
    rows = []
    for u, sts in zip(us, [("ok", "unparsed"), ("unparsed", "truncated"),
                           ("ok", "ok")], strict=True):
        rows += [dict(u.row(), status=st, error=None) for st in sts]
    plan = pd.DataFrame([u.row() for u in us])
    a = attrition(pd.DataFrame(rows), ["model"], manifest=plan).iloc[0]
    assert (a["n_completed"], a["n_unparsed"], a["n_truncated"]) == (1, 1, 1)
    assert a["n_rows"] == 6


def test_rows_from_before_status_get_one_derived(tmp_path):
    """A sink with no `status` column at all: derived from `error` alone."""
    out = tmp_path / "old.jsonl"
    out.write_text(json.dumps({"unit_id": "a", "error": None}) + "\n"
                   + json.dumps({"unit_id": "b", "error": "fatal: x"}) + "\n")
    assert to_frame(out)["status"].tolist() == ["ok", "error"]


def test_a_row_with_an_error_is_never_ok(tmp_path):
    """B18.  Hand-written sink rows: a row stored as ok but carrying an
    error is an error; a tool_error with an error keeps its own failure
    status; error rows stay error; a row with no error keeps ok; a row
    missing its status among rows that have one is derived from `error`."""
    out = tmp_path / "r.jsonl"
    rows = [{"unit_id": "a", "status": "ok", "error": "x"},
            {"unit_id": "b", "status": "tool_error", "error": "search down"},
            {"unit_id": "c", "status": "error", "error": "HTTP 400"},
            {"unit_id": "d", "status": "ok", "error": None},
            {"unit_id": "e", "error": "fatal: x"}]
    out.write_text("".join(json.dumps(r) + "\n" for r in rows))
    got = to_frame(out).set_index("unit_id")["status"].to_dict()
    assert got == {"a": "error", "b": "tool_error", "c": "error", "d": "ok",
                   "e": "error"}


# --- backend contract at the runner ------------------------------------------

def test_a_target_sent_to_a_generating_backend_is_an_error_row(tmp_path):
    from evalcore import EchoBackend

    def render_t(u: Unit) -> Request:
        return Request(u.id, [{"role": "user", "content": "2 + 2 ="}], {},
                       target=" 4")

    df = execute(one_cell(2), EchoBackend(), render_t,
             echo_score, tmp_path / "r.jsonl", workers=1, stream=io.StringIO())
    assert (df["status"] == "error").all()
    assert df["error"].str.contains("cannot teacher-force").all()


def test_a_backend_must_say_what_determines_its_outputs(tmp_path):
    class MyModel(Backend):
        def __init__(self, path: str) -> None:
            self.path = path

        def complete(self, req: Request) -> Response:
            return Response(req.unit_id, text=self.path)

    with pytest.raises(TypeError, match="MyModel must define identity"):
        run(one_cell(2), MyModel("ckpt-A"), tmp_path / "r.jsonl")
    assert not (tmp_path / "r.jsonl").exists()          # refused up front


# --- the cache and the resumed sink ---------------------------------------------

def test_run_cache_resume_and_frame(tmp_path):
    cells = grid({"model": ["m1", "m2"], "temperature": [0.0]},
                 tags={"condition": "base"})
    its = items(8, aug=2)
    out, cache = tmp_path / "runs.jsonl", Cache(tmp_path / "cache")

    n = len(cells) * len(its)
    be1 = Answers()
    df1 = execute(units(cells, its), be1, render, echo_score, out,
              cache=cache, workers=2, log_every=1000)
    assert len(df1) == n and sum(be1.calls.values()) == n
    assert cache.hits == 0 and cache.misses == n
    assert df1["error"].isna().all()

    # second pass: every unit already in the sink -> nothing dispatched.
    # Checked on the backend, not the frame: to_frame deduplicates on
    # unit_id, so the frame looks the same whether or not work was redone.
    be2 = Answers()
    df2 = execute(units(cells, its), be2, render, echo_score, out,
              cache=cache, workers=2, log_every=1000)
    assert sum(be2.calls.values()) == 0
    assert len(df2) == len(df1)
    assert len(out.read_text().splitlines()) == n      # nothing re-appended

    # cache hit path: same requests, fresh sink
    c2 = Cache(tmp_path / "cache")
    execute(units(cells, its), Answers(), render, echo_score,
        tmp_path / "runs2.jsonl", cache=c2, workers=2, log_every=1000)
    assert c2.hits == len(cells) * len(its) and c2.misses == 0


def test_a_cache_hit_gives_the_same_row_as_a_fresh_call(tmp_path):
    """B19.  Two sinks share one cache.  The first run calls the backend and
    its response is truncated; the second must be served from the cache
    and produce the same row (truncated, not scored), differing only in
    `cached`.  Expected: by construction the backend is called once and the
    scorer never (a truncated response is not scored)."""
    scored: list[str] = []

    def score(u: Unit, r: Response) -> dict[str, Any]:
        scored.append(u.id)
        return {"correct": 1.0}

    us = one_cell(1)
    cache, be = Cache(tmp_path / "cache"), Truncates(odd_only=False)
    run(us, be, tmp_path / "a.jsonl", cache=cache, score=score)
    run(us, be, tmp_path / "b.jsonl", cache=cache, score=score)
    fresh, hit = raw_rows(tmp_path / "a.jsonl")[-1], raw_rows(tmp_path / "b.jsonl")[-1]
    assert be.n == 1 and scored == []
    assert fresh["cached"] is False and hit["cached"] is True
    keep = ("unit_id", "request_key", "text", "error", "status", "meta_truncated")
    assert_same({k: hit.get(k) for k in keep}, {k: fresh.get(k) for k in keep},
                what="cached row vs fresh row")
    assert hit["status"] == "truncated"


def test_the_frame_carries_the_cluster_into_summarize(tmp_path):
    """A value constant within each cluster and varying across clusters has
    MS_w = 0, so rho = 1 exactly and n_eff = N / (1 + (m-1)) = k: twenty
    clusters of five rows are worth twenty rows.  The expected value comes
    from eq. (3), not from running the estimator; it holds only if the
    runner wrote each augmentation's parent as its cluster."""
    cells = grid({"model": ["m1"], "temperature": [0.0]})
    its = items(20, aug=4)
    df = execute(units(cells, its), Answers(), render, echo_score,
             tmp_path / "r.jsonl", workers=1, log_every=1000)
    df["v"] = df["cluster"].str[1:].astype(int) % 3   # gold: shared in cluster
    s = summarize(df, "v", by=["model"], n_boot=500)
    assert len(s) == 1
    row = s.iloc[0]
    assert row["n_clusters"] == 20 and row["n_rows"] == 100
    assert row["rho"] == pytest.approx(1.0)
    assert row["deff"] == pytest.approx(5.0)
    assert row["n_eff"] == pytest.approx(20.0)


# --- meta: numbers stay numbers, arrays go to a sidecar ---------------------------

def _numpy_backend():
    def fn(req):
        return Response(req.unit_id, text="x", meta={
            "truncated": False, "score": np.float32(0.25), "n": np.int64(3),
            "flag": np.bool_(True), "heads": np.arange(6.0).reshape(2, 3)})
    return FunctionBackend(fn, identity={"m": "numpy"})


@pytest.mark.parametrize("use_cache", [False, True])
def test_numpy_scalars_stay_numbers_and_arrays_go_to_a_sidecar(tmp_path,
                                                                use_cache):
    out = tmp_path / "r.jsonl"
    df = execute(units(grid({"a": [1]}), items(1)), _numpy_backend(), render,
             echo_score, out, workers=1, stream=io.StringIO(),
             cache=Cache(tmp_path / "c") if use_cache else None)
    row = df.iloc[0]
    assert row["status"] == "ok"
    raw = json.loads(out.read_text().splitlines()[-1])   # types as stored
    assert raw["meta_score"] == 0.25 and isinstance(raw["meta_score"], float)
    assert raw["meta_n"] == 3 and isinstance(raw["meta_n"], int)
    assert raw["meta_flag"] is True
    assert "meta_heads" not in df.columns            # not inlined as text
    arrays = load_arrays(out, row["meta_arrays"])
    assert np.array_equal(arrays["heads"], np.arange(6.0).reshape(2, 3))


def test_meta_arrays_resolve_from_their_own_sink_under_a_shared_cache(tmp_path):
    """B20.  The backend returns heads = arange(6).reshape(2, 3).  Run once
    into a/r.jsonl, then into b/deep/r.jsonl with the same cache (a cache
    hit).  Each sink's own `meta_arrays` must load that array from that
    sink."""
    us = list(units(grid({"a": [1]}), items(1)))
    cache = Cache(tmp_path / "cache")
    out1, out2 = tmp_path / "a" / "r.jsonl", tmp_path / "b" / "deep" / "r.jsonl"
    run(us, _numpy_backend(), out1, cache=cache)
    df2 = run(us, _numpy_backend(), out2, cache=cache)
    assert bool(df2["cached"].iloc[0])                  # the premise: a hit
    got = load_arrays(out2, df2["meta_arrays"].iloc[0])
    assert np.array_equal(got["heads"], np.arange(6.0).reshape(2, 3))


def test_a_torch_scalar_in_meta_is_stored_as_a_number(tmp_path):
    """B21.  meta {"s": torch.tensor(0.25)}: 0.25 is exact in binary, so the
    stored value must be the float 0.25 (as for numpy scalars)."""
    torch = pytest.importorskip("torch")

    def fn(req: Request) -> Response:
        return Response(req.unit_id, text="x", meta={"s": torch.tensor(0.25)})

    out = tmp_path / "r.jsonl"
    run(list(units(grid({"a": [1]}), items(1))),
        FunctionBackend(fn, identity={"m": "torch"}), out)
    s = raw_rows(out)[-1]["meta_s"]
    assert isinstance(s, float) and s == 0.25


def test_nested_numpy_in_meta_survives_the_cache(tmp_path):
    """B21.  meta {"d": {"x": np.float32(0.5)}} with a cache: 0.5 is exact in
    float32, so the stored value must be {"x": 0.5}, as it is without a
    cache."""
    def fn(req: Request) -> Response:
        return Response(req.unit_id, text="x",
                        meta={"d": {"x": np.float32(0.5)}})

    out = tmp_path / "r.jsonl"
    run(list(units(grid({"a": [1]}), items(1))),
        FunctionBackend(fn, identity={"m": "nested"}), out,
        cache=Cache(tmp_path / "cache"))
    d = raw_rows(out)[-1]["meta_d"]
    assert d == {"x": 0.5} and isinstance(d["x"], float)
