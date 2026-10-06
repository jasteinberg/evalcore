"""Harness failure modes that yield a plausible number instead of a crash.

Each test pins one way the runner can lose, duplicate, mis-join or mis-score
a unit while every statistic downstream still computes.  Expected values
come from the construction of the fixture -- which unit asked what, how many
calls a backend received -- never from running the code under test.

Row counts are taken from the raw JSONL wherever duplication is the
question: `to_frame` deduplicates on unit_id, so a frame cannot show a
duplicated row even when the sink holds one.
"""

from __future__ import annotations

import dataclasses
import importlib.util
import io
import json
import random
import threading
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import evalcore.core.backends.base as backends_mod
from evalcore import (
    Backend,
    Cache,
    Cell,
    Fatal,
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
    read_manifest,
    summarize,
    to_frame,
    units,
    write_manifest,
)
from evalcore.core.backends import ADAPTERS, DECODE_KEYS, request_key

REPO = Path(__file__).resolve().parents[2]


def items(n: int) -> list[Item]:
    return [Item(f"q{i}", {"q": f"question {i}"}) for i in range(n)]


def render(u: Unit) -> Request:
    return Request(u.id, [{"role": "user", "content": u.item.payload["q"]}],
                   dict(u.cell.params))


def echo_score(u: Unit, r: Response) -> dict[str, Any]:
    return {"answer": r.text}


def raw_rows(path: Path) -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def cache_files(root: Path) -> list[Path]:
    return sorted(root.rglob("*.json"))


@pytest.fixture
def no_backoff(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retries sleep U(0, 2^k) s; the schedule is not what is under test."""
    monkeypatch.setattr(backends_mod.time, "sleep", lambda s: None)


class Answers(Backend):
    """Answers each request with its own prompt, so a mis-join is visible as
    a row whose `answer` names another item.  Counts calls per prompt."""

    def identity(self) -> dict:
        return {"backend": "answers"}

    name = "answers"

    def __init__(self) -> None:
        self.calls: Counter[str] = Counter()
        self.lock = threading.Lock()

    def complete(self, req: Request) -> Response:
        q = req.messages[-1]["content"]
        with self.lock:
            self.calls[q] += 1
        return Response(req.unit_id, text=q)


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


@pytest.mark.parametrize("batch_size", [1, 4])
def test_fail_once_then_succeed_writes_exactly_one_row(tmp_path, no_backoff,
                                                       batch_size):
    n = 8
    out, cache = tmp_path / "r.jsonl", Cache(tmp_path / "cache")
    be = FailsOnce()
    execute(units(grid({"model": ["m"]}), items(n)), be, render, echo_score, out,
        cache=cache, workers=1, batch_size=batch_size, stream=io.StringIO())

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


@pytest.mark.parametrize("fail", [_raise_fatal, _raise_transient,
                                  _return_error, _raise_unknown],
                         ids=["fatal", "transient-exhausted", "returned-error",
                              "unknown-exception"])
def test_an_error_is_never_cached(tmp_path, no_backoff, fail):
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

    us = list(units(grid({"model": ["m"]}), items(4)))
    out, cache = tmp_path / "r.jsonl", Cache(tmp_path / "cache")
    df1 = execute(us, Flaky(failing=True), render, echo_score, out, cache=cache,
              workers=1, attempts=3, stream=io.StringIO())
    assert df1.set_index("item_id").loc["q1", "error"]     # it did fail
    assert len(cache_files(tmp_path / "cache")) == 3       # error not stored

    be = Flaky(failing=False)
    df2 = execute(us, be, render, echo_score, out, cache=Cache(tmp_path / "cache"),
              workers=1, resume=False, stream=io.StringIO())
    assert be.calls == Counter({bad: 1})                   # only the failure
    assert df2["error"].isna().all()
    assert df2.set_index("item_id").loc["q1", "answer"] == bad


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

    df = execute(units(grid({"model": ["m"]}), items(n)), Reverse(), render,
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
    df = execute(units(grid({"model": ["m"]}), items(n)), Permuted(), render,
             echo_score, tmp_path / "r.jsonl", workers=1, batch_size=n,
             stream=io.StringIO())
    assert len(df) == n and df["error"].isna().all()
    assert (df["answer"] == "question " + df["item_id"].str[1:]).all()


def test_batch_that_drops_a_response_gives_an_error_row(tmp_path):
    """A short batch must cost exactly the unit it lost: an error row for it,
    the right answer for every other unit, no crash."""
    n = 6
    df = execute(units(grid({"model": ["m"]}), items(n)), Permuted(drop=2),
             render, echo_score, tmp_path / "r.jsonl", workers=1,
             batch_size=n, stream=io.StringIO())
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
    us = list(units(grid({"model": ["m"]}), items(4)))
    out = tmp_path / "r.jsonl"
    execute(us, Answers(), render, echo_score, out, workers=1,
        stream=io.StringIO())
    lines = out.read_text().splitlines()
    out.write_text("\n".join(lines[:3]) + "\n" + lines[3][:40])   # the crash

    with pytest.warns(RuntimeWarning, match="1 unreadable line"):
        df = to_frame(out)
    assert sorted(df["item_id"]) == ["q0", "q1", "q2"]

    log, be = io.StringIO(), Answers()
    with pytest.warns(RuntimeWarning, match="1 unreadable line"):
        df = execute(us, be, render, echo_score, out, workers=1, stream=log)
    assert "unreadable" in log.getvalue()
    assert be.calls == Counter({"question 3": 1})          # only the lost unit
    assert sorted(df["item_id"]) == ["q0", "q1", "q2", "q3"]
    assert df.set_index("item_id").loc["q3", "answer"] == "question 3"
    json.loads(out.read_text().splitlines()[-1])           # last row intact


def test_manifest_append_after_a_truncated_line_keeps_every_unit(tmp_path):
    us = list(units(grid({"model": ["m"]}), items(6)))
    mp = tmp_path / "plan.manifest.jsonl"
    write_manifest(us[:3], mp)
    keep = mp.read_text().splitlines()
    mp.write_text("\n".join(keep[:2]) + "\n" + keep[2][:30])     # crash
    write_manifest(us, mp)
    assert set(read_manifest(mp)["unit_id"]) == {u.id for u in us}


# --- parse failures ---------------------------------------------------------

def test_a_scorer_exception_is_an_error_row_not_an_incorrect_answer(tmp_path):
    """A parser that raises must leave the metric MISSING, counted as an
    error in attrition -- never a 0 averaged into accuracy."""
    def strict_score(u: Unit, r: Response) -> dict[str, Any]:
        digits = [c for c in r.text if c.isdigit()]
        if not digits:
            raise ValueError("no answer found")
        return {"correct": float(digits[0] == "1")}

    class Mixed(Backend):
        def identity(self) -> dict:
            return {"backend": "mixed"}

        def complete(self, req: Request) -> Response:
            q = req.messages[-1]["content"]
            return Response(req.unit_id,
                            text="I cannot say" if q.endswith("2") else q)

    df = execute(units(grid({"model": ["m"]}), items(4)), Mixed(), render,
             strict_score, tmp_path / "r.jsonl", workers=1,
             stream=io.StringIO())
    got = df.set_index("item_id")
    assert got.loc["q2", "error"].startswith("score: ValueError")
    assert pd.isna(got.loc["q2", "correct"])
    assert got["correct"].count() == 3                     # not 4 with a zero
    a = attrition(df, ["model"]).iloc[0]
    assert a["n_err"] == 1 and a["n_rows"] == 4


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


# --- truncation -------------------------------------------------------------

class Truncates(Backend):
    """Every second item comes back cut off at the budget."""

    def identity(self) -> dict:
        return {"backend": "truncates"}

    def complete(self, req: Request) -> Response:
        i = int(req.messages[-1]["content"].split()[-1])
        return Response(req.unit_id, text="The answer is",
                        meta={"truncated": i % 2 == 1})


def test_truncation_flag_reaches_the_frame(tmp_path):
    df = execute(units(grid({"model": ["m"]}), items(4)), Truncates(), render,
             echo_score, tmp_path / "r.jsonl", workers=1, stream=io.StringIO())
    got = df.set_index("item_id")["meta_truncated"]
    assert got.to_dict() == {"q0": False, "q1": True, "q2": False, "q3": True}


def test_a_truncated_generation_is_not_scored(tmp_path):
    scored: list[str] = []

    def score(u: Unit, r: Response) -> dict[str, Any]:
        scored.append(u.item.item_id)
        return {"correct": 0.0}

    df = execute(units(grid({"model": ["m"]}), items(4)), Truncates(), render,
             score, tmp_path / "r.jsonl", workers=1, stream=io.StringIO())
    assert sorted(scored) == ["q0", "q2"]
    got = df.set_index("item_id")
    assert got.loc[["q1", "q3"], "correct"].isna().all()
    assert got.loc[["q1", "q3"], "meta_truncated"].all()
    assert got["status"].to_dict() == {"q0": "ok", "q1": "truncated",
                                       "q2": "ok", "q3": "truncated"}


# --- content addressing: every field of the request ------------------------
# Two values per decode key, chosen to differ; nested values included because
# `stop` is a list and a key that only looked at its first element would pass
# a scalar-only test.
DECODE_VALUES: dict[str, tuple[Any, Any]] = {
    "temperature": (0.0, 0.7), "top_p": (1.0, 0.9), "top_k": (1, 40),
    "max_tokens": (8, 256), "seed": (0, 1), "stop": (["\n"], ["\n", "##"]),
    "stop_sequences": (["</a>"], ["</b>"]), "repetition_penalty": (1.0, 1.1),
}
MSGS = [{"role": "system", "content": "be terse"},
        {"role": "user", "content": "2+2?"}]


def test_request_fields_are_the_ones_the_cache_key_covers():
    """If Request grows a field (tools, a response schema), this fails until
    cache_key is taught about it -- a field outside the key is a stale hit."""
    assert {f.name for f in dataclasses.fields(Request)} == {
        "unit_id", "messages", "params", "target", "tools", "input_ids",
        "candidates"}


def test_input_ids_and_candidates_move_the_cache_key():
    base = Request("u", MSGS, {})
    a = Request("u", MSGS, {}, input_ids=[1, 2, 3])
    b = Request("u", MSGS, {}, input_ids=[1, 2, 4])
    c = Request("u", MSGS, {}, input_ids=[1, 2, 3], candidates=[5, 6])
    d = Request("u", MSGS, {}, input_ids=[1, 2, 3], candidates=[6, 5])
    assert len({base.cache_key, a.cache_key, b.cache_key, c.cache_key,
                d.cache_key}) == 5                       # order matters too


def test_tools_move_the_cache_key():
    spec = {"name": "search", "description": "d", "parameters": {}}
    a = Request("u", MSGS, {"model": "m"})
    b = Request("u", MSGS, {"model": "m"}, tools=[spec])
    c = Request("u", MSGS, {"model": "m"}, tools=[{**spec, "description": "e"}])
    assert len({a.cache_key, b.cache_key, c.cache_key}) == 3


def test_target_moves_the_cache_key_and_absence_keeps_old_keys():
    base = Request("u", MSGS, {"model": "m"})
    a = Request("u", MSGS, {"model": "m"}, target=" 46")
    b = Request("u", MSGS, {"model": "m"}, target=" 47")
    assert len({base.cache_key, a.cache_key, b.cache_key}) == 3
    # a request with no target keeps the key it had before targets existed
    from evalcore.core.spec import digest
    assert base.cache_key == digest({"m": [dict(x) for x in MSGS],
                                     "p": {"model": "m"}})


def test_decode_values_cover_every_decode_key():
    assert set(DECODE_VALUES) == set(DECODE_KEYS)


@pytest.mark.parametrize("key", [*sorted(DECODE_KEYS), "model"])
def test_every_decode_key_moves_the_cache_key(key):
    a, b = DECODE_VALUES.get(key, ("model-a", "model-b"))
    base = {"model": "m", "max_tokens": 16}
    k_a = Request("u", MSGS, {**base, key: a}).cache_key
    k_b = Request("u", MSGS, {**base, key: b}).cache_key
    k_absent = Request("u", MSGS, {k: v for k, v in base.items()
                                   if k != key}).cache_key
    assert len({k_a, k_b, k_absent}) == 3


@pytest.mark.parametrize("key", [*sorted(DECODE_KEYS), "model", "template"])
def test_every_param_moves_the_cell_and_unit_id(key):
    a, b = DECODE_VALUES.get(key, ("x", "y"))
    item = Item("q0", {"q": "?"})
    ca, cb = Cell({"model": "m", key: a}), Cell({"model": "m", key: b})
    assert ca.id != cb.id
    assert Unit(ca, item).id != Unit(cb, item).id


def test_every_message_field_moves_the_cache_key():
    base = Request("u", MSGS, {"model": "m"}).cache_key
    variants = [
        [MSGS[0], {**MSGS[1], "content": "2+3?"}],          # content
        [MSGS[0], {**MSGS[1], "role": "assistant"}],        # role
        [MSGS[0], {**MSGS[1], "name": "alice"}],            # extra field
        [MSGS[1], MSGS[0]],                                 # order
        [MSGS[1]],                                          # system dropped
        [{**MSGS[0], "content": "be verbose"}, MSGS[1]],    # system edited
    ]
    keys = [Request("u", v, {"model": "m"}).cache_key for v in variants]
    assert base not in keys and len(set(keys)) == len(keys)


def test_unit_id_covers_cell_item_and_repeat():
    c, i = Cell({"model": "m"}), Item("q0", {"q": "?"})
    base = Unit(c, i, 0).id
    assert Unit(Cell({"model": "n"}), i, 0).id != base
    assert Unit(c, Item("q1", {"q": "?"}), 0).id != base
    assert Unit(c, i, 1).id != base
    assert Unit(Cell({"model": "m"}, tags={"arm": "renamed"}), i, 0).id == base


@pytest.mark.parametrize("flavour", sorted(ADAPTERS))
def test_adapters_refuse_decode_keys_they_would_drop(flavour):
    """A key in the cell params that never reaches the provider makes two
    cells with different ids send identical requests: a sweep over it is a
    clean null result about nothing.  Each key must be either forwarded or
    refused."""
    adapt, _ = ADAPTERS[flavour]
    for key in DECODE_KEYS:
        a, b = DECODE_VALUES[key]
        p = {"max_tokens": 16}
        try:
            _, body_a = adapt("m", MSGS, {**p, key: a})
            _, body_b = adapt("m", MSGS, {**p, key: b})
        except Fatal as exc:
            assert key in str(exc), f"{flavour}: refusal must name {key!r}"
            continue
        assert body_a != body_b, f"{flavour} silently drops {key!r}"


def test_repeats_are_independent_samples_under_the_cache(tmp_path):
    rng = random.Random(0)

    class Sampler(Backend):
        def identity(self) -> dict:
            return {"backend": "sampler"}

        def complete(self, req: Request) -> Response:
            return Response(req.unit_id, text=f"{rng.random():.12f}")

    cells = grid({"model": ["m"], "temperature": [1.0]})
    df = execute(units(cells, items(1), repeats=5), Sampler(), render, echo_score,
             tmp_path / "r.jsonl", cache=Cache(tmp_path / "c"), workers=1,
             stream=io.StringIO())
    assert df["answer"].nunique() == 5
    assert not df["cached"].any()


def test_backend_defaults_are_part_of_the_cache_key(tmp_path):
    class ModelFromDefaults(Backend):
        def identity(self) -> dict:
            return self._base_identity()

        def __init__(self, model: str) -> None:
            self.default_params = {"model": model}

        def complete(self, req: Request) -> Response:
            return Response(req.unit_id, text=self.default_params["model"])

    def render_no_model(u: Unit) -> Request:
        return Request(u.id, [{"role": "user", "content": "q"}],
                       {"temperature": 0.0})

    us, cache = list(units(grid({"arm": ["x"]}), items(2))), Cache(tmp_path)
    execute(us, ModelFromDefaults("model-a"), render_no_model, echo_score,
        tmp_path / "a.jsonl", cache=cache, workers=1, stream=io.StringIO())
    b = execute(us, ModelFromDefaults("model-b"), render_no_model, echo_score,
            tmp_path / "b.jsonl", cache=cache, workers=1, stream=io.StringIO())
    assert (b["answer"] == "model-b").all()


def test_resume_does_not_reuse_a_row_rendered_from_a_different_prompt(
        tmp_path):
    template = {"plain": "{q}"}

    def render_t(u: Unit) -> Request:
        body = template[u.cell.get("template")].format(q=u.item.payload["q"])
        return Request(u.id, [{"role": "user", "content": body}],
                       {"model": "m"})

    us = list(units(grid({"template": ["plain"]}), items(2)))
    out = tmp_path / "r.jsonl"
    execute(us, Answers(), render_t, echo_score, out, workers=1,
        stream=io.StringIO())
    template["plain"] = "Answer briefly. {q}"               # edited in code
    df = execute(us, Answers(), render_t, echo_score, out, workers=1,
             stream=io.StringIO())
    assert df["answer"].str.startswith("Answer briefly.").all()


def test_hf_generate_refuses_decode_keys_it_would_drop():
    """`_run_batch` reads max_tokens, temperature and repetition_penalty and
    nothing else, so any other decode key in a generate request is dropped."""
    from evalcore import HFBackend

    class NoTorch(HFBackend):
        def _run_batch(self, reqs):
            return [Response(r.unit_id, text="ran") for r in reqs]

    be = NoTorch(model=None, tokenizer=None, generate=True)
    for key in sorted(set(DECODE_KEYS) - {"max_tokens", "temperature",
                                          "repetition_penalty"}):
        req = Request("u", MSGS, {"max_tokens": 8, key: DECODE_VALUES[key][1]})
        with pytest.raises(Fatal, match=key):
            be.complete_batch([req])
    ok = Request("u", MSGS, {"max_tokens": 8, "temperature": 0.7,
                             "repetition_penalty": 1.1})
    assert be.complete_batch([ok])[0].text == "ran"


# --- the request key: what resume and the cache are keyed on ---------------

def test_every_row_records_its_request_key(tmp_path):
    us = list(units(grid({"model": ["m"]}), items(3), repeats=2))
    be = Answers()
    df = execute(us, be, render, echo_score, tmp_path / "r.jsonl", workers=1,
             stream=io.StringIO())
    want = {u.id: request_key(render(u), u.repeat, be) for u in us}
    assert dict(zip(df["unit_id"], df["request_key"], strict=True)) == want
    assert df["request_key"].nunique() == 6                # repeats distinct


def test_an_edited_item_reruns_that_unit_only(tmp_path):
    """A corrected item keeps its item_id, hence its unit_id.  Resume must
    redo exactly the units it feeds, say so, and leave the rest alone."""
    its = items(4)
    us = list(units(grid({"model": ["m1", "m2"]}), its))
    out = tmp_path / "r.jsonl"
    execute(us, Answers(), render, echo_score, out, workers=1,
        stream=io.StringIO())

    fixed = [Item("q2", {"q": "question 2, corrected"}) if i.item_id == "q2"
             else i for i in its]
    log, be = io.StringIO(), Answers()
    df = execute(units(grid({"model": ["m1", "m2"]}), fixed), be, render,
             echo_score, out, workers=1, stream=log)
    assert be.calls == Counter({"question 2, corrected": 2})   # one per cell
    assert "2 finished units were made from a different request" in \
        log.getvalue()
    assert len(df) == 8
    assert (df.set_index("item_id").loc["q2", "answer"]
            == "question 2, corrected").all()


def test_rows_without_a_request_key_are_redone(tmp_path):
    """A row with no request_key cannot be checked against the request it
    would render to now, so it is not trusted as done: it is stale."""
    us = list(units(grid({"model": ["m"]}), items(3)))
    out = tmp_path / "r.jsonl"
    execute(us, Answers(), render, echo_score, out, workers=1,
        stream=io.StringIO())
    rows = raw_rows(out)
    out.write_text("".join(json.dumps({k: v for k, v in r.items()
                                       if k != "request_key"}) + "\n"
                           for r in rows))
    log, be = io.StringIO(), Answers()
    execute(us, be, render, echo_score, out, workers=1, stream=log)
    assert sum(be.calls.values()) == 3
    assert "3 finished units were made from a different request" in log.getvalue()


def test_backend_identity_tracks_output_not_credentials():
    httpx = pytest.importorskip("httpx")  # noqa: F841 - HTTPBackend needs it
    from evalcore import HTTPBackend

    def ident(**kw):
        be = HTTPBackend("openai", api_key=kw.pop("api_key", "k1"), **kw)
        try:
            return be.identity()
        finally:
            be.close()

    base = ident(default_params={"model": "a"})
    assert ident(default_params={"model": "a"}, api_key="k2") == base
    assert ident(default_params={"model": "a"},
                 extra_headers={"x-trace": "1"}) == base
    assert ident(default_params={"model": "b"}) != base
    assert ident(default_params={"model": "a"},
                 base_url="http://localhost:8000") != base
    assert "k1" not in json.dumps(base)


# --- row status: five states, one partition ----------------------------------

@pytest.mark.filterwarnings("ignore:.*carry caveats:RuntimeWarning")  # tiny fixture
def test_every_planned_unit_lands_in_exactly_one_state(tmp_path):
    """One unit per state, fixed by construction:
    q0, q5 ok; q1 truncated; q2 unparsed; q3 call refused; q4 scorer bug;
    q6 planned but never attempted (limit=).  The metric must see only the
    two ok rows, and every other unit must be accounted for, by name."""
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

    us = list(units(grid({"model": ["m"]}), items(7)))
    out = tmp_path / "r.jsonl"
    write_manifest(us, manifest_path(out))                  # plan all seven
    df = execute(us, Mixed(), render, scorer, out, workers=1, limit=6,
             stream=io.StringIO())

    got = df.set_index("item_id")
    assert got["status"].to_dict() == {
        "q0": "ok", "q1": "truncated", "q2": "unparsed", "q3": "error",
        "q4": "error", "q5": "ok"}
    assert got["correct"].count() == 2                      # only ok rows
    assert got.loc["q2", "parse_error"] == "no letter A-D"
    assert pd.isna(got.loc["q2", "error"])                  # not a failed call

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
    us = list(units(grid({"model": ["m"]}), items(3)))
    rows = []
    for u, sts in zip(us, [("ok", "unparsed"), ("unparsed", "truncated"),
                           ("ok", "ok")], strict=True):
        rows += [dict(u.row(), status=st, error=None) for st in sts]
    plan = pd.DataFrame([u.row() for u in us])
    a = attrition(pd.DataFrame(rows), ["model"], manifest=plan).iloc[0]
    assert (a["n_completed"], a["n_unparsed"], a["n_truncated"]) == (1, 1, 1)
    assert a["n_rows"] == 6


def test_rows_from_before_status_get_one_derived(tmp_path):
    out = tmp_path / "old.jsonl"
    out.write_text(json.dumps({"unit_id": "a", "error": None}) + "\n"
                   + json.dumps({"unit_id": "b", "error": "fatal: x"}) + "\n")
    assert to_frame(out)["status"].tolist() == ["ok", "error"]


# --- review findings (2 Oct) -------------------------------------------------

def test_resume_retries_units_whose_last_row_is_an_error(tmp_path, no_backoff):
    """An error row records a failed attempt, not a result.  A rerun must try
    those units again (and say so), unless asked not to."""
    us = list(units(grid({"model": ["m"]}), items(4)))
    out = tmp_path / "r.jsonl"

    class Down(Answers):
        """Same identity as Answers (so its rows are not stale against it),
        but every call fails."""

        def complete(self, req):
            raise Transient("HTTP 503")

    execute(us, Down(), render, echo_score, out, workers=1, attempts=2,
        stream=io.StringIO())
    kept = execute(us, Answers(), render, echo_score, out, workers=1,
               retry_errors=False, stream=io.StringIO())
    assert (kept["status"] == "error").all()

    log, be = io.StringIO(), Answers()
    df = execute(us, be, render, echo_score, out, workers=1, stream=log)
    assert sum(be.calls.values()) == 4
    assert (df["status"] == "ok").all()
    assert "4 units errored last time" in log.getvalue()


# --- backend contract (2 Oct) -------------------------------------------------

def test_a_target_sent_to_a_generating_backend_is_an_error_row(tmp_path):
    from evalcore import EchoBackend

    def render_t(u: Unit) -> Request:
        return Request(u.id, [{"role": "user", "content": "2 + 2 ="}], {},
                       target=" 4")

    df = execute(units(grid({"model": ["m"]}), items(2)), EchoBackend(), render_t,
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
        execute(units(grid({"model": ["m"]}), items(2)), MyModel("ckpt-A"),
            render, echo_score, tmp_path / "r.jsonl", workers=1,
            stream=io.StringIO())
    assert not (tmp_path / "r.jsonl").exists()          # refused up front


# --- numpy in probe output (2 Oct) --------------------------------------------

def _numpy_backend():
    import numpy as np

    from evalcore import FunctionBackend

    def fn(req):
        return Response(req.unit_id, text="x", meta={
            "truncated": False, "score": np.float32(0.25), "n": np.int64(3),
            "flag": np.bool_(True), "heads": np.arange(6.0).reshape(2, 3)})
    return FunctionBackend(fn, identity={"m": "numpy"})


@pytest.mark.parametrize("use_cache", [False, True])
def test_numpy_scalars_stay_numbers_and_arrays_go_to_a_sidecar(tmp_path,
                                                                use_cache):
    import numpy as np

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
    from evalcore.core.records import load_arrays
    arrays = load_arrays(out, row["meta_arrays"])
    assert np.array_equal(arrays["heads"], np.arange(6.0).reshape(2, 3))
