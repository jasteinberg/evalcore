"""The request key: what the cache and resume are keyed on.

request_key = digest(cache_key, backend identity, repeat), and cache_key
covers every field of the request as actually made.  A field outside the key
is a stale cache hit or a resumed row from another prompt -- a plausible
number from the wrong request.  Each test changes one thing that changes
the output and asserts the key moved (or, for what cannot change the
output, that it did not).
"""

from __future__ import annotations

import dataclasses
import io
import json
import random
from collections import Counter

import pytest
from support.runner import (
    DECODE_VALUES,
    Answers,
    echo_score,
    items,
    raw_rows,
    render,
)

from evalcore import (
    Backend,
    Cache,
    Fatal,
    Item,
    Request,
    Response,
    Unit,
    execute,
    grid,
    units,
)
from evalcore.core.backends import ADAPTERS, DECODE_KEYS, EchoBackend, request_key

MSGS = [{"role": "system", "content": "be terse"},
        {"role": "user", "content": "2+2?"}]


# --- content addressing: every field of the request ------------------------

def test_request_fields_are_the_ones_the_cache_key_covers():
    """If Request grows a field (tools, a response schema), this fails until
    cache_key is taught about it -- a field outside the key is a stale hit."""
    assert {f.name for f in dataclasses.fields(Request)} == {
        "unit_id", "messages", "params", "target", "tools", "input_ids",
        "candidates", "repeat"}
    # `repeat` is the exception: request_key covers it (below), and putting
    # it in cache_key as well would move every existing key with repeat >= 1


def test_request_key_moves_with_repeat_and_refuses_a_conflicting_one():
    b = EchoBackend()
    req = Request("u", MSGS, {})
    keys = {request_key(req, r, b) for r in range(3)}
    assert len(keys) == 3
    assert request_key(dataclasses.replace(req, repeat=2), 2, b) == \
        request_key(req, 2, b)               # the field itself moves nothing
    with pytest.raises(ValueError, match="repeat=2"):
        request_key(dataclasses.replace(req, repeat=2), 1, b)


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


# --- a param the key covers must reach the model ----------------------------

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


def test_hf_generate_refuses_decode_keys_it_would_drop():
    """`_run_batch` reads max_tokens, temperature and repetition_penalty and
    nothing else, so any other decode key in a generate request is dropped.
    The request comes back as an error naming the key, without running."""
    from evalcore import HFBackend

    class NoTorch(HFBackend):
        def _run_batch(self, reqs):
            return [Response(r.unit_id, text="ran") for r in reqs]

    be = NoTorch(model=None, tokenizer=None, generate=True)
    for key in sorted(set(DECODE_KEYS) - {"max_tokens", "temperature",
                                          "repetition_penalty"}):
        req = Request("u", MSGS, {"max_tokens": 8, key: DECODE_VALUES[key][1]})
        (r,) = be.complete_batch([req])
        assert r.text is None and key in r.error, key
    ok = Request("u", MSGS, {"max_tokens": 8, "temperature": 0.7,
                             "repetition_penalty": 1.1})
    assert be.complete_batch([ok])[0].text == "ran"


# --- backend identity and the repeat index ----------------------------------

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

    us, cache = list(units(grid({"condition": ["x"]}), items(2))), Cache(tmp_path)
    execute(us, ModelFromDefaults("model-a"), render_no_model, echo_score,
        tmp_path / "a.jsonl", cache=cache, workers=1, stream=io.StringIO())
    b = execute(us, ModelFromDefaults("model-b"), render_no_model, echo_score,
            tmp_path / "b.jsonl", cache=cache, workers=1, stream=io.StringIO())
    assert (b["answer"] == "model-b").all()


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


# --- resume: a finished row counts only for the request that made it --------

def test_every_row_records_its_request_key(tmp_path):
    us = list(units(grid({"model": ["m"]}), items(3), repeats=2))
    be = Answers()
    df = execute(us, be, render, echo_score, tmp_path / "r.jsonl", workers=1,
             stream=io.StringIO())
    want = {u.id: request_key(render(u), u.repeat, be) for u in us}
    assert dict(zip(df["unit_id"], df["request_key"], strict=True)) == want
    assert df["request_key"].nunique() == 6                # repeats distinct


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
