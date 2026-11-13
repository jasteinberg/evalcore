"""HFBackend sampling (class docstring, "Sampling"): each row draws from its
own generator, seeded from its request and repeat.

Expected values: the Gumbel-max frequencies are checked against the closed
form softmax(z / T); the batch, order, shard and global-seed tests are
invariances (one row's text under every arrangement must be the same); the
request keys are the documented formula written out by hand."""

from __future__ import annotations

import math

import pytest

from evalcore import HFBackend, Request
from evalcore.core.backends.base import request_key
from evalcore.core.runner import execute
from evalcore.core.shards import shard_of
from evalcore.core.spec import Cell, Item, Unit, digest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

from support.toy_hf import CharTok, tiny_qwen, user  # noqa: E402

from evalcore.core.backends.hf import _row_gumbel  # noqa: E402

PROMPTS = ["abcde", "the river", "where is it, q", "x", "a b c", "qq rr"]


@pytest.fixture(scope="module")
def qwen():
    return tiny_qwen(len(CharTok().vocab), n_layers=2, width=32)


def test_gumbel_argmax_frequencies_are_softmax_of_z_over_t():
    """Closed form: for z = (0, 1, 2, -inf) at T = 0.5, P(k) is
    exp(2 z_k) / sum, i.e. (1, e^2, e^4, 0) / (1 + e^2 + e^4).  N rows,
    each with its own seed, are one draw each; every frequency must sit
    within 5 binomial standard errors, and the masked token is never
    drawn."""
    n, t = 40_000, 0.5
    z = torch.tensor([0.0, 1.0, 2.0, -math.inf]).repeat(n, 1)
    proc = _row_gumbel(list(range(n)), t)
    drawn = proc(torch.zeros((n, 1), dtype=torch.long), z).argmax(-1)
    w = [1.0, math.e ** 2, math.e ** 4, 0.0]
    want = [x / sum(w) for x in w]
    freq = torch.bincount(drawn, minlength=4).double() / n
    for k in range(3):
        se = math.sqrt(want[k] * (1 - want[k]) / n)
        assert abs(float(freq[k]) - want[k]) < 5 * se, (k, float(freq[k]))
    assert int(freq[3] * n) == 0


def _units(repeats: int) -> list[Unit]:
    cell = Cell({})
    return [Unit(cell, Item(f"i{k}", {"prompt": p}), r)
            for k, p in enumerate(PROMPTS) for r in range(repeats)]


def _render(u: Unit) -> Request:
    return Request(u.id, [{"role": "user", "content": u.item.payload["prompt"]}],
                   dict(u.cell.params))


def _texts(qwen, tmp_path, name, units, global_seed, **kw) -> dict[str, str]:
    """unit_id -> text, from a fresh backend (no cache), with torch's
    global generator set to `global_seed` first."""
    be = HFBackend(qwen, CharTok(), device="cpu", generate=True,
                   default_params={"max_tokens": 8, "temperature": 1.0})
    torch.manual_seed(global_seed)
    df = execute(units, be, _render, lambda u, r: {}, tmp_path / f"{name}.jsonl",
                 workers=1, stream=None, events=False, **kw)
    # a row that used all 8 tokens is "truncated", which is still a text
    assert df["status"].isin(["ok", "truncated"]).all(), df["error"].tolist()
    return dict(zip(df["unit_id"], df["text"], strict=True))


def test_a_sampled_row_is_the_same_alone_batched_reordered_and_sharded(
        qwen, tmp_path):
    """Invariance: each unit's text is the same run one at a time, in
    batches of 4 (which pad the prompts differently), in reverse order with
    batches of 3, and split over two shards, with torch's global generator
    seeded differently every time.  Repeats of one prompt are distinct
    draws, so not all of a prompt's repeats agree."""
    us = _units(repeats=3)
    alone = _texts(qwen, tmp_path, "alone", us, 1, batch_size=1)
    assert alone == _texts(qwen, tmp_path, "b4", us, 2, batch_size=4)
    assert alone == _texts(qwen, tmp_path, "rev", us[::-1], 3, batch_size=3)
    sharded: dict[str, str] = {}
    for i in range(2):
        mine = [u for u in us if shard_of(u.id, 2) == i]
        assert mine
        sharded.update(_texts(qwen, tmp_path, f"s{i}", mine, 4 + i,
                              batch_size=4))
    assert sharded == alone
    by_prompt: dict[str, set[str]] = {}
    for u in us:
        by_prompt.setdefault(u.item.item_id, set()).add(alone[u.id])
    assert any(len(v) > 1 for v in by_prompt.values())


def test_the_seed_is_recorded_and_follows_the_request_and_repeat():
    """The documented seed: 63 bits of digest(cache_key, repeat), written by
    hand here.  Two repeats of one request get different seeds."""
    a = user("u", "ab", {"temperature": 1.0})
    b = Request(**{**a.__dict__, "repeat": 1})
    for r in (a, b):
        want = int(digest({"request": r.cache_key, "repeat": r.repeat}, n=8),
                   16) >> 1
        assert HFBackend.sample_seed(r) == want
    assert HFBackend.sample_seed(a) != HFBackend.sample_seed(b)


def test_only_a_sampled_request_key_carries_the_sampler(qwen):
    """request_key = digest(cache_key, identity, repeat [, context]), written
    out by hand.  A greedy request keeps the key without a context, and a
    sampled one adds {"sampler": "row-gumbel"}; with generate off there is
    no sampling and no context."""
    be = HFBackend(qwen, CharTok(), device="cpu", generate=True,
                   default_params={"max_tokens": 4})
    greedy = user("g", "ab", {"temperature": 0.0})
    hot = user("h", "ab", {"temperature": 0.8})
    ident = be.identity()
    assert request_key(greedy, 0, be) == digest(
        {"request": greedy.cache_key, "backend": ident, "repeat": 0})
    assert request_key(hot, 0, be) == digest(
        {"request": hot.cache_key, "backend": ident, "repeat": 0,
         "context": {"sampler": "row-gumbel"}})
    fwd = HFBackend(qwen, CharTok(), device="cpu", generate=False)
    assert fwd.key_context(hot) is None
