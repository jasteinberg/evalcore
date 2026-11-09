"""Gold controls: the oracle and gold-removed arms against rankings written
out by hand from their definitions (search.py, "Controls."), the refusals,
and both arms run and scored through the ordinary runner."""

from __future__ import annotations

import io

import pytest
from support.retrievers import Fixed

from evalcore import Item, grid
from evalcore.core.search import (
    BM25,
    GoldControl,
    RetrieverBackend,
    gold_by_query,
    search_tool,
)
from evalcore.core.spec import digest
from evalcore.evaluation.retrieval import score_ranked
from evalcore.tasks.base import Task
from evalcore.tasks.search_qa import make_world, search_qa_items

# full ranking pi = d (2), c (1), b (0.5), a (0)
SCORES = {"a": 0.0, "b": 0.5, "c": 1.0, "d": 2.0}


def test_oracle_puts_gold_first_in_base_order_then_fills_to_k():
    """G = {a, c}: gold in pi order is c, a; pi minus G is d, b."""
    r = GoldControl(Fixed(SCORES), {"q": ["a", "c"]}, "oracle")
    assert r.retrieve(["q"], k=3) == [[("c", 1.0), ("a", 0.0), ("d", 2.0)]]
    assert r.retrieve(["q"], k=2) == [[("c", 1.0), ("a", 0.0)]]


def test_removed_is_the_base_ranking_without_gold():
    """G = {d}: pi minus G is c, b, a.  At k = 4 only n - |G| = 3 remain."""
    r = GoldControl(Fixed(SCORES), {"q": ["d"]}, "removed")
    assert r.retrieve(["q"], k=3) == [[("c", 1.0), ("b", 0.5), ("a", 0.0)]]
    assert r.retrieve(["q"], k=4) == r.retrieve(["q"], k=3)


_DOCS = {"d1": "the cat sat", "d2": "the dog sat on the dog mat",
         "d3": "a bird", "d4": "fish swim"}


def test_bm25_controls_fill_with_documents_bm25_would_drop():
    """"cat" matches d1 only, so pi = d1, then d2, d3, d4 (all 0, by id).
    BM25 itself returns one document; the controls return k = 3."""
    bm = BM25(_DOCS)
    assert [d for d, _ in bm.retrieve(["cat"], 3)[0]] == ["d1"]
    oracle = GoldControl(bm, {"cat": ["d3"]}, "oracle")
    assert [d for d, _ in oracle.retrieve(["cat"], 3)[0]] == ["d3", "d1", "d2"]
    removed = GoldControl(bm, {"cat": ["d1"]}, "removed")
    assert [d for d, _ in removed.retrieve(["cat"], 3)[0]] == ["d2", "d3", "d4"]


def test_scores_are_the_base_scores():
    bm = BM25(_DOCS)
    want = dict(zip(bm.ids, bm.scores("cat"), strict=True))
    for mode in ("oracle", "removed"):
        for d, s in GoldControl(bm, {"cat": ["d3"]}, mode).retrieve(["cat"], 3)[0]:
            assert s == want[d]


def test_repeat_is_ignored():
    r = GoldControl(Fixed(SCORES), {"q": ["a"]}, "oracle")
    assert r.retrieve(["q"], 3, repeat=0) == r.retrieve(["q"], 3, repeat=5)


def test_refusals():
    base = Fixed(SCORES)
    with pytest.raises(ValueError, match="mode"):
        GoldControl(base, {"q": ["a"]}, "both")
    r = GoldControl(base, {"q": ["a", "b", "c"]}, "oracle")
    with pytest.raises(KeyError, match="no gold entry"):
        r.retrieve(["other"], 3)
    with pytest.raises(ValueError, match="do not fit"):
        r.retrieve(["q"], 2)
    with pytest.raises(ValueError, match="k must"):
        r.retrieve(["q"], 0)
    with pytest.raises(ValueError, match="not in the corpus"):
        GoldControl(base, {"q": ["zz"]}, "removed").retrieve(["q"], 2)


def test_identity_covers_mode_and_gold_but_not_gold_order():
    base = Fixed(SCORES)
    ident = GoldControl(base, {"q": ["a", "c"]}, "oracle").identity()
    assert ident == GoldControl(base, {"q": ["c", "a"]}, "oracle").identity()
    assert ident["base"] == base.identity()
    others = [GoldControl(base, {"q": ["a", "c"]}, "removed"),
              GoldControl(base, {"q": ["a"]}, "oracle"),
              GoldControl(base, {"p": ["a", "c"]}, "oracle")]
    assert len({digest(ident)} | {digest(o.identity()) for o in others}) == 4
    tool = search_tool(GoldControl(BM25(_DOCS), {"cat": ["d1"]}, "oracle"), _DOCS)
    assert tool.version.startswith("gold-oracle-")


def _item(i, q, gold):
    return Item(f"i{i}", {"messages": [{"role": "system", "content": "s"},
                                       {"role": "user", "content": q}],
                          "gold_docs": gold})


def test_gold_by_query_reads_the_last_user_message():
    items = [_item(0, "q1", ["b", "a"]), _item(1, "q2", {"c": 2, "d": 1}),
             _item(2, "q1", ["a", "b"])]               # same query, same gold
    assert gold_by_query(items) == {"q1": ["a", "b"], "q2": ["c", "d"]}
    with pytest.raises(ValueError, match="two different gold sets"):
        gold_by_query([*items, _item(3, "q1", ["a"])])


@pytest.mark.parametrize(("mode", "recall"), [("oracle", 1.0), ("removed", 0.0)])
def test_controls_through_the_runner(tmp_path, mode, recall):
    """Every search_qa question has at most 2 gold documents, so at k = 3 the
    oracle holds all of them (recall@3 = 1) and removed none (= 0), and
    both read exactly 3 documents."""
    world = make_world(6, 0)
    items = search_qa_items(world, 0)
    assert max(len(it.payload["gold_docs"]) for it in items) <= 2
    task = Task("sqa", items, score=score_ranked(ks=(3,)))
    r = GoldControl(BM25(world.docs), gold_by_query(items), mode)
    df = task.run(grid({"arm": [mode]}), RetrieverBackend(r, k=3),
                  tmp_path / "r.jsonl", workers=1, batch_size=8,
                  stream=io.StringIO())
    assert (df["status"] == "ok").all() and len(df) == len(items)
    assert df["recall@3"].tolist() == [recall] * len(items)
    assert df["depth"].tolist() == [3] * len(items)
