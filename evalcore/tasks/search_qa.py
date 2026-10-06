r"""Questions that can only be answered by searching, generated.

A seeded world of invented entities -- towns, rivers, people -- each with a
few facts, written up as one short document per entity.  Names are built
from syllables, so no model knows them: an answer has to come from the
documents, which makes this a test of searching and composing, not of
memory.

    1 hop   How many people live in Askern?                 one search
    2 hop   How many people live in the town where the      search the river,
            Velor rises?                                    then the town
    2 hop   Which river flows through the town where        search the person,
            Tamsin Orrel was born?                          then the town

Every town shares a name stem with a distractor town, so a search returns
near-misses and the model has to read which document answers.

`meta["hops"]` counts the facts an answer composes, not a minimum number of
searches: town documents also name their river, so a well-chosen query can
return the river's document and the town's together (observed in a live
run, 2 Oct).  Report `meta_n_tool_calls` beside accuracy for search
efficiency.  Run the
same items with no tools for the closed-book baseline: accuracy there
should sit at the floor, and the gap is what tool use buys.

`search_qa(...)` returns the Task and a BM25 CorpusSearch over the corpus;
wrap a tool-capable backend as `AgentBackend(backend, [search.tool])`.
Each item also carries `gold_docs`, the documents its answer composes, so
the same questions score a retriever on its own (core.search's
RetrieverBackend with evaluation.retrieval.score_ranked).  The person ->
river question needs the person's and the town's documents (the town
document names the river), not the river's.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any

from ..core.search import CorpusSearch
from ..core.spec import Item
from .base import Task
from .formats import render_canonical
from .graders import make_score

__all__ = ["World", "make_world", "search_qa", "search_qa_items"]

_ON, _NUC, _CODA = ("b br d k l m n r s st t th v z".split(),
                    "a e i o u ae ia".split(), "n r s l th nd rn st".split())

SYSTEM = ("Answer the question. You can search a document collection; the "
          "answer is in it. Reply with the answer only.")


def _name(rng: random.Random, syllables: int = 2) -> str:
    s = "".join(rng.choice(_ON) + rng.choice(_NUC)
                for _ in range(syllables)) + rng.choice(_CODA)
    return s.capitalize()


@dataclass
class World:
    towns: dict[str, dict[str, Any]]
    rivers: dict[str, dict[str, Any]]
    people: dict[str, dict[str, Any]]
    docs: dict[str, str]
    doc_of: dict[str, str] = field(default_factory=dict)  # entity -> doc id


def make_world(n_towns: int = 20, seed: int = 0) -> World:
    """Towns in near-name pairs, one river rising in each pair's first town,
    one person born in each town."""
    rng = random.Random(seed)
    used: set[str] = set()

    def fresh(k: int = 2) -> str:
        while True:
            n = _name(rng, k)
            if n not in used:
                used.add(n)
                return n

    towns: dict[str, dict[str, Any]] = {}
    while len(towns) < n_towns:
        stem = fresh(2)
        for suffix in ("", "ford")[: max(1, min(2, n_towns - len(towns)))]:
            name = stem + suffix
            used.add(name)
            towns[name] = {"population": rng.randrange(800, 90_000, 10)}
    names = list(towns)
    rivers = {}
    for t in names[::2]:
        r = fresh(2)
        rivers[r] = {"source": t}
        towns[t]["river"] = r
    for t in names[1::2]:                     # the near-name twin: same river
        twin = t[:-4] if t.endswith("ford") else None
        if twin in towns and "river" in towns[twin]:
            towns[t]["river"] = towns[twin]["river"]
    people = {}
    for t in names:
        p = f"{fresh(2)} {fresh(2)}"
        people[p] = {"born": t}
    docs, doc_of = {}, {}
    for k, (t, f) in enumerate(towns.items()):
        river = f" The {f['river']} flows through it." if "river" in f else ""
        doc_of[t] = f"town{k:03d}"
        docs[doc_of[t]] = (f"{t} is a town with a population of "
                           f"{f['population']:,} people.{river}")
    for k, (r, f) in enumerate(rivers.items()):
        doc_of[r] = f"river{k:03d}"
        docs[doc_of[r]] = f"The {r} is a river that rises in {f['source']}."
    for k, (p, f) in enumerate(people.items()):
        doc_of[p] = f"person{k:03d}"
        docs[doc_of[p]] = f"{p} is a surveyor who was born in {f['born']}."
    return World(towns, rivers, people, docs, doc_of)


def search_qa_items(world: World, seed: int = 0) -> list[Item]:
    """One item per question; gold answers come from the world, so they
    are exact by construction."""
    rng = random.Random(seed)
    doc = world.doc_of
    qs: list[tuple[str, str, Any, Any, int, list[str]]] = []
    for t, f in world.towns.items():
        qs.append(("pop", f"How many people live in {t}?", f["population"],
                   {"name": "numeric", "tol": 0}, 1, [doc[t]]))
    for r, f in world.rivers.items():
        qs.append(("river_pop", f"How many people live in the town where the "
                   f"{r} rises?", world.towns[f["source"]]["population"],
                   {"name": "numeric", "tol": 0}, 2,
                   [doc[r], doc[f["source"]]]))
    rivers = sorted(world.rivers)
    for p, f in world.people.items():
        river = world.towns[f["born"]].get("river")
        if river:
            qs.append(("person_river", f"Which river flows through the town "
                       f"where {p} was born?", river,
                       {"name": "label", "options": rivers}, 2,
                       [doc[p], doc[f["born"]]]))
    rng.shuffle(qs)
    return [Item(f"sqa{seed}-{k:04d}",
                 {"messages": [{"role": "system", "content": SYSTEM},
                               {"role": "user", "content": q}],
                  "gold": gold, "grader": spec, "gold_docs": gold_docs,
                  "meta": {"type": t, "hops": hops}},
                 group=t)
            for k, (t, q, gold, spec, hops, gold_docs) in enumerate(qs)]


def search_qa(n_towns: int = 20, seed: int = 0, k: int = 3
              ) -> tuple[Task, CorpusSearch]:
    """The Task and a BM25 search over its world's documents (`.tool` for
    the agent loop, `.docs`).  For another retriever, build
    `CorpusSearch(world.docs, retriever)` over `make_world(n_towns, seed)`."""
    world = make_world(n_towns, seed)
    return (Task(f"search_qa_{n_towns}", search_qa_items(world, seed),
                 render=render_canonical, score=make_score()),
            CorpusSearch(world.docs, k=k))
