"""Config space: cells, deterministic identity, grid expansion.

A *cell* is one point in the configuration space of an experiment
(model x prompt variant x decode params x dataset slice x seed).  A *unit*
is one (cell, item, repeat) triple -- the thing that produces exactly one
model call and exactly one row in the tidy frame.

Identity is content-addressed: cell_id = blake2b of the canonical JSON of
the cell's fields.  This makes the cache key, the resume key, and the join
key into the analysis frame the same string, so a partially-completed run
merges with a later one without bookkeeping.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any


def canonical(obj: Any) -> str:
    """Stable JSON: sorted keys, no whitespace, no NaN."""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      allow_nan=False, default=str)


def digest(obj: Any, n: int = 16) -> str:
    return hashlib.blake2b(canonical(obj).encode(), digest_size=n).hexdigest()


@dataclass(frozen=True)
class Cell:
    """One configuration.  `params` carries everything that can change the
    model's behaviour; `tags` carries labels that must NOT affect identity
    (a human-readable arm name, a note).  Keeping tags out of the digest is
    what lets you rename an arm without invalidating the cache."""

    params: Mapping[str, Any]
    tags: Mapping[str, Any] = field(default_factory=dict)

    @property
    def id(self) -> str:
        return digest(dict(self.params))

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def merged(self) -> dict[str, Any]:
        """Flat dict for the tidy frame: params first, tags cannot shadow."""
        out = dict(self.tags)
        out.update(self.params)
        out["cell_id"] = self.id
        return out

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"Cell({self.id[:8]}, {canonical(dict(self.params))})"


def grid(axes: Mapping[str, Sequence[Any]],
         exclude: Iterable[Mapping[str, Any]] = (),
         tags: Mapping[str, Any] | None = None) -> list[Cell]:
    """Cartesian product of `axes`, minus any cell matching an `exclude`
    pattern (a partial dict; a cell is dropped if it matches on every key
    the pattern names).

    >>> len(grid({"model": ["a", "b"], "temp": [0.0, 1.0]}))
    4
    >>> len(grid({"model": ["a", "b"], "temp": [0.0, 1.0]},
    ...          exclude=[{"model": "a", "temp": 1.0}]))
    3
    """
    keys = list(axes)
    cells: list[Cell] = []
    for combo in itertools.product(*(axes[k] for k in keys)):
        params = dict(zip(keys, combo, strict=True))
        if any(all(params.get(k) == v for k, v in pat.items())
               for pat in exclude):
            continue
        cells.append(Cell(params=params, tags=dict(tags or {})))
    return cells


@dataclass(frozen=True)
class Item:
    """One dataset element.  `item_id` is the CLUSTER label used by every
    bootstrap downstream, so it must be stable across arms and must be
    shared by all augmentations derived from the same source element --
    that sharing is the whole reason the design effect is computable."""

    item_id: str
    payload: Mapping[str, Any]
    group: str | None = None          # optional stratum (subject, difficulty)
    parent_id: str | None = None      # set on augmentations; None if original

    @property
    def cluster(self) -> str:
        """The resampling unit: an augmentation clusters with its parent."""
        return self.parent_id or self.item_id


@dataclass(frozen=True)
class Unit:
    """(cell, item, repeat) -- one model call, one row."""

    cell: Cell
    item: Item
    repeat: int = 0

    @property
    def id(self) -> str:
        return digest({"cell": self.cell.id, "item": self.item.item_id,
                       "repeat": self.repeat})

    def row(self) -> dict[str, Any]:
        r = self.cell.merged()
        r.update(item_id=self.item.item_id, cluster=self.item.cluster,
                 group=self.item.group, repeat=self.repeat, unit_id=self.id)
        return r


def units(cells: Sequence[Cell], items: Sequence[Item],
          repeats: int = 1) -> Iterator[Unit]:
    for cell in cells:
        for item in items:
            for r in range(repeats):
                yield Unit(cell=cell, item=item, repeat=r)
