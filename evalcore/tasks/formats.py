"""The canonical task format, and translation into it from any other.

One record per item, as JSON or JSONL:

    {"id": "q17",                                  required, unique
     "prompt": "..."  or  "messages": [{"role": "user", "content": "..."}],
     "gold": ...,                                  required if graded
     "grader": "numeric" | {"name": "numeric", "rel": 0.01},
     "target": " 1046",                            teacher-forced scoring
     "gold_docs": ["d3", "d9"] or {"d3": 2, ...},  relevant documents
                                                   (retrieval scoring)
     "cluster": "q17",                             items sharing a source
     "group": "arithmetic",                        a stratum
     "meta": {...}}                                anything else

A file in another shape is translated by a `mapping`: either a dict from
canonical field to a dotted path in the foreign record ("answers.0" indexes
a list), or a function from a foreign record to a canonical one.  Most
formats need only the dict.

Validation is strict and names the record: a missing id, a duplicate id, a
prompt AND messages (or neither), an unknown field (a typo such as "glod"
would otherwise silently drop the gold), an unknown grader, a grader with
no gold.  Nothing is defaulted.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterable, Mapping
from pathlib import Path
from typing import Any

from ..core.backends import Request
from ..core.spec import Item, Unit
from .graders import GRADERS

__all__ = ["CANONICAL_FIELDS", "TaskFormatError", "load_tasks",
           "render_canonical", "save_tasks"]

CANONICAL_FIELDS = ("id", "prompt", "messages", "gold", "grader", "target",
                    "gold_docs", "cluster", "group", "meta")
_PAYLOAD = ("gold", "grader", "target", "gold_docs", "meta")
_MISSING = object()


class TaskFormatError(ValueError):
    """A task record that cannot be read as specified; names the record."""


def _get(rec: Any, path: str) -> Any:
    cur = rec
    for part in path.split("."):
        if isinstance(cur, Mapping) and part in cur:
            cur = cur[part]
        elif isinstance(cur, list) and part.lstrip("-").isdigit() \
                and -len(cur) <= int(part) < len(cur):
            cur = cur[int(part)]
        else:
            return _MISSING
    return cur


def _records(source: str | Path | Iterable[Mapping[str, Any]]) -> list[Any]:
    if not isinstance(source, (str, Path)):
        return list(source)
    p = Path(source)
    text = p.read_text()
    if p.suffix == ".jsonl":
        out = []
        for n, line in enumerate(text.splitlines(), 1):
            if line.strip():
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError as exc:
                    raise TaskFormatError(f"{p}:{n}: not JSON: {exc}") from exc
        return out
    data = json.loads(text)
    if isinstance(data, Mapping):
        for key in ("items", "data", "examples", "tasks"):
            if isinstance(data.get(key), list):
                return data[key]
        raise TaskFormatError(f"{p}: a JSON object, but no list under "
                              f"items/data/examples/tasks")
    return data


def _translate(rec: Any, mapping: Mapping[str, Any] | Callable | None,
               i: int) -> dict[str, Any]:
    if mapping is None:
        if not isinstance(rec, Mapping):
            raise TaskFormatError(f"record {i}: not an object")
        return dict(rec)
    if callable(mapping):
        out = mapping(rec)
        if not isinstance(out, Mapping):
            raise TaskFormatError(f"record {i}: mapping function returned "
                                  f"{type(out).__name__}, not a dict")
        return dict(out)
    out = {}
    for field, how in mapping.items():
        if field not in CANONICAL_FIELDS:
            raise TaskFormatError(f"mapping names unknown field {field!r}; "
                                  f"canonical fields: {CANONICAL_FIELDS}")
        val = how(rec) if callable(how) else _get(rec, how)
        if val is _MISSING:
            raise TaskFormatError(f"record {i}: no value at path {how!r} "
                                  f"(for {field!r})")
        out[field] = val
    return out


def _validate(c: dict[str, Any], i: int) -> None:
    unknown = set(c) - set(CANONICAL_FIELDS)
    if unknown:
        raise TaskFormatError(f"record {i}: unknown field(s) {sorted(unknown)}; "
                              f"put extra data under 'meta'")
    if c.get("id") in (None, ""):
        raise TaskFormatError(f"record {i}: no id")
    if ("prompt" in c) == ("messages" in c):
        raise TaskFormatError(f"record {i} ({c['id']}): give exactly one of "
                              f"'prompt' and 'messages'")
    if "messages" in c:
        msgs = c["messages"]
        if not (isinstance(msgs, list) and msgs and all(
                isinstance(m, Mapping) and {"role", "content"} <= set(m)
                for m in msgs)):
            raise TaskFormatError(f"record {i} ({c['id']}): messages must be "
                                  f"a non-empty list of {{role, content}}")
    if "grader" in c:
        spec = c["grader"]
        name = spec if isinstance(spec, str) else (
            spec.get("name") if isinstance(spec, Mapping) else None)
        if name not in GRADERS:
            raise TaskFormatError(f"record {i} ({c['id']}): unknown grader "
                                  f"{name!r}; known: {sorted(GRADERS)}")
        if "gold" not in c and name != "abstain":
            raise TaskFormatError(f"record {i} ({c['id']}): grader {name!r} "
                                  f"needs a 'gold'")
    if "gold_docs" in c:
        g = c["gold_docs"]
        ok = (isinstance(g, list) and all(isinstance(d, str) for d in g)) or (
            isinstance(g, Mapping) and all(isinstance(d, str) and
                                           isinstance(v, (int, float))
                                           for d, v in g.items()))
        if not ok or not g:
            raise TaskFormatError(f"record {i} ({c['id']}): gold_docs must be "
                                  f"a non-empty list of ids or {{id: gain}}")


def load_tasks(source: str | Path | Iterable[Mapping[str, Any]],
               mapping: Mapping[str, Any] | Callable | None = None
               ) -> list[Item]:
    """Read canonical (or, with `mapping`, foreign) task records as Items.
    The payload holds `messages`, and whichever of gold, grader, target,
    gold_docs and meta the record has."""
    items, seen = [], set()
    for i, rec in enumerate(_records(source)):
        c = _translate(rec, mapping, i)
        _validate(c, i)
        iid = str(c["id"])
        if iid in seen:
            raise TaskFormatError(f"record {i}: duplicate id {iid!r}")
        seen.add(iid)
        msgs = (c["messages"] if "messages" in c
                else [{"role": "user", "content": str(c["prompt"])}])
        payload = {"messages": [dict(m) for m in msgs]}
        payload.update({k: c[k] for k in _PAYLOAD if k in c})
        cl = c.get("cluster")
        items.append(Item(iid, payload, group=c.get("group"),
                          parent_id=None if cl in (None, iid) else str(cl)))
    return items


def save_tasks(items: Iterable[Item], path: str | Path) -> Path:
    """Write Items as canonical JSONL (the inverse of `load_tasks`)."""
    path = Path(path)
    with path.open("w") as fh:
        for it in items:
            rec = {"id": it.item_id, "messages": it.payload["messages"]}
            rec.update({k: it.payload[k] for k in _PAYLOAD if k in it.payload})
            if it.parent_id is not None:
                rec["cluster"] = it.parent_id
            if it.group is not None:
                rec["group"] = it.group
            fh.write(json.dumps(rec) + "\n")
    return path


def render_canonical(u: Unit) -> Request:
    """Request for a canonical item: its messages, the cell's params as the
    request params (model, temperature, max_tokens...), and its target if
    it has one.  Labels that must not reach the backend belong in the
    cell's tags, not its params."""
    p = u.item.payload
    return Request(u.id, [dict(m) for m in p["messages"]],
                   dict(u.cell.params), target=p.get("target"))
