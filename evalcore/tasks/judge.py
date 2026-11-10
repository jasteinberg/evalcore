"""Judging as a task: the answers of a finished run become the items of a
second run, in which each judge returns a structured verdict.

    items  = judge_items(frame, source_task)        one item per answer
    task   = judge_task(items, criteria)
    frame  = run_panel(task, judges, out_dir)       one run per judge
    long   = verdicts_long(frame, criteria, source=frame0,
                           reference={"correct": "correct"})

and evaluation/judges.py takes the long frame from there (calibrate,
debias, agreement, error_correlation, majority).

Because judging is an ordinary run, it inherits the cache, resume and
attrition: a judge's failed call is an error row, a verdict that cannot be
read is an `unparsed` row (never a fail), and re-reading verdicts costs a
re-score, not a re-run.  A judged item's id is the source unit's id, and
its cluster is the source item's cluster, so every bootstrap over verdicts
resamples source items, as over the answers themselves.

The verdict is a call to the `verdict` tool: one boolean per criterion and
a one-sentence reason.  A judge whose provider allows it should force the
call, with params {"tool_choice": TOOL}.  Some models refuse forced tool
use (Claude Sonnet 5.5 returns a 400), so forcing is a per-judge param,
not built in: the prompt asks for the call either way, and the judge's
params, recorded in its rows, say which it was.  The reply is read
strictly: exactly one call to `verdict`, exactly the expected keys,
booleans where booleans belong.  Anything else is a ParseFailure, so an
unforced judge that answers in prose is counted, not guessed at.
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from ..core.backends import Backend, Request, Response
from ..core.runner import ParseFailure
from ..core.spec import Cell, Item, Unit
from .base import Task

__all__ = ["INSTRUCTIONS", "TOOL", "Criterion", "judge_items", "judge_task",
           "run_panel", "verdict_tool", "verdicts_long"]

TOOL = "verdict"
REASON = "reason"
_NAME = re.compile(r"[a-z][a-z0-9_]*")

INSTRUCTIONS = (
    "You grade a response to a question. Decide each criterion on its own, "
    "then call the verdict tool once. A criterion passes only if the "
    "response clearly meets it.")
WITH_REFERENCE = " The reference answer is correct."


@dataclass(frozen=True)
class Criterion:
    """One rubric line.  `question` is put to the judge so that yes means
    pass; `name` is its key in the verdict and the column `v_<name>`."""

    name: str
    question: str

    def __post_init__(self) -> None:
        if not _NAME.fullmatch(self.name) or self.name == REASON:
            raise ValueError(f"criterion name {self.name!r}: use lower-case "
                             f"letters, digits and _, not {REASON!r}")


def verdict_tool(criteria: Sequence[Criterion]) -> dict[str, Any]:
    names = [c.name for c in criteria]
    if not names or len(set(names)) != len(names):
        raise ValueError(f"criteria must be non-empty and distinct: {names}")
    props: dict[str, Any] = {c.name: {"type": "boolean",
                                      "description": c.question}
                             for c in criteria}
    props[REASON] = {"type": "string",
                     "description": "One short sentence."}
    return {"name": TOOL, "description": "Record the verdict.",
            "parameters": {"type": "object", "properties": props,
                           "required": [*names, REASON]}}


def _question(item: Item) -> str:
    """The last user message of a canonical item."""
    users = [m["content"] for m in item.payload["messages"]
             if m["role"] == "user"]
    if not users:
        raise ValueError(f"item {item.item_id} has no user message")
    return str(users[-1])


def judge_items(frame: pd.DataFrame, source: Task,
                statuses: Sequence[str] = ("ok", "unparsed"),
                reference: bool = True, id_col: str = "unit_id"
                ) -> list[Item]:
    """One item per answer in `frame` whose status is in `statuses` (the
    rows that have a text).  Truncated and failed rows have no answer to
    judge and stay in the source run's attrition.  `reference` shows the
    judge the item's gold answer.  `id_col` names the column that
    identifies an answer, and becomes the judge item's id: unit_id within
    one run, or a column you build when stacking runs whose unit ids
    coincide (conditions that differ only in tags)."""
    by_id = {it.item_id: it for it in source.items}
    if len(by_id) != len(source.items):
        raise ValueError("source task has duplicate item ids")
    if frame[id_col].duplicated().any():
        raise ValueError(f"frame has duplicate {id_col} values: pass an "
                         f"id_col that identifies each answer")
    unknown = sorted(set(frame["item_id"]) - set(by_id))
    if unknown:
        raise ValueError(f"frame rows for items not in the source task: "
                         f"{unknown[:3]}")
    out = []
    for r in frame[frame["status"].isin(statuses)].itertuples(index=False):
        if not isinstance(r.text, str) or not r.text.strip():
            raise ValueError(f"answer {getattr(r, id_col)} has status "
                             f"{r.status} and no text to judge")
        it = by_id[r.item_id]
        payload: dict[str, Any] = {"question": _question(it),
                                   "answer": r.text,
                                   "source_status": r.status}
        if reference:
            payload["reference"] = str(it.payload["gold"])
        out.append(Item(str(getattr(r, id_col)), payload, group=r.group,
                        parent_id=r.cluster))
    return out


def _prompt(p: Mapping[str, Any]) -> str:
    parts = [f"Question:\n{p['question']}"]
    if "reference" in p:
        parts.append(f"Reference answer:\n{p['reference']}")
    parts.append(f"Response:\n{p['answer']}")
    return "\n\n".join(parts)


def judge_task(items: Sequence[Item], criteria: Sequence[Criterion],
               name: str = "judge", instructions: str = INSTRUCTIONS,
               version: str = "1") -> Task:
    """A Task whose requests offer the verdict tool.  The cell's params
    (model, max_tokens, and tool_choice=TOOL where the provider allows a
    forced call) pass through as the request params."""
    tool = verdict_tool(criteria)
    names = [c.name for c in criteria]
    refs = {"reference" in it.payload for it in items}
    if len(refs) > 1:
        raise ValueError("some items show a reference answer and some do not")
    system = instructions + (WITH_REFERENCE if refs == {True} else "")

    def render(u: Unit) -> Request:
        return Request(u.id, [{"role": "system", "content": system},
                              {"role": "user", "content": _prompt(u.item.payload)}],
                       dict(u.cell.params), tools=[tool])

    def score(u: Unit, rp: Response) -> dict[str, Any]:
        calls = rp.meta.get("tool_calls") or []
        if len(calls) != 1 or calls[0].get("name") != TOOL:
            raise ParseFailure(f"expected one {TOOL} call, got "
                               f"{[c.get('name') for c in calls]}")
        args = calls[0].get("arguments")
        if not isinstance(args, Mapping):
            raise ParseFailure(f"verdict arguments are not an object: {args!r}")
        if set(args) != {*names, REASON}:
            raise ParseFailure(f"verdict keys {sorted(args)}, expected "
                               f"{sorted([*names, REASON])}")
        bad = [n for n in names if not isinstance(args[n], bool)]
        if bad:
            raise ParseFailure(f"verdict values not boolean: {bad}")
        return {**{f"v_{n}": float(args[n]) for n in names},
                "verdict_reason": str(args[REASON])}

    return Task(name, list(items), render=render, score=score, version=version)


def run_panel(task: Task, judges: Sequence[tuple[str, Backend, Mapping[str, Any]]],
              out_dir: str | Path, **kw: Any) -> pd.DataFrame:
    """Run `task` once per judge (name, backend, params), each with its own
    backend instance, to out_dir/<name>.jsonl, and stack the frames.  The
    judge's name is a cell tag, so it labels rows without entering any
    key.  Keyword arguments pass to Task.run (cache, workers, ...)."""
    names = [j[0] for j in judges]
    if len(set(names)) != len(names):
        raise ValueError(f"judge names must be distinct: {names}")
    if len({id(j[1]) for j in judges}) != len(judges):
        # each run closes its backend when it ends, so a shared instance
        # would be closed under every judge after the first
        raise ValueError("each judge needs its own backend instance (a run "
                         "closes its backend when it ends)")
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    frames = [task.run([Cell(dict(params), {"judge": name})], backend,
                       out_dir / f"{name}.jsonl", **kw)
              for name, backend, params in judges]
    return pd.concat(frames, ignore_index=True)


def verdicts_long(judged: pd.DataFrame, criteria: Sequence[Criterion],
                  source: pd.DataFrame | None = None,
                  reference: Mapping[str, str] | None = None,
                  id_col: str = "unit_id") -> pd.DataFrame:
    """The long frame evaluation/judges.py reads: one row per (answer,
    judge, criterion) from the judge rows with status ok.  `reference`
    maps a criterion to a metric column of the `source` frame (an exact
    grader's `correct`), joined on the source's `id_col` (as passed to
    judge_items); a source row without
    that metric (unparsed, not scored) gives a missing reference."""
    ok = judged[judged["status"] == "ok"]
    rows = []
    for c in criteria:
        col = f"v_{c.name}"
        rows.append(pd.DataFrame({"answer": ok["item_id"].to_numpy(),
                                  "cluster": ok["cluster"].to_numpy(),
                                  "judge": ok["judge"].to_numpy(),
                                  "criterion": c.name,
                                  "verdict": ok[col].to_numpy(dtype=float)}))
    out = pd.concat(rows, ignore_index=True)
    if reference:
        if source is None:
            raise ValueError("reference columns need the source frame")
        unknown = sorted(set(reference) - {c.name for c in criteria})
        if unknown:
            raise ValueError(f"reference for unknown criteria {unknown}")
        src = source.set_index(id_col)
        out["reference"] = [
            float(src.at[a, reference[c]]) if c in reference else float("nan")
            for a, c in zip(out["answer"], out["criterion"], strict=True)]
    return out
