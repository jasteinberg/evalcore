r"""Question answering over structured execution logs, generated.

Every trace is generated from a seed: a tree of operations ("spans"), each
with an id, a parent, a start and end time, a status and a few attributes.
Children run one after another inside their parent's interval; some
operations fail, and a failed operation is sometimes retried.  Nothing is
hand-written, so there are as many traces as wanted, of any size and depth,
and every question's gold answer is computed from the trace -- exact by
construction.

The same trace can be serialised three ways, which is an experimental axis
in itself: `json` (a list of span records), `tree` (indented by depth), and
`log` (a chronological START/END event log, the shape real logs have).

Question types, each with a deterministic grader:

    count      how many times did operation X run?          numeric
    slowest    which operation took longest (not the root)?  label
    first_fail which operation failed first?                 label
    any_fail   did any operation fail?                       yes_no
    parent     which span directly started span S?           exact (span id)
    retries    how many retries did X need?                  numeric
    total_ms   total time spent in X, over all its runs?     numeric
    failed_ids which spans failed? cite their ids            set_f1
    attr       what was attribute A of span S?               numeric
    absent     an attribute the span does not have            abstain

`absent` has no answer in the trace: a correct response declines, and an
answer is counted as a hallucination, separately from "wrong".

Questions about one span record where that span sits in the serialised
context (`meta["position"]`, 0 = start, 1 = end), and `position=` asks
the generator to pick target spans near a given relative position -- the
axis of a lost-in-the-middle study.  All questions about one trace share
it as their cluster, so the bootstrap resamples traces, not questions.
"""

from __future__ import annotations

import json
import random
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..core.spec import Item
from .base import Task
from .formats import render_canonical
from .graders import make_score

__all__ = ["FORMATS", "QUESTION_TYPES", "Span", "generate_trace",
           "log_qa_items", "log_qa_task", "serialise"]

OPERATIONS = ("load_config", "open_session", "fetch_page", "parse_rows",
              "query_index", "rank_results", "write_cache", "read_cache",
              "resize_image", "compress_blob", "validate_input", "sync_clock",
              "render_report", "send_batch", "merge_shards", "check_quota")
ATTRS = ("rows", "bytes", "items", "shards")
FORMATS = ("json", "tree", "log")
QUESTION_TYPES = ("count", "slowest", "first_fail", "any_fail", "parent",
                  "retries", "total_ms", "failed_ids", "attr", "absent")


@dataclass
class Span:
    id: str
    name: str
    parent: str | None
    start_ms: int
    end_ms: int
    status: str = "ok"
    attempt: int = 1
    attrs: dict[str, int] = field(default_factory=dict)

    @property
    def ms(self) -> int:
        return self.end_ms - self.start_ms


def generate_trace(rng: random.Random, n_spans: int = 20, max_depth: int = 3,
                   p_error: float = 0.15, p_retry: float = 0.6) -> list[Span]:
    """A trace of about `n_spans` spans, in start-time order (the root
    first).  Durations are log-normal; a parent's interval contains its
    children run in sequence with small gaps."""
    spans: list[Span] = []

    def new_id() -> str:
        return f"s{len(spans):03d}"

    def build(name: str, parent: str | None, start: int, depth: int,
              attempt: int = 1) -> Span:
        sp = Span(new_id(), name, parent, start, start)
        spans.append(sp)
        t = start + rng.randint(1, 5)
        n_kids = 0 if depth >= max_depth or len(spans) >= n_spans else \
            rng.randint(1, 4) if depth == 0 else rng.choice((0, 0, 1, 2, 3))
        for _ in range(n_kids):
            if len(spans) >= n_spans:
                break
            child = build(rng.choice(OPERATIONS), sp.id, t, depth + 1)
            t = child.end_ms + rng.randint(1, 5)
            if child.status == "error" and rng.random() < p_retry \
                    and len(spans) < n_spans:
                retry = build(child.name, sp.id, t, depth + 1,
                              attempt=child.attempt + 1)
                t = retry.end_ms + rng.randint(1, 5)
        if n_kids == 0:
            t += max(1, int(rng.lognormvariate(3.0, 1.0)))
            if parent is not None and rng.random() < p_error:
                sp.status = "error"
        sp.end_ms = t
        sp.attempt = attempt
        for a in rng.sample(ATTRS, rng.randint(0, 2)):
            sp.attrs[a] = rng.randint(1, 5000)
        return sp

    build("job", None, 0, 0)
    while len(spans) < n_spans:        # top up under the root if short
        root = spans[0]
        extra = build(rng.choice(OPERATIONS), root.id, root.end_ms + 1, 1)
        root.end_ms = extra.end_ms + rng.randint(1, 5)
    spans.sort(key=lambda s: (s.start_ms, s.id))
    return spans


def serialise(spans: Sequence[Span], fmt: str) -> str:
    """The trace as text in one of FORMATS."""
    if fmt == "json":
        return json.dumps([{**asdict(s), "duration_ms": s.ms} for s in spans],
                          indent=1)
    if fmt == "tree":
        kids: dict[str | None, list[Span]] = {}
        for s in spans:
            kids.setdefault(s.parent, []).append(s)
        out: list[str] = []

        def walk(pid: str | None, depth: int) -> None:
            for s in kids.get(pid, []):
                att = "".join(f" {k}={v}" for k, v in sorted(s.attrs.items()))
                retry = f" attempt={s.attempt}" if s.attempt > 1 else ""
                out.append(f"{'  ' * depth}{s.name} [{s.id}] {s.ms} ms "
                           f"{s.status}{retry}{att}")
                walk(s.id, depth + 1)

        walk(None, 0)
        return "\n".join(out)
    if fmt == "log":
        events = []
        for s in spans:
            events.append((s.start_ms, 0, s.id,
                           f"t={s.start_ms:07d}ms START {s.name} id={s.id} "
                           f"parent={s.parent or '-'} attempt={s.attempt}"))
            att = "".join(f" {k}={v}" for k, v in sorted(s.attrs.items()))
            events.append((s.end_ms, 1, s.id,
                           f"t={s.end_ms:07d}ms END {s.name} id={s.id} "
                           f"status={s.status} duration={s.ms}ms{att}"))
        events.sort()
        return "\n".join(e[3] for e in events)
    raise ValueError(f"fmt must be one of {FORMATS}")


def _position(text: str, span_id: str) -> float:
    i = text.find(span_id)
    return i / max(1, len(text) - 1)


def _questions(spans: list[Span], text: str, rng: random.Random,
               types: Sequence[str], position: float | None
               ) -> list[tuple[str, str, Any, Any, dict]]:
    """(type, question, gold, grader, extra meta) for each type that the
    trace supports; a type with no well-defined answer here is skipped
    (e.g. `slowest` with a tie, `retries` with no retried operation)."""
    root, body = spans[0], spans[1:]
    names = sorted({s.name for s in spans})
    by_name: dict[str, list[Span]] = {}
    for s in body:
        by_name.setdefault(s.name, []).append(s)
    failed = [s for s in body if s.status == "error"]

    def pick(cands: list[Span]) -> Span:
        if position is None:
            return rng.choice(cands)
        return min(cands, key=lambda s: abs(_position(text, s.id) - position))

    out: list[tuple[str, str, Any, Any, dict[str, Any]]] = []
    for t in types:
        if t == "count":
            n = rng.choice(sorted(by_name))
            out.append((t, f"How many times did the operation {n} run, "
                        f"counting retries? Answer with a number.",
                        len(by_name[n]), "numeric", {}))
        elif t == "slowest":
            ms = sorted((s.ms for s in body), reverse=True)
            if len(ms) > 1 and ms[0] != ms[1]:
                top = max(body, key=lambda s: s.ms)
                out.append((t, "Which operation took the longest, not "
                            "counting the root job? Answer with its name.",
                            top.name, {"name": "label", "options": names},
                            {"span": top.id}))
        elif t == "first_fail" and failed:
            first = min(failed, key=lambda s: (s.end_ms, s.id))
            out.append((t, "Which operation was the first to fail? Answer "
                        "with its name.", first.name,
                        {"name": "label", "options": names},
                        {"span": first.id}))
        elif t == "any_fail":
            out.append((t, "Did any operation fail? Answer yes or no.",
                        "yes" if failed else "no", "yes_no", {}))
        elif t == "parent":
            s = pick([s for s in body if s.parent != root.id] or body)
            out.append((t, f"Which span directly started span {s.id}? "
                        f"Answer with its span id.", s.parent,
                        {"name": "exact", "extract": r"\b(s\d{3})\b"},
                        {"span": s.id}))
        elif t == "retries":
            retried = sorted({s.name for s in body if s.attempt > 1})
            if retried:
                n = rng.choice(retried)
                k = max(s.attempt for s in by_name[n]) - 1
                out.append((t, f"How many retries did {n} need? Answer with "
                            f"a number.", k, "numeric", {}))
        elif t == "total_ms":
            n = rng.choice(sorted(by_name))
            out.append((t, f"What was the total time spent in {n}, summed "
                        f"over all its runs, in milliseconds? Answer with a "
                        f"number.", sum(s.ms for s in by_name[n]), "numeric",
                        {}))
        elif t == "failed_ids" and failed:
            out.append((t, "Which spans failed? List their span ids.",
                        sorted(s.id for s in failed),
                        {"name": "set_f1", "pattern": r"\bs\d{3}\b"}, {}))
        elif t == "attr":
            have = [s for s in body if s.attrs]
            if have:
                s = pick(have)
                a = rng.choice(sorted(s.attrs))
                out.append((t, f"What was the value of {a} for span {s.id}? "
                            f"Answer with a number.", s.attrs[a], "numeric",
                            {"span": s.id}))
        elif t == "absent":
            s = pick(body)
            missing = [a for a in ATTRS if a not in s.attrs]
            if missing:
                a = rng.choice(missing)
                out.append((t, f"What was the value of {a} for span {s.id}? "
                            f"If the trace does not say, say so.", None,
                            "abstain", {"span": s.id}))
    return out


SYSTEM = ("You are given an execution trace of a program. Answer the question "
          "using only the trace.")


def log_qa_items(n_traces: int = 50, fmt: str = "log", n_spans: int = 20,
                 max_depth: int = 3, types: Sequence[str] = QUESTION_TYPES,
                 position: float | None = None, seed: int = 0,
                 **trace_kw: Any) -> list[Item]:
    """Items for `n_traces` generated traces, one per supported question
    type per trace.  The cluster of every item is its trace."""
    unknown = set(types) - set(QUESTION_TYPES)
    if unknown:
        raise ValueError(f"unknown question types {sorted(unknown)}")
    rng = random.Random(seed)
    items = []
    for k in range(n_traces):
        spans = generate_trace(rng, n_spans, max_depth, **trace_kw)
        text = serialise(spans, fmt)
        trace_id = f"trace{seed}-{k:04d}"
        for t, q, gold, spec, extra in _questions(spans, text, rng, types,
                                                  position):
            meta = {"type": t, "fmt": fmt, "n_spans": len(spans),
                    "context_chars": len(text), "trace": trace_id, **extra}
            if "span" in extra:
                meta["position"] = _position(text, extra["span"])
            items.append(Item(
                f"{trace_id}-{t}",
                {"messages": [{"role": "system", "content": SYSTEM},
                              {"role": "user", "content":
                               f"Trace:\n{text}\n\nQuestion: {q}"}],
                 "gold": gold, "grader": spec, "meta": meta},
                group=t, parent_id=trace_id))
    return items


def log_qa_task(n_traces: int = 50, fmt: str = "log", n_spans: int = 20,
                max_depth: int = 3, types: Sequence[str] = QUESTION_TYPES,
                position: float | None = None, seed: int = 0,
                p_error: float = 0.15, p_retry: float = 0.6) -> Task:
    """The items as a Task with the canonical render and grader-driven
    score; works with any generating backend."""
    items = log_qa_items(n_traces, fmt, n_spans, max_depth, types, position,
                         seed, p_error=p_error, p_retry=p_retry)
    return Task(f"log_qa_{fmt}", items, render=render_canonical,
                score=make_score())
