"""Conformance check for a backend, to run before a sweep.

The runner relies on a few properties no type signature expresses: one
response per request, in order, with the request's unit_id; an identity
that is stable and serialisable; failures raised as Transient or Fatal.
A backend that breaks one of them does not crash the harness -- it
produces rows that look like results.  This check makes the calls and
says which property failed.  It calls the backend for real: pass cheap
requests (the default sends four short prompts)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field

from ..spec import canonical
from .base import Backend, Request, Response

_DEFAULT = ["Reply with the single word: ok", "What is 2 + 2?",
            "Name a colour.", "Say yes or no."]


@dataclass
class CheckReport:
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    passed: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems

    def __str__(self) -> str:
        lines = [f"{'PASS' if self.ok else 'FAIL'}: {len(self.passed)} checks "
                 f"passed, {len(self.problems)} problems, "
                 f"{len(self.warnings)} warnings"]
        lines += [f"  problem: {p}" for p in self.problems]
        lines += [f"  warning: {w}" for w in self.warnings]
        return "\n".join(lines)


def check_backend(backend: Backend, requests: Sequence[Request] | None = None,
                  params: dict | None = None) -> CheckReport:
    """Run the contract checks; see the module docstring.  `params` are
    added to the default requests (e.g. the model and max_tokens an HTTP
    backend requires)."""
    rep = CheckReport()
    reqs = list(requests) if requests is not None else [
        Request(f"check-{i}", [{"role": "user", "content": c}],
                dict(params or {})) for i, c in enumerate(_DEFAULT)]

    try:
        ident = backend.identity()
        if canonical(ident) != canonical(backend.identity()):
            rep.problems.append("identity() is not stable between calls")
        else:
            rep.passed.append("identity is defined, serialisable and stable")
    except TypeError as exc:
        rep.problems.append(str(exc))
    except ValueError as exc:
        rep.problems.append(f"identity() is not JSON-serialisable: {exc}")

    try:
        out = backend.complete_batch(reqs)
    except Exception as exc:  # noqa: BLE001 - reported, not raised
        rep.problems.append(f"complete_batch raised {type(exc).__name__}: "
                            f"{exc}")
        return rep
    if len(out) != len(reqs):
        rep.problems.append(f"complete_batch returned {len(out)} responses "
                            f"for {len(reqs)} requests")
    elif not all(isinstance(r, Response) for r in out):
        rep.problems.append("complete_batch returned something other than "
                            "Response objects")
    elif [r.unit_id for r in out] != [q.unit_id for q in reqs]:
        rep.problems.append("responses do not carry their requests' unit_ids "
                            "in request order")
    else:
        rep.passed.append("one response per request, in order, ids preserved")
        bad = [r.error for r in out if not r.ok]
        if bad:
            rep.problems.append(f"{len(bad)} of {len(out)} check requests "
                                f"failed, e.g. {bad[0]}")
        good = [r for r in out if r.ok]
        if good and not all("truncated" in (r.meta or {}) for r in good):
            rep.warnings.append(
                "responses carry no meta['truncated']: a generation cut off "
                "at its token budget cannot be told from a finished one, and "
                "will be scored")
        elif good:
            rep.passed.append("truncation is reported")

    single = backend.complete(reqs[0])
    if not isinstance(single, Response) or single.unit_id != reqs[0].unit_id:
        rep.problems.append("complete() does not return a Response with the "
                            "request's unit_id")
    else:
        rep.passed.append("complete() matches the contract")

    if backend.supports_target:
        t = Request("check-target", [{"role": "user", "content": "1 + 1 ="}],
                    {}, target=" 2")
        r = backend.complete_batch([t])[0]
        if not r.ok or not isinstance(r.meta.get("logp"), list):
            rep.problems.append("supports_target is True but a target request "
                                "returned no per-token logp")
        else:
            rep.passed.append("teacher-forced scoring returns per-token logp")
    return rep
