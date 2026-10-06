"""Backends: the only part of the harness that knows how a model is called.

Contract.  A backend maps a list of `Request` to a list of `Response` of the
same length and order.  It may raise `Transient` (retry) or `Fatal` (do not).
It must NOT know about cells, items, clusters, or metrics -- everything above
this line is backend-agnostic, which is what lets one core serve both a
rate-limited HTTP API and a local batched forward pass.

Batching is expressed by `complete_batch`; the default maps `complete` over
the list.  An HTTP backend leaves that alone and gets its throughput from the
runner's thread pool.  A local HF backend overrides it and gets its
throughput from a real batched forward, with the runner at one worker.
"""

from __future__ import annotations

import random
import time
from collections.abc import Callable, Collection, Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

from ..spec import digest


class Transient(RuntimeError):
    """Rate limit, timeout, 5xx -- worth retrying.  `retry_after` is the
    delay the server asked for, in seconds, when it said; the retry loop
    waits at least that long instead of its own shorter guess."""

    def __init__(self, msg: str = "", retry_after: float | None = None) -> None:
        super().__init__(msg)
        self.retry_after = retry_after



class Fatal(RuntimeError):
    """Bad request, auth failure, content filter -- retrying will not help."""


# Shapes that cross the backend boundary, named so signatures say what they
# carry.  A message is {"role", "content"} plus, for tool use, "tool_calls"
# or "tool_call_id"/"name"/"is_error"; a tool spec is {"name",
# "description", "parameters"}; provider JSON is whatever the API returns.
Message = Mapping[str, Any]
ToolSpec = Mapping[str, Any]
JSONDict = dict[str, Any]


@dataclass(frozen=True)
class Request:
    """One model call.  `target`, when set, asks for teacher-forced scoring
    of that continuation instead of a generation: the per-token log
    probability the model assigns to each target token given the prompt and
    the true prefix before it.  `input_ids` gives the prompt as token ids
    (for prompts built in token space, where text would not round-trip),
    and `candidates` asks for the next-token distribution restricted to a
    set of token ids.  These need a backend that reads logits
    (`supports_target`, i.e. `HFBackend`)."""

    unit_id: str
    messages: Sequence[Mapping[str, Any]]
    params: Mapping[str, Any] = field(default_factory=dict)
    target: str | None = None
    tools: Sequence[Mapping[str, Any]] | None = None
    input_ids: Sequence[int] | None = None
    candidates: Sequence[int] | None = None

    @property
    def cache_key(self) -> str:
        key: dict[str, Any] = {"m": [dict(x) for x in self.messages],
                               "p": dict(self.params)}
        if self.target is not None:      # absent, so existing keys still hold
            key["t"] = self.target
        if self.tools:
            key["tools"] = [dict(t) for t in self.tools]
        if self.input_ids is not None:
            key["ids"] = [int(i) for i in self.input_ids]
        if self.candidates is not None:
            key["cands"] = [int(i) for i in self.candidates]
        return digest(key)


@dataclass
class Response:
    unit_id: str
    text: str | None = None
    meta: dict[str, Any] = field(default_factory=dict)
    error: str | None = None
    cached: bool = False

    @property
    def ok(self) -> bool:
        return self.error is None


# --- decode parameters ------------------------------------------------------
# Named explicitly because they change what the model emits, and therefore must
# be identical across any cells that will later be differenced.  A parameter a
# backend silently defaults is invisible in the run record and produces cells
# that look comparable and are not.
DECODE_KEYS = ("temperature", "top_p", "top_k", "max_tokens", "seed", "stop",
               "stop_sequences", "repetition_penalty")


def merge_params(default: Mapping[str, Any] | None,
                 params: Mapping[str, Any]) -> dict[str, Any]:
    """Request params win, backend defaults fill gaps, nothing is invented."""
    return {**dict(default or {}), **dict(params)}


def decode_signature(params: Mapping[str, Any]) -> str:
    """Digest of the decode-relevant params only -- used to prove a batch is
    homogeneous before it is run as one."""
    return digest({k: params[k] for k in DECODE_KEYS if k in params})


def require(params: Mapping[str, Any], key: str, who: str) -> Any:
    """Fail loudly rather than substitute a provider-specific default.

    Anthropic requires max_tokens, OpenAI does not, and a local generate call
    has its own idea; left implicit, one spec run against three backends gets
    three different generation budgets and the comparison is meaningless."""
    if params.get(key) is None:
        raise Fatal(
            f"{who} requires '{key}': set it in the cell params or in the "
            f"backend's default_params. Refusing to substitute a default, "
            f"because a silent per-provider value makes cells incomparable.")
    return params[key]


def refuse_dropped(params: Mapping[str, Any], forwarded: Collection[str],
                   who: str) -> None:
    """Refuse a decode key that is set but would never reach the model.

    Such a key is in the cell params, so it moves the cell id and the cache
    key, while the request actually sent is identical: two cells that look
    different are the same cell, and a sweep over the key reads as a clean
    null result.  Refusing is deliberate where mapping might seem kinder --
    which fields a provider accepts is a fact to check against its current
    API reference, not to assume, and a refusal names the key so it can be
    added to the adapter once checked."""
    dropped = sorted(k for k in DECODE_KEYS
                     if params.get(k) is not None and k not in forwarded)
    if dropped:
        raise Fatal(
            f"{who} does not forward decode param(s) {dropped}: remove them "
            f"from the params sent to this backend, or teach the adapter to "
            f"send them.  Dropping them silently would make cells that "
            f"differ only in these keys identical requests.")


class Backend:
    name = "base"
    # True only for backends that read logits and so can score a
    # Request.target by teacher forcing.  Anything else would ignore the
    # target and generate, and the row would look like a result.
    supports_target = False
    # True only for backends whose adapter can send Request.tools and
    # return the model's calls in meta["tool_calls"].
    supports_tools = False

    def identity(self) -> dict[str, Any]:
        """Everything about this backend that can change a response without
        appearing in the request.  Part of the request key, so a sweep
        against another model, checkpoint or default budget can never be
        served this one's cache or resume onto its rows.

        Every backend must define it; there is no default.  A default can
        only see what the base class knows (a name, `default_params`), and
        a backend configured in `__init__` -- a model path, a revision, a
        temperature held as an attribute -- would then share keys across
        configurations.  Return a JSON-serialisable dict of what determines
        the outputs; leave out what does not (a timeout, an API key), or
        every change to it costs a full re-run.  Built-in backends start
        from `_base_identity()`."""
        raise TypeError(
            f"{type(self).__name__} must define identity(): return a "
            f"JSON-serialisable dict of what determines its outputs (model "
            f"path, revision, settings). It keys the cache and resume, so "
            f"without it two configurations would share results.")

    def _base_identity(self) -> dict[str, Any]:
        """Name and merged-in default params: the part every backend has."""
        return {"backend": self.name,
                "default_params": dict(getattr(self, "default_params", {})
                                       or {})}

    def refusal(self, req: Request) -> str | None:
        """Why this backend cannot serve `req` at all, or None.

        Checked before any call: by the runner for every request (a refusal
        becomes an error row, never a call), and by Experiment on the first
        request at construction, so a task the backend cannot serve fails
        before any spend.  A backend with requirements of its own extends
        this and calls super()."""
        name = type(self).__name__
        if req.tools and not self.supports_tools:
            return (f"{name} cannot send tools (supports_tools is False); "
                    f"wrap a tool-capable backend in AgentBackend")
        logit_only = [f for f, v in (("target", req.target),
                                     ("input_ids", req.input_ids),
                                     ("candidates", req.candidates))
                      if v is not None]
        if logit_only and not self.supports_target:
            return (f"{name} cannot teacher-force or score "
                    f"{', '.join(logit_only)} (supports_target is False); use "
                    f"a backend that reads logits, such as HFBackend")
        return None

    def complete(self, req: Request) -> Response:
        raise NotImplementedError

    def complete_batch(self, reqs: Sequence[Request]) -> list[Response]:
        return [self.complete(r) for r in reqs]

    def close(self) -> None:
        pass


def filled(out: Sequence[Response | None]) -> list[Response]:
    """A response list built slot by slot, checked complete: every request
    has its response.  A missing one is a harness bug, reported by position
    rather than surfacing later as a mis-joined row."""
    missing = [i for i, r in enumerate(out) if r is None]
    if missing:
        raise AssertionError(f"internal: no response for request(s) {missing}")
    return [r for r in out if r is not None]


def request_key(req: Request, repeat: int, backend: Backend) -> str:
    r"""Content address of one model call as actually made.

        request_key = digest(request.cache_key, backend.identity(), repeat)

    `request.cache_key` covers the messages and params; `identity()` covers
    what the backend merges in (a `model` or `max_tokens` set as a default
    is otherwise invisible); the repeat index makes repeats distinct samples
    rather than one sample read back k times from the cache, which at
    temperature > 0 would set the within-cluster variance of repeats to 0
    and with it every rho, DEFF and n_eff computed from them.

    The cache is keyed on it, and every row records it, so resume can tell
    a finished unit from one whose prompt, item, backend defaults or repeat
    changed since its row was written."""
    return digest({"request": req.cache_key, "backend": backend.identity(),
                   "repeat": repeat})


def with_retries(fn: Callable[[Request], Response], req: Request,
                 attempts: int = 5, base: float = 1.0, cap: float = 30.0,
                 seed: int | None = None,
                 on_retry: Callable[..., None] | None = None) -> Response:
    """Exponential backoff with full jitter: sleep ~ U(0, min(cap, base 2^k)).

    Full jitter rather than fixed backoff because every worker hits the same
    429 at the same instant; deterministic sleeps re-synchronise the thundering
    herd on every retry, while U(0, .) spreads the retries uniformly over the
    window and is what actually drains a rate-limit queue."""
    return _retrying(lambda: fn(req),
                     lambda e: Response(req.unit_id, error=e),
                     seed if seed is not None else req.unit_id,
                     attempts, base, cap, on_retry)


def batch_with_retries(fn: Callable[[Sequence[Request]], list[Response]],
                       reqs: Sequence[Request], attempts: int = 5,
                       base: float = 1.0, cap: float = 30.0,
                       seed: int | None = None,
                       on_retry: Callable[..., None] | None = None
                       ) -> list[Response]:
    """`with_retries` for a whole batch: one policy for both paths.

    A batch is retried as a unit, since one generate() call or one batched
    request is what failed.  A terminal failure becomes one error row per
    request with the same prefixes as the single-request path -- never an
    exception out of the runner, which would turn a single 429 into a dead
    sweep."""
    return _retrying(lambda: fn(reqs),
                     lambda e: [Response(r.unit_id, error=e) for r in reqs],
                     seed if seed is not None else
                     (reqs[0].unit_id if reqs else 0),
                     attempts, base, cap, on_retry)


# Longest server-requested wait honoured between attempts.  A server asking
# for longer (a daily quota) is not going to clear within a run; waiting
# would only hang the sweep, so the unit fails now, saying so.
MAX_SERVER_WAIT = 300.0


def _retrying(call: Callable[[], Any], fail: Callable[[str], Any],
              seed: Any, attempts: int, base: float, cap: float,
              on_retry: Callable[..., None] | None = None) -> Any:
    """`on_retry(attempt, wait_s, reason, server_wait_s)` is called before
    each sleep, so a retry leaves a record (the run's event log)."""
    rng = random.Random(seed)
    last = ""
    for k in range(attempts):
        try:
            return call()
        except Fatal as exc:
            return fail(f"fatal: {exc}")
        except Transient as exc:
            last = str(exc)
            if k == attempts - 1:
                break
            wait = rng.uniform(0, min(cap, base * 2 ** k))
            if exc.retry_after is not None:
                if exc.retry_after > MAX_SERVER_WAIT:
                    return fail(f"transient: server asks to wait "
                                f"{exc.retry_after:.0f}s (> {MAX_SERVER_WAIT:.0f}s"
                                f" cap): {last}")
                # at least what was asked, plus jitter so workers that hit
                # the same limit do not all return in the same instant
                wait = exc.retry_after + wait
            if on_retry is not None:
                on_retry(k + 1, wait, last, exc.retry_after)
            time.sleep(wait)
        except Exception as exc:  # noqa: BLE001 - an unknown failure is a row, not a crash
            return fail(f"{type(exc).__name__}: {exc}")
    return fail(f"transient x{attempts}: {last}")
