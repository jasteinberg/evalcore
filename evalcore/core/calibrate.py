"""Measure a backend on the experiment's own requests, then size the run.

`calibrate` makes a few real calls and records what a run needs to know:
latency, tokens per call, how often the budget truncates, the provider's
rate-limit headers as returned, and -- for a local model -- throughput per
batch size.  `derive` turns that into run settings (workers, batch size)
and lists every assumption it made, so a sizing decision is never silent.

The requests should be the experiment's own: latency and tokens depend on
prompt and output length, so a toy prompt underestimates both.
"""

from __future__ import annotations

import math
import statistics
import time
import warnings
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any

from .backends import Backend, Request, Response
from .spec import digest

__all__ = ["Calibration", "calibrate", "calibration_caveats", "derive"]

MIN_OK_CALLS = 3


@dataclass
class Calibration:
    identity: str                         # digest of backend.identity()
    measured_at: str
    n_calls: int
    latency_median_s: float | None = None
    latency_p95_s: float | None = None
    in_tokens: float | None = None        # mean per call
    out_tokens: float | None = None
    truncated_rate: float | None = None
    rate_limits: dict[str, str] = field(default_factory=dict)
    throughput: dict[int, float] = field(default_factory=dict)  # batch -> req/s
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _mean(xs: Sequence[float | None]) -> float | None:
    vals = [x for x in xs if x is not None]
    return float(statistics.fmean(vals)) if vals else None


def calibrate(backend: Backend, requests: Sequence[Request], n: int = 5,
              batch_sizes: Sequence[int] = (1, 2, 4, 8, 16, 32)
              ) -> Calibration:
    """Real calls on up to `n` of `requests` (or, for a local model, the
    throughput ladder over `batch_sizes`).  Failures are recorded in
    `errors`, not raised: a calibration that reports a 400 is a result."""
    cal = Calibration(digest(backend.identity()),
                      datetime.now(timezone.utc).isoformat(timespec="seconds"),
                      0)
    sample = list(requests)[:n]
    if not sample:
        cal.errors.append("no requests to calibrate on")
        return cal
    if backend.supports_target:               # a local model: batch ladder
        return _throughput(backend, list(requests), batch_sizes, cal)
    lat, resps = [], []
    for r in sample:
        t0 = time.perf_counter()
        resp = _once(backend, r)
        lat.append(time.perf_counter() - t0)
        resps.append(resp)
    ok = [r for r in resps if r.ok]
    cal.n_calls = len(resps)
    cal.errors = [r.error for r in resps if r.error]
    if lat:
        lat_sorted = sorted(lat)
        cal.latency_median_s = float(statistics.median(lat))
        cal.latency_p95_s = float(lat_sorted[min(len(lat) - 1,
                                                 math.ceil(0.95 * len(lat)) - 1)])
    cal.in_tokens = _mean([r.meta.get("in_tokens") for r in ok])
    cal.out_tokens = _mean([r.meta.get("out_tokens") for r in ok])
    if ok:
        cal.truncated_rate = sum(bool(r.meta.get("truncated")) for r in ok) / len(ok)
    cal.rate_limits = dict(getattr(backend, "last_rate_limits", {}) or {})
    return cal


def _once(backend: Backend, r: Request) -> Response:
    try:
        return backend.complete(r)
    except Exception as exc:  # noqa: BLE001 - a failed call is a calibration result
        return Response(r.unit_id, error=f"{type(exc).__name__}: {exc}")


def _throughput(backend: Backend, reqs: list[Request],
                sizes: Sequence[int], cal: Calibration) -> Calibration:
    """Requests per second at each batch size, up the ladder until memory
    runs out or doubling the batch buys less than 10%."""
    best = 0.0
    for b in sizes:
        if b > len(reqs):
            break
        t0 = time.perf_counter()
        try:
            out = backend.complete_batch(reqs[:b])
        except RuntimeError as exc:           # torch's out-of-memory
            cal.errors.append(f"batch {b}: {exc}"[:300])
            break
        rate = b / max(time.perf_counter() - t0, 1e-9)
        cal.throughput[b] = rate
        cal.n_calls += 1
        cal.errors += [r.error for r in out if r.error]
        if rate < 1.1 * best:
            break
        best = max(best, rate)
    return cal


def _limit(rate_limits: dict[str, str], kind: str) -> tuple[str, float] | None:
    """The tightest `<kind>` limit among headers such as
    anthropic-ratelimit-requests-limit or x-ratelimit-limit-requests."""
    found = []
    for k, v in rate_limits.items():
        name = k.lower()
        if kind in name and "limit" in name and not any(
                w in name for w in ("remaining", "reset")):
            try:
                found.append((k, float(v)))
            except ValueError:
                continue
    return min(found, key=lambda kv: kv[1]) if found else None


def calibration_caveats(cal: Calibration) -> list[str]:
    """Reasons the derived settings rest on too little to trust as they are.

    * Few successful calls (HTTP): the median latency L of n calls has a
      relative spread of order 1/sqrt(n), and W is proportional to L; below
      MIN_OK_CALLS it is one or two draws from a latency distribution that
      is typically right-skewed, and the token counts likewise.
    * Failed calls: what failed in calibration fails in the run.
    * One batch size (local model): the chosen batch is not a comparison.
    """
    out = []
    if cal.throughput:
        if len(cal.throughput) < 2:
            (b,) = cal.throughput
            out.append(f"only batch {b} was measured: the batch size is not "
                       f"a comparison")
        return out
    n_ok = cal.n_calls - len(cal.errors)
    if n_ok < MIN_OK_CALLS:
        out.append(f"{n_ok} successful calibration call(s): latency and "
                   f"tokens per call, and so the worker count, rest on that")
    if cal.errors:
        out.append(f"{len(cal.errors)} of {cal.n_calls} calibration calls "
                   f"failed, e.g. {cal.errors[0][:120]}")
    return out


def derive(cal: Calibration, max_workers: int = 16, safety: float = 0.8
           ) -> dict[str, Any]:
    """Run settings from a calibration, with the assumptions behind them
    and any `calibration_caveats` (also raised as one RuntimeWarning).

    HTTP: a worker issues about 60 / L calls per minute (L the median
    latency), so W workers stay under a requests-per-minute limit R when
    W <= safety * R * L / 60, and under a tokens-per-minute limit T when
    W <= safety * T / (tokens per call) * L / 60.  Local model: workers = 1
    and the batch size with the highest measured throughput."""
    out = _derive(cal, max_workers, safety)
    out["caveats"] = calibration_caveats(cal)
    if out["caveats"]:
        warnings.warn("calibration carries caveats: " + "; ".join(out["caveats"]),
                      RuntimeWarning, stacklevel=2)
    return out


def _derive(cal: Calibration, max_workers: int, safety: float
            ) -> dict[str, Any]:
    notes: list[str] = []
    if cal.throughput:
        b = max(cal.throughput, key=lambda k: cal.throughput[k])
        notes.append(f"batch {b} had the highest throughput "
                     f"({cal.throughput[b]:.1f} req/s)")
        return {"workers": 1, "batch_size": b, "assumptions": notes}
    L = cal.latency_median_s
    if not L:
        return {"workers": 1, "batch_size": 1,
                "assumptions": ["no successful calls: one worker"]}
    bounds = [max_workers]
    rpm = _limit(cal.rate_limits, "request")
    tpm = _limit(cal.rate_limits, "token")
    if rpm:
        bounds.append(math.floor(safety * rpm[1] * L / 60))
        notes.append(f"{rpm[0]}={rpm[1]:g} read as per minute")
    # all tokens per call, even against an output-only limit: an overcount,
    # so it errs towards fewer workers
    per_call = (cal.in_tokens or 0) + (cal.out_tokens or 0)
    if tpm and per_call:
        bounds.append(math.floor(safety * tpm[1] / per_call * L / 60))
        notes.append(f"{tpm[0]}={tpm[1]:g} read as per minute, "
                     f"{per_call:.0f} tokens per call")
    if not rpm and not tpm:
        bounds.append(2)
        notes.append("no rate-limit headers: starting at 2 workers; 429s are "
                     "retried with the server's requested delay")
    workers = max(1, min(bounds))
    notes.append(f"workers = min{tuple(bounds)} = {workers} (safety {safety})")
    return {"workers": workers, "batch_size": 1, "assumptions": notes}
