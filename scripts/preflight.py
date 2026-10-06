#!/usr/bin/env python3
"""Check an API endpoint before a sweep, and measure what sizing it needs.

One provider and one model per invocation.  The script sends `--n` requests
through the same client, adapter and parser the harness uses (`HTTPBackend`),
so a pass here means the harness's own request path works, not a
hand-written copy of it.  It reports:

* whether auth, the request shape and the response parse work.  A 4xx
  usually names the offending field;
* the reply, the token accounting and the stop reason, including whether
  the truncation flag fires at your `--max-tokens`;
* round-trip latency, median and max over the n calls;
* the rate-limit headers exactly as the server sent them;
* the wall-clock arithmetic for `--units` requests at several worker counts.

Wall-clock time for U units at W workers and per-call latency L is U L / W,
and the request rate is 60 W / L per minute.  Choose W so that rate stays
under the requests-per-minute limit in the headers.  L depends on prompt
and output length, so the estimate is only as good as the probe: pass a
representative prompt (`--prompt-file`) and the real `--max-tokens`.  The
defaults are a minimal "is it alive" probe and will underestimate L.

Model names have no default, because they go stale.  `--list-models` asks
the provider what it serves.

    python scripts/preflight.py --provider openai --list-models
    python scripts/preflight.py --provider openai --model <name>
    python scripts/preflight.py --provider anthropic --model <name> \\
        --prompt-file prompt.txt --max-tokens 512 --n 10 --units 2400

The measurement is `evalcore.core.calibrate` -- the same code the pipeline
runs before an experiment -- so this script only parses arguments and
prints.  Exit status: 0 if every call succeeded, 1 if one failed, 2 if no
API key.

Drafted with the assistance of Claude (Anthropic).
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from evalcore.core.backends import ADAPTERS, Fatal, HTTPBackend, Request
from evalcore.core.calibrate import Calibration, calibrate, derive

ALIVE = "Reply with the single word: ok"
MODEL_LIST_PATHS = {"anthropic": "/v1/models", "openai": "/v1/models",
                    "gemini": "/v1beta/models"}
WORKERS = (1, 4, 8, 16, 32)

G, R, Y, DIM, END = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def wall_clock(units: int, latency: float, workers: int) -> tuple[float, float]:
    """(minutes for `units` calls, requests per minute) at `workers`."""
    return units * latency / workers / 60.0, 60.0 * workers / latency


def list_models(be: HTTPBackend, provider: str) -> int:
    r = be.client.get(MODEL_LIST_PATHS[provider])
    if r.status_code >= 400:
        print(f"{R}{provider}: HTTP {r.status_code} {r.text[:160]}{END}")
        return 1
    js = r.json()
    rows = js.get("data") or js.get("models") or []
    names = sorted({(m.get("id") or m.get("name", "")).split("/")[-1]
                    for m in rows})
    print(f"{G}{provider}: {len(names)} models{END}")
    for name in names:
        print(f"   {name}")
    return 0


def report(provider: str, model: str, params: dict, prompt_chars: int,
           cal: Calibration, units: int | None) -> None:
    ok = not cal.errors
    print(f"\n{G if ok else R}{provider} / {model}: "
          f"{'OK' if ok else 'FAILED'}{END}")
    print(f"  {DIM}probe      {prompt_chars} prompt chars, params {params}{END}")
    for e in cal.errors[:1]:
        print(f"  {e[:600]}")
    if cal.latency_median_s is not None and ok:
        print(f"  tokens     in={cal.in_tokens} out={cal.out_tokens}  "
              f"truncated rate={cal.truncated_rate}")
        if cal.truncated_rate:
            print(f"  {Y}hit max_tokens: at this budget the harness marks rows "
                  f"'truncated' and does not score them{END}")
        print(f"  latency    median {cal.latency_median_s * 1000:.0f} ms   "
              f"p95 {cal.latency_p95_s * 1000:.0f} ms   (n={cal.n_calls})")
    if cal.rate_limits:
        print("  rate limits, as returned:")
        for k, v in sorted(cal.rate_limits.items()):
            print(f"    {DIM}{k:42s}{END} {v}")
    elif ok:
        print(f"  {Y}no rate-limit headers returned: find the safe worker "
              f"count empirically, starting low{END}")
    if ok:
        d = derive(cal)
        print(f"  suggested  workers={d['workers']}  ({'; '.join(d['assumptions'])})")
    if units and ok and cal.latency_median_s:
        print(f"\n  {units} units at median latency "
              f"{cal.latency_median_s:.2f} s per call:")
        for w in WORKERS:
            minutes, rpm = wall_clock(units, cal.latency_median_s, w)
            print(f"    {w:3d} workers   {rpm:7.0f} req/min   {minutes:7.1f} min")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(
        description="Check an endpoint and measure latency and rate limits "
                    "before a sweep.")
    ap.add_argument("--provider", choices=sorted(ADAPTERS), required=True)
    ap.add_argument("--model", help="model name as the provider spells it "
                    "(required unless --list-models)")
    ap.add_argument("--list-models", action="store_true",
                    help="print the models the provider serves, and exit")
    ap.add_argument("--base-url", help="override the provider's base URL "
                    "(a proxy or a compatible server)")
    ap.add_argument("--prompt-file", type=Path,
                    help="a representative prompt; default is a one-word "
                         "liveness probe")
    ap.add_argument("--max-tokens", type=int, default=8,
                    help="use the sweep's real budget for a latency estimate "
                         "(default 8, liveness only)")
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--n", type=int, default=5, help="calls to time")
    ap.add_argument("--units", type=int,
                    help="planned number of calls, for the wall-clock table")
    a = ap.parse_args(argv)
    if not a.list_models and not a.model:
        ap.error("--model is required (use --list-models to see names)")
    if a.n < 1:
        ap.error("--n must be at least 1")
    return a


def main(argv: list[str] | None = None) -> int:
    a = parse_args(argv)
    try:
        be = HTTPBackend(a.provider, base_url=a.base_url)
    except Fatal as exc:
        print(f"{Y}{a.provider}: {exc}{END}")
        return 2
    try:
        if a.list_models:
            return list_models(be, a.provider)
        prompt = a.prompt_file.read_text() if a.prompt_file else ALIVE
        params = {"model": a.model, "max_tokens": a.max_tokens,
                  "temperature": a.temperature}
        reqs = [Request(f"probe-{k}", [{"role": "user", "content": prompt}],
                        params) for k in range(a.n)]
        cal = calibrate(be, reqs, n=a.n)
        report(a.provider, a.model, params, len(prompt), cal, a.units)
    finally:
        be.close()
    if not cal.errors:
        print("\nAlso worth checking before a long run: the account's "
              "spending limit, and that\nno other traffic shares this key's "
              "rate limit.")
    return 0 if not cal.errors else 1


if __name__ == "__main__":
    sys.exit(main())
