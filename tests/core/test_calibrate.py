"""Calibration measures the backend the run will use, on its own requests,
and derive() turns that into settings with every assumption written down."""

from __future__ import annotations

import json

import pytest

from evalcore import Request
from evalcore.core.backends import HTTPBackend
from evalcore.core.calibrate import Calibration, calibrate, derive

httpx = pytest.importorskip("httpx")

MSGS = [{"role": "user", "content": "Reply with the single word: ok"}]


def backend(handler, flavour="openai"):
    seen = []

    def record(req):
        seen.append(req)
        return handler(req)

    be = HTTPBackend(flavour, api_key="test-key",
                     default_params={"model": "m", "max_tokens": 8})
    real = be.client
    be.client = httpx.Client(base_url=real.base_url, headers=real.headers,
                             transport=httpx.MockTransport(record))
    real.close()
    return be, seen


def reply(finish="stop", headers=None, tokens=(12, 1)):
    def handler(req):
        return httpx.Response(200, headers=headers or {}, json={
            "choices": [{"message": {"content": " ok "}, "finish_reason": finish}],
            "usage": {"prompt_tokens": tokens[0],
                      "completion_tokens": tokens[1]}})
    return handler


REQS = [Request(f"r{k}", MSGS, {}) for k in range(4)]


def test_calls_go_through_the_backends_own_request_path():
    be, seen = backend(reply(headers={"x-ratelimit-limit-requests": "500",
                                      "date": "x"}))
    cal = calibrate(be, REQS, n=4)
    assert cal.n_calls == 4 and len(seen) == 4 and not cal.errors
    assert (cal.in_tokens, cal.out_tokens, cal.truncated_rate) == (12, 1, 0.0)
    assert cal.rate_limits == {"x-ratelimit-limit-requests": "500"}
    body = json.loads(seen[0].content)
    assert body["model"] == "m" and body["max_completion_tokens"] == 8
    assert seen[0].headers["authorization"] == "Bearer test-key"


def test_truncation_rate_is_measured():
    be, _ = backend(reply(finish="length"))
    assert calibrate(be, REQS, n=2).truncated_rate == 1.0


def test_a_failing_endpoint_is_a_result_not_an_exception():
    be, _ = backend(lambda req: httpx.Response(400, json={"error": {
        "message": "Unsupported parameter: 'max_tokens'"}}))
    cal = calibrate(be, REQS, n=2)
    assert cal.n_calls == 2 and len(cal.errors) == 2
    assert "Unsupported parameter" in cal.errors[0]
    assert cal.in_tokens is None and cal.truncated_rate is None


def cal_with(lat, limits, tokens=(1000, 100)):
    return Calibration("id", "t", 5, latency_median_s=lat,
                       in_tokens=tokens[0], out_tokens=tokens[1],
                       rate_limits=limits)


def test_derive_respects_request_and_token_limits():
    """W <= safety R L / 60 and W <= safety T / tokens L / 60.
    R = 500/min, L = 1.2 s: 0.8 * 500 * 1.2 / 60 = 8.
    T = 200k/min, 1100 tokens/call: 0.8 * 200000 / 1100 * 1.2 / 60 = 2.9 -> 2."""
    d = derive(cal_with(1.2, {"x-ratelimit-limit-requests": "500"}))
    assert d["workers"] == 8
    d = derive(cal_with(1.2, {"x-ratelimit-limit-requests": "500",
                              "x-ratelimit-limit-tokens": "200000"}))
    assert d["workers"] == 2
    assert any("read as per minute" in n for n in d["assumptions"])


def test_derive_without_headers_starts_low_and_says_why():
    d = derive(cal_with(0.7, {}))
    assert d["workers"] == 2
    assert any("no rate-limit headers" in n for n in d["assumptions"])


def test_derive_caps_at_max_workers():
    d = derive(cal_with(2.0, {"anthropic-ratelimit-requests-limit": "10000"}),
               max_workers=16)
    assert d["workers"] == 16


def test_local_models_get_the_fastest_measured_batch():
    cal = Calibration("id", "t", 3, throughput={1: 10.0, 8: 55.0, 16: 54.0})
    assert derive(cal) == {"workers": 1, "batch_size": 8,
                           "assumptions": [
                               "batch 8 had the highest throughput (55.0 req/s)"],
                           "caveats": []}


def test_settings_from_well_measured_calls_carry_no_caveats():
    import warnings
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        assert derive(cal_with(1.0, {}))["caveats"] == []


def test_settings_from_too_few_successful_calls_are_flagged():
    """5 calls, 3 failed: two latencies behind the worker count."""
    cal = Calibration("id", "t", 5, latency_median_s=1.0, in_tokens=10,
                      out_tokens=2, errors=["HTTP 429: slow down"] * 3)
    with pytest.warns(RuntimeWarning, match="calibration carries caveats"):
        d = derive(cal)
    assert d["caveats"] == [
        "2 successful calibration call(s): latency and tokens per call, and "
        "so the worker count, rest on that",
        "3 of 5 calibration calls failed, e.g. HTTP 429: slow down"]


def test_a_single_measured_batch_size_is_flagged():
    cal = Calibration("id", "t", 1, throughput={1: 10.0},
                      errors=["batch 2: CUDA out of memory"])
    with pytest.warns(RuntimeWarning):
        d = derive(cal)
    assert d["batch_size"] == 1
    assert d["caveats"] == ["only batch 1 was measured: the batch size is "
                            "not a comparison"]
