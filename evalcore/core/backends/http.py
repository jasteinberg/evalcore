"""Chat-completions backends over raw httpx, one adapter per provider."""

from __future__ import annotations

import json
import os
import time
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import httpx

from .base import (
    Backend,
    Fatal,
    JSONDict,
    Message,
    Request,
    Response,
    ToolSpec,
    Transient,
    merge_params,
    refuse_dropped,
    require,
)

# --- HTTP chat backends -----------------------------------------------------
# Raw httpx rather than a vendor SDK: a vendored copy has to be readable and
# runnable by someone who did not set up this environment, and adapters of a
# few dozen lines each are cheaper to audit than a pinned SDK.
#
# Messages arrive in the harness's neutral form (core/tools.py):
#   {"role": "system" | "user" | "assistant", "content": str}
#   {"role": "assistant", "content": str | None, "tool_calls": [...]}
#   {"role": "tool", "tool_call_id", "name", "content", "is_error"?}
# and model tool calls leave as meta["tool_calls"] = [{"id", "name",
# "arguments"}].  Wire formats were written against each provider's
# reference (2 Oct 2026), and calls are detected by their presence in the
# response, never by a finish-reason value.  Arguments that are not valid
# JSON are passed on as the raw string, so the tool refuses them as a
# model-side ToolInputError rather than the harness guessing.
#
# A provider may attach opaque data to a model turn that must come back
# verbatim -- Gemini's `thoughtSignature` on functionCall parts (a missing
# one is a 400; found in a live run), Anthropic's thinking blocks.  The
# parser keeps the turn as meta["raw_turn"] = {flavour: raw}, the agent
# carries it on the assistant message, and the adapter of the SAME flavour
# echoes it unchanged instead of rebuilding the turn.


def _runs(messages: Sequence[Message]) -> list[tuple[str, Any]]:
    """Group consecutive tool messages: providers want all results of one
    assistant turn together."""
    out: list[tuple[str, Any]] = []
    buf: list[Message] = []
    for m in messages:
        if m["role"] == "tool":
            buf.append(m)
            continue
        if buf:
            out.append(("tool", buf))
            buf = []
        out.append((m["role"], m))
    if buf:
        out.append(("tool", buf))
    return out


def _json_args(raw: Any) -> Any:
    if isinstance(raw, Mapping):
        return dict(raw)
    try:
        val = json.loads(raw)
    except (TypeError, ValueError):
        return raw
    return val if isinstance(val, Mapping) else raw


def _anthropic_adapter(model: str, messages: Sequence[Message],
                       params: Mapping[str, Any],
                       tools: Sequence[ToolSpec] | None = None
                       ) -> tuple[str, JSONDict]:
    refuse_dropped(params, {"max_tokens", "temperature", "top_p",
                            "stop_sequences"}, "anthropic adapter")
    sys_msgs = [m["content"] for m in messages if m["role"] == "system"]
    msgs: list[JSONDict] = []
    for role, m in _runs([m for m in messages if m["role"] != "system"]):
        if role == "tool":
            # one user message, tool_result blocks first (the API requires it)
            msgs.append({"role": "user", "content": [
                {"type": "tool_result", "tool_use_id": t["tool_call_id"],
                 "content": t["content"], **({"is_error": True}
                                             if t.get("is_error") else {})}
                for t in m]})
        elif role == "assistant" and (m.get("raw_turn") or {}).get("anthropic"):
            msgs.append({"role": "assistant",
                         "content": m["raw_turn"]["anthropic"]})
        elif role == "assistant" and m.get("tool_calls"):
            blocks = ([{"type": "text", "text": m["content"]}]
                      if m.get("content") else [])
            blocks += [{"type": "tool_use", "id": c["id"], "name": c["name"],
                        "input": c["arguments"] if isinstance(
                            c["arguments"], Mapping) else {}}
                       for c in m["tool_calls"]]
            msgs.append({"role": "assistant", "content": blocks})
        else:
            msgs.append({"role": m["role"], "content": m["content"]})
    body = {"model": model,
            "max_tokens": require(params, "max_tokens", "anthropic"),
            "messages": msgs}
    if sys_msgs:
        body["system"] = "\n\n".join(sys_msgs)
    if tools:
        body["tools"] = [{"name": t["name"], "description": t["description"],
                          "input_schema": t["parameters"]} for t in tools]
    for k in ("temperature", "top_p", "stop_sequences"):
        if k in params:
            body[k] = params[k]
    return "/v1/messages", body


def _anthropic_parse(js: Mapping[str, Any]) -> tuple[str, JSONDict]:
    blocks = js.get("content", [])
    text = "".join(b.get("text", "") for b in blocks if b.get("type") == "text")
    calls = [{"id": b["id"], "name": b["name"], "arguments": b.get("input", {})}
             for b in blocks if b.get("type") == "tool_use"]
    u = js.get("usage", {})
    meta = {"in_tokens": u.get("input_tokens"),
            "out_tokens": u.get("output_tokens"),
            "stop_reason": js.get("stop_reason"),
            "truncated": js.get("stop_reason") == "max_tokens"}
    if calls:
        meta["tool_calls"] = calls
        meta["raw_turn"] = {"anthropic": blocks}
    return text, meta


def _openai_adapter(model: str, messages: Sequence[Message],
                    params: Mapping[str, Any],
                    tools: Sequence[ToolSpec] | None = None
                    ) -> tuple[str, JSONDict]:
    refuse_dropped(params, {"temperature", "top_p", "max_tokens", "seed",
                            "stop"}, "openai adapter")
    msgs: list[JSONDict] = []
    for m in messages:
        if m["role"] == "tool":
            msgs.append({"role": "tool", "tool_call_id": m["tool_call_id"],
                         "content": m["content"]})
        elif m["role"] == "assistant" and m.get("tool_calls"):
            msgs.append({"role": "assistant", "content": m.get("content"),
                         "tool_calls": [
                             {"id": c["id"], "type": "function",
                              "function": {"name": c["name"],
                                           "arguments": c["arguments"]
                                           if isinstance(c["arguments"], str)
                                           else json.dumps(c["arguments"])}}
                             for c in m["tool_calls"]]})
        else:
            msgs.append({"role": m["role"], "content": m["content"]})
    body: JSONDict = {"model": model, "messages": msgs}
    if tools:
        body["tools"] = [{"type": "function",
                          "function": {"name": t["name"],
                                       "description": t["description"],
                                       "parameters": t["parameters"]}}
                         for t in tools]
    for k in ("temperature", "top_p", "seed", "logprobs", "top_logprobs",
              "stop"):
        if k in params:
            body[k] = params[k]
    # `max_tokens` is deprecated in Chat Completions and rejected by newer
    # models (found live, 2 Oct); `max_completion_tokens` replaces it.  It
    # also counts reasoning tokens, so on a reasoning model a small budget
    # can be spent before any visible text -- the row is then `truncated`.
    if "max_tokens" in params:
        body["max_completion_tokens"] = params["max_tokens"]
    return "/v1/chat/completions", body


def _openai_parse(js: Mapping[str, Any]) -> tuple[str | None, JSONDict]:
    ch = js["choices"][0]
    msg = ch.get("message", {})
    u = js.get("usage", {})
    meta = {"in_tokens": u.get("prompt_tokens"),
            "out_tokens": u.get("completion_tokens"),
            "stop_reason": ch.get("finish_reason"),
            "truncated": ch.get("finish_reason") == "length",
            "logprobs": ch.get("logprobs")}
    calls = [{"id": c["id"], "name": c["function"]["name"],
              "arguments": _json_args(c["function"].get("arguments"))}
             for c in msg.get("tool_calls") or [] if c.get("type") == "function"]
    if calls:
        meta["tool_calls"] = calls
    return msg.get("content", ""), meta


def _gemini_adapter(model: str, messages: Sequence[Message],
                    params: Mapping[str, Any],
                    tools: Sequence[ToolSpec] | None = None
                    ) -> tuple[str, JSONDict]:
    refuse_dropped(params, {"temperature", "top_p", "max_tokens", "stop"},
                   "gemini adapter")
    sys_msgs = [m["content"] for m in messages if m["role"] == "system"]
    contents: list[JSONDict] = []
    for role, m in _runs([m for m in messages if m["role"] != "system"]):
        if role == "tool":
            # function responses go back in a user turn; `response` must be
            # an object, so the text is wrapped
            contents.append({"role": "user", "parts": [
                {"functionResponse": {
                    "name": t["name"],
                    "response": {"error" if t.get("is_error") else "result":
                                 t["content"]},
                    **({"id": t["tool_call_id"]}
                       if not str(t["tool_call_id"]).startswith("gemini-")
                       else {})}}
                for t in m]})
        elif role == "assistant" and (m.get("raw_turn") or {}).get("gemini"):
            contents.append({"role": "model", "parts": m["raw_turn"]["gemini"]})
        elif role == "assistant":
            parts = [{"text": m["content"]}] if m.get("content") else []
            parts += [{"functionCall": {
                "name": c["name"], "args": c["arguments"] if isinstance(
                    c["arguments"], Mapping) else {},
                **({"id": c["id"]} if not str(c["id"]).startswith("gemini-")
                   else {})}} for c in m.get("tool_calls") or []]
            contents.append({"role": "model", "parts": parts})
        else:
            contents.append({"role": "user", "parts": [{"text": m["content"]}]})
    body: JSONDict = {"contents": contents}
    if sys_msgs:
        body["systemInstruction"] = {"parts": [{"text": "\n\n".join(sys_msgs)}]}
    if tools:
        body["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"],
             "parameters": t["parameters"]} for t in tools]}]
    gc = {}
    for src, dst in (("temperature", "temperature"), ("top_p", "topP"),
                     ("max_tokens", "maxOutputTokens"), ("stop", "stopSequences")):
        if src in params:
            gc[dst] = params[src]
    if gc:
        body["generationConfig"] = gc
    return f"/v1beta/models/{model}:generateContent", body


def _gemini_parse(js: Mapping[str, Any]) -> tuple[str, JSONDict]:
    cands = js.get("candidates") or [{}]
    parts = cands[0].get("content", {}).get("parts", [])
    u = js.get("usageMetadata", {})
    # the reference marks functionCall.id optional; a call without one gets
    # a local id that is never sent back to the API
    calls = [{"id": p["functionCall"].get("id") or f"gemini-{k}",
              "name": p["functionCall"]["name"],
              "arguments": p["functionCall"].get("args", {})}
             for k, p in enumerate(parts) if "functionCall" in p]
    meta = {"in_tokens": u.get("promptTokenCount"),
            "out_tokens": u.get("candidatesTokenCount"),
            "stop_reason": cands[0].get("finishReason"),
            "truncated": cands[0].get("finishReason") == "MAX_TOKENS"}
    if calls:
        meta["tool_calls"] = calls
        meta["raw_turn"] = {"gemini": parts}
    return "".join(p.get("text", "") for p in parts), meta


RATE_LIMIT_CUES = ("ratelimit", "rate-limit", "quota", "retry-after")


def _error_text(r: httpx.Response) -> str:
    """The provider's own error: status, message and a bounded dump of its
    details.  The details are where the actionable part lives (which quota,
    which field) -- a 200-character cut lost them in the first live run."""
    try:
        err = r.json().get("error")
    except ValueError:
        err = None
    if not isinstance(err, Mapping):
        return f"HTTP {r.status_code}: {r.text[:2000]}"
    head = " ".join(str(err[k]) for k in ("status", "type") if err.get(k))
    out = f"HTTP {r.status_code}: {head + ': ' if head else ''}{err.get('message', '')}"
    if err.get("details"):
        out += f" | details: {json.dumps(err['details'])[:1500]}"
    return out


def _retry_after(r: httpx.Response) -> float | None:
    """Seconds the server asked to wait: the standard Retry-After header
    (seconds form), or a google.rpc.RetryInfo `retryDelay` such as "37s" in
    the error details.  The RetryInfo form follows Google's published
    error model; it has not yet been seen in a live run."""
    h = r.headers.get("retry-after")
    if h:
        try:
            return float(h)
        except ValueError:
            pass                      # an HTTP-date: fall back to backoff
    try:
        details = r.json().get("error", {}).get("details", [])
    except (ValueError, AttributeError):
        return None
    for d in details if isinstance(details, list) else []:
        delay = d.get("retryDelay") if isinstance(d, Mapping) else None
        if isinstance(delay, str) and delay.endswith("s"):
            try:
                return float(delay[:-1])
            except ValueError:
                return None
    return None


ADAPTERS = {"anthropic": (_anthropic_adapter, _anthropic_parse),
            "openai": (_openai_adapter, _openai_parse),
            "gemini": (_gemini_adapter, _gemini_parse)}


class HTTPBackend(Backend):
    """Thread-safe chat-completions client.  Classifies 408/409/429/5xx and
    transport errors as Transient and everything else as Fatal, so the retry
    policy is decided once here rather than at each call site."""

    name = "http"
    supports_tools = True
    # read-only: shared by every instance, so a mutation would leak
    BASES = MappingProxyType({"anthropic": "https://api.anthropic.com",
                              "openai": "https://api.openai.com",
                              "gemini": "https://generativelanguage.googleapis.com"})
    ENV = MappingProxyType({"anthropic": "ANTHROPIC_API_KEY",
                            "openai": "OPENAI_API_KEY",
                            "gemini": "GEMINI_API_KEY"})

    def __init__(self, flavour: str = "anthropic", api_key: str | None = None,
                 base_url: str | None = None, timeout: float = 120.0,
                 extra_headers: Mapping[str, str] | None = None,
                 default_params: Mapping[str, Any] | None = None) -> None:
        import httpx
        if flavour not in ADAPTERS:
            raise ValueError(f"flavour must be one of {sorted(ADAPTERS)}")
        env = self.ENV[flavour]
        key = api_key or os.environ.get(env)
        if not key:
            raise Fatal(f"no API key: set ${env}")
        headers = {"content-type": "application/json"}
        if flavour == "anthropic":
            headers |= {"x-api-key": key, "anthropic-version": "2023-06-01"}
        elif flavour == "gemini":
            headers |= {"x-goog-api-key": key}
        else:
            headers |= {"authorization": f"Bearer {key}"}
        headers |= dict(extra_headers or {})
        self.flavour = flavour
        self.base_url = base_url or self.BASES[flavour]
        self.default_params = dict(default_params or {})
        self.adapt, self.parse = ADAPTERS[flavour]
        # rate-limit headers of the latest response, verbatim, for
        # calibration; replaced (not merged) on every response
        self.last_rate_limits: dict[str, str] = {}
        self.client = httpx.Client(base_url=self.base_url,
                                   headers=headers, timeout=timeout)

    def identity(self) -> dict[str, Any]:
        # extra_headers are left out on purpose: they carry auth and
        # tracing, and keying on them would re-run a sweep for a new key.
        return {**self._base_identity(), "flavour": self.flavour,
                "base_url": self.base_url}

    def complete(self, req: Request) -> Response:
        import httpx
        params = merge_params(self.default_params, req.params)
        model = require(params, "model", self.flavour)
        params.pop("model")
        path, body = self.adapt(model, req.messages, params, req.tools)
        t0 = time.perf_counter()
        try:
            r = self.client.post(path, json=body)
        except httpx.HTTPError as exc:
            raise Transient(f"transport: {exc}") from exc
        dt = time.perf_counter() - t0
        self.last_rate_limits = {k: v for k, v in r.headers.items()
                                 if any(c in k.lower() for c in RATE_LIMIT_CUES)}
        if r.status_code in (408, 409, 429) or r.status_code >= 500:
            raise Transient(_error_text(r), retry_after=_retry_after(r))
        if r.status_code >= 400:
            raise Fatal(_error_text(r))
        text, meta = self.parse(r.json())
        return Response(req.unit_id, text=text,
                        meta={**meta, "latency_s": dt, "backend": self.flavour,
                              "model": model})

    def close(self) -> None:
        self.client.close()
