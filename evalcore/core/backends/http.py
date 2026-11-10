"""Chat-completions backends over raw httpx, one adapter per provider."""

from __future__ import annotations

import json
import os
import threading
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
    refuse_unread,
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
# one is a 400, found in a live run), Anthropic's thinking blocks.  The
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


def _split_system(messages: Sequence[Message]
                  ) -> tuple[str | None, list[tuple[str, Any]]]:
    """The system messages joined (None if there are none), and the rest
    grouped by `_runs`: Anthropic and Gemini take the system prompt as a
    separate field."""
    sys_msgs = [m["content"] for m in messages if m["role"] == "system"]
    rest = _runs([m for m in messages if m["role"] != "system"])
    return ("\n\n".join(sys_msgs) if sys_msgs else None), rest


def _args_object(call: Mapping[str, Any]) -> Any:
    """A call's arguments as the object Anthropic and Gemini require.  A raw
    string (arguments that were not valid JSON) is sent as {}."""
    return call["arguments"] if isinstance(call["arguments"], Mapping) else {}


def _gemini_id(call_id: Any) -> dict[str, Any]:
    """{"id": call_id} for an id the API issued, {} for a local
    "gemini-<k>" id, which must never be sent back."""
    return {} if str(call_id).startswith("gemini-") else {"id": call_id}


def _forced_tool(params: Mapping[str, Any], tools: Sequence[ToolSpec] | None,
                 who: str) -> str | None:
    """The tool the model must call, from params["tool_choice"] (a tool
    name), or None.  A forced call is how a structured answer is asked for
    (a judge's verdict): left to the model's choice, a reply in prose
    instead would be a parse failure on some rows only.  A name that is
    not among the request's tools is refused, since no reply could
    satisfy it."""
    name = params.get("tool_choice")
    if name is None:
        return None
    names = [t["name"] for t in tools or []]
    if name not in names:
        raise Fatal(f"{who}: tool_choice {name!r} is not one of the "
                    f"request's tools {names}")
    return str(name)


# each adapter sends exactly these params.  The forwarding code below reads
# the same sets, so the refusal and the body cannot drift apart
ANTHROPIC_SENT = frozenset({"max_tokens", "temperature", "top_p",
                            "stop_sequences", "tool_choice"})
OPENAI_SENT = frozenset({"temperature", "top_p", "seed", "logprobs",
                         "top_logprobs", "stop", "max_tokens", "tool_choice"})
GEMINI_SENT = frozenset({"temperature", "top_p", "max_tokens", "stop",
                         "tool_choice"})


def _anthropic_adapter(model: str, messages: Sequence[Message],
                       params: Mapping[str, Any],
                       tools: Sequence[ToolSpec] | None = None
                       ) -> tuple[str, JSONDict]:
    refuse_unread(params, ANTHROPIC_SENT, "anthropic adapter")
    system, runs = _split_system(messages)
    msgs: list[JSONDict] = []
    for role, m in runs:
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
                        "input": _args_object(c)} for c in m["tool_calls"]]
            msgs.append({"role": "assistant", "content": blocks})
        else:
            msgs.append({"role": m["role"], "content": m["content"]})
    body = {"model": model,
            "max_tokens": require(params, "max_tokens", "anthropic"),
            "messages": msgs}
    if system is not None:
        body["system"] = system
    if tools:
        body["tools"] = [{"name": t["name"], "description": t["description"],
                          "input_schema": t["parameters"]} for t in tools]
    forced = _forced_tool(params, tools, "anthropic adapter")
    if forced is not None:
        body["tool_choice"] = {"type": "tool", "name": forced}
    body |= {k: params[k] for k in sorted(ANTHROPIC_SENT
                                          - {"max_tokens", "tool_choice"})
             if k in params}
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
    refuse_unread(params, OPENAI_SENT, "openai adapter")
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
    forced = _forced_tool(params, tools, "openai adapter")
    if forced is not None:
        body["tool_choice"] = {"type": "function",
                               "function": {"name": forced}}
    body |= {k: params[k] for k in sorted(OPENAI_SENT
                                          - {"max_tokens", "tool_choice"})
             if k in params}
    # `max_tokens` is deprecated in Chat Completions and rejected by newer
    # models (found live, 2 Oct).  `max_completion_tokens` replaces it.  It
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
            "out_tokens": u.get("completion_tokens"),     # reasoning included
            "thinking_tokens": (u.get("completion_tokens_details") or {}
                                ).get("reasoning_tokens"),
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
    refuse_unread(params, GEMINI_SENT, "gemini adapter")
    system, runs = _split_system(messages)
    contents: list[JSONDict] = []
    for role, m in runs:
        if role == "tool":
            # function responses go back in a user turn, and `response`
            # must be an object, so the text is wrapped
            contents.append({"role": "user", "parts": [
                {"functionResponse": {
                    "name": t["name"],
                    "response": {"error" if t.get("is_error") else "result":
                                 t["content"]},
                    **_gemini_id(t["tool_call_id"])}}
                for t in m]})
        elif role == "assistant" and (m.get("raw_turn") or {}).get("gemini"):
            contents.append({"role": "model", "parts": m["raw_turn"]["gemini"]})
        elif role == "assistant":
            parts = [{"text": m["content"]}] if m.get("content") else []
            parts += [{"functionCall": {"name": c["name"],
                                        "args": _args_object(c),
                                        **_gemini_id(c["id"])}}
                      for c in m.get("tool_calls") or []]
            contents.append({"role": "model", "parts": parts})
        else:
            contents.append({"role": "user", "parts": [{"text": m["content"]}]})
    body: JSONDict = {"contents": contents}
    if system is not None:
        body["systemInstruction"] = {"parts": [{"text": system}]}
    if tools:
        body["tools"] = [{"functionDeclarations": [
            {"name": t["name"], "description": t["description"],
             "parameters": t["parameters"]} for t in tools]}]
    forced = _forced_tool(params, tools, "gemini adapter")
    if forced is not None:
        body["toolConfig"] = {"functionCallingConfig": {
            "mode": "ANY", "allowedFunctionNames": [forced]}}
    gc ={dst: params[src] for src, dst in (
        ("temperature", "temperature"), ("top_p", "topP"),
        ("max_tokens", "maxOutputTokens"), ("stop", "stopSequences"))
        if src in params}
    if gc:
        body["generationConfig"] = gc
    return f"/v1beta/models/{model}:generateContent", body


def _gemini_out(u: Mapping[str, Any]) -> int | None:
    """Billed output tokens.  Gemini reports thinking apart from the reply
    (thoughtsTokenCount next to candidatesTokenCount) and bills both, while
    Anthropic's output_tokens and OpenAI's completion_tokens already include
    thinking.  Summing here gives out_tokens one meaning on every provider:
    what the call's output is billed as."""
    parts = [u.get("candidatesTokenCount"), u.get("thoughtsTokenCount")]
    if all(x is None for x in parts):
        return None
    return sum(x for x in parts if x is not None)


def _gemini_parse(js: Mapping[str, Any]) -> tuple[str, JSONDict]:
    # A blocked prompt is answered 200 with promptFeedback.blockReason and
    # no candidates.  Read as a reply it would be an ok row with text "",
    # i.e. a wrong answer.  It is a refusal of the request, and fatal because
    # resending the same prompt is blocked the same way.
    if not js.get("candidates"):
        reason = (js.get("promptFeedback") or {}).get("blockReason")
        raise Fatal(f"gemini blocked the prompt (blockReason={reason})"
                    if reason else "gemini reply has no candidates")
    cands = js["candidates"]
    parts = cands[0].get("content", {}).get("parts", [])
    u = js.get("usageMetadata", {})
    # the reference marks functionCall.id optional.  A call without one gets
    # a local id that is never sent back to the API
    calls = [{"id": p["functionCall"].get("id") or f"gemini-{k}",
              "name": p["functionCall"]["name"],
              "arguments": p["functionCall"].get("args", {})}
             for k, p in enumerate(parts) if "functionCall" in p]
    meta = {"in_tokens": u.get("promptTokenCount"),
            "out_tokens": _gemini_out(u),
            "thinking_tokens": u.get("thoughtsTokenCount"),
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
    error model, but it has not yet been seen in a live run."""
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
        # calibration, replaced (not merged) on every response
        self.last_rate_limits: dict[str, str] = {}
        self._client_args: dict[str, Any] = {
            "base_url": self.base_url, "headers": headers, "timeout": timeout}
        self._client_lock = threading.Lock()
        self.client = httpx.Client(**self._client_args)

    def _http(self) -> Any:
        """The client, reopened if `close` closed it: `execute` closes its
        backend at the end, and the same backend may run again in this
        process (a retry of the error rows, a notebook re-run, several
        shards in one loop).  Without this every call then failed."""
        import httpx
        with self._client_lock:
            if self.client.is_closed:
                self.client = httpx.Client(**self._client_args)
            return self.client

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
            r = self._http().post(path, json=body)
        except httpx.HTTPError as exc:
            raise Transient(f"transport: {exc}") from exc
        dt = time.perf_counter() - t0
        self.last_rate_limits = {k: v for k, v in r.headers.items()
                                 if any(c in k.lower() for c in RATE_LIMIT_CUES)}
        _raise_for_status(r)
        text, meta = self.parse(r.json())
        return Response(req.unit_id, text=text,
                        meta={**meta, "latency_s": dt, "backend": self.flavour,
                              "model": model})

    def close(self) -> None:
        self.client.close()


def _raise_for_status(r: Any) -> None:
    """408, 409, 429 and 5xx are worth retrying (Transient, honouring
    Retry-After).  Any other 4xx is the request's fault (Fatal)."""
    if r.status_code in (408, 409, 429) or r.status_code >= 500:
        raise Transient(_error_text(r), retry_after=_retry_after(r))
    if r.status_code >= 400:
        raise Fatal(_error_text(r))
