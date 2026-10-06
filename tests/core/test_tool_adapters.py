"""Provider tool-calling formats.  Expected shapes follow each provider's
reference as read on 2 Oct 2026 (Anthropic Messages tool use; OpenAI Chat
Completions types; Gemini generateContent), and the end-to-end tests check
what actually goes over the wire on the second turn."""

from __future__ import annotations

import json

import pytest

from evalcore import AgentBackend, CorpusSearch, HTTPBackend, Request
from evalcore.core.backends.http import (
    _anthropic_adapter,
    _anthropic_parse,
    _gemini_adapter,
    _gemini_parse,
    _openai_adapter,
    _openai_parse,
)

httpx = pytest.importorskip("httpx")

SPEC = {"name": "search", "description": "Search the docs.",
        "parameters": {"type": "object",
                       "properties": {"query": {"type": "string"}},
                       "required": ["query"]}}
CONV = [{"role": "system", "content": "Be brief."},
        {"role": "user", "content": "Where does the Velor rise?"},
        {"role": "assistant", "content": "Let me look.",
         "tool_calls": [{"id": "call_1", "name": "search",
                         "arguments": {"query": "Velor"}},
                        {"id": "call_2", "name": "search",
                         "arguments": {"query": "Velor source"}}]},
        {"role": "tool", "tool_call_id": "call_1", "name": "search",
         "content": "[d1] The Velor rises in the Askern hills."},
        {"role": "tool", "tool_call_id": "call_2", "name": "search",
         "content": "error: bad query", "is_error": True}]


def test_anthropic_request_shape():
    _, body = _anthropic_adapter("m", CONV, {"max_tokens": 64}, [SPEC])
    assert body["system"] == "Be brief."
    assert body["tools"] == [{"name": "search", "description": "Search the docs.",
                              "input_schema": SPEC["parameters"]}]
    user, asst, results = body["messages"]
    assert user == {"role": "user", "content": "Where does the Velor rise?"}
    assert asst == {"role": "assistant", "content": [
        {"type": "text", "text": "Let me look."},
        {"type": "tool_use", "id": "call_1", "name": "search",
         "input": {"query": "Velor"}},
        {"type": "tool_use", "id": "call_2", "name": "search",
         "input": {"query": "Velor source"}}]}
    # both results in ONE user message, tool_result blocks only
    assert results == {"role": "user", "content": [
        {"type": "tool_result", "tool_use_id": "call_1",
         "content": "[d1] The Velor rises in the Askern hills."},
        {"type": "tool_result", "tool_use_id": "call_2",
         "content": "error: bad query", "is_error": True}]}


def test_anthropic_parse_of_the_documented_example():
    js = {"id": "msg_01Aq9w938a90dw8q", "model": "m", "stop_reason": "tool_use",
          "role": "assistant", "content": [
              {"type": "text",
               "text": "I'll check the current weather in San Francisco for you."},
              {"type": "tool_use", "id": "toolu_01A09q90qw90lq917835lq9",
               "name": "get_weather",
               "input": {"location": "San Francisco, CA", "unit": "celsius"}}]}
    text, meta = _anthropic_parse(js)
    assert text.startswith("I'll check")
    assert meta["tool_calls"] == [{"id": "toolu_01A09q90qw90lq917835lq9",
                                   "name": "get_weather",
                                   "arguments": {"location": "San Francisco, CA",
                                                 "unit": "celsius"}}]
    assert meta["truncated"] is False


def test_openai_request_shape():
    _, body = _openai_adapter("m", CONV, {}, [SPEC])
    assert body["tools"] == [{"type": "function", "function": {
        "name": "search", "description": "Search the docs.",
        "parameters": SPEC["parameters"]}}]
    asst = body["messages"][2]
    assert asst["role"] == "assistant" and asst["content"] == "Let me look."
    assert asst["tool_calls"][0] == {"id": "call_1", "type": "function",
                                     "function": {"name": "search",
                                                  "arguments": '{"query": "Velor"}'}}
    assert body["messages"][3:] == [
        {"role": "tool", "tool_call_id": "call_1",
         "content": "[d1] The Velor rises in the Askern hills."},
        {"role": "tool", "tool_call_id": "call_2", "content": "error: bad query"}]


def test_openai_parse_keeps_unparseable_arguments_raw():
    js = {"choices": [{"finish_reason": "tool_calls", "message": {
        "role": "assistant", "content": None, "tool_calls": [
            {"id": "a", "type": "function",
             "function": {"name": "search", "arguments": '{"query": "Velor"}'}},
            {"id": "b", "type": "function",
             "function": {"name": "search", "arguments": '{"query": Velor'}}]}}]}
    text, meta = _openai_parse(js)
    assert text is None
    assert meta["tool_calls"][0]["arguments"] == {"query": "Velor"}
    assert meta["tool_calls"][1]["arguments"] == '{"query": Velor'


def test_gemini_request_shape_and_local_ids():
    _, body = _gemini_adapter("m", CONV, {}, [SPEC])
    assert body["tools"] == [{"functionDeclarations": [
        {"name": "search", "description": "Search the docs.",
         "parameters": SPEC["parameters"]}]}]
    model_turn, responses = body["contents"][1], body["contents"][2]
    assert model_turn["role"] == "model"
    assert model_turn["parts"][1] == {"functionCall": {
        "name": "search", "args": {"query": "Velor"}, "id": "call_1"}}
    assert responses["role"] == "user"
    assert responses["parts"] == [
        {"functionResponse": {"name": "search", "id": "call_1", "response":
                              {"result": "[d1] The Velor rises in the Askern hills."}}},
        {"functionResponse": {"name": "search", "id": "call_2",
                              "response": {"error": "error: bad query"}}}]
    # a call that arrived without an id gets a local one, never sent back
    _, meta = _gemini_parse({"candidates": [{"content": {"parts": [
        {"functionCall": {"name": "search", "args": {"query": "x"}}}]}}]})
    local = meta["tool_calls"][0]
    assert local["id"] == "gemini-0"
    _, body = _gemini_adapter("m", [
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": None, "tool_calls": [local]},
        {"role": "tool", "tool_call_id": "gemini-0", "name": "search",
         "content": "r"}], {})
    assert "id" not in body["contents"][1]["parts"][0]["functionCall"]
    assert "id" not in body["contents"][2]["parts"][0]["functionResponse"]


DOCS = {"d1": "The Velor river rises in the Askern hills."}


def scripted_server(flavour):
    """Turn 1: a tool call in the provider's own shape.  Turn 2: a final
    answer.  Records every request body."""
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        first = len(seen) == 1
        if flavour == "anthropic":
            js = {"stop_reason": "tool_use" if first else "end_turn",
                  "content": [{"type": "tool_use", "id": "toolu_1",
                               "name": "search", "input": {"query": "Velor"}}]
                  if first else [{"type": "text", "text": "In the Askern hills."}]}
        elif flavour == "openai":
            msg = ({"role": "assistant", "content": None, "tool_calls": [
                {"id": "call_1", "type": "function", "function": {
                    "name": "search", "arguments": '{"query": "Velor"}'}}]}
                if first else {"role": "assistant",
                               "content": "In the Askern hills."})
            js = {"choices": [{"message": msg, "finish_reason":
                               "tool_calls" if first else "stop"}]}
        else:
            part = ({"functionCall": {"name": "search", "args": {"query": "Velor"}}}
                    if first else {"text": "In the Askern hills."})
            js = {"candidates": [{"content": {"role": "model", "parts": [part]},
                                  "finishReason": "STOP"}]}
        return httpx.Response(200, json=js)
    return handler, seen


@pytest.mark.parametrize("flavour", ["anthropic", "openai", "gemini"])
def test_a_two_turn_episode_over_the_wire(flavour):
    handler, seen = scripted_server(flavour)
    http = HTTPBackend(flavour, api_key="k",
                       default_params={"model": "m", "max_tokens": 64})
    real = http.client
    http.client = httpx.Client(base_url=real.base_url, headers=real.headers,
                               transport=httpx.MockTransport(handler))
    real.close()
    agent = AgentBackend(http, [CorpusSearch(DOCS).tool])
    r = agent.complete(Request("u", [{"role": "user", "content": "Where?"}], {}))
    assert r.ok and r.text == "In the Askern hills."
    assert r.meta["n_tool_calls"] == 1 and len(seen) == 2
    second = json.dumps(seen[1])
    assert "The Velor river rises in the Askern hills." in second   # result sent
    assert "tools" in seen[1] and "tools" in seen[0]


def test_garbled_arguments_become_a_model_side_input_error():
    from evalcore import FunctionBackend, Response

    def policy(req):
        if not any(m.get("role") == "tool" for m in req.messages):
            return Response("x", meta={"tool_calls": [
                {"id": "a", "name": "search", "arguments": '{"query": Velor'}]})
        last = req.messages[-1]
        assert last["is_error"] and "must be an object" in last["content"]
        return Response("x", text="retrying failed", meta={"truncated": False})

    agent = AgentBackend(FunctionBackend(policy, identity={"p": "garble"},
                                         supports_tools=True),
                         [CorpusSearch(DOCS).tool])
    r = agent.complete(Request("u", [{"role": "user", "content": "q"}], {}))
    assert r.ok and r.meta["tool_input_errors"] == 1


def signature_enforcing_server():
    """Gemini as observed live on 2 Oct: a functionCall part carries a
    sibling `thoughtSignature`, and a follow-up request whose model turn
    lacks it is rejected with 400."""
    seen = []

    def handler(req):
        body = json.loads(req.content)
        seen.append(body)
        if len(seen) == 1:
            return httpx.Response(200, json={"candidates": [{"content": {
                "role": "model", "parts": [{
                    "functionCall": {"name": "search", "args": {"query": "Velor"},
                                     "id": "call_9"},
                    "thoughtSignature": "SIG-abc123"}]},
                "finishReason": "STOP"}]})
        model_parts = [p for c in body["contents"] if c["role"] == "model"
                       for p in c["parts"] if "functionCall" in p]
        if not all(p.get("thoughtSignature") == "SIG-abc123" for p in model_parts):
            return httpx.Response(400, json={"error": {
                "code": 400, "message": "Function call is missing a "
                "thought_signature in functionCall parts."}})
        return httpx.Response(200, json={"candidates": [{"content": {
            "role": "model", "parts": [{"text": "In the Askern hills."}]},
            "finishReason": "STOP"}]})
    return handler, seen


def test_gemini_thought_signatures_are_echoed_back():
    handler, _seen = signature_enforcing_server()
    http = HTTPBackend("gemini", api_key="k",
                       default_params={"model": "m", "max_tokens": 64})
    real = http.client
    http.client = httpx.Client(base_url=real.base_url, headers=real.headers,
                               transport=httpx.MockTransport(handler))
    real.close()
    r = AgentBackend(http, [CorpusSearch(DOCS).tool]).complete(
        Request("u", [{"role": "user", "content": "Where?"}], {}))
    assert r.ok, r.error
    assert r.text == "In the Askern hills."


# --- found in the first live run (2 Oct) ---------------------------------------

def _backend_with(handler):
    http = HTTPBackend("gemini", api_key="k",
                       default_params={"model": "m", "max_tokens": 8})
    real = http.client
    http.client = httpx.Client(base_url=real.base_url, headers=real.headers,
                               transport=httpx.MockTransport(handler))
    real.close()
    return http


def test_an_error_keeps_the_providers_message_and_details():
    long_msg = "You exceeded your current quota, please check your plan. " * 5
    detail = {"@type": "type.googleapis.com/google.rpc.QuotaFailure",
              "violations": [{"quotaMetric": "generate_requests_per_minute"}]}

    def handler(req):
        return httpx.Response(400, json={"error": {
            "code": 400, "status": "INVALID_ARGUMENT", "message": long_msg,
            "details": [detail]}})

    from evalcore import Fatal
    with pytest.raises(Fatal) as exc:
        _backend_with(handler).complete(Request("u", [{"role": "user",
                                                       "content": "q"}], {}))
    assert long_msg.strip() in str(exc.value)
    assert "generate_requests_per_minute" in str(exc.value)


@pytest.mark.parametrize("how", ["header", "retryinfo"])
def test_a_rate_limit_waits_as_long_as_the_server_asks(monkeypatch, how):
    from evalcore.core.backends import base
    slept = []
    monkeypatch.setattr(base.time, "sleep", slept.append)
    calls = []

    def handler(req):
        calls.append(1)
        if len(calls) == 1:
            if how == "header":
                return httpx.Response(429, headers={"retry-after": "37"},
                                      json={"error": {"code": 429}})
            return httpx.Response(429, json={"error": {"code": 429, "details": [
                {"@type": "type.googleapis.com/google.rpc.RetryInfo",
                 "retryDelay": "37s"}]}})
        return httpx.Response(200, json={"candidates": [{"content": {
            "parts": [{"text": "ok"}]}, "finishReason": "STOP"}]})

    from evalcore import with_retries
    r = with_retries(_backend_with(handler).complete,
                     Request("u", [{"role": "user", "content": "q"}], {}))
    assert r.ok and r.text == "ok"
    assert slept and slept[0] >= 37        # at least what the server asked


def test_a_server_asking_for_more_than_the_cap_fails_fast(monkeypatch):
    from evalcore import with_retries
    from evalcore.core.backends import base
    slept = []
    monkeypatch.setattr(base.time, "sleep", slept.append)

    def handler(req):
        return httpx.Response(429, headers={"retry-after": "3600"},
                              json={"error": {"code": 429, "message": "daily"}})

    r = with_retries(_backend_with(handler).complete,
                     Request("u", [{"role": "user", "content": "q"}], {}))
    assert not r.ok and "asks to wait 3600s" in r.error and slept == []


def test_openai_budget_is_sent_as_max_completion_tokens():
    _, body = _openai_adapter("m", [{"role": "user", "content": "q"}],
                              {"max_tokens": 64})
    assert body["max_completion_tokens"] == 64 and "max_tokens" not in body
