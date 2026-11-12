"""VLLMBackend against a scripted vLLM server (httpx.MockTransport).

The server's answers follow vLLM's OpenAI-compatible API: `GET /v1/models`
lists each served model with the weights it loaded (`root`), the base of
an adapter (`parent`) and `max_model_len`; `POST /v1/chat/completions`
takes OpenAI's body plus vLLM's own sampling params.  Expected values are
fixed by the script, never read back from the code under test."""

from __future__ import annotations

import json
from typing import Any

import pytest

from evalcore import ConfigError, Fatal, HTTPBackend, Request, VLLMBackend, from_config
from evalcore.core.backends import request_key

httpx = pytest.importorskip("httpx")

Q = [{"role": "user", "content": "q"}]
REPLY = {"choices": [{"finish_reason": "stop",
                      "message": {"role": "assistant", "content": "hi"}}],
         "usage": {"prompt_tokens": 3, "completion_tokens": 1}}


def served(root: str = "/models/qwen-7b", name: str = "qwen") -> dict:
    return {"object": "list", "data": [
        {"id": name, "object": "model", "root": root, "parent": None,
         "max_model_len": 4096}]}


def server(url: str = "http://10.0.0.7:8000", models: dict | None = None,
           revision: str | None = "abc123", api_key: str | None = None,
           ) -> tuple[VLLMBackend, list[Any]]:
    """A VLLMBackend whose client reaches a scripted server, and the list
    of requests it received."""
    seen: list[Any] = []
    models = served() if models is None else models

    def handle(req: Any) -> Any:
        seen.append(req)
        if req.url.path == "/v1/models":
            return httpx.Response(200, json=models)
        return httpx.Response(200, json=REPLY)

    be = VLLMBackend(url, revision=revision, api_key=api_key,
                     default_params={"model": "qwen", "max_tokens": 8})
    be._client_args = {**be._client_args,
                       "transport": httpx.MockTransport(handle)}
    be.client.close()
    be.client = httpx.Client(**be._client_args)
    return be, seen


def key(be: VLLMBackend) -> str:
    return request_key(Request("u", Q, {}), 0, be)


def test_the_key_follows_the_weights_not_the_address():
    """A relaunched server has a new address and the same weights: same
    key, so the cache survives.  Other weights or another declared
    revision: another key."""
    a, _ = server("http://10.0.0.7:8000")
    b, _ = server("http://node-12.cluster:9000")
    c, _ = server(models=served(root="/models/qwen-7b-ft"))
    d, _ = server(revision="def456")
    assert key(a) == key(b)
    assert len({key(a), key(c), key(d)}) == 3


def test_the_server_is_asked_once_for_its_models():
    be, seen = server()
    be.identity()
    be.identity()
    be.complete(Request("u", Q, {}))
    assert [r.url.path for r in seen] == ["/v1/models",
                                          "/v1/chat/completions"]


def test_a_model_the_server_does_not_serve_is_refused_before_a_call():
    be, seen = server()
    why = be.refusal(Request("u", Q, {"model": "llama"}))
    assert why is not None and "llama" in why and "qwen" in why
    assert be.refusal(Request("u", Q, {})) is None
    assert [r.url.path for r in seen] == ["/v1/models"]


def test_vllm_sampling_params_are_sent_and_the_budget_is_max_tokens():
    be, seen = server()
    r = be.complete(Request("u", Q, {"top_k": 20, "min_p": 0.05,
                                     "repetition_penalty": 1.1, "seed": 3}))
    body = json.loads(seen[-1].content)
    assert (body["top_k"], body["min_p"], body["repetition_penalty"],
            body["seed"], body["max_tokens"]) == (20, 0.05, 1.1, 3, 8)
    assert "max_completion_tokens" not in body
    assert r.text == "hi" and r.meta["in_tokens"] == 3


def test_an_unread_param_is_refused():
    be, _ = server()
    with pytest.raises(Fatal, match="frequency_typo"):
        be.complete(Request("u", Q, {"frequency_typo": 1}))


@pytest.mark.parametrize("api_key,env,want", [
    (None, None, None),
    ("s3cret", None, "Bearer s3cret"),
    (None, "from-env", "Bearer from-env")])
def test_a_key_is_sent_only_when_there_is_one(monkeypatch, api_key, env, want):
    """A server started without --api-key reads none, so none is needed."""
    if env is not None:
        monkeypatch.setenv("VLLM_API_KEY", env)
    be, seen = server(api_key=api_key)
    be.complete(Request("u", Q, {}))
    assert seen[-1].headers.get("authorization") == want


def test_an_unreachable_server_is_refused_at_identity():
    def down(req: Any) -> Any:
        raise httpx.ConnectError("connection refused")
    be = VLLMBackend("http://10.0.0.7:8000", revision=None)
    be.client.close()
    be.client = httpx.Client(base_url=be.base_url,
                             transport=httpx.MockTransport(down))
    with pytest.raises(Fatal, match="unreachable"):
        be.identity()


def test_a_server_with_no_model_is_refused():
    be, _ = server(models={"object": "list", "data": []})
    with pytest.raises(Fatal, match="serves no model"):
        be.identity()


def test_the_vllm_flavour_is_only_vllm_backends():
    """HTTPBackend keys on the URL, which is what VLLMBackend avoids."""
    with pytest.raises(ValueError, match="VLLMBackend"):
        HTTPBackend("vllm", base_url="http://x:8000")


def test_a_config_must_state_the_revision(tmp_path):
    """`revision` has no default: a config that leaves it out is refused
    by name before any server is contacted."""
    tasks = tmp_path / "t.jsonl"
    tasks.write_text(json.dumps({"id": "t0", "prompt": "say 1", "gold": 1,
                                 "grader": "numeric"}))
    cfg = {"out": str(tmp_path / "r.jsonl"),
           "task": {"type": "file", "path": str(tasks)},
           "backend": {"type": "vllm", "base_url": "http://x:8000",
                       "default_params": {"model": "qwen"}},
           "grid": {"model": ["qwen"]}, "metric": "correct"}
    with pytest.raises(ConfigError, match="revision"):
        from_config(cfg)
