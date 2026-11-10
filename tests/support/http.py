"""HTTPBackend against a scripted server: httpx.MockTransport serves every
call, so nothing touches the network.  Import after
`pytest.importorskip("httpx")`."""

from __future__ import annotations

from collections.abc import Callable

import httpx

from evalcore import HTTPBackend

# A minimal successful reply in each provider's own shape, text "hi".
OK_REPLY = {
    "anthropic": {"content": [{"type": "text", "text": "hi"}],
                  "stop_reason": "end_turn", "usage": {}},
    "openai": {"choices": [{"finish_reason": "stop",
                            "message": {"role": "assistant", "content": "hi"}}]},
    "gemini": {"candidates": [{"content": {"role": "model",
                                           "parts": [{"text": "hi"}]},
                               "finishReason": "STOP"}]},
}


def mock_http(flavour: str,
              handler: Callable[[httpx.Request], httpx.Response],
              default_params: dict | None = None, api_key: str = "k",
              ) -> tuple[HTTPBackend, list[httpx.Request]]:
    """An HTTPBackend whose client sends every request to `handler`, and the
    list of requests it sent (in order).  `default_params` defaults to model
    "m" with an 8-token budget."""
    seen: list[httpx.Request] = []

    def record(req: httpx.Request) -> httpx.Response:
        seen.append(req)
        return handler(req)

    be = HTTPBackend(flavour, api_key=api_key, default_params=(
        {"model": "m", "max_tokens": 8} if default_params is None
        else default_params))
    real = be.client
    be.client = httpx.Client(base_url=real.base_url, headers=real.headers,
                             transport=httpx.MockTransport(record))
    real.close()
    # a client reopened after close() must reach the same scripted server
    be._client_args = {**be._client_args,
                       "transport": httpx.MockTransport(record)}
    return be, seen
