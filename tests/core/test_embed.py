"""Embedders: the cache keys exactly what changes a vector, batching never
changes one, and the HTTP adapters speak each provider's format."""

from __future__ import annotations

import json

import numpy as np
import pytest

from evalcore.core.embed import EmbeddingCache, FunctionEmbedder, HTTPEmbedder


def counting(dim=4, ident=None):
    calls = []

    def fn(texts, role):
        calls.append((role, list(texts)))
        return np.array([[len(t), ord(t[0]), role == "query", 1.0][:dim]
                         for t in texts], float)
    return FunctionEmbedder(fn, ident or {"model": "toy"}, batch_size=2), calls


def test_each_distinct_text_is_embedded_once_in_order(tmp_path):
    emb, calls = counting()
    v = emb.embed(["bb", "a", "bb", "ccc"], cache=EmbeddingCache(tmp_path))
    assert v.dtype == np.float32 and v.shape == (4, 4)
    assert v[:, 0].tolist() == [2, 1, 2, 3]
    assert [t for _, b in calls for t in b] == ["bb", "a", "ccc"]  # batches of 2


def test_the_cache_survives_a_reopen_and_embeds_only_new_texts(tmp_path):
    emb, calls = counting()
    first = emb.embed(["a", "bb"], cache=EmbeddingCache(tmp_path))
    calls.clear()
    again = emb.embed(["bb", "a", "new"], cache=EmbeddingCache(tmp_path))
    assert calls == [("document", ["new"])]
    assert np.array_equal(again[:2], first[::-1])


def test_role_and_identity_are_part_of_the_key(tmp_path):
    cache = EmbeddingCache(tmp_path)
    emb, calls = counting()
    emb.embed(["a"], "document", cache)
    emb.embed(["a"], "query", cache)
    other, other_calls = counting(ident={"model": "toy", "revision": "2"})
    other.embed(["a"], "document", cache)
    assert len(calls) == 2 and len(other_calls) == 1


def test_a_wrong_shape_from_the_model_is_refused():
    emb = FunctionEmbedder(lambda texts, role: np.zeros((1, 3)), {"m": 1})
    with pytest.raises(ValueError, match="shape"):
        emb.embed(["a", "b"])


# --- HF ---------------------------------------------------------------------------

torch = pytest.importorskip("torch")


class Tok:
    """Characters as tokens; pads on `side`, returns torch tensors."""

    def __init__(self, side="right"):
        self.padding_side, self.pad_token, self.eos_token = side, "#", "#"

    def __call__(self, texts, padding=False, truncation=False, max_length=None,
                 return_tensors=None):
        if isinstance(texts, str):
            return {"input_ids": [ord(c) % 50 + 1 for c in texts]}
        ids = [[ord(c) % 50 + 1 for c in t] for t in texts]
        if truncation:
            ids = [x[:max_length] for x in ids]
        T = max(map(len, ids))
        pad = [[0] * (T - len(x)) for x in ids]
        rows = [x + p if self.padding_side == "right" else p + x
                for x, p in zip(ids, pad, strict=True)]
        mask = [[int(t != 0) for t in r] for r in rows]
        return {"input_ids": torch.tensor(rows),
                "attention_mask": torch.tensor(mask)}


class Enc(torch.nn.Module):
    """A causal toy: h_t is the masked running mean of the embeddings up to
    t, so a padded position cannot change a real one."""

    def __init__(self):
        super().__init__()
        g = torch.Generator().manual_seed(0)
        self.emb = torch.nn.Parameter(torch.randn(64, 8, generator=g))

    def forward(self, input_ids, attention_mask, output_hidden_states=False):
        m = attention_mask.unsqueeze(-1).float()
        e = self.emb[input_ids] * m
        h = e.cumsum(1) / m.cumsum(1).clamp(min=1)
        return type("Out", (), {"hidden_states": (e, h)})()


@pytest.mark.parametrize("pooling", ["mean", "cls", "last"])
@pytest.mark.parametrize("side", ["right", "left"])
def test_hf_pooling_is_the_same_batched_as_alone(pooling, side):
    from evalcore.core.embed import HFEmbedder
    emb = HFEmbedder(Enc(), Tok(side), pooling=pooling, device="cpu")
    texts = ["a", "hello there", "abc"]
    batch = emb._embed(texts, "document")
    alone = np.vstack([emb._embed([t], "document") for t in texts])
    if pooling == "cls" and side == "left":
        # cls reads position 0, which left padding fills: a real mismatch,
        # and the reason cls is for right-padded encoders
        assert not np.allclose(batch, alone, atol=1e-6)
    else:
        np.testing.assert_allclose(batch, alone, atol=1e-6)


def test_hf_truncation_is_counted_and_warned():
    from evalcore.core.embed import HFEmbedder
    emb = HFEmbedder(Enc(), Tok(), max_length=3, device="cpu")
    with pytest.warns(RuntimeWarning, match="1 text"):
        emb._embed(["ab", "abcdef"], "query")
    assert emb.n_truncated == 1


def test_hf_prefixes_and_commit_are_in_the_identity():
    from evalcore.core.embed import HFEmbedder
    a = HFEmbedder(Enc(), Tok(), prefixes={"query": "query: "})
    b = HFEmbedder(Enc(), Tok())
    assert a.identity() != b.identity()
    assert a.identity()["prefixes"]["document"] == ""


# --- HTTP -----------------------------------------------------------------------------

httpx = pytest.importorskip("httpx")


def http(flavour, handler, **kw):
    seen = []

    def record(req):
        seen.append(req)
        return handler(req)
    emb = HTTPEmbedder(flavour, "m", api_key="k", **kw)
    emb.client = httpx.Client(base_url=emb.base_url, headers=emb.client.headers,
                              transport=httpx.MockTransport(record))
    return emb, seen


def test_openai_request_and_out_of_order_data():
    emb, seen = http("openai", lambda r: httpx.Response(200, json={
        "data": [{"index": 1, "embedding": [0.0, 1.0]},
                 {"index": 0, "embedding": [1.0, 0.0]}],
        "usage": {"prompt_tokens": 7, "total_tokens": 7}}), dimensions=2)
    v = emb.embed(["x", "y"], "query")
    body = json.loads(seen[0].content)
    assert seen[0].url.path == "/v1/embeddings"
    assert body == {"model": "m", "input": ["x", "y"], "encoding_format": "float",
                    "dimensions": 2}
    assert v.tolist() == [[1.0, 0.0], [0.0, 1.0]]       # sorted by index
    assert emb.in_tokens == 7


def test_gemini_request_carries_the_role_as_task_type():
    emb, seen = http("gemini", lambda r: httpx.Response(200, json={
        "embeddings": [{"values": [1.0, 2.0]}],
        "usageMetadata": {"promptTokenCount": 3}}))
    emb.embed(["x"], "query")
    emb.embed(["x"], "document")
    q, d = (json.loads(r.content) for r in seen)
    assert seen[0].url.path == "/v1beta/models/m:batchEmbedContents"
    # top-level fields: embedContentConfig is ignored live (embed.py)
    assert q["requests"][0] == {"model": "models/m",
                                "content": {"parts": [{"text": "x"}]},
                                "taskType": "RETRIEVAL_QUERY"}
    assert d["requests"][0]["taskType"] == "RETRIEVAL_DOCUMENT"


def test_a_dimension_the_server_ignored_is_refused():
    emb, _ = http("gemini", lambda r: httpx.Response(200, json={
        "embeddings": [{"values": [0.0] * 8}]}), dimensions=4)
    with pytest.raises(RuntimeError, match="asked for 4 dimensions, got 8"):
        emb.embed(["x"])


def test_a_rate_limit_is_retried_and_a_bad_request_stops():
    replies = iter([httpx.Response(429, headers={"retry-after": "0"},
                                   json={"error": {"message": "slow"}}),
                    httpx.Response(200, json={"data": [{"index": 0,
                                                        "embedding": [1.0]}]})])
    emb, seen = http("openai", lambda r: next(replies))
    assert emb.embed(["x"]).tolist() == [[1.0]] and len(seen) == 2
    bad, _ = http("openai", lambda r: httpx.Response(
        400, json={"error": {"message": "bad model"}}))
    with pytest.raises(RuntimeError, match="bad model"):
        bad.embed(["x"])
