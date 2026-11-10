"""Embedders: the cache keys exactly what changes a vector, batching never
changes one, and the HTTP adapters speak each provider's format."""

from __future__ import annotations

import json
import types

import numpy as np
import pytest

from evalcore.core.embed import (
    EmbeddingCache,
    FunctionEmbedder,
    HFEmbedder,
    HTTPEmbedder,
)

httpx = pytest.importorskip("httpx")


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


def test_the_cache_survives_a_killed_write(tmp_path):
    """B36.  One complete chunk (key "k", vector [1, 2]) and the half-written
    temp file a kill inside `np.savez` leaves.  Reopening must give exactly
    the complete chunk's row: the temp is not a committed chunk."""
    root = tmp_path / "emb"
    EmbeddingCache(root).put(["k"], np.array([[1.0, 2.0]], np.float32))
    np.savez(root / "whole.npz", keys=np.array(["j"]),
             vectors=np.zeros((1, 2), np.float32))
    blob = (root / "whole.npz").read_bytes()
    (root / "whole.npz").unlink()
    (root / "dead.tmp.npz").write_bytes(blob[:len(blob) // 2])
    rows = EmbeddingCache(root).rows
    assert list(rows) == ["k"]
    assert np.array_equal(rows["k"], np.array([1.0, 2.0], np.float32))


# --- HF ---------------------------------------------------------------------------

def toy_hf():
    """(torch, MeanEncoder, EmbedTok), skipping where torch is absent.  Per
    test rather than per module, so the HTTP tests below still run there."""
    torch = pytest.importorskip("torch")
    from support.toy_hf import EmbedTok, MeanEncoder
    return torch, MeanEncoder, EmbedTok


@pytest.mark.parametrize("pooling", ["mean", "cls", "last"])
@pytest.mark.parametrize("side", ["right", "left"])
def test_hf_pooling_is_the_same_batched_as_alone(pooling, side):
    _, MeanEncoder, EmbedTok = toy_hf()
    emb = HFEmbedder(MeanEncoder(), EmbedTok(side), pooling=pooling, device="cpu")
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
    _, MeanEncoder, EmbedTok = toy_hf()
    emb = HFEmbedder(MeanEncoder(), EmbedTok(), max_length=3, device="cpu")
    with pytest.warns(RuntimeWarning, match="1 text"):
        emb._embed(["ab", "abcdef"], "query")
    assert emb.n_truncated == 1


def test_hf_prefixes_and_commit_are_in_the_identity():
    """B26 (commit).  Two checkpoints of one model name (commits "a" and
    "b") hold different weights, so their identities must differ."""
    _, MeanEncoder, EmbedTok = toy_hf()
    a = HFEmbedder(MeanEncoder(), EmbedTok(), prefixes={"query": "query: "})
    b = HFEmbedder(MeanEncoder(), EmbedTok())
    assert a.identity() != b.identity()
    assert a.identity()["prefixes"]["document"] == ""
    ca, cb = MeanEncoder(), MeanEncoder()
    ca.config = types.SimpleNamespace(_commit_hash="a")
    cb.config = types.SimpleNamespace(_commit_hash="b")
    assert (HFEmbedder(ca, EmbedTok(), device="cpu").identity()
            != HFEmbedder(cb, EmbedTok(), device="cpu").identity())


def test_hf_identity_covers_the_dtype():
    """B26.  The same weights in float32 and in bfloat16 give different
    vectors (bfloat16 keeps 8 mantissa bits, and the premise is asserted), so
    the two embedders must have different identities -- else they share
    cache entries."""
    torch, MeanEncoder, EmbedTok = toy_hf()
    f32 = HFEmbedder(MeanEncoder(), EmbedTok(), device="cpu")
    b16 = HFEmbedder(MeanEncoder().to(torch.bfloat16), EmbedTok(), device="cpu")
    assert not np.array_equal(f32._embed(["hello"], "document"),
                              b16._embed(["hello"], "document"))
    assert f32.identity() != b16.identity()


# --- HTTP -----------------------------------------------------------------------------

def http(flavour, handler, **kw):
    """An HTTPEmbedder whose transport records each request and answers
    with `handler(request)`."""
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
