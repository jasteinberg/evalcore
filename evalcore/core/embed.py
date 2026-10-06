"""Text embeddings as model calls: identified, cached, and batched.

An `Embedder` maps texts to vectors.  Like a backend, it has an
`identity()` -- everything that changes the vectors -- and every vector is
cached under

    key = digest(embedder identity, role, text)

so re-embedding a corpus is free, adding documents embeds only the new
ones, and a changed embedder (model, revision, pooling, prefix, dimension)
misses exactly its own entries.  `role` is "query" or "document": many
embedding models encode the two differently (an instruction prefix, a task
type), and a query must not be served a document's vector.

    FunctionEmbedder   any fn(texts, role) -> (n, d) array, with an identity
    HFEmbedder         a local transformers encoder or decoder, pooled
    HTTPEmbedder       OpenAI or Gemini embedding endpoints

torch and httpx are imported inside the classes that need them, as in the
backends.
"""

from __future__ import annotations

import os
import threading
import uuid
import warnings
from abc import ABC, abstractmethod
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal

import numpy as np

from .backends.base import Fatal, Transient, _retrying
from .spec import digest

__all__ = ["Embedder", "EmbeddingCache", "FunctionEmbedder", "HFEmbedder",
           "HTTPEmbedder", "Role"]

Role = Literal["query", "document"]
ROLES = ("query", "document")


class EmbeddingCache:
    """Vectors on disk, one key per (embedder, role, text).

    Each `put` writes one .npz chunk holding its keys and vectors together,
    atomically (temp file, then rename), so a killed write leaves either a
    whole chunk or none -- never keys without vectors.  Every chunk is read
    at open; the store is append-only, and deleting the directory is the
    only way to forget.  Safe across threads of one process."""

    def __init__(self, root: str | Path) -> None:
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.rows: dict[str, np.ndarray] = {}
        self.hits = self.misses = 0
        for f in sorted(self.root.glob("*.npz")):
            with np.load(f) as z:
                self.rows.update(zip(z["keys"].tolist(), z["vectors"],
                                     strict=True))

    def get(self, keys: Sequence[str]) -> list[np.ndarray | None]:
        out = [self.rows.get(k) for k in keys]
        found = sum(v is not None for v in out)
        self.hits += found
        self.misses += len(out) - found
        return out

    def put(self, keys: Sequence[str], vectors: np.ndarray) -> None:
        if not len(keys):
            return
        name = uuid.uuid4().hex
        tmp = self.root / f"{name}.tmp.npz"
        np.savez(tmp, keys=np.array(keys), vectors=vectors)
        os.replace(tmp, self.root / f"{name}.npz")
        with self.lock:
            self.rows.update(zip(keys, vectors, strict=True))


class Embedder(ABC):
    """Subclasses define `identity()` and `_embed(texts, role)`; `embed`
    adds the cache, de-duplication and batching."""

    batch_size: int = 64

    @abstractmethod
    def identity(self) -> dict[str, Any]:
        """Everything that determines the vectors (as Backend.identity)."""

    @abstractmethod
    def _embed(self, texts: list[str], role: Role) -> np.ndarray:
        """(len(texts), d) for one batch, uncached."""

    def embed(self, texts: Sequence[str], role: Role = "document",
              cache: EmbeddingCache | None = None) -> np.ndarray:
        """(len(texts), d) float32, in the order given.  Each distinct text
        not in the cache is embedded once, in batches of `batch_size`."""
        if role not in ROLES:
            raise ValueError(f"role must be one of {ROLES}, got {role!r}")
        texts = list(texts)
        if not texts:
            raise ValueError("no texts to embed")
        ident = self.identity()              # once, not per text
        keys = [digest({"embedder": ident, "role": role, "text": t})
                for t in texts]
        found: dict[str, np.ndarray] = {}
        if cache is not None:
            uniq = list(dict.fromkeys(keys))
            found = {k: v for k, v in zip(uniq, cache.get(uniq), strict=True)
                     if v is not None}
        todo = list(dict.fromkeys((k, t) for k, t in zip(keys, texts, strict=True)
                                  if k not in found))
        for a in range(0, len(todo), self.batch_size):
            part = todo[a:a + self.batch_size]
            vecs = np.asarray(self._embed([t for _, t in part], role), np.float32)
            if vecs.shape[0] != len(part) or vecs.ndim != 2:
                raise ValueError(f"{type(self).__name__} returned shape "
                                 f"{vecs.shape} for {len(part)} texts")
            if cache is not None:
                cache.put([k for k, _ in part], vecs)
            found.update(zip((k for k, _ in part), vecs, strict=True))
        return np.stack([found[k] for k in keys]).astype(np.float32, copy=False)


class FunctionEmbedder(Embedder):
    """`fn(texts, role) -> (n, d)`.  `identity` is required, as for
    FunctionBackend: a function's name says nothing about its weights."""

    def __init__(self, fn: Callable[[list[str], Role], Any],
                 identity: Mapping[str, Any], batch_size: int = 64) -> None:
        self.fn, self._identity, self.batch_size = fn, dict(identity), batch_size

    def identity(self) -> dict[str, Any]:
        return {"embedder": "function", **self._identity}

    def _embed(self, texts: list[str], role: Role) -> np.ndarray:
        return np.asarray(self.fn(texts, role))


class HFEmbedder(Embedder):
    """A transformers model's final hidden states, pooled.

    pooling  "mean"  average over real tokens (attention mask) -- the usual
                     choice for sentence encoders
             "cls"   the first position (BERT-style encoders)
             "last"  the last real token (decoder-only embedding models,
                     which attend only leftwards, so only the last position
                     has seen the whole text)
    prefixes {"query": ..., "document": ...}: an instruction some models are
             trained with ("query: " / "passage: "); part of the identity.

    Texts longer than `max_length` tokens are truncated, counted in
    `n_truncated`, and warned about: the vector then describes a prefix."""

    def __init__(self, model: Any, tokenizer: Any, pooling: str = "mean",
                 prefixes: Mapping[str, str] | None = None,
                 max_length: int = 512, device: Any = None,
                 batch_size: int = 32) -> None:
        if pooling not in ("mean", "cls", "last"):
            raise ValueError("pooling must be mean, cls or last")
        self.model, self.tok, self.pooling = model, tokenizer, pooling
        self.prefixes = {"query": "", "document": "", **(prefixes or {})}
        self.max_length, self.batch_size = max_length, batch_size
        self.device = device
        self.n_truncated = 0
        if getattr(self.tok, "pad_token", None) is None:
            self.tok.pad_token = self.tok.eos_token

    def identity(self) -> dict[str, Any]:
        cfg = getattr(self.model, "config", None)
        return {"embedder": "hf",
                "model": getattr(self.model, "name_or_path", None),
                "commit": getattr(cfg, "_commit_hash", None),
                "pooling": self.pooling, "prefixes": self.prefixes,
                "max_length": self.max_length}

    def _embed(self, texts: list[str], role: Role) -> np.ndarray:
        import torch
        dev = self.device or next(self.model.parameters()).device
        full = [self.prefixes[role] + t for t in texts]
        n_tok = [len(self.tok(t)["input_ids"]) for t in full]
        cut = sum(n > self.max_length for n in n_tok)
        if cut:
            self.n_truncated += cut
            warnings.warn(f"{cut} text(s) longer than max_length="
                          f"{self.max_length} tokens were truncated",
                          RuntimeWarning, stacklevel=3)
        enc = self.tok(full, padding=True, truncation=True,
                       max_length=self.max_length, return_tensors="pt")
        enc = {k: v.to(dev) for k, v in enc.items()}
        with torch.no_grad():
            out = self.model(**enc, output_hidden_states=True)
        h = out.hidden_states[-1]                          # (n, T, d)
        mask = enc["attention_mask"]
        if self.pooling == "mean":
            m = mask.unsqueeze(-1).to(h.dtype)
            v = (h * m).sum(1) / m.sum(1).clamp(min=1)
        elif self.pooling == "cls":
            v = h[:, 0]
        else:
            # last real position per row, whichever side the padding is on
            pos = torch.arange(mask.shape[1], device=mask.device)
            last = (pos * mask).argmax(1)
            v = h[torch.arange(len(h)), last]
        return v.float().cpu().numpy()


class HTTPEmbedder(Embedder):
    """OpenAI (`/v1/embeddings`) or Gemini (`:batchEmbedContents`).

    Written against the providers' API references of 6 Oct 2026 and
    checked live the same day.  OpenAI takes up to 2048 inputs and 300k
    tokens per request and an optional `dimensions` (text-embedding-3 and
    later); it has no query/document distinction, so both roles get the
    same vector.  Gemini maps the role to taskType RETRIEVAL_QUERY /
    RETRIEVAL_DOCUMENT.  The reference marks the per-request `taskType`
    and `outputDimensionality` deprecated in favour of `embedContentConfig`,
    but live, batchEmbedContents parses embedContentConfig (an unknown key
    in it is a 400) and IGNORES it: 3072 dimensions back for 256 asked,
    and identical query and document vectors.  The top-level fields work,
    so they are sent.  A returned dimension other than the one asked for
    raises, so a setting silently dropped again cannot pass unnoticed.
    Rate limits and 5xx are retried with the backends' policy (honouring
    Retry-After); anything else raises Fatal."""

    ENV = MappingProxyType({"openai": "OPENAI_API_KEY",
                            "gemini": "GEMINI_API_KEY"})
    BASES = MappingProxyType({"openai": "https://api.openai.com",
                              "gemini": "https://generativelanguage.googleapis.com"})

    def __init__(self, flavour: str, model: str, dimensions: int | None = None,
                 api_key: str | None = None, base_url: str | None = None,
                 timeout: float = 60.0, batch_size: int = 100,
                 attempts: int = 5) -> None:
        import httpx
        if flavour not in self.ENV:
            raise ValueError(f"flavour must be one of {sorted(self.ENV)}")
        key = api_key or os.environ.get(self.ENV[flavour])
        if not key:
            raise Fatal(f"no API key: set ${self.ENV[flavour]}")
        headers = ({"authorization": f"Bearer {key}"} if flavour == "openai"
                   else {"x-goog-api-key": key})
        self.flavour, self.model, self.dimensions = flavour, model, dimensions
        self.base_url = base_url or self.BASES[flavour]
        self.batch_size, self.attempts = batch_size, attempts
        self.in_tokens = 0
        self.client = httpx.Client(base_url=self.base_url, timeout=timeout,
                                   headers={"content-type": "application/json",
                                            **headers})

    def identity(self) -> dict[str, Any]:
        return {"embedder": "http", "flavour": self.flavour, "model": self.model,
                "dimensions": self.dimensions, "base_url": self.base_url}

    def _request(self, texts: list[str], role: Role
                 ) -> tuple[str, dict[str, Any]]:
        if self.flavour == "openai":
            body: dict[str, Any] = {"model": self.model, "input": texts,
                                    "encoding_format": "float"}
            if self.dimensions:
                body["dimensions"] = self.dimensions
            return "/v1/embeddings", body
        cfg: dict[str, Any] = {"taskType": "RETRIEVAL_QUERY" if role == "query"
                               else "RETRIEVAL_DOCUMENT"}
        if self.dimensions:
            cfg["outputDimensionality"] = self.dimensions
        name = f"models/{self.model}"
        # top level, not embedContentConfig: see the class docstring
        return (f"/v1beta/{name}:batchEmbedContents",
                {"requests": [{"model": name, "content": {"parts": [{"text": t}]},
                               **cfg} for t in texts]})

    def _post(self, texts: list[str], role: Role) -> np.ndarray:
        import httpx

        from .backends.http import _error_text, _retry_after
        path, body = self._request(texts, role)
        try:
            r = self.client.post(path, json=body)
        except httpx.HTTPError as exc:
            raise Transient(f"transport: {exc}") from exc
        if r.status_code in (408, 409, 429) or r.status_code >= 500:
            raise Transient(_error_text(r), retry_after=_retry_after(r))
        if r.status_code >= 400:
            raise Fatal(_error_text(r))
        js = r.json()
        if self.flavour == "openai":
            data = sorted(js["data"], key=lambda e: e["index"])
            self.in_tokens += int(js.get("usage", {}).get("prompt_tokens", 0))
            out = np.array([e["embedding"] for e in data], np.float32)
        else:
            self.in_tokens += int(js.get("usageMetadata", {})
                                  .get("promptTokenCount", 0))
            out = np.array([e["values"] for e in js["embeddings"]], np.float32)
        if self.dimensions and out.shape[1] != self.dimensions:
            raise Fatal(f"asked for {self.dimensions} dimensions, got "
                        f"{out.shape[1]}: the request's settings were ignored")
        return out

    def _embed(self, texts: list[str], role: Role) -> np.ndarray:
        failed: list[str] = []
        out = _retrying(lambda: self._post(texts, role), failed.append,
                        seed=texts[0], attempts=self.attempts, base=1.0,
                        cap=30.0)
        if failed:
            # an embedding failure has no row to become: the corpus or the
            # query set is incomplete, so stop rather than index a hole
            raise RuntimeError(f"embedding failed: {failed[0]}")
        return out

    def close(self) -> None:
        self.client.close()
