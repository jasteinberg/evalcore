r"""Two-stage retrieval: any retriever's top `depth`, rescored by a
PairScorer (a function, or an HF cross-encoder).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any, Protocol

import numpy as np

from ..embed import _checkpoint, _on_device
from .base import Hit, Retriever, _uses_repeat


class PairScorer(Protocol):
    """Relevance of each document to one query, higher is better."""

    def identity(self) -> dict[str, Any]: ...
    def score(self, query: str, docs: Sequence[str]) -> np.ndarray: ...


class FunctionScorer:
    """`fn(query, docs) -> scores`, with a required identity (as for
    FunctionEmbedder)."""

    def __init__(self, fn: Any, identity: Mapping[str, Any]) -> None:
        self.fn, self._identity = fn, dict(identity)

    def identity(self) -> dict[str, Any]:
        return {"scorer": "function", **self._identity}

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        return np.asarray(self.fn(query, list(docs)), float)


class HFCrossEncoder:
    """A sequence-classification model reading (query, document) as one
    pair, with a single relevance logit (`num_labels == 1`, as MS MARCO
    cross-encoders and bge-rerankers are).  A model with more labels is
    refused rather than guessing which logit means "relevant".  Pairs
    longer than `max_length` tokens lose the end of the document."""

    def __init__(self, model: Any, tokenizer: Any, max_length: int = 512,
                 batch_size: int = 32, device: Any = None) -> None:
        n = getattr(getattr(model, "config", None), "num_labels", 1)
        if n != 1:
            raise ValueError(f"cross-encoder must have one output logit, "
                             f"this one has num_labels={n}")
        self.model, self.tok = model, tokenizer
        self.max_length, self.batch_size, self.device = max_length, batch_size, device

    def identity(self) -> dict[str, Any]:
        return {"scorer": "hf-cross-encoder", **_checkpoint(self.model),
                "max_length": self.max_length}

    def score(self, query: str, docs: Sequence[str]) -> np.ndarray:
        import torch
        out = []
        for a in range(0, len(docs), self.batch_size):
            part = list(docs[a:a + self.batch_size])
            enc = _on_device(self.model, self.device, self.tok(
                [query] * len(part), part, padding=True,
                truncation="only_second", max_length=self.max_length,
                return_tensors="pt"))
            with torch.no_grad():
                logits = self.model(**enc).logits
            out.append(logits.reshape(len(part)).float().cpu().numpy())
        return np.concatenate(out) if out else np.zeros(0)


class Reranker:
    """Two stages: `first` retrieves `depth` candidates, `scorer` rescores
    them against the query, and the top k by (score desc, id asc) are
    returned.  A document the first stage misses at `depth` cannot be
    recovered, so recall@depth of the first stage bounds the reranker's
    recall at any k.  Evaluate the first stage at `depth` to see that
    ceiling."""

    def __init__(self, first: Retriever, scorer: PairScorer,
                 docs: Mapping[str, str], depth: int) -> None:
        self.first, self.scorer, self.depth = first, scorer, depth
        self.docs = dict(docs)

    @property
    def uses_repeat(self) -> bool:
        """Whether the first stage's draw depends on the repeat index: a
        reranked sampled draw is still a random draw."""
        return _uses_repeat(self.first)

    def identity(self) -> dict[str, Any]:
        return {"retriever": "rerank", "first": self.first.identity(),
                "scorer": self.scorer.identity(), "depth": self.depth}

    def retrieve(self, queries: Sequence[str], k: int,
                 repeat: int = 0) -> list[list[Hit]]:
        out: list[list[Hit]] = []
        first = self.first.retrieve(queries, self.depth, repeat=repeat)
        for q, cands in zip(queries, first, strict=True):
            ids = [d for d, _ in cands]
            if not ids:
                out.append([])
                continue
            sc = self.scorer.score(q, [self.docs[d] for d in ids])
            order = sorted(range(len(ids)), key=lambda i: (-sc[i], ids[i]))[:k]
            out.append([(ids[i], float(sc[i])) for i in order])
        return out
