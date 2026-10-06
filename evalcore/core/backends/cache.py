"""On-disk response cache, content-addressed by request_key."""

from __future__ import annotations

import json
import os
import uuid
from pathlib import Path


class Cache:
    """Content-addressed on-disk cache, one JSON file per key under a
    two-character shard directory.  The runner keys it on `request_key`:
    the request as actually made (messages, params, backend identity and
    defaults, repeat index), so the same call is never paid for twice across
    runs, dry runs, or crashes -- which is what makes a long unattended sweep
    survivable -- and a different call is never served this one's answer.
    It holds responses only, never scores, so a metric change is a re-score.
    Delete the directory to force a refresh; never mutate it."""

    def __init__(self, root: str | Path, enabled: bool = True) -> None:
        self.root = Path(root)
        self.enabled = enabled
        self.hits = self.misses = 0
        if enabled:
            self.root.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        return self.root / key[:2] / f"{key}.json"

    def has(self, key: str) -> bool:
        """Whether `key` is cached, without counting a hit or a miss (for a
        dry run's 'already paid for')."""
        return self.enabled and self._path(key).exists()

    def get(self, key: str) -> dict | None:
        if not self.enabled:
            return None
        p = self._path(key)
        if not p.exists():
            self.misses += 1
            return None
        try:
            out = json.loads(p.read_text())
            self.hits += 1
            return out
        except json.JSONDecodeError:
            self.misses += 1
            return None

    def put(self, key: str, payload: dict) -> None:
        if not self.enabled:
            return
        p = self._path(key)
        p.parent.mkdir(parents=True, exist_ok=True)
        # The temp name must be unique per writer, not per key.  Two workers
        # can hold the same key at once -- any design with repeats > 1 at a
        # non-zero temperature produces identical requests by construction --
        # and with a shared "<key>.tmp" the second os.replace raises
        # FileNotFoundError because the first already moved the file out from
        # under it.  Unique temp, last writer wins, still atomic.
        tmp = p.with_suffix(f".{uuid.uuid4().hex}.tmp")
        tmp.write_text(json.dumps(payload))
        os.replace(tmp, p)   # atomic: no half-written cache entry
