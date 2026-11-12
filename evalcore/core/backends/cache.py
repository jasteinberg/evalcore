"""On-disk response cache, content-addressed by request_key."""

from __future__ import annotations

import json
from pathlib import Path

from ..records import write_atomic


class Cache:
    """Content-addressed on-disk cache, one JSON file per key under a
    two-character shard directory.  The runner keys it on `request_key`:
    the request as actually made (messages, params, backend identity and
    defaults, repeat index).  So the same call is never paid for twice
    across runs, dry runs or crashes, which is what lets a long unattended
    sweep survive a crash, and a different call is never served this one's
    answer.  It holds responses only, never scores, so a metric change is a
    re-score.  Delete the directory to force a refresh.  Never edit its
    files."""

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
        # two workers can hold one key at once, and write_atomic says why that
        # needs a unique temp
        write_atomic(p, lambda tmp: tmp.write_text(json.dumps(payload)))
