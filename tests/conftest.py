"""Golden snapshots: the `golden` fixture compares a result with the copy
stored in tests/golden/<module>/<name>.json.

A snapshot detects change, not error (evalcore/testing.py says why), so a
test using it carries the `golden` marker, and snapshots are written only
on request:

    make snapshots      # pytest -m golden --update-golden, then the diff

Review `git diff tests/golden` before committing: every changed number is
a behaviour change someone has to account for."""

from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest

from evalcore.testing import assert_same, to_plain

GOLDEN = Path(__file__).resolve().parent / "golden"


def pytest_addoption(parser):
    parser.addoption("--update-golden", action="store_true",
                     help="rewrite tests/golden snapshots instead of comparing")


@pytest.fixture
def golden(request) -> Callable[..., None]:
    if request.node.get_closest_marker("golden") is None:
        raise pytest.UsageError(
            f"{request.node.nodeid} uses the golden fixture without "
            f"@pytest.mark.golden")
    update = request.config.getoption("--update-golden")
    module = request.node.module.__name__.rsplit(".", 1)[-1]

    def check(name: str, value: Any, rtol: float = 0.0,
              atol: float = 0.0) -> None:
        path = GOLDEN / module / f"{name}.json"
        plain = to_plain(value)
        if update:
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(plain, indent=1, sort_keys=True) + "\n")
            return
        if not path.exists():
            pytest.fail(f"no snapshot at {path.relative_to(GOLDEN.parent)}: "
                        f"create it with `make snapshots` and review the diff")
        stored = json.loads(path.read_text())
        assert_same(plain, stored, rtol=rtol, atol=atol,
                    what=f"result vs snapshot {module}/{name} (current vs stored)")

    return check
