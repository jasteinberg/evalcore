#!/usr/bin/env python3
"""Fail when library code changes and no test changes with it.

A change under `evalcore/` that touches nothing under `tests/` is either a
refactor that must not change behaviour (the existing tests are its guard)
or a new behaviour with no test (the case this check exists to stop).  The
script cannot tell them apart, so it fails and names the override.  It is a
reminder at the boundary, not a proof of coverage: a test file touched for
another reason satisfies it.

    python scripts/check_tests_touched.py --staged          # pre-commit hook
    python scripts/check_tests_touched.py --base origin/main  # pull request

Override: `ALLOW_NO_TESTS=1` in the environment (hook), or the
`no-tests-needed` label (pull request).
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
from collections.abc import Sequence


def untested(changed: Sequence[str]) -> list[str]:
    """Library files among `changed`, if no file under tests/ changed;
    else an empty list.  Paths are repo-relative with forward slashes."""
    src = [f for f in changed if f.startswith("evalcore/") and f.endswith(".py")]
    if any(f.startswith("tests/") for f in changed):
        return []
    return src


def changed_files(staged: bool, base: str | None) -> list[str]:
    cmd = (["git", "diff", "--cached", "--name-only"] if staged
           else ["git", "diff", "--name-only", f"{base}...HEAD"])
    out = subprocess.run(cmd, check=True, capture_output=True, text=True)
    return [line for line in out.stdout.splitlines() if line]


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    which = ap.add_mutually_exclusive_group(required=True)
    which.add_argument("--staged", action="store_true",
                       help="check the files staged for commit")
    which.add_argument("--base", help="check BASE...HEAD (a pull request)")
    return ap.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    if args.staged and os.environ.get("ALLOW_NO_TESTS") == "1":
        return 0
    src = untested(changed_files(args.staged, args.base))
    if not src:
        return 0
    print("Library code changed with no change under tests/:", file=sys.stderr)
    for f in src:
        print(f"  {f}", file=sys.stderr)
    print("Add or update a test that would fail without this change.  For a "
          "pure refactor or a docstring edit, override with ALLOW_NO_TESTS=1 "
          "(commit) or the no-tests-needed label (pull request).",
          file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
