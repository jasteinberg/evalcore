"""scripts/check_tests_touched.py: the verdict on a list of changed paths,
and that the mode is required rather than defaulted."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

PATH = Path(__file__).resolve().parents[2] / "scripts" / "check_tests_touched.py"
spec = importlib.util.spec_from_file_location("check_tests_touched", PATH)
ctt = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ctt)


@pytest.mark.parametrize("changed,want", [
    (["evalcore/core/runner.py"], ["evalcore/core/runner.py"]),
    (["evalcore/core/runner.py", "tests/core/test_harness.py"], []),
    (["evalcore/a.py", "evalcore/b.py", "README.md"],
     ["evalcore/a.py", "evalcore/b.py"]),
    (["README.md", "Makefile"], []),                  # no library code
    (["tests/core/test_harness.py"], []),             # tests only
    (["evalcore/py.typed"], []),                      # not a .py file
    (["examples/evalcore/x.py"], []),                 # prefix, not substring
])
def test_verdict_on_changed_paths(changed, want):
    assert ctt.untested(changed) == want


def test_a_mode_is_required():
    with pytest.raises(SystemExit):
        ctt.parse_args([])
    with pytest.raises(SystemExit):
        ctt.parse_args(["--staged", "--base", "origin/main"])
