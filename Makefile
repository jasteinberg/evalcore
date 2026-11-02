# The same checks CI runs (.github/workflows/ci.yml), in one command.
# Uses whichever python is active; override with `make check PYTHON=...`.
PYTHON ?= python

.PHONY: check lint typecheck test hooks snapshots

check: lint typecheck test

lint:
	$(PYTHON) -m ruff check .

typecheck:
	$(PYTHON) -m mypy evalcore

test:
	$(PYTHON) -m pytest -q -W error::RuntimeWarning

# opt in to the pre-commit hook (.githooks/pre-commit) for this clone
hooks:
	git config core.hooksPath .githooks

# rewrite the golden snapshots (tests/golden/), then show what moved; review
# the diff before committing, since every change there is a behaviour change
snapshots:
	$(PYTHON) -m pytest -q -m golden --update-golden
	git diff --stat -- tests/golden
