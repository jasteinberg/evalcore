"""Golden snapshots of end-to-end outputs.  These catch any change in what
the harness produces for a fixed input; they do not say the output is right
(the hand-value tests elsewhere do that).  Regenerate with `make snapshots`.

Tolerance: rtol 1e-12 on floats.  The same inputs give the same arithmetic
on every platform except for summation order inside numpy's vectorised
reductions, which moves a double by a few ulp (~1e-16 relative) per sum;
1e-12 leaves four orders of margin and still catches any real change."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from evalcore import FunctionBackend, Item, Response, grid, paired_bootstrap
from evalcore.pipeline import Experiment, RunOptions
from evalcore.tasks.arithmetic import arithmetic_items
from evalcore.tasks.base import Task

pytestmark = [pytest.mark.golden, pytest.mark.filterwarnings(
    "ignore:.*carry caveats:RuntimeWarning")]

RTOL = 1e-12


def _model(req):
    """Deterministic answers that depend on the model param and the item, so
    the two conditions differ on some items and agree on others."""
    q = req.messages[-1]["content"]
    i = int(q.split()[-1])
    strong = req.params["model"] == "strong"
    right = (i % 3 != 0) if strong else (i % 2 == 0)
    return Response(req.unit_id, text=str(i if right else -1),
                    meta={"truncated": False})


def _items(n=24):
    # augmentations: items 12..23 are rephrasings of 0..11, same cluster
    out = []
    for i in range(n):
        src = i % 12
        out.append(Item(f"q{i}", {"messages": [{"role": "user",
                                                "content": f"value of {i}"}],
                                  "gold": i, "grader": "numeric"},
                        parent_id=None if i < 12 else f"q{src}"))
    return out


def test_experiment_end_to_end(tmp_path, golden):
    exp = Experiment(Task("toy", _items()),
                     FunctionBackend(_model, identity={"model": "golden"}),
                     grid({"model": ["weak", "strong"]}), tmp_path / "r.jsonl",
                     metric="correct", calibrate=None,
                     options=RunOptions(workers=1))
    res = exp.run()
    golden("summary", res.analysis["summary"], rtol=RTOL)
    golden("attrition", res.analysis["attrition"], rtol=RTOL)
    frame = res.frame.sort_values("unit_id")[["item_id", "cluster", "model",
                                               "status", "correct"]]
    golden("rows", frame.reset_index(drop=True), rtol=RTOL)


def test_paired_bootstrap(golden):
    rng = np.random.default_rng(7)
    k, m = 30, 3
    rows = []
    for c in range(k):
        level = rng.normal()
        for _ in range(m):
            for condition, shift in (("a", 0.3), ("b", 0.0)):
                rows.append({"cluster": f"c{c}", "condition": condition,
                             "v": level + shift + 0.5 * rng.normal()})
    df = pd.DataFrame(rows)
    res = paired_bootstrap(df, "v", "condition", "a", "b", n_boot=2000, seed=1)
    golden("paired", res.as_row(), rtol=RTOL)


def test_arithmetic_items_default_mode(golden):
    its = arithmetic_items(digits=2, n_items=12, seed=3)
    golden("arith_d2_n12_s3", [[i.item_id, dict(i.payload)] for i in its])
