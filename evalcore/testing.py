"""Regression checks: two outputs that must agree, and stored snapshots.

`assert_same(a, b, rtol=, atol=)` compares two results that must be equal
-- batched vs one at a time, a cached prefix vs a full pass, a resumed run
vs an uninterrupted one -- through nested dicts, lists, numpy arrays,
array-likes with `.detach()` (torch, read without importing it) and pandas
frames.  Every mismatch is reported with its path, not only the first, so a
failure says how much moved and where.  A number x matches y when

    |x - y| <= atol + rtol * |y|,

NaN matches NaN at the same position, and strings, bools and None must be
equal exactly.  The tolerance is the caller's to state and justify: zero
for paths that should be bitwise equal, a rounding bound otherwise.

`to_plain(obj)` turns the same structures into JSON-able data, which is how
a snapshot (tests/golden/) is stored.  A snapshot detects CHANGE, not
error: it holds what the code produced at a commit someone checked, so it
complements a test against an independent expected value and never
replaces one.
"""

from __future__ import annotations

import math
from collections.abc import Mapping
from numbers import Number
from typing import Any

import numpy as np
import pandas as pd

__all__ = ["assert_same", "mismatches", "to_plain"]

MAX_REPORTED = 20


def _as_array(x: Any) -> np.ndarray | None:
    if isinstance(x, np.ndarray):
        return x
    if hasattr(x, "detach") and hasattr(x, "cpu"):       # a torch tensor
        return np.asarray(x.detach().cpu().numpy())
    return None


def _num_close(x: float, y: float, rtol: float, atol: float) -> bool:
    if math.isnan(x) or math.isnan(y):
        return math.isnan(x) and math.isnan(y)
    if math.isinf(x) or math.isinf(y):
        return x == y
    return abs(x - y) <= atol + rtol * abs(y)


def mismatches(a: Any, b: Any, rtol: float = 0.0, atol: float = 0.0,
               path: str = "root") -> list[str]:
    """Every place where `a` and `b` differ beyond the tolerance, as
    "path: description" strings; empty when they agree."""
    out: list[str] = []
    _walk(a, b, rtol, atol, path, out)
    return out


def _walk(a: Any, b: Any, rtol: float, atol: float, path: str,  # noqa: PLR0911 - one return per type
          out: list[str]) -> None:
    if isinstance(a, pd.DataFrame) or isinstance(b, pd.DataFrame):
        if not (isinstance(a, pd.DataFrame) and isinstance(b, pd.DataFrame)):
            out.append(f"{path}: {type(a).__name__} vs {type(b).__name__}")
            return
        if list(a.columns) != list(b.columns):
            out.append(f"{path}: columns {list(a.columns)} vs {list(b.columns)}")
            return
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} rows vs {len(b)}")
            return
        for c in a.columns:
            _walk(a[c].tolist(), b[c].tolist(), rtol, atol, f"{path}[{c!r}]", out)
        return
    if isinstance(a, pd.Series) and isinstance(b, pd.Series):
        _walk(a.tolist(), b.tolist(), rtol, atol, path, out)
        return
    xa, xb = _as_array(a), _as_array(b)
    if xa is not None or xb is not None:
        if xa is None or xb is None:
            out.append(f"{path}: {type(a).__name__} vs {type(b).__name__}")
            return
        if xa.shape != xb.shape:
            out.append(f"{path}: shape {xa.shape} vs {xb.shape}")
            return
        _walk(xa.tolist(), xb.tolist(), rtol, atol, path, out)
        return
    if isinstance(a, Mapping) and isinstance(b, Mapping):
        for k in sorted(set(a) | set(b), key=str):
            if k not in a or k not in b:
                out.append(f"{path}[{k!r}]: only in {'b' if k not in a else 'a'}")
            else:
                _walk(a[k], b[k], rtol, atol, f"{path}[{k!r}]", out)
        return
    if isinstance(a, (list, tuple)) and isinstance(b, (list, tuple)):
        if len(a) != len(b):
            out.append(f"{path}: length {len(a)} vs {len(b)}")
            return
        for i, (x, y) in enumerate(zip(a, b, strict=True)):
            _walk(x, y, rtol, atol, f"{path}[{i}]", out)
        return
    numeric = (Number, np.number)
    if (isinstance(a, numeric) and isinstance(b, numeric)
            and not isinstance(a, (bool, np.bool_))
            and not isinstance(b, (bool, np.bool_))):
        x, y = float(a), float(b)          # type: ignore[arg-type]
        if not _num_close(x, y, rtol, atol):
            out.append(f"{path}: {x!r} vs {y!r} (|diff| {abs(x - y):.3g})")
        return
    if a is None and b is None:
        return
    if type(a) is not type(b) and not (isinstance(a, (bool, np.bool_))
                                       and isinstance(b, (bool, np.bool_))):
        out.append(f"{path}: {type(a).__name__} {a!r} vs "
                   f"{type(b).__name__} {b!r}")
        return
    if a != b:
        out.append(f"{path}: {a!r} vs {b!r}")


def assert_same(a: Any, b: Any, rtol: float = 0.0, atol: float = 0.0,
                what: str = "outputs") -> None:
    """Raise AssertionError listing every mismatch between `a` and `b`."""
    found = mismatches(a, b, rtol, atol)
    if found:
        shown = "\n  ".join(found[:MAX_REPORTED])
        more = (f"\n  ... and {len(found) - MAX_REPORTED} more"
                if len(found) > MAX_REPORTED else "")
        raise AssertionError(
            f"{what} differ in {len(found)} place(s) "
            f"(rtol={rtol}, atol={atol}):\n  {shown}{more}")


def to_plain(obj: Any) -> Any:  # noqa: PLR0911 - one return per type
    """JSON-able form of a result: frames become {"columns", "rows"},
    arrays become nested lists, numpy scalars become Python numbers.
    `assert_same` compares a stored snapshot with the plain form of the
    current result."""
    if isinstance(obj, pd.DataFrame):
        return {"columns": [str(c) for c in obj.columns],
                "rows": [[to_plain(v) for v in row]
                         for row in obj.itertuples(index=False, name=None)]}
    if isinstance(obj, pd.Series):
        return [to_plain(v) for v in obj.tolist()]
    arr = _as_array(obj)
    if arr is not None:
        return to_plain(arr.tolist())
    if isinstance(obj, Mapping):
        return {str(k): to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_plain(v) for v in obj]
    if isinstance(obj, (bool, np.bool_)):
        return bool(obj)
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating, float)):
        return float(obj)
    if obj is pd.NA or obj is pd.NaT:
        return None
    return obj
