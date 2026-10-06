"""Overlap between generated text and a reference corpus.

The question this answers is "did the model produce this, or retrieve it?", and
the reason it belongs in an eval harness is that memorisation is invisible to
the metric: a model reciting its training data scores *well*, and a human
reading the output sees fluent, correct text.  Nothing in the score, the
confidence interval, or the sample flags it.

Two measures, both standard:

* n-gram overlap at fixed n -- the fraction of the candidate's n-grams that
  appear anywhere in the reference.  Cheap, and the usual choice in the
  decontamination literature (n = 8 to 13 for natural language).
* longest match -- the length of the longest contiguous run shared with the
  reference.  A maximum rather than a mean, so it is robust to a candidate
  that is mostly original with one memorised passage in it.

Neither number means anything alone.  Two samples of the same author, or two
documents from the same domain, share long runs by coincidence: function words,
stock phrases, formatting.  The interpretable quantity is the *difference*
between the candidate and a control -- reference-adjacent text known not to be
in the reference.  A candidate scoring at or below its control is producing;
one scoring well above it is retrieving.  `report()` takes the control as a
first-class argument for that reason.

Works on any sequence: a string (character-level), a list of tokens, a list of
token ids.  Costs O(len(reference)) per probed n, so `longest_match` caps the
search and says so rather than degrading quietly on a large corpus.
"""

from __future__ import annotations

from collections.abc import Hashable, Mapping, Sequence
from typing import Any

import pandas as pd

__all__ = ["grams", "longest_match", "ngram_overlap", "report"]


def grams(seq: Sequence[Any], n: int) -> list[Hashable]:
    """All contiguous n-grams of `seq`, as hashable objects.

    Strings slice to strings and are already hashable; anything else is
    tupled, so token lists and id lists work unchanged."""
    if n <= 0:
        raise ValueError("n must be positive")
    if len(seq) < n:
        return []
    if isinstance(seq, str):
        return [seq[i:i + n] for i in range(len(seq) - n + 1)]
    return [tuple(seq[i:i + n]) for i in range(len(seq) - n + 1)]


def ngram_overlap(candidate: Sequence[Any], reference: Sequence[Any],
                  n: int) -> float:
    """Fraction of the candidate's n-grams that occur in the reference.

    Type-level, not token-level: a phrase the candidate repeats ten times
    counts once, so a single memorised passage cannot be inflated by
    repetition into a high score."""
    cand = set(grams(candidate, n))
    if not cand:
        return float("nan")
    ref = set(grams(reference, n))
    return len(cand & ref) / len(cand)


def longest_match(candidate: Sequence[Any], reference: Sequence[Any],
                  max_n: int = 64) -> int:
    """Length of the longest contiguous run the candidate shares with the
    reference, capped at `max_n`.

    Binary search on the length: presence of a shared run of length L implies
    a shared run of every length below L, so the predicate is monotone.  The
    cap is real -- a return value equal to `max_n` means "at least max_n", not
    "exactly max_n", and callers should raise the cap rather than believe it.
    """
    hi = min(max_n, len(candidate), len(reference))
    if hi <= 0:
        return 0

    def shares(n: int) -> bool:
        cand = set(grams(candidate, n))
        return any(g in cand for g in grams(reference, n))

    if not shares(1):
        return 0
    lo = 1
    while lo < hi:                      # invariant: shares(lo), maybe not hi
        mid = (lo + hi + 1) // 2
        if shares(mid):
            lo = mid
        else:
            hi = mid - 1
    return lo


def report(candidates: Mapping[str, Sequence[Any]],
           reference: Sequence[Any],
           control: Sequence[Any] | None = None,
           ns: Sequence[int] = (8, 16),
           max_n: int = 64) -> pd.DataFrame:
    """Overlap table with the control as its own row.

    Read the table by comparing rows, never by reading a row alone.  The
    control row is the overlap you get *without* memorisation, for this
    reference and this genre; candidates at or below it are producing.

    Omitting `control` is supported and discouraged -- the numbers are then
    uncalibrated, and the missing row is the reason a reviewer cannot tell
    whether 0.31 is alarming or ordinary.
    """
    rows: list[dict[str, Any]] = []
    named = dict(candidates)
    if control is not None:
        named = {**named, "control": control}
    for name, cand in named.items():
        row: dict[str, Any] = {"name": name, "length": len(cand),
                               "is_control": name == "control"
                               and control is not None}
        for n in ns:
            row[f"overlap_{n}"] = ngram_overlap(cand, reference, n)
        row["longest_match"] = longest_match(cand, reference, max_n)
        row["capped"] = row["longest_match"] >= max_n
        rows.append(row)
    df = pd.DataFrame(rows).set_index("name")

    if control is not None:
        base = df.loc["control"]
        for col in [f"overlap_{n}" for n in ns] + ["longest_match"]:
            df[f"{col}_excess"] = df[col] - base[col]
    return df
