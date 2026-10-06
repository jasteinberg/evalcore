"""End-to-end on EchoBackend: grid -> runner -> cache -> resume -> frame."""

import pytest

from evalcore import (
    Cache,
    Cell,
    EchoBackend,
    Item,
    Request,
    Unit,
    execute,
    grid,
    summarize,
    units,
)


class CountingEcho(EchoBackend):
    """Echo that counts calls, so "nothing was dispatched" is checkable."""

    def __init__(self):
        super().__init__()
        self.calls = 0

    def complete(self, req):
        self.calls += 1
        return super().complete(req)


def items(n=12, aug=2):
    out = []
    for i in range(n):
        src = f"q{i:03d}"
        out.append(Item(src, {"q": f"question {i}", "gold": str(i % 3)}))
        for a in range(aug):
            out.append(Item(f"{src}-a{a}", {"q": f"QUESTION {i} v{a}",
                                            "gold": str(i % 3)},
                            parent_id=src))
    return out


def render(u: Unit) -> Request:
    return Request(u.id, [{"role": "user", "content": u.item.payload["q"]}],
                   {"model": u.cell.get("model"),
                    "temperature": u.cell.get("temperature")})


def score(u: Unit, r) -> dict:
    return {"correct": float(len(r.text) % 3 == int(u.item.payload["gold"])),
            "n_chars": len(r.text)}


def test_ids_are_stable_and_tag_free():
    a = Cell({"model": "m", "temperature": 0.0}, {"arm": "baseline"})
    b = Cell({"temperature": 0.0, "model": "m"}, {"arm": "renamed"})
    assert a.id == b.id                       # key order and tags do not count
    assert Cell({"model": "m", "temperature": 0.7}).id != a.id


def test_grid_exclusions():
    g = grid({"model": ["a", "b"], "temperature": [0.0, 1.0]},
             exclude=[{"model": "a", "temperature": 1.0}])
    assert len(g) == 3
    assert all(not (c.get("model") == "a" and c.get("temperature") == 1.0)
               for c in g)


def test_augmentations_cluster_with_their_parent():
    its = items(n=4, aug=3)
    assert len(its) == 16
    assert len({i.cluster for i in its}) == 4


def test_run_cache_resume_and_frame(tmp_path):
    cells = grid({"model": ["m1", "m2"], "temperature": [0.0]},
                 tags={"arm": "base"})
    its = items(n=8, aug=2)
    out, cache = tmp_path / "runs.jsonl", Cache(tmp_path / "cache")

    n = len(cells) * len(its)
    be1 = CountingEcho()
    df1 = execute(units(cells, its), be1, render, score, out,
              cache=cache, workers=2, log_every=1000)
    assert len(df1) == n and be1.calls == n
    assert cache.hits == 0 and cache.misses == n
    assert df1["error"].isna().all()

    # second pass: every unit already in the sink -> nothing dispatched.
    # Checked on the backend, not the frame: to_frame deduplicates on
    # unit_id, so the frame looks the same whether or not work was redone.
    be2 = CountingEcho()
    df2 = execute(units(cells, its), be2, render, score, out,
              cache=cache, workers=2, log_every=1000)
    assert be2.calls == 0
    assert len(df2) == len(df1)
    assert len(out.read_text().splitlines()) == n      # nothing re-appended

    # cache hit path: same requests, fresh sink
    c2 = Cache(tmp_path / "cache")
    execute(units(cells, its), EchoBackend(), render, score,
        tmp_path / "runs2.jsonl", cache=c2, workers=2, log_every=1000)
    assert c2.hits == len(cells) * len(its) and c2.misses == 0


def test_summarize_reports_neff_from_the_cluster_structure(tmp_path):
    """A value constant within each cluster and varying across clusters has
    MS_w = 0, so rho = 1 exactly and n_eff = N / (1 + (m-1)) = k: twenty
    clusters of five rows are worth twenty rows.  The expected value comes
    from eq. (3), not from running the estimator."""
    cells = grid({"model": ["m1"], "temperature": [0.0]})
    its = items(n=20, aug=4)
    df = execute(units(cells, its), EchoBackend(), render, score,
             tmp_path / "r.jsonl", workers=1, log_every=1000)
    df["v"] = df["cluster"].str[1:].astype(int) % 3   # gold: shared in cluster
    s = summarize(df, "v", by=["model"], n_boot=500)
    assert len(s) == 1
    row = s.iloc[0]
    assert row["n_clusters"] == 20 and row["n_rows"] == 100
    assert row["rho"] == pytest.approx(1.0)
    assert row["deff"] == pytest.approx(5.0)
    assert row["n_eff"] == pytest.approx(20.0)


def test_cache_put_is_concurrent_safe(tmp_path):
    """Two workers can hold the same cache key at once.

    Any grid with repeats > 1 at a non-zero temperature builds identical
    requests by construction -- the repeat index is not part of the request --
    so the collision is the normal case, not a rare race. With a temp file
    named per key rather than per writer, the second os.replace raised
    FileNotFoundError and killed the batch.
    """
    import concurrent.futures as cf

    cache = Cache(tmp_path / "cache")
    key = "a" * 32
    with cf.ThreadPoolExecutor(16) as pool:
        errs = [e for e in pool.map(
            lambda i: _put_or_err(cache, key, {"text": f"x{i}"}), range(200))
            if e is not None]

    assert errs == []
    assert cache.get(key)["text"].startswith("x")
    assert list((tmp_path / "cache").rglob("*.tmp")) == []


def _put_or_err(cache, key, payload):
    try:
        cache.put(key, payload)
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - collects the bug
        return f"{type(exc).__name__}: {exc}"
    return None
