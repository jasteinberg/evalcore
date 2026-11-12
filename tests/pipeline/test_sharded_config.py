"""A config run as shards from the command line, then merged.  The merged
result must be the unsharded run's result: that invariance is the expected
value, not anything computed by the shard code."""

from __future__ import annotations

import json

import pandas as pd
import pytest

from evalcore import to_frame
from evalcore.__main__ import main
from evalcore.config import from_config

pytestmark = pytest.mark.filterwarnings("ignore:.*carry caveats:RuntimeWarning")

N_TASKS, N = 12, 3


def config(tmp_path, sub: str):
    tasks = tmp_path / "tasks.jsonl"
    if not tasks.exists():
        tasks.write_text("\n".join(json.dumps(
            {"id": f"t{i}", "prompt": f"say {i}", "gold": i if i % 3 else -1,
             "grader": "numeric"}) for i in range(N_TASKS)))
    c = {"out": str(tmp_path / sub / "r.jsonl"),
         "task": {"type": "file", "path": str(tasks)},
         "backend": {"type": "echo"}, "grid": {"model": ["echo"]},
         "metric": "correct", "options": {"workers": 1}}
    path = tmp_path / f"{sub}.json"
    path.write_text(json.dumps(c))
    return path, tmp_path / sub / "r.jsonl"


def table(path) -> pd.DataFrame:
    return pd.read_csv(path).sort_index(axis=1)


def test_shards_then_merge_equal_one_unsharded_run(tmp_path, capsys):
    whole_cfg, whole = config(tmp_path, "whole")
    assert main(["run", str(whole_cfg)]) == 0
    capsys.readouterr()
    cfg, out = config(tmp_path, "sharded")
    for i in range(N):
        assert main(["run", str(cfg), "--shard", f"{i}/{N}"]) == 0
    assert "summary:" not in capsys.readouterr().out   # no per-shard analysis
    assert not out.exists()
    assert main(["merge", str(cfg), "--shards", str(N)]) == 0
    assert "complete" in capsys.readouterr().out

    cols = ["unit_id", "item_id", "text", "correct", "status"]
    got = to_frame(out)[cols].sort_values("unit_id").reset_index(drop=True)
    want = to_frame(whole)[cols].sort_values("unit_id").reset_index(drop=True)
    pd.testing.assert_frame_equal(got, want)
    adir = lambda p: p.with_name(p.name + ".analysis")        # noqa: E731
    pd.testing.assert_frame_equal(table(adir(out) / "attrition.csv"),
                                  table(adir(whole) / "attrition.csv"))
    s, w = (table(adir(p) / "summary.csv") for p in (out, whole))
    # by hand: echo reverses "say i" to "i yas" reversed digitwise, so the
    # numeric grader reads i only for palindromic i, and gold is -1 when
    # 3 | i.  Correct: 1, 2, 4, 5, 7, 8, 11 (10 reads as 1).
    assert s["point"].tolist() == w["point"].tolist() == [7 / 12]


def test_after_a_merge_an_unsharded_run_only_analyses(tmp_path):
    cfg, _ = config(tmp_path, "s")
    for i in range(N):
        main(["run", str(cfg), "--shard", f"{i}/{N}"])
    main(["merge", str(cfg), "--shards", str(N)])
    res = from_config(cfg).run()
    assert res.skipped and "already complete" in res.skipped


def test_a_shard_that_never_ran_is_missing_until_it_runs(tmp_path, capsys):
    cfg, out = config(tmp_path, "s")
    for i in (0, 2):
        main(["run", str(cfg), "--shard", f"{i}/{N}"])
    main(["merge", str(cfg), "--shards", str(N)])
    assert "incomplete" in capsys.readouterr().out
    att = pd.read_csv(out.with_name(out.name + ".analysis") / "attrition.csv")
    n_rows = len(to_frame(out))
    assert att["n_planned"].sum() == N_TASKS
    assert att["n_missing"].sum() == N_TASKS - n_rows > 0

    main(["run", str(cfg), "--shard", f"1/{N}"])               # late shard
    main(["merge", str(cfg), "--shards", str(N)])               # re-merge
    assert len(to_frame(out)) == N_TASKS


def test_merge_will_not_overwrite_a_sink_it_did_not_write(tmp_path):
    cfg, out = config(tmp_path, "s")
    main(["run", str(cfg)])                                     # unsharded
    main(["run", str(cfg), "--shard", f"0/{N}"])
    before = out.read_text()
    with pytest.raises(FileExistsError, match="not a previous merge"):
        main(["merge", str(cfg), "--shards", str(N)])
    assert out.read_text() == before


def test_shard_slurm_reads_the_array_task(tmp_path, monkeypatch):
    cfg, out = config(tmp_path, "s")
    monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "11")
    monkeypatch.setenv("SLURM_ARRAY_TASK_MIN", "10")
    monkeypatch.setenv("SLURM_ARRAY_TASK_COUNT", "3")
    assert main(["run", str(cfg), "--shard", "slurm"]) == 0
    assert out.with_name("r.shard1-of-3.jsonl").exists()


def test_shard_k8s_reads_the_pod_index(tmp_path, monkeypatch):
    cfg, out = config(tmp_path, "s")
    monkeypatch.setenv("JOB_COMPLETION_INDEX", "2")
    assert main(["run", str(cfg), "--shard", "k8s/3"]) == 0
    assert out.with_name("r.shard2-of-3.jsonl").exists()


@pytest.mark.parametrize("bad", ["3", "a/3", "3/3", "1/0", "k8s", "k8s/x"])
def test_a_malformed_shard_argument_is_refused(tmp_path, bad):
    cfg, _ = config(tmp_path, "s")
    with pytest.raises((SystemExit, ValueError)):
        main(["run", str(cfg), "--shard", bad])


def test_merge_refuses_shards_that_ran_another_job(tmp_path):
    """batch_size enters the job hash and not the unit ids: shards run at
    the default, then the config changed, must not merge as complete."""
    cfg, out = config(tmp_path, "s")
    for i in range(N):
        main(["run", str(cfg), "--shard", f"{i}/{N}"])
    c = json.loads(cfg.read_text())
    c["options"]["batch_size"] = 2
    cfg.write_text(json.dumps(c))
    with pytest.raises(ValueError, match="config changed since the shards"):
        main(["merge", str(cfg), "--shards", str(N)])
    assert not out.exists()


def test_a_remerge_keeps_rows_a_resume_added_after_the_merge(tmp_path):
    """Two shards merged, then an unsharded resume fills the third shard's
    units into the merged sink: those rows exist only there, so a second
    merge must refuse rather than overwrite them."""
    cfg, out = config(tmp_path, "s")
    for i in (0, 2):
        main(["run", str(cfg), "--shard", f"{i}/{N}"])
    main(["merge", str(cfg), "--shards", str(N)])
    main(["run", str(cfg)])                                     # resume
    assert len(to_frame(out)) == N_TASKS
    before = out.read_text()
    with pytest.raises(FileExistsError, match="changed since it was merged"):
        main(["merge", str(cfg), "--shards", str(N)])
    assert out.read_text() == before


def test_merge_says_which_shard_is_not_complete(tmp_path, capsys):
    """Shards 0 and 2 ran, 1 never started: the marker lists each shard's
    status and the command prints the one that is not complete."""
    from evalcore.pipeline import read_marker
    cfg, out = config(tmp_path, "s")
    for i in (0, 2):
        main(["run", str(cfg), "--shard", f"{i}/{N}"])
    capsys.readouterr()
    main(["merge", str(cfg), "--shards", str(N)])
    assert "shard 1: never started" in capsys.readouterr().out
    marker = read_marker(out.with_name(out.name + ".done"))
    assert marker["shards"] == {"0": "complete", "1": "never started",
                                "2": "complete"}


def test_the_merge_digest_is_of_the_bytes_on_disk(tmp_path):
    """The recorded sha256 is the digest of the merged file's bytes, so a
    re-merge of an untouched merge is accepted on any platform."""
    import hashlib
    cfg, out = config(tmp_path, "s")
    for i in range(N):
        main(["run", str(cfg), "--shard", f"{i}/{N}"])
    main(["merge", str(cfg), "--shards", str(N)])
    mark = json.loads(out.with_name(out.name + ".shards.json").read_text())
    assert mark["sha256"] == hashlib.sha256(out.read_bytes()).hexdigest()
    assert main(["merge", str(cfg), "--shards", str(N)]) == 0
