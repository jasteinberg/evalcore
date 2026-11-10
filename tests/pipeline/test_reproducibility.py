"""What a run writes so it can be reproduced: the resolved config, and the
code (source digest, commit, and the patch of uncommitted changes).

Expected values: factory defaults read off the signatures by hand, the
round-trip invariance (a resolved config rebuilds the same job), and for
the code, a clean clone of the commit with the patch applied must hash to
the digest of the dirty tree that ran."""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

import evalcore
from evalcore.config import ConfigError, from_config
from evalcore.pipeline import provenance, source_digest

pytestmark = pytest.mark.filterwarnings("ignore:.*carry caveats:RuntimeWarning")

PKG = Path(evalcore.__file__).resolve().parent


def search_cfg(tmp_path) -> dict:
    return {"out": str(tmp_path / "runs" / "r.jsonl"),
            "task": {"type": "search_qa", "n_towns": 4},
            "backend": {"type": "http", "flavour": "anthropic",
                        "default_params": {"model": "m", "max_tokens": 8}},
            "grid": {},
            "search": {"retriever": {"type": "sampled", "base": "bm25",
                                     "beta": 2.0, "seed": 1}, "k": 3},
            "analyses": {"a": "attrition"},
            "tags": {"note": "attrition"}}


def test_every_default_is_written_out_nested_specs_included(tmp_path):
    """By hand from the signatures: search_qa(n_towns, seed=0, k=3),
    http(..., base_url=None, timeout=120.0), bm25(k1=1.2, b=0.75),
    AgentBackend(max_turns=8, max_tool_calls=16)."""
    c = from_config(search_cfg(tmp_path)).config
    assert c["task"] == {"type": "search_qa", "n_towns": 4, "seed": 0, "k": 3}
    assert c["backend"]["base_url"] is None
    assert c["backend"]["timeout"] == 120.0
    assert c["search"]["retriever"] == {
        "type": "sampled", "base": {"type": "bm25", "k1": 1.2, "b": 0.75},
        "beta": 2.0, "seed": 1}
    assert c["agent"] == {"max_turns": 8, "max_tool_calls": 16}
    assert c["repeats"] == 1 and c["options"]["attempts"] == 5
    # a spec given as a bare name resolves, the same text as a tag does not
    assert c["analyses"]["a"] == {"type": "attrition"}
    assert c["tags"] == {"note": "attrition"}


def test_a_resolved_config_rebuilds_the_same_job(tmp_path):
    exp = from_config(search_cfg(tmp_path))
    again = from_config(exp.config)
    assert again.job_hash() == exp.job_hash()
    assert again.config == exp.config                    # a fixed point


def test_a_config_that_is_not_json_is_refused(tmp_path):
    with pytest.raises(ConfigError, match="not JSON"):
        from_config({**search_cfg(tmp_path), "tags": {"when": object()}})


def file_cfg(tmp_path) -> dict:
    tasks = tmp_path / "tasks.jsonl"
    tasks.write_text("\n".join(json.dumps(
        {"id": f"t{i}", "prompt": f"say {i}", "gold": i, "grader": "numeric"})
        for i in range(3)))
    return {"out": str(tmp_path / "runs" / "r.jsonl"),
            "task": {"type": "file", "path": str(tasks)},
            "backend": {"type": "echo"}, "grid": {"model": ["echo"]},
            "metric": "correct", "options": {"workers": 1}}


def test_a_config_run_writes_its_resolved_config(tmp_path):
    exp = from_config(file_cfg(tmp_path))
    res = exp.run()
    path = Path(str(exp.out) + ".config.json")
    assert json.loads(path.read_text()) == exp.config
    report = json.loads(res.report_path.read_text())
    assert report["outputs"]["config"] == str(path)
    assert "script" not in report["outputs"]


SCRIPT = """\
import sys
from evalcore import EchoBackend, Experiment, Task, grid
task = Task.from_file(sys.argv[1])
Experiment(task, EchoBackend(), grid({"model": ["echo"]}), sys.argv[2],
           metric="correct").run()
"""


def test_a_script_run_keeps_a_copy_of_its_script(tmp_path):
    tasks = Path(file_cfg(tmp_path)["task"]["path"])
    script = tmp_path / "my_sweep.py"
    script.write_text(SCRIPT)
    out = tmp_path / "runs" / "s.jsonl"
    proc = subprocess.run([sys.executable, str(script), str(tasks), str(out)],
                          capture_output=True, text=True, check=False,
                          cwd=tmp_path)
    assert proc.returncode == 0, proc.stderr
    assert Path(str(out) + ".script.py").read_text() == SCRIPT
    assert not Path(str(out) + ".config.json").exists()
    prov = json.loads(Path(str(out) + ".run.json").read_text())["provenance"]
    assert prov["script"]["argv"][1:] == [str(tasks), str(out)]


def test_source_digest_names_the_code_not_the_place(tmp_path):
    copy = tmp_path / "evalcore"
    shutil.copytree(PKG, copy, ignore=shutil.ignore_patterns("__pycache__"))
    assert source_digest(copy) == source_digest(PKG)
    f = copy / "core" / "spec.py"
    f.write_text(f.read_text() + "\n# an edit\n")
    assert source_digest(copy) != source_digest(PKG)


def git(cwd: Path, *args: str) -> str:
    return subprocess.run(["git", "-C", str(cwd), "-c", "user.email=t@t",
                           "-c", "user.name=t", *args], capture_output=True,
                          text=True, check=True).stdout


def test_commit_plus_patch_rebuild_the_code_that_ran(tmp_path):
    """A dirty checkout (an edited file and a new one, whose name has a
    space) is recorded.  A clean
    clone of the commit, with the patch applied as provenance says, must
    hash to the digest of the tree that ran."""
    if shutil.which("git") is None:
        pytest.skip("no git")
    repo = tmp_path / "repo"
    shutil.copytree(PKG, repo / "src" / "evalcore",
                    ignore=shutil.ignore_patterns("__pycache__"))
    git(tmp_path, "init", "-q", str(repo))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    pkg = repo / "src" / "evalcore"
    (pkg / "core" / "spec.py").write_text(
        (pkg / "core" / "spec.py").read_text() + "\n# local edit\n")
    (pkg / "scratch plugin.py").write_text("X = 1\n")   # a space in the name

    prov = provenance(pkg)
    assert prov["dirty"] is True
    assert prov["untracked"] == ["scratch plugin.py"]
    assert prov["package_dir"] == "src/evalcore/"
    patch = tmp_path / "run.repo.patch"
    patch.write_text(prov["diff"])

    clean = tmp_path / "clean"
    subprocess.run(["git", "clone", "-q", str(repo), str(clean)], check=True)
    git(clean, "checkout", "-q", prov["commit"])
    assert source_digest(clean / "src" / "evalcore") != prov["source_digest"]
    git(clean, "apply", f"--directory={prov['package_dir']}", str(patch))
    assert source_digest(clean / "src" / "evalcore") == prov["source_digest"]


def test_a_clean_checkout_has_no_patch(tmp_path):
    if shutil.which("git") is None:
        pytest.skip("no git")
    repo = tmp_path / "repo"
    shutil.copytree(PKG, repo / "evalcore",
                    ignore=shutil.ignore_patterns("__pycache__"))
    git(tmp_path, "init", "-q", str(repo))
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    prov = provenance(repo / "evalcore")
    assert prov["dirty"] is False and prov["diff"] == ""


def test_a_dirty_run_writes_its_patch_beside_the_sink(tmp_path, monkeypatch):
    from evalcore import pipeline
    real = pipeline.provenance
    monkeypatch.setattr(pipeline, "provenance",
                        lambda: {**real(), "dirty": True, "diff": "PATCH\n"})
    exp = from_config(file_cfg(tmp_path))
    res = exp.run()
    patch = Path(str(exp.out) + ".repo.patch")
    assert patch.read_text() == "PATCH\n"
    report = json.loads(res.report_path.read_text())
    assert report["outputs"]["repo_patch"] == str(patch)
    assert "diff" not in report["provenance"]          # in the file, not here


def test_a_resolved_config_finds_its_plugins_from_anywhere(tmp_path,
                                                          monkeypatch):
    """A .py plugin is named relative to the config file.  The resolved
    config lives beside the sink, so it must name the plugin absolutely."""
    cdir = tmp_path / "configs"
    cdir.mkdir()
    (cdir / "my_plugin.py").write_text("X = 1\n")
    cfg = {**file_cfg(tmp_path), "plugins": ["my_plugin.py"]}
    (cdir / "c.json").write_text(json.dumps(cfg))
    monkeypatch.chdir(tmp_path)                    # not where the plugin is
    exp = from_config(cdir / "c.json")
    assert exp.config["plugins"] == [str((cdir / "my_plugin.py").resolve())]
    assert from_config(exp.config).job_hash() == exp.job_hash()
