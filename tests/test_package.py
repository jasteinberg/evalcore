"""The package as a user meets it: the public API, and the worked example."""

from __future__ import annotations

import importlib.util
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

import evalcore

REPO = Path(__file__).resolve().parents[1]


PUBLIC = {
    "Experiment", "RunOptions", "from_config", "register", "ConfigError",
    "Cell", "Item", "Unit", "grid", "units", "digest", "canonical",
    "Backend", "Request", "Response", "Transient", "Fatal", "DECODE_KEYS",
    "HTTPBackend", "HFBackend", "EchoBackend", "FunctionBackend",
    "AgentBackend", "Probe", "FunctionProbe", "Cache", "check_backend",
    "CheckReport", "with_retries",
    "Tool", "ToolInputError", "ToolUnavailable",
    "EmbeddingCache", "FunctionEmbedder", "HFEmbedder", "HTTPEmbedder",
    "BM25", "CorpusSearch", "DenseRetriever", "ExactIndex", "FaissIndex",
    "RetrieverBackend", "Reranker", "FunctionScorer", "HFCrossEncoder",
    "ann_recall", "search_tool", "rank_metrics", "score_ranked",
    "retrieval_summary",
    "plan", "execute", "ParseFailure", "calibrate", "derive",
    "to_frame", "load_arrays", "manifest_path", "read_manifest",
    "write_manifest",
    "read_events", "summarize_events", "format_summary", "parse_log",
    "Task", "load_tasks", "save_tasks", "grade", "make_score",
    "BootResult", "cluster_bootstrap", "paired_bootstrap", "paired_pvalue",
    "icc1", "effective_n", "wilson_interval", "mcnemar_exact",
    "benjamini_hochberg", "holm", "summarize", "attrition",
    "axis_curves", "axis_data", "axis_bootstrap", "value_curves", "crossing",
    "composition", "susceptibility", "subsampled_sharpness", "explode_tokens",
    "floors", "floors_from_frame", "headroom", "majority_rate",
    "prior_matched_rate", "contamination_report", "ngram_overlap",
    "longest_match",
}


def test_d31_public_api_is_the_written_contract():
    """D31.  The literal set above IS the contract: `__all__` equals it, has
    no duplicates, and every name resolves on the package."""
    assert len(evalcore.__all__) == len(set(evalcore.__all__))
    assert set(evalcore.__all__) == PUBLIC
    missing = [n for n in PUBLIC if not hasattr(evalcore, n)]
    assert missing == []


def test_d33_worked_example_runs_with_its_output_under_tmp_path(tmp_path):
    """D33.  Contract: the example runs in a fresh process, exits 0, and
    writes its sink where it is told (here tmp_path), so a rerun reflects
    edits instead of resuming a stale sink elsewhere."""
    out = tmp_path / "demo"
    proc = subprocess.run([sys.executable, str(REPO / "examples" /
                                                "worked_example.py"),
                           str(out)], cwd=tmp_path, capture_output=True,
                          env={**os.environ, "PYTHONPATH": str(REPO)},
                          text=True, timeout=120, check=False)
    assert proc.returncode == 0, proc.stderr
    assert (out / "rows.jsonl").exists()


def run_example(name: str, out: Path) -> subprocess.CompletedProcess:
    return subprocess.run([sys.executable, str(REPO / "examples" / name),
                           str(out)], cwd=out.parent, capture_output=True,
                          env={**os.environ, "PYTHONPATH": str(REPO)},
                          text=True, timeout=300, check=False)


def test_agent_example_reaches_every_episode_outcome(tmp_path):
    """The example's claim: its items end ok, ok after a corrected call,
    tool_error and truncated, and attrition accounts for all five."""
    proc = run_example("agent_example.py", tmp_path / "demo")
    assert proc.returncode == 0, proc.stderr
    rows = [json.loads(x) for x in
            (tmp_path / "demo" / "rows.jsonl").read_text().splitlines()]
    status = {r["item_id"].split("-")[0]: r["status"] for r in rows}
    assert status == {"plain": "ok", "typo": "ok", "outage": "tool_error",
                      "stubborn": "truncated"}
    assert sum(r["status"] == "ok" for r in rows) == 3


def test_probe_example_steers_the_margin_upward(tmp_path):
    """The example's claim: adding v = W_U[A] - W_U[B] at the read position
    raises the margin log p(A) - log p(B), monotonically in alpha."""
    pytest.importorskip("torch")
    pytest.importorskip("transformers")
    # in-process: a fresh interpreter would pay the transformers import
    # (over 10 s) that this session has already paid
    spec = importlib.util.spec_from_file_location(
        "probe_example", REPO / "examples" / "probe_example.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    mod.main(str(tmp_path / "demo"))
    rows = [json.loads(x) for x in
            (tmp_path / "demo" / "rows.jsonl").read_text().splitlines()]
    assert {r["status"] for r in rows} == {"ok"}
    assert all(r["meta_arrays"] for r in rows)
    by_alpha: dict[float, list[float]] = {}
    for r in rows:
        by_alpha.setdefault(r["alpha"], []).append(r["margin"])
    means = [sum(v) / len(v) for _, v in sorted(by_alpha.items())]
    assert means == sorted(means) and means[-1] > means[0]
