"""Experiments from JSON, and the command line."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

import evalcore.config as config_mod
from evalcore.__main__ import main
from evalcore.config import ConfigError, from_config

pytestmark = pytest.mark.filterwarnings(
    # plumbing fixtures are deliberately tiny; caveats are tested in test_stats
    "ignore:.*carry caveats:RuntimeWarning")

CORE_TESTS = Path(__file__).parents[1] / "core"


def tasks_file(tmp_path, n=4):
    p = tmp_path / "tasks.jsonl"
    p.write_text("\n".join(json.dumps({"id": f"t{i}", "prompt": f"say {i}",
                                       "gold": i, "grader": "numeric"})
                           for i in range(n)))
    return p


def cfg(tmp_path, **over):
    c = {"out": str(tmp_path / "runs" / "r.jsonl"),
         "task": {"type": "file", "path": str(tasks_file(tmp_path))},
         "backend": {"type": "echo"}, "grid": {"model": ["echo"]},
         "metric": "correct", "options": {"workers": 1}}
    c.update(over)
    return c


def test_a_file_task_runs_end_to_end_from_config(tmp_path):
    res = from_config(cfg(tmp_path)).run()
    assert len(res.frame) == 4
    # echo reverses "say 3" -> "3 yas": numeric grader reads 3
    assert (res.frame["correct"] == 1.0).all()
    report = json.loads(res.report_path.read_text())
    assert report["experiment"]["task"] == "tasks"
    assert report["experiment"]["cells"] == [{"model": "echo"}]


@pytest.mark.parametrize("bad,where", [
    ({"repeat": 3}, "config: unknown key(s) ['repeat']"),
    ({"options": {"worker": 2}}, "options: unknown key(s) ['worker']"),
    ({"stages": {"calibrat": False}}, "stages: unknown key(s) ['calibrat']"),
    ({"backend": {"type": "echo", "flavor": "x"}},
     "backend: unknown key(s) ['flavor'] for backend 'echo'"),
    ({"backend": {"type": "http"}}, "backend: backend 'http' requires ['flavour']"),
    ({"task": {"type": "arith"}}, "task: unknown task 'arith'; registered:"),
    ({"task": {"path": "x"}}, 'task: give a {"type": ...}'),
    ({"stages": {"calibrate": True}}, "true is the default"),
    ({"analyses": {"s": {"type": "summary"}}},
     "give 'metric' (shorthand) or 'analyses'"),
])
def test_config_mistakes_are_named(tmp_path, bad, where):
    with pytest.raises(ConfigError) as e:
        from_config(cfg(tmp_path, **bad))
    assert where in str(e.value)


def test_a_misspelt_analysis_key_is_refused_when_the_config_is_read(tmp_path):
    c = cfg(tmp_path, analyses={"s": {"type": "summary", "metric": "correct",
                                      "nboot": 10}})
    del c["metric"]
    with pytest.raises(ConfigError, match=r"analyses.s: unknown key\(s\) \['nboot'\]"):
        from_config(c)


def test_a_required_key_is_named(tmp_path):
    c = cfg(tmp_path)
    del c["grid"]
    with pytest.raises(ConfigError, match="'grid' is required"):
        from_config(c)


def test_a_task_the_backend_cannot_serve_fails_before_any_spend(tmp_path,
                                                               monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    with pytest.raises(ValueError, match="cannot teacher-force"):
        from_config(cfg(tmp_path, task={"type": "arithmetic", "n_items": 4},
                        backend={"type": "http", "flavour": "anthropic",
                                 "default_params": {"model": "m",
                                                    "max_tokens": 8}}))
    with pytest.raises(TypeError, match="cannot send tools"):
        from_config(cfg(tmp_path, task={"type": "search_qa", "n_towns": 4}))


def test_stages_switch_off_from_config(tmp_path):
    exp = from_config(cfg(tmp_path, stages={"calibrate": False,
                                            "analyse": False}))
    res = exp.run()
    assert res.calibration is None and res.analysis == {}


def test_cli_run_dry_run_and_summarize(tmp_path, capsys):
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg(tmp_path)))
    assert main(["run", str(path), "--dry-run"]) == 0
    out = capsys.readouterr().out
    assert "halted: dry run only" in out and "'to_call': 4" in out
    assert not (tmp_path / "runs" / "r.jsonl").exists()
    assert main(["run", str(path)]) == 0
    assert "summary:" in capsys.readouterr().out
    assert main(["summarize", str(tmp_path / "runs" / "r.jsonl")]) == 0
    assert "status: ok 4" in capsys.readouterr().out


def test_a_local_model_task_from_config(tmp_path, monkeypatch):
    """Teacher-forced arithmetic through the whole pipeline on a toy model
    (the HF loader is replaced so nothing is downloaded)."""
    pytest.importorskip("torch")
    sys.path.insert(0, str(CORE_TESTS))
    from test_hf_scoring import KeepLM, ToyTok

    monkeypatch.setattr(config_mod, "_hf_load", lambda spec, head: (
        KeepLM(len(ToyTok().vocab)), ToyTok()))
    res = from_config(cfg(tmp_path, task={"type": "arithmetic", "n_items": 6},
                          backend={"type": "hf", "model": "toy"},
                          metric="em_tf")).run()
    assert len(res.frame) == 6 and (res.frame["status"] == "ok").all()
    assert set(res.calibration.throughput) >= {1, 2}      # batch ladder ran
    assert res.settings["workers"] == 1


# --- your own components ----------------------------------------------------------

PLUGIN = '''
from evalcore import FunctionBackend
from evalcore.registry import register


@register("backend", "always", replace=True)
def always(answer: str, tag: str = "v1"):
    """A model that gives the same answer to everything."""
    return FunctionBackend(lambda req: answer, identity={"always": answer,
                                                         "tag": tag})


@register("stage", "marker", replace=True)
def marker(note: str):
    def stage(exp, res):
        exp.path("marker.txt").write_text(note)
    return stage
'''


def test_a_plugin_registers_a_backend_and_a_stage_used_by_name(tmp_path):
    (tmp_path / "my_plugin.py").write_text(PLUGIN)
    c = cfg(tmp_path, plugins=["my_plugin.py"],
            backend={"type": "always", "answer": "2"},
            stages={"report": {"type": "marker", "note": "replaced"}})
    path = tmp_path / "c.json"
    path.write_text(json.dumps(c))
    res = from_config(path).run()           # the plugin path is relative to c.json
    assert res.frame.set_index("item_id")["correct"].to_dict() == {
        "t0": 0.0, "t1": 0.0, "t2": 1.0, "t3": 0.0}
    assert (tmp_path / "runs" / "r.jsonl.marker.txt").read_text() == "replaced"
    assert res.report_path is None           # the report stage was replaced


def test_a_plugin_factory_is_checked_against_its_signature(tmp_path):
    (tmp_path / "my_plugin.py").write_text(PLUGIN)
    base = cfg(tmp_path, plugins=[str(tmp_path / "my_plugin.py")])
    with pytest.raises(ConfigError, match=r"requires \['answer'\]"):
        from_config({**base, "backend": {"type": "always"}})
    with pytest.raises(ConfigError, match=r"unknown key\(s\) \['answr'\]"):
        from_config({**base, "backend": {"type": "always", "answr": "2"}})


def test_a_missing_plugin_is_named(tmp_path):
    with pytest.raises(ConfigError, match="no file"):
        from_config(cfg(tmp_path, plugins=["nope.py"]))
    with pytest.raises(ModuleNotFoundError):
        from_config(cfg(tmp_path, plugins=["no_such_module_xyz"]))


def test_registering_a_taken_name_needs_replace():
    from evalcore.registry import register
    with pytest.raises(ValueError, match="already registered"):
        register("backend", "echo")(lambda: None)


# --- retrieval from config ------------------------------------------------------------

def retrieval_cfg(tmp_path, retriever=None, **over):
    return cfg(tmp_path, task={"type": "search_qa", "n_towns": 6},
               backend={"type": "retriever", "k": 5,
                        "retriever": retriever or {"type": "bm25"}},
               score={"type": "ranked", "ks": [1, 3]},
               metric="recall@3", **over)


def test_a_retriever_is_evaluated_from_config(tmp_path):
    """One-hop questions: recall@3 = 1 (test_search explains why)."""
    res = from_config(retrieval_cfg(tmp_path)).run()
    f = res.frame
    assert (f["status"] == "ok").all() and (f["depth"] == 5).all()
    assert (f.loc[f["group"] == "pop", "recall@3"] == 1.0).all()
    assert "summary" in res.analysis


def test_retrieval_analysis_from_config(tmp_path):
    c = retrieval_cfg(tmp_path, analyses={
        "ranking": {"type": "retrieval", "n_boot": 200},
        "attrition": {"type": "attrition"}})
    del c["metric"]
    res = from_config(c).run()
    assert set(res.analysis["ranking"]["metric"]) == {
        "recall@1", "hit@1", "ndcg@1", "recall@3", "hit@3", "ndcg@3", "rr"}


def test_a_file_task_with_a_corpus_is_evaluated(tmp_path):
    tasks = tmp_path / "q.jsonl"
    tasks.write_text("\n".join(json.dumps(r) for r in [
        {"id": "q1", "prompt": "where do cats sit", "gold": "-",
         "grader": "exact", "gold_docs": ["c"]},
        {"id": "q2", "prompt": "dogs on mats", "gold": "-", "grader": "exact",
         "gold_docs": ["d"]}]))
    corpus = tmp_path / "docs.jsonl"
    corpus.write_text("\n".join(json.dumps(r) for r in [
        {"id": "c", "text": "cats sit"}, {"id": "d", "text": "dogs mats"}]))
    c = cfg(tmp_path, task={"type": "file", "path": str(tasks),
                            "corpus": str(corpus)},
            backend={"type": "retriever", "retriever": {"type": "bm25"},
                     "k": 2},
            score={"type": "ranked", "ks": [1]}, metric="recall@1",
            stages={"calibrate": False})          # two items: nothing to size
    res = from_config(c).run()
    assert res.frame.sort_values("item_id")["recall@1"].tolist() == [1.0, 1.0]


@pytest.mark.parametrize("over,where", [
    ({"backend": {"type": "retriever", "retriever": {"type": "bm25"}}},
     r"backend 'retriever' requires \['k'\]"),
    ({"backend": {"type": "retriever", "k": 3,
                  "retriever": {"type": "dense", "embedder": {"type": "hf",
                                                              "model": "x"}}}},
     r"retriever.embedder: embedder 'hf' requires \['pooling'\]"),
    ({"backend": {"type": "retriever", "k": 3,
                  "retriever": {"type": "rerank", "first": {"type": "bm25"},
                                "pair_scorer": {"type": "hf", "model": "x"}}}},
     r"retriever 'rerank' requires \['depth'\]"),
    ({"backend": {"type": "retriever", "k": 3,
                  "retriever": {"type": "bm25", "docs": {}}}},
     r"\['docs'\] cannot be set here"),
    ({"agent": {}}, "agent / search: not used when the backend is a retriever"),
])
def test_retrieval_config_mistakes_are_named(tmp_path, over, where):
    with pytest.raises(ConfigError, match=where):
        from_config({**retrieval_cfg(tmp_path), **over})


def test_a_retriever_needs_a_corpus_and_a_ranked_score(tmp_path):
    with pytest.raises(ConfigError, match="needs a corpus"):
        from_config(cfg(tmp_path, backend={"type": "retriever", "k": 2,
                                           "retriever": {"type": "bm25"}}))
    c = retrieval_cfg(tmp_path)
    del c["score"]
    with pytest.raises(ConfigError, match="scored by ranking"):
        from_config(c)


def test_items_without_gold_documents_are_refused(tmp_path):
    corpus = tmp_path / "docs.jsonl"
    corpus.write_text(json.dumps({"id": "a", "text": "x"}))
    c = cfg(tmp_path, task={"type": "file", "path": str(tasks_file(tmp_path)),
                            "corpus": str(corpus)},
            backend={"type": "retriever", "retriever": {"type": "bm25"},
                     "k": 2},
            score={"type": "ranked", "ks": [1]})
    with pytest.raises(ConfigError, match=r"4 item\(s\) have no 'gold_docs'"):
        from_config(c)


def test_a_dense_retriever_from_config(tmp_path, monkeypatch):
    pytest.importorskip("torch")
    sys.path.insert(0, str(CORE_TESTS))
    from test_embed import Enc, Tok
    monkeypatch.setattr(config_mod, "_hf_load", lambda spec, head: (Enc(), Tok()))
    c = retrieval_cfg(tmp_path, retriever={
        "type": "dense", "cache_dir": str(tmp_path / "emb"),
        "embedder": {"type": "hf", "model": "toy", "pooling": "mean"}})
    res = from_config(c).run()
    assert (res.frame["status"] == "ok").all()
    assert any((tmp_path / "emb").glob("*.npz"))       # vectors were cached


def test_the_search_section_chooses_the_agent_tool(tmp_path, monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "k")
    base = cfg(tmp_path, task={"type": "search_qa", "n_towns": 4},
               backend={"type": "http", "flavour": "anthropic",
                        "default_params": {"model": "m", "max_tokens": 8}})
    default = from_config(base).backend.tools["search"]
    tuned = from_config({**base, "search": {
        "retriever": {"type": "bm25", "k1": 2.0}, "k": 3}}).backend.tools["search"]
    assert default.version != tuned.version
    with pytest.raises(ConfigError, match="no corpus to search"):
        from_config({**cfg(tmp_path), "search": {"retriever": {"type": "bm25"},
                                                 "k": 3}})
