"""Experiments from JSON, and the command line."""

from __future__ import annotations

import json

import pytest
from support.scripted import Counting

import evalcore.config as config_mod
from evalcore import Cell, FunctionBackend
from evalcore.__main__ import main
from evalcore.config import ConfigError, from_config
from evalcore.registry import REGISTRIES

pytestmark = pytest.mark.filterwarnings(
    # plumbing fixtures are deliberately tiny; caveats are tested in test_stats
    "ignore:.*carry caveats:RuntimeWarning")


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


@pytest.mark.parametrize("over,bad", [
    ({"agent": {"max_turn": 3}}, "agent: unknown key(s) ['max_turn']"),
    ({"search": {"retriever": "bm25", "k": 3, "kk": 1}},
     "search: unknown key(s) ['kk']"),
])
def test_d16_agent_and_search_sub_keys_are_strict(tmp_path, over, bad):
    """D16.  Hand-written typos: each must be refused by name before any
    backend is built."""
    with pytest.raises(ConfigError) as e:
        from_config(cfg(tmp_path, task={"type": "search_qa", "n_towns": 4},
                        **over))
    assert bad in str(e.value)


@pytest.mark.parametrize("lines,bad", [
    ([{"id": "a", "txt": "x"}], "each line must be exactly"),
    ([{"id": "a", "text": "x"}, {"id": "a", "text": "y"}], "duplicate id 'a'"),
])
def test_d16_corpus_lines_are_strict(tmp_path, lines, bad):
    """D16.  Hand-written corpus faults: a misspelt field and a duplicate
    document id are refused, naming the line."""
    corpus = tmp_path / "docs.jsonl"
    corpus.write_text("\n".join(json.dumps(r) for r in lines))
    task = {"type": "file", "corpus": str(corpus),
            "path": str(tasks_file(tmp_path, n=1))}
    with pytest.raises(ConfigError, match=bad):
        from_config(cfg(tmp_path, task=task))


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


def test_d21_cli_exits_nonzero_on_a_halted_run(tmp_path, monkeypatch, capsys):
    """D21.  Contract: a run halted over budget returns 1 from the CLI and
    writes no sink.  Cost hand-computed: calibration measures 100 in / 5
    out tokens per call, so 4 units at $1000 per 1M tokens cost
    (400 + 20) * 1000 / 1e6 = $0.42 > max_cost $0.01."""
    monkeypatch.setitem(REGISTRIES["backend"].factories, "counting",
                        lambda: FunctionBackend(Counting(),
                                                identity={"model": "c"}))
    path = tmp_path / "c.json"
    path.write_text(json.dumps(cfg(tmp_path, backend={"type": "counting"},
                                   grid={"model": ["c"]},
                                   options={"workers": 1, "max_cost": 0.01,
                                            "prices": {"in": 1000.0,
                                                       "out": 1000.0}})))
    assert main(["run", str(path)]) == 1
    assert "halted: estimated cost $0.42 exceeds" in capsys.readouterr().out
    assert not (tmp_path / "runs" / "r.jsonl").exists()


@pytest.fixture
def toy_lm(monkeypatch):
    """The HF loader replaced by a toy causal LM, so nothing is downloaded."""
    pytest.importorskip("torch")
    from support.toy_hf import CharTok, KeepLM

    monkeypatch.setattr(config_mod, "_hf_load", lambda *a: (
        KeepLM(len(CharTok().vocab)), CharTok()))


def test_a_local_model_task_from_config(tmp_path, toy_lm):
    """Teacher-forced arithmetic through the whole pipeline on a toy model."""
    res = from_config(cfg(tmp_path, task={"type": "arithmetic", "n_items": 6},
                          backend={"type": "hf", "model": "toy"},
                          grid={}, tags={"model": "toy"},
                          metric="em_tf")).run()
    assert len(res.frame) == 6 and (res.frame["status"] == "ok").all()
    assert set(res.calibration.throughput) >= {1, 2}      # batch ladder ran
    assert res.settings["workers"] == 1
    # the label is a column in every row and stays out of the cell id
    assert (res.frame["model"] == "toy").all()
    assert (res.frame["cell_id"] == Cell({}).id).all()


def test_a_label_in_an_hf_grid_is_refused_before_any_call(tmp_path, toy_lm):
    """HF reads no "model" param (the model is loaded), so a grid axis of
    labels would make cells that differ in id and are the same request; the
    refusal names the key, at construction, before calibration spends."""
    with pytest.raises(ValueError, match=r"6 of 6 requests.*'model'.*tags"):
        from_config(cfg(tmp_path, task={"type": "arithmetic", "n_items": 6},
                        backend={"type": "hf", "model": "toy"},
                        grid={"model": ["toy"]}, metric="em_tf"))
    assert not (tmp_path / "runs" / "r.jsonl").exists()   # no sink, no call


def test_tags_must_be_a_mapping(tmp_path):
    with pytest.raises(ConfigError, match="tags"):
        from_config(cfg(tmp_path, tags=["model", "echo"]))


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
        {"id": "q1", "prompt": "where do cats sit", "gold_docs": ["c"]},
        {"id": "q2", "prompt": "dogs on mats", "gold_docs": ["d"]}]))
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
    from support.toy_hf import EmbedTok, MeanEncoder
    monkeypatch.setattr(config_mod, "_hf_load",
                        lambda *a: (MeanEncoder(), EmbedTok()))
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
