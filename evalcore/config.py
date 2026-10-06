"""An Experiment from a JSON file -- the route for runs that need no code.

    {"out": "runs/add_d2.jsonl",
     "task":    {"type": "arithmetic", "digits": 2, "n_items": 512},
     "backend": {"type": "hf", "model": "EleutherAI/pythia-70m"},
     "grid":    {"model": ["pythia-70m"]},
     "repeats": 1,
     "metric":  "em_tf",
     "options": {"cache_dir": "cache", "max_cost": 5.0,
                 "prices": {"in": 1.0, "out": 5.0}},
     "stages":  {"calibrate": false}}

Every component is a spec {"type": <name>, ...}, built by the registry of
its kind (registry.py): the keys are the factory's arguments, checked
against its signature.  `python -m evalcore list` prints every registered
name with its keys.  The built-ins:

  task      file (path, name, mapping, corpus), arithmetic, log_qa, search_qa
  backend   http (flavour, default_params, base_url, timeout), hf (model,
            revision, dtype, device, generate, default_params), echo,
            retriever (retriever, k)
  score     graded (the items' graders; the default), ranked (ks),
            teacher_forced
  retriever bm25, dense (embedder, index, cache_dir), rerank (first,
            pair_scorer, depth)
  embedder  http (flavour, model, dimensions), hf (model, pooling, ...)
  index     exact, faiss
  pair_scorer  hf (a cross-encoder)
  analysis  attrition, summary, retrieval, value_curve, axis_curve

Top-level keys:

  out, task, backend, grid       required
  repeats                        default 1
  score                          how rows are scored, if not the task's own
  metric | analyses              "metric": "correct" is shorthand for
                                 attrition + summary; "analyses" is
                                 {name: analysis spec}
  agent                          {"max_turns", "max_tool_calls"}, used when
                                 the task brings tools
  search                         {"retriever": R, "k": 3}: the agent's
                                 search tool over the task's corpus
  options                        RunOptions fields
  stages                         {stage: false} switches a stage off;
                                 {stage: spec} replaces it with a
                                 registered stage
  plugins                        modules or .py files to import first, so
                                 their @register factories exist

Your own components register under a name and are then used like the
built-ins (see registry.py).  A retriever backend needs a score: give
"score": {"type": "ranked", "ks": [...]}.
"""

from __future__ import annotations

import importlib
import importlib.util
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

from . import analyses as an
from .core.agent import AgentBackend
from .core.backends import Backend, EchoBackend, HFBackend, HTTPBackend
from .core.embed import Embedder, EmbeddingCache, HFEmbedder, HTTPEmbedder
from .core.search import (
    BM25,
    CorpusSearch,
    DenseRetriever,
    ExactIndex,
    FaissIndex,
    HFCrossEncoder,
    Index,
    PairScorer,
    Reranker,
    Retriever,
    RetrieverBackend,
)
from .core.spec import grid
from .core.tools import Tool
from .evaluation.retrieval import score_ranked
from .pipeline import Experiment, RunOptions, Stage
from .registry import ConfigError, build, register
from .tasks.arithmetic import arithmetic_task
from .tasks.base import Task
from .tasks.graders import make_score, score_teacher_forced
from .tasks.log_qa import log_qa_task
from .tasks.search_qa import search_qa

__all__ = ["ConfigError", "TaskSpec", "from_config"]

_TOP = {"out", "task", "backend", "grid", "repeats", "score", "metric",
        "analyses", "options", "stages", "agent", "search", "plugins"}


def _check(section: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(section) - allowed
    if unknown:
        raise ConfigError(f"{where}: unknown key(s) {sorted(unknown)}; "
                          f"allowed: {sorted(allowed)}")


@dataclass
class TaskSpec:
    """What a task factory returns when a Task alone is not enough: the
    corpus it searches (for retrievers and the search tool) and the tools
    it brings.  A factory may return a plain Task instead."""

    task: Task
    docs: dict[str, str] | None = None
    tools: list[Tool] = field(default_factory=list)


# --- tasks --------------------------------------------------------------------

def _corpus(path: str) -> dict[str, str]:
    docs: dict[str, str] = {}
    for n, line in enumerate(Path(path).read_text().splitlines(), 1):
        if not line.strip():
            continue
        rec = json.loads(line)
        if set(rec) != {"id", "text"}:
            raise ConfigError(f"corpus {path}:{n}: each line must be exactly "
                              f'{{"id", "text"}}, got {sorted(rec)}')
        if str(rec["id"]) in docs:
            raise ConfigError(f"corpus {path}:{n}: duplicate id {rec['id']!r}")
        docs[str(rec["id"])] = str(rec["text"])
    return docs


@register("task", "file")
def _file_task(path: str, name: str | None = None,
               mapping: Mapping[str, Any] | None = None,
               corpus: str | None = None) -> TaskSpec:
    """Canonical (or, with `mapping`, foreign) JSONL; `corpus` is a JSONL
    of {"id", "text"} for tasks scored by retrieval or searched by tool."""
    return TaskSpec(Task.from_file(path, name, mapping),
                    _corpus(corpus) if corpus is not None else None)


register("task", "arithmetic")(arithmetic_task)
register("task", "log_qa")(log_qa_task)


@register("task", "search_qa")
def _search_qa(n_towns: int = 20, seed: int = 0, k: int = 3) -> TaskSpec:
    task, search = search_qa(n_towns, seed, k)
    return TaskSpec(task, search.docs, [search.tool])


# --- backends -----------------------------------------------------------------

@register("backend", "http")
def _http(flavour: str, default_params: Mapping[str, Any] | None = None,
          base_url: str | None = None, timeout: float = 120.0) -> Backend:
    """API key from the environment ($ANTHROPIC_API_KEY, ...)."""
    return HTTPBackend(flavour, base_url=base_url, timeout=timeout,
                       default_params=default_params)


def _hf_load(spec: Mapping[str, Any], head: str) -> tuple[Any, Any]:
    """A transformers model and tokenizer at `revision`, on `device`."""
    import torch
    import transformers as tf
    cls = {"causal": tf.AutoModelForCausalLM, "encoder": tf.AutoModel,
           "pair": tf.AutoModelForSequenceClassification}[head]
    rev = spec.get("revision")
    model = cls.from_pretrained(spec["model"], revision=rev,
                                dtype=getattr(torch, spec["dtype"])).eval()
    if spec.get("device"):
        model = model.to(spec["device"])
    return model, tf.AutoTokenizer.from_pretrained(spec["model"], revision=rev)


@register("backend", "hf")
def _hf(model: str, revision: str | None = None, dtype: str = "float32",
        device: str | None = None, generate: bool = False,
        default_params: Mapping[str, Any] | None = None) -> Backend:
    m, tok = _hf_load({"model": model, "revision": revision, "dtype": dtype,
                       "device": device}, "causal")
    return HFBackend(m, tok, device=device, generate=generate,
                     default_params=default_params)


@register("backend", "echo")
def _echo(latency: float = 0.0) -> Backend:
    """Reverses the prompt: offline plumbing checks."""
    return EchoBackend(latency=latency)


@register("backend", "retriever")
def _retriever_backend(retriever: Mapping[str, Any], k: int, *,
                       task: TaskSpec) -> Backend:
    """A retriever evaluated on its own over the task's corpus."""
    if task.docs is None:
        raise ConfigError("backend: a retriever needs a corpus (search_qa, or "
                          "a file task with 'corpus')")
    return RetrieverBackend(build("retriever", retriever, "backend.retriever",
                                  docs=task.docs), k=k)


# --- scores -------------------------------------------------------------------

@register("score", "graded")
def _graded(default: str | Mapping[str, Any] | None = None) -> Any:
    """Each item's own grader and gold (`default` for items naming none)."""
    return make_score(default=default)


@register("score", "ranked")
def _ranked(ks: Sequence[int], *, task: TaskSpec) -> Any:
    """recall@k, hit@k, nDCG@k for k in ks, and RR, against gold_docs."""
    missing = [it.item_id for it in task.task.items
               if "gold_docs" not in it.payload]
    if missing:
        raise ConfigError(f"score: {len(missing)} item(s) have no 'gold_docs' "
                          f"(e.g. {missing[0]!r}); ranked scoring needs them")
    return score_ranked(ks=ks)


register("score", "teacher_forced")(lambda: score_teacher_forced)


# --- retrieval components -----------------------------------------------------

@register("retriever", "bm25")
def _bm25(k1: float = 1.2, b: float = 0.75, *,
          docs: Mapping[str, str]) -> Retriever:
    return BM25(docs, k1, b)


@register("retriever", "dense")
def _dense(embedder: Mapping[str, Any], index: Mapping[str, Any] | None = None,
           cache_dir: str | None = None, ann_sample: int = 200, ann_k: int = 10,
           min_ann_recall: float = 0.95, *,
           docs: Mapping[str, str]) -> Retriever:
    """`index` defaults to exact search."""
    emb: Embedder = build("embedder", embedder, "retriever.embedder")
    idx: Index = build("index", index or {"type": "exact"}, "retriever.index")
    return DenseRetriever(docs, emb, idx,
                          cache=EmbeddingCache(cache_dir) if cache_dir else None,
                          ann_sample=ann_sample, ann_k=ann_k,
                          min_ann_recall=min_ann_recall)


@register("retriever", "rerank")
def _rerank(first: Mapping[str, Any], pair_scorer: Mapping[str, Any],
            depth: int, *, docs: Mapping[str, str]) -> Retriever:
    scorer: PairScorer = build("pair_scorer", pair_scorer,
                               "retriever.pair_scorer")
    return Reranker(build("retriever", first, "retriever.first", docs=docs),
                    scorer, docs, depth)


@register("embedder", "http")
def _http_embedder(flavour: str, model: str, dimensions: int | None = None,
                   base_url: str | None = None, timeout: float = 60.0,
                   batch_size: int = 100) -> Embedder:
    """API key from the environment, never from the config file."""
    return HTTPEmbedder(flavour, model, dimensions, base_url=base_url,
                        timeout=timeout, batch_size=batch_size)


@register("embedder", "hf")
def _hf_embedder(model: str, pooling: str, revision: str | None = None,
                 dtype: str = "float32", device: str | None = None,
                 prefixes: Mapping[str, str] | None = None,
                 max_length: int = 512) -> Embedder:
    m, tok = _hf_load({"model": model, "revision": revision, "dtype": dtype,
                       "device": device}, "encoder")
    return HFEmbedder(m, tok, pooling=pooling, prefixes=prefixes,
                      max_length=max_length, device=device)


register("index", "exact")(ExactIndex)
register("index", "faiss")(FaissIndex)


@register("pair_scorer", "hf")
def _hf_cross_encoder(model: str, revision: str | None = None,
                      dtype: str = "float32", device: str | None = None,
                      max_length: int = 512) -> PairScorer:
    m, tok = _hf_load({"model": model, "revision": revision, "dtype": dtype,
                       "device": device}, "pair")
    return HFCrossEncoder(m, tok, max_length=max_length, device=device)


# --- analyses -----------------------------------------------------------------
# Explicit signatures rather than **kwargs, so a misspelt key is refused
# when the config is read, not hours later in the analyse stage.

register("analysis", "attrition")(an.attrition_table)


@register("analysis", "summary")
def _summary(metric: str, by: Sequence[str] | None = None, n_boot: int = 5000,
             seed: int = 0, alpha: float = 0.05, method: str = "bca",
             label: str | None = None, floor: Mapping[str, float] | None = None,
             higher_is_better: bool = True) -> an.Analysis:
    return an.summary(metric, by, n_boot=n_boot, seed=seed, alpha=alpha,
                      method=method, label=label, floor=floor,
                      higher_is_better=higher_is_better)


@register("analysis", "retrieval")
def _retrieval(metrics: Sequence[str] | None = None,
               by: Sequence[str] | None = None, n_boot: int = 5000,
               seed: int = 0, alpha: float = 0.05,
               method: str = "bca") -> an.Analysis:
    return an.retrieval_table(metrics, by, n_boot=n_boot, seed=seed,
                              alpha=alpha, method=method)


@register("analysis", "value_curve")
def _value_curve(x: str, value: str, levels: Sequence[float] = (),
                 n_boot: int = 1000, seed: int = 0, alpha: float = 0.05,
                 log_x: bool = True) -> an.Analysis:
    return an.value_curve(x, value, levels, n_boot=n_boot, seed=seed,
                          alpha=alpha, log_x=log_x)


@register("analysis", "axis_curve")
def _axis_curve(x: str, n_boot: int = 1000, seed: int = 0,
                alpha: float = 0.05, log_x: bool = True) -> an.Analysis:
    return an.axis_curve(x, n_boot=n_boot, seed=seed, alpha=alpha, log_x=log_x)


# --- the experiment -----------------------------------------------------------

def _plugins(names: Sequence[str], base: Path | None) -> None:
    """Import each plugin (a module name, or a .py path relative to the
    config file), so its @register factories exist before anything is
    built."""
    for name in names:
        if name.endswith(".py"):
            path = Path(name) if base is None else base / name
            if not path.exists():
                raise ConfigError(f"plugins: no file {str(path)!r}")
            mod_name = f"evalcore_plugin_{path.stem}"
            spec = importlib.util.spec_from_file_location(mod_name, path)
            assert spec is not None and spec.loader is not None
            module = importlib.util.module_from_spec(spec)
            sys.modules[mod_name] = module
            spec.loader.exec_module(module)
        else:
            importlib.import_module(name)


def _stages(spec: Mapping[str, Any]) -> dict[str, Stage | None]:
    _check(spec, set(Experiment.STAGES), "stages")
    out: dict[str, Stage | None] = {}
    for name, s in spec.items():
        if s is False:
            out[name] = None
        elif s is True:
            raise ConfigError(f"stages.{name}: true is the default; give "
                              f"false (off) or a registered stage")
        else:
            out[name] = build("stage", s, f"stages.{name}")
    return out


def from_config(source: str | Path | Mapping[str, Any]) -> Experiment:
    """An Experiment from a config file or dict (see the module docstring)."""
    base = Path(source).parent if isinstance(source, (str, Path)) else None
    cfg = (json.loads(Path(source).read_text())
           if isinstance(source, (str, Path)) else dict(source))
    _check(cfg, _TOP, "config")
    for key in ("out", "task", "backend", "grid"):
        if key not in cfg:
            raise ConfigError(f"config: '{key}' is required")
    _plugins(cfg.get("plugins", []), base)

    made = build("task", cfg["task"], "task")
    ts = made if isinstance(made, TaskSpec) else TaskSpec(made)
    backend: Backend = build("backend", cfg["backend"], "backend", task=ts)
    retrieving = isinstance(backend, RetrieverBackend)
    if retrieving and "score" not in cfg:
        raise ConfigError('score: a retriever backend is scored by ranking; '
                          'give "score": {"type": "ranked", "ks": [...]}')
    task = (replace(ts.task, score=build("score", cfg["score"], "score",
                                         task=ts))
            if "score" in cfg else ts.task)

    tools = [] if retrieving else list(ts.tools)
    if retrieving and ("agent" in cfg or "search" in cfg):
        raise ConfigError("agent / search: not used when the backend is a "
                          "retriever")
    if "search" in cfg:
        search = dict(cfg["search"])
        _check(search, {"retriever", "k"}, "search")
        for key in ("retriever", "k"):
            if key not in search:
                raise ConfigError(f"search: '{key}' is required")
        if ts.docs is None or not tools:
            raise ConfigError("search: this task has no corpus to search")
        tools = [CorpusSearch(ts.docs, build("retriever", search["retriever"],
                                             "search.retriever", docs=ts.docs),
                              k=search["k"]).tool]
    if tools:
        agent = dict(cfg.get("agent", {}))
        _check(agent, {"max_turns", "max_tool_calls"}, "agent")
        backend = AgentBackend(backend, tools, **agent)
    elif "agent" in cfg:
        raise ConfigError("agent: this task brings no tools")

    if "metric" in cfg and "analyses" in cfg:
        raise ConfigError("give 'metric' (shorthand) or 'analyses', not both")
    analyses = ({name: build("analysis", spec, f"analyses.{name}")
                 for name, spec in cfg["analyses"].items()}
                if "analyses" in cfg else None)
    opts = dict(cfg.get("options", {}))
    _check(opts, set(RunOptions.__dataclass_fields__), "options")
    return Experiment(task, backend, grid(cfg["grid"]), cfg["out"],
                      repeats=cfg.get("repeats", 1), metric=cfg.get("metric"),
                      analyses=analyses, options=RunOptions(**opts),
                      **_stages(cfg.get("stages", {})))
