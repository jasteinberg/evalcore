"""evalcore -- a small, backend-agnostic evaluation harness.

Start here:

    from evalcore import Experiment, Task, grid
    exp = Experiment(task, backend, grid({"model": [...]}), out="runs/x.jsonl")
    exp.run()     # calibrate -> dry run -> execute -> analyse -> report

or, with no code, `python -m evalcore run experiment.json` (see config.py).

The package in three layers, each importable on its own:

  core/        how model calls are made, keyed, cached, recorded, resumed
    spec         Cell, Item, Unit: content-addressed identity, grids
    backends/    Request/Response, the Backend contract, HTTP (three
                 providers), local HF (teacher forcing, candidate scoring,
                 probes), the cache, conformance checks
    runner       plan() then execute(): resume decisions, retries, rows, status
    records      the append-only sink, plan manifest, array sidecars
    events       the run's event log: read, summarise, parse foreign logs
    calibrate    measure a backend, derive workers / batch size
    agent, tools multi-turn tool use; hooks: forward hooks done safely
    embed        embedders (function, HF, OpenAI / Gemini), the vector cache
    search       BM25, exact and FAISS indexes, dense retrieval, reranking,
                 the search tool, retrievers as backends
  tasks/       what is asked and how it is scored
    base, formats, graders   Task, the canonical task format, graders
    arithmetic, log_qa, search_qa   built-in generated tasks
  evaluation/  what the numbers mean
    stats        cluster/paired bootstrap, ICC, n_eff, multiple comparisons
    curves       curves along a model axis: chi, composition, crossings
    attrition    the six-state accounting of a plan
    retrieval    recall@k, hit@k, nDCG@k, MRR with cluster intervals
    baselines, contamination, plots

  pipeline     Experiment: the stages in order, each replaceable
  analyses     what the analyse stage computes: attrition, summary,
               retrieval, curves, or any function of the frame
  config       an Experiment from JSON
  registry     named factories: any component, yours included, by name

Nothing above `core/backends` knows whether a model is reached over HTTP
or run locally.
"""


from .config import from_config
from .core.agent import AgentBackend
from .core.backends import (
    DECODE_KEYS,
    Backend,
    Cache,
    CheckReport,
    EchoBackend,
    Fatal,
    FunctionBackend,
    FunctionProbe,
    HFBackend,
    HTTPBackend,
    Probe,
    Request,
    Response,
    Transient,
    check_backend,
    with_retries,
)
from .core.calibrate import calibrate, derive
from .core.embed import EmbeddingCache, FunctionEmbedder, HFEmbedder, HTTPEmbedder
from .core.events import (
    format_summary,
    parse_log,
    read_events,
    summarize_events,
)
from .core.records import (
    load_arrays,
    manifest_path,
    read_manifest,
    to_frame,
    write_manifest,
)
from .core.runner import ParseFailure, execute, plan
from .core.search import (
    BM25,
    CorpusSearch,
    DenseRetriever,
    ExactIndex,
    FaissIndex,
    FunctionScorer,
    HFCrossEncoder,
    Reranker,
    RetrieverBackend,
    ann_recall,
    search_tool,
)
from .core.spec import Cell, Item, Unit, canonical, digest, grid, units
from .core.tools import Tool, ToolInputError, ToolUnavailable
from .evaluation.attrition import attrition
from .evaluation.baselines import (
    floors,
    floors_from_frame,
    headroom,
    majority_rate,
    prior_matched_rate,
)
from .evaluation.contamination import longest_match, ngram_overlap
from .evaluation.contamination import report as contamination_report
from .evaluation.curves import (
    axis_bootstrap,
    axis_curves,
    axis_data,
    composition,
    crossing,
    explode_tokens,
    subsampled_sharpness,
    susceptibility,
    value_curves,
)
from .evaluation.retrieval import rank_metrics, retrieval_summary, score_ranked
from .evaluation.stats import (
    BootResult,
    benjamini_hochberg,
    cluster_bootstrap,
    effective_n,
    holm,
    icc1,
    mcnemar_exact,
    paired_bootstrap,
    paired_pvalue,
    summarize,
    wilson_interval,
)
from .pipeline import Experiment, RunOptions
from .registry import ConfigError, register
from .tasks.base import Task
from .tasks.formats import load_tasks, save_tasks
from .tasks.graders import grade, make_score

__version__ = "0.1.0"

__all__ = [  # noqa: RUF022 - grouped by layer, for readers
    # the pipeline
    "Experiment", "RunOptions", "from_config", "register", "ConfigError",
    # identity and grids
    "Cell", "Item", "Unit", "grid", "units", "digest", "canonical",
    # backends
    "Backend", "Request", "Response", "Transient", "Fatal", "DECODE_KEYS",
    "HTTPBackend", "HFBackend", "EchoBackend", "FunctionBackend",
    "AgentBackend", "Probe", "FunctionProbe", "Cache", "check_backend",
    "CheckReport", "with_retries",
    # tools
    "Tool", "ToolInputError", "ToolUnavailable",
    # retrieval
    "EmbeddingCache", "FunctionEmbedder", "HFEmbedder", "HTTPEmbedder",
    "BM25", "CorpusSearch", "DenseRetriever", "ExactIndex", "FaissIndex",
    "RetrieverBackend", "Reranker", "FunctionScorer", "HFCrossEncoder",
    "ann_recall", "search_tool", "rank_metrics", "score_ranked",
    "retrieval_summary",
    # running and records
    "plan", "execute", "ParseFailure", "calibrate", "derive",
    "to_frame", "load_arrays", "manifest_path", "read_manifest",
    "write_manifest",
    # events
    "read_events", "summarize_events", "format_summary", "parse_log",
    # tasks
    "Task", "load_tasks", "save_tasks", "grade", "make_score",
    # statistics
    "BootResult", "cluster_bootstrap", "paired_bootstrap", "paired_pvalue",
    "icc1", "effective_n", "wilson_interval", "mcnemar_exact",
    "benjamini_hochberg", "holm", "summarize", "attrition",
    # curves
    "axis_curves", "axis_data", "axis_bootstrap", "value_curves", "crossing",
    "composition", "susceptibility", "subsampled_sharpness", "explode_tokens",
    # floors and contamination
    "floors", "floors_from_frame", "headroom", "majority_rate",
    "prior_matched_rate", "contamination_report", "ngram_overlap",
    "longest_match",
]
