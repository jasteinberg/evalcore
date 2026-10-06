# evalcore

A small, backend-agnostic harness for running evaluations of language
models and analysing them honestly.

Drafted with the assistance of Claude (Anthropic).

The same core runs a rate-limited API sweep and a batched local forward
pass with activation capture. Every model call is content-addressed, so a
sweep that dies eight hours in resumes where it stopped, and nothing is
paid for twice. Failures are recorded as rows rather than dropped, and every
interval resamples the right unit (the cluster, not the row), so a table
says what the data can actually resolve.

## Quick start

```bash
python -m pip install -e ".[http,dev]"     # extras: http, hf, retrieval, dev
pytest -q
```

From Python:

```python
from evalcore import Experiment, HTTPBackend, RunOptions, grid
from evalcore.tasks.log_qa import log_qa_task

exp = Experiment(
    task=log_qa_task(n_traces=20),
    backend=HTTPBackend("anthropic", default_params={"max_tokens": 200}),
    cells=grid({"model": ["claude-haiku-4-5"]}),
    out="runs/log_qa.jsonl",
    metric="correct",
    options=RunOptions(prices={"in": 1.0, "out": 5.0}, max_cost=2.0),
)
res = exp.run()        # calibrate -> dry run -> execute -> analyse -> report
print(res.analysis["summary"])
```

Or with no code, from a JSON file (the format is documented at the top of
`evalcore/config.py`):

```bash
python -m evalcore run experiment.json --dry-run   # plan and cost only
python -m evalcore run experiment.json             # halts above max_cost unless --yes
python -m evalcore summarize runs/log_qa.jsonl     # what the run did, from its event log
python -m evalcore list                            # every component a config can name, with its keys
```

A config names every component by type, with that component's own
arguments:

```json
{"out": "runs/log_qa.jsonl",
 "task":    {"type": "log_qa", "n_traces": 20},
 "backend": {"type": "http", "flavour": "anthropic",
             "default_params": {"max_tokens": 200}},
 "grid":    {"model": ["claude-haiku-4-5"]},
 "metric":  "correct",
 "options": {"prices": {"in": 1.0, "out": 5.0}, "max_cost": 2.0}}
```

## The harness loop

```
  cells (grid) x items (Task) x repeats
        │
        ▼
  units ──────────► plan()  render each unit to a Request, key it, compare
                      │     with the sink: done / stale / errored / new
                      ▼
               manifest written (what was planned, before any call)
                      │
        ┌─────────────┴──────────────── for each batch of units to do ──┐
        │  Request ──► capability check ──► Cache ──hit──► Response     │
        │                                     │miss                      │
        │                                     ▼                          │
        │                 Backend.complete_batch (with retries)          │
        │                 [HTTP │ HF + Probe │ Agent+tools │ Retriever]  │
        │                                     │                          │
        │        arrays ──► .npz sidecar      ▼                          │
        │                               Response ──► Cache               │
        │                                     │                          │
        │      status: ok / error / tool_error / truncated / unparsed    │
        │      score(unit, response)   (only for ok rows)                │
        │                                     │                          │
        │      row ──► sink.jsonl (append, flushed)   event ──► events   │
        └────────────────────────────────────────────────────────────────┘
                      │
                      ▼
  to_frame ──► attrition (+ missing, from the manifest)
           ──► summarize (cluster bootstrap, n_eff, caveats)
           ──► curves / retrieval metrics / paired comparisons ──► run.json
```

The `Experiment` in `pipeline.py` wraps this loop in five stages, each a
plain function `stage(exp, res) -> None` that can be replaced, or switched
off with `None`:

| stage | what it does |
|---|---|
| `calibrate` | a few real calls on the experiment's own requests; latency, tokens, rate-limit headers, or the throughput of each batch size for a local model; derives workers and batch size and lists every assumption |
| `dry_run` | the resume decisions and the estimated spend without calling anything; halts above `max_cost` |
| `execute` | the loop above |
| `analyse` | every analysis in `analyses` (default: attrition, plus the bootstrap summary of `metric`), each written as CSV |
| `report` | `run.json`: the experiment, options, calibration, plan, event summary, outputs and provenance (evalcore commit, library versions) |

`execute=None` re-analyses an existing sink without touching the model.
What `analyse` computes is pluggable too: `analyses={"curve":
value_curve("N", "logp", levels=[0.0]), "mine": my_fn}` takes any function
`(frame, ctx) -> table`, and `analyses.py` has attrition, summary,
retrieval, value curves and axis curves ready-made.

## Where things live

The package has three layers. Each can be imported on its own, and nothing
above `core/backends` knows whether a model is reached over HTTP or run
locally.

**`core/`: how calls are made, keyed, cached, recorded and resumed**

| module | contents |
|---|---|
| `spec.py` | `Cell`, `Item`, `Unit`, `grid`, `digest`: content-addressed identity |
| `backends/base.py` | `Request`, `Response`, the `Backend` contract, `request_key`, retries |
| `backends/http.py` | `HTTPBackend`: Anthropic, OpenAI and Gemini, including tool calls |
| `backends/hf.py` | `HFBackend`: local generation, teacher-forced scoring, candidate scoring |
| `backends/probe.py` | `Probe`, `FunctionProbe`: capture from inside each forward pass |
| `backends/cache.py` | `Cache`: one JSON file per request key |
| `backends/echo.py`, `function.py` | `EchoBackend`, `FunctionBackend`: offline and user-supplied models |
| `backends/check.py` | `check_backend`: a conformance check for a backend you write |
| `runner.py` | `plan()` and `execute()`: resume, batching, status, scoring |
| `records.py` | the JSONL sink, the plan manifest, `.npz` array sidecars, `to_frame` |
| `events.py` | the run's event log; `summarize_events`; `parse_log` for foreign logs |
| `calibrate.py` | `calibrate`, `derive` |
| `agent.py`, `tools.py` | `AgentBackend` (the multi-turn tool loop), `Tool` |
| `hooks.py` | `hooked`: forward hooks that must match, are counted, and are always removed |
| `embed.py` | `Embedder` (function, HF, OpenAI, Gemini) and the per-text vector cache |
| `search.py` | `BM25`, `ExactIndex`, `FaissIndex`, `DenseRetriever`, `Reranker`, `CorpusSearch`, `RetrieverBackend` |

**`tasks/`: what is asked and how it is scored**

| module | contents |
|---|---|
| `base.py` | `Task`: items, `render(unit) -> Request`, `score(unit, response) -> dict` |
| `formats.py` | the canonical task format; `load_tasks` translates foreign JSON with a field mapping |
| `graders.py` | `exact`, `contains`, `numeric`, `choice`, `yes_no`, `set_f1`, `abstain`, `label`; `score_teacher_forced` |
| `arithmetic.py` | addition by teacher forcing (the emergence experiments) |
| `log_qa.py` | questions over generated execution traces, with exact gold |
| `search_qa.py` | questions answerable only by searching an invented corpus; carries the gold documents |

**`evaluation/`: what the numbers mean**

| module | contents |
|---|---|
| `stats.py` | cluster and paired bootstrap, ICC, design effect, `n_eff`, Wilson, McNemar, Holm and Benjamini-Hochberg, `summarize` |
| `attrition.py` | per-cell accounting of the plan in six states |
| `curves.py` | curves along a model axis: susceptibility, composition, crossings, with item-paired intervals |
| `retrieval.py` | recall@k, hit@k, nDCG@k, MRR |
| `baselines.py`, `contamination.py`, `plots.py` | floors to beat, n-gram overlap checks, figures |

**Top level**

| module | contents |
|---|---|
| `pipeline.py` | `Experiment`: the stages in order, each replaceable |
| `analyses.py` | what the analyse stage computes; any function of the frame |
| `registry.py` | named factories for every component kind, yours included |
| `config.py` | an `Experiment` from JSON, built through the registry |
| `__main__.py` | the command line: `run`, `summarize`, `list` |

## Extension points

Everything model-specific is supplied by the caller. The core defines a
small interface at each point and checks it.

| to add | implement | notes |
|---|---|---|
| a model | `Backend.identity()` and `complete` (or `complete_batch`) | `identity()` is required: it keys the cache, so it must name everything that changes an output |
| a quick model | `FunctionBackend(fn, identity=...)` | `fn(request) -> str` |
| activation capture | `Probe.identity`, `hooks`, `collect` | hooks see the padded batch; `collect` returns one dict per request; arrays go to sidecars |
| a task | `Task(name, items, render, score)`, or a JSONL file | `Task.from_file(path, mapping=...)` for a foreign format |
| a grader | `@grader("name")` in `tasks/graders.py` | raise `ParseFailure` when the answer cannot be read |
| a tool | `Tool(name, description, parameters, fn)` | `ToolInputError` goes back to the model; `ToolUnavailable` ends the episode as `tool_error` |
| embeddings | `Embedder.identity` and `_embed` | the vector cache comes with it |
| a vector index | `Index.identity`, `build`, `search` | set `exact = False` and the recall cost is measured |
| a retriever | `identity` and `retrieve(queries, k)` | runs through the harness via `RetrieverBackend` |
| a reranker | `PairScorer.identity` and `score(query, docs)` | wrap in `Reranker(first, scorer, docs, depth)` |
| a pipeline stage | a function `(exp, res) -> None` | pass it as `Experiment(..., report=my_stage)` |
| an analysis | a function `(frame, ctx) -> DataFrame` (or a dict of them) | `ctx.by` is the grid axes; pass it in `analyses` |

### Your components in a config

Anything above can be named in a JSON config once it is registered. The
factory's signature is the schema: keys it does not take are refused by
name, and parameters without a default are required.

```python
# my_plugin.py
from evalcore import register

@register("backend", "vllm")
def vllm(url: str, model: str, timeout: float = 60.0):
    return MyVLLMBackend(url, model, timeout)

@register("analysis", "worst_items")
def worst_items(metric: str, n: int = 20):
    return lambda frame, ctx: frame.nsmallest(n, metric)
```

```json
{"plugins": ["my_plugin.py"],
 "backend": {"type": "vllm", "url": "http://localhost:8000", "model": "m"},
 "analyses": {"worst": {"type": "worst_items", "metric": "correct"}},
 "...": "..."}
```

The kinds are `task`, `backend`, `score`, `retriever`, `embedder`, `index`,
`pair_scorer`, `analysis` and `stage`. A factory that needs something from
the experiment declares it as a keyword-only parameter (`*, docs` for a
retriever's corpus, `*, task` for a scorer); the experiment supplies it,
and a config cannot override it.
## What the harness guarantees

- **One key per call.** `request_key = digest(request, backend.identity(),
  repeat)`. The cache, resume and every row use it. A changed prompt, model,
  revision, probe, default parameter or tool re-runs exactly the affected
  units, and a different call is never served this one's answer.
- **Scoring is separate from calling.** The cache holds responses, never
  scores, so changing a metric costs a re-score, not a re-run.
- **Nothing silently disappears.** Errors are rows. Every row has a status,
  only `ok` rows are scored, and `attrition` with the manifest accounts for
  every planned unit: `n_planned = ok + error + tool_error + truncated +
  unparsed + missing`.
- **Append-only, resumable.** Rows are flushed one at a time. A unit is done
  only if its last row is a result for the request it renders to now.
- **Numbers carry caveats.** A summary computed correctly but not to be read
  at face value says so in a `caveats` column and a warning: too few
  clusters for the interval to be honest, zero spread, many rows excluded,
  a crossing most replicates never reach, a calibration resting on two calls.

## The one idea

The unit of resampling is the **cluster**: a source item together with
every repeat and every augmentation derived from it, never the row. Under
`X_ij = mu + a_i + e_ij` with `k` clusters of `m` rows,

```
Var(theta) = (s2_a + s2_e/m) / k          honest
Var_naive  = (s2_a + s2_e) / (km)         row-resampling
DEFF = Var/Var_naive = 1 + (m-1) rho,     rho = s2_a / (s2_a + s2_e)
n_eff = N / DEFF
```

At `rho -> 1`, ten paraphrases of one question buy you one question; at
`rho -> 0` they buy ten. `summarize()` prints `n_eff` beside `n_rows` for
exactly this reason.

Measured on the simulated check in `tests/` (k=40, m=6, rho=0.44, DEFF=3.2,
N=240, n_eff=76), coverage of a nominal 95% interval:

| method | coverage |
|---|---|
| cluster bootstrap | 0.943 |
| row bootstrap | 0.687 |

The bootstrap needs enough clusters to see their spread. With few, its
interval is too narrow, and `summarize` flags the cell:

| clusters | coverage of a nominal 95% interval |
|---|---|
| 3 | 0.68 |
| 5 | 0.83 |
| 10 | 0.88 |
| 20 | 0.93 |
| 30+ | ~0.95 |

## Retrieval

A retriever is evaluated like a model. Any task whose items carry
`gold_docs` can be run against `RetrieverBackend`, and its rows are scored
with ranking metrics:

```python
from evalcore import BM25, RetrieverBackend, grid
from evalcore.evaluation.retrieval import retrieval_summary, score_ranked
from evalcore.tasks.base import Task
from evalcore.tasks.search_qa import make_world, search_qa_items

world = make_world(20)
task = Task("search_qa_retrieval", search_qa_items(world),
            score=score_ranked(ks=(1, 3, 10)))
df = task.run(grid({"retriever": ["bm25"]}), RetrieverBackend(BM25(world.docs), k=10),
              "runs/bm25.jsonl")
print(retrieval_summary(df, ["retriever"]))
```

Dense retrieval embeds the corpus once (vectors are cached per text), and
an approximate FAISS index has its recall measured against the exact
search, so an approximation shows up as a number. The same retrievers serve
as the agent's search tool (`CorpusSearch`).

FAISS (`pip install -e ".[retrieval]"`) and torch cannot share a process
when both come from pip wheels on macOS: each bundles its own OpenMP
runtime, and the process aborts. `FaissIndex` detects this and raises with
the remedy.

## Outputs

For `out="runs/x.jsonl"`:

| file | contents |
|---|---|
| `runs/x.jsonl` | one row per attempt (append-only) |
| `runs/x.jsonl.manifest.jsonl` | every planned unit, written before the first call |
| `runs/x.jsonl.events.jsonl` | run start and end, each unit, each retry |
| `runs/x.jsonl.arrays/` | `.npz` sidecars from probes (or under the cache dir, if there is one) |
| `runs/x.jsonl.calibration.json` | the calibration, reused while fresh |
| `runs/x.jsonl.analysis/*.csv` | attrition and summary tables |
| `runs/x.jsonl.run.json` | the report |

## Vendoring

The core depends only on numpy, pandas, scipy and matplotlib. `httpx`,
`torch` and `transformers`, and `faiss` are imported inside the classes
that use them. To ship evalcore inside a bundle someone else must run, copy
the `evalcore/` directory next to the entry script; no install step is
needed. **Carry `LICENSE` with it:** the MIT terms require the copyright
notice to travel with any substantial portion of the code.

## License

MIT; see [LICENSE](LICENSE). Copyright (c) 2026 Julia Steinberg.

This is a general-purpose tool, developed independently and not derived
from any third party's confidential material.
