# Glossary

Drafted with the assistance of Claude (Anthropic)

One page, covering the vocabulary used across this package. Several terms are
overloaded, and six collide badly enough to cause a wrong answer if left
ambiguous. Those are listed at the end.

---

## Configuration

**Cell** — one point in the configuration space: model x prompt variant x
decode params x dataset slice x seed. Carries `params` (identity-bearing) and
`tags` (labels that must not affect identity).

**Item** — one datapoint to be evaluated, independent of configuration. May
carry `parent_id`, which places it in an augmentation cluster with its parent.

**Unit** — one `(cell, item, repeat)` triple. Produces exactly one model call
and exactly one row in the tidy frame. The atom of the runner.

**Grid** — expansion of named axes into cells, with exclusions.

**`cell_id` / `unit_id`** — content-addressed identity: blake2b of the
canonical JSON of the identity-bearing fields. The cache key, the resume key,
and the analysis join key are the same string, which is what lets a partial
run merge with a later one.

**Axis** — one named dimension of the grid (`model`, `temperature`, `template`).

**Arm** — a set of cells being contrasted. An arm is a comparison you intend
to make, not a structure the package enforces.

## Execution

**Backend** — maps requests to responses. Knows nothing about cells, items,
clusters, or metrics.

**Flavour** — which provider dialect an `HTTPBackend` speaks (`anthropic`,
`openai`, `gemini`). Not the model.

**Request / Response** — the transport-level pair. One unit produces one
request.

**`params`** — everything on a cell that can change model behaviour. A
superset of the decode params: it also carries `model`.

**Decode params** — the subset of `params` listed in `DECODE_KEYS` that changes
what the model emits: `temperature`, `top_p`, `top_k`, `max_tokens`, `seed`,
`stop`, `stop_sequences`. Cells that will be differenced must agree on these.

**`default_params`** — backend-level fallbacks. Request params win; nothing is
invented if neither supplies a value.

**Transient / Fatal** — retry classification, decided once in the backend
rather than at each call site.

**Attrition** — planned units that did not become usable rows, tabulated by
cell. Three states, and they partition the plan: completed, errored (errors are
rows, not exceptions), and *never attempted*. The third is visible only against
the manifest, because a cell that died before its first call has no rows to be
counted in.

**Manifest** — the plan, written to disk before the first call: one record per
planned unit carrying the same identity columns as the frame, so it joins on
`unit_id`. A record of intent, not a second source of truth — resume reads the
JSONL, and a stale or missing manifest costs information, never correctness.

**Truncated** — the generation stopped because it hit `max_tokens`, not
because it finished. Normalised across providers into `meta["truncated"]`.

## Analysis

**Score** — the per-row number a scorer assigns to one response.

**Metric** — a scalar computed over many rows (accuracy, mean score, pass@k).

**Estimand** — the population quantity a metric is meant to estimate. The
distinction matters when the sampling unit and the reporting unit differ.

**Cluster** — the resampling unit for the bootstrap. Defaults to the item, and
becomes the augmentation family when `parent_id` is set.

**ICC(1)** — intraclass correlation: the fraction of score variance that is
between clusters rather than within them.

**`n_eff`** — effective sample size under clustering,
`n_eff = N / (1 + (m-1) rho)`. Printed beside `n_rows` because the two differ
by a factor that is easy to forget and hard to notice.

**BCa** — bias-corrected and accelerated bootstrap interval.

**Paired bootstrap** — resamples clusters once and differences two arms within
each resample, so the correlation between arms is retained.

---

## The six that collide

**`n`.** Three different things. `n_rows` is rows in the frame. `n_eff` is the
effective sample size after clustering, always smaller. Bare `n` in
`wilson_interval(k, n)` and `digest(obj, n)` is a trial count and a digest
length respectively — neither is a sample size in the statistical sense. Never
write bare `n` in a results table.

**Unit.** In this package a *unit* is a `(cell, item, repeat)` triple. In the
statistics literature the "unit" is usually the thing you resample, which here
is the **cluster**. When reading `n_eff`, "unit" means cluster; when reading
the runner, it means one model call.

**`params`.** Cell params (everything identity-bearing, including `model`) are
a superset of decode params (`DECODE_KEYS`, which exclude `model`). The
comparability guarantee is stated over decode params only — two cells differing
in `model` are still comparable, which is usually the whole point.

**Run.** Either one execution of a sweep, or one model call. Prefer *sweep* for
the former and *request* or *call* for the latter.  Three functions run
things, at three levels: `Experiment.run()` runs the pipeline stages,
`Task.run()` runs one task's units, and `runner.execute()` is the loop
underneath both.

**rho.** The intraclass correlation in `icc1` and `effective_n`. Not a
correlation between arms, and not a rank correlation. If a document needs both,
name the second one explicitly.

**Score / metric.** A score is per-row, a metric is per-cell. `summarize`
consumes scores and emits metrics; a "score" appearing in a results table with
a confidence interval is a metric that has been mislabelled.

---

## Naming rules

- Anything reported in a table carries its denominator: `n_rows`, `n_eff`,
  `n_clusters` — never bare `n`.
- Symbols that appear in both prose and code use the code spelling in prose
  (`n_eff`, not $n_{\text{eff}}$) so the two are greppable together.
- A quantity that exists in two versions gets two names, not one name and a
  qualifier: `score` and `metric`, never "row-level metric".
