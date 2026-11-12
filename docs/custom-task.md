# Writing your own task

Drafted with the assistance of Claude (Anthropic).

A task is three things: the items, how each item becomes a request
(`render`), and how a response is scored (`score`). This guide builds one
task four ways, from a data file with no code to a module inside evalcore.
Each step adds one thing to the step before it, so stop at the first step
that does what you need.

| step | you write | the task is used from |
|---|---|---|
| 1. A task file | a JSONL file | a config (`"type": "file"`) |
| 2. A task in Python | a function that returns a `Task` | Python |
| 3. A named task | the same function, registered | a config (`"type": "letter_count"`) |
| 4. Your own render and score | `render`, a grader | either |
| 5. A built-in task | a module in `evalcore/tasks/` and a test | either |

The running example asks how many times a letter occurs in a word. The gold
answer is `word.count(letter)`, so it is known exactly. Every file below is
complete, and `tests/test_custom_task_doc.py` writes each one out and runs
it, so the code in this guide is the code that is tested.

## 1. A task file

One JSON record per line, in the canonical format (documented at the top of
`evalcore/tasks/formats.py`):

<!-- file: letters.jsonl -->
```json
{"id": "w0", "prompt": "How many times does the letter 'r' occur in the word 'strawberry'? Answer with a number.", "gold": 3, "grader": "numeric"}
{"id": "w0-p", "prompt": "Count the occurrences of 'r' in \"strawberry\". Reply with the count only.", "gold": 3, "grader": "numeric", "cluster": "w0"}
{"id": "w1", "prompt": "How many times does the letter 's' occur in the word 'mississippi'? Answer with a number.", "gold": 4, "grader": "numeric"}
```

- `id` is required and unique. It identifies the item in every row, in the
  cache key and on resume, so it must not change between runs.
- `grader` names a grader in `tasks/graders.py` (`exact`, `contains`,
  `numeric`, `choice`, `yes_no`, `set_f1`, `label`), with options as
  `{"name": "numeric", "rel": 0.01}`. The grader reads `gold`.
- `cluster` puts a paraphrase in the cluster of its source item. Every
  interval resamples clusters, so `w0` and `w0-p` count as one source item,
  not two (see "The one idea" in the README).

Every record is checked when the file is loaded, and an error names the
record: a missing or duplicate id, an unknown field, an unknown grader or
grader option, or a gold the grader can never score.

A config runs the file:

<!-- file: letters.json -->
```json
{"out": "runs/letters.jsonl",
 "task":    {"type": "file", "path": "letters.jsonl"},
 "backend": {"type": "http", "flavour": "anthropic",
             "default_params": {"max_tokens": 50}},
 "grid":    {"model": ["claude-haiku-4-5"]},
 "metric":  "correct",
 "options": {"prices": {"in": 1.0, "out": 5.0}, "max_cost": 1.0}}
```

```bash
python -m evalcore run letters.json --dry-run    # the plan and its cost, no calls
python -m evalcore run letters.json
```

The `path` is read relative to the config file, so the config and its data
move together. The `out` path is relative to the working directory. A file in another shape needs no conversion: `"mapping": {"id":
"qid", "prompt": "question", "gold": "answers.0"}` names the field of each
canonical key in the foreign record.

## 2. A task in Python

When the items are generated rather than written by hand, build the same
records in a function. `load_tasks` checks them exactly as it checks a file:

<!-- file: letter_task.py -->
```python
import random

from evalcore import register
from evalcore.tasks.base import Task
from evalcore.tasks.formats import load_tasks

WORDS = ["strawberry", "mississippi", "bookkeeper", "committee", "banana",
         "parallel", "assessment", "occurrence", "necessary", "raspberry"]


@register("task", "letter_count")
def letter_count_task(n_words: int = 10, seed: int = 0) -> Task:
    """How many times a letter occurs in a word, asked two ways.  The
    letter is drawn from the word, so the gold count is at least 1."""
    rng = random.Random(seed)
    records = []
    for i in range(n_words):
        word = WORDS[i % len(WORDS)]
        letter = rng.choice(sorted(set(word)))
        gold = word.count(letter)
        records.append({
            "id": f"w{i}", "gold": gold, "grader": "numeric",
            "prompt": f"How many times does the letter '{letter}' occur "
                      f"in the word '{word}'? Answer with a number."})
        records.append({
            "id": f"w{i}-p", "gold": gold, "grader": "numeric",
            "cluster": f"w{i}",
            "prompt": f"Count the occurrences of '{letter}' in \"{word}\". "
                      f"Reply with the count only."})
    return Task("letter_count", load_tasks(records))
```

Ignore the `@register` line until step 3. The function returns a `Task`
with the default `render` (the item's own prompt) and `score` (the item's
own grader). An `Experiment` runs it. Here the model is a stand-in that
always answers 2, so the run makes no API call:

<!-- file: run_letters.py -->
```python
from evalcore import Experiment, FunctionBackend, grid
from letter_task import letter_count_task

always_2 = FunctionBackend(lambda request: "2",
                           identity={"model": "always-2"})
exp = Experiment(task=letter_count_task(n_words=10), backend=always_2,
                 cells=grid({}), out="runs/always_2.jsonl", metric="correct")
res = exp.run()
print(res.analysis["summary"][["point", "n_clusters", "n_rows", "n_eff"]])
```

For a real model, the backend is `HTTPBackend("anthropic",
default_params={"max_tokens": 50})` and the cells are
`grid({"model": ["claude-haiku-4-5"]})`, as in the README's Quick start.

It prints

```
   point  n_clusters  n_rows  n_eff
0    0.1          10      20   10.0
```

One word in ten has a gold count of 2, hence `point` = 0.1. `n_eff` is the
number of independent items the 20 rows are worth, $N/\mathrm{DEFF}$ with
$\mathrm{DEFF} = 1 + (m-1)\rho$ for $m = 2$ rows per cluster. The stand-in
gives both phrasings of a word the same answer, so $\rho = 1$ and the
paraphrases add nothing: $n_{\text{eff}} = 10$. A model that sometimes
answers the two phrasings differently has $\rho < 1$ and $n_{\text{eff}}$
between 10 and 20.

## 3. A named task

The `@register("task", "letter_count")` line in `letter_task.py` gives the
function a name that a config can use. Its arguments become the config's
keys, with the same defaults, and a key the function does not take is
refused. The config lists the file under `plugins`, which is imported
before anything is built:

<!-- file: letter_count.json -->
```json
{"plugins": ["letter_task.py"],
 "out": "runs/letter_count.jsonl",
 "task":    {"type": "letter_count", "n_words": 10},
 "backend": {"type": "http", "flavour": "anthropic",
             "default_params": {"max_tokens": 50}},
 "grid":    {"model": ["claude-haiku-4-5"]},
 "metric":  "correct",
 "options": {"prices": {"in": 1.0, "out": 5.0}, "max_cost": 1.0}}
```

A `.py` plugin is found relative to the config file, like a task file. With the plugin
listed, `python -m evalcore list` shows `letter_count(n_words=10, seed=0)`
among the tasks. The resolved config, `runs/letter_count.jsonl.config.json`,
records `"seed": 0` although the config left it out.

## 4. Your own render and score

The defaults send the item's prompt as the only message and grade the
response with the item's grader. Replace either when that is not the
experiment. Below, `render` adds a system prompt that fixes the answer
format, and a new grader reads only an answer in that format:

<!-- file: letter_task_strict.py -->
```python
import re
from dataclasses import replace

from evalcore import ParseFailure, Request, Unit
from evalcore.tasks.base import Task
from evalcore.tasks.graders import grader
from letter_task import letter_count_task

SYSTEM = "End your reply with a line of the form 'Answer: N'."


def render(u: Unit) -> Request:
    messages = [{"role": "system", "content": SYSTEM},
                *u.item.payload["messages"]]
    return Request(u.id, messages, dict(u.cell.params))


@grader("answer_line")
def answer_line(text: str, gold: int) -> dict:
    """Correct iff the last 'Answer: N' line gives the gold count.  A
    response without one is unparsed, not wrong."""
    found = re.findall(r"^Answer:\s*(-?\d+)\s*$", text, flags=re.MULTILINE)
    if not found:
        raise ParseFailure("no 'Answer: N' line")
    return {"correct": float(int(found[-1]) == gold), "pred": int(found[-1])}


def strict_task(n_words: int = 10) -> Task:
    base = letter_count_task(n_words)
    items = [replace(it, payload={**it.payload, "grader": "answer_line"})
             for it in base.items]
    return Task("letter_count_strict", items, render=render, version="1")
```

- `render(unit) -> Request` keeps `unit.id` as the request's id and passes
  the cell's params (the model, `max_tokens`) through unchanged. Labels that
  the backend must not receive belong in the cell's tags.
- A grader returns a dict of metrics. It raises `ParseFailure` when it
  cannot read an answer, and the row's status is then `unparsed`, which
  `attrition` counts separately from a wrong answer.
- Changing what a task asks changes the rendered requests, and resume
  re-runs exactly those units. Changing only how it scores does not: bump
  `version`, and the rows scored under the old version are re-scored from
  their stored responses, with no model call.

`score` can also be replaced whole: `Task(name, items, score=fn)` with
`fn(unit, response) -> dict`, for scoring that is not a grader of one
response against one gold.

## 5. A built-in task

To ship a task with evalcore itself:

1. Put the function in `evalcore/tasks/<name>.py`, without `@register`.
2. Register it in `evalcore/config.py`, beside the others:
   `register("task", "<name>")(<name>_task)`.
3. Add its row to the table of task names in the README.
4. Add a test under `tests/tasks/` whose expected values come from the
   construction of the items, never from running the task. For this task:
   every gold equals `word.count(letter)` recomputed from the prompt text,
   each paraphrase shares its source's cluster, and the same `seed` gives
   the same items.
