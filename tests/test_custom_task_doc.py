"""docs/custom-task.md: every file the guide shows is written out and run,
so the guide cannot drift from the API.  Each step runs in a fresh
process, because registering a task or grader changes global registries.

Expected values come from the construction of the items (two rows per
word, one cluster per word, prompts without digits), not from running
the task."""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
DOC = REPO / "docs" / "custom-task.md"
BLOCK = re.compile(r"<!-- file: (\S+) -->\n```\w*\n(.*?)```", re.DOTALL)


@pytest.fixture
def guide(tmp_path: Path) -> Path:
    files = dict(BLOCK.findall(DOC.read_text()))
    assert sorted(files) == ["letter_count.json", "letter_task.py",
                             "letter_task_strict.py", "letters.json",
                             "letters.jsonl", "run_letters.py"]
    for name, text in files.items():
        (tmp_path / name).write_text(text)
    return tmp_path


def py(cwd: Path, code: str) -> str:
    """Run `code` in a fresh interpreter in `cwd`, with a dummy API key so
    an HTTP backend can be built (nothing here calls it).  The repo comes
    first on the path, so the evalcore under test is this checkout's, not
    whichever copy the interpreter has installed."""
    env = {**os.environ, "ANTHROPIC_API_KEY": "dummy",
           "PYTHONPATH": os.pathsep.join([str(REPO), str(cwd)])}
    p = subprocess.run([sys.executable, "-c", textwrap.dedent(code)],
                       cwd=cwd, env=env, capture_output=True, text=True,
                       check=False)
    assert p.returncode == 0, p.stderr
    return p.stdout


def offline(cwd: Path, config: str) -> str:
    """The config with its backend replaced by the echo backend (it returns
    the prompt reversed), written beside it.  Every row is then unparsed,
    so there is no metric column to summarise, and `metric` is dropped."""
    cfg = json.loads((cwd / config).read_text())
    cfg["backend"] = {"type": "echo"}
    cfg["grid"] = {"model": ["echo"]}
    cfg["options"] = {}
    cfg.pop("metric")
    name = "offline_" + config
    (cwd / name).write_text(json.dumps(cfg))
    return name


def test_step1_task_file(guide: Path) -> None:
    py(guide, "from evalcore import from_config; from_config('letters.json')")
    out = py(guide, f"""
        from evalcore import from_config, to_frame
        res = from_config({offline(guide, 'letters.json')!r}).run()
        df = to_frame(res.out)
        print(len(df), sorted(set(df['status'])), df['cluster'].nunique())
        """)
    # the reversed prompts hold no digit, so no row can be graded
    assert out.split("\n")[-2] == "3 ['unparsed'] 2"


def test_step2_task_in_python(guide: Path) -> None:
    py(guide, "exec(open('run_letters.py').read())")
    out = py(guide, """
        from evalcore import to_frame
        df = to_frame('runs/always_2.jsonl')
        per = df.groupby('cluster')['correct'].nunique()
        print(len(df), df['cluster'].nunique(), int(per.max()),
              sorted(set(df['pred'])))
        """)
    # 10 words, two phrasings each, one cluster per word; a paraphrase has
    # its source's gold, so both rows of a cluster are right or both wrong
    assert out.split("\n")[-2] == "20 10 1 [2.0]"


def test_step3_named_task(guide: Path) -> None:
    out = py(guide, """
        import json
        from evalcore import from_config
        exp = from_config('letter_count.json')
        print(json.dumps(exp.config['task'], sort_keys=True))
        """)
    assert json.loads(out.split("\n")[-2]) == {
        "type": "letter_count", "n_words": 10, "seed": 0}
    out = py(guide, f"""
        from evalcore import from_config, to_frame
        res = from_config({offline(guide, 'letter_count.json')!r}).run()
        print(len(to_frame(res.out)))
        """)
    assert out.split("\n")[-2] == "20"


def test_step4_own_render_and_grader(guide: Path) -> None:
    out = py(guide, """
        from evalcore import FunctionBackend, grid, to_frame
        from letter_task_strict import strict_task

        def model(answer):
            def fn(request):
                assert request.messages[0]["role"] == "system"
                return answer
            return FunctionBackend(fn, identity={"model": answer})

        task = strict_task(3)
        for answer in ("I count 3.", "Answer: 3"):
            df = task.run(grid({}), model(answer), f"runs/{answer}.jsonl",
                          stream=None)
            pred = df["pred"].dropna() if "pred" in df else []
            print(sorted(set(df["status"])), sorted(set(pred)))
        """)
    lines = out.strip().split("\n")
    assert lines[-2] == "['unparsed'] []"      # no 'Answer: N' line
    assert lines[-1] == "['ok'] [3]"
