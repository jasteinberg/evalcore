"""Agent example: a multi-turn tool loop, and the row status of each outcome.

A scripted model answers sums by calling an `add` tool, so the run costs
nothing and every outcome is known in advance.  `AgentBackend` wraps it:
one unit is one episode and one row, and the row's meta holds the whole
trajectory.  The items are rigged to reach every outcome the loop has:

    plain         calls add, reads the result, answers       -> ok
    typo          first call has a bad argument name.  The
                  ToolInputError goes back to the model as
                  the tool's output, and it corrects itself   -> ok
    outage        the tool's backend is down (ToolUnavailable) -> tool_error
    stubborn      keeps calling the tool, never answers        -> truncated

`attrition` then accounts for every planned unit, and only the ok rows
are scored.

    python examples/agent_example.py OUT_DIR

This example calls `execute` (core/runner.py), the harness loop alone,
so that each step is visible.  `Experiment.run()` (pipeline.py) runs the
same loop as its execute stage, with calibration, a cost check, analysis
and a report around it.  `stream=None` turns off the progress lines.

Drafted with the assistance of Claude (Anthropic).
"""
import json
import sys
from pathlib import Path

import pandas as pd

from evalcore import (
    AgentBackend,
    FunctionBackend,
    Item,
    Request,
    Response,
    Tool,
    ToolUnavailable,
    Unit,
    attrition,
    execute,
    grid,
    units,
)
from evalcore.core.records import to_frame

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 20)


def add(arguments) -> str:
    if arguments["a"] == 13:                 # the "outage" item's numbers
        raise ToolUnavailable("calculator service returned 503")
    return str(arguments["a"] + arguments["b"])


ADD = Tool("add", "Add two integers.",
           {"type": "object",
            "properties": {"a": {"type": "integer"}, "b": {"type": "integer"}},
            "required": ["a", "b"]},
           add)


def scripted_model(req: Request) -> Response:
    """Reads the conversation so far and decides the next turn."""
    kind, a, b = json.loads(req.messages[0]["content"])
    tool_turns = [m for m in req.messages if m["role"] == "tool"]
    last = tool_turns[-1]["content"] if tool_turns else None

    def call(args: dict) -> Response:
        n = len(tool_turns)
        return Response(req.unit_id, text=None, meta={
            "tool_calls": [{"id": f"c{n}", "name": "add", "arguments": args}]})

    if kind == "stubborn" or last is None:
        return call({"a": a, "bb": b} if kind == "typo" else {"a": a, "b": b})
    if last.startswith("error:"):            # read the tool's complaint, retry
        return call({"a": a, "b": b})
    return Response(req.unit_id, text=f"The sum is {last}.")


def items() -> list[Item]:
    cases = [("plain", 2, 3), ("plain", 40, 2), ("typo", 7, 8),
             ("outage", 13, 1), ("stubborn", 5, 5)]
    return [Item(f"{k}-{a}-{b}", {"kind": k, "a": a, "b": b})
            for k, a, b in cases]


def render(u: Unit) -> Request:
    p = u.item.payload
    return Request(u.id, [{"role": "user",
                           "content": json.dumps([p["kind"], p["a"], p["b"]])}])


def score(u: Unit, r: Response) -> dict:
    p = u.item.payload
    return {"correct": float(r.text == f"The sum is {p['a'] + p['b']}.")}


def main(out_dir: str) -> None:
    out = Path(out_dir) / "rows.jsonl"
    inner = FunctionBackend(scripted_model, identity={"model": "scripted-adder",
                                                      "version": 1},
                            supports_tools=True)
    agent = AgentBackend(inner, [ADD], max_turns=4)
    cells = grid({"agent": ["scripted"]})
    execute(units(cells, items()), agent, render, score, out, workers=1,
            stream=None)

    df = to_frame(out)
    df["kind"] = df["item_id"].str.split("-").str[0]
    # a response's meta reaches the frame as meta_* columns
    print(df[["kind", "status", "meta_turns", "meta_n_tool_calls",
              "meta_tool_input_errors", "meta_truncated_by", "correct",
              "error"]].to_string(index=False))

    typo = df[df["kind"] == "typo"].iloc[0]
    print("\ntrajectory of the 'typo' episode:")
    for t, turn in enumerate(typo["meta_trajectory"]):
        for c, res in zip(turn["tool_calls"], turn["results"], strict=True):
            print(f"  turn {t}: add({c['arguments']}) -> {res['kind']}: "
                  f"{res['content']}")
        if not turn["tool_calls"]:
            print(f"  turn {t}: answer {turn['text']!r}")

    print("\nattrition, against the plan manifest:")
    print(attrition(df, ["agent"], manifest=f"{out}.manifest.jsonl")
          .to_string(index=False))


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("usage: python examples/agent_example.py OUT_DIR")
    main(sys.argv[1])
