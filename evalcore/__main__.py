"""python -m evalcore run config.json [--dry-run] [--yes] [--recalibrate]
python -m evalcore summarize runs/sweep.jsonl
python -m evalcore list [kind] [--plugin module_or_file.py ...]"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from .config import _plugins, from_config
from .core.events import format_summary, summarize_events


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="python -m evalcore")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run", help="run an experiment from a JSON config")
    r.add_argument("config")
    r.add_argument("--dry-run", action="store_true",
                   help="calibrate and plan, report the cost, call nothing more")
    r.add_argument("--yes", action="store_true",
                   help="proceed even above options.max_cost")
    r.add_argument("--recalibrate", action="store_true")
    s = sub.add_parser("summarize", help="summarise a sink's event log")
    s.add_argument("sink")
    ls = sub.add_parser("list", help="registered components and their keys")
    ls.add_argument("kind", nargs="?", help="one kind (backend, task, ...)")
    ls.add_argument("--plugin", action="append", default=[],
                    help="import a plugin first (repeatable)")
    a = ap.parse_args(argv)

    if a.cmd == "list":
        from .registry import REGISTRIES
        _plugins(a.plugin, None)
        for kind, reg in REGISTRIES.items():
            if a.kind and kind != a.kind:
                continue
            print(f"{kind}:")
            for name in reg.names():
                keys, context = reg.schema(name)
                args = ", ".join(k if v == "required" else f"{k}={v!r}"
                                 for k, v in keys.items())
                given = f"   [given: {', '.join(context)}]" if context else ""
                print(f"  {name}({args}){given}")
        return 0

    if a.cmd == "summarize":
        print(format_summary(summarize_events(a.sink)))
        return 0
    exp = from_config(a.config)
    exp.options = dataclasses.replace(
        exp.options, dry_run_only=a.dry_run or exp.options.dry_run_only,
        confirm=a.yes or exp.options.confirm,
        recalibrate=a.recalibrate or exp.options.recalibrate)
    res = exp.run()
    if res.dry_run:
        print("plan:", {k: v for k, v in res.dry_run.items()})
    if res.halted:
        print(f"halted: {res.halted}")
    for name, table in res.analysis.items():
        print(f"\n{name}:\n{table.to_string(index=False)}")
    print(f"\nreport: {res.report_path}")
    return 1 if res.halted and res.halted != "dry run only" else 0


if __name__ == "__main__":
    sys.exit(main())
