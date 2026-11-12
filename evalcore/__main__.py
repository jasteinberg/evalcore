"""python -m evalcore run config.json [--dry-run] [--yes] [--recalibrate]
                                     [--shard I/N | --shard slurm]
python -m evalcore merge config.json --shards N
python -m evalcore summarize runs/sweep.jsonl
python -m evalcore list [kind] [--plugin module_or_file.py ...]"""

from __future__ import annotations

import argparse
import dataclasses
import sys

from .config import _plugins, from_config
from .core.events import format_summary, summarize_events
from .core.shards import check_shard, shard_from_env, shard_from_k8s


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
    r.add_argument("--shard", metavar="I/N|slurm|k8s/N",
                   help="run only shard I of N (\"slurm\": this array task's, "
                        "\"k8s/N\": this Indexed Job pod's, of N)")
    m = sub.add_parser("merge", help="combine a sharded run, then analyse it")
    m.add_argument("config")
    m.add_argument("--shards", type=int, required=True, metavar="N")
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
    if a.cmd == "merge":
        res = exp.merge(a.shards)
        print(f"merged {a.shards} shards into {exp.out}: "
              f"{res.marker['status'] if res.marker else '?'}")
        for i, st in (res.marker or {}).get("shards", {}).items():
            if st != "complete":
                print(f"  shard {i}: {st}")
        for name, table in res.analysis.items():
            print(f"\n{name}:\n{table.to_string(index=False)}")
        print(f"\nreport: {res.report_path}")
        return 0
    if a.shard:
        exp = exp.sharded(*_shard_arg(a.shard))
    exp.options = dataclasses.replace(
        exp.options, dry_run_only=a.dry_run or exp.options.dry_run_only,
        confirm=a.yes or exp.options.confirm,
        recalibrate=a.recalibrate or exp.options.recalibrate)
    res = exp.run()
    if res.dry_run:
        print("plan:", res.dry_run)
    if res.halted:
        print(f"halted: {res.halted}")
    for name, table in res.analysis.items():
        print(f"\n{name}:\n{table.to_string(index=False)}")
    print(f"\nreport: {res.report_path}")
    return 1 if res.halted and res.halted != "dry run only" else 0


def _shard_arg(text: str) -> tuple[int, int]:
    """"3/16" -> (3, 16), "slurm" -> this array task's shard, "k8s/16" ->
    this Indexed Job pod's shard of 16."""
    if text == "slurm":
        return shard_from_env()
    i, sep, n = text.partition("/")
    if not sep or not n.isdigit() or not (i.isdigit() or i == "k8s"):
        raise SystemExit(f"--shard: expected I/N, slurm or k8s/N, got "
                         f"{text!r}")
    if i == "k8s":
        return shard_from_k8s(int(n))
    return check_shard((int(i), int(n)))


if __name__ == "__main__":
    sys.exit(main())
