#!/usr/bin/env python
"""Gather every eval under an eval folder into one results table.

Reads each `summary.tsv` written by eval_archive.sh (one per evaluated model) and writes them all to
`<eval root>/all_results.tsv`, one row per model x task, with an `__overall__` row per model. The
model's benchmark, suite, arm and budget are read from its folder name where it follows the short
format - `lib<suite>_<arm>_<budget>` (LIBERO) or `mw_<arm>_<budget>` (Meta-World) - and left blank
otherwise; the folder name itself is always kept in the `scenario` column.

Usage:
    python examples/federated/gather_evals.py [EVAL_ROOT]     # default: outputs/eval
"""

import csv
import re
import sys
from collections import defaultdict
from pathlib import Path

ARMS = {"cen": "centralized", "fed": "federated", "loc": "local"}
SHORT_NAME = re.compile(r"^(?:lib(?P<suite>[a-z0-9]+)|(?P<mw>mw))_(?P<arm>[a-z]+)_(?P<budget>\d+)$")
FIXED_COLUMNS = ["scenario", "benchmark", "suite", "arm", "budget", "task", "mean", "std", "n_seeds",
                 "episodes_per_seed"]  # fmt: skip


def describe(scenario: str) -> dict[str, str]:
    """Benchmark, suite, arm and budget from a short-format folder name; blanks if it is not one."""
    match = SHORT_NAME.match(scenario)
    if not match:
        return {"benchmark": "", "suite": "", "arm": "", "budget": ""}
    if match["mw"]:
        benchmark, suite = "metaworld", ""
    else:
        benchmark, suite = "libero", f"libero_{match['suite']}"
    return {
        "benchmark": benchmark,
        "suite": suite,
        "arm": ARMS.get(match["arm"], match["arm"]),
        "budget": match["budget"],
    }


def main() -> None:
    eval_root = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/eval")
    summaries = sorted(eval_root.rglob("summary.tsv"))
    if not summaries:
        sys.exit(f"no summary.tsv found under {eval_root}")

    rows, seed_columns = [], set()
    for path in summaries:
        scenario = path.parent.relative_to(eval_root).as_posix()
        with path.open() as f:
            for record in csv.DictReader(f, delimiter="\t"):
                seed_columns.update(k for k in record if k.startswith("seed"))
                rows.append({"scenario": scenario, **describe(scenario), **record})

    seeds = sorted(seed_columns, key=lambda s: int(s[4:]))
    out = eval_root / "all_results.tsv"
    with out.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIXED_COLUMNS + seeds, delimiter="\t", restval="")
        writer.writeheader()
        writer.writerows(rows)

    # Overview: overall success per scenario, grouped so arms sit side by side at each budget.
    overall = defaultdict(dict)
    unparsed = []
    for row in rows:
        if row["task"] != "__overall__":
            continue
        if row["benchmark"]:
            key = (row["benchmark"], row["suite"], int(row["budget"]))
            overall[key][row["arm"]] = f"{float(row['mean']):.1f} +/- {float(row['std']):.1f}"
        else:
            unparsed.append(f"{row['scenario']}: {float(row['mean']):.1f} +/- {float(row['std']):.1f}")
    arms = sorted({arm for by_arm in overall.values() for arm in by_arm})
    print(f"{len(summaries)} evaluated models -> {out}\n")
    print(f"{'benchmark':<11}{'suite':<16}{'budget':>7}" + "".join(f"{a:>16}" for a in arms))
    for (benchmark, suite, budget), by_arm in sorted(overall.items()):
        print(
            f"{benchmark:<11}{suite or '-':<16}{budget:>7}"
            + "".join(f"{by_arm.get(a, '-'):>16}" for a in arms)
        )
    if unparsed:
        print("\nother folders (name not in the short format):")
        print("\n".join(f"  {line}" for line in unparsed))
    print("\n(overall success %, mean +/- std over seeds)")


if __name__ == "__main__":
    main()
