#!/usr/bin/env python
"""Turn the benchmark and smoke-test outputs into the study's throughput report.

  python examples/federated/vla_study/throughput.py [STUDY_ROOT] > throughput.md

Reads $STUDY_ROOT/bench/results/*.json (study.py bench-workers), $STUDY_ROOT/bench/eval/*/
result.json (study.py bench-eval) and $STUDY_ROOT/smoke/*/summaries/*.json (study.py smoke). A
speed not measured yet falls back to grid.DEFAULT_SPS and is marked "estimate".
"""

import contextlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import grid  # noqa: E402

# Earlier federated studies on an A100 80GB (W&B project fedvla): SmolVLA with only its action
# expert trainable, batch 64. Context for the LoRA and full numbers measured here.
EARLIER = {"LIBERO": 1.9, "Meta-World": 2.4}
CHECKPOINTS_PER_RUN = 10  # rounds 20, 40, ..., 200


def load(paths) -> list[dict]:
    out = []
    for path in paths:
        with contextlib.suppress(OSError, json.JSONDecodeError):
            out.append(json.loads(Path(path).read_text()))
    return out


def main() -> int:
    s = grid.Settings.from_env()
    root = Path(sys.argv[1]) if len(sys.argv) > 1 else s.study_root
    bench = load(sorted((root / "bench" / "results").glob("*.json")))
    evals = load(sorted((root / "bench" / "eval").glob("*/result.json")))
    smoke = load(sorted((root / "smoke").glob("*/summaries/*.json")))
    n_bench, n_seeds = len(s.benchmarks), len(s.seeds)

    print("# Throughput of the federated SmolVLA study\n")
    print(
        "Earlier SmolVLA runs (action expert only, batch 64, A100 80GB): "
        + ", ".join(f"{v} steps/s on {k}" for k, v in EARLIER.items())
        + ".\n"
    )

    print(f"## Training: {s.num_clients} clients, batch {s.batch_size}, by data-loader workers per client\n")
    print("| mode | benchmark | workers/client | steps/s | GPU peak (GB) | RAM incl. workers (GB) | GPU |")
    print("|---|---|---|---|---|---|---|")
    best: dict[str, float] = {}
    for r in sorted(bench, key=lambda r: (r["mode"], r["benchmark"], r["workers_per_client"])):
        sps = r.get("steps_per_s")
        print(
            f"| {r['mode']} | {r['benchmark']} | {r['workers_per_client']} | {sps if sps is not None else 'failed'} "
            f"| {r.get('gpu_mem_peak_gb')} | {r.get('rss_with_workers_gb')} | {r.get('host', '')} |"
        )
        if sps and r["benchmark"] == "libero_object" and not math.isnan(sps):
            best[r["mode"]] = max(best.get(r["mode"], 0.0), sps)

    print("\n## Disk per checkpoint and per resumable state (smoke tests)\n")
    print("| case | checkpoint (GB) | federated state (GB) | trainable params |")
    print("|---|---|---|---|")
    for r in smoke:
        if "checkpoint_gb" in r:
            ck = max(r["checkpoint_gb"].values()) if r["checkpoint_gb"] else None
            trainable = (r.get("trainable_params") or {}).get("trainable")
            print(f"| {r['case']} | {ck} | {r.get('fl_state_gb')} | {trainable} |")

    print("\n## Evaluation speed (episodes per hour)\n")
    print("| device | benchmark | episodes/hour | threads |")
    print("|---|---|---|---|")
    eval_rate: dict[str, list[float]] = defaultdict(list)
    for r in evals:
        print(f"| {r['device']} ({r.get('gpu') or r.get('host', '')}) | {r['benchmark']} | "
              f"{r['timing']['episodes_per_hour']} | {r.get('cpu_threads')} |")  # fmt: skip
        eval_rate[r["device"]].append(r["timing"]["episodes_per_hour"])

    steps = s.rounds * s.local_steps * s.num_clients
    runs_p1 = n_bench * n_seeds * 2  # federated + centralized
    runs_p2 = n_bench * n_seeds * 2  # IID + shared-core, federated only
    print(f"\n## The planned grid on one GPU\n\nOne run = {s.rounds} rounds x {s.local_steps} local steps x "
          f"{s.num_clients} clients = {steps:,} optimizer steps. Per mode, Phase 1 is {runs_p1} runs "
          f"({n_bench} benchmarks x {n_seeds} seeds x federated + centralized) and Phase 2 adds {runs_p2} "
          "federated runs.\n")  # fmt: skip
    print("| mode | steps/s | source | hours/run | Phase 1 GPU-hours | Phase 1 days | Phase 2 GPU-hours |")
    print("|---|---|---|---|---|---|---|")
    for mode in grid.vla_study.MODES:
        sps = best.get(mode) or grid.DEFAULT_SPS[mode]
        source = "measured" if mode in best else "estimate"
        hours = steps / sps / 3600
        print(f"| {mode} | {sps:.2f} | {source} | {hours:.0f} | {hours * runs_p1:,.0f} | "
              f"{hours * runs_p1 / 24:.0f} | {hours * runs_p2:,.0f} |")  # fmt: skip

    print("\n### Reducing the budget (Phase 1, both modes; time scales with rounds x local steps)\n")
    print("| rounds | local steps | steps/run | Phase 1 GPU-hours | days on one GPU |")
    print("|---|---|---|---|---|")
    for rounds in (200, 100, 50):
        for local in (250, 100, 50):
            run_steps = rounds * local * s.num_clients
            total = sum(
                run_steps / (best.get(m) or grid.DEFAULT_SPS[m]) / 3600 * runs_p1
                for m in grid.vla_study.MODES
            )
            print(f"| {rounds} | {local} | {run_steps:,} | {total:,.0f} | {total / 24:.0f} |")

    if eval_rate:
        episodes = CHECKPOINTS_PER_RUN * 10 * s.eval_episodes
        print(f"\n### Evaluation load\n\nEach run is evaluated on {episodes:,} episodes ({CHECKPOINTS_PER_RUN} "
              f"checkpoints x 10 tasks x {s.eval_episodes} episodes); Phase 1 has {2 * runs_p1} runs over both modes.\n")  # fmt: skip
        for device, rates in eval_rate.items():
            rate = sum(rates) / len(rates)
            print(f"- {device}: {rate:.0f} episodes/hour -> {episodes / rate:.1f} hours per run, "
                  f"{2 * runs_p1 * episodes / rate:,.0f} hours for Phase 1")  # fmt: skip
    return 0


if __name__ == "__main__":
    sys.exit(main())
