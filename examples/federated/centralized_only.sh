#!/usr/bin/env bash
#
# Centralized SmolVLA on Meta-World MT50 - the ceiling arm, on its own.
#
# Trains one policy on the pooled episodes of several tasks and evaluates it in the simulator,
# once per seed. This is the same centralized arm examples/federated/noniid.sh runs, extracted
# so it can be run alone: to establish the ceiling before spending GPU on federated runs, or to
# add seeds to a ceiling you already have.
#
# The task list still defines which episodes are used, but nothing is partitioned here - the
# data is simply pooled. NUM_CLIENTS therefore does not apply, and neither do the server or
# FedProx settings.
#
# Usage:
#   bash examples/federated/centralized_only.sh                   # default budget
#   STEPS=100 SKIP_EVAL=1 bash ...                                # fast plumbing check
#   TASKS=reach-v3,window-open-v3,handle-press-side-v3 bash ...    # different tasks
#   SEEDS="1000 2000 3000 4000 5000" bash ...                      # more seeds
#   BATCH_SIZE=128 NUM_WORKERS=8 bash ...                          # A100 80GB
#   BATCH_SIZE=8 MIXED_PRECISION=no bash ...                       # small / older GPU
#
set -euo pipefail

# ----------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------
DATASET="${DATASET:-lerobot/metaworld_mt50}"
OUT_ROOT="${OUT_ROOT:-./outputs/fl-study-$(date +%Y%m%d_%H%M%S)}"

# Tasks whose episodes are pooled into one training set. Any number of them; nothing is
# partitioned here. Avoid push-v3 and push-back-v3: they share a description string, so the
# resolver cannot tell which 50 episodes you meant and rejects them.
TASKS="${TASKS:-drawer-close-v3,button-press-v3,door-close-v3}"

# Optimizer steps for the run. Named directly rather than derived from rounds x local steps,
# since there are no rounds here. To match a federated run, set it to that run's
# ROUNDS x LOCAL_STEPS (equal progress) or ROUNDS x LOCAL_STEPS x NUM_CLIENTS (equal compute).
STEPS="${STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Dataloader workers. One training process here, so this is the total - unlike the federated
# scripts, where each client holds its own dataloader and the count multiplies by NUM_CLIENTS.
# Keep it at or below the cores your scheduler actually allocated.
NUM_WORKERS="${NUM_WORKERS:-4}"

# bf16 autocast. Native on A100 and later; roughly halves activation memory and materially
# raises throughput on a VLM this size. Set to "no" on hardware without bf16 support.
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"

# draccus parses CLI values as YAML, where a bare `no` is the boolean False - so
# `--accelerator.mixed_precision=no` fails to parse even though "no" is the field's own default.
# Omitting the flag entirely selects that default, which is what we do here.
MP_ARGS=()
if [[ "$MIXED_PRECISION" != "no" ]]; then
  MP_ARGS=(--accelerator.mixed_precision="$MIXED_PRECISION")
fi
# One run per seed. Seeds vary model init and data order, so the spread across them is the
# training variance any single-seed number should be read against.
SEEDS="${SEEDS:-1000 2000 3000}"

EVAL_EPISODES="${EVAL_EPISODES:-15}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# MuJoCo rendering backend, used by the eval rollouts (Meta-World renders rgb_array frames).
#   egl    - GPU-accelerated headless. The default, and what the LeRobot sim docs use.
#   osmesa - software fallback; no GPU needed, noticeably slower, needs libOSMesa installed.
#   glfw   - requires an attached display; will fail on a headless box.
# Exported so the lerobot-eval subprocesses inherit it.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# NOTE: BATCH_SIZE is an experimental variable, not only a throughput knob. Numbers are not
# comparable across runs with different batch sizes, and the LR preset was tuned at a much
# smaller one - so keep it equal to whatever the federated runs you compare against used.

TRAIN=(python -m lerobot.scripts.lerobot_train)
EVAL=(python -m lerobot.scripts.lerobot_eval)

POLICY_ARGS=(
  --policy.type=smolvla
  --policy.load_vlm_weights=true
  --policy.push_to_hub=false
)

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.tsv"
: > "$SUMMARY"

# ----------------------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------------------

# Resolve comma-separated Meta-World slugs to a draccus episode list, e.g. "[900,901,...]".
#
# Episode indices are looked up through each task's natural-language description, because a
# task's position in the dataset does NOT follow its task_id (the two disagree for 23 of the
# 50 tasks). Every task occupies exactly one contiguous block of 50 episodes.
episodes_for_tasks() {
  python - "$1" "$DATASET" <<'PYEOF'
import json, sys
from pathlib import Path

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

slugs = sys.argv[1].split(",")
import lerobot.envs
cfg = json.loads((Path(lerobot.envs.__file__).parent / "metaworld_config.json").read_text())
desc = cfg["TASK_DESCRIPTIONS"]

meta = LeRobotDatasetMetadata(sys.argv[2])
tasks = [t[0] if t else "" for t in meta.episodes["tasks"]]

out = []
for slug in slugs:
    if slug not in desc:
        sys.exit(f"unknown Meta-World task: {slug!r}")
    hits = [i for i, t in enumerate(tasks) if t == desc[slug]]
    if not hits:
        sys.exit(f"no episodes found for {slug!r} in this dataset")
    # push-v3 and push-back-v3 share one description string, so a lookup by description
    # returns both blocks. Refuse rather than silently doubling the data.
    if len(hits) != 50:
        sys.exit(f"{slug!r} matched {len(hits)} episodes, expected 50 (ambiguous description?)")
    out.extend(hits)

print("[" + ",".join(str(i) for i in sorted(out)) + "]")
PYEOF
}

# Read the headline success rate out of a finished eval run. The payload is
# {"overall": {...}} for multi-task evals and {"aggregated": {...}} for single-env ones, so
# look for pc_success wherever it lives rather than assuming one shape.
success_rate() {
  local f="$1/eval_info.json"
  [[ -f "$f" ]] || { echo "n/a"; return; }
  python - "$f" <<'METRIC_EOF' 2>/dev/null || echo "n/a"
import json, sys

payload = json.load(open(sys.argv[1]))
for key in ("overall", "aggregated"):
    node = payload.get(key)
    if isinstance(node, dict) and "pc_success" in node:
        print(f"{node['pc_success']:.1f}%")
        break
else:
    print("n/a")
METRIC_EOF
}

# Per-task success. A policy trained on pooled tasks can average well while never solving one
# of them, and that per-task floor is what a federated run has to be compared against.
per_task_rates() {
  local f="$1/eval_info.json"
  [[ -f "$f" ]] || return
  python - "$f" <<'PERTASK_EOF' 2>/dev/null || true
import json, sys

payload = json.load(open(sys.argv[1]))
for name, info in sorted((payload.get("per_group") or {}).items()):
    rate = info.get("pc_success")
    if rate is not None:
        print(f"      {name:32} {rate:5.1f}%  (n={info.get('n_episodes', '?')})")
PERTASK_EOF
}

run_eval() {
  local name="$1" ckpt="$2" tasks="$3"
  if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "  [eval skipped]"
    return
  fi
  echo "  --> evaluating $name on $tasks"
  "${EVAL[@]}" \
    --policy.path="$ckpt/checkpoints/last/pretrained_model" \
    --env.type=metaworld --env.task="$tasks" \
    --eval.batch_size=1 --eval.n_episodes="$EVAL_EPISODES" \
    --output_dir="$ckpt/eval" \
    > "$ckpt/eval.log" 2>&1 || echo "  !! eval failed, see $ckpt/eval.log"
}

# `cmd | tee log | grep ... || true` cannot report a failed training run: tee succeeds, grep's
# status is discarded, and the script would sail on to report "n/a" as though eval were the
# problem. Assert on the artifact instead, and stop the study where it actually broke.
assert_trained() {
  local dir="$1" label="$2"
  if [[ ! -f "$dir/checkpoints/last/pretrained_model/model.safetensors" ]]; then
    echo
    echo "!! $label produced no checkpoint - training failed."
    echo "!! Full log: $dir.log"
    tail -20 "$dir.log" 2>/dev/null || true
    exit 1
  fi
}

record() { printf '%s\t%s\t%s\n' "$1" "$2" "$3" >> "$SUMMARY"; }

banner() {
  echo
  echo "========================================================================"
  echo "  $*"
  echo "========================================================================"
}

# ----------------------------------------------------------------------------------------
# Centralized - all episodes pooled on one machine
# ----------------------------------------------------------------------------------------

banner "Centralized - pooled episodes of: $TASKS"

EPS="$(episodes_for_tasks "$TASKS")"
echo "episodes: $(echo "$EPS" | tr -cd ',' | wc -c | awk '{print $1+1}') selected"

echo "seeds: $SEEDS"

for SEED in $SEEDS; do
  banner "seed $SEED"
  CEN="$OUT_ROOT/centralized-s$SEED"
  rm -rf "$CEN"

  echo "--> centralized (seed $SEED, $STEPS steps)"
  "${TRAIN[@]}" "${POLICY_ARGS[@]}" \
    --dataset.repo_id="$DATASET" --dataset.episodes="$EPS" \
    --steps="$STEPS" --save_freq="$STEPS" \
    --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed="$SEED" \
    "${MP_ARGS[@]}" \
    --output_dir="$CEN" \
    2>&1 | tee "$CEN.log" | grep -E "^INFO.*step" || true
  assert_trained "$CEN" "centralized-s$SEED"

  run_eval "centralized-s$SEED" "$CEN" "$TASKS"
  record "centralized" "$SEED" "$(success_rate "$CEN/eval")"
done

# ----------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------
banner "Results (pc_success over $(wc -w <<< "$SEEDS") seeds)"

python - "$SUMMARY" <<'AGG_EOF'
import statistics
import sys
from collections import defaultdict

rows = defaultdict(dict)
for line in open(sys.argv[1]):
    arm, seed, rate = line.rstrip("\n").split("\t")
    rows[arm][seed] = rate

seeds = sorted({s for per_seed in rows.values() for s in per_seed})
print(f"{'arm':<14}" + "".join(f"{('seed ' + s):>12}" for s in seeds) + f"{'mean':>10}{'std':>9}")
for arm in ("centralized",):
    if arm not in rows:
        continue
    cells, values = [], []
    for s in seeds:
        raw = rows[arm].get(s, "n/a")
        cells.append(f"{raw:>12}")
        if raw.endswith("%"):
            values.append(float(raw[:-1]))
    # A mean over one usable seed is not a mean; say so rather than printing a bare number.
    summary = f"{statistics.mean(values):>9.1f}%" if values else f"{'n/a':>10}"
    spread = f"{statistics.stdev(values):>8.1f}%" if len(values) > 1 else f"{'-':>9}"
    print(f"{arm:<14}" + "".join(cells) + summary + spread)
AGG_EOF

echo
echo "Per-task breakdown (per seed):"
for SEED in $SEEDS; do
  if [[ -f "$OUT_ROOT/centralized-s$SEED/eval/eval_info.json" ]]; then
    echo "  seed=$SEED"
    per_task_rates "$OUT_ROOT/centralized-s$SEED/eval"
  fi
done

cat <<'EOS'

How to read this:
  * This is the ceiling: the same data, the same budget, pooled on one machine. A federated run
    on the same TASKS, BATCH_SIZE and step budget is only interpretable against it.
  * Per-task success matters more than the mean - a policy can average well while never solving
    one of the pooled tasks, and that is the baseline a federated run must be compared against
    task by task, not in aggregate.
  * The seed-to-seed std is the noise floor for every comparison you make later. If it is large,
    raise EVAL_EPISODES or add seeds before reading anything into a federated gap.
EOS
echo "Raw logs and checkpoints: $OUT_ROOT"
