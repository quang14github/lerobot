#!/usr/bin/env bash
#
# Federated vs centralized SmolVLA on Meta-World MT50, under task heterogeneity.
#
# Every client holds a DIFFERENT, disjoint task. This is the case federated learning exists
# for, and the one plain FedAvg struggles with: a client that has never seen `door-close-v3`
# must still contribute to a global policy that can do it.
#
# Both arms train on exactly the same episodes with the same policy, seed and batch size; the
# centralized arm simply gets the data pooled on one machine, which makes it the ceiling.
#
# For the IID control - the same comparison with every client holding slices of ONE task - run
# examples/federated/run_fl_vs_centralized.sh, which does both. That gap is the noise floor
# this one should be read against: whatever the gap here exceeds it by is the cost of
# heterogeneity rather than of federation.
#
# Usage:
#   bash examples/federated/noniid.sh                             # default budget
#   ROUNDS=5 LOCAL_STEPS=10 SKIP_EVAL=1 bash ...                  # fast plumbing check
#   SERVER_TYPE=fedadam SERVER_LR=1e-3 bash ...                   # adaptive aggregation
#   PROX_MU=0.05 bash ...                                         # the FedProx penalty
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

# One task per client, so the count here must match NUM_CLIENTS (checked below). Avoid
# push-v3 and push-back-v3: they share a description string and the resolver rejects them.
TASKS="${TASKS:-drawer-close-v3,button-press-v3,door-close-v3}"

NUM_CLIENTS="${NUM_CLIENTS:-3}"
ROUNDS="${ROUNDS:-100}"
LOCAL_STEPS="${LOCAL_STEPS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Dataloader workers PER CLIENT. Each client keeps its own dataloader alive across rounds (that
# is what stops every round replaying the same first batches), so the process count is
# NUM_CLIENTS x NUM_WORKERS - 12 at the defaults. On a scheduler that pins you to
# --cpus-per-task, size this as (allocated cores / NUM_CLIENTS), not as the node's core count,
# or the workers oversubscribe and the GPU starves anyway.
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
# One run per seed. Seeds vary model init and data order only - the partition itself is held
# fixed by PARTITION_SEED, so every seed sees the same task-to-client assignment and the
# spread across seeds measures training variance rather than luck of the split.
SEEDS="${SEEDS:-1000 2000 3000}"
PARTITION_SEED="${PARTITION_SEED:-42}"

SERVER_TYPE="${SERVER_TYPE:-fedavg}"
SERVER_LR="${SERVER_LR:-1.0}"
PROX_MU="${PROX_MU:-0.0}"

EVAL_EPISODES="${EVAL_EPISODES:-15}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# MuJoCo rendering backend, used by the eval rollouts (Meta-World renders rgb_array frames).
#   egl    - GPU-accelerated headless. The default, and what the LeRobot sim docs use.
#   osmesa - software fallback; no GPU needed, noticeably slower, needs libOSMesa installed.
#   glfw   - requires an attached display; will fail on a headless box.
# Exported so the lerobot-eval subprocesses inherit it.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

# NOTE: BATCH_SIZE is an experimental variable, not only a throughput knob. Both arms share it
# so the comparison stays internally valid, but numbers are not comparable across runs with
# different batch sizes, and the LR preset was tuned at a much smaller one.
#
# Compute budget. The federation performs ROUNDS x LOCAL_STEPS x NUM_CLIENTS gradient updates
# in aggregate, so the centralized arm is given the same total to keep the comparison about
# *where the data lives* rather than about who got more compute. Set CENTRAL_STEPS yourself to
# compare on a different basis (e.g. ROUNDS x LOCAL_STEPS to match global progress instead).
CENTRAL_STEPS="${CENTRAL_STEPS:-$((ROUNDS * LOCAL_STEPS))}"

FL_TRAIN=(python -m lerobot.scripts.lerobot_fl_train)
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

# Per-task success, which is what exposes the federated failure mode: a global model can hold a
# respectable average while having collapsed completely on one client's task.
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
# Non-IID - every client holds a DIFFERENT task
# ----------------------------------------------------------------------------------------

banner "Non-IID - $NUM_CLIENTS clients, one task each: $TASKS"

# partition.strategy=task hands each client a disjoint set of tasks, so an uneven count either
# starves a client or silently gives one of them two tasks. Fail here with the real reason
# rather than inside the partitioner.
NUM_TASKS="$(awk -F, '{print NF}' <<< "$TASKS")"
if [[ "$NUM_TASKS" -ne "$NUM_CLIENTS" ]]; then
  echo "!! NUM_CLIENTS ($NUM_CLIENTS) must equal the number of entries in TASKS ($NUM_TASKS)"
  echo "!! TASKS = $TASKS"
  exit 1
fi

EPS="$(episodes_for_tasks "$TASKS")"
echo "episodes: $(echo "$EPS" | tr -cd ',' | wc -c | awk '{print $1+1}') selected"

echo "seeds: $SEEDS"

for SEED in $SEEDS; do
  banner "seed $SEED"
  FED="$OUT_ROOT/federated-s$SEED"
  CEN="$OUT_ROOT/centralized-s$SEED"
  rm -rf "$FED" "$CEN"

  echo "--> federated (seed $SEED)"
  "${FL_TRAIN[@]}" "${POLICY_ARGS[@]}" \
    --dataset.repo_id="$DATASET" --dataset.episodes="$EPS" \
    --partition.strategy=task --partition.num_clients="$NUM_CLIENTS" \
    --partition.seed="$PARTITION_SEED" \
    --fl.rounds="$ROUNDS" --fl.local_steps="$LOCAL_STEPS" \
    --fl.prox_mu="$PROX_MU" \
    --server.type="$SERVER_TYPE" --server.lr="$SERVER_LR" \
    --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed="$SEED" \
    "${MP_ARGS[@]}" \
    --output_dir="$FED" \
    2>&1 | tee "$FED.log" | grep -E "^INFO.*(round|Partitioned|client)|^    clients:" || true
  assert_trained "$FED" "federated-s$SEED"

  echo "--> centralized (seed $SEED, $CENTRAL_STEPS steps)"
  "${TRAIN[@]}" "${POLICY_ARGS[@]}" \
    --dataset.repo_id="$DATASET" --dataset.episodes="$EPS" \
    --steps="$CENTRAL_STEPS" --save_freq="$CENTRAL_STEPS" \
    --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed="$SEED" \
    "${MP_ARGS[@]}" \
    --output_dir="$CEN" \
    2>&1 | tee "$CEN.log" | grep -E "^INFO.*step" || true
  assert_trained "$CEN" "centralized-s$SEED"

  run_eval "federated-s$SEED"   "$FED" "$TASKS"
  run_eval "centralized-s$SEED" "$CEN" "$TASKS"
  record "federated"   "$SEED" "$(success_rate "$FED/eval")"
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
for arm in ("federated", "centralized"):
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
  for arm in federated centralized; do
    if [[ -f "$OUT_ROOT/$arm-s$SEED/eval/eval_info.json" ]]; then
      echo "  $arm seed=$SEED"
      per_task_rates "$OUT_ROOT/$arm-s$SEED/eval"
    fi
  done
done

cat <<'EOS'

How to read this:
  * Per-task success is the number that matters, not the mean. The characteristic federated
    failure here is a respectable average hiding one client's task that collapsed entirely.
  * Compare against the IID control from run_fl_vs_centralized.sh. A gap that appears in BOTH
    is the cost of federation; the extra gap here is the cost of heterogeneity.
  * Watch `fl/client_loss_std` across rounds in the federated log. Under IID it collapses once
    clients agree. If it stays high or grows here, that divergence IS the client drift.
  * If federated trails badly, the two knobs to reach for are SERVER_TYPE=fedadam
    SERVER_LR=1e-3, and PROX_MU=0.05. Lowering LOCAL_STEPS also reduces drift directly.
  * Read the gap against the seed-to-seed std, not against zero. With EVAL_EPISODES small the
    binomial noise alone is large, so a gap under ~1 std is not a result yet - raise
    EVAL_EPISODES or add seeds before believing it.
EOS
echo "Raw logs and checkpoints: $OUT_ROOT"
