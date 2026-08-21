#!/usr/bin/env bash
#
# Federated vs centralized SmolVLA on Meta-World MT50, across two data distributions.
#
#   Scenario A (IID)      - every client holds episodes of the SAME task.
#   Scenario B (non-IID)  - every client holds a DIFFERENT, disjoint task.
#
# Each scenario runs a federated arm and a centralized arm on exactly the same episodes,
# with the same policy, seed, batch size and total number of gradient updates, then evaluates
# both in the simulator. Centralized is the ceiling: it is the same experiment with the data
# pooled on one machine. The interesting quantity is not either number on its own, it is how
# much of the gap opens up when you go from A to B.
#
# Usage:
#   bash examples/federated/run_fl_vs_centralized.sh              # default budget
#   ROUNDS=5 LOCAL_STEPS=10 EVAL_EPISODES=1 bash ...              # fast smoke run
#   SERVER_TYPE=fedadam SERVER_LR=1e-3 bash ...                   # try adaptive aggregation
#   PROX_MU=0.05 bash ...                                         # add the FedProx penalty
#
set -euo pipefail

# ----------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------
DATASET="${DATASET:-lerobot/metaworld_mt50}"
OUT_ROOT="${OUT_ROOT:-./outputs/fl-study-$(date +%Y%m%d_%H%M%S)}"

# Scenario A uses one task; scenario B uses these three, one per client.
TASK_A="${TASK_A:-drawer-close-v3}"
TASKS_B="${TASKS_B:-drawer-close-v3,button-press-v3,door-close-v3}"

NUM_CLIENTS="${NUM_CLIENTS:-3}"
ROUNDS="${ROUNDS:-2}"
LOCAL_STEPS="${LOCAL_STEPS:-5}"
BATCH_SIZE="${BATCH_SIZE:-4}"
NUM_WORKERS="${NUM_WORKERS:-2}"
SEED="${SEED:-1000}"

SERVER_TYPE="${SERVER_TYPE:-fedavg}"
SERVER_LR="${SERVER_LR:-1.0}"
PROX_MU="${PROX_MU:-0.0}"

EVAL_EPISODES="${EVAL_EPISODES:-1}"
SKIP_EVAL="${SKIP_EVAL:-0}"

# MuJoCo rendering backend, used by the eval rollouts (Meta-World renders rgb_array frames).
#   egl    - GPU-accelerated headless. The default, and what the LeRobot sim docs use.
#   osmesa - software fallback; no GPU needed, noticeably slower, needs libOSMesa installed.
#   glfw   - requires an attached display; will fail on a headless box.
# Exported so the lerobot-eval subprocesses inherit it.
export MUJOCO_GL="${MUJOCO_GL:-egl}"

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

record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$SUMMARY"; }

banner() {
  echo
  echo "========================================================================"
  echo "  $*"
  echo "========================================================================"
}

# ----------------------------------------------------------------------------------------
# Scenario A - all clients hold the SAME task (IID)
# ----------------------------------------------------------------------------------------
# The clients' shards are statistically identical, so federated averaging should land close to
# centralized. If it does not, the problem is the setup, not the heterogeneity.
#
# strategy=uniform, not task: with a single task there is nothing for `task` to split on, and
# the partitioner correctly refuses that configuration.

banner "Scenario A (IID) - $NUM_CLIENTS clients, all on $TASK_A"
EPS_A="$(episodes_for_tasks "$TASK_A")"
echo "episodes: $(echo "$EPS_A" | tr -cd ',' | wc -c | awk '{print $1+1}') selected"

rm -rf "$OUT_ROOT/A-federated" "$OUT_ROOT/A-centralized"

echo "--> [A] federated"
"${FL_TRAIN[@]}" "${POLICY_ARGS[@]}" \
  --dataset.repo_id="$DATASET" --dataset.episodes="$EPS_A" \
  --partition.strategy=uniform --partition.num_clients="$NUM_CLIENTS" \
  --fl.rounds="$ROUNDS" --fl.local_steps="$LOCAL_STEPS" \
  --fl.prox_mu="$PROX_MU" \
  --server.type="$SERVER_TYPE" --server.lr="$SERVER_LR" \
  --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed="$SEED" \
  --output_dir="$OUT_ROOT/A-federated" \
  2>&1 | tee "$OUT_ROOT/A-federated.log" | grep -E "^INFO.*(round|Partitioned|client)" || true
assert_trained "$OUT_ROOT/A-federated" "A-federated"

echo "--> [A] centralized ($CENTRAL_STEPS steps)"
"${TRAIN[@]}" "${POLICY_ARGS[@]}" \
  --dataset.repo_id="$DATASET" --dataset.episodes="$EPS_A" \
  --steps="$CENTRAL_STEPS" --save_freq="$CENTRAL_STEPS" \
  --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed="$SEED" \
  --output_dir="$OUT_ROOT/A-centralized" \
  2>&1 | tee "$OUT_ROOT/A-centralized.log" | grep -E "^INFO.*step" || true
assert_trained "$OUT_ROOT/A-centralized" "A-centralized"

run_eval "A-federated"   "$OUT_ROOT/A-federated"   "$TASK_A"
run_eval "A-centralized" "$OUT_ROOT/A-centralized" "$TASK_A"
record "A-iid" "federated"   "$TASK_A" "$(success_rate "$OUT_ROOT/A-federated/eval")"
record "A-iid" "centralized" "$TASK_A" "$(success_rate "$OUT_ROOT/A-centralized/eval")"

# ----------------------------------------------------------------------------------------
# Scenario B - every client holds a DIFFERENT task (non-IID)
# ----------------------------------------------------------------------------------------
# This is the case federated learning exists for, and the one plain FedAvg struggles with. The
# centralized arm sees all three tasks pooled; each federated client sees exactly one and must
# still contribute to a policy that can do all three.

banner "Scenario B (non-IID) - $NUM_CLIENTS clients, one task each: $TASKS_B"
EPS_B="$(episodes_for_tasks "$TASKS_B")"
echo "episodes: $(echo "$EPS_B" | tr -cd ',' | wc -c | awk '{print $1+1}') selected"

rm -rf "$OUT_ROOT/B-federated" "$OUT_ROOT/B-centralized"

echo "--> [B] federated"
"${FL_TRAIN[@]}" "${POLICY_ARGS[@]}" \
  --dataset.repo_id="$DATASET" --dataset.episodes="$EPS_B" \
  --partition.strategy=task --partition.num_clients="$NUM_CLIENTS" \
  --fl.rounds="$ROUNDS" --fl.local_steps="$LOCAL_STEPS" \
  --fl.prox_mu="$PROX_MU" \
  --server.type="$SERVER_TYPE" --server.lr="$SERVER_LR" \
  --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed="$SEED" \
  --output_dir="$OUT_ROOT/B-federated" \
  2>&1 | tee "$OUT_ROOT/B-federated.log" | grep -E "^INFO.*(round|Partitioned|client)" || true
assert_trained "$OUT_ROOT/B-federated" "B-federated"

echo "--> [B] centralized ($CENTRAL_STEPS steps)"
"${TRAIN[@]}" "${POLICY_ARGS[@]}" \
  --dataset.repo_id="$DATASET" --dataset.episodes="$EPS_B" \
  --steps="$CENTRAL_STEPS" --save_freq="$CENTRAL_STEPS" \
  --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed="$SEED" \
  --output_dir="$OUT_ROOT/B-centralized" \
  2>&1 | tee "$OUT_ROOT/B-centralized.log" | grep -E "^INFO.*step" || true
assert_trained "$OUT_ROOT/B-centralized" "B-centralized"

run_eval "B-federated"   "$OUT_ROOT/B-federated"   "$TASKS_B"
run_eval "B-centralized" "$OUT_ROOT/B-centralized" "$TASKS_B"
record "B-noniid" "federated"   "$TASKS_B" "$(success_rate "$OUT_ROOT/B-federated/eval")"
record "B-noniid" "centralized" "$TASKS_B" "$(success_rate "$OUT_ROOT/B-centralized/eval")"

# ----------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------
banner "Results (pc_success)"
printf '%-10s %-12s %-14s %s\n' SCENARIO ARM SUCCESS TASKS
while IFS=$'\t' read -r scenario arm tasks rate; do
  printf '%-10s %-12s %-14s %s\n' "$scenario" "$arm" "$rate" "$tasks"
done < "$SUMMARY"

echo
echo "Per-task breakdown:"
for run in A-federated A-centralized B-federated B-centralized; do
  if [[ -f "$OUT_ROOT/$run/eval/eval_info.json" ]]; then
    echo "  $run"
    per_task_rates "$OUT_ROOT/$run/eval"
  fi
done

cat <<'EOS'

How to read this:
  * The A gap (federated vs centralized on IID data) is your noise floor. It should be small.
  * The B gap is the real result. Whatever it exceeds the A gap by is the cost of
    heterogeneity, not of federation.
  * Also compare the two federated arms' `fl/client_loss_std` in the logs: in A it collapses
    once the clients agree, in B it stays high or grows. That divergence IS the client drift.
  * If B is much worse, re-run it with SERVER_TYPE=fedadam SERVER_LR=1e-3, or PROX_MU=0.05.
EOS
echo "Raw logs and checkpoints: $OUT_ROOT"
