#!/usr/bin/env bash
#
# Asymmetric federation: two clients share one task, a third holds a different one.
#
#   client 0:  half the episodes of SHARED_TASK
#   client 1:  the other half of SHARED_TASK
#   client 2:  all of MINORITY_TASK
#
# "Sharing a task" means disjoint episodes of it - data never moves in federated learning, so
# clients 0 and 1 hold different demonstrations of the same skill. They agree on what to
# optimise; client 2 does not.
#
# This sits between the two settings the other scripts cover. run_fl_vs_centralized.sh makes
# every client statistically identical; noniid.sh gives every client its own task. Here the
# federation has a MAJORITY that agrees and a MINORITY that does not, which is the shape most
# real fleets have - and the one where averaging can quietly serve the majority at the
# minority's expense.
#
# Watch two things the symmetric settings cannot show:
#   * whether MINORITY_TASK is served as well as SHARED_TASK by the global model;
#   * whether client 2 would have done better alone (the fed-loc column).
#
# Note that client 2 holds twice the episodes of either majority client, so under the default
# --fl.client_weighting=frames it carries roughly half the aggregate weight despite being one
# client in three. Set CLIENT_WEIGHTING=uniform to give it a third instead: that single knob
# changes whether the minority is outvoted by clients or by data.
#
# Usage:
#   bash examples/federated/majority_minority.sh
#   BUDGETS="5" LOCAL_STEPS=2 SKIP_EVAL=1 bash ...                 # fast plumbing check
#   CLIENT_WEIGHTING=uniform bash ...                              # one client, one vote
#   SHARED_CLIENTS=4 bash ...                                      # a bigger majority
#   SERVER_TYPE=fedadam SERVER_LR=1e-3 bash ...                    # adaptive aggregation
#   PROX_MU=0.05 bash ...                                          # the FedProx penalty
#   WANDB_PROJECT=fl-majority bash ...                             # stream to W&B
#
set -euo pipefail

# ----------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------
DATASET="${DATASET:-lerobot/metaworld_mt50}"
OUT_ROOT="${OUT_ROOT:-./outputs/fl-study-$(date +%Y%m%d_%H%M%S)}"

# The task the majority shares, split between SHARED_CLIENTS of them, and the task only the
# last client holds. Avoid push-v3 and push-back-v3: they share a description string and the
# resolver rejects them.
SHARED_TASK="${SHARED_TASK:-coffee-pull-v3}"
MINORITY_TASK="${MINORITY_TASK:-door-close-v3}"

# How many clients share SHARED_TASK. One more client is added for MINORITY_TASK.
SHARED_CLIENTS="${SHARED_CLIENTS:-2}"
NUM_CLIENTS=$((SHARED_CLIENTS + 1))
MINORITY_CLIENT=$SHARED_CLIENTS

# "frames" weights each client by how much data it holds, which recovers the centralized
# objective; "uniform" gives every client an equal vote regardless of volume.
CLIENT_WEIGHTING="${CLIENT_WEIGHTING:-uniform}"

# Compute budgets, as federated rounds. Every arm is rebuilt at each budget so the question
# "does the federated gap close with more compute, or persist?" gets a real answer rather than
# a single snapshot.
#
# These are separate runs, not checkpoints pulled from one long run. A round-100 checkpoint of
# a 400-round run is NOT the 100-round model: cfg.steps == rounds drives the LR schedule, so
# the long run is still mid-anneal there and would look artificially weak. Because the budgets
# are geometric the honest version is barely more expensive - 125+500+2000 updates is 1.3x the
# 2000-update run on its own.
BUDGETS="${BUDGETS:-25 100 400}"
LOCAL_STEPS="${LOCAL_STEPS:-5}"
BATCH_SIZE="${BATCH_SIZE:-64}"

# Dataloader workers PER CLIENT. Each client keeps its own dataloader alive across rounds (that
# is what stops every round replaying the same first batches), so the process count is
# NUM_CLIENTS x NUM_WORKERS - 12 at the defaults. On a scheduler that pins you to
# --cpus-per-task, size this as (allocated cores / NUM_CLIENTS), not as the node's core count,
# or the workers oversubscribe and the GPU starves anyway.
NUM_WORKERS="${NUM_WORKERS:-12}"

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

# Also train one model per client on that client's shard alone, with no aggregation. This is
# the floor federation has to beat: what each client could already achieve by itself. It costs
# NUM_CLIENTS extra training runs per seed, so it is the expensive addition here.
RUN_LOCAL="${RUN_LOCAL:-1}"
# Evaluate each local model on EVERY task, not only the one it trained on. Its own-task score
# still feeds the fed-loc comparison; the rest measure what specialisation costs - a local
# specialist on a task it has never seen. That contrast is the strongest argument FOR
# federation, and the own-task-only view cannot make it. One eval per local model either way,
# just over more tasks, so the cost is roughly NUM_CLIENTS x the local eval time.
LOCAL_EVAL_ALL_TASKS="${LOCAL_EVAL_ALL_TASKS:-0}"

# Weights & Biases. Empty project disables it entirely, which is the default. When set, every
# training run streams its own curves (the federated arm additionally logs fl/client_loss_std
# and the per-client losses), and the finished study is uploaded as one extra run holding the
# results table and the gap-vs-budget curves.
WANDB_PROJECT="${WANDB_PROJECT:-fedvla}"
WANDB_ENTITY="${WANDB_ENTITY:-qduongminh3tcd}"

WANDB_ARGS=()
if [[ -n "$WANDB_PROJECT" ]]; then
  WANDB_ARGS=(--wandb.enable=true --wandb.project="$WANDB_PROJECT")
  [[ -n "$WANDB_ENTITY" ]] && WANDB_ARGS+=(--wandb.entity="$WANDB_ENTITY")
fi

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
# How the baselines are sized against a federated budget of R rounds:
#   local       R x LOCAL_STEPS               - the updates one client performs, so "did joining
#                                               beat going alone?" is asked at equal per-model
#                                               progress.
#   centralized R x LOCAL_STEPS x NUM_CLIENTS - the updates the federation performs in total, so
#                                               the ceiling is not handicapped on compute.
# Set MATCH_CENTRAL_TO_CLIENT=1 to size centralized like local instead (equal global progress
# rather than equal total compute); the two framings answer different questions.
MATCH_CENTRAL_TO_CLIENT="${MATCH_CENTRAL_TO_CLIENT:-0}"

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

# One client's shard, read back out of the very partition the federated arm uses, so the local
# arm provably trains on the same episodes that client held. Emits "<episode list>\t<slugs>";
# the slugs are needed because --env.task takes task names while the partition speaks in the
# dataset's natural-language descriptions.
client_shard() {
  python - "$DATASET" "$1" "$2" "$3" "$4" "$ASSIGNMENT" <<'SHARD_EOF'
import json
import sys
from pathlib import Path

import lerobot.envs
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.federated.config import PartitionConfig
from lerobot.federated.partition import partition_episodes

dataset, eps_json, num_clients, client_id, seed, assignment = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
)
meta = LeRobotDatasetMetadata(dataset)
shards = partition_episodes(
    meta,
    PartitionConfig(
        strategy="explicit", num_clients=num_clients, seed=seed, task_assignment=json.loads(assignment)
    ),
    episodes=json.loads(eps_json),
)
shard = shards[client_id]

cfg = json.loads((Path(lerobot.envs.__file__).parent / "metaworld_config.json").read_text())
desc2slug = {}
for slug, description in cfg["TASK_DESCRIPTIONS"].items():
    desc2slug.setdefault(description, []).append(slug)

slugs = []
for description in shard.tasks:
    matches = desc2slug.get(description, [])
    if len(matches) != 1:
        sys.exit(f"description {description!r} maps to {len(matches)} task slugs; cannot target eval")
    slugs.append(matches[0])

print("[" + ",".join(str(e) for e in shard.episodes) + "]\t" + ",".join(sorted(slugs)))
SHARD_EOF
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
  [[ -f "$f" ]] || return 0
  python - "$f" <<'PERTASK_EOF' 2>/dev/null || true
import json, sys

payload = json.load(open(sys.argv[1]))
for name, info in sorted((payload.get("per_group") or {}).items()):
    rate = info.get("pc_success")
    if rate is not None:
        print(f"      {name:32} {rate:5.1f}%  (n={info.get('n_episodes', '?')})")
PERTASK_EOF
}

# Split one local model's per-task rates into the task(s) it owns and the ones it never saw,
# recording them under different arms so the summary can show both without a second eval.
record_local_rates() {
  local seed="$1" evaldir="$2" own="$3" out=""
  [[ -f "$evaldir/eval_info.json" ]] || return 0
  out="$(
    python - "$evaldir/eval_info.json" "$own" <<'LOC_EOF' 2>/dev/null || true
import json
import sys

payload = json.load(open(sys.argv[1]))
own = set(sys.argv[2].split(","))
for name, info in sorted((payload.get("per_group") or {}).items()):
    rate = info.get("pc_success")
    if rate is not None:
        print(f"{'local' if name in own else 'local_offtask'}\t{name}\t{rate:.1f}%")
LOC_EOF
  )"
  while IFS=$'\t' read -r arm task rate; do
    [[ -n "$task" ]] && record "$arm" "$seed" "$task" "$rate"
  done <<< "$out"
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

record() { printf '%s\t%s\t%s\t%s\t%s\n' "$1" "$2" "$BUDGET" "$3" "$4" >> "$SUMMARY"; }

# One row per task per arm per seed, so the summary can average a task over seeds and put the
# two arms side by side. Without this the per-task numbers exist only in each run's
# eval_info.json and can only be eyeballed one seed at a time.
record_per_task() {
  local arm="$1" seed="$2" evaldir="$3" out=""
  [[ -f "$evaldir/eval_info.json" ]] || return 0
  # The heredoc is captured first and the loop reads it afterwards: a heredoc whose command is
  # piped into a multi-line `while` swallows the loop body as heredoc content.
  out="$(
    python - "$evaldir/eval_info.json" <<'PT_EOF' 2>/dev/null || true
import json, sys

payload = json.load(open(sys.argv[1]))
for name, info in sorted((payload.get("per_group") or {}).items()):
    rate = info.get("pc_success")
    if rate is not None:
        print(f"{name}\t{rate:.1f}%")
PT_EOF
  )"
  while IFS=$'\t' read -r task rate; do
    [[ -n "$task" ]] && record "$arm" "$seed" "$task" "$rate"
  done <<< "$out"
}

banner() {
  echo
  echo "========================================================================"
  echo "  $*"
  echo "========================================================================"
}

# ----------------------------------------------------------------------------------------
# Majority/minority split
# ----------------------------------------------------------------------------------------

banner "Majority/minority - $SHARED_CLIENTS clients share $SHARED_TASK, 1 holds $MINORITY_TASK"

if [[ "$SHARED_TASK" == "$MINORITY_TASK" ]]; then
  echo "!! SHARED_TASK and MINORITY_TASK must differ; both are '$SHARED_TASK'."
  exit 1
fi
if (( SHARED_CLIENTS < 2 )); then
  echo "!! SHARED_CLIENTS must be >= 2 for there to be a majority; got $SHARED_CLIENTS."
  exit 1
fi

# client -> tasks. Clients 0..SHARED_CLIENTS-1 all name SHARED_TASK, so the partitioner splits
# its episodes evenly between them; the last client names MINORITY_TASK alone and gets all of it.
ASSIGNMENT="$(python - "$SHARED_CLIENTS" "$SHARED_TASK" "$MINORITY_TASK" <<'ASSIGN_EOF'
import json
import sys
from pathlib import Path

import lerobot.envs

# The partitioner groups episodes by the dataset's natural-language task description, not by
# the Meta-World slug, so the assignment has to be written in descriptions. Slugs stay the
# user-facing vocabulary because that is what --env.task takes.
shared_clients, shared_slug, minority_slug = int(sys.argv[1]), sys.argv[2], sys.argv[3]
cfg = json.loads((Path(lerobot.envs.__file__).parent / "metaworld_config.json").read_text())
desc = cfg["TASK_DESCRIPTIONS"]
for slug in (shared_slug, minority_slug):
    if slug not in desc:
        sys.exit(f"unknown Meta-World task: {slug!r}")

assignment = {str(c): [desc[shared_slug]] for c in range(shared_clients)}
assignment[str(shared_clients)] = [desc[minority_slug]]
print(json.dumps(assignment))
ASSIGN_EOF
)"
echo "assignment: $ASSIGNMENT"

TASKS="$SHARED_TASK,$MINORITY_TASK"
EPS="$(episodes_for_tasks "$TASKS")"
echo "episodes: $(echo "$EPS" | tr -cd ',' | wc -c | awk '{print $1+1}') selected"

# Everything one (budget, seed) group does: both arms, their evals, and the local models.
run_seed_group() {
  local BUDGET="$1" SEED="$2"

  local CLIENT_UPDATES=$((BUDGET * LOCAL_STEPS))
  local LOCAL_ARM_STEPS=$CLIENT_UPDATES
  local CENTRAL_STEPS
  if [[ "$MATCH_CENTRAL_TO_CLIENT" == "1" ]]; then
    CENTRAL_STEPS=$CLIENT_UPDATES
  else
    CENTRAL_STEPS=$((CLIENT_UPDATES * NUM_CLIENTS))
  fi

  banner "budget $BUDGET - seed $SEED"
  FED="$OUT_ROOT/federated-b$BUDGET-s$SEED"
  CEN="$OUT_ROOT/centralized-b$BUDGET-s$SEED"
  rm -rf "$FED" "$CEN"

  echo "--> federated (budget $BUDGET, seed $SEED)"
  "${FL_TRAIN[@]}" "${POLICY_ARGS[@]}" \
    --dataset.repo_id="$DATASET" --dataset.episodes="$EPS" \
    --partition.strategy=explicit --partition.num_clients="$NUM_CLIENTS" \
    --partition.task_assignment="$ASSIGNMENT" \
    --partition.seed="$PARTITION_SEED" \
    --fl.client_weighting="$CLIENT_WEIGHTING" \
    --fl.rounds="$BUDGET" --fl.local_steps="$LOCAL_STEPS" \
    --fl.prox_mu="$PROX_MU" \
    --server.type="$SERVER_TYPE" --server.lr="$SERVER_LR" \
    --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed="$SEED" \
    "${MP_ARGS[@]}" "${WANDB_ARGS[@]}" --job_name="fed-b$BUDGET-s$SEED" \
    --output_dir="$FED" \
    2>&1 | tee "$FED.log" | grep -E "^INFO.*(round|Partitioned|client)|^    clients:" || true
  assert_trained "$FED" "federated-b$BUDGET-s$SEED"

  echo "--> centralized (budget $BUDGET, seed $SEED, $CENTRAL_STEPS steps)"
  "${TRAIN[@]}" "${POLICY_ARGS[@]}" \
    --dataset.repo_id="$DATASET" --dataset.episodes="$EPS" \
    --steps="$CENTRAL_STEPS" --save_freq="$CENTRAL_STEPS" \
    --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed="$SEED" \
    "${MP_ARGS[@]}" "${WANDB_ARGS[@]}" --job_name="cen-b$BUDGET-s$SEED" \
    --output_dir="$CEN" \
    2>&1 | tee "$CEN.log" | grep -E "^INFO.*step" || true
  assert_trained "$CEN" "centralized-b$BUDGET-s$SEED"

  run_eval "federated-b$BUDGET-s$SEED"   "$FED" "$TASKS"
  run_eval "centralized-b$BUDGET-s$SEED" "$CEN" "$TASKS"
  record "federated"   "$SEED" __overall__ "$(success_rate "$FED/eval")"
  record "centralized" "$SEED" __overall__ "$(success_rate "$CEN/eval")"
  record_per_task "federated"   "$SEED" "$FED/eval"
  record_per_task "centralized" "$SEED" "$CEN/eval"

  if [[ "$RUN_LOCAL" == "1" ]]; then
    for ((c = 0; c < NUM_CLIENTS; c++)); do
      IFS=$'\t' read -r C_EPS C_TASKS < <(client_shard "$EPS" "$NUM_CLIENTS" "$c" "$PARTITION_SEED")
      LOC="$OUT_ROOT/local-c$c-b$BUDGET-s$SEED"
      rm -rf "$LOC"
      echo "--> local: client $c alone on $C_TASKS (budget $BUDGET, seed $SEED, $LOCAL_ARM_STEPS steps)"
      "${TRAIN[@]}" "${POLICY_ARGS[@]}" \
        --dataset.repo_id="$DATASET" --dataset.episodes="$C_EPS" \
        --steps="$LOCAL_ARM_STEPS" --save_freq="$LOCAL_ARM_STEPS" \
        --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed="$SEED" \
        "${MP_ARGS[@]}" "${WANDB_ARGS[@]}" --job_name="loc-c$c-b$BUDGET-s$SEED" \
        --output_dir="$LOC" \
        2>&1 | tee "$LOC.log" | grep -E "^INFO.*step" || true
      assert_trained "$LOC" "local-c$c-b$BUDGET-s$SEED"
      # Own tasks answer "what does this client achieve unaided?"; all tasks additionally
      # answer "what did specialising cost it elsewhere?".
      if [[ "$LOCAL_EVAL_ALL_TASKS" == "1" ]]; then
        run_eval "local-c$c-b$BUDGET-s$SEED" "$LOC" "$TASKS"
      else
        run_eval "local-c$c-b$BUDGET-s$SEED" "$LOC" "$C_TASKS"
      fi
      record_local_rates "$SEED" "$LOC/eval" "$C_TASKS"
    done
  fi
}

echo "seeds: $SEEDS"
echo "budgets (rounds): $BUDGETS"

for BUDGET in $BUDGETS; do
  banner "budget: $BUDGET rounds x $LOCAL_STEPS local steps = $((BUDGET * LOCAL_STEPS)) updates/client"
  for SEED in $SEEDS; do
    run_seed_group "$BUDGET" "$SEED"
  done
done

# ----------------------------------------------------------------------------------------
# Summary
# ----------------------------------------------------------------------------------------
banner "Results (pc_success over $(wc -w <<< "$SEEDS") seeds)"

python - "$SUMMARY" "$LOCAL_STEPS" <<'AGG_EOF'
import statistics
import sys
from collections import defaultdict

# (budget, task, arm) -> {seed: [rate, ...]}. A list, not a scalar: `local_offtask` gets one row
# per non-owner client per seed, and keying on the seed alone would keep only the last of them.
rows = defaultdict(lambda: defaultdict(list))
for line in open(sys.argv[1]):
    arm, seed, budget, task, rate = line.rstrip("\n").split("\t")
    rows[(int(budget), task, arm)][seed].append(rate)

local_steps = int(sys.argv[2])

budgets = sorted({b for b, _, _ in rows})
seeds = sorted({s for per_seed in rows.values() for s in per_seed})
tasks = sorted({t for _, t, _ in rows if t != "__overall__"})
ARMS = ("local", "federated", "centralized")
has_offtask = any(a == "local_offtask" for _, _, a in rows)


def values(budget, task, arm):
    """Every numeric rate for one cell, over all seeds and all contributing models."""
    return [
        float(v[:-1])
        for per_seed in (rows.get((budget, task, arm)) or {}).values()
        for v in per_seed
        if v.endswith("%")
    ]


def overall(budget, arm):
    """One arm's headline numbers across seeds at a budget.

    federated/centralized are single models and report their own overall. `local` is not one
    model but NUM_CLIENTS of them, each scored on the task it owns, so there is no overall to
    read - it is macro-averaged over tasks instead, weighting every task equally.
    """
    direct = rows.get((budget, "__overall__", arm))
    if direct:
        return [float(v[0][:-1]) for v in direct.values() if v[0].endswith("%")]
    out = []
    for seed in seeds:
        per_task = [
            float(v[:-1])
            for t in tasks
            for v in (rows.get((budget, t, arm)) or {}).get(seed, [])
            if v.endswith("%")
        ]
        if per_task:
            out.append(statistics.mean(per_task))
    return out


def cell(vs, width=15):
    if not vs:
        return f"{'n/a':>{width}}"
    text = f"{statistics.mean(vs):.1f}%"
    # A spread over one value is not a spread; leave it off rather than printing +/-0.0.
    if len(vs) > 1:
        text += f" +/-{statistics.stdev(vs):.1f}"
    return f"{text:>{width}}"


def delta(a, b, width=10):
    return f"{statistics.mean(a) - statistics.mean(b):>{width - 1}.1f}%" if a and b else f"{'n/a':>{width}}"


print("Overall, by compute budget (mean over seeds):")
print(f"{'budget':>8}{'updates/cli':>13}" + "".join(f"{a:>15}" for a in ARMS) + f"{'fed-loc':>10}{'cen-fed':>10}")
for b in budgets:
    loc, fed, cen = (overall(b, a) for a in ARMS)
    print(
        f"{b:>8}{b * local_steps:>13}"
        + cell(loc) + cell(fed) + cell(cen) + delta(fed, loc) + delta(cen, fed)
    )

print("\nPer task and budget (gap = centralized - federated):")
off_head = f"{'local-off':>15}" if has_offtask else ""
print(
    f"{'task':<24}{'budget':>8}{'local':>15}{off_head}{'federated':>15}{'centralized':>15}"
    f"{'fed-loc':>10}{'cen-fed':>10}"
)
for task in tasks:
    for b in budgets:
        loc, fed, cen = (values(b, task, a) for a in ARMS)
        off = values(b, task, "local_offtask")
        off_cell = cell(off) if has_offtask else ""
        print(
            f"{task[:23]:<24}{b:>8}{cell(loc)}{off_cell}{cell(fed)}{cell(cen)}"
            f"{delta(fed, loc)}{delta(cen, fed)}"
        )
    print()

if has_offtask:
    print("  local     = the client that OWNS this task, trained alone on it")
    print("  local-off = the other clients' local models, which never saw this task")
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

cat <<EOS

How to read this:
  * The comparison that matters is $MINORITY_TASK against $SHARED_TASK. Two of $NUM_CLIENTS
    clients push the global model toward the shared task every round; the minority client
    pushes alone. If its task scores materially worse, averaging served the majority.
  * fed-loc on the minority row answers the question its owner would actually ask: was joining
    worth it? Negative means that client would have done better keeping its data out.
  * The majority clients are the control. They hold the SAME task as each other, so their rows
    show what federation costs when clients already agree - the floor any minority gap should
    be read against.
  * Re-run with CLIENT_WEIGHTING=uniform. With "frames" the minority client holds twice the
    episodes of either majority client and carries about half the weight; with "uniform" it
    carries a third. If the minority gap opens up under uniform, the minority was being
    protected by its data volume rather than by the algorithm.
  * If the minority is underserved, SERVER_TYPE=fedadam and PROX_MU=0.05 are the two knobs
    that target client drift directly.
EOS
# The per-run curves live in their own W&B runs; this uploads the STUDY - the thing those runs
# cannot show individually, because every number here is a comparison across them.
if [[ -n "$WANDB_PROJECT" ]]; then
  echo
  echo "Uploading study results to W&B project '$WANDB_PROJECT'..."
  python - "$SUMMARY" "$WANDB_PROJECT" "$WANDB_ENTITY" "$LOCAL_STEPS" "$(basename "$OUT_ROOT")" <<'WB_EOF' || echo "!! W&B upload failed (results are still in $SUMMARY)"
import statistics
import sys
from collections import defaultdict

import wandb

summary_path, project, entity, local_steps, run_name = sys.argv[1:6]
local_steps = int(local_steps)

rows = defaultdict(lambda: defaultdict(list))
raw = []
for line in open(summary_path):
    arm, seed, budget, task, rate = line.rstrip("\n").split("\t")
    rows[(int(budget), task, arm)][seed].append(rate)
    raw.append((arm, int(seed), int(budget), task, rate))

budgets = sorted({b for b, _, _ in rows})
tasks = sorted({t for _, t, _ in rows if t != "__overall__"})
ARMS = ("local", "federated", "centralized")


def values(budget, task, arm):
    return [
        float(v[:-1])
        for per_seed in (rows.get((budget, task, arm)) or {}).values()
        for v in per_seed
        if v.endswith("%")
    ]


def overall(budget, arm):
    direct = rows.get((budget, "__overall__", arm))
    if direct:
        return [float(v[0][:-1]) for v in direct.values() if v[0].endswith("%")]
    seeds = {s for key in rows for s in rows[key]}
    out = []
    for seed in sorted(seeds):
        per_task = [
            float(v[:-1])
            for t in tasks
            for v in (rows.get((budget, t, arm)) or {}).get(seed, [])
            if v.endswith("%")
        ]
        if per_task:
            out.append(statistics.mean(per_task))
    return out


run = wandb.init(
    project=project,
    entity=entity or None,
    name=f"summary-{run_name}",
    job_type="study-summary",
    tags=["summary"],
)

table = wandb.Table(columns=["arm", "seed", "budget", "task", "pc_success"])
for arm, seed, budget, task, rate in raw:
    table.add_data(arm, seed, budget, task, float(rate[:-1]) if rate.endswith("%") else None)
run.log({"results": table})

# One step per budget, so W&B draws success and both gaps against compute. This is the whole
# experiment in one chart: whether the federated gap closes as the budget grows or persists.
for b in budgets:
    loc, fed, cen = (overall(b, a) for a in ARMS)
    payload = {"budget_rounds": b, "updates_per_client": b * local_steps}
    for name, vs in (("local", loc), ("federated", fed), ("centralized", cen)):
        if vs:
            payload[f"overall/{name}"] = statistics.mean(vs)
            if len(vs) > 1:
                payload[f"overall/{name}_std"] = statistics.stdev(vs)
    if fed and loc:
        payload["gap/fed_minus_local"] = statistics.mean(fed) - statistics.mean(loc)
    if cen and fed:
        payload["gap/central_minus_fed"] = statistics.mean(cen) - statistics.mean(fed)
    for task in tasks:
        tl, tf, tc = (values(b, task, a) for a in ARMS)
        if tf:
            payload[f"task/{task}/federated"] = statistics.mean(tf)
        if tf and tl:
            payload[f"task/{task}/fed_minus_local"] = statistics.mean(tf) - statistics.mean(tl)
        if tc and tf:
            payload[f"task/{task}/central_minus_fed"] = statistics.mean(tc) - statistics.mean(tf)
    run.log(payload)

# Final-budget numbers as run summary fields, so studies are sortable in the W&B runs table.
if budgets:
    last = budgets[-1]
    for name, vs in zip(ARMS, (overall(last, a) for a in ARMS), strict=True):
        if vs:
            run.summary[f"final/{name}"] = statistics.mean(vs)
run.finish()
print("uploaded")
WB_EOF
fi

echo "Raw logs and checkpoints: $OUT_ROOT"
