#!/usr/bin/env bash
#
# Federated vs centralized SmolVLA on Meta-World MT50, under task heterogeneity.
#
# Every client holds a DIFFERENT, disjoint set of tasks. This is the case federated learning
# exists for, and the one plain FedAvg struggles with: a client that has never seen
# `door-close-v3` must still contribute to a global policy that can do it.
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
#   BUDGETS="25 100 400" bash ...                                  # three compute budgets
#   BUDGETS=100 bash ...                                           # a single budget
#   BUDGETS="5" LOCAL_STEPS=2 SKIP_EVAL=1 bash ...                 # fast plumbing check
#   SERVER_TYPE=fedadam SERVER_LR=1e-3 bash ...                   # adaptive aggregation
#   PROX_MU=0.05 bash ...                                         # the FedProx penalty
#   TASKS_PER_CLIENT=1 TASKS=reach-v3,window-open-v3,door-open-v3 bash ...  # one task each
#   TASKS=<9 comma-separated slugs> bash ...                       # different tasks
#   WANDB_PROJECT=fl-noniid bash ...                               # stream to Weights & Biases
#   ENV_EVAL_EVERY=10 bash ...                                     # rollouts during training
#   RUN_LOCAL=0 bash ...                                           # skip the expensive arm
#   RUN_FEDERATED=1 RUN_CENTRALIZED=0 RUN_LOCAL=0 bash ...          # one arm only
#   SEEDS="1000 2000 3000 4000 5000" bash ...                      # more seeds
#   BATCH_SIZE=128 NUM_WORKERS=8 bash ...                          # A100 80GB
#   BATCH_SIZE=8 MIXED_PRECISION=no bash ...                       # small / older GPU
#
# LIBERO - separating TASK heterogeneity from ENVIRONMENT heterogeneity:
#   BENCHMARK=libero SUITE=libero_goal    bash ...   # task difference: one kitchen scene, each
#                                                    #   client pursues a different goal in it
#   BENCHMARK=libero SUITE=libero_spatial bash ...   # environment difference (layout): every
#                                                    #   client does "black bowl -> plate", but
#                                                    #   in a different scene arrangement
#   BENCHMARK=libero SUITE=libero_object  bash ...   # environment difference (objects): every
#                                                    #   client does "<object> -> basket", with a
#                                                    #   different object
#   BENCHMARK=libero SUITE=libero_goal TASKS=0,1,2,3,4,5 NUM_CLIENTS=3 TASKS_PER_CLIENT=2 bash ...
# Under LIBERO, TASKS are task ids (0-9) within SUITE. Keep NUM_CLIENTS, TASKS_PER_CLIENT and the
# task count identical across the three suites so the only thing that changes is the KIND of
# heterogeneity. Results are keyed "<suite>_<id>"; the id -> instruction legend is printed at
# startup.
#
set -euo pipefail

# ----------------------------------------------------------------------------------------
# Configuration
# ----------------------------------------------------------------------------------------
# metaworld | libero. Everything downstream - episode lookup, the eval env, per-task result
# parsing - dispatches on this.
BENCHMARK="${BENCHMARK:-metaworld}"
# LIBERO suite. libero_goal is the task-heterogeneity condition; libero_spatial and
# libero_object are the environment-heterogeneity conditions (layouts and objects respectively).
SUITE="${SUITE:-libero_goal}"

case "$BENCHMARK" in
  metaworld)
    DATASET="${DATASET:-lerobot/metaworld_mt50}"
    STUDY_TAG="metaworld"
    JOB_PREFIX=""
    ;;
  libero)
    case "$SUITE" in
      libero_goal | libero_spatial | libero_object | libero_10) ;;
      *)
        echo "!! SUITE=$SUITE is not a LIBERO suite (libero_goal, libero_spatial, libero_object, libero_10)."
        exit 1
        ;;
    esac
    # Video-encoded and 37x smaller than HuggingFaceVLA/libero; same demonstrations and schema.
    DATASET="${DATASET:-lerobot/libero}"
    STUDY_TAG="$SUITE"
    # Runs from different suites land in the same W&B project, so their names must not collide.
    JOB_PREFIX="$SUITE-"
    ;;
  *)
    echo "!! BENCHMARK=$BENCHMARK is not supported; use metaworld or libero."
    exit 1
    ;;
esac
OUT_ROOT="${OUT_ROOT:-./outputs/fl-study-$STUDY_TAG-$(date +%Y%m%d_%H%M%S)}"

# Tasks are dealt out so that each client holds TASKS_PER_CLIENT of them, disjointly - the
# partition is still fully non-IID, but each client's shard is TASKS_PER_CLIENT times larger.
#
# Shard size matters here beyond throughput. With one task per client a shard is ~4k frames, so
# at BATCH_SIZE=64 a client re-reads its own data every 64 steps and local overfitting starts
# masquerading as client drift. More tasks per client keeps the heterogeneity while removing
# that confound. TASKS must hold exactly NUM_CLIENTS x TASKS_PER_CLIENT entries.
#
# Meta-World: task slugs. Avoid push-v3 and push-back-v3: they share a description string and the
# resolver rejects them.
# LIBERO: task ids within SUITE. LIBERO tasks hold 29-50 demos each (failed ones were filtered),
# so unlike Meta-World the shards are not exactly equal in size.
if [[ "$BENCHMARK" == "libero" ]]; then
  TASKS="${TASKS:-0,1,2}"
else
  TASKS="${TASKS:-coffee-pull-v3,door-close-v3,drawer-close-v3}"
fi
TASKS_PER_CLIENT="${TASKS_PER_CLIENT:-1}"

NUM_CLIENTS="${NUM_CLIENTS:-3}"

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

# Which arms to run. Turn one off to reuse results you already have, or to get a first answer
# cheaply - the summary simply omits any arm with no rows, and $SUMMARY is append-only, so runs
# with different arms enabled can be concatenated and aggregated together afterwards.
#
#   federated    the federation itself.
#   centralized  the same episodes pooled on one machine - the ceiling.
#   local        one model per client on its own shard, no aggregation - the floor federation
#                has to beat. The expensive one: NUM_CLIENTS extra training runs per seed.
RUN_FEDERATED="${RUN_FEDERATED:-1}"
RUN_CENTRALIZED="${RUN_CENTRALIZED:-1}"
RUN_LOCAL="${RUN_LOCAL:-1}"
# Evaluate each local model on EVERY task, not only the one it trained on. Its own-task score
# still feeds the fed-loc comparison; the rest measure what specialisation costs - a local
# specialist on a task it has never seen. That contrast is the strongest argument FOR
# federation, and the own-task-only view cannot make it. One eval per local model either way,
# just over more tasks, so the cost is roughly NUM_CLIENTS x the local eval time.
LOCAL_EVAL_ALL_TASKS="${LOCAL_EVAL_ALL_TASKS:-1}"

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

# Roll out in the simulator DURING training, not only on the final checkpoint: every N rounds
# for the federated arm, every N x LOCAL_STEPS optimizer steps for the others, so all arms are
# measured at the same intervals of per-model progress. 0 disables it, which is the default.
#
# Mostly redundant with BUDGETS, which already measures success against compute - and measures
# it better, because each budget is a separate run whose LR schedule has fully annealed, while
# a mid-training rollout catches the model with its LR still high. Worth enabling for a single
# diagnostic run (to see WHEN a federated arm diverges, not just that it did); expensive to
# leave on across a 45-run grid, since every rollout is a full episode of simulation.
ENV_EVAL_EVERY="${ENV_EVAL_EVERY:-0}"
# Episodes per task for those in-training rollouts. Deliberately smaller than EVAL_EPISODES:
# these are for the shape of the curve, not for the numbers you report.
ENV_EVAL_EPISODES="${ENV_EVAL_EPISODES:-5}"

EVAL_EPISODES="${EVAL_EPISODES:-15}"
SKIP_EVAL="${SKIP_EVAL:-1}"

# MuJoCo rendering backend, used by the eval rollouts (Meta-World and LIBERO/robosuite both
# render rgb_array frames).
#   egl    - GPU-accelerated headless. The default, and what the LeRobot sim docs use.
#   osmesa - software fallback; no GPU needed, noticeably slower, needs libOSMesa installed.
#   glfw   - requires an attached display; will fail on a headless box.
# LIBERO caveat: robosuite overrides every value except osmesa and glx with egl. Where EGL lacks
# headless device support (WSL2 among them: "Cannot initialize a EGL device display"), use
# MUJOCO_GL=glx with a display (WSLg provides one), or osmesa.
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

# In-training rollouts need an environment; without --env.type the frequency flags are inert.
# The env itself (type + which tasks) comes from env_target at each call site, since every arm
# rolls out on a different task set.
ENV_ARGS=()
FED_ENV_ARGS=()
if (( ENV_EVAL_EVERY > 0 )); then
  ENV_ARGS=(
    --eval.batch_size=1
    --eval.n_episodes="$ENV_EVAL_EPISODES"
  )
  FED_ENV_ARGS=("${ENV_ARGS[@]}" --fl.env_eval_freq_rounds="$ENV_EVAL_EVERY")
fi

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

# Every task the benchmark offers, as "<key>\t<instruction>" lines. Keys are what the rest of the
# script speaks - Meta-World slugs, or "<suite>_<id>" for LIBERO - while the dataset only knows
# the instruction strings, so this table is the one place the two are joined. It lists the WHOLE
# benchmark, not only the selected tasks, so a selected task whose instruction is shared with an
# unselected one is still caught.
write_task_table() {
  python - "$BENCHMARK" "$SUITE" <<'TT_EOF'
import contextlib
import json
import sys
from pathlib import Path

benchmark, suite = sys.argv[1:3]
if benchmark == "metaworld":
    import lerobot.envs

    cfg = json.loads((Path(lerobot.envs.__file__).parent / "metaworld_config.json").read_text())
    table = list(cfg["TASK_DESCRIPTIONS"].items())
else:
    # LIBERO prints progress to stdout, which is where this table goes.
    with contextlib.redirect_stdout(sys.stderr):
        from libero.libero import benchmark as libero_benchmark

        tasks = libero_benchmark.get_benchmark_dict()[suite]()
        table = [(f"{suite}_{i}", tasks.get_task(i).language) for i in range(tasks.n_tasks)]
for key, description in table:
    print(f"{key}\t{description}")
TT_EOF
}

# Resolve comma-separated task keys to a draccus episode list, e.g. "[900,901,...]".
#
# Episodes are matched through each task's instruction, because a task's position in the dataset
# does NOT follow its id: in Meta-World the two disagree for 23 of the 50 tasks, and in LIBERO
# the episodes of different tasks are interleaved. The key -> instruction legend goes to stderr.
episodes_for_tasks() {
  python - "$1" "$DATASET" "$TASK_TABLE" <<'PYEOF'
import sys

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.federated.partition import episode_tasks

keys = sys.argv[1].split(",")
desc = dict(line.rstrip("\n").split("\t", 1) for line in open(sys.argv[3]))

meta = LeRobotDatasetMetadata(sys.argv[2])
task_of = episode_tasks(meta, list(range(meta.total_episodes)))

out = []
for key in keys:
    if key not in desc:
        sys.exit(f"unknown task {key!r}; known: {', '.join(desc)}")
    # Meta-World's push-v3 and push-back-v3 share one instruction, so a lookup by instruction
    # would return both tasks' episodes. Refuse rather than silently doubling the data.
    twins = [k for k, d in desc.items() if d == desc[key] and k != key]
    if twins:
        sys.exit(f"{key!r} shares its instruction with {twins}; their episodes cannot be told apart")
    hits = [ep for ep, task in task_of.items() if task == desc[key]]
    if not hits:
        sys.exit(f"no episodes found for {key!r} ({desc[key]!r}) in this dataset")
    print(f"  {key:<20} {len(hits):>3} episodes  {desc[key]}", file=sys.stderr)
    out.extend(hits)

print("[" + ",".join(str(i) for i in sorted(out)) + "]")
PYEOF
}

# One client's shard, read back out of the very partition the federated arm uses, so the local
# arm provably trains on the same episodes that client held. Emits "<episode list>\t<task keys>";
# the keys are needed because the eval env takes task names/ids while the partition speaks in
# the dataset's natural-language instructions.
client_shard() {
  python - "$DATASET" "$1" "$2" "$3" "$4" "$TASK_TABLE" <<'SHARD_EOF'
import json
import sys

from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata
from lerobot.federated.config import PartitionConfig
from lerobot.federated.partition import partition_episodes

dataset, eps_json, num_clients, client_id, seed, table = (
    sys.argv[1], sys.argv[2], int(sys.argv[3]), int(sys.argv[4]), int(sys.argv[5]), sys.argv[6]
)
meta = LeRobotDatasetMetadata(dataset)
shards = partition_episodes(
    meta,
    PartitionConfig(strategy="task", num_clients=num_clients, seed=seed),
    episodes=json.loads(eps_json),
)
shard = shards[client_id]

desc2key = {}
for line in open(table):
    key, description = line.rstrip("\n").split("\t", 1)
    desc2key.setdefault(description, []).append(key)

keys = []
for description in shard.tasks:
    matches = desc2key.get(description, [])
    if len(matches) != 1:
        sys.exit(f"instruction {description!r} maps to {len(matches)} task keys; cannot target eval")
    keys.append(matches[0])

print("[" + ",".join(str(e) for e in shard.episodes) + "]\t" + ",".join(sorted(keys)))
SHARD_EOF
}

# The env flags that roll out exactly the given comma-separated task keys, one flag per line
# (read them with mapfile). LIBERO evaluates a suite restricted to task ids; Meta-World takes the
# slugs directly.
env_target() {
  if [[ "$BENCHMARK" == "libero" ]]; then
    local ids
    ids="$(tr ',' '\n' <<< "$1" | sed "s/^${SUITE}_//" | paste -sd, -)"
    printf '%s\n' --env.type=libero --env.task="$SUITE" --env.task_ids="[$ids]"
  else
    printf '%s\n' --env.type=metaworld --env.task="$1"
  fi
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

# "<task key>\t<rate>%\t<episodes>" for every task in a finished eval. Meta-World evals are
# launched with one group per slug, so per_group is already per task. LIBERO groups by suite,
# which would collapse every task into one number, so its per-task rates are rebuilt from
# per_task and keyed "<suite>_<id>" to match the task keys used everywhere else.
task_rates() {
  local f="$1/eval_info.json"
  [[ -f "$f" ]] || return 0
  python - "$f" "$BENCHMARK" <<'TR_EOF' 2>/dev/null || true
import json
import sys

payload = json.load(open(sys.argv[1]))
rows = {}
if sys.argv[2] == "libero":
    for entry in payload.get("per_task") or []:
        successes = entry["metrics"].get("successes") or []
        if successes:
            key = f"{entry['task_group']}_{entry['task_id']}"
            rows[key] = (100 * sum(successes) / len(successes), len(successes))
else:
    for name, info in (payload.get("per_group") or {}).items():
        if info.get("pc_success") is not None:
            rows[name] = (info["pc_success"], info.get("n_episodes", "?"))
for name, (rate, n) in sorted(rows.items()):
    print(f"{name}\t{rate:.1f}%\t{n}")
TR_EOF
}

# Per-task success, which is what exposes the federated failure mode: a global model can hold a
# respectable average while having collapsed completely on one client's task.
per_task_rates() {
  local task rate n
  while IFS=$'\t' read -r task rate n; do
    [[ -n "$task" ]] && printf '      %-32s %6s  (n=%s)\n' "$task" "$rate" "$n"
  done <<< "$(task_rates "$1")"
}

# Split one local model's per-task rates into the task(s) it owns and the ones it never saw,
# recording them under different arms so the summary can show both without a second eval.
record_local_rates() {
  local seed="$1" evaldir="$2" own=",$3," task rate
  while IFS=$'\t' read -r task rate _; do
    [[ -n "$task" ]] || continue
    if [[ "$own" == *",$task,"* ]]; then
      record local "$seed" "$task" "$rate"
    else
      record local_offtask "$seed" "$task" "$rate"
    fi
  done <<< "$(task_rates "$evaldir")"
}

run_eval() {
  local name="$1" ckpt="$2" tasks="$3" target=()
  if [[ "$SKIP_EVAL" == "1" ]]; then
    echo "  [eval skipped]"
    return
  fi
  mapfile -t target < <(env_target "$tasks")
  echo "  --> evaluating $name on $tasks"
  "${EVAL[@]}" \
    --policy.path="$ckpt/checkpoints/last/pretrained_model" \
    "${target[@]}" \
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
  local arm="$1" seed="$2" evaldir="$3" task rate
  while IFS=$'\t' read -r task rate _; do
    [[ -n "$task" ]] && record "$arm" "$seed" "$task" "$rate"
  done <<< "$(task_rates "$evaldir")"
}

banner() {
  echo
  echo "========================================================================"
  echo "  $*"
  echo "========================================================================"
}

# ----------------------------------------------------------------------------------------
# Non-IID - every client holds a DIFFERENT task
# ----------------------------------------------------------------------------------------

banner "Non-IID - $NUM_CLIENTS clients x $TASKS_PER_CLIENT disjoint tasks each"

# partition.strategy=task deals disjoint task groups to the clients, so the task count has to
# divide evenly. Fail here with the real reason rather than inside the partitioner.
NUM_TASKS="$(awk -F, '{print NF}' <<< "$TASKS")"
EXPECTED_TASKS=$((NUM_CLIENTS * TASKS_PER_CLIENT))
if [[ "$NUM_TASKS" -ne "$EXPECTED_TASKS" ]]; then
  echo "!! TASKS has $NUM_TASKS entries, but NUM_CLIENTS ($NUM_CLIENTS) x TASKS_PER_CLIENT"
  echo "!! ($TASKS_PER_CLIENT) = $EXPECTED_TASKS are needed."
  echo "!! TASKS = $TASKS"
  exit 1
fi

if [[ "$BENCHMARK" == "libero" ]]; then
  # LIBERO asks on stdin for a dataset path the first time it is imported, which would hang (or
  # die on EOF in) every subprocess below. Declining once writes its default config.
  if [[ ! -f "${LIBERO_CONFIG_PATH:-$HOME/.libero}/config.yaml" ]]; then
    echo N | python -c "import libero.libero" > /dev/null
  fi
  # TASKS are ids within the suite; everything downstream uses "<suite>_<id>" keys, the same
  # names the eval results come back under.
  TASK_KEYS="$(tr ',' '\n' <<< "$TASKS" | sed "s/^/${SUITE}_/" | paste -sd, -)"
else
  TASK_KEYS="$TASKS"
fi

TASK_TABLE="$OUT_ROOT/tasks.tsv"
write_task_table > "$TASK_TABLE"

echo "tasks ($STUDY_TAG):"
EPS="$(episodes_for_tasks "$TASK_KEYS")"
echo "episodes: $(echo "$EPS" | tr -cd ',' | wc -c | awk '{print $1+1}') selected"

# Env flags covering every task, for the arms that roll out on all of them.
mapfile -t ALL_TARGET < <(env_target "$TASK_KEYS")

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

  if [[ "$RUN_FEDERATED" == "1" ]]; then
    rm -rf "$FED"
    echo "--> federated (budget $BUDGET, seed $SEED)"
    "${FL_TRAIN[@]}" "${POLICY_ARGS[@]}" \
      --dataset.repo_id="$DATASET" --dataset.episodes="$EPS" \
      --partition.strategy=task --partition.num_clients="$NUM_CLIENTS" \
      --partition.seed="$PARTITION_SEED" \
      --fl.rounds="$BUDGET" --fl.local_steps="$LOCAL_STEPS" \
      --fl.prox_mu="$PROX_MU" \
      --server.type="$SERVER_TYPE" --server.lr="$SERVER_LR" \
      ${FED_ENV_ARGS[@]:+"${FED_ENV_ARGS[@]}" "${ALL_TARGET[@]}"} \
      --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed="$SEED" \
      "${MP_ARGS[@]}" "${WANDB_ARGS[@]}" --job_name="${JOB_PREFIX}fed-b$BUDGET-s$SEED" \
      --output_dir="$FED" \
      2>&1 | tee "$FED.log" | grep -E "^INFO.*(round|Partitioned|client)|^    clients:" || true
    assert_trained "$FED" "federated-b$BUDGET-s$SEED"
    run_eval "federated-b$BUDGET-s$SEED" "$FED" "$TASK_KEYS"
    record "federated" "$SEED" __overall__ "$(success_rate "$FED/eval")"
    record_per_task "federated" "$SEED" "$FED/eval"
  fi

  if [[ "$RUN_CENTRALIZED" == "1" ]]; then
    rm -rf "$CEN"
    echo "--> centralized (budget $BUDGET, seed $SEED, $CENTRAL_STEPS steps)"
    "${TRAIN[@]}" "${POLICY_ARGS[@]}" \
      --dataset.repo_id="$DATASET" --dataset.episodes="$EPS" \
      --steps="$CENTRAL_STEPS" --save_freq="$CENTRAL_STEPS" \
      ${ENV_ARGS[@]:+"${ENV_ARGS[@]}" "${ALL_TARGET[@]}" --env_eval_freq="$((ENV_EVAL_EVERY * LOCAL_STEPS))"} \
      --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed="$SEED" \
      "${MP_ARGS[@]}" "${WANDB_ARGS[@]}" --job_name="${JOB_PREFIX}cen-b$BUDGET-s$SEED" \
      --output_dir="$CEN" \
      2>&1 | tee "$CEN.log" | grep -E "^INFO.*step" || true
    assert_trained "$CEN" "centralized-b$BUDGET-s$SEED"
    run_eval "centralized-b$BUDGET-s$SEED" "$CEN" "$TASK_KEYS"
    record "centralized" "$SEED" __overall__ "$(success_rate "$CEN/eval")"
    record_per_task "centralized" "$SEED" "$CEN/eval"
  fi

  if [[ "$RUN_LOCAL" == "1" ]]; then
    for ((c = 0; c < NUM_CLIENTS; c++)); do
      IFS=$'\t' read -r C_EPS C_TASKS < <(client_shard "$EPS" "$NUM_CLIENTS" "$c" "$PARTITION_SEED")
      mapfile -t C_TARGET < <(env_target "$C_TASKS")
      LOC="$OUT_ROOT/local-c$c-b$BUDGET-s$SEED"
      rm -rf "$LOC"
      echo "--> local: client $c alone on $C_TASKS (budget $BUDGET, seed $SEED, $LOCAL_ARM_STEPS steps)"
      "${TRAIN[@]}" "${POLICY_ARGS[@]}" \
        --dataset.repo_id="$DATASET" --dataset.episodes="$C_EPS" \
        --steps="$LOCAL_ARM_STEPS" --save_freq="$LOCAL_ARM_STEPS" \
        ${ENV_ARGS[@]:+"${ENV_ARGS[@]}" "${C_TARGET[@]}" --env_eval_freq="$((ENV_EVAL_EVERY * LOCAL_STEPS))"} \
        --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed="$SEED" \
        "${MP_ARGS[@]}" "${WANDB_ARGS[@]}" --job_name="${JOB_PREFIX}loc-c$c-b$BUDGET-s$SEED" \
        --output_dir="$LOC" \
        2>&1 | tee "$LOC.log" | grep -E "^INFO.*step" || true
      assert_trained "$LOC" "local-c$c-b$BUDGET-s$SEED"
      # Own tasks answer "what does this client achieve unaided?"; all tasks additionally
      # answer "what did specialising cost it elsewhere?".
      if [[ "$LOCAL_EVAL_ALL_TASKS" == "1" ]]; then
        run_eval "local-c$c-b$BUDGET-s$SEED" "$LOC" "$TASK_KEYS"
      else
        run_eval "local-c$c-b$BUDGET-s$SEED" "$LOC" "$C_TASKS"
      fi
      record_local_rates "$SEED" "$LOC/eval" "$C_TASKS"
    done
  fi
}

if [[ "$RUN_FEDERATED$RUN_CENTRALIZED$RUN_LOCAL" != *1* ]]; then
  echo "!! all three arms are disabled; nothing to run."
  echo "!! set at least one of RUN_FEDERATED / RUN_CENTRALIZED / RUN_LOCAL to 1."
  exit 1
fi

echo "seeds: $SEEDS"
echo "arms: federated=$RUN_FEDERATED centralized=$RUN_CENTRALIZED local=$RUN_LOCAL"
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
echo "Per-task breakdown (per budget and seed):"
for BUDGET in $BUDGETS; do
  for SEED in $SEEDS; do
    for arm in federated centralized; do
      if [[ -f "$OUT_ROOT/$arm-b$BUDGET-s$SEED/eval/eval_info.json" ]]; then
        echo "  $arm budget=$BUDGET seed=$SEED"
        per_task_rates "$OUT_ROOT/$arm-b$BUDGET-s$SEED/eval"
      fi
    done
  done
done
echo "Task legend: $TASK_TABLE"

cat <<'EOS'

How to read this:
  * Per-task success is the number that matters, not the mean. The characteristic federated
    failure here is a respectable average hiding one client's task that collapsed entirely,
    which is why the per-task table is sorted by gap: the worst-served task reads first.
  * A large gap on ONE task with small gaps elsewhere is the interesting shape - it means
    averaging sacrificed a specific client rather than degrading everything evenly.
  * fed-loc is the question every client actually cares about: did joining the federation beat
    training alone on my own data? Negative on a task means that client would have been better
    off keeping its data and skipping the federation entirely - the sharpest possible negative
    result, and the one worth reporting.
  * cen-fed is the cost of not pooling. Together the two bracket federation: local <= federated
    <= centralized is the outcome federation is supposed to deliver, and any inversion is a
    finding.
  * local-off is the cost of specialising. A local model is competitive on its own task and
    near zero elsewhere; federated should be close to local on each task while remaining
    competent on all of them. The distance between the local and local-off columns is what
    federation actually buys, and it is the argument the own-task numbers alone cannot make.
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



