#!/usr/bin/env bash
#
# Two-stage study: pretrain centrally, then federate on unseen tasks.
#
#   Stage 1  One centralized SmolVLA run on PRETRAIN_TASKS. Produces a checkpoint whose feature
#            shapes match the dataset and whose action expert is no longer random - the shared
#            base a fleet would be handed before any federation begins. Run once, reused by
#            every arm and seed, so the comparison is about the federated stage alone.
#
#   Stage 2  Federated training on FL_TASKS, which stage 1 never saw. One task per client, so
#            each client must contribute a skill the others cannot. Two arms:
#
#              expert       the action expert trains in full; the VLM stays frozen.
#                           This is SmolVLA's default (train_expert_only=true).
#              expert_lora  the action expert still trains in full, and the VLM additionally
#                           gets LoRA adapters on its attention projections, so the perceptual
#                           backbone can adapt too.
#
# The two arms exchange almost identical payloads - 381 MB vs 385 MB per client per round -
# because the LoRA adapters are only ~2.5M parameters against the expert's ~98M. So this
# compares them at matched communication cost, and any difference is attributable to the VLM
# being adaptable rather than to one arm having been given a larger budget.
#
# Usage:
#   bash examples/federated/pretrain_then_federate.sh
#   ROUNDS=10 LOCAL_STEPS=5 PRETRAIN_STEPS=20 SKIP_EVAL=1 bash ...   # plumbing check
#   ARMS="expert_lora" bash ...                                      # one arm
#   PRETRAIN_CKPT=/path/to/pretrained_model bash ...                 # reuse a stage-1 run
#   EVAL_FORGETTING=1 bash ...                                       # also score the old tasks
#   LORA_R=64 bash ...                                               # more adapter capacity
#   LORA_ALPHA=32 bash ...                                           # stronger adaptation (2x r)
#
set -euo pipefail

# ---------------------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------------------
DATASET="${DATASET:-lerobot/metaworld_mt50}"
OUT_ROOT="${OUT_ROOT:-./outputs/pretrain-federate-$(date +%Y%m%d_%H%M%S)}"

PRETRAIN_TASKS="${PRETRAIN_TASKS:-button-press-v3,door-close-v3,drawer-close-v3}"
FL_TASKS="${FL_TASKS:-coffee-button-v3,dial-turn-v3,door-open-v3}"

NUM_CLIENTS="${NUM_CLIENTS:-3}"
ROUNDS="${ROUNDS:-100}"
LOCAL_STEPS="${LOCAL_STEPS:-5}"
PRETRAIN_STEPS="${PRETRAIN_STEPS:-5000}"

BATCH_SIZE="${BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
MIXED_PRECISION="${MIXED_PRECISION:-bf16}"
SEEDS="${SEEDS:-1000 2000 3000}"
PARTITION_SEED="${PARTITION_SEED:-42}"

LORA_R="${LORA_R:-32}"
# LoRA's contribution is scaled by lora_alpha/r. PEFT's fallback is alpha=8, so leaving this
# unset would run r=16 adapters at half strength - and a rank sweep would silently weaken the
# updates as rank rose (r=64 would scale by 8/64). Pinning alpha to r keeps the scaling at 1.0
# and makes rank the only thing a rank sweep changes.
LORA_ALPHA="${LORA_ALPHA:-$LORA_R}"
# Only the VLM: `\.vlm\.` appears in the backbone's module paths and not in the expert's
# (`...vlm_with_expert.lm_expert...`), so the adapters cannot leak onto the expert.
LORA_TARGETS="${LORA_TARGETS:-.*\.vlm\..*(q_proj|k_proj|v_proj|o_proj)}"
# PEFT freezes every base parameter when it wraps a model. These modules are exempted and stay
# fully trainable, which is what keeps "full action expert" true in the expert_lora arm.
LORA_FULL_MODULES='["lm_expert","state_proj","action_in_proj","action_out_proj"]'

ARMS="${ARMS:-expert expert_lora}"
EVAL_EPISODES="${EVAL_EPISODES:-15}"
SKIP_EVAL="${SKIP_EVAL:-0}"
# Also evaluate on PRETRAIN_TASKS, to see what federating on new tasks did to the old ones.
EVAL_FORGETTING="${EVAL_FORGETTING:-1}"
PRETRAIN_CKPT="${PRETRAIN_CKPT:-}"

export MUJOCO_GL="${MUJOCO_GL:-egl}"

MP_ARGS=()
if [[ "$MIXED_PRECISION" != "no" ]]; then
  MP_ARGS=(--accelerator.mixed_precision="$MIXED_PRECISION")
fi

FL_TRAIN=(python -m lerobot.scripts.lerobot_fl_train)
TRAIN=(python -m lerobot.scripts.lerobot_train)
EVAL=(python -m lerobot.scripts.lerobot_eval)

mkdir -p "$OUT_ROOT"
SUMMARY="$OUT_ROOT/summary.tsv"
: > "$SUMMARY"

banner() {
  echo
  echo "========================================================================"
  echo "  $*"
  echo "========================================================================"
}

# Resolve Meta-World slugs to a draccus episode list. Tasks are looked up by their
# natural-language description because a task's block position does not follow its task_id.
episodes_for_tasks() {
  python - "$1" "$DATASET" <<'PYEOF'
import json
import sys
from pathlib import Path

import lerobot.envs
from lerobot.datasets.dataset_metadata import LeRobotDatasetMetadata

slugs = sys.argv[1].split(",")
cfg = json.loads((Path(lerobot.envs.__file__).parent / "metaworld_config.json").read_text())
desc = cfg["TASK_DESCRIPTIONS"]

meta = LeRobotDatasetMetadata(sys.argv[2])
tasks = [t[0] if t else "" for t in meta.episodes["tasks"]]

out = []
for slug in slugs:
    if slug not in desc:
        sys.exit(f"unknown Meta-World task: {slug!r}")
    hits = [i for i, t in enumerate(tasks) if t == desc[slug]]
    # push-v3 and push-back-v3 share one description, so a lookup returns both blocks.
    if len(hits) != 50:
        sys.exit(f"{slug!r} matched {len(hits)} episodes, expected 50 (ambiguous description?)")
    out.extend(hits)
print("[" + ",".join(str(i) for i in sorted(out)) + "]")
PYEOF
}

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

per_task_rates() {
  local f="$1/eval_info.json"
  [[ -f "$f" ]] || return
  python - "$f" <<'PERTASK_EOF' 2>/dev/null || true
import json, sys

payload = json.load(open(sys.argv[1]))
for name, info in sorted((payload.get("per_group") or {}).items()):
    rate = info.get("pc_success")
    if rate is not None:
        print(f"      {name:34} {rate:5.1f}%  (n={info.get('n_episodes', '?')})")
PERTASK_EOF
}

run_eval() {
  local label="$1" ckpt="$2" tasks="$3" subdir="$4"
  if [[ "$SKIP_EVAL" == "1" ]]; then echo "  [eval skipped: $label]"; return; fi
  echo "  --> evaluating $label on $tasks"
  "${EVAL[@]}" \
    --policy.path="$ckpt/checkpoints/last/pretrained_model" \
    --env.type=metaworld --env.task="$tasks" \
    --eval.batch_size=1 --eval.n_episodes="$EVAL_EPISODES" \
    --output_dir="$ckpt/$subdir" \
    > "$ckpt/$subdir.log" 2>&1 || echo "  !! eval failed, see $ckpt/$subdir.log"
}

assert_trained() {
  local dir="$1" label="$2"
  local ckpt="$dir/checkpoints/last/pretrained_model"
  # A PEFT run writes adapter_model.safetensors instead of a full model.safetensors, so accept
  # either rather than declaring a successful LoRA run a failure.
  if [[ ! -f "$ckpt/model.safetensors" && ! -f "$ckpt/adapter_model.safetensors" ]]; then
    echo
    echo "!! $label produced no checkpoint - training failed."
    echo "!! Full log: $dir.log"
    tail -20 "$dir.log" 2>/dev/null || true
    exit 1
  fi
}

record() { printf '%s\t%s\t%s\t%s\n' "$1" "$2" "$3" "$4" >> "$SUMMARY"; }

# ---------------------------------------------------------------------------------------
# Stage 1 - centralized pretraining
# ---------------------------------------------------------------------------------------
banner "Stage 1: pretrain on $PRETRAIN_TASKS"
PRETRAIN_EPS="$(episodes_for_tasks "$PRETRAIN_TASKS")"

if [[ -n "$PRETRAIN_CKPT" ]]; then
  echo "reusing supplied checkpoint: $PRETRAIN_CKPT"
  BASE_CKPT="$(readlink -f "$PRETRAIN_CKPT")"
else
  PRE="$OUT_ROOT/pretrain"
  rm -rf "$PRE"
  echo "--> centralized, $PRETRAIN_STEPS steps"
  "${TRAIN[@]}" \
    --policy.type=smolvla --policy.load_vlm_weights=true --policy.push_to_hub=false \
    --dataset.repo_id="$DATASET" --dataset.episodes="$PRETRAIN_EPS" \
    --steps="$PRETRAIN_STEPS" --save_freq="$PRETRAIN_STEPS" \
    --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed="$PARTITION_SEED" \
    "${MP_ARGS[@]}" --output_dir="$PRE" \
    2>&1 | tee "$PRE.log" | grep -E "^INFO.*step" || true
  assert_trained "$PRE" "pretrain"
  BASE_CKPT="$(readlink -f "$PRE/checkpoints/last/pretrained_model")"
  run_eval "pretrain" "$PRE" "$PRETRAIN_TASKS" "eval"
  record pretrain base "$PRETRAIN_TASKS" "$(success_rate "$PRE/eval")"
fi
echo "base checkpoint: $BASE_CKPT"

# ---------------------------------------------------------------------------------------
# Stage 2 - federated on unseen tasks
# ---------------------------------------------------------------------------------------
banner "Stage 2: federate on $FL_TASKS (unseen in stage 1)"

NUM_FL_TASKS="$(awk -F, '{print NF}' <<< "$FL_TASKS")"
if [[ "$NUM_FL_TASKS" -ne "$NUM_CLIENTS" ]]; then
  echo "!! FL_TASKS has $NUM_FL_TASKS entries but NUM_CLIENTS is $NUM_CLIENTS;"
  echo "!! partition.strategy=task gives each client a disjoint task, so they must match."
  exit 1
fi
for t in ${PRETRAIN_TASKS//,/ }; do
  if [[ ",$FL_TASKS," == *",$t,"* ]]; then
    echo "!! '$t' appears in both PRETRAIN_TASKS and FL_TASKS; stage 2 must be unseen."
    exit 1
  fi
done

FL_EPS="$(episodes_for_tasks "$FL_TASKS")"
echo "episodes: $(tr -cd ',' <<< "$FL_EPS" | wc -c | awk '{print $1+1}') across $NUM_CLIENTS clients"

for SEED in $SEEDS; do
  for ARM in $ARMS; do
    banner "seed $SEED - arm $ARM"
    RUN="$OUT_ROOT/$ARM-s$SEED"
    rm -rf "$RUN"

    ARM_ARGS=()
    case "$ARM" in
      expert)
        # SmolVLA's default: the VLM is frozen wholesale, only the expert moves.
        ARM_ARGS=(--policy.train_expert_only=true)
        ;;
      expert_lora)
        # train_expert_only=false is required first, or the VLM stays frozen and the adapters
        # sit on parameters that never receive gradients.
        ARM_ARGS=(
          --policy.train_expert_only=false
          --peft.method_type=lora
          --peft.r="$LORA_R"
          --peft.lora_alpha="$LORA_ALPHA"
          --peft.target_modules="$LORA_TARGETS"
          --peft.full_training_modules="$LORA_FULL_MODULES"
        )
        ;;
      *) echo "!! unknown arm '$ARM' (expected: expert, expert_lora)"; exit 1 ;;
    esac

    "${FL_TRAIN[@]}" \
      --policy.path="$BASE_CKPT" --policy.push_to_hub=false \
      "${ARM_ARGS[@]}" \
      --dataset.repo_id="$DATASET" --dataset.episodes="$FL_EPS" \
      --partition.strategy=task --partition.num_clients="$NUM_CLIENTS" \
      --partition.seed="$PARTITION_SEED" \
      --fl.rounds="$ROUNDS" --fl.local_steps="$LOCAL_STEPS" \
      --batch_size="$BATCH_SIZE" --num_workers="$NUM_WORKERS" --seed="$SEED" \
      "${MP_ARGS[@]}" --output_dir="$RUN" \
      2>&1 | tee "$RUN.log" | grep -E "^INFO.*(round|Partitioned|Exchanged|upload)|^    clients:" || true
    assert_trained "$RUN" "$ARM-s$SEED"

    run_eval "$ARM-s$SEED" "$RUN" "$FL_TASKS" "eval"
    record "$ARM" "$SEED" new "$(success_rate "$RUN/eval")"
    if [[ "$EVAL_FORGETTING" == "1" ]]; then
      run_eval "$ARM-s$SEED (pretrain tasks)" "$RUN" "$PRETRAIN_TASKS" "eval_pretrain"
      record "$ARM" "$SEED" old "$(success_rate "$RUN/eval_pretrain")"
    fi
  done
done

# ---------------------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------------------
banner "Results (pc_success)"
python - "$SUMMARY" <<'AGG_EOF'
import statistics
import sys
from collections import defaultdict

rows = defaultdict(dict)
base = None
for line in open(sys.argv[1]):
    arm, seed, scope, rate = line.rstrip("\n").split("\t")
    if arm == "pretrain":
        base = rate
        continue
    rows[(scope, arm)][seed] = rate

if base:
    print(f"stage-1 base on its own pretrain tasks: {base}\n")

seeds = sorted({s for per_seed in rows.values() for s in per_seed})
labels = {"new": "FL tasks (unseen in stage 1)", "old": "pretrain tasks (forgetting)"}
for scope in ("new", "old"):
    if not any(sc == scope for sc, _ in rows):
        continue
    print(labels[scope])
    print(f"{'arm':<14}" + "".join(f"{('seed ' + s):>12}" for s in seeds) + f"{'mean':>10}{'std':>9}")
    for arm in ("expert", "expert_lora"):
        if (scope, arm) not in rows:
            continue
        cells, values = [], []
        for s in seeds:
            raw = rows[(scope, arm)].get(s, "n/a")
            cells.append(f"{raw:>12}")
            if raw.endswith("%"):
                values.append(float(raw[:-1]))
        mean = f"{statistics.mean(values):>9.1f}%" if values else f"{'n/a':>10}"
        std = f"{statistics.stdev(values):>8.1f}%" if len(values) > 1 else f"{'-':>9}"
        print(f"{arm:<14}" + "".join(cells) + mean + std)
    print()
AGG_EOF

echo "Per-task breakdown:"
for SEED in $SEEDS; do
  for ARM in $ARMS; do
    if [[ -f "$OUT_ROOT/$ARM-s$SEED/eval/eval_info.json" ]]; then
      echo "  $ARM seed=$SEED"
      per_task_rates "$OUT_ROOT/$ARM-s$SEED/eval"
    fi
  done
done

cat <<'EOS'

How to read this:
  * The two arms exchange near-identical payloads (381 vs 385 MB per client per round), so a
    win for expert_lora is a win for VLM adaptability, not for a bigger budget.
  * expert_lora ahead means the frozen backbone was the bottleneck: the new tasks needed
    perceptual adaptation the action expert alone could not supply.
  * expert ahead, or a tie, means the pretrained backbone already generalized to the new tasks
    and the adapters only added optimization noise - the cheaper arm is then the right default.
  * Read both against the seed-to-seed std before believing either.
  * With EVAL_FORGETTING=1 the second table shows what federating on new tasks cost the old
    ones. A LoRA arm that wins on new tasks while losing more of the old ones is trading, not
    improving - and that trade is the more interesting result.
EOS
echo
echo "Raw logs and checkpoints: $OUT_ROOT"
