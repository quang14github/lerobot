#!/usr/bin/env bash
#
# Evaluate a model archive written by examples/federated/noniid.sh, without unpacking it by hand.
#
# lerobot-eval only reads a directory or a Hub repo, so the archive is extracted to a temporary
# directory once, evaluated, and the temporary copy deleted afterwards - even if the eval fails or
# is interrupted. The archive itself is never modified.
#
# The eval runs once per seed in EVAL_SEEDS (three by default) and reports the mean and spread
# across them, the protocol LIBERO results are reported with. The seeds must differ: LIBERO starts
# every episode from a fixed initial state, so repeats with one seed would replay identical
# rollouts. What varies between seeds is the policy's own sampling noise (and the env seeds).
#
# With no extra arguments each run reuses what the study used: the same suite and task ids, one
# rollout at a time, and the same number of episodes per task (from the archive's eval_args.txt).
# Anything you pass after the archive is forwarded to lerobot-eval and wins over those defaults.
#
# Usage:
#   bash examples/federated/eval_archive.sh outputs/models/<study>/<study>__centralized-b500-s1000.tar.gz
#   bash ... <archive> --eval.n_episodes=10                       # fewer episodes per task
#   bash ... <archive> --env.task_ids=[2]                         # one task only
#   bash ... <archive> --output_dir=/somewhere/else               # choose where results go
#   EVAL_SEEDS="1000" bash ... <archive>                          # a single run
#   EVAL_SEEDS="1000 2000 3000 4000 5000" bash ... <archive>      # five runs
#   MUJOCO_GL=osmesa bash ... <archive>                           # pick the renderer (WSL: glx by default)
#
# Results go to outputs/eval/<study>/<run>/, named after the archive's own identity rather than
# where it sits, so archives downloaded into any folder - or renamed - land in the right place:
#   1. the study.env inside the archive (STUDY and RUN), embedded by noniid.sh;
#   2. otherwise a "<study>__<run>.tar.gz" file name;
#   3. otherwise the folder holding it (outputs/models/<study>/<run>.tar.gz), or just
#      outputs/eval/<run>/ for an archive placed directly in outputs/models.
#   Inside:
#       eval_args.txt       what produced this eval: archive, seeds, flags
#       study.env           the training study's settings, when the archive carries them
#       seed1000/ ...       one lerobot-eval output per seed (eval_info.json, videos/)
#       summary.tsv         per task and overall: mean, std and every seed's success rate
# A re-evaluation of the same archive gets its own <run>_<date-time>/ folder instead of mixing
# into the first. Passing --output_dir uses that folder instead, with the same layout inside.
#
# Also reads archives made by hand with `tar -czf x.tar.gz <path>/pretrained_model`; those carry
# no eval_args.txt, so pass the --env.* flags yourself.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
EVAL_SEEDS="${EVAL_SEEDS:-1000 2000 3000}"

# robosuite (LIBERO's simulator) renders with EGL unless MUJOCO_GL is glx or osmesa, and WSL2's EGL
# has no headless device support, so every rollout would fail with "Cannot initialize a EGL
# device display". Under WSL, default to glx (WSLg provides the display); elsewhere, and whenever
# MUJOCO_GL is already set, leave it alone.
#
# glx alone renders on the CPU: Mesa picks its llvmpipe software rasterizer. Its d3d12 driver
# renders on the GPU instead, through WSL's DirectX bridge, and the adapter name makes it pick the
# NVIDIA GPU over an integrated one. Measured on a LIBERO eval: 36.6 s/episode -> 21.7 s/episode,
# with frames matching the CPU render. Any value you set yourself wins
# (GALLIUM_DRIVER=llvmpipe restores CPU rendering).
if grep -qi microsoft /proc/version 2>/dev/null; then
  export MUJOCO_GL="${MUJOCO_GL:-glx}"
  if [[ "$MUJOCO_GL" == "glx" ]]; then
    export GALLIUM_DRIVER="${GALLIUM_DRIVER:-d3d12}"
    export MESA_D3D12_DEFAULT_ADAPTER_NAME="${MESA_D3D12_DEFAULT_ADAPTER_NAME:-NVIDIA}"
  fi
  echo "--> WSL detected: MUJOCO_GL=$MUJOCO_GL"
else
  # Headless machines (HPC nodes, a Jetson): EGL renders on the GPU without a display. robosuite
  # forces it anyway; MetaWorld's renderer would otherwise try to open a glfw window and fail.
  export MUJOCO_GL="${MUJOCO_GL:-egl}"
fi

# Each seed names its own run folder, so the seeds must be distinct whole numbers.
read -ra SEED_LIST <<< "$EVAL_SEEDS"
if ((${#SEED_LIST[@]} == 0)) || [[ ! " ${SEED_LIST[*]} " =~ ^(\ [0-9]+)+\ $ ]] \
  || (($(printf '%s\n' "${SEED_LIST[@]}" | sort -u | wc -l) != ${#SEED_LIST[@]})); then
  echo "!! EVAL_SEEDS must be distinct whole numbers, e.g. \"1000 2000 3000\" (got \"$EVAL_SEEDS\")" >&2
  exit 1
fi

if [[ $# -lt 1 || ! -f "$1" ]]; then
  echo "usage: $0 <model archive .tar.gz> [lerobot-eval args...]" >&2
  exit 1
fi
ARCHIVE="$1"
shift

# --output_dir picks the folder that holds the per-seed runs, so it is taken out of the forwarded
# arguments; --seed is refused because each run's seed comes from EVAL_SEEDS.
ARGS=()
OUTPUT_DIR=""
while (($#)); do
  case "$1" in
    --output_dir=*) OUTPUT_DIR="${1#--output_dir=}" ;;
    --output_dir)
      [[ $# -ge 2 ]] || { echo "!! --output_dir needs a value" >&2; exit 1; }
      OUTPUT_DIR="$2"
      shift
      ;;
    --seed | --seed=*)
      echo "!! set the seeds with EVAL_SEEDS=\"...\" instead of --seed (now: $EVAL_SEEDS)" >&2
      exit 1
      ;;
    *) ARGS+=("$1") ;;
  esac
  shift
done

# The policy must run on the GPU. lerobot's policy config only logs a warning and switches to the
# CPU when CUDA is unusable, which quietly turns hours of eval into days. So CUDA is checked first
# with a real computation - torch.cuda.is_available() alone can be True on a build that cannot run
# here, such as a datacenter wheel on a Jetson - and the eval refuses to start without it. Set
# ALLOW_CPU=1, or pass --policy.device=cpu yourself, to run the policy on the CPU deliberately.
ALLOW_CPU="${ALLOW_CPU:-0}"
if [[ " ${ARGS[*]} " == *" --policy.device=cpu "* ]]; then
  echo "--> policy on the CPU (--policy.device=cpu)"
else
  GPU_NAME="$(timeout 300 python -c '
import torch
x = torch.randn(512, 512, device="cuda")
assert torch.isfinite(x @ x).all(), "non-finite result"
print(torch.cuda.get_device_name(0))
' 2>/dev/null | tail -1 || true)"
  if [[ -n "$GPU_NAME" ]]; then
    echo "--> policy on the GPU: $GPU_NAME"
  elif [[ "$ALLOW_CPU" == "1" ]]; then
    echo "--> no usable CUDA GPU; policy on the CPU (ALLOW_CPU=1)"
    ARGS+=(--policy.device=cpu)
  else
    echo "!! no usable CUDA GPU: a test computation on cuda failed, so lerobot would quietly run the" >&2
    echo "!!   policy on the CPU. Check that torch is a CUDA build for this machine:" >&2
    echo "!!   python -c \"import torch; print(torch.__version__, torch.cuda.is_available())\"" >&2
    echo "!!   Set ALLOW_CPU=1 to run on the CPU anyway." >&2
    exit 1
  fi
fi

WORKDIR="$(mktemp -d)"
trap 'rm -rf "$WORKDIR"' EXIT

echo "--> extracting $ARCHIVE"
tar -xzf "$ARCHIVE" -C "$WORKDIR"

# The model directory is the one holding config.json. Archives from noniid.sh keep it at the top
# level; hand-made ones may bury it under the absolute path they were created from.
MODEL_DIR="$(dirname "$(find "$WORKDIR" -name config.json -path '*pretrained_model/*' -print -quit)")"
if [[ ! -f "$MODEL_DIR/model.safetensors" && ! -f "$MODEL_DIR/adapter_model.safetensors" ]]; then
  echo "!! no pretrained_model/ with weights found inside $ARCHIVE" >&2
  exit 1
fi

DEFAULTS=()
if [[ -f "$WORKDIR/eval_args.txt" ]]; then
  mapfile -t DEFAULTS < "$WORKDIR/eval_args.txt"
elif [[ " ${ARGS[*]} " != *" --env.type="* ]]; then
  echo "!! $ARCHIVE has no eval_args.txt; pass the env yourself, e.g." >&2
  echo "!!   --env.type=libero --env.task=libero_spatial --env.task_ids=[0,1,2]" >&2
  exit 1
fi
ENV_TYPE="$(printf '%s\n' "${DEFAULTS[@]}" "${ARGS[@]}" | sed -n 's/^--env\.type=//p' | tail -1)"

# Rendering: report what actually draws the frames, on every machine, by creating the same GL
# context the simulator will. For LIBERO that is robosuite's own context class, which insists on
# an EGL *device* where MuJoCo's EGL quietly falls back to software - probing with MuJoCo would
# report "fine" for an eval that then crashes. A software renderer only warns: the frames are
# still correct, just slower to produce.
if [[ "$ENV_TYPE" == "libero" ]]; then
  PROBE='
from robosuite.utils.binding_utils import GLContext
ctx = GLContext(max_width=16, max_height=16, device_id=-1)
ctx.make_current()'
else
  PROBE='
import mujoco
ctx = mujoco.GLContext(16, 16)
ctx.make_current()'
fi
PROBE_ENV=()
if [[ "$MUJOCO_GL" != "glx" && -z "${PYOPENGL_PLATFORM:-}" ]]; then
  PROBE_ENV=(PYOPENGL_PLATFORM="$MUJOCO_GL")
fi
RENDERER="$(env "${PROBE_ENV[@]}" timeout 60 python -c "$PROBE
from OpenGL.GL import GL_RENDERER, glGetString
print('RENDERER=' + glGetString(GL_RENDERER).decode())" 2>/dev/null | sed -n 's/^RENDERER=//p' | tail -1 || true)"
if [[ -z "$RENDERER" ]]; then
  echo "!! the simulator could not create its MUJOCO_GL=$MUJOCO_GL rendering context, so the eval will"
  echo "!!   most likely fail. Try MUJOCO_GL=osmesa (CPU) or MUJOCO_GL=glx (needs a display)."
elif [[ "$RENDERER" =~ llvmpipe|softpipe|swrast ]]; then
  echo "!! rendering on the CPU ($RENDERER, MUJOCO_GL=$MUJOCO_GL): correct but slow"
else
  echo "--> rendering on the GPU: $RENDERER (MUJOCO_GL=$MUJOCO_GL)"
fi

if [[ -n "$OUTPUT_DIR" ]]; then
  mkdir -p "$OUTPUT_DIR"
else
  # Name the results after the archive's identity (see the header for the order of precedence).
  # The values are read with sed rather than by sourcing study.env, so nothing in an archive runs.
  STUDY=""
  RUN=""
  if [[ -f "$WORKDIR/study.env" ]]; then
    STUDY="$(sed -n "s/^STUDY='\(.*\)'\$/\1/p" "$WORKDIR/study.env")"
    RUN="$(sed -n "s/^RUN='\(.*\)'\$/\1/p" "$WORKDIR/study.env")"
  fi
  if [[ -z "$STUDY" || -z "$RUN" ]]; then
    NAME="$(basename "$ARCHIVE" .tar.gz)"
    ARCHIVE_DIR="$(dirname "$(realpath "$ARCHIVE")")"
    if [[ "$NAME" == *__* ]]; then
      STUDY="${NAME%%__*}"
      RUN="${NAME#*__}"
    elif [[ "$ARCHIVE_DIR" == "$(realpath -m "$REPO_ROOT/outputs/models")" ]]; then
      STUDY=""
      RUN="$NAME"
    else
      STUDY="$(basename "$ARCHIVE_DIR")"
      RUN="$NAME"
    fi
  fi
  OUTPUT_DIR="$REPO_ROOT/outputs/eval/${STUDY:+$STUDY/}$RUN"
  mkdir -p "$(dirname "$OUTPUT_DIR")"
  # Claim the folder with a plain mkdir, which fails if it already exists, so two eval - even
  # ones started in the same second - can never share a folder.
  if ! mkdir "$OUTPUT_DIR" 2>/dev/null; then
    BASE="${OUTPUT_DIR}_$(date +%Y%m%d-%H%M%S)"
    OUTPUT_DIR="$BASE"
    N=1
    until mkdir "$OUTPUT_DIR" 2>/dev/null; do
      N=$((N + 1))
      if ((N > 100)); then
        echo "!! cannot create an eval folder under $(dirname "$OUTPUT_DIR")" >&2
        exit 1
      fi
      OUTPUT_DIR="$BASE-$N"
    done
  fi
fi

{
  echo "# lerobot-eval of $(realpath "$ARCHIVE"), started $(date '+%Y-%m-%d %H:%M:%S')"
  echo "# seeds: $EVAL_SEEDS"
  printf '%s\n' "${DEFAULTS[@]}" "${ARGS[@]}"
} > "$OUTPUT_DIR/eval_args.txt"
if [[ -f "$WORKDIR/study.env" ]]; then
  cp "$WORKDIR/study.env" "$OUTPUT_DIR/study.env"
fi

echo "--> ${#SEED_LIST[@]} run(s), seeds $EVAL_SEEDS, results in $OUTPUT_DIR"
echo "--> flags: ${DEFAULTS[*]} ${ARGS[*]}"

# A failed seed is reported and skipped rather than aborting the others: hours of finished runs
# are worth keeping, and the summary says how many seeds it is built from.
FAILED=()
for i in "${!SEED_LIST[@]}"; do
  SEED="${SEED_LIST[$i]}"
  echo
  echo "==> run $((i + 1))/${#SEED_LIST[@]}: seed $SEED"
  # lerobot-eval keeps the last value of a repeated flag, so the user's arguments override the
  # archive's defaults, and the per-run seed and folder come last of all.
  if ! python -m lerobot.scripts.lerobot_eval --policy.path="$MODEL_DIR" \
    "${DEFAULTS[@]}" "${ARGS[@]}" --seed="$SEED" --output_dir="$OUTPUT_DIR/seed$SEED"; then
    echo "!! seed $SEED failed; continuing with the remaining seeds"
    FAILED+=("$SEED")
  fi
done

# Per-task success comes from per_task, keyed "<suite>_<id>" for LIBERO (whose per_group would
# lump the whole suite together) and by task name otherwise - the same keys noniid.sh uses.
echo
python - "$OUTPUT_DIR" "$ENV_TYPE" <<'AGG_EOF'
import json
import statistics
import sys
from collections import defaultdict
from pathlib import Path

out_dir, env_type = Path(sys.argv[1]), sys.argv[2]
runs = sorted(out_dir.glob("seed*/eval_info.json"), key=lambda p: int(p.parent.name[4:]))
if not runs:
    sys.exit("!! no seed produced an eval_info.json; nothing to summarize")

rates = defaultdict(dict)  # task -> {seed: pc_success}
episodes = {}
for path in runs:
    seed = int(path.parent.name[4:])
    info = json.loads(path.read_text())
    for entry in info.get("per_task") or []:
        successes = entry["metrics"].get("successes") or []
        if not successes:
            continue
        key = f"{entry['task_group']}_{entry['task_id']}" if env_type == "libero" else entry["task_group"]
        rates[key][seed] = 100 * sum(successes) / len(successes)
        episodes[key] = len(successes)
    rates["__overall__"][seed] = info["overall"]["pc_success"]
    episodes["__overall__"] = info["overall"]["n_episodes"]

seeds = sorted({s for per_seed in rates.values() for s in per_seed})


def spread(values):
    return statistics.stdev(values) if len(values) > 1 else 0.0


lines = ["task\tmean\tstd\tn_seeds\tepisodes_per_seed\t" + "\t".join(f"seed{s}" for s in seeds)]
print(f"{'task':<24}{'mean':>8}{'std':>8}  per seed ({', '.join(map(str, seeds))})")
for task in sorted(rates, key=lambda t: (t == "__overall__", t)):
    values = [rates[task][s] for s in seeds if s in rates[task]]
    mean, std = statistics.mean(values), spread(values)
    per_seed = "\t".join(f"{rates[task][s]:.1f}" if s in rates[task] else "n/a" for s in seeds)
    lines.append(f"{task}\t{mean:.1f}\t{std:.1f}\t{len(values)}\t{episodes[task]}\t{per_seed}")
    shown = ", ".join(f"{rates[task][s]:.1f}" if s in rates[task] else "n/a" for s in seeds)
    print(f"{task[:23]:<24}{mean:>7.1f}%{std:>7.1f}  ({shown})")

(out_dir / "summary.tsv").write_text("\n".join(lines) + "\n")
print(f"\nsummary: {out_dir / 'summary.tsv'}")
AGG_EOF

if ((${#FAILED[@]})); then
  echo "!! seed(s) ${FAILED[*]} failed; the summary covers only the others. Logs are above."
  exit 1
fi
