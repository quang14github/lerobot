#!/usr/bin/env bash
#
# Evaluate every model archive in a folder, one after another and unattended, then gather all
# results into one file. Meant to run on the eval machine (e.g. a Jetson) inside tmux or nohup,
# while push_archives.sh fills the folder from the machine that holds the archives.
#
# For each <name>.tar.gz in MODELS_DIR (default outputs/models), in name order:
#   - skip it if EVAL_ROOT/<name>/summary.tsv already exists (evaluated before);
#   - evaluate it with eval_archive.sh into EVAL_ROOT/<name>/ - the flat, short-name layout;
#   - delete the archive once its summary.tsv exists, freeing the disk for the next one. A failed
#     eval keeps its archive (retried on the next run) and the queue moves on.
# Archives still being copied are hidden .<name>.partial files and are ignored until complete. The
# queue keeps waiting for more while MODELS_DIR/.push_in_progress exists (push_archives.sh holds
# it for the whole transfer) or until nothing new has arrived for IDLE_HOURS, then writes
# EVAL_ROOT/all_results.tsv with gather_evals.py.
#
# Env flags: archives made by noniid.sh carry their own (eval_args.txt); others get them from their
# name: lib<suite>_* -> libero_<suite> with LIBERO_TASK_IDS, mw_* -> Meta-World with MW_TASKS.
#
# Usage (from the repo root, with the lerobot environment on PATH):
#   tmux new -s evalq 'bash examples/federated/eval_queue.sh 2>&1 | tee -a outputs/eval/queue.log'
#   EPISODES=10 BATCH=5 EVAL_SEEDS="1000 2000 3000" bash examples/federated/eval_queue.sh
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
MODELS_DIR="${MODELS_DIR:-$REPO_ROOT/outputs/models}"
EVAL_ROOT="${EVAL_ROOT:-$REPO_ROOT/outputs/eval}"
EVAL_ARCHIVE="${EVAL_ARCHIVE:-$REPO_ROOT/examples/federated/eval_archive.sh}"
EPISODES="${EPISODES:-10}"
BATCH="${BATCH:-10}"
export EVAL_SEEDS="${EVAL_SEEDS:-1000 2000 3000}"
LIBERO_TASK_IDS="${LIBERO_TASK_IDS:-[0,1,2]}"
MW_TASKS="${MW_TASKS:-coffee-pull-v3,door-close-v3,drawer-close-v3}"
KEEP_ARCHIVES="${KEEP_ARCHIVES:-0}"
IDLE_HOURS="${IDLE_HOURS:-3}"
POLL_SECONDS="${POLL_SECONDS:-60}"

LOG_DIR="$EVAL_ROOT/_logs"
mkdir -p "$LOG_DIR"
cd "$REPO_ROOT"

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

DONE=()
FAILED=()
declare -A TRIED  # names that failed in this run, so the loop does not retry them forever

evaluate() {
  local archive="$1" name out log flags=() status=0 t0 suite overall
  name="$(basename "$archive" .tar.gz)"
  out="$EVAL_ROOT/$name"
  log="$LOG_DIR/$name.log"
  # A folder without summary.tsv is a run that died midway; set it aside rather than mix into it.
  if [[ -e "$out" ]]; then
    mv "$out" "${out}_incomplete_$(date +%Y%m%d-%H%M%S)"
  fi
  if ! tar -xzOf "$archive" eval_args.txt > /dev/null 2>&1; then
    case "$name" in
      lib*_*)
        suite="${name%%_*}"
        flags=(--env.type=libero --env.task="libero_${suite#lib}" --env.task_ids="$LIBERO_TASK_IDS")
        ;;
      mw_*) flags=(--env.type=metaworld --env.task="$MW_TASKS") ;;
      *)
        echo "[$(stamp)] !! $name: no eval_args.txt inside and no env known from its name; skipped"
        FAILED+=("$name")
        TRIED[$name]=1
        return
        ;;
    esac
  fi

  echo "[$(stamp)] START $name"
  t0=$(date +%s)
  bash "$EVAL_ARCHIVE" "$archive" "${flags[@]}" --eval.n_episodes="$EPISODES" --eval.batch_size="$BATCH" \
    --output_dir="$out" > "$log" 2>&1 || status=$?
  if [[ $status -eq 0 && -f "$out/summary.tsv" ]]; then
    overall="$(awk -F'\t' '$1=="__overall__" {print $2 "% +/- " $3}' "$out/summary.tsv")"
    echo "[$(stamp)] DONE  $name in $((($(date +%s) - t0) / 60)) min: overall $overall"
    DONE+=("$name")
    if [[ "$KEEP_ARCHIVES" != "1" ]]; then
      rm -f "$archive"
    fi
  else
    echo "[$(stamp)] !! FAILED $name (exit $status) after $((($(date +%s) - t0) / 60)) min; archive kept, log: $log"
    FAILED+=("$name")
    TRIED[$name]=1
  fi
}

echo "[$(stamp)] queue started: $MODELS_DIR -> $EVAL_ROOT ($EPISODES episodes x seeds $EVAL_SEEDS, batch $BATCH)"
last_progress=$(date +%s)
while true; do
  next=""
  for archive in "$MODELS_DIR"/*.tar.gz; do
    [[ -e "$archive" ]] || continue
    name="$(basename "$archive" .tar.gz)"
    [[ -z "${TRIED[$name]:-}" ]] || continue
    if [[ -f "$EVAL_ROOT/$name/summary.tsv" ]]; then
      echo "[$(stamp)] skip  $name (already evaluated)"
      if [[ "$KEEP_ARCHIVES" != "1" ]]; then
        rm -f "$archive"
      fi
      continue
    fi
    next="$archive"
    break
  done

  if [[ -n "$next" ]]; then
    evaluate "$next"
    last_progress=$(date +%s)
    continue
  fi
  # Nothing to do right now: finish, unless more archives are on their way.
  partials=("$MODELS_DIR"/.*.partial)
  if [[ ! -e "$MODELS_DIR/.push_in_progress" && ! -e "${partials[0]}" ]]; then
    break
  fi
  if (($(date +%s) - last_progress > IDLE_HOURS * 3600)); then
    echo "[$(stamp)] nothing new for $IDLE_HOURS h although a transfer looked unfinished; stopping"
    break
  fi
  sleep "$POLL_SECONDS"
done

echo "[$(stamp)] queue finished: ${#DONE[@]} evaluated${DONE[*]:+ (${DONE[*]})}, ${#FAILED[@]} failed${FAILED[*]:+ (${FAILED[*]})}"
python "$REPO_ROOT/examples/federated/gather_evals.py" "$EVAL_ROOT" || echo "!! could not gather the results"
