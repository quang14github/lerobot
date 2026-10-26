#!/usr/bin/env bash
#
# Copy model archives from this machine to an eval machine, for eval_queue.sh running there.
#
#   bash examples/federated/push_archives.sh user@host [archive ...]
#   DRY_RUN=1 bash examples/federated/push_archives.sh user@host     # only list what would be sent
#
# With no archives listed, it sends every outputs/models/*.tar.gz that neither this machine nor
# the remote has evaluated (no outputs/eval/<name>/summary.tsv) and that the remote does not
# already hold.
#
# Safe to run while the remote queue is evaluating:
#   - each file is uploaded as a hidden .<name>.partial and renamed only when complete, so the
#     queue never picks up a half-copied archive;
#   - before each upload it waits until the remote disk has room for the archive plus
#     EVAL_TMP_GB (the eval's temporary extraction) plus MIN_FREE_GB, so a small disk is refilled
#     as the queue deletes evaluated archives instead of running out;
#   - REMOTE_MODELS/.push_in_progress tells the queue that more archives are coming; it is removed
#     when the transfer ends, also on failure or Ctrl-C. This machine must stay on until then.
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "usage: $0 user@host [archive ...]" >&2
  exit 1
fi
HOST="$1"
shift
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
REMOTE_REPO="${REMOTE_REPO:-dev/lerobot}"  # relative to the remote home
REMOTE_MODELS="$REMOTE_REPO/outputs/models"
REMOTE_EVAL="$REMOTE_REPO/outputs/eval"
EVAL_TMP_GB="${EVAL_TMP_GB:-1}"
MIN_FREE_GB="${MIN_FREE_GB:-1}"
SSH=(ssh -o BatchMode=yes -o ServerAliveInterval=30 "$HOST")

stamp() { date '+%Y-%m-%d %H:%M:%S'; }

ARCHIVES=("$@")
if ((${#ARCHIVES[@]} == 0)); then
  # Names the remote has evaluated or already holds, fetched in one round trip. find rather than
  # globs: the remote login shell may be zsh, which errors on a glob that matches nothing.
  mapfile -t REMOTE_KNOWN < <("${SSH[@]}" "find ~/$REMOTE_EVAL -mindepth 2 -maxdepth 2 -name summary.tsv 2>/dev/null | awk -F/ '{print \$(NF-1)}'; find ~/$REMOTE_MODELS -maxdepth 1 -name '*.tar.gz' 2>/dev/null | sed -e 's|.*/||' -e 's/\\.tar\\.gz\$//'")
  for archive in "$REPO_ROOT"/outputs/models/*.tar.gz; do
    name="$(basename "$archive" .tar.gz)"
    if [[ -f "$REPO_ROOT/outputs/eval/$name/summary.tsv" ]] || printf '%s\n' "${REMOTE_KNOWN[@]}" | grep -qxF "$name"; then
      continue
    fi
    ARCHIVES+=("$archive")
  done
fi
if ((${#ARCHIVES[@]} == 0)); then
  echo "nothing to send: every archive is evaluated or already on $HOST"
  exit 0
fi
if [[ "${DRY_RUN:-0}" == "1" ]]; then
  echo "would send ${#ARCHIVES[@]} archive(s) to $HOST:$REMOTE_MODELS:"
  printf '  %s\n' "${ARCHIVES[@]##*/}"
  exit 0
fi

CURRENT=""
cleanup() {
  "${SSH[@]}" "rm -f ~/$REMOTE_MODELS/.push_in_progress${CURRENT:+ ~/$REMOTE_MODELS/.$CURRENT.partial}" || true
}
trap cleanup EXIT
"${SSH[@]}" "mkdir -p ~/$REMOTE_MODELS && touch ~/$REMOTE_MODELS/.push_in_progress"

echo "[$(stamp)] sending ${#ARCHIVES[@]} archive(s) to $HOST:$REMOTE_MODELS"
i=0
for archive in "${ARCHIVES[@]}"; do
  i=$((i + 1))
  CURRENT="$(basename "$archive")"
  need_kb=$(($(stat -c %s "$archive") / 1024 + (EVAL_TMP_GB + MIN_FREE_GB) * 1024 * 1024))
  while true; do
    free_kb="$("${SSH[@]}" "df -Pk ~/$REMOTE_MODELS | awk 'NR==2 {print \$4}'")"
    ((free_kb >= need_kb)) && break
    echo "[$(stamp)] waiting for space on $HOST: $((free_kb / 1024 / 1024)) GB free, need $((need_kb / 1024 / 1024)) GB"
    sleep 120
  done
  echo "[$(stamp)] ($i/${#ARCHIVES[@]}) $CURRENT ($(du -h "$archive" | cut -f1))"
  scp -o BatchMode=yes -q "$archive" "$HOST:$REMOTE_MODELS/.$CURRENT.partial"
  "${SSH[@]}" "mv ~/$REMOTE_MODELS/.$CURRENT.partial ~/$REMOTE_MODELS/$CURRENT"
done
CURRENT=""
echo "[$(stamp)] all archives sent; this machine can be switched off"
