#!/usr/bin/env bash
set -euo pipefail

# Wait for the natural Phase 1 run to finish, then launch Phase 2/3/4.
# This is workstation-only and does not start any GPU job until the natural
# run has written its status.txt and no main_cuda process remains.

ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
NATURAL_DIR="${NATURAL_DIR:-tmp_codex_ops/beta_staged_handoff_validation/phase1_natural_T380_S005_10000}"
POLL_SECONDS="${POLL_SECONDS:-120}"
MAX_WAIT_SECONDS="${MAX_WAIT_SECONDS:-0}"

cd "$ROOT"

start_ts="$(date +%s)"
mkdir -p tmp_codex_ops/beta_staged_handoff_validation
watch_log="tmp_codex_ops/beta_staged_handoff_validation/phase234_watcher.log"

echo "[watch] started $(date -Is)" | tee -a "$watch_log"
echo "[watch] natural_dir=$NATURAL_DIR poll_seconds=$POLL_SECONDS max_wait_seconds=$MAX_WAIT_SECONDS" | tee -a "$watch_log"

while [ ! -f "$NATURAL_DIR/status.txt" ]; do
  now_ts="$(date +%s)"
  elapsed=$((now_ts - start_ts))
  if [ "$MAX_WAIT_SECONDS" -gt 0 ] && [ "$elapsed" -gt "$MAX_WAIT_SECONDS" ]; then
    echo "[watch] timeout after ${elapsed}s; natural run still incomplete" | tee -a "$watch_log"
    exit 124
  fi
  if [ -f "$NATURAL_DIR/stdout.log" ]; then
    last_step="$(python3 - <<'PY'
import re, pathlib
p = pathlib.Path("tmp_codex_ops/beta_staged_handoff_validation/phase1_natural_T380_S005_10000/stdout.log")
text = p.read_text(errors="ignore") if p.exists() else ""
steps = [int(x) for x in re.findall(r"step=(\d+)", text)]
print(max(steps or [0]))
PY
)"
    echo "[watch] $(date -Is) waiting natural status; last_step=$last_step elapsed=${elapsed}s" | tee -a "$watch_log"
  else
    echo "[watch] $(date -Is) waiting natural stdout; elapsed=${elapsed}s" | tee -a "$watch_log"
  fi
  sleep "$POLL_SECONDS"
done

echo "[watch] natural status: $(cat "$NATURAL_DIR/status.txt")" | tee -a "$watch_log"
echo "[watch] launching phase234 runner $(date -Is)" | tee -a "$watch_log"

scripts/run_beta_staged_handoff_phase23_workstation.sh 2>&1 | tee -a "$watch_log"

echo "[watch] complete $(date -Is)" | tee -a "$watch_log"
