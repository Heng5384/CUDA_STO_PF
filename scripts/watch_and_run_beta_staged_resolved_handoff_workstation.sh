#!/usr/bin/env bash
set -euo pipefail
ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
POLL_SECONDS="${POLL_SECONDS:-300}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/beta_staged_resolved_handoff_debug}"
cd "$ROOT"
mkdir -p "$OUT_ROOT"
log="$OUT_ROOT/watcher.log"
echo "[watch] started $(date -Is) poll_seconds=$POLL_SECONDS" | tee -a "$log"
while true; do
  if ! pgrep -af main_cuda | grep -v "watch_and_run_beta_staged_resolved_handoff" >/tmp/beta_resolved_watch_running.txt; then
    echo "[watch] $(date -Is) no main_cuda active; starting resolved handoff runner" | tee -a "$log"
    scripts/run_beta_staged_resolved_handoff_workstation.sh >>"$OUT_ROOT/runner.nohup.log" 2>&1
    rc=$?
    echo "[watch] $(date -Is) runner_exit=$rc" | tee -a "$log"
    exit "$rc"
  fi
  echo "[watch] $(date -Is) waiting; active main_cuda:" | tee -a "$log"
  cat /tmp/beta_resolved_watch_running.txt | tee -a "$log"
  sleep "$POLL_SECONDS"
done
