#!/usr/bin/env bash
set -euo pipefail

# The current GPU case is allowed to finish naturally. Its serial launcher is
# paused by the caller so this diagnostic can run before older queued cases.
ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
CURRENT_MAIN_PID="${CURRENT_MAIN_PID:?CURRENT_MAIN_PID is required}"
PAUSED_RUNNER_PID="${PAUSED_RUNNER_PID:?PAUSED_RUNNER_PID is required}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/rsmd_y_solver_attribution}"
REPORT_ROOT="${REPORT_ROOT:-reports/rsmd_y_solver_attribution}"
STATUS_FILE="${OUT_ROOT}/priority_finalizer.status"
POLL_SECONDS="${POLL_SECONDS:-20}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"

resume_runner() {
  if kill -0 "$PAUSED_RUNNER_PID" 2>/dev/null; then
    kill -CONT "$PAUSED_RUNNER_PID"
  fi
}
trap resume_runner EXIT

while kill -0 "$CURRENT_MAIN_PID" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done
while pgrep -x main_cuda >/dev/null; do
  sleep "$POLL_SECONDS"
done

set +e
make main_cuda NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}" \
  CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}" -j1
build_rc=$?
run_rc=125
analysis_rc=125
if [ "$build_rc" -eq 0 ]; then
  bash scripts/run_rsmd_y_solver_attribution_workstation.sh
  run_rc=$?
  python3 scripts/analyze_rsmd_y_solver_attribution.py \
    --manifest params/rsmd_y_solver_attribution/rsmd_y_solver_attribution_manifest.csv \
    --run-root "$OUT_ROOT" --report-root "$REPORT_ROOT"
  analysis_rc=$?
fi
set -e

printf 'build_rc=%s\nrun_rc=%s\nanalysis_rc=%s\n' \
  "$build_rc" "$run_rc" "$analysis_rc" > "$STATUS_FILE"
exit $((build_rc != 0 || run_rc != 0 || analysis_rc != 0))
