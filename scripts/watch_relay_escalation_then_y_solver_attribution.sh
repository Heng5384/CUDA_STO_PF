#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
PREVIOUS="${PREVIOUS:-tmp_codex_ops/rsmd_effective_relay_rate_escalation}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/rsmd_y_solver_attribution}"
REPORT_ROOT="${REPORT_ROOT:-reports/rsmd_y_solver_attribution}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"
while [ ! -f "$PREVIOUS/finalizer.status" ]; do sleep "$POLL_SECONDS"; done
while pgrep -x main_cuda >/dev/null; do sleep "$POLL_SECONDS"; done
make main_cuda NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}" CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}" -j1
set +e
bash scripts/run_rsmd_y_solver_attribution_workstation.sh >"$OUT_ROOT/runner.log" 2>&1
run_rc=$?
python3 scripts/analyze_rsmd_y_solver_attribution.py \
  --manifest params/rsmd_y_solver_attribution/rsmd_y_solver_attribution_manifest.csv \
  --run-root "$OUT_ROOT" --report-root "$REPORT_ROOT"
analysis_rc=$?
set -e
printf 'run_rc=%s\nanalysis_rc=%s\n' "$run_rc" "$analysis_rc" >"$OUT_ROOT/finalizer.status"
exit $((run_rc != 0 || analysis_rc != 0))
