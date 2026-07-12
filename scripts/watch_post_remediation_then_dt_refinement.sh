#!/usr/bin/env bash
set -euo pipefail

# Waits for the already-running serial matrix. It never terminates or overlaps
# that workload; it merely starts the pre-registered dt-only refinement after
# the GPU becomes idle.
ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
PREVIOUS_RUN_ROOT="${PREVIOUS_RUN_ROOT:-tmp_codex_ops/post_remediation_growth_validation_transport_fixed}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/post_remediation_dt_refinement}"
REPORT_ROOT="${REPORT_ROOT:-reports/post_remediation_dt_refinement}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"
while ! grep -qx 'post_remediation_growth_validation_workstation_complete' "$PREVIOUS_RUN_ROOT/runner.log" 2>/dev/null; do
  sleep "$POLL_SECONDS"
done
while pgrep -x main_cuda >/dev/null; do
  sleep "$POLL_SECONDS"
done

nohup env OUT_ROOT="$OUT_ROOT" bash scripts/run_post_remediation_dt_refinement_workstation.sh \
  >"$OUT_ROOT/runner.log" 2>&1 < /dev/null &
runner_pid=$!
wait "$runner_pid"
set +e
python3 scripts/analyze_post_remediation_dt_refinement.py \
  --manifest params/post_remediation_dt_refinement/post_remediation_dt_refinement_manifest.csv \
  --run-root "$OUT_ROOT" --report-root "$REPORT_ROOT"
dt_analysis_rc=$?
set -e
set +e
python3 scripts/assemble_interface_supply_growth_evidence.py
assemble_rc=$?
set -e
printf 'dt_analysis_rc=%s\nassemble_rc=%s\n' "$dt_analysis_rc" "$assemble_rc" \
  >"$OUT_ROOT/finalizer.status"
exit $((dt_analysis_rc != 0 || assemble_rc != 0))
