#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/staged_handoff_profile_audit}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"

if pgrep -af main_cuda | grep -v "run_staged_handoff_profile_audit_workstation" >/tmp/staged_profile_audit_running.txt; then
  echo "[abort] main_cuda is already running; not starting profile audit validation." >&2
  cat /tmp/staged_profile_audit_running.txt >&2
  exit 75
fi

set +e
make main_cuda NVCC="$NVCC" CUDA_ROOT="$CUDA_ROOT" -j1 >"$OUT_ROOT/build.stdout.log" 2>"$OUT_ROOT/build.stderr.log"
build_rc=$?
set -e
echo "EXIT $build_rc" >"$OUT_ROOT/build.status.txt"
if [ "$build_rc" -ne 0 ]; then
  echo "[fail] build failed" >&2
  exit "$build_rc"
fi

run_case() {
  local name="$1"
  shift
  local dir="$OUT_ROOT/$name"
  mkdir -p "$dir"
  echo "[run] $name"
  set +e
  ./main_cuda "$@" >"$dir/stdout.log" 2>"$dir/stderr.log"
  local rc=$?
  set -e
  echo "EXIT $rc" >"$dir/status.txt"
  if [ "$rc" -ne 0 ]; then
    echo "[fail] $name exited with $rc" >&2
    return "$rc"
  fi
  return 0
}

overall_rc=0

COMMON_FORCED_PROFILE=(
  --Nx 128 --Ny 128 --Nz 128
  --nsteps 40
  --out-every 200
  --csv-out-every 200
  --beta_staged_accumulation_enabled 1
  --beta_staged_accumulation_interval_steps 10
  --beta_staged_accumulation_GP_capture_radius_nm 12
  --beta_staged_accumulation_matrix_draw_radius_nm 64
  --beta_staged_accumulation_max_fraction_per_step 0.1
  --beta_staged_insert_when_target_reached 1
  --beta_staged_debug_accelerated_accumulation 1
  --beta_staged_debug_accumulation_rate_multiplier 1000000
  --beta_staged_debug_stop_after_resolved_insert 0
)

COMMON_REF=(
  --Nx 128 --Ny 128 --Nz 128
  --nsteps 1000
  --out-every 1000
  --csv-out-every 1000
  --beta_staged_accumulation_enabled 1
  --beta_staged_accumulation_interval_steps 10
  --beta_staged_accumulation_GP_capture_radius_nm 12
  --beta_staged_accumulation_matrix_draw_radius_nm 2
  --beta_staged_accumulation_max_fraction_per_step 0.1
  --beta_staged_insert_when_target_reached 1
)

run_case phase3_forced_profile_T380_S05 \
  "${COMMON_FORCED_PROFILE[@]}" \
  --pf-param-file params/beta_capacity_gated_handoff/T380_S05_staged_handoff_diag_20.params \
  --init-case-tag profile_audit_forced_T380_S05 || overall_rc=1

run_case phase3_forced_profile_T400_S05 \
  "${COMMON_FORCED_PROFILE[@]}" \
  --pf-param-file params/beta_capacity_gated_handoff/T400_S05_staged_handoff_diag_20.params \
  --init-case-tag profile_audit_forced_T400_S05 || overall_rc=1

run_case phase6_reference_T380_S0p5_1000 \
  "${COMMON_REF[@]}" \
  --pf-param-file params/beta_enabled_long_coupling_diagnostic/T380_beta_enabled_S0p5_1000.params \
  --init-case-tag profile_audit_reference_T380_S0p5_1000 || overall_rc=1

run_case phase6_reference_T400_S0p5_1000 \
  "${COMMON_REF[@]}" \
  --pf-param-file params/beta_enabled_long_coupling_diagnostic/T400_beta_enabled_S0p5_1000.params \
  --init-case-tag profile_audit_reference_T400_S0p5_1000 || overall_rc=1

run_case phase6_reference_T450_S0p5_1000 \
  "${COMMON_REF[@]}" \
  --pf-param-file params/beta_capacity_gated_handoff/T450_S05_capacity_gated_smoke_1000.params \
  --init-case-tag profile_audit_reference_T450_S0p5_1000 || overall_rc=1

echo "staged_handoff_profile_audit_workstation_jobs_complete"
exit "$overall_rc"
