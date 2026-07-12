#!/usr/bin/env bash
set -euo pipefail

# Workstation-only staged GP-to-beta validation runner.
# This script is meant to be launched only after the natural 10000-step run has
# finished, so it refuses to start if another main_cuda process is active.

ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/beta_staged_handoff_validation}"

cd "$ROOT"

if pgrep -af main_cuda | grep -v "run_beta_staged_handoff_phase23_workstation" >/tmp/beta_phase23_running.txt; then
  echo "[abort] main_cuda is already running; not starting Phase 2/3 jobs." >&2
  cat /tmp/beta_phase23_running.txt >&2
  exit 75
fi

make main_cuda NVCC="$NVCC" CUDA_ROOT="$CUDA_ROOT" -j1

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
}

COMMON_ACCUM=(
  --Nx 128 --Ny 128 --Nz 128
  --nsteps 10000
  --out-every 10000
  --csv-out-every 10000
  --beta_staged_accumulation_enabled 1
  --beta_staged_accumulation_interval_steps 10
  --beta_staged_accumulation_GP_capture_radius_nm 12
  --beta_staged_accumulation_matrix_draw_radius_nm 2
  --beta_staged_accumulation_max_fraction_per_step 0.1
  --beta_staged_insert_when_target_reached 1
)

COMMON_ACCEL=(
  --Nx 128 --Ny 128 --Nz 128
  --nsteps 2000
  --out-every 2000
  --csv-out-every 2000
  --beta_staged_accumulation_enabled 1
  --beta_staged_accumulation_interval_steps 10
  --beta_staged_accumulation_GP_capture_radius_nm 12
  --beta_staged_accumulation_matrix_draw_radius_nm 2
  --beta_staged_accumulation_max_fraction_per_step 0.1
  --beta_staged_insert_when_target_reached 1
  --beta_staged_debug_accelerated_accumulation 1
  --beta_staged_debug_accumulation_rate_multiplier 1000000
  --beta_staged_debug_stop_after_resolved_insert 1
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

run_case phase2_forced_T380_S05_10000 \
  "${COMMON_ACCUM[@]}" \
  --pf-param-file params/beta_capacity_gated_handoff/T380_S05_staged_handoff_diag_20.params \
  --init-case-tag phase2_forced_T380_S05_10000

run_case phase2_forced_T400_S05_10000 \
  "${COMMON_ACCUM[@]}" \
  --pf-param-file params/beta_capacity_gated_handoff/T400_S05_staged_handoff_diag_20.params \
  --init-case-tag phase2_forced_T400_S05_10000

run_case phase3_accelerated_handoff_T380_S05 \
  "${COMMON_ACCEL[@]}" \
  --pf-param-file params/beta_capacity_gated_handoff/T380_S05_staged_handoff_diag_20.params \
  --init-case-tag phase3_accelerated_handoff_T380_S05

run_case phase3_accelerated_handoff_T400_S05 \
  "${COMMON_ACCEL[@]}" \
  --pf-param-file params/beta_capacity_gated_handoff/T400_S05_staged_handoff_diag_20.params \
  --init-case-tag phase3_accelerated_handoff_T400_S05

run_case phase4_reference_T380_S0p5_1000 \
  "${COMMON_REF[@]}" \
  --pf-param-file params/beta_enabled_long_coupling_diagnostic/T380_beta_enabled_S0p5_1000.params \
  --init-case-tag phase4_reference_T380_S0p5_1000

run_case phase4_reference_T400_S0p5_1000 \
  "${COMMON_REF[@]}" \
  --pf-param-file params/beta_enabled_long_coupling_diagnostic/T400_beta_enabled_S0p5_1000.params \
  --init-case-tag phase4_reference_T400_S0p5_1000

run_case phase4_reference_T450_S0p5_1000 \
  "${COMMON_REF[@]}" \
  --pf-param-file params/beta_capacity_gated_handoff/T450_S05_capacity_gated_smoke_1000.params \
  --init-case-tag phase4_reference_T450_S0p5_1000

echo "phase234_workstation_jobs_complete"
