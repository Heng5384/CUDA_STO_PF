#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/scaled_resolved_seed_growth_and_overshoot_audit}"
REPORT_ROOT="${REPORT_ROOT:-reports/scaled_resolved_seed_growth_and_overshoot_audit}"

cd "$ROOT"
mkdir -p "$OUT_ROOT" "$REPORT_ROOT/data"

if pgrep -af main_cuda | grep -v "run_scaled_resolved_seed_growth_and_overshoot_audit_workstation" >/tmp/scaled_seed_growth_running.txt; then
  echo "[abort] main_cuda is already running; not starting scaled seed audit." >&2
  cat /tmp/scaled_seed_growth_running.txt >&2
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
  timeout 7200 ./main_cuda "$@" >"$dir/stdout.log" 2>"$dir/stderr.log"
  local rc=$?
  set -e
  echo "EXIT $rc" >"$dir/status.txt"
  if [ "$rc" -ne 0 ]; then
    echo "[fail] $name exited with $rc" >&2
    return "$rc"
  fi
}

# This is a diagnostic completion path, not physical capacity/rate calibration.
# It still goes through staged embryo -> accumulation -> target refresh ->
# resolved dynamic-continued profile handoff -> PF evolution.
COMMON_ARGS=(
  --Nx 128 --Ny 128 --Nz 128
  --nsteps 1100
  --out-every 250
  --csv-out-every 50
  --resolved_handoff_xB_write_mode preserve_profile_xB_alpha_in_support
  --scheduled-nuc-source-lambda-nm 0.6
  --scheduled-nuc-target-lambda-nm 4.0
  --scheduled-nuc-scale-interface-width 0
  --scheduled-nuc-scale-xB-profile-width 0
  --beta_staged_accumulation_enabled 1
  --beta_staged_accumulation_interval_steps 10
  --beta_staged_accumulation_GP_capture_radius_nm 50
  --beta_staged_accumulation_matrix_draw_radius_nm 50
  --beta_staged_accumulation_max_fraction_per_step 1.0
  --beta_staged_accumulation_max_inventory_per_step 1e300
  --beta_staged_insert_when_target_reached 1
  --beta_staged_debug_accelerated_accumulation 1
  --beta_staged_debug_accumulation_rate_multiplier 1000000
  --beta_staged_debug_stop_after_resolved_insert 0
)

run_case scaled_T380 \
  "${COMMON_ARGS[@]}" \
  --pf-param-file params/beta_capacity_gated_handoff/T380_S05_staged_handoff_diag_20.params \
  --init-case-tag scaled_T380_growth_overshoot

run_case scaled_T400 \
  "${COMMON_ARGS[@]}" \
  --pf-param-file params/beta_capacity_gated_handoff/T400_S05_staged_handoff_diag_20.params \
  --init-case-tag scaled_T400_growth_overshoot

python3 scripts/analyze_scaled_resolved_seed_growth_and_overshoot.py \
  --run-root "$OUT_ROOT" \
  --report-root "$REPORT_ROOT" \
  >"$OUT_ROOT/analyzer.stdout.log" \
  2>"$OUT_ROOT/analyzer.stderr.log"
echo "EXIT $?" >"$OUT_ROOT/analyzer.status.txt"

echo "scaled_resolved_seed_growth_and_overshoot_workstation_jobs_complete"
