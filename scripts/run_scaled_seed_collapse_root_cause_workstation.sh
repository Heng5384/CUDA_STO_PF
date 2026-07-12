#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/scaled_seed_collapse_root_cause_attribution}"
REPORT_ROOT="${REPORT_ROOT:-reports/scaled_seed_collapse_root_cause_attribution}"

cd "$ROOT"
mkdir -p "$OUT_ROOT" "$REPORT_ROOT"

if pgrep -af main_cuda | grep -v "run_scaled_seed_collapse_root_cause_workstation" >/tmp/scaled_seed_rootcause_running.txt; then
  echo "[abort] main_cuda is already running; not starting root-cause diagnostics." >&2
  cat /tmp/scaled_seed_rootcause_running.txt >&2
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
  local name="$1"; shift
  local scale="$1"; shift
  local dt="$1"; shift
  local dir="$OUT_ROOT/$name"
  mkdir -p "$dir"
  echo "[run] $name scale=$scale dt=$dt"
  set +e
  timeout 3600 ./main_cuda \
    --Nx 128 --Ny 128 --Nz 128 \
    --dt "$dt" \
    --nsteps 260 \
    --out-every 260 \
    --csv-out-every 10 \
    --resolved_handoff_xB_write_mode preserve_profile_xB_alpha_in_support \
    --scheduled-nuc-source-lambda-nm 0.6 \
    --scheduled-nuc-target-lambda-nm 4.0 \
    --scheduled-nuc-scale-interface-width "$scale" \
    --scheduled-nuc-scale-xB-profile-width "$scale" \
    --beta_staged_accumulation_enabled 1 \
    --beta_staged_accumulation_interval_steps 10 \
    --beta_staged_accumulation_GP_capture_radius_nm 50 \
    --beta_staged_accumulation_matrix_draw_radius_nm 50 \
    --beta_staged_accumulation_max_fraction_per_step 1.0 \
    --beta_staged_accumulation_max_inventory_per_step 1e300 \
    --beta_staged_insert_when_target_reached 1 \
    --beta_staged_debug_accelerated_accumulation 1 \
    --beta_staged_debug_accumulation_rate_multiplier 1000000 \
    --beta_staged_debug_stop_after_resolved_insert 0 \
    --phi_eta_rhs_attribution_diag_enabled 1 \
    --phi_eta_rhs_attribution_diag_every 1 \
    --phi_eta_rhs_attribution_diag_max_steps 80 \
    --phi_eta_rhs_attribution_diag_prefix phi_eta_rhs_attribution \
    --pf-param-file params/beta_capacity_gated_handoff/T400_S05_staged_handoff_diag_20.params \
    --init-case-tag "$name" \
    >"$dir/stdout.log" 2>"$dir/stderr.log"
  local rc=$?
  set -e
  echo "EXIT $rc" >"$dir/status.txt"
  if [ "$rc" -ne 0 ]; then
    echo "[warn] $name exited with $rc" >&2
  fi
}

run_case T400_scale1_dt0p02 1.0 0.02
run_case T400_scale2_dt0p02 2.0 0.02
run_case T400_scale3_dt0p02 3.0 0.02
run_case T400_scale4_dt0p02 4.0 0.02
run_case T400_scale6p666_dt0p02 6.6666666667 0.02
run_case T400_scale6p666_dt0p001 6.6666666667 0.001

python3 scripts/analyze_scaled_seed_collapse_root_cause.py \
  --run-root "$OUT_ROOT" \
  --scaled-audit-root reports/scaled_resolved_seed_growth_and_overshoot_audit \
  --unscaled-root reports/post_handoff_seed_stability_after_xB_writeback_fix \
  --report-root "$REPORT_ROOT" \
  >"$OUT_ROOT/analyzer.stdout.log" \
  2>"$OUT_ROOT/analyzer.stderr.log"
echo "EXIT $?" >"$OUT_ROOT/analyzer.status.txt"

echo "scaled_seed_collapse_root_cause_workstation_jobs_complete"
