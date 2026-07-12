#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/beta_seed_xBcrit_titration}"
REPORT_ROOT="${REPORT_ROOT:-reports/beta_seed_xBcrit_titration}"
NPOST="${NPOST:-1000}"
CSV_EVERY="${CSV_EVERY:-10}"
XBS="${XBS:-0.0078305391025,0.010,0.012,0.014,0.016,0.018,0.020,0.024,0.030}"
TEMPS="${TEMPS:-380,400}"

cd "$ROOT"
mkdir -p "$OUT_ROOT" "$REPORT_ROOT"

if pgrep -af main_cuda | grep -v "run_beta_seed_xBcrit_titration_workstation" >/tmp/beta_seed_xBcrit_main_cuda_running.txt; then
  echo "[abort] main_cuda is already running; not starting beta seed xBcrit titration." >&2
  cat /tmp/beta_seed_xBcrit_main_cuda_running.txt >&2
  exit 75
fi

python3 scripts/prepare_beta_seed_xBcrit_titration_params.py --temperatures "$TEMPS" --xbs "$XBS"

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
  local T="$1"
  local xb="$2"
  local param_file="$3"
  local xb_tag="${xb//./p}"
  xb_tag="${xb_tag//-/_}"
  local name="T${T}_xB${xb_tag}_stable_unscaled_noAQ"
  local dir="$OUT_ROOT/$name"
  mkdir -p "$dir"
  echo "[run] $name param=$param_file"
  set +e
  timeout 21600 ./main_cuda \
    --Nx 128 --Ny 128 --Nz 128 \
    --nsteps "$((NPOST + 60))" \
    --out-every "$((NPOST + 60))" \
    --csv-out-every "$CSV_EVERY" \
    --pf-param-file "$param_file" \
    --init-case-tag "$name" \
    --resolved_handoff_xB_write_mode preserve_profile_xB_alpha_in_support \
    --scheduled-nuc-source-lambda-nm 0.6 \
    --scheduled-nuc-target-lambda-nm 0.6 \
    --scheduled-nuc-scale-interface-width 1.0 \
    --scheduled-nuc-scale-xB-profile-width 1.0 \
    --beta_staged_accumulation_enabled 1 \
    --beta_staged_accumulation_interval_steps 10 \
    --beta_staged_accumulation_GP_capture_radius_nm 0 \
    --beta_staged_accumulation_matrix_draw_radius_nm 50 \
    --beta_staged_accumulation_max_fraction_per_step 1.0 \
    --beta_staged_accumulation_max_inventory_per_step 1e300 \
    --beta_staged_insert_when_target_reached 1 \
    --beta_staged_debug_accelerated_accumulation 1 \
    --beta_staged_debug_accumulation_rate_multiplier 1000000 \
    --beta_staged_debug_stop_after_resolved_insert 0 \
    >"$dir/stdout.log" 2>"$dir/stderr.log"
  local rc=$?
  set -e
  echo "EXIT $rc" >"$dir/status.txt"
  if [ "$rc" -ne 0 ]; then
    echo "[warn] $name exited with $rc" >&2
  fi
}

tail -n +2 params/beta_seed_xBcrit_titration/manifest.csv | while IFS=, read -r T xb param_file; do
  run_case "$T" "$xb" "$param_file"
done

if python3 - <<'PY' >/dev/null 2>&1
import csv
PY
then
  python3 scripts/analyze_beta_seed_xBcrit_titration.py \
    --run-root "$OUT_ROOT" \
    --report-root "$REPORT_ROOT" \
    >"$OUT_ROOT/analyzer.stdout.log" \
    2>"$OUT_ROOT/analyzer.stderr.log"
  echo "EXIT $?" >"$OUT_ROOT/analyzer.status.txt"
fi

echo "beta_seed_xBcrit_titration_workstation_jobs_complete"
