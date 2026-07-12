#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/t400_xbcrit_high_range_refinement}"
REPORT_ROOT="${REPORT_ROOT:-reports/t400_xbcrit_high_range_refinement}"
POST_CODE_TIME="${POST_CODE_TIME:-20.0}"
XBS="${XBS:-0.018,0.020,0.022,0.024,0.026,0.030}"
DTS="${DTS:-0.020,0.010,0.005}"
RESUME="${RESUME:-1}"

cd "$ROOT"
mkdir -p "$OUT_ROOT" "$REPORT_ROOT"

if pgrep -af main_cuda | grep -v "run_t400_xbcrit_high_range_refinement_workstation" >/tmp/t400_xbcrit_main_cuda_running.txt; then
  echo "[abort] main_cuda is already running; not starting T400 high-xB refinement." >&2
  cat /tmp/t400_xbcrit_main_cuda_running.txt >&2
  exit 75
fi

python3 scripts/prepare_t400_xbcrit_high_range_refinement_params.py \
  --xbs "$XBS" \
  --dts "$DTS" \
  --post-code-time "$POST_CODE_TIME"

set +e
make main_cuda NVCC="$NVCC" CUDA_ROOT="$CUDA_ROOT" -j1 >"$OUT_ROOT/build.stdout.log" 2>"$OUT_ROOT/build.stderr.log"
build_rc=$?
set -e
echo "EXIT $build_rc" >"$OUT_ROOT/build.status.txt"
if [ "$build_rc" -ne 0 ]; then
  echo "[fail] build failed" >&2
  python3 scripts/analyze_t400_xbcrit_high_range_refinement.py --run-root "$OUT_ROOT" --report-root "$REPORT_ROOT" || true
  exit "$build_rc"
fi

tail -n +2 params/t400_xbcrit_high_range_refinement/manifest.csv | while IFS=, read -r T xb dt dt_label nsteps csv_every case_tag param_file; do
  dir="$OUT_ROOT/$case_tag"
  mkdir -p "$dir"
  if [ "$RESUME" = "1" ] && [ -f "$dir/status.txt" ] && grep -q "EXIT 0" "$dir/status.txt"; then
    echo "[skip] $case_tag already completed"
    continue
  fi
  echo "[run] $case_tag param=$param_file nsteps=$nsteps csv_every=$csv_every"
  set +e
  timeout 28800 ./main_cuda \
    --Nx 128 --Ny 128 --Nz 128 \
    --nsteps "$nsteps" \
    --out-every "$nsteps" \
    --csv-out-every "$csv_every" \
    --pf-param-file "$param_file" \
    --dt "$dt" \
    --init-case-tag "$case_tag" \
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
  rc=$?
  set -e
  echo "EXIT $rc" >"$dir/status.txt"
  if [ "$rc" -ne 0 ]; then
    echo "[warn] $case_tag exited with $rc" >&2
  fi
done

python3 scripts/analyze_t400_xbcrit_high_range_refinement.py \
  --run-root "$OUT_ROOT" \
  --report-root "$REPORT_ROOT" \
  >"$OUT_ROOT/analyzer.stdout.log" \
  2>"$OUT_ROOT/analyzer.stderr.log"

cat "$REPORT_ROOT/final_terminal_output.txt"
echo "t400_xbcrit_high_range_refinement_workstation_complete"
