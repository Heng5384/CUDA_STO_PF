#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/diagnostic_rsmd_T380_refined_required_supply_map}"
REPORT_ROOT="${REPORT_ROOT:-reports/diagnostic_rsmd_T380_refined_required_supply_map}"
TIMEOUT_S="${TIMEOUT_S:-43200}"

cd "$ROOT"
mkdir -p "$OUT_ROOT" "$REPORT_ROOT/data"

if pgrep -af main_cuda | grep -v "run_diagnostic_rsmd_T380_refined_workstation" >/tmp/diagnostic_rsmd_refined_main_cuda_running.txt; then
  echo "[abort] main_cuda is already running; not starting refined diagnostic RSMD T380 sweep." >&2
  cat /tmp/diagnostic_rsmd_refined_main_cuda_running.txt >&2
  exit 75
fi

python3 scripts/prepare_diagnostic_rsmd_T380_refined_params.py

set +e
make main_cuda NVCC="$NVCC" CUDA_ROOT="$CUDA_ROOT" -j1 >"$OUT_ROOT/build.stdout.log" 2>"$OUT_ROOT/build.stderr.log"
build_rc=$?
set -e
echo "EXIT $build_rc" >"$OUT_ROOT/build.status.txt"
if [ "$build_rc" -ne 0 ]; then
  echo "[fail] build failed" >&2
  exit "$build_rc"
fi

manifest="params/diagnostic_rsmd_T380_refined_required_supply_map/manifest.csv"
python3 - "$manifest" <<'PY' >"$OUT_ROOT/selected_cases.tsv"
import csv, sys
with open(sys.argv[1], newline="") as fh:
    for row in csv.DictReader(fh):
        if row.get("run_selected") == "1":
            print("\t".join([
                row["case"], row["param_file"], row["diagnostic_rsmd_enabled"],
                row["xB_halo_target"], row["R_exchange_nm"], row["chi_rel"],
                row["kernel_radius_dx"], row["nsteps"], row["selection_reason"],
            ]))
PY

while IFS=$'\t' read -r case_name param_file enabled target rex chi kernel nsteps reason; do
  [ -n "$case_name" ] || continue
  case_dir="$OUT_ROOT/$case_name"
  mkdir -p "$case_dir"
  if [ -f "$case_dir/status.txt" ] && grep -q "EXIT 0" "$case_dir/status.txt"; then
    echo "[skip] $case_name already completed"
    continue
  fi
  echo "[run] $case_name target=$target R=$rex chi=$chi kernel=$kernel nsteps=$nsteps reason=$reason"
  set +e
  timeout "$TIMEOUT_S" ./main_cuda \
    --Nx 128 --Ny 128 --Nz 128 \
    --nsteps "$nsteps" \
    --out-every "$nsteps" \
    --csv-out-every 50 \
    --beta_staged_accumulation_enabled 1 \
    --beta_staged_accumulation_interval_steps 10 \
    --beta_staged_accumulation_GP_capture_radius_nm 12 \
    --beta_staged_accumulation_matrix_draw_radius_nm 64 \
    --beta_staged_accumulation_max_fraction_per_step 0.1 \
    --beta_staged_accumulation_max_inventory_per_step 1e300 \
    --beta_staged_insert_when_target_reached 1 \
    --beta_staged_debug_accelerated_accumulation 1 \
    --beta_staged_debug_accumulation_rate_multiplier 1000000 \
    --beta_staged_debug_stop_after_resolved_insert 0 \
    --resolved_handoff_xB_write_mode preserve_profile_xB_alpha_in_support \
    --scheduled-nuc-source-lambda-nm 0.6 \
    --scheduled-nuc-target-lambda-nm 0.6 \
    --scheduled-nuc-scale-interface-width 1.0 \
    --scheduled-nuc-scale-xB-profile-width 1.0 \
    --diagnostic_rsmd_enabled "$enabled" \
    --diagnostic_rsmd_xB_halo_target "$target" \
    --diagnostic_rsmd_R_exchange_nm "$rex" \
    --diagnostic_rsmd_chi_rel "$chi" \
    --diagnostic_rsmd_kernel_radius_dx "$kernel" \
    --diagnostic_rsmd_release_window_steps "$nsteps" \
    --diagnostic_rsmd_provenance required_supply_diagnostic \
    --pf-param-file "$param_file" \
    --init-case-tag "$case_name" \
    >"$case_dir/stdout.log" 2>"$case_dir/stderr.log"
  rc=$?
  set -e
  echo "EXIT $rc" >"$case_dir/status.txt"
  if [ "$rc" -ne 0 ]; then
    echo "[warn] $case_name exited with $rc" >&2
  fi
done <"$OUT_ROOT/selected_cases.tsv"

python3 scripts/analyze_diagnostic_rsmd_T380_evidence_and_refined.py \
  --previous-report-root reports/diagnostic_rsmd_source_engine_T380 \
  --refined-out-root "$OUT_ROOT" \
  --refined-report-root "$REPORT_ROOT" \
  --final-report reports/diagnostic_rsmd_T380_refined_required_supply_final_report.md

echo "diagnostic_rsmd_T380_refined_workstation_complete"
