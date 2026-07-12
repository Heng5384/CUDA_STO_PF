#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/gp_required_supply_informed_rsmd}"
REPORT_ROOT="${REPORT_ROOT:-reports/gp_required_supply_informed_rsmd}"
TIMEOUT_S="${TIMEOUT_S:-43200}"
MAX_CASES="${MAX_CASES:-0}"
CASE_FILTER="${CASE_FILTER:-}"
INCLUDE_COUNTERFACTUAL_RUNS="${INCLUDE_COUNTERFACTUAL_RUNS:-0}"

cd "$ROOT"
mkdir -p "$OUT_ROOT" "$REPORT_ROOT"

if pgrep -af main_cuda | grep -v "run_gp_required_supply_informed_rsmd_workstation" >/tmp/gp_required_supply_rsmd_main_cuda_running.txt; then
  echo "[abort] main_cuda is already running; not starting GP required-supply-informed RSMD sweep." >&2
  cat /tmp/gp_required_supply_rsmd_main_cuda_running.txt >&2
  exit 75
fi

prepare_args=()
if [ "$INCLUDE_COUNTERFACTUAL_RUNS" = "1" ]; then
  prepare_args+=(--include-counterfactual-runs)
fi
python3 scripts/prepare_gp_required_supply_informed_rsmd.py "${prepare_args[@]}"

set +e
make main_cuda NVCC="$NVCC" CUDA_ROOT="$CUDA_ROOT" -j1 >"$OUT_ROOT/build.stdout.log" 2>"$OUT_ROOT/build.stderr.log"
build_rc=$?
set -e
echo "EXIT $build_rc" >"$OUT_ROOT/build.status.txt"
if [ "$build_rc" -ne 0 ]; then
  echo "[fail] build failed" >&2
  exit "$build_rc"
fi

manifest="params/gp_required_supply_informed_rsmd/manifest.csv"
python3 - "$manifest" "$CASE_FILTER" "$MAX_CASES" <<'PY' >"$OUT_ROOT/selected_cases.tsv"
import csv
import sys

manifest, case_filter, max_cases = sys.argv[1], sys.argv[2], int(sys.argv[3])
count = 0
with open(manifest, newline="") as fh:
    for row in csv.DictReader(fh):
        if row.get("run_selected") != "1":
            continue
        if case_filter and case_filter not in row["case"]:
            continue
        print("\t".join([
            row["case"], row["param_file"], row["T_C"], row["scenario_id"],
            row["xB_ceiling_eff"], row["R_exchange_nm"], row["chi_rel"],
            row["kernel_radius_dx"], row["nsteps"], row["provenance"],
        ]))
        count += 1
        if max_cases > 0 and count >= max_cases:
            break
PY

while IFS=$'\t' read -r case_name param_file T scenario target rex chi kernel nsteps provenance; do
  [ -n "$case_name" ] || continue
  case_dir="$OUT_ROOT/$case_name"
  mkdir -p "$case_dir"
  if [ -f "$case_dir/status.txt" ] && grep -q "EXIT 0" "$case_dir/status.txt"; then
    echo "[skip] $case_name already completed"
    continue
  fi
  echo "[run] $case_name T=$T scenario=$scenario target=$target R=$rex chi=$chi kernel=$kernel nsteps=$nsteps"
  echo "RUNNING" >"$case_dir/status.txt"
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
    --diagnostic_rsmd_enabled 1 \
    --diagnostic_rsmd_xB_halo_target "$target" \
    --diagnostic_rsmd_R_exchange_nm "$rex" \
    --diagnostic_rsmd_chi_rel "$chi" \
    --diagnostic_rsmd_kernel_radius_dx "$kernel" \
    --diagnostic_rsmd_release_window_steps "$nsteps" \
    --diagnostic_rsmd_provenance "$provenance" \
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

python3 scripts/analyze_gp_required_supply_informed_rsmd.py \
  --run-root "$OUT_ROOT" \
  --report-root "$REPORT_ROOT"

echo "gp_required_supply_informed_rsmd_workstation_complete"
