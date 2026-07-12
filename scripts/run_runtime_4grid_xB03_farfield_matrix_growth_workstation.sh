#!/usr/bin/env bash
set -euo pipefail

ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/runtime_4grid_xB03_farfield_matrix_growth_test}"
REPORT_ROOT="${REPORT_ROOT:-reports/runtime_4grid_xB03_farfield_matrix_growth_test}"

cd "$ROOT"
mkdir -p "$OUT_ROOT" "$REPORT_ROOT"

if pgrep -af main_cuda | grep -v "run_runtime_4grid_xB03_farfield_matrix_growth_workstation" >/tmp/runtime_4grid_xB03_main_cuda_running.txt; then
  echo "[abort] main_cuda is already running; not starting xB03 4-grid diagnostic." >&2
  cat /tmp/runtime_4grid_xB03_main_cuda_running.txt >&2
  exit 75
fi

python3 scripts/prepare_runtime_4grid_xB03_farfield_params.py

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
  local param_file="$1"; shift
  local nsteps="$1"; shift
  local scale_phi="$1"; shift
  local target_lambda="$1"; shift
  local csv_every="$1"; shift
  local dir="$OUT_ROOT/$name"
  mkdir -p "$dir"
  echo "[run] $name nsteps=$nsteps scale=$scale_phi target_lambda=$target_lambda param=$param_file"
  set +e
  timeout 21600 ./main_cuda \
    --Nx 128 --Ny 128 --Nz 128 \
    --nsteps "$nsteps" \
    --out-every "$nsteps" \
    --csv-out-every "$csv_every" \
    --resolved_handoff_xB_write_mode preserve_profile_xB_alpha_in_support \
    --scheduled-nuc-source-lambda-nm 0.6 \
    --scheduled-nuc-target-lambda-nm "$target_lambda" \
    --scheduled-nuc-scale-interface-width "$scale_phi" \
    --scheduled-nuc-scale-xB-profile-width "$scale_phi" \
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
    --phi_eta_rhs_attribution_diag_enabled 1 \
    --phi_eta_rhs_attribution_diag_every 1 \
    --phi_eta_rhs_attribution_diag_max_steps 160 \
    --phi_eta_rhs_attribution_diag_prefix phi_eta_rhs_attribution \
    --pf-param-file "$param_file" \
    --init-case-tag "$name" \
    >"$dir/stdout.log" 2>"$dir/stderr.log"
  local rc=$?
  set -e
  echo "EXIT $rc" >"$dir/status.txt"
  if [ "$rc" -ne 0 ]; then
    echo "[warn] $name exited with $rc" >&2
  fi
  return 0
}

t400_unscaled="params/beta_capacity_gated_handoff/T400_4grid_xB03_matrix_unscaled_handoff_diag.params"
t400_scaled="params/beta_capacity_gated_handoff/T400_4grid_xB03_matrix_scaled_handoff_diag.params"
t380_unscaled="params/beta_capacity_gated_handoff/T380_4grid_xB03_matrix_unscaled_handoff_diag.params"
t380_scaled="params/beta_capacity_gated_handoff/T380_4grid_xB03_matrix_scaled_handoff_diag.params"

run_case T400_4grid_xB03_unscaled_1040 "$t400_unscaled" 1040 1.0 0.6 10
run_case T400_4grid_xB03_scaled_300 "$t400_scaled" 300 6.6666666667 4.0 10
run_case T380_4grid_xB03_unscaled_1040 "$t380_unscaled" 1040 1.0 0.6 10

python3 - <<'PY' > tmp_codex_ops/runtime_4grid_xB03_farfield_matrix_growth_test/scaled_survival_gate.txt
from pathlib import Path
import csv
import re

root = Path("tmp_codex_ops/runtime_4grid_xB03_farfield_matrix_growth_test/T400_4grid_xB03_scaled_300")
stdout = root / "stdout.log"
txt = stdout.read_text(errors="ignore") if stdout.exists() else ""
m = re.search(r"case_output_dir\s*:\s*(\S+)", txt)
case_dir = Path(m.group(1)) if m else root
p = case_dir / "handoff_profile_probes.csv"
last = None
first = None
if p.exists():
    with p.open(newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("probe_label") == "PROBE_STEP_END" and float(row.get("beta_phi_sum", "0") or 0) > 0:
                first = first or row
                last = row
survives = False
if first and last:
    h0 = float(first.get("beta_phi_sum", 0.0) or 0.0)
    hf = float(last.get("beta_phi_sum", 0.0) or 0.0)
    pf = float(last.get("beta_phi_max", 0.0) or 0.0)
    survives = hf > 0.5 * h0 and pf > 0.5
print("survives=" + ("true" if survives else "false"))
PY

if grep -q "survives=true" "$OUT_ROOT/scaled_survival_gate.txt"; then
  run_case T400_4grid_xB03_scaled_1040 "$t400_scaled" 1040 6.6666666667 4.0 10
  run_case T380_4grid_xB03_scaled_300 "$t380_scaled" 300 6.6666666667 4.0 10
else
  echo "[gate] T400 xB03 4-grid scaled did not survive 300-step smoke; skipping scaled extension/T380 scaled." | tee "$OUT_ROOT/scaled_skip_reason.txt"
fi

if python3 - <<'PY' >/dev/null 2>&1
import pandas  # noqa: F401
PY
then
  python3 scripts/analyze_runtime_4grid_xB03_farfield_matrix_growth.py \
    --run-root "$OUT_ROOT" \
    --report-root "$REPORT_ROOT" \
    >"$OUT_ROOT/analyzer.stdout.log" \
    2>"$OUT_ROOT/analyzer.stderr.log"
  echo "EXIT $?" >"$OUT_ROOT/analyzer.status.txt"
else
  echo "EXIT 77 (pandas unavailable; analyze after rsync to local)" >"$OUT_ROOT/analyzer.status.txt"
  echo "[skip] pandas unavailable on workstation; run analyzer locally after syncing results." | tee "$OUT_ROOT/analyzer.stdout.log"
  : >"$OUT_ROOT/analyzer.stderr.log"
fi

echo "runtime_4grid_xB03_farfield_matrix_growth_workstation_jobs_complete"
