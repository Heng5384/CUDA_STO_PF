#!/usr/bin/env bash
# One serialized PF-only 380 C, 6 h -> 48 h effective-population pilot.
# Every accepted one-hour boundary is a checksummed PF zero-mode checkpoint.

#SBATCH --job-name=pf_6h48h
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
FIXTURE_ROOT="${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
if [[ -e "${OUT_ROOT}" ]]; then
  echo "[fatal] refusing to overwrite output root: ${OUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUT_ROOT}"
exec >"${OUT_ROOT}/driver.stdout.log" 2>"${OUT_ROOT}/driver.stderr.log"
trap 'rc=$?; if [[ "$rc" -ne 0 ]]; then echo "FAILED_EXIT_${rc}" >"${OUT_ROOT}/status.txt"; fi' EXIT

cd "${PROJECT_ROOT}"
source jobs/_site_env.sh
site_prepare_cuda_env
echo "RUNNING" >"${OUT_ROOT}/status.txt"
make main_cuda CUDA_ARCH="${CUDA_ARCH:-sm_80}" CUDA_ROOT="${CUDA_ROOT}"
make test_pf_zero_mode_checkpoint

export TEMP_C=380
export DT=0.02
export PHYS_DX_REF_M=1.0e-9
export PF_DX_M=1.0e-9
export PHYS_LAMBDA_SM_M=4.0e-9
export PHYS_GAMMA_JM2=0.168
export PHYS_L_REF_FACTOR=5.0
export PHYS_D_RATIO=0.01
PF_PARAM_GENERATED="$(site_generate_pf_param_file pf_6h48h_T380)"
PF_PARAM_FILE="${OUT_ROOT}/pf_input.params"
python3 - "${PF_PARAM_GENERATED}" "${PF_PARAM_FILE}" <<'PY'
import pathlib, sys
src, dst = map(pathlib.Path, sys.argv[1:])
overrides = {
    "pf_composition_mode": "legacy",
    "pf_y_update_mode": "lagged_rhs",
    "pf_baseline_control_mode": "full",
    "enable_legacy_gp_storage_coupling": "0",
    "enable_gp_assisted_beta_nucleation": "0",
    "enable_gp_runtime_library_nucleation": "0",
    "gp_nuc_enabled": "0",
    "gp_to_beta_enabled": "0",
    "gp_initial_population_enabled": "0",
    "gp_birth_enabled": "0",
    "gp_growth_enabled": "0",
    "gp_radius_evolution_enabled": "0",
    "gp_inventory_growth_enabled": "0",
    "gp_literature_model_enabled": "0",
    "diagnostic_rsmd_enabled": "0",
    "enable_dynamic_continue_bridge": "0",
    "enable_Y_rhs_picard": "0",
    "disable_Y_rhs_gamma_term": "0",
    "y_update_mass_projection_enabled": "0",
    "y_update_mass_projection_report_enabled": "0",
    "dynamics_mass_diag_enabled": "1",
    "dynamics_mass_diag_interval": "1",
}
seen = set()
lines = []
for raw in src.read_text().splitlines():
    line = raw.strip()
    if line and not line.startswith("#") and "=" in line:
        key = line.split("=", 1)[0].strip()
        if key in overrides:
            lines.append(f"{key}={overrides[key]}")
            seen.add(key)
            continue
    lines.append(raw)
for key, value in overrides.items():
    if key not in seen:
        lines.append(f"{key}={value}")
dst.write_text("\n".join(lines) + "\n")
PY
cp "${PF_PARAM_GENERATED}" "${OUT_ROOT}/converter_pf_input.params"
cp jobs/generated/physical_inputs_pf_6h48h_T380.json "${OUT_ROOT}/physical_inputs.json"
cp jobs/generated/pf_payload_pf_6h48h_T380.json "${OUT_ROOT}/converter_payload.json"
cp "${FIXTURE_ROOT}/fixture_manifest.json" "${OUT_ROOT}/fixture_manifest.json"
cp "${FIXTURE_ROOT}/fixture_hashes.md" "${OUT_ROOT}/fixture_hashes.md"

python3 - "${PF_PARAM_FILE}" "${OUT_ROOT}/unit_contract.json" <<'PY'
import json, pathlib, sys
vals = {}
for raw in pathlib.Path(sys.argv[1]).read_text().splitlines():
    line = raw.strip()
    if line and not line.startswith('#') and '=' in line:
        k, v = line.split('=', 1); vals[k.strip()] = v.strip()
keys = ('temperature_C', 'dt', 't_real_unit', 'dx', 'lambda_sm_m', 'ic_phi_iface_w', 'D_alpha', 'D_compound', 'L_phi')
out = {k: float(vals[k]) for k in keys if k in vals}
out['physical_dt_s'] = out['dt'] * out['t_real_unit']
pathlib.Path(sys.argv[2]).write_text(json.dumps(out, indent=2) + '\n')
print(json.dumps(out, indent=2))
PY

DT_CODE="$(awk -F= '$1=="dt" {print $2}' "${PF_PARAM_FILE}" | tail -n1)"
T_REAL_UNIT="$(awk -F= '$1=="t_real_unit" {print $2}' "${PF_PARAM_FILE}" | tail -n1)"
STEPS_PER_HOUR="$(python3 - "${DT_CODE}" "${T_REAL_UNIT}" <<'PY'
import math, sys
dt, t0 = map(float, sys.argv[1:])
print(int(round(3600.0 / (dt * t0))))
PY
)"
PHYSICAL_DT="$(python3 - "${DT_CODE}" "${T_REAL_UNIT}" <<'PY'
import sys
print('%.17g' % (float(sys.argv[1]) * float(sys.argv[2])))
PY
)"
echo "dt_code=${DT_CODE}" >"${OUT_ROOT}/time_contract.txt"
echo "t_real_unit_s=${T_REAL_UNIT}" >>"${OUT_ROOT}/time_contract.txt"
echo "dt_physical_s=${PHYSICAL_DT}" >>"${OUT_ROOT}/time_contract.txt"
echo "steps_per_physical_hour=${STEPS_PER_HOUR}" >>"${OUT_ROOT}/time_contract.txt"

mkdir -p "${OUT_ROOT}/segments" "${OUT_ROOT}/checkpoints" "${OUT_ROOT}/analysis"
N=128
MATRIX_XB=0.006219279767278563
CURRENT_CHECKPOINT=""

run_segment() {
  local age_h="$1"
  local elapsed_h=$((age_h - 6))
  local endpoint_step=$((elapsed_h * STEPS_PER_HOUR))
  local seg="${OUT_ROOT}/segments/age_${age_h}h"
  local results="${seg}/results"
  mkdir -p "${results}"
  local checkpoint_path="${OUT_ROOT}/checkpoint_current.chk"
  local args=("${N}" "${N}" "${N}" "${DT_CODE}" "${endpoint_step}" "${endpoint_step}" "${endpoint_step}" 0
    --pf-param-file "${PF_PARAM_FILE}" --mode dynamics
    --ic-23d-xB-out "${MATRIX_XB}"
    --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1
    --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1
    --pf-zero-mode-tol-rel 1e-12 --pf-zero-mode-max-iter 24
    --pf-checkpoint-every "${STEPS_PER_HOUR}"
    --pf-checkpoint-path "${checkpoint_path}"
    --init-case-tag "pf_6h48h_age_${age_h}h")
  if [[ -z "${CURRENT_CHECKPOINT}" ]]; then
    args+=(--radius-phys-nm 0 --init-mode raw_fields
      --init-phi-raw "${FIXTURE_ROOT}/phi.raw.f64"
      --init-xB-raw "${FIXTURE_ROOT}/xB.raw.f64"
      --init-meta "${FIXTURE_ROOT}/raw_init_meta.json")
  else
    args+=(--pf-restart-from "${CURRENT_CHECKPOINT}")
  fi
  echo "SEGMENT_START age_h=${age_h} endpoint_step=${endpoint_step}"
  CUDA_STO_RESULTS_ROOT="${results}" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
    ./main_cuda "${args[@]}" >"${seg}/stdout.log" 2>"${seg}/stderr.log"
  if grep -qiE 'fatal|nan|inf|PF zero-mode rejected|mass projection' "${seg}/stderr.log"; then
    echo "[fatal] disallowed diagnostic in ${seg}/stderr.log" >&2
    return 3
  fi
  cp "${checkpoint_path}" "${OUT_ROOT}/checkpoints/step_${endpoint_step}.chk"
  CURRENT_CHECKPOINT="${OUT_ROOT}/checkpoints/step_${endpoint_step}.chk"
  grep -q 'PF_ZERO_MODE_FINAL_AUDIT status=PASS' "${seg}/stdout.log"
}

# The fixture is the 6 h state; the 42 one-hour segments reach 48 h.
for age in $(seq 1 42); do
  absolute_age=$((age + 6))
  run_segment "${absolute_age}"
done

python3 scripts/analyze_pf_zero_mode_checkpoints.py \
  --glob "${OUT_ROOT}/checkpoints/step_*.chk" \
  --out "${OUT_ROOT}/analysis" --dx-nm 1.0 --mean-ctot 0.03

python3 - "${OUT_ROOT}" "${STEPS_PER_HOUR}" "${PHYSICAL_DT}" <<'PY'
import csv, json, pathlib, sys
root = pathlib.Path(sys.argv[1])
pop = list(csv.DictReader((root/'analysis/population_time_series.csv').open()))
mass = list(csv.DictReader((root/'analysis/mass_and_zero_mode_time_series.csv').open()))
count = [int(r['resolved_particle_count']) for r in pop]
radii = {}
for row in csv.DictReader((root/'analysis/particle_trajectories.csv').open()):
    radii.setdefault(int(row['particle_id']), []).append(float(row['equivalent_radius_nm']))
max_mass = max(abs(float(r['mass_error_mean'])) for r in mass)
matrix_xag = [float(r['matrix_xAg_alpha']) for r in mass]
monotonic_count = all(a >= b for a,b in zip(count, count[1:]))
initial_mean_r = sum(v[0] for v in radii.values()) / max(len(radii), 1)
last_vals = [v[-1] for v in radii.values() if v]
last_mean_r = sum(last_vals) / max(len(last_vals), 1)
trend = monotonic_count and (count[-1] < count[0] or last_mean_r > initial_mean_r * 1.01)
summary = {
  'status': 'PASS_T380_6H_48H_COARSENING_TREND_PILOT' if (max_mass <= 1e-10 and trend and min(matrix_xag) >= 0.0058 and max(matrix_xag) <= 0.0066) else 'BLOCKED_COARSENING_TREND_NOT_RESOLVED',
  'elapsed_physical_time_s': 42 * 3600,
  'dt_physical_s': float(sys.argv[3]), 'steps_per_hour': int(sys.argv[2]),
  'initial_resolved_beta_count': count[0], 'final_resolved_beta_count': count[-1],
  'resolved_density_ratio_48h_over_6h': count[-1] / count[0],
  'initial_mean_equivalent_radius_nm': initial_mean_r,
  'final_mean_equivalent_radius_nm': last_mean_r,
  'max_mass_error_mean': max_mass, 'matrix_xAg_min': min(matrix_xag), 'matrix_xAg_max': max(matrix_xag),
  'count_nonincreasing': monotonic_count, 'trend_materially_resolved': trend,
  'zero_mode_status': 'PASS', 'GP_used': False, 'external_source_used': False,
}
(root/'pilot_summary.json').write_text(json.dumps(summary, indent=2) + '\n')
(root/'status.txt').write_text(summary['status'] + '\n')
print(json.dumps(summary, indent=2))
PY

echo "PASS_PF_6H48H_PILOT_CHAIN_COMPLETE" >"${OUT_ROOT}/chain_status.txt"
