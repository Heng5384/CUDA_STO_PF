#!/usr/bin/env bash
# Pure-PF timing matrix under the converter-derived 1 nm / 4 nm T400 contract.

#SBATCH --job-name=pf_zm_perf
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
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
make main_cuda CUDA_ARCH="${CUDA_ARCH}" CUDA_ROOT="${CUDA_ROOT}"

export TEMP_C=400
export DT=0.02
export VF_INIT=0.01
export PHYS_DX_REF_M=1.0e-9
export PF_DX_M=1.0e-9
export PHYS_LAMBDA_SM_M=4.0e-9
export PHYS_GAMMA_JM2=0.2
export PHYS_L_REF_FACTOR=5.0
export PHYS_D_RATIO=0.01
PF_PARAM_FILE="$(site_generate_pf_param_file pf_zero_mode_perf_T400)"
cp "${PF_PARAM_FILE}" "${OUT_ROOT}/pf_input.params"
cp jobs/generated/physical_inputs_pf_zero_mode_perf_T400.json \
  "${OUT_ROOT}/physical_inputs.json"
cp jobs/generated/pf_payload_pf_zero_mode_perf_T400.json \
  "${OUT_ROOT}/converter_payload.json"

python3 - "${PF_PARAM_FILE}" "${OUT_ROOT}/unit_contract_audit.json" <<'PY'
import json
import pathlib
import sys

params = {}
for raw in pathlib.Path(sys.argv[1]).read_text().splitlines():
    line = raw.strip()
    if line and not line.startswith("#") and "=" in line:
        key, value = line.split("=", 1)
        params[key.strip()] = value.strip()

expected = {
    "dx": 1.0,
    "dy": 1.0,
    "dz": 1.0,
    "lambda_sm_m": 4.0e-9,
    "ic_phi_iface_w": 2.0,
    "kappa_phi": 2.0,
    "D_alpha": 400.0,
    "D_compound": 4.0,
    "L_phi": 9.599212970350374,
    "t_real_unit": 41.12958542455477,
}
actual = {key: float(params[key]) for key in expected}
failures = []
for key, target in expected.items():
    tolerance = 1.0e-12 * max(abs(target), 1.0)
    if abs(actual[key] - target) > tolerance:
        failures.append({"key": key, "actual": actual[key], "expected": target})
audit = {
    "status": (
        "PASS_ETA_1NM_4NM_T400_CONVERTER_CONTRACT"
        if not failures else "FAIL_ETA_1NM_4NM_T400_CONVERTER_CONTRACT"
    ),
    "actual": actual,
    "expected": expected,
    "failures": failures,
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(audit, indent=2) + "\n")
if failures:
    raise SystemExit(audit["status"])
PY

run_case() {
  local label="$1"
  local n="$2"
  local steps="$3"
  local zero_mode="$4"
  local stdout="${OUT_ROOT}/${label}.stdout.log"
  local stderr="${OUT_ROOT}/${label}.stderr.log"
  local results_root="${OUT_ROOT}/results/${label}"
  local zero_args=()
  mkdir -p "${results_root}"
  if [[ "${zero_mode}" == "ON" ]]; then
    zero_args=(
      --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1
      --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1
      --pf-zero-mode-tol-rel 1e-12
      --pf-zero-mode-max-iter 24
    )
  else
    zero_args=(--pf-zero-mode OFF)
  fi
  set +e
  CUDA_STO_RESULTS_ROOT="${results_root}" \
    ./main_cuda "${n}" "${n}" "${n}" "${DT}" \
      "${steps}" "${steps}" "${steps}" 0 \
      --pf-param-file "${PF_PARAM_FILE}" \
      --mode dynamics \
      --radius 0.8 \
      --ic-23d-xB-out 0.03 \
      "${zero_args[@]}" \
      --init-case-tag "${label}" \
      >"${stdout}" 2>"${stderr}"
  local rc=$?
  set -e
  echo "${rc}" >"${OUT_ROOT}/${label}.exit_code"
  if [[ "${rc}" -ne 0 ]]; then
    echo "[fatal] ${label} exited ${rc}" >&2
    return "${rc}"
  fi
  if grep -vFx \
      -e "[warn] 界面分辨率不足: lambda_sm/dx = 4.000000 <= 4.000000，建议减小 dx 或增大 lambda_sm。" \
      -e "[warn] Seed radius is not large compared to interface width: phi_center_ideal=0.690 (<0.95). This often indicates a radius-unit mismatch (physical nm vs internal units) or too-small R." \
      -e "[warn] geometry summary skipped: no valid nucleus component found." \
      "${stderr}" | grep -q .; then
    echo "[fatal] ${label} produced unexpected stderr" >&2
    return 3
  fi
}

for n in 64 128 195; do
  run_case "N${n}_baseline_off" "${n}" 192 OFF
  run_case "N${n}_zero_mode" "${n}" 192 ON
done

python3 - "${OUT_ROOT}" <<'PY'
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])
rows = []
for n in (64, 128, 195):
    parsed = {}
    for mode in ("baseline_off", "zero_mode"):
        label = f"N{n}_{mode}"
        text = (root / f"{label}.stdout.log").read_text()
        gpu = re.findall(r"平均每步:\s*([+\-0-9.eE]+) ms", text)
        wall = re.findall(r"平均每步墙钟时间:\s*([+\-0-9.eE]+) s", text)
        audit = re.findall(
            r"PF_ZERO_MODE_FINAL_AUDIT status=(\S+).*?"
            r"mean_mass_error=([+\-0-9.eE]+).*?"
            r"zero_mode_wall_s=([+\-0-9.eE]+)",
            text,
        )
        parsed[mode] = {
            "gpu_event_ms_per_step": float(gpu[-1]),
            "wall_s_per_step": float(wall[-1]),
            "final_audit": audit[-1] if audit else None,
        }
    if not parsed["zero_mode"]["final_audit"]:
        raise SystemExit(f"missing zero-mode audit for N={n}")
    if parsed["zero_mode"]["final_audit"][0] != "PASS":
        raise SystemExit(f"zero-mode audit failed for N={n}")
    base_gpu = parsed["baseline_off"]["gpu_event_ms_per_step"]
    zero_gpu = parsed["zero_mode"]["gpu_event_ms_per_step"]
    base_wall = parsed["baseline_off"]["wall_s_per_step"]
    zero_wall = parsed["zero_mode"]["wall_s_per_step"]
    rows.append({
        "grid": [n, n, n],
        "steps": 192,
        "physical_dt_s": 0.8225917084910954,
        "physical_endpoint_s": 157.9376080302903,
        "baseline": parsed["baseline_off"],
        "zero_mode": parsed["zero_mode"],
        "gpu_event_overhead_fraction": zero_gpu / base_gpu - 1.0,
        "wall_overhead_fraction": zero_wall / base_wall - 1.0,
    })

summary = {
    "status": "PASS_PF_ZERO_MODE_ETA_1NM_4NM_PERFORMANCE_MATRIX_V1",
    "physics_contract": {
        "temperature_C": 400,
        "dx_nm": 1.0,
        "lambda_sm_nm": 4.0,
        "dt_code": 0.02,
        "t_real_unit_s": 41.12958542455477,
    },
    "rows": rows,
}
(root / "performance_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
PY

echo "PASS_PF_ZERO_MODE_ETA_1NM_4NM_PERFORMANCE_MATRIX_V1" \
  >"${OUT_ROOT}/status.txt"
