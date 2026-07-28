#!/usr/bin/env bash
# Pure-PF conserved-Y zero-mode qualification:
# build + host provenance test + OFF baseline + continuous/restart byte audit.

#SBATCH --job-name=pf_zm_q1
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
echo "PROJECT_ROOT=${PROJECT_ROOT}"
echo "OUT_ROOT=${OUT_ROOT}"
echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
echo "CUDA_ARCH=${CUDA_ARCH}"
nvidia-smi

make main_cuda CUDA_ARCH="${CUDA_ARCH}" CUDA_ROOT="${CUDA_ROOT}"
make test_pf_zero_mode_checkpoint CXX=c++
sha256sum main_cuda main_cuda.cu cuda_kernels.cu \
  pf_zero_mode_checkpoint.cpp pf_zero_mode_checkpoint.h \
  >"${OUT_ROOT}/source_binary_hashes.sha256"

export TEMP_C=400
export DT=0.02
export VF_INIT=0.01
export PHYS_DX_REF_M=1.0e-9
export PF_DX_M=1.0e-9
export PHYS_LAMBDA_SM_M=4.0e-9
export PHYS_GAMMA_JM2=0.2
export PHYS_L_REF_FACTOR=5.0
export PHYS_D_RATIO=0.01
PF_PARAM_FILE="$(site_generate_pf_param_file pf_zero_mode_q1_T400)"
cp "${PF_PARAM_FILE}" "${OUT_ROOT}/pf_input.params"
cp jobs/generated/physical_inputs_pf_zero_mode_q1_T400.json \
  "${OUT_ROOT}/physical_inputs.json"
cp jobs/generated/pf_payload_pf_zero_mode_q1_T400.json \
  "${OUT_ROOT}/converter_payload.json"

python3 - "${PF_PARAM_FILE}" "${OUT_ROOT}/unit_contract_audit.json" <<'PY'
import json
import math
import pathlib
import sys

params = {}
for raw in pathlib.Path(sys.argv[1]).read_text().splitlines():
    line = raw.strip()
    if not line or line.startswith("#") or "=" not in line:
        continue
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
failed = []
for key, target in expected.items():
    error = abs(actual[key] - target)
    tolerance = 1.0e-12 * max(abs(target), 1.0)
    if error > tolerance:
        failed.append(
            {"key": key, "actual": actual[key], "expected": target,
             "absolute_error": error, "tolerance": tolerance}
        )
audit = {
    "status": (
        "PASS_ETA_1NM_4NM_T400_CONVERTER_CONTRACT"
        if not failed else "FAIL_ETA_1NM_4NM_T400_CONVERTER_CONTRACT"
    ),
    "actual": actual,
    "expected": expected,
    "failures": failed,
}
pathlib.Path(sys.argv[2]).write_text(json.dumps(audit, indent=2) + "\n")
if failed:
    raise SystemExit(audit["status"])
print(audit["status"])
PY

COMMON=(
  64 64 64 "${DT}"
  --pf-param-file "${PF_PARAM_FILE}"
  --mode dynamics
  --radius 0.8
  --ic-23d-xB-out 0.03
)

run_case() {
  local label="$1"
  shift
  local stdout="${OUT_ROOT}/${label}.stdout.log"
  local stderr="${OUT_ROOT}/${label}.stderr.log"
  local results_root="${OUT_ROOT}/results/${label}"
  mkdir -p "${results_root}"
  set +e
  CUDA_STO_RESULTS_ROOT="${results_root}" \
    ./main_cuda "$@" >"${stdout}" 2>"${stderr}"
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

# Same PF physics, old Y k=0 handling, used only as the timing comparator.
run_case baseline_off \
  "${COMMON[@]:0:4}" 96 96 96 0 \
  "${COMMON[@]:4}" \
  --pf-zero-mode OFF \
  --init-case-tag pf_zm_baseline_off

# Uninterrupted reference to absolute step 96.
run_case zero_continuous \
  "${COMMON[@]:0:4}" 96 96 96 0 \
  "${COMMON[@]:4}" \
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 \
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 \
  --pf-zero-mode-tol-rel 1e-12 \
  --pf-zero-mode-max-iter 24 \
  --pf-checkpoint-every 48 \
  --pf-checkpoint-path "${OUT_ROOT}/continuous_final.chk" \
  --init-case-tag pf_zm_continuous

# First half creates the exact restart boundary.
run_case zero_first_half \
  "${COMMON[@]:0:4}" 48 48 48 0 \
  "${COMMON[@]:4}" \
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 \
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 \
  --pf-zero-mode-tol-rel 1e-12 \
  --pf-zero-mode-max-iter 24 \
  --pf-checkpoint-every 48 \
  --pf-checkpoint-path "${OUT_ROOT}/half.chk" \
  --init-case-tag pf_zm_first_half

# Restart reaches the same absolute step and writes a second final checkpoint.
run_case zero_restart \
  "${COMMON[@]:0:4}" 96 96 96 0 \
  "${COMMON[@]:4}" \
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 \
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 \
  --pf-zero-mode-tol-rel 1e-12 \
  --pf-zero-mode-max-iter 24 \
  --pf-restart-from "${OUT_ROOT}/half.chk" \
  --pf-checkpoint-every 48 \
  --pf-checkpoint-path "${OUT_ROOT}/restart_final.chk" \
  --init-case-tag pf_zm_restart

cmp "${OUT_ROOT}/continuous_final.chk" "${OUT_ROOT}/restart_final.chk"
sha256sum "${OUT_ROOT}/continuous_final.chk" \
  "${OUT_ROOT}/restart_final.chk" >"${OUT_ROOT}/restart_hashes.sha256"

python3 - "${OUT_ROOT}" <<'PY'
import csv
import json
import pathlib
import re
import sys

root = pathlib.Path(sys.argv[1])

def parse_run(label):
    text = (root / f"{label}.stdout.log").read_text()
    audit = re.findall(
        r"PF_ZERO_MODE_FINAL_AUDIT status=(\S+).*?"
        r"mean_mass_error=([+\-0-9.eE]+).*?"
        r"zero_mode_wall_s=([+\-0-9.eE]+).*?"
        r"checkpoint_wall_s=([+\-0-9.eE]+)",
        text,
    )
    perf = re.findall(r"平均每步墙钟时间:\s*([+\-0-9.eE]+) s", text)
    return {
        "label": label,
        "final_audit": audit[-1] if audit else None,
        "wall_s_per_step": float(perf[-1]) if perf else None,
        "gp_enabled_false": "gp_enabled=false" in text
            or "gp_paths_enabled=false" in text,
    }

summary = {
    "status": "PASS_PF_ZERO_MODE_CUDA_CONTINUOUS_RESTART_BYTEWISE_V1",
    "restart_bytewise_equal": True,
    "runs": [
        parse_run("baseline_off"),
        parse_run("zero_continuous"),
        parse_run("zero_first_half"),
        parse_run("zero_restart"),
    ],
}
base = summary["runs"][0]["wall_s_per_step"]
zero = summary["runs"][1]["wall_s_per_step"]
summary["zero_mode_overhead_fraction"] = (
    (zero / base - 1.0) if base and zero else None
)
if summary["runs"][1]["final_audit"] is None:
    raise SystemExit("missing zero-mode final audit")
if summary["runs"][1]["final_audit"][0] != "PASS":
    raise SystemExit("zero-mode final audit failed")
if not all(run["gp_enabled_false"] for run in summary["runs"][1:]):
    raise SystemExit("GP-off marker missing")
(root / "qualification_summary.json").write_text(
    json.dumps(summary, indent=2) + "\n"
)
PY

echo "PASS_PF_ZERO_MODE_CUDA_CONTINUOUS_RESTART_BYTEWISE_V1" \
  >"${OUT_ROOT}/status.txt"
