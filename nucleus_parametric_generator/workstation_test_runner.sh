#!/usr/bin/env bash
set -euo pipefail

REMOTE_HOST="${REMOTE_HOST:-cluster-direct}"
REMOTE_ROOT="${REMOTE_ROOT:-/data/home/luozhiheng/CUDA_STO_PF}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-${REMOTE_ROOT}/nucleus_parametric_generator/workstation_validation}"
SUMMARY_PATH="${SUMMARY_PATH:-}"
XB_OUT="${XB_OUT:-0.03}"
N_STEPS="${N_STEPS:-1}"
DT="${DT:-0.001}"
VALIDATION_N="${VALIDATION_N:-300}"

echo "[runner] host=${REMOTE_HOST}"
echo "[runner] root=${REMOTE_ROOT}"

ssh "${REMOTE_HOST}" "mkdir -p '${REMOTE_ROOT}/nucleus_parametric_generator' '${REMOTE_WORKDIR}' '${REMOTE_ROOT}/jobs/generated'"
rsync -a nucleus_parametric_generator/ "${REMOTE_HOST}:${REMOTE_ROOT}/nucleus_parametric_generator/"

ssh "${REMOTE_HOST}" "cd '${REMOTE_ROOT}' && REMOTE_ROOT='${REMOTE_ROOT}' REMOTE_WORKDIR='${REMOTE_WORKDIR}' SUMMARY_PATH='${SUMMARY_PATH}' XB_OUT='${XB_OUT}' N_STEPS='${N_STEPS}' DT='${DT}' VALIDATION_N='${VALIDATION_N}' bash -s" <<'REMOTE_SCRIPT'
set -euo pipefail

REMOTE_ROOT="${REMOTE_ROOT:-/data/home/luozhiheng/CUDA_STO_PF}"
REMOTE_WORKDIR="${REMOTE_WORKDIR:-${REMOTE_ROOT}/nucleus_parametric_generator/workstation_validation}"
SUMMARY_PATH="${SUMMARY_PATH:-}"
XB_OUT="${XB_OUT:-0.03}"
N_STEPS="${N_STEPS:-1}"
DT="${DT:-0.001}"
VALIDATION_N="${VALIDATION_N:-300}"

mkdir -p "${REMOTE_WORKDIR}"

if [[ -z "${SUMMARY_PATH}" ]]; then
  SUMMARY_PATH="$(find Results/workflows -path '*/continue_dyn_1/summary.txt' | sort | head -n 1)"
fi
if [[ -z "${SUMMARY_PATH}" || ! -f "${SUMMARY_PATH}" ]]; then
  echo "{\"accepted\": false, \"failure\": \"no_dynamic_continue_summary_found\"}" > "${REMOTE_WORKDIR}/acceptance_summary.json"
  exit 1
fi
PF_PARAM_FILE="${PF_PARAM_FILE:-$(dirname "${SUMMARY_PATH}")/pf_input.params}"

echo "[remote] summary=${SUMMARY_PATH}"
echo "[remote] pf_param_file=${PF_PARAM_FILE}"

if command -v nvcc >/dev/null 2>&1; then
  nvcc -O3 -std=c++11 -x cu nucleus_parametric_generator/parametric_nucleus_builder.cu -o "${REMOTE_WORKDIR}/parametric_nucleus_builder"
else
  g++ -O3 -std=c++11 -x c++ nucleus_parametric_generator/parametric_nucleus_builder.cu -o "${REMOTE_WORKDIR}/parametric_nucleus_builder"
fi
g++ -O3 -std=c++11 nucleus_parametric_generator/dynamic_continue_parser.cpp -o "${REMOTE_WORKDIR}/dynamic_continue_parser"

"${REMOTE_WORKDIR}/dynamic_continue_parser" "${SUMMARY_PATH}" "${REMOTE_WORKDIR}/parametric_descriptor.json" \
  --xB "${XB_OUT}" \
  --override-grid "${VALIDATION_N}" "${VALIDATION_N}" "${VALIDATION_N}" \
  --center-mode grid_center
"${REMOTE_WORKDIR}/parametric_nucleus_builder" "${REMOTE_WORKDIR}/parametric_descriptor.json" "${REMOTE_WORKDIR}" "generated_before_runtime"

runtime_status="not_run"
runtime_pass=false
if [[ -x ./main_cuda ]]; then
  pf_args=()
  if [[ -f "${PF_PARAM_FILE}" ]]; then
    pf_args=(--pf-param-file "${PF_PARAM_FILE}")
  fi
  set +e
  ./main_cuda \
    "${pf_args[@]}" \
    --mode=dynamics \
    --Nx "${VALIDATION_N}" \
    --Ny "${VALIDATION_N}" \
    --Nz "${VALIDATION_N}" \
    --init-mode raw_fields \
    --init-phi-raw "${REMOTE_WORKDIR}/phi_init.raw" \
    --init-xB-raw "${REMOTE_WORKDIR}/xB_init.raw" \
    --init-meta "${REMOTE_WORKDIR}/init_meta.json" \
    --nsteps "${N_STEPS}" \
    --dt "${DT}" \
    --out-every "${N_STEPS}" \
    --csv-out-every 1 \
    --output-dir "${REMOTE_WORKDIR}/pf_smoke" \
    > "${REMOTE_WORKDIR}/pf_smoke.out" 2> "${REMOTE_WORKDIR}/pf_smoke.err"
  code=$?
  set -e
  if grep -Eiq 'CUDA driver version is insufficient for CUDA runtime version' "${REMOTE_WORKDIR}/pf_smoke.err"; then
    runtime_status="blocked_cuda_driver_version"
  elif [[ "${code}" -eq 0 ]] && ! grep -Eiq 'nan|inf|overflow|fatal' "${REMOTE_WORKDIR}/pf_smoke.out" "${REMOTE_WORKDIR}/pf_smoke.err"; then
    runtime_status="passed_1_step_raw_fields"
    runtime_pass=true
  else
    runtime_status="failed_code_${code}"
  fi
else
  runtime_status="main_cuda_missing"
fi

"${REMOTE_WORKDIR}/parametric_nucleus_builder" "${REMOTE_WORKDIR}/parametric_descriptor.json" "${REMOTE_WORKDIR}" "${runtime_status}" || true

python3 - <<'PY'
from pathlib import Path
import json

root = Path("nucleus_parametric_generator/workstation_validation")
acc_path = root / "acceptance_summary.json"
data = json.loads(acc_path.read_text())
runtime_status = data.get("runtime_status", "unknown")
runtime_pass = runtime_status.startswith("passed")
accepted = bool(data.get("accepted")) and runtime_pass
data["workstation_runtime_passed"] = runtime_pass
data["accepted"] = accepted
data["acceptance_policy"] = "generator acceptance AND main_cuda raw_fields 1-step smoke pass"
acc_path.write_text(json.dumps(data, indent=2))

report = root / "validation_report.md"
report.write_text(
    "# Workstation Parametric Nucleus Validation\n\n"
    f"- descriptor: `{root / 'parametric_descriptor.json'}`\n"
    f"- raw bundle: `{root}`\n"
    f"- runtime_status: `{runtime_status}`\n"
    f"- accepted: `{accepted}`\n"
    f"- mass_error: `{data.get('mass_error')}`\n"
    f"- phi_max: `{data.get('phi_max')}`\n"
    f"- connected_components: `{data.get('connected_components')}`\n",
    encoding="utf-8",
)
print(json.dumps({
    "acceptance_summary": str(acc_path),
    "validation_report": str(report),
    "accepted": accepted,
    "runtime_status": runtime_status,
}, indent=2))
PY
REMOTE_SCRIPT

rsync -a "${REMOTE_HOST}:${REMOTE_WORKDIR}/acceptance_summary.json" nucleus_parametric_generator/acceptance_summary.json
rsync -a "${REMOTE_HOST}:${REMOTE_WORKDIR}/validation_report.md" nucleus_parametric_generator/validation_report.md

echo "workstation_validation_complete"
