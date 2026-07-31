#!/usr/bin/env bash
set -euo pipefail

# Numerical qualification only: identical 246^3 fixture and physics are run
# with the frozen fixed-iteration solver and the optional warm-start/residual
# solver.  Each nested run independently performs continuous/restart checks.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
FIXTURE_ROOT="${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
PARAM_FILE="${PARAM_FILE:?PARAM_FILE is required}"
BASELINE_ROOT_REUSE="${BASELINE_ROOT_REUSE:-}"

[[ ! -e "${RUN_ROOT}" ]] || {
  echo "[fatal] refusing to overwrite RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
}
for path in \
  "${SOURCE_ROOT}/main_cuda" \
  "${SOURCE_ROOT}/jobs/run_pf_246cube_short_restart_v1.sh" \
  "${SOURCE_ROOT}/scripts/audit_pf_elastic_warm_start_residual_v1.py" \
  "${SOURCE_ROOT}/scripts/analyze_pf_zero_mode_checkpoints.py" \
  "${PARAM_FILE}"; do
  [[ -f "${path}" ]] || {
    echo "[fatal] missing qualification input: ${path}" >&2
    exit 2
  }
done

mkdir -p "${RUN_ROOT}/provenance"
exec >"${RUN_ROOT}/driver.stdout.log" 2>"${RUN_ROOT}/driver.stderr.log"
trap 'rc=$?; if [[ "${rc}" -ne 0 ]]; then printf "BLOCKED_ELASTIC_WARM_START_RESIDUAL_V1\\nexit_code=%s\\n" "${rc}" >"${RUN_ROOT}/status.txt"; fi' EXIT

BASE_PARAM="${RUN_ROOT}/provenance/fixed_iteration.params"
ACCEL_PARAM="${RUN_ROOT}/provenance/warm_start_residual.params"
cp "${PARAM_FILE}" "${BASE_PARAM}"
cp "${PARAM_FILE}" "${ACCEL_PARAM}"
printf '\nelastic_warm_start_enabled = 0\nelastic_residual_control_enabled = 0\nelastic_iter_max = 20\n' \
  >>"${BASE_PARAM}"
printf '%s\n' \
  '' \
  'elastic_warm_start_enabled = 1' \
  'elastic_residual_control_enabled = 1' \
  'elastic_iter_min = 2' \
  'elastic_iter_max = 32' \
  'elastic_residual_tolerance = 1.0e-6' \
  'elastic_residual_absolute_floor = 1.0e-30' \
  'elastic_fail_on_nonconvergence = 1' \
  >>"${ACCEL_PARAM}"

sha256sum \
  "${SOURCE_ROOT}/main_cuda" \
  "${SOURCE_ROOT}/main_cuda.cu" \
  "${SOURCE_ROOT}/cuda_kernels.cu" \
  "${SOURCE_ROOT}/cuda_kernels.h" \
  "${SOURCE_ROOT}/pf_params.h" \
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.cpp" \
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.h" \
  "${SOURCE_ROOT}/jobs/run_pf_246cube_short_restart_v1.sh" \
  "${SOURCE_ROOT}/scripts/audit_pf_elastic_warm_start_residual_v1.py" \
  "${SOURCE_ROOT}/scripts/analyze_pf_zero_mode_checkpoints.py" \
  "${BASE_PARAM}" "${ACCEL_PARAM}" \
  >"${RUN_ROOT}/provenance/input_hashes.sha256"

if [[ -n "${BASELINE_ROOT_REUSE}" ]]; then
  grep -qx "PASS_246CUBE_SHORT_RESTART_AND_OBSERVABLES_V1" \
    "${BASELINE_ROOT_REUSE}/status.txt"
  current_binary_sha="$(sha256sum "${SOURCE_ROOT}/main_cuda" | awk '{print $1}')"
  baseline_binary_sha="$(
    awk '$2 ~ /main_cuda$/ {print $1; exit}' \
      "${BASELINE_ROOT_REUSE}/provenance/input_hashes.sha256"
  )"
  [[ -n "${baseline_binary_sha}" &&
      "${current_binary_sha}" == "${baseline_binary_sha}" ]] || {
    echo "[fatal] reused baseline binary hash mismatch" >&2
    exit 2
  }
  BASELINE_ROOT="${BASELINE_ROOT_REUSE}"
  printf 'baseline_reused=true\nbaseline_root=%s\nbaseline_binary_sha256=%s\n' \
    "${BASELINE_ROOT}" "${baseline_binary_sha}" \
    >"${RUN_ROOT}/provenance/baseline_reuse.txt"
  sha256sum \
    "${BASELINE_ROOT}/status.txt" \
    "${BASELINE_ROOT}/continuous/final.chk" \
    "${BASELINE_ROOT}/restart_second_half/final.chk" \
    "${BASELINE_ROOT}/audit/audit.json" \
    >"${RUN_ROOT}/provenance/baseline_reuse_hashes.sha256"
else
  BASELINE_ROOT="${RUN_ROOT}/baseline_fixed20"
  env \
    SOURCE_ROOT="${SOURCE_ROOT}" \
    FIXTURE_ROOT="${FIXTURE_ROOT}" \
    RUN_ROOT="${BASELINE_ROOT}" \
    PARAM_FILE="${BASE_PARAM}" \
    TRACKER_LDFLAGS="${TRACKER_LDFLAGS:-}" \
    bash "${SOURCE_ROOT}/jobs/run_pf_246cube_short_restart_v1.sh"
fi

env \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  FIXTURE_ROOT="${FIXTURE_ROOT}" \
  RUN_ROOT="${RUN_ROOT}/warm_start_residual" \
  PARAM_FILE="${ACCEL_PARAM}" \
  TRACKER_LDFLAGS="${TRACKER_LDFLAGS:-}" \
  bash "${SOURCE_ROOT}/jobs/run_pf_246cube_short_restart_v1.sh"

python3 "${SOURCE_ROOT}/scripts/audit_pf_elastic_warm_start_residual_v1.py" \
  --source-root "${SOURCE_ROOT}" \
  --baseline-root "${BASELINE_ROOT}" \
  --accelerated-root "${RUN_ROOT}/warm_start_residual" \
  --out "${RUN_ROOT}/audit"

grep -qx "PASS_ELASTIC_WARM_START_RESIDUAL_V1" \
  "${RUN_ROOT}/audit/status.txt"
cp "${RUN_ROOT}/audit/status.txt" "${RUN_ROOT}/status.txt"
cat "${RUN_ROOT}/audit/final_terminal_output.txt"
