#!/usr/bin/env bash
set -euo pipefail

# Chained, sparse-checkpoint 6--8 h screening for one fixture.  Requires the
# exact PASS from the fixture's independent 256-step restart qualification.
# This runner never advances past registered step 7266.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
FIXTURE_ROOT="${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
STAGE7_ROOT="${STAGE7_ROOT:?STAGE7_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
PARAM_FILE="${PARAM_FILE:?PARAM_FILE is required}"
DT_CODE="0.02"
GRID_N=246
INITIAL_STATE_CLASS="MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
LIBRARY_SHA256="58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe"
ENDPOINTS=(512 1024 1536 2048 2560 3072 3584 4096 4608 5120 5632 6144 6656 7168 7266)

[[ ! -e "${RUN_ROOT}" ]] || {
  echo "[fatal] refusing to overwrite RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
}
grep -qx "PASS_246CUBE_SHORT_RESTART_AND_OBSERVABLES_V1" \
  "${STAGE7_ROOT}/status.txt"
for path in \
  "${SOURCE_ROOT}/main_cuda" \
  "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_6h8h_screening_v1.py" \
  "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${FIXTURE_ROOT}/phi.raw.f64" \
  "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
  "${FIXTURE_ROOT}/init_meta.json" \
  "${STAGE7_ROOT}/continuous/final.chk" \
  "${PARAM_FILE}"; do
  [[ -f "${path}" ]] || {
    echo "[fatal] missing required input: ${path}" >&2
    exit 2
  }
done

mkdir -p \
  "${RUN_ROOT}/provenance" \
  "${RUN_ROOT}/segments" \
  "${RUN_ROOT}/checkpoints"
exec >"${RUN_ROOT}/driver.stdout.log" 2>"${RUN_ROOT}/driver.stderr.log"
MONITOR_PID=""
cleanup() {
  local rc=$?
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ "${rc}" -ne 0 ]]; then
    printf 'BLOCKED_246CUBE_6H8H_SCREENING_DRIVER_V1\nexit_code=%s\n' "${rc}" \
      >"${RUN_ROOT}/status.txt"
  fi
}
trap cleanup EXIT

FIXTURE_SHA256="$(sha256sum "${FIXTURE_ROOT}/fixture_manifest.json" | awk '{print $1}')"
cp "${PARAM_FILE}" "${RUN_ROOT}/provenance/pf_input_dt0p02.params"
cp "${FIXTURE_ROOT}/init_meta.json" \
  "${RUN_ROOT}/provenance/init_meta_dt0p02.json"
sha256sum \
  "${SOURCE_ROOT}/main_cuda" \
  "${SOURCE_ROOT}/main_cuda.cu" \
  "${SOURCE_ROOT}/cuda_kernels.cu" \
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.cpp" \
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.h" \
  "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_6h8h_screening_v1.py" \
  "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
  "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${STAGE7_ROOT}/continuous/final.chk" \
  >"${RUN_ROOT}/provenance/input_hashes.sha256"

TRACKER_LINK_FLAGS=()
if [[ -n "${TRACKER_LDFLAGS:-}" ]]; then
  read -r -a TRACKER_LINK_FLAGS <<<"${TRACKER_LDFLAGS}"
fi
c++ -std=c++17 -O3 \
  "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  "${TRACKER_LINK_FLAGS[@]}" \
  -o "${RUN_ROOT}/provenance/analyze_pf_246cube_particle_lineage_v1"
sha256sum "${RUN_ROOT}/provenance/analyze_pf_246cube_particle_lineage_v1" \
  >"${RUN_ROOT}/provenance/analysis_binary.sha256"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -lms 1000 \
    >"${RUN_ROOT}/gpu_samples.csv" 2>/dev/null &
  MONITOR_PID="$!"
else
  : >"${RUN_ROOT}/gpu_samples.csv"
fi

IDENTITY_FLAGS=(
  --pf-initial-state-class "${INITIAL_STATE_CLASS}"
  --pf-fixture-manifest-sha256 "${FIXTURE_SHA256}"
  --pf-profile-library-manifest-sha256 "${LIBRARY_SHA256}"
)
ZERO_MODE_FLAGS=(
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1
  --pf-zero-mode-tol-rel 1e-12
  --pf-zero-mode-max-iter 24
)

CURRENT_CHECKPOINT="${STAGE7_ROOT}/continuous/final.chk"
CHECKPOINT_ARGS=(
  --checkpoint "${STAGE7_ROOT}/continuous/final.chk"
)
for endpoint in "${ENDPOINTS[@]}"; do
  SEGMENT="${RUN_ROOT}/segments/step_${endpoint}"
  CHECKPOINT="${RUN_ROOT}/checkpoints/step_${endpoint}.chk"
  mkdir -p "${SEGMENT}/results"
  CUDA_STO_RESULTS_ROOT="${SEGMENT}/results" \
    CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
    "${SOURCE_ROOT}/main_cuda" \
    "${GRID_N}" "${GRID_N}" "${GRID_N}" "${DT_CODE}" \
    "${endpoint}" "${endpoint}" 1 1 \
    --pf-param-file "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
    --mode dynamics \
    --pf-restart-from "${CURRENT_CHECKPOINT}" \
    "${ZERO_MODE_FLAGS[@]}" "${IDENTITY_FLAGS[@]}" \
    --enable-dynamics-mass-diagnostics \
    --dynamics-mass-diag-interval "${endpoint}" \
    --pf-checkpoint-every "${endpoint}" \
    --pf-checkpoint-path "${CHECKPOINT}" \
    --init-case-tag "pf_246cube_6h8h_step_${endpoint}_v1" \
    >"${SEGMENT}/stdout.log" 2>"${SEGMENT}/stderr.log"
  [[ ! -s "${SEGMENT}/stderr.log" ]]
  grep -q '^PF_ZERO_MODE_FINAL_AUDIT status=PASS ' \
    "${SEGMENT}/stdout.log"
  MASS_SOURCE="$(
    find "${SEGMENT}/results" -type f \
      -name dynamics_mass_diagnostics.csv -print -quit
  )"
  [[ -n "${MASS_SOURCE}" && -s "${MASS_SOURCE}" ]]
  cp "${MASS_SOURCE}" "${SEGMENT}/dynamics_mass_diagnostics.csv"
  CURRENT_CHECKPOINT="${CHECKPOINT}"
  CHECKPOINT_ARGS+=(--checkpoint "${CHECKPOINT}")
done

"${RUN_ROOT}/provenance/analyze_pf_246cube_particle_lineage_v1" \
  --initial-phi "${FIXTURE_ROOT}/phi.raw.f64" \
  --initial-xb "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
  "${CHECKPOINT_ARGS[@]}" \
  --out "${RUN_ROOT}/lineage" \
  --grid "${GRID_N}" --dx-nm 1 --threshold 1e-4 \
  --physical-dt-s 0.9909260953431841 --start-age-h 6 \
  --target-mean 0.03 --expected-initial-count 96 \
  --allow-dissolution

python3 "${SOURCE_ROOT}/scripts/audit_pf_246cube_6h8h_screening_v1.py" \
  --fixture-manifest "${FIXTURE_ROOT}/fixture_manifest.json" \
  --stage7-checkpoint "${STAGE7_ROOT}/continuous/final.chk" \
  --run-root "${RUN_ROOT}" \
  --lineage-root "${RUN_ROOT}/lineage" \
  --gpu-samples "${RUN_ROOT}/gpu_samples.csv" \
  --out "${RUN_ROOT}/audit"

grep -qx "PASS_246CUBE_6H8H_SHORT_SCREENING_V1" \
  "${RUN_ROOT}/audit/status.txt"
cp "${RUN_ROOT}/audit/status.txt" "${RUN_ROOT}/status.txt"
cat "${RUN_ROOT}/status.txt"
