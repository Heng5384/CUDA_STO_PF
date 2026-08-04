#!/usr/bin/env bash
set -euo pipefail

# Read-only mechanics replay for one registered Method-1 replicate.  The 6 h
# field is the immutable initial fixture; later ages are exact V4 checkpoints.

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${INPUT_ROOT:?INPUT_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${PARAM_FILE:?PARAM_FILE is required}"
: "${AUTHORITY_AUDIT:?AUTHORITY_AUDIT is required}"
: "${REGISTERED_OBSERVABLES:?REGISTERED_OBSERVABLES is required}"
: "${REPLICATE:?REPLICATE is required}"
LIBRARY_SHA256="${LIBRARY_SHA256:-de4142e0268e379f70fd1c860aab9e004421d07df4f8d872f4eaeb3dbef7af5b}"
GRID_N=246

[[ "${REPLICATE}" =~ ^[ABC]$ ]] || { echo "[fatal] REPLICATE must be A, B, or C" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "[fatal] refusing to overwrite RUN_ROOT: ${RUN_ROOT}" >&2; exit 2; }
for input in \
  "${SOURCE_ROOT}/main_cuda" "${SOURCE_ROOT}/main_cuda.cu" \
  "${SOURCE_ROOT}/cuda_kernels.cu" \
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.cpp" \
  "${PARAM_FILE}" "${AUTHORITY_AUDIT}" "${REGISTERED_OBSERVABLES}" \
  "${INPUT_ROOT}/phi.raw.f64" "${INPUT_ROOT}/xB_alpha.raw.f64" \
  "${INPUT_ROOT}/init_meta.json" "${INPUT_ROOT}/fixture_manifest.json"; do
  [[ -f "${input}" ]] || { echo "[fatal] missing input: ${input}" >&2; exit 2; }
done

mkdir -p "${RUN_ROOT}/provenance"
exec >"${RUN_ROOT}/driver.stdout.log" 2>"${RUN_ROOT}/driver.stderr.log"
trap 'rc=$?; if [[ "${rc}" -ne 0 ]]; then printf "BLOCKED_HISTORICAL_MECHANICS_ACCEPTED_FIELD_REPLAY_V1\nexit_code=%s\n" "${rc}" >"${RUN_ROOT}/status.txt"; fi' EXIT
printf 'RUNNING_HISTORICAL_MECHANICS_ACCEPTED_FIELD_REPLAY_V1\n' >"${RUN_ROOT}/status.txt"

cp "${PARAM_FILE}" "${RUN_ROOT}/provenance/production_physics.params"
cp "${AUTHORITY_AUDIT}" "${RUN_ROOT}/provenance/authority_audit.json"
cp "${REGISTERED_OBSERVABLES}" "${RUN_ROOT}/provenance/registered_observables.csv"
fixture_sha="$(sha256sum "${INPUT_ROOT}/fixture_manifest.json" | awk '{print $1}')"
audit_fixture_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fixture_manifest_sha256"])' "${AUTHORITY_AUDIT}")"
[[ "${fixture_sha}" == "${audit_fixture_sha}" ]]

sha256sum \
  "${SOURCE_ROOT}/main_cuda" "${SOURCE_ROOT}/main_cuda.cu" \
  "${SOURCE_ROOT}/cuda_kernels.cu" \
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.cpp" \
  "${RUN_ROOT}/provenance/production_physics.params" \
  "${RUN_ROOT}/provenance/authority_audit.json" \
  "${RUN_ROOT}/provenance/registered_observables.csv" \
  "${INPUT_ROOT}/fixture_manifest.json" \
  "${INPUT_ROOT}/phi.raw.f64" "${INPUT_ROOT}/xB_alpha.raw.f64" \
  "${INPUT_ROOT}/init_meta.json" \
  >"${RUN_ROOT}/provenance/input_hashes.sha256"
{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'replicate=%s\n' "${REPLICATE}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-NATIVE_WORKSTATION}"
  nvidia-smi --query-gpu=index,name,uuid,memory.total \
    --format=csv,noheader,nounits 2>/dev/null || true
} >"${RUN_ROOT}/provenance/device_identity.txt"

identity=(
  --pf-initial-state-class MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1
  --pf-fixture-manifest-sha256 "${fixture_sha}"
  --pf-profile-library-manifest-sha256 "${LIBRARY_SHA256}"
)
zero_mode=(
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1
  --pf-zero-mode-tol-rel 1e-12
  --pf-zero-mode-max-iter 24
)

# Registered 6 h authority is the exact initial fixture (step 0), for which no
# V4 checkpoint exists.  Solve mechanics and stop before the first PF update.
age_root="${RUN_ROOT}/age_6h"
mkdir -p "${age_root}/results"
CUDA_STO_RESULTS_ROOT="${age_root}/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
  "${SOURCE_ROOT}/main_cuda" \
  "${GRID_N}" "${GRID_N}" "${GRID_N}" 0.02 1 1 1 1 \
  --pf-param-file "${RUN_ROOT}/provenance/production_physics.params" \
  --mode dynamics \
  --init-mode raw_fields \
  --init-phi-raw "${INPUT_ROOT}/phi.raw.f64" \
  --init-xB-raw "${INPUT_ROOT}/xB_alpha.raw.f64" \
  --init-meta "${INPUT_ROOT}/init_meta.json" \
  "${zero_mode[@]}" "${identity[@]}" \
  --pf-checkpoint-every 0 \
  --mechanics-sync-diagnostic-dir "${age_root}/fields" \
  --mechanics-sync-diagnostic-step 1 \
  --init-case-tag "mechanics_replay_${REPLICATE}_6h_v1" \
  >"${age_root}/stdout.log" 2>"${age_root}/stderr.log"
[[ ! -s "${age_root}/stderr.log" ]]
grep -q '^MECHANICS_DIAGNOSTIC_STOP_BEFORE_PF_UPDATE ' "${age_root}/stdout.log"

ages=(12 18 24 36 48)
steps=(21798 43596 65393 108989 152585)
for index in "${!ages[@]}"; do
  age="${ages[$index]}"
  step="${steps[$index]}"
  checkpoint="${INPUT_ROOT}/checkpoints/step_${step}.chk"
  [[ -f "${checkpoint}" ]]
  expected_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["checkpoint_hashes"][sys.argv[2]])' "${AUTHORITY_AUDIT}" "${step}")"
  actual_sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
  [[ "${actual_sha}" == "${expected_sha}" ]] || {
    echo "[fatal] checkpoint hash mismatch replicate=${REPLICATE} step=${step}" >&2
    exit 2
  }
  age_root="${RUN_ROOT}/age_${age}h"
  mkdir -p "${age_root}/results"
  CUDA_STO_RESULTS_ROOT="${age_root}/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
    "${SOURCE_ROOT}/main_cuda" \
    "${GRID_N}" "${GRID_N}" "${GRID_N}" 0.02 "$((step + 1))" "$((step + 1))" 1 1 \
    --pf-param-file "${RUN_ROOT}/provenance/production_physics.params" \
    --mode dynamics --pf-restart-from "${checkpoint}" \
    "${zero_mode[@]}" "${identity[@]}" \
    --pf-checkpoint-every 0 \
    --mechanics-only-replay-dir "${age_root}/fields" \
    --mechanics-replay-initialization checkpoint_warm \
    --init-case-tag "mechanics_replay_${REPLICATE}_${age}h_v1" \
    >"${age_root}/stdout.log" 2>"${age_root}/stderr.log"
  [[ ! -s "${age_root}/stderr.log" ]]
  grep -q '^MECHANICS_DIAGNOSTIC_STOP_BEFORE_PF_UPDATE ' "${age_root}/stdout.log"
  printf '%s  %s\n' "${actual_sha}" "${checkpoint}" \
    >>"${RUN_ROOT}/provenance/selected_checkpoint_hashes.sha256"
done

find "${RUN_ROOT}" -path '*/fields/*' -type f -print0 | sort -z | xargs -0 sha256sum \
  >"${RUN_ROOT}/provenance/replay_field_hashes.sha256"
printf 'PASS_HISTORICAL_MECHANICS_ACCEPTED_FIELD_REPLAY_V1\n' >"${RUN_ROOT}/status.txt"
cat "${RUN_ROOT}/status.txt"
