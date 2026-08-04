#!/usr/bin/env bash
#SBATCH --job-name=pf400_v2_rep24
#SBATCH --partition=gpu_vip_24h
#SBATCH --qos=gpu_vip_24h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Accepted-field mechanics replay for a new PF input to the frozen transport-V2
# equations.  The source checkpoints and fixture are immutable and read-only.

set -euo pipefail
: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${PRODUCTION_ROOT:?PRODUCTION_ROOT is required}"
: "${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
: "${REPLAY_ROOT:?REPLAY_ROOT is required}"
[[ ! -e "${REPLAY_ROOT}" ]] || { echo "[fatal] refusing to overwrite REPLAY_ROOT" >&2; exit 2; }
mkdir -p "${REPLAY_ROOT}/provenance"
exec >"${REPLAY_ROOT}/driver.stdout.log" 2>"${REPLAY_ROOT}/driver.stderr.log"
trap 'rc=$?; if [[ "${rc}" -ne 0 ]]; then printf "BLOCKED_400CUBE_V2_INTERIM_24H_ACCEPTED_FIELD_REPLAY_V1\nexit_code=%s\n" "${rc}" >"${REPLAY_ROOT}/status.txt"; fi' EXIT
printf 'RUNNING_400CUBE_V2_INTERIM_24H_ACCEPTED_FIELD_REPLAY_V1\n' >"${REPLAY_ROOT}/status.txt"

param="${PRODUCTION_ROOT}/provenance/pf_input_dt0p02.params"
checkpoint="${PRODUCTION_ROOT}/checkpoints/step_65393.chk"
for path in "${SOURCE_ROOT}/main_cuda" "${SOURCE_ROOT}/main_cuda.cu" \
  "${SOURCE_ROOT}/cuda_kernels.cu" "${SOURCE_ROOT}/pf_zero_mode_checkpoint.cpp" \
  "${param}" "${checkpoint}" "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${FIXTURE_ROOT}/phi.raw.f64" "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
  "${FIXTURE_ROOT}/init_meta.json"; do
  [[ -f "${path}" ]] || { echo "[fatal] missing input: ${path}" >&2; exit 2; }
done

fixture_sha="$(sha256sum "${FIXTURE_ROOT}/fixture_manifest.json" | awk '{print $1}')"
expected_fixture_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fixture_manifest_sha256"])' "${PRODUCTION_ROOT}/provenance/campaign_manifest.json")"
[[ "${fixture_sha}" == "${expected_fixture_sha}" ]]
library_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["profile_library_manifest_sha256"])' "${FIXTURE_ROOT}/fixture_manifest.json")"
expected_checkpoint_sha="$(awk '{print $1}' "${PRODUCTION_ROOT}/segments/step_65393/checkpoint.sha256")"
actual_checkpoint_sha="$(sha256sum "${checkpoint}" | awk '{print $1}')"
[[ "${actual_checkpoint_sha}" == "${expected_checkpoint_sha}" ]]

cp "${param}" "${REPLAY_ROOT}/provenance/production_physics.params"
cp "${PRODUCTION_ROOT}/provenance/campaign_manifest.json" "${REPLAY_ROOT}/provenance/"
sha256sum "${SOURCE_ROOT}/main_cuda" "${SOURCE_ROOT}/main_cuda.cu" \
  "${SOURCE_ROOT}/cuda_kernels.cu" "${SOURCE_ROOT}/pf_zero_mode_checkpoint.cpp" \
  "${param}" "${checkpoint}" "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${FIXTURE_ROOT}/phi.raw.f64" "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
  "${FIXTURE_ROOT}/init_meta.json" >"${REPLAY_ROOT}/provenance/input_hashes.sha256"
{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-UNSCHEDULED}"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader,nounits || true
} >"${REPLAY_ROOT}/provenance/device_identity.txt"

identity=(
  --pf-initial-state-class MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1
  --pf-fixture-manifest-sha256 "${fixture_sha}"
  --pf-profile-library-manifest-sha256 "${library_sha}"
)
zero_mode=(
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1
  --pf-zero-mode-tol-rel 1e-12 --pf-zero-mode-max-iter 24
)

age_root="${REPLAY_ROOT}/age_6h"
mkdir -p "${age_root}/results"
CUDA_STO_RESULTS_ROOT="${age_root}/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
  "${SOURCE_ROOT}/main_cuda" 400 400 400 0.02 1 1 1 1 \
  --pf-param-file "${REPLAY_ROOT}/provenance/production_physics.params" \
  --mode dynamics --init-mode raw_fields \
  --init-phi-raw "${FIXTURE_ROOT}/phi.raw.f64" \
  --init-xB-raw "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
  --init-meta "${FIXTURE_ROOT}/init_meta.json" \
  "${zero_mode[@]}" "${identity[@]}" --pf-checkpoint-every 0 \
  --mechanics-sync-diagnostic-dir "${age_root}/fields" \
  --mechanics-sync-diagnostic-step 1 \
  --init-case-tag mechanics_v2_input_400cube_6h \
  >"${age_root}/stdout.log" 2>"${age_root}/stderr.log"
[[ ! -s "${age_root}/stderr.log" ]]
grep -q '^MECHANICS_DIAGNOSTIC_STOP_BEFORE_PF_UPDATE ' "${age_root}/stdout.log"

age_root="${REPLAY_ROOT}/age_24h"
mkdir -p "${age_root}/results"
CUDA_STO_RESULTS_ROOT="${age_root}/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
  "${SOURCE_ROOT}/main_cuda" 400 400 400 0.02 65394 65394 1 1 \
  --pf-param-file "${REPLAY_ROOT}/provenance/production_physics.params" \
  --mode dynamics --pf-restart-from "${checkpoint}" \
  "${zero_mode[@]}" "${identity[@]}" --pf-checkpoint-every 0 \
  --mechanics-only-replay-dir "${age_root}/fields" \
  --mechanics-replay-initialization checkpoint_warm \
  --init-case-tag mechanics_v2_input_400cube_24h \
  >"${age_root}/stdout.log" 2>"${age_root}/stderr.log"
[[ ! -s "${age_root}/stderr.log" ]]
grep -q '^MECHANICS_DIAGNOSTIC_STOP_BEFORE_PF_UPDATE ' "${age_root}/stdout.log"

find "${REPLAY_ROOT}" -path '*/fields/*' -type f -print0 | sort -z | xargs -0 sha256sum \
  >"${REPLAY_ROOT}/provenance/replay_field_hashes.sha256"
printf 'PASS_400CUBE_V2_INTERIM_24H_ACCEPTED_FIELD_REPLAY_V1\n' >"${REPLAY_ROOT}/status.txt"
cat "${REPLAY_ROOT}/status.txt"
