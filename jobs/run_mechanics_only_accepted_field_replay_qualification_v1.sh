#!/usr/bin/env bash
set -euo pipefail

# Extremely short synchronized online/offline qualification.  This creates a
# fresh 32^3 diagnostic trajectory and never reads or writes a production run.

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${PARAM_FILE:?PARAM_FILE is required}"
GRID_N="${GRID_N:-32}"

[[ ! -e "${RUN_ROOT}" ]] || {
  echo "[fatal] refusing to overwrite RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
}
for input in \
  "${SOURCE_ROOT}/main_cuda" \
  "${SOURCE_ROOT}/main_cuda.cu" \
  "${SOURCE_ROOT}/cuda_kernels.cu" \
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.cpp" \
  "${SOURCE_ROOT}/scripts/qualify_mechanics_only_accepted_field_replay_v1.py" \
  "${PARAM_FILE}"; do
  [[ -f "${input}" ]] || { echo "[fatal] missing input: ${input}" >&2; exit 2; }
done

mkdir -p "${RUN_ROOT}/provenance" \
  "${RUN_ROOT}/online_results" \
  "${RUN_ROOT}/warm_replay_results" \
  "${RUN_ROOT}/zero_replay_results"
exec >"${RUN_ROOT}/driver.stdout.log" 2>"${RUN_ROOT}/driver.stderr.log"
trap 'rc=$?; if [[ "${rc}" -ne 0 ]]; then printf "BLOCKED_MECHANICS_ONLY_ACCEPTED_FIELD_REPLAY_V1\nexit_code=%s\n" "${rc}" >"${RUN_ROOT}/status.txt"; fi' EXIT
printf 'RUNNING_MECHANICS_ONLY_ACCEPTED_FIELD_REPLAY_V1\n' >"${RUN_ROOT}/status.txt"

cp "${PARAM_FILE}" "${RUN_ROOT}/provenance/production_physics.params"
sha256sum \
  "${SOURCE_ROOT}/main_cuda" \
  "${SOURCE_ROOT}/main_cuda.cu" \
  "${SOURCE_ROOT}/cuda_kernels.cu" \
  "${SOURCE_ROOT}/cuda_kernels.h" \
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.cpp" \
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.h" \
  "${SOURCE_ROOT}/scripts/qualify_mechanics_only_accepted_field_replay_v1.py" \
  "${RUN_ROOT}/provenance/production_physics.params" \
  >"${RUN_ROOT}/provenance/input_hashes.sha256"
{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-UNSCHEDULED}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-UNSET}"
  nvidia-smi --query-gpu=index,name,uuid,memory.total \
    --format=csv,noheader,nounits 2>/dev/null || true
} >"${RUN_ROOT}/provenance/device_identity.txt"

common=(
  "${GRID_N}" "${GRID_N}" "${GRID_N}" 0.02 2 2 1 1
  --pf-param-file "${RUN_ROOT}/provenance/production_physics.params"
  --mode dynamics
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1
  --pf-zero-mode-tol-rel 1e-12
  --pf-zero-mode-max-iter 24
)

checkpoint="${RUN_ROOT}/accepted_step_1.chk"
CUDA_STO_RESULTS_ROOT="${RUN_ROOT}/online_results" \
CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
  "${SOURCE_ROOT}/main_cuda" "${common[@]}" \
  --pf-checkpoint-every 1 \
  --pf-checkpoint-path "${checkpoint}" \
  --mechanics-sync-diagnostic-dir "${RUN_ROOT}/online_synchronized" \
  --mechanics-sync-diagnostic-step 2 \
  --init-case-tag mechanics_online_v1 \
  >"${RUN_ROOT}/online.stdout.log" 2>"${RUN_ROOT}/online.stderr.log"
[[ -s "${checkpoint}" && ! -s "${RUN_ROOT}/online.stderr.log" ]]
grep -q '^MECHANICS_DIAGNOSTIC_STOP_BEFORE_PF_UPDATE ' \
  "${RUN_ROOT}/online.stdout.log"

CUDA_STO_RESULTS_ROOT="${RUN_ROOT}/warm_replay_results" \
CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
  "${SOURCE_ROOT}/main_cuda" "${common[@]}" \
  --pf-restart-from "${checkpoint}" \
  --pf-checkpoint-every 0 \
  --mechanics-only-replay-dir "${RUN_ROOT}/warm_replay" \
  --mechanics-replay-initialization checkpoint_warm \
  --init-case-tag mechanics_warm_replay_v1 \
  >"${RUN_ROOT}/warm_replay.stdout.log" \
  2>"${RUN_ROOT}/warm_replay.stderr.log"
[[ ! -s "${RUN_ROOT}/warm_replay.stderr.log" ]]

CUDA_STO_RESULTS_ROOT="${RUN_ROOT}/zero_replay_results" \
CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
  "${SOURCE_ROOT}/main_cuda" "${common[@]}" \
  --pf-restart-from "${checkpoint}" \
  --pf-checkpoint-every 0 \
  --mechanics-only-replay-dir "${RUN_ROOT}/zero_replay" \
  --mechanics-replay-initialization zero \
  --init-case-tag mechanics_zero_replay_v1 \
  >"${RUN_ROOT}/zero_replay.stdout.log" \
  2>"${RUN_ROOT}/zero_replay.stderr.log"
[[ ! -s "${RUN_ROOT}/zero_replay.stderr.log" ]]

python3 "${SOURCE_ROOT}/scripts/qualify_mechanics_only_accepted_field_replay_v1.py" \
  --online "${RUN_ROOT}/online_synchronized" \
  --warm-replay "${RUN_ROOT}/warm_replay" \
  --zero-replay "${RUN_ROOT}/zero_replay" \
  --out "${RUN_ROOT}/qualification" \
  >"${RUN_ROOT}/qualification_terminal.json"
grep -qx 'PASS_MECHANICS_ONLY_ACCEPTED_FIELD_REPLAY_V1' \
  "${RUN_ROOT}/qualification/status.txt"

sha256sum "${checkpoint}" \
  "${RUN_ROOT}/online_synchronized/accepted_phi.raw.f64" \
  "${RUN_ROOT}/online_synchronized/accepted_xB.raw.f64" \
  "${RUN_ROOT}/online_synchronized/replay_summary.txt" \
  "${RUN_ROOT}/warm_replay/replay_summary.txt" \
  "${RUN_ROOT}/zero_replay/replay_summary.txt" \
  "${RUN_ROOT}/qualification/audit.json" \
  >"${RUN_ROOT}/provenance/output_hashes.sha256"
cp "${RUN_ROOT}/qualification/status.txt" "${RUN_ROOT}/status.txt"
cat "${RUN_ROOT}/status.txt"
