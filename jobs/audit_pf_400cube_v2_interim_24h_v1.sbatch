#!/usr/bin/env bash
#SBATCH --job-name=pf400_v2_24h
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Isolated analysis of immutable checkpoints.  This script never writes below
# PRODUCTION_ROOT or FIXTURE_ROOT and cannot submit, resume, or cancel PF jobs.

set -euo pipefail
: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${PRODUCTION_ROOT:?PRODUCTION_ROOT is required}"
: "${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
: "${AUDIT_ROOT:?AUDIT_ROOT is required}"
[[ ! -e "${AUDIT_ROOT}" ]] || { echo "[fatal] refusing to overwrite AUDIT_ROOT" >&2; exit 2; }
for path in \
  "${SOURCE_ROOT}/main_cuda" \
  "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  "${SOURCE_ROOT}/scripts/audit_pf_400cube_v2_interim_24h_cluster_v1.py" \
  "${SOURCE_ROOT}/scripts/build_pf_400cube_initial_component_anchors_v1.py" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_hourly_merge_dissolution_v1.py" \
  "${FIXTURE_ROOT}/phi.raw.f64" "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
  "${FIXTURE_ROOT}/init_meta.json" "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${FIXTURE_ROOT}/initial_particles.csv" \
  "${PRODUCTION_ROOT}/checkpoints/step_65393.chk"; do
  [[ -f "${path}" ]] || { echo "[fatal] missing input: ${path}" >&2; exit 2; }
done

mkdir -p "${AUDIT_ROOT}/provenance"
exec >"${AUDIT_ROOT}/driver.stdout.log" 2>"${AUDIT_ROOT}/driver.stderr.log"
trap 'rc=$?; if [[ "${rc}" -ne 0 ]]; then printf "BLOCKED_400CUBE_V2_INTERIM_24H_CLUSTER_EXTRACTION_V1\nexit_code=%s\n" "${rc}" >"${AUDIT_ROOT}/status.txt"; fi' EXIT
printf 'RUNNING_400CUBE_V2_INTERIM_24H_CLUSTER_EXTRACTION_V1\n' >"${AUDIT_ROOT}/status.txt"

python3 "${SOURCE_ROOT}/scripts/audit_pf_400cube_v2_interim_24h_cluster_v1.py" \
  --production-root "${PRODUCTION_ROOT}" --out "${AUDIT_ROOT}/integrity"
grep -qx 'PASS_400CUBE_V2_INTERIM_24H_CLUSTER_EXTRACTION_GATE_V1' "${AUDIT_ROOT}/integrity/status.txt"

cp "${PRODUCTION_ROOT}/provenance/campaign_manifest.json" "${AUDIT_ROOT}/provenance/"
cp "${PRODUCTION_ROOT}/provenance/fixture_manifest.json" "${AUDIT_ROOT}/provenance/"
cp "${PRODUCTION_ROOT}/provenance/pf_input_dt0p02.params" "${AUDIT_ROOT}/provenance/production_physics.params"
cp "${PRODUCTION_ROOT}/provenance/input_hashes.sha256" "${AUDIT_ROOT}/provenance/production_input_hashes.sha256"
printf '%s\n' 21798 43596 65393 >"${AUDIT_ROOT}/provenance/selected_steps.txt"
sha256sum \
  "${SOURCE_ROOT}/main_cuda" "${SOURCE_ROOT}/main_cuda.cu" \
  "${SOURCE_ROOT}/cuda_kernels.cu" "${SOURCE_ROOT}/pf_zero_mode_checkpoint.cpp" \
  "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  "${SOURCE_ROOT}/scripts/audit_pf_400cube_v2_interim_24h_cluster_v1.py" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_hourly_merge_dissolution_v1.py" \
  "${AUDIT_ROOT}/provenance/production_physics.params" \
  "${FIXTURE_ROOT}/fixture_manifest.json" "${FIXTURE_ROOT}/phi.raw.f64" \
  "${FIXTURE_ROOT}/xB_alpha.raw.f64" "${FIXTURE_ROOT}/init_meta.json" \
  "${PRODUCTION_ROOT}/checkpoints/step_21798.chk" \
  "${PRODUCTION_ROOT}/checkpoints/step_43596.chk" \
  "${PRODUCTION_ROOT}/checkpoints/step_65393.chk" \
  >"${AUDIT_ROOT}/provenance/input_hashes.sha256"
{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-UNSCHEDULED}"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader,nounits || true
} >"${AUDIT_ROOT}/provenance/device_identity.txt"

c++ -std=c++17 -O3 "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  -lstdc++fs -o "${AUDIT_ROOT}/provenance/analyze_pf_periodic_particle_lineage"
python3 "${SOURCE_ROOT}/scripts/build_pf_400cube_initial_component_anchors_v1.py" \
  --fixture-manifest "${FIXTURE_ROOT}/fixture_manifest.json" \
  --initial-particles "${FIXTURE_ROOT}/initial_particles.csv" \
  --out "${AUDIT_ROOT}/initial_component_anchors.csv"

checkpoint_args=()
for step in 21798 43596 65393; do
  checkpoint_args+=(--checkpoint "${PRODUCTION_ROOT}/checkpoints/step_${step}.chk")
done
run_tracker() {
  local name="$1"
  local threshold="$2"
  local out="${AUDIT_ROOT}/tracker_${name}"
  set +e
  "${AUDIT_ROOT}/provenance/analyze_pf_periodic_particle_lineage" \
    --initial-phi "${FIXTURE_ROOT}/phi.raw.f64" \
    --initial-xb "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
    "${checkpoint_args[@]}" --out "${out}" --grid 400 --dx-nm 1 \
    --threshold "${threshold}" --physical-dt-s 0.9909260953431841 \
    --start-age-h 6 --target-mean 0.03 --expected-initial-count 64 \
    --allow-dissolution --allow-merge-groups \
    >"${AUDIT_ROOT}/tracker_${name}.stdout.log" \
    2>"${AUDIT_ROOT}/tracker_${name}.stderr.log"
  local rc=$?
  set -e
  [[ "${rc}" -eq 0 || "${rc}" -eq 2 ]]
}
run_tracker low 0.0001
run_tracker medium 0.001
run_tracker strong 0.005

set +e
python3 "${SOURCE_ROOT}/scripts/audit_pf_246cube_hourly_merge_dissolution_v1.py" \
  --low "${AUDIT_ROOT}/tracker_low" --medium "${AUDIT_ROOT}/tracker_medium" \
  --strong "${AUDIT_ROOT}/tracker_strong" \
  --initial-components "${AUDIT_ROOT}/initial_component_anchors.csv" \
  --expected-steps "${AUDIT_ROOT}/provenance/selected_steps.txt" \
  --expected-initial-count 64 --domain-nm 400 \
  --physical-dt-s 0.9909260953431841 --start-age-h 6 \
  --out "${AUDIT_ROOT}/merge_aware" \
  >"${AUDIT_ROOT}/merge_aware.stdout.log" 2>"${AUDIT_ROOT}/merge_aware.stderr.log"
merge_rc=$?
set -e
[[ "${merge_rc}" -eq 0 ]]
grep -qx 'PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1' "${AUDIT_ROOT}/merge_aware/status.txt"

fixture_sha="$(sha256sum "${FIXTURE_ROOT}/fixture_manifest.json" | awk '{print $1}')"
library_sha="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["profile_library_manifest_sha256"])' "${FIXTURE_ROOT}/fixture_manifest.json")"
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

age_root="${AUDIT_ROOT}/age_6h"
mkdir -p "${age_root}/results"
CUDA_STO_RESULTS_ROOT="${age_root}/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
  "${SOURCE_ROOT}/main_cuda" 400 400 400 0.02 1 1 1 1 \
  --pf-param-file "${AUDIT_ROOT}/provenance/production_physics.params" \
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

age_root="${AUDIT_ROOT}/age_24h"
mkdir -p "${age_root}/results"
CUDA_STO_RESULTS_ROOT="${age_root}/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
  "${SOURCE_ROOT}/main_cuda" 400 400 400 0.02 65394 65394 1 1 \
  --pf-param-file "${AUDIT_ROOT}/provenance/production_physics.params" \
  --mode dynamics --pf-restart-from "${PRODUCTION_ROOT}/checkpoints/step_65393.chk" \
  "${zero_mode[@]}" "${identity[@]}" --pf-checkpoint-every 0 \
  --mechanics-only-replay-dir "${age_root}/fields" \
  --mechanics-replay-initialization checkpoint_warm \
  --init-case-tag mechanics_v2_input_400cube_24h \
  >"${age_root}/stdout.log" 2>"${age_root}/stderr.log"
[[ ! -s "${age_root}/stderr.log" ]]
grep -q '^MECHANICS_DIAGNOSTIC_STOP_BEFORE_PF_UPDATE ' "${age_root}/stdout.log"

find "${AUDIT_ROOT}" -path '*/fields/*' -type f -print0 | sort -z | xargs -0 sha256sum \
  >"${AUDIT_ROOT}/provenance/replay_field_hashes.sha256"
find "${AUDIT_ROOT}" -type f ! -path '*/fields/*' ! -name output_hashes.sha256 -print0 \
  | sort -z | xargs -0 sha256sum >"${AUDIT_ROOT}/provenance/output_hashes.sha256"
printf 'PASS_400CUBE_V2_INTERIM_24H_CLUSTER_EXTRACTION_V1\n' >"${AUDIT_ROOT}/status.txt"
cat "${AUDIT_ROOT}/status.txt"
