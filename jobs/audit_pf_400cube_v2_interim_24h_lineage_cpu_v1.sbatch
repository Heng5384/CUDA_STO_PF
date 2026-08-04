#!/usr/bin/env bash
#SBATCH --job-name=pf400_v2_lin24
#SBATCH --partition=uvip_parallel_intel
#SBATCH --qos=uvip_parallel_intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=02:00:00
#SBATCH --mem=120G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Read-only, hourly, three-threshold lineage audit through exact 24 h.  No GPU
# is requested and nothing below PRODUCTION_ROOT or FIXTURE_ROOT is written.

set -euo pipefail
: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${PRODUCTION_ROOT:?PRODUCTION_ROOT is required}"
: "${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
: "${AUDIT_ROOT:?AUDIT_ROOT is required}"
[[ ! -e "${AUDIT_ROOT}" ]] || { echo "[fatal] refusing to overwrite AUDIT_ROOT" >&2; exit 2; }
mkdir -p "${AUDIT_ROOT}/provenance"
exec >"${AUDIT_ROOT}/driver.stdout.log" 2>"${AUDIT_ROOT}/driver.stderr.log"
trap 'rc=$?; if [[ "${rc}" -ne 0 ]]; then printf "BLOCKED_400CUBE_V2_INTERIM_24H_HOURLY_LINEAGE_V1\nexit_code=%s\n" "${rc}" >"${AUDIT_ROOT}/status.txt"; fi' EXIT
printf 'RUNNING_400CUBE_V2_INTERIM_24H_HOURLY_LINEAGE_V1\n' >"${AUDIT_ROOT}/status.txt"

c++ -std=c++17 -O3 "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  -lstdc++fs -o "${AUDIT_ROOT}/provenance/analyze_pf_periodic_particle_lineage"
python3 "${SOURCE_ROOT}/scripts/build_pf_400cube_initial_component_anchors_v1.py" \
  --fixture-manifest "${FIXTURE_ROOT}/fixture_manifest.json" \
  --initial-particles "${FIXTURE_ROOT}/initial_particles.csv" \
  --out "${AUDIT_ROOT}/initial_component_anchors.csv"

mapfile -t steps < <(find "${PRODUCTION_ROOT}/checkpoints" -maxdepth 1 -type f -name 'step_*.chk' \
  -printf '%f\n' | sed -E 's/^step_([0-9]+)\.chk$/\1/' | awk '$1 <= 65393' | sort -n)
expected=(3633 7266 10899 14532 18165 21798 25431 29064 32697 36330 39963 43596 47229 50862 54495 58128 61761 65393)
[[ "${steps[*]}" == "${expected[*]}" ]] || { echo "[fatal] 0--24 h checkpoint schedule mismatch" >&2; exit 3; }
printf '%s\n' "${steps[@]}" >"${AUDIT_ROOT}/provenance/selected_steps.txt"
checkpoint_args=()
: >"${AUDIT_ROOT}/provenance/selected_checkpoint_hashes.sha256"
for step in "${steps[@]}"; do
  checkpoint="${PRODUCTION_ROOT}/checkpoints/step_${step}.chk"
  checkpoint_args+=(--checkpoint "${checkpoint}")
  cat "${PRODUCTION_ROOT}/segments/step_${step}/checkpoint.sha256" \
    >>"${AUDIT_ROOT}/provenance/selected_checkpoint_hashes.sha256"
done

sha256sum \
  "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  "${SOURCE_ROOT}/scripts/build_pf_400cube_initial_component_anchors_v1.py" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_hourly_merge_dissolution_v1.py" \
  "${FIXTURE_ROOT}/fixture_manifest.json" "${FIXTURE_ROOT}/phi.raw.f64" \
  "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
  >"${AUDIT_ROOT}/provenance/input_hashes.sha256"
{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-UNSCHEDULED}"
  printf 'production_root=%s\n' "${PRODUCTION_ROOT}"
  printf 'analysis_scope=READ_ONLY_EXACT_6H_TO_24H_HOURLY_LINEAGE\n'
} >"${AUDIT_ROOT}/provenance/device_identity.txt"

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

find "${AUDIT_ROOT}" -type f ! -name output_hashes.sha256 -print0 | sort -z | xargs -0 sha256sum \
  >"${AUDIT_ROOT}/provenance/output_hashes.sha256"
printf 'PASS_400CUBE_V2_INTERIM_24H_HOURLY_LINEAGE_V1\n' >"${AUDIT_ROOT}/status.txt"
cat "${AUDIT_ROOT}/status.txt"
