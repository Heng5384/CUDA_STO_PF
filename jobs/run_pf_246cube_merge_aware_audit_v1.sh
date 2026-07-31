#!/usr/bin/env bash
set -euo pipefail

# Read-only supplemental lineage audit for an already completed 246^3
# short-screening checkpoint chain.  It never advances the PF solver.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
FIXTURE_ROOT="${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
STAGE7_ROOT="${STAGE7_ROOT:?STAGE7_ROOT is required}"
SCREEN_ROOT="${SCREEN_ROOT:?SCREEN_ROOT is required}"
AUDIT_ROOT="${AUDIT_ROOT:?AUDIT_ROOT is required}"
GRID_N=246

[[ ! -e "${AUDIT_ROOT}" ]] || {
  echo "[fatal] refusing to overwrite AUDIT_ROOT: ${AUDIT_ROOT}" >&2
  exit 2
}
for path in \
  "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_merge_aware_lineage_v1.py" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_6h8h_screening_v1.py" \
  "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${FIXTURE_ROOT}/initial_components.csv" \
  "${FIXTURE_ROOT}/phi.raw.f64" \
  "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
  "${STAGE7_ROOT}/continuous/final.chk" \
  "${SCREEN_ROOT}/gpu_samples.csv"; do
  [[ -f "${path}" ]] || {
    echo "[fatal] missing required input: ${path}" >&2
    exit 2
  }
done
grep -qx "PASS_246CUBE_SHORT_RESTART_AND_OBSERVABLES_V1" \
  "${STAGE7_ROOT}/status.txt"

ENDPOINTS=(512 1024 1536 2048 2560 3072 3584 4096 4608 5120 5632 6144 6656 7168 7266)
CHECKPOINT_ARGS=(--checkpoint "${STAGE7_ROOT}/continuous/final.chk")
for endpoint in "${ENDPOINTS[@]}"; do
  checkpoint="${SCREEN_ROOT}/checkpoints/step_${endpoint}.chk"
  [[ -f "${checkpoint}" ]] || {
    echo "[fatal] missing screening checkpoint: ${checkpoint}" >&2
    exit 2
  }
  CHECKPOINT_ARGS+=(--checkpoint "${checkpoint}")
done

mkdir -p "${AUDIT_ROOT}/provenance"
exec >"${AUDIT_ROOT}/driver.stdout.log" 2>"${AUDIT_ROOT}/driver.stderr.log"
cleanup() {
  local rc=$?
  if [[ "${rc}" -ne 0 ]]; then
    printf 'BLOCKED_246CUBE_MERGE_AWARE_AUDIT_V1\nexit_code=%s\n' "${rc}" \
      >"${AUDIT_ROOT}/status.txt"
  fi
}
trap cleanup EXIT

TRACKER_LINK_FLAGS=()
if [[ -n "${TRACKER_LDFLAGS:-}" ]]; then
  read -r -a TRACKER_LINK_FLAGS <<<"${TRACKER_LDFLAGS}"
fi
c++ -std=c++17 -O3 \
  "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  "${TRACKER_LINK_FLAGS[@]}" \
  -o "${AUDIT_ROOT}/provenance/analyze_pf_246cube_particle_lineage_v1"
sha256sum \
  "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_merge_aware_lineage_v1.py" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_6h8h_screening_v1.py" \
  "${AUDIT_ROOT}/provenance/analyze_pf_246cube_particle_lineage_v1" \
  "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${FIXTURE_ROOT}/initial_components.csv" \
  "${STAGE7_ROOT}/continuous/final.chk" \
  "${SCREEN_ROOT}/checkpoints/step_7266.chk" \
  >"${AUDIT_ROOT}/provenance/input_hashes.sha256"

run_tracker() {
  local threshold="$1"
  local output="$2"
  "${AUDIT_ROOT}/provenance/analyze_pf_246cube_particle_lineage_v1" \
    --initial-phi "${FIXTURE_ROOT}/phi.raw.f64" \
    --initial-xb "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
    "${CHECKPOINT_ARGS[@]}" \
    --out "${output}" \
    --grid "${GRID_N}" --dx-nm 1 --threshold "${threshold}" \
    --physical-dt-s 0.9909260953431841 --start-age-h 6 \
    --target-mean 0.03 --expected-initial-count 96 \
    --allow-dissolution --allow-merge-groups
  grep -qx "PASS_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1" \
    "${output}/status.txt"
}

run_tracker 0.0001 "${AUDIT_ROOT}/h1e-4"
run_tracker 0.001 "${AUDIT_ROOT}/h1e-3"
run_tracker 0.005 "${AUDIT_ROOT}/h5e-3"

python3 "${SOURCE_ROOT}/scripts/audit_pf_246cube_merge_aware_lineage_v1.py" \
  --low "${AUDIT_ROOT}/h1e-4" \
  --medium "${AUDIT_ROOT}/h1e-3" \
  --strong "${AUDIT_ROOT}/h5e-3" \
  --initial-components "${FIXTURE_ROOT}/initial_components.csv" \
  --out "${AUDIT_ROOT}/merge_aware"
grep -qx "PASS_246CUBE_RESOLVED_MERGE_AWARE_LINEAGE_V1" \
  "${AUDIT_ROOT}/merge_aware/status.txt"

python3 "${SOURCE_ROOT}/scripts/audit_pf_246cube_6h8h_screening_v1.py" \
  --fixture-manifest "${FIXTURE_ROOT}/fixture_manifest.json" \
  --stage7-checkpoint "${STAGE7_ROOT}/continuous/final.chk" \
  --run-root "${SCREEN_ROOT}" \
  --lineage-root "${AUDIT_ROOT}/h1e-4" \
  --gpu-samples "${SCREEN_ROOT}/gpu_samples.csv" \
  --merge-aware-audit "${AUDIT_ROOT}/merge_aware/audit.json" \
  --out "${AUDIT_ROOT}/screening_audit"
grep -qx "PASS_246CUBE_6H8H_SHORT_SCREENING_V1" \
  "${AUDIT_ROOT}/screening_audit/status.txt"
cp "${AUDIT_ROOT}/screening_audit/status.txt" "${AUDIT_ROOT}/status.txt"
cat "${AUDIT_ROOT}/status.txt"
