#!/usr/bin/env bash
set -euo pipefail

# Extend the qualified quarter-nm elastic target-profile library to 6.0-15.0 nm
# on the workstation.  Existing 8.0-11.5 nm profiles are reused without field
# modification; the 6.0-7.75 nm and 12.0-15.0 nm entries are generated with
# the same 96^3, dt=0.025, T=380, elastic conserve-mass contract.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
BASE_LIBRARY_ROOT="${BASE_LIBRARY_ROOT:?BASE_LIBRARY_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
RUN_LABEL="${RUN_LABEL:-workstation_6to15}"
GRID_N="${GRID_N:-96}"
EXTRA_RADII_NM="${EXTRA_RADII_NM:-11.75 12.0 12.25 12.5 12.75 13.0 13.25 13.5 13.75 14.0 14.25 14.5 14.75 15.0}"
EXPECTED_RADII_NM="${EXPECTED_RADII_NM:-6.0 6.25 6.5 6.75 7.0 7.25 7.5 7.75 8.0 8.25 8.5 8.75 9.0 9.25 9.5 9.75 10.0 10.25 10.5 10.75 11.0 11.25 11.5 11.75 12.0 12.25 12.5 12.75 13.0 13.25 13.5 13.75 14.0 14.25 14.5 14.75 15.0}"
MINIMIZE_DT="${MINIMIZE_DT:-0.025}"
MAX_ITER="${MAX_ITER:-16000}"
MIN_PSEUDO_TIME="${MIN_PSEUDO_TIME:-310}"
OUT_EVERY="${OUT_EVERY:-${MAX_ITER}}"
CSV_OUT_EVERY="${CSV_OUT_EVERY:-10}"
XB_FAR="${XB_FAR:-0.006219279767278563}"
MASS_TOL="${MASS_TOL:-1e-12}"
VOLUME_TOL="${VOLUME_TOL:-2e-4}"
CONVERGENCE_STEPS="${CONVERGENCE_STEPS:-20}"
RMS_DPHI_TOL="${RMS_DPHI_TOL:-8e-6}"
RMS_DY_TOL="${RMS_DY_TOL:-2e-5}"
RMS_RES_TOL="${RMS_RES_TOL:-1e-3}"
ENERGY_REL_TOL="${ENERGY_REL_TOL:-1e-8}"
POST_PROJECTION_ITERS="${POST_PROJECTION_ITERS:-0}"
PHYSICAL_OVERRIDE_FILE="${PHYSICAL_OVERRIDE_FILE:-${SOURCE_ROOT}/data/qualification/pf_elastic_target_profile_v1/physical_override_T380_dx1nm_lambda4nm.json}"
ALLOW_MIXED_BINARY_SAME_SOURCE="${ALLOW_MIXED_BINARY_SAME_SOURCE:-1}"
RESUME_EXISTING="${RESUME_EXISTING:-0}"

if [[ -e "${RUN_ROOT}" && "${RESUME_EXISTING}" != "1" ]]; then
  echo "[fatal] refusing to overwrite existing RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
fi
[[ -x "${SOURCE_ROOT}/main_cuda" ]] || { echo "[fatal] compiled binary is missing" >&2; exit 2; }
[[ -f "${PHYSICAL_OVERRIDE_FILE}" ]] || { echo "[fatal] physical override is missing" >&2; exit 2; }

mkdir -p "${RUN_ROOT}/cases" "${RUN_ROOT}/profiles" "${RUN_ROOT}/provenance"

# Reuse the individually qualified 8.0-11.5 nm profiles without touching fields.
for dir in "${BASE_LIBRARY_ROOT}"/profiles/R*; do
  tag="$(basename "${dir}")"
  if [[ ! -f "${dir}/profile_manifest.json" ]] ||
     ! grep -q '^profile_status=PASS_ELASTIC_CONSTRAINED_TARGET_PROFILE_V1$' \
       "${dir}/final_terminal_output.txt"; then
    echo "[fatal] base profile has no exact PASS: ${dir}" >&2
    exit 2
  fi
  if [[ -e "${RUN_ROOT}/profiles/${tag}/profile_manifest.json" ]]; then
    echo "[resume] keeping existing profile ${tag}" >&2
  else
    cp -a "${dir}" "${RUN_ROOT}/profiles/${tag}"
    {
      printf 'radius_source=base_library\n'
      printf 'source_profile=%s\n' "${dir}"
      printf 'source_profile_manifest_sha256=%s\n' \
        "$(sha256sum "${dir}/profile_manifest.json" | awk '{print $1}')"
    } >"${RUN_ROOT}/provenance/${tag}.source.txt"
  fi
done

SOURCE_FILES=(
  main_cuda.cu
  cuda_kernels.cu
  cuda_kernels.h
  cuda_common.cu
  cuda_common.h
  pf_params.h
  pf_zero_mode_checkpoint.cpp
  pf_zero_mode_checkpoint.h
  Unit_Psedobinary.py
  scripts/materialize_pf_elastic_target_profile_v1.py
  scripts/assemble_pf_elastic_target_profile_library_v1.py
  scripts/test_pf_elastic_target_profile_v1.py
  jobs/run_pf_elastic_target_profile_library_v1.sh
)
for relative in "${SOURCE_FILES[@]}"; do
  [[ -f "${SOURCE_ROOT}/${relative}" ]] || {
    echo "[fatal] source-tree member is missing: ${relative}" >&2
    exit 2
  }
done
{
  for relative in "${SOURCE_FILES[@]}"; do
    printf '%s  %s\n' \
      "$(sha256sum "${SOURCE_ROOT}/${relative}" | awk '{print $1}')" \
      "${relative}"
  done
} | LC_ALL=C sort >"${RUN_ROOT}/provenance/source_files.sha256"
SOURCE_TREE_SHA256="$(sha256sum "${RUN_ROOT}/provenance/source_files.sha256" | awk '{print $1}')"
BINARY_SHA256="$(sha256sum "${SOURCE_ROOT}/main_cuda" | awk '{print $1}')"
if [[ -n "${SOURCE_COMMIT_OVERRIDE:-}" ]]; then
  SOURCE_COMMIT="${SOURCE_COMMIT_OVERRIDE}"
elif SOURCE_COMMIT="$(git -C "${SOURCE_ROOT}" rev-parse --verify HEAD 2>/dev/null)"; then
  :
else
  SOURCE_COMMIT="NO_GIT_COMMIT"
fi
sha256sum "${SOURCE_ROOT}/main_cuda" "${PHYSICAL_OVERRIDE_FILE}" \
  >"${RUN_ROOT}/provenance/runtime_inputs.sha256"

export PHYSICAL_OVERRIDE_FILE
export TEMP_C=380
export DT="${MINIMIZE_DT}"
PF_PARAM_FILE="$(site_generate_pf_param_file "elastic_target_profile_${RUN_LABEL}_T380")"
cp "${PF_PARAM_FILE}" "${RUN_ROOT}/provenance/pf_input.params"
cp "${PHYSICAL_OVERRIDE_FILE}" "${RUN_ROOT}/provenance/physical_override.json"

profile_dirs=("${RUN_ROOT}"/profiles/R*)
for RADIUS_NM in ${EXTRA_RADII_NM}; do
  SAFE_RADIUS="${RADIUS_NM//./p}"
  CASE_TAG="elastic_target_R${SAFE_RADIUS}_${RUN_LABEL}"
  CASE_ROOT="${RUN_ROOT}/cases/R${SAFE_RADIUS}"
  PROFILE_ROOT="${RUN_ROOT}/profiles/R${SAFE_RADIUS}"
  if [[ -e "${PROFILE_ROOT}/profile_manifest.json" ]]; then
    echo "[resume] keeping existing profile R${SAFE_RADIUS}" >&2
    profile_dirs+=("${PROFILE_ROOT}")
    continue
  fi
  mkdir -p "${CASE_ROOT}"
  V0="$(python3 - "${RADIUS_NM}" "${GRID_N}" <<'PY'
import math
import sys
r = float(sys.argv[1])
n = int(sys.argv[2])
print(f"{(4.0 * math.pi * r**3 / 3.0) / n**3:.17e}")
PY
)"
  (
    cd "${CASE_ROOT}"
    "${SOURCE_ROOT}/main_cuda" \
      "${GRID_N}" "${GRID_N}" "${GRID_N}" \
      "${MINIMIZE_DT}" "${MAX_ITER}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" 1 \
      --pf-param-file "${RUN_ROOT}/provenance/pf_input.params" \
      --mode=minimize \
      --minimize-full-model \
      --minimize-mass-constraint \
      --minimize-mass-tolerance-relative "${MASS_TOL}" \
      --minimize-mass-max-iterations 64 \
      --minimize-max-iter "${MAX_ITER}" \
      --minimize-dt "${MINIMIZE_DT}" \
      --V0 "${V0}" \
      --radius-phys-nm "${RADIUS_NM}" \
      --elastic 1 \
      --ic-23d-xB-out "${XB_FAR}" \
      --init-shape sphere \
      --init-case-tag "${CASE_TAG}" \
      --minimize-rms-dphi-threshold "${RMS_DPHI_TOL}" \
      --minimize-rms-dY-threshold "${RMS_DY_TOL}" \
      --minimize-rms-res-threshold "${RMS_RES_TOL}" \
      --minimize-vol-err-rel-threshold "${VOLUME_TOL}" \
      --minimize-energy-diff-rel-threshold "${ENERGY_REL_TOL}" \
      --minimize-convergence-steps "${CONVERGENCE_STEPS}" \
      --minimize-min-pseudo-time "${MIN_PSEUDO_TIME}" \
      --minimize-post-projection-iters "${POST_PROJECTION_ITERS}" \
      >"${CASE_ROOT}/run.stdout" 2>"${CASE_ROOT}/run.stderr"
  )
  [[ ! -s "${CASE_ROOT}/run.stderr" ]] || {
    echo "[fatal] non-empty stderr for R=${RADIUS_NM}" >&2
    exit 2
  }
  grep -q '^MINIMIZE_TARGET_PROFILE_CONVERGENCE_FINAL_AUDIT status=PASS ' \
    "${CASE_ROOT}/run.stdout" || {
    echo "[fatal] convergence contract failed for R=${RADIUS_NM}" >&2
    exit 2
  }
  grep -q '^MINIMIZE_MASS_CONSTRAINT_FINAL_AUDIT status=PASS ' \
    "${CASE_ROOT}/run.stdout" || {
    echo "[fatal] mass-constraint PASS missing for R=${RADIUS_NM}" >&2
    exit 2
  }
  PHI_RAW="$(find "${CASE_ROOT}/Results" -type f -name 'phi_final_constraint.raw.f64' -print -quit)"
  XB_RAW="$(find "${CASE_ROOT}/Results" -type f -name 'xB_final_constraint.raw.f64' -print -quit)"
  TRACE="$(find "${CASE_ROOT}/Results" -type f -name 'minimize_mass_constraint_trace.csv' -print -quit)"
  EFFECTIVE_PARAM="$(find "${CASE_ROOT}/Results" -type f -name 'pf_input.params' | sort | tail -1)"
  if [[ -z "${PHI_RAW}" || -z "${XB_RAW}" || -z "${TRACE}" || -z "${EFFECTIVE_PARAM}" ]]; then
    echo "[fatal] incomplete final artifact set for R=${RADIUS_NM}" >&2
    exit 2
  fi
  python3 "${SOURCE_ROOT}/scripts/materialize_pf_elastic_target_profile_v1.py" \
    --phi-raw "${PHI_RAW}" \
    --xb-raw "${XB_RAW}" \
    --grid-nx "${GRID_N}" \
    --grid-ny "${GRID_N}" \
    --grid-nz "${GRID_N}" \
    --constraint-trace "${TRACE}" \
    --run-log "${CASE_ROOT}/run.stdout" \
    --param-file "${EFFECTIVE_PARAM}" \
    --out "${PROFILE_ROOT}" \
    --target-radius-nm "${RADIUS_NM}" \
    --temperature-c 380 \
    --dx-nm 1 \
    --lambda-sm-nm 4 \
    --v-b 1 \
    --orientation-label variant_100_identity \
    --source-commit "${SOURCE_COMMIT}" \
    --source-tree-sha256 "${SOURCE_TREE_SHA256}" \
    --binary-sha256 "${BINARY_SHA256}" \
    >"${CASE_ROOT}/materialize.stdout" 2>"${CASE_ROOT}/materialize.stderr"
  [[ ! -s "${CASE_ROOT}/materialize.stderr" ]] || {
    echo "[fatal] materializer stderr for R=${RADIUS_NM}" >&2
    exit 2
  }
  profile_dirs+=("${PROFILE_ROOT}")
  {
    printf 'radius_source=new_workstation_generation\n'
    printf 'minimize_dt=%s\n' "${MINIMIZE_DT}"
    printf 'max_iter=%s\n' "${MAX_ITER}"
    printf 'min_pseudo_time=%s\n' "${MIN_PSEUDO_TIME}"
  } >"${RUN_ROOT}/provenance/R${SAFE_RADIUS}.source.txt"
done

assembler_args=(
  --profiles "${profile_dirs[@]}"
  --expected-radii-nm ${EXPECTED_RADII_NM}
  --out "${RUN_ROOT}/library"
)
if [[ "${ALLOW_MIXED_BINARY_SAME_SOURCE}" == "1" ]]; then
  assembler_args+=(--allow-mixed-binary-same-source)
fi
python3 "${SOURCE_ROOT}/scripts/assemble_pf_elastic_target_profile_library_v1.py" \
  "${assembler_args[@]}" \
  >"${RUN_ROOT}/library_assembly.stdout" \
  2>"${RUN_ROOT}/library_assembly.stderr"
[[ ! -s "${RUN_ROOT}/library_assembly.stderr" ]] || {
  echo "[fatal] library assembler produced stderr" >&2
  exit 2
}
grep -q '^library_status=PASS_ELASTIC_TARGET_PROFILE_LIBRARY_ASSEMBLY_V1$' \
  "${RUN_ROOT}/library/final_terminal_output.txt"

{
  printf 'runner_status=PASS_PF_ELASTIC_TARGET_PROFILE_LIBRARY_V1\n'
  printf 'run_label=%s\n' "${RUN_LABEL}"
  printf 'grid=%sx%sx%s\n' "${GRID_N}" "${GRID_N}" "${GRID_N}"
  printf 'radii_nm=%s\n' "${EXPECTED_RADII_NM}"
  printf 'minimize_dt=%s\n' "${MINIMIZE_DT}"
  printf 'min_pseudo_time=%s\n' "${MIN_PSEUDO_TIME}"
  printf 'elastic_enabled=true\n'
  printf 'gp_enabled=false\n'
  printf 'source_commit=%s\n' "${SOURCE_COMMIT}"
  printf 'source_tree_sha256=%s\n' "${SOURCE_TREE_SHA256}"
  printf 'binary_sha256=%s\n' "${BINARY_SHA256}"
  printf 'library_manifest_sha256=%s\n' \
    "$(sha256sum "${RUN_ROOT}/library/library_manifest.json" | awk '{print $1}')"
} >"${RUN_ROOT}/status.txt"
cat "${RUN_ROOT}/status.txt"
