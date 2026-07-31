#!/usr/bin/env bash
set -euo pipefail

# Generate a non-overwriting, hash-pinned T380 constrained elastic profile
# ladder.  This script is valid both on workstation and inside a Slurm job.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
RUN_LABEL="${RUN_LABEL:-workstation}"
GRID_N="${GRID_N:-96}"
RADII_NM="${RADII_NM:-8.0 8.5 9.0 9.5 10.0 10.5 11.0 11.5}"
MINIMIZE_DT="${MINIMIZE_DT:-0.1}"
MAX_ITER="${MAX_ITER:-8000}"
OUT_EVERY="${OUT_EVERY:-${MAX_ITER}}"
CSV_OUT_EVERY="${CSV_OUT_EVERY:-10}"
XB_FAR="${XB_FAR:-0.006219279767278563}"
MASS_TOL="${MASS_TOL:-1e-12}"
VOLUME_TOL="${VOLUME_TOL:-2e-4}"
CONVERGENCE_STEPS="${CONVERGENCE_STEPS:-50}"
MIN_PSEUDO_TIME="${MIN_PSEUDO_TIME:-300}"
RMS_DPHI_TOL="${RMS_DPHI_TOL:-8e-6}"
RMS_DY_TOL="${RMS_DY_TOL:-2e-5}"
RMS_RES_TOL="${RMS_RES_TOL:-1e-3}"
ENERGY_REL_TOL="${ENERGY_REL_TOL:-1e-8}"
POST_PROJECTION_ITERS="${POST_PROJECTION_ITERS:-0}"
PHYSICAL_OVERRIDE_FILE="${PHYSICAL_OVERRIDE_FILE:-${SOURCE_ROOT}/data/qualification/pf_elastic_target_profile_v1/physical_override_T380_dx1nm_lambda4nm.json}"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "[fatal] refusing to overwrite existing RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
fi
if [[ ! -x "${SOURCE_ROOT}/main_cuda" ]]; then
  echo "[fatal] compiled binary is missing: ${SOURCE_ROOT}/main_cuda" >&2
  exit 2
fi
if [[ ! -f "${PHYSICAL_OVERRIDE_FILE}" ]]; then
  echo "[fatal] physical override is missing: ${PHYSICAL_OVERRIDE_FILE}" >&2
  exit 2
fi
python3 - \
  "${MINIMIZE_DT}" "${MIN_PSEUDO_TIME}" "${MAX_ITER}" \
  "${CONVERGENCE_STEPS}" <<'PY'
import math
import sys

dt, minimum, max_iter = float(sys.argv[1]), float(sys.argv[2]), int(sys.argv[3])
consecutive = int(sys.argv[4])
if not math.isfinite(dt) or dt <= 0.0:
    raise SystemExit("[fatal] MINIMIZE_DT must be finite and > 0")
if not math.isfinite(minimum) or minimum < 0.0:
    raise SystemExit("[fatal] MIN_PSEUDO_TIME must be finite and >= 0")
required = math.ceil(minimum / dt) + consecutive
if max_iter < required:
    raise SystemExit(
        f"[fatal] MAX_ITER={max_iter} cannot reach MIN_PSEUDO_TIME={minimum} "
        f"at dt={dt} plus {consecutive} consecutive convergence steps; "
        f"need at least {required}"
    )
PY
mkdir -p "${RUN_ROOT}/cases" "${RUN_ROOT}/profiles" "${RUN_ROOT}/provenance"

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
  if [[ ! -f "${SOURCE_ROOT}/${relative}" ]]; then
    echo "[fatal] source-tree member is missing: ${relative}" >&2
    exit 2
  fi
  printf '%s  %s\n' \
    "$(sha256sum "${SOURCE_ROOT}/${relative}" | awk '{print $1}')" \
    "${relative}"
done | LC_ALL=C sort >"${RUN_ROOT}/provenance/source_files.sha256"
SOURCE_TREE_SHA256="$(sha256sum "${RUN_ROOT}/provenance/source_files.sha256" | awk '{print $1}')"
BINARY_SHA256="$(sha256sum "${SOURCE_ROOT}/main_cuda" | awk '{print $1}')"
if [[ -n "${SOURCE_COMMIT_OVERRIDE:-}" ]]; then
  SOURCE_COMMIT="${SOURCE_COMMIT_OVERRIDE}"
elif SOURCE_COMMIT="$(git -C "${SOURCE_ROOT}" rev-parse --verify HEAD 2>/dev/null)"; then
  :
else
  SOURCE_COMMIT="NO_GIT_COMMIT"
fi
if [[ "${SOURCE_COMMIT}" == *$'\n'* || -z "${SOURCE_COMMIT}" ]]; then
  echo "[fatal] SOURCE_COMMIT must be one non-empty line" >&2
  exit 2
fi
sha256sum "${SOURCE_ROOT}/main_cuda" "${PHYSICAL_OVERRIDE_FILE}" \
  >"${RUN_ROOT}/provenance/runtime_inputs.sha256"

export PHYSICAL_OVERRIDE_FILE
export TEMP_C=380
export DT="${MINIMIZE_DT}"
PF_PARAM_FILE="$(site_generate_pf_param_file "elastic_target_profile_${RUN_LABEL}_T380")"
cp "${PF_PARAM_FILE}" "${RUN_ROOT}/provenance/pf_input.params"
cp "${PHYSICAL_OVERRIDE_FILE}" "${RUN_ROOT}/provenance/physical_override.json"

PROFILE_DIRS=()
for RADIUS_NM in ${RADII_NM}; do
  SAFE_RADIUS="${RADIUS_NM//./p}"
  CASE_TAG="elastic_target_R${SAFE_RADIUS}_${RUN_LABEL}"
  CASE_ROOT="${RUN_ROOT}/cases/R${SAFE_RADIUS}"
  PROFILE_ROOT="${RUN_ROOT}/profiles/R${SAFE_RADIUS}"
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
  if [[ -s "${CASE_ROOT}/run.stderr" ]]; then
    echo "[fatal] non-empty stderr for R=${RADIUS_NM}" >&2
    exit 2
  fi
  if ! grep -q \
      "^MINIMIZE_TARGET_PROFILE_CONVERGENCE_FINAL_AUDIT status=PASS " \
      "${CASE_ROOT}/run.stdout"; then
    echo "[fatal] target profile did not satisfy convergence contract for R=${RADIUS_NM}" >&2
    exit 2
  fi
  if ! grep -q "^MINIMIZE_MASS_CONSTRAINT_FINAL_AUDIT status=PASS " \
      "${CASE_ROOT}/run.stdout"; then
    echo "[fatal] missing exact mass-constraint PASS for R=${RADIUS_NM}" >&2
    exit 2
  fi
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
  if [[ -s "${CASE_ROOT}/materialize.stderr" ]]; then
    echo "[fatal] materializer stderr for R=${RADIUS_NM}" >&2
    exit 2
  fi
  PROFILE_DIRS+=("${PROFILE_ROOT}")
done

python3 "${SOURCE_ROOT}/scripts/assemble_pf_elastic_target_profile_library_v1.py" \
  --profiles "${PROFILE_DIRS[@]}" \
  --expected-radii-nm ${RADII_NM} \
  --out "${RUN_ROOT}/library" \
  >"${RUN_ROOT}/library_assembly.stdout" \
  2>"${RUN_ROOT}/library_assembly.stderr"
if [[ -s "${RUN_ROOT}/library_assembly.stderr" ]]; then
  echo "[fatal] library assembler produced stderr" >&2
  exit 2
fi
grep -q '^library_status=PASS_ELASTIC_TARGET_PROFILE_LIBRARY_ASSEMBLY_V1$' \
  "${RUN_ROOT}/library/final_terminal_output.txt"
{
  printf 'runner_status=PASS_PF_ELASTIC_TARGET_PROFILE_LIBRARY_V1\n'
  printf 'run_label=%s\n' "${RUN_LABEL}"
  printf 'grid=%sx%sx%s\n' "${GRID_N}" "${GRID_N}" "${GRID_N}"
  printf 'radii_nm=%s\n' "${RADII_NM}"
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
