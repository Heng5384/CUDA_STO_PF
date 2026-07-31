#!/usr/bin/env bash
set -euo pipefail

# Construct a validation-only V4 handoff profile.  The six resolved-beta
# profiles from the selected V2 fixture are held byte-identical; only the
# conserved composition field is equilibrated under the exact mass ledger.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
V2_FIXTURE_ROOT="${V2_FIXTURE_ROOT:?V2_FIXTURE_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
PHYSICAL_OVERRIDE_FILE="${PHYSICAL_OVERRIDE_FILE:-${SOURCE_ROOT}/data/qualification/pf_elastic_target_profile_v1/physical_override_T380_dx1nm_lambda4nm.json}"
MINIMIZE_DT="${MINIMIZE_DT:-0.1}"
MAX_ITER="${MAX_ITER:-8000}"
OUT_EVERY="${OUT_EVERY:-${MAX_ITER}}"
CSV_OUT_EVERY="${CSV_OUT_EVERY:-10}"
MASS_TOL="${MASS_TOL:-1e-12}"
CONVERGENCE_STEPS="${CONVERGENCE_STEPS:-50}"
MIN_PSEUDO_TIME="${MIN_PSEUDO_TIME:-300}"
RMS_DY_TOL="${RMS_DY_TOL:-2e-5}"
ENERGY_REL_TOL="${ENERGY_REL_TOL:-1e-8}"

[[ ! -e "${RUN_ROOT}" ]] || { echo "[fatal] refusing to overwrite RUN_ROOT: ${RUN_ROOT}" >&2; exit 2; }
for path in \
  "${SOURCE_ROOT}/main_cuda" \
  "${SOURCE_ROOT}/scripts/materialize_pf_elastic_multi_particle_common_matrix_seed_v3.py" \
  "${SOURCE_ROOT}/scripts/materialize_pf_elastic_multi_particle_common_matrix_profile_v3.py" \
  "${V2_FIXTURE_ROOT}/fixture_manifest.json" \
  "${PHYSICAL_OVERRIDE_FILE}"; do
  [[ -f "${path}" ]] || { echo "[fatal] missing required input: ${path}" >&2; exit 2; }
done

mkdir -p "${RUN_ROOT}/provenance" "${RUN_ROOT}/minimize"
SOURCE_FILES=(
  main_cuda.cu cuda_kernels.cu cuda_kernels.h cuda_common.cu cuda_common.h pf_params.h
  pf_zero_mode_checkpoint.cpp pf_zero_mode_checkpoint.h Unit_Psedobinary.py
  scripts/materialize_pf_elastic_multi_particle_common_matrix_seed_v3.py
  scripts/materialize_pf_elastic_multi_particle_common_matrix_profile_v3.py
  scripts/materialize_pf_elastic_multi_particle_6h_fixture_v1.py
  jobs/run_pf_elastic_multi_particle_fixed_phi_profile_v4.sh
)
for relative in "${SOURCE_FILES[@]}"; do
  [[ -f "${SOURCE_ROOT}/${relative}" ]] || { echo "[fatal] source member missing: ${relative}" >&2; exit 2; }
  printf '%s  %s\n' "$(sha256sum "${SOURCE_ROOT}/${relative}" | awk '{print $1}')" "${relative}"
done | LC_ALL=C sort >"${RUN_ROOT}/provenance/source_files.sha256"
sha256sum "${SOURCE_ROOT}/main_cuda" "${PHYSICAL_OVERRIDE_FILE}" "${V2_FIXTURE_ROOT}/fixture_manifest.json" \
  >"${RUN_ROOT}/provenance/runtime_inputs.sha256"

python3 "${SOURCE_ROOT}/scripts/materialize_pf_elastic_multi_particle_common_matrix_seed_v3.py" \
  --v2-fixture "${V2_FIXTURE_ROOT}" --out "${RUN_ROOT}/seed" \
  >"${RUN_ROOT}/seed.stdout" 2>"${RUN_ROOT}/seed.stderr"
[[ ! -s "${RUN_ROOT}/seed.stderr" ]] || { echo "[fatal] common-matrix seed stderr" >&2; exit 2; }

export PHYSICAL_OVERRIDE_FILE TEMP_C=380 DT="${MINIMIZE_DT}"
PF_PARAM_FILE="$(site_generate_pf_param_file 'elastic_multi_particle_fixed_phi_v4_T380')"
cp "${PF_PARAM_FILE}" "${RUN_ROOT}/provenance/pf_input.params"
cp "${PHYSICAL_OVERRIDE_FILE}" "${RUN_ROOT}/provenance/physical_override.json"
GRID_N="$(python3 - "${RUN_ROOT}/seed/seed_manifest.json" <<'PY'
import json,sys
g=json.load(open(sys.argv[1]))['grid']; assert g['Nx']==g['Ny']==g['Nz']; print(g['Nx'])
PY
)"

(
  cd "${RUN_ROOT}/minimize"
  CUDA_STO_RESULTS_ROOT="${RUN_ROOT}/minimize/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
    "${SOURCE_ROOT}/main_cuda" \
      "${GRID_N}" "${GRID_N}" "${GRID_N}" \
      "${MINIMIZE_DT}" "${MAX_ITER}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" 1 \
      --pf-param-file "${RUN_ROOT}/provenance/pf_input.params" \
      --mode=minimize --minimize-full-model --minimize-mass-constraint --minimize-freeze-phi \
      --minimize-mass-tolerance-relative "${MASS_TOL}" --minimize-mass-max-iterations 64 \
      --minimize-max-iter "${MAX_ITER}" --minimize-dt "${MINIMIZE_DT}" \
      --elastic 1 --minimize-rms-dY-threshold "${RMS_DY_TOL}" \
      --minimize-energy-diff-rel-threshold "${ENERGY_REL_TOL}" \
      --minimize-convergence-steps "${CONVERGENCE_STEPS}" \
      --minimize-min-pseudo-time "${MIN_PSEUDO_TIME}" \
      --minimize-post-projection-iters 0 \
      --init-mode raw_fields --init-phi-raw "${RUN_ROOT}/seed/phi.raw.f64" \
      --init-xB-raw "${RUN_ROOT}/seed/xB_alpha.raw.f64" \
      --init-dY-dt-prev-raw "${RUN_ROOT}/seed/dY_dt_prev.raw.f64" \
      --init-meta "${RUN_ROOT}/seed/init_meta.json" --init-case-tag elastic_multi_particle_fixed_phi_v4 \
      >"${RUN_ROOT}/minimize/run.stdout" 2>"${RUN_ROOT}/minimize/run.stderr"
)
[[ ! -s "${RUN_ROOT}/minimize/run.stderr" ]] || { echo "[fatal] fixed-phi minimizer stderr" >&2; exit 2; }
grep -q '^MINIMIZE_TARGET_PROFILE_CONVERGENCE_FINAL_AUDIT status=PASS mode=FIXED_PHI_CONSERVED_COMPOSITION_TARGET_PROFILE_V4 ' "${RUN_ROOT}/minimize/run.stdout"
grep -q '^MINIMIZE_MASS_CONSTRAINT_FINAL_AUDIT status=PASS ' "${RUN_ROOT}/minimize/run.stdout"
grep -q '^MINIMIZE_FIXED_PHI_COMPOSITION_FINAL_AUDIT status=PASS ' "${RUN_ROOT}/minimize/run.stdout"

PHI_RAW="$(find "${RUN_ROOT}/minimize/results" -type f -name phi_final_constraint.raw.f64 -print -quit)"
XB_RAW="$(find "${RUN_ROOT}/minimize/results" -type f -name xB_final_constraint.raw.f64 -print -quit)"
Y_RAW="$(find "${RUN_ROOT}/minimize/results" -type f -name Y_final_constraint.raw.f64 -print -quit)"
DY_RAW="$(find "${RUN_ROOT}/minimize/results" -type f -name dY_dt_prev_final_constraint.raw.f64 -print -quit)"
[[ -n "${PHI_RAW}" && -n "${XB_RAW}" && -n "${Y_RAW}" && -n "${DY_RAW}" ]] || { echo "[fatal] fixed-phi minimizer raw field set incomplete" >&2; exit 2; }
python3 "${SOURCE_ROOT}/scripts/materialize_pf_elastic_multi_particle_common_matrix_profile_v3.py" \
  --seed "${RUN_ROOT}/seed" --minimizer-stdout "${RUN_ROOT}/minimize/run.stdout" \
  --phi-raw "${PHI_RAW}" --xB-raw "${XB_RAW}" --Y-raw "${Y_RAW}" --dY-dt-prev-raw "${DY_RAW}" \
  --require-fixed-phi --out "${RUN_ROOT}/profile" >"${RUN_ROOT}/profile.stdout" 2>"${RUN_ROOT}/profile.stderr"
[[ ! -s "${RUN_ROOT}/profile.stderr" ]] || { echo "[fatal] V4 profile materializer stderr" >&2; exit 2; }

cp "${RUN_ROOT}/profile/fixture_manifest.json" "${RUN_ROOT}/fixture_manifest.json"
{
  echo 'fixed_phi_common_matrix_profile_status=PASS_PF_ELASTIC_MULTI_PARTICLE_FIXED_PHI_COMMON_MATRIX_PROFILE_V4'
  printf 'fixture_manifest_sha256=%s\n' "$(sha256sum "${RUN_ROOT}/profile/fixture_manifest.json" | awk '{print $1}')"
  printf 'source_tree_sha256=%s\n' "$(sha256sum "${RUN_ROOT}/provenance/source_files.sha256" | awk '{print $1}')"
  echo 'phi_evolution=frozen_byte_identical_to_seed'
  echo 'physical_parameters_changed=false'
  echo 'full_6h_to_48h_production_started=false'
} | tee "${RUN_ROOT}/status.txt"
