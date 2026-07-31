#!/usr/bin/env bash
set -euo pipefail

# Validation-only V5 preparation: jointly relax phi/Y for the separated E2
# particles while independently preserving every particle's initial h-volume.
# This never starts a 6--48 h production continuation.

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
# V5 uses the same registered *numerical* target-profile stopping contract as
# the frozen single-particle elastic library.  The direct-rate gate is 1e-5;
# its energy-plateau branch is therefore 5e-5 inside main_cuda.  These are
# not case-specific physics knobs.
RMS_DPHI_TOL="${RMS_DPHI_TOL:-1e-5}"
RMS_DY_TOL="${RMS_DY_TOL:-2e-5}"
RMS_RES_TOL="${RMS_RES_TOL:-1e-3}"
ENERGY_REL_TOL="${ENERGY_REL_TOL:-1e-8}"
COMPONENT_VOL_TOL="${COMPONENT_VOL_TOL:-1e-8}"
# V5 projects the phase gradient onto one independent h-volume tangent plane
# per particle.  A full current-field multiplier is the deterministic
# constrained-gradient update; this is numerical stabilization only and does
# not alter the free energy or any physical parameter.
COMPONENT_LAMBDA_RELAXATION="${COMPONENT_LAMBDA_RELAXATION:-1.0}"

python3 - "${COMPONENT_LAMBDA_RELAXATION}" "${RMS_DPHI_TOL}" "${RMS_DY_TOL}" "${RMS_RES_TOL}" "${ENERGY_REL_TOL}" "${COMPONENT_VOL_TOL}" <<'PY'
import math, sys
x, *thresholds = (float(arg) for arg in sys.argv[1:])
if not math.isfinite(x) or x < 0.0 or x > 1.0:
    raise SystemExit("COMPONENT_LAMBDA_RELAXATION must be finite and in [0,1]")
if any((not math.isfinite(value) or value <= 0.0) for value in thresholds):
    raise SystemExit("V5 minimization thresholds must be finite and positive")
PY

[[ ! -e "${RUN_ROOT}" ]] || { echo "[fatal] refusing to overwrite RUN_ROOT: ${RUN_ROOT}" >&2; exit 2; }
for path in \
  "${SOURCE_ROOT}/main_cuda" \
  "${SOURCE_ROOT}/scripts/materialize_pf_elastic_multi_particle_common_matrix_seed_v3.py" \
  "${SOURCE_ROOT}/scripts/materialize_pf_elastic_multi_particle_component_constraint_v5.py" \
  "${SOURCE_ROOT}/scripts/materialize_pf_elastic_multi_particle_component_profile_v5.py" \
  "${V2_FIXTURE_ROOT}/fixture_manifest.json" \
  "${PHYSICAL_OVERRIDE_FILE}"; do
  [[ -f "${path}" ]] || { echo "[fatal] missing required input: ${path}" >&2; exit 2; }
done

mkdir -p "${RUN_ROOT}/provenance" "${RUN_ROOT}/minimize"
SOURCE_FILES=(
  main_cuda.cu cuda_kernels.cu cuda_kernels.h cuda_common.cu cuda_common.h pf_params.h
  pf_zero_mode_checkpoint.cpp pf_zero_mode_checkpoint.h Unit_Psedobinary.py
  scripts/materialize_pf_elastic_multi_particle_6h_fixture_v1.py
  scripts/materialize_pf_elastic_multi_particle_common_matrix_seed_v3.py
  scripts/materialize_pf_elastic_multi_particle_component_constraint_v5.py
  scripts/materialize_pf_elastic_multi_particle_component_profile_v5.py
  jobs/run_pf_elastic_multi_particle_component_profile_v5.sh
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
[[ ! -s "${RUN_ROOT}/seed.stderr" ]] || { echo "[fatal] V5 common-matrix seed stderr" >&2; exit 2; }
python3 "${SOURCE_ROOT}/scripts/materialize_pf_elastic_multi_particle_component_constraint_v5.py" \
  --seed "${RUN_ROOT}/seed" --out "${RUN_ROOT}/component_constraint" \
  >"${RUN_ROOT}/component_constraint.stdout" 2>"${RUN_ROOT}/component_constraint.stderr"
[[ ! -s "${RUN_ROOT}/component_constraint.stderr" ]] || { echo "[fatal] V5 component constraint stderr" >&2; exit 2; }

# The seed is a physical fixture; the following joint relaxation is explicitly
# pseudo-time.  Give main_cuda a matching metadata copy so the raw-field
# integrity check does not emit a spurious stderr warning merely because the
# pseudo-time step differs from the seed's recommended dynamics step.
python3 - "${RUN_ROOT}/seed/init_meta.json" "${RUN_ROOT}/minimize/init_meta.json" "${MINIMIZE_DT}" <<'PY'
import json, math, pathlib, sys
src = pathlib.Path(sys.argv[1])
dst = pathlib.Path(sys.argv[2])
dt = float(sys.argv[3])
if not math.isfinite(dt) or dt <= 0.0:
    raise SystemExit("MINIMIZE_DT must be finite and positive")
payload = json.loads(src.read_text(encoding="utf-8"))
payload["dt_recommended"] = dt
payload["time_level_contract"] = (
    "nonphysical_component_constrained_minimization_pseudotime_v5"
)
dst.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

export PHYSICAL_OVERRIDE_FILE TEMP_C=380 DT="${MINIMIZE_DT}"
PF_PARAM_FILE="$(site_generate_pf_param_file 'elastic_multi_particle_component_v5_T380')"
cp "${PF_PARAM_FILE}" "${RUN_ROOT}/provenance/pf_input.params"
cp "${PHYSICAL_OVERRIDE_FILE}" "${RUN_ROOT}/provenance/physical_override.json"
read -r GRID_N COMPONENT_COUNT TARGET_H_SUMS < <(python3 - "${RUN_ROOT}/seed/seed_manifest.json" "${RUN_ROOT}/component_constraint/component_constraint_manifest.json" <<'PY'
import json,sys
seed=json.load(open(sys.argv[1])); c=json.load(open(sys.argv[2])); g=seed['grid']
assert g['Nx']==g['Ny']==g['Nz']
values=','.join(format(float(x['target_h_sum']), '.17g') for x in c['components'])
print(g['Nx'], c['ownership_contract']['component_count'], values)
PY
)
LABEL_RAW="${RUN_ROOT}/component_constraint/component_ownership_labels.raw.i32"

(
  cd "${RUN_ROOT}/minimize"
  CUDA_STO_RESULTS_ROOT="${RUN_ROOT}/minimize/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
    "${SOURCE_ROOT}/main_cuda" \
      "${GRID_N}" "${GRID_N}" "${GRID_N}" \
      "${MINIMIZE_DT}" "${MAX_ITER}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" 1 \
      --pf-param-file "${RUN_ROOT}/provenance/pf_input.params" \
      --mode=minimize --minimize-full-model --minimize-mass-constraint \
      --minimize-component-volume-constraints \
      --minimize-component-count "${COMPONENT_COUNT}" \
      --minimize-component-label-raw "${LABEL_RAW}" \
      --minimize-component-target-h-sums "${TARGET_H_SUMS}" \
      --V0 0 \
      --minimize-mass-tolerance-relative "${MASS_TOL}" --minimize-mass-max-iterations 64 \
      --minimize-max-iter "${MAX_ITER}" --minimize-dt "${MINIMIZE_DT}" \
      --elastic 1 --minimize-rms-dphi-threshold "${RMS_DPHI_TOL}" \
      --minimize-rms-dY-threshold "${RMS_DY_TOL}" \
      --minimize-rms-res-threshold "${RMS_RES_TOL}" \
      --minimize-energy-diff-rel-threshold "${ENERGY_REL_TOL}" \
      --minimize-vol-err-rel-threshold "${COMPONENT_VOL_TOL}" \
      --eta-lambda-vol "${COMPONENT_LAMBDA_RELAXATION}" \
      --minimize-convergence-steps "${CONVERGENCE_STEPS}" \
      --minimize-min-pseudo-time "${MIN_PSEUDO_TIME}" \
      --minimize-post-projection-iters 0 \
      --init-mode raw_fields --init-phi-raw "${RUN_ROOT}/seed/phi.raw.f64" \
      --init-xB-raw "${RUN_ROOT}/seed/xB_alpha.raw.f64" \
      --init-dY-dt-prev-raw "${RUN_ROOT}/seed/dY_dt_prev.raw.f64" \
      --init-meta "${RUN_ROOT}/minimize/init_meta.json" --init-case-tag elastic_multi_particle_component_v5 \
      >"${RUN_ROOT}/minimize/run.stdout" 2>"${RUN_ROOT}/minimize/run.stderr"
)
[[ ! -s "${RUN_ROOT}/minimize/run.stderr" ]] || { echo "[fatal] V5 minimizer stderr" >&2; exit 2; }
grep -q '^MINIMIZE_TARGET_PROFILE_CONVERGENCE_FINAL_AUDIT status=PASS mode=COMPONENT_CONSERVED_H_VOLUME_TARGET_PROFILE_V5 ' "${RUN_ROOT}/minimize/run.stdout"
grep -q '^MINIMIZE_MASS_CONSTRAINT_FINAL_AUDIT status=PASS ' "${RUN_ROOT}/minimize/run.stdout"
grep -q '^MINIMIZE_COMPONENT_VOLUME_FINAL_AUDIT status=PASS ' "${RUN_ROOT}/minimize/run.stdout"

PHI_RAW="$(find "${RUN_ROOT}/minimize/results" -type f -name phi_final_constraint.raw.f64 -print -quit)"
XB_RAW="$(find "${RUN_ROOT}/minimize/results" -type f -name xB_final_constraint.raw.f64 -print -quit)"
Y_RAW="$(find "${RUN_ROOT}/minimize/results" -type f -name Y_final_constraint.raw.f64 -print -quit)"
DY_RAW="$(find "${RUN_ROOT}/minimize/results" -type f -name dY_dt_prev_final_constraint.raw.f64 -print -quit)"
[[ -n "${PHI_RAW}" && -n "${XB_RAW}" && -n "${Y_RAW}" && -n "${DY_RAW}" ]] || { echo "[fatal] V5 minimizer raw field set incomplete" >&2; exit 2; }
python3 "${SOURCE_ROOT}/scripts/materialize_pf_elastic_multi_particle_component_profile_v5.py" \
  --seed "${RUN_ROOT}/seed" --constraint "${RUN_ROOT}/component_constraint" \
  --minimizer-stdout "${RUN_ROOT}/minimize/run.stdout" \
  --phi-raw "${PHI_RAW}" --xB-raw "${XB_RAW}" --Y-raw "${Y_RAW}" --dY-dt-prev-raw "${DY_RAW}" \
  --dynamic-time-level fresh_zero \
  --out "${RUN_ROOT}/profile" >"${RUN_ROOT}/profile.stdout" 2>"${RUN_ROOT}/profile.stderr"
[[ ! -s "${RUN_ROOT}/profile.stderr" ]] || { echo "[fatal] V5 profile materializer stderr" >&2; exit 2; }

cp "${RUN_ROOT}/profile/fixture_manifest.json" "${RUN_ROOT}/fixture_manifest.json"
{
  echo 'component_volume_profile_status=PASS_PF_ELASTIC_MULTI_PARTICLE_COMPONENT_VOLUME_PROFILE_V5'
  printf 'fixture_manifest_sha256=%s\n' "$(sha256sum "${RUN_ROOT}/profile/fixture_manifest.json" | awk '{print $1}')"
  printf 'source_tree_sha256=%s\n' "$(sha256sum "${RUN_ROOT}/provenance/source_files.sha256" | awk '{print $1}')"
  echo 'phi_Y_joint_relaxation=true'
  echo 'per_particle_h_volume_fixed=true'
  echo 'inter_component_volume_exchange=false'
  printf 'component_lambda_relaxation=%s\n' "${COMPONENT_LAMBDA_RELAXATION}"
  printf 'rms_dphi_direct_threshold=%s\n' "${RMS_DPHI_TOL}"
  printf 'rms_res_projected_kkt_threshold=%s\n' "${RMS_RES_TOL}"
  echo 'physical_parameters_changed=false'
  echo 'full_6h_to_48h_production_started=false'
} | tee "${RUN_ROOT}/status.txt"
