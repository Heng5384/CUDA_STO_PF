#!/usr/bin/env bash
set -euo pipefail

# Non-overwriting workstation-only production-dt selection on the frozen
# conditional handoff fixture.  This is a short 6 h + ~254 s qualification,
# not a 6--48 h production trajectory.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
FIXTURE_ROOT="${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
DYNAMIC_OVERRIDE_FILE="${DYNAMIC_OVERRIDE_FILE:-${SOURCE_ROOT}/data/qualification/pf_elastic_target_profile_v1/physical_override_T380_dx1nm_lambda4nm_dynamic_dt0p02.json}"
INITIAL_STATE_CLASS="MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
LIBRARY_SHA256="${LIBRARY_SHA256:-58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe}"
EXPECTED_FIXTURE_SCHEMA="${EXPECTED_FIXTURE_SCHEMA:-PF_MASS_CONSERVING_LIBRARY_HANDOFF_MANIFEST_V1}"
EXPECTED_OPTIMIZER_INVOKED="${EXPECTED_OPTIMIZER_INVOKED:-false}"

[[ ! -e "${RUN_ROOT}" ]] || {
  echo "[fatal] refusing to overwrite RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
}
for path in \
  "${SOURCE_ROOT}/main_cuda" \
  "${SOURCE_ROOT}/scripts/qualify_pf_production_dt_observables_v1.py" \
  "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${FIXTURE_ROOT}/phi.raw.f64" \
  "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
  "${FIXTURE_ROOT}/init_meta.json" \
  "${DYNAMIC_OVERRIDE_FILE}"; do
  [[ -f "${path}" ]] || {
    echo "[fatal] missing required input: ${path}" >&2
    exit 2
  }
done

mkdir -p "${RUN_ROOT}/provenance"
FIXTURE_SHA256="$(sha256sum "${FIXTURE_ROOT}/fixture_manifest.json" | awk '{print $1}')"
GRID_N="$(python3 - "${FIXTURE_ROOT}/fixture_manifest.json" <<'PY'
import json,sys
g=json.load(open(sys.argv[1]))["grid"]
assert g["Nx"]==g["Ny"]==g["Nz"]
print(g["Nx"])
PY
)"
python3 - "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${RUN_ROOT}/provenance/fixture_preflight.json" "${LIBRARY_SHA256}" \
  "${EXPECTED_FIXTURE_SCHEMA}" "${EXPECTED_OPTIMIZER_INVOKED}" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); m=json.loads(p.read_text())
if m.get("schema")!=sys.argv[4]: raise SystemExit("wrong fixture schema")
if m.get("initial_state_class")!="MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1": raise SystemExit("wrong initial-state class")
if m.get("validation_only") is not True: raise SystemExit("fixture is not validation-only")
if m.get("profile_library_manifest_sha256")!=sys.argv[3]: raise SystemExit("wrong profile library")
if m.get("full_field_xB_dt_MAE_blocking") is not False: raise SystemExit("wrong dt policy")
if any(m["physical_contract"].get(k) for k in ("GP_enabled","GP_birth_enabled","GP_release_enabled","external_source_enabled","new_beta_nucleation_enabled")): raise SystemExit("forbidden path enabled")
expected_optimizer = sys.argv[5].lower() == "true"
if m.get("assembly_contract",{}).get("optimizer_invoked") is not expected_optimizer: raise SystemExit("unexpected optimizer provenance")
pathlib.Path(sys.argv[2]).write_text(json.dumps({
 "fixture_manifest_sha256":hashlib.sha256(p.read_bytes()).hexdigest(),
 "initial_state_class":m["initial_state_class"],
 "profile_library_manifest_sha256":m["profile_library_manifest_sha256"],
 "target_global_inventory":m["target_global_inventory"],
},indent=2,sort_keys=True)+"\n")
PY

SOURCE_FILES=(
  main_cuda.cu cuda_kernels.cu cuda_kernels.h cuda_common.cu cuda_common.h
  pf_params.h pf_zero_mode_checkpoint.cpp pf_zero_mode_checkpoint.h
  Unit_Psedobinary.py
  scripts/analyze_pf_zero_mode_checkpoints.py
  scripts/qualify_pf_elastic_multi_particle_6h_dynamics_v1.py
  scripts/qualify_pf_production_dt_observables_v1.py
  jobs/run_pf_production_dt_selection_v1.sh
)
for relative in "${SOURCE_FILES[@]}"; do
  [[ -f "${SOURCE_ROOT}/${relative}" ]] || {
    echo "[fatal] missing source identity member: ${relative}" >&2
    exit 2
  }
  printf '%s  %s\n' \
    "$(sha256sum "${SOURCE_ROOT}/${relative}" | awk '{print $1}')" \
    "${relative}"
done | LC_ALL=C sort >"${RUN_ROOT}/provenance/runtime_source_files.sha256"
printf 'runtime_source_tree_sha256=%s\nruntime_binary_sha256=%s\nfixture_manifest_sha256=%s\nprofile_library_manifest_sha256=%s\n' \
  "$(sha256sum "${RUN_ROOT}/provenance/runtime_source_files.sha256" | awk '{print $1}')" \
  "$(sha256sum "${SOURCE_ROOT}/main_cuda" | awk '{print $1}')" \
  "${FIXTURE_SHA256}" "${LIBRARY_SHA256}" \
  >"${RUN_ROOT}/provenance/runtime_identity.txt"

export PHYSICAL_OVERRIDE_FILE="${DYNAMIC_OVERRIDE_FILE}" TEMP_C=380
PARAM02="$(site_generate_pf_param_file "production_dt_selection_v1_dt0p02")"
cp "${PARAM02}" "${RUN_ROOT}/provenance/pf_input_dt0p02.params"
for item in "dt0p01 0.01" "dt0p005 0.005"; do
  read -r label dt <<<"${item}"
  sed "s/^dt=.*/dt=${dt}/" "${PARAM02}" \
    >"${RUN_ROOT}/provenance/pf_input_${label}.params"
done
T_REAL_UNIT_S="$(awk -F= '$1=="t_real_unit"{print $2}' "${PARAM02}")"
cp "${FIXTURE_ROOT}/init_meta.json" "${RUN_ROOT}/provenance/init_meta_dt0p02.json"
python3 - "${FIXTURE_ROOT}/init_meta.json" "${RUN_ROOT}/provenance" <<'PY'
import json,pathlib,sys
source=json.load(open(sys.argv[1])); root=pathlib.Path(sys.argv[2])
for label,dt in (("dt0p01",0.01),("dt0p005",0.005)):
 value=dict(source); value["dt_recommended"]=dt
 (root/f"init_meta_{label}.json").write_text(json.dumps(value,indent=2,sort_keys=True)+"\n")
PY

IDENTITY_FLAGS=(
  --pf-initial-state-class "${INITIAL_STATE_CLASS}"
  --pf-fixture-manifest-sha256 "${FIXTURE_SHA256}"
  --pf-profile-library-manifest-sha256 "${LIBRARY_SHA256}"
)
ZERO_MODE_FLAGS=(
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1
  --pf-zero-mode-tol-rel 1e-12
  --pf-zero-mode-max-iter 24
)

run_fresh() {
  local label="$1" dt="$2" steps="$3" params="$4" meta="$5"
  local root="${RUN_ROOT}/${label}"
  mkdir -p "${root}/results"
  CUDA_STO_RESULTS_ROOT="${root}/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
    "${SOURCE_ROOT}/main_cuda" "${GRID_N}" "${GRID_N}" "${GRID_N}" \
    "${dt}" "${steps}" "${steps}" 1 1 \
    --pf-param-file "${params}" --mode dynamics \
    --init-mode raw_fields \
    --init-phi-raw "${FIXTURE_ROOT}/phi.raw.f64" \
    --init-xB-raw "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
    --init-meta "${meta}" \
    "${ZERO_MODE_FLAGS[@]}" "${IDENTITY_FLAGS[@]}" \
    --enable-dynamics-mass-diagnostics \
    --dynamics-mass-diag-interval "${steps}" \
    --pf-checkpoint-every "${steps}" \
    --pf-checkpoint-path "${root}/final.chk" \
    --init-case-tag "production_dt_${label}" \
    >"${root}/stdout.log" 2>"${root}/stderr.log"
  [[ ! -s "${root}/stderr.log" ]] || {
    echo "[fatal] non-empty stderr: ${label}" >&2
    exit 2
  }
  grep -q '^PF_ZERO_MODE_FINAL_AUDIT status=PASS ' "${root}/stdout.log"
  local mass_diagnostics
  mass_diagnostics="$(
    find "${root}/results" -type f -name dynamics_mass_diagnostics.csv \
      -print -quit
  )"
  [[ -n "${mass_diagnostics}" && -f "${mass_diagnostics}" ]] || {
    echo "[fatal] missing dynamics mass diagnostics: ${label}" >&2
    exit 2
  }
  cp "${mass_diagnostics}" "${root}/dynamics_mass_diagnostics.csv"
}

run_fresh dt0p02 0.02 256 \
  "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
  "${RUN_ROOT}/provenance/init_meta_dt0p02.json"
run_fresh dt0p01 0.01 512 \
  "${RUN_ROOT}/provenance/pf_input_dt0p01.params" \
  "${RUN_ROOT}/provenance/init_meta_dt0p01.json"
run_fresh dt0p005 0.005 1024 \
  "${RUN_ROOT}/provenance/pf_input_dt0p005.params" \
  "${RUN_ROOT}/provenance/init_meta_dt0p005.json"

CASE_ARGS=(
  --case dt0p02 "${RUN_ROOT}/dt0p02/final.chk" \
    "${RUN_ROOT}/dt0p02/stdout.log" "${RUN_ROOT}/dt0p02/stderr.log" \
    "${RUN_ROOT}/dt0p02/dynamics_mass_diagnostics.csv"
  --case dt0p01 "${RUN_ROOT}/dt0p01/final.chk" \
    "${RUN_ROOT}/dt0p01/stdout.log" "${RUN_ROOT}/dt0p01/stderr.log" \
    "${RUN_ROOT}/dt0p01/dynamics_mass_diagnostics.csv"
  --case dt0p005 "${RUN_ROOT}/dt0p005/final.chk" \
    "${RUN_ROOT}/dt0p005/stdout.log" "${RUN_ROOT}/dt0p005/stderr.log" \
    "${RUN_ROOT}/dt0p005/dynamics_mass_diagnostics.csv"
)

python3 "${SOURCE_ROOT}/scripts/qualify_pf_production_dt_observables_v1.py" \
  --fixture-manifest "${FIXTURE_ROOT}/fixture_manifest.json" \
  --fixture-schema "${EXPECTED_FIXTURE_SCHEMA}" \
  "${CASE_ARGS[@]}" --t-real-unit-s "${T_REAL_UNIT_S}" \
  --out "${RUN_ROOT}/preliminary" \
  >"${RUN_ROOT}/preliminary.stdout" 2>"${RUN_ROOT}/preliminary.stderr"
[[ ! -s "${RUN_ROOT}/preliminary.stderr" ]] || {
  echo "[fatal] preliminary analyzer stderr" >&2
  exit 2
}
grep -q '^final_status=PASS_PRODUCTION_DT_OBSERVABLE_CONVERGENCE_V1$' \
  "${RUN_ROOT}/preliminary/final_terminal_output.txt"

read -r SELECTED_LABEL SELECTED_DT SELECTED_STEPS <<<"$(python3 - "${RUN_ROOT}/preliminary/production_dt_audit.json" <<'PY'
import json,sys
m=json.load(open(sys.argv[1])); label=m["selected_label"]; row=m["cases"][label]
print(label,row["dt_code"],row["step"])
PY
)"
SELECTED_PARAMS="${RUN_ROOT}/provenance/pf_input_${SELECTED_LABEL}.params"
SELECTED_META="${RUN_ROOT}/provenance/init_meta_${SELECTED_LABEL}.json"
HALF_STEPS="$((SELECTED_STEPS / 2))"

run_fresh restart_first_half "${SELECTED_DT}" "${HALF_STEPS}" \
  "${SELECTED_PARAMS}" "${SELECTED_META}"
mv "${RUN_ROOT}/restart_first_half/final.chk" \
  "${RUN_ROOT}/restart_first_half/half.chk"

mkdir -p "${RUN_ROOT}/restart_second_half/results"
CUDA_STO_RESULTS_ROOT="${RUN_ROOT}/restart_second_half/results" \
  CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
  "${SOURCE_ROOT}/main_cuda" "${GRID_N}" "${GRID_N}" "${GRID_N}" \
  "${SELECTED_DT}" "${SELECTED_STEPS}" "${SELECTED_STEPS}" 1 1 \
  --pf-param-file "${SELECTED_PARAMS}" --mode dynamics \
  --pf-restart-from "${RUN_ROOT}/restart_first_half/half.chk" \
  "${ZERO_MODE_FLAGS[@]}" "${IDENTITY_FLAGS[@]}" \
  --enable-dynamics-mass-diagnostics \
  --dynamics-mass-diag-interval "${SELECTED_STEPS}" \
  --pf-checkpoint-every "${SELECTED_STEPS}" \
  --pf-checkpoint-path "${RUN_ROOT}/restart_second_half/final.chk" \
  --init-case-tag "production_dt_restart_${SELECTED_LABEL}" \
  >"${RUN_ROOT}/restart_second_half/stdout.log" \
  2>"${RUN_ROOT}/restart_second_half/stderr.log"
[[ ! -s "${RUN_ROOT}/restart_second_half/stderr.log" ]] || {
  echo "[fatal] restart stderr is non-empty" >&2
  exit 2
}
grep -q '^PF_ZERO_MODE_FINAL_AUDIT status=PASS ' \
  "${RUN_ROOT}/restart_second_half/stdout.log"

python3 "${SOURCE_ROOT}/scripts/qualify_pf_production_dt_observables_v1.py" \
  --fixture-manifest "${FIXTURE_ROOT}/fixture_manifest.json" \
  --fixture-schema "${EXPECTED_FIXTURE_SCHEMA}" \
  "${CASE_ARGS[@]}" --t-real-unit-s "${T_REAL_UNIT_S}" \
  --restart-checkpoint "${RUN_ROOT}/restart_second_half/final.chk" \
  --out "${RUN_ROOT}/final_audit" \
  >"${RUN_ROOT}/final_audit.stdout" 2>"${RUN_ROOT}/final_audit.stderr"
[[ ! -s "${RUN_ROOT}/final_audit.stderr" ]] || {
  echo "[fatal] final analyzer stderr" >&2
  exit 2
}
grep -q '^final_status=PASS_PRODUCTION_DT_OBSERVABLE_CONVERGENCE_V1$' \
  "${RUN_ROOT}/final_audit/final_terminal_output.txt"
cp "${RUN_ROOT}/final_audit/final_terminal_output.txt" "${RUN_ROOT}/status.txt"
cat "${RUN_ROOT}/status.txt"
