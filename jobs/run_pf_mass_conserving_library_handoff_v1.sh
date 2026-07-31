#!/usr/bin/env bash
set -euo pipefail

# Short, non-overwriting workstation qualification.  This runner never starts
# the 6--48 h production trajectory and never invokes a minimizer.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
FIXTURE_ROOT="${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
DYNAMIC_OVERRIDE_FILE="${DYNAMIC_OVERRIDE_FILE:-${SOURCE_ROOT}/data/qualification/pf_elastic_target_profile_v1/physical_override_T380_dx1nm_lambda4nm_dynamic_dt0p02.json}"
DT_CODE="${DT_CODE:-0.02}"
REFINED_DT_CODE="${REFINED_DT_CODE:-0.01}"
STEPS="${STEPS:-64}"
REFINED_STEPS="${REFINED_STEPS:-128}"
INITIAL_STATE_CLASS="MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
LIBRARY_SHA256="58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe"

[[ ! -e "${RUN_ROOT}" ]] || {
  echo "[fatal] refusing to overwrite RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
}
for path in \
  "${SOURCE_ROOT}/main_cuda" \
  "${SOURCE_ROOT}/scripts/qualify_pf_elastic_multi_particle_6h_dynamics_v2.py" \
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
python3 - "${FIXTURE_ROOT}/fixture_manifest.json" "${RUN_ROOT}/provenance/fixture_preflight.json" "${INITIAL_STATE_CLASS}" "${LIBRARY_SHA256}" <<'PY'
import hashlib,json,math,pathlib,sys
path=pathlib.Path(sys.argv[1]); root=path.parent; m=json.loads(path.read_text())
if m.get("schema")!="PF_MASS_CONSERVING_LIBRARY_HANDOFF_MANIFEST_V1": raise SystemExit("wrong fixture schema")
if m.get("initial_state_class")!=sys.argv[3]: raise SystemExit("wrong initial-state class")
if m.get("profile_library_manifest_sha256")!=sys.argv[4]: raise SystemExit("wrong library identity")
if m.get("validation_only") is not True: raise SystemExit("fixture is not validation-only")
for key,value in {
 "common_multi_particle_equilibrium_required":False,
 "common_multi_particle_equilibrium_claim":False,
 "initial_relaxation_is_physical_evolution":True,
 "particle_profiles_locked_after_t0":False,
 "full_field_xB_dt_MAE_blocking":False,
}.items():
 if m.get(key) is not value: raise SystemExit("wrong semantic field: "+key)
if m.get("assembly_contract",{}).get("optimizer_invoked") is not False: raise SystemExit("optimizer is prohibited")
if any(m["physical_contract"].get(k) for k in ("GP_enabled","GP_birth_enabled","GP_release_enabled","external_source_enabled","new_beta_nucleation_enabled")): raise SystemExit("forbidden path enabled")
for name in ("phi","xB_alpha","C_B_tot","Y","dY_dt_prev","delta_C_relaxation_total"):
 row=m["fields"][name]; p=root/row["path"]
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row["sha256"]: raise SystemExit("fixture hash mismatch: "+name)
inventory=m["initial_canonical_inventory"]
if inventory.get("status")!="PASS_MACHINE_PRECISION" or float(inventory["field_relative_error"])>1e-14: raise SystemExit("initial inventory does not close")
required=("particle_id","library_entry_id","registered_radius_nm","library_entry_sha256","center","orientation","phi_profile_hash","delta_C_relaxation_hash","effective_h_volume","canonical_particle_inventory")
rows=m.get("particle_library_mappings",[])
if not rows or any(any(k not in row for k in required) for row in rows): raise SystemExit("incomplete particle-library mapping")
pathlib.Path(sys.argv[2]).write_text(json.dumps({
 "fixture_manifest_sha256":hashlib.sha256(path.read_bytes()).hexdigest(),
 "profile_library_manifest_sha256":m["profile_library_manifest_sha256"],
 "particle_count":len(rows),
 "target_global_inventory":m["target_global_inventory"],
 "initial_canonical_inventory":inventory,
},indent=2,sort_keys=True)+"\n")
PY

SOURCE_FILES=(
  main_cuda.cu cuda_kernels.cu cuda_kernels.h cuda_common.cu cuda_common.h
  pf_params.h pf_zero_mode_checkpoint.cpp pf_zero_mode_checkpoint.h
  Unit_Psedobinary.py
  scripts/analyze_pf_zero_mode_checkpoints.py
  scripts/qualify_pf_elastic_multi_particle_6h_dynamics_v2.py
  jobs/run_pf_mass_conserving_library_handoff_v1.sh
)
for relative in "${SOURCE_FILES[@]}"; do
  [[ -f "${SOURCE_ROOT}/${relative}" ]] || {
    echo "[fatal] missing runtime source identity member: ${relative}" >&2
    exit 2
  }
  printf '%s  %s\n' "$(sha256sum "${SOURCE_ROOT}/${relative}" | awk '{print $1}')" "${relative}"
done | LC_ALL=C sort >"${RUN_ROOT}/provenance/runtime_source_files.sha256"
printf 'runtime_source_tree_sha256=%s\nruntime_binary_sha256=%s\nfixture_manifest_sha256=%s\nprofile_library_manifest_sha256=%s\n' \
  "$(sha256sum "${RUN_ROOT}/provenance/runtime_source_files.sha256" | awk '{print $1}')" \
  "$(sha256sum "${SOURCE_ROOT}/main_cuda" | awk '{print $1}')" \
  "${FIXTURE_SHA256}" "${LIBRARY_SHA256}" \
  >"${RUN_ROOT}/provenance/runtime_identity.txt"

export PHYSICAL_OVERRIDE_FILE="${DYNAMIC_OVERRIDE_FILE}" TEMP_C=380
PARAM02="$(site_generate_pf_param_file "mass_conserving_library_handoff_v1_dt0p02")"
cp "${PARAM02}" "${RUN_ROOT}/provenance/pf_input_dt0p02.params"
PARAM01="${RUN_ROOT}/provenance/pf_input_dt0p01.params"
sed "s/^dt=.*/dt=${REFINED_DT_CODE}/" "${PARAM02}" >"${PARAM01}"
cp "${FIXTURE_ROOT}/init_meta.json" "${RUN_ROOT}/provenance/init_meta_dt0p02.json"
python3 - "${FIXTURE_ROOT}/init_meta.json" "${RUN_ROOT}/provenance/init_meta_dt0p01.json" "${REFINED_DT_CODE}" <<'PY'
import json,pathlib,sys
x=json.load(open(sys.argv[1]))
x["dt_recommended"]=float(sys.argv[3])
pathlib.Path(sys.argv[2]).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
PY
GRID_N="$(python3 - "${FIXTURE_ROOT}/fixture_manifest.json" <<'PY'
import json,sys
g=json.load(open(sys.argv[1]))["grid"]
assert g["Nx"]==g["Ny"]==g["Nz"]
print(g["Nx"])
PY
)"

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
  local label="$1" dt="$2" steps="$3" params="$4" meta="$5" checkpoint="$6"
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
    --pf-checkpoint-every "${steps}" \
    --pf-checkpoint-path "${checkpoint}" \
    --init-case-tag mass_conserving_library_handoff_v1 \
    >"${root}/stdout.log" 2>"${root}/stderr.log"
  [[ ! -s "${root}/stderr.log" ]] || {
    echo "[fatal] non-empty stderr: ${label}" >&2
    exit 2
  }
  grep -q '^PF_ZERO_MODE_FINAL_AUDIT status=PASS ' "${root}/stdout.log"
}

run_fresh one_step "${DT_CODE}" 1 \
  "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
  "${RUN_ROOT}/provenance/init_meta_dt0p02.json" \
  "${RUN_ROOT}/one_step/final.chk"
run_fresh continuous "${DT_CODE}" "${STEPS}" \
  "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
  "${RUN_ROOT}/provenance/init_meta_dt0p02.json" \
  "${RUN_ROOT}/continuous/final.chk"
run_fresh restart_first_half "${DT_CODE}" "$((STEPS / 2))" \
  "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
  "${RUN_ROOT}/provenance/init_meta_dt0p02.json" \
  "${RUN_ROOT}/restart_first_half/half.chk"

mkdir -p "${RUN_ROOT}/restart_second_half/results"
CUDA_STO_RESULTS_ROOT="${RUN_ROOT}/restart_second_half/results" \
  CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
  "${SOURCE_ROOT}/main_cuda" "${GRID_N}" "${GRID_N}" "${GRID_N}" \
  "${DT_CODE}" "${STEPS}" "${STEPS}" 1 1 \
  --pf-param-file "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
  --mode dynamics \
  --pf-restart-from "${RUN_ROOT}/restart_first_half/half.chk" \
  "${ZERO_MODE_FLAGS[@]}" "${IDENTITY_FLAGS[@]}" \
  --pf-checkpoint-every "${STEPS}" \
  --pf-checkpoint-path "${RUN_ROOT}/restart_second_half/final.chk" \
  --init-case-tag mass_conserving_library_handoff_v1 \
  >"${RUN_ROOT}/restart_second_half/stdout.log" \
  2>"${RUN_ROOT}/restart_second_half/stderr.log"
[[ ! -s "${RUN_ROOT}/restart_second_half/stderr.log" ]] || {
  echo "[fatal] non-empty stderr: restart_second_half" >&2
  exit 2
}

run_fresh refined "${REFINED_DT_CODE}" "${REFINED_STEPS}" \
  "${RUN_ROOT}/provenance/pf_input_dt0p01.params" \
  "${RUN_ROOT}/provenance/init_meta_dt0p01.json" \
  "${RUN_ROOT}/refined/final.chk"

run_mismatch_rejection() {
  local label="$1" fixture_sha="$2" library_sha="$3"
  local root="${RUN_ROOT}/${label}"
  mkdir -p "${root}/results"
  set +e
  CUDA_STO_RESULTS_ROOT="${root}/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
    "${SOURCE_ROOT}/main_cuda" "${GRID_N}" "${GRID_N}" "${GRID_N}" \
    "${DT_CODE}" "${STEPS}" "${STEPS}" 1 1 \
    --pf-param-file "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
    --mode dynamics \
    --pf-restart-from "${RUN_ROOT}/restart_first_half/half.chk" \
    "${ZERO_MODE_FLAGS[@]}" \
    --pf-initial-state-class "${INITIAL_STATE_CLASS}" \
    --pf-fixture-manifest-sha256 "${fixture_sha}" \
    --pf-profile-library-manifest-sha256 "${library_sha}" \
    --init-case-tag mass_conserving_library_handoff_v1 \
    >"${root}/stdout.log" 2>"${root}/stderr.log"
  local rc=$?
  set -e
  [[ "${rc}" -ne 0 ]] || {
    echo "[fatal] provenance mismatch was not rejected: ${label}" >&2
    exit 2
  }
  grep -q 'checkpoint provenance mismatch' "${root}/stderr.log"
}

run_mismatch_rejection fixture_mismatch \
  "0${FIXTURE_SHA256:1}" "${LIBRARY_SHA256}"
run_mismatch_rejection library_mismatch \
  "${FIXTURE_SHA256}" "0${LIBRARY_SHA256:1}"

python3 "${SOURCE_ROOT}/scripts/qualify_pf_elastic_multi_particle_6h_dynamics_v2.py" \
  --fixture-manifest "${FIXTURE_ROOT}/fixture_manifest.json" \
  --initial-probe-checkpoint "${RUN_ROOT}/one_step/final.chk" \
  --continuous-checkpoint "${RUN_ROOT}/continuous/final.chk" \
  --restart-checkpoint "${RUN_ROOT}/restart_second_half/final.chk" \
  --refined-checkpoint "${RUN_ROOT}/refined/final.chk" \
  --initial-probe-stdout "${RUN_ROOT}/one_step/stdout.log" \
  --continuous-stdout "${RUN_ROOT}/continuous/stdout.log" \
  --restart-stdout "${RUN_ROOT}/restart_second_half/stdout.log" \
  --refined-stdout "${RUN_ROOT}/refined/stdout.log" \
  --out "${RUN_ROOT}/audit" \
  >"${RUN_ROOT}/audit.stdout" 2>"${RUN_ROOT}/audit.stderr"
[[ ! -s "${RUN_ROOT}/audit.stderr" ]] || {
  echo "[fatal] analyzer stderr" >&2
  exit 2
}
grep -q '^handoff_status=PASS_MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1$' \
  "${RUN_ROOT}/audit/final_terminal_output.txt"
cp "${RUN_ROOT}/audit/final_terminal_output.txt" "${RUN_ROOT}/status.txt"
printf 'fixture_mismatch_rejection=PASS\nlibrary_mismatch_rejection=PASS\n' \
  >>"${RUN_ROOT}/status.txt"
cat "${RUN_ROOT}/status.txt"
