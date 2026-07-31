#!/usr/bin/env bash
set -euo pipefail

# Non-overwriting E2 raw-field handoff/dt/restart qualification for the V2
# phase-consistent composition fixture.  This runner never starts production.
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

[[ ! -e "${RUN_ROOT}" ]] || { echo "[fatal] refusing to overwrite RUN_ROOT: ${RUN_ROOT}" >&2; exit 2; }
for path in "${SOURCE_ROOT}/main_cuda" "${SOURCE_ROOT}/scripts/qualify_pf_elastic_multi_particle_6h_dynamics_v2.py" "${FIXTURE_ROOT}/fixture_manifest.json" "${FIXTURE_ROOT}/phi.raw.f64" "${FIXTURE_ROOT}/xB_alpha.raw.f64" "${FIXTURE_ROOT}/init_meta.json" "${DYNAMIC_OVERRIDE_FILE}"; do
  [[ -f "${path}" ]] || { echo "[fatal] missing required input: ${path}" >&2; exit 2; }
done

mkdir -p "${RUN_ROOT}/provenance"
python3 - "${FIXTURE_ROOT}/fixture_manifest.json" "${RUN_ROOT}/provenance/fixture_preflight.json" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).parent; m=json.loads(pathlib.Path(sys.argv[1]).read_text())
if m.get("schema") != "PF_ELASTIC_MULTI_PARTICLE_6H_FIXTURE_V2": raise SystemExit("wrong fixture schema")
if m.get("fixture_kind") != "E2" or m.get("validation_only") is not True: raise SystemExit("only validation-only E2 enters handoff")
if m.get("combination",{}).get("composition_contract") != "PHASE_CONSISTENT_DELTA_X_ALPHA_V2": raise SystemExit("wrong composition contract")
if any(m["physical_contract"].get(k) for k in ("GP_enabled","GP_birth_enabled","GP_release_enabled","external_source_enabled","new_beta_nucleation_enabled")): raise SystemExit("forbidden path enabled")
for name in ("phi","xB_alpha","C_B_tot","Y","dY_dt_prev"):
 row=m["fields"][name]; p=root/row["path"]
 if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row["sha256"]: raise SystemExit("fixture hash mismatch: "+name)
pathlib.Path(sys.argv[2]).write_text(json.dumps({"fixture_manifest_sha256":hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(),"fixture_particle_count":len(m["particles"]),"fixture_grid":m["grid"],"fixture_inventory":m["inventory"],"composition_contract":m["combination"]["composition_contract"]},indent=2,sort_keys=True)+"\n")
PY

SOURCE_FILES=(main_cuda.cu cuda_kernels.cu cuda_kernels.h cuda_common.cu cuda_common.h pf_params.h pf_zero_mode_checkpoint.cpp pf_zero_mode_checkpoint.h Unit_Psedobinary.py scripts/materialize_pf_elastic_target_profile_v1.py scripts/assemble_pf_elastic_target_profile_library_v1.py scripts/test_pf_elastic_target_profile_v1.py jobs/run_pf_elastic_target_profile_library_v1.sh)
for relative in "${SOURCE_FILES[@]}"; do
  [[ -f "${SOURCE_ROOT}/${relative}" ]] || { echo "[fatal] missing source identity member: ${relative}" >&2; exit 2; }
  printf '%s  %s\n' "$(sha256sum "${SOURCE_ROOT}/${relative}" | awk '{print $1}')" "${relative}"
done | LC_ALL=C sort >"${RUN_ROOT}/provenance/runtime_source_files.sha256"
RUNTIME_SOURCE_TREE_SHA256="$(sha256sum "${RUN_ROOT}/provenance/runtime_source_files.sha256" | awk '{print $1}')"
FIXTURE_SOURCE_TREE_SHA256="$(python3 - "${FIXTURE_ROOT}/fixture_manifest.json" <<'PY'
import json,sys
print(json.load(open(sys.argv[1]))["selected_library"]["source_tree_sha256"])
PY
)"
[[ "${RUNTIME_SOURCE_TREE_SHA256}" == "${FIXTURE_SOURCE_TREE_SHA256}" ]] || { echo "[fatal] source tree mismatch" >&2; exit 2; }
printf 'runtime_source_tree_sha256=%s\nfixture_source_tree_sha256=%s\nruntime_binary_sha256=%s\n' "${RUNTIME_SOURCE_TREE_SHA256}" "${FIXTURE_SOURCE_TREE_SHA256}" "$(sha256sum "${SOURCE_ROOT}/main_cuda" | awk '{print $1}')" >"${RUN_ROOT}/provenance/runtime_identity.txt"

export PHYSICAL_OVERRIDE_FILE="${DYNAMIC_OVERRIDE_FILE}" TEMP_C=380
PARAM02="$(site_generate_pf_param_file "elastic_multi_particle_E2_v2_dt0p02")"
cp "${PARAM02}" "${RUN_ROOT}/provenance/pf_input_dt0p02.params"
PARAM01="${RUN_ROOT}/provenance/pf_input_dt0p01.params"
sed "s/^dt=.*/dt=${REFINED_DT_CODE}/" "${PARAM02}" >"${PARAM01}"
cp "${FIXTURE_ROOT}/init_meta.json" "${RUN_ROOT}/provenance/init_meta_dt0p02.json"
python3 - "${FIXTURE_ROOT}/init_meta.json" "${RUN_ROOT}/provenance/init_meta_dt0p01.json" "${REFINED_DT_CODE}" <<'PY'
import json,pathlib,sys
x=json.load(open(sys.argv[1])); x["dt_recommended"]=float(sys.argv[3]); pathlib.Path(sys.argv[2]).write_text(json.dumps(x,indent=2,sort_keys=True)+"\n")
PY
GRID_N="$(python3 - "${FIXTURE_ROOT}/fixture_manifest.json" <<'PY'
import json,sys
g=json.load(open(sys.argv[1]))["grid"]; assert g["Nx"]==g["Ny"]==g["Nz"]; print(g["Nx"])
PY
)"

run_fresh() {
  local label="$1" dt="$2" steps="$3" params="$4" meta="$5" checkpoint="$6" root="${RUN_ROOT}/$1"
  mkdir -p "${root}/results"
  CUDA_STO_RESULTS_ROOT="${root}/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 "${SOURCE_ROOT}/main_cuda" "${GRID_N}" "${GRID_N}" "${GRID_N}" "${dt}" "${steps}" "${steps}" 1 1 \
    --pf-param-file "${params}" --mode dynamics --init-mode raw_fields --init-phi-raw "${FIXTURE_ROOT}/phi.raw.f64" --init-xB-raw "${FIXTURE_ROOT}/xB_alpha.raw.f64" --init-meta "${meta}" \
    --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 --pf-zero-mode-tol-rel 1e-12 --pf-zero-mode-max-iter 24 \
    --pf-checkpoint-every "${steps}" --pf-checkpoint-path "${checkpoint}" --init-case-tag elastic_multi_particle_E2_v2 >"${root}/stdout.log" 2>"${root}/stderr.log"
  [[ ! -s "${root}/stderr.log" ]] || { echo "[fatal] non-empty stderr: ${label}" >&2; exit 2; }
  grep -q '^PF_ZERO_MODE_FINAL_AUDIT status=PASS ' "${root}/stdout.log"
}

run_fresh one_step "${DT_CODE}" 1 "${RUN_ROOT}/provenance/pf_input_dt0p02.params" "${RUN_ROOT}/provenance/init_meta_dt0p02.json" "${RUN_ROOT}/one_step/final.chk"
run_fresh continuous "${DT_CODE}" "${STEPS}" "${RUN_ROOT}/provenance/pf_input_dt0p02.params" "${RUN_ROOT}/provenance/init_meta_dt0p02.json" "${RUN_ROOT}/continuous/final.chk"
run_fresh restart_first_half "${DT_CODE}" "$((STEPS / 2))" "${RUN_ROOT}/provenance/pf_input_dt0p02.params" "${RUN_ROOT}/provenance/init_meta_dt0p02.json" "${RUN_ROOT}/restart_first_half/half.chk"
mkdir -p "${RUN_ROOT}/restart_second_half/results"
CUDA_STO_RESULTS_ROOT="${RUN_ROOT}/restart_second_half/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 "${SOURCE_ROOT}/main_cuda" "${GRID_N}" "${GRID_N}" "${GRID_N}" "${DT_CODE}" "${STEPS}" "${STEPS}" 1 1 \
  --pf-param-file "${RUN_ROOT}/provenance/pf_input_dt0p02.params" --mode dynamics --pf-restart-from "${RUN_ROOT}/restart_first_half/half.chk" \
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 --pf-zero-mode-tol-rel 1e-12 --pf-zero-mode-max-iter 24 \
  --pf-checkpoint-every "${STEPS}" --pf-checkpoint-path "${RUN_ROOT}/restart_second_half/final.chk" --init-case-tag elastic_multi_particle_E2_v2 >"${RUN_ROOT}/restart_second_half/stdout.log" 2>"${RUN_ROOT}/restart_second_half/stderr.log"
[[ ! -s "${RUN_ROOT}/restart_second_half/stderr.log" ]] || { echo "[fatal] non-empty stderr: restart_second_half" >&2; exit 2; }
run_fresh refined "${REFINED_DT_CODE}" "${REFINED_STEPS}" "${RUN_ROOT}/provenance/pf_input_dt0p01.params" "${RUN_ROOT}/provenance/init_meta_dt0p01.json" "${RUN_ROOT}/refined/final.chk"

python3 "${SOURCE_ROOT}/scripts/qualify_pf_elastic_multi_particle_6h_dynamics_v2.py" --fixture-manifest "${FIXTURE_ROOT}/fixture_manifest.json" --initial-probe-checkpoint "${RUN_ROOT}/one_step/final.chk" --continuous-checkpoint "${RUN_ROOT}/continuous/final.chk" --restart-checkpoint "${RUN_ROOT}/restart_second_half/final.chk" --refined-checkpoint "${RUN_ROOT}/refined/final.chk" --initial-probe-stdout "${RUN_ROOT}/one_step/stdout.log" --continuous-stdout "${RUN_ROOT}/continuous/stdout.log" --restart-stdout "${RUN_ROOT}/restart_second_half/stdout.log" --refined-stdout "${RUN_ROOT}/refined/stdout.log" --out "${RUN_ROOT}/audit" >"${RUN_ROOT}/audit.stdout" 2>"${RUN_ROOT}/audit.stderr"
[[ ! -s "${RUN_ROOT}/audit.stderr" ]] || { echo "[fatal] analyzer stderr" >&2; exit 2; }
grep -q '^handoff_status=PASS_CONSERVED_ELASTIC_MULTI_PARTICLE_HANDOFF_RESTART_DT_V2$' "${RUN_ROOT}/audit/final_terminal_output.txt"
cp "${RUN_ROOT}/audit/final_terminal_output.txt" "${RUN_ROOT}/status.txt"
cat "${RUN_ROOT}/status.txt"
