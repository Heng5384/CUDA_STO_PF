#!/usr/bin/env bash
set -euo pipefail

# Non-overwriting short dynamics qualification for a frozen V3 constrained
# common-matrix fixture. This never starts 6--48 h production.

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
FIXTURE_SCHEMA="${FIXTURE_SCHEMA:-PF_ELASTIC_MULTI_PARTICLE_COMMON_MATRIX_PROFILE_V3}"
FIXTURE_KIND="${FIXTURE_KIND:-E2_COMMON_MATRIX_CONSTRAINED_PROFILE}"
FINAL_PROFILE="${FINAL_PROFILE:-FULL_MODEL_ELASTIC_MASS_CONSTRAINED_MINIMIZATION_V3}"
HANDOFF_STATUS="${HANDOFF_STATUS:-PASS_CONSERVED_ELASTIC_MULTI_PARTICLE_HANDOFF_RESTART_DT_V3}"
CASE_TAG="${CASE_TAG:-elastic_multi_particle_common_matrix_v3}"
RUNNER_PROVENANCE_EXTRA_REL="${RUNNER_PROVENANCE_EXTRA_REL:-}"

[[ ! -e "${RUN_ROOT}" ]] || { echo "[fatal] refusing to overwrite RUN_ROOT: ${RUN_ROOT}" >&2; exit 2; }
for path in "${SOURCE_ROOT}/main_cuda" "${SOURCE_ROOT}/scripts/qualify_pf_elastic_multi_particle_6h_dynamics_v2.py" "${FIXTURE_ROOT}/fixture_manifest.json" "${FIXTURE_ROOT}/phi.raw.f64" "${FIXTURE_ROOT}/xB_alpha.raw.f64" "${FIXTURE_ROOT}/dY_dt_prev.raw.f64" "${FIXTURE_ROOT}/init_meta.json" "${DYNAMIC_OVERRIDE_FILE}"; do
  [[ -f "${path}" ]] || { echo "[fatal] missing required input: ${path}" >&2; exit 2; }
done

mkdir -p "${RUN_ROOT}/provenance"
python3 - "${FIXTURE_ROOT}/fixture_manifest.json" "${RUN_ROOT}/provenance/fixture_preflight.json" "${FIXTURE_SCHEMA}" "${FIXTURE_KIND}" "${FINAL_PROFILE}" <<'PY'
import hashlib,json,pathlib,sys
root=pathlib.Path(sys.argv[1]).parent
m=json.loads(pathlib.Path(sys.argv[1]).read_text())
if m.get('schema') != sys.argv[3]: raise SystemExit('wrong fixture schema')
if m.get('fixture_kind') != sys.argv[4]: raise SystemExit('wrong fixture kind')
if m.get('validation_only') is not True: raise SystemExit('fixture must remain validation-only')
if m.get('composition_contract',{}).get('final_profile') != sys.argv[5]: raise SystemExit('missing final-profile contract')
time_level=m.get('time_level_contract',{})
if time_level.get('dynamic_raw_history_required') is not True and time_level.get('dynamic_time_level_kind') != 'fresh_zero':
    raise SystemExit('missing explicit dynamic time-level handoff contract')
for name in ('phi','xB_alpha','C_B_tot','Y','dY_dt_prev'):
    row=m['fields'][name]; p=root/row['path']
    if not p.is_file() or hashlib.sha256(p.read_bytes()).hexdigest()!=row['sha256']:
        raise SystemExit('fixture hash mismatch: '+name)
pathlib.Path(sys.argv[2]).write_text(json.dumps({'fixture_manifest_sha256':hashlib.sha256(pathlib.Path(sys.argv[1]).read_bytes()).hexdigest(),'fixture_grid':m['grid'],'fixture_inventory':m['inventory'],'time_level_contract':m['time_level_contract'],'selected_library':m['selected_library']},indent=2,sort_keys=True)+'\n')
PY

SOURCE_FILES=(main_cuda.cu cuda_kernels.cu cuda_kernels.h cuda_common.cu cuda_common.h pf_params.h pf_zero_mode_checkpoint.cpp pf_zero_mode_checkpoint.h Unit_Psedobinary.py scripts/qualify_pf_elastic_multi_particle_6h_dynamics_v1.py scripts/qualify_pf_elastic_multi_particle_6h_dynamics_v2.py jobs/run_pf_elastic_multi_particle_common_matrix_qualification_v3.sh)
if [[ -n "${RUNNER_PROVENANCE_EXTRA_REL}" ]]; then
  SOURCE_FILES+=("${RUNNER_PROVENANCE_EXTRA_REL}")
fi
for relative in "${SOURCE_FILES[@]}"; do
  [[ -f "${SOURCE_ROOT}/${relative}" ]] || { echo "[fatal] source member missing: ${relative}" >&2; exit 2; }
  printf '%s  %s\n' "$(sha256sum "${SOURCE_ROOT}/${relative}" | awk '{print $1}')" "${relative}"
done | LC_ALL=C sort >"${RUN_ROOT}/provenance/runtime_source_files.sha256"
sha256sum "${SOURCE_ROOT}/main_cuda" "${DYNAMIC_OVERRIDE_FILE}" >"${RUN_ROOT}/provenance/runtime_inputs.sha256"

export PHYSICAL_OVERRIDE_FILE="${DYNAMIC_OVERRIDE_FILE}" TEMP_C=380
PARAM02="$(site_generate_pf_param_file 'elastic_multi_particle_common_matrix_v3_dt0p02')"
PARAM_PRODUCTION="${RUN_ROOT}/provenance/pf_input_production.params"
PARAM_REFINED="${RUN_ROOT}/provenance/pf_input_refined.params"
sed "s/^dt=.*/dt=${DT_CODE}/" "${PARAM02}" >"${PARAM_PRODUCTION}"
sed "s/^dt=.*/dt=${REFINED_DT_CODE}/" "${PARAM02}" >"${PARAM_REFINED}"
python3 - "${FIXTURE_ROOT}/init_meta.json" "${RUN_ROOT}/provenance/init_meta_production.json" "${DT_CODE}" <<'PY'
import json,pathlib,sys
x=json.load(open(sys.argv[1])); x['dt_recommended']=float(sys.argv[3]); pathlib.Path(sys.argv[2]).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
PY
python3 - "${FIXTURE_ROOT}/init_meta.json" "${RUN_ROOT}/provenance/init_meta_refined.json" "${REFINED_DT_CODE}" <<'PY'
import json,pathlib,sys
x=json.load(open(sys.argv[1])); x['dt_recommended']=float(sys.argv[3]); pathlib.Path(sys.argv[2]).write_text(json.dumps(x,indent=2,sort_keys=True)+'\n')
PY
GRID_N="$(python3 - "${FIXTURE_ROOT}/fixture_manifest.json" <<'PY'
import json,sys
g=json.load(open(sys.argv[1]))['grid']; assert g['Nx']==g['Ny']==g['Nz']; print(g['Nx'])
PY
)"
DYNAMIC_HISTORY_REQUIRED="$(python3 - "${FIXTURE_ROOT}/fixture_manifest.json" <<'PY'
import json,sys
print('1' if json.load(open(sys.argv[1]))['time_level_contract'].get('dynamic_raw_history_required') else '0')
PY
)"
if [[ "${DYNAMIC_HISTORY_REQUIRED}" == "1" ]]; then
  DYNAMIC_HISTORY_ARGS=(--init-dY-dt-prev-raw "${FIXTURE_ROOT}/dY_dt_prev.raw.f64")
else
  DYNAMIC_HISTORY_ARGS=()
fi

run_fresh() {
  local label="$1" dt="$2" steps="$3" params="$4" meta="$5" checkpoint="$6" root="${RUN_ROOT}/$1"
  mkdir -p "${root}/results"
  CUDA_STO_RESULTS_ROOT="${root}/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 "${SOURCE_ROOT}/main_cuda" "${GRID_N}" "${GRID_N}" "${GRID_N}" "${dt}" "${steps}" "${steps}" 1 1 --pf-param-file "${params}" --mode dynamics --init-mode raw_fields --init-phi-raw "${FIXTURE_ROOT}/phi.raw.f64" --init-xB-raw "${FIXTURE_ROOT}/xB_alpha.raw.f64" "${DYNAMIC_HISTORY_ARGS[@]}" --init-meta "${meta}" --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 --pf-zero-mode-tol-rel 1e-12 --pf-zero-mode-max-iter 24 --pf-checkpoint-every "${steps}" --pf-checkpoint-path "${checkpoint}" --init-case-tag "${CASE_TAG}" >"${root}/stdout.log" 2>"${root}/stderr.log"
  [[ ! -s "${root}/stderr.log" ]] || { echo "[fatal] non-empty stderr: ${label}" >&2; exit 2; }
  grep -q '^PF_ZERO_MODE_FINAL_AUDIT status=PASS ' "${root}/stdout.log"
  if [[ "${DYNAMIC_HISTORY_REQUIRED}" == "1" ]]; then
    grep -q 'dY_dt_prev initialization : provenance_pinned_raw_history' "${root}/stdout.log"
  else
    grep -q 'dY_dt_prev initialization : zero_for_fresh_dynamic_start' "${root}/stdout.log"
  fi
}

run_fresh one_step "${DT_CODE}" 1 "${PARAM_PRODUCTION}" "${RUN_ROOT}/provenance/init_meta_production.json" "${RUN_ROOT}/one_step/final.chk"
run_fresh continuous "${DT_CODE}" "${STEPS}" "${PARAM_PRODUCTION}" "${RUN_ROOT}/provenance/init_meta_production.json" "${RUN_ROOT}/continuous/final.chk"
run_fresh restart_first_half "${DT_CODE}" "$((STEPS / 2))" "${PARAM_PRODUCTION}" "${RUN_ROOT}/provenance/init_meta_production.json" "${RUN_ROOT}/restart_first_half/half.chk"
mkdir -p "${RUN_ROOT}/restart_second_half/results"
CUDA_STO_RESULTS_ROOT="${RUN_ROOT}/restart_second_half/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 "${SOURCE_ROOT}/main_cuda" "${GRID_N}" "${GRID_N}" "${GRID_N}" "${DT_CODE}" "${STEPS}" "${STEPS}" 1 1 --pf-param-file "${PARAM_PRODUCTION}" --mode dynamics --pf-restart-from "${RUN_ROOT}/restart_first_half/half.chk" --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 --pf-zero-mode-tol-rel 1e-12 --pf-zero-mode-max-iter 24 --pf-checkpoint-every "${STEPS}" --pf-checkpoint-path "${RUN_ROOT}/restart_second_half/final.chk" --init-case-tag "${CASE_TAG}" >"${RUN_ROOT}/restart_second_half/stdout.log" 2>"${RUN_ROOT}/restart_second_half/stderr.log"
[[ ! -s "${RUN_ROOT}/restart_second_half/stderr.log" ]] || { echo "[fatal] non-empty stderr: restart_second_half" >&2; exit 2; }
grep -q '^PF_ZERO_MODE_FINAL_AUDIT status=PASS ' "${RUN_ROOT}/restart_second_half/stdout.log"
run_fresh refined "${REFINED_DT_CODE}" "${REFINED_STEPS}" "${PARAM_REFINED}" "${RUN_ROOT}/provenance/init_meta_refined.json" "${RUN_ROOT}/refined/final.chk"

python3 "${SOURCE_ROOT}/scripts/qualify_pf_elastic_multi_particle_6h_dynamics_v2.py" --fixture-manifest "${FIXTURE_ROOT}/fixture_manifest.json" --initial-probe-checkpoint "${RUN_ROOT}/one_step/final.chk" --continuous-checkpoint "${RUN_ROOT}/continuous/final.chk" --restart-checkpoint "${RUN_ROOT}/restart_second_half/final.chk" --refined-checkpoint "${RUN_ROOT}/refined/final.chk" --initial-probe-stdout "${RUN_ROOT}/one_step/stdout.log" --continuous-stdout "${RUN_ROOT}/continuous/stdout.log" --restart-stdout "${RUN_ROOT}/restart_second_half/stdout.log" --refined-stdout "${RUN_ROOT}/refined/stdout.log" --out "${RUN_ROOT}/audit" >"${RUN_ROOT}/audit.stdout" 2>"${RUN_ROOT}/audit.stderr"
[[ ! -s "${RUN_ROOT}/audit.stderr" ]] || { echo "[fatal] V3 analyzer stderr" >&2; exit 2; }
grep -q "^handoff_status=${HANDOFF_STATUS}$" "${RUN_ROOT}/audit/final_terminal_output.txt"
cp "${RUN_ROOT}/audit/final_terminal_output.txt" "${RUN_ROOT}/status.txt"
cat "${RUN_ROOT}/status.txt"
