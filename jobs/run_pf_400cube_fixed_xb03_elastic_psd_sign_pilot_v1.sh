#!/usr/bin/env bash
set -euo pipefail

# Independent 400^3 conditional 6 h -> 48 h feasibility pilot.  It is never
# used to overwrite or reinterpret a 246^3 production path, and it suppresses
# bulk VTK output throughout the production trajectory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
FIXTURE_ROOT="${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
SPEC_PATH="${SPEC_PATH:?SPEC_PATH is required}"
GRID_N=400
DT_CODE=0.02
DT_PHYSICAL_S=0.9909260953431841
FINAL_STEP=152585
CHECKPOINT_CADENCE=3633
INITIAL_STATE_CLASS="MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
VALIDATE_ONLY=0
if [[ "${1:-}" == "--validate-only" ]]; then VALIDATE_ONLY=1; elif [[ "$#" -ne 0 ]]; then echo "usage: $0 [--validate-only]" >&2; exit 2; fi

[[ -x "${SOURCE_ROOT}/main_cuda" ]] || { echo "[fatal] main_cuda is missing" >&2; exit 2; }
[[ -f "${FIXTURE_ROOT}/fixture_manifest.json" && -f "${FIXTURE_ROOT}/status.txt" ]] || { echo "[fatal] fixture is incomplete" >&2; exit 2; }
[[ "$(<"${FIXTURE_ROOT}/status.txt")" == "PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_FIXTURE_V1" ]] || { echo "[fatal] fixture lacks exact PASS" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "[fatal] refusing to overwrite RUN_ROOT: ${RUN_ROOT}" >&2; exit 2; }

fixture_sha="$(sha256sum "${FIXTURE_ROOT}/fixture_manifest.json" | awk '{print $1}')"
library_sha="$(python3 - "${FIXTURE_ROOT}/fixture_manifest.json" <<'PY'
import json, pathlib, sys
m=json.loads(pathlib.Path(sys.argv[1]).read_text())
assert m['grid']=={'Nx':400,'Ny':400,'Nz':400,'dx_nm':1.0,'lambda_sm_nm':4.0}
assert m['initial_state_class']=='MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1'
assert m['component_contract']['actual_count']==64
assert m['physical_contract']['elasticity_enabled'] is True
for key in ('GP_enabled','GP_birth_enabled','GP_release_enabled','external_source_enabled','new_beta_nucleation_enabled'):
    assert m['physical_contract'][key] is False, key
print(m['profile_library_manifest_sha256'])
PY
)"

export TEMP_C=380 DT="${DT_CODE}" PHYS_DX_REF_M=1.0e-9 PF_DX_M=1.0e-9
export PHYS_LAMBDA_SM_M=4.0e-9 PHYS_GAMMA_JM2=0.168 PHYS_L_REF_FACTOR=5.0 PHYS_D_RATIO=0.01
export PHYSICAL_OVERRIDE_FILE="${SOURCE_ROOT}/data/qualification/pf_elastic_target_profile_v1/physical_override_T380_dx1nm_lambda4nm_dynamic_dt0p02.json"
PARAM_FILE="$(site_generate_pf_param_file pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1)"
python3 - "${PARAM_FILE}" <<'PY'
import pathlib, sys
p={}
for raw in pathlib.Path(sys.argv[1]).read_text().splitlines():
    if raw.strip() and not raw.lstrip().startswith('#') and '=' in raw:
        k,v=raw.split('=',1); p[k.strip()]=v.strip()
assert abs(float(p['dt'])-0.02)<1e-15
assert abs(float(p['dt'])*float(p['t_real_unit'])-0.9909260953431841)<1e-12
for key in ('elastic_warm_start_enabled','elastic_residual_control_enabled','elastic_fail_on_nonconvergence'):
    assert p[key]=='1', key
PY

endpoints=()
for ((step=CHECKPOINT_CADENCE; step<FINAL_STEP; step+=CHECKPOINT_CADENCE)); do endpoints+=("${step}"); done
endpoints+=(21798 43596 65393 108989 152585)
mapfile -t endpoints < <(printf '%s\n' "${endpoints[@]}" | sort -n -u)

if [[ "${VALIDATE_ONLY}" -eq 1 ]]; then
  printf '%s\n' "PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_PRODUCTION_PREFLIGHT_V1" "fixture_manifest_sha256=${fixture_sha}" "profile_library_manifest_sha256=${library_sha}" "parameter_sha256=$(sha256sum "${PARAM_FILE}" | awk '{print $1}')" "binary_sha256=$(sha256sum "${SOURCE_ROOT}/main_cuda" | awk '{print $1}')" "checkpoint_count=${#endpoints[@]}" "run_root=${RUN_ROOT}"
  exit 0
fi

mkdir -p "${RUN_ROOT}/provenance" "${RUN_ROOT}/segments" "${RUN_ROOT}/checkpoints"
exec >"${RUN_ROOT}/driver.stdout.log" 2>"${RUN_ROOT}/driver.stderr.log"
monitor_pid=""
cleanup() {
  rc=$?
  if [[ -n "${monitor_pid}" ]]; then kill "${monitor_pid}" 2>/dev/null || true; wait "${monitor_pid}" 2>/dev/null || true; fi
  if [[ "${rc}" -ne 0 ]]; then printf 'BLOCKED_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_PILOT_V1\nexit_code=%s\n' "${rc}" >"${RUN_ROOT}/status.txt"; fi
}
trap cleanup EXIT
printf 'RUNNING_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_PILOT_V1\n' >"${RUN_ROOT}/status.txt"
cp "${PARAM_FILE}" "${RUN_ROOT}/provenance/pf_input_dt0p02.params"
cp "${FIXTURE_ROOT}/fixture_manifest.json" "${RUN_ROOT}/provenance/fixture_manifest.json"
cp "${FIXTURE_ROOT}/init_meta.json" "${RUN_ROOT}/provenance/init_meta.json"
cp "${SPEC_PATH}" "${RUN_ROOT}/provenance/fixture_spec.json"
printf '%s\n' "${endpoints[@]}" >"${RUN_ROOT}/provenance/checkpoint_steps.txt"
python3 - "${RUN_ROOT}/provenance/campaign_manifest.json" "${fixture_sha}" "${library_sha}" "${SOURCE_ROOT}" <<'PY'
import json, pathlib, subprocess, sys
try: commit=subprocess.check_output(['git','-C',sys.argv[4],'rev-parse','HEAD'],text=True).strip()
except Exception: commit='NO_GIT_COMMIT'
payload={'schema':'PF_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_CAMPAIGN_V1','source_commit':commit,'fixture_manifest_sha256':sys.argv[2],'profile_library_manifest_sha256':sys.argv[3],'grid':[400,400,400],'dx_nm':1.0,'temperature_C':380.0,'dt_code':0.02,'dt_physical_s':0.9909260953431841,'final_step':152585,'checkpoint_cadence_steps':3633,'registered_science_steps':[0,21798,43596,65393,108989,152585],'GP_enabled':False,'GP_birth_enabled':False,'GP_release_enabled':False,'external_source_enabled':False,'new_beta_nucleation_enabled':False,'elasticity_enabled':True,'elastic_solver_mode':'ELASTIC_WARM_START_RESIDUAL_V1','user_authorized_skipped_qualifications':['one_step_restart','100_to_1000_step_smoke','6_to_12h_short_smoke']}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
sha256sum "${SOURCE_ROOT}/main_cuda" "${SOURCE_ROOT}/main_cuda.cu" "${SOURCE_ROOT}/cuda_kernels.cu" "${SOURCE_ROOT}/pf_zero_mode_checkpoint.cpp" "${SOURCE_ROOT}/pf_zero_mode_checkpoint.h" "${SOURCE_ROOT}/scripts/build_pf_400cube_fixed_xb03_elastic_psd_sign_fixture_v1.py" "${SOURCE_ROOT}/scripts/audit_pf_400cube_fixed_xb03_elastic_psd_sign_fixture_v1.py" "${SOURCE_ROOT}/jobs/run_pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1.sh" "${RUN_ROOT}/provenance/pf_input_dt0p02.params" "${FIXTURE_ROOT}/fixture_manifest.json" "${FIXTURE_ROOT}/phi.raw.f64" "${FIXTURE_ROOT}/xB_alpha.raw.f64" "${FIXTURE_ROOT}/init_meta.json" "${RUN_ROOT}/provenance/campaign_manifest.json" >"${RUN_ROOT}/provenance/input_hashes.sha256"
{ echo "hostname=$(hostname)"; echo "slurm_job_id=${SLURM_JOB_ID:-UNSCHEDULED}"; nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader,nounits || true; } >"${RUN_ROOT}/provenance/device_identity.txt"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 1000 >"${RUN_ROOT}/gpu_samples.csv" 2>/dev/null & monitor_pid="$!"

identity=(--pf-initial-state-class "${INITIAL_STATE_CLASS}" --pf-fixture-manifest-sha256 "${fixture_sha}" --pf-profile-library-manifest-sha256 "${library_sha}")
zero=(--pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 --pf-zero-mode-tol-rel 1e-12 --pf-zero-mode-max-iter 24)
current=""; checkpoint_args=()
for endpoint in "${endpoints[@]}"; do
  segment="${RUN_ROOT}/segments/step_${endpoint}"; checkpoint="${RUN_ROOT}/checkpoints/step_${endpoint}.chk"; mkdir -p "${segment}/results"
  if [[ -z "${current}" ]]; then init=(--init-mode raw_fields --init-phi-raw "${FIXTURE_ROOT}/phi.raw.f64" --init-xB-raw "${FIXTURE_ROOT}/xB_alpha.raw.f64" --init-meta "${FIXTURE_ROOT}/init_meta.json"); else init=(--pf-restart-from "${current}"); fi
  CUDA_STO_SUPPRESS_VTK_OUTPUT=1 CUDA_STO_RESULTS_ROOT="${segment}/results" "${SOURCE_ROOT}/main_cuda" "${GRID_N}" "${GRID_N}" "${GRID_N}" "${DT_CODE}" "${endpoint}" "${endpoint}" 1 1 --pf-param-file "${RUN_ROOT}/provenance/pf_input_dt0p02.params" --mode dynamics "${init[@]}" "${zero[@]}" "${identity[@]}" --enable-dynamics-mass-diagnostics --dynamics-mass-diag-interval "${endpoint}" --pf-checkpoint-every "${endpoint}" --pf-checkpoint-path "${checkpoint}" --init-case-tag "pf_400cube_fixed_xb03_step_${endpoint}" >"${segment}/stdout.log" 2>"${segment}/stderr.log"
  [[ ! -s "${segment}/stderr.log" ]] || { echo "[fatal] nonempty stderr at step ${endpoint}" >&2; exit 3; }
  grep -q '^PF_ZERO_MODE_FINAL_AUDIT status=PASS ' "${segment}/stdout.log"
  mass="$(find "${segment}/results" -type f -name dynamics_mass_diagnostics.csv -print -quit)"; [[ -n "${mass}" && -s "${mass}" ]] || { echo "[fatal] missing mass diagnostic at step ${endpoint}" >&2; exit 3; }
  cp "${mass}" "${segment}/dynamics_mass_diagnostics.csv"; sha256sum "${checkpoint}" >"${segment}/checkpoint.sha256"; printf 'PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1\n' >"${segment}/status.txt"
  current="${checkpoint}"; checkpoint_args+=(--checkpoint "${checkpoint}")
done
sha256sum "${RUN_ROOT}"/checkpoints/*.chk | sort -k2 >"${RUN_ROOT}/provenance/checkpoint_chain.sha256"
# The same three-threshold, merge-aware tracker contract that was accepted
# for the earlier C path is executed on the complete 400-cube checkpoint
# chain.  It is read-only and may fail closed on identity while preserving the
# independent global PSD for transport analysis.
c++ -std=c++17 -O3 "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" -lstdc++fs -o "${RUN_ROOT}/provenance/analyze_pf_periodic_particle_lineage"
sha256sum "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" "${SOURCE_ROOT}/scripts/audit_pf_246cube_hourly_merge_dissolution_v1.py" "${SOURCE_ROOT}/scripts/build_pf_400cube_initial_component_anchors_v1.py" "${SOURCE_ROOT}/scripts/audit_pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1.py" "${SOURCE_ROOT}/scripts/analyze_pf_400cube_fixed_xb03_elastic_psd_sign_transport_v1.py" "${RUN_ROOT}/provenance/analyze_pf_periodic_particle_lineage" >"${RUN_ROOT}/provenance/analysis_hashes.sha256"
python3 "${SOURCE_ROOT}/scripts/build_pf_400cube_initial_component_anchors_v1.py" --fixture-manifest "${FIXTURE_ROOT}/fixture_manifest.json" --initial-particles "${FIXTURE_ROOT}/initial_particles.csv" --out "${RUN_ROOT}/analysis_initial_component_anchors.csv" >"${RUN_ROOT}/anchors.stdout.log" 2>"${RUN_ROOT}/anchors.stderr.log"
[[ ! -s "${RUN_ROOT}/anchors.stderr.log" ]] || { echo "[fatal] anchor registration stderr is nonempty" >&2; exit 3; }

run_tracker() {
  local name="$1" threshold="$2" output="${RUN_ROOT}/tracker_${name}"
  set +e
  "${RUN_ROOT}/provenance/analyze_pf_periodic_particle_lineage" --initial-phi "${FIXTURE_ROOT}/phi.raw.f64" --initial-xb "${FIXTURE_ROOT}/xB_alpha.raw.f64" "${checkpoint_args[@]}" --out "${output}" --grid 400 --dx-nm 1 --threshold "${threshold}" --physical-dt-s "${DT_PHYSICAL_S}" --start-age-h 6 --target-mean 0.03 --expected-initial-count 64 --allow-dissolution --allow-merge-groups >"${RUN_ROOT}/tracker_${name}.stdout.log" 2>"${RUN_ROOT}/tracker_${name}.stderr.log"
  local rc=$?
  set -e
  [[ "${rc}" -eq 0 || "${rc}" -eq 2 ]] || { echo "[fatal] tracker ${name} returned ${rc}" >&2; exit 3; }
  for file in lineage_summary.txt particle_events.csv particle_lineage.csv ensemble_observables.csv status.txt; do
    [[ -s "${output}/${file}" ]] || { echo "[fatal] tracker ${name} omitted ${file}" >&2; exit 3; }
  done
}
run_tracker low 0.0001
run_tracker medium 0.001
run_tracker strong 0.005

set +e
python3 "${SOURCE_ROOT}/scripts/audit_pf_246cube_hourly_merge_dissolution_v1.py" --low "${RUN_ROOT}/tracker_low" --medium "${RUN_ROOT}/tracker_medium" --strong "${RUN_ROOT}/tracker_strong" --initial-components "${RUN_ROOT}/analysis_initial_component_anchors.csv" --expected-steps "${RUN_ROOT}/provenance/checkpoint_steps.txt" --expected-initial-count 64 --domain-nm 400 --physical-dt-s "${DT_PHYSICAL_S}" --start-age-h 6 --out "${RUN_ROOT}/merge_aware" >"${RUN_ROOT}/merge_aware.stdout.log" 2>"${RUN_ROOT}/merge_aware.stderr.log"
merge_rc=$?
set -e
[[ "${merge_rc}" -eq 0 || "${merge_rc}" -eq 2 ]] || { echo "[fatal] merge-aware audit returned ${merge_rc}" >&2; exit 3; }

set +e
python3 "${SOURCE_ROOT}/scripts/audit_pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1.py" --fixture-manifest "${FIXTURE_ROOT}/fixture_manifest.json" --run-root "${RUN_ROOT}" --lineage-root "${RUN_ROOT}/tracker_low" --merge-aware-audit "${RUN_ROOT}/merge_aware/audit.json" --gpu-samples "${RUN_ROOT}/gpu_samples.csv" --out "${RUN_ROOT}/audit" >"${RUN_ROOT}/audit.stdout.log" 2>"${RUN_ROOT}/audit.stderr.log"
audit_rc=$?
python3 "${SOURCE_ROOT}/scripts/analyze_pf_400cube_fixed_xb03_elastic_psd_sign_transport_v1.py" --observables "${RUN_ROOT}/audit/hourly_observables.csv" --particles "${RUN_ROOT}/audit/hourly_particle_psd.csv" --yu-config "${SOURCE_ROOT}/data/qualification/yu2024_transport_v1/yu_48h_parameters.json" --contract "${SOURCE_ROOT}/data/qualification/pf_full_psd_no_dislocation_transport_v1/transport_parameter_contract.json" --transport-module "${SOURCE_ROOT}/scripts/pf_full_psd_no_dislocation_transport_v1.py" --out "${RUN_ROOT}/transport" >"${RUN_ROOT}/transport.stdout.log" 2>"${RUN_ROOT}/transport.stderr.log"
transport_rc=$?
set -e
if [[ "${audit_rc}" -eq 0 && "${transport_rc}" -eq 0 ]] && grep -qx 'PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_PILOT_V1' "${RUN_ROOT}/audit/status.txt" && grep -qx 'PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_TRANSPORT_V1' "${RUN_ROOT}/transport/status.txt"; then
  printf 'PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_PILOT_V1\n' >"${RUN_ROOT}/status.txt"
else
  printf 'BLOCKED_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_PILOT_V1\naudit_exit=%s\ntransport_exit=%s\nmerge_aware_exit=%s\n' "${audit_rc}" "${transport_rc}" "${merge_rc}" >"${RUN_ROOT}/status.txt"
fi
cat "${RUN_ROOT}/status.txt"
