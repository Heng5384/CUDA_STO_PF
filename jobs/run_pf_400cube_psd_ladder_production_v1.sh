#!/usr/bin/env bash
set -euo pipefail

# One 400^3 PSD/spatial/density ladder case: claim/authority, 256-step
# continuous/restart, first-hour and 6-8 h qualification, then the full
# 6 h -> 48 h checkpoint chain and frozen no-dislocation transport audit.
# Bulk VTK output remains suppressed; checkpoints are the authoritative field
# state and can be exported separately by a read-only post-processing tool.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
CASE_ID="${CASE_ID:?CASE_ID is required}"
QUEUE="${QUEUE:?QUEUE is required}"
PARAM_FILE="${PARAM_FILE:-}"
GRID_N="${GRID_N:-400}"
EXPECTED_GRID_N="${EXPECTED_GRID_N:-400}"
SMOKE_RESTART_ONLY="${SMOKE_RESTART_ONLY:-0}"
DT_CODE=0.02
DT_PHYSICAL_S=0.9909260953431841
FINAL_STEP=152585
CHECKPOINT_CADENCE=3633
INITIAL_STATE_CLASS="MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
JOB_ID="${SLURM_JOB_ID:-UNSCHEDULED}_${SLURM_ARRAY_TASK_ID:-0}"
STATIC_FIXTURE_PASS="PASS_400CUBE_PSD_CAMPAIGN_INTEGER_GATE_FIXTURE_V2"
STATIC_AUDIT_PASS="PASS_400CUBE_PSD_CAMPAIGN_INTEGER_GATE_STATIC_AUDIT_V2"
SEGMENT_PASS="PASS_400CUBE_PSD_LADDER_SEGMENT_V1"
SHORT_QUAL_PASS="PASS_400CUBE_PSD_LADDER_SHORT_QUALIFICATION_V1"
FINAL_PASS="PASS_400CUBE_PSD_LADDER_PRODUCTION_V1"
FINAL_BLOCKED="BLOCKED_400CUBE_PSD_LADDER_PRODUCTION_V1"

PLAN_MANIFEST="${CAMPAIGN_ROOT}/manifests/plan/campaign_case_manifest.json"
[[ -f "${PLAN_MANIFEST}" ]] || { echo "[fatal] campaign plan manifest missing" >&2; exit 2; }
CASE_LABEL="$(python3 - "${PLAN_MANIFEST}" "${CASE_ID}" <<'PY'
import json, pathlib, sys
rows=json.loads(pathlib.Path(sys.argv[1]).read_text())["cases"]
for row in rows:
    if row["case_id"]==sys.argv[2]:
        print(row["case_label"]); break
else:
    raise SystemExit("case_id not found")
PY
)"
FIXTURE_ROOT="${CAMPAIGN_ROOT}/fixtures/${CASE_ID}"
SPEC_PATH="${CAMPAIGN_ROOT}/manifests/plan/case_specs/${CASE_ID}_${CASE_LABEL}.json"
CLAIM_ROOT="${CAMPAIGN_ROOT}/claims"
STATUS_ROOT="${CAMPAIGN_ROOT}/statuses"
AUTHORITY_TMP="${CAMPAIGN_ROOT}/authority_tmp/${CASE_ID}"
AUTHORITY_ROOT="${CAMPAIGN_ROOT}/authority/${CASE_ID}"
RUN_ROOT="${CAMPAIGN_ROOT}/attempts/${CASE_ID}/${QUEUE}_${JOB_ID}"

mkdir -p "${CLAIM_ROOT}" "${STATUS_ROOT}" "${CAMPAIGN_ROOT}/authority_tmp"

skip_and_exit() {
  local reason="$1"
  printf '%s\n' "${reason}" >"${CAMPAIGN_ROOT}/statuses/${CASE_ID}.${QUEUE}.${JOB_ID}.skip"
  echo "${reason}"
  exit 0
}

if [[ -f "${AUTHORITY_ROOT}/PASS" ]]; then
  skip_and_exit "SKIPPED_ALREADY_AUTHORITY"
fi

CLAIM_DIR="${CLAIM_ROOT}/${CASE_ID}.claim"
if ! mkdir "${CLAIM_DIR}" 2>/dev/null; then
  skip_and_exit "SKIPPED_CLAIMED_BY_OTHER_QUEUE"
fi

cleanup() {
  local rc=$?
  if [[ -n "${monitor_pid:-}" ]]; then
    kill "${monitor_pid}" 2>/dev/null || true
    wait "${monitor_pid}" 2>/dev/null || true
  fi
  if [[ "${rc}" -ne 0 && ! -f "${STATUS_ROOT}/${CASE_ID}.SCIENTIFIC_FAIL" ]]; then
    printf 'RETRYABLE_CASE_FAILURE\nexit_code=%s\n' "${rc}" >"${RUN_ROOT}/status.txt"
    rmdir "${CLAIM_DIR}" 2>/dev/null || true
  fi
}
trap cleanup EXIT

[[ -x "${SOURCE_ROOT}/main_cuda" ]] || { echo "[fatal] main_cuda is missing" >&2; exit 2; }
[[ -f "${FIXTURE_ROOT}/fixture_manifest.json" && -f "${FIXTURE_ROOT}/status.txt" ]] || { echo "[fatal] fixture is incomplete" >&2; exit 2; }
[[ "$(<"${FIXTURE_ROOT}/status.txt")" == "${STATIC_FIXTURE_PASS}" ]] || { echo "[fatal] fixture lacks exact PASS" >&2; exit 2; }
[[ -f "${FIXTURE_ROOT}/static_audit/status.txt" && "$(<"${FIXTURE_ROOT}/static_audit/status.txt")" == "${STATIC_AUDIT_PASS}" ]] || { echo "[fatal] fixture lacks static audit PASS" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "[fatal] attempt root already exists: ${RUN_ROOT}" >&2; exit 2; }

fixture_sha="$(sha256sum "${FIXTURE_ROOT}/fixture_manifest.json" | awk '{print $1}')"
library_sha="$(python3 - "${FIXTURE_ROOT}/fixture_manifest.json" "${EXPECTED_GRID_N}" <<'PY'
import json, pathlib, sys
m=json.loads(pathlib.Path(sys.argv[1]).read_text())
expected_n=int(sys.argv[2])
assert m['grid']=={'Nx':expected_n,'Ny':expected_n,'Nz':expected_n,'dx_nm':1.0,'lambda_sm_nm':4.0}
assert m['initial_state_class']=='MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1'
assert m['physical_contract']['elasticity_enabled'] is True
for key in ('GP_enabled','GP_birth_enabled','GP_release_enabled','external_source_enabled','new_beta_nucleation_enabled'):
    assert m['physical_contract'][key] is False, key
print(m['profile_library_manifest_sha256'])
PY
)"
initial_count="$(python3 - "${FIXTURE_ROOT}/fixture_manifest.json" <<'PY'
import json, pathlib, sys
print(int(json.loads(pathlib.Path(sys.argv[1]).read_text())['component_contract']['expected_count']))
PY
)"

export TEMP_C=380 DT="${DT_CODE}" PHYS_DX_REF_M=1.0e-9 PF_DX_M=1.0e-9
export PHYS_LAMBDA_SM_M=4.0e-9 PHYS_GAMMA_JM2=0.168 PHYS_L_REF_FACTOR=5.0 PHYS_D_RATIO=0.01
export PHYSICAL_OVERRIDE_FILE="${SOURCE_ROOT}/data/qualification/pf_elastic_target_profile_v1/physical_override_T380_dx1nm_lambda4nm_dynamic_dt0p02.json"
if [[ -z "${PARAM_FILE}" ]]; then
  PARAM_FILE="$(site_generate_pf_param_file "pf_400cube_psd_ladder_${CASE_ID}")"
fi
[[ -f "${PARAM_FILE}" ]] || { echo "[fatal] parameter file missing: ${PARAM_FILE}" >&2; exit 2; }
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

mkdir -p "${RUN_ROOT}/provenance" "${RUN_ROOT}/segments" "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/qualification"
exec >"${RUN_ROOT}/driver.stdout.log" 2>"${RUN_ROOT}/driver.stderr.log"
monitor_pid=""
printf 'RUNNING_400CUBE_PSD_LADDER_PRODUCTION_V1\n' >"${RUN_ROOT}/status.txt"
cp "${PARAM_FILE}" "${RUN_ROOT}/provenance/pf_input_dt0p02.params"
cp "${FIXTURE_ROOT}/fixture_manifest.json" "${RUN_ROOT}/provenance/fixture_manifest.json"
cp "${FIXTURE_ROOT}/init_meta.json" "${RUN_ROOT}/provenance/init_meta.json"
cp "${SPEC_PATH}" "${RUN_ROOT}/provenance/fixture_spec.json"
printf '%s\n' "${endpoints[@]}" >"${RUN_ROOT}/provenance/checkpoint_steps.txt"
python3 - "${RUN_ROOT}/provenance/campaign_manifest.json" "${fixture_sha}" "${library_sha}" "${SOURCE_ROOT}" "${CASE_ID}" "${QUEUE}" "${JOB_ID}" "${initial_count}" "${GRID_N}" <<'PY'
import json, pathlib, subprocess, sys
try: commit=subprocess.check_output(['git','-C',sys.argv[4],'rev-parse','HEAD'],text=True,stderr=subprocess.DEVNULL).strip()
except Exception: commit='NO_GIT_COMMIT'
payload={
 'schema':'PF_400CUBE_PSD_SPATIAL_DENSITY_LADDER_CAMPAIGN_V1',
 'source_commit':commit,
 'case_id':sys.argv[5],
 'queue':sys.argv[6],
 'slurm_job_id':sys.argv[7],
 'initial_particle_count':int(sys.argv[8]),
 'fixture_manifest_sha256':sys.argv[2],
 'profile_library_manifest_sha256':sys.argv[3],
 'grid':[int(sys.argv[9])]*3,'dx_nm':1.0,'temperature_C':380.0,
 'dt_code':0.02,'dt_physical_s':0.9909260953431841,
 'final_step':152585,'checkpoint_cadence_steps':3633,
 'registered_science_steps':[0,21798,43596,65393,108989,152585],
 'inventory_policy':'COMMON_INTEGER_REALIZABLE_H_VOLUME',
 'h_volume_relative_tolerance':1e-7,
 'GP_enabled':False,'GP_birth_enabled':False,'GP_release_enabled':False,
 'external_source_enabled':False,'new_beta_nucleation_enabled':False,
 'elasticity_enabled':True,'elastic_solver_mode':'ELASTIC_WARM_START_RESIDUAL_V1',
 'qualification_contract':'256_STEP_CONTINUOUS_RESTART_PLUS_FIRST_HOUR_PLUS_6_TO_8H_RUN_IN_DRIVER',
}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
sha256sum \
  "${SOURCE_ROOT}/main_cuda" \
  "${SOURCE_ROOT}/main_cuda.cu" \
  "${SOURCE_ROOT}/cuda_kernels.cu" \
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.cpp" \
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.h" \
  "${SOURCE_ROOT}/scripts/audit_pf_400cube_psd_ladder_production_v1.py" \
  "${SOURCE_ROOT}/scripts/analyze_pf_400cube_psd_ladder_transport_v1.py" \
  "${SOURCE_ROOT}/jobs/run_pf_400cube_psd_ladder_production_v1.sh" \
  "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
  "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${FIXTURE_ROOT}/phi.raw.f64" \
  "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
  "${FIXTURE_ROOT}/init_meta.json" \
  "${RUN_ROOT}/provenance/campaign_manifest.json" >"${RUN_ROOT}/provenance/input_hashes.sha256"
{
  echo "hostname=$(hostname)"
  echo "slurm_job_id=${JOB_ID}"
  echo "queue=${QUEUE}"
  echo "case_id=${CASE_ID}"
  echo "fixture_manifest_sha256=${fixture_sha}"
  echo "binary_sha256=$(sha256sum "${SOURCE_ROOT}/main_cuda" | awk '{print $1}')"
  echo "parameter_sha256=$(sha256sum "${PARAM_FILE}" | awk '{print $1}')"
  echo "claim_dir=${CLAIM_DIR}"
  nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total --format=csv,noheader,nounits || true
} >"${RUN_ROOT}/provenance/claim_and_identity.txt"
nvidia-smi --query-gpu=timestamp,index,utilization.gpu,memory.used --format=csv,noheader,nounits -lms 1000 >"${RUN_ROOT}/gpu_samples.csv" 2>/dev/null & monitor_pid="$!"

identity=(--pf-initial-state-class "${INITIAL_STATE_CLASS}" --pf-fixture-manifest-sha256 "${fixture_sha}" --pf-profile-library-manifest-sha256 "${library_sha}")
zero=(--pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 --pf-zero-mode-tol-rel 1e-12 --pf-zero-mode-max-iter 24)

run_main() {
  local label="$1" endpoint="$2" checkpoint="$3"
  local root="${RUN_ROOT}/qualification/${label}"
  local init_args=()
  if [[ "${label}" == "restart_second_half" ]]; then
    init_args=(--pf-restart-from "${RUN_ROOT}/checkpoints/step_0128_restart.chk")
  elif [[ "${label}" == "continuous_256" || "${label}" == "restart_first_half" ]]; then
    init_args=(--init-mode raw_fields --init-phi-raw "${FIXTURE_ROOT}/phi.raw.f64" --init-xB-raw "${FIXTURE_ROOT}/xB_alpha.raw.f64" --init-meta "${FIXTURE_ROOT}/init_meta.json")
  else
    init_args=(--pf-restart-from "${RUN_ROOT}/checkpoints/step_0256.chk")
  fi
  mkdir -p "${root}/results"
  CUDA_STO_SUPPRESS_VTK_OUTPUT=1 CUDA_STO_RESULTS_ROOT="${root}/results" \
    "${SOURCE_ROOT}/main_cuda" "${GRID_N}" "${GRID_N}" "${GRID_N}" "${DT_CODE}" \
    "${endpoint}" "${endpoint}" 1 1 \
    --pf-param-file "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
    --mode dynamics "${init_args[@]}" "${zero[@]}" "${identity[@]}" \
    --enable-dynamics-mass-diagnostics --dynamics-mass-diag-interval "${endpoint}" \
    --pf-checkpoint-every "${endpoint}" --pf-checkpoint-path "${checkpoint}" \
    --init-case-tag "pf_400cube_psd_ladder_${CASE_ID}_${label}" \
    >"${root}/stdout.log" 2>"${root}/stderr.log"
  [[ ! -s "${root}/stderr.log" ]] || { echo "[fatal] nonempty stderr in ${label}" >&2; exit 3; }
  grep -q '^PF_ZERO_MODE_FINAL_AUDIT status=PASS ' "${root}/stdout.log"
  local mass
  mass="$(find "${root}/results" -type f -name dynamics_mass_diagnostics.csv -print -quit)"
  [[ -n "${mass}" && -s "${mass}" ]] || { echo "[fatal] missing mass diagnostics in ${label}" >&2; exit 3; }
  cp "${mass}" "${root}/dynamics_mass_diagnostics.csv"
}

run_main continuous_256 256 "${RUN_ROOT}/checkpoints/step_0256.chk"
run_main restart_first_half 128 "${RUN_ROOT}/checkpoints/step_0128_restart.chk"
run_main restart_second_half 256 "${RUN_ROOT}/checkpoints/step_0256_restart.chk"
cmp -s "${RUN_ROOT}/checkpoints/step_0256.chk" "${RUN_ROOT}/checkpoints/step_0256_restart.chk"

if [[ "${SMOKE_RESTART_ONLY}" == "1" ]]; then
  printf 'PASS_400CUBE_PSD_LADDER_WORKSTATION_SMOKE_V1\n' >"${RUN_ROOT}/qualification/status.txt"
  printf 'smoke_mode=restart_only_256\ncontinuous_vs_restart_256=BITWISE_IDENTICAL\n' >"${RUN_ROOT}/qualification/summary.txt"
  cp "${RUN_ROOT}/qualification/status.txt" "${RUN_ROOT}/status.txt"
  cat "${RUN_ROOT}/status.txt"
  exit 0
fi

run_segment() {
  local endpoint="$1"
  local segment="${RUN_ROOT}/segments/step_${endpoint}"
  local checkpoint="${RUN_ROOT}/checkpoints/step_${endpoint}.chk"
  mkdir -p "${segment}/results"
  CUDA_STO_SUPPRESS_VTK_OUTPUT=1 CUDA_STO_RESULTS_ROOT="${segment}/results" \
    "${SOURCE_ROOT}/main_cuda" "${GRID_N}" "${GRID_N}" "${GRID_N}" "${DT_CODE}" \
    "${endpoint}" "${endpoint}" 1 1 \
    --pf-param-file "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
    --mode dynamics --pf-restart-from "${current}" "${zero[@]}" "${identity[@]}" \
    --enable-dynamics-mass-diagnostics --dynamics-mass-diag-interval "${endpoint}" \
    --pf-checkpoint-every "${endpoint}" --pf-checkpoint-path "${checkpoint}" \
    --init-case-tag "pf_400cube_psd_ladder_${CASE_ID}_step_${endpoint}" \
    >"${segment}/stdout.log" 2>"${segment}/stderr.log"
  [[ ! -s "${segment}/stderr.log" ]] || { echo "[fatal] nonempty stderr at step ${endpoint}" >&2; exit 3; }
  grep -q '^PF_ZERO_MODE_FINAL_AUDIT status=PASS ' "${segment}/stdout.log"
  local mass
  mass="$(find "${segment}/results" -type f -name dynamics_mass_diagnostics.csv -print -quit)"
  [[ -n "${mass}" && -s "${mass}" ]] || { echo "[fatal] missing mass diagnostic at step ${endpoint}" >&2; exit 3; }
  cp "${mass}" "${segment}/dynamics_mass_diagnostics.csv"
  sha256sum "${checkpoint}" >"${segment}/checkpoint.sha256"
  printf '%s\n' "${SEGMENT_PASS}" >"${segment}/status.txt"
  current="${checkpoint}"
  checkpoint_args+=(--checkpoint "${checkpoint}")
}

current="${RUN_ROOT}/checkpoints/step_0256.chk"
run_segment 3633
run_segment 7266
printf '%s\n' "${SHORT_QUAL_PASS}" >"${RUN_ROOT}/qualification/status.txt"
printf 'short_qualification_status=%s\ncontinuous_vs_restart_256=BITWISE_IDENTICAL\nfirst_hour_checkpoint=step_03633\nstage_6h_to_8h_endpoint=step_07266\n' "${SHORT_QUAL_PASS}" >"${RUN_ROOT}/qualification/summary.txt"
printf 'PASS_400CUBE_PSD_LADDER_SHORT_QUALIFICATION_V1\n' >"${STATUS_ROOT}/${CASE_ID}.SHORT_QUALIFICATION_PASS"

current="${RUN_ROOT}/checkpoints/step_7266.chk"
for endpoint in "${endpoints[@]}"; do
  if (( endpoint > 7266 )); then
    run_segment "${endpoint}"
  fi
done

sha256sum "${RUN_ROOT}"/checkpoints/*.chk | sort -k2 >"${RUN_ROOT}/provenance/checkpoint_chain.sha256"
c++ -std=c++17 -O3 "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" -lstdc++fs -o "${RUN_ROOT}/provenance/analyze_pf_periodic_particle_lineage"
sha256sum \
  "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_hourly_merge_dissolution_v1.py" \
  "${SOURCE_ROOT}/scripts/build_pf_400cube_initial_component_anchors_v1.py" \
  "${SOURCE_ROOT}/scripts/audit_pf_400cube_psd_ladder_production_v1.py" \
  "${SOURCE_ROOT}/scripts/analyze_pf_400cube_psd_ladder_transport_v1.py" \
  "${RUN_ROOT}/provenance/analyze_pf_periodic_particle_lineage" >"${RUN_ROOT}/provenance/analysis_hashes.sha256"
python3 "${SOURCE_ROOT}/scripts/build_pf_400cube_initial_component_anchors_v1.py" \
  --fixture-manifest "${FIXTURE_ROOT}/fixture_manifest.json" \
  --initial-particles "${FIXTURE_ROOT}/initial_particles.csv" \
  --out "${RUN_ROOT}/analysis_initial_component_anchors.csv" \
  >"${RUN_ROOT}/anchors.stdout.log" 2>"${RUN_ROOT}/anchors.stderr.log"
[[ ! -s "${RUN_ROOT}/anchors.stderr.log" ]] || { echo "[fatal] anchor registration stderr is nonempty" >&2; exit 3; }

run_tracker() {
  local name="$1" threshold="$2" output="${RUN_ROOT}/tracker_${name}"
  set +e
  "${RUN_ROOT}/provenance/analyze_pf_periodic_particle_lineage" \
    --initial-phi "${FIXTURE_ROOT}/phi.raw.f64" \
    --initial-xb "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
    "${checkpoint_args[@]}" --out "${output}" --grid 400 --dx-nm 1 \
    --threshold "${threshold}" --physical-dt-s "${DT_PHYSICAL_S}" \
    --start-age-h 6 --target-mean 0.03 --expected-initial-count "${initial_count}" \
    --allow-dissolution --allow-merge-groups \
    >"${RUN_ROOT}/tracker_${name}.stdout.log" 2>"${RUN_ROOT}/tracker_${name}.stderr.log"
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
python3 "${SOURCE_ROOT}/scripts/audit_pf_246cube_hourly_merge_dissolution_v1.py" \
  --low "${RUN_ROOT}/tracker_low" --medium "${RUN_ROOT}/tracker_medium" \
  --strong "${RUN_ROOT}/tracker_strong" \
  --initial-components "${RUN_ROOT}/analysis_initial_component_anchors.csv" \
  --expected-steps "${RUN_ROOT}/provenance/checkpoint_steps.txt" \
  --expected-initial-count "${initial_count}" --domain-nm 400 \
  --physical-dt-s "${DT_PHYSICAL_S}" --start-age-h 6 --out "${RUN_ROOT}/merge_aware" \
  >"${RUN_ROOT}/merge_aware.stdout.log" 2>"${RUN_ROOT}/merge_aware.stderr.log"
merge_rc=$?
set -e
[[ "${merge_rc}" -eq 0 || "${merge_rc}" -eq 2 ]] || { echo "[fatal] merge-aware audit returned ${merge_rc}" >&2; exit 3; }

set +e
python3 "${SOURCE_ROOT}/scripts/audit_pf_400cube_psd_ladder_production_v1.py" \
  --fixture-manifest "${FIXTURE_ROOT}/fixture_manifest.json" \
  --run-root "${RUN_ROOT}" --lineage-root "${RUN_ROOT}/tracker_low" \
  --merge-aware-audit "${RUN_ROOT}/merge_aware/audit.json" \
  --gpu-samples "${RUN_ROOT}/gpu_samples.csv" --out "${RUN_ROOT}/audit" \
  >"${RUN_ROOT}/audit.stdout.log" 2>"${RUN_ROOT}/audit.stderr.log"
audit_rc=$?
python3 "${SOURCE_ROOT}/scripts/analyze_pf_400cube_psd_ladder_transport_v1.py" \
  --observables "${RUN_ROOT}/audit/hourly_observables.csv" \
  --particles "${RUN_ROOT}/audit/hourly_particle_psd.csv" \
  --yu-config "${SOURCE_ROOT}/data/qualification/yu2024_transport_v1/yu_48h_parameters.json" \
  --contract "${SOURCE_ROOT}/data/qualification/pf_full_psd_no_dislocation_transport_v1/transport_parameter_contract.json" \
  --transport-module "${SOURCE_ROOT}/scripts/pf_full_psd_no_dislocation_transport_v1.py" \
  --out "${RUN_ROOT}/transport" \
  >"${RUN_ROOT}/transport.stdout.log" 2>"${RUN_ROOT}/transport.stderr.log"
transport_rc=$?
set -e

if [[ "${audit_rc}" -eq 0 && "${transport_rc}" -eq 0 ]] \
   && grep -qx "${FINAL_PASS}" "${RUN_ROOT}/audit/status.txt" \
   && grep -qx 'PASS_400CUBE_PSD_LADDER_TRANSPORT_V1' "${RUN_ROOT}/transport/status.txt"; then
  printf '%s\n' "${FINAL_PASS}" >"${RUN_ROOT}/status.txt"
  mkdir -p "${AUTHORITY_TMP}"
  printf '%s\n' "${FINAL_PASS}" >"${AUTHORITY_TMP}/PASS"
  cp "${RUN_ROOT}/provenance/campaign_manifest.json" "${AUTHORITY_TMP}/campaign_manifest.json"
  cp "${RUN_ROOT}/audit/audit.json" "${AUTHORITY_TMP}/audit.json"
  cp "${RUN_ROOT}/transport/transport_manifest.json" "${AUTHORITY_TMP}/transport_manifest.json"
  printf 'source_queue=%s\nsource_job_id=%s\ncase_id=%s\nfixture_manifest_sha256=%s\n' \
    "${QUEUE}" "${JOB_ID}" "${CASE_ID}" "${fixture_sha}" >"${AUTHORITY_TMP}/source.txt"
  mv "${AUTHORITY_TMP}" "${AUTHORITY_ROOT}"
  printf 'PASS_400CUBE_PSD_LADDER_PRODUCTION_V1\n' >"${STATUS_ROOT}/${CASE_ID}.AUTHORITY_PASS"
else
  printf 'BLOCKED_400CUBE_PSD_LADDER_PRODUCTION_V1\naudit_exit=%s\ntransport_exit=%s\nmerge_aware_exit=%s\n' \
    "${audit_rc}" "${transport_rc}" "${merge_rc}" >"${RUN_ROOT}/status.txt"
fi
cat "${RUN_ROOT}/status.txt"
