#!/usr/bin/env bash
set -euo pipefail

# One non-overwriting 246^3 fixture: 256 continuous steps versus
# 128 + checkpoint/restart + 128.  This runner never starts the 6--8 h or
# 6--48 h trajectory.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
FIXTURE_ROOT="${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
PARAM_FILE="${PARAM_FILE:?PARAM_FILE is required}"
DT_CODE="0.02"
STEPS=256
HALF_STEPS=128
GRID_N=246
INITIAL_STATE_CLASS="MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
LIBRARY_SHA256="${LIBRARY_SHA256:-58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe}"
EXPECTED_OPTIMIZER_INVOKED="${EXPECTED_OPTIMIZER_INVOKED:-false}"

[[ ! -e "${RUN_ROOT}" ]] || {
  echo "[fatal] refusing to overwrite RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
}
for path in \
  "${SOURCE_ROOT}/main_cuda" \
  "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_short_restart_v1.py" \
  "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${FIXTURE_ROOT}/phi.raw.f64" \
  "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
  "${FIXTURE_ROOT}/init_meta.json" \
  "${PARAM_FILE}"; do
  [[ -f "${path}" ]] || {
    echo "[fatal] missing required input: ${path}" >&2
    exit 2
  }
done

mkdir -p "${RUN_ROOT}/provenance"
exec >"${RUN_ROOT}/driver.stdout.log" 2>"${RUN_ROOT}/driver.stderr.log"
MONITOR_PID=""
cleanup() {
  local rc=$?
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ "${rc}" -ne 0 ]]; then
    printf 'BLOCKED_246CUBE_SHORT_RESTART_DRIVER_V1\nexit_code=%s\n' "${rc}" \
      >"${RUN_ROOT}/status.txt"
  fi
}
trap cleanup EXIT

FIXTURE_SHA256="$(sha256sum "${FIXTURE_ROOT}/fixture_manifest.json" | awk '{print $1}')"
python3 - "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${RUN_ROOT}/provenance/fixture_preflight.json" "${FIXTURE_SHA256}" \
  "${LIBRARY_SHA256}" "${EXPECTED_OPTIMIZER_INVOKED}" <<'PY'
import hashlib,json,pathlib,sys
p=pathlib.Path(sys.argv[1]); m=json.loads(p.read_text())
if hashlib.sha256(p.read_bytes()).hexdigest()!=sys.argv[3]: raise SystemExit("fixture identity changed")
if m.get("schema")!="PF_246CUBE_LIBRARY_HANDOFF_MANIFEST_V1": raise SystemExit("wrong fixture schema")
if m.get("initial_state_class")!="MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1": raise SystemExit("wrong initial-state class")
if m.get("profile_library_manifest_sha256")!=sys.argv[4]: raise SystemExit("wrong profile library")
if m.get("grid")!={"Nx":246,"Ny":246,"Nz":246,"dx_nm":1.0,"lambda_sm_nm":4.0}: raise SystemExit("wrong production grid")
if m.get("component_contract",{}).get("actual_count")!=96: raise SystemExit("wrong particle count")
if any(m.get("physical_contract",{}).get(k) for k in ("GP_enabled","GP_birth_enabled","GP_release_enabled","external_source_enabled","new_beta_nucleation_enabled")): raise SystemExit("forbidden path enabled")
expected_optimizer = sys.argv[5].lower() == "true"
if m.get("assembly_contract",{}).get("optimizer_invoked") is not expected_optimizer: raise SystemExit("unexpected optimizer provenance")
pathlib.Path(sys.argv[2]).write_text(json.dumps({
 "fixture_manifest_sha256":sys.argv[3],
 "replicate_id":m["replicate_id"],
 "target_global_inventory":m["target_global_inventory"],
 "initial_canonical_inventory":m["initial_canonical_inventory"],
},indent=2,sort_keys=True)+"\n")
PY

cp "${PARAM_FILE}" "${RUN_ROOT}/provenance/pf_input_dt0p02.params"
cp "${FIXTURE_ROOT}/init_meta.json" \
  "${RUN_ROOT}/provenance/init_meta_dt0p02.json"
sha256sum \
  "${SOURCE_ROOT}/main_cuda" \
  "${SOURCE_ROOT}/main_cuda.cu" \
  "${SOURCE_ROOT}/cuda_kernels.cu" \
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.cpp" \
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.h" \
  "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_short_restart_v1.py" \
  "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
  "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${FIXTURE_ROOT}/phi.raw.f64" \
  "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
  "${FIXTURE_ROOT}/init_meta.json" \
  >"${RUN_ROOT}/provenance/input_hashes.sha256"

TRACKER_LINK_FLAGS=()
if [[ -n "${TRACKER_LDFLAGS:-}" ]]; then
  read -r -a TRACKER_LINK_FLAGS <<<"${TRACKER_LDFLAGS}"
fi
c++ -std=c++17 -O3 \
  "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  "${TRACKER_LINK_FLAGS[@]}" \
  -o "${RUN_ROOT}/provenance/analyze_pf_246cube_particle_lineage_v1"
sha256sum "${RUN_ROOT}/provenance/analyze_pf_246cube_particle_lineage_v1" \
  >"${RUN_ROOT}/provenance/analysis_binary.sha256"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -lms 500 \
    >"${RUN_ROOT}/gpu_samples.csv" 2>/dev/null &
  MONITOR_PID="$!"
else
  : >"${RUN_ROOT}/gpu_samples.csv"
fi

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

copy_mass_diagnostics() {
  local root="$1"
  local source
  source="$(
    find "${root}/results" -type f -name dynamics_mass_diagnostics.csv \
      -print -quit
  )"
  [[ -n "${source}" && -s "${source}" ]] || {
    echo "[fatal] missing dynamics mass diagnostics: ${root}" >&2
    return 2
  }
  cp "${source}" "${root}/dynamics_mass_diagnostics.csv"
}

run_fresh() {
  local label="$1" endpoint="$2" checkpoint="$3"
  local root="${RUN_ROOT}/${label}"
  mkdir -p "${root}/results"
  CUDA_STO_RESULTS_ROOT="${root}/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
    "${SOURCE_ROOT}/main_cuda" \
    "${GRID_N}" "${GRID_N}" "${GRID_N}" "${DT_CODE}" \
    "${endpoint}" "${endpoint}" 1 1 \
    --pf-param-file "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
    --mode dynamics \
    --init-mode raw_fields \
    --init-phi-raw "${FIXTURE_ROOT}/phi.raw.f64" \
    --init-xB-raw "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
    --init-meta "${RUN_ROOT}/provenance/init_meta_dt0p02.json" \
    "${ZERO_MODE_FLAGS[@]}" "${IDENTITY_FLAGS[@]}" \
    --enable-dynamics-mass-diagnostics \
    --dynamics-mass-diag-interval 32 \
    --pf-checkpoint-every "${endpoint}" \
    --pf-checkpoint-path "${checkpoint}" \
    --init-case-tag "pf_246cube_${label}_v1" \
    >"${root}/stdout.log" 2>"${root}/stderr.log"
  [[ ! -s "${root}/stderr.log" ]]
  grep -q '^PF_ZERO_MODE_FINAL_AUDIT status=PASS ' "${root}/stdout.log"
  copy_mass_diagnostics "${root}"
}

run_fresh continuous "${STEPS}" "${RUN_ROOT}/continuous/final.chk"
run_fresh restart_first_half "${HALF_STEPS}" \
  "${RUN_ROOT}/restart_first_half/half.chk"

mkdir -p "${RUN_ROOT}/restart_second_half/results"
CUDA_STO_RESULTS_ROOT="${RUN_ROOT}/restart_second_half/results" \
  CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
  "${SOURCE_ROOT}/main_cuda" \
  "${GRID_N}" "${GRID_N}" "${GRID_N}" "${DT_CODE}" \
  "${STEPS}" "${STEPS}" 1 1 \
  --pf-param-file "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
  --mode dynamics \
  --pf-restart-from "${RUN_ROOT}/restart_first_half/half.chk" \
  "${ZERO_MODE_FLAGS[@]}" "${IDENTITY_FLAGS[@]}" \
  --enable-dynamics-mass-diagnostics \
  --dynamics-mass-diag-interval 32 \
  --pf-checkpoint-every "${STEPS}" \
  --pf-checkpoint-path "${RUN_ROOT}/restart_second_half/final.chk" \
  --init-case-tag "pf_246cube_restart_second_half_v1" \
  >"${RUN_ROOT}/restart_second_half/stdout.log" \
  2>"${RUN_ROOT}/restart_second_half/stderr.log"
[[ ! -s "${RUN_ROOT}/restart_second_half/stderr.log" ]]
grep -q '^PF_ZERO_MODE_FINAL_AUDIT status=PASS ' \
  "${RUN_ROOT}/restart_second_half/stdout.log"
copy_mass_diagnostics "${RUN_ROOT}/restart_second_half"

cmp -s \
  "${RUN_ROOT}/continuous/final.chk" \
  "${RUN_ROOT}/restart_second_half/final.chk"

"${RUN_ROOT}/provenance/analyze_pf_246cube_particle_lineage_v1" \
  --initial-phi "${FIXTURE_ROOT}/phi.raw.f64" \
  --initial-xb "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
  --checkpoint "${RUN_ROOT}/continuous/final.chk" \
  --out "${RUN_ROOT}/lineage" \
  --grid "${GRID_N}" --dx-nm 1 --threshold 1e-4 \
  --physical-dt-s 0.9909260953431841 --start-age-h 6 \
  --target-mean 0.03 --expected-initial-count 96

python3 "${SOURCE_ROOT}/scripts/audit_pf_246cube_short_restart_v1.py" \
  --library-sha256 "${LIBRARY_SHA256}" \
  --fixture-manifest "${FIXTURE_ROOT}/fixture_manifest.json" \
  --continuous-checkpoint "${RUN_ROOT}/continuous/final.chk" \
  --restart-checkpoint "${RUN_ROOT}/restart_second_half/final.chk" \
  --continuous-stdout "${RUN_ROOT}/continuous/stdout.log" \
  --restart-stdout "${RUN_ROOT}/restart_second_half/stdout.log" \
  --continuous-stderr "${RUN_ROOT}/continuous/stderr.log" \
  --restart-stderr "${RUN_ROOT}/restart_second_half/stderr.log" \
  --continuous-mass-csv \
    "${RUN_ROOT}/continuous/dynamics_mass_diagnostics.csv" \
  --restart-mass-csv \
    "${RUN_ROOT}/restart_second_half/dynamics_mass_diagnostics.csv" \
  --lineage-root "${RUN_ROOT}/lineage" \
  --gpu-samples "${RUN_ROOT}/gpu_samples.csv" \
  --out "${RUN_ROOT}/audit"

grep -qx "PASS_246CUBE_SHORT_RESTART_AND_OBSERVABLES_V1" \
  "${RUN_ROOT}/audit/status.txt"
cp "${RUN_ROOT}/audit/status.txt" "${RUN_ROOT}/status.txt"
cat "${RUN_ROOT}/status.txt"
