#!/usr/bin/env bash
set -euo pipefail

# Non-overwriting 246^3 / 96-particle conditional-path production runner.
# It starts from the original 6 h raw fixture, advances to registered step
# 152585, and preserves approximately hourly checkpoints plus every frozen
# science time.
# Multi-threshold merge-aware lineage is evaluated over the complete chain.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
FIXTURE_ROOT="${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
PARAM_FILE="${PARAM_FILE:?PARAM_FILE is required}"
REPLICATE="${REPLICATE:?REPLICATE is required (A, B, or C)}"
VALIDATE_ONLY=0
if [[ "${1:-}" == "--validate-only" ]]; then
  VALIDATE_ONLY=1
elif [[ "$#" -ne 0 ]]; then
  echo "usage: $0 [--validate-only]" >&2
  exit 2
fi

GRID_N=246
DT_CODE="0.02"
DT_PHYSICAL_S="0.9909260953431841"
FINAL_STEP=152585
CHECKPOINT_CADENCE=3633
INITIAL_STATE_CLASS="MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
LIBRARY_SHA256="${LIBRARY_SHA256:-58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe}"
CLUSTER_BINARY_SHA256="efb99c707acf7f22899425b8742c7dc06bb3231b9f21604cb5d75cf4565d8c94"
EXPECTED_BINARY_SHA256="${EXPECTED_BINARY_SHA256:-${CLUSTER_BINARY_SHA256}}"
PARAM_SHA256="${EXPECTED_PARAM_SHA256:-ecbdd0ac070bdf5e5d214322b5248a08f5ca5dd4e0670427513f5ef977ea977a}"
EXPECTED_OPTIMIZER_INVOKED="${EXPECTED_OPTIMIZER_INVOKED:-false}"
REGISTERED_STEPS=(21798 43596 65393 108989 152585)

case "${REPLICATE}" in
  A)
    DEFAULT_FIXTURE_SHA256="63a5080b01962bf19f72a37541ffde4302fec3a0e4dc9e519b0f759f30367de6"
    ;;
  B)
    DEFAULT_FIXTURE_SHA256="b37682e5aea7cc294a675ce562a34fb0d306181990d40090df1d20779a93980c"
    ;;
  C)
    DEFAULT_FIXTURE_SHA256="12c265be4e352392385e689c87ecaea1d18dc364428c111fab5bceb1eac0f981"
    ;;
  *)
    echo "[fatal] REPLICATE must be A, B, or C" >&2
    exit 2
    ;;
esac
EXPECTED_FIXTURE_SHA256="${EXPECTED_FIXTURE_SHA256:-${DEFAULT_FIXTURE_SHA256}}"

ENDPOINTS=()
for ((step=CHECKPOINT_CADENCE; step<FINAL_STEP; step+=CHECKPOINT_CADENCE)); do
  ENDPOINTS+=("${step}")
done
ENDPOINTS+=("${REGISTERED_STEPS[@]}")
SORTED_ENDPOINTS=()
while IFS= read -r step; do
  SORTED_ENDPOINTS+=("${step}")
done < <(printf '%s\n' "${ENDPOINTS[@]}" | sort -n -u)
ENDPOINTS=("${SORTED_ENDPOINTS[@]}")

required_inputs=(
  "${SOURCE_ROOT}/main_cuda"
  "${SOURCE_ROOT}/main_cuda.cu"
  "${SOURCE_ROOT}/cuda_kernels.cu"
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.cpp"
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.h"
  "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp"
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_merge_aware_lineage_v1.py"
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_6h48h_production_v1.py"
  "${FIXTURE_ROOT}/fixture_manifest.json"
  "${FIXTURE_ROOT}/initial_components.csv"
  "${FIXTURE_ROOT}/initial_particles.csv"
  "${FIXTURE_ROOT}/phi.raw.f64"
  "${FIXTURE_ROOT}/xB_alpha.raw.f64"
  "${FIXTURE_ROOT}/init_meta.json"
  "${PARAM_FILE}"
)
for path in "${required_inputs[@]}"; do
  [[ -f "${path}" ]] || {
    echo "[fatal] missing required input: ${path}" >&2
    exit 2
  }
done
[[ ! -e "${RUN_ROOT}" ]] || {
  echo "[fatal] refusing to overwrite RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
}

actual_binary_sha="$(sha256sum "${SOURCE_ROOT}/main_cuda" | awk '{print $1}')"
actual_param_sha="$(sha256sum "${PARAM_FILE}" | awk '{print $1}')"
actual_fixture_sha="$(
  sha256sum "${FIXTURE_ROOT}/fixture_manifest.json" | awk '{print $1}'
)"
[[ "${actual_binary_sha}" == "${EXPECTED_BINARY_SHA256}" ]] || {
  echo "[fatal] runtime binary hash mismatch: ${actual_binary_sha}" >&2
  exit 2
}
[[ "${actual_param_sha}" == "${PARAM_SHA256}" ]] || {
  echo "[fatal] production parameter hash mismatch: ${actual_param_sha}" >&2
  exit 2
}
[[ "${actual_fixture_sha}" == "${EXPECTED_FIXTURE_SHA256}" ]] || {
  echo "[fatal] fixture hash mismatch: ${actual_fixture_sha}" >&2
  exit 2
}

python3 - \
  "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${REPLICATE}" \
  "${EXPECTED_FIXTURE_SHA256}" \
  "${FIXTURE_ROOT}" \
  "${LIBRARY_SHA256}" \
  "${EXPECTED_OPTIMIZER_INVOKED}" <<'PY'
import hashlib
import json
import pathlib
import sys

path = pathlib.Path(sys.argv[1])
replicate = sys.argv[2]
expected_sha = sys.argv[3]
fixture_root = pathlib.Path(sys.argv[4])
manifest = json.loads(path.read_text(encoding="utf-8"))
if hashlib.sha256(path.read_bytes()).hexdigest() != expected_sha:
    raise SystemExit("fixture identity changed during preflight")
if manifest.get("schema") != "PF_246CUBE_LIBRARY_HANDOFF_MANIFEST_V1":
    raise SystemExit("wrong fixture schema")
if manifest.get("replicate_id") != f"replicate_{replicate}":
    raise SystemExit("replicate label mismatch")
if manifest.get("initial_state_class") != (
    "MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1"
):
    raise SystemExit("wrong initial-state class")
if manifest.get("profile_library_manifest_sha256") != sys.argv[5]:
    raise SystemExit("wrong profile library")
expected_optimizer = sys.argv[6].lower() == "true"
if manifest.get("assembly_contract", {}).get("optimizer_invoked") is not expected_optimizer:
    raise SystemExit("unexpected optimizer provenance")
if expected_optimizer and not manifest.get("inventory_selection"):
    raise SystemExit("optimized fixture does not pin inventory selection")
if manifest.get("grid") != {
    "Nx": 246,
    "Ny": 246,
    "Nz": 246,
    "dx_nm": 1.0,
    "lambda_sm_nm": 4.0,
}:
    raise SystemExit("wrong production grid")
if manifest.get("component_contract", {}).get("actual_count") != 96:
    raise SystemExit("wrong particle count")
physical = manifest.get("physical_contract", {})
for key in (
    "GP_enabled",
    "GP_birth_enabled",
    "GP_release_enabled",
    "external_source_enabled",
    "new_beta_nucleation_enabled",
):
    if physical.get(key) is not False:
        raise SystemExit(f"forbidden path is not explicitly false: {key}")
if physical.get("elasticity_enabled") is not True:
    raise SystemExit("elasticity is not enabled")
if physical.get("elastic_boundary") != "periodic_fixed_cell":
    raise SystemExit("wrong elastic boundary")
if physical.get("orientation_label") != "variant_100_identity":
    raise SystemExit("wrong orientation")
if physical.get("eigenstrain") != [0.046, -0.022, -0.017, 0.0, 0.0, 0.0]:
    raise SystemExit("wrong eigenstrain")
inventory = manifest.get("initial_canonical_inventory", {})
if (
    inventory.get("status") != "PASS_MACHINE_PRECISION"
    or float(inventory.get("field_relative_error", 1.0)) > 1.0e-14
):
    raise SystemExit("initial inventory is not machine-precision PASS")

for entry_name in ("phi", "xB_alpha"):
    entry = manifest.get("fields", {}).get(entry_name, {})
    field_path = fixture_root / entry.get("path", "")
    if not field_path.is_file():
        raise SystemExit(f"missing fixture field: {entry_name}")
    actual = hashlib.sha256(field_path.read_bytes()).hexdigest()
    if actual != entry.get("sha256"):
        raise SystemExit(f"fixture field hash mismatch: {entry_name}")

for entry_name in ("init_meta", "initial_components", "initial_particles"):
    entry = manifest.get(entry_name, {})
    auxiliary_path = fixture_root / entry.get("path", "")
    if not auxiliary_path.is_file():
        raise SystemExit(f"missing fixture auxiliary: {entry_name}")
    actual = hashlib.sha256(auxiliary_path.read_bytes()).hexdigest()
    if actual != entry.get("sha256"):
        raise SystemExit(f"fixture auxiliary hash mismatch: {entry_name}")
PY

TARGET_MEAN_C_BTOT="$(
  python3 - "${FIXTURE_ROOT}/fixture_manifest.json" <<'PY'
import json
import pathlib
import sys

manifest = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
value = float(manifest["target_global_inventory"]["mean_C_B_tot"])
if not 0.0 < value < 1.0:
    raise SystemExit("invalid target mean C_B_tot")
print(f"{value:.17g}")
PY
)"

if [[ "${VALIDATE_ONLY}" -eq 1 ]]; then
  available_bytes="$(df -PB1 "$(dirname "${RUN_ROOT}")" | awk 'NR==2 {print $4}')"
  checkpoint_bytes="$(
    python3 -c "print(476382752 * ${#ENDPOINTS[@]})"
  )"
  if (( available_bytes < checkpoint_bytes + 50000000000 )); then
    echo "[fatal] insufficient disk headroom for production root" >&2
    exit 2
  fi
  printf '%s\n' \
    "PASS_246CUBE_6H48H_PRODUCTION_PREFLIGHT_V1" \
    "replicate=${REPLICATE}" \
    "fixture_manifest_sha256=${actual_fixture_sha}" \
    "binary_sha256=${actual_binary_sha}" \
    "parameter_sha256=${actual_param_sha}" \
    "dt_code=${DT_CODE}" \
    "dt_physical_s=${DT_PHYSICAL_S}" \
    "final_step=${FINAL_STEP}" \
    "checkpoint_count=${#ENDPOINTS[@]}" \
    "checkpoint_cadence=${CHECKPOINT_CADENCE}" \
    "run_root=${RUN_ROOT}"
  exit 0
fi

mkdir -p \
  "${RUN_ROOT}/provenance" \
  "${RUN_ROOT}/segments" \
  "${RUN_ROOT}/checkpoints"
exec >"${RUN_ROOT}/driver.stdout.log" 2>"${RUN_ROOT}/driver.stderr.log"
MONITOR_PID=""
cleanup() {
  local rc=$?
  if [[ -n "${MONITOR_PID}" ]]; then
    kill "${MONITOR_PID}" 2>/dev/null || true
    wait "${MONITOR_PID}" 2>/dev/null || true
  fi
  if [[ "${rc}" -ne 0 ]]; then
    printf 'BLOCKED_246CUBE_6H48H_PRODUCTION_DRIVER_V1\nexit_code=%s\n' \
      "${rc}" >"${RUN_ROOT}/status.txt"
  fi
}
trap cleanup EXIT
printf 'RUNNING_246CUBE_6H48H_PRODUCTION_V1\n' >"${RUN_ROOT}/status.txt"

cp "${PARAM_FILE}" "${RUN_ROOT}/provenance/pf_input_dt0p02.params"
cp "${FIXTURE_ROOT}/init_meta.json" \
  "${RUN_ROOT}/provenance/init_meta_dt0p02.json"
printf '%s\n' "${ENDPOINTS[@]}" >"${RUN_ROOT}/provenance/checkpoint_steps.txt"
python3 - \
  "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${RUN_ROOT}/provenance/campaign_manifest.json" \
  "${REPLICATE}" \
  "${actual_fixture_sha}" \
  "${SLURM_JOB_ID:-UNSCHEDULED}" <<'PY'
import json
import pathlib
import sys

fixture = json.loads(pathlib.Path(sys.argv[1]).read_text(encoding="utf-8"))
out = {
    "schema": "PF_246CUBE_6H48H_PRODUCTION_CAMPAIGN_V1",
    "replicate": f"replicate_{sys.argv[3]}",
    "slurm_job_id": sys.argv[5],
    "fixture_manifest_sha256": sys.argv[4],
    "profile_library_manifest_sha256": fixture[
        "profile_library_manifest_sha256"
    ],
    "target_mean_C_Btot": fixture["target_global_inventory"]["mean_C_B_tot"],
    "dt_code": 0.02,
    "dt_physical_s": 0.9909260953431841,
    "final_step": 152585,
    "registered_science_steps": [0, 21798, 43596, 65393, 108989, 152585],
    "checkpoint_cadence_steps": 3633,
    "GP_enabled": False,
    "GP_birth_enabled": False,
    "GP_release_enabled": False,
    "external_source_enabled": False,
    "new_beta_nucleation_enabled": False,
    "concentration_gate_stops_path": False,
    "elastic_solver_mode": "ELASTIC_WARM_START_RESIDUAL_V1",
    "elastic_warm_start_enabled": True,
    "elastic_residual_control_enabled": True,
    "elastic_iter_min": 2,
    "elastic_iter_max": 32,
    "elastic_residual_tolerance": 1.0e-6,
    "elastic_fail_on_nonconvergence": True,
    "elastic_checkpoint_contract": "V4_WHEN_ZERO_MODE_CHECKPOINTED",
}
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(out, indent=2, sort_keys=True) + "\n",
    encoding="utf-8",
)
PY
sha256sum \
  "${SOURCE_ROOT}/main_cuda" \
  "${SOURCE_ROOT}/main_cuda.cu" \
  "${SOURCE_ROOT}/cuda_kernels.cu" \
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.cpp" \
  "${SOURCE_ROOT}/pf_zero_mode_checkpoint.h" \
  "${SOURCE_ROOT}/tools/analyze_pf_246cube_particle_lineage_v1.cpp" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_merge_aware_lineage_v1.py" \
  "${SOURCE_ROOT}/scripts/audit_pf_246cube_6h48h_production_v1.py" \
  "${SOURCE_ROOT}/jobs/run_pf_246cube_6h48h_production_v1.sh" \
  "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
  "${FIXTURE_ROOT}/fixture_manifest.json" \
  "${FIXTURE_ROOT}/initial_components.csv" \
  "${FIXTURE_ROOT}/initial_particles.csv" \
  "${FIXTURE_ROOT}/phi.raw.f64" \
  "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
  "${FIXTURE_ROOT}/init_meta.json" \
  "${RUN_ROOT}/provenance/campaign_manifest.json" \
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

{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-UNSCHEDULED}"
  printf 'cuda_visible_devices=%s\n' "${CUDA_VISIBLE_DEVICES:-UNSET}"
  if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=index,name,uuid,memory.total \
      --format=csv,noheader,nounits
  fi
} >"${RUN_ROOT}/provenance/device_identity.txt"

if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi \
    --query-gpu=timestamp,index,utilization.gpu,memory.used \
    --format=csv,noheader,nounits -lms 1000 \
    >"${RUN_ROOT}/gpu_samples.csv" 2>/dev/null &
  MONITOR_PID="$!"
else
  : >"${RUN_ROOT}/gpu_samples.csv"
fi

IDENTITY_FLAGS=(
  --pf-initial-state-class "${INITIAL_STATE_CLASS}"
  --pf-fixture-manifest-sha256 "${actual_fixture_sha}"
  --pf-profile-library-manifest-sha256 "${LIBRARY_SHA256}"
)
ZERO_MODE_FLAGS=(
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1
  --pf-zero-mode-tol-rel 1e-12
  --pf-zero-mode-max-iter 24
)

CURRENT_CHECKPOINT=""
CHECKPOINT_ARGS=()
for endpoint in "${ENDPOINTS[@]}"; do
  segment="${RUN_ROOT}/segments/step_${endpoint}"
  checkpoint="${RUN_ROOT}/checkpoints/step_${endpoint}.chk"
  mkdir -p "${segment}/results"
  common_args=(
    "${GRID_N}" "${GRID_N}" "${GRID_N}" "${DT_CODE}"
    "${endpoint}" "${endpoint}" 1 1
    --pf-param-file "${RUN_ROOT}/provenance/pf_input_dt0p02.params"
    --mode dynamics
  )
  if [[ -z "${CURRENT_CHECKPOINT}" ]]; then
    initialization_args=(
      --init-mode raw_fields
      --init-phi-raw "${FIXTURE_ROOT}/phi.raw.f64"
      --init-xB-raw "${FIXTURE_ROOT}/xB_alpha.raw.f64"
      --init-meta "${RUN_ROOT}/provenance/init_meta_dt0p02.json"
    )
  else
    initialization_args=(--pf-restart-from "${CURRENT_CHECKPOINT}")
  fi
  CUDA_STO_RESULTS_ROOT="${segment}/results" \
    CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
    "${SOURCE_ROOT}/main_cuda" \
    "${common_args[@]}" \
    "${initialization_args[@]}" \
    "${ZERO_MODE_FLAGS[@]}" "${IDENTITY_FLAGS[@]}" \
    --enable-dynamics-mass-diagnostics \
    --dynamics-mass-diag-interval "${endpoint}" \
    --pf-checkpoint-every "${endpoint}" \
    --pf-checkpoint-path "${checkpoint}" \
    --init-case-tag "pf_246cube_6h48h_${REPLICATE}_step_${endpoint}_v1" \
    >"${segment}/stdout.log" 2>"${segment}/stderr.log"
  [[ ! -s "${segment}/stderr.log" ]]
  grep -q '^PF_ZERO_MODE_FINAL_AUDIT status=PASS ' \
    "${segment}/stdout.log"
  mass_source="$(
    find "${segment}/results" -type f \
      -name dynamics_mass_diagnostics.csv -print -quit
  )"
  [[ -n "${mass_source}" && -s "${mass_source}" ]]
  cp "${mass_source}" "${segment}/dynamics_mass_diagnostics.csv"
  sha256sum "${checkpoint}" >"${segment}/checkpoint.sha256"
  printf 'PASS_246CUBE_6H48H_SEGMENT_V1\n' >"${segment}/status.txt"
  printf '%s %s\n' "${endpoint}" "${checkpoint}" \
    >"${RUN_ROOT}/latest_checkpoint.txt"
  CURRENT_CHECKPOINT="${checkpoint}"
  CHECKPOINT_ARGS+=(--checkpoint "${checkpoint}")
done
printf 'PF_COMPLETE_PENDING_MULTI_THRESHOLD_ANALYSIS\n' \
  >"${RUN_ROOT}/status.txt"

run_tracker() {
  local threshold="$1"
  local output="$2"
  "${RUN_ROOT}/provenance/analyze_pf_246cube_particle_lineage_v1" \
    --initial-phi "${FIXTURE_ROOT}/phi.raw.f64" \
    --initial-xb "${FIXTURE_ROOT}/xB_alpha.raw.f64" \
    "${CHECKPOINT_ARGS[@]}" \
    --out "${output}" \
    --grid "${GRID_N}" --dx-nm 1 --threshold "${threshold}" \
    --physical-dt-s "${DT_PHYSICAL_S}" --start-age-h 6 \
    --target-mean "${TARGET_MEAN_C_BTOT}" --expected-initial-count 96 \
    --allow-dissolution --allow-merge-groups
  grep -qx "PASS_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1" \
    "${output}/status.txt"
}

run_tracker 0.0001 "${RUN_ROOT}/lineage_h1e-4"
run_tracker 0.001 "${RUN_ROOT}/lineage_h1e-3"
run_tracker 0.005 "${RUN_ROOT}/lineage_h5e-3"

python3 "${SOURCE_ROOT}/scripts/audit_pf_246cube_merge_aware_lineage_v1.py" \
  --low "${RUN_ROOT}/lineage_h1e-4" \
  --medium "${RUN_ROOT}/lineage_h1e-3" \
  --strong "${RUN_ROOT}/lineage_h5e-3" \
  --initial-components "${FIXTURE_ROOT}/initial_components.csv" \
  --out "${RUN_ROOT}/merge_aware"
grep -qx "PASS_246CUBE_RESOLVED_MERGE_AWARE_LINEAGE_V1" \
  "${RUN_ROOT}/merge_aware/status.txt"

python3 "${SOURCE_ROOT}/scripts/audit_pf_246cube_6h48h_production_v1.py" \
  --library-sha256 "${LIBRARY_SHA256}" \
  --fixture-manifest "${FIXTURE_ROOT}/fixture_manifest.json" \
  --run-root "${RUN_ROOT}" \
  --lineage-root "${RUN_ROOT}/lineage_h1e-4" \
  --merge-aware-audit "${RUN_ROOT}/merge_aware/audit.json" \
  --gpu-samples "${RUN_ROOT}/gpu_samples.csv" \
  --out "${RUN_ROOT}/audit"
grep -qx "PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1" \
  "${RUN_ROOT}/audit/status.txt"
cp "${RUN_ROOT}/audit/status.txt" "${RUN_ROOT}/status.txt"
cat "${RUN_ROOT}/status.txt"
