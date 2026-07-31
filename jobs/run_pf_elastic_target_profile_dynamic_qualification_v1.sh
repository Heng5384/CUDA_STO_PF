#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
PROFILE_ROOT="${PROFILE_ROOT:?PROFILE_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
DYNAMIC_OVERRIDE_FILE="${DYNAMIC_OVERRIDE_FILE:-${SOURCE_ROOT}/data/qualification/pf_elastic_target_profile_v1/physical_override_T380_dx1nm_lambda4nm_dynamic_dt0p02.json}"
DT_CODE="${DT_CODE:-0.02}"
REFINED_DT_CODE="${REFINED_DT_CODE:-0.01}"
STEPS="${STEPS:-64}"
REFINED_STEPS="${REFINED_STEPS:-128}"

if [[ -e "${RUN_ROOT}" ]]; then
  echo "[fatal] refusing to overwrite RUN_ROOT: ${RUN_ROOT}" >&2
  exit 2
fi
for path in \
  "${SOURCE_ROOT}/main_cuda" \
  "${PROFILE_ROOT}/profile_manifest.json" \
  "${PROFILE_ROOT}/phi.raw.f64" \
  "${PROFILE_ROOT}/xB_alpha.raw.f64" \
  "${PROFILE_ROOT}/init_meta.json" \
  "${DYNAMIC_OVERRIDE_FILE}"; do
  if [[ ! -e "${path}" ]]; then
    echo "[fatal] required input is missing: ${path}" >&2
    exit 2
  fi
done
mkdir -p "${RUN_ROOT}/provenance"

SOURCE_FILES=(
  main_cuda.cu
  cuda_kernels.cu
  cuda_kernels.h
  cuda_common.cu
  cuda_common.h
  pf_params.h
  pf_zero_mode_checkpoint.cpp
  pf_zero_mode_checkpoint.h
  Unit_Psedobinary.py
  scripts/materialize_pf_elastic_target_profile_v1.py
  scripts/assemble_pf_elastic_target_profile_library_v1.py
  scripts/test_pf_elastic_target_profile_v1.py
  jobs/run_pf_elastic_target_profile_library_v1.sh
)
for relative in "${SOURCE_FILES[@]}"; do
  if [[ ! -f "${SOURCE_ROOT}/${relative}" ]]; then
    echo "[fatal] runtime source-tree member is missing: ${relative}" >&2
    exit 2
  fi
  printf '%s  %s\n' \
    "$(sha256sum "${SOURCE_ROOT}/${relative}" | awk '{print $1}')" \
    "${relative}"
done | LC_ALL=C sort >"${RUN_ROOT}/provenance/runtime_source_files.sha256"
RUNTIME_SOURCE_TREE_SHA256="$(
  sha256sum "${RUN_ROOT}/provenance/runtime_source_files.sha256" |
    awk '{print $1}'
)"
PROFILE_SOURCE_TREE_SHA256="$(
  python3 - "${PROFILE_ROOT}/profile_manifest.json" <<'PY'
import json
import sys
print(json.load(open(sys.argv[1]))["source_tree_sha256"])
PY
)"
if [[ "${RUNTIME_SOURCE_TREE_SHA256}" != "${PROFILE_SOURCE_TREE_SHA256}" ]]; then
  echo "[fatal] runtime/profile source-tree mismatch: runtime=${RUNTIME_SOURCE_TREE_SHA256} profile=${PROFILE_SOURCE_TREE_SHA256}" >&2
  exit 2
fi
{
  printf 'runtime_source_tree_sha256=%s\n' "${RUNTIME_SOURCE_TREE_SHA256}"
  printf 'profile_source_tree_sha256=%s\n' "${PROFILE_SOURCE_TREE_SHA256}"
  printf 'source_tree_match=true\n'
} >"${RUN_ROOT}/provenance/source_tree_identity.txt"

python3 - "${PROFILE_ROOT}" "${RUN_ROOT}/provenance/profile_audit.json" <<'PY'
import hashlib
import json
import pathlib
import sys

root = pathlib.Path(sys.argv[1])
manifest_path = root / "profile_manifest.json"
manifest = json.loads(manifest_path.read_text())
if manifest.get("schema") != "PF_ELASTIC_TARGET_PROFILE_V1":
    raise SystemExit("wrong profile schema")
if "runtime_load_contract" not in manifest:
    raise SystemExit("profile has no runtime raw-load contract")
for name in ("phi", "xB_alpha"):
    field = manifest["fields"][name]
    path = root / field["path"]
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    if digest != field["sha256"]:
        raise SystemExit(f"field hash mismatch: {path}")
payload = {
    "profile_manifest": str(manifest_path),
    "profile_manifest_sha256": hashlib.sha256(
        manifest_path.read_bytes()
    ).hexdigest(),
    "target_radius_nm": manifest["geometry"]["target_equivalent_radius_nm"],
    "source_tree_sha256": manifest["source_tree_sha256"],
    "binary_sha256": manifest["binary_sha256"],
}
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY

export PHYSICAL_OVERRIDE_FILE="${DYNAMIC_OVERRIDE_FILE}"
export TEMP_C=380
PF_PARAM_DT02="$(site_generate_pf_param_file "elastic_profile_dynamic_dt0p02")"
cp "${PF_PARAM_DT02}" "${RUN_ROOT}/provenance/pf_input_dt0p02.params"
PF_PARAM_DT01="${RUN_ROOT}/provenance/pf_input_dt0p01.params"
python3 - "${PF_PARAM_DT02}" "${PF_PARAM_DT01}" "${REFINED_DT_CODE}" <<'PY'
import pathlib
import sys

source, target, dt = pathlib.Path(sys.argv[1]), pathlib.Path(sys.argv[2]), sys.argv[3]
lines = []
seen = False
for raw in source.read_text().splitlines():
    if raw.startswith("dt="):
        lines.append(f"dt={float(dt):.16e}")
        seen = True
    else:
        lines.append(raw)
if not seen:
    raise SystemExit("parameter file has no dt")
target.write_text("\n".join(lines) + "\n")
PY
META_DT01="${RUN_ROOT}/provenance/init_meta_dt0p01.json"
python3 - "${PROFILE_ROOT}/init_meta.json" "${META_DT01}" "${REFINED_DT_CODE}" <<'PY'
import json
import pathlib
import sys

payload = json.loads(pathlib.Path(sys.argv[1]).read_text())
payload["dt_recommended"] = float(sys.argv[3])
pathlib.Path(sys.argv[2]).write_text(
    json.dumps(payload, indent=2, sort_keys=True) + "\n"
)
PY
sha256sum \
  "${SOURCE_ROOT}/main_cuda" \
  "${PROFILE_ROOT}/profile_manifest.json" \
  "${PROFILE_ROOT}/phi.raw.f64" \
  "${PROFILE_ROOT}/xB_alpha.raw.f64" \
  "${PROFILE_ROOT}/init_meta.json" \
  "${DYNAMIC_OVERRIDE_FILE}" \
  "${SOURCE_ROOT}/scripts/qualify_pf_elastic_target_profile_dynamics_v1.py" \
  >"${RUN_ROOT}/provenance/runtime_inputs.sha256"

GRID_N="$(python3 - "${PROFILE_ROOT}/profile_manifest.json" <<'PY'
import json
import sys
d = json.load(open(sys.argv[1]))
g = d["grid"]
if not (g["Nx"] == g["Ny"] == g["Nz"]):
    raise SystemExit("dynamic qualifier currently requires a cubic profile")
print(g["Nx"])
PY
)"
CASE_TAG="elastic_target_profile_dynamic_v1"

run_fresh() {
  local label="$1"
  local dt="$2"
  local nsteps="$3"
  local param_file="$4"
  local meta_file="$5"
  local checkpoint="$6"
  local case_root="${RUN_ROOT}/${label}"
  mkdir -p "${case_root}/results"
  CUDA_STO_RESULTS_ROOT="${case_root}/results" \
  CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
    "${SOURCE_ROOT}/main_cuda" \
      "${GRID_N}" "${GRID_N}" "${GRID_N}" \
      "${dt}" "${nsteps}" "${nsteps}" 1 1 \
      --pf-param-file "${param_file}" \
      --mode dynamics \
      --init-mode raw_fields \
      --init-phi-raw "${PROFILE_ROOT}/phi.raw.f64" \
      --init-xB-raw "${PROFILE_ROOT}/xB_alpha.raw.f64" \
      --init-meta "${meta_file}" \
      --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 \
      --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 \
      --pf-zero-mode-tol-rel 1e-12 \
      --pf-zero-mode-max-iter 24 \
      --pf-checkpoint-every "${nsteps}" \
      --pf-checkpoint-path "${checkpoint}" \
      --init-case-tag "${CASE_TAG}" \
      >"${case_root}/stdout.log" 2>"${case_root}/stderr.log"
  if [[ -s "${case_root}/stderr.log" ]]; then
    echo "[fatal] non-empty stderr for ${label}" >&2
    exit 2
  fi
  grep -q "^PF_ZERO_MODE_FINAL_AUDIT status=PASS " \
    "${case_root}/stdout.log"
}

CONTINUOUS_CHK="${RUN_ROOT}/continuous/final.chk"
INITIAL_PROBE_CHK="${RUN_ROOT}/initial_probe/step1.chk"
HALF_CHK="${RUN_ROOT}/restart_first_half/half.chk"
RESTART_CHK="${RUN_ROOT}/restart_second_half/final.chk"
REFINED_CHK="${RUN_ROOT}/refined/final.chk"
mkdir -p \
  "$(dirname "${CONTINUOUS_CHK}")" \
  "$(dirname "${INITIAL_PROBE_CHK}")" \
  "$(dirname "${HALF_CHK}")" \
  "$(dirname "${RESTART_CHK}")" \
  "$(dirname "${REFINED_CHK}")"

run_fresh initial_probe "${DT_CODE}" 1 \
  "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
  "${PROFILE_ROOT}/init_meta.json" "${INITIAL_PROBE_CHK}"
run_fresh continuous "${DT_CODE}" "${STEPS}" \
  "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
  "${PROFILE_ROOT}/init_meta.json" "${CONTINUOUS_CHK}"
run_fresh restart_first_half "${DT_CODE}" "$((STEPS / 2))" \
  "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
  "${PROFILE_ROOT}/init_meta.json" "${HALF_CHK}"

case_root="${RUN_ROOT}/restart_second_half"
mkdir -p "${case_root}/results"
CUDA_STO_RESULTS_ROOT="${case_root}/results" \
CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
  "${SOURCE_ROOT}/main_cuda" \
    "${GRID_N}" "${GRID_N}" "${GRID_N}" \
    "${DT_CODE}" "${STEPS}" "${STEPS}" 1 1 \
    --pf-param-file "${RUN_ROOT}/provenance/pf_input_dt0p02.params" \
    --mode dynamics \
    --pf-restart-from "${HALF_CHK}" \
    --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 \
    --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 \
    --pf-zero-mode-tol-rel 1e-12 \
    --pf-zero-mode-max-iter 24 \
    --pf-checkpoint-every "${STEPS}" \
    --pf-checkpoint-path "${RESTART_CHK}" \
    --init-case-tag "${CASE_TAG}" \
    >"${case_root}/stdout.log" 2>"${case_root}/stderr.log"
if [[ -s "${case_root}/stderr.log" ]]; then
  echo "[fatal] non-empty stderr for restart_second_half" >&2
  exit 2
fi
grep -q "^PF_ZERO_MODE_FINAL_AUDIT status=PASS " "${case_root}/stdout.log"

run_fresh refined "${REFINED_DT_CODE}" "${REFINED_STEPS}" \
  "${RUN_ROOT}/provenance/pf_input_dt0p01.params" \
  "${META_DT01}" "${REFINED_CHK}"

python3 "${SOURCE_ROOT}/scripts/qualify_pf_elastic_target_profile_dynamics_v1.py" \
  --profile-manifest "${PROFILE_ROOT}/profile_manifest.json" \
  --initial-probe-checkpoint "${INITIAL_PROBE_CHK}" \
  --continuous-checkpoint "${CONTINUOUS_CHK}" \
  --restart-checkpoint "${RESTART_CHK}" \
  --refined-checkpoint "${REFINED_CHK}" \
  --initial-probe-stdout "${RUN_ROOT}/initial_probe/stdout.log" \
  --continuous-stdout "${RUN_ROOT}/continuous/stdout.log" \
  --restart-stdout "${RUN_ROOT}/restart_second_half/stdout.log" \
  --refined-stdout "${RUN_ROOT}/refined/stdout.log" \
  --out "${RUN_ROOT}/audit" \
  >"${RUN_ROOT}/audit.stdout" 2>"${RUN_ROOT}/audit.stderr"
if [[ -s "${RUN_ROOT}/audit.stderr" ]]; then
  echo "[fatal] dynamic qualification analyzer produced stderr" >&2
  exit 2
fi
grep -q \
  "^dynamic_profile_status=PASS_ELASTIC_TARGET_PROFILE_DYNAMIC_RESTART_AND_DT_V1$" \
  "${RUN_ROOT}/audit/final_terminal_output.txt"
cp "${RUN_ROOT}/audit/final_terminal_output.txt" "${RUN_ROOT}/status.txt"
cat "${RUN_ROOT}/status.txt"
