#!/usr/bin/env bash
# Controlled GPU validation of the frozen 96^3 CUDA A--E fixture.
# This file only runs a supplied job allocation; it never calls sbatch.

#SBATCH --job-name=kwn_pf_cuda_ae_v1
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=12:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${RUN_ROOT:?RUN_ROOT is required and must not already exist}"
: "${PARAM_FILE:?PARAM_FILE is required (frozen PF parameter file)}"

SOURCE_ROOT="${SOURCE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
ASSET_ROOT="${ASSET_ROOT:-${SOURCE_ROOT}/outputs/kwn_pf_cuda_runtime_closure_v1/cuda_ae_assets}"
CONTRACT_PATH="${CONTRACT_PATH:-${SOURCE_ROOT}/contracts/pf_kwn_validation_contract_v1.json}"
PROFILE_LIBRARY_MANIFEST_SHA256="${PROFILE_LIBRARY_MANIFEST_SHA256:-58803a8bc6679b823e45e7a7b85df16ae68efa55338d52d4c4151b414a5ef0fe}"
MAX_STAGE="${MAX_STAGE:-R2}"
CUDA_ARCH="${CUDA_ARCH:-sm_80}"

case "${MAX_STAGE}" in R0|R1|R2|R3|R4) ;; *) echo "[fatal] MAX_STAGE must be R0..R4" >&2; exit 2 ;; esac
[[ "${CUDA_ARCH}" == "sm_80" ]] || { echo "[fatal] this validation is pinned to CUDA_ARCH=sm_80" >&2; exit 2; }
[[ -d "${SOURCE_ROOT}/.git" && -f "${SOURCE_ROOT}/main_cuda.cu" ]] || { echo "[fatal] invalid SOURCE_ROOT" >&2; exit 2; }
[[ -f "${PARAM_FILE}" && -f "${ASSET_ROOT}/cuda_ae_asset_index.json" && -f "${CONTRACT_PATH}" ]] || { echo "[fatal] missing PF input, assets, or contract" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "[fatal] refusing to overwrite RUN_ROOT: ${RUN_ROOT}" >&2; exit 2; }

# This determines whether a controlled-build provenance record is meaningful.
# If it fails, stage/commit the intended source instead of launching an
# unbound CUDA binary.
if [[ -n "$(git -C "${SOURCE_ROOT}" status --porcelain=v1 --untracked-files=all)" ]]; then
  echo "[fatal] controlled CUDA build requires a clean SOURCE_ROOT" >&2
  exit 2
fi

mkdir -p "${RUN_ROOT}"
exec >"${RUN_ROOT}/driver.stdout.log" 2>"${RUN_ROOT}/driver.stderr.log"
trap 'rc=$?; if [[ "$rc" -ne 0 ]]; then printf "FAIL_CUDA_AE_RUNTIME_CLOSURE_V1\nexit_code=%s\n" "$rc" >"${RUN_ROOT}/status.txt"; fi' EXIT
printf 'RUNNING_CUDA_AE_RUNTIME_CLOSURE_V1\n' >"${RUN_ROOT}/status.txt"

cd "${SOURCE_ROOT}"
# shellcheck source=jobs/_site_env.sh
source jobs/_site_env.sh
site_prepare_cuda_env
[[ "${CUDA_ARCH}" == "sm_80" ]] || { echo "[fatal] site CUDA architecture is not sm_80" >&2; exit 2; }

{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  hostname
  git rev-parse HEAD
  nvcc --version
  nvidia-smi
  env | LC_ALL=C sort | grep -E '^(SLURM_|CUDA_|NVCC=|LD_LIBRARY_PATH=)' || true
} >"${RUN_ROOT}/runtime_environment.txt"
cp "${PARAM_FILE}" "${RUN_ROOT}/pf_input.params"
PARAM_USED="${RUN_ROOT}/pf_input.params"
sha256sum "${PARAM_FILE}" "${PARAM_USED}" >"${RUN_ROOT}/pf_input.sha256"

# The asset index records preparer's absolute local paths.  Copy/rewrite only
# the asset-root descendants, retain original and copied hashes, then validate
# the relocated index before any GPU work.
python3 scripts/stage_kwn_pf_cuda_ae_assets_for_cluster_v1.py \
  --asset-root "${ASSET_ROOT}" --out "${RUN_ROOT}/staged_assets" \
  >"${RUN_ROOT}/asset_staging.json"
ASSET_INDEX="${RUN_ROOT}/staged_assets/cuda_ae_asset_index.json"
python3 scripts/audit_kwn_pf_cuda_ae_runtime_v1.py \
  --asset-index "${ASSET_INDEX}" --profile-library-sha256 "${PROFILE_LIBRARY_MANIFEST_SHA256}" \
  --emit-run-manifest "${RUN_ROOT}/checked_asset_cases.tsv"

ASSET_CONTRACT_HASH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["contract"]["hash"])' "${ASSET_INDEX}")"
FIXTURE_HASH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["fixture"]["hash"])' "${ASSET_INDEX}")"
[[ "${ASSET_CONTRACT_HASH}" == "d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333" ]] || { echo "[fatal] unexpected validation contract" >&2; exit 2; }
[[ "${FIXTURE_HASH}" == "f1247cb66419af764b97de2f7843fc6de2049459d78550bc603edd8e88d9134f" ]] || { echo "[fatal] unexpected fixture hash" >&2; exit 2; }

make controlled_cuda "CUDA_ROOT=${CUDA_ROOT}" "NVCC=${NVCC}" "CUDA_ARCH=sm_80" \
  "CUDA_CONTROLLED_BUILD_DIR=${RUN_ROOT}/build"
BIN="${RUN_ROOT}/build/main_cuda"
[[ -x "${BIN}" ]] || { echo "[fatal] controlled binary is missing" >&2; exit 2; }
"${BIN}" --provenance >"${RUN_ROOT}/build/main_cuda.embedded_provenance.json"
python3 -c '
import hashlib,json,pathlib,sys
m=json.load(open(sys.argv[1])); e=json.load(open(sys.argv[2])); expected=sys.argv[3]
assert m["status"] == "PASS_CONTROLLED_CUDA_BINARY_PROVENANCE_V1"
assert m["controlled_binary_eligible"] is True and m["build_provenance"] == e
assert e["contract"]["canonical_hash"] == expected
b=pathlib.Path(sys.argv[1]).parent/m["artifacts"]["binary"]["path"]
assert hashlib.sha256(b.read_bytes()).hexdigest() == m["artifacts"]["binary"]["sha256"]
' "${RUN_ROOT}/build/controlled_binary_manifest.json" "${RUN_ROOT}/build/main_cuda.embedded_provenance.json" "${ASSET_CONTRACT_HASH}"
python3 -c '
import hashlib,json,os,pathlib,subprocess,sys
root=pathlib.Path(sys.argv[1]); source=pathlib.Path(sys.argv[2]); staged=pathlib.Path(sys.argv[3])
def sha(p):
 h=hashlib.sha256(); h.update(pathlib.Path(p).read_bytes()); return h.hexdigest()
document={
 "schema_version":"PF_CUDA_AE_RUNTIME_RUN_MANIFEST_V1", "status":"PREPARED_NOT_YET_INTERPRETED", "validation_only":True,
 "historical_as_run_claim":False, "source_commit":subprocess.check_output(["git","-C",str(source),"rev-parse","HEAD"],text=True).strip(),
 "slurm_job_id":os.environ.get("SLURM_JOB_ID"), "grid":[96,96,96], "cuda_arch":"sm_80", "max_stage":sys.argv[4],
 "contract_hash":sys.argv[5], "fixture_hash":sys.argv[6], "profile_library_manifest_sha256":sys.argv[7],
 "parameter_file":{"path":str(root/"pf_input.params"),"sha256":sha(root/"pf_input.params")},
 "asset_relocation":json.load(open(root/"asset_staging.json")),
 "controlled_binary_manifest":json.load(open(root/"build/controlled_binary_manifest.json")),
 "embedded_binary_provenance":json.load(open(root/"build/main_cuda.embedded_provenance.json")),
 "prohibited_dynamics":{"GP_release":"OFF","GP_to_beta_conversion":"OFF","beta_birth":"OFF","online_KWN_callback":"OFF","matrix_global_reset":"OFF","composition_clamp":"FORBIDDEN"},
}
(root/"run_manifest.json").write_text(json.dumps(document,indent=2,sort_keys=True)+"\\n")
' "${RUN_ROOT}" "${SOURCE_ROOT}" "${RUN_ROOT}/staged_assets" "${MAX_STAGE}" "${ASSET_CONTRACT_HASH}" "${FIXTURE_HASH}" "${PROFILE_LIBRARY_MANIFEST_SHA256}"

declare -A PHI XB META SIDECAR SOURCE_HASH PACKAGE_HASH
while IFS=$'\t' read -r case phi xb meta sidecar source_hash package_hash _fixture _index; do
  [[ "${case}" == "case" ]] && continue
  PHI["${case}"]="${phi}"; XB["${case}"]="${xb}"; META["${case}"]="${meta}"
  SIDECAR["${case}"]="${sidecar}"; SOURCE_HASH["${case}"]="${source_hash}"; PACKAGE_HASH["${case}"]="${package_hash}"
done <"${RUN_ROOT}/checked_asset_cases.tsv"
for case in A B C D E; do
  [[ -n "${PHI[$case]:-}" && -n "${XB[$case]:-}" && -n "${META[$case]:-}" ]] || { echo "[fatal] missing checked asset case ${case}" >&2; exit 2; }
done
mkdir -p "${RUN_ROOT}/checkpoints" "${RUN_ROOT}/logs" "${RUN_ROOT}/results" "${RUN_ROOT}/compact_audit"

run_logged() {
  local label="$1" stdout="$2" stderr="$3"; shift 3
  printf '%q ' "$@" >"${stdout%.stdout.log}.command.txt"; printf '\n' >>"${stdout%.stdout.log}.command.txt"
  set +e; "$@" >"${stdout}" 2>"${stderr}"; local rc=$?; set -e
  printf '%s\n' "${rc}" >"${stdout%.stdout.log}.exit_code"
  [[ "${rc}" -eq 0 ]] || { echo "[fatal] ${label} exited ${rc}" >&2; return "${rc}"; }
}

log_base() {
  local case="$1" step="$2" tag="$3"
  if [[ "${tag}" == R* ]]; then printf '%s/logs/%s/step_%s' "${RUN_ROOT}" "${case}" "${step}";
  else printf '%s/logs/qualification/%s/step_%s_%s' "${RUN_ROOT}" "${case}" "${step}" "${tag}"; fi
}

fresh() {
  local case="$1" step="$2" checkpoint="$3" tag="$4" initial_only="$5"
  local base; base="$(log_base "${case}" "${step}" "${tag}")"
  mkdir -p "$(dirname "${checkpoint}")" "$(dirname "${base}")" "${RUN_ROOT}/results/${case}/${tag}"
  local a=("${BIN}" 96 96 96 0.02 "${step}" "${step}" "${step}" 1 --pf-param-file "${PARAM_USED}" --mode dynamics
    --init-mode raw_fields --init-phi-raw "${PHI[$case]}" --init-xB-raw "${XB[$case]}" --init-meta "${META[$case]}"
    --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 --pf-zero-mode-tol-rel 1e-12 --pf-zero-mode-max-iter 24
    --pf-checkpoint-every "${step}" --pf-checkpoint-path "${checkpoint}"
    --pf-initial-state-class MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1 --pf-fixture-manifest-sha256 "${FIXTURE_HASH}"
    --pf-profile-library-manifest-sha256 "${PROFILE_LIBRARY_MANIFEST_SHA256}" --init-case-tag "cuda_ae_${case}_${tag}")
  # main_cuda validates nsteps>0 before its R0 early exit; the flag itself
  # guarantees no accepted PF step, so use a harmless positive endpoint.
  if [[ "${initial_only}" == 1 ]]; then a[5]=1; a[6]=1; a[7]=1; a+=(--pf-initial-checkpoint-only --pf-checkpoint-every 0); fi
  if [[ "${SIDECAR[$case]}" != "-" ]]; then
    a+=(--pf-auxiliary-sidecar "${SIDECAR[$case]}" --pf-auxiliary-source-handoff-sha256 "${SOURCE_HASH[$case]}" --pf-auxiliary-package-sha256 "${PACKAGE_HASH[$case]}" --pf-auxiliary-fixture-sha256 "${FIXTURE_HASH}")
  fi
  CUDA_STO_RESULTS_ROOT="${RUN_ROOT}/results/${case}/${tag}" run_logged "${case}:${tag}" "${base}.stdout.log" "${base}.stderr.log" "${a[@]}"
}

restart() {
  local case="$1" step="$2" prior="$3" checkpoint="$4" tag="$5"
  local base; base="$(log_base "${case}" "${step}" "${tag}")"
  mkdir -p "$(dirname "${checkpoint}")" "$(dirname "${base}")" "${RUN_ROOT}/results/${case}/${tag}"
  local a=("${BIN}" 96 96 96 0.02 "${step}" "${step}" "${step}" 1 --pf-param-file "${PARAM_USED}" --mode dynamics
    --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 --pf-zero-mode-tol-rel 1e-12 --pf-zero-mode-max-iter 24
    --pf-restart-from "${prior}" --pf-checkpoint-every "${step}" --pf-checkpoint-path "${checkpoint}"
    --pf-initial-state-class MASS_CONSERVING_LIBRARY_ASSEMBLED_CONDITIONAL_HANDOFF_V1 --pf-fixture-manifest-sha256 "${FIXTURE_HASH}"
    --pf-profile-library-manifest-sha256 "${PROFILE_LIBRARY_MANIFEST_SHA256}" --init-case-tag "cuda_ae_${case}_${tag}")
  CUDA_STO_RESULTS_ROOT="${RUN_ROOT}/results/${case}/${tag}" run_logged "${case}:${tag}" "${base}.stdout.log" "${base}.stderr.log" "${a[@]}"
}

gate() {
  python3 scripts/audit_kwn_pf_cuda_ae_runtime_v1.py --asset-index "${ASSET_INDEX}" --run-root "${RUN_ROOT}" --out-dir "${RUN_ROOT}/compact_audit" --contract "${CONTRACT_PATH}" --profile-library-sha256 "${PROFILE_LIBRARY_MANIFEST_SHA256}" --require-stage "$1"
}

for case in A B C D E; do fresh "${case}" 0 "${RUN_ROOT}/checkpoints/${case}/step_0.pfzck" R0 1; done
gate R0
[[ "${MAX_STAGE}" == R0 ]] && { echo PASS_R0_CUDA_AE_RUNTIME_GATE >"${RUN_ROOT}/status.txt"; exit 0; }

for case in A B C D E; do
  restart "${case}" 1 "${RUN_ROOT}/checkpoints/${case}/step_0.pfzck" "${RUN_ROOT}/checkpoints/${case}/step_1.pfzck" R1a
  restart "${case}" 10 "${RUN_ROOT}/checkpoints/${case}/step_1.pfzck" "${RUN_ROOT}/checkpoints/${case}/step_10.pfzck" R1b
done
gate R1
[[ "${MAX_STAGE}" == R1 ]] && { echo PASS_R1_CUDA_AE_RUNTIME_GATE >"${RUN_ROOT}/status.txt"; exit 0; }

for case in A B C D E; do
  restart "${case}" 36 "${RUN_ROOT}/checkpoints/${case}/step_10.pfzck" "${RUN_ROOT}/checkpoints/${case}/step_36.pfzck" R2a
  restart "${case}" 363 "${RUN_ROOT}/checkpoints/${case}/step_36.pfzck" "${RUN_ROOT}/checkpoints/${case}/step_363.pfzck" R2b
done
gate R2
[[ "${MAX_STAGE}" == R2 ]] && { echo PASS_R2_CUDA_AE_RUNTIME_GATE >"${RUN_ROOT}/status.txt"; exit 0; }

for case in A B C D E; do
  restart "${case}" 3633 "${RUN_ROOT}/checkpoints/${case}/step_363.pfzck" "${RUN_ROOT}/checkpoints/${case}/step_3633.pfzck" R3a
  restart "${case}" 10899 "${RUN_ROOT}/checkpoints/${case}/step_3633.pfzck" "${RUN_ROOT}/checkpoints/${case}/step_10899.pfzck" R3b
  restart "${case}" 21798 "${RUN_ROOT}/checkpoints/${case}/step_10899.pfzck" "${RUN_ROOT}/checkpoints/${case}/step_21798.pfzck" R3c
done
for case in A B E; do
  q="${RUN_ROOT}/restart_qualification/${case}"
  fresh "${case}" 21798 "${q}/continuous_6h.pfzck" continuous_6h 0
  fresh "${case}" 10899 "${q}/first_3h.pfzck" first_3h 0
  restart "${case}" 21798 "${q}/first_3h.pfzck" "${q}/restart_6h.pfzck" restart_6h
done
gate R3
[[ "${MAX_STAGE}" == R3 ]] && { echo PASS_R3_CUDA_AE_RUNTIME_GATE_DIAGNOSTICS_PENDING >"${RUN_ROOT}/status.txt"; exit 0; }

for case in A B C D E; do
  restart "${case}" 43596 "${RUN_ROOT}/checkpoints/${case}/step_21798.pfzck" "${RUN_ROOT}/checkpoints/${case}/step_43596.pfzck" R4a
  restart "${case}" 87191 "${RUN_ROOT}/checkpoints/${case}/step_43596.pfzck" "${RUN_ROOT}/checkpoints/${case}/step_87191.pfzck" R4b
  restart "${case}" 174382 "${RUN_ROOT}/checkpoints/${case}/step_87191.pfzck" "${RUN_ROOT}/checkpoints/${case}/step_174382.pfzck" R4c
done
q="${RUN_ROOT}/restart_qualification/E"
fresh E 174382 "${q}/continuous_48h.pfzck" continuous_48h 0
fresh E 87191 "${q}/first_24h.pfzck" first_24h 0
restart E 174382 "${q}/first_24h.pfzck" "${q}/restart_48h.pfzck" restart_48h
gate R4
echo PASS_R4_CUDA_AE_RUNTIME_GATE_DIAGNOSTICS_PENDING >"${RUN_ROOT}/status.txt"
