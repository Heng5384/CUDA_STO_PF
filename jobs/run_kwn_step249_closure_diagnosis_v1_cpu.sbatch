#!/usr/bin/env bash
#SBATCH --job-name=kwn_step249_diag
#SBATCH --partition=uvip_serial
#SBATCH --qos=uvip_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Frozen, CPU-only step-249 nonlinear-map diagnosis.  This job never starts
# PF/CUDA, never advances candidate step 249, and never invokes a safeguard.

set -euo pipefail

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${FIXTURE_COMPAT_ROOT:?FIXTURE_COMPAT_ROOT is required}"
: "${PYTHON_BIN:=/data/home/luozhiheng/tmp/kwn_characteristic_reference_runtime_py311_np202_sp113_v1/bin/python}"
: "${FORMAL_TRACE_CSV:=/data/home/luozhiheng/tmp/kwn_characteristic_reference_v1_2ee679437421_20260902T202810Z/formal_all_14d_v4/outputs/kwn_characteristic_reference_v1/characteristic_fixed_point_trace.csv}"

readonly REQUIRED_BRANCH="codex/kwn-step249-closure-diagnosis-v1"
readonly BASE_COMMIT="512cc0a54cea75e23306d699a4224489226e812f"
readonly V1_CLOSURE_COMMIT="2ee679437421c750ccf2f2173d208df73bf28475"
readonly FROZEN_START_COMMIT="9269e07a2d0fafbf35be950b858373fb71b54e4b"
readonly TASK_NAME="kwn_step249_closure_diagnosis_v1"
readonly EXPECTED_FORMAL_TRACE_SHA256="e63e546ccd31d93d105b2bf40e3b5abb0fee9e9015f06eace45610a5654dad65"

[[ ! -e "${RUN_ROOT}" ]] || { echo "[fatal] refusing to overwrite RUN_ROOT=${RUN_ROOT}" >&2; exit 2; }
mkdir -- "${RUN_ROOT}"
mkdir -- "${RUN_ROOT}/logs"
exec >"${RUN_ROOT}/logs/driver.stdout.log" 2>"${RUN_ROOT}/logs/driver.stderr.log"

preflight_fail() {
  printf '[fatal] %s\n' "$*" >&2
  printf 'NONPASS_KWN_STEP249_CLOSURE_DIAGNOSIS_V1_PREFLIGHT\n' >"${RUN_ROOT}/status.txt"
  exit 2
}

[[ -e "${SOURCE_ROOT}/.git" ]] || preflight_fail "SOURCE_ROOT must be a Git worktree"
[[ -x "${PYTHON_BIN}" ]] || preflight_fail "PYTHON_BIN must be executable"
[[ -r "${FORMAL_TRACE_CSV}" ]] || preflight_fail "formal trace is unavailable"
[[ -d "${FIXTURE_COMPAT_ROOT}" ]] || preflight_fail "fixture compatibility root is unavailable"

source_commit="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" || preflight_fail "could not resolve source commit"
source_branch="$(git -C "${SOURCE_ROOT}" branch --show-current)" || preflight_fail "could not resolve source branch"
source_status="$(git -C "${SOURCE_ROOT}" status --short)" || preflight_fail "could not resolve source status"
[[ "${source_branch}" == "${REQUIRED_BRANCH}" ]] || preflight_fail "source branch differs from ${REQUIRED_BRANCH}"
[[ -z "${source_status}" ]] || preflight_fail "source tree is dirty"
git -C "${SOURCE_ROOT}" merge-base --is-ancestor "${BASE_COMMIT}" HEAD || preflight_fail "source does not descend from 512cc0a"
git -C "${SOURCE_ROOT}" merge-base --is-ancestor "${V1_CLOSURE_COMMIT}" HEAD || preflight_fail "source does not descend from 2ee6794"
git -C "${SOURCE_ROOT}" merge-base --is-ancestor "${FROZEN_START_COMMIT}" HEAD || preflight_fail "source does not descend from 9269e07"

python_version="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')" || preflight_fail "could not query Python"
numpy_version="$("${PYTHON_BIN}" -c 'import numpy; print(numpy.__version__)')" || preflight_fail "could not import NumPy"
scipy_version="$("${PYTHON_BIN}" -c 'import scipy; print(scipy.__version__)')" || preflight_fail "could not import SciPy"
[[ "${python_version}" == "3.11.13" ]] || preflight_fail "unexpected Python ${python_version}"
[[ "${numpy_version}" == "2.0.2" ]] || preflight_fail "unexpected NumPy ${numpy_version}"
[[ "${scipy_version}" == "1.13.1" ]] || preflight_fail "unexpected SciPy ${scipy_version}"

formal_trace_sha256="$(sha256sum "${FORMAL_TRACE_CSV}" | awk '{print $1}')"
[[ "${formal_trace_sha256}" == "${EXPECTED_FORMAL_TRACE_SHA256}" ]] || preflight_fail "formal trace hash differs"

fixture_root_realpath="$(realpath "${FIXTURE_COMPAT_ROOT}")" || preflight_fail "could not resolve fixture root"
runtime_fixture_link="${SOURCE_ROOT}/../CUDA_STO_PF"
if [[ -e "${runtime_fixture_link}" || -L "${runtime_fixture_link}" ]]; then
  [[ -L "${runtime_fixture_link}" ]] || preflight_fail "existing runtime fixture path is not a symlink"
  [[ "$(realpath "${runtime_fixture_link}")" == "${fixture_root_realpath}" ]] || preflight_fail "existing runtime fixture link differs"
else
  ln -s "${FIXTURE_COMPAT_ROOT}" "${runtime_fixture_link}" || preflight_fail "could not create fixture compatibility link"
fi

fixture_profile_root="${FIXTURE_COMPAT_ROOT}/data/qualification/pf_elastic_target_profile_quarter_nm_v2/profiles"
[[ -d "${fixture_profile_root}" ]] || preflight_fail "frozen fixture profiles are unavailable"
fixture_files=(
  R10p5/profile_manifest.json R10p5/phi.raw.f64 R10p5/delta_C_relaxation.raw.f64 R10p5/h_phi.raw.f64
  R9p5/profile_manifest.json R9p5/phi.raw.f64 R9p5/delta_C_relaxation.raw.f64 R9p5/h_phi.raw.f64
  R8p0/profile_manifest.json R8p0/phi.raw.f64 R8p0/delta_C_relaxation.raw.f64 R8p0/h_phi.raw.f64
)
for fixture_file in "${fixture_files[@]}"; do
  [[ -r "${fixture_profile_root}/${fixture_file}" ]] || preflight_fail "missing fixture input ${fixture_file}"
done

fixture_hashes="${RUN_ROOT}/fixture_input_sha256.txt"
(
  for fixture_file in "${fixture_files[@]}"; do
    sha256sum "${fixture_profile_root}/${fixture_file}"
  done
) >"${fixture_hashes}"

{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-UNSCHEDULED}"
  printf 'source_root=%s\n' "${SOURCE_ROOT}"
  printf 'source_commit=%s\n' "${source_commit}"
  printf 'source_branch=%s\n' "${source_branch}"
  printf 'source_status=%s\n' "${source_status}"
  printf 'base_commit=%s\n' "${BASE_COMMIT}"
  printf 'python_version=%s\n' "${python_version}"
  printf 'numpy_version=%s\n' "${numpy_version}"
  printf 'scipy_version=%s\n' "${scipy_version}"
  printf 'formal_trace_csv=%s\n' "${FORMAL_TRACE_CSV}"
  printf 'formal_trace_sha256=%s\n' "${formal_trace_sha256}"
  printf 'fixture_root=%s\n' "${fixture_root_realpath}"
  printf 'fixture_input_sha256_file=%s\n' "${fixture_hashes}"
  printf 'fixture_input_set_sha256=%s\n' "$(sha256sum "${fixture_hashes}" | awk '{print $1}')"
  printf 'scope=STEP248_REPLAY_THEN_READ_ONLY_STEP249_RAW_MAP_NO_PF_CUDA\n'
} >"${RUN_ROOT}/launch_provenance.txt"

export PYTHONPATH="${SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

set +e
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/run_kwn_step249_closure_diagnosis_v1.py" diagnose \
  --output-root "${RUN_ROOT}/outputs/${TASK_NAME}" \
  --report-root "${RUN_ROOT}/reports/${TASK_NAME}" \
  --formal-trace-csv "${FORMAL_TRACE_CSV}"
runner_rc=$?
set -e

printf '%s\n' "${runner_rc}" >"${RUN_ROOT}/runner_exit_code.txt"
if [[ "${runner_rc}" -eq 0 ]]; then
  printf 'PASS_KWN_STEP249_CLOSURE_DIAGNOSIS_V1\n' >"${RUN_ROOT}/status.txt"
else
  printf 'NONPASS_KWN_STEP249_CLOSURE_DIAGNOSIS_V1\n' >"${RUN_ROOT}/status.txt"
fi
exit "${runner_rc}"
