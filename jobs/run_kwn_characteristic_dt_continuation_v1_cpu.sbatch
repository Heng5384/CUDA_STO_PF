#!/usr/bin/env bash
#SBATCH --job-name=kwn_dtcont_v1
#SBATCH --partition=uvip_serial
#SBATCH --qos=uvip_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Phase A only: CPU-only KWN characteristic dt-continuation local-window
# qualification.  This job never starts PF/CUDA and never advances into the
# CR1 ladder or cohort-parity phases.  A caller must stage a clean worktree,
# bind its exact head through EXPECTED_SOURCE_COMMIT, and provide a new
# non-existent RUN_ROOT.

set -euo pipefail

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${FIXTURE_COMPAT_ROOT:?FIXTURE_COMPAT_ROOT is required}"
: "${EXPECTED_SOURCE_COMMIT:?EXPECTED_SOURCE_COMMIT is required}"
: "${PYTHON_BIN:=/data/home/luozhiheng/tmp/kwn_characteristic_reference_runtime_py311_np202_sp113_v1/bin/python}"
: "${FORMAL_TRACE_CSV:=/data/home/luozhiheng/tmp/kwn_characteristic_reference_v1_2ee679437421_20260902T202810Z/formal_all_14d_v4/outputs/kwn_characteristic_reference_v1/characteristic_fixed_point_trace.csv}"

readonly REQUIRED_BRANCH="codex/kwn-characteristic-dt-continuation-v1"
readonly CONTINUATION_START_COMMIT="e372a6e6b3401db697bf6a9d1a8a5d70fb27ec8a"
readonly STEP249_DIAGNOSIS_COMMIT="e372a6e6b3401db697bf6a9d1a8a5d70fb27ec8a"
readonly V1_CLOSURE_COMMIT="2ee679437421c750ccf2f2173d208df73bf28475"
readonly FROZEN_START_COMMIT="9269e07a2d0fafbf35be950b858373fb71b54e4b"
readonly TASK_NAME="kwn_characteristic_dt_continuation_v1"
readonly PHASE_A_COMMAND="local-window"
readonly EXPECTED_FORMAL_TRACE_SHA256="e63e546ccd31d93d105b2bf40e3b5abb0fee9e9015f06eace45610a5654dad65"
readonly EXPECTED_FIXTURE_INPUT_SET_SHA256="c97de66ec1a89c00f4badd2d3546e9f744ae9dfd10fdadc10cb9941de10da9ae"

[[ "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || {
  printf '[fatal] EXPECTED_SOURCE_COMMIT must be a lowercase 40-character Git commit\n' >&2
  exit 2
}
[[ ! -e "${RUN_ROOT}" ]] || {
  printf '[fatal] refusing to overwrite RUN_ROOT=%s\n' "${RUN_ROOT}" >&2
  exit 2
}
mkdir -- "${RUN_ROOT}"
mkdir -- "${RUN_ROOT}/logs"
exec >"${RUN_ROOT}/logs/driver.stdout.log" 2>"${RUN_ROOT}/logs/driver.stderr.log"

preflight_fail() {
  printf '[fatal] %s\n' "$*" >&2
  printf 'NONPASS_KWN_CHARACTERISTIC_DT_CONTINUATION_V1_PREFLIGHT\n' >"${RUN_ROOT}/status.txt"
  exit 2
}

[[ -e "${SOURCE_ROOT}/.git" ]] || preflight_fail "SOURCE_ROOT must be a Git worktree"
[[ -x "${PYTHON_BIN}" ]] || preflight_fail "PYTHON_BIN must be executable"
[[ -r "${FORMAL_TRACE_CSV}" ]] || preflight_fail "formal step-244 trace is unavailable"
[[ -d "${FIXTURE_COMPAT_ROOT}" ]] || preflight_fail "fixture compatibility root is unavailable"
[[ -r "${SOURCE_ROOT}/scripts/run_kwn_characteristic_dt_continuation_v1.py" ]] || \
  preflight_fail "Phase-A runner script is unavailable"

source_commit="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" || preflight_fail "could not resolve source commit"
source_branch="$(git -C "${SOURCE_ROOT}" branch --show-current)" || preflight_fail "could not resolve source branch"
source_status="$(git -C "${SOURCE_ROOT}" status --short)" || preflight_fail "could not resolve source status"
git -C "${SOURCE_ROOT}" rev-parse --verify "${EXPECTED_SOURCE_COMMIT}^{commit}" >/dev/null || \
  preflight_fail "EXPECTED_SOURCE_COMMIT is not present in SOURCE_ROOT"
[[ "${source_branch}" == "${REQUIRED_BRANCH}" ]] || \
  preflight_fail "source branch differs from ${REQUIRED_BRANCH}: ${source_branch}"
[[ "${source_commit}" == "${EXPECTED_SOURCE_COMMIT}" ]] || \
  preflight_fail "source HEAD differs from EXPECTED_SOURCE_COMMIT"
[[ -z "${source_status}" ]] || preflight_fail "source tree is dirty"
git -C "${SOURCE_ROOT}" merge-base --is-ancestor "${CONTINUATION_START_COMMIT}" HEAD || \
  preflight_fail "source does not descend from frozen continuation start e372a6e"
git -C "${SOURCE_ROOT}" merge-base --is-ancestor "${STEP249_DIAGNOSIS_COMMIT}" HEAD || \
  preflight_fail "source does not descend from frozen step-249 diagnosis e372a6e"
git -C "${SOURCE_ROOT}" merge-base --is-ancestor "${V1_CLOSURE_COMMIT}" HEAD || \
  preflight_fail "source does not descend from frozen v1 closure 2ee6794"
git -C "${SOURCE_ROOT}" merge-base --is-ancestor "${FROZEN_START_COMMIT}" HEAD || \
  preflight_fail "source does not descend from frozen start 9269e07"

python_version="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')" || \
  preflight_fail "could not query frozen Python version"
numpy_version="$("${PYTHON_BIN}" -c 'import numpy; print(numpy.__version__)')" || \
  preflight_fail "could not import NumPy in frozen runtime"
scipy_version="$("${PYTHON_BIN}" -c 'import scipy; print(scipy.__version__)')" || \
  preflight_fail "could not import SciPy in frozen runtime"
[[ "${python_version}" == "3.11.13" ]] || preflight_fail "unexpected Python ${python_version}"
[[ "${numpy_version}" == "2.0.2" ]] || preflight_fail "unexpected NumPy ${numpy_version}"
[[ "${scipy_version}" == "1.13.1" ]] || preflight_fail "unexpected SciPy ${scipy_version}"

formal_trace_sha256="$(sha256sum "${FORMAL_TRACE_CSV}" | awk '{print $1}')"
[[ "${formal_trace_sha256}" == "${EXPECTED_FORMAL_TRACE_SHA256}" ]] || \
  preflight_fail "formal trace hash differs from the frozen reference"

fixture_root_realpath="$(realpath "${FIXTURE_COMPAT_ROOT}")" || \
  preflight_fail "could not resolve fixture compatibility root"
runtime_fixture_link="${SOURCE_ROOT}/../CUDA_STO_PF"
if [[ -e "${runtime_fixture_link}" || -L "${runtime_fixture_link}" ]]; then
  [[ -L "${runtime_fixture_link}" ]] || \
    preflight_fail "existing runtime fixture path must be a symlink"
  [[ "$(realpath "${runtime_fixture_link}")" == "${fixture_root_realpath}" ]] || \
    preflight_fail "existing runtime fixture link differs from frozen fixture root"
  runtime_fixture_link_created=false
else
  ln -s "${FIXTURE_COMPAT_ROOT}" "${runtime_fixture_link}" || \
    preflight_fail "could not create fixture compatibility link"
  runtime_fixture_link_created=true
fi

fixture_profile_root="${FIXTURE_COMPAT_ROOT}/data/qualification/pf_elastic_target_profile_quarter_nm_v2/profiles"
runtime_profile_root="${runtime_fixture_link}/data/qualification/pf_elastic_target_profile_quarter_nm_v2/profiles"
[[ -d "${fixture_profile_root}" ]] || preflight_fail "frozen fixture profiles are unavailable"
[[ -d "${runtime_profile_root}" ]] || preflight_fail "runtime fixture profiles are unavailable"
[[ "$(realpath "${runtime_profile_root}")" == "$(realpath "${fixture_profile_root}")" ]] || \
  preflight_fail "runtime fixture profiles differ from frozen fixture profiles"
fixture_files=(
  R10p5/profile_manifest.json R10p5/phi.raw.f64 R10p5/delta_C_relaxation.raw.f64 R10p5/h_phi.raw.f64
  R9p5/profile_manifest.json R9p5/phi.raw.f64 R9p5/delta_C_relaxation.raw.f64 R9p5/h_phi.raw.f64
  R8p0/profile_manifest.json R8p0/phi.raw.f64 R8p0/delta_C_relaxation.raw.f64 R8p0/h_phi.raw.f64
)
for fixture_file in "${fixture_files[@]}"; do
  [[ -r "${fixture_profile_root}/${fixture_file}" ]] || \
    preflight_fail "missing frozen fixture input ${fixture_file}"
done
fixture_hashes="${RUN_ROOT}/fixture_input_sha256.txt"
(
  for fixture_file in "${fixture_files[@]}"; do
    sha256sum "${fixture_profile_root}/${fixture_file}"
  done
) >"${fixture_hashes}"
fixture_input_set_sha256="$(sha256sum "${fixture_hashes}" | awk '{print $1}')"
[[ "${fixture_input_set_sha256}" == "${EXPECTED_FIXTURE_INPUT_SET_SHA256}" ]] || \
  preflight_fail "fixture input set hash differs from frozen compatibility input set"

export PYTHONPATH="${SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-UNSCHEDULED}"
  printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION:-UNSPECIFIED}"
  printf 'slurm_qos=%s\n' "${SLURM_JOB_QOS:-UNSPECIFIED}"
  printf 'source_root=%s\n' "${SOURCE_ROOT}"
  printf 'source_commit=%s\n' "${source_commit}"
  printf 'expected_source_commit=%s\n' "${EXPECTED_SOURCE_COMMIT}"
  printf 'source_branch=%s\n' "${source_branch}"
  printf 'source_status=%s\n' "${source_status}"
  printf 'continuation_start_commit=%s\n' "${CONTINUATION_START_COMMIT}"
  printf 'step249_diagnosis_commit=%s\n' "${STEP249_DIAGNOSIS_COMMIT}"
  printf 'v1_closure_commit=%s\n' "${V1_CLOSURE_COMMIT}"
  printf 'frozen_start_commit=%s\n' "${FROZEN_START_COMMIT}"
  printf 'python_bin=%s\n' "${PYTHON_BIN}"
  printf 'python_version=%s\n' "${python_version}"
  printf 'numpy_version=%s\n' "${numpy_version}"
  printf 'scipy_version=%s\n' "${scipy_version}"
  printf 'formal_trace_csv=%s\n' "${FORMAL_TRACE_CSV}"
  printf 'formal_trace_sha256=%s\n' "${formal_trace_sha256}"
  printf 'fixture_root=%s\n' "${fixture_root_realpath}"
  printf 'fixture_input_sha256_file=%s\n' "${fixture_hashes}"
  printf 'fixture_input_set_sha256=%s\n' "${fixture_input_set_sha256}"
  printf 'runtime_fixture_link=%s\n' "${runtime_fixture_link}"
  printf 'runtime_fixture_link_created=%s\n' "${runtime_fixture_link_created}"
  printf 'phase=%s\n' "${PHASE_A_COMMAND}"
  printf 'scope=CHARACTERISTIC_DT_CONTINUATION_V1_PHASE_A_LOCAL_WINDOW_CPU_ONLY_NO_PF_CUDA_NO_CR1_NO_COHORT\n'
} >"${RUN_ROOT}/launch_provenance.txt"

set +e
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/run_kwn_characteristic_dt_continuation_v1.py" "${PHASE_A_COMMAND}" \
  --output-root "${RUN_ROOT}/outputs/${TASK_NAME}" \
  --report-root "${RUN_ROOT}/reports/${TASK_NAME}" \
  --formal-trace-csv "${FORMAL_TRACE_CSV}"
runner_rc=$?
set -e

printf '%s\n' "${runner_rc}" >"${RUN_ROOT}/runner_exit_code.txt"
if [[ "${runner_rc}" -eq 0 ]]; then
  printf 'PASS_KWN_CHARACTERISTIC_DT_CONTINUATION_V1_PHASE_A_LOCAL_WINDOW\n' >"${RUN_ROOT}/status.txt"
else
  printf 'NONPASS_KWN_CHARACTERISTIC_DT_CONTINUATION_V1_PHASE_A_LOCAL_WINDOW\n' >"${RUN_ROOT}/status.txt"
fi
exit "${runner_rc}"
