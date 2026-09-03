#!/usr/bin/env bash
#SBATCH --job-name=kwn_closure_v2
#SBATCH --partition=uvip_serial
#SBATCH --qos=uvip_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=04:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Deterministic, CPU-only step-245 CR1 closure work.  Select only one narrow
# mode with CLOSURE_V2_MODE: ``diagnose`` (default) or ``closure-regression``.
# Neither mode launches PF/CUDA or changes physics/registered timesteps.

set -euo pipefail

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${FIXTURE_COMPAT_ROOT:?FIXTURE_COMPAT_ROOT is required}"
: "${PYTHON_BIN:=/data/home/luozhiheng/tmp/kwn_characteristic_reference_runtime_py311_np202_sp113_v1/bin/python}"
: "${FORMAL_TRACE_CSV:=/data/home/luozhiheng/tmp/kwn_characteristic_reference_v1_2ee679437421_20260902T202810Z/formal_all_14d_v4/outputs/kwn_characteristic_reference_v1/characteristic_fixed_point_trace.csv}"

readonly BASE_COMMIT="2ee679437421c750ccf2f2173d208df73bf28475"
readonly REQUIRED_BRANCH="codex/kwn-characteristic-closure-v2"
readonly TASK_NAME="kwn_characteristic_closure_v2"
: "${CLOSURE_V2_MODE:=diagnose}"

case "${CLOSURE_V2_MODE}" in
  diagnose)
    runner_script="${SOURCE_ROOT}/scripts/diagnose_kwn_characteristic_closure_v2.py"
    runner_command="diagnose"
    status_label="KWN_CHARACTERISTIC_CLOSURE_DIAGNOSIS_V2"
    scope_label="KWN_STEP245_DIAGNOSTIC_ONLY_NO_PF_CUDA"
    ;;
  closure-regression)
    runner_script="${SOURCE_ROOT}/scripts/run_kwn_characteristic_closure_v2.py"
    runner_command="closure-regression"
    status_label="KWN_CHARACTERISTIC_CLOSURE_LOCAL_REGRESSION_V2"
    scope_label="KWN_STEP245_SAFEGUARDED_CLOSURE_LOCAL_REGRESSION_NO_PF_CUDA"
    ;;
  *)
    printf '[fatal] unsupported CLOSURE_V2_MODE=%s\n' "${CLOSURE_V2_MODE}" >&2
    exit 2
    ;;
esac

[[ ! -e "${RUN_ROOT}" ]] || { echo "[fatal] refusing to overwrite RUN_ROOT=${RUN_ROOT}" >&2; exit 2; }
mkdir -- "${RUN_ROOT}"
mkdir -- "${RUN_ROOT}/logs"
exec >"${RUN_ROOT}/logs/driver.stdout.log" 2>"${RUN_ROOT}/logs/driver.stderr.log"

preflight_fail() {
  printf '[fatal] %s\n' "$*" >&2
  printf 'NONPASS_%s_PREFLIGHT\n' "${status_label}" >"${RUN_ROOT}/status.txt"
  exit 2
}

[[ -e "${SOURCE_ROOT}/.git" ]] || preflight_fail "SOURCE_ROOT must be a Git worktree"
[[ -x "${PYTHON_BIN}" ]] || preflight_fail "PYTHON_BIN must be executable"
[[ -r "${FORMAL_TRACE_CSV}" ]] || preflight_fail "formal step-244 trace is unavailable"
runtime_python_version="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')" || \
  preflight_fail "could not query frozen Python version"
runtime_numpy_version="$("${PYTHON_BIN}" -c 'import numpy; print(numpy.__version__)')" || \
  preflight_fail "could not import NumPy in frozen runtime"
runtime_scipy_version="$("${PYTHON_BIN}" -c 'import scipy; print(scipy.__version__)')" || \
  preflight_fail "could not import SciPy in frozen runtime"
[[ "${runtime_python_version}" == "3.11.13" ]] || \
  preflight_fail "frozen Python version differs: ${runtime_python_version}"
[[ "${runtime_numpy_version}" == "2.0.2" ]] || \
  preflight_fail "frozen NumPy version differs: ${runtime_numpy_version}"
[[ "${runtime_scipy_version}" == "1.13.1" ]] || \
  preflight_fail "frozen SciPy version differs: ${runtime_scipy_version}"

source_commit="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" || preflight_fail "could not resolve source commit"
source_branch="$(git -C "${SOURCE_ROOT}" branch --show-current)" || preflight_fail "could not resolve source branch"
source_status="$(git -C "${SOURCE_ROOT}" status --short)" || preflight_fail "could not resolve source status"
git -C "${SOURCE_ROOT}" merge-base --is-ancestor "${BASE_COMMIT}" HEAD || {
  preflight_fail "source does not descend from frozen v1 closure commit"
}
[[ "${source_branch}" == "${REQUIRED_BRANCH}" ]] || {
  preflight_fail "source branch differs from ${REQUIRED_BRANCH}: ${source_branch}"
}
[[ -z "${source_status}" ]] || preflight_fail "source tree is dirty"

fixture_profile_root="${FIXTURE_COMPAT_ROOT}/data/qualification/pf_elastic_target_profile_quarter_nm_v2/profiles"
[[ -d "${FIXTURE_COMPAT_ROOT}" ]] || preflight_fail "declared frozen fixture root is unavailable"
fixture_compat_realpath="$(realpath "${FIXTURE_COMPAT_ROOT}")" || preflight_fail "could not resolve frozen fixture root"
runtime_fixture_link="${SOURCE_ROOT}/../CUDA_STO_PF"
if [[ -e "${runtime_fixture_link}" || -L "${runtime_fixture_link}" ]]; then
  runtime_fixture_link_created=false
  [[ -L "${runtime_fixture_link}" ]] || preflight_fail "existing runtime fixture path must be a symbolic link"
  runtime_fixture_link_realpath="$(realpath "${runtime_fixture_link}")" || \
    preflight_fail "could not resolve existing runtime fixture link"
  [[ "${runtime_fixture_link_realpath}" == "${fixture_compat_realpath}" ]] || {
    preflight_fail "existing runtime fixture link differs from declared frozen fixture root"
  }
else
  ln -s "${FIXTURE_COMPAT_ROOT}" "${runtime_fixture_link}" || \
    preflight_fail "could not create runtime frozen-fixture compatibility link"
  runtime_fixture_link_created=true
  runtime_fixture_link_realpath="${fixture_compat_realpath}"
fi
runtime_profile_root="${runtime_fixture_link}/data/qualification/pf_elastic_target_profile_quarter_nm_v2/profiles"
[[ -d "${runtime_profile_root}" ]] || {
  preflight_fail "runtime profile path is unavailable: ${runtime_profile_root}"
}
[[ -d "${fixture_profile_root}" ]] || preflight_fail "declared frozen fixture profile path is unavailable"
fixture_profile_realpath="$(realpath "${fixture_profile_root}")" || \
  preflight_fail "could not resolve declared frozen fixture profile path"
runtime_profile_realpath="$(realpath "${runtime_profile_root}")" || \
  preflight_fail "could not resolve runtime fixture profile path"
[[ "${runtime_profile_realpath}" == "${fixture_profile_realpath}" ]] || {
  preflight_fail "runtime profile path differs from declared frozen fixture root"
}
fixture_files=(
  R10p5/profile_manifest.json R10p5/phi.raw.f64 R10p5/delta_C_relaxation.raw.f64 R10p5/h_phi.raw.f64
  R9p5/profile_manifest.json R9p5/phi.raw.f64 R9p5/delta_C_relaxation.raw.f64 R9p5/h_phi.raw.f64
  R8p0/profile_manifest.json R8p0/phi.raw.f64 R8p0/delta_C_relaxation.raw.f64 R8p0/h_phi.raw.f64
)
for fixture_file in "${fixture_files[@]}"; do
  [[ -r "${fixture_profile_root}/${fixture_file}" ]] || {
    preflight_fail "missing frozen fixture input: ${fixture_profile_root}/${fixture_file}"
  }
done

export PYTHONPATH="${SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export MKL_NUM_THREADS=1

fixture_input_hashes="${RUN_ROOT}/fixture_input_sha256.txt"
(
  for fixture_file in "${fixture_files[@]}"; do
    sha256sum "${fixture_profile_root}/${fixture_file}"
  done
) >"${fixture_input_hashes}"
fixture_input_set_sha256="$(sha256sum "${fixture_input_hashes}" | awk '{print $1}')"
formal_trace_sha256="$(sha256sum "${FORMAL_TRACE_CSV}" | awk '{print $1}')"

{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-UNSCHEDULED}"
  printf 'source_root=%s\n' "${SOURCE_ROOT}"
  printf 'source_commit=%s\n' "${source_commit}"
  printf 'source_branch=%s\n' "${source_branch}"
  printf 'source_status=%s\n' "${source_status}"
  printf 'base_commit=%s\n' "${BASE_COMMIT}"
  printf 'python_bin=%s\n' "${PYTHON_BIN}"
  printf 'python_version=%s\n' "${runtime_python_version}"
  printf 'numpy_version=%s\n' "${runtime_numpy_version}"
  printf 'scipy_version=%s\n' "${runtime_scipy_version}"
  printf 'fixture_profile_root=%s\n' "${fixture_profile_root}"
  printf 'fixture_profile_realpath=%s\n' "${fixture_profile_realpath}"
  printf 'runtime_fixture_link=%s\n' "${runtime_fixture_link}"
  printf 'runtime_fixture_link_realpath=%s\n' "${runtime_fixture_link_realpath}"
  printf 'runtime_fixture_link_created=%s\n' "${runtime_fixture_link_created}"
  printf 'runtime_profile_root=%s\n' "${runtime_profile_root}"
  printf 'runtime_profile_realpath=%s\n' "${runtime_profile_realpath}"
  printf 'fixture_input_sha256_file=%s\n' "${fixture_input_hashes}"
  printf 'fixture_input_set_sha256=%s\n' "${fixture_input_set_sha256}"
  printf 'formal_trace_csv=%s\n' "${FORMAL_TRACE_CSV}"
  printf 'formal_trace_sha256=%s\n' "${formal_trace_sha256}"
  printf 'closure_v2_mode=%s\n' "${CLOSURE_V2_MODE}"
  printf 'scope=%s\n' "${scope_label}"
} >"${RUN_ROOT}/launch_provenance.txt"

set +e
"${PYTHON_BIN}" "${runner_script}" "${runner_command}" \
  --output-root "${RUN_ROOT}/outputs/${TASK_NAME}" \
  --report-root "${RUN_ROOT}/reports/${TASK_NAME}" \
  --formal-trace-csv "${FORMAL_TRACE_CSV}"
runner_rc=$?
set -e

printf '%s\n' "${runner_rc}" >"${RUN_ROOT}/runner_exit_code.txt"
if [[ "${runner_rc}" -eq 0 ]]; then
  printf 'PASS_%s\n' "${status_label}" >"${RUN_ROOT}/status.txt"
else
  printf 'NONPASS_%s\n' "${status_label}" >"${RUN_ROOT}/status.txt"
fi
exit "${runner_rc}"
