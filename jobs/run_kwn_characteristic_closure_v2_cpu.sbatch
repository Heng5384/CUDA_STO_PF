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

# Deterministic, CPU-only step-245 CR1 closure diagnosis.  This is not a
# PF/CUDA run and it never accepts a candidate macro step.

set -euo pipefail

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${FIXTURE_COMPAT_ROOT:?FIXTURE_COMPAT_ROOT is required}"
: "${PYTHON_BIN:=/data/home/luozhiheng/tmp/kwn_characteristic_reference_runtime_py311_np202_sp113_v1/bin/python}"
: "${FORMAL_TRACE_CSV:=/data/home/luozhiheng/tmp/kwn_characteristic_reference_v1_2ee679437421_20260902T202810Z/formal_all_14d_v4/outputs/kwn_characteristic_reference_v1/characteristic_fixed_point_trace.csv}"

readonly BASE_COMMIT="2ee679437421c750ccf2f2173d208df73bf28475"
readonly REQUIRED_BRANCH="codex/kwn-characteristic-closure-v2"
readonly TASK_NAME="kwn_characteristic_closure_v2"

[[ -d "${SOURCE_ROOT}/.git" ]] || { echo "[fatal] SOURCE_ROOT must be a Git worktree" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "[fatal] refusing to overwrite RUN_ROOT=${RUN_ROOT}" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "[fatal] PYTHON_BIN must be executable" >&2; exit 2; }
[[ -r "${FORMAL_TRACE_CSV}" ]] || { echo "[fatal] formal step-244 trace is unavailable" >&2; exit 2; }

source_commit="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
source_branch="$(git -C "${SOURCE_ROOT}" branch --show-current)"
source_status="$(git -C "${SOURCE_ROOT}" status --short)"
git -C "${SOURCE_ROOT}" merge-base --is-ancestor "${BASE_COMMIT}" HEAD || {
  echo "[fatal] source does not descend from frozen v1 closure commit" >&2
  exit 2
}
[[ "${source_branch}" == "${REQUIRED_BRANCH}" ]] || {
  echo "[fatal] source branch differs from ${REQUIRED_BRANCH}: ${source_branch}" >&2
  exit 2
}
[[ -z "${source_status}" ]] || { echo "[fatal] source tree is dirty" >&2; exit 2; }

fixture_profile_root="${FIXTURE_COMPAT_ROOT}/data/qualification/pf_elastic_target_profile_quarter_nm_v2/profiles"
fixture_files=(
  R10p5/profile_manifest.json R10p5/phi.raw.f64 R10p5/delta_C_relaxation.raw.f64 R10p5/h_phi.raw.f64
  R9p5/profile_manifest.json R9p5/phi.raw.f64 R9p5/delta_C_relaxation.raw.f64 R9p5/h_phi.raw.f64
  R8p0/profile_manifest.json R8p0/phi.raw.f64 R8p0/delta_C_relaxation.raw.f64 R8p0/h_phi.raw.f64
)
for fixture_file in "${fixture_files[@]}"; do
  [[ -r "${fixture_profile_root}/${fixture_file}" ]] || {
    echo "[fatal] missing frozen fixture input: ${fixture_profile_root}/${fixture_file}" >&2
    exit 2
  }
done

mkdir -p "${RUN_ROOT}/logs"
exec >"${RUN_ROOT}/logs/driver.stdout.log" 2>"${RUN_ROOT}/logs/driver.stderr.log"

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

{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-UNSCHEDULED}"
  printf 'source_root=%s\n' "${SOURCE_ROOT}"
  printf 'source_commit=%s\n' "${source_commit}"
  printf 'source_branch=%s\n' "${source_branch}"
  printf 'source_status=%s\n' "${source_status}"
  printf 'base_commit=%s\n' "${BASE_COMMIT}"
  printf 'python_bin=%s\n' "${PYTHON_BIN}"
  printf 'python_version=%s\n' "$("${PYTHON_BIN}" -c 'import sys; print(sys.version.replace("\n", " "))')"
  printf 'numpy_version=%s\n' "$("${PYTHON_BIN}" -c 'import numpy; print(numpy.__version__)')"
  printf 'fixture_profile_root=%s\n' "${fixture_profile_root}"
  printf 'fixture_input_sha256_file=%s\n' "${fixture_input_hashes}"
  printf 'formal_trace_csv=%s\n' "${FORMAL_TRACE_CSV}"
  printf 'scope=KWN_STEP245_DIAGNOSTIC_ONLY_NO_PF_CUDA\n'
} >"${RUN_ROOT}/launch_provenance.txt"

set +e
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/diagnose_kwn_characteristic_closure_v2.py" diagnose \
  --output-root "${RUN_ROOT}/outputs/${TASK_NAME}" \
  --report-root "${RUN_ROOT}/reports/${TASK_NAME}" \
  --formal-trace-csv "${FORMAL_TRACE_CSV}"
runner_rc=$?
set -e

printf '%s\n' "${runner_rc}" >"${RUN_ROOT}/runner_exit_code.txt"
if [[ "${runner_rc}" -eq 0 ]]; then
  printf 'PASS_KWN_CHARACTERISTIC_CLOSURE_DIAGNOSIS_V2\n' >"${RUN_ROOT}/status.txt"
else
  printf 'NONPASS_KWN_CHARACTERISTIC_CLOSURE_DIAGNOSIS_V2\n' >"${RUN_ROOT}/status.txt"
fi
exit "${runner_rc}"
