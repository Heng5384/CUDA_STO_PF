#!/usr/bin/env bash
#SBATCH --job-name=kwn_trace_audit_v1
#SBATCH --partition=uvip_serial
#SBATCH --qos=uvip_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=16G
#SBATCH --time=10:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Frozen autonomous production-trace audit only.  It intentionally excludes
# PF/CUDA, CR2, dynamic closure/time-reference work, cohort/implicit ladders,
# GP release, and every physical retuning path.
set -euo pipefail

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${FIXTURE_COMPAT_ROOT:?FIXTURE_COMPAT_ROOT is required}"
: "${EXPECTED_SOURCE_COMMIT:?EXPECTED_SOURCE_COMMIT is required}"
: "${BASELINE_ROOT:?BASELINE_ROOT is required}"
: "${CONTINUATION_RESTART_NPZ:?CONTINUATION_RESTART_NPZ is required}"
: "${PHASE:?PHASE is required}"
: "${PYTHON_BIN:=/data/home/luozhiheng/tmp/kwn_characteristic_reference_runtime_py311_np202_sp113_v1/bin/python}"
: "${MPMATH_SOURCE_DIR:=/data/home/luozhiheng/miniconda3/envs/grace/lib/python3.11/site-packages}"

readonly REQUIRED_BRANCH="codex/kwn-production-trace-audit-v1"
readonly FROZEN_ANCESTOR="7759c83ca4da7fd930e0d67dfbd15788095702cd"
readonly TASK_NAME="kwn_production_trace_audit_v1"
readonly EXPECTED_RUNTIME_SHA256="f67109f65ae2d56a54065316d5475b0de009040c856cde3377c1d07a6b1ace0d"
readonly EXPECTED_RESTART_SHA256="c0bc8dd946769550d780763d5a73446b929bac15c5bf7a04895a6288a2314770"
readonly EXPECTED_MPMATH_SOURCE_MANIFEST_SHA256="f08fb116da8ea63f88f14b873bfa0f79ae100ce631945d06f7432b60f79f4cf3"
readonly EXPECTED_FIXTURE_INPUT_SET_SHA256="aaeee1d4c4795f23c4711b35a22d3d700d42f8da08e278348e004724a1722835"

[[ "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ "${PHASE}" == "diagnose" || "${PHASE}" == "qualify" ]] || exit 2
[[ ! -e "${RUN_ROOT}" ]] || exit 2
mkdir -- "${RUN_ROOT}" "${RUN_ROOT}/logs"
exec >"${RUN_ROOT}/logs/driver.stdout.log" 2>"${RUN_ROOT}/logs/driver.stderr.log"

preflight_fail() {
  printf '[fatal] %s\n' "$*" >&2
  printf 'NONPASS_KWN_PRODUCTION_TRACE_AUDIT_V1_PREFLIGHT\n' >"${RUN_ROOT}/status.txt"
  exit 2
}

[[ -e "${SOURCE_ROOT}/.git" ]] || preflight_fail "SOURCE_ROOT must be a Git worktree"
[[ -x "${PYTHON_BIN}" ]] || preflight_fail "frozen Python runtime is unavailable"
[[ -r "${SOURCE_ROOT}/scripts/run_kwn_production_trace_audit_v1.py" ]] || preflight_fail "trace-audit runner is unavailable"
[[ -r "${SOURCE_ROOT}/jobs/run_kwn_production_trace_audit_v1_cpu.sbatch" ]] || preflight_fail "trace-audit launcher is unavailable"
[[ -r "${CONTINUATION_RESTART_NPZ}" ]] || preflight_fail "step-244 restart is unavailable"
[[ -r "${BASELINE_ROOT}/status.txt" ]] || preflight_fail "baseline result is unavailable"
[[ -d "${MPMATH_SOURCE_DIR}/mpmath" ]] || preflight_fail "pinned mpmath source is unavailable"

source_commit="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" || preflight_fail "cannot resolve source commit"
source_branch="$(git -C "${SOURCE_ROOT}" branch --show-current)" || preflight_fail "cannot resolve source branch"
source_status="$(git -C "${SOURCE_ROOT}" status --short)" || preflight_fail "cannot resolve source status"
[[ "${source_branch}" == "${REQUIRED_BRANCH}" ]] || preflight_fail "wrong source branch"
[[ "${source_commit}" == "${EXPECTED_SOURCE_COMMIT}" ]] || preflight_fail "wrong source commit"
[[ -z "${source_status}" ]] || preflight_fail "source tree is dirty"
git -C "${SOURCE_ROOT}" merge-base --is-ancestor "${FROZEN_ANCESTOR}" HEAD || preflight_fail "source does not descend from frozen trace baseline"

runtime_sha256="$(sha256sum "${PYTHON_BIN}" | awk '{print $1}')"
[[ "${runtime_sha256}" == "${EXPECTED_RUNTIME_SHA256}" ]] || preflight_fail "frozen Python runtime differs"
python_version="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')" || preflight_fail "cannot query Python"
numpy_version="$("${PYTHON_BIN}" -c 'import numpy; print(numpy.__version__)')" || preflight_fail "cannot query NumPy"
scipy_version="$("${PYTHON_BIN}" -c 'import scipy; print(scipy.__version__)')" || preflight_fail "cannot query SciPy"
[[ "${python_version}" == "3.11.13" && "${numpy_version}" == "2.0.2" && "${scipy_version}" == "1.13.1" ]] || preflight_fail "runtime version differs"

restart_sha256="$(sha256sum "${CONTINUATION_RESTART_NPZ}" | awk '{print $1}')"
[[ "${restart_sha256}" == "${EXPECTED_RESTART_SHA256}" ]] || preflight_fail "restart hash differs"
[[ "$(cat "${BASELINE_ROOT}/status.txt")" == "COMPLETED_KWN_FROZEN_EXACT_FLOW_REFERENCE_V1" ]] || preflight_fail "baseline did not complete"

fixture_root_realpath="$(realpath "${FIXTURE_COMPAT_ROOT}")" || preflight_fail "cannot resolve fixture root"
runtime_fixture_link="${SOURCE_ROOT}/../CUDA_STO_PF"
if [[ -e "${runtime_fixture_link}" || -L "${runtime_fixture_link}" ]]; then
  [[ -L "${runtime_fixture_link}" ]] || preflight_fail "fixture compatibility path is not a symlink"
  [[ "$(realpath "${runtime_fixture_link}")" == "${fixture_root_realpath}" ]] || preflight_fail "fixture compatibility symlink differs"
else
  ln -s "${FIXTURE_COMPAT_ROOT}" "${runtime_fixture_link}" || preflight_fail "cannot create fixture compatibility symlink"
fi
fixture_profile_root="${FIXTURE_COMPAT_ROOT}/data/qualification/pf_elastic_target_profile_quarter_nm_v2/profiles"
fixture_files=(
  R10p5/profile_manifest.json R10p5/phi.raw.f64 R10p5/delta_C_relaxation.raw.f64 R10p5/h_phi.raw.f64
  R9p5/profile_manifest.json R9p5/phi.raw.f64 R9p5/delta_C_relaxation.raw.f64 R9p5/h_phi.raw.f64
  R8p0/profile_manifest.json R8p0/phi.raw.f64 R8p0/delta_C_relaxation.raw.f64 R8p0/h_phi.raw.f64
)
for fixture_file in "${fixture_files[@]}"; do
  [[ -r "${fixture_profile_root}/${fixture_file}" ]] || preflight_fail "missing fixture ${fixture_file}"
done
fixture_hashes="${RUN_ROOT}/fixture_input_sha256.txt"
(
  for fixture_file in "${fixture_files[@]}"; do
    printf '%s  %s\n' "$(sha256sum "${fixture_profile_root}/${fixture_file}" | awk '{print $1}')" "${fixture_file}"
  done
) >"${fixture_hashes}"
fixture_input_set_sha256="$(sha256sum "${fixture_hashes}" | awk '{print $1}')"
[[ "${fixture_input_set_sha256}" == "${EXPECTED_FIXTURE_INPUT_SET_SHA256}" ]] || preflight_fail "fixture input set hash differs"

vendor_root="${RUN_ROOT}/vendor"
mkdir -- "${vendor_root}"
source_manifest="${RUN_ROOT}/mpmath_source_files.sha256"
(
  cd "${MPMATH_SOURCE_DIR}"
  find mpmath -type f -name '*.py' ! -path '*/tests/*' ! -path '*/__pycache__/*' -print0 |
    LC_ALL=C sort -z | xargs -0 sha256sum
) >"${source_manifest}"
mpmath_source_manifest_sha256="$(sha256sum "${source_manifest}" | awk '{print $1}')"
[[ "${mpmath_source_manifest_sha256}" == "${EXPECTED_MPMATH_SOURCE_MANIFEST_SHA256}" ]] || preflight_fail "mpmath source manifest differs"
while IFS= read -r -d '' source_file; do
  target_file="${vendor_root}/${source_file}"
  mkdir -p "$(dirname "${target_file}")"
  cp -- "${MPMATH_SOURCE_DIR}/${source_file}" "${target_file}"
done < <(cd "${MPMATH_SOURCE_DIR}" && find mpmath -type f -name '*.py' ! -path '*/tests/*' ! -path '*/__pycache__/*' -print0)
vendor_manifest="${RUN_ROOT}/mpmath_vendor_files.sha256"
(
  cd "${vendor_root}"
  find mpmath -type f -name '*.py' ! -path '*/tests/*' ! -path '*/__pycache__/*' -print0 |
    LC_ALL=C sort -z | xargs -0 sha256sum
) >"${vendor_manifest}"
[[ "$(sha256sum "${vendor_manifest}" | awk '{print $1}')" == "${EXPECTED_MPMATH_SOURCE_MANIFEST_SHA256}" ]] || preflight_fail "vendored mpmath manifest differs"

export PYTHONPATH="${vendor_root}:${SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
"${PYTHON_BIN}" - "${vendor_root}" <<'PY' || preflight_fail "isolated mpmath shadow did not preserve frozen runtime"
import sys
from pathlib import Path
import mpmath
import numpy
import scipy
root = Path(sys.argv[1]).resolve()
assert root in Path(mpmath.__file__).resolve().parents
assert mpmath.__version__ == "1.3.0"
assert numpy.__version__ == "2.0.2"
assert scipy.__version__ == "1.13.1"
PY

test_log="${RUN_ROOT}/logs/full_unittest.log"
set +e
(cd "${SOURCE_ROOT}" && "${PYTHON_BIN}" -m unittest discover -s tests -v) >"${test_log}" 2>&1
test_rc=$?
set -e
[[ "${test_rc}" -eq 0 ]] || preflight_fail "full unit suite failed; see ${test_log}"
test_summary="$(grep -E '^Ran [0-9]+ tests' "${test_log}" | tail -n 1 || true)"
[[ -n "${test_summary}" ]] || preflight_fail "full unit suite emitted no test count"
test_status="PASS_REMOTE_FULL_UNITTEST_DISCOVER:${test_summary}"

{
  printf 'source_root=%s\nsource_commit=%s\nsource_branch=%s\nsource_status=%s\n' "${SOURCE_ROOT}" "${source_commit}" "${source_branch}" "${source_status}"
  printf 'phase=%s\nbaseline_root=%s\nrestart_sha256=%s\n' "${PHASE}" "${BASELINE_ROOT}" "${restart_sha256}"
  printf 'frozen_python_sha256=%s\npython=%s\nnumpy=%s\nscipy=%s\n' "${runtime_sha256}" "${python_version}" "${numpy_version}" "${scipy_version}"
  printf 'fixture_input_set_sha256=%s\nmpmath_manifest_sha256=%s\nfull_unit_test_status=%s\n' "${fixture_input_set_sha256}" "${mpmath_source_manifest_sha256}" "${test_status}"
  printf 'scope=FROZEN_PRODUCTION_TRACE_AUDIT_ONLY_NO_PF_CUDA_NO_CR2_NO_DYNAMIC_TIME_REFERENCE\n'
} >"${RUN_ROOT}/launch_provenance.txt"

set +e
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/run_kwn_production_trace_audit_v1.py" audit \
  --phase "${PHASE}" \
  --baseline-root "${BASELINE_ROOT}" \
  --restart-checkpoint "${CONTINUATION_RESTART_NPZ}" \
  --mpmath-vendor-root "${vendor_root}" \
  --output-root "${RUN_ROOT}/outputs/${TASK_NAME}" \
  --report-root "${RUN_ROOT}/reports/${TASK_NAME}" \
  --test-status "${test_status}"
runner_rc=$?
set -e
printf '%s\n' "${runner_rc}" >"${RUN_ROOT}/runner_exit_code.txt"
if [[ "${runner_rc}" -eq 0 ]]; then
  printf 'COMPLETED_KWN_PRODUCTION_TRACE_AUDIT_V1\n' >"${RUN_ROOT}/status.txt"
else
  printf 'NONPASS_KWN_PRODUCTION_TRACE_AUDIT_V1\n' >"${RUN_ROOT}/status.txt"
fi
exit "${runner_rc}"
