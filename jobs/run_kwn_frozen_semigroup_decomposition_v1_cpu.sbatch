#!/usr/bin/env bash
#SBATCH --job-name=kwn_fsemi_dec_v1
#SBATCH --partition=uvip_serial
#SBATCH --qos=uvip_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8G
#SBATCH --time=08:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# CPU-only, frozen-matrix diagnostic.  This wrapper reaches neither PF/CUDA
# nor CR2, a time-reference workflow, an implicit ladder, or any GP path.
set -euo pipefail

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${FIXTURE_COMPAT_ROOT:?FIXTURE_COMPAT_ROOT is required}"
: "${EXPECTED_SOURCE_COMMIT:?EXPECTED_SOURCE_COMMIT is required}"
: "${CONTINUATION_RESTART_NPZ:?CONTINUATION_RESTART_NPZ is required}"
: "${PYTHON_BIN:=/data/home/luozhiheng/tmp/kwn_characteristic_reference_runtime_py311_np202_sp113_v1/bin/python}"
: "${FORMAL_TRACE_CSV:=/data/home/luozhiheng/tmp/kwn_characteristic_reference_v1_2ee679437421_20260902T202810Z/formal_all_14d_v4/outputs/kwn_characteristic_reference_v1/characteristic_fixed_point_trace.csv}"
: "${FORMAL_PHASE_A_CSV:=/data/home/luozhiheng/tmp/kwn_characteristic_dt_continuation_v1_1ab2c61_20260904T045458Z/local_window/outputs/kwn_characteristic_dt_continuation_v1/step245_substep_ladder.csv}"
: "${FROZEN_SEMIGROUP_CSV:=/data/home/luozhiheng/tmp/kwn_cr1_semigroup_audit_v1_52055be_20260904T123324Z/formal/outputs/kwn_cr1_semigroup_audit_v1/frozen_matrix_semigroup.csv}"

readonly REQUIRED_BRANCH="codex/kwn-frozen-semigroup-decomposition-v1"
readonly FROZEN_PARENT_COMMIT="52055befdb5093c348617defeaf03b149cba3c18"
readonly TASK_NAME="kwn_frozen_semigroup_decomposition_v1"
readonly EXPECTED_FORMAL_TRACE_SHA256="e63e546ccd31d93d105b2bf40e3b5abb0fee9e9015f06eace45610a5654dad65"
readonly EXPECTED_FORMAL_PHASE_A_SHA256="a91d1940fe471b7f2ebfde5402e177b06fa7b55056bcacdce93519c668755703"
readonly EXPECTED_FROZEN_SEMIGROUP_SHA256="2eed7069411a21b5d23de6d7749f88af3c93729790ebd884e8df956c2393dc1c"
readonly EXPECTED_RESTART_STATE_HASH="51c243b61f4278fdf8f5bae9afe50672f79a6e0be57260b9c7cc0fdb46dca4c2"
readonly EXPECTED_RESTART_CHECKPOINT_SHA256="c0bc8dd946769550d780763d5a73446b929bac15c5bf7a04895a6288a2314770"
readonly EXPECTED_FIXTURE_INPUT_SET_SHA256="aaeee1d4c4795f23c4711b35a22d3d700d42f8da08e278348e004724a1722835"

[[ "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ ! -e "${RUN_ROOT}" ]] || exit 2
mkdir -- "${RUN_ROOT}" "${RUN_ROOT}/logs"
exec >"${RUN_ROOT}/logs/driver.stdout.log" 2>"${RUN_ROOT}/logs/driver.stderr.log"

preflight_fail() {
  printf '[fatal] %s\n' "$*" >&2
  printf 'NONPASS_KWN_FROZEN_SEMIGROUP_DECOMPOSITION_V1_PREFLIGHT\n' >"${RUN_ROOT}/status.txt"
  exit 2
}

[[ -e "${SOURCE_ROOT}/.git" ]] || preflight_fail "SOURCE_ROOT must be a Git worktree"
[[ -x "${PYTHON_BIN}" ]] || preflight_fail "PYTHON_BIN must be executable"
[[ -r "${FORMAL_TRACE_CSV}" && -r "${FORMAL_PHASE_A_CSV}" && -r "${FROZEN_SEMIGROUP_CSV}" ]] || preflight_fail "frozen CSV input unavailable"
[[ -r "${CONTINUATION_RESTART_NPZ}" ]] || preflight_fail "frozen restart unavailable"
[[ -d "${FIXTURE_COMPAT_ROOT}" ]] || preflight_fail "fixture root unavailable"
[[ -r "${SOURCE_ROOT}/scripts/run_kwn_frozen_semigroup_decomposition_v1.py" ]] || preflight_fail "runner unavailable"

source_commit="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" || preflight_fail "could not resolve source commit"
source_branch="$(git -C "${SOURCE_ROOT}" branch --show-current)" || preflight_fail "could not resolve source branch"
source_status="$(git -C "${SOURCE_ROOT}" status --short)" || preflight_fail "could not resolve source status"
[[ "${source_branch}" == "${REQUIRED_BRANCH}" ]] || preflight_fail "wrong source branch"
[[ "${source_commit}" == "${EXPECTED_SOURCE_COMMIT}" ]] || preflight_fail "wrong source commit"
[[ -z "${source_status}" ]] || preflight_fail "dirty source tree"
git -C "${SOURCE_ROOT}" merge-base --is-ancestor "${FROZEN_PARENT_COMMIT}" HEAD || preflight_fail "source does not descend from frozen parent"

python_version="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')" || preflight_fail "could not query Python"
numpy_version="$("${PYTHON_BIN}" -c 'import numpy; print(numpy.__version__)')" || preflight_fail "could not query NumPy"
scipy_version="$("${PYTHON_BIN}" -c 'import scipy; print(scipy.__version__)')" || preflight_fail "could not query SciPy"
[[ "${python_version}" == "3.11.13" && "${numpy_version}" == "2.0.2" && "${scipy_version}" == "1.13.1" ]] || preflight_fail "runtime differs from frozen environment"

formal_trace_sha256="$(sha256sum "${FORMAL_TRACE_CSV}" | awk '{print $1}')"
formal_phase_a_sha256="$(sha256sum "${FORMAL_PHASE_A_CSV}" | awk '{print $1}')"
frozen_semigroup_sha256="$(sha256sum "${FROZEN_SEMIGROUP_CSV}" | awk '{print $1}')"
restart_sha256="$(sha256sum "${CONTINUATION_RESTART_NPZ}" | awk '{print $1}')"
[[ "${formal_trace_sha256}" == "${EXPECTED_FORMAL_TRACE_SHA256}" ]] || preflight_fail "formal trace hash differs"
[[ "${formal_phase_a_sha256}" == "${EXPECTED_FORMAL_PHASE_A_SHA256}" ]] || preflight_fail "formal Phase-A hash differs"
[[ "${frozen_semigroup_sha256}" == "${EXPECTED_FROZEN_SEMIGROUP_SHA256}" ]] || preflight_fail "frozen semigroup hash differs"
[[ "${restart_sha256}" == "${EXPECTED_RESTART_CHECKPOINT_SHA256}" ]] || preflight_fail "restart hash differs"

fixture_root_realpath="$(realpath "${FIXTURE_COMPAT_ROOT}")" || preflight_fail "could not resolve fixture root"
runtime_fixture_link="${SOURCE_ROOT}/../CUDA_STO_PF"
if [[ -e "${runtime_fixture_link}" || -L "${runtime_fixture_link}" ]]; then
  [[ -L "${runtime_fixture_link}" ]] || preflight_fail "fixture compatibility path is not a symlink"
  [[ "$(realpath "${runtime_fixture_link}")" == "${fixture_root_realpath}" ]] || preflight_fail "fixture link differs"
else
  ln -s "${FIXTURE_COMPAT_ROOT}" "${runtime_fixture_link}" || preflight_fail "could not create fixture link"
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
    fixture_content_sha256="$(sha256sum "${fixture_profile_root}/${fixture_file}" | awk '{print $1}')"
    printf '%s  %s\n' "${fixture_content_sha256}" "${fixture_file}"
  done
) >"${fixture_hashes}"
fixture_input_set_sha256="$(sha256sum "${fixture_hashes}" | awk '{print $1}')"
[[ "${fixture_input_set_sha256}" == "${EXPECTED_FIXTURE_INPUT_SET_SHA256}" ]] || preflight_fail "fixture set hash differs"

export PYTHONPATH="${SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""

restart_identity="$("${PYTHON_BIN}" - "${SOURCE_ROOT}" "${CONTINUATION_RESTART_NPZ}" <<'PY'
import sys
from pathlib import Path

sys.path.insert(0, sys.argv[1])
sys.path.insert(0, str(Path(sys.argv[1]) / "src"))

from kwn_mvp.characteristic_dt_continuation import accepted_state_hash
from kwn_mvp.characteristic_reference import CharacteristicReferenceSolver
from scripts.frozen_canonical_smooth_population_v1 import build_frozen_canonical_context
from scripts.run_kwn_cr1_fine_substep_pathology_v1 import checkpoint_bound_config

context = build_frozen_canonical_context()
config, _ = checkpoint_bound_config(context=context, restart_checkpoint=Path(sys.argv[2]))
solver = CharacteristicReferenceSolver.load_checkpoint(config=config, path=sys.argv[2])
print(f"{solver.step} {accepted_state_hash(solver)}")
PY
)" || preflight_fail "could not load frozen checkpoint"
[[ "${restart_identity}" == "244 ${EXPECTED_RESTART_STATE_HASH}" ]] || preflight_fail "restart state identity differs"

test_log="${RUN_ROOT}/logs/full_unittest.log"
set +e
(cd "${SOURCE_ROOT}" && "${PYTHON_BIN}" -m unittest discover -s tests -v) >"${test_log}" 2>&1
test_rc=$?
set -e
[[ "${test_rc}" -eq 0 ]] || preflight_fail "full unit suite failed; see ${test_log}"
test_summary="$(grep -E '^Ran [0-9]+ tests' "${test_log}" | tail -n 1 || true)"
[[ -n "${test_summary}" ]] || preflight_fail "full unit suite emitted no test count"
test_count="$(printf '%s\n' "${test_summary}" | sed -E 's/^Ran ([0-9]+) tests.*/\1/')"
[[ "${test_count}" =~ ^[1-9][0-9]*$ ]] || preflight_fail "full unit suite ran zero or an invalid number of tests"
test_status="PASS_REMOTE_FULL_UNITTEST_DISCOVER:${test_summary}"

{
  printf 'source_root=%s\nsource_commit=%s\nsource_branch=%s\nsource_status=%s\n' "${SOURCE_ROOT}" "${source_commit}" "${source_branch}" "${source_status}"
  printf 'frozen_parent_commit=%s\n' "${FROZEN_PARENT_COMMIT}"
  printf 'formal_trace_sha256=%s\nformal_phase_a_sha256=%s\nfrozen_semigroup_sha256=%s\nrestart_sha256=%s\n' "${formal_trace_sha256}" "${formal_phase_a_sha256}" "${frozen_semigroup_sha256}" "${restart_sha256}"
  printf 'fixture_input_set_sha256=%s\nfull_unit_test_status=%s\n' "${fixture_input_set_sha256}" "${test_status}"
  printf 'scope=FROZEN_MATRIX_CR1_SEMIGROUP_DECOMPOSITION_ONLY_NO_PF_CUDA_NO_CR2_NO_TIME_REFERENCE\n'
} >"${RUN_ROOT}/launch_provenance.txt"

set +e
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/run_kwn_frozen_semigroup_decomposition_v1.py" audit \
  --restart-checkpoint "${CONTINUATION_RESTART_NPZ}" \
  --formal-trace-csv "${FORMAL_TRACE_CSV}" \
  --formal-phase-a-csv "${FORMAL_PHASE_A_CSV}" \
  --frozen-semigroup-csv "${FROZEN_SEMIGROUP_CSV}" \
  --output-root "${RUN_ROOT}/outputs/${TASK_NAME}" \
  --report-root "${RUN_ROOT}/reports/${TASK_NAME}" \
  --test-status "${test_status}"
runner_rc=$?
set -e
printf '%s\n' "${runner_rc}" >"${RUN_ROOT}/runner_exit_code.txt"
if [[ "${runner_rc}" -eq 0 ]]; then
  printf 'COMPLETED_KWN_FROZEN_SEMIGROUP_DECOMPOSITION_V1\n' >"${RUN_ROOT}/status.txt"
else
  printf 'NONPASS_KWN_FROZEN_SEMIGROUP_DECOMPOSITION_V1\n' >"${RUN_ROOT}/status.txt"
fi
exit "${runner_rc}"
