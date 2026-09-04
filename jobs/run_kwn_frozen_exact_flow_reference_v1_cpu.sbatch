#!/usr/bin/env bash
#SBATCH --job-name=kwn_fexact_v1
#SBATCH --partition=uvip_serial
#SBATCH --qos=uvip_serial
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=12G
#SBATCH --time=10:00:00
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# CPU-only frozen-autonomous diagnostic.  This wrapper reaches no PF/CUDA,
# CR2, production solver source modification, dynamic-matrix time reference,
# implicit ladder, cohort, GP, or physical-retuning path.
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
: "${FORMAL_FROZEN_U0:=/data/home/luozhiheng/tmp/kwn_frozen_semigroup_decomposition_v1_ad7f9b66a2e_20260904T153753Z/formal/outputs/kwn_frozen_semigroup_decomposition_v1/frozen_u0.npz}"
: "${FORMAL_THREE_PATH_STATES:=/data/home/luozhiheng/tmp/kwn_frozen_semigroup_decomposition_v1_ad7f9b66a2e_20260904T153753Z/formal/outputs/kwn_frozen_semigroup_decomposition_v1/three_path_states.npz}"
: "${FORMAL_H_REFINEMENT_CSV:=/data/home/luozhiheng/tmp/kwn_frozen_semigroup_decomposition_v1_ad7f9b66a2e_20260904T153753Z/formal/outputs/kwn_frozen_semigroup_decomposition_v1/h_refinement_decomposition.csv}"
: "${MPMATH_SOURCE_DIR:=/data/home/luozhiheng/miniconda3/envs/grace/lib/python3.11/site-packages}"

readonly REQUIRED_BRANCH="codex/kwn-frozen-exact-flow-reference-v1"
readonly FROZEN_PARENT_COMMIT="ad7f9b66a2e750bdf80a0c101d7ad7c167297c5d"
readonly TASK_NAME="kwn_frozen_exact_flow_reference_v1"
readonly EXPECTED_RUNTIME_SHA256="f67109f65ae2d56a54065316d5475b0de009040c856cde3377c1d07a6b1ace0d"
readonly EXPECTED_RESTART_SHA256="c0bc8dd946769550d780763d5a73446b929bac15c5bf7a04895a6288a2314770"
readonly EXPECTED_FORMAL_TRACE_SHA256="e63e546ccd31d93d105b2bf40e3b5abb0fee9e9015f06eace45610a5654dad65"
readonly EXPECTED_FORMAL_PHASE_A_SHA256="a91d1940fe471b7f2ebfde5402e177b06fa7b55056bcacdce93519c668755703"
readonly EXPECTED_FROZEN_SEMIGROUP_SHA256="2eed7069411a21b5d23de6d7749f88af3c93729790ebd884e8df956c2393dc1c"
readonly EXPECTED_FORMAL_FROZEN_U0_SHA256="fd777c36018027c4b499c2a1878f6b5b8d68d239866c98e0e1c762c9b3a6dd8"
readonly EXPECTED_THREE_PATH_STATES_SHA256="3f51d0e1b6ef9e1406a4355dce44bb16cbde40f3aaa5d8a30d8aa06fe83dd552"
readonly EXPECTED_H_REFINEMENT_SHA256="06a1b65de7e559fa44bb03211ba5f0cf27c948895220fd01da4812fb4da9eee5"
readonly EXPECTED_RESTART_STATE_HASH="51c243b61f4278fdf8f5bae9afe50672f79a6e0be57260b9c7cc0fdb46dca4c2"
readonly EXPECTED_FIXTURE_INPUT_SET_SHA256="aaeee1d4c4795f23c4711b35a22d3d700d42f8da08e278348e004724a1722835"
readonly EXPECTED_MPMATH_SOURCE_MANIFEST_SHA256="f08fb116da8ea63f88f14b873bfa0f79ae100ce631945d06f7432b60f79f4cf3"

[[ "${EXPECTED_SOURCE_COMMIT}" =~ ^[0-9a-f]{40}$ ]] || exit 2
[[ ! -e "${RUN_ROOT}" ]] || exit 2
mkdir -- "${RUN_ROOT}" "${RUN_ROOT}/logs"
exec >"${RUN_ROOT}/logs/driver.stdout.log" 2>"${RUN_ROOT}/logs/driver.stderr.log"

preflight_fail() {
  printf '[fatal] %s\n' "$*" >&2
  printf 'NONPASS_KWN_FROZEN_EXACT_FLOW_REFERENCE_V1_PREFLIGHT\n' >"${RUN_ROOT}/status.txt"
  exit 2
}

[[ -e "${SOURCE_ROOT}/.git" ]] || preflight_fail "SOURCE_ROOT must be a clean Git worktree"
[[ -x "${PYTHON_BIN}" ]] || preflight_fail "frozen PYTHON_BIN is unavailable"
[[ -r "${SOURCE_ROOT}/scripts/run_kwn_frozen_exact_flow_reference_v1.py" ]] || preflight_fail "exact-flow runner is unavailable"
[[ -r "${SOURCE_ROOT}/jobs/run_kwn_frozen_exact_flow_reference_v1_cpu.sbatch" ]] || preflight_fail "launch wrapper is unavailable"
[[ -r "${CONTINUATION_RESTART_NPZ}" && -d "${FIXTURE_COMPAT_ROOT}" ]] || preflight_fail "frozen restart or fixture root is unavailable"
[[ -d "${MPMATH_SOURCE_DIR}/mpmath" ]] || preflight_fail "pinned pure-Python mpmath source is unavailable"
for input in "${FORMAL_TRACE_CSV}" "${FORMAL_PHASE_A_CSV}" "${FROZEN_SEMIGROUP_CSV}" "${FORMAL_FROZEN_U0}" "${FORMAL_THREE_PATH_STATES}" "${FORMAL_H_REFINEMENT_CSV}"; do
  [[ -r "${input}" ]] || preflight_fail "a required inherited formal artifact is unavailable"
done

source_commit="$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" || preflight_fail "cannot resolve source commit"
source_branch="$(git -C "${SOURCE_ROOT}" branch --show-current)" || preflight_fail "cannot resolve source branch"
source_status="$(git -C "${SOURCE_ROOT}" status --short)" || preflight_fail "cannot resolve source status"
[[ "${source_branch}" == "${REQUIRED_BRANCH}" ]] || preflight_fail "wrong source branch"
[[ "${source_commit}" == "${EXPECTED_SOURCE_COMMIT}" ]] || preflight_fail "wrong source commit"
[[ -z "${source_status}" ]] || preflight_fail "source tree is dirty"
git -C "${SOURCE_ROOT}" merge-base --is-ancestor "${FROZEN_PARENT_COMMIT}" HEAD || preflight_fail "source does not descend from frozen parent"

runtime_sha256="$(sha256sum "${PYTHON_BIN}" | awk '{print $1}')"
[[ "${runtime_sha256}" == "${EXPECTED_RUNTIME_SHA256}" ]] || preflight_fail "frozen Python binary differs"
python_version="$("${PYTHON_BIN}" -c 'import sys; print(".".join(map(str, sys.version_info[:3])))')" || preflight_fail "cannot query Python"
numpy_version="$("${PYTHON_BIN}" -c 'import numpy; print(numpy.__version__)')" || preflight_fail "cannot query NumPy"
scipy_version="$("${PYTHON_BIN}" -c 'import scipy; print(scipy.__version__)')" || preflight_fail "cannot query SciPy"
[[ "${python_version}" == "3.11.13" && "${numpy_version}" == "2.0.2" && "${scipy_version}" == "1.13.1" ]] || preflight_fail "runtime differs from frozen Python/NumPy/SciPy"

restart_sha256="$(sha256sum "${CONTINUATION_RESTART_NPZ}" | awk '{print $1}')"
formal_trace_sha256="$(sha256sum "${FORMAL_TRACE_CSV}" | awk '{print $1}')"
formal_phase_a_sha256="$(sha256sum "${FORMAL_PHASE_A_CSV}" | awk '{print $1}')"
frozen_semigroup_sha256="$(sha256sum "${FROZEN_SEMIGROUP_CSV}" | awk '{print $1}')"
formal_u0_sha256="$(sha256sum "${FORMAL_FROZEN_U0}" | awk '{print $1}')"
three_path_sha256="$(sha256sum "${FORMAL_THREE_PATH_STATES}" | awk '{print $1}')"
h_refinement_sha256="$(sha256sum "${FORMAL_H_REFINEMENT_CSV}" | awk '{print $1}')"
[[ "${restart_sha256}" == "${EXPECTED_RESTART_SHA256}" ]] || preflight_fail "restart hash differs"
[[ "${formal_trace_sha256}" == "${EXPECTED_FORMAL_TRACE_SHA256}" ]] || preflight_fail "formal trace hash differs"
[[ "${formal_phase_a_sha256}" == "${EXPECTED_FORMAL_PHASE_A_SHA256}" ]] || preflight_fail "formal Phase-A hash differs"
[[ "${frozen_semigroup_sha256}" == "${EXPECTED_FROZEN_SEMIGROUP_SHA256}" ]] || preflight_fail "old frozen semigroup hash differs"
[[ "${formal_u0_sha256}" == "${EXPECTED_FORMAL_FROZEN_U0_SHA256}" ]] || preflight_fail "formal frozen U0 hash differs"
[[ "${three_path_sha256}" == "${EXPECTED_THREE_PATH_STATES_SHA256}" ]] || preflight_fail "formal A/B/C state hash differs"
[[ "${h_refinement_sha256}" == "${EXPECTED_H_REFINEMENT_SHA256}" ]] || preflight_fail "formal h-refinement hash differs"

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

# Preserve the frozen NumPy/SciPy interpreter.  Only pure-Python mpmath is
# copied into an isolated vendor root for the required high-precision shadow;
# no grace/mace site-packages or compiled extension is imported.
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
mpmath_vendor_manifest_sha256="$(sha256sum "${vendor_manifest}" | awk '{print $1}')"
[[ "${mpmath_vendor_manifest_sha256}" == "${EXPECTED_MPMATH_SOURCE_MANIFEST_SHA256}" ]] || preflight_fail "vendored mpmath manifest differs"

export PYTHONPATH="${vendor_root}:${SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
export OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 CUDA_VISIBLE_DEVICES=""
runtime_shadow="$("${PYTHON_BIN}" - "${vendor_root}" <<'PY'
import sys
from pathlib import Path
import numpy
import scipy
import mpmath
root = Path(sys.argv[1]).resolve()
path = Path(mpmath.__file__).resolve()
assert root in path.parents
assert mpmath.__version__ == "1.3.0"
assert numpy.__version__ == "2.0.2"
assert scipy.__version__ == "1.13.1"
print(f"{mpmath.__version__} {path}")
PY
)" || preflight_fail "isolated mpmath shadow did not preserve frozen runtime"

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
)" || preflight_fail "cannot load exact frozen checkpoint"
[[ "${restart_identity}" == "244 ${EXPECTED_RESTART_STATE_HASH}" ]] || preflight_fail "restart state identity differs"

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
  printf 'frozen_python_sha256=%s\npython=%s\nnumpy=%s\nscipy=%s\n' "${runtime_sha256}" "${python_version}" "${numpy_version}" "${scipy_version}"
  printf 'restart_sha256=%s\nformal_trace_sha256=%s\nformal_phase_a_sha256=%s\nfrozen_semigroup_sha256=%s\n' "${restart_sha256}" "${formal_trace_sha256}" "${formal_phase_a_sha256}" "${frozen_semigroup_sha256}"
  printf 'formal_u0_sha256=%s\nthree_path_sha256=%s\nh_refinement_sha256=%s\n' "${formal_u0_sha256}" "${three_path_sha256}" "${h_refinement_sha256}"
  printf 'fixture_input_set_sha256=%s\nmpmath_manifest_sha256=%s\nmpmath_runtime=%s\nfull_unit_test_status=%s\n' "${fixture_input_set_sha256}" "${mpmath_vendor_manifest_sha256}" "${runtime_shadow}" "${test_status}"
  printf 'scope=FROZEN_AUTONOMOUS_EXACT_FLOW_REFERENCE_ONLY_NO_PF_CUDA_NO_CR2_NO_DYNAMIC_TIME_REFERENCE\n'
} >"${RUN_ROOT}/launch_provenance.txt"

set +e
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/run_kwn_frozen_exact_flow_reference_v1.py" audit \
  --restart-checkpoint "${CONTINUATION_RESTART_NPZ}" \
  --formal-trace-csv "${FORMAL_TRACE_CSV}" \
  --formal-phase-a-csv "${FORMAL_PHASE_A_CSV}" \
  --frozen-semigroup-csv "${FROZEN_SEMIGROUP_CSV}" \
  --formal-frozen-u0 "${FORMAL_FROZEN_U0}" \
  --formal-three-path-states "${FORMAL_THREE_PATH_STATES}" \
  --formal-h-refinement-csv "${FORMAL_H_REFINEMENT_CSV}" \
  --mpmath-vendor-root "${vendor_root}" \
  --output-root "${RUN_ROOT}/outputs/${TASK_NAME}" \
  --report-root "${RUN_ROOT}/reports/${TASK_NAME}" \
  --test-status "${test_status}"
runner_rc=$?
set -e
printf '%s\n' "${runner_rc}" >"${RUN_ROOT}/runner_exit_code.txt"
if [[ "${runner_rc}" -eq 0 ]]; then
  printf 'COMPLETED_KWN_FROZEN_EXACT_FLOW_REFERENCE_V1\n' >"${RUN_ROOT}/status.txt"
else
  printf 'NONPASS_KWN_FROZEN_EXACT_FLOW_REFERENCE_V1\n' >"${RUN_ROOT}/status.txt"
fi
exit "${runner_rc}"
