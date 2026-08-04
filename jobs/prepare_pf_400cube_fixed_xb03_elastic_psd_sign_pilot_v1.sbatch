#!/usr/bin/env bash
#SBATCH --job-name=pf400_psd_prepare
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

# Build the previously unavailable 17--20 nm target-profile library, then
# materialize and statically audit the 400^3 fixture.  Only a complete static
# PASS causes this job to submit the distinct 48 h production allocation.

set -euo pipefail
: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${PREP_ROOT:?PREP_ROOT is required}"
: "${PRODUCTION_ROOT:?PRODUCTION_ROOT is required}"
SPEC_PATH="${SPEC_PATH:-${SOURCE_ROOT}/data/qualification/pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1/fixture_spec_native160_v1.json}"
PROFILE_GRID_N="${PROFILE_GRID_N:-160}"
PROFILE_MAX_ITER="${PROFILE_MAX_ITER:-60000}"
REUSE_PROFILE_LIBRARY_ROOT="${REUSE_PROFILE_LIBRARY_ROOT:-}"
[[ ! -e "${PREP_ROOT}" ]] || { echo "[fatal] refusing to overwrite PREP_ROOT: ${PREP_ROOT}" >&2; exit 2; }
[[ ! -e "${PRODUCTION_ROOT}" ]] || { echo "[fatal] refusing to overwrite PRODUCTION_ROOT: ${PRODUCTION_ROOT}" >&2; exit 2; }

mkdir -p "${PREP_ROOT}"
exec >"${PREP_ROOT}/driver.stdout.log" 2>"${PREP_ROOT}/driver.stderr.log"
trap 'rc=$?; if [[ "$rc" -ne 0 ]]; then printf "BLOCKED_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_PREPARATION_V1\\nexit_code=%s\\n" "$rc" >"${PREP_ROOT}/status.txt"; fi' EXIT
printf 'RUNNING_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_PREPARATION_V1\n' >"${PREP_ROOT}/status.txt"

cd "${SOURCE_ROOT}"
source jobs/_site_env.sh
site_prepare_cuda_env
make main_cuda CUDA_ROOT="${CUDA_ROOT}" CUDA_ARCH="${CUDA_ARCH:-sm_80}" NVCC="$(command -v nvcc)"
python3 -m py_compile scripts/build_pf_400cube_fixed_xb03_elastic_psd_sign_fixture_v1.py scripts/audit_pf_400cube_fixed_xb03_elastic_psd_sign_fixture_v1.py
git rev-parse HEAD >"${PREP_ROOT}/source_commit.txt"
sha256sum main_cuda main_cuda.cu cuda_kernels.cu cuda_kernels.h cuda_common.cu cuda_common.h pf_zero_mode_checkpoint.cpp pf_zero_mode_checkpoint.h scripts/build_pf_400cube_fixed_xb03_elastic_psd_sign_fixture_v1.py scripts/audit_pf_400cube_fixed_xb03_elastic_psd_sign_fixture_v1.py jobs/run_pf_elastic_target_profile_library_v1.sh jobs/run_pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1.sh jobs/submit_pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1.sbatch jobs/prepare_pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1.sbatch "${SPEC_PATH}" >"${PREP_ROOT}/source_and_spec.sha256"

# This is the same fully elastic target-profile minimization contract used by
# the frozen library, evaluated at new, exact registered radii.  It is not a
# stretched or interpolated copy of the 8--11.5 nm entries.
# The preserved V1 preparation stopped at 14,000 iterations with R17 still
# monotonically reducing its full-model energy.  Only the offline numerical
# iteration cap is increased for this new non-overwriting preparation root;
# the minimization dt, physics, inventory, and acceptance thresholds remain
# exactly unchanged.
if [[ -n "${REUSE_PROFILE_LIBRARY_ROOT}" ]]; then
  PROFILE_LIBRARY_ROOT="${REUSE_PROFILE_LIBRARY_ROOT}"
  [[ -d "${PROFILE_LIBRARY_ROOT}" ]] || { echo "[fatal] reused profile library is missing: ${PROFILE_LIBRARY_ROOT}" >&2; exit 2; }
  [[ -f "${PROFILE_LIBRARY_ROOT}/library_manifest.json" ]] || { echo "[fatal] reused profile library lacks manifest" >&2; exit 2; }
  python3 - "${PROFILE_LIBRARY_ROOT}" "${PREP_ROOT}/reused_profile_library_provenance.json" <<'PY'
import hashlib, json, pathlib, sys
root = pathlib.Path(sys.argv[1]).resolve()
manifest_path = root / 'library_manifest.json'
manifest = json.loads(manifest_path.read_text())
shape = tuple(int(manifest['grid'][axis]) for axis in ('Nx', 'Ny', 'Nz'))
radii = sorted(float(row['target_radius_nm']) for row in manifest['profiles'])
expected = [17.0, 17.5, 18.0, 18.5, 19.0, 19.5, 20.0]
if manifest.get('schema') != 'PF_ELASTIC_TARGET_PROFILE_LIBRARY_V1' or shape != (160, 160, 160) or radii != expected:
    raise SystemExit('[fatal] reused profile library does not match frozen 17--20 nm / 160^3 contract')
digest = hashlib.sha256(manifest_path.read_bytes()).hexdigest()
payload = {'status': 'PASS_REUSED_HASH_PINNED_PROFILE_LIBRARY_V1', 'library_root': str(root), 'library_manifest_sha256': digest, 'native_grid': list(shape), 'radii_nm': radii}
pathlib.Path(sys.argv[2]).write_text(json.dumps(payload, indent=2, sort_keys=True) + '\\n')
print(json.dumps(payload, sort_keys=True))
PY
else
  PROFILE_LIBRARY_ROOT="${PREP_ROOT}/profile_library/library"
  RUN_ROOT="${PREP_ROOT}/profile_library" RUN_LABEL="cluster_400cube_psd_sign" GRID_N="${PROFILE_GRID_N}" RADII_NM="17.0 17.5 18.0 18.5 19.0 19.5 20.0" MINIMIZE_DT=0.025 MAX_ITER="${PROFILE_MAX_ITER}" MIN_PSEUDO_TIME=300 CONVERGENCE_STEPS=20 OUT_EVERY="${PROFILE_MAX_ITER}" CSV_OUT_EVERY=10 XB_FAR=0.006219279767278563 SOURCE_COMMIT_OVERRIDE="$(<"${PREP_ROOT}/source_commit.txt")" bash jobs/run_pf_elastic_target_profile_library_v1.sh
  grep -qx 'runner_status=PASS_PF_ELASTIC_TARGET_PROFILE_LIBRARY_V1' "${PREP_ROOT}/profile_library/status.txt"
fi

python3 scripts/build_pf_400cube_fixed_xb03_elastic_psd_sign_fixture_v1.py --spec "${SPEC_PATH}" --library-root "${PROFILE_LIBRARY_ROOT}" --out "${PREP_ROOT}/fixture" | tee "${PREP_ROOT}/fixture_materialization.json"
grep -qx 'PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_FIXTURE_V1' "${PREP_ROOT}/fixture/status.txt"
python3 scripts/audit_pf_400cube_fixed_xb03_elastic_psd_sign_fixture_v1.py --spec "${SPEC_PATH}" --fixture "${PREP_ROOT}/fixture" --library-root "${PROFILE_LIBRARY_ROOT}" --out "${PREP_ROOT}/static_audit" | tee "${PREP_ROOT}/static_audit_terminal.json"
grep -qx 'PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_FIXTURE_STATIC_V1' "${PREP_ROOT}/static_audit/status.txt"

# This remains a zero-macrostep structural preflight: it confirms the exact
# production binary, parameter contract, fixture hashes, and vacant output
# root without running a one-step/restart or short physical smoke.
SOURCE_ROOT="${SOURCE_ROOT}" FIXTURE_ROOT="${PREP_ROOT}/fixture" RUN_ROOT="${PRODUCTION_ROOT}" SPEC_PATH="${SPEC_PATH}" bash jobs/run_pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1.sh --validate-only | tee "${PREP_ROOT}/production_preflight.txt"

production_job="$(sbatch --parsable --export=ALL,SOURCE_ROOT="${SOURCE_ROOT}",FIXTURE_ROOT="${PREP_ROOT}/fixture",RUN_ROOT="${PRODUCTION_ROOT}",SPEC_PATH="${SPEC_PATH}" jobs/submit_pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1.sbatch)"
tail_job="$(sbatch --parsable --dependency="afterany:${production_job}" --export=ALL,TARGET_ROOT="${PRODUCTION_ROOT}" jobs/tail_pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1.sbatch)"
library_sha="$(sha256sum "${PROFILE_LIBRARY_ROOT}/library_manifest.json" | awk '{print $1}')"
fixture_sha="$(sha256sum "${PREP_ROOT}/fixture/fixture_manifest.json" | awk '{print $1}')"
param_sha="$(awk -F= '/^parameter_sha256=/{print $2}' "${PREP_ROOT}/production_preflight.txt")"
binary_sha="$(awk -F= '/^binary_sha256=/{print $2}' "${PREP_ROOT}/production_preflight.txt")"
sbatch_sha="$(sha256sum jobs/submit_pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1.sbatch | awk '{print $1}')"
python3 - "${PREP_ROOT}/submission.json" "${production_job}" "${tail_job}" "${PRODUCTION_ROOT}" "${sbatch_sha}" "${binary_sha}" "${param_sha}" "${fixture_sha}" "${library_sha}" <<'PY'
import json, pathlib, sys
payload={'status':'SUBMITTED_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_PILOT_V1','job_id':sys.argv[2],'tail_job_id':sys.argv[3],'queue_state':'PENDING_OR_RUNNING_QUERY_SQUEUE','output_root':sys.argv[4],'sbatch_sha256':sys.argv[5],'binary_sha256':sys.argv[6],'parameter_sha256':sys.argv[7],'fixture_sha256':sys.argv[8],'profile_library_sha256':sys.argv[9],'checkpoint_schedule_sha256':None}
pathlib.Path(sys.argv[1]).write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
PY
sha256sum "${PREP_ROOT}/fixture/fixture_manifest.json" "${PROFILE_LIBRARY_ROOT}/library_manifest.json" "${PREP_ROOT}/production_preflight.txt" >"${PREP_ROOT}/submission_inputs.sha256"
printf 'SUBMITTED_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_PILOT_V1\njob_id=%s\ntail_job_id=%s\n' "${production_job}" "${tail_job}" >"${PREP_ROOT}/status.txt"
cat "${PREP_ROOT}/status.txt"
