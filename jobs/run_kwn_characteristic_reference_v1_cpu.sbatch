#!/usr/bin/env bash
#SBATCH --job-name=kwn_charref_v1
#SBATCH --partition=parallel_vip_24h
#SBATCH --qos=parallel_vip_24h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
# The fixture reconstruction peaks far below 1 GiB; 16 GiB preserves a large
# margin while allowing the CPU-only numerical gate to use available nodes.
#SBATCH --mem=16G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Run the KWN-only characteristic-reference gate from an isolated, clean Git
# source tree.  This job requests no GPU and never invokes PF/CUDA.

set -euo pipefail

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
# The cluster login default is a minimal Python 3.13 environment without
# SciPy.  Pin the validated scientific interpreter; callers may override it
# with another absolute executable only when they also preserve its provenance.
: "${PYTHON_BIN:=/data/home/luozhiheng/miniconda3/envs/grace/bin/python}"

# The inherited fixture loader intentionally reads the frozen raw profiles
# through this sibling compatibility path.  The isolated payload builder must
# stage a read-only, hash-validated profile library there; source code alone
# deliberately contains only its manifests, not the raw 96^3 fields.
fixture_compat_root="$(dirname "${SOURCE_ROOT}")/CUDA_STO_PF"
fixture_profile_root="${fixture_compat_root}/data/qualification/pf_elastic_target_profile_quarter_nm_v2/profiles"
fixture_files=(
  R10p5/profile_manifest.json R10p5/phi.raw.f64 R10p5/delta_C_relaxation.raw.f64 R10p5/h_phi.raw.f64
  R9p5/profile_manifest.json R9p5/phi.raw.f64 R9p5/delta_C_relaxation.raw.f64 R9p5/h_phi.raw.f64
  R8p0/profile_manifest.json R8p0/phi.raw.f64 R8p0/delta_C_relaxation.raw.f64 R8p0/h_phi.raw.f64
)

[[ -d "${SOURCE_ROOT}/.git" ]] || { echo "[fatal] SOURCE_ROOT must be a Git worktree" >&2; exit 2; }
[[ ! -e "${RUN_ROOT}" ]] || { echo "[fatal] refusing to overwrite RUN_ROOT=${RUN_ROOT}" >&2; exit 2; }
[[ -x "${PYTHON_BIN}" ]] || { echo "[fatal] PYTHON_BIN must be an executable path: ${PYTHON_BIN}" >&2; exit 2; }

mkdir -p "${RUN_ROOT}/logs"
exec >"${RUN_ROOT}/logs/driver.stdout.log" 2>"${RUN_ROOT}/logs/driver.stderr.log"

export PYTHONPATH="${SOURCE_ROOT}/src${PYTHONPATH:+:${PYTHONPATH}}"
# Some compute-node batch shells omit /usr/bin even though the runner needs
# the system Git binary to bind source provenance before numerical work.
export PATH="/usr/bin:/bin${PATH:+:${PATH}}"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export OPENBLAS_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
export MKL_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"

if ! "${PYTHON_BIN}" -c 'import numpy, scipy'; then
  printf 'NONPASS_KWN_CHARACTERISTIC_REFERENCE_ENVIRONMENT_V1\n' >"${RUN_ROOT}/status.txt"
  exit 2
fi
for fixture_file in "${fixture_files[@]}"; do
  if [[ ! -r "${fixture_profile_root}/${fixture_file}" ]]; then
    printf 'NONPASS_KWN_CHARACTERISTIC_REFERENCE_FIXTURE_STAGING_V1\n' >"${RUN_ROOT}/status.txt"
    printf '[fatal] missing frozen fixture input: %s\n' "${fixture_profile_root}/${fixture_file}" >&2
    exit 2
  fi
done
fixture_input_hashes="${RUN_ROOT}/fixture_input_sha256.txt"
(
  for fixture_file in "${fixture_files[@]}"; do
    sha256sum "${fixture_profile_root}/${fixture_file}"
  done
) >"${fixture_input_hashes}"
fixture_input_set_sha256="$(sha256sum "${fixture_input_hashes}" | awk '{print $1}')"

task_name="kwn_characteristic_reference_v1"
output_root="${RUN_ROOT}/outputs/${task_name}"
report_root="${RUN_ROOT}/reports/${task_name}"
max_wall_s="${MAX_WALL_S:-82800}"
max_steps="${MAX_STEPS:-2000000}"

{
  printf 'hostname=%s\n' "$(hostname)"
  printf 'slurm_job_id=%s\n' "${SLURM_JOB_ID:-UNSCHEDULED}"
  printf 'slurm_partition=%s\n' "${SLURM_JOB_PARTITION:-UNSPECIFIED}"
  printf 'slurm_qos=%s\n' "${SLURM_JOB_QOS:-UNSPECIFIED}"
  printf 'slurm_memory_per_node_mb=%s\n' "${SLURM_MEM_PER_NODE:-UNSPECIFIED}"
  printf 'source_root=%s\n' "${SOURCE_ROOT}"
  printf 'source_commit=%s\n' "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)"
  printf 'source_branch=%s\n' "$(git -C "${SOURCE_ROOT}" branch --show-current)"
  printf 'source_status=%s\n' "$(git -C "${SOURCE_ROOT}" status --short)"
  printf 'python_bin=%s\n' "${PYTHON_BIN}"
  printf 'python_version=%s\n' "$("${PYTHON_BIN}" -c 'import sys; print(sys.version.replace("\n", " "))')"
  printf 'numpy_version=%s\n' "$("${PYTHON_BIN}" -c 'import numpy; print(numpy.__version__)')"
  printf 'scipy_version=%s\n' "$("${PYTHON_BIN}" -c 'import scipy; print(scipy.__version__)')"
  printf 'fixture_profile_root=%s\n' "${fixture_profile_root}"
  printf 'fixture_profile_root_resolved=%s\n' "$(readlink -f "${fixture_profile_root}")"
  printf 'fixture_input_sha256_file=%s\n' "${fixture_input_hashes}"
  printf 'fixture_input_set_sha256=%s\n' "${fixture_input_set_sha256}"
  printf 'max_wall_s=%s\n' "${max_wall_s}"
  printf 'max_steps=%s\n' "${max_steps}"
  printf 'scope=KWN_BETA_ONLY_NUMERICAL_REFERENCE_NO_PF_CUDA\n'
} >"${RUN_ROOT}/launch_provenance.txt"

set +e
"${PYTHON_BIN}" "${SOURCE_ROOT}/scripts/run_kwn_characteristic_reference_v1.py" all \
  --output-root "${output_root}" \
  --report-root "${report_root}" \
  --max-wall-s "${max_wall_s}" \
  --max-steps "${max_steps}"
runner_rc=$?
set -e

printf '%s\n' "${runner_rc}" >"${RUN_ROOT}/runner_exit_code.txt"
if [[ "${runner_rc}" -eq 0 ]]; then
  printf 'PASS_KWN_CHARACTERISTIC_REFERENCE_DRIVER_V1\n' >"${RUN_ROOT}/status.txt"
else
  printf 'NONPASS_KWN_CHARACTERISTIC_REFERENCE_DRIVER_V1\n' >"${RUN_ROOT}/status.txt"
fi
exit "${runner_rc}"
