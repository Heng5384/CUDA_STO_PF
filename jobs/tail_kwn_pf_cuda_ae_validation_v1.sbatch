#!/usr/bin/env bash
# Read-only afterany queue tail for the CUDA A--E validation root.
# Submit after the target, for example:
#   TARGET_ROOT="$RUN_ROOT" sbatch --dependency=afterany:<target_job_id> \
#     jobs/tail_kwn_pf_cuda_ae_validation_v1.sbatch

#SBATCH --job-name=kwn_pf_cuda_ae_tail_v1
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:10:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
: "${TARGET_ROOT:?TARGET_ROOT is required}"

tail_dir="${TARGET_ROOT}/queue_tail"
mkdir -p "${tail_dir}"
{
  date -u +'%Y-%m-%dT%H:%M:%SZ'
  printf 'dependency=%s\n' "${SLURM_JOB_DEPENDENCY:-<unset>}"
  printf 'target_root=%s\n' "${TARGET_ROOT}"
  if [[ -f "${TARGET_ROOT}/status.txt" ]]; then
    printf 'target_status:\n'
    sed -n '1,40p' "${TARGET_ROOT}/status.txt"
  else
    printf 'target_status=NOT_WRITTEN\n'
  fi
} >"${tail_dir}/afterany_observation.txt"

# Deliberately no resume, cancellation, submission, or numerical-output edit.
