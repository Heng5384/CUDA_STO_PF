#!/usr/bin/env bash
# gpu_uvip queue-policy tail. It never mutates or reruns the qualification.

#SBATCH --job-name=pf_zm_tail
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=00:05:00
#SBATCH --mem=1G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
mkdir -p "${OUT_ROOT}"
{
  echo "SLURM_JOB_ID=${SLURM_JOB_ID}"
  echo "dependency_satisfied=true"
  if [[ -f "${OUT_ROOT}/status.txt" ]]; then
    echo "target_status=$(tr -d '\r\n' <"${OUT_ROOT}/status.txt")"
  else
    echo "target_status=MISSING"
  fi
} >"${OUT_ROOT}/tail_status.txt"
