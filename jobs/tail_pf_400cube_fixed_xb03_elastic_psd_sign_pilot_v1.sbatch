#!/usr/bin/env bash
#SBATCH --job-name=pf400_xb03_tail
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

# Queue-preserving, read-only tail.  It never resumes or changes the pilot.
set -euo pipefail
: "${TARGET_ROOT:?TARGET_ROOT is required}"
if [[ -f "${TARGET_ROOT}/status.txt" ]]; then
  cp "${TARGET_ROOT}/status.txt" "${TARGET_ROOT}/tail_observed_status.txt"
  printf 'READ_ONLY_TAIL_OBSERVED\n' >>"${TARGET_ROOT}/tail_observed_status.txt"
else
  printf 'READ_ONLY_TAIL_TARGET_MISSING\n' >"${TARGET_ROOT}/tail_observed_status.txt"
fi
