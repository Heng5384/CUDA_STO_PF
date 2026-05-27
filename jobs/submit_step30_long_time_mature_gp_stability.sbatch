#!/usr/bin/env bash
# Step 30: long-time mature observed-GP stability study
#
# Submits 4 production-like GP-only cases as a Slurm array and
# postprocesses them into a shared summary/report with a dependent job.

#SBATCH --job-name=step30_gp_stability
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null
#SBATCH --array=0-3

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_site_env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/jobs"
elif [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/_site_env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"
site_redirect_slurm_logs "step30_gp_stability"
site_print_env_banner
site_build_main_cuda

OUT_ROOT="${OUT_ROOT:-/tmp/step30_long_time_mature_gp_stability}"
export OUT_ROOT
export BIN="${PROJECT_ROOT}/main_cuda"

CASE_KEYS=(
  "xb03_r1_p02_dep5"
  "xb03_r1_p03_dep5"
  "xb05_r1_p02_dep5"
  "xb05_r1_p03_dep5"
)

IDX="${SLURM_ARRAY_TASK_ID:-0}"
CASE_KEY="${CASE_KEYS[$IDX]}"
export CASE_KEY

echo "[step30] CASE_KEY=${CASE_KEY}"
"${PROJECT_ROOT}/run_step30_long_time_mature_gp_stability.sh"
