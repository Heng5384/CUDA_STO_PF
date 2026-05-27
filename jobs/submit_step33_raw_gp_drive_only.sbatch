#!/usr/bin/env bash
# Step 33: raw GP reaction-drive only validation

#SBATCH --job-name=step33_raw_gp_drive
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

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
site_redirect_slurm_logs "step33_raw_gp_drive"
site_print_env_banner
site_build_main_cuda

OUT_ROOT="${OUT_ROOT:-/tmp/step33_raw_gp_drive_only}"
export OUT_ROOT
export BIN="${PROJECT_ROOT}/main_cuda"
export BASE_PARAM_FILE="${PROJECT_ROOT}/Results/interface_width_audit/pf_strict_dualdx_final.params"

"${PROJECT_ROOT}/run_step33_raw_gp_drive_only.sh"
