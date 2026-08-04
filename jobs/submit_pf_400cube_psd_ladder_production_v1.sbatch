#!/usr/bin/env bash
#SBATCH --job-name=pf400_psd_ladder
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
: "${CASE_ID:?CASE_ID is required}"
: "${QUEUE:?QUEUE is required}"
cd "${SOURCE_ROOT}"
source jobs/_site_env.sh
site_prepare_cuda_env
[[ -x main_cuda ]] || { echo "[fatal] main_cuda is missing; build it before submission" >&2; exit 2; }
exec env \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
  CASE_ID="${CASE_ID}" \
  QUEUE="${QUEUE}" \
  bash jobs/run_pf_400cube_psd_ladder_production_v1.sh
