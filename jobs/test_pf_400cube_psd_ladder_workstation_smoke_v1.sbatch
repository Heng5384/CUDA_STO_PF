#!/usr/bin/env bash
#SBATCH --job-name=pf400_psd_smoke
#SBATCH --partition=gpu_vip_24h
#SBATCH --qos=gpu_vip_24h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=01:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

export SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
export CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
export CASE_ID="${CASE_ID:-001}"
export QUEUE="${QUEUE:-workstation_smoke}"
export PARAM_FILE="${PARAM_FILE:?PARAM_FILE is required}"
export GRID_N="${GRID_N:-246}"
export EXPECTED_GRID_N="${EXPECTED_GRID_N:-246}"
export SMOKE_RESTART_ONLY=1

cd "${SOURCE_ROOT}"
bash jobs/run_pf_400cube_psd_ladder_production_v1.sh
