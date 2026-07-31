#!/usr/bin/env bash
#SBATCH --job-name=pf246_restart
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

SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
FIXTURE_ROOT="${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
PARAM_FILE="${PARAM_FILE:?PARAM_FILE is required}"

cd "${SOURCE_ROOT}"
source jobs/_site_env.sh
site_prepare_cuda_env
exec env \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  FIXTURE_ROOT="${FIXTURE_ROOT}" \
  RUN_ROOT="${RUN_ROOT}" \
  PARAM_FILE="${PARAM_FILE}" \
  TRACKER_LDFLAGS="-lstdc++fs" \
  bash jobs/run_pf_246cube_short_restart_v1.sh
