#!/usr/bin/env bash
#SBATCH --job-name=pf_profile_6to15
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=2-00:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${BASE_LIBRARY_ROOT:?BASE_LIBRARY_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"

cd "${SOURCE_ROOT}"
source jobs/_site_env.sh
site_prepare_cuda_env
[[ -x main_cuda ]] || { echo "[fatal] main_cuda is missing" >&2; exit 2; }
sha256sum main_cuda >"${RUN_ROOT}.binary.sha256" 2>/dev/null || true
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader \
  >"${RUN_ROOT}.gpu.csv" 2>/dev/null || true

export SOURCE_ROOT BASE_LIBRARY_ROOT RUN_ROOT
export SOURCE_COMMIT_OVERRIDE="${SOURCE_COMMIT_OVERRIDE:-NO_GIT_COMMIT}"
export GRID_N="${GRID_N:-96}"
export MINIMIZE_DT="${MINIMIZE_DT:-0.025}"
export MAX_ITER="${MAX_ITER:-16000}"
export MIN_PSEUDO_TIME="${MIN_PSEUDO_TIME:-310}"
export CONVERGENCE_STEPS="${CONVERGENCE_STEPS:-20}"
export RUN_LABEL="${RUN_LABEL:-workstation_6to15}"
export CSV_OUT_EVERY="${CSV_OUT_EVERY:-10}"
export ALLOW_MIXED_BINARY_SAME_SOURCE="${ALLOW_MIXED_BINARY_SAME_SOURCE:-1}"
exec bash jobs/run_pf_elastic_target_profile_extension_6to15_v1.sh
