#!/usr/bin/env bash
#SBATCH --job-name=pf_elastic_profile_v1
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --output=/data/home/luozhiheng/tmp/pf_elastic_target_profile_cluster_logs_v1_20260730/%x_%j.out
#SBATCH --error=/data/home/luozhiheng/tmp/pf_elastic_target_profile_cluster_logs_v1_20260730/%x_%j.err

set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
RUN_ROOT="${RUN_ROOT:?RUN_ROOT is required}"
SOURCE_COMMIT_OVERRIDE="${SOURCE_COMMIT_OVERRIDE:?SOURCE_COMMIT_OVERRIDE is required}"
RADII_NM="${RADII_NM:-8.0 8.5 9.0 9.5 10.0 10.5 11.0 11.5}"
MINIMIZE_DT="${MINIMIZE_DT:-0.05}"
MAX_ITER="${MAX_ITER:-5000}"
MIN_PSEUDO_TIME="${MIN_PSEUDO_TIME:-300}"

module load cuda/cuda-12.8
CUDA_ROOT="$(cd "$(dirname "$(command -v nvcc)")/.." && pwd)"
export CUDA_ROOT
export CUDA_ARCH="${CUDA_ARCH:-sm_80}"
export SOURCE_ROOT RUN_ROOT SOURCE_COMMIT_OVERRIDE RADII_NM
export MINIMIZE_DT MAX_ITER
export MIN_PSEUDO_TIME
export GRID_N="${GRID_N:-96}"
export RUN_LABEL="${RUN_LABEL:-cluster_dt_refinement}"
export CSV_OUT_EVERY="${CSV_OUT_EVERY:-5}"
export OUT_EVERY="${OUT_EVERY:-${MAX_ITER}}"
export CONVERGENCE_STEPS="${CONVERGENCE_STEPS:-20}"
export POST_PROJECTION_ITERS=0

cd "${SOURCE_ROOT}"
mkdir -p "${SOURCE_ROOT}/jobs/logs" "${SOURCE_ROOT}/jobs/generated"
python3 scripts/test_pf_elastic_target_profile_v1.py \
  >"${SOURCE_ROOT}/jobs/logs/elastic_target_profile_static_tests_${SLURM_JOB_ID}.json"
make main_cuda CUDA_ROOT="${CUDA_ROOT}" CUDA_ARCH="${CUDA_ARCH}" \
  NVCC="$(command -v nvcc)"
sha256sum main_cuda >"${SOURCE_ROOT}/jobs/logs/main_cuda_${SLURM_JOB_ID}.sha256"
nvidia-smi --query-gpu=index,name,uuid,driver_version,memory.total \
  --format=csv,noheader \
  >"${SOURCE_ROOT}/jobs/logs/gpu_${SLURM_JOB_ID}.csv"

jobs/run_pf_elastic_target_profile_library_v1.sh
