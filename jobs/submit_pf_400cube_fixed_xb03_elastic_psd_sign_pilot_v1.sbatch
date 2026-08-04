#!/usr/bin/env bash
#SBATCH --job-name=pf400_xb03_psd_sign
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
: "${RUN_ROOT:?RUN_ROOT is required}"
: "${SPEC_PATH:?SPEC_PATH is required}"
cd "${SOURCE_ROOT}"
source jobs/_site_env.sh
site_prepare_cuda_env
make main_cuda CUDA_ROOT="${CUDA_ROOT}" CUDA_ARCH="${CUDA_ARCH:-sm_80}" NVCC="$(command -v nvcc)"
exec env SOURCE_ROOT="${SOURCE_ROOT}" FIXTURE_ROOT="${FIXTURE_ROOT}" RUN_ROOT="${RUN_ROOT}" SPEC_PATH="${SPEC_PATH}" bash jobs/run_pf_400cube_fixed_xb03_elastic_psd_sign_pilot_v1.sh
