#!/usr/bin/env bash
#SBATCH --job-name=pf_elastic_ws
#SBATCH --partition=gpu_vip_24h
#SBATCH --qos=gpu_vip_24h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
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
BUILD_ROOT="${RUN_ROOT}.build"
[[ ! -e "${BUILD_ROOT}" ]] || {
  echo "[fatal] refusing to overwrite BUILD_ROOT: ${BUILD_ROOT}" >&2
  exit 2
}
mkdir -p "${BUILD_ROOT}"
exec >"${BUILD_ROOT}/stdout.log" 2>"${BUILD_ROOT}/stderr.log"
trap 'rc=$?; if [[ "${rc}" -ne 0 ]]; then printf "BLOCKED_ELASTIC_WARM_START_RESIDUAL_BUILD_V1\\nexit_code=%s\\n" "${rc}" >"${BUILD_ROOT}/status.txt"; fi' EXIT

make main_cuda CUDA_ARCH="${CUDA_ARCH:-sm_80}" \
  CUDA_ROOT="${CUDA_ROOT}" NVCC="${NVCC}"
make test_pf_zero_mode_checkpoint CXX=c++
printf 'PASS_ELASTIC_WARM_START_RESIDUAL_BUILD_V1\n' \
  >"${BUILD_ROOT}/status.txt"
sha256sum \
  main_cuda main_cuda.cu cuda_kernels.cu cuda_kernels.h pf_params.h \
  pf_zero_mode_checkpoint.cpp pf_zero_mode_checkpoint.h \
  >"${BUILD_ROOT}/build_hashes.sha256"

exec env \
  SOURCE_ROOT="${SOURCE_ROOT}" \
  FIXTURE_ROOT="${FIXTURE_ROOT}" \
  RUN_ROOT="${RUN_ROOT}" \
  PARAM_FILE="${PARAM_FILE}" \
  TRACKER_LDFLAGS="-lstdc++fs" \
  bash jobs/run_pf_elastic_warm_start_residual_v1.sh
