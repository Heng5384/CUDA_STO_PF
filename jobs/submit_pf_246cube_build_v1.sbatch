#!/usr/bin/env bash
#SBATCH --job-name=pf246_build
#SBATCH --partition=gpu_vip_24h
#SBATCH --qos=gpu_vip_24h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=00:30:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
BUILD_ROOT="${BUILD_ROOT:?BUILD_ROOT is required}"
[[ ! -e "${BUILD_ROOT}" ]] || {
  echo "[fatal] refusing to overwrite BUILD_ROOT: ${BUILD_ROOT}" >&2
  exit 2
}
mkdir -p "${BUILD_ROOT}"
exec >"${BUILD_ROOT}/stdout.log" 2>"${BUILD_ROOT}/stderr.log"
trap 'rc=$?; if [[ "${rc}" -ne 0 ]]; then printf "BLOCKED_PF_246CUBE_CLUSTER_BUILD_V1\\nexit_code=%s\\n" "${rc}" >"${BUILD_ROOT}/status.txt"; fi' EXIT

cd "${SOURCE_ROOT}"
source jobs/_site_env.sh
site_prepare_cuda_env
make main_cuda CUDA_ARCH="${CUDA_ARCH:-sm_80}" CUDA_ROOT="${CUDA_ROOT}"
make test_pf_zero_mode_checkpoint CXX=c++
c++ -std=c++17 -O3 tools/analyze_pf_246cube_particle_lineage_v1.cpp \
  -lstdc++fs \
  -o "${BUILD_ROOT}/analyze_pf_246cube_particle_lineage_v1"
sha256sum \
  main_cuda main_cuda.cu cuda_kernels.cu cuda_common.cu \
  pf_zero_mode_checkpoint.cpp pf_zero_mode_checkpoint.h \
  tools/analyze_pf_246cube_particle_lineage_v1.cpp \
  "${BUILD_ROOT}/analyze_pf_246cube_particle_lineage_v1" \
  >"${BUILD_ROOT}/build_hashes.sha256"
printf 'PASS_PF_246CUBE_CLUSTER_BUILD_V1\n' >"${BUILD_ROOT}/status.txt"
