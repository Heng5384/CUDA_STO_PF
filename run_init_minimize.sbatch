#!/bin/bash
#SBATCH --job-name=cuda_minB_el_full_cases
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --output=logs/run_minimize_init_cases_%j.out
#SBATCH --error=logs/run_minimize_init_cases_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/lib/site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"
site_print_env_banner
site_build_main_cuda

# =========================
# 网格与输出
# =========================
NX=512
NY=512
NZ=512
OUT_EVERY=10000
CSV_OUT_EVERY=10

# =========================
# Minimization 参数
# =========================
NSTEPS=30000
MIN_DT=1e-1

MIN_RMS_DPHI_THRESHOLD=1e-5
MIN_RMS_DY_THRESHOLD=1e-5
MIN_ENERGY_DIFF_REL_THRESHOLD=1e-7
MIN_RMS_RES_FOR_ENERGY_PLATEAU=1e-4
MIN_RMS_RES_THRESHOLD=1e-4
MIN_VOL_ERR_REL_THRESHOLD=3e-2
MIN_CONVERGENCE_STEPS=100
MIN_DT_SAFETY_LIMIT=1e-6
ETA_LAMBDA_VOL=0.9

POST_PROJ_ITERS=1

# =========================
# 固定物理参数（不再 sweep）
# =========================
V0=0.0
ELASTIC=1
XB_OUT=0.03
TEMP_C=400.0

# 👉 固定 radius（只改这里）
RADIUS=10.0

echo "====================================================="
echo "固定 radius = ${RADIUS}，开始多初值测试"
echo "====================================================="

# =========================
# case 列表（你可以随便扩展）
# =========================
CASE_LIST=(0 1 2 3 4 5 6 7)

for TID in "${CASE_LIST[@]}"; do

  CASE_TAG="test_${TID}"

  echo "-----------------------------------------------------"
  echo "运行 case: init-test-id = ${TID}"
  echo "case_tag = ${CASE_TAG}"
  echo "-----------------------------------------------------"

  set -x
  ./main_cuda "${NX}" "${NY}" "${NZ}" "${MIN_DT}" "${NSTEPS}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" "${ELASTIC}" \
    --mode=minimize \
    --minimize-full-model \
    --minimize-max-iter "${NSTEPS}" \
    --minimize-dt "${MIN_DT}" \
    --V0 "${V0}" \
    --radius "${RADIUS}" \
    --elastic "${ELASTIC}" \
    --ic-23d-xB-out "${XB_OUT}" \
    --temperature-C "${TEMP_C}" \
    --init-test-id "${TID}" \
    --init-case-tag "${CASE_TAG}" \
    --minimize-rms-dphi-threshold "${MIN_RMS_DPHI_THRESHOLD}" \
    --minimize-energy-diff-rel-threshold "${MIN_ENERGY_DIFF_REL_THRESHOLD}" \
    --minimize-rms-res-for-energy-plateau "${MIN_RMS_RES_FOR_ENERGY_PLATEAU}" \
    --minimize-convergence-steps "${MIN_CONVERGENCE_STEPS}" \
    --minimize-dt-safety-limit "${MIN_DT_SAFETY_LIMIT}" \
    --minimize-rms-res-threshold "${MIN_RMS_RES_THRESHOLD}" \
    --minimize-vol-err-rel-threshold "${MIN_VOL_ERR_REL_THRESHOLD}" \
    --eta-lambda-vol "${ETA_LAMBDA_VOL}" \
    --minimize-post-projection-iters "${POST_PROJ_ITERS}"
  set +x

  echo "完成 case ${TID}"
  echo ""

done

echo "======================================"
echo "所有初值测试完成"
echo "======================================"
