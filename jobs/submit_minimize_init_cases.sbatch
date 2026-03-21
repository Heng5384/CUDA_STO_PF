#!/usr/bin/env bash
# 标准化脚本：Slurm 多 case 最小化模板实例
#
# 用途：
# - 在 Slurm 上提交多初值最小化测试
# - 针对固定 radius 跑多组 init-test-id
#
# 可复制修改：
# - 改 #SBATCH 资源区
# - 改默认参数区
# - 改循环体里的 CASE_TAG / main_cuda flags

#SBATCH --job-name=min_init_cases
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

# ------------------------------------------------------------
# A. 启动与项目定位
# ------------------------------------------------------------
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
site_redirect_slurm_logs "minimize_init_cases"
site_print_env_banner
site_build_main_cuda

# ------------------------------------------------------------
# B. 默认参数区
# ------------------------------------------------------------
NX=${NX:-512}
NY=${NY:-512}
NZ=${NZ:-512}
OUT_EVERY=${OUT_EVERY:-10000}
CSV_OUT_EVERY=${CSV_OUT_EVERY:-10}

NSTEPS=${NSTEPS:-30000}
MIN_DT=${MIN_DT:-1e-1}

MIN_RMS_DPHI_THRESHOLD=${MIN_RMS_DPHI_THRESHOLD:-1e-5}
MIN_RMS_DY_THRESHOLD=${MIN_RMS_DY_THRESHOLD:-1e-5}
MIN_ENERGY_DIFF_REL_THRESHOLD=${MIN_ENERGY_DIFF_REL_THRESHOLD:-1e-7}
MIN_RMS_RES_FOR_ENERGY_PLATEAU=${MIN_RMS_RES_FOR_ENERGY_PLATEAU:-1e-4}
MIN_RMS_RES_THRESHOLD=${MIN_RMS_RES_THRESHOLD:-1e-4}
MIN_VOL_ERR_REL_THRESHOLD=${MIN_VOL_ERR_REL_THRESHOLD:-3e-2}
MIN_CONVERGENCE_STEPS=${MIN_CONVERGENCE_STEPS:-100}
MIN_DT_SAFETY_LIMIT=${MIN_DT_SAFETY_LIMIT:-1e-6}
ETA_LAMBDA_VOL=${ETA_LAMBDA_VOL:-0.9}
POST_PROJ_ITERS=${POST_PROJ_ITERS:-1}

V0=${V0:-0.0}
ELASTIC=${ELASTIC:-1}
XB_OUT=${XB_OUT:-0.03}
TEMP_C=${TEMP_C:-400.0}
RADIUS=${RADIUS:-10.0}

CASE_LIST=(${CASE_LIST:-0 1 2 3 4 5 6 7})

echo "====================================================="
echo "固定 radius = ${RADIUS}，开始 Slurm 多初值测试"
echo "====================================================="

# ------------------------------------------------------------
# C. 循环执行
# ------------------------------------------------------------
for TID in "${CASE_LIST[@]}"; do
  CASE_TAG="test_${TID}"

  echo "-----------------------------------------------------"
  echo "运行 case: init-test-id = ${TID}"
  echo "case_tag = ${CASE_TAG}"
  echo "-----------------------------------------------------"

  PF_PARAM_FILE="$(site_generate_pf_param_file "slurm_min_init_${CASE_TAG}_T${TEMP_C}")"
  echo "PF_PARAM = ${PF_PARAM_FILE}"

  # ----------------------------------------------------------
  # D. main_cuda 调用
  # ----------------------------------------------------------
  set -x
  ./main_cuda "${NX}" "${NY}" "${NZ}" "${MIN_DT}" "${NSTEPS}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" "${ELASTIC}" \
    --pf-param-file "${PF_PARAM_FILE}" \
    --mode=minimize \
    --minimize-full-model \
    --minimize-max-iter "${NSTEPS}" \
    --minimize-dt "${MIN_DT}" \
    --V0 "${V0}" \
    --radius "${RADIUS}" \
    --elastic "${ELASTIC}" \
    --ic-23d-xB-out "${XB_OUT}" \
    --init-test-id "${TID}" \
    --init-case-tag "${CASE_TAG}" \
    --minimize-rms-dphi-threshold "${MIN_RMS_DPHI_THRESHOLD}" \
    --minimize-rms-dY-threshold "${MIN_RMS_DY_THRESHOLD}" \
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
