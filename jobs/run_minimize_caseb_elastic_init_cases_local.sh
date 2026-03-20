#!/usr/bin/env bash
set -euo pipefail

# 标准化脚本：本地多 case 最小化模板实例（Case B 弹性版）
#
# 用途：
# - 本地执行 Case B 弹性最小化的多初值测试
# - 保留该实验使用的默认参数
#
# 可复制修改：
# - 改默认参数区
# - 改 CASE_LIST 与 EXTRA_ARGS 分支
# - 改 main_cuda flags 形成新功能脚本

# ------------------------------------------------------------
# A. 启动与项目定位
# ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"
site_prepare_job_dirs
cd "${PROJECT_ROOT}"
site_print_env_banner
site_build_main_cuda

# ------------------------------------------------------------
# B. 默认参数区
# ------------------------------------------------------------
NX=${NX:-256}
NY=${NY:-256}
NZ=${NZ:-256}
OUT_EVERY=${OUT_EVERY:-5000}
CSV_OUT_EVERY=${CSV_OUT_EVERY:-10}

NSTEPS=${NSTEPS:-30000}
MIN_DT=${MIN_DT:-1e-1}

MIN_RMS_DPHI_THRESHOLD=${MIN_RMS_DPHI_THRESHOLD:-1e-5}
MIN_RMS_DY_THRESHOLD=${MIN_RMS_DY_THRESHOLD:-1e-5}
MIN_ENERGY_DIFF_REL_THRESHOLD=${MIN_ENERGY_DIFF_REL_THRESHOLD:-1e-7}
MIN_RMS_RES_FOR_ENERGY_PLATEAU=${MIN_RMS_RES_FOR_ENERGY_PLATEAU:-1e-4}
MIN_RMS_RES_THRESHOLD=${MIN_RMS_RES_THRESHOLD:-1e-4}
MIN_VOL_ERR_REL_THRESHOLD=${MIN_VOL_ERR_REL_THRESHOLD:-5e-3}
MIN_CONVERGENCE_STEPS=${MIN_CONVERGENCE_STEPS:-100}
MIN_DT_SAFETY_LIMIT=${MIN_DT_SAFETY_LIMIT:-1e-6}
ETA_LAMBDA_VOL=${ETA_LAMBDA_VOL:-0.9}
POST_PROJ_ITERS=${POST_PROJ_ITERS:-1}

V0=${V0:-0.0}
RADIUS=${RADIUS:-18.0}
ELASTIC=${ELASTIC:-1}
XB_OUT=${XB_OUT:-0.03}
TEMP_C=${TEMP_C:-380.0}

CASE_LIST=${CASE_LIST:-"0 1 2 3 4 5 6 7 custom1 custom2"}

echo ""
echo "=========================================="
echo "运行 Case B 弹性多初值最小化（本地）"
echo "=========================================="
echo "  Grid: ${NX} x ${NY} x ${NZ}"
echo "  Radius: ${RADIUS}"
echo "  Elastic: ${ELASTIC}"
echo "  Steps: ${NSTEPS}"
echo "  Case list: ${CASE_LIST}"
echo "=========================================="
echo ""

# ------------------------------------------------------------
# C. 循环执行
# ------------------------------------------------------------
for CASE in ${CASE_LIST}; do
  echo "-----------------------------------------------------"
  echo "Running case: ${CASE}"
  echo "-----------------------------------------------------"

  EXTRA_ARGS=""

  if [[ "${CASE}" =~ ^[0-9]+$ ]]; then
    TID=${CASE}
    CASE_TAG="test_${TID}"
    EXTRA_ARGS="--init-test-id ${TID}"
  elif [[ "${CASE}" == "custom1" ]]; then
    CASE_TAG="tilt20_strong"
    EXTRA_ARGS="\
      --init-shape ellipsoid \
      --init-axis-ratio-rx 0.8 \
      --init-axis-ratio-ry 1.0 \
      --init-axis-ratio-rz 1.0 \
      --init-tilt-theta-deg 20 \
      --init-tilt-phi-deg 30 \
    "
  elif [[ "${CASE}" == "custom2" ]]; then
    CASE_TAG="noise_shift"
    EXTRA_ARGS="\
      --init-shape ellipsoid \
      --init-axis-ratio-rx 0.85 \
      --init-axis-ratio-ry 1.0 \
      --init-axis-ratio-rz 1.0 \
      --init-center-shift-x 2.0 \
      --init-center-shift-y -1.0 \
      --init-phi-noise-amp 5e-4 \
      --init-phi-noise-seed 2026 \
    "
  else
    echo "[warning] unknown case: ${CASE}, skip"
    continue
  fi

  echo "CASE_TAG = ${CASE_TAG}"
  echo "ARGS     = ${EXTRA_ARGS}"
  echo ""

  PF_PARAM_FILE="$(site_generate_pf_param_file "min_caseb_${CASE_TAG}_T${TEMP_C}")"
  echo "PF_PARAM = ${PF_PARAM_FILE}"
  echo ""

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
    ${EXTRA_ARGS} \
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

  echo "完成 case: ${CASE_TAG}"
  echo ""
done

echo "=========================================="
echo "全部初值测试完成"
echo "=========================================="
