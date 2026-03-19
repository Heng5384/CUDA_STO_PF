#!/bin/bash

# ============================================================
# 本地运行：单一 radius + 多初值测试（完整版）
# - 支持 init-test-id sweep
# - 支持自定义 case
# - 支持环境变量覆盖参数
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/lib/site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"
site_print_env_banner
site_build_main_cuda

# ============================================================
# 网格与输出（支持外部覆盖）
# ============================================================
NX=${NX:-256}
NY=${NY:-256}
NZ=${NZ:-256}

OUT_EVERY=${OUT_EVERY:-5000}
CSV_OUT_EVERY=${CSV_OUT_EVERY:-10}

# ============================================================
# Minimization 参数
# ============================================================
NSTEPS=${NSTEPS:-30000}
MIN_DT=${MIN_DT:-1e-1}

# 收敛判据
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

# ============================================================
# 物理参数（关键）
# ============================================================
V0=${V0:-0.0}
RADIUS=${RADIUS:-18.0}
ELASTIC=${ELASTIC:-1}
XB_OUT=${XB_OUT:-0.03}
TEMP_C=${TEMP_C:-380.0}

# ============================================================
# case 列表（可自定义）
# ============================================================
CASE_LIST=${CASE_LIST:-"0 1 2 3 4 5 6 7 custom1 custom2"}

echo ""
echo "=========================================="
echo "运行多初值测试（本地）"
echo "=========================================="
echo "  Grid: ${NX} x ${NY} x ${NZ}"
echo "  Radius: ${RADIUS}"
echo "  Elastic: ${ELASTIC}"
echo "  Steps: ${NSTEPS}"
echo "  Case list: ${CASE_LIST}"
echo "=========================================="
echo ""

# ============================================================
# 主循环
# ============================================================
for CASE in ${CASE_LIST}; do

  echo "-----------------------------------------------------"
  echo "Running case: ${CASE}"
  echo "-----------------------------------------------------"

  EXTRA_ARGS=""

  if [[ "${CASE}" =~ ^[0-9]+$ ]]; then
    # ===== 标准 preset =====
    TID=${CASE}
    CASE_TAG="test_${TID}"
    EXTRA_ARGS="--init-test-id ${TID}"

  elif [[ "${CASE}" == "custom1" ]]; then
    # ===== 强破对称（推荐重点看）=====
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
    # ===== 噪声 + 偏移 =====
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
    ${EXTRA_ARGS} \
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

  echo "完成 case: ${CASE_TAG}"
  echo ""

done

echo "=========================================="
echo "全部初值测试完成"
echo "=========================================="