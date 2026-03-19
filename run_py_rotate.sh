#!/bin/bash

# ============================================================
# 本地运行：单一 case
# perturbation 预测方向下的 oblate 初始核
# - Grid: 256 x 256 x 256
# - 短轴沿预测方向 e3 = n
# - 预测方向来自 Python:
#     theta = 101 deg
#     phi   = 39 deg
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/lib/site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"
site_print_env_banner
site_build_main_cuda

# ============================================================
# 网格与输出
# ============================================================
NX=256
NY=256
NZ=256

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
# 物理参数
# ============================================================
V0=${V0:-0.0}
RADIUS=${RADIUS:-7.0}
ELASTIC=${ELASTIC:-1}
XB_OUT=${XB_OUT:-0.03}
TEMP_C=${TEMP_C:-380.0}

# ============================================================
# perturbation 预测方向
# 短轴 e3 = n
# ============================================================
PRED_THETA=101
PRED_PHI=39

# ============================================================
# oblate 形貌参数
# 由于 qz 对应 e3=n，所以 Rz 要最小，表示短轴沿预测方向
# 这里先用一个比较明确的扁椭球比例
# ============================================================
OBLATE_RX=${OBLATE_RX:-1.25}
OBLATE_RY=${OBLATE_RY:-1.25}
OBLATE_RZ=${OBLATE_RZ:-0.70}

CASE_TAG=${CASE_TAG:-pert_r7_oblate_pred_256}

echo ""
echo "=========================================="
echo "运行单一 case：perturbation 预测方向短轴初始核"
echo "=========================================="
echo "  Grid: ${NX} x ${NY} x ${NZ}"
echo "  Radius: ${RADIUS}"
echo "  Elastic: ${ELASTIC}"
echo "  Steps: ${NSTEPS}"
echo "  Theta: ${PRED_THETA}"
echo "  Phi: ${PRED_PHI}"
echo "  Axis ratio: rx=${OBLATE_RX}, ry=${OBLATE_RY}, rz=${OBLATE_RZ}"
echo "  Case tag: ${CASE_TAG}"
echo "=========================================="
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
  --init-shape ellipsoid \
  --init-axis-ratio-rx "${OBLATE_RX}" \
  --init-axis-ratio-ry "${OBLATE_RY}" \
  --init-axis-ratio-rz "${OBLATE_RZ}" \
  --init-tilt-theta-deg "${PRED_THETA}" \
  --init-tilt-phi-deg "${PRED_PHI}" \
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

echo ""
echo "=========================================="
echo "完成：${CASE_TAG}"
echo "=========================================="