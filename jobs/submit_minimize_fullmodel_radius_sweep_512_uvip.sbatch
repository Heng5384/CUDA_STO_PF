#!/usr/bin/env bash
# 标准化脚本：Slurm 半径扫描模板实例（512^3, uvip, minimize full-model, rotated oblate）
#
# 用途：
# - 在 gpu_uvip 队列上提交 512x512x512 的 minimize full-model 半径扫描
# - 只使用“旋转角度的 oblate 初始核”，不跑球形对照
# - 默认扫 radius = 100, 105, ..., 200
# - 默认物理与运行参数：
#   TEMP_C=400
#   XB_OUT=0.03
#   ELASTIC=1
#   init-shape=ellipsoid
#   rotate(theta,phi)=101/39   # 之前选定的最低弹性能预测方向
#   axis-ratio=(1.25, 1.25, 0.70)
#
# 可通过环境变量覆盖的常用参数：
#   NX NY NZ
#   NSTEPS MIN_DT OUT_EVERY CSV_OUT_EVERY
#   TEMP_C XB_OUT ELASTIC
#   PRED_THETA PRED_PHI OBLATE_RX OBLATE_RY OBLATE_RZ
#   R_MIN R_MAX R_STEP

#SBATCH --job-name=min_full_r512_sweep
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=48:00:00
#SBATCH --mem=0
#SBATCH --output=jobs/logs/min_full_r512_sweep_%j.out
#SBATCH --error=jobs/logs/min_full_r512_sweep_%j.err

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
site_print_env_banner
site_build_main_cuda

# ------------------------------------------------------------
# B. 默认参数区
# ------------------------------------------------------------
NX=${NX:-512}
NY=${NY:-512}
NZ=${NZ:-512}

NSTEPS=${NSTEPS:-30000}
MIN_DT=${MIN_DT:-1e-1}
OUT_EVERY=${OUT_EVERY:-5000}
CSV_OUT_EVERY=${CSV_OUT_EVERY:-10}

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

PRED_THETA=${PRED_THETA:-101}
PRED_PHI=${PRED_PHI:-39}
OBLATE_RX=${OBLATE_RX:-1.25}
OBLATE_RY=${OBLATE_RY:-1.25}
OBLATE_RZ=${OBLATE_RZ:-0.70}

R_MIN=${R_MIN:-100}
R_MAX=${R_MAX:-200}
R_STEP=${R_STEP:-5}

echo "====================================================="
echo "512^3 minimize full-model 半径扫描"
echo "  Grid: ${NX} x ${NY} x ${NZ}"
echo "  Queue: gpu_uvip"
echo "  TEMP_C: ${TEMP_C}"
echo "  XB_OUT: ${XB_OUT}"
echo "  ELASTIC: ${ELASTIC}"
echo "  Init shape: rotated oblate"
echo "  Theta/Phi: ${PRED_THETA}/${PRED_PHI}"
echo "  Axis ratio: rx=${OBLATE_RX}, ry=${OBLATE_RY}, rz=${OBLATE_RZ}"
echo "  Radius sweep: ${R_MIN} -> ${R_MAX} step ${R_STEP}"
echo "====================================================="
echo ""

RADII=$(python3 - "$R_MIN" "$R_MAX" "$R_STEP" <<'PYEOF'
import sys

rmin, rmax, step = map(float, sys.argv[1:4])
vals = []
r = rmin
eps = 1e-12
while r <= rmax + eps:
    vals.append(f"{r:.6g}")
    r += step
print(" ".join(vals))
PYEOF
)

# ------------------------------------------------------------
# C. 循环执行
# ------------------------------------------------------------
for RADIUS in ${RADII}; do
  RADIUS_TAG="${RADIUS//./p}"
  CASE_TAG="rotate_oblate_r${RADIUS_TAG}"

  echo "-----------------------------------------------------"
  echo "运行 radius=${RADIUS}"
  echo "CASE_TAG=${CASE_TAG}"
  echo "-----------------------------------------------------"

  PF_PARAM_FILE="$(site_generate_pf_param_file "slurm_min_full_r${RADIUS_TAG}_T${TEMP_C}")"
  echo "PF_PARAM=${PF_PARAM_FILE}"

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

  echo "完成 radius=${RADIUS}"
  echo ""
done

echo "====================================================="
echo "全部半径扫描完成"
echo "====================================================="
