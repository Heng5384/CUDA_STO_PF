#!/usr/bin/env bash
# 标准化脚本：Slurm 半径扫描模板实例（512^3, uvip, minimize full-model）
#
# 用途：
# - 在 gpu_uvip 队列上提交 512x512x512 的 minimize full-model 半径扫描
# - 对每个 radius，依次运行 3 个初始核子：
#   1. A 定向 oblate
#   2. B 定向 oblate
#   3. sphere 球核对照
# - 默认扫描半径：
#   50, 60, 70, 80, 90
# - 默认物理与运行参数：
#   TEMP_C=400
#   XB_OUT=0.03
#   ELASTIC=1
#   NSTEPS=30000
#   init-shape=ellipsoid / sphere
#   axis-ratio=(1.25, 1.00, 0.60)
#
# 2 个定向 oblate 方向：
#   A: n=( 0.786687491,  0.581195444,  0.208169755) -> theta=77.985, phi=36.457
#   B: n=( 0.744921649, -0.625009732, -0.233355034) -> theta=103.495, phi=320.002

#SBATCH --job-name=min_full_r50_90
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=96:00:00
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
site_redirect_slurm_logs "min_full_r50_90"
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
MIN_RMS_RES_FOR_ENERGY_PLATEAU=${MIN_RMS_RES_FOR_ENERGY_PLATEAU:-1e-5}
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

R_MIN=${R_MIN:-50}
R_MAX=${R_MAX:-90}
R_STEP=${R_STEP:-10}

OBLATE_RX=${OBLATE_RX:-1.25}
OBLATE_RY=${OBLATE_RY:-1.00}
OBLATE_RZ=${OBLATE_RZ:-0.60}

echo "====================================================="
echo "512^3 minimize full-model 半径扫描"
echo "  Grid: ${NX} x ${NY} x ${NZ}"
echo "  Queue: gpu_uvip"
echo "  TEMP_C: ${TEMP_C}"
echo "  XB_OUT: ${XB_OUT}"
echo "  ELASTIC: ${ELASTIC}"
echo "  Radius sweep: ${R_MIN} -> ${R_MAX} (step ${R_STEP})"
echo "  Cases: A, B, sphere"
echo "  Axis ratio for oblate: rx=${OBLATE_RX}, ry=${OBLATE_RY}, rz=${OBLATE_RZ}"
echo "====================================================="
echo ""

CASE_NAMES=(A B sphere)
CASE_SHAPES=(ellipsoid ellipsoid sphere)
CASE_THETA=(77.985 103.495 0.0)
CASE_PHI=(36.457 320.002 0.0)
CASE_NX=(0.786687491 0.744921649 0.0)
CASE_NY=(0.581195444 -0.625009732 0.0)
CASE_NZ=(0.208169755 -0.233355034 1.0)

# ------------------------------------------------------------
# C. 双循环执行：半径 x 形核方式
# ------------------------------------------------------------
for ((RADIUS=R_MIN; RADIUS<=R_MAX; RADIUS+=R_STEP)); do
  echo "#####################################################"
  echo "运行 radius=${RADIUS}"
  echo "#####################################################"

  for idx in "${!CASE_NAMES[@]}"; do
    CASE_NAME="${CASE_NAMES[$idx]}"
    CASE_SHAPE="${CASE_SHAPES[$idx]}"
    THETA="${CASE_THETA[$idx]}"
    PHI="${CASE_PHI[$idx]}"
    NX_DIR="${CASE_NX[$idx]}"
    NY_DIR="${CASE_NY[$idx]}"
    NZ_DIR="${CASE_NZ[$idx]}"
    CASE_TAG="dirscan_${CASE_NAME}_r${RADIUS}"

    echo "-----------------------------------------------------"
    echo "运行方向=${CASE_NAME}"
    echo "CASE_TAG=${CASE_TAG}"
    echo "shape=${CASE_SHAPE}"
    if [[ "${CASE_SHAPE}" == "ellipsoid" ]]; then
      echo "normal=(${NX_DIR}, ${NY_DIR}, ${NZ_DIR})"
      echo "theta/phi=${THETA}/${PHI}"
    else
      echo "normal=<sphere>"
    fi
    echo "-----------------------------------------------------"

    PF_PARAM_FILE="$(site_generate_pf_param_file "slurm_${CASE_TAG}_T${TEMP_C}")"
    echo "PF_PARAM=${PF_PARAM_FILE}"

    set -x
    CMD=(
      ./main_cuda "${NX}" "${NY}" "${NZ}" "${MIN_DT}" "${NSTEPS}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" "${ELASTIC}"
      --pf-param-file "${PF_PARAM_FILE}"
      --mode=minimize
      --minimize-full-model
      --minimize-max-iter "${NSTEPS}"
      --minimize-dt "${MIN_DT}"
      --V0 "${V0}"
      --radius "${RADIUS}"
      --elastic "${ELASTIC}"
      --ic-23d-xB-out "${XB_OUT}"
      --init-shape "${CASE_SHAPE}"
      --init-case-tag "${CASE_TAG}"
      --minimize-rms-dphi-threshold "${MIN_RMS_DPHI_THRESHOLD}"
      --minimize-rms-dY-threshold "${MIN_RMS_DY_THRESHOLD}"
      --minimize-energy-diff-rel-threshold "${MIN_ENERGY_DIFF_REL_THRESHOLD}"
      --minimize-rms-res-for-energy-plateau "${MIN_RMS_RES_FOR_ENERGY_PLATEAU}"
      --minimize-convergence-steps "${MIN_CONVERGENCE_STEPS}"
      --minimize-dt-safety-limit "${MIN_DT_SAFETY_LIMIT}"
      --minimize-rms-res-threshold "${MIN_RMS_RES_THRESHOLD}"
      --minimize-vol-err-rel-threshold "${MIN_VOL_ERR_REL_THRESHOLD}"
      --eta-lambda-vol "${ETA_LAMBDA_VOL}"
      --minimize-post-projection-iters "${POST_PROJ_ITERS}"
    )

    if [[ "${CASE_SHAPE}" == "ellipsoid" ]]; then
      CMD+=(
        --init-axis-ratio-rx "${OBLATE_RX}"
        --init-axis-ratio-ry "${OBLATE_RY}"
        --init-axis-ratio-rz "${OBLATE_RZ}"
        --init-tilt-theta-deg "${THETA}"
        --init-tilt-phi-deg "${PHI}"
      )
    fi

    "${CMD[@]}"
    set +x

    echo "完成方向=${CASE_NAME}"
    echo ""
  done
done

echo "====================================================="
echo "半径扫描完成"
echo "====================================================="
