#!/usr/bin/env bash
# 专用脚本：在 gpu_uvip 队列补跑 r100 的 sphere 对照
#
# 用途：
# - 在
#   Results/chel_T400_cuda_512x512x512_dt0.1_steps30000_r100.00_xB0.030
#   同一 output_root 下生成 dirscan_sphere_r100
# - 参数与当前 A/B r100 大目录保持一致，仅把初始核换成 sphere

#SBATCH --job-name=dirscan_sphere_r100
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
site_redirect_slurm_logs "dirscan_sphere_r100_uvip"
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
RADIUS=${RADIUS:-100}
CASE_TAG=${CASE_TAG:-dirscan_sphere_r100}

echo "====================================================="
echo "512^3 minimize full-model sphere 对照"
echo "  Grid: ${NX} x ${NY} x ${NZ}"
echo "  Queue: gpu_uvip"
echo "  TEMP_C: ${TEMP_C}"
echo "  XB_OUT: ${XB_OUT}"
echo "  ELASTIC: ${ELASTIC}"
echo "  Radius: ${RADIUS}"
echo "  Init shape: sphere"
echo "  Case tag: ${CASE_TAG}"
echo "====================================================="

# ------------------------------------------------------------
# C. 生成 PF 参数
# ------------------------------------------------------------
PF_PARAM_FILE="$(site_generate_pf_param_file "slurm_${CASE_TAG}_T${TEMP_C}")"
echo "PF_PARAM=${PF_PARAM_FILE}"

# ------------------------------------------------------------
# D. main_cuda 调用
# ------------------------------------------------------------
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
  --init-shape sphere
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

set -x
"${CMD[@]}"
set +x
