#!/usr/bin/env bash
# 专用脚本：在 gpu_vip_24h 队列继续/补跑 dirscan_{A,B}_r100
#
# 用途：
# - 针对
#   Results/chel_T400_cuda_512x512x512_dt0.1_steps30000_r100.00_xB0.030
#   下的 dirscan_A_r100 或 dirscan_B_r100 继续跑 minimize full-model
# - 运行时自动选择最合适的继续场：
#   1. 优先用 phi_final/xB_final
#   2. 若 final 不存在，则回退到 phi_init/xB_init
#
# 提交示例：
#   sbatch --job-name=cont_dirscan_A_r100_24h --dependency=afterany:44420 --export=ALL,CASE_NAME=A jobs/submit_continue_dirscan_r100_24h.sbatch
#   sbatch --job-name=cont_dirscan_B_r100_24h --dependency=afterany:44420 --export=ALL,CASE_NAME=B jobs/submit_continue_dirscan_r100_24h.sbatch

#SBATCH --job-name=cont_dirscan_r100
#SBATCH --partition=gpu_vip_24h
#SBATCH --qos=gpu_vip_24h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
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
site_redirect_slurm_logs "cont_dirscan_r100_24h"
site_print_env_banner
site_build_main_cuda

# ------------------------------------------------------------
# B. 默认参数区
# ------------------------------------------------------------
NX=${NX:-512}
NY=${NY:-512}
NZ=${NZ:-512}

# 24h 队列下单个 case 更现实的补跑步数。
NSTEPS=${NSTEPS:-8000}
MIN_DT=${MIN_DT:-1e-1}
OUT_EVERY=${OUT_EVERY:-5000}
CSV_OUT_EVERY=${CSV_OUT_EVERY:-10}
ELASTIC=${ELASTIC:-1}

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
MODE=${MODE:-dynamics-continue}

TEMP_C=${TEMP_C:-400.0}
XB_OUT=${XB_OUT:-0.03}
RADIUS=${RADIUS:-100}
V0=${V0:-0.0}
CASE_NAME=${CASE_NAME:-A}

RESULT_ROOT=${RESULT_ROOT:-${PROJECT_ROOT}/Results/chel_T400_cuda_512x512x512_dt0.1_steps30000_r100.00_xB0.030}
CASE_DIR="${RESULT_ROOT}/dirscan_${CASE_NAME}_r${RADIUS}"

if [[ ! -d "${CASE_DIR}" ]]; then
  echo "[fatal] case 目录不存在: ${CASE_DIR}" >&2
  exit 2
fi

PHI_FINAL="${CASE_DIR}/phi_final_dirscan_${CASE_NAME}_r${RADIUS}.vtk"
XB_FINAL="${CASE_DIR}/xB_final_dirscan_${CASE_NAME}_r${RADIUS}.vtk"
PHI_INIT="${CASE_DIR}/phi_init_dirscan_${CASE_NAME}_r${RADIUS}.vtk"
XB_INIT="${CASE_DIR}/xB_init_dirscan_${CASE_NAME}_r${RADIUS}.vtk"

CONTINUE_PHI_VTK=""
CONTINUE_XB_VTK=""
SOURCE_KIND=""

if [[ -s "${PHI_FINAL}" ]]; then
  CONTINUE_PHI_VTK="${PHI_FINAL}"
  SOURCE_KIND="final"
  if [[ -s "${XB_FINAL}" ]]; then
    CONTINUE_XB_VTK="${XB_FINAL}"
  fi
elif [[ -s "${PHI_INIT}" ]]; then
  CONTINUE_PHI_VTK="${PHI_INIT}"
  SOURCE_KIND="init-fallback"
  if [[ -s "${XB_INIT}" ]]; then
    CONTINUE_XB_VTK="${XB_INIT}"
  fi
else
  echo "[fatal] 未找到可继续的 phi VTK: ${PHI_FINAL} 或 ${PHI_INIT}" >&2
  exit 3
fi

echo "====================================================="
echo "continue dirscan_${CASE_NAME}_r${RADIUS}"
echo "  Grid: ${NX} x ${NY} x ${NZ}"
echo "  Queue: gpu_vip_24h"
echo "  TEMP_C: ${TEMP_C}"
echo "  XB_OUT: ${XB_OUT}"
echo "  ELASTIC: ${ELASTIC}"
echo "  Radius: ${RADIUS}"
echo "  Case dir: ${CASE_DIR}"
echo "  Continue source: ${SOURCE_KIND}"
echo "  Continue phi: ${CONTINUE_PHI_VTK}"
if [[ -n "${CONTINUE_XB_VTK}" ]]; then
  echo "  Continue xB: ${CONTINUE_XB_VTK}"
else
  echo "  Continue xB: <missing -> rebuild from phi>"
fi
echo "  Extra steps: ${NSTEPS}"
echo "====================================================="

# ------------------------------------------------------------
# C. 生成 fallback PF 参数
# ------------------------------------------------------------
PF_PARAM_FILE="$(site_generate_pf_param_file "continue_dirscan_${CASE_NAME}_r${RADIUS}_24h_T${TEMP_C}")"
echo "FALLBACK_PF_PARAM=${PF_PARAM_FILE}"

# ------------------------------------------------------------
# D. main_cuda 调用
# ------------------------------------------------------------
CMD=(
  ./main_cuda "${NX}" "${NY}" "${NZ}" "${MIN_DT}" "${NSTEPS}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" "${ELASTIC}"
  --pf-param-file "${PF_PARAM_FILE}"
  --mode="${MODE}"
  --minimize-full-model
  --minimize-max-iter "${NSTEPS}"
  --minimize-dt "${MIN_DT}"
  --V0 "${V0}"
  --radius "${RADIUS}"
  --elastic "${ELASTIC}"
  --ic-23d-xB-out "${XB_OUT}"
  --continue-phi-vtk "${CONTINUE_PHI_VTK}"
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

if [[ -n "${CONTINUE_XB_VTK}" ]]; then
  CMD+=(--continue-xB-vtk "${CONTINUE_XB_VTK}")
fi

set -x
"${CMD[@]}"
set +x
