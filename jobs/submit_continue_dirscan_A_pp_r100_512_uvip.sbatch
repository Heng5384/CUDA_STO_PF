#!/usr/bin/env bash
# 专用脚本：继续跑 dirscan_A_pp_r100 的 512^3 full-model minimize
#
# 当前目标：
# - 从已有结果
#   /data/home/luozhiheng/CUDA_STO_PF/Results/chel_T400_cuda_512x512x512_dt0.1_steps21000_r100.00_xB0.030/dirscan_A_pp_r100
#   的 phi_final_dirscan_A_pp_r100.vtk 继续跑
# - 额外再跑 20000 步
# - 维持：
#   TEMP_C=400
#   XB_OUT=0.03
#   ELASTIC=1
#   radius=100
#   dt=0.1
#
# 注意：
# - 该目录当前没有 xB_final.vtk，也没有 pf_input.params
# - 因此这次 continue 会：
#   1. 从 phi_final_dirscan_A_pp_r100.vtk 读入 phi
#   2. 按 full-model init 的质量守恒逻辑重建 xB/Y
#   3. 物理参数回退到当前仓库里的 physical_inputs.example.json + 下方环境变量覆盖
# - 使用新代码后，continue 输出会继续写回原来的 output_root 下，并自动生成 continue_1 / continue_2 ...

#SBATCH --job-name=cont_App_r100
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
site_redirect_slurm_logs "cont_App_r100"
site_print_env_banner
site_build_main_cuda

# ------------------------------------------------------------
# B. 默认参数区
# ------------------------------------------------------------
NX=${NX:-512}
NY=${NY:-512}
NZ=${NZ:-512}

NSTEPS=${NSTEPS:-20000}
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

CONTINUE_PHI_VTK=${CONTINUE_PHI_VTK:-/data/home/luozhiheng/CUDA_STO_PF/Results/chel_T400_cuda_512x512x512_dt0.1_steps21000_r100.00_xB0.030/dirscan_A_pp_r100/phi_final_dirscan_A_pp_r100.vtk}
CONTINUE_XB_VTK=${CONTINUE_XB_VTK:-}

echo "====================================================="
echo "continue dirscan_A_pp_r100"
echo "  Grid: ${NX} x ${NY} x ${NZ}"
echo "  Queue: gpu_uvip"
echo "  TEMP_C: ${TEMP_C}"
echo "  XB_OUT: ${XB_OUT}"
echo "  ELASTIC: ${ELASTIC}"
echo "  Radius: ${RADIUS}"
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
PF_PARAM_FILE="$(site_generate_pf_param_file "continue_dirscan_A_pp_r100_T${TEMP_C}")"
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
