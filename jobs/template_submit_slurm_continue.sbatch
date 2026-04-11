#!/usr/bin/env bash
# 标准模板：Slurm continue 提交脚本
#
# 用途：
# - 适合从已有 VTK 结果继续跑 minimize / dynamics
# - full-model 下优先使用结果目录中的 pf_input.params
# - 如果结果目录没有 pf_input.params，则回退到当前脚本生成的 fallback PF 参数

#SBATCH --job-name=template_continue
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
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
site_redirect_slurm_logs "template_continue"
site_print_env_banner
site_build_main_cuda

# ------------------------------------------------------------
# B. 默认参数区
# ------------------------------------------------------------
NX=${NX:-64}
NY=${NY:-64}
NZ=${NZ:-64}
MIN_DT=${MIN_DT:-0.01}
NSTEPS=${NSTEPS:-100}
OUT_EVERY=${OUT_EVERY:-50}
CSV_OUT_EVERY=${CSV_OUT_EVERY:-10}
ELASTIC=${ELASTIC:-0}

MINIMIZE_FULL_MODEL=${MINIMIZE_FULL_MODEL:-1}
MIN_RMS_DPHI_THRESHOLD=${MIN_RMS_DPHI_THRESHOLD:-1e-5}
MIN_RMS_DY_THRESHOLD=${MIN_RMS_DY_THRESHOLD:-1e-5}
MODE=${MODE:-dynamics-continue}

CONTINUE_PHI_VTK=${CONTINUE_PHI_VTK:-}
CONTINUE_XB_VTK=${CONTINUE_XB_VTK:-}

TEMP_C=${TEMP_C:-380.0}
DX_M=${DX_M:-1e-10}
VF_TARGET=${VF_TARGET:-0.04}

RADIUS=${RADIUS:-5.0}
XB_OUT=${XB_OUT:-0.03}

if [[ -z "${CONTINUE_PHI_VTK}" ]]; then
  echo "[fatal] CONTINUE_PHI_VTK 未设置。" >&2
  exit 2
fi

# ------------------------------------------------------------
# C. 生成 fallback PF 参数并执行
# ------------------------------------------------------------
PF_PARAM_FILE="$(site_generate_pf_param_file "template_continue_T${TEMP_C}")"
echo "FALLBACK_PF_PARAM=${PF_PARAM_FILE}"

CMD=(
  ./main_cuda "${NX}" "${NY}" "${NZ}" "${MIN_DT}" "${NSTEPS}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" "${ELASTIC}"
  --pf-param-file "${PF_PARAM_FILE}"
  --mode="${MODE}"
  --minimize-max-iter "${NSTEPS}"
  --minimize-dt "${MIN_DT}"
  --radius-phys-nm "${RADIUS}"
  --ic-23d-xB-out "${XB_OUT}"
  --continue-phi-vtk "${CONTINUE_PHI_VTK}"
  --minimize-rms-dphi-threshold "${MIN_RMS_DPHI_THRESHOLD}"
  --minimize-rms-dY-threshold "${MIN_RMS_DY_THRESHOLD}"
)

if [[ "${MINIMIZE_FULL_MODEL}" == "1" ]]; then
  CMD+=(--minimize-full-model)
fi

if [[ -n "${CONTINUE_XB_VTK}" ]]; then
  CMD+=(--continue-xB-vtk "${CONTINUE_XB_VTK}")
fi

set -x
"${CMD[@]}"
set +x
