#!/usr/bin/env bash
set -euo pipefail

# 标准模板：本地/交互式 GPU 的 continue 运行脚本
#
# 用途：
# - 适合从某个已有 VTK 结果继续跑 minimize / dynamics
# - full-model 下如果不给 CONTINUE_XB_VTK，会自动按结果目录内的 pf_input.params
#   与当前物理参数逻辑，从 phi 重建 xB/Y
#
# 复制后通常只需要修改这几块：
# 1. “默认参数区”：CONTINUE_PHI_VTK、网格、步数
# 2. “fallback PF 参数”：当结果目录没有 pf_input.params 时使用
# 3. “main_cuda 命令”：补充你的特定 flags

# ------------------------------------------------------------
# A. 启动与项目定位
# ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"
site_prepare_job_dirs
cd "${PROJECT_ROOT}"
site_print_env_banner

# ------------------------------------------------------------
# B. 编译策略
# ------------------------------------------------------------
DO_BUILD=${DO_BUILD:-1}
if [[ "${DO_BUILD}" == "1" ]]; then
  site_build_main_cuda
fi

# ------------------------------------------------------------
# C. 默认参数区
# ------------------------------------------------------------
NX=${NX:-64}
NY=${NY:-64}
NZ=${NZ:-64}
DT=${DT:-0.01}
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

# fallback 物理参数：仅当 continue 结果目录没有 pf_input.params 时才作为后备输入
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
# D. 生成 fallback PF 参数文件
# ------------------------------------------------------------
PF_PARAM_FILE="$(site_generate_pf_param_file "template_local_continue_T${TEMP_C}")"
echo "[info] fallback_pf_param_file=${PF_PARAM_FILE}"

# ------------------------------------------------------------
# E. 执行主程序
# ------------------------------------------------------------
CMD=(
  ./main_cuda "${NX}" "${NY}" "${NZ}" "${DT}" "${NSTEPS}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" "${ELASTIC}"
  --pf-param-file "${PF_PARAM_FILE}"
  --mode="${MODE}"
  --minimize-max-iter "${NSTEPS}"
  --minimize-dt "${DT}"
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
