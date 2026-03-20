#!/usr/bin/env bash
set -euo pipefail

# 标准模板：本地/交互式 GPU 单案例运行脚本
#
# 用途：
# - 适合单案例 dynamics 或 minimize smoke test
# - 适合复制后改成新的本地功能脚本
#
# 复制后通常只需要修改这几块：
# 1. “默认参数区”：网格、步数、物理/运行参数
# 2. “PF 参数生成 tag”：给生成文件起一个稳定名字
# 3. “main_cuda 命令”：补充你的特定功能 flags

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
# 默认每次运行前重新编译；如果已有可用二进制，可改成 0。
DO_BUILD=${DO_BUILD:-1}
if [[ "${DO_BUILD}" == "1" ]]; then
  site_build_main_cuda
fi

# ------------------------------------------------------------
# C. 默认参数区
# ------------------------------------------------------------
# 运行控制参数
NX=${NX:-64}
NY=${NY:-64}
NZ=${NZ:-64}
DT=${DT:-0.01}
NSTEPS=${NSTEPS:-10}
OUT_EVERY=${OUT_EVERY:-10}
CSV_OUT_EVERY=${CSV_OUT_EVERY:-1}
ELASTIC=${ELASTIC:-0}
MIN_RMS_DPHI_THRESHOLD=${MIN_RMS_DPHI_THRESHOLD:-1e-5}
MIN_RMS_DY_THRESHOLD=${MIN_RMS_DY_THRESHOLD:-1e-5}

# 物理参数覆盖（进入 physical_inputs -> .params）
TEMP_C=${TEMP_C:-380.0}
DX_M=${DX_M:-1e-10}
VF_TARGET=${VF_TARGET:-0.04}

# 运行层参数（直接进 main_cuda flags）
RADIUS=${RADIUS:-5.0}
XB_OUT=${XB_OUT:-0.03}

# ------------------------------------------------------------
# D. 生成本次实际使用的 PF 参数文件
# ------------------------------------------------------------
PF_PARAM_FILE="$(site_generate_pf_param_file "template_local_single_T${TEMP_C}")"
echo "[info] pf_param_file=${PF_PARAM_FILE}"

# ------------------------------------------------------------
# E. 执行主程序
# ------------------------------------------------------------
set -x
./main_cuda "${NX}" "${NY}" "${NZ}" "${DT}" "${NSTEPS}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" "${ELASTIC}" \
  --pf-param-file "${PF_PARAM_FILE}" \
  --mode=minimize \
  --minimize-full-model \
  --minimize-max-iter "${NSTEPS}" \
  --minimize-dt "${DT}" \
  --radius "${RADIUS}" \
  --ic-23d-xB-out "${XB_OUT}" \
  --minimize-rms-dphi-threshold "${MIN_RMS_DPHI_THRESHOLD}" \
  --minimize-rms-dY-threshold "${MIN_RMS_DY_THRESHOLD}"
set +x
