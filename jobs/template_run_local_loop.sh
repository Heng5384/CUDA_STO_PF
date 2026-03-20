#!/usr/bin/env bash
set -euo pipefail

# 标准模板：本地循环脚本
#
# 用途：
# - 适合本地循环多个 radius / case / temperature
# - 适合复制后改成新的多案例脚本
#
# 复制后主要改：
# 1. 默认参数区
# 2. 循环变量生成逻辑
# 3. LOOP_TAG / CASE_TAG 的命名规则
# 4. main_cuda 的特定 flags

# ------------------------------------------------------------
# A. 启动与项目定位
# ------------------------------------------------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"
site_prepare_job_dirs
cd "${PROJECT_ROOT}"
site_print_env_banner
site_build_main_cuda

# ------------------------------------------------------------
# B. 默认参数区
# ------------------------------------------------------------
NX=${NX:-64}
NY=${NY:-64}
NZ=${NZ:-64}
MIN_DT=${MIN_DT:-0.01}
NSTEPS=${NSTEPS:-10}
OUT_EVERY=${OUT_EVERY:-10}
CSV_OUT_EVERY=${CSV_OUT_EVERY:-1}
ELASTIC=${ELASTIC:-0}
MIN_RMS_DPHI_THRESHOLD=${MIN_RMS_DPHI_THRESHOLD:-1e-5}
MIN_RMS_DY_THRESHOLD=${MIN_RMS_DY_THRESHOLD:-1e-5}

TEMP_C=${TEMP_C:-380.0}
VF_TARGET=${VF_TARGET:-0.04}
XB_OUT=${XB_OUT:-0.03}

# 示例：循环两个 radius
LOOP_VALUES=${LOOP_VALUES:-"5 6"}

# ------------------------------------------------------------
# C. 循环执行
# ------------------------------------------------------------
for LOOP_VALUE in ${LOOP_VALUES}; do
  CASE_TAG="template_case_${LOOP_VALUE}"
  PF_PARAM_FILE="$(site_generate_pf_param_file "${CASE_TAG}_T${TEMP_C}")"

  echo "====================================================="
  echo "运行 LOOP_VALUE=${LOOP_VALUE}"
  echo "CASE_TAG=${CASE_TAG}"
  echo "PF_PARAM=${PF_PARAM_FILE}"
  echo "====================================================="

  set -x
  ./main_cuda "${NX}" "${NY}" "${NZ}" "${MIN_DT}" "${NSTEPS}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" "${ELASTIC}" \
    --pf-param-file "${PF_PARAM_FILE}" \
    --mode=minimize \
    --minimize-full-model \
    --minimize-max-iter "${NSTEPS}" \
    --minimize-dt "${MIN_DT}" \
    --radius "${LOOP_VALUE}" \
    --ic-23d-xB-out "${XB_OUT}" \
    --init-case-tag "${CASE_TAG}" \
    --minimize-rms-dphi-threshold "${MIN_RMS_DPHI_THRESHOLD}" \
    --minimize-rms-dY-threshold "${MIN_RMS_DY_THRESHOLD}"
  set +x
done
