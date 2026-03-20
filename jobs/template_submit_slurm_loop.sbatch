#!/usr/bin/env bash
# 标准模板：Slurm 循环/扫描脚本
#
# 用途：
# - 适合单个 sbatch 内循环多个 radius / shape / init case
# - 适合复制后改成新的参数扫描脚本

#SBATCH --job-name=template_loop
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=jobs/logs/template_loop_%j.out
#SBATCH --error=jobs/logs/template_loop_%j.err

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

# 示例：循环半径列表
LOOP_VALUES=${LOOP_VALUES:-"5 6"}

# ------------------------------------------------------------
# C. 循环执行
# ------------------------------------------------------------
for LOOP_VALUE in ${LOOP_VALUES}; do
  CASE_TAG="template_loop_${LOOP_VALUE}"
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
