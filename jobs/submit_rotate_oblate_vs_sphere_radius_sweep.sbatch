#!/usr/bin/env bash
# 标准化脚本：Slurm 多形状半径扫描模板实例
#
# 用途：
# - 在 Slurm 上比较 rotate-oblate 与 sphere 两类初始核
# - 并对半径列表做批量扫描
#
# 可复制修改：
# - 改 #SBATCH 资源区
# - 改默认参数区
# - 改 run_case() 中的 shape 分支

#SBATCH --job-name=rotate_vs_sphere_sweep
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=jobs/logs/rotate_vs_sphere_radius_sweep_%j.out
#SBATCH --error=jobs/logs/rotate_vs_sphere_radius_sweep_%j.err

set -euo pipefail

# ------------------------------------------------------------
# A. 启动与项目定位
# ------------------------------------------------------------
if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_site_env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/jobs"
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
NX=${NX:-400}
NY=${NY:-400}
NZ=${NZ:-400}
OUT_EVERY=${OUT_EVERY:-5000}
CSV_OUT_EVERY=${CSV_OUT_EVERY:-10}

NSTEPS=${NSTEPS:-30000}
MIN_DT=${MIN_DT:-1e-1}
MIN_RMS_DPHI_THRESHOLD=${MIN_RMS_DPHI_THRESHOLD:-1e-5}
MIN_RMS_DY_THRESHOLD=${MIN_RMS_DY_THRESHOLD:-1e-5}
MIN_ENERGY_DIFF_REL_THRESHOLD=${MIN_ENERGY_DIFF_REL_THRESHOLD:-1e-8}
MIN_RMS_RES_FOR_ENERGY_PLATEAU=${MIN_RMS_RES_FOR_ENERGY_PLATEAU:-1e-4}
MIN_RMS_RES_THRESHOLD=${MIN_RMS_RES_THRESHOLD:-1e-4}
MIN_VOL_ERR_REL_THRESHOLD=${MIN_VOL_ERR_REL_THRESHOLD:-5e-3}
MIN_CONVERGENCE_STEPS=${MIN_CONVERGENCE_STEPS:-100}
MIN_DT_SAFETY_LIMIT=${MIN_DT_SAFETY_LIMIT:-1e-6}
ETA_LAMBDA_VOL=${ETA_LAMBDA_VOL:-0.9}
POST_PROJ_ITERS=${POST_PROJ_ITERS:-1}

V0=${V0:-0.0}
ELASTIC=${ELASTIC:-1}
XB_OUT=${XB_OUT:-0.03}
TEMP_C=${TEMP_C:-380.0}

PRED_THETA=${PRED_THETA:-101}
PRED_PHI=${PRED_PHI:-39}
OBLATE_RX=${OBLATE_RX:-1.25}
OBLATE_RY=${OBLATE_RY:-1.25}
OBLATE_RZ=${OBLATE_RZ:-0.70}

R_LIST=${R_LIST:-"5 6 7 8 9 10"}

echo ""
echo "====================================================="
echo "Rotate-oblate vs sphere 半径扫描开始"
echo "  Grid: ${NX} x ${NY} x ${NZ}"
echo "  Radius list: ${R_LIST}"
echo "  Rotate theta/phi: ${PRED_THETA}/${PRED_PHI}"
echo "  Rotate axis ratio: rx=${OBLATE_RX}, ry=${OBLATE_RY}, rz=${OBLATE_RZ}"
echo "====================================================="
echo ""

# ------------------------------------------------------------
# C. 单 case 执行函数
# ------------------------------------------------------------
run_case() {
  local radius="$1"
  local shape_mode="$2"
  local case_tag="$3"

  echo "-----------------------------------------------------"
  echo "运行 case_tag=${case_tag}"
  echo "  radius=${radius}, shape=${shape_mode}"
  echo "-----------------------------------------------------"

  PF_PARAM_FILE="$(site_generate_pf_param_file "slurm_${case_tag}_T${TEMP_C}")"
  echo "PF_PARAM = ${PF_PARAM_FILE}"

  if [[ "$shape_mode" == "rotate_oblate" ]]; then
    # --------------------------------------------------------
    # D1. rotate_oblate 分支
    # --------------------------------------------------------
    set -x
    ./main_cuda "${NX}" "${NY}" "${NZ}" "${MIN_DT}" "${NSTEPS}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" "${ELASTIC}" \
      --pf-param-file "${PF_PARAM_FILE}" \
      --mode=minimize \
      --minimize-full-model \
      --minimize-max-iter "${NSTEPS}" \
      --minimize-dt "${MIN_DT}" \
      --V0 "${V0}" \
      --radius "${radius}" \
      --elastic "${ELASTIC}" \
      --ic-23d-xB-out "${XB_OUT}" \
      --init-shape ellipsoid \
      --init-axis-ratio-rx "${OBLATE_RX}" \
      --init-axis-ratio-ry "${OBLATE_RY}" \
      --init-axis-ratio-rz "${OBLATE_RZ}" \
      --init-tilt-theta-deg "${PRED_THETA}" \
      --init-tilt-phi-deg "${PRED_PHI}" \
      --init-case-tag "${case_tag}" \
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
  elif [[ "$shape_mode" == "sphere" ]]; then
    # --------------------------------------------------------
    # D2. sphere 分支
    # --------------------------------------------------------
    set -x
    ./main_cuda "${NX}" "${NY}" "${NZ}" "${MIN_DT}" "${NSTEPS}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" "${ELASTIC}" \
      --pf-param-file "${PF_PARAM_FILE}" \
      --mode=minimize \
      --minimize-full-model \
      --minimize-max-iter "${NSTEPS}" \
      --minimize-dt "${MIN_DT}" \
      --V0 "${V0}" \
      --radius "${radius}" \
      --elastic "${ELASTIC}" \
      --ic-23d-xB-out "${XB_OUT}" \
      --init-shape sphere \
      --init-case-tag "${case_tag}" \
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
  else
    echo "[fatal] unknown shape_mode=${shape_mode}" >&2
    return 2
  fi

  echo "完成 case_tag=${case_tag}"
  echo ""
}

# ------------------------------------------------------------
# E. 循环执行
# ------------------------------------------------------------
for RADIUS in ${R_LIST}; do
  run_case "${RADIUS}" rotate_oblate "rotate_oblate_r${RADIUS}"
  run_case "${RADIUS}" sphere "sphere_r${RADIUS}"
done

echo "====================================================="
echo "全部 radius + shape 扫描完成"
echo "====================================================="
