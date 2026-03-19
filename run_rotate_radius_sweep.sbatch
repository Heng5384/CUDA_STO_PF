#!/bin/bash
#SBATCH --job-name=cuda_rotate_r5_10
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=logs/run_rotate_radius_sweep_%j.out
#SBATCH --error=logs/run_rotate_radius_sweep_%j.err

set -euo pipefail

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
source "${SCRIPT_DIR}/scripts/lib/site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"
site_print_env_banner
site_build_main_cuda

# ============================================================
# 网格与输出：沿用 run_py_rotate.sh
# ============================================================
NX=${NX:-400}
NY=${NY:-400}
NZ=${NZ:-400}
OUT_EVERY=${OUT_EVERY:-5000}
CSV_OUT_EVERY=${CSV_OUT_EVERY:-10}

# ============================================================
# Minimization 参数：沿用 run_py_rotate.sh
# ============================================================
NSTEPS=${NSTEPS:-30000}
MIN_DT=${MIN_DT:-1e-1}
MIN_RMS_DPHI_THRESHOLD=${MIN_RMS_DPHI_THRESHOLD:-5e-5}
MIN_RMS_DY_THRESHOLD=${MIN_RMS_DY_THRESHOLD:-1e-5}
MIN_ENERGY_DIFF_REL_THRESHOLD=${MIN_ENERGY_DIFF_REL_THRESHOLD:-1e-8}
MIN_RMS_RES_FOR_ENERGY_PLATEAU=${MIN_RMS_RES_FOR_ENERGY_PLATEAU:-1e-4}
MIN_RMS_RES_THRESHOLD=${MIN_RMS_RES_THRESHOLD:-1e-4}
MIN_VOL_ERR_REL_THRESHOLD=${MIN_VOL_ERR_REL_THRESHOLD:-5e-3}
MIN_CONVERGENCE_STEPS=${MIN_CONVERGENCE_STEPS:-100}
MIN_DT_SAFETY_LIMIT=${MIN_DT_SAFETY_LIMIT:-1e-6}
ETA_LAMBDA_VOL=${ETA_LAMBDA_VOL:-0.9}
POST_PROJ_ITERS=${POST_PROJ_ITERS:-1}

# ============================================================
# 物理参数：沿用 run_py_rotate.sh
# ============================================================
V0=${V0:-0.0}
ELASTIC=${ELASTIC:-1}
XB_OUT=${XB_OUT:-0.03}
TEMP_C=${TEMP_C:-380.0}

# ============================================================
# rotate case：沿用 run_py_rotate.sh 的预测方向和扁椭球比例
# ============================================================
PRED_THETA=${PRED_THETA:-101}
PRED_PHI=${PRED_PHI:-39}
OBLATE_RX=${OBLATE_RX:-1.25}
OBLATE_RY=${OBLATE_RY:-1.25}
OBLATE_RZ=${OBLATE_RZ:-0.70}

# ============================================================
# 半径扫描：5 到 10（含端点）
# 如需覆盖，可在提交前设置：R_LIST="5 6 7 8 9 10"
# ============================================================
R_LIST=${R_LIST:-"5 6 7 8 9 10"}

echo ""
echo "====================================================="
echo "Rotate + Sphere 半径扫描开始"
echo "  Grid: ${NX} x ${NY} x ${NZ}"
echo "  Radius list: ${R_LIST}"
echo "  Rotate theta/phi: ${PRED_THETA}/${PRED_PHI}"
echo "  Rotate axis ratio: rx=${OBLATE_RX}, ry=${OBLATE_RY}, rz=${OBLATE_RZ}"
echo "====================================================="
echo ""

run_case() {
  local radius="$1"
  local shape_mode="$2"
  local case_tag="$3"

  echo "-----------------------------------------------------"
  echo "运行 case_tag=${case_tag}"
  echo "  radius=${radius}, shape=${shape_mode}"
  echo "-----------------------------------------------------"

  if [[ "$shape_mode" == "rotate_oblate" ]]; then
    set -x
    ./main_cuda "${NX}" "${NY}" "${NZ}" "${MIN_DT}" "${NSTEPS}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" "${ELASTIC}" \
      --mode=minimize \
      --minimize-full-model \
      --minimize-max-iter "${NSTEPS}" \
      --minimize-dt "${MIN_DT}" \
      --V0 "${V0}" \
      --radius "${radius}" \
      --elastic "${ELASTIC}" \
      --ic-23d-xB-out "${XB_OUT}" \
      --temperature-C "${TEMP_C}" \
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
    set -x
    ./main_cuda "${NX}" "${NY}" "${NZ}" "${MIN_DT}" "${NSTEPS}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" "${ELASTIC}" \
      --mode=minimize \
      --minimize-full-model \
      --minimize-max-iter "${NSTEPS}" \
      --minimize-dt "${MIN_DT}" \
      --V0 "${V0}" \
      --radius "${radius}" \
      --elastic "${ELASTIC}" \
      --ic-23d-xB-out "${XB_OUT}" \
      --temperature-C "${TEMP_C}" \
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

for RADIUS in ${R_LIST}; do
  run_case "${RADIUS}" rotate_oblate "rotate_oblate_r${RADIUS}_pred_256"
  run_case "${RADIUS}" sphere        "sphere_r${RADIUS}_256"
done

echo "====================================================="
echo "全部 radius + shape 扫描完成"
echo "====================================================="
