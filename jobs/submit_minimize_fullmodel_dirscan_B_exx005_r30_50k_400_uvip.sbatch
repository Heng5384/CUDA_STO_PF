#!/usr/bin/env bash
# 专用脚本：gpu_uvip 上提交 400^3、r=30、dirscan_B 起始核子 + x 轴 0.05 应变的 minimize full-model
#
# 用途：
# - 在 workstation 的 gpu_uvip 队列上提交 400x400x400 的 minimize full-model 单 case
# - 固定 radius=30，固定 B normal 方向的 oblate 初始核子
# - 固定外加应变：E0_xx=0.05，其余 E0 分量为 0
# - 默认物理与运行参数：
#   TEMP_C=400
#   XB_OUT=0.03
#   ELASTIC=1
#   NSTEPS=50000
#   axis-ratio=(1.25, 1.00, 0.60)
#
# B 方向：
#   n=( 0.744921649, -0.625009732, -0.233355034) -> theta=103.495, phi=320.002

#SBATCH --job-name=B_exx005_r30_50k
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
site_redirect_slurm_logs "B_exx005_r30_50k_400_uvip"
site_print_env_banner
site_build_main_cuda

NX=${NX:-400}
NY=${NY:-400}
NZ=${NZ:-400}

NSTEPS=${NSTEPS:-50000}
MIN_DT=${MIN_DT:-1e-1}
OUT_EVERY=${OUT_EVERY:-5000}
CSV_OUT_EVERY=${CSV_OUT_EVERY:-10}

MIN_RMS_DPHI_THRESHOLD=${MIN_RMS_DPHI_THRESHOLD:-2e-7}
MIN_RMS_DY_THRESHOLD=${MIN_RMS_DY_THRESHOLD:-2e-7}
MIN_ENERGY_DIFF_REL_THRESHOLD=${MIN_ENERGY_DIFF_REL_THRESHOLD:-1e-7}
MIN_RMS_RES_FOR_ENERGY_PLATEAU=${MIN_RMS_RES_FOR_ENERGY_PLATEAU:-1e-5}
MIN_RMS_RES_THRESHOLD=${MIN_RMS_RES_THRESHOLD:-2e-4}
MIN_VOL_ERR_REL_THRESHOLD=${MIN_VOL_ERR_REL_THRESHOLD:-5e-2}
MIN_CONVERGENCE_STEPS=${MIN_CONVERGENCE_STEPS:-100}
MIN_DT_SAFETY_LIMIT=${MIN_DT_SAFETY_LIMIT:-1e-6}
ETA_LAMBDA_VOL=${ETA_LAMBDA_VOL:-0.9}
POST_PROJ_ITERS=${POST_PROJ_ITERS:-1}
ELASTIC_ITER_MAX=${ELASTIC_ITER_MAX:-50}

V0=${V0:-0.0}
ELASTIC=${ELASTIC:-1}
XB_OUT=${XB_OUT:-0.03}
TEMP_C=${TEMP_C:-400.0}
RADIUS=${RADIUS:-30}
CASE_TAG=${CASE_TAG:-dirscan_B_exx005_r30}

OBLATE_RX=${OBLATE_RX:-1.25}
OBLATE_RY=${OBLATE_RY:-1.00}
OBLATE_RZ=${OBLATE_RZ:-0.60}
THETA=${THETA:-103.495}
PHI=${PHI:-320.002}

E0_XX=${E0_XX:-0.05}
E0_YY=${E0_YY:-0.0}
E0_ZZ=${E0_ZZ:-0.0}
E0_YZ=${E0_YZ:-0.0}
E0_XZ=${E0_XZ:-0.0}
E0_XY=${E0_XY:-0.0}

echo "====================================================="
echo "400^3 minimize full-model dirscan_B + x-axis strain"
echo "  Grid: ${NX} x ${NY} x ${NZ}"
echo "  Queue: gpu_uvip"
echo "  TEMP_C: ${TEMP_C}"
echo "  XB_OUT: ${XB_OUT}"
echo "  ELASTIC: ${ELASTIC}"
echo "  Radius: ${RADIUS}"
echo "  Axis ratio: rx=${OBLATE_RX}, ry=${OBLATE_RY}, rz=${OBLATE_RZ}"
echo "  theta/phi: ${THETA}/${PHI}"
echo "  Applied strain: E0_xx=${E0_XX}"
echo "  elastic_iter_max: ${ELASTIC_ITER_MAX}"
echo "  Case tag: ${CASE_TAG}"
echo "====================================================="

PF_PARAM_FILE="$(site_generate_pf_param_file "slurm_${CASE_TAG}_T${TEMP_C}")"
cat >> "${PF_PARAM_FILE}" <<EOF
E0_xx=${E0_XX}
E0_yy=${E0_YY}
E0_zz=${E0_ZZ}
E0_yz=${E0_YZ}
E0_xz=${E0_XZ}
E0_xy=${E0_XY}
elastic_iter_max=${ELASTIC_ITER_MAX}
EOF
echo "PF_PARAM=${PF_PARAM_FILE}"

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
  --init-shape ellipsoid
  --init-axis-ratio-rx "${OBLATE_RX}"
  --init-axis-ratio-ry "${OBLATE_RY}"
  --init-axis-ratio-rz "${OBLATE_RZ}"
  --init-tilt-theta-deg "${THETA}"
  --init-tilt-phi-deg "${PHI}"
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
