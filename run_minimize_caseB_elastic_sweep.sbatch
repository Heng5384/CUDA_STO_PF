#!/bin/bash
#SBATCH --job-name=cuda_minB_el_full_sweep
#SBATCH --partition=gpu_vip_24h
#SBATCH --qos=gpu_vip_24h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=logs/run_minimize_caseB_elastic_sweep_%j.out
#SBATCH --error=logs/run_minimize_caseB_elastic_sweep_%j.err

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/scripts/lib/site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"
site_print_env_banner
site_build_main_cuda

# 网格与输出
NX=512
NY=512
NZ=512
OUT_EVERY=5000
CSV_OUT_EVERY=10

# Minimization(full-model) 基本参数
NSTEPS=30000
MIN_DT=1e-1

# 收敛判据参数
MIN_RMS_DPHI_THRESHOLD=1e-5
MIN_RMS_DY_THRESHOLD=1e-5
MIN_ENERGY_DIFF_REL_THRESHOLD=1e-7
MIN_RMS_RES_FOR_ENERGY_PLATEAU=1e-4
MIN_RMS_RES_THRESHOLD=1e-4
MIN_VOL_ERR_REL_THRESHOLD=3e-2
MIN_CONVERGENCE_STEPS=100
MIN_DT_SAFETY_LIMIT=1e-6
ETA_LAMBDA_VOL=0.9

# 后投影修正子步数（0=关闭，对应 CSV 列 post_proj_iters_used）
POST_PROJ_ITERS=1

# 物理与初始条件参数（除半径外与单次 sbatch 相同）
V0=0.0
ELASTIC=1
XB_OUT=0.03
TEMP_C=380.0

# Sweep 设置：r 从 R_MIN 到 R_MAX，步长 R_STEP
R_MIN=${R_MIN:-18.0}
R_MAX=${R_MAX:-19.0}
R_STEP=${R_STEP:-1.0}

echo "Sweep 半径: R from ${R_MIN} to ${R_MAX} step ${R_STEP}"
echo ""

RADII=$(python3 - "$R_MIN" "$R_MAX" "$R_STEP" <<'PYEOF'
import sys
rmin, rmax, step = map(float, sys.argv[1:4])
vals = []
r = rmin
eps = 1e-12
while r <= rmax + eps:
    vals.append(f"{r:.6g}")
    r += step
print(" ".join(vals))
PYEOF
)

for RADIUS in ${RADII}; do
  echo "====================================================="
  echo "运行 radius = ${RADIUS}"
  echo "====================================================="

  echo ""
  echo "Minimization 参数配置（sweep 单个半径）："
  echo "  网格尺寸: ${NX} x ${NY} x ${NZ}"
  echo "  迭代步数: ${NSTEPS}"
  echo "  时间步长: ${MIN_DT}"
  echo ""
  echo "  收敛判据参数: rms_dphi=${MIN_RMS_DPHI_THRESHOLD}, rms_dY=${MIN_RMS_DY_THRESHOLD},"
  echo "                rel_dF=${MIN_ENERGY_DIFF_REL_THRESHOLD}, rms_res=${MIN_RMS_RES_THRESHOLD},"
  echo "                vol_err_rel=${MIN_VOL_ERR_REL_THRESHOLD}, steps=${MIN_CONVERGENCE_STEPS}"
  echo "  post_proj_iters: ${POST_PROJ_ITERS}"
  echo ""
  echo "  物理参数: V0=${V0}, radius=${RADIUS}, elastic=${ELASTIC}, xB_out=${XB_OUT}, T_C=${TEMP_C}"
  echo "====================================================="
  echo ""

  set -x
  ./main_cuda "${NX}" "${NY}" "${NZ}" "${MIN_DT}" "${NSTEPS}" "${OUT_EVERY}" "${CSV_OUT_EVERY}" "${ELASTIC}" \
    --mode=minimize \
    --minimize-full-model \
    --minimize-max-iter "${NSTEPS}" \
    --minimize-dt "${MIN_DT}" \
    --V0 "${V0}" \
    --radius "${RADIUS}" \
    --elastic "${ELASTIC}" \
    --ic-23d-xB-out "${XB_OUT}" \
    --temperature-C "${TEMP_C}" \
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

  echo "完成 radius=${RADIUS} 的模拟。"
  echo ""
done

echo "Sweep 全部完成。"
