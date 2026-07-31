#!/usr/bin/env bash
# CPU-only post-processing of existing 6--8 h and 14--17 h PF fields.

#SBATCH --job-name=pf_shape_neck
#SBATCH --partition=parallel_pub_4core_3h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/home/luozhiheng/tmp/codex_pf_dynamic_microstructure_audit_v1_20260730}"
SHAPE_OUT="${SHAPE_OUT:-/data/home/luozhiheng/tmp/pf_shape_tensor_6h8h_v1_20260730}"
NECK_OUT="${NECK_OUT:-/data/home/luozhiheng/tmp/pf_merge_neck_persistence_v1_20260730}"
HIGHRES_ROOT="${HIGHRES_ROOT:-/data/home/luozhiheng/tmp/pf_elastic_6h8h_highres_energy_v1_20260730}"
MERGE_ROOT="${MERGE_ROOT:-/data/home/luozhiheng/tmp/pf_random_psd_v2_246x96_elastic_12h48h_20260730/Results/chel_T380_cuda_246x246x246_dt0.02_steps152600_xB0.030}"
MERGE_ROWS="${MERGE_ROWS:-/data/home/luozhiheng/tmp/pf_merge_aware_14h17h_v3_20260730/window_particle_rows.csv}"

if [[ -e "${SHAPE_OUT}" || -e "${NECK_OUT}" ]]; then
  echo "[fatal] refusing to overwrite analysis roots" >&2
  exit 2
fi
mkdir -p "${SHAPE_OUT}" "${NECK_OUT}"
exec >"${SHAPE_OUT}/driver.stdout.log" 2>"${SHAPE_OUT}/driver.stderr.log"
trap 'rc=$?; if [[ "$rc" -ne 0 ]]; then echo "FAILED_EXIT_${rc}" >"${SHAPE_OUT}/status.txt"; fi' EXIT

python3 "${PROJECT_ROOT}/scripts/audit_pf_particle_shape_tensor_6_8h_v1.py" \
  --snapshot-csv "${HIGHRES_ROOT}/morphology_6h8h/morphology_6_8h_time_series.csv" \
  --out-dir "${SHAPE_OUT}" \
  --grid 246 --dx-nm 1 --particle-h-threshold 1e-4

python3 "${PROJECT_ROOT}/scripts/audit_pf_merge_neck_persistence_v1.py" \
  --snapshots \
  "14=${MERGE_ROOT}/phi_29064.vtk" \
  "15=${MERGE_ROOT}/phi_32697.vtk" \
  "16=${MERGE_ROOT}/phi_36330.vtk" \
  "17=${MERGE_ROOT}/phi_39963.vtk" \
  --particle-rows "${MERGE_ROWS}" \
  --parent-ids 46 68 \
  --anchor-age-h 15 \
  --thresholds 1e-4 5e-4 1e-3 5e-3 1e-2 5e-2 1e-1 2.5e-1 5e-1 7.5e-1 \
  --anchor-search-nm 12 \
  --grid 246 --dx-nm 1 \
  --out-dir "${NECK_OUT}"

grep -qx "PASS_PARTICLE_SHAPE_TENSOR_6_8H_V1" "${SHAPE_OUT}/status.txt"
grep -qE "^(PASS_MULTI_THRESHOLD_NECK_PERSISTS_16H_17H|TREND_ONLY_DIFFUSE_TAIL_NECK)$" \
  "${NECK_OUT}/status.txt"
sha256sum \
  "${PROJECT_ROOT}/scripts/audit_pf_particle_shape_tensor_6_8h_v1.py" \
  "${PROJECT_ROOT}/scripts/audit_pf_merge_neck_persistence_v1.py" \
  "${SHAPE_OUT}/particle_shape_tensor_trajectories.csv" \
  "${NECK_OUT}/neck_threshold_matrix.csv" \
  >"${SHAPE_OUT}/analysis_hashes.sha256"
echo "PASS_PF_SHAPE_AND_NECK_POSTPROCESS_V1" >"${SHAPE_OUT}/combined_status.txt"
