#!/usr/bin/env bash
# Short 246^3 continuous/restart audit for PF-only elastic diagnostics.

#SBATCH --job-name=pf_el_diag_rst
#SBATCH --partition=gpu_vip_24h
#SBATCH --qos=gpu_vip_24h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/home/luozhiheng/tmp/codex_pf_dynamic_microstructure_audit_v1_20260730}"
OUT_ROOT="${OUT_ROOT:-/data/home/luozhiheng/tmp/pf_elastic_diag_restart_v1_20260730}"
PARAM_SOURCE="${PARAM_SOURCE:-/data/home/luozhiheng/tmp/pf_random_psd_v2_preflight_246x96_6h12h_20260730/pf_input.params}"
FIXTURE_ROOT="${FIXTURE_ROOT:-/data/home/luozhiheng/tmp/codex_pf_random_psd_v2_246x96_6b69895_20260730/data/qualification/pf_6h_48h_random_psd_v2_246x96}"

if [[ -e "${OUT_ROOT}" ]]; then
  echo "[fatal] refusing to overwrite output root: ${OUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUT_ROOT}"
exec >"${OUT_ROOT}/driver.stdout.log" 2>"${OUT_ROOT}/driver.stderr.log"
trap 'rc=$?; if [[ "$rc" -ne 0 ]]; then echo "FAILED_EXIT_${rc}" >"${OUT_ROOT}/status.txt"; fi' EXIT

cd "${PROJECT_ROOT}"
source jobs/_site_env.sh
site_prepare_cuda_env
echo "BUILDING" >"${OUT_ROOT}/status.txt"
make main_cuda CUDA_ARCH="${CUDA_ARCH:-sm_80}" CUDA_ROOT="${CUDA_ROOT}"
make test_pf_zero_mode_checkpoint CXX=c++
cp main_cuda "${OUT_ROOT}/main_cuda"
cp "${PARAM_SOURCE}" "${OUT_ROOT}/pf_input.params"
sha256sum main_cuda main_cuda.cu cuda_kernels.cu \
  pf_zero_mode_checkpoint.cpp pf_zero_mode_checkpoint.h \
  "${OUT_ROOT}/pf_input.params" \
  "${FIXTURE_ROOT}/phi.raw" "${FIXTURE_ROOT}/xB_alpha.raw" \
  "${FIXTURE_ROOT}/init_meta.json" >"${OUT_ROOT}/input_hashes.sha256"

COMMON_FLAGS=(
  --pf-param-file "${OUT_ROOT}/pf_input.params"
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1
  --pf-zero-mode-tol-rel 1e-12
  --pf-zero-mode-max-iter 24
  --enable-dynamics-mass-diagnostics
  --dynamics-mass-diag-interval 64
)
RAW_FLAGS=(
  --init-mode raw_fields
  --init-phi-raw "${FIXTURE_ROOT}/phi.raw"
  --init-xB-raw "${FIXTURE_ROOT}/xB_alpha.raw"
  --init-meta "${FIXTURE_ROOT}/init_meta.json"
)

run_case() {
  local label="$1"
  local final_step="$2"
  local output_interval="$3"
  shift 3
  local results="${OUT_ROOT}/results/${label}"
  mkdir -p "${results}"
  CUDA_STO_RESULTS_ROOT="${results}" \
    "${OUT_ROOT}/main_cuda" 246 246 246 0.02 "${final_step}" "${output_interval}" 64 1 \
      "${COMMON_FLAGS[@]}" "$@" \
      >"${OUT_ROOT}/${label}.stdout.log" 2>"${OUT_ROOT}/${label}.stderr.log"
  grep -q "PF_ZERO_MODE_FINAL_AUDIT status=PASS" "${OUT_ROOT}/${label}.stdout.log"
}

echo "RUNNING" >"${OUT_ROOT}/status.txt"
run_case continuous 512 512 \
  --pf-checkpoint-every 256 \
  --pf-checkpoint-path "${OUT_ROOT}/continuous_final.chk" \
  --init-case-tag elastic_diag_continuous \
  "${RAW_FLAGS[@]}"

run_case first_half 256 256 \
  --pf-checkpoint-every 256 \
  --pf-checkpoint-path "${OUT_ROOT}/half.chk" \
  --init-case-tag elastic_diag_first_half \
  "${RAW_FLAGS[@]}"

run_case restart 512 512 \
  --pf-restart-from "${OUT_ROOT}/half.chk" \
  --pf-checkpoint-every 256 \
  --pf-checkpoint-path "${OUT_ROOT}/restart_final.chk" \
  --init-case-tag elastic_diag_restart

python3 "${PROJECT_ROOT}/scripts/analyze_pf_elastic_diag_restart_v1.py" \
  --continuous-root "${OUT_ROOT}/results/continuous" \
  --first-half-root "${OUT_ROOT}/results/first_half" \
  --restart-root "${OUT_ROOT}/results/restart" \
  --continuous-checkpoint "${OUT_ROOT}/continuous_final.chk" \
  --restart-checkpoint "${OUT_ROOT}/restart_final.chk" \
  --out-dir "${OUT_ROOT}/restart_validation" \
  --final-step 512

grep -qx "PASS_PF_ELASTIC_DIAGNOSTIC_AND_FIELD_RESTART_V1" \
  "${OUT_ROOT}/restart_validation/status.txt"
echo "PASS_PF_ELASTIC_DIAGNOSTIC_AND_FIELD_RESTART_V1" >"${OUT_ROOT}/status.txt"
