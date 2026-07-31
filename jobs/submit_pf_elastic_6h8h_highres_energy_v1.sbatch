#!/usr/bin/env bash
# Two-physical-hour PF-only elastic audit: high-cadence VTK plus runtime gel_hat.

#SBATCH --job-name=pf_6h8h_elhr
#SBATCH --partition=gpu_vip_24h
#SBATCH --qos=gpu_vip_24h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=04:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/data/home/luozhiheng/tmp/codex_pf_dynamic_microstructure_audit_v1_20260730}"
OUT_ROOT="${OUT_ROOT:-/data/home/luozhiheng/tmp/pf_elastic_6h8h_highres_energy_v1_20260730}"
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

echo "RUNNING" >"${OUT_ROOT}/status.txt"
cd "${OUT_ROOT}"
"${OUT_ROOT}/main_cuda" 246 246 246 0.02 7266 256 256 1 \
  --pf-param-file "${OUT_ROOT}/pf_input.params" \
  --pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1 \
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 \
  --pf-zero-mode-tol-rel 1e-12 \
  --pf-zero-mode-max-iter 24 \
  --init-mode raw_fields \
  --init-phi-raw "${FIXTURE_ROOT}/phi.raw" \
  --init-xB-raw "${FIXTURE_ROOT}/xB_alpha.raw" \
  --init-meta "${FIXTURE_ROOT}/init_meta.json" \
  --pf-checkpoint-every 256 \
  --pf-checkpoint-path "${OUT_ROOT}/checkpoint_6h8h_highres.chk" \
  --enable-dynamics-mass-diagnostics \
  --dynamics-mass-diag-interval 256

CASE_ROOT="${OUT_ROOT}/Results/chel_T380_cuda_246x246x246_dt0.02_steps7266_xB0.030/raw_fields_relax"
test -s "${CASE_ROOT}/dynamics_mass_diagnostics.csv"
python3 "${PROJECT_ROOT}/scripts/extract_pf_runtime_elastic_scalar_v1.py" \
  --diagnostics "${CASE_ROOT}/dynamics_mass_diagnostics.csv" \
  --out-dir "${OUT_ROOT}/runtime_elastic_scalar" \
  --age-start-h 6 \
  --code-time-to-physical-s 49.546304767159207 \
  --elastic-energy-scale-j-m3 5.04e8

python3 - "${PROJECT_ROOT}" "${CASE_ROOT}" "${OUT_ROOT}/morphology_6h8h" <<'PY'
import pathlib
import subprocess
import sys

project = pathlib.Path(sys.argv[1])
case = pathlib.Path(sys.argv[2])
out = pathlib.Path(sys.argv[3])
factor = 49.546304767159207
dt = 0.02
specs = ["6=%s=%s" % (case / "phi_init.vtk", case / "xB_0.vtk")]
for step in range(256, 7169, 256):
    age = 6.0 + step * dt * factor / 3600.0
    specs.append("%.15g=%s=%s" % (age, case / ("phi_%d.vtk" % step), case / ("xB_%d.vtk" % step)))
final_index = len(range(256, 7169, 256)) + 1
specs.append("%.15g=%s=%s" % (
    6.0 + 7266 * dt * factor / 3600.0,
    case / ("phi_%d.vtk" % final_index),
    case / ("xB_%d.vtk" % final_index),
))
cmd = [
    "python3", str(project / "scripts/audit_pf_high_time_morphology_6_8h_v1.py"),
    "--snapshots",
] + specs + [
    "--out-dir", str(out), "--grid", "246", "--dx-nm", "1",
    "--particle-h-threshold", "1e-4",
]
subprocess.check_call(cmd)
PY

grep -qx "PASS_RUNTIME_ELASTIC_SCALAR_EXTRACTED" \
  "${OUT_ROOT}/runtime_elastic_scalar/status.txt"
grep -qx "PASS_HIGH_TIME_MORPHOLOGY_OBSERVATION_V1" \
  "${OUT_ROOT}/morphology_6h8h/status.txt"
echo "PASS_PF_ELASTIC_6H8H_HIGHRES_AND_RUNTIME_ENERGY_V1" >"${OUT_ROOT}/status.txt"
