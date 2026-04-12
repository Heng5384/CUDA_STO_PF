#!/usr/bin/env bash
# Serial CNT scan job: read a single guide CSV and run all scan points sequentially.
#
# Required env:
#   GUIDE_CSV   path to guide_cnt_scan.csv
#
# Optional env overrides:
#   MANIFEST_NAME / JOB_TAG

#SBATCH --job-name=cnt_scan_serial
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

if [[ -z "${GUIDE_CSV:-}" ]]; then
  echo "[fatal] GUIDE_CSV is required." >&2
  exit 2
fi

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

MANIFEST_NAME=${MANIFEST_NAME:-cnt_scan_serial}

site_redirect_slurm_logs "${MANIFEST_NAME}"
site_print_env_banner
site_build_main_cuda

guide_rows="$(
python3 - "${GUIDE_CSV}" <<'PY'
import csv
import sys
from pathlib import Path

guide = Path(sys.argv[1]).expanduser().resolve()
with guide.open(newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))

for idx, row in enumerate(rows):
    print(f"__ROW_BEGIN__ {idx}")
    for key, value in row.items():
        if value is None or value == "":
            continue
        print(f"{key}={value}")
    print("__ROW_END__")
PY
)"

run_case() {
  local row_idx="$1"
  local exports_block="$2"
  eval "${exports_block}"

  export PHYSICAL_INPUT_JSON="${physical_input_json_rel}"
  export DT="${dt}"
  export CUDA_STO_RESULTS_ROOT="${PROJECT_ROOT}/${raw_results_root_rel}"
  mkdir -p "${CUDA_STO_RESULTS_ROOT}"
  PF_PARAM_FILE="$(site_generate_pf_param_file "${case_tag}")"
  echo "[info] row=${row_idx} case=${case_tag} radius_nm=${radius_nm} pf_param=${PF_PARAM_FILE}"

  CMD=(
    ./main_cuda "${nx}" "${ny}" "${nz}" "${dt}" "${nsteps}" "${out_every}" "${csv_out_every}" "${elastic}"
    --pf-param-file "${PF_PARAM_FILE}"
    --mode=minimize
    --minimize-full-model
    --minimize-max-iter "${nsteps}"
    --minimize-dt "${dt}"
    --radius-phys-nm "${radius_nm}"
    --elastic "${elastic}"
    --ic-23d-xB-out "${xB_out}"
    --init-case-tag "${case_tag}"
    --minimize-rms-dphi-threshold "1e-5"
    --minimize-rms-dY-threshold "1e-5"
    --minimize-energy-diff-rel-threshold "1e-7"
    --minimize-rms-res-for-energy-plateau "1e-5"
    --minimize-convergence-steps "100"
    --minimize-dt-safety-limit "1e-6"
    --minimize-rms-res-threshold "1e-4"
    --minimize-vol-err-rel-threshold "3e-2"
    --eta-lambda-vol "0.9"
    --minimize-post-projection-iters "1"
  )
  set -x
  "${CMD[@]}"
  set +x
}

current_idx=""
current_block=""
while IFS= read -r line; do
  if [[ "${line}" == __ROW_BEGIN__* ]]; then
    current_idx="${line#* }"
    current_block=""
  elif [[ "${line}" == __ROW_END__ ]]; then
    echo "[info] starting row ${current_idx}"
    run_case "${current_idx}" "${current_block}"
    current_block=""
    current_idx=""
  else
    if [[ -n "${current_block}" ]]; then
      current_block+=$'\n'
    fi
    current_block+="export ${line}"
  fi
done <<< "${guide_rows}"
