#!/usr/bin/env bash
# Serial continue-dynamic job: read a single guide CSV and run all continue cases sequentially.
#
# Required env:
#   GUIDE_CSV   path to guide_continue_dynamic.csv

#SBATCH --job-name=continue_dyn_serial
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

MANIFEST_NAME=${MANIFEST_NAME:-continue_dyn_serial}

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

continue_case_complete() {
  local run_root="${PROJECT_ROOT}/${output_root_rel}"
  local continue_dir="${run_root}/continue_dyn_1"
  local final_phi="${continue_dir}/phi_${nsteps}.vtk"
  local final_xb="${continue_dir}/xB_${nsteps}.vtk"
  local summary_txt="${continue_dir}/summary.txt"

  if [[ -f "${summary_txt}" ]]; then
    return 0
  fi
  if [[ -f "${final_phi}" && -f "${final_xb}" ]]; then
    return 0
  fi
  return 1
}

generate_continue_pf_param_file() {
  local row_idx="$1"
  local source_summary_path="${continue_source_summary_path}"
  local source_dir
  local source_pf
  local safe_tag
  local out_pf

  source_dir="$(dirname "${source_summary_path}")"
  source_pf="${source_dir}/pf_input.params"
  if [[ ! -f "${source_pf}" ]]; then
    echo "[fatal] source pf_input.params not found: ${source_pf}" >&2
    return 2
  fi

  safe_tag="$(site_make_safe_tag "${MANIFEST_NAME}_${row_idx}_${base_case_tag}")"
  out_pf="${PROJECT_ROOT}/jobs/generated/pf_params_${safe_tag}.params"
  python3 - "${source_pf}" "${out_pf}" "${dt}" <<'PY'
import sys
from pathlib import Path

source_pf = Path(sys.argv[1])
out_pf = Path(sys.argv[2])
dt_value = float(sys.argv[3])

lines = source_pf.read_text(encoding="utf-8").splitlines()
updated = []
replaced = False
for line in lines:
    if line.startswith("dt="):
        updated.append(f"dt={dt_value:.16e}")
        replaced = True
    else:
        updated.append(line)
if not replaced:
    updated.append(f"dt={dt_value:.16e}")
out_pf.write_text("\n".join(updated) + "\n", encoding="utf-8")
PY
  printf '%s\n' "${out_pf}"
}

run_case() {
  local row_idx="$1"
  local exports_block="$2"
  eval "${exports_block}"

  if continue_case_complete; then
    echo "[skip] row=${row_idx} base=${base_case_tag} continue_dyn_1 already complete"
    return 0
  fi

  export CUDA_STO_RESULTS_ROOT="${PROJECT_ROOT}/${raw_results_root_rel}"
  mkdir -p "${CUDA_STO_RESULTS_ROOT}"
  PF_PARAM_FILE="$(generate_continue_pf_param_file "${row_idx}")"
  echo "[info] row=${row_idx} base=${base_case_tag} start_radius_nm=${start_radius_nm} source_radius_nm=${source_radius_nm}"

  CMD=(
    ./main_cuda "${nx}" "${ny}" "${nz}" "${dt}" "${nsteps}" "${out_every}" "${csv_out_every}" "${elastic}"
    --pf-param-file "${PF_PARAM_FILE}"
    --mode=dynamics-continue
    --radius-phys-nm "${start_radius_nm}"
    --elastic "${elastic}"
    --ic-23d-xB-out "${xB_out}"
    --continue-phi-vtk "${continue_phi_vtk}"
    --continue-xB-vtk "${continue_xb_vtk}"
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
