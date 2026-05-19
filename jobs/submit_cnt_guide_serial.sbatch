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

pick_existing_result() {
  local preferred="$1"
  local glob_pattern="$2"
  local matches=()
  if [[ -f "${preferred}" ]]; then
    printf '%s\n' "${preferred}"
    return 0
  fi
  shopt -s nullglob
  matches=(${glob_pattern})
  shopt -u nullglob
  if (( ${#matches[@]} > 0 )); then
    printf '%s\n' "${matches[0]}"
    return 0
  fi
  return 1
}

cnt_case_complete() {
  local case_dir="${PROJECT_ROOT}/${case_dir_rel}"
  local energy_csv=""
  local phi_final=""

  energy_csv="$(pick_existing_result "${PROJECT_ROOT}/${energy_csv_rel}" "${case_dir}"/energy_minimize_*.csv || true)"
  phi_final="$(pick_existing_result "${PROJECT_ROOT}/${phi_final_rel}" "${case_dir}"/phi_final_*.vtk || true)"

  [[ -n "${energy_csv}" && -f "${energy_csv}" && -n "${phi_final}" && -f "${phi_final}" ]]
}

write_reference_energy_csv() {
  local case_dir="${PROJECT_ROOT}/${case_dir_rel}"
  local energy_csv=""
  local reference_csv="${PROJECT_ROOT}/${reference_energy_rel}"
  energy_csv="$(pick_existing_result "${PROJECT_ROOT}/${energy_csv_rel}" "${case_dir}"/energy_minimize_*.csv || true)"
  if [[ -z "${energy_csv}" ]]; then
    echo "[warn] reference energy csv missing in ${case_dir}" >&2
    return 1
  fi
  python3 - "${energy_csv}" "${reference_csv}" "${base_case_tag}" "${case_dir}" "${T_C}" "${xB_out}" "${mode}" "${strain}" \
    "${E0_xx}" "${E0_yy}" "${E0_zz}" "${E0_xy}" "${E0_xz}" "${E0_yz}" "${nx}" "${ny}" "${nz}" "${pf_dx_nm}" "${lambda_sm_nm}" <<'PY'
import csv, json, math, sys
from pathlib import Path

energy_csv = Path(sys.argv[1])
out_csv = Path(sys.argv[2])
base_case_tag = sys.argv[3]
case_dir = sys.argv[4]
T_C = float(sys.argv[5])
xB_out = float(sys.argv[6])
mode = sys.argv[7]
strain = float(sys.argv[8])
E0_xx, E0_yy, E0_zz, E0_xy, E0_xz, E0_yz = map(float, sys.argv[9:15])
nx, ny, nz = map(int, sys.argv[15:18])
dx_nm = float(sys.argv[18])
lambda_sm_nm = float(sys.argv[19])

rows = list(csv.DictReader(energy_csv.open(newline="", encoding="utf-8")))
last = rows[-1] if rows else {}
gamma = float("nan")
w_phys = float("nan")
phys_json = energy_csv.parents[3] / "input" / "physical_inputs.json"
if phys_json.exists():
    payload = json.loads(phys_json.read_text(encoding="utf-8"))
    gamma = float(payload.get("gamma", float("nan")))
    lambda_sm = float(payload.get("lambda_sm", float("nan")))
    if math.isfinite(gamma) and math.isfinite(lambda_sm) and lambda_sm > 0.0:
        w_phys = 12.0 * gamma / lambda_sm
v_box = nx * ny * nz * (dx_nm * 1.0e-9) ** 3
f_surf = float(last.get("F_surf_hat", "nan"))
f_chem = float(last.get("F_chem_excess_hat", "nan"))
f_el = float(last.get("F_el_hat", "nan"))
f_total = float(last.get("F_total_excess_hat", "nan"))
mean_h = float(last.get("mean_h", "nan"))
record = {
    "base_case_tag": base_case_tag,
    "case_dir": case_dir,
    "reference_type": "matrix_only_same_strain",
    "T_input": T_C,
    "T_K": T_C + 273.15,
    "xB0": xB_out,
    "xB_out": xB_out,
    "mode": mode,
    "exx": E0_xx,
    "eyy": E0_yy,
    "ezz": E0_zz,
    "exy": E0_xy,
    "exz": E0_xz,
    "eyz": E0_yz,
    "Nx": nx,
    "Ny": ny,
    "Nz": nz,
    "dx_nm": dx_nm,
    "dy_nm": dx_nm,
    "dz_nm": dx_nm,
    "V_box_m3": v_box,
    "gamma_Jm2": gamma,
    "lambda_sm_m": lambda_sm_nm * 1.0e-9,
    "w_phys_J_m3": w_phys,
    "F_surf_ref_hat": f_surf,
    "F_chem_ref_hat": f_chem,
    "F_el_ref_hat": f_el,
    "F_total_ref_hat": f_total,
    "F_total_ref_J": (f_total * w_phys * v_box) if math.isfinite(f_total) and math.isfinite(w_phys) else float("nan"),
    "mean_phi": 0.0,
    "mean_h": mean_h,
    "mean_xB": xB_out,
    "voxel_count": 0,
    "status": "complete" if rows else "missing_energy",
    "warnings": "" if abs(mean_h) < 1.0e-8 else "mean_h_not_close_to_zero",
}
out_csv.parent.mkdir(parents=True, exist_ok=True)
with out_csv.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(record.keys()))
    writer.writeheader()
    writer.writerow(record)
PY
}

run_case() {
  local row_idx="$1"
  local exports_block="$2"
  eval "${exports_block}"

  if cnt_case_complete; then
    echo "[skip] row=${row_idx} case=${case_tag} already complete"
    return 0
  fi

  export PHYSICAL_INPUT_JSON="${physical_input_json_rel}"
  export DT="${dt}"
  export CUDA_STO_RESULTS_ROOT="${PROJECT_ROOT}/${raw_results_root_rel}"
  mkdir -p "${CUDA_STO_RESULTS_ROOT}"
  PF_PARAM_FILE="$(site_generate_pf_param_file "${case_tag}")"
  echo "[info] row=${row_idx} case=${case_tag} radius_nm=${radius_nm} pf_param=${PF_PARAM_FILE}"

  local row_type_local="${row_type:-scan_point}"
  local is_reference_case="0"
  local minimize_max_iter_local="${nsteps}"
  local minimize_convergence_steps_local="100"
  if [[ "${row_type_local}" == "reference" || "${case_tag}" == *_ref_matrix_only ]]; then
    is_reference_case="1"
    local raw_init_dir="${PROJECT_ROOT}/${raw_init_dir_rel}"
    python3 "${PROJECT_ROOT}/tools/analysis/write_uniform_matrix_raw_init.py" \
      --out-dir "${raw_init_dir}" \
      --nx "${nx}" \
      --ny "${ny}" \
      --nz "${nz}" \
      --dx-nm "${pf_dx_nm}" \
      --interface-width-nm "${lambda_sm_nm}" \
      --xb-out "${xB_out}" >/dev/null
    minimize_max_iter_local="1"
    minimize_convergence_steps_local="1"
  fi
  echo "[info] row=${row_idx} row_type=${row_type_local} is_reference_case=${is_reference_case} minimize_max_iter=${minimize_max_iter_local} minimize_convergence_steps=${minimize_convergence_steps_local}"

  CMD=(
    ./main_cuda "${nx}" "${ny}" "${nz}" "${dt}" "${nsteps}" "${out_every}" "${csv_out_every}" "${elastic}"
    --pf-param-file "${PF_PARAM_FILE}"
    --mode=minimize
    --minimize-full-model
    --minimize-max-iter "${minimize_max_iter_local}"
    --minimize-dt "${dt}"
    --radius-phys-nm "${radius_nm}"
    --elastic "${elastic}"
    --ic-23d-xB-out "${xB_out}"
    --init-case-tag "${case_tag}"
    --minimize-rms-dphi-threshold "1e-5"
    --minimize-rms-dY-threshold "1e-5"
    --minimize-energy-diff-rel-threshold "1e-7"
    --minimize-rms-res-for-energy-plateau "1e-5"
    --minimize-convergence-steps "${minimize_convergence_steps_local}"
    --minimize-dt-safety-limit "1e-6"
    --minimize-rms-res-threshold "1e-4"
    --minimize-vol-err-rel-threshold "3e-2"
    --eta-lambda-vol "0.9"
    --minimize-post-projection-iters "1"
  )
  if [[ "${is_reference_case}" == "1" ]]; then
    CMD+=(
      --init-mode raw_fields
      --init-phi-raw "${PROJECT_ROOT}/${raw_init_dir_rel}/phi_init.raw"
      --init-xB-raw "${PROJECT_ROOT}/${raw_init_dir_rel}/xB_init.raw"
      --init-meta "${PROJECT_ROOT}/${raw_init_dir_rel}/init_meta.json"
      --radius-phys-nm "0.0"
    )
  fi
  set -x
  "${CMD[@]}"
  set +x
  if [[ "${is_reference_case}" == "1" ]]; then
    write_reference_energy_csv || echo "[warn] reference_energy.csv post-write failed for ${case_tag}; summary script will reconstruct it from energy_minimize_*.csv" >&2
  fi
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
