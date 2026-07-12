#!/usr/bin/env bash
# Required-supply-informed GP supply RSMD scenario array.
# One array task runs one manifest row from params/gp_required_supply_informed_rsmd/manifest.csv.

#SBATCH --job-name=gp_rsmd_supply
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --array=0-89%1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" && -f "${SLURM_SUBMIT_DIR}/jobs/_site_env.sh" ]]; then
  SCRIPT_DIR="${SLURM_SUBMIT_DIR}/jobs"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi

source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"
cd "${PROJECT_ROOT}"

OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/gp_required_supply_informed_rsmd_cluster}"
REPORT_ROOT="${REPORT_ROOT:-reports/gp_required_supply_informed_rsmd_cluster}"
MANIFEST="${MANIFEST:-params/gp_required_supply_informed_rsmd/manifest.csv}"

mkdir -p "${OUT_ROOT}" "${REPORT_ROOT}" jobs/logs jobs/generated

site_redirect_slurm_logs "gp_required_supply_rsmd_${SLURM_ARRAY_JOB_ID:-job}_${SLURM_ARRAY_TASK_ID:-task}"
site_print_env_banner

python3 scripts/prepare_gp_required_supply_informed_rsmd.py

if [[ ! -f "${MANIFEST}" ]]; then
  echo "[fatal] manifest missing: ${MANIFEST}" >&2
  exit 2
fi

task_row="$(
python3 - "${MANIFEST}" "${SLURM_ARRAY_TASK_ID:-0}" <<'PY'
import csv
import sys

manifest = sys.argv[1]
task_id = int(sys.argv[2])
rows = []
with open(manifest, newline="") as fh:
    for row in csv.DictReader(fh):
        if row.get("run_selected") == "1":
            rows.append(row)
if task_id < 0 or task_id >= len(rows):
    raise SystemExit(f"array task id {task_id} outside selected rows 0..{len(rows)-1}")
row = rows[task_id]
print("\t".join([
    row["case"],
    row["param_file"],
    row["T_C"],
    row["scenario_id"],
    row["xB_ceiling_eff"],
    row["R_exchange_nm"],
    row["chi_rel"],
    row["kernel_radius_dx"],
    row["nsteps"],
    row["provenance"],
]))
PY
)"

IFS=$'\t' read -r case_name param_file T scenario target rex chi kernel nsteps provenance <<<"${task_row}"
case_dir="${OUT_ROOT}/${case_name}"
mkdir -p "${case_dir}"

if [[ -f "${case_dir}/status.txt" ]] && grep -q "EXIT 0" "${case_dir}/status.txt"; then
  echo "[skip] ${case_name} already completed"
  python3 scripts/analyze_gp_required_supply_informed_rsmd.py \
    --run-root "${OUT_ROOT}" \
    --report-root "${REPORT_ROOT}"
  exit 0
fi

site_build_main_cuda >"${OUT_ROOT}/build_${SLURM_ARRAY_JOB_ID:-job}_${SLURM_ARRAY_TASK_ID:-task}.stdout.log" \
  2>"${OUT_ROOT}/build_${SLURM_ARRAY_JOB_ID:-job}_${SLURM_ARRAY_TASK_ID:-task}.stderr.log"
echo "EXIT 0" >"${OUT_ROOT}/build.status.txt"

echo "RUNNING" >"${case_dir}/status.txt"
echo "[run] ${case_name} T=${T} scenario=${scenario} target=${target} R=${rex} chi=${chi} kernel=${kernel} nsteps=${nsteps}"

set +e
./main_cuda \
  --Nx 128 --Ny 128 --Nz 128 \
  --nsteps "${nsteps}" \
  --out-every "${nsteps}" \
  --csv-out-every 50 \
  --beta_staged_accumulation_enabled 1 \
  --beta_staged_accumulation_interval_steps 10 \
  --beta_staged_accumulation_GP_capture_radius_nm 12 \
  --beta_staged_accumulation_matrix_draw_radius_nm 64 \
  --beta_staged_accumulation_max_fraction_per_step 0.1 \
  --beta_staged_accumulation_max_inventory_per_step 1e300 \
  --beta_staged_insert_when_target_reached 1 \
  --beta_staged_debug_accelerated_accumulation 1 \
  --beta_staged_debug_accumulation_rate_multiplier 1000000 \
  --beta_staged_debug_stop_after_resolved_insert 0 \
  --resolved_handoff_xB_write_mode preserve_profile_xB_alpha_in_support \
  --scheduled-nuc-source-lambda-nm 0.6 \
  --scheduled-nuc-target-lambda-nm 0.6 \
  --scheduled-nuc-scale-interface-width 1.0 \
  --scheduled-nuc-scale-xB-profile-width 1.0 \
  --diagnostic_rsmd_enabled 1 \
  --diagnostic_rsmd_xB_halo_target "${target}" \
  --diagnostic_rsmd_R_exchange_nm "${rex}" \
  --diagnostic_rsmd_chi_rel "${chi}" \
  --diagnostic_rsmd_kernel_radius_dx "${kernel}" \
  --diagnostic_rsmd_release_window_steps "${nsteps}" \
  --diagnostic_rsmd_provenance "${provenance}" \
  --pf-param-file "${param_file}" \
  --init-case-tag "${case_name}" \
  >"${case_dir}/stdout.log" 2>"${case_dir}/stderr.log"
rc=$?
set -e

echo "EXIT ${rc}" >"${case_dir}/status.txt"
if [[ "${rc}" -ne 0 ]]; then
  echo "[warn] ${case_name} exited with ${rc}" >&2
fi

python3 scripts/analyze_gp_required_supply_informed_rsmd.py \
  --run-root "${OUT_ROOT}" \
  --report-root "${REPORT_ROOT}"

exit "${rc}"
