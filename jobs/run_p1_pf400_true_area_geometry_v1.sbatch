#!/usr/bin/env bash
#SBATCH --job-name=p1_pf400_area
#SBATCH --partition=uvip_parallel_intel
#SBATCH --qos=uvip_parallel_intel
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --array=0-27%4

# Read-only checkpoint geometry extraction.  This job never invokes the PF
# executable and writes only beneath the separate P1 output root.
set -euo pipefail

: "${P1_ROOT:?P1_ROOT is required}"
: "${ROUTE_CSV:?ROUTE_CSV is required}"
: "${PYTHON_DEPS:?PYTHON_DEPS is required}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${P1_ROOT}}"

row=$(awk -F, -v task="${SLURM_ARRAY_TASK_ID}" 'NR>1 && $1==task {print; exit}' "${ROUTE_CSV}")
if [[ -z "${row}" ]]; then
  echo "missing route for array task ${SLURM_ARRAY_TASK_ID}" >&2
  exit 2
fi
IFS=, read -r task_id case_id attempt_id time_h step source_type source_path <<<"${row}"
out="${OUTPUT_ROOT}/geometry/case_${case_id}/${time_h}h"
if [[ -e "${out}" ]]; then
  echo "refusing to overwrite ${out}" >&2
  exit 3
fi
mkdir -p "$(dirname "${out}")"

common=(
  "${P1_ROOT}/scripts/measure_pf400_periodic_interface_v1.py"
  --case-id "${case_id}"
  --time-h "${time_h}"
  --out "${out}"
)
if [[ "${source_type}" == "fixture" ]]; then
  command=(python3 "${common[@]}" --fixture "${source_path}" --step "${step}")
elif [[ "${source_type}" == "checkpoint" ]]; then
  command=(python3 "${common[@]}" --checkpoint "${source_path}")
else
  echo "unsupported source type ${source_type}" >&2
  exit 4
fi

export PYTHONPATH="${PYTHON_DEPS}"
"${command[@]}"
