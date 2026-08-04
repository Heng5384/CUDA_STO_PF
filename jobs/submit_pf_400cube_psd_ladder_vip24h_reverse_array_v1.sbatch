#!/usr/bin/env bash
#SBATCH --job-name=pf400_psd_vip_rev
#SBATCH --partition=gpu_vip_24h
#SBATCH --qos=gpu_vip_24h
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --mem=0
#SBATCH --array=1-21%1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail

: "${SOURCE_ROOT:?SOURCE_ROOT is required}"
: "${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"

cd "${SOURCE_ROOT}"
source jobs/_site_env.sh
site_prepare_cuda_env
[[ -x main_cuda ]] || { echo "[fatal] main_cuda is missing" >&2; exit 2; }

row="$(python3 - "${CAMPAIGN_ROOT}/manifests/plan/campaign_case_manifest.json" "${SLURM_ARRAY_TASK_ID}" <<'PY'
import json, pathlib, sys
cases = json.loads(pathlib.Path(sys.argv[1]).read_text())["cases"]
cases = sorted(cases, key=lambda row: int(row["case_id"]), reverse=True)
index = int(sys.argv[2]) - 1
if index < 0 or index >= len(cases):
    raise SystemExit("array index out of range")
row = cases[index]
print(f"{row['case_id']}\t{row['case_label']}")
PY
)"
IFS=$'\t' read -r CASE_ID CASE_LABEL <<<"${row}"
PARAM_FILE="${CAMPAIGN_ROOT}/params/${CASE_ID}_${CASE_LABEL}.params"
[[ -f "${PARAM_FILE}" ]] || { echo "[fatal] parameter file missing: ${PARAM_FILE}" >&2; exit 2; }

export SOURCE_ROOT CAMPAIGN_ROOT CASE_ID CASE_LABEL PARAM_FILE QUEUE=gpu_vip_24h
exec bash jobs/run_pf_400cube_psd_ladder_production_v1.sh
