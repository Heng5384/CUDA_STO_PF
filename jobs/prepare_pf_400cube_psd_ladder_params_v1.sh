#!/usr/bin/env bash
set -euo pipefail

# Pre-generate and freeze the common physical parameter file for every ladder
# case.  Only the case tag differs; all physical inputs are identical.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/_site_env.sh"
site_setup_project_root "${SCRIPT_DIR}"

SOURCE_ROOT="${SOURCE_ROOT:-${PROJECT_ROOT}}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
PLAN_MANIFEST="${CAMPAIGN_ROOT}/manifests/plan/campaign_case_manifest.json"

[[ -f "${PLAN_MANIFEST}" ]] || { echo "[fatal] campaign plan manifest missing" >&2; exit 2; }
mkdir -p "${CAMPAIGN_ROOT}/params"

export TEMP_C=380 DT=0.02 PHYS_DX_REF_M=1.0e-9 PF_DX_M=1.0e-9
export PHYS_LAMBDA_SM_M=4.0e-9 PHYS_GAMMA_JM2=0.168 PHYS_L_REF_FACTOR=5.0 PHYS_D_RATIO=0.01
export PHYSICAL_OVERRIDE_FILE="${SOURCE_ROOT}/data/qualification/pf_elastic_target_profile_v1/physical_override_T380_dx1nm_lambda4nm_dynamic_dt0p02.json"

mapfile -t rows < <(python3 - "${PLAN_MANIFEST}" <<'PY'
import json, pathlib, sys
for row in json.loads(pathlib.Path(sys.argv[1]).read_text())["cases"]:
    print(f"{row['case_id']}\t{row['case_label']}")
PY
)
[[ "${#rows[@]}" -eq 21 ]] || { echo "[fatal] expected 21 cases" >&2; exit 2; }

: >"${CAMPAIGN_ROOT}/params/parameter_manifest.sha256"
for row in "${rows[@]}"; do
  IFS=$'\t' read -r case_id case_label <<<"${row}"
  generated="$(site_generate_pf_param_file "pf_400cube_psd_ladder_${case_id}")"
  target="${CAMPAIGN_ROOT}/params/${case_id}_${case_label}.params"
  cp "${generated}" "${target}"
  sha256sum "${target}" >>"${CAMPAIGN_ROOT}/params/parameter_manifest.sha256"
done
sort -k2 "${CAMPAIGN_ROOT}/params/parameter_manifest.sha256" -o "${CAMPAIGN_ROOT}/params/parameter_manifest.sha256"
echo "PASS_400CUBE_PSD_LADDER_PARAMETER_FREEZE_V1"
