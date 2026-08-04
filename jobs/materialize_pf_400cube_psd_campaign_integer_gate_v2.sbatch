#!/usr/bin/env bash
#SBATCH --job-name=pf400_psd_v2_static
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00

# Build the 21 independent V2 fixtures only.  This is a CPU-side immutable
# input-materialization job, not a PF dynamics job and not a GPU submission.
set -euo pipefail

SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
PLAN_ROOT="${PLAN_ROOT:?PLAN_ROOT is required}"
LIBRARY_ROOT="${LIBRARY_ROOT:?LIBRARY_ROOT is required}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"

[[ -f "${SOURCE_ROOT}/scripts/materialize_pf_400cube_psd_campaign_integer_gate_resolution_v2.py" ]]
[[ -f "${SOURCE_ROOT}/scripts/audit_pf_400cube_psd_campaign_integer_gate_v2.py" ]]
[[ -f "${PLAN_ROOT}/campaign_case_manifest.json" ]]
[[ -f "${LIBRARY_ROOT}/library/library_manifest.json" || -f "${LIBRARY_ROOT}/library_manifest.json" ]]
[[ ! -e "${CAMPAIGN_ROOT}" ]] || { echo "[fatal] refusing to overwrite campaign root: ${CAMPAIGN_ROOT}" >&2; exit 2; }

mkdir -p "${CAMPAIGN_ROOT}/fixtures" "${CAMPAIGN_ROOT}/reports" "${CAMPAIGN_ROOT}/manifests"
cp -a "${PLAN_ROOT}/." "${CAMPAIGN_ROOT}/manifests/plan"
printf 'RUNNING_400CUBE_PSD_CAMPAIGN_INTEGER_STATIC_MATERIALIZATION_V2\n' >"${CAMPAIGN_ROOT}/status.txt"

mapfile -t case_rows < <(python3 - "${CAMPAIGN_ROOT}/manifests/plan/campaign_case_manifest.json" <<'PY'
import json, pathlib, sys
for row in json.loads(pathlib.Path(sys.argv[1]).read_text())['cases']:
    print(f"{row['case_id']}\t{row['case_label']}")
PY
)
[[ "${#case_rows[@]}" -eq 21 ]] || { echo "[fatal] expected 21 unique cases" >&2; exit 2; }

for row in "${case_rows[@]}"; do
    IFS=$'\t' read -r case_id case_label <<<"${row}"
    spec="${CAMPAIGN_ROOT}/manifests/plan/case_specs/${case_id}_${case_label}.json"
    fixture="${CAMPAIGN_ROOT}/fixtures/${case_id}"
    log="${CAMPAIGN_ROOT}/fixtures/${case_id}.materialize.log"
    python3 "${SOURCE_ROOT}/scripts/materialize_pf_400cube_psd_campaign_integer_gate_resolution_v2.py" \
        --spec "${spec}" --library-root "${LIBRARY_ROOT}" --out "${fixture}" --reverse-order-check >"${log}" 2>&1
    python3 "${SOURCE_ROOT}/scripts/audit_pf_400cube_psd_campaign_integer_gate_v2.py" \
        fixture --fixture "${fixture}" --out "${fixture}/static_audit" >"${CAMPAIGN_ROOT}/fixtures/${case_id}.audit.log" 2>&1
    grep -qx 'PASS_400CUBE_PSD_CAMPAIGN_INTEGER_GATE_STATIC_AUDIT_V2' "${fixture}/static_audit/status.txt"
done

python3 "${SOURCE_ROOT}/scripts/audit_pf_400cube_psd_campaign_integer_gate_v2.py" \
    aggregate --campaign "${CAMPAIGN_ROOT}" --case-manifest "${CAMPAIGN_ROOT}/manifests/plan/campaign_case_manifest.json" --out "${CAMPAIGN_ROOT}/reports" >"${CAMPAIGN_ROOT}/aggregate.stdout.log" 2>"${CAMPAIGN_ROOT}/aggregate.stderr.log"
grep -qx 'PASS_21_FIXTURES_STATIC_QUALIFICATION_V2' "${CAMPAIGN_ROOT}/reports/status.txt"
printf 'PASS_21_FIXTURES_STATIC_QUALIFICATION_V2\n' >"${CAMPAIGN_ROOT}/status.txt"
cat "${CAMPAIGN_ROOT}/status.txt"
