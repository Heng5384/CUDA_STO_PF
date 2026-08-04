#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "usage: $0 SOURCE_ROOT TRACKER_ROOT FIXTURE_ROOT EXPECTED_STEPS OUT_ROOT" >&2
  exit 2
fi

SOURCE_ROOT="$1"
TRACKER_ROOT="$2"
FIXTURE_ROOT="$3"
EXPECTED_STEPS="$4"
OUT_ROOT="$5"
AUDITOR="${SOURCE_ROOT}/scripts/audit_pf_246cube_hourly_merge_dissolution_v1.py"

if [[ -e "${OUT_ROOT}" ]]; then
  echo "[fatal] refusing to overwrite output: ${OUT_ROOT}" >&2
  exit 2
fi
for path in \
  "${AUDITOR}" \
  "${EXPECTED_STEPS}" \
  "${FIXTURE_ROOT}/initial_components.csv"
do
  if [[ ! -f "${path}" ]]; then
    echo "[fatal] missing required input: ${path}" >&2
    exit 2
  fi
done
for threshold in low medium strong; do
  for filename in \
    lineage_summary.txt particle_events.csv particle_lineage.csv \
    ensemble_observables.csv status.txt
  do
    path="${TRACKER_ROOT}/tracker_${threshold}/${filename}"
    if [[ ! -s "${path}" ]]; then
      echo "[fatal] missing tracker input: ${path}" >&2
      exit 2
    fi
  done
done

mkdir -p "${OUT_ROOT}/provenance"
sha256sum \
  "${AUDITOR}" \
  "${EXPECTED_STEPS}" \
  "${FIXTURE_ROOT}/initial_components.csv" \
  "${TRACKER_ROOT}"/tracker_{low,medium,strong}/{lineage_summary.txt,particle_events.csv,particle_lineage.csv,ensemble_observables.csv,status.txt} \
  >"${OUT_ROOT}/provenance/input_hashes.sha256"

python3 "${AUDITOR}" \
  --low "${TRACKER_ROOT}/tracker_low" \
  --medium "${TRACKER_ROOT}/tracker_medium" \
  --strong "${TRACKER_ROOT}/tracker_strong" \
  --initial-components "${FIXTURE_ROOT}/initial_components.csv" \
  --expected-steps "${EXPECTED_STEPS}" \
  --out "${OUT_ROOT}/audit"

grep -qx "PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1" \
  "${OUT_ROOT}/audit/status.txt"
cp "${OUT_ROOT}/audit/status.txt" "${OUT_ROOT}/status.txt"
cat "${OUT_ROOT}/status.txt"
