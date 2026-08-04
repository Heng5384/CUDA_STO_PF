#!/usr/bin/env bash
set -uo pipefail

# Persistent submission watcher for the 21-case dual-queue ladder campaign.
# It waits for static fixture qualification, freezes parameter files, then
# submits gpu_uvip forward and gpu_vip_24h reverse jobs one active job per
# queue at a time, matching the cluster QOS limits.

SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
SBATCH="${SBATCH:?SBATCH is required}"
WATCH_LOG="${CAMPAIGN_ROOT}/submission_watcher.log"
AGGREGATE_PASS="PASS_21_FIXTURES_STATIC_QUALIFICATION_V2"
SUBMITTER="${SOURCE_ROOT}/scripts/submit_pf_400cube_psd_ladder_dual_queue_v1.py"
PARAMS_PREP="${SOURCE_ROOT}/jobs/prepare_pf_400cube_psd_ladder_params_v1.sh"

: >"${WATCH_LOG}"
for attempt in $(seq 1 100000); do
  if [[ ! -f "${CAMPAIGN_ROOT}/reports/status.txt" ]]; then
    echo "[watch:${attempt}] fixtures aggregate status not present" >>"${WATCH_LOG}"
    sleep 60
    continue
  fi
  aggregate="$(cat "${CAMPAIGN_ROOT}/reports/status.txt")"
  if [[ "${aggregate}" != "${AGGREGATE_PASS}" ]]; then
    echo "[watch:${attempt}] fixtures aggregate status=${aggregate}" >>"${WATCH_LOG}"
    sleep 60
    continue
  fi

  if [[ ! -f "${CAMPAIGN_ROOT}/params/parameter_manifest.sha256" ]]; then
    echo "[watch:${attempt}] freezing parameter files" >>"${WATCH_LOG}"
    env SOURCE_ROOT="${SOURCE_ROOT}" CAMPAIGN_ROOT="${CAMPAIGN_ROOT}" \
      bash "${PARAMS_PREP}" >>"${WATCH_LOG}" 2>&1 || {
        echo "[watch:${attempt}] parameter freeze failed" >>"${WATCH_LOG}"
        sleep 120
        continue
      }
  fi

  echo "[watch:${attempt}] submitting next eligible jobs" >>"${WATCH_LOG}"
  python3 "${SUBMITTER}" \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --source-root "${SOURCE_ROOT}" \
    --sbatch "${SBATCH}" \
    --submit-uvip --submit-vip --max-active 1 --force-queue gpu_uvip \
    >>"${WATCH_LOG}" 2>&1 || {
      echo "[watch:${attempt}] submission attempt failed" >>"${WATCH_LOG}"
      sleep 300
      continue
    }

  all_submitted="$(python3 - "${CAMPAIGN_ROOT}/submission_validation.json" <<'PY' 2>/dev/null || printf '0'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(1 if int(value.get("submitted_count", 0)) == 42 else 0)
PY
)"
  if [[ "${all_submitted}" == "1" ]]; then
    echo "[watch] all 42 logical submissions complete" >>"${WATCH_LOG}"
    exit 0
  fi
  sleep 300
done
