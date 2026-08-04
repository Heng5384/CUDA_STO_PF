#!/usr/bin/env bash
set -uo pipefail

# Reverse-chain submission watcher for gpu_vip_24h.  The QOS allows at most
# 4 submitted jobs and 1 active job per user, so this watcher keeps a 4-job
# sliding window and submits 021 -> 020 -> ... -> 001 as slots free.

SOURCE_ROOT="${SOURCE_ROOT:?SOURCE_ROOT is required}"
CAMPAIGN_ROOT="${CAMPAIGN_ROOT:?CAMPAIGN_ROOT is required}"
SBATCH="${SBATCH:?SBATCH is required}"
WATCH_LOG="${CAMPAIGN_ROOT}/submission_watcher_vip.log"
AGGREGATE_PASS="PASS_21_FIXTURES_STATIC_QUALIFICATION_V2"
SUBMITTER="${SOURCE_ROOT}/scripts/submit_pf_400cube_psd_ladder_dual_queue_v1.py"

: >"${WATCH_LOG}"
for attempt in $(seq 1 100000); do
  if [[ ! -f "${CAMPAIGN_ROOT}/reports/status.txt" ]]; then
    echo "[vip-watch:${attempt}] fixtures aggregate status not present" >>"${WATCH_LOG}"
    sleep 60
    continue
  fi
  aggregate="$(cat "${CAMPAIGN_ROOT}/reports/status.txt")"
  if [[ "${aggregate}" != "${AGGREGATE_PASS}" ]]; then
    echo "[vip-watch:${attempt}] fixtures aggregate status=${aggregate}" >>"${WATCH_LOG}"
    sleep 60
    continue
  fi

  echo "[vip-watch:${attempt}] submitting next eligible reverse jobs" >>"${WATCH_LOG}"
  python3 "${SUBMITTER}" \
    --campaign-root "${CAMPAIGN_ROOT}" \
    --source-root "${SOURCE_ROOT}" \
    --sbatch "${SBATCH}" \
    --submit-vip --force-queue gpu_vip_24h --max-submitted 4 \
    >>"${WATCH_LOG}" 2>&1 || {
      echo "[vip-watch:${attempt}] submission attempt failed" >>"${WATCH_LOG}"
      sleep 300
      continue
    }

  vip_submitted="$(python3 - "${CAMPAIGN_ROOT}/submission_validation.json" <<'PY' 2>/dev/null || printf '0'
import json, pathlib, sys
value = json.loads(pathlib.Path(sys.argv[1]).read_text())
print(int(value.get("vip_submitted_count", 0)))
PY
)"
  if [[ "${vip_submitted}" == "21" ]]; then
    echo "[vip-watch] all 21 reverse submissions complete" >>"${WATCH_LOG}"
    exit 0
  fi
  sleep 300
done
