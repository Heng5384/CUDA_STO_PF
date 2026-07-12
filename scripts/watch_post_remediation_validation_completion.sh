#!/usr/bin/env bash
set -euo pipefail

# CPU-only watcher for the workstation validation runner.  It never starts,
# stops, or overlaps CUDA work; once every manifest case has EXIT 0 it only
# regenerates the evidence tables from the completed runtime logs.
ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
RUN_ROOT="${RUN_ROOT:-tmp_codex_ops/post_remediation_growth_validation}"
MANIFEST="${MANIFEST:-params/post_remediation_growth_validation/post_remediation_validation_manifest.csv}"
REPORT_ROOT="${REPORT_ROOT:-reports/post_remediation_growth_validation}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "$ROOT"
while :; do
  mapfile -t cases < <(python3 - "$MANIFEST" <<'PY'
import csv
import sys

for row in csv.DictReader(open(sys.argv[1], newline="")):
    print(row["run_id"])
PY
)

  failed=0
  pending=0
  for case_name in "${cases[@]}"; do
    status_file="$RUN_ROOT/$case_name/status.txt"
    status="$(cat "$status_file" 2>/dev/null || printf 'NOT_RUN')"
    case "$status" in
      "EXIT 0") ;;
      RUNNING|NOT_RUN) pending=1 ;;
      *)
        echo "[watcher] non-success status for $case_name: $status" >&2
        failed=1
        ;;
    esac
  done

  if [ "$failed" -ne 0 ]; then
    echo "post_remediation_validation_watcher=FAILED_CASE_STATUS"
    exit 1
  fi
  if [ "$pending" -eq 0 ]; then
    python3 scripts/analyze_post_remediation_growth_validation.py \
      --manifest "$MANIFEST" \
      --run-root "$RUN_ROOT" \
      --report-root "$REPORT_ROOT"
    python3 scripts/verify_post_remediation_growth_validation.py \
      --report-root "$REPORT_ROOT"
    echo "post_remediation_validation_watcher=COMPLETE"
    exit 0
  fi
  sleep "$POLL_SECONDS"
done
