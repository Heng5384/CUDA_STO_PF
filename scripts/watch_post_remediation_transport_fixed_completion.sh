#!/usr/bin/env bash
set -euo pipefail

# CPU-only finalizer for the post-fix serial validation matrix.
ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
RUN_ROOT="${RUN_ROOT:-tmp_codex_ops/post_remediation_growth_validation_transport_fixed}"
MANIFEST="${MANIFEST:-params/post_remediation_growth_validation/post_remediation_validation_manifest.csv}"
REPORT_ROOT="${REPORT_ROOT:-reports/post_remediation_growth_validation_transport_fixed}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "$ROOT"
while ! grep -qx 'post_remediation_growth_validation_workstation_complete' "$RUN_ROOT/runner.log" 2>/dev/null; do
  if grep -qE '^\[warn\].* exited [^0]' "$RUN_ROOT/runner.log" 2>/dev/null; then
    echo "[finalizer] post-fix runtime case failed; preserving logs and stopping." >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

while IFS=, read -r run_id _; do
  [ "$run_id" = "run_id" ] && continue
  status="$(cat "$RUN_ROOT/$run_id/status.txt" 2>/dev/null || printf 'NOT_RUN')"
  if [ "$status" != 'EXIT 0' ]; then
    echo "[finalizer] incomplete post-fix case: $run_id -> $status" >&2
    exit 1
  fi
done <"$MANIFEST"

python3 scripts/analyze_post_remediation_growth_validation.py \
  --manifest "$MANIFEST" \
  --run-root "$RUN_ROOT" \
  --report-root "$REPORT_ROOT"
python3 scripts/verify_post_remediation_growth_validation.py \
  --report-root "$REPORT_ROOT"
echo 'post_remediation_transport_fixed_finalized' >"$RUN_ROOT/finalizer.status"
