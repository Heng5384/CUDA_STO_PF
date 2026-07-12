#!/usr/bin/env bash
set -euo pipefail

# Chain the numerical repair only after the existing pre-repair baseline has
# completed successfully.  This watcher never terminates or overlaps CUDA.
ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
BASELINE_ROOT="${BASELINE_ROOT:-tmp_codex_ops/post_remediation_growth_validation}"
BASELINE_MANIFEST="${BASELINE_MANIFEST:-params/post_remediation_growth_validation/post_remediation_validation_manifest.csv}"
BASELINE_LOG="$BASELINE_ROOT/nominal_completion_runner.log"
SMOKE_ROOT="${SMOKE_ROOT:-tmp_codex_ops/transport_mobility_consistency_smoke}"
SMOKE_MANIFEST="${SMOKE_MANIFEST:-params/transport_mobility_consistency_smoke/transport_mobility_consistency_smoke_manifest.csv}"
POLL_SECONDS="${POLL_SECONDS:-60}"

cd "$ROOT"
while ! grep -qx 'post_remediation_growth_validation_workstation_complete' "$BASELINE_LOG" 2>/dev/null; do
  if grep -qE '^\[warn\].* exited [^0]' "$BASELINE_LOG" 2>/dev/null; then
    echo "[watcher] baseline reports a failed case; refusing repaired smoke." >&2
    exit 1
  fi
  sleep "$POLL_SECONDS"
done

while IFS=, read -r run_id _; do
  [ "$run_id" = "run_id" ] && continue
  status="$(cat "$BASELINE_ROOT/$run_id/status.txt" 2>/dev/null || printf 'NOT_RUN')"
  if [ "$status" != 'EXIT 0' ]; then
    echo "[watcher] baseline status is not EXIT 0: $run_id -> $status" >&2
    exit 1
  fi
done <"$BASELINE_MANIFEST"

while pgrep -x main_cuda >/dev/null; do
  sleep "$POLL_SECONDS"
done

mkdir -p "$SMOKE_ROOT"
echo "baseline_complete_then_transport_smoke" >"$SMOKE_ROOT/chain_status.txt"
bash scripts/run_transport_mobility_consistency_smoke_workstation.sh \
  >"$SMOKE_ROOT/runner.log" 2>&1
python3 scripts/analyze_transport_mobility_consistency_smoke.py \
  --manifest "$SMOKE_MANIFEST" \
  --run-root "$SMOKE_ROOT" \
  --report-root reports/transport_mobility_consistency_smoke \
  >>"$SMOKE_ROOT/runner.log" 2>&1
echo 'transport_mobility_consistency_smoke_complete' >>"$SMOKE_ROOT/chain_status.txt"
