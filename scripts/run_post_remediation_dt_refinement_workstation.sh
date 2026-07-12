#!/usr/bin/env bash
set -euo pipefail

# Serial workstation-only refinement. Never overlap a current CUDA/PF run.
ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/post_remediation_dt_refinement}"
MANIFEST="${MANIFEST:-params/post_remediation_dt_refinement/post_remediation_dt_refinement_manifest.csv}"
TIMEOUT_S="${TIMEOUT_S:-43200}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"
if pgrep -x main_cuda >/dev/null; then
  echo "[abort] main_cuda is already running; refusing to overlap workstation GPU work." >&2
  exit 75
fi

python3 scripts/prepare_post_remediation_dt_refinement.py

while IFS=, read -r run_id _; do
  [ "$run_id" = "run_id" ] && continue
  read -r param_file nsteps < <(python3 - "$MANIFEST" "$run_id" <<'PY'
import csv, sys
for row in csv.DictReader(open(sys.argv[1], newline='')):
    if row['run_id'] == sys.argv[2]:
        print(row['param_file'], row['nsteps'])
        break
else:
    raise SystemExit(f"case not found: {sys.argv[2]}")
PY
)
  case_dir="$OUT_ROOT/$run_id"
  mkdir -p "$case_dir"
  if [ -f "$case_dir/status.txt" ] && grep -qx 'EXIT 0' "$case_dir/status.txt"; then
    echo "[skip] $run_id"
    continue
  fi
  if pgrep -x main_cuda >/dev/null; then
    echo "[abort] main_cuda appeared while queued: $run_id" >&2
    exit 75
  fi
  echo "[run] $run_id"
  echo RUNNING >"$case_dir/status.txt"
  set +e
  timeout "$TIMEOUT_S" ./main_cuda --pf-param-file "$param_file" \
    --Nx 128 --Ny 128 --Nz 128 --nsteps "$nsteps" --out-every "$nsteps" \
    --csv-out-every 50 --init-case-tag "$run_id" \
    >"$case_dir/stdout.log" 2>"$case_dir/stderr.log"
  rc=$?
  set -e
  echo "EXIT $rc" >"$case_dir/status.txt"
  if [ "$rc" -ne 0 ]; then
    echo "[warn] $run_id exited $rc" >&2
    exit "$rc"
  fi
done <"$MANIFEST"

echo post_remediation_dt_refinement_workstation_complete
