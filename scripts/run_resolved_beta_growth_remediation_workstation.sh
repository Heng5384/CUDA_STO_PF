#!/usr/bin/env bash
set -euo pipefail

# Workstation-only runner. It refuses to overlap another PF/CUDA run.
ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/resolved_beta_growth_remediation}"
TIMEOUT_S="${TIMEOUT_S:-43200}"
CASE_LIST_FILE="${CASE_LIST_FILE:-}"
PREPARE_SCRIPT="${PREPARE_SCRIPT:-scripts/prepare_resolved_beta_growth_remediation_params.py}"
MANIFEST="${MANIFEST:-params/resolved_beta_growth_remediation/manifest.csv}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"

if pgrep -x main_cuda >/tmp/resolved_beta_growth_remediation_main_cuda.txt; then
  echo "[abort] main_cuda is already running; refusing to overlap workstation GPU work." >&2
  ps -o pid,etime,%cpu,%mem,cmd -p "$(tr '\n' ',' </tmp/resolved_beta_growth_remediation_main_cuda.txt | sed 's/,$//')" >&2 || true
  exit 75
fi

python3 "$PREPARE_SCRIPT"

set +e
make main_cuda NVCC="$NVCC" CUDA_ROOT="$CUDA_ROOT" -j1 >"$OUT_ROOT/build.stdout.log" 2>"$OUT_ROOT/build.stderr.log"
build_rc=$?
set -e
echo "EXIT $build_rc" >"$OUT_ROOT/build.status.txt"
if [ "$build_rc" -ne 0 ]; then
  exit "$build_rc"
fi

manifest="$MANIFEST"
if [ -n "$CASE_LIST_FILE" ]; then
  if [ ! -f "$CASE_LIST_FILE" ]; then
    echo "[abort] CASE_LIST_FILE not found: $CASE_LIST_FILE" >&2
    exit 66
  fi
  cases=( $(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$CASE_LIST_FILE") )
else
  mapfile -t cases < <(python3 - "$manifest" <<'PY'
import csv, sys
with open(sys.argv[1], newline='') as fh:
    for row in csv.DictReader(fh):
        print(row['case'])
PY
)
fi

for case_name in "${cases[@]}"; do
  read -r param_file nsteps < <(python3 - "$manifest" "$case_name" <<'PY'
import csv, sys
for row in csv.DictReader(open(sys.argv[1], newline='')):
    if row['case'] == sys.argv[2]:
        print(row['param_file'], row['nsteps'])
        break
else:
    raise SystemExit(f"case not in manifest: {sys.argv[2]}")
PY
)
  case_dir="$OUT_ROOT/$case_name"
  mkdir -p "$case_dir"
  if [ -f "$case_dir/status.txt" ] && grep -q 'EXIT 0' "$case_dir/status.txt"; then
    echo "[skip] $case_name already completed"
    continue
  fi
  echo "[run] $case_name param=$param_file nsteps=$nsteps"
  echo RUNNING >"$case_dir/status.txt"
  set +e
  timeout "$TIMEOUT_S" ./main_cuda \
    --pf-param-file "$param_file" \
    --Nx 128 --Ny 128 --Nz 128 \
    --nsteps "$nsteps" --out-every "$nsteps" --csv-out-every 50 \
    --init-case-tag "$case_name" \
    >"$case_dir/stdout.log" 2>"$case_dir/stderr.log"
  rc=$?
  set -e
  echo "EXIT $rc" >"$case_dir/status.txt"
  if [ "$rc" -ne 0 ]; then
    echo "[warn] $case_name exited with $rc" >&2
  fi
done

echo "resolved_beta_growth_remediation_workstation_complete"
