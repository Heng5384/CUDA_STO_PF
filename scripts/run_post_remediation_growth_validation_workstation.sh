#!/usr/bin/env bash
set -euo pipefail

# Workstation-only and serial: do not overlap another CUDA/PF task.
ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/post_remediation_growth_validation}"
MANIFEST="${MANIFEST:-params/post_remediation_growth_validation/post_remediation_validation_manifest.csv}"
CASE_LIST_FILE="${CASE_LIST_FILE:-}"
TIMEOUT_S="${TIMEOUT_S:-43200}"

cd "$ROOT"
mkdir -p "$OUT_ROOT"
if pgrep -x main_cuda >/tmp/post_remediation_main_cuda.txt; then
  echo "[abort] main_cuda is already running; refusing to overlap workstation GPU work." >&2
  exit 75
fi

python3 scripts/prepare_post_remediation_growth_validation_params.py
make main_cuda NVCC="$NVCC" CUDA_ROOT="$CUDA_ROOT" -j1 >"$OUT_ROOT/build.stdout.log" 2>"$OUT_ROOT/build.stderr.log"
echo "EXIT 0" >"$OUT_ROOT/build.status.txt"

if [ -n "$CASE_LIST_FILE" ]; then
  mapfile -t cases < <(sed -e 's/#.*//' -e '/^[[:space:]]*$/d' "$CASE_LIST_FILE")
else
  mapfile -t cases < <(python3 - "$MANIFEST" <<'PY'
import csv, sys
for row in csv.DictReader(open(sys.argv[1], newline='')):
    print(row['run_id'])
PY
)
fi

for case_name in "${cases[@]}"; do
  read -r param_file nsteps < <(python3 - "$MANIFEST" "$case_name" <<'PY'
import csv, sys
for row in csv.DictReader(open(sys.argv[1], newline='')):
    if row['run_id'] == sys.argv[2]:
        print(row['param_file'], row['nsteps'])
        break
else:
    raise SystemExit(f"case not found: {sys.argv[2]}")
PY
)
  case_dir="$OUT_ROOT/$case_name"
  mkdir -p "$case_dir"
  if [ -f "$case_dir/status.txt" ] && grep -qx 'EXIT 0' "$case_dir/status.txt"; then
    echo "[skip] $case_name"
    continue
  fi
  if pgrep -x main_cuda >/tmp/post_remediation_main_cuda.txt; then
    echo "[abort] main_cuda appeared while queued: $case_name" >&2
    exit 75
  fi
  echo "[run] $case_name"
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
    echo "[warn] $case_name exited $rc" >&2
  fi
done

echo "post_remediation_growth_validation_workstation_complete"
