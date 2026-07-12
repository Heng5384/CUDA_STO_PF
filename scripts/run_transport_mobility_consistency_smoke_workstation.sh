#!/usr/bin/env bash
set -euo pipefail

# Workstation-only, serial validation. Refuse to overlap an existing CUDA run.
ROOT="${ROOT:-/home/zhiheng/PF/CUDA_STO_PF}"
NVCC="${NVCC:-/usr/local/cuda-12.9/bin/nvcc}"
CUDA_ROOT="${CUDA_ROOT:-/usr/local/cuda-12.9}"
OUT_ROOT="${OUT_ROOT:-tmp_codex_ops/transport_mobility_consistency_smoke}"
MANIFEST="${MANIFEST:-params/transport_mobility_consistency_smoke/transport_mobility_consistency_smoke_manifest.csv}"
TIMEOUT_S="${TIMEOUT_S:-14400}"

cd "$ROOT"
if pgrep -x main_cuda >/dev/null; then
  echo "[abort] main_cuda is already running; refusing to overlap workstation GPU work." >&2
  exit 75
fi
mkdir -p "$OUT_ROOT"

python3 scripts/prepare_transport_mobility_consistency_smoke.py
make main_cuda NVCC="$NVCC" CUDA_ROOT="$CUDA_ROOT" -j1 \
  >"$OUT_ROOT/build.stdout.log" 2>"$OUT_ROOT/build.stderr.log"
echo "EXIT 0" >"$OUT_ROOT/build.status.txt"

while IFS=, read -r run_id temperature_C group nsteps source_mode f_max xB_target exchange seed thermo provenance param_file; do
  [ "$run_id" = "run_id" ] && continue
  # Python's csv module emits CRLF by default; strip the final field before
  # passing it to main_cuda so a valid path cannot acquire a literal '\r'.
  param_file="${param_file%$'\r'}"
  if [ ! -f "$param_file" ]; then
    echo "[fatal] smoke parameter file is missing: $param_file" >&2
    exit 2
  fi
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
  timeout "$TIMEOUT_S" ./main_cuda \
    --pf-param-file "$param_file" \
    --Nx 128 --Ny 128 --Nz 128 \
    --nsteps "$nsteps" --out-every "$nsteps" --csv-out-every 10 \
    --init-case-tag "$run_id" \
    >"$case_dir/stdout.log" 2>"$case_dir/stderr.log"
  rc=$?
  set -e
  echo "EXIT $rc" >"$case_dir/status.txt"
  [ "$rc" -eq 0 ] || echo "[warn] $run_id exited $rc" >&2
done <"$MANIFEST"

echo "transport_mobility_consistency_smoke_complete"
