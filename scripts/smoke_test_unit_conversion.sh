#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

main_cuda_bin="./main_cuda"
if [[ ! -x "$main_cuda_bin" ]]; then
  echo "[fatal] $main_cuda_bin not found or not executable" >&2
  exit 2
fi

workdir="$(mktemp -d "${TMPDIR:-/tmp}/pf_unitconv_smoke.XXXXXX")"
trap 'rm -rf "$workdir"' EXIT

good_params="$workdir/pf_smoke_good.params"
bad_t_params="$workdir/pf_smoke_bad_t.params"
bad_l_params="$workdir/pf_smoke_bad_l.params"
bad_schema_params="$workdir/pf_smoke_bad_schema.params"
legacy_params="$workdir/pf_smoke_legacy.params"

run_main() {
  local params="$1"
  local stdout_file="$2"
  local stderr_file="$3"
  set +e
  "$main_cuda_bin" --Nx 32 --Ny 32 --Nz 32 --pf-param-file "$params" --nsteps 1 >"$stdout_file" 2>"$stderr_file"
  local status=$?
  set -e
  return "$status"
}

make_good_params() {
  python3 Unit_Psedobinary.py \
    --input-json physical_inputs.example.json \
    --output-pf-param-file "$good_params" >/dev/null
}

edit_param_file() {
  local src="$1"
  local dst="$2"
  local mode="$3"
  python3 - "$src" "$dst" "$mode" <<'PY'
from pathlib import Path
import sys

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
mode = sys.argv[3]
lines = src.read_text().splitlines()
out = []
for line in lines:
    if not line or line.startswith("#"):
        out.append(line)
        continue
    if "=" not in line:
        out.append(line)
        continue
    key, value = line.split("=", 1)
    if mode == "t_real_unit_x2" and key == "t_real_unit":
        out.append(f"{key}={float(value) * 2.0:.16e}")
    elif mode == "gp_L_eta_code_x101" and key == "gp_L_eta_code":
        out.append(f"{key}={float(value) * 1.01:.16e}")
    elif mode == "drop_schema" and key == "pf_params_schema_version":
        continue
    elif mode == "legacy_gp_W_eta" and key in {"gp_W_eta_code", "gp_W_eta_phys"}:
        continue
    else:
        out.append(line)

if mode == "legacy_gp_W_eta":
    out.append("gp_W_eta=5.0000000000000000e-01")

dst.write_text("\n".join(out) + "\n")
PY
}

check_summary_rows() {
  local stdout_file="$1"
  python3 - "$stdout_file" <<'PY'
from pathlib import Path
import sys

wanted = {"W_phi", "kappa_phi", "L_phi", "W_eta", "kappa_eta", "L_eta"}
seen = {}
for line in Path(sys.argv[1]).read_text().splitlines():
    parts = line.split()
    if len(parts) >= 5 and parts[0] in wanted:
        seen[parts[0]] = float(parts[4])

missing = wanted - seen.keys()
if missing:
    raise SystemExit(f"missing summary rows: {sorted(missing)}")

bad = {k: v for k, v in seen.items() if abs(v - 1.0) >= 1e-6}
if bad:
    raise SystemExit(f"ratio mismatch: {bad}")
PY
}

no_fatal() {
  local stdout_file="$1"
  local stderr_file="$2"
  if grep -q "\[fatal\]" "$stdout_file" "$stderr_file"; then
    echo "[fail] unexpected [fatal] in positive case" >&2
    return 1
  fi
  return 0
}

case_good() {
  local stdout_file="$workdir/good.stdout"
  local stderr_file="$workdir/good.stderr"
  if ! run_main "$good_params" "$stdout_file" "$stderr_file"; then
    echo "[fail] good case exited non-zero" >&2
    return 1
  fi
  if ! grep -q "\[Unit Conversion Summary\]" "$stdout_file"; then
    echo "[fail] good case missing Unit Conversion Summary" >&2
    return 1
  fi
  check_summary_rows "$stdout_file"
  no_fatal "$stdout_file" "$stderr_file"
  grep -A 20 "\[Unit Conversion Summary\]" "$stdout_file"
}

case_bad_t() {
  local stdout_file="$workdir/bad_t.stdout"
  local stderr_file="$workdir/bad_t.stderr"
  edit_param_file "$good_params" "$bad_t_params" "t_real_unit_x2"
  if run_main "$bad_t_params" "$stdout_file" "$stderr_file"; then
    echo "[fail] bad_t case unexpectedly succeeded" >&2
    return 1
  fi
  if ! grep -Eq "D_alpha consistency mismatch|gp_L_eta_code and gp_L_eta_phys are inconsistent" "$stderr_file"; then
    echo "[fail] bad_t case did not hit the expected unit-consistency guard" >&2
    return 1
  fi
}

case_bad_l() {
  local stdout_file="$workdir/bad_l.stdout"
  local stderr_file="$workdir/bad_l.stderr"
  edit_param_file "$good_params" "$bad_l_params" "gp_L_eta_code_x101"
  if run_main "$bad_l_params" "$stdout_file" "$stderr_file"; then
    echo "[fail] bad_L case unexpectedly succeeded" >&2
    return 1
  fi
  grep -q "gp_L_eta_code and gp_L_eta_phys are inconsistent" "$stderr_file"
}

case_bad_schema() {
  local stdout_file="$workdir/bad_schema.stdout"
  local stderr_file="$workdir/bad_schema.stderr"
  edit_param_file "$good_params" "$bad_schema_params" "drop_schema"
  if run_main "$bad_schema_params" "$stdout_file" "$stderr_file"; then
    echo "[fail] bad_schema case unexpectedly succeeded" >&2
    return 1
  fi
  grep -q "pf_params_schema_version" "$stderr_file"
}

case_legacy() {
  local stdout_file="$workdir/legacy.stdout"
  local stderr_file="$workdir/legacy.stderr"
  edit_param_file "$good_params" "$legacy_params" "legacy_gp_W_eta"
  if ! run_main "$legacy_params" "$stdout_file" "$stderr_file"; then
    echo "[fail] legacy case exited non-zero" >&2
    return 1
  fi
  grep -q "gp_W_eta is deprecated" "$stderr_file"
}

pass_count=0
total_count=5

make_good_params

if case_good; then
  echo "[ok] good_case"
  pass_count=$((pass_count + 1))
else
  echo "[fail] good_case" >&2
fi

if case_bad_t; then
  echo "[ok] bad_t_case"
  pass_count=$((pass_count + 1))
else
  echo "[fail] bad_t_case" >&2
fi

if case_bad_l; then
  echo "[ok] bad_L_case"
  pass_count=$((pass_count + 1))
else
  echo "[fail] bad_L_case" >&2
fi

if case_bad_schema; then
  echo "[ok] bad_schema_case"
  pass_count=$((pass_count + 1))
else
  echo "[fail] bad_schema_case" >&2
fi

if case_legacy; then
  echo "[ok] legacy_case"
  pass_count=$((pass_count + 1))
else
  echo "[fail] legacy_case" >&2
fi

echo "PASS ${pass_count}/${total_count}"
if [[ "$pass_count" -eq "$total_count" ]]; then
  exit 0
fi
exit 1
