#!/usr/bin/env bash
# Short qualification for the raw-fixture + conserved-Y zero-mode handoff.

#SBATCH --job-name=pf_6h48h_pre
#SBATCH --partition=gpu_uvip
#SBATCH --qos=gpu_uvip
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --gres=gpu:1
#SBATCH --time=02:00:00
#SBATCH --mem=0
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

set -euo pipefail
PROJECT_ROOT="${PROJECT_ROOT:?PROJECT_ROOT is required}"
OUT_ROOT="${OUT_ROOT:?OUT_ROOT is required}"
FIXTURE_ROOT="${FIXTURE_ROOT:?FIXTURE_ROOT is required}"
if [[ -e "${OUT_ROOT}" ]]; then
  echo "[fatal] refusing to overwrite output root: ${OUT_ROOT}" >&2
  exit 2
fi
mkdir -p "${OUT_ROOT}"
exec >"${OUT_ROOT}/driver.stdout.log" 2>"${OUT_ROOT}/driver.stderr.log"
trap 'rc=$?; if [[ "$rc" -ne 0 ]]; then echo "FAILED_EXIT_${rc}" >"${OUT_ROOT}/status.txt"; fi' EXIT
cd "${PROJECT_ROOT}"
source jobs/_site_env.sh
site_prepare_cuda_env
echo RUNNING >"${OUT_ROOT}/status.txt"
make main_cuda CUDA_ARCH="${CUDA_ARCH:-sm_80}" CUDA_ROOT="${CUDA_ROOT}"
make test_pf_zero_mode_checkpoint

export TEMP_C=380 DT=0.02 PHYS_DX_REF_M=1.0e-9 PF_DX_M=1.0e-9
export PHYS_LAMBDA_SM_M=4.0e-9 PHYS_GAMMA_JM2=0.168 PHYS_L_REF_FACTOR=5.0 PHYS_D_RATIO=0.01
generated="$(site_generate_pf_param_file pf_6h48h_pre_T380)"
params="${OUT_ROOT}/pf_input.params"
python3 - "${generated}" "${params}" <<'PY'
import pathlib, sys
src, dst = map(pathlib.Path, sys.argv[1:])
overrides = {
    "pf_composition_mode":"legacy", "pf_y_update_mode":"lagged_rhs",
    "pf_baseline_control_mode":"full", "enable_legacy_gp_storage_coupling":"0",
    "enable_gp_assisted_beta_nucleation":"0", "enable_gp_runtime_library_nucleation":"0",
    "gp_nuc_enabled":"0", "gp_to_beta_enabled":"0", "gp_initial_population_enabled":"0",
    "gp_growth_enabled":"0", "gp_radius_evolution_enabled":"0", "gp_inventory_growth_enabled":"0",
    "gp_literature_model_enabled":"0", "diagnostic_rsmd_enabled":"0",
    "enable_dynamic_continue_bridge":"0", "enable_Y_rhs_picard":"0",
    "disable_Y_rhs_gamma_term":"0", "y_update_mass_projection_enabled":"0",
    "y_update_mass_projection_report_enabled":"0",
}
lines=[]; seen=set()
for raw in src.read_text().splitlines():
    line=raw.strip(); key=line.split("=",1)[0].strip() if "=" in line else ""
    if key in overrides:
        lines.append(f"{key}={overrides[key]}"); seen.add(key)
    else: lines.append(raw)
for key,val in overrides.items():
    if key not in seen: lines.append(f"{key}={val}")
dst.write_text("\n".join(lines)+"\n")
PY
cp "${generated}" "${OUT_ROOT}/converter_pf_input.params"
cp "${FIXTURE_ROOT}/fixture_manifest.json" "${OUT_ROOT}/fixture_manifest.json"

common=(128 128 128)
raw=(--radius-phys-nm 0 --init-mode raw_fields
  --init-phi-raw "${FIXTURE_ROOT}/phi.raw.f64"
  --init-xB-raw "${FIXTURE_ROOT}/xB.raw.f64"
  --init-meta "${FIXTURE_ROOT}/raw_init_meta.json")
zm=(--pf-zero-mode PF_CONSERVED_Y_ZERO_MODE_V1
  --pf-zero-mode-backend HOST_NEWTON_BISECTION_V1 --pf-zero-mode-tol-rel 1e-12 --pf-zero-mode-max-iter 24)
run() {
  local tag="$1"; shift
  local root="${OUT_ROOT}/${tag}"; mkdir -p "${root}/results"
  CUDA_STO_RESULTS_ROOT="${root}/results" CUDA_STO_SUPPRESS_VTK_OUTPUT=1 \
    ./main_cuda "${common[@]}" "$@" --pf-param-file "${params}" --mode dynamics \
      --ic-23d-xB-out 0.006219279767278563 --init-case-tag "${tag}" \
      >"${root}/stdout.log" 2>"${root}/stderr.log"
  grep -q 'PF_ZERO_MODE_FINAL_AUDIT status=PASS' "${root}/stdout.log"
  if grep -qiE 'fatal|nan|inf|mass projection' "${root}/stderr.log"; then return 3; fi
}
run continuous 0.02 128 128 128 0 "${zm[@]}" --pf-checkpoint-every 64 \
  --pf-checkpoint-path "${OUT_ROOT}/continuous.chk" "${raw[@]}"
run first_half 0.02 64 64 64 0 "${zm[@]}" --pf-checkpoint-every 64 \
  --pf-checkpoint-path "${OUT_ROOT}/half.chk" "${raw[@]}"
run restart 0.02 128 128 128 0 "${zm[@]}" --pf-checkpoint-every 64 \
  --pf-checkpoint-path "${OUT_ROOT}/restart.chk" --pf-restart-from "${OUT_ROOT}/half.chk"
cmp "${OUT_ROOT}/continuous.chk" "${OUT_ROOT}/restart.chk"
sha256sum "${OUT_ROOT}/continuous.chk" "${OUT_ROOT}/restart.chk" >"${OUT_ROOT}/restart_hashes.sha256"
run dt_half 0.01 256 256 256 0 "${zm[@]}" "${raw[@]}"
python3 - "${OUT_ROOT}" <<'PY'
import json, pathlib, re, sys
root=pathlib.Path(sys.argv[1])
rows={}
for tag in ('continuous','restart','dt_half'):
    text=(root/f'{tag}/stdout.log').read_text()
    audit=re.findall(r'PF_ZERO_MODE_FINAL_AUDIT status=(\S+).*?mean_mass_error=([+\-0-9.eE]+)',text)
    rows[tag]={'final_audit':audit[-1] if audit else None,
               'stderr':(root/f'{tag}/stderr.log').read_text()}
if not rows['continuous']['final_audit'] or not rows['restart']['final_audit'] or not rows['dt_half']['final_audit']:
    raise SystemExit('missing final audit')
summary={'status':'PASS_PF_6H48H_RAW_FIXTURE_ZERO_MODE_PREFLIGHT_V1',
         'restart_bytewise_equal':True,'dt_half_audit_pass':True,'runs':rows}
(root/'qualification_summary.json').write_text(json.dumps(summary,indent=2)+'\n')
(root/'status.txt').write_text(summary['status']+'\n')
print(json.dumps(summary,indent=2))
PY
