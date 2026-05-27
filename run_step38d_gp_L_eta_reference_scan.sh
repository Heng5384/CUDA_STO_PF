#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zhiheng/PF/CUDA_STO_PF}"
cd "$ROOT_DIR"

BASE_PARAMS="${BASE_PARAMS:-$ROOT_DIR/Results/chel_T380_cuda_96x96x96_step39S/step39S_dt1e-4_L1e-5_t2/pf_input.params}"
L_ETA_DIFF_REF_CODE="${L_ETA_DIFF_REF_CODE:-4.7495061735e+01}"
DT_CODE="${DT_CODE:-1e-4}"
NSTEPS="${NSTEPS:-20000}"
OUT_EVERY="${OUT_EVERY:-1000}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/Results/ch_T380_cuda_96x96x96_dt${DT_CODE}_steps${NSTEPS}_xB0.030}"
SCRIPT_OUT="${SCRIPT_OUT:-$ROOT_DIR/reports/data}"

mkdir -p "$RESULT_ROOT" "$SCRIPT_OUT"

F_LIST=("1e-6" "1e-5" "1e-4" "1e-3" "1e-2")
CASES=()

make_case_name() {
  local f="$1"
  case "$f" in
    1e-6) echo "gp_eta_kinetic_scan_f1em6" ;;
    1e-5) echo "gp_eta_kinetic_scan_f1em5" ;;
    1e-4) echo "gp_eta_kinetic_scan_f1em4" ;;
    1e-3) echo "gp_eta_kinetic_scan_f1em3" ;;
    1e-2) echo "gp_eta_kinetic_scan_f1em2" ;;
    *) echo "gp_eta_kinetic_scan_${f}" | tr '+.' '__' ;;
  esac
}

for F_ETA in "${F_LIST[@]}"; do
  CASE_NAME="$(make_case_name "$F_ETA")"
  CASE_DIR="$RESULT_ROOT/$CASE_NAME"
  mkdir -p "$CASE_DIR"
  CASES+=("$CASE_NAME")

  GP_L_ETA=$(python3 - <<PY
f = float("${F_ETA}")
Lref = float("${L_ETA_DIFF_REF_CODE}")
print("{:.12e}".format(f * Lref))
PY
)

  python3 - <<PY
from pathlib import Path

base = Path("${BASE_PARAMS}")
out = Path("${CASE_DIR}") / "pf_input.params"
lines = base.read_text(encoding="utf-8").splitlines()

drop_prefixes = {
    "gp_W_eta=",
    "gp_kappa_eta=",
    "gp_W_eta_phys=",
    "gp_kappa_eta_phys=",
    "gp_W_eta_code=",
    "gp_kappa_eta_code=",
    "gp_L_eta=",
    "gp_elastic_enabled=",
    "gp_elastic_active_eta=",
    "gp_elastic_active_phi=",
    "gp_eps_iso=",
    "gp_to_beta_enabled=",
    "gp_nuc_enabled=",
    "y_update_mass_projection_enabled=",
    "gp_y_update_mode=",
    "gp_init_mode=",
    "gp_obs_target_radius_nm=",
    "gp_obs_eta_peak=",
    "gp_obs_iface_width_nm=",
    "gp_obs_profile_type=",
    "gp_obs_match_mode=",
    "gp_obs_compensation_mode=",
    "gp_obs_depletion_radius_factor=",
    "gp_obs_depletion_smooth_width_factor=",
    "gp_obs_min_xB_alpha=",
    "gp_obs_max_xB_alpha=",
    "gp_drive_reference_mode=",
    "gp_delta_g_stab=",
    "gp_xB_fixed=",
    "gp_raw_reaction_drive_only=",
    "thermo_convex_extrapolation_enabled=",
    "diag_vtk_enabled=",
    "dt=",
    "temperature_C=",
    "ic_23d_xB_out=",
    "ic_vf_target_phi=",
    "ic_vf_init_phi=",
}

kept = []
for line in lines:
    s = line.strip()
    if not s or s.startswith("#"):
        kept.append(line)
        continue
    if any(s.startswith(prefix) for prefix in drop_prefixes):
        continue
    kept.append(line)

append = [
    "model_mode=gp_zone",
    "gp_init_mode=observed_gp_diffuse",
    "gp_nuc_enabled=0",
    "gp_to_beta_enabled=0",
    "y_update_mass_projection_enabled=0",
    "gp_y_update_mode=conservative_y_rhs",
    "gp_elastic_enabled=0",
    "gp_elastic_active_eta=0",
    "gp_elastic_active_phi=0",
    "thermo_convex_extrapolation_enabled=0",
    "gp_drive_reference_mode=mechanical_mixture",
    "gp_xB_fixed=0.35",
    "gp_delta_g_stab=0.0",
    "gp_raw_reaction_drive_only=1",
    "temperature_C=380",
    "dt=${DT_CODE}",
    "gp_L_eta=${GP_L_ETA}",
    "gp_gamma_alpha_gp=5.0e-2",
    "gp_l_eta_nm=1.0",
    "gp_D_ratio=1.0",
    "gp_xB_eq_alpha_for_eta=0.03",
    "gp_W_eta_phys=6.59167373e8",
    "gp_kappa_eta_phys=6.82679420e-11",
    "gp_eps_iso=0.0",
    "gp_obs_target_radius_nm=1.0",
    "gp_obs_eta_peak=1.0",
    "gp_obs_iface_width_nm=0.20",
    "gp_obs_profile_type=tanh",
    "gp_obs_match_mode=match_integral_h_volume",
    "gp_obs_compensation_mode=smooth_radial_depletion",
    "gp_obs_depletion_radius_factor=2.0",
    "gp_obs_depletion_smooth_width_factor=0.5",
    "gp_obs_min_xB_alpha=1e-6",
    "gp_obs_max_xB_alpha=0.999999",
    "ic_vf_target_phi=1.0e-6",
    "ic_vf_init_phi=0.0",
    "ic_23d_xB_out=0.03",
    "diag_vtk_enabled=1",
]

out.write_text("\n".join(kept + append) + "\n", encoding="utf-8")
PY

  cat > "$CASE_DIR/scan_meta.json" <<JSON
{
  "case": "$CASE_NAME",
  "f_eta": ${F_ETA},
  "L_eta_diff_ref_code": ${L_ETA_DIFF_REF_CODE},
  "gp_L_eta": ${GP_L_ETA},
  "dt": ${DT_CODE},
  "steps": ${NSTEPS},
  "temperature_C": 380.0,
  "xB0": 0.03,
  "gp_W_eta_phys": 6.59167373e8,
  "gp_kappa_eta_phys": 6.82679420e-11,
  "elastic_enabled": 0,
  "gp_obs_target_radius_nm": 1.0,
  "gp_obs_eta_peak": 1.0,
  "gp_obs_depletion_radius_factor": 2.0
}
JSON

  echo "[scan] running $CASE_NAME with f_eta=$F_ETA gp_L_eta=$GP_L_ETA"
  ./main_cuda 96 96 96 "$DT_CODE" 1 1 1 0 \
    --pf-param-file "$CASE_DIR/pf_input.params" \
    --init-case-tag "$CASE_NAME" \
    --radius 0 \
    --xB_matrix 0.03 \
    --nsteps "$NSTEPS" \
    --out-every "$OUT_EVERY" \
    --csv-out-every "$OUT_EVERY" \
    --diag-vtk-enabled 1 \
    > "$CASE_DIR/run.log" 2>&1
done

python3 "$ROOT_DIR/analysis/postprocess_step38d_gp_L_eta_scan.py" \
  --root-result-dir "$RESULT_ROOT" \
  --out-dir "$SCRIPT_OUT" \
  --cases "${CASES[@]}"

echo "[done] step38d gp_L_eta scan complete"
