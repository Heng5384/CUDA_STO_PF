#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zhiheng/PF/CUDA_STO_PF}"
cd "$ROOT_DIR"

BASE_PARAMS="${BASE_PARAMS:-$ROOT_DIR/Results/ch_T380_cuda_96x96x96_dt0.0001_steps100_xB0.030/step38h_gp_eta_rhs_attr_f1em5_scaled/pf_input.params}"
DT_CODE="${DT_CODE:-1e-4}"
NSTEPS="${NSTEPS:-500}"
OUT_EVERY="${OUT_EVERY:-100}"
CSV_OUT_EVERY="${CSV_OUT_EVERY:-100}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/Results/ch_T380_cuda_96x96x96_dt${DT_CODE}_steps${NSTEPS}_xB0.030_step38j}"
POSTPROCESS_ROOT="${POSTPROCESS_ROOT:-$ROOT_DIR/Results/ch_T380_cuda_96x96x96_dt0.0001_steps${NSTEPS}_xB0.030}"
SCRIPT_OUT="${SCRIPT_OUT:-$ROOT_DIR/reports/data/step38j_eta_S218b_scan}"

mkdir -p "$RESULT_ROOT" "$SCRIPT_OUT"

MRATIO_LIST=("1e-4" "1e-3" "1e-2" "1e-1" "1" "10")
CASES=()

make_case_name() {
  local r="$1"
  case "$r" in
    1e-4) echo "gp_eta_S218b_Mratio_1em4" ;;
    1e-3) echo "gp_eta_S218b_Mratio_1em3" ;;
    1e-2) echo "gp_eta_S218b_Mratio_1em2" ;;
    1e-1) echo "gp_eta_S218b_Mratio_1em1" ;;
    1) echo "gp_eta_S218b_Mratio_1" ;;
    10) echo "gp_eta_S218b_Mratio_10" ;;
    *) echo "gp_eta_S218b_Mratio_${r}" | tr '+.' '__' ;;
  esac
}

for MRATIO in "${MRATIO_LIST[@]}"; do
  CASE_NAME="$(make_case_name "$MRATIO")"
  CASE_DIR="$RESULT_ROOT/$CASE_NAME"
  mkdir -p "$CASE_DIR"
  CASES+=("$CASE_NAME")

  python3 - <<PY
from pathlib import Path

base = Path("${BASE_PARAMS}")
out = Path("${CASE_DIR}") / "pf_input.params"
lines = base.read_text(encoding="utf-8").splitlines()

drop_prefixes = {
    "gp_W_eta=", "gp_kappa_eta=",
    "gp_W_eta_phys=", "gp_kappa_eta_phys=",
    "gp_W_eta_code=", "gp_kappa_eta_code=",
    "gp_L_eta=", "gp_L_eta_mode=",
    "gp_M_eta_phys=", "gp_M_eta_ratio_to_crit=", "gp_M_int_eta=",
    "gp_elastic_enabled=", "gp_elastic_active_eta=", "gp_elastic_active_phi=",
    "gp_eps_iso=", "gp_to_beta_enabled=", "gp_nuc_enabled=",
    "y_update_mass_projection_enabled=", "gp_y_update_mode=",
    "gp_init_mode=", "gp_obs_target_radius_nm=", "gp_obs_eta_peak=",
    "gp_obs_iface_width_nm=", "gp_obs_profile_type=", "gp_obs_match_mode=",
    "gp_obs_compensation_mode=", "gp_obs_depletion_radius_factor=",
    "gp_obs_depletion_smooth_width_factor=", "gp_obs_min_xB_alpha=",
    "gp_obs_max_xB_alpha=", "gp_drive_reference_mode=", "gp_delta_g_stab=",
    "gp_xB_fixed=", "gp_raw_reaction_drive_only=", "thermo_convex_extrapolation_enabled=",
    "diag_vtk_enabled=", "dt=", "temperature_C=", "ic_23d_xB_out=",
    "ic_vf_target_phi=", "ic_vf_init_phi=",
    "phi_eta_rhs_attribution_diag_enabled=", "phi_eta_step_delta_diag_enabled=",
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
    "gp_xB_fixed=0.35",
    "gp_delta_g_stab=0.0",
    "gp_raw_reaction_drive_only=1",
    "gp_raw_reaction_drive_use_raw_units_debug=0",
    "gp_L_eta_mode=sto_S218b",
    "gp_M_eta_ratio_to_crit=${MRATIO}",
    "temperature_C=380",
    "dt=${DT_CODE}",
    "gp_gamma_alpha_gp=5.0e-2",
    "gp_l_eta_nm=1.0",
    "gp_D_ratio=1.0",
    "gp_xB_eq_alpha_for_eta=0.03",
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
    "phi_eta_rhs_attribution_diag_enabled=1",
    "phi_eta_rhs_attribution_diag_every=1",
    "phi_eta_rhs_attribution_diag_max_steps=100",
    "phi_eta_rhs_attribution_diag_prefix=step38j_rhs_attr",
]

out.write_text("\n".join(kept + append) + "\n", encoding="utf-8")
PY

  cat > "$CASE_DIR/scan_meta.json" <<JSON
{
  "case": "$CASE_NAME",
  "M_ratio": ${MRATIO},
  "dt": ${DT_CODE},
  "steps": ${NSTEPS},
  "temperature_C": 380.0,
  "xB0": 0.03,
  "gp_gamma_alpha_gp": 5.0e-2,
  "gp_l_eta_nm": 1.0,
  "gp_L_eta_mode": "sto_S218b"
}
JSON

  echo "[step38j] running $CASE_NAME with M_ratio=$MRATIO"
  ./main_cuda 96 96 96 "$DT_CODE" 1 1 1 0 \
    --pf-param-file "$CASE_DIR/pf_input.params" \
    --init-case-tag "$CASE_NAME" \
    --radius 0 \
    --xB_matrix 0.03 \
    --nsteps "$NSTEPS" \
    --out-every "$OUT_EVERY" \
    --csv-out-every "$CSV_OUT_EVERY" \
    --diag-vtk-enabled 1 \
    > "$CASE_DIR/run.log" 2>&1
done

python3 "$ROOT_DIR/analysis/postprocess_step38j_eta_S218b_Mratio_scan.py" \
  --root-result-dir "$POSTPROCESS_ROOT" \
  --out-dir "$SCRIPT_OUT" \
  --cases "${CASES[@]}"

echo "[done] step38j eta S218b M-ratio smoke scan complete"
