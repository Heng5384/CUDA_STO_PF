#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="${ROOT_DIR:-/home/zhiheng/PF/CUDA_STO_PF}"
cd "$ROOT_DIR"

BASE_PARAMS="${BASE_PARAMS:-$ROOT_DIR/.tmp_step38f_base_pf.params}"
L_ETA_DIFF_REF_CODE="${L_ETA_DIFF_REF_CODE:-4.7495061735e+01}"
DT_CODE="${DT_CODE:-1e-4}"
TEMPERATURE_C="${TEMPERATURE_C:-380}"
XB0="${XB0:-0.03}"
RESULT_ROOT="${RESULT_ROOT:-$ROOT_DIR/Results/ch_T380_cuda_96x96x96_phi_eta_rhs_attribution}"
SCRIPT_OUT="${SCRIPT_OUT:-$ROOT_DIR/reports/data/step38f_phi_eta_rhs_attribution}"

mkdir -p "$RESULT_ROOT" "$SCRIPT_OUT"

if [[ ! -f "$BASE_PARAMS" ]]; then
  python3 "$ROOT_DIR/Unit_Psedobinary.py" \
    --input-json "$ROOT_DIR/physical_inputs.example.json" \
    --output-pf-param-file "$BASE_PARAMS"
fi

declare -a CASE_DIRS=()

run_case() {
  local case_name="$1"
  local f_eta="$2"
  local nsteps="$3"
  local out_every="$4"

  local case_dir="$RESULT_ROOT/$case_name"
  mkdir -p "$case_dir"
  CASE_DIRS+=("$case_dir")

  local gp_L_eta
  gp_L_eta=$(python3 - <<PY
f = float("${f_eta}")
Lref = float("${L_ETA_DIFF_REF_CODE}")
print("{:.12e}".format(f * Lref))
PY
)

  python3 - <<PY
from pathlib import Path

base = Path("${BASE_PARAMS}")
out = Path("${case_dir}") / "pf_input.params"
lines = base.read_text(encoding="utf-8").splitlines()
drop_prefixes = {
    "gp_W_eta=","gp_kappa_eta=","gp_W_eta_phys=","gp_kappa_eta_phys=",
    "gp_W_eta_code=","gp_kappa_eta_code=","gp_L_eta=",
    "gp_elastic_enabled=","gp_elastic_active_eta=","gp_elastic_active_phi=",
    "gp_eps_iso=","gp_to_beta_enabled=","gp_nuc_enabled=",
    "y_update_mass_projection_enabled=","gp_y_update_mode=","gp_init_mode=",
    "gp_obs_target_radius_nm=","gp_obs_eta_peak=","gp_obs_iface_width_nm=",
    "gp_obs_profile_type=","gp_obs_match_mode=","gp_obs_compensation_mode=",
    "gp_obs_depletion_radius_factor=","gp_obs_depletion_smooth_width_factor=",
    "gp_obs_min_xB_alpha=","gp_obs_max_xB_alpha=","gp_drive_reference_mode=",
    "gp_delta_g_stab=","gp_xB_fixed=","gp_raw_reaction_drive_only=",
    "thermo_convex_extrapolation_enabled=","diag_vtk_enabled=","dt=",
    "temperature_C=","ic_23d_xB_out=","ic_vf_target_phi=","ic_vf_init_phi=",
    "phi_eta_rhs_attribution_diag_enabled=","phi_eta_rhs_attribution_diag_every=",
    "phi_eta_rhs_attribution_diag_max_steps=","phi_eta_rhs_attribution_diag_prefix=",
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
    "temperature_C=${TEMPERATURE_C}",
    "dt=${DT_CODE}",
    "gp_L_eta=${gp_L_eta}",
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
    "ic_23d_xB_out=${XB0}",
    "diag_vtk_enabled=0",
    "phi_eta_rhs_attribution_diag_enabled=1",
    "phi_eta_rhs_attribution_diag_every=1",
    "phi_eta_rhs_attribution_diag_max_steps=${nsteps}",
    "phi_eta_rhs_attribution_diag_prefix=phi_eta_rhs_attribution",
]
out.write_text("\n".join(kept + append) + "\n", encoding="utf-8")
PY

  cat > "$case_dir/scan_meta.json" <<JSON
{
  "case": "$case_name",
  "f_eta": ${f_eta},
  "L_eta_diff_ref_code": ${L_ETA_DIFF_REF_CODE},
  "gp_L_eta": ${gp_L_eta},
  "dt": ${DT_CODE},
  "steps": ${nsteps},
  "temperature_C": ${TEMPERATURE_C},
  "xB0": ${XB0}
}
JSON

  echo "[step38f] running $case_name f_eta=$f_eta gp_L_eta=$gp_L_eta"
  ./main_cuda 96 96 96 "$DT_CODE" 1 1 1 0 \
    --pf-param-file "$case_dir/pf_input.params" \
    --init-case-tag "$case_name" \
    --radius 0 \
    --xB_matrix "$XB0" \
    --nsteps "$nsteps" \
    --out-every "$out_every" \
    --csv-out-every "$out_every" \
    > "$case_dir/run.log" 2>&1
}

run_case "gp_eta_rhs_attr_f1em5" "1e-5" "500" "100"
run_case "gp_eta_rhs_attr_f1em4" "1e-4" "500" "100"
run_case "gp_eta_rhs_attr_f1em3" "1e-3" "50" "10"

python3 "$ROOT_DIR/analysis/postprocess_phi_eta_rhs_attribution.py" \
  --cases "${CASE_DIRS[@]}" \
  --out-dir "$SCRIPT_OUT"

echo "[done] step38f phi-vs-eta rhs attribution diagnostics complete"
