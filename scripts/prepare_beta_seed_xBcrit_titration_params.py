#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARAM_DIR = ROOT / "params/beta_seed_xBcrit_titration"
BASE_DIR = ROOT / "params/beta_capacity_gated_handoff"
XB_GRID = [0.0078305391025, 0.010, 0.012, 0.014, 0.016, 0.018, 0.020, 0.024, 0.030]
BASE_FILES = {
    380: "T380_S05_capacity_gated_smoke_1000.params",
    400: "T400_S05_capacity_gated_smoke_1000.params",
}


def parse_params(path: Path) -> tuple[list[str], dict[str, str]]:
    order: list[str] = []
    vals: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        vals[key] = value.strip()
        if key not in order:
            order.append(key)
    return order, vals


def write_params(path: Path, order: list[str], vals: dict[str, str], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [header.rstrip(), ""]
    for key in order:
        if key in vals:
            lines.append(f"{key}={vals[key]}")
    extras = [key for key in vals if key not in order]
    if extras:
        lines.append("")
        lines.append("# beta_seed_xBcrit_titration diagnostic overrides")
        for key in sorted(extras):
            lines.append(f"{key}={vals[key]}")
    path.write_text("\n".join(lines) + "\n")


def fmt_xb(xb: float) -> str:
    return f"{xb:.12f}".rstrip("0").rstrip(".").replace(".", "p")


def make_case(T: int, xb: float) -> Path:
    order, vals = parse_params(BASE_DIR / BASE_FILES[T])
    xb_txt = f"{xb:.16e}"
    overrides = {
        "temperature_C": f"{float(T):.16e}",
        "ic_23d_xB_out": xb_txt,
        "ic_xB_eq_matrix": xb_txt,
        "gp_initial_xB_tot": xb_txt,
        "gp_literature_model_enabled": "0",
        "gp_birth_model": "prescribed_sites",
        "gp_population_source": "diagnostic_xBcrit_prescribed_epsilon_marker",
        "gp_literature_xAg_mode": "from_current_mean_xB_alpha",
        "gp_initial_mass_mode": "matrix_composition_fixed",
        "gp_smooth_depletion_enabled": "0",
        "gp_birth_connect_to_smooth_local_depletion": "0",
        "gp_initial_population_enabled": "0",
        "gp_initial_population_source": "none",
        "gp_site_mode": "single",
        "gp_n_sites": "1",
        "gp_debug_site_ix": "64",
        "gp_debug_site_iy": "64",
        "gp_debug_site_iz": "64",
        "gp_site_B_mass_equiv": "1.0e-300",
        "gp_site_S_factor": "0.5",
        "gp_growth_enabled": "0",
        "gp_radius_evolution_enabled": "0",
        "gp_inventory_growth_enabled": "0",
        "gp_literature_birth_requires_post_Y_projection": "0",
        "enable_legacy_gp_storage_coupling": "0",
        "gp_runtime_s_gp_scalar": "0.5",
        # Diagnostic titration scans matrix xB while holding the selected
        # dynamic-continued seed fixed for each temperature. This tolerance is
        # intentionally broad so low-xB uniform-matrix points do not reject the
        # T380/T400 production seed by catalog-composition mismatch.
        "gp_runtime_catalog_T_tol_C": "1.0",
        "gp_runtime_catalog_xB_tol": "1.0",
        "gp_stochastic_S_GP": "0.5",
        "beta_debug_force_single_event": "1",
        "beta_debug_force_step": "10",
        "beta_debug_position_mode": "max_capacity_near_GP",
        "beta_debug_inventory_mode": "full_physical_seed",
        "beta_debug_capacity_fraction": "0.5",
        "beta_debug_max_events_total": "1",
        "beta_debug_do_not_reduce_requested_mass": "1",
        "beta_debug_matrix_draw_radius_nm": "50",
        "beta_debug_GP_capture_mode": "none",
        "beta_debug_GP_capture_radius_nm": "0",
        "beta_handoff_policy": "staged_GP_to_beta_conversion",
        "beta_capacity_gate_enabled": "1",
        "beta_capacity_gate_matrix_draw_radius_nm": "50",
        "beta_capacity_gate_GP_capture_mode": "none",
        "beta_capacity_gate_GP_capture_radius_nm": "0",
        "beta_capacity_gate_max_reasonable_radius_nm": "50",
        "beta_capacity_gate_allow_direct_if_capacity_ratio_ge": "1000000000",
        "beta_staged_conversion_enabled": "1",
        "beta_staged_conversion_target": "dynamic_continue_seed_mass",
        "beta_staged_conversion_initial_inventory_mode": "consume_available_GP_and_matrix",
        "beta_staged_conversion_release_mode": "inventory_accumulation",
        "beta_staged_conversion_insert_when_capacity_reached": "1",
        "beta_staged_conversion_max_subgrid_steps": "100000",
        "beta_staged_conversion_mass_tolerance_rel": "1e-10",
        "resolved_handoff_xB_write_mode": "preserve_profile_xB_alpha_in_support",
        "scheduled_nuc_source_lambda_nm": "0.6",
        "scheduled_nuc_target_lambda_nm": "0.6",
        "scheduled_nuc_scale_interface_width": "1.0",
        "scheduled_nuc_scale_xB_profile_width": "1.0",
        "beta_staged_accumulation_enabled": "1",
        "beta_staged_accumulation_interval_steps": "10",
        "beta_staged_accumulation_GP_capture_radius_nm": "0",
        "beta_staged_accumulation_matrix_draw_radius_nm": "50",
        "beta_staged_accumulation_max_fraction_per_step": "1.0",
        "beta_staged_accumulation_max_inventory_per_step": "1.0e300",
        "beta_staged_insert_when_target_reached": "1",
        "beta_staged_debug_accelerated_accumulation": "1",
        "beta_staged_debug_accumulation_rate_multiplier": "1000000",
        "beta_staged_debug_stop_after_resolved_insert": "0",
        "analytic_fallback_used": "0",
    }
    vals.update(overrides)
    out = PARAM_DIR / f"T{T}_xB{fmt_xb(xb)}_stable_unscaled_noAQ.params"
    header = f"""# Diagnostic beta seed xBcrit titration parameter file
# Base: {BASE_FILES[T]}
# T_C={T}
# uniform matrix xB_alpha={xb:.12e}
# Stable baseline: dx=1 nm, lambda_sm=0.6 nm, scale_phi=1, scale_xB=1
# AQ GP population/J_GP/RSMD/release disabled; epsilon marker only anchors staged handoff.
"""
    write_params(out, order, vals, header)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperatures", default="380,400")
    parser.add_argument("--xbs", default=",".join(f"{x:.12g}" for x in XB_GRID))
    args = parser.parse_args()
    temps = [int(x.strip()) for x in args.temperatures.split(",") if x.strip()]
    xbs = [float(x.strip()) for x in args.xbs.split(",") if x.strip()]
    manifest = PARAM_DIR / "manifest.csv"
    rows = ["T_C,xB_uniform,param_file"]
    for T in temps:
        for xb in xbs:
            path = make_case(T, xb)
            rows.append(f"{T},{xb:.12e},{path.relative_to(ROOT)}")
            print(path.relative_to(ROOT))
    manifest.write_text("\n".join(rows) + "\n")


if __name__ == "__main__":
    main()
