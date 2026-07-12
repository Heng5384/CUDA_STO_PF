#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARAM_DIR = ROOT / "params/beta_capacity_gated_handoff"


def parse_params(path: Path) -> tuple[list[str], dict[str, str]]:
    order: list[str] = []
    vals: dict[str, str] = {}
    for line in path.read_text().splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        k, v = s.split("=", 1)
        k = k.strip()
        vals[k] = v.strip()
        if k not in order:
            order.append(k)
    return order, vals


def write_params(path: Path, order: list[str], vals: dict[str, str], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [header.rstrip(), ""]
    for k in order:
        if k in vals:
            lines.append(f"{k}={vals[k]}")
    extras = sorted(k for k in vals if k not in order)
    if extras:
        lines.append("")
        lines.append("# Additional diagnostic overrides")
        for k in extras:
            lines.append(f"{k}={vals[k]}")
    path.write_text("\n".join(lines) + "\n")


def make_case(base: str, out: str, T: int, scaled: bool, nsteps: int) -> None:
    order, vals = parse_params(PARAM_DIR / base)
    scale = "6.6666666667" if scaled else "1.0"
    target_lambda = "4.0" if scaled else "0.6"
    common = {
        "temperature_C": f"{float(T):.16e}",
        "ic_23d_xB_out": "3.0000000000000000e-02",
        "ic_xB_eq_matrix": "3.0000000000000000e-02",
        "gp_initial_xB_tot": "3.0000000000000000e-02",
        "gp_birth_model": "prescribed_sites",
        "gp_literature_model_enabled": "0",
        "gp_population_source": "diagnostic_xB03_prescribed_epsilon_marker",
        "gp_initial_population_enabled": "0",
        "gp_initial_population_source": "none",
        "gp_initial_mass_mode": "matrix_composition_fixed",
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
        "enable_legacy_gp_storage_coupling": "0",
        "gp_smooth_depletion_enabled": "0",
        "gp_birth_connect_to_smooth_local_depletion": "0",
        "gp_literature_birth_requires_post_Y_projection": "0",
        "gp_debug_xB_max": "1.0",
        "gp_runtime_s_gp_scalar": "0.5",
        "gp_stochastic_S_GP": "0.5",
        "beta_debug_force_single_event": "1",
        "beta_debug_force_step": "10",
        "beta_debug_position_mode": "max_capacity_near_GP",
        "beta_debug_inventory_mode": "full_physical_seed",
        "beta_debug_do_not_reduce_requested_mass": "1",
        "beta_debug_max_events_total": "1",
        "beta_debug_matrix_draw_radius_nm": "50",
        "beta_debug_GP_capture_mode": "none",
        "beta_debug_GP_capture_radius_nm": "0",
        "beta_capacity_gate_GP_capture_mode": "none",
        "beta_capacity_gate_GP_capture_radius_nm": "0",
        "beta_capacity_gate_matrix_draw_radius_nm": "50",
        "beta_capacity_gate_max_reasonable_radius_nm": "50",
        "beta_capacity_gate_allow_direct_if_capacity_ratio_ge": "1000000000",
        "beta_handoff_policy": "staged_GP_to_beta_conversion",
        "beta_staged_accumulation_enabled": "1",
        "beta_staged_accumulation_interval_steps": "10",
        "beta_staged_accumulation_GP_capture_radius_nm": "0",
        "beta_staged_accumulation_matrix_draw_radius_nm": "50",
        "beta_staged_accumulation_max_fraction_per_step": "1.0",
        "beta_staged_accumulation_max_inventory_per_step": "1.0e300",
        "beta_staged_debug_accelerated_accumulation": "1",
        "beta_staged_debug_accumulation_rate_multiplier": "1000000",
        "beta_staged_debug_stop_after_resolved_insert": "0",
        "beta_staged_insert_when_target_reached": "1",
        "resolved_handoff_xB_write_mode": "preserve_profile_xB_alpha_in_support",
        "scheduled_nuc_source_lambda_nm": "0.6",
        "scheduled_nuc_target_lambda_nm": target_lambda,
        "scheduled_nuc_scale_interface_width": scale,
        "scheduled_nuc_scale_xB_profile_width": scale,
        "post_conversion_y_update_audit_enabled": "1",
        "post_conversion_y_update_audit_steps": "4",
        "y_update_k0_audit_enabled": "1",
        "y_update_k0_audit_steps": "4",
        "y_update_mass_projection_report_enabled": "1",
    }
    vals.update(common)
    mode = "scaled" if scaled else "unscaled"
    header = f"""# Diagnostic xB_far=0.03 matrix-only counterfactual for PASS_4GRID_RUNTIME_XB03_FARFIELD_MATRIX_GROWTH_TEST
# Base: {base}
# Runtime PF interface: dx=1 nm, lambda_sm_m=4e-9 m, ic_phi_iface_w=2.0, kappa_phi=2.0
# Matrix mode: uniform xB_alpha=0.03, AQ GP population disabled, epsilon prescribed marker only as forced-event anchor
# Profile mode: {mode}; suggested nsteps={nsteps}
"""
    write_params(PARAM_DIR / out, order, vals, header)


def main() -> None:
    make_case(
        "T400_4grid_runtime_interface_unscaled_handoff_diag.params",
        "T400_4grid_xB03_matrix_unscaled_handoff_diag.params",
        400,
        False,
        1040,
    )
    make_case(
        "T400_4grid_runtime_interface_scaled_handoff_diag.params",
        "T400_4grid_xB03_matrix_scaled_handoff_diag.params",
        400,
        True,
        300,
    )
    make_case(
        "T380_4grid_runtime_interface_scaled_handoff_diag.params",
        "T380_4grid_xB03_matrix_unscaled_handoff_diag.params",
        380,
        False,
        1040,
    )
    make_case(
        "T380_4grid_runtime_interface_scaled_handoff_diag.params",
        "T380_4grid_xB03_matrix_scaled_handoff_diag.params",
        380,
        True,
        300,
    )


if __name__ == "__main__":
    main()
