#!/usr/bin/env python3
"""Prepare the fixed-rate RSMD history and dt validation matrix."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "params" / "rsmd_post_history_sync_validation"
BASE = {
    380: ROOT / "params/rsmd_source_history_sync_extended/T380_rsmd_source_history_sync_chi4_dt0p002_n500_continuous.params",
    400: ROOT / "params/rsmd_source_history_sync_extended/T400_rsmd_source_history_sync_chi2_dt0p002_n500_continuous.params",
}


def parse(path: Path) -> tuple[list[str], dict[str, str]]:
    order: list[str] = []
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = (part.strip() for part in line.split("=", 1))
        if key not in values:
            order.append(key)
        values[key] = value
    return order, values


def write_case(T: int, run_id: str, overrides: dict[str, str]) -> Path:
    order, values = parse(BASE[T])
    values.update(overrides)
    values["init_case_tag"] = run_id
    lines = [
        "# RSMD post-history-sync validation; scenario bracket, not calibrated kinetics.",
        "# Static GP reservoir; no direct phi/beta write; no GP thermodynamic claim.",
        "",
    ]
    lines.extend(f"{key}={values[key]}" for key in order)
    lines.extend(f"{key}={values[key]}" for key in sorted(values) if key not in order)
    path = OUT / f"{run_id}.params"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    common = {
        "diagnostic_rsmd_enabled": "1",
        "diagnostic_rsmd_xB_halo_target": "0.024",
        "diagnostic_rsmd_delivery_mode": "seed_interface_alpha_shell",
        "diagnostic_rsmd_interface_shell_width_nm": "4.0",
        "diagnostic_rsmd_interface_shell_kernel": "compact_bell",
        "diagnostic_rsmd_h_src_max": "0.49",
        "diagnostic_rsmd_kernel_radius_dx": "1.5",
        "diagnostic_rsmd_provenance": "scenario_bracket_not_calibrated",
        "diagnostic_rsmd_interface_diag_enabled": "1",
        "resolved_handoff_xB_write_mode": "preserve_profile_xB_alpha_in_support",
        "scheduled_nuc_scale_interface_width": "1.0",
        "scheduled_nuc_scale_xB_profile_width": "1.0",
        "gp_growth_enabled": "0",
        "gp_radius_evolution_enabled": "0",
        "gp_inventory_growth_enabled": "0",
        "enable_legacy_gp_storage_coupling": "0",
        "phi_eta_rhs_attribution_diag_enabled": "1",
        "dynamics_mass_diag_enabled": "1",
        "dynamics_mass_diag_interval": "1",
        "out_every": "1000000",
    }
    conditions = {
        380: {"chi": "2.0", "R": "12.0", "seed_R": "5.358726"},
        400: {"chi": "1.125", "R": "16.0", "seed_R": "4.735881"},
    }
    rows: list[dict[str, str]] = []

    # The source starts after the controlled staged handoff near step 40.
    # Each dt case retains 0.92 code-time units after handoff and f_step/dt=0.08.
    dt_matrix = [(0.001, 960, 0.00008, 50), (0.002, 500, 0.00016, 25),
                 (0.005, 224, 0.00040, 10)]
    for T, cond in conditions.items():
        for dt, steps, f_step, diag_every in dt_matrix:
            dt_tag = str(dt).replace("0.", "0p")
            run_id = f"T{T}_dt{dt_tag}_history_local_zero_restart"
            overrides = {
                **common,
                "temperature_C": str(T),
                "diagnostic_rsmd_T_only": str(T),
                "diagnostic_rsmd_chi_rel": cond["chi"],
                "diagnostic_rsmd_R_exchange_nm": cond["R"],
                "diagnostic_rsmd_seed_R_eff_h_nm": cond["seed_R"],
                "diagnostic_rsmd_reset_Y_history_after_source": "1",
                "diagnostic_rsmd_history_restart_mode": "1",
                "diagnostic_rsmd_f_max_per_step": f"{f_step:.8g}",
                "diagnostic_rsmd_release_window_steps": str(round(0.92 / dt)),
                "dt": f"{dt:.8g}",
                "dt_code": f"{dt:.8g}",
                "nsteps": str(steps),
                "csv_out_every": str(diag_every),
                "diagnostic_rsmd_interface_diag_every": str(diag_every),
                "phi_eta_rhs_attribution_diag_every": str(diag_every),
                "phi_eta_rhs_attribution_diag_max_steps": str(steps),
            }
            path = write_case(T, run_id, overrides)
            rows.append({
                "case": run_id, "phase": "dt_convergence", "T_C": str(T),
                "dt": str(dt), "nsteps": str(steps), "chi_rel": cond["chi"],
                "f_max_per_step": str(f_step), "f_over_dt": str(f_step / dt),
                "history_mode": "1_local_zero_restart", "param_file": str(path),
                "extra_args": "",
            })

        # Mode 0 and Mode 1 are additional controls at dt=0.002. Mode 2 reuses
        # the dt-convergence row above.
        for mode, reset, extra in (
            ("0_stale_history", "0", ""),
            ("1_local_zero_restart", "1", ""),
        ):
            run_id = f"T{T}_dt0p002_history_{mode}"
            overrides = {
                **common,
                "temperature_C": str(T),
                "diagnostic_rsmd_T_only": str(T),
                "diagnostic_rsmd_chi_rel": cond["chi"],
                "diagnostic_rsmd_R_exchange_nm": cond["R"],
                "diagnostic_rsmd_seed_R_eff_h_nm": cond["seed_R"],
                "diagnostic_rsmd_reset_Y_history_after_source": reset,
                "diagnostic_rsmd_history_restart_mode": reset,
                "diagnostic_rsmd_f_max_per_step": "0.00016",
                "diagnostic_rsmd_release_window_steps": "460",
                "dt": "0.002", "dt_code": "0.002", "nsteps": "500",
                "csv_out_every": "25", "diagnostic_rsmd_interface_diag_every": "25",
                "phi_eta_rhs_attribution_diag_every": "25",
                "phi_eta_rhs_attribution_diag_max_steps": "500",
            }
            path = write_case(T, run_id, overrides)
            rows.append({
                "case": run_id, "phase": "history_scheme", "T_C": str(T),
                "dt": "0.002", "nsteps": "500", "chi_rel": cond["chi"],
                "f_max_per_step": "0.00016", "f_over_dt": "0.08",
                "history_mode": mode, "param_file": str(path), "extra_args": extra,
            })
        run_id = f"T{T}_dt0p002_history_2_rhs_mask_first_order_restart"
        overrides = {
            **common,
            "temperature_C": str(T),
            "diagnostic_rsmd_T_only": str(T),
            "diagnostic_rsmd_chi_rel": cond["chi"],
            "diagnostic_rsmd_R_exchange_nm": cond["R"],
            "diagnostic_rsmd_seed_R_eff_h_nm": cond["seed_R"],
            "diagnostic_rsmd_reset_Y_history_after_source": "1",
            "diagnostic_rsmd_history_restart_mode": "2",
            "diagnostic_rsmd_f_max_per_step": "0.00016",
            "diagnostic_rsmd_release_window_steps": "460",
            "dt": "0.002", "dt_code": "0.002", "nsteps": "500",
            "csv_out_every": "25", "diagnostic_rsmd_interface_diag_every": "25",
            "phi_eta_rhs_attribution_diag_every": "25",
            "phi_eta_rhs_attribution_diag_max_steps": "500",
        }
        path = write_case(T, run_id, overrides)
        rows.append({
            "case": run_id, "phase": "history_scheme", "T_C": str(T),
            "dt": "0.002", "nsteps": "500", "chi_rel": cond["chi"],
            "f_max_per_step": "0.00016", "f_over_dt": "0.08",
            "history_mode": "2_rhs_mask_first_order_restart", "param_file": str(path),
            "extra_args": "",
        })

    with (OUT / "run_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT / "run_manifest.csv")


if __name__ == "__main__":
    main()
