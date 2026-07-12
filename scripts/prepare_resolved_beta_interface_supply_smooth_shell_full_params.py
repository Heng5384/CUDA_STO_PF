#!/usr/bin/env python3
"""Prepare the full physical-window validation after a smooth-shell smoke."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "params/gp_required_supply_informed_rsmd"
OUT = ROOT / "params/resolved_beta_interface_supply_smooth_shell_full"
MANIFEST = OUT / "manifest.csv"


def parse_params(path: Path) -> tuple[list[str], dict[str, str]]:
    order: list[str] = []
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key not in values:
            order.append(key)
        values[key] = value.strip()
    return order, values


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = SOURCE / "T380_gp_supply_upper_guard_xB0p024_R12_chi1_k1p5_1000post.params"
    order, values = parse_params(base)
    case = "T380_upper_guard_dt0p005_bell_shell4nm_f0p0005_hsrc0p49_full19s"
    values.update({
        "dt": "5.0000000000000001e-03",
        "dt_code": "5.0000000000000001e-03",
        "nsteps": "4160",
        "out_every": "4160",
        "csv_out_every": "50",
        "diagnostic_rsmd_release_window_steps": "4160",
        "diagnostic_rsmd_delivery_mode": "seed_interface_alpha_shell",
        "diagnostic_rsmd_interface_shell_width_nm": "4.0",
        "diagnostic_rsmd_interface_shell_kernel": "compact_bell",
        "diagnostic_rsmd_h_src_max": "0.49",
        "diagnostic_rsmd_f_max_per_step": "5.0000000000000001e-04",
        "diagnostic_rsmd_provenance": "scenario_bracket_not_calibrated",
        "phi_eta_rhs_attribution_diag_enabled": "1",
        "phi_eta_rhs_attribution_diag_every": "10",
        "phi_eta_rhs_attribution_diag_max_steps": "120",
        "phi_eta_rhs_attribution_diag_prefix": case,
        "resolved_handoff_xB_write_mode": "preserve_profile_xB_alpha_in_support",
        "scheduled_nuc_scale_interface_width": "1.0",
        "scheduled_nuc_scale_xB_profile_width": "1.0",
    })
    param_path = OUT / f"{case}.params"
    lines = [
        "# Full-window smooth spatial required-supply relay validation.",
        "# Compact-bell 4 nm matrix-side shell; no direct beta write.",
        "# Scenario diagnostic only, not calibrated GP release thermodynamics.",
        "",
    ]
    lines.extend(f"{key}={values[key]}" for key in order if key in values)
    lines.extend(f"{key}={values[key]}" for key in sorted(values) if key not in order)
    param_path.write_text("\n".join(lines) + "\n")
    row = {
        "case": case,
        "temperature_C": 380,
        "base_param_file": str(base.relative_to(ROOT)),
        "param_file": str(param_path.relative_to(ROOT)),
        "dt_code": 0.005,
        "nsteps": 4160,
        "physical_duration_s_approx": 4160 * 0.005 * 0.9254156720524821,
        "delivery_mode": "seed_interface_alpha_shell",
        "interface_shell_width_nm": 4.0,
        "interface_shell_kernel": "compact_bell",
        "diagnostic_rsmd_f_max_per_step": 0.0005,
        "physical_release_rate_proxy_f_over_dt": 0.1,
        "diagnostic_rsmd_h_src_max": 0.49,
        "xB_target": values["diagnostic_rsmd_xB_halo_target"],
        "R_exchange_nm": values["diagnostic_rsmd_R_exchange_nm"],
        "provenance": "scenario_bracket_not_calibrated",
        "purpose": "full_window_clean_growth_validation",
    }
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row))
        writer.writeheader()
        writer.writerow(row)
    print(f"prepared_cases=1\nmanifest={MANIFEST}")


if __name__ == "__main__":
    main()
