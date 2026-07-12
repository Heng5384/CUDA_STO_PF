#!/usr/bin/env python3
"""Prepare bounded GP-to-alpha-interface relay diagnostics.

This is deliberately not a GP release thermodynamic law.  It tests whether
inventory already held by GPs that meet the existing R_exchange criterion can
stabilize a beta seed when delivered to the alpha-side seed interface instead
of deposited around each remote GP center.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "params/gp_required_supply_informed_rsmd"
OUT = ROOT / "params/resolved_beta_interface_supply_remediation"
MANIFEST = OUT / "manifest.csv"

BASES = {
    "T380": "T380_gp_supply_upper_guard_xB0p024_R12_chi1_k1p5_1000post.params",
    "T400": "T400_gp_supply_upper_guard_xB0p024_R16_chi1_k1p5_1000post.params",
}


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


def write_params(path: Path, order: list[str], values: dict[str, str], header: str) -> None:
    lines = [header.rstrip(), ""]
    lines.extend(f"{key}={values[key]}" for key in order if key in values)
    lines.extend(f"{key}={values[key]}" for key in sorted(values) if key not in order)
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for temp, base_name in BASES.items():
        order, values = parse_params(SOURCE / base_name)
        label = f"{temp}_upper_guard_dt0p01_interface_shell2nm_hsrc0p49"
        values.update({
            "dt": "1.0000000000000000e-02",
            "dt_code": "1.0000000000000000e-02",
            "nsteps": "2080",
            "out_every": "2080",
            "csv_out_every": "50",
            "diagnostic_rsmd_release_window_steps": "2080",
            "diagnostic_rsmd_delivery_mode": "seed_interface_alpha_shell",
            "diagnostic_rsmd_interface_shell_width_nm": "2.0",
            "diagnostic_rsmd_h_src_max": "0.49",
            "diagnostic_rsmd_provenance": "scenario_bracket_not_calibrated",
            "phi_eta_rhs_attribution_diag_enabled": "1",
            "phi_eta_rhs_attribution_diag_every": "10",
            "phi_eta_rhs_attribution_diag_max_steps": "120",
            "phi_eta_rhs_attribution_diag_prefix": label,
            "resolved_handoff_xB_write_mode": "preserve_profile_xB_alpha_in_support",
            "scheduled_nuc_scale_interface_width": "1.0",
            "scheduled_nuc_scale_xB_profile_width": "1.0",
        })
        param_path = OUT / f"{label}.params"
        write_params(param_path, order, values, f"""# Bounded GP-to-interface required-supply relay diagnostic
# Base: {base_name}
# Case: {label}
# Existing eligible GP inventory is deposited only into the matrix-side alpha
# interface shell. This is scenario evidence, not calibrated GP thermodynamics.
""")
        rows.append({
            "case": label,
            "temperature_C": int(temp[1:]),
            "base_param_file": str((SOURCE / base_name).relative_to(ROOT)),
            "param_file": str(param_path.relative_to(ROOT)),
            "dt_code": 0.01,
            "nsteps": 2080,
            "physical_duration_s_approx": 2080 * 0.01 * 0.9254156720524821,
            "delivery_mode": "seed_interface_alpha_shell",
            "interface_shell_width_nm": 2.0,
            "diagnostic_rsmd_h_src_max": 0.49,
            "xB_target": values["diagnostic_rsmd_xB_halo_target"],
            "R_exchange_nm": values["diagnostic_rsmd_R_exchange_nm"],
            "chi_rel": values["diagnostic_rsmd_chi_rel"],
            "provenance": "scenario_bracket_not_calibrated",
            "physical_claim": "required_supply_relay_diagnostic_not_calibrated_GP_thermodynamics",
        })
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"prepared_cases={len(rows)}")
    print(f"manifest={MANIFEST}")


if __name__ == "__main__":
    main()
