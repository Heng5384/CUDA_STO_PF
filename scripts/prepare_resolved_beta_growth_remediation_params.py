#!/usr/bin/env python3
"""Prepare small, reproducible no-growth remediation diagnostics.

These cases do not alter the PF equation or GP inventory policy.  They test
whether the existing high-supply outcome is controlled by timestep size and/or
the alpha-side source mask.  All rows retain scenario provenance; they are not
calibrated GP-release thermodynamics.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "params/gp_required_supply_informed_rsmd"
OUT = ROOT / "params/resolved_beta_growth_remediation"
MANIFEST = OUT / "manifest.csv"

N_STEPS_BY_DT = {"0p02": 1040, "0p01": 2080}

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
    for key in order:
        if key in values:
            lines.append(f"{key}={values[key]}")
    for key in sorted(key for key in values if key not in order):
        lines.append(f"{key}={values[key]}")
    path.write_text("\n".join(lines) + "\n")


def make_case(label: str, base_name: str, dt: float, h_src_max: float) -> dict[str, object]:
    order, values = parse_params(SOURCE / base_name)
    nsteps = N_STEPS_BY_DT["0p01" if abs(dt - 0.01) < 1.0e-15 else "0p02"]
    values.update({
        "dt": f"{dt:.16e}",
        "dt_code": f"{dt:.16e}",
        "nsteps": str(nsteps),
        "out_every": str(nsteps),
        "csv_out_every": "50",
        "diagnostic_rsmd_release_window_steps": str(nsteps),
        "diagnostic_rsmd_h_src_max": f"{h_src_max:.16e}",
        "diagnostic_rsmd_provenance": "scenario_bracket_not_calibrated",
        "phi_eta_rhs_attribution_diag_enabled": "1",
        "phi_eta_rhs_attribution_diag_every": "10",
        "phi_eta_rhs_attribution_diag_max_steps": "120",
        "phi_eta_rhs_attribution_diag_prefix": label,
        "resolved_handoff_xB_write_mode": "preserve_profile_xB_alpha_in_support",
        "scheduled_nuc_scale_interface_width": "1.0",
        "scheduled_nuc_scale_xB_profile_width": "1.0",
    })
    path = OUT / f"{label}.params"
    header = f"""# Resolved-beta no-growth remediation diagnostic
# Base: {base_name}
# Case: {label}
# Purpose: timestep / alpha-side source-mask sensitivity at the pre-existing upper_guard target.
# Source remains matrix-side only; this is not calibrated GP release thermodynamics.
"""
    write_params(path, order, values, header)
    return {
        "case": label,
        "temperature_C": int(label[1:4]),
        "base_param_file": str((SOURCE / base_name).relative_to(ROOT)),
        "param_file": str(path.relative_to(ROOT)),
        "dt_code": dt,
        "nsteps": nsteps,
        "physical_duration_s_approx": nsteps * dt * 0.9254156720524821,
        "diagnostic_rsmd_h_src_max": h_src_max,
        "xB_target": values["diagnostic_rsmd_xB_halo_target"],
        "R_exchange_nm": values["diagnostic_rsmd_R_exchange_nm"],
        "chi_rel": values["diagnostic_rsmd_chi_rel"],
        "provenance": "scenario_bracket_not_calibrated",
        "purpose": "dt_and_alpha_side_source_mask_sensitivity",
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for temp, base in BASES.items():
        for dt_tag, dt in (("dt0p02", 0.02), ("dt0p01", 0.01)):
            for mask_tag, h_src_max in (("hsrc0p10", 0.10), ("hsrc0p49", 0.49)):
                rows.append(make_case(f"{temp}_upper_guard_{dt_tag}_{mask_tag}", base, dt, h_src_max))
    fields = list(rows[0])
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"prepared_cases={len(rows)}")
    print(f"manifest={MANIFEST}")


if __name__ == "__main__":
    main()
