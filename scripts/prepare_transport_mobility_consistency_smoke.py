#!/usr/bin/env python3
"""Prepare early-time controls for the transport-mobility consistency fix."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "params/gp_required_supply_informed_rsmd"
OUT = ROOT / "params/transport_mobility_consistency_smoke"
MANIFEST = OUT / "transport_mobility_consistency_smoke_manifest.csv"

BASE = {
    "T380": "T380_gp_supply_upper_guard_xB0p024_R12_chi1_k1p5_1000post.params",
    "T400": "T400_gp_supply_upper_guard_xB0p024_R16_chi1_k1p5_1000post.params",
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


def write(path: Path, order: list[str], values: dict[str, str]) -> None:
    lines = [
        "# Early-time validation of the numerical transport-mobility consistency fix.",
        "# Same physics and scenario relay as the upper-guard baseline; only cadence differs.",
        "",
    ]
    lines += [f"{key}={values[key]}" for key in order if key in values]
    lines += [f"{key}={values[key]}" for key in sorted(values) if key not in order]
    path.write_text("\n".join(lines) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for temp in ("T380", "T400"):
        for source_mode, fmax, group in (
            ("seed_interface_alpha_shell", "0.0", "no_source"),
            ("seed_interface_alpha_shell", "4.0000000000000002e-04", "upper_guard_source"),
        ):
            order, values = parse(SOURCE / BASE[temp])
            run_id = f"{temp}_{group}_transport_mobility_fix_240steps"
            values.update({
                "dt": "5.0000000000000001e-03",
                "dt_code": "5.0000000000000001e-03",
                "nsteps": "240",
                "out_every": "240",
                "csv_out_every": "10",
                "diagnostic_rsmd_enabled": "1",
                "diagnostic_rsmd_delivery_mode": source_mode,
                "diagnostic_rsmd_f_max_per_step": fmax,
                "diagnostic_rsmd_release_window_steps": "240",
                "diagnostic_rsmd_interface_diag_enabled": "1",
                "diagnostic_rsmd_interface_diag_every": "10",
                "phi_eta_rhs_attribution_diag_enabled": "1",
                "phi_eta_rhs_attribution_diag_every": "10",
                "phi_eta_rhs_attribution_diag_max_steps": "240",
                "phi_eta_rhs_attribution_diag_prefix": run_id,
                "resolved_handoff_xB_write_mode": "preserve_profile_xB_alpha_in_support",
                "scheduled_nuc_scale_interface_width": "1.0",
                "scheduled_nuc_scale_xB_profile_width": "1.0",
                "gp_growth_enabled": "0",
                "gp_inventory_growth_enabled": "0",
            })
            param = OUT / f"{run_id}.params"
            write(param, order, values)
            rows.append({
                "run_id": run_id,
                "temperature_C": int(temp[1:]),
                "group": group,
                "nsteps": 240,
                "source_mode": source_mode,
                "f_max_per_step": fmax,
                "xB_halo_target": values["diagnostic_rsmd_xB_halo_target"],
                "R_exchange_nm": values["diagnostic_rsmd_R_exchange_nm"],
                "seed_library_expected": "nlib_00006" if temp == "T380" else "nlib_dc_T400_xB003",
                "thermo_convex_extrapolation_enabled": values.get("thermo_convex_extrapolation_enabled", "0"),
                "provenance": "numerical_transport_consistency_validation",
                "param_file": str(param.relative_to(ROOT)),
            })
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"prepared_cases={len(rows)}")
    print(f"manifest={MANIFEST}")


if __name__ == "__main__":
    main()
