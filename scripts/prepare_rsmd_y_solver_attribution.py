#!/usr/bin/env python3
"""Prepare workstation-only attribution cases for the post-handoff Y update.

These cases retain the resolved dynamic-continued seed, static GP reservoir,
and scenario-only interface-shell source.  They vary only existing numerical
controls of the Y update to identify whether a local xB bound hit originates
in the explicit gamma*dY/dt closure or the high-xB thermo extrapolation.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "params/rsmd_effective_relay_rate_escalation/"
    "T380_source_on_upper_guard_chi4_dt0p001_n5200_relay_escalation.params"
)
OUT = ROOT / "params/rsmd_y_solver_attribution"
MANIFEST = OUT / "rsmd_y_solver_attribution_manifest.csv"
N_STEPS = 450

CASES: tuple[tuple[str, str, dict[str, str]], ...] = (
    (
        "T380_y_rhs_explicit_gamma_control_dt0p001_n450",
        "Existing production-equation explicit gamma*dYdt history control.",
        {},
    ),
    (
        "T380_y_rhs_picard8_dt0p001_n450",
        "Same equation with eight fixed-point iterations for gamma*dYdt closure.",
        {
            "enable_Y_rhs_picard": "1",
            "Y_rhs_picard_iters": "8",
            "Y_rhs_picard_omega": "1.0",
        },
    ),
    (
        "T380_y_rhs_gamma_disabled_diagnostic_dt0p001_n450",
        "Diagnostic-only removal of gamma*dYdt to attribute the instability.",
        {"disable_Y_rhs_gamma_term": "1"},
    ),
    (
        "T380_thermo_convex_guard_dt0p001_n450",
        "Existing high-xB convex extrapolation guard; inactive in the physical halo range.",
        {"thermo_convex_extrapolation_enabled": "1"},
    ),
    (
        "T380_rsmd_source_history_sync_dt0p001_n450",
        "Resets only the stale lagged dY/dt history after each external RSMD source transaction.",
        {"diagnostic_rsmd_reset_Y_history_after_source": "1"},
    ),
)


def parse(path: Path) -> tuple[list[str], dict[str, str]]:
    order: list[str] = []
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key, value = key.strip(), value.strip()
        if key not in values:
            order.append(key)
        values[key] = value
    return order, values


def write(path: Path, order: list[str], values: dict[str, str], description: str) -> None:
    header = [
        "# Workstation-only Y-solver attribution diagnostic.",
        "# Same GP/CNT/seed/inventory physics as the chi=4 scenario bracket.",
        "# This file changes only numerical Y-update controls noted below.",
        f"# {description}",
        "",
    ]
    body = [f"{key}={values[key]}" for key in order if key in values]
    body.extend(f"{key}={values[key]}" for key in sorted(values) if key not in order)
    path.write_text("\n".join(header + body) + "\n")


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"missing base parameter file: {SOURCE}")
    OUT.mkdir(parents=True, exist_ok=True)
    order, base = parse(SOURCE)
    rows: list[dict[str, object]] = []
    for run_id, purpose, overrides in CASES:
        values = dict(base)
        values.update(
            {
                "nsteps": str(N_STEPS),
                "out_every": str(N_STEPS),
                "csv_out_every": "25",
                "diagnostic_rsmd_release_window_steps": str(N_STEPS),
                "diagnostic_rsmd_interface_diag_every": "25",
                "dynamics_mass_diag_enabled": "1",
                "dynamics_mass_diag_interval": "1",
                "phi_eta_rhs_attribution_diag_enabled": "1",
                "phi_eta_rhs_attribution_diag_every": "25",
                "phi_eta_rhs_attribution_diag_max_steps": str(N_STEPS),
                "phi_eta_rhs_attribution_diag_prefix": run_id,
                "init_case_tag": run_id,
                "gp_growth_enabled": "0",
                "gp_inventory_growth_enabled": "0",
            }
        )
        values.update(overrides)
        path = OUT / f"{run_id}.params"
        write(path, order, values, purpose)
        rows.append(
            {
                "run_id": run_id,
                "param_file": str(path.relative_to(ROOT)),
                "nsteps": N_STEPS,
                "temperature_C": 380,
                "dt_code": values["dt"],
                "chi_rel": values["diagnostic_rsmd_chi_rel"],
                "xB_halo_target": values["diagnostic_rsmd_xB_halo_target"],
                "enable_Y_rhs_picard": values.get("enable_Y_rhs_picard", "0"),
                "Y_rhs_picard_iters": values.get("Y_rhs_picard_iters", "0"),
                "disable_Y_rhs_gamma_term": values.get("disable_Y_rhs_gamma_term", "0"),
                "thermo_convex_extrapolation_enabled": values.get(
                    "thermo_convex_extrapolation_enabled", "0"
                ),
                "diagnostic_rsmd_reset_Y_history_after_source": values.get(
                    "diagnostic_rsmd_reset_Y_history_after_source", "0"
                ),
                "purpose": purpose,
                "provenance": "numerical_attribution_not_physics_recalibration",
            }
        )
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"prepared_cases={len(rows)}")
    print(f"manifest={MANIFEST}")


if __name__ == "__main__":
    main()
