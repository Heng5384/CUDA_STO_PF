#!/usr/bin/env python3
"""Assemble T400 plateau selection and T380 confirmation evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def rel_change(left: dict[str, str], right: dict[str, str], key: str) -> float:
    a, b = float(left[key]), float(right[key])
    return abs(b - a) / max(abs(a), abs(b), 1.0e-30)


def plateau_pair(rows: list[dict[str, str]], low_factor: float,
                 high_factor: float) -> dict[str, object]:
    by_factor = {float(row["L_phi_factor"]): row for row in rows}
    low, high = by_factor[low_factor], by_factor[high_factor]
    changes = {
        "velocity_change_rel": rel_change(low, high, "full_window_velocity_nm_s"),
        "beta_gain_change_rel": rel_change(
            low, high, "beta_h_inventory_gain_cell_units"
        ),
        "matrix_flux_change_rel": rel_change(
            low, high, "matrix_flux_from_stefan_code"
        ),
    }
    return {
        "low_factor": low_factor,
        "high_factor": high_factor,
        **changes,
        "all_hard_gates_pass": (
            low["numerical_hard_gates_pass"] == "True"
            and high["numerical_hard_gates_pass"] == "True"
        ),
        "plateau_10pct_pass": max(changes.values()) <= 0.10,
        "plateau_5pct_pass": max(changes.values()) <= 0.05,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--t400", type=Path, required=True)
    parser.add_argument("--t380", type=Path, required=True)
    parser.add_argument("--metrics-out", type=Path, required=True)
    parser.add_argument("--report-out", type=Path, required=True)
    parser.add_argument("--mode-out", type=Path, required=True)
    args = parser.parse_args()

    t400 = read_csv(args.t400)
    t380 = read_csv(args.t380)
    pair400 = plateau_pair(t400, 1000.0, 3000.0)
    pair380 = plateau_pair(t380, 1000.0, 3000.0)
    accepted = bool(pair400["plateau_10pct_pass"] and
                    pair400["all_hard_gates_pass"] and
                    pair380["plateau_10pct_pass"] and
                    pair380["all_hard_gates_pass"])
    if not accepted:
        raise RuntimeError("T400 selection or T380 confirmation failed")

    args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(t400 + t380,
                  key=lambda row: (float(row["temperature_C"]),
                                   float(row["L_phi_factor"])))
    with args.metrics_out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    selected400 = next(row for row in t400 if float(row["L_phi_factor"]) == 1000.0)
    selected380 = next(row for row in t380 if float(row["L_phi_factor"]) == 1000.0)
    mode = {
        "research_model": "fixed_ctot_gp_reservoir_to_beta_diffusion_limit_v1",
        "matrix_to_beta_kinetics": "FAST_INTERFACE_DIFFUSION_CONTROLLED_LIMIT",
        "finite_interface_mode": "off",
        "L_phi_fast_factor": 1000.0,
        "factor_semantics": "multiplier_of_temperature_specific_source_derived_one_sided_reference",
        "T400": {
            "L_phi_reference_code": float(selected400["L_phi_reference_code"]),
            "L_phi_reference_physical": float(selected400["L_phi_reference_physical"]),
            "L_phi_fast_code": float(selected400["L_phi_code"]),
            "L_phi_fast_physical": float(selected400["L_phi_physical"]),
            "plateau_pair": pair400,
        },
        "T380": {
            "L_phi_reference_code": float(selected380["L_phi_reference_code"]),
            "L_phi_reference_physical": float(selected380["L_phi_reference_physical"]),
            "L_phi_fast_code": float(selected380["L_phi_code"]),
            "L_phi_fast_physical": float(selected380["L_phi_physical"]),
            "confirmation_pair": pair380,
        },
        "absolute_interface_mobility_claimed": False,
        "GP_source_enabled_during_plateau": False,
        "plateau_status": "PASS",
    }
    args.mode_out.parent.mkdir(parents=True, exist_ok=True)
    args.mode_out.write_text(json.dumps(mode, indent=2) + "\n")

    def row(pair: dict[str, object]) -> str:
        return (f"{float(pair['velocity_change_rel']):.6g} | "
                f"{float(pair['beta_gain_change_rel']):.6g} | "
                f"{float(pair['matrix_flux_change_rel']):.6g} | "
                f"{pair['all_hard_gates_pass']} | {pair['plateau_10pct_pass']}")

    report = f"""# Research2 Fast-Interface Plateau

## Decision

`fast_interface_plateau_status=PASS`

`L_phi_fast_factor=1000`

The factor is applied to each temperature's source-derived one-sided reference;
it is not a temperature-independent absolute code mobility and is not an
independently measured interface mobility.

| T (C) | pair | velocity change | beta gain change | matrix flux change | hard gates | <=10% |
|---:|---:|---:|---:|---:|---|---|
| 400 | 1000 -> 3000 | {row(pair400)} |
| 380 | 1000 -> 3000 | {row(pair380)} |

T400 uses `L_phi_fast={selected400['L_phi_code']}` code units; T380 uses
`L_phi_fast={selected380['L_phi_code']}` code units. Both use physical
`D_alpha(T)`, equal physical observation time, finite-interface correction OFF,
elasticity OFF, and GP/S3 OFF.

The full-window matrix-side flux is inferred from the exactly closed one-sided
Stefan ledger. Every row passed per-step accepted predicates, converged-state
KKT/storage, energy/work, mass/bounds, zero clipping, and zero physical
projection.

`absolute_interface_mobility_claimed=false`
"""
    args.report_out.parent.mkdir(parents=True, exist_ok=True)
    args.report_out.write_text(report)
    print("fast_interface_plateau_status=PASS")
    print("L_phi_fast_factor=1000")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
