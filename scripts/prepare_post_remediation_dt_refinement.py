#!/usr/bin/env python3
"""Prepare a timestep-only stability refinement for interface-shell RSMD tests.

The four rows preserve the selected dynamic-continued seed, AQ GP realization,
upper-guard ceiling, reservoir radius, inventory policy, and all PF terms. The
source-on rows use dt=0.001 with five times as many steps, so their total code
and mapped physical duration match the 1040-step dt=0.005 controls. Because
the release limit is defined per step, its value is reduced in the same ratio
to preserve the original physical-time release ceiling; matched source-off rows
add a short PF-only tail.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "params" / "gp_required_supply_informed_rsmd"
OUT = ROOT / "params" / "post_remediation_dt_refinement"
MANIFEST = OUT / "post_remediation_dt_refinement_manifest.csv"

CASES = {
    "T380": {
        "base": "T380_gp_supply_upper_guard_xB0p024_R12_chi1_k1p5_1000post.params",
        "seed": "nlib_00006",
        "R_exchange_nm": 12.0,
    },
    "T400": {
        "base": "T400_gp_supply_upper_guard_xB0p024_R16_chi1_k1p5_1000post.params",
        "seed": "nlib_dc_T400_xB003",
        "R_exchange_nm": 16.0,
    },
}

DT_CODE = 1.0e-3
SOURCE_ON_STEPS = 5200
SOURCE_OFF_TAIL_STEPS = 1000
PHYSICAL_SECONDS_PER_CODE_TIME = 0.9254156720524821
REFERENCE_DT_CODE = 5.0e-3
REFERENCE_F_MAX_PER_STEP = 4.0e-4
F_MAX_PER_STEP = REFERENCE_F_MAX_PER_STEP * DT_CODE / REFERENCE_DT_CODE


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


def write_params(path: Path, order: list[str], values: dict[str, str]) -> None:
    header = [
        "# Timestep-only interface-shell RSMD stability refinement.",
        "# Scenario relay only: no calibrated GP release thermodynamics.",
        "# All physics and inventory settings match the upper-guard short control.",
        "# Per-step release is reduced to preserve the dt=0.005 physical-time rate.",
        "",
    ]
    body = [f"{key}={values[key]}" for key in order if key in values]
    body.extend(f"{key}={values[key]}" for key in sorted(values) if key not in order)
    path.write_text("\n".join(header + body) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for temp, meta in CASES.items():
        base = SOURCE / str(meta["base"])
        order, values = parse_params(base)
        for group, nsteps, release_steps in (
            ("source_on", SOURCE_ON_STEPS, SOURCE_ON_STEPS),
            ("source_off_tail", SOURCE_ON_STEPS + SOURCE_OFF_TAIL_STEPS, SOURCE_ON_STEPS),
        ):
            run_id = f"{temp}_{group}_upper_guard_dt0p001_n{nsteps}_stability_refinement"
            case_values = dict(values)
            case_values.update({
                "dt": f"{DT_CODE:.16e}",
                "dt_code": f"{DT_CODE:.16e}",
                "nsteps": str(nsteps),
                "out_every": str(nsteps),
                "csv_out_every": "50",
                "diagnostic_rsmd_enabled": "1",
                "diagnostic_rsmd_delivery_mode": "seed_interface_alpha_shell",
                "diagnostic_rsmd_interface_shell_width_nm": "4.0",
                "diagnostic_rsmd_interface_shell_kernel": "compact_bell",
                "diagnostic_rsmd_interface_diag_enabled": "1",
                "diagnostic_rsmd_interface_diag_every": "100",
                "diagnostic_rsmd_h_src_max": "0.49",
                "diagnostic_rsmd_f_max_per_step": f"{F_MAX_PER_STEP:.16e}",
                "diagnostic_rsmd_release_window_steps": str(release_steps),
                "diagnostic_rsmd_provenance": "scenario_bracket_not_calibrated",
                "phi_eta_rhs_attribution_diag_enabled": "1",
                "phi_eta_rhs_attribution_diag_every": "200",
                "phi_eta_rhs_attribution_diag_max_steps": str(nsteps),
                "phi_eta_rhs_attribution_diag_prefix": run_id,
                "resolved_handoff_xB_write_mode": "preserve_profile_xB_alpha_in_support",
                "scheduled_nuc_scale_interface_width": "1.0",
                "scheduled_nuc_scale_xB_profile_width": "1.0",
                "gp_growth_enabled": "0",
                "gp_inventory_growth_enabled": "0",
            })
            param_path = OUT / f"{run_id}.params"
            write_params(param_path, order, case_values)
            rows.append({
                "run_id": run_id,
                "temperature_C": int(temp[1:]),
                "group": group,
                "param_file": str(param_path.relative_to(ROOT)),
                "dt_code": DT_CODE,
                "nsteps": nsteps,
                "release_window_steps": release_steps,
                "code_time": DT_CODE * nsteps,
                "physical_duration_s": DT_CODE * nsteps * PHYSICAL_SECONDS_PER_CODE_TIME,
                "scenario": "upper_guard",
                "delivery_mode": "seed_interface_alpha_shell",
                "xB_halo_target": case_values["diagnostic_rsmd_xB_halo_target"],
                "R_exchange_nm": meta["R_exchange_nm"],
                "f_max_per_step": case_values["diagnostic_rsmd_f_max_per_step"],
                "seed_library_expected": meta["seed"],
                "provenance": "scenario_bracket_not_calibrated",
                "purpose": "dt_only_stability_refinement_matched_source_on_and_tail",
            })
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"prepared_cases={len(rows)}")
    print(f"manifest={MANIFEST}")


if __name__ == "__main__":
    main()
