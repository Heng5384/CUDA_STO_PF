#!/usr/bin/env python3
"""Pre-register a stronger, explicitly scenario-only GP relay-rate bracket.

The preceding dt=0.001 chi=1 refinement tests the original physical-time
release ceiling.  These cases retain the same seed, halo ceiling, static GP
reservoirs, and PF/CNT model, while multiplying only the *scenario relay rate*
by chi_rel=4.  It is deliberately not a GP thermodynamic calibration.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "params" / "gp_required_supply_informed_rsmd"
OUT = ROOT / "params" / "rsmd_effective_relay_rate_escalation"
MANIFEST = OUT / "rsmd_effective_relay_rate_escalation_manifest.csv"

CASES = {
    "T380": {
        "base": "T380_gp_supply_upper_guard_xB0p024_R12_chi1_k1p5_1000post.params",
        "seed": "nlib_00006", "exchange": 12.0,
    },
    "T400": {
        "base": "T400_gp_supply_upper_guard_xB0p024_R16_chi1_k1p5_1000post.params",
        "seed": "nlib_dc_T400_xB003", "exchange": 16.0,
    },
}

DT = 1.0e-3
F_MAX_BASE_DT001 = 8.0e-5
CHI = 4.0
SOURCE_ON_STEPS = 5200
SOURCE_OFF_TAIL_STEPS = 1000
TIME_UNIT_S = 0.9254156720524821


def parse(path: Path) -> tuple[list[str], dict[str, str]]:
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


def write(path: Path, order: list[str], values: dict[str, str]) -> None:
    header = [
        "# Effective GP-to-matrix relay-rate escalation; scenario-only diagnostic.",
        "# No GP solvus, release thermodynamics, direct beta write, or PF/CNT change.",
        "# chi_rel=4 multiplies the dt=0.001 physical relay-rate bracket only.",
        "",
    ]
    body = [f"{key}={values[key]}" for key in order if key in values]
    body.extend(f"{key}={values[key]}" for key in sorted(values) if key not in order)
    path.write_text("\n".join(header + body) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for temp, meta in CASES.items():
        order, base_values = parse(SOURCE / str(meta["base"]))
        for group, nsteps, release_steps in (
            ("source_on", SOURCE_ON_STEPS, SOURCE_ON_STEPS),
            ("source_off_tail", SOURCE_ON_STEPS + SOURCE_OFF_TAIL_STEPS, SOURCE_ON_STEPS),
        ):
            run_id = f"{temp}_{group}_upper_guard_chi4_dt0p001_n{nsteps}_relay_escalation"
            values = dict(base_values)
            values.update({
                "dt": f"{DT:.16e}", "dt_code": f"{DT:.16e}",
                "nsteps": str(nsteps), "out_every": str(nsteps), "csv_out_every": "50",
                "diagnostic_rsmd_enabled": "1",
                "diagnostic_rsmd_delivery_mode": "seed_interface_alpha_shell",
                "diagnostic_rsmd_interface_shell_width_nm": "4.0",
                "diagnostic_rsmd_interface_shell_kernel": "compact_bell",
                "diagnostic_rsmd_interface_diag_enabled": "1",
                "diagnostic_rsmd_interface_diag_every": "100",
                "diagnostic_rsmd_h_src_max": "0.49",
                "diagnostic_rsmd_f_max_per_step": f"{F_MAX_BASE_DT001:.16e}",
                "diagnostic_rsmd_chi_rel": f"{CHI:.16e}",
                "diagnostic_rsmd_release_window_steps": str(release_steps),
                "diagnostic_rsmd_provenance": "scenario_bracket_not_calibrated",
                "phi_eta_rhs_attribution_diag_enabled": "1",
                "phi_eta_rhs_attribution_diag_every": "200",
                "phi_eta_rhs_attribution_diag_max_steps": str(nsteps),
                "phi_eta_rhs_attribution_diag_prefix": run_id,
                "resolved_handoff_xB_write_mode": "preserve_profile_xB_alpha_in_support",
                "scheduled_nuc_scale_interface_width": "1.0",
                "scheduled_nuc_scale_xB_profile_width": "1.0",
                "gp_growth_enabled": "0", "gp_inventory_growth_enabled": "0",
            })
            param_path = OUT / f"{run_id}.params"
            write(param_path, order, values)
            rows.append({
                "run_id": run_id, "temperature_C": int(temp[1:]), "group": group,
                "param_file": str(param_path.relative_to(ROOT)), "dt_code": DT,
                "nsteps": nsteps, "release_window_steps": release_steps,
                "code_time": nsteps * DT,
                "physical_duration_s": nsteps * DT * TIME_UNIT_S,
                "xB_halo_target": values["diagnostic_rsmd_xB_halo_target"],
                "R_exchange_nm": meta["exchange"],
                "f_max_per_step": F_MAX_BASE_DT001, "chi_rel": CHI,
                "effective_physical_relay_rate_multiplier": CHI,
                "seed_library_expected": meta["seed"],
                "provenance": "scenario_bracket_not_calibrated",
                "purpose": "effective_required_supply_rate_bracket_not_calibrated_GP_release",
            })
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"prepared_cases={len(rows)}")
    print(f"manifest={MANIFEST}")


if __name__ == "__main__":
    main()
