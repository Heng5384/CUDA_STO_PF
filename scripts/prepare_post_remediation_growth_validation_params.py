#!/usr/bin/env python3
"""Pre-register the workstation post-remediation GP-to-beta validation matrix.

The matrix changes no PF, seed, or inventory physics.  Within each temperature
the control and remediated rows share the same seed library, AQ-GP realization,
ceiling, exchange radius, chi, timestep, and output cadence.  Only the source
placement mode differs between the matched short controls.
"""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "params/gp_required_supply_informed_rsmd"
OUT = ROOT / "params/post_remediation_growth_validation"
MANIFEST = OUT / "post_remediation_validation_manifest.csv"

BASES = {
    "T380": {
        "nominal": "T380_gp_supply_nominal_xB0p0118638113025_R12_chi1_k1p5_1000post.params",
        "upper_guard": "T380_gp_supply_upper_guard_xB0p024_R12_chi1_k1p5_1000post.params",
    },
    "T400": {
        "nominal": "T400_gp_supply_nominal_xB0p0184841486239_R16_chi1_k1p5_1000post.params",
        "upper_guard": "T400_gp_supply_upper_guard_xB0p024_R16_chi1_k1p5_1000post.params",
    },
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


def make_case(temp: str, kind: str, scenario: str, nsteps: int, release_steps: int) -> dict[str, object]:
    base = SOURCE / BASES[temp][scenario]
    order, values = parse_params(base)
    case = f"{temp}_{kind}_{scenario}_dt0p005_n{nsteps}"
    if kind == "source_off_tail":
        case += f"_release{release_steps}"
    source_mode = "seed_interface_alpha_shell"
    f_max = 4.0e-4
    if kind == "no_source":
        f_max = 0.0
    elif kind == "pre_gp_centered":
        source_mode = "gp_centered_kernel"
    values.update({
        "dt": "5.0000000000000001e-03",
        "dt_code": "5.0000000000000001e-03",
        "nsteps": str(nsteps),
        "out_every": str(nsteps),
        "csv_out_every": "50",
        "diagnostic_rsmd_enabled": "1",
        "diagnostic_rsmd_delivery_mode": source_mode,
        "diagnostic_rsmd_interface_shell_width_nm": "4.0",
        "diagnostic_rsmd_interface_shell_kernel": "compact_bell",
        "diagnostic_rsmd_interface_diag_enabled": "1",
        "diagnostic_rsmd_interface_diag_every": "100",
        "diagnostic_rsmd_h_src_max": "0.49",
        "diagnostic_rsmd_f_max_per_step": f"{f_max:.16e}",
        "diagnostic_rsmd_release_window_steps": str(release_steps),
        "diagnostic_rsmd_provenance": "scenario_bracket_not_calibrated",
        "phi_eta_rhs_attribution_diag_enabled": "1",
        "phi_eta_rhs_attribution_diag_every": "200",
        "phi_eta_rhs_attribution_diag_max_steps": str(nsteps),
        "phi_eta_rhs_attribution_diag_prefix": case,
        "resolved_handoff_xB_write_mode": "preserve_profile_xB_alpha_in_support",
        "scheduled_nuc_scale_interface_width": "1.0",
        "scheduled_nuc_scale_xB_profile_width": "1.0",
        "gp_growth_enabled": "0",
        "gp_inventory_growth_enabled": "0",
    })
    path = OUT / f"{case}.params"
    write_params(
        path,
        order,
        values,
        "# Pre-registered post-remediation growth validation.\n"
        "# Scenario-only GP reservoir relay; no calibrated GP release thermodynamics.\n"
        "# No direct phi/beta write, no matrix reset, no profile scaling.",
    )
    return {
        "run_id": case,
        "temperature_C": int(temp[1:]),
        "group": kind,
        "scenario": scenario,
        "base_param_file": str(base.relative_to(ROOT)),
        "param_file": str(path.relative_to(ROOT)),
        "nsteps": nsteps,
        "release_window_steps": release_steps,
        "physical_duration_s": nsteps * 0.005 * 0.9254156720524821,
        "delivery_mode": source_mode,
        "f_max_per_step": f_max,
        "xB_halo_target": values["diagnostic_rsmd_xB_halo_target"],
        "R_exchange_nm": values["diagnostic_rsmd_R_exchange_nm"],
        "chi_rel": values["diagnostic_rsmd_chi_rel"],
        "seed_library_expected": "nlib_00006" if temp == "T380" else "nlib_dc_T400_xB003",
        "provenance": "scenario_bracket_not_calibrated",
        "pre_registered_purpose": {
            "no_source": "matched no-release control",
            "pre_gp_centered": "matched legacy GP-centred source control",
            "post_interface": "remediated interface-shell source response",
            "extended_interface": "longer remediated source-on window",
            "source_off_tail": "source-on then fixed-field PF tail",
        }[kind],
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for temp in ("T380", "T400"):
        rows.extend([
            make_case(temp, "no_source", "upper_guard", 1040, 1040),
            make_case(temp, "pre_gp_centered", "upper_guard", 1040, 1040),
            make_case(temp, "post_interface", "nominal", 1040, 1040),
            make_case(temp, "post_interface", "upper_guard", 1040, 1040),
            make_case(temp, "extended_interface", "nominal", 4160, 4160),
            make_case(temp, "source_off_tail", "nominal", 4160, 2080),
        ])
    with MANIFEST.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"prepared_cases={len(rows)}")
    print(f"manifest={MANIFEST}")


if __name__ == "__main__":
    main()
