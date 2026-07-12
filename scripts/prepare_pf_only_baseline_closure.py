#!/usr/bin/env python3
"""Prepare workstation-only PF baseline operator-decomposition cases."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "params/rsmd_equal_time_domain_low_overshoot/T400_PFonly_dt0p002_equal_time.params"
OUT = ROOT / "params/pf_only_baseline_closure"


CASES = [
    ("P0_full", "full", 0, 1, 0, "full PF baseline"),
    ("P1_frozen_phi", "frozen_phi", 0, 1, 0, "Y transport plus projection, phi frozen"),
    ("P2_transport_no_projection", "transport_no_projection", 0, 0, 0,
     "Y transport only, phi frozen, projection bypassed"),
    ("P3_projection_only", "projection_only", 0, 1, 0,
     "projection only after exact composition restore, phi frozen"),
    ("P4_frozen_phi_history_off", "frozen_phi", 1, 1, 0,
     "Y transport plus projection, phi frozen, lagged gamma term off"),
    ("P5_full_history_off", "full", 1, 1, 0,
     "full PF with lagged gamma term off"),
    ("P6_phi_only", "phi_only", 0, 0, 0,
     "phi only with composition restored and projection bypassed"),
    ("P9_frozen_phi_picard16", "frozen_phi", 0, 1, 16,
     "diagnostic within-step Picard, 16 iterations"),
    ("P10_full_picard16", "full", 0, 1, 16,
     "full PF diagnostic within-step Picard, 16 iterations"),
]


def parse_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if line and not line.startswith("#") and "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def write_params(path: Path, values: dict[str, str]) -> None:
    path.write_text("# PF-only baseline numerical-closure diagnostic. No RSMD mass source.\n" +
                    "\n".join(f"{key}={value}" for key, value in values.items()) + "\n")


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    base = parse_params(BASE)
    rows: list[dict[str, object]] = []
    for name, mode, history_off, projection, picard, notes in CASES:
        values = dict(base)
        values.update({
            "dt": "0.002", "dt_code": "0.002", "nsteps": "500",
            "csv_out_every": "10", "out_every": "500",
            "init_case_tag": f"T400_PFBASE_{name}_dt0p002",
            "diagnostic_rsmd_enabled": "1",
            "diagnostic_rsmd_f_max_per_step": "0.0",
            "diagnostic_rsmd_chi_rel": "0.0",
            "diagnostic_rsmd_release_window_steps": "460",
            "diagnostic_rsmd_interface_diag_enabled": "1",
            "diagnostic_rsmd_interface_diag_every": "10",
            "diagnostic_rsmd_control_mode": "full_coupled",
            "pf_baseline_control_mode": mode,
            "disable_Y_rhs_gamma_term": str(history_off),
            "y_update_mass_projection_enabled": str(projection),
            "y_update_mass_projection_report_enabled": "1",
            "dynamics_mass_diag_enabled": "1",
            "dynamics_mass_diag_interval": "10",
            "enable_Y_rhs_picard": "1" if picard else "0",
            "Y_rhs_picard_iters": str(picard if picard else 1),
            "Y_rhs_picard_omega": "1.0",
        })
        path = OUT / f"T400_{name}_dt0p002.params"
        write_params(path, values)
        rows.append({
            "case": values["init_case_tag"], "operator_case": name,
            "T_C": 400, "dt": 0.002, "nsteps": 500,
            "post_handoff_code_time": 0.92,
            "post_handoff_physical_time_s": 0.92 * float(values["t_real_unit_s"]),
            "pf_baseline_control_mode": mode,
            "history_gamma_disabled": history_off,
            "projection_enabled": projection,
            "picard_iters": picard,
            "RSMD_mass_source_enabled": False,
            "param_file": str(path), "notes": notes,
        })
    rows.extend([
        {**rows[1], "case": "P8_shared_with_P1", "operator_case": "P8",
         "param_file": "SHARED_WITH_P1", "notes": "h frozen at resolved initial profile; identical to P1"},
        {**rows[6], "case": "P7_shared_with_P6", "operator_case": "P7",
         "param_file": "SHARED_WITH_P6", "notes": "phi plus fixed composition; identical to P6"},
    ])
    with (OUT / "operator_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    dense = dict(base)
    dense.update({
        "dt": "0.002", "dt_code": "0.002", "nsteps": "120",
        "csv_out_every": "1", "out_every": "120",
        "init_case_tag": "T400_PFBASE_P0_full_dense_trace_dt0p002",
        "diagnostic_rsmd_enabled": "1",
        "diagnostic_rsmd_f_max_per_step": "0.0",
        "diagnostic_rsmd_chi_rel": "0.0",
        "diagnostic_rsmd_release_window_steps": "80",
        "diagnostic_rsmd_interface_diag_enabled": "1",
        "diagnostic_rsmd_interface_diag_every": "1",
        "diagnostic_rsmd_control_mode": "full_coupled",
        "pf_baseline_control_mode": "full",
        "pf_y_update_mode": "lagged_rhs",
        "disable_Y_rhs_gamma_term": "0",
        "y_update_mass_projection_enabled": "1",
        "y_update_mass_projection_report_enabled": "1",
        "dynamics_mass_diag_enabled": "1",
        "dynamics_mass_diag_interval": "1",
        "enable_Y_rhs_picard": "0",
    })
    write_params(OUT / "T400_P0_full_dense_trace_dt0p002.params", dense)
    print(f"prepared_cases={len(CASES)}")


if __name__ == "__main__":
    main()
