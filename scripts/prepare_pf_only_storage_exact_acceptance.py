#!/usr/bin/env python3
"""Prepare equal-time PF-only storage-exact remediation cases."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "params/rsmd_equal_time_domain_low_overshoot"
OUT = ROOT / "params/pf_only_baseline_closure/remediation"


def parse(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        text = raw.strip()
        if text and not text.startswith("#") and "=" in text:
            key, value = text.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for temp in (400, 380):
        for dt_tag in ("0p0005", "0p001", "0p002"):
            source = BASE / f"T{temp}_PFonly_dt{dt_tag}_equal_time.params"
            values = parse(source)
            dt = float(values["dt"])
            diagnostic_every = max(1, round(0.02 / dt))
            tag = f"T{temp}_PFBASE_storage_weighted_dt{dt_tag}"
            values.update({
                "init_case_tag": tag,
                "pf_baseline_control_mode": "full",
                "pf_y_update_mode": "x_transport_projection_split",
                "pf_matrix_storage_floor": "0.1",
                "diagnostic_rsmd_enabled": "1",
                "diagnostic_rsmd_f_max_per_step": "0.0",
                "diagnostic_rsmd_chi_rel": "0.0",
                "diagnostic_rsmd_control_mode": "full_coupled",
                "diagnostic_rsmd_interface_diag_enabled": "1",
                "diagnostic_rsmd_interface_diag_every": str(diagnostic_every),
                "y_update_mass_projection_enabled": "1",
                "y_update_mass_projection_report_enabled": "1",
                "disable_Y_rhs_gamma_term": "0",
                "enable_Y_rhs_picard": "0",
                "dynamics_mass_diag_enabled": "1",
                "dynamics_mass_diag_interval": str(diagnostic_every),
            })
            target = OUT / f"{tag}.params"
            target.write_text(
                "# PF-only same-step storage remediation; S3 mass source is frozen off.\n" +
                "\n".join(f"{key}={value}" for key, value in values.items()) + "\n"
            )
            records.append({
                "case": tag, "T_C": temp, "dt": dt, "nsteps": int(values["nsteps"]),
                "post_handoff_code_time": 0.92,
                "post_handoff_physical_time_s": 0.92 * float(values["t_real_unit_s"]),
                "fix_A": "storage_weighted_semi_implicit_matrix_transport",
                "fix_B": "protected_halpha_floor_plus_full_storage_projection",
                "S3_mass_source": "OFF_FROZEN", "param_file": str(target),
            })
    with (OUT / "acceptance_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]))
        writer.writeheader(); writer.writerows(records)
    print(f"prepared={len(records)}")


if __name__ == "__main__":
    main()
