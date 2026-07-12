#!/usr/bin/env python3
"""Prepare the T400 counterpart of the source-history synchronization smoke."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = (
    ROOT
    / "params/rsmd_effective_relay_rate_escalation/"
    "T400_source_on_upper_guard_chi4_dt0p001_n5200_relay_escalation.params"
)
OUT = ROOT / "params/rsmd_y_solver_attribution"
RUN_ID = "T400_rsmd_source_history_sync_dt0p001_n450"


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


def write_case(order: list[str], base: dict[str, str], run_id: str,
               overrides: dict[str, str]) -> Path:
    values = dict(base)
    values.update(overrides)
    values["init_case_tag"] = run_id
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / f"{run_id}.params"
    header = [
        "# T400 source-history synchronization smoke.",
        "# Static GP reservoir; no direct beta write or GP growth.",
        "# Only the stale local dY/dt history is reset after an external source transaction.",
        "",
    ]
    body = [f"{key}={values[key]}" for key in order if key in values]
    body.extend(f"{key}={values[key]}" for key in sorted(values) if key not in order)
    path.write_text("\n".join(header + body) + "\n")
    return path


def main() -> None:
    if not SOURCE.exists():
        raise SystemExit(f"missing T400 base: {SOURCE}")
    order, values = parse(SOURCE)
    common = {
        "csv_out_every": "25",
        "diagnostic_rsmd_interface_diag_every": "25",
        "diagnostic_rsmd_reset_Y_history_after_source": "1",
        "resolved_handoff_xB_write_mode": "preserve_profile_xB_alpha_in_support",
        "scheduled_nuc_scale_interface_width": "1.0",
        "scheduled_nuc_scale_xB_profile_width": "1.0",
        "gp_growth_enabled": "0",
        "gp_inventory_growth_enabled": "0",
    }
    cases = [
        (RUN_ID, {**common, "nsteps": "450", "out_every": "450",
                  "diagnostic_rsmd_release_window_steps": "450"}),
        ("T400_rsmd_source_history_sync_convex_guard_dt0p001_n450",
         {**common, "nsteps": "450", "out_every": "450",
          "diagnostic_rsmd_release_window_steps": "450",
          "thermo_convex_extrapolation_enabled": "1"}),
        ("T400_rsmd_source_history_sync_convex_guard_dt0p001_n1000",
         {**common, "nsteps": "1000", "out_every": "1000",
          "diagnostic_rsmd_release_window_steps": "1000",
          "thermo_convex_extrapolation_enabled": "1"}),
        ("T400_rsmd_source_history_sync_chi2_dt0p0005_n1000",
         {**common, "dt": "5.0e-4", "dt_code": "5.0e-4",
          "nsteps": "1000", "out_every": "1000",
          "diagnostic_rsmd_release_window_steps": "1000",
          # Keep the diagnostic source flux per code-time constant when dt is halved.
          "diagnostic_rsmd_f_max_per_step": "4.0e-5",
          "diagnostic_rsmd_chi_rel": "2.0",
          "thermo_convex_extrapolation_enabled": "0"}),
    ]
    for run_id, overrides in cases:
        print(write_case(order, values, run_id, overrides))


if __name__ == "__main__":
    main()
