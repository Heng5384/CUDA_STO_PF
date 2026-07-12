#!/usr/bin/env python3
"""Prepare rate-preserving extended RSMD history-sync validation cases."""

from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "params" / "rsmd_source_history_sync_extended"


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


def write(source: Path, run_id: str, overrides: dict[str, str]) -> Path:
    order, values = parse(source)
    values.update(overrides)
    values["init_case_tag"] = run_id
    lines = [
        "# Extended source-window validation; scenario bracket, not calibrated kinetics.",
        "# The source rate, ceiling, static GP semantics, and PF physics are unchanged.",
        "",
    ]
    lines.extend(f"{key}={values[key]}" for key in order)
    lines.extend(f"{key}={values[key]}" for key in sorted(values) if key not in order)
    path = OUT / f"{run_id}.params"
    path.write_text("\n".join(lines) + "\n")
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    common = {
        "diagnostic_rsmd_reset_Y_history_after_source": "1",
        "resolved_handoff_xB_write_mode": "preserve_profile_xB_alpha_in_support",
        "scheduled_nuc_scale_interface_width": "1.0",
        "scheduled_nuc_scale_xB_profile_width": "1.0",
        "gp_growth_enabled": "0",
        "gp_inventory_growth_enabled": "0",
        "thermo_convex_extrapolation_enabled": "0",
        "csv_out_every": "25",
        "diagnostic_rsmd_interface_diag_every": "25",
    }
    cases = [
        (
            ROOT / "params/rsmd_y_solver_attribution/T380_rsmd_source_history_sync_dt0p001_n450.params",
            "T380_rsmd_source_history_sync_chi4_dt0p001_n1000_continuous",
            {**common, "nsteps": "1000", "out_every": "1000",
             "diagnostic_rsmd_release_window_steps": "1000"},
        ),
        (
            ROOT / "params/rsmd_y_solver_attribution/T400_rsmd_source_history_sync_chi2_dt0p0005_n1000.params",
            "T400_rsmd_source_history_sync_chi2_dt0p0005_n2000_continuous",
            {**common, "dt": "5.0e-4", "dt_code": "5.0e-4", "nsteps": "2000",
             "out_every": "2000", "diagnostic_rsmd_release_window_steps": "2000",
             "diagnostic_rsmd_chi_rel": "2.0",
             "diagnostic_rsmd_f_max_per_step": "4.0e-5"},
        ),
    ]
    # Equal code-time and equal source flux per code-time: changing dt changes
    # only the time discretization, not the scenario supply strength.
    t380_source = ROOT / "params/rsmd_y_solver_attribution/T380_rsmd_source_history_sync_dt0p001_n450.params"
    t400_source = ROOT / "params/rsmd_y_solver_attribution/T400_rsmd_source_history_sync_chi2_dt0p0005_n1000.params"
    for dt, steps, f_step, tag in (
        ("2.0e-3", "500", "1.6e-4", "dt0p002_n500"),
        ("5.0e-3", "200", "4.0e-4", "dt0p005_n200"),
    ):
        cases.append((
            t380_source,
            f"T380_rsmd_source_history_sync_chi4_{tag}_continuous",
            {**common, "dt": dt, "dt_code": dt, "nsteps": steps,
             "out_every": steps, "diagnostic_rsmd_release_window_steps": steps,
             "diagnostic_rsmd_chi_rel": "4.0",
             "diagnostic_rsmd_f_max_per_step": f_step},
        ))
    for dt, steps, f_step, tag in (
        ("1.0e-3", "1000", "8.0e-5", "dt0p001_n1000"),
        ("2.0e-3", "500", "1.6e-4", "dt0p002_n500"),
        ("5.0e-3", "200", "4.0e-4", "dt0p005_n200"),
    ):
        cases.append((
            t400_source,
            f"T400_rsmd_source_history_sync_chi2_{tag}_continuous",
            {**common, "dt": dt, "dt_code": dt, "nsteps": steps,
             "out_every": steps, "diagnostic_rsmd_release_window_steps": steps,
             "diagnostic_rsmd_chi_rel": "2.0",
             "diagnostic_rsmd_f_max_per_step": f_step},
        ))
    for source, run_id, overrides in cases:
        if not source.exists():
            raise SystemExit(f"missing source parameter file: {source}")
        print(write(source, run_id, overrides))


if __name__ == "__main__":
    main()
