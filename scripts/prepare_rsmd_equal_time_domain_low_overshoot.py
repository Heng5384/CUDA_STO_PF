#!/usr/bin/env python3
"""Prepare workstation-only RSMD operator and equal-time validation cases."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "params" / "rsmd_equal_time_domain_low_overshoot"
BASE = {
    380: ROOT / "params/rsmd_post_history_sync_validation/T380_dt0p002_history_local_zero_restart.params",
    400: ROOT / "params/rsmd_post_history_sync_validation/T400_dt0p002_history_local_zero_restart.params",
}


def read_params(path: Path) -> tuple[list[str], dict[str, str]]:
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


def write_case(T: int, case: str, overrides: dict[str, str]) -> Path:
    order, values = read_params(BASE[T])
    values.update(overrides)
    values["init_case_tag"] = case
    lines = [
        "# RSMD numerical coupling audit; scenario bracket, not calibrated kinetics.",
        "# GP inventory and beta/PF physics are unchanged.",
        "",
    ]
    lines.extend(f"{key}={values[key]}" for key in order)
    lines.extend(f"{key}={values[key]}" for key in sorted(values) if key not in order)
    path = OUT / f"{case}.params"
    path.write_text("\n".join(lines) + "\n")
    return path


def row(case: str, phase: str, T: int, dt: float, nsteps: int, scheme: str,
        operator: str, integrator: str, substep: float, headroom: int,
        source_enabled: int = 1, control_mode: str = "full_coupled") -> dict[str, str]:
    f_step = 0.08 * dt
    diag_every = max(1, round(0.05 / dt))
    overrides = {
        "temperature_C": str(T),
        "diagnostic_rsmd_T_only": str(T),
        "diagnostic_rsmd_enabled": str(source_enabled),
        "diagnostic_rsmd_f_max_per_step": f"{f_step:.12g}",
        "diagnostic_rsmd_release_window_steps": str(round(0.92 / dt)),
        "diagnostic_rsmd_operator_split": operator,
        "diagnostic_rsmd_source_integrator": integrator,
        "diagnostic_rsmd_source_substep_dt_code": f"{substep:.12g}",
        "diagnostic_rsmd_headroom_weighted": str(headroom),
        "diagnostic_rsmd_control_mode": control_mode,
        "diagnostic_rsmd_history_restart_mode":
            "0" if control_mode == "source_only_frozen_field" else "1",
        "diagnostic_rsmd_reset_Y_history_after_source":
            "0" if control_mode == "source_only_frozen_field" else "1",
        "dt": f"{dt:.12g}",
        "dt_code": f"{dt:.12g}",
        "nsteps": str(nsteps),
        "csv_out_every": str(diag_every),
        "diagnostic_rsmd_interface_diag_every": str(diag_every),
        "phi_eta_rhs_attribution_diag_every": str(diag_every),
        "phi_eta_rhs_attribution_diag_max_steps": str(nsteps),
        "dynamics_mass_diag_interval": "1",
        "out_every": "1000000",
    }
    path = write_case(T, case, overrides)
    return {
        "case": case, "phase": phase, "T_C": str(T), "dt": str(dt),
        "nsteps": str(nsteps), "scheme": scheme, "operator_split": operator,
        "source_integrator": integrator, "source_substep_dt_code": str(substep),
        "headroom_weighted": str(headroom), "source_enabled": str(source_enabled),
        "control_mode": control_mode,
        "f_max_per_step": str(f_step), "f_over_dt": "0.08",
        "param_file": str(path),
    }


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str]] = []

    smoke = [
        ("S0", "post_pf_lie", "legacy_explicit", 0.0, 0),
        ("S1", "pre_pf_lie", "legacy_explicit", 0.0, 0),
        ("S2", "strang", "exact_exponential", 0.0005, 0),
        ("S3", "post_pf_lie", "exact_exponential", 0.0005, 1),
    ]
    for scheme, operator, integrator, substep, headroom in smoke:
        case = f"T400_smoke_{scheme}_dt0p002"
        rows.append(row(case, "scheme_smoke", 400, 0.002, 140, scheme,
                        operator, integrator, substep, headroom))

    # S2 isolates symmetric/exact time integration; S3 adds headroom weighting.
    candidates = [
        ("S2", "strang", "exact_exponential", 0.0005, 0),
        ("S3", "post_pf_lie", "exact_exponential", 0.0005, 1),
    ]
    for T in (380, 400):
        for dt in (0.0005, 0.001, 0.002):
            # The controlled resolved handoff occurs at code step 40 for every
            # dt. Keep the post-handoff code-time window exactly 0.92.
            nsteps = 40 + round(0.92 / dt)
            tag = str(dt).replace("0.", "0p")
            for scheme, operator, integrator, substep, headroom in candidates:
                case = f"T{T}_{scheme}_dt{tag}_equal_time"
                rows.append(row(case, "dt_convergence", T, dt, nsteps, scheme,
                                operator, integrator, substep, headroom))
            case = f"T{T}_PFonly_dt{tag}_equal_time"
            rows.append(row(case, "pf_only", T, dt, nsteps, "PF_ONLY",
                            "post_pf_lie", "legacy_explicit", 0.0, 0,
                            source_enabled=0))
            if T == 400:
                rows.append(row(f"T400_SOURCEONLY_dt{tag}_equal_time", "source_only",
                                T, dt, nsteps, "S3_SOURCE_ONLY", "post_pf_lie",
                                "exact_exponential", 0.0005, 1,
                                control_mode="source_only_frozen_field"))
                rows.append(row(f"T400_SOURCEDIFF_dt{tag}_equal_time", "source_diffusion",
                                T, dt, nsteps, "S3_SOURCE_DIFFUSION", "post_pf_lie",
                                "exact_exponential", 0.0005, 1,
                                control_mode="source_diffusion_frozen_phi"))

    with (OUT / "run_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(OUT / "run_manifest.csv")


if __name__ == "__main__":
    main()
