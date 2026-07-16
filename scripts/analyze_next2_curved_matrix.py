#!/usr/bin/env python3
"""Analyze the complete Next2 moving curved OFF/ON, dt/dt2 matrix."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import Unit_Psedobinary as unit  # noqa: E402
from scripts.analyze_finite_interface_stefan_windows import (  # noqa: E402
    h,
    parse_params,
)
from scripts.analyze_prompt7f_circular_benchmark import (  # noqa: E402
    checkpoint_map,
    circular_control_volume_ledger,
    equivalent_radii,
    load_field,
    parse_failure,
    radial_correction_flux,
    radial_surface_metrics,
)


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        if not rows:
            handle.write("")
            return
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fit_window(rows: list[dict[str, float]]) -> tuple[float, float]:
    if len(rows) < 2:
        return math.nan, math.nan
    time = np.array([row["time_code"] for row in rows])
    radius = np.array([row["h_volume_radius_nm"] for row in rows])
    slope, intercept = np.polyfit(time, radius, 1)
    residual = radius - (slope * time + intercept)
    return float(slope), float(np.sqrt(np.mean(residual**2)))


def analyze_case(
    input_case: Path,
    run_case: Path,
    sharp_reference: dict[str, object],
) -> tuple[
    list[dict[str, object]], list[dict[str, object]], dict[str, object],
    dict[str, object], dict[str, object]
]:
    manifest = json.loads((input_case / "input/benchmark_manifest.json").read_text())
    params = parse_params(input_case / "input/benchmark.params")
    shape = tuple(int(value) for value in manifest["grid"])
    log = (run_case / "run.log").read_text(errors="replace")
    phi_paths = checkpoint_map(run_case, "phi")
    x_paths = checkpoint_map(run_case, "xB_alpha")
    c_paths = checkpoint_map(run_case, "Ctot")
    checkpoint_steps = sorted(set(phi_paths) & set(x_paths) & set(c_paths))
    phi0 = load_field(input_case / "input/phi_init.raw", shape)
    x0 = load_field(input_case / "input/xB_init.raw", shape)
    c0 = load_field(input_case / "input/Ctot_init.raw", shape)
    dx_nm = float(manifest["dx_nm"])
    dx_code = float(manifest.get("dx_code", params["dx"]))
    dt = float(manifest["dt_code"])
    mass0 = float(np.sum(c0))
    trajectory: list[dict[str, object]] = []
    loaded: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {
        0: (phi0, x0, c0)
    }
    for step in [0] + checkpoint_steps:
        if step == 0:
            phi, x_field, ctot = phi0, x0, c0
        else:
            phi = load_field(phi_paths[step], shape)
            x_field = load_field(x_paths[step], shape)
            ctot = load_field(c_paths[step], shape)
            loaded[step] = (phi, x_field, ctot)
        r_h, r_half = equivalent_radii(phi, dx_nm)
        plane_phi = phi.mean(axis=1)
        plane_c = ctot.mean(axis=1)
        h_area = float(np.sum(h(plane_phi))) * dx_nm**2
        total = float(np.sum(plane_c)) * dx_nm**2
        matrix = total - h_area
        trajectory.append({
            "case": manifest["case"],
            "R_over_lambda": manifest["R_over_lambda"],
            "correction": manifest["finite_interface_correction"],
            "dt_code": dt,
            "step": step,
            "time_code": step * dt,
            "h_volume_radius_nm": r_h,
            "phi_half_radius_nm": r_half,
            "beta_h_storage_nm2": h_area,
            "matrix_storage_nm2": matrix,
            "total_Ctot_inventory_nm2": total,
            "mass_error_rel": abs(float(np.sum(ctot)) - mass0)
                / max(abs(mass0), 1.0e-300),
            "phi_min": float(np.min(phi)),
            "phi_max": float(np.max(phi)),
            "xB_alpha_min": float(np.min(x_field)),
            "xB_alpha_max": float(np.max(x_field)),
        })
    requested = int(manifest["nsteps"])
    accepted = len(re.findall(r"CTOT_MIMETIC_BE_ACCEPT", log))
    retries = len(re.findall(r"CTOT_COUPLED_STEP_RETRY", log))
    rejects = len(re.findall(r"CTOT_COUPLED_STEP_REJECT", log))
    failure = parse_failure(log)
    kkt_values = [float(value) for value in re.findall(
        r"final_KKT=([0-9.eE+-]+)", log
    )]
    projection_values = [float(value) for value in re.findall(
        r"projection_mass=([0-9.eE+-]+)", log
    )]
    clip_values = [int(value) for value in re.findall(r"clip_count=(\d+)", log)]

    n = len(trajectory)
    windows = {
        "early": trajectory[:max(2, n // 3 + 1)],
        "middle": trajectory[max(0, n // 3 - 1):max(n // 3 + 1, 2 * n // 3 + 1)],
        "full": trajectory,
    }
    sharp_velocity = float(sharp_reference["velocity_full_nm_per_code_time"])
    metrics: list[dict[str, object]] = []
    for window_name, window_rows in windows.items():
        velocity, fit_rmse = fit_window(window_rows)
        metrics.append({
            "case": manifest["case"],
            "R_over_lambda": manifest["R_over_lambda"],
            "correction": manifest["finite_interface_correction"],
            "dt_code": dt,
            "window": window_name,
            "window_start_step": window_rows[0]["step"],
            "window_end_step": window_rows[-1]["step"],
            "PF_h_velocity_nm_per_code_time": velocity,
            "fit_RMSE_nm": fit_rmse,
            "sharp_velocity_nm_per_code_time": sharp_velocity,
            "velocity_error_rel": abs(velocity - sharp_velocity)
                / max(abs(sharp_velocity), 1.0e-300),
            "velocity_direction_match": int(
                velocity == 0.0 or sharp_velocity == 0.0
                or math.copysign(1.0, velocity) == math.copysign(1.0, sharp_velocity)
            ),
            "relative_radius_change": abs(
                float(window_rows[-1]["h_volume_radius_nm"])
                - float(window_rows[0]["h_volume_radius_nm"])
            ) / max(float(window_rows[0]["h_volume_radius_nm"]), 1.0e-300),
            "accepted_steps": accepted,
            "requested_steps": requested,
            "retry_count": retries,
            "reject_count": rejects,
            "max_mass_error_rel": max(
                float(row["mass_error_rel"]) for row in trajectory
            ),
            "max_phase_KKT": max(kkt_values, default=math.nan),
            "projection_mass_max": max(
                (abs(value) for value in projection_values), default=0.0
            ),
            "clip_count_max": max(clip_values, default=0),
            "status": "PASS" if (
                accepted == requested and retries == 0 and rejects == 0
                and max((float(row["mass_error_rel"]) for row in trajectory), default=0.0)
                <= 1.0e-10
            ) else "FAIL",
        })

    final_step = checkpoint_steps[-1]
    previous_step = checkpoint_steps[-2] if len(checkpoint_steps) >= 2 else 0
    previous_phi, _, previous_c = loaded[previous_step]
    final_phi, final_x, final_c = loaded[final_step]
    final_radius = float(trajectory[-1]["phi_half_radius_nm"])
    surface = radial_surface_metrics(
        final_phi,
        final_x,
        dx_nm,
        dx_code,
        float(manifest["interface_width_nm"]),
        final_radius,
        float(params["temperature_C"]) + 273.15,
        float(params["mu_reference_scale"]),
        float(params["D_alpha"]),
    )
    correction_enabled = int(manifest["finite_interface_correction"]) != 0
    correction_flux = radial_correction_flux(
        final_phi,
        previous_phi,
        final_x,
        (final_step - previous_step) * dt,
        dx_code,
        float(manifest["interface_width_nm"]) * dx_code / dx_nm,
        float(params["v_B"]),
    ) if correction_enabled else 0.0
    total_flux = float(surface["matrix_flux"]) + correction_flux
    stefan_velocity = total_flux / max(1.0 - float(surface["x_surface"]), 1.0e-300)
    control = circular_control_volume_ledger(
        previous_phi,
        previous_c,
        final_phi,
        final_c,
        dx_nm,
        (final_step - previous_step) * dt,
        final_radius + 2.0 * float(manifest["interface_width_nm"]),
    )
    temperature = float(params["temperature_C"]) + 273.15
    x_eq = unit.xAg2Te_eq_from_T(temperature)
    mu_eq = (
        unit.mu_Ag2Te(temperature, x_eq) - unit.mu_PbTe(temperature, x_eq)
    ) / float(params["mu_reference_scale"])
    flux_row = {
        "case": manifest["case"],
        "R_over_lambda": manifest["R_over_lambda"],
        "correction": manifest["finite_interface_correction"],
        "dt_code": dt,
        "matrix_flux_code": surface["matrix_flux"],
        "finite_interface_flux_code": correction_flux,
        "beta_side_flux_code": 0.0,
        "total_matrix_side_flux_code": total_flux,
        "stefan_velocity_code": stefan_velocity,
        "PF_h_velocity_nm_per_code_time": next(
            float(row["PF_h_velocity_nm_per_code_time"])
            for row in metrics if row["window"] == "full"
        ),
        "stefan_residual_h": next(
            float(row["PF_h_velocity_nm_per_code_time"])
            for row in metrics if row["window"] == "full"
        ) - stefan_velocity,
        "control_volume_Hdot": control["Hdot"],
        "control_volume_Qdot": control["Qdot"],
        "control_volume_Cdot": control["Cdot"],
        "control_volume_closure": control["closure"],
        "mu_surface": surface["mu_surface"],
        "mu_equilibrium": mu_eq,
        "chemical_potential_jump": float(surface["mu_surface"]) - mu_eq,
        "xB_surface": surface["x_surface"],
    }

    energy_files = list((run_case / "results").rglob("ctot_energy_work.csv"))
    energy_row: dict[str, object] = {
        "case": manifest["case"],
        "R_over_lambda": manifest["R_over_lambda"],
        "correction": manifest["finite_interface_correction"],
        "dt_code": dt,
        "energy_rows": 0,
        "max_energy_balance_rel": math.nan,
        "min_base_transport_dissipation": math.nan,
        "max_abs_finite_interface_work": math.nan,
        "energy_predicate_all_pass": 0,
    }
    if energy_files:
        with energy_files[0].open(newline="") as handle:
            energy = list(csv.DictReader(handle))
        if energy:
            balance = [abs(float(row["energy_balance_rel"])) for row in energy]
            energy_row.update({
                "energy_rows": len(energy),
                "max_energy_balance_rel": max(balance),
                "min_base_transport_dissipation": min(
                    float(row["D_transport"]) for row in energy
                ),
                "max_abs_finite_interface_work": max(
                    abs(float(row["W_finite_interface"])) for row in energy
                ),
                "energy_predicate_all_pass": int(all(
                    int(float(row["monotone_pass"])) != 0
                    and int(float(row["balance_pass"])) != 0
                    and int(float(row["accepted"])) != 0
                    for row in energy
                )),
            })
    status_row = {
        "case": manifest["case"],
        "accepted_steps": accepted,
        "requested_steps": requested,
        "retry_count": retries,
        "reject_count": rejects,
        "failure_reason": failure["failure_reason"],
        "lower_bound_violations": failure["lower_violations"],
        "upper_bound_violations": failure["upper_violations"],
        "max_lower_defect": failure["max_lower_defect"],
        "status": "PASS" if accepted == requested else "FAIL",
    }
    return metrics, trajectory, flux_row, energy_row, status_row


def minimum_contiguous_ratio(
    errors: dict[int, float], threshold: float
) -> str:
    ratios = sorted(errors)
    for index, ratio in enumerate(ratios):
        if all(errors[value] <= threshold for value in ratios[index:]):
            return str(ratio)
    return "NOT_DEMONSTRATED"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--sharp-reference", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    args = parser.parse_args()
    references = json.loads(args.sharp_reference.read_text())
    manifests = json.loads(
        (args.input_root / "next2_moving_manifest.json").read_text()
    )
    metrics: list[dict[str, object]] = []
    timeseries: list[dict[str, object]] = []
    flux_rows: list[dict[str, object]] = []
    energy_rows: list[dict[str, object]] = []
    statuses: list[dict[str, object]] = []
    for manifest in manifests:
        label = str(manifest["case"])
        result = analyze_case(
            args.input_root / label,
            args.run_root / label,
            references[str(manifest["R_over_lambda"])],
        )
        case_metrics, case_timeseries, flux, energy, status = result
        metrics.extend(case_metrics)
        timeseries.extend(case_timeseries)
        flux_rows.append(flux)
        energy_rows.append(energy)
        statuses.append(status)

    full = {
        (int(row["R_over_lambda"]), int(row["correction"]), float(row["dt_code"])): row
        for row in metrics if row["window"] == "full"
    }
    flux_map = {
        (int(row["R_over_lambda"]), int(row["correction"]), float(row["dt_code"])): row
        for row in flux_rows
    }
    dt_rows: list[dict[str, object]] = []
    for ratio in (5, 8, 10, 15, 20):
        for correction in (0, 1):
            coarse = full[(ratio, correction, 6.25e-6)]
            fine = full[(ratio, correction, 3.125e-6)]
            coarse_flux = flux_map[(ratio, correction, 6.25e-6)]
            fine_flux = flux_map[(ratio, correction, 3.125e-6)]
            sharp = float(fine["sharp_velocity_nm_per_code_time"])
            velocity_difference = abs(
                float(coarse["PF_h_velocity_nm_per_code_time"])
                - float(fine["PF_h_velocity_nm_per_code_time"])
            )
            velocity_scale = max(
                abs(sharp),
                abs(float(coarse["PF_h_velocity_nm_per_code_time"])),
                abs(float(fine["PF_h_velocity_nm_per_code_time"])),
                1.0e-12,
            )
            velocity_relative = velocity_difference / velocity_scale
            mu_difference = abs(
                float(coarse_flux["chemical_potential_jump"])
                - float(fine_flux["chemical_potential_jump"])
            )
            mu_scale = max(
                abs(float(fine_flux["chemical_potential_jump"])), 1.0e-12
            )
            converged = (
                velocity_relative <= 0.02
                and mu_difference / mu_scale <= 0.02
                and coarse["status"] == "PASS"
                and fine["status"] == "PASS"
            )
            dt_rows.append({
                "R_over_lambda": ratio,
                "correction": correction,
                "coarse_dt": 6.25e-6,
                "fine_dt": 3.125e-6,
                "velocity_coarse": coarse["PF_h_velocity_nm_per_code_time"],
                "velocity_fine": fine["PF_h_velocity_nm_per_code_time"],
                "velocity_abs_difference": velocity_difference,
                "velocity_relative_difference_scaled": velocity_relative,
                "chemical_jump_coarse": coarse_flux["chemical_potential_jump"],
                "chemical_jump_fine": fine_flux["chemical_potential_jump"],
                "chemical_jump_relative_difference": mu_difference / mu_scale,
                "stefan_residual_coarse": coarse_flux["stefan_residual_h"],
                "stefan_residual_fine": fine_flux["stefan_residual_h"],
                "mass_error_coarse": coarse["max_mass_error_rel"],
                "mass_error_fine": fine["max_mass_error_rel"],
                "dt_convergence_status": "PASS" if converged else "FAIL",
            })

    fine_errors: dict[tuple[int, int], float] = {
        (ratio, correction): float(
            full[(ratio, correction, 3.125e-6)]["velocity_error_rel"]
        ) for ratio in (5, 8, 10, 15, 20) for correction in (0, 1)
    }
    improvements = {
        ratio: fine_errors[(ratio, 1)] < fine_errors[(ratio, 0)]
        for ratio in (5, 8, 10, 15, 20)
    }
    dt_on_pass = all(
        row["dt_convergence_status"] == "PASS"
        for row in dt_rows if int(row["correction"]) == 1
    )
    on_errors = {ratio: fine_errors[(ratio, 1)] for ratio in improvements}
    min2 = minimum_contiguous_ratio(on_errors, 0.02)
    min5 = minimum_contiguous_ratio(on_errors, 0.05)
    systematic = all(improvements.values())
    accepted = systematic and dt_on_pass and min5 != "NOT_DEMONSTRATED"
    decision = {
        "correction_ON_systematic_improvement": systematic,
        "dt_convergence_ON": dt_on_pass,
        "minimum_R_over_lambda_for_2percent": min2,
        "minimum_R_over_lambda_for_5percent": min5,
        "finite_interface_correction_decision": (
            "PASS_CURVED_FINITE_INTERFACE_CORRECTION" if accepted else
            "CURRENT_FINITE_INTERFACE_CORRECTION_REJECTED_FOR_CURVED_QUANTITATIVE_PRODUCTION"
        ),
        "improvement_by_ratio": improvements,
        "fine_dt_errors_OFF": {str(r): fine_errors[(r, 0)] for r in improvements},
        "fine_dt_errors_ON": {str(r): fine_errors[(r, 1)] for r in improvements},
    }
    output = args.output_root
    output.mkdir(parents=True, exist_ok=True)
    write_rows(output / "next2_curved_velocity_metrics.csv", metrics)
    write_rows(output / "next2_curved_radius_timeseries.csv", timeseries)
    write_rows(output / "next2_curved_flux_stefan.csv", flux_rows)
    write_rows(output / "next2_curved_energy_work.csv", energy_rows)
    write_rows(output / "next2_dt_convergence.csv", dt_rows)
    write_rows(output / "next2_moving_case_status.csv", statuses)
    (output / "next2_correction_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    print(f"moving_cases_analyzed={len(statuses)}")
    print(f"moving_cases_passed={sum(row['status'] == 'PASS' for row in statuses)}")
    print(f"finite_interface_correction_decision={decision['finite_interface_correction_decision']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
