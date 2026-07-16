#!/usr/bin/env python3
"""Windowed, matrix-side Stefan audit for one-sided planar Ctot runs."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def parse_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def crossing_positions(phi: np.ndarray, dx: float) -> list[float]:
    positions: list[float] = []
    for index, left in enumerate(phi):
        right = phi[(index + 1) % phi.size]
        if left == 0.5:
            positions.append(index * dx)
        elif (left - 0.5) * (right - 0.5) < 0.0:
            positions.append((index + (0.5 - left) / (right - left)) * dx)
    if len(positions) != 2:
        raise ValueError(f"expected two phi=0.5 crossings, got {positions}")
    return positions


def linear_surface_extrapolation(
    face_positions: np.ndarray,
    values: np.ndarray,
    surface_position: float,
    distances: np.ndarray,
    min_distance: float,
    max_distance: float,
    degree: int = 1,
) -> tuple[float, int, float]:
    mask = ((distances >= min_distance) & (distances <= max_distance) &
            np.isfinite(values))
    count = int(mask.sum())
    if count < degree + 2:
        return math.nan, count, math.nan
    coordinates = face_positions[mask] - surface_position
    coefficients = np.polyfit(coordinates, values[mask], degree)
    fitted = np.polyval(coefficients, coordinates)
    rms = float(np.sqrt(np.mean((fitted - values[mask]) ** 2)))
    return float(coefficients[-1]), count, rms


def checkpoint_map(run_dir: Path, suffix: str) -> dict[int, Path]:
    result: dict[int, Path] = {}
    pattern = re.compile(r"ctot_checkpoint_step(\d+)_" + re.escape(suffix) + r"\.raw$")
    for path in run_dir.rglob(f"ctot_checkpoint_step*_{suffix}.raw"):
        match = pattern.search(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def matrix_mu_and_mobility(
    phi: np.ndarray,
    x: np.ndarray,
    temperature_K: float,
    mu_scale: float,
    D_alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    mu = np.array([
        (unit.mu_Ag2Te(temperature_K, float(value)) -
         unit.mu_PbTe(temperature_K, float(value))) / mu_scale
        for value in x
    ])
    gamma = np.array([
        unit.g_alpha_second_unified(temperature_K, float(value)) / mu_scale
        for value in x
    ])
    mobility = (1.0 - h(phi)) * D_alpha / gamma
    return mu, mobility


def matrix_surface_metrics(
    phi: np.ndarray,
    x: np.ndarray,
    surface: float,
    dx_code: float,
    dx_nm: float,
    temperature_K: float,
    mu_scale: float,
    D_alpha: float,
) -> dict[str, float | int]:
    nx = phi.size
    mu, mobility = matrix_mu_and_mobility(
        phi, x, temperature_K, mu_scale, D_alpha
    )
    next_index = np.roll(np.arange(nx), -1)
    M_next = mobility[next_index]
    M_face = np.zeros(nx)
    positive = (mobility > 0.0) & (M_next > 0.0)
    M_face[positive] = (2.0 * mobility[positive] * M_next[positive] /
                        (mobility[positive] + M_next[positive]))
    J_face = M_face * (mu[next_index] - mu) / dx_code
    face_positions = (np.arange(nx, dtype=float) + 0.5) * dx_code
    distances = (face_positions - surface) % (nx * dx_code)
    pure_matrix = (phi < 0.01) & (phi[next_index] < 0.01)
    mu_face = 0.5 * (mu + mu[next_index])
    x_face = 0.5 * (x + x[next_index])
    min_fit = max(0.5 * dx_code, 0.10 / (dx_nm / dx_code))
    max_fit = 2.0 / (dx_nm / dx_code)
    result: dict[str, float | int] = {}
    for name, values in (
        ("flux", J_face), ("mu", mu_face), ("x", x_face)
    ):
        value, count, rms = linear_surface_extrapolation(
            face_positions, np.where(pure_matrix, values, np.nan),
            surface, distances, min_fit, max_fit, degree=3
        )
        result[name] = value
        result[f"{name}_count"] = count
        result[f"{name}_rms"] = rms
    result["stefan_velocity_code"] = (
        float(result["flux"]) / (1.0 - float(result["x"]))
    )
    return result


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def analyze_case(case: Path) -> list[dict[str, object]]:
    input_dir = case / "input"
    run_dir = case / "run"
    manifest = json.loads((input_dir / "benchmark_manifest.json").read_text())
    params = parse_params(input_dir / "benchmark.params")
    grid = tuple(int(value) for value in manifest["grid"])
    nx, ny, nz = grid
    dx_nm = float(manifest["dx_nm"])
    dx_code = float(params["dx"])
    dt_code = float(manifest["dt_code"])
    t0_s = float(params["t_real_unit_s"])
    speed_scale_nm_s = (dx_nm / dx_code) / t0_s
    temperature_K = float(params["temperature_C"]) + 273.15
    mu_scale = float(params["mu_reference_scale"])
    D_alpha = float(params["D_alpha"])
    x_eq = float(manifest["xB_eq"])
    phi_by_step = checkpoint_map(run_dir, "phi")
    x_by_step = checkpoint_map(run_dir, "xB_alpha")
    C_by_step = checkpoint_map(run_dir, "Ctot")
    phi_by_step[0] = input_dir / "phi_init.raw"
    x_by_step[0] = input_dir / "xB_init.raw"
    C_by_step[0] = input_dir / "Ctot_init.raw"
    steps = sorted(set(phi_by_step) & set(x_by_step) & set(C_by_step))
    if len(steps) < 2:
        raise ValueError(f"{case}: fewer than two complete states")
    states: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for step in steps:
        phi = np.fromfile(phi_by_step[step], dtype=np.float64).reshape(grid).mean((1, 2))
        x = np.fromfile(x_by_step[step], dtype=np.float64).reshape(grid).mean((1, 2))
        C = np.fromfile(C_by_step[step], dtype=np.float64).reshape(grid).mean((1, 2))
        states[step] = phi, x, C

    energy_files = list(run_dir.rglob("ctot_energy_work.csv"))
    max_mass = math.nan
    max_balance = math.nan
    if len(energy_files) == 1:
        with energy_files[0].open(newline="") as handle:
            energy_rows = list(csv.DictReader(handle))
        if energy_rows:
            max_balance = max(abs(float(row["energy_balance_rel"])) for row in energy_rows)
    log = (run_dir / "run.log").read_text() if (run_dir / "run.log").is_file() else ""
    mass_values = [float(value) for value in re.findall(
        r"CTOT_(?:FV|SPECTRAL)_BE_ACCEPT[^\n]*mass_error=([^ ]+)", log)]
    accepted_steps = len(re.findall(
        r"CTOT_(?:FV|SPECTRAL)_BE_ACCEPT[^\n]+", log))
    retry_count = len(re.findall(r"CTOT_COUPLED_STEP_RETRY", log))
    if mass_values:
        max_mass = max(map(abs, mass_values))

    rows: list[dict[str, object]] = []
    for old_step, new_step in zip(steps, steps[1:]):
        phi_old, x_old, C_old = states[old_step]
        phi_new, x_new, C_new = states[new_step]
        old_cross = crossing_positions(phi_old, dx_code)
        new_cross = crossing_positions(phi_new, dx_code)
        old_right, new_right = old_cross[1], new_cross[1]
        elapsed_code = (new_step - old_step) * dt_code
        velocity_code = (new_right - old_right) / elapsed_code
        velocity_nm_s = velocity_code * speed_scale_nm_s

        old_surface = matrix_surface_metrics(
            phi_old, x_old, old_right, dx_code, dx_nm,
            temperature_K, mu_scale, D_alpha)
        new_surface = matrix_surface_metrics(
            phi_new, x_new, new_right, dx_code, dx_nm,
            temperature_K, mu_scale, D_alpha)
        stefan_velocity_nm_s = 0.5 * (
            float(old_surface["stefan_velocity_code"]) +
            float(new_surface["stefan_velocity_code"])
        ) * speed_scale_nm_s

        t_start = float(manifest["sharp_start_time_s"])
        t_old = old_step * dt_code * t0_s
        t_new = new_step * dt_code * t0_s
        D_nm2_s = (float(manifest["sharp_diffusion_length_nm"]) ** 2 /
                   t_start)
        eta = float(manifest["sharp_similarity_parameter"])
        sharp_displacement = 2.0 * eta * math.sqrt(D_nm2_s) * (
            math.sqrt(t_start + t_new) - math.sqrt(t_start + t_old))
        sharp_velocity = sharp_displacement / (t_new - t_old)
        delta_C = float((C_new - C_old).sum())
        delta_h = float((h(phi_new) - h(phi_old)).sum())
        delta_q = float(((C_new - h(phi_new)) -
                         (C_old - h(phi_old))).sum())
        mu_eq = ((unit.mu_Ag2Te(temperature_K, x_eq) -
                  unit.mu_PbTe(temperature_K, x_eq)) / mu_scale)
        rows.append({
            "case": case.name,
            "old_step": old_step,
            "new_step": new_step,
            "window_elapsed_s": (new_step - old_step) * dt_code * t0_s,
            "interface_resolution": manifest["interface_resolution"],
            "dx_nm": dx_nm,
            "L_phi_factor": manifest["L_phi_factor"],
            "finite_interface_correction": int(
                params.get("ctot_finite_interface_antitrapping_enabled", "0")),
            "accepted_steps": accepted_steps,
            "requested_steps": manifest["nsteps"],
            "retry_count": retry_count,
            "equal_requested_dt": retry_count == 0,
            "phi_half_velocity_nm_s": velocity_nm_s,
            "matrix_flux_old_extrapolated_code": old_surface["flux"],
            "matrix_flux_new_extrapolated_code": new_surface["flux"],
            "matrix_flux_fit_points": new_surface["flux_count"],
            "matrix_flux_fit_rms": new_surface["flux_rms"],
            "local_stefan_velocity_nm_s": stefan_velocity_nm_s,
            "local_stefan_ratio_phi_over_flux": velocity_nm_s / stefan_velocity_nm_s,
            "sharp_similarity_velocity_nm_s": sharp_velocity,
            "phi_velocity_ratio_to_sharp": velocity_nm_s / sharp_velocity,
            "matrix_xB_surface_old_extrapolated": old_surface["x"],
            "matrix_xB_surface_new_extrapolated": new_surface["x"],
            "matrix_xB_fit_points": new_surface["x_count"],
            "matrix_xB_fit_rms": new_surface["x_rms"],
            "mu_surface_old_extrapolated": old_surface["mu"],
            "mu_surface_new_extrapolated": new_surface["mu"],
            "mu_fit_points": new_surface["mu_count"],
            "mu_fit_rms": new_surface["mu_rms"],
            "mu_equilibrium": mu_eq,
            "mu_surface_old_excess": float(old_surface["mu"]) - mu_eq,
            "mu_surface_new_excess": float(new_surface["mu"]) - mu_eq,
            "delta_C_cell_units": delta_C * ny * nz,
            "stefan_storage_residual_rel": abs(delta_h + delta_q) /
                max(abs(delta_h), abs(delta_q), 1.0),
            "runtime_mass_error_max": max_mass,
            "energy_balance_rel_max": max_balance,
        })
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for case in sorted(path for path in args.matrix_root.iterdir() if path.is_dir()):
        if (case / "input/benchmark_manifest.json").is_file():
            rows.extend(analyze_case(case))
    if not rows:
        raise SystemExit("no analyzable cases")
    write_rows(args.output, rows)
    print(f"window_rows={len(rows)}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
