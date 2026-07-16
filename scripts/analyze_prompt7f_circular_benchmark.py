#!/usr/bin/env python3
"""Analyze Prompt 7f CUDA circular correction-on runs without changing them."""

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
import Unit_Psedobinary as unit  # noqa: E402
from analyze_finite_interface_stefan_windows import (  # noqa: E402
    h,
    matrix_mu_and_mobility,
    parse_params,
)


def checkpoint_map(case: Path, suffix: str) -> dict[int, Path]:
    pattern = re.compile(rf"ctot_checkpoint_step(\d+)_{re.escape(suffix)}\.raw$")
    result: dict[int, Path] = {}
    for path in case.rglob(f"ctot_checkpoint_step*_{suffix}.raw"):
        match = pattern.match(path.name)
        if match:
            result[int(match.group(1))] = path
    return result


def load_field(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float64)
    if values.size != math.prod(shape):
        raise ValueError(f"field size mismatch for {path}: {values.size} vs {shape}")
    return values.reshape(shape)


def phi_half_radius(phi2: np.ndarray, dx_nm: float, n_angles: int = 360) -> float:
    nx, nz = phi2.shape
    cx = 0.5 * nx
    cz = 0.5 * nz
    samples = np.arange(0.0, 0.45 * min(nx, nz), 0.25)
    radii: list[float] = []
    for angle in np.linspace(0.0, 2.0 * math.pi, n_angles, endpoint=False):
        x = cx + samples * math.cos(angle)
        z = cz + samples * math.sin(angle)
        i0 = np.floor(x).astype(int) % nx
        k0 = np.floor(z).astype(int) % nz
        i1 = (i0 + 1) % nx
        k1 = (k0 + 1) % nz
        tx = x - np.floor(x)
        tz = z - np.floor(z)
        values = ((1.0 - tx) * (1.0 - tz) * phi2[i0, k0] +
                  tx * (1.0 - tz) * phi2[i1, k0] +
                  (1.0 - tx) * tz * phi2[i0, k1] +
                  tx * tz * phi2[i1, k1])
        crossings = np.where((values[:-1] >= 0.5) & (values[1:] < 0.5))[0]
        if crossings.size:
            index = int(crossings[0])
            fraction = ((0.5 - values[index]) /
                        (values[index + 1] - values[index]))
            radii.append(float(samples[index] + fraction *
                               (samples[index + 1] - samples[index])) * dx_nm)
    return float(np.mean(radii)) if radii else math.nan


def equivalent_radii(phi: np.ndarray, dx_nm: float) -> tuple[float, float]:
    plane = phi.mean(axis=1)
    area_h = float(h(plane).sum()) * dx_nm * dx_nm
    return math.sqrt(area_h / math.pi), phi_half_radius(plane, dx_nm)


def circular_control_volume_ledger(
    phi_before: np.ndarray,
    ctot_before: np.ndarray,
    phi_after: np.ndarray,
    ctot_after: np.ndarray,
    dx_nm: float,
    dt_code: float,
    control_radius_nm: float,
) -> dict[str, float]:
    if not dt_code > 0.0:
        return {key: math.nan for key in (
            "Hdot", "Qdot", "Cdot", "closure", "velocity_h",
            "control_flux", "matrix_storage_velocity")}
    phi0 = phi_before.mean(axis=1)
    phi1 = phi_after.mean(axis=1)
    c0 = ctot_before.mean(axis=1)
    c1 = ctot_after.mean(axis=1)
    nx, nz = phi0.shape
    xx = np.arange(nx, dtype=float) * dx_nm
    zz = np.arange(nz, dtype=float) * dx_nm
    rr = np.sqrt(
        (xx[:, None] - 0.5 * nx * dx_nm) ** 2 +
        (zz[None, :] - 0.5 * nz * dx_nm) ** 2
    )
    mask = rr <= control_radius_nm
    cell_area = dx_nm * dx_nm
    h0, h1 = h(phi0), h(phi1)
    H0, H1 = float(h0[mask].sum()) * cell_area, float(h1[mask].sum()) * cell_area
    Q0 = float((c0[mask] - h0[mask]).sum()) * cell_area
    Q1 = float((c1[mask] - h1[mask]).sum()) * cell_area
    C0, C1 = float(c0[mask].sum()) * cell_area, float(c1[mask].sum()) * cell_area
    Hdot, Qdot, Cdot = ((H1 - H0) / dt_code,
                         (Q1 - Q0) / dt_code,
                         (C1 - C0) / dt_code)
    radius0 = math.sqrt(max(float(h0.sum()) * cell_area, 0.0) / math.pi)
    radius1 = math.sqrt(max(float(h1.sum()) * cell_area, 0.0) / math.pi)
    circumference = 2.0 * math.pi * max(0.5 * (radius0 + radius1), 1.0e-300)
    return {
        "Hdot": Hdot,
        "Qdot": Qdot,
        "Cdot": Cdot,
        "closure": Hdot + Qdot - Cdot,
        "velocity_h": Hdot / circumference,
        "control_flux": Cdot / circumference,
        "matrix_storage_velocity": Qdot / circumference,
    }


def radial_surface_metrics(
    phi: np.ndarray,
    x: np.ndarray,
    dx_nm: float,
    dx_code: float,
    lambda_nm: float,
    radius_nm: float,
    temperature_K: float,
    mu_scale: float,
    D_alpha: float,
) -> dict[str, float]:
    phi2 = phi.mean(axis=1)
    x2 = x.mean(axis=1)
    nx, nz = phi2.shape
    xx = np.arange(nx, dtype=float) * dx_nm
    zz = np.arange(nz, dtype=float) * dx_nm
    cx = 0.5 * nx * dx_nm
    cz = 0.5 * nz * dx_nm
    rr = np.sqrt((xx[:, None] - cx) ** 2 + (zz[None, :] - cz) ** 2)
    mu, mobility = matrix_mu_and_mobility(
        phi2.ravel(), x2.ravel(), temperature_K, mu_scale, D_alpha
    )
    mu = mu.reshape(phi2.shape)
    mobility = mobility.reshape(phi2.shape)
    mask = ((rr >= radius_nm + 0.5 * lambda_nm) &
            (rr <= radius_nm + 2.5 * lambda_nm) & (phi2 < 0.1))
    if int(mask.sum()) < 8:
        return {key: math.nan for key in (
            "matrix_flux", "mu_surface", "x_surface", "mu_gradient",
            "mobility_surface")}
    r_code = rr[mask] * (dx_code / dx_nm)
    center_code = radius_nm * (dx_code / dx_nm)
    mu_fit = np.polyfit(r_code - center_code, mu[mask], 1)
    x_fit = np.polyfit(r_code - center_code, x2[mask], 1)
    m_fit = np.polyfit(r_code - center_code, mobility[mask], 1)
    mu_surface = float(mu_fit[1])
    x_surface = float(x_fit[1])
    mobility_surface = max(float(m_fit[1]), 0.0)
    return {
        "matrix_flux": mobility_surface * float(mu_fit[0]),
        "mu_surface": mu_surface,
        "x_surface": x_surface,
        "mu_gradient": float(mu_fit[0]),
        "mobility_surface": mobility_surface,
    }


def radial_correction_flux(
    phi: np.ndarray,
    phi_previous: np.ndarray,
    x: np.ndarray,
    dt_interval: float,
    dx_code: float,
    lambda_code: float,
    v_B: float,
) -> float:
    if not (dt_interval > 0.0):
        return math.nan
    phi2 = phi.mean(axis=1)
    old2 = phi_previous.mean(axis=1)
    x2 = x.mean(axis=1)
    nx, nz = phi2.shape
    gx = (np.roll(phi2, -1, axis=0) - np.roll(phi2, 1, axis=0)) / (2.0 * dx_code)
    gz = (np.roll(phi2, -1, axis=1) - np.roll(phi2, 1, axis=1)) / (2.0 * dx_code)
    magnitude = np.sqrt(gx * gx + gz * gz)
    xx = np.arange(nx, dtype=float)[:, None] - 0.5 * nx
    zz = np.arange(nz, dtype=float)[None, :] - 0.5 * nz
    rr = np.sqrt(xx * xx + zz * zz)
    radial_dot = np.zeros_like(phi2)
    valid = (magnitude > 1.0e-14) & (rr > 0.0)
    radial_dot[valid] = ((gx[valid] * (xx / np.maximum(rr, 1.0e-300))[valid] +
                          gz[valid] * (zz / np.maximum(rr, 1.0e-300))[valid]) /
                         magnitude[valid])
    H = h(phi2)
    shape = np.zeros_like(phi2)
    interface = ((phi2 > 1.0e-12) & (phi2 < 1.0 - 1.0e-12))
    shape[interface] = (H[interface] * (1.0 - H[interface]) /
                        (4.0 * phi2[interface] * (1.0 - phi2[interface])))
    phi_t = (phi2 - old2) / dt_interval
    flux = lambda_code * shape * (x2 - v_B) * phi_t * radial_dot
    sample = valid & (phi2 >= 0.35) & (phi2 <= 0.65)
    return float(np.mean(flux[sample])) if int(sample.sum()) else math.nan


def parse_failure(log_text: str) -> dict[str, float | int | str]:
    matches = re.findall(
        r"BE_target_lower_violations=([0-9.eE+-]+) max_lower_defect=([0-9.eE+-]+) "
        r"BE_target_upper_violations=([0-9.eE+-]+) max_upper_defect=([0-9.eE+-]+)",
        log_text,
    )
    lower = upper = 0
    lower_defect = upper_defect = 0.0
    for item in matches:
        lower = max(lower, int(float(item[0])))
        lower_defect = max(lower_defect, float(item[1]))
        upper = max(upper, int(float(item[2])))
        upper_defect = max(upper_defect, float(item[3]))
    reason = "NONE"
    if "energy_work_predicate_failed" in log_text:
        reason = "ENERGY_WORK_PREDICATE_FAILED"
    elif lower or upper:
        reason = "BE_TARGET_INADMISSIBLE"
    elif "[fatal]" in log_text or "[reject]" in log_text:
        reason = "RUNTIME_REJECTED"
    return {
        "lower_violations": lower,
        "upper_violations": upper,
        "max_lower_defect": lower_defect,
        "max_upper_defect": upper_defect,
        "failure_reason": reason,
    }


def analyze_case(case: Path) -> dict[str, object]:
    manifest = json.loads((case / "input/benchmark_manifest.json").read_text())
    params = parse_params(case / "input/benchmark.params")
    shape = tuple(int(value) for value in manifest["grid"])
    log_path = case / "run.log"
    if not log_path.exists():
        log_path = case / "run/run.log"
    log_text = log_path.read_text(errors="replace")
    failure = parse_failure(log_text)
    accepted_steps = len(re.findall(
        r"CTOT_(?:MIMETIC|FV|SPECTRAL)_BE_ACCEPT", log_text))
    requested_match = re.search(r"^\s*steps\s*:\s*(\d+)\s*$", log_text, re.MULTILINE)
    requested_steps = (int(requested_match.group(1)) if requested_match
                       else int(manifest["nsteps"]))
    retries = len(re.findall(r"physical_step_id,attempt_id", log_text)) - 1
    phi_paths = checkpoint_map(case, "phi")
    x_paths = checkpoint_map(case, "xB_alpha")
    c_paths = checkpoint_map(case, "Ctot")
    steps = [0] + sorted(set(phi_paths) & set(x_paths) & set(c_paths))
    phi0 = load_field(case / "input/phi_init.raw", shape)
    x0 = load_field(case / "input/xB_init.raw", shape)
    c0 = load_field(case / "input/Ctot_init.raw", shape)
    dx_nm = float(manifest["dx_nm"])
    dx_code = float(manifest["dx_code"])
    lambda_nm = float(manifest["interface_width_nm"])
    dt = float(manifest["dt_code"])
    t0 = float(params["t_real_unit"])
    trajectory: list[tuple[int, float, float, float, float]] = []
    mass0 = float(c0.sum())
    max_mass_rel = 0.0
    final_phi, final_x, final_c = phi0, x0, c0
    previous_phi = phi0
    previous_c = c0
    previous_step = 0
    for step in steps:
        if step == 0:
            phi, x, c = phi0, x0, c0
        else:
            phi = load_field(phi_paths[step], shape)
            x = load_field(x_paths[step], shape)
            c = load_field(c_paths[step], shape)
        r_h, r_half = equivalent_radii(phi, dx_nm)
        mass_rel = abs(float(c.sum()) - mass0) / max(abs(mass0), 1.0e-300)
        max_mass_rel = max(max_mass_rel, mass_rel)
        trajectory.append((step, step * dt, r_h, r_half, mass_rel))
        if step < steps[-1]:
            previous_phi = phi
            previous_c = c
            previous_step = step
        final_phi, final_x, final_c = phi, x, c
    if len(trajectory) >= 2:
        fit = np.polyfit([row[1] for row in trajectory],
                         [row[3] * dx_code / dx_nm for row in trajectory], 1)
        velocity_code = float(fit[0])
        fit_h = np.polyfit([row[1] for row in trajectory],
                           [row[2] * dx_code / dx_nm for row in trajectory], 1)
        velocity_h_code = float(fit_h[0])
    else:
        velocity_code = math.nan
        velocity_h_code = math.nan
    final_radius = trajectory[-1][3]
    surface = radial_surface_metrics(
        final_phi, final_x, dx_nm, dx_code, lambda_nm, final_radius,
        float(params["temperature_C"]) + 273.15,
        float(params["mu_reference_scale"]), float(params["D_alpha"]),
    )
    correction_enabled = int(float(
        params.get("ctot_finite_interface_antitrapping_enabled", "0")
    )) != 0
    correction_flux = (radial_correction_flux(
        final_phi, previous_phi, final_x,
        (steps[-1] - previous_step) * dt, dx_code,
        lambda_nm * dx_code / dx_nm, float(params["v_B"]),
    ) if correction_enabled else 0.0)
    total_flux = surface["matrix_flux"] + correction_flux
    # The runtime equation is C_t=div(J_code), where J_code=M grad(mu).
    # For an outward beta normal and zero beta-side flux,
    # V_n (C_beta-C_alpha)=J_matrix dot n.
    stefan_velocity = total_flux / max(
        1.0 - surface["x_surface"], 1.0e-300)
    stefan_residual = velocity_code - stefan_velocity
    control = circular_control_volume_ledger(
        previous_phi, previous_c, final_phi, final_c, dx_nm,
        (steps[-1] - previous_step) * dt,
        final_radius + 2.0 * lambda_nm,
    )
    x_eq = unit.xAg2Te_eq_from_T(float(params["temperature_C"]) + 273.15)
    mu_eq = ((unit.mu_Ag2Te(float(params["temperature_C"]) + 273.15, x_eq) -
              unit.mu_PbTe(float(params["temperature_C"]) + 273.15, x_eq)) /
             float(params["mu_reference_scale"]))
    gamma = float(params["gamma_Jm2"])
    Vm = unit.USER_PHYSICAL_INPUTS.Vm_compound
    capillary_expected = (gamma * Vm /
                          max(final_radius * 1.0e-9, 1.0e-300) /
                          float(params["mu_reference_scale"]))
    energy_files = list(case.rglob("ctot_energy_work.csv"))
    max_energy_balance = max_correction_work = math.nan
    min_transport_dissipation = math.nan
    if energy_files:
        with energy_files[0].open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        if rows:
            energy_balance = [float(row["energy_balance_rel"]) for row in rows]
            energy_balance = [value for value in energy_balance if math.isfinite(value)]
            max_energy_balance = max(energy_balance) if energy_balance else math.nan
            max_correction_work = max(abs(float(row["W_finite_interface"])) for row in rows)
            min_transport_dissipation = min(float(row["D_transport"]) for row in rows)
    return {
        "case": case.name,
        "mode": manifest["mode"],
        "interface_points": manifest["runtime_lambda_over_dx"],
        "grid": "x".join(str(value) for value in shape),
        "dx_nm": dx_nm,
        "radius_initial_nm": trajectory[0][3],
        "radius_final_nm": final_radius,
        "radius_h_final_nm": trajectory[-1][2],
        "accepted_steps": accepted_steps,
        "requested_steps": requested_steps,
        "production_acceptance_allowed": manifest["production_acceptance_allowed"],
        "velocity_code": velocity_code,
        "velocity_h_code": velocity_h_code,
        "matrix_flux_code": surface["matrix_flux"],
        "finite_interface_flux_code": correction_flux,
        "finite_interface_correction_enabled": int(correction_enabled),
        "total_radial_flux_code": total_flux,
        "stefan_velocity_code": stefan_velocity,
        "stefan_residual_code": stefan_residual,
        "stefan_residual_h_code": velocity_h_code - stefan_velocity,
        "control_volume_Hdot": control["Hdot"],
        "control_volume_Qdot": control["Qdot"],
        "control_volume_Cdot": control["Cdot"],
        "control_volume_closure": control["closure"],
        "control_volume_velocity_h": control["velocity_h"],
        "control_volume_flux_velocity": control["control_flux"],
        "control_volume_matrix_storage_velocity": control["matrix_storage_velocity"],
        "mu_surface": surface["mu_surface"],
        "mu_equilibrium": mu_eq,
        "mu_jump": surface["mu_surface"] - mu_eq,
        "capillary_shift_expected_cylindrical": capillary_expected,
        "xB_surface": surface["x_surface"],
        "beta_side_flux": 0.0,
        "max_mass_error_rel": max_mass_rel,
        "max_energy_balance_rel": max_energy_balance,
        "min_transport_dissipation": min_transport_dissipation,
        "max_abs_correction_work": max_correction_work,
        "lower_bound_violations": failure["lower_violations"],
        "upper_bound_violations": failure["upper_violations"],
        "max_lower_defect": failure["max_lower_defect"],
        "retry_count": max(retries, 0),
        "failure_reason": failure["failure_reason"],
        "status": "PASS" if accepted_steps == requested_steps else "FAIL",
        "provenance": manifest["provenance"],
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    cases = sorted(path for path in args.matrix_root.iterdir()
                   if (path / "input/benchmark_manifest.json").exists())
    rows = [analyze_case(path) for path in cases]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"cases={len(rows)}")
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
