#!/usr/bin/env python3
"""Analyze Research2 source-free planar L_phi plateau checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

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


def crossings(line: np.ndarray, dx_nm: float) -> list[float]:
    found = []
    for idx, left in enumerate(line):
        right = line[(idx + 1) % line.size]
        if (left - 0.5) * (right - 0.5) < 0.0:
            found.append((idx + (0.5 - left) / (right - left)) * dx_nm)
    if len(found) != 2:
        raise RuntimeError(f"expected two phi=0.5 crossings, got {found}")
    return sorted(found)


def final_checkpoint(run_dir: Path, field: str) -> tuple[int, Path]:
    paths = list(run_dir.rglob(f"ctot_checkpoint_step*_{field}.raw"))
    parsed = []
    for path in paths:
        match = re.search(r"step(\d+)_", path.name)
        if match:
            parsed.append((int(match.group(1)), path))
    if not parsed:
        raise FileNotFoundError(f"no {field} checkpoint under {run_dir}")
    return max(parsed, key=lambda item: item[0])


def read_max_csv_value(run_dir: Path, pattern: str, candidates: list[str]) -> float:
    paths = list(run_dir.rglob(pattern))
    if not paths:
        return math.nan
    values = []
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                for key in candidates:
                    if key in row and row[key] not in ("", None):
                        try:
                            values.append(abs(float(row[key])))
                        except ValueError:
                            pass
                        break
    return max(values, default=math.nan)


def read_csv_rows(run_dir: Path, pattern: str) -> list[dict[str, str]]:
    paths = list(run_dir.rglob(pattern))
    if len(paths) != 1:
        raise RuntimeError(
            f"expected exactly one {pattern} under {run_dir}, got {len(paths)}"
        )
    with paths[0].open(newline="") as handle:
        return list(csv.DictReader(handle))


def accepted_state_gate_metrics(
    run_dir: Path, expected_steps: int
) -> tuple[bool, float, float, int]:
    predicates = read_csv_rows(run_dir, "ctot_acceptance_predicate.csv")
    if len(predicates) != expected_steps:
        return False, math.nan, math.nan, len(predicates)
    predicate_columns = [
        key for key in predicates[0]
        if key not in ("physical_step_id", "attempt_id")
    ]
    predicates_pass = all(
        all(row.get(key) == "1" for key in predicate_columns)
        for row in predicates
    )

    outer = read_csv_rows(run_dir, "ctot_outer_iterations.csv")
    converged = [row for row in outer if row.get("status") == "CONVERGED"]
    accepted_kkt = max(
        (abs(float(row["phase_KKT_residual"])) for row in converged),
        default=math.nan,
    )
    accepted_storage = max(
        (abs(float(row["local_phase_storage_residual"])) for row in converged),
        default=math.nan,
    )
    final_outer_pass = len(converged) == expected_steps

    energy = read_csv_rows(run_dir, "ctot_energy_work.csv")
    energy_pass = len(energy) == expected_steps and all(
        row.get("accepted") == "1" and row.get("monotone_pass") == "1"
        and row.get("balance_pass") == "1"
        for row in energy
    )
    return (
        predicates_pass and final_outer_pass and energy_pass,
        accepted_kkt,
        accepted_storage,
        len(predicates),
    )


def candidate_mu_and_mobility(
    phi: np.ndarray, x_b: np.ndarray, params: dict[str, str]
) -> tuple[np.ndarray, np.ndarray]:
    temperature_k = float(params["temperature_C"]) + 273.15
    mu_ref = float(params["mu_reference_scale"])
    vm_a0 = float(params["Vm_alpha_0"])
    dvm = float(params["dVm_alpha_dxB"])
    vm_b = float(params["Vm_compound"])
    if "mu0_compound" in params:
        mu0_b = float(params["mu0_compound"])
    else:
        x_eq = unit.xAg2Te_eq_from_T(temperature_k)
        v_a = float(params["v_A"])
        v_b = float(params["v_B"])
        mu_a_eq = unit.mu_PbTe(temperature_k, x_eq) / mu_ref
        mu_b_eq = unit.mu_Ag2Te(temperature_k, x_eq) / mu_ref
        mu0_b = (v_a * mu_a_eq + v_b * mu_b_eq) / (v_a + v_b)
    d_alpha = float(params["D_alpha"])
    hp = h(phi)
    x = np.clip(x_b, 1.0e-12, 1.0 - 1.0e-12)
    mu_a = np.array([unit.mu_PbTe(temperature_k, value) for value in x]) / mu_ref
    mu_b = np.array([unit.mu_Ag2Te(temperature_k, value) for value in x]) / mu_ref
    vm_a = vm_a0 + dvm * x
    c_bulk = 1.0 / (vm_a * (1.0 - hp) + vm_b * hp)
    mu_mix = (1.0 - hp) * ((1.0 - x) * mu_a + x * mu_b) + hp * mu0_b
    mu = c_bulk * (mu_b - mu_a - c_bulk * mu_mix * dvm)
    c_alpha = 1.0 / vm_a
    gamma = c_alpha * np.array([
        unit.g_alpha_second_unified(temperature_k, value) for value in x
    ]) / mu_ref
    mobility = (1.0 - hp) * d_alpha / gamma
    mobility_mode = params.get("coarse_interface_mobility_mode", "off")
    a_m = float(params.get("coarse_interface_mobility_a_M", "0"))
    if mobility_mode == "INTERFACE_BAND_BOOST_V1":
        mobility *= 1.0 + a_m * (4.0 * hp * (1.0 - hp))
    elif mobility_mode != "off":
        raise ValueError(f"unsupported coarse mobility mode {mobility_mode}")
    matrix_support_eps = float(params.get("ctot_matrix_support_eps", "1e-10"))
    mobility[(1.0 - hp) <= matrix_support_eps] = 0.0
    return mu, mobility


def face_flux(line_mu: np.ndarray, line_mobility: np.ndarray, dx_code: float) -> np.ndarray:
    right_mu = np.roll(line_mu, -1)
    right_m = np.roll(line_mobility, -1)
    denom = line_mobility + right_m
    active = (line_mobility > 0.0) & (right_m > 0.0)
    m_face = np.zeros_like(line_mobility)
    np.divide(2.0 * line_mobility * right_m, denom, out=m_face, where=active)
    return m_face * (right_mu - line_mu) / dx_code


def periodic_distance(x: np.ndarray, point: float, length: float) -> np.ndarray:
    delta = np.abs(x - point)
    return np.minimum(delta, length - delta)


def relative_change(a: float, b: float) -> float:
    return abs(b - a) / max(abs(a), abs(b), 1.0e-30)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--mode-json", type=Path, required=True)
    args = parser.parse_args()

    metrics: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    for case_dir in sorted(path for path in args.matrix_root.iterdir()
                           if path.is_dir()):
        status_path = case_dir / "status.json"
        manifest_path = case_dir / "input/benchmark_manifest.json"
        if not status_path.is_file() or not manifest_path.is_file():
            continue
        status = json.loads(status_path.read_text())
        manifest = json.loads(manifest_path.read_text())
        params = parse_params(case_dir / "input/benchmark.params")
        grid = tuple(map(int, manifest["grid"]))
        nx, ny, nz = grid
        area_cells = ny * nz
        dx_nm = float(manifest["dx_nm"])
        dx_code = float(manifest["dx_code"])
        domain_nm = nx * dx_nm
        phi0 = np.fromfile(case_dir / "input/phi_init.raw", dtype=np.float64)
        x0 = np.fromfile(case_dir / "input/xB_init.raw", dtype=np.float64)
        c0 = np.fromfile(case_dir / "input/Ctot_init.raw", dtype=np.float64)
        final_step, phi_path = final_checkpoint(case_dir / "run", "phi")
        c_step, c_path = final_checkpoint(case_dir / "run", "Ctot")
        x_step, x_path = final_checkpoint(case_dir / "run", "xB_alpha")
        if not (final_step == c_step == x_step):
            raise RuntimeError(f"checkpoint step mismatch in {case_dir}")
        phi1 = np.fromfile(phi_path, dtype=np.float64)
        x1 = np.fromfile(x_path, dtype=np.float64)
        c1 = np.fromfile(c_path, dtype=np.float64)
        if any(array.size != nx * ny * nz for array in (phi0, x0, c0, phi1, x1, c1)):
            raise RuntimeError(f"checkpoint shape mismatch in {case_dir}")

        p0 = phi0.reshape(grid).mean(axis=(1, 2))
        p1 = phi1.reshape(grid).mean(axis=(1, 2))
        x0_line = x0.reshape(grid).mean(axis=(1, 2))
        x1_line = x1.reshape(grid).mean(axis=(1, 2))
        c0_line = c0.reshape(grid).mean(axis=(1, 2))
        c1_line = c1.reshape(grid).mean(axis=(1, 2))
        cross0 = crossings(p0, dx_nm)
        cross1 = crossings(p1, dx_nm)
        elapsed_s = float(manifest["final_physical_time_s"])
        displacement_nm = 0.5 * ((cross1[1] - cross1[0]) -
                                 (cross0[1] - cross0[0]))
        velocity_nm_s = displacement_nm / elapsed_s
        h0 = h(phi0)
        h1 = h(phi1)
        delta_h = float((h1 - h0).sum())
        delta_q = float(((c1 - h1) - (c0 - h0)).sum())
        beta_gain_per_area = delta_h / area_cells
        ledger_flux = beta_gain_per_area / (2.0 * float(manifest["final_code_time"]))

        mu, mobility = candidate_mu_and_mobility(p1, x1_line, params)
        flux = face_flux(mu, mobility, dx_code)
        interface_faces = []
        for crossing in cross1:
            interface_faces.append(int(math.floor(crossing / dx_nm)) % nx)
        direct_flux = float(np.mean([abs(flux[index]) for index in interface_faces]))

        coordinates = np.arange(nx) * dx_nm
        d_interface = np.minimum(periodic_distance(coordinates, cross1[0], domain_nm),
                                 periodic_distance(coordinates, cross1[1], domain_nm))
        matrix_mask = p1 < 0.1
        far_mask = matrix_mask & (d_interface > 0.25 * (cross1[0] + domain_nm - cross1[1]))
        if not np.any(far_mask):
            far_mask = matrix_mask & (d_interface > 2.0)
        near_mask = matrix_mask & (d_interface <= 2.0)
        far_x = float(np.mean(x1_line[far_mask]))
        interface_x = float(np.min(x1_line[near_mask]))
        depletion_amplitude = max(far_x - interface_x, 0.0)
        if depletion_amplitude > 0.0:
            depleted = matrix_mask & ((far_x - x1_line) >= 0.1 * depletion_amplitude)
            depletion_width = float(np.max(d_interface[depleted])) if np.any(depleted) else 0.0
        else:
            depletion_width = 0.0

        run_log = (case_dir / "run/run.log").read_text(errors="replace")
        accepts = re.findall(r"CTOT_MIMETIC_BE_ACCEPT[^\n]+", run_log)
        mass_values = [float(value) for value in
                       re.findall(r"CTOT_MIMETIC_BE_ACCEPT[^\n]*mass_error=([^ ]+)", run_log)]
        nonlinear = [int(value) for value in re.findall(
            r"CTOT_MIMETIC_BE_ACCEPT[^\n]*nonlinear_iters=(\d+)", run_log)]
        outer = [int(value) for value in re.findall(
            r"CTOT_MIMETIC_BE_ACCEPT[^\n]*outer_iters=(\d+)", run_log)]
        pcg = [int(value) for value in re.findall(r"pcg_iters=(\d+)", run_log)]
        (accepted_state_audit_pass, kkt_max, storage_max,
         acceptance_predicate_rows) = accepted_state_gate_metrics(
             case_dir / "run", int(status["nsteps"])
         )
        energy_balance_max = read_max_csv_value(
            case_dir / "run", "ctot_energy_work.csv",
            ["energy_balance_rel", "balance_relative_residual"])
        mass_rel = abs(float((c1 - c0).sum())) / max(abs(float(c0.sum())), 1.0)
        stefan_rel = abs(delta_h + delta_q) / max(abs(delta_h), abs(delta_q), 1.0)
        q1 = c1 - h1
        c_lower_margin = float(np.min(c1 - h1))
        c_upper_margin = float(np.min(1.0 - c1))
        q_lower_margin = float(np.min(q1))
        q_upper_margin = float(np.min((1.0 - h1) - q1))
        numerical_pass = (
            int(status["returncode"]) == 0 and len(accepts) == int(status["nsteps"]) and
            accepted_state_audit_pass and
            max(map(abs, mass_values), default=math.inf) <= 1.0e-10 and
            mass_rel <= 1.0e-10 and stefan_rel <= 1.0e-10 and
            float(phi1.min()) >= -1.0e-12 and float(phi1.max()) <= 1.0 + 1.0e-12 and
            float(x1.min()) >= 0.0 and float(x1.max()) <= 1.0 and
            min(c_lower_margin, c_upper_margin, q_lower_margin, q_upper_margin) >= -1.0e-12 and
            all("clip_count=0" in line and "projection_mass=0" in line for line in accepts)
        ) if accepts else False
        metrics.append({
            "case": case_dir.name,
            "temperature_C": status["temperature_C"],
            "L_phi_factor": status["L_phi_factor"],
            "L_phi_reference_code": status["L_phi_reference_code"],
            "L_phi_reference_physical": status["L_phi_reference_physical"],
            "L_phi_code": status["L_phi_code"],
            "L_phi_physical": (float(status["L_phi_factor"]) *
                               float(status["L_phi_reference_physical"])),
            "dx_nm": dx_nm,
            "interface_resolution": manifest["interface_resolution"],
            "dt_code": status["dt_code"],
            "nsteps": status["nsteps"],
            "elapsed_s": elapsed_s,
            "phi_half_displacement_nm": displacement_nm,
            "full_window_velocity_nm_s": velocity_nm_s,
            "beta_h_inventory_gain_cell_units": delta_h,
            "beta_h_gain_per_interface_area_cells": beta_gain_per_area,
            "matrix_flux_from_stefan_code": ledger_flux,
            "matrix_side_face_flux_direct_code": direct_flux,
            "far_field_xB": far_x,
            "interface_side_matrix_xB": interface_x,
            "depletion_amplitude": depletion_amplitude,
            "depletion_width_nm": depletion_width,
            "stefan_storage_residual_rel": stefan_rel,
            "mass_error_rel_recomputed": mass_rel,
            "runtime_mass_error_max": max(map(abs, mass_values), default=math.nan),
            "phi_min": float(phi1.min()),
            "phi_max": float(phi1.max()),
            "xB_min": float(x1.min()),
            "xB_max": float(x1.max()),
            "phase_kkt_max": kkt_max,
            "phase_storage_residual_max": storage_max,
            "energy_balance_rel_max": energy_balance_max,
            "accepted_state_audit_pass": accepted_state_audit_pass,
            "acceptance_predicate_rows": acceptance_predicate_rows,
            "Ctot_lower_margin_min": c_lower_margin,
            "Ctot_upper_margin_min": c_upper_margin,
            "q_alpha_lower_margin_min": q_lower_margin,
            "q_alpha_upper_margin_min": q_upper_margin,
            "nonlinear_iters_max": max(nonlinear, default=-1),
            "outer_iters_max": max(outer, default=-1),
            "phase_pcg_iters_max": max(pcg, default=-1),
            "retry_count": status["retry_count"],
            "wall_time_s": status["wall_time_s"],
            "finite_interface_mode": "off",
            "numerical_hard_gates_pass": numerical_pass,
        })
        for idx in range(nx):
            profiles.append({
                "case": case_dir.name,
                "L_phi_factor": status["L_phi_factor"],
                "x_nm": coordinates[idx],
                "phi_initial": p0[idx],
                "phi_final": p1[idx],
                "xB_initial": x0_line[idx],
                "xB_final": x1_line[idx],
                "Ctot_initial": c0_line[idx],
                "Ctot_final": c1_line[idx],
                "mu_final": mu[idx],
                "matrix_face_flux_final": flux[idx],
                "distance_to_interface_nm": d_interface[idx],
            })

    if not metrics:
        raise RuntimeError("no completed Research2 plateau cases found")
    metrics.sort(key=lambda row: float(row["L_phi_factor"]))
    comparisons = []
    selected = None
    for low, high in zip(metrics, metrics[1:]):
        velocity_change = relative_change(float(low["full_window_velocity_nm_s"]),
                                          float(high["full_window_velocity_nm_s"]))
        gain_change = relative_change(float(low["beta_h_inventory_gain_cell_units"]),
                                      float(high["beta_h_inventory_gain_cell_units"]))
        flux_change = relative_change(float(low["matrix_flux_from_stefan_code"]),
                                      float(high["matrix_flux_from_stefan_code"]))
        same_direction = math.copysign(1.0, float(low["full_window_velocity_nm_s"])) == \
            math.copysign(1.0, float(high["full_window_velocity_nm_s"]))
        hard = bool(low["numerical_hard_gates_pass"]) and bool(high["numerical_hard_gates_pass"])
        passed = max(velocity_change, gain_change, flux_change) <= 0.10 and same_direction and hard
        strict = max(velocity_change, gain_change, flux_change) <= 0.05 and same_direction and hard
        comparison = {
            "low_factor": low["L_phi_factor"],
            "high_factor": high["L_phi_factor"],
            "velocity_change_rel": velocity_change,
            "beta_gain_change_rel": gain_change,
            "matrix_flux_change_rel": flux_change,
            "same_direction": same_direction,
            "hard_gates_pass": hard,
            "plateau_10pct_pass": passed,
            "plateau_5pct_pass": strict,
        }
        comparisons.append(comparison)
        if selected is None and passed:
            selected = low

    args.metrics.parent.mkdir(parents=True, exist_ok=True)
    with args.metrics.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    args.profiles.parent.mkdir(parents=True, exist_ok=True)
    with args.profiles.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(profiles[0]))
        writer.writeheader()
        writer.writerows(profiles)

    plateau_status = "PASS" if selected else "BLOCKED_NO_FAST_INTERFACE_PLATEAU"
    mode = {
        "research_model": "fixed_ctot_gp_reservoir_to_beta_diffusion_limit_v1",
        "matrix_to_beta_kinetics": "FAST_INTERFACE_DIFFUSION_CONTROLLED_LIMIT",
        "finite_interface_mode": "off",
        "L_phi_reference_code": metrics[0]["L_phi_reference_code"],
        "L_phi_reference_physical": metrics[0]["L_phi_reference_physical"],
        "plateau_status": plateau_status,
        "L_phi_fast_factor": selected["L_phi_factor"] if selected else None,
        "L_phi_fast_code": selected["L_phi_code"] if selected else None,
        "L_phi_fast_physical": selected["L_phi_physical"] if selected else None,
        "absolute_interface_mobility_claimed": False,
        "source_free_only": True,
    }
    args.mode_json.parent.mkdir(parents=True, exist_ok=True)
    args.mode_json.write_text(json.dumps(mode, indent=2) + "\n")

    table = "\n".join(
        f"| {c['low_factor']} | {c['high_factor']} | "
        f"{c['velocity_change_rel']:.6g} | {c['beta_gain_change_rel']:.6g} | "
        f"{c['matrix_flux_change_rel']:.6g} | {c['plateau_10pct_pass']} |"
        for c in comparisons
    )
    report = f"""# Research2 Fast-Interface Plateau

## Scope

T{metrics[0]['temperature_C']} source-free periodic planar beta slab,
`ctot_mimetic_be / mimetic_shared_face_v1`, physical `D_alpha(T)`, elasticity
OFF, GP/S3 OFF, automatic dt growth OFF, and finite-interface correction OFF.
The reference is the source-derived one-sided diagonal value
`L_phi_ref={metrics[0]['L_phi_reference_code']}` code units
(`{metrics[0]['L_phi_reference_physical']}` physical units). This scan is an internal limiting
scenario test, not an independent mobility measurement.

`matrix_flux_from_stefan_code` is the full-window matrix-side transfer inferred
from the exactly closed `delta h + delta q_alpha` ledger for the one-sided
`D_beta=0` slab. The final-time direct face flux is retained as a profile
diagnostic, not substituted for the full-window plateau observable.

Every plateau row additionally requires all per-step accepted-state predicates,
converged-state KKT/storage, energy/work acceptance, mass/storage closure,
bounds, zero clipping, and zero physical projection.

## Adjacent-pair plateau test

| low factor | high factor | velocity change | beta gain change | flux change | pass <=10% |
|---:|---:|---:|---:|---:|---|
{table}

## Decision

`fast_interface_plateau_status={plateau_status}`

`L_phi_fast_factor={selected['L_phi_factor'] if selected else 'NOT_SELECTED'}`

`L_phi_fast_code={selected['L_phi_code'] if selected else 'NOT_SELECTED'}`

The selected value, when present, is the smallest tested row that passes
against the next larger value. If no row is selected, GP source integration is
gated and only the planar range may be extended.

`absolute_interface_mobility_claimed=false`
"""
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(report)
    print(f"plateau_status={plateau_status}")
    print(f"L_phi_fast={selected['L_phi_code'] if selected else 'NOT_SELECTED'}")
    return 0 if selected else 2


if __name__ == "__main__":
    raise SystemExit(main())
