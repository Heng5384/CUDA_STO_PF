#!/usr/bin/env python3
"""Analyze matched Correction-2 runtime cases against their common sharp state."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402
from scripts.correction1_planar_sharp_oracle import PlanarSharpOracle  # noqa: E402
from scripts.analyze_research2_fast_interface_plateau import (  # noqa: E402
    candidate_mu_and_mobility,
    face_flux,
    parse_params,
)


def h_switch(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def final_checkpoint(case_dir: Path, field: str) -> tuple[int, Path]:
    paths = list((case_dir / "run").rglob(f"ctot_checkpoint_step*_{field}.raw"))
    parsed: list[tuple[int, Path]] = []
    for path in paths:
        match = re.search(r"step(\d+)_", path.name)
        if match:
            parsed.append((int(match.group(1)), path))
    if not parsed:
        raise FileNotFoundError(f"no {field} checkpoint under {case_dir}")
    return max(parsed)


def crossings(line: np.ndarray, dx_nm: float) -> list[float]:
    found: list[float] = []
    for index, left in enumerate(line):
        right = line[(index + 1) % line.size]
        if (left - 0.5) * (right - 0.5) < 0.0:
            found.append((index + (0.5 - left) / (right - left)) * dx_nm)
    if len(found) != 2:
        raise RuntimeError(f"expected two crossings, found {found}")
    return sorted(found)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def unique_result_file(case_dir: Path, name: str) -> Path:
    paths = list((case_dir / "run").rglob(name))
    if len(paths) != 1:
        raise RuntimeError(f"expected one {name} under {case_dir}, got {len(paths)}")
    return paths[0]


def continue_sharp(case_dir: Path, elapsed_s: float) -> dict[str, float | int]:
    payload = np.load(case_dir / "sharp_state.npz")
    state0 = np.asarray(payload["state"], dtype=np.float64)
    temperature_c = float(payload["temperature_c"])
    matrix_x = float(payload["matrix_x"])
    x_eq = unit.xAg2Te_eq_from_T(temperature_c + 273.15)
    # The historical constructor creates an unused growth-only similarity
    # target. The conservative ALE equations also support dissolution, so
    # mirror the matched-state generator and restore the requested composition.
    oracle = PlanarSharpOracle(
        temperature_c=temperature_c,
        domain_nm=float(payload["domain_nm"]),
        matrix_x=max(matrix_x, x_eq + 1.0e-12),
        start_time_s=float(payload["preage_s"]),
        cells=int(payload["sharp_cells"]),
        pf_dx_nm=json.loads((case_dir / "runtime_manifest.json").read_text())["dx_nm"],
    )
    oracle.matrix_x = matrix_x
    mass0 = oracle.inventory(state0)
    solution = solve_ivp(
        oracle.rhs, (0.0, elapsed_s), state0, method="BDF",
        rtol=2.0e-10, atol=2.0e-12,
        max_step=max(elapsed_s / 200.0, 1.0e-10),
    )
    if not solution.success:
        raise RuntimeError(solution.message)
    state1 = solution.y[:, -1]
    radius0 = float(state0[-1])
    radius1 = float(state1[-1])
    return {
        "sharp_radius_initial_nm": radius0,
        "sharp_radius_final_nm": radius1,
        "sharp_displacement_nm": radius1 - radius0,
        "sharp_velocity_nm_s": (radius1 - radius0) / elapsed_s,
        "sharp_velocity_initial_nm_s": oracle.velocity(state0),
        "sharp_velocity_final_nm_s": oracle.velocity(state1),
        "sharp_mass_error_rel": abs(oracle.inventory(state1) - mass0)
        / max(abs(mass0), 1.0),
        "sharp_solver_steps": len(solution.t) - 1,
    }


def analyze_case(case_dir: Path) -> dict[str, object]:
    manifest = json.loads((case_dir / "runtime_manifest.json").read_text())
    nx, ny, nz = map(int, manifest["grid"])
    shape = (nx, ny, nz)
    phi0 = np.fromfile(case_dir / "phi_init.raw", dtype=np.float64).reshape(shape)
    c0 = np.fromfile(case_dir / "Ctot_init.raw", dtype=np.float64).reshape(shape)
    step_phi, phi_path = final_checkpoint(case_dir, "phi")
    step_c, c_path = final_checkpoint(case_dir, "Ctot")
    step_x, x_path = final_checkpoint(case_dir, "xB_alpha")
    if not (step_phi == step_c == step_x):
        raise RuntimeError("checkpoint step mismatch")
    phi1 = np.fromfile(phi_path, dtype=np.float64).reshape(shape)
    c1 = np.fromfile(c_path, dtype=np.float64).reshape(shape)
    x1 = np.fromfile(x_path, dtype=np.float64).reshape(shape)
    h0 = h_switch(phi0)
    h1 = h_switch(phi1)
    dx_nm = float(manifest["dx_nm"])
    elapsed_s = float(manifest["elapsed_s"])
    area = ny * nz
    delta_h = float((h1 - h0).sum())
    h_displacement = delta_h * dx_nm / (2.0 * area)
    p0 = phi0.mean(axis=(1, 2))
    p1 = phi1.mean(axis=(1, 2))
    cross0 = crossings(p0, dx_nm)
    cross1 = crossings(p1, dx_nm)
    crossing_displacement = 0.5 * (
        (cross1[1] - cross1[0]) - (cross0[1] - cross0[0])
    )
    q0 = c0 - h0
    q1 = c1 - h1
    stefan_residual = delta_h + float((q1 - q0).sum())
    mass_error = abs(float((c1 - c0).sum())) / max(abs(float(c0.sum())), 1.0)

    log = (case_dir / "run/run.log").read_text(errors="replace")
    accepts = re.findall(r"CTOT_MIMETIC_BE_ACCEPT[^\n]+", log)
    retries = len(re.findall(r"CTOT_COUPLED_STEP_RETRY", log))
    rejects = len(re.findall(r"CTOT_MIMETIC_BE_REJECT", log))
    mass_runtime = [abs(float(value)) for value in re.findall(
        r"CTOT_MIMETIC_BE_ACCEPT[^\n]*mass_error=([^ ]+)", log
    )]
    predicate = read_rows(unique_result_file(case_dir, "ctot_acceptance_predicate.csv"))
    predicate_pass = len(predicate) == int(manifest["nsteps"]) and all(
        all(value == "1" for key, value in row.items()
            if key not in ("physical_step_id", "attempt_id"))
        for row in predicate
    )
    energy = read_rows(unique_result_file(case_dir, "ctot_energy_work.csv"))
    energy_pass = len(energy) == int(manifest["nsteps"]) and all(
        row["accepted"] == "1" and row["monotone_pass"] == "1"
        and row["balance_pass"] == "1" for row in energy
    )
    energy_rel_max = max(
        (abs(float(row["energy_balance_rel"])) for row in energy), default=math.nan
    )
    outer = read_rows(unique_result_file(case_dir, "ctot_outer_iterations.csv"))
    converged = [row for row in outer if row["status"] == "CONVERGED"]
    kkt_max = max((abs(float(row["phase_KKT_residual"])) for row in converged),
                  default=math.nan)
    storage_max = max(
        (abs(float(row["local_phase_storage_residual"])) for row in converged),
        default=math.nan,
    )
    sharp = continue_sharp(case_dir, elapsed_s)
    pf_velocity = h_displacement / elapsed_s
    sharp_velocity = float(sharp["sharp_velocity_nm_s"])
    velocity_error = abs(pf_velocity - sharp_velocity) / max(abs(sharp_velocity), 1.0e-300)
    line_x = x1.mean(axis=(1, 2))
    initial_line_x = np.fromfile(
        case_dir / "xB_init.raw", dtype=np.float64
    ).reshape(shape).mean(axis=(1, 2))
    matrix_mask = p1 < 0.1
    initial_matrix_mask = p0 < 0.1
    far_x = float(np.max(line_x[matrix_mask]))
    initial_far_x = float(np.max(initial_line_x[initial_matrix_mask]))
    near_x = float(np.min(line_x[matrix_mask]))
    params = parse_params(case_dir / "runtime.params")
    mu_line, mobility_line = candidate_mu_and_mobility(p1, line_x, params)
    direct_flux = face_flux(mu_line, mobility_line, float(manifest["dx_code"]))
    interface_faces = [int(math.floor(value / dx_nm)) % nx for value in cross1]
    matrix_face_flux_direct_code = float(np.mean([
        abs(direct_flux[index]) for index in interface_faces
    ]))
    temperature_k = 673.15
    x_eq = unit.xAg2Te_eq_from_T(temperature_k)
    mu_ref = float(params["mu_reference_scale"])
    mu_b_eq = unit.mu_Ag2Te(temperature_k, x_eq)
    interface_mu_jump = (unit.mu_Ag2Te(temperature_k, near_x) - mu_b_eq) / mu_ref
    far_reaction_drive = (unit.mu_Ag2Te(temperature_k, far_x) - mu_b_eq) / mu_ref
    numerical_pass = (
        step_phi == int(manifest["nsteps"])
        and len(accepts) == int(manifest["nsteps"])
        and retries == 0 and rejects == 0 and predicate_pass and energy_pass
        and mass_error <= 1.0e-10
        and max(mass_runtime, default=math.inf) <= 1.0e-10
        and abs(stefan_residual) / max(abs(delta_h), 1.0) <= 1.0e-10
        and storage_max <= 1.0e-12
        and float(phi1.min()) >= -1.0e-12 and float(phi1.max()) <= 1.0 + 1.0e-12
        and float(x1.min()) >= 0.0 and float(x1.max()) <= 1.0
        and all("clip_count=0" in row and "projection_mass=0" in row for row in accepts)
    )
    return {
        "case": case_dir.name,
        "study": manifest.get("study", "unspecified"),
        "drive_amplitude": manifest.get("drive_amplitude", math.nan),
        "state_case": manifest["state_case"],
        "matrix_xB": manifest["matrix_xB"],
        "preage_Fo": manifest["preage_Fo"],
        "target_incremental_Fo": manifest["target_incremental_Fo"],
        "L_phi_ratio": manifest["L_phi_ratio"],
        "dt_code": manifest["dt_code"],
        "nsteps": manifest["nsteps"],
        "elapsed_s": elapsed_s,
        "accepted_steps": len(accepts),
        "retry_count": retries,
        "reject_count": rejects,
        "PF_h_displacement_nm": h_displacement,
        "PF_crossing_displacement_nm": crossing_displacement,
        "PF_velocity_h_nm_s": pf_velocity,
        "PF_velocity_crossing_nm_s": crossing_displacement / elapsed_s,
        **sharp,
        "PF_sharp_velocity_error_rel": velocity_error,
        "beta_h_gain_cell_units": delta_h,
        "matrix_q_change_cell_units": float((q1 - q0).sum()),
        "matrix_transfer_rate_inferred_nm_s": h_displacement / elapsed_s,
        "matrix_face_flux_direct_code": matrix_face_flux_direct_code,
        "interface_muB_jump_hat": interface_mu_jump,
        "far_field_reaction_drive_hat": far_reaction_drive,
        "stefan_storage_residual_rel": abs(stefan_residual) / max(abs(delta_h), 1.0),
        "mass_error_rel_recomputed": mass_error,
        "runtime_mass_error_max": max(mass_runtime, default=math.nan),
        "phase_KKT_max": kkt_max,
        "phase_storage_residual_max": storage_max,
        "energy_balance_rel_max": energy_rel_max,
        "far_field_xB": far_x,
        "initial_far_field_xB": initial_far_x,
        "interface_side_xB": near_x,
        "phi_min": float(phi1.min()),
        "phi_max": float(phi1.max()),
        "xB_min": float(x1.min()),
        "xB_max": float(x1.max()),
        "acceptance_predicate_pass": predicate_pass,
        "energy_work_pass": energy_pass,
        "finite_interface_mode": "off",
        "elasticity": "off",
        "GP_S3": "off",
        "numerical_hard_gates_pass": numerical_pass,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for path in sorted(args.root.iterdir()):
        if (path / "runtime_manifest.json").is_file() and (path / "run/run.log").is_file():
            rows.append(analyze_case(path))
    if not rows:
        raise RuntimeError("no completed runtime cases")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"correction2_analyzed_cases={len(rows)}")
    print(f"correction2_numerical_pass={sum(bool(row['numerical_hard_gates_pass']) for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
