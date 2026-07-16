#!/usr/bin/env python3
"""Analyze JC4 CUDA long-planar runs against the independent sharp oracle."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re
import sys
from typing import Iterable

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/pf_ctot_production_candidate"
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402


def h_switch(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        raise ValueError(f"refusing to write empty table {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def unique_result_file(case: Path, name: str) -> Path:
    found = list((case / "run").rglob(name))
    if len(found) != 1:
        raise RuntimeError(f"expected one {name} under {case}, got {len(found)}")
    return found[0]


def unique_result_glob(case: Path, pattern: str) -> Path:
    found = list((case / "run").rglob(pattern))
    if len(found) != 1:
        raise RuntimeError(
            f"expected one result matching {pattern} under {case}, got {len(found)}"
        )
    return found[0]


def interpolate_sharp_state(rows: list[dict[str, str]], time_s: float) -> dict[str, float]:
    times = np.array([float(row["time_s"]) for row in rows])
    if time_s < times[0] - 1.0e-12 or time_s > times[-1] + 1.0e-9:
        raise ValueError(
            f"PF time {time_s:.17e} lies outside sharp reference "
            f"[{times[0]:.17e}, {times[-1]:.17e}]"
        )
    result: dict[str, float] = {"time_s": time_s}
    for key in ("radius_nm", "displacement_nm", "matrix_mean_xB"):
        values = np.array([float(row[key]) for row in rows])
        result[key] = float(np.interp(time_s, times, values))
    return result


def final_checkpoint(case: Path, field: str) -> tuple[int, Path]:
    parsed: list[tuple[int, Path]] = []
    for path in (case / "run").rglob(f"ctot_checkpoint_step*_{field}.raw"):
        match = re.search(r"step(\d+)_", path.name)
        if match:
            parsed.append((int(match.group(1)), path))
    if not parsed:
        raise FileNotFoundError(f"no {field} checkpoint under {case}")
    return max(parsed)


def checkpoint_series(case: Path, field: str) -> list[tuple[int, Path]]:
    parsed: list[tuple[int, Path]] = []
    for path in (case / "run").rglob(f"ctot_checkpoint_step*_{field}.raw"):
        match = re.search(r"step(\d+)_", path.name)
        if match:
            parsed.append((int(match.group(1)), path))
    return sorted(parsed)


def crossings(phi: np.ndarray, dx_nm: float) -> list[float]:
    result: list[float] = []
    for index, left in enumerate(phi):
        right = phi[(index + 1) % phi.size]
        if (left - 0.5) * (right - 0.5) < 0.0:
            result.append((index + (0.5 - left) / (right - left)) * dx_nm)
    return sorted(result)


def half_width_from_crossings(phi: np.ndarray, dx_nm: float) -> float:
    points = crossings(phi, dx_nm)
    if len(points) != 2:
        return math.nan
    return 0.5 * (points[1] - points[0])


def growth_law_exponent(times: np.ndarray, displacement: np.ndarray) -> float:
    magnitude = np.abs(displacement)
    if magnitude.size < 4 or float(np.max(magnitude)) <= 0.0:
        return math.nan
    mask = (times > 0.0) & (magnitude >= max(0.5, 0.1 * float(np.max(magnitude))))
    if int(np.count_nonzero(mask)) < 3:
        return math.nan
    return float(np.polyfit(np.log(times[mask]), np.log(magnitude[mask]), 1)[0])


def analyze_case(case: Path) -> tuple[dict[str, object], list[dict[str, object]],
                                      list[dict[str, object]]]:
    manifest = json.loads((case / "runtime_manifest.json").read_text())
    nx, ny, nz = map(int, manifest["grid"])
    shape = (nx, ny, nz)
    dx = float(manifest["dx_nm"])
    phi0 = np.fromfile(case / "phi_init.raw", dtype=np.float64).reshape(shape)
    x0 = np.fromfile(case / "xB_init.raw", dtype=np.float64).reshape(shape)
    C0 = np.fromfile(case / "Ctot_init.raw", dtype=np.float64).reshape(shape)
    step_phi, phi_path = final_checkpoint(case, "phi")
    step_x, x_path = final_checkpoint(case, "xB_alpha")
    step_C, C_path = final_checkpoint(case, "Ctot")
    if not (step_phi == step_x == step_C):
        raise RuntimeError("final checkpoint fields do not share a step")
    phi1 = np.fromfile(phi_path, dtype=np.float64).reshape(shape)
    x1 = np.fromfile(x_path, dtype=np.float64).reshape(shape)
    C1 = np.fromfile(C_path, dtype=np.float64).reshape(shape)
    h0, h1 = h_switch(phi0), h_switch(phi1)
    area = ny * nz
    h0_line = float(np.sum(h0) / area)
    h1_line = float(np.sum(h1) / area)
    initial_half_h = 0.5 * h0_line * dx
    final_half_h = 0.5 * h1_line * dx
    displacement_h = final_half_h - initial_half_h
    p0 = phi0.mean(axis=(1, 2))
    p1 = phi1.mean(axis=(1, 2))
    initial_half_cross = half_width_from_crossings(p0, dx)
    final_half_cross = half_width_from_crossings(p1, dx)
    displacement_cross = final_half_cross - initial_half_cross
    q0 = C0 - h0
    q1 = C1 - h1
    delta_h = float(np.sum(h1 - h0))
    stefan = delta_h + float(np.sum(q1 - q0))
    mass_error = abs(float(np.sum(C1 - C0))) / max(abs(float(np.sum(C0))), 1.0)

    sharp = read_rows(case / "sharp_reference.csv")
    sharp_t = np.array([float(row["time_s"]) for row in sharp])
    sharp_d = np.array([float(row["displacement_nm"]) for row in sharp])
    time_scale_s = float(manifest["elapsed_s"]) / float(manifest["final_time_code"])
    trajectory: list[dict[str, object]] = []
    trajectory_sources: list[tuple[int, float, float]] = [(0, 0.0, initial_half_h)]
    for checkpoint_step, checkpoint_path in checkpoint_series(case, "phi"):
        checkpoint_phi = np.fromfile(
            checkpoint_path, dtype=np.float64
        ).reshape(shape)
        checkpoint_h = h_switch(checkpoint_phi)
        half_h = 0.5 * float(np.sum(checkpoint_h) / area) * dx
        metadata_path = checkpoint_path.with_name(
            checkpoint_path.name.replace("_phi.raw", "_meta.json")
        )
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        trajectory_sources.append((
            checkpoint_step, float(metadata["time_code"]), half_h
        ))
    for checkpoint_step, time_code, half_h in trajectory_sources:
        time_s = time_code * time_scale_s
        displacement = half_h - initial_half_h
        reference = float(np.interp(time_s, sharp_t, sharp_d))
        error = abs(displacement - reference)
        allowed = max(0.5 * dx, 0.15 * abs(reference))
        trajectory.append({
            "case_id": manifest["case_id"],
            "step": checkpoint_step,
            "time_code": time_code,
            "time_s": time_s,
            "PF_half_width_h_nm": half_h,
            "PF_displacement_h_nm": displacement,
            "sharp_displacement_nm": reference,
            "absolute_trajectory_error_nm": error,
            "allowed_error_nm": allowed,
            "trajectory_gate_pass": error <= allowed,
        })
    actual_elapsed_s = float(trajectory[-1]["time_s"])
    sharp_final = interpolate_sharp_state(sharp, actual_elapsed_s)
    sharp_disp = float(sharp_final["displacement_nm"])

    accepts_text = (case / "run/run.log").read_text(errors="replace")
    accepts = re.findall(r"CTOT_MIMETIC_BE_ACCEPT[^\n]+", accepts_text)
    retries = accepts_text.count("CTOT_COUPLED_STEP_RETRY")
    rejects = accepts_text.count("CTOT_MIMETIC_BE_REJECT")
    runtime_mass = [abs(float(v)) for v in re.findall(
        r"CTOT_MIMETIC_BE_ACCEPT[^\n]*mass_error=([^ ]+)", accepts_text
    )]
    no_clip_projection = all(
        "clip_count=0" in row and "projection_mass=0" in row for row in accepts
    )
    predicate = read_rows(unique_result_file(case, "ctot_acceptance_predicate.csv"))
    predicate_pass = len(predicate) == int(manifest["nsteps"]) and all(
        row.get("accepted") == "1" and row.get("physical_projection_zero") == "1"
        and row.get("no_mass_loss_clipping") == "1" for row in predicate
    )
    energy = read_rows(unique_result_file(case, "ctot_energy_work.csv"))
    energy_pass = len(energy) == int(manifest["nsteps"]) and all(
        row.get("accepted") == "1" and row.get("monotone_pass") == "1"
        and row.get("balance_pass") == "1" for row in energy
    )
    outer = read_rows(unique_result_file(case, "ctot_outer_iterations.csv"))
    converged = [row for row in outer if row.get("status") == "CONVERGED"]
    kkt_max = max((abs(float(row["phase_KKT_residual"])) for row in converged),
                  default=math.nan)
    storage_max = max((abs(float(row["local_phase_storage_residual"]))
                       for row in converged), default=math.nan)

    alpha1 = 1.0 - h1
    matrix_weight = float(np.sum(alpha1))
    matrix_mean = float(np.sum(alpha1 * x1) / matrix_weight)
    sharp_matrix_mean = float(sharp_final["matrix_mean_xB"])
    x_eq = unit.xAg2Te_eq_from_T(float(manifest["temperature_C"]) + 273.15)
    profile_rows: list[dict[str, object]] = []
    p_x0 = x0.mean(axis=(1, 2))
    p_x1 = x1.mean(axis=(1, 2))
    p_C0 = C0.mean(axis=(1, 2))
    p_C1 = C1.mean(axis=(1, 2))
    for i in range(nx):
        profile_rows.append({
            "case_id": manifest["case_id"],
            "coordinate_nm": (i + 0.5) * dx,
            "phi_initial": p0[i],
            "phi_final": p1[i],
            "xB_alpha_initial": p_x0[i],
            "xB_alpha_final": p_x1[i],
            "Ctot_initial": p_C0[i],
            "Ctot_final": p_C1[i],
            "h_initial": float(h_switch(np.array([p0[i]]))[0]),
            "h_final": float(h_switch(np.array([p1[i]]))[0]),
        })

    transfer_error = abs(displacement_h - sharp_disp) / max(abs(sharp_disp), 1.0e-300)
    phase_fraction_error = abs(final_half_h - float(sharp_final["radius_nm"])) / max(
        abs(float(sharp_final["radius_nm"])), 1.0e-300
    )
    final_allowed = max(0.5 * dx, 0.15 * abs(sharp_disp))
    direction_pass = displacement_h > 0.0 if manifest["direction"] == "growth" else displacement_h < 0.0
    displacement_pass = abs(displacement_h) >= 5.0 * dx
    matrix_reference_error = abs(matrix_mean - sharp_matrix_mean)
    equilibrium_residual = abs(matrix_mean - x_eq)
    numerical_pass = (
        step_phi == int(manifest["nsteps"])
        and len(accepts) == int(manifest["nsteps"])
        and retries == 0 and rejects == 0
        and predicate_pass and energy_pass and no_clip_projection
        and mass_error <= 1.0e-10
        and max(runtime_mass, default=math.inf) <= 1.0e-10
        and abs(stefan) / max(abs(delta_h), 1.0) <= 1.0e-10
        and storage_max <= 1.0e-12
        and float(np.min(phi1)) >= -1.0e-12
        and float(np.max(phi1)) <= 1.0 + 1.0e-12
        and float(np.min(x1)) >= 0.0 and float(np.max(x1)) <= 1.0
    )
    research_gate = (
        numerical_pass and direction_pass and displacement_pass
        and abs(displacement_h - sharp_disp) <= final_allowed
        and transfer_error <= 0.15 and phase_fraction_error <= 0.05
        and matrix_reference_error <= 5.0e-4
    )
    metric = {
        "case_id": manifest["case_id"],
        "direction": manifest["direction"],
        "grid": f"{nx}x{ny}x{nz}",
        "dx_nm": dx,
        "lambda_nm": manifest["lambda_nm"],
        "dt_code": manifest["dt_code"],
        "nsteps": manifest["nsteps"],
        "requested_elapsed_s": manifest["elapsed_s"],
        "actual_elapsed_s": actual_elapsed_s,
        "actual_to_requested_time_ratio": (
            actual_elapsed_s / float(manifest["elapsed_s"])
        ),
        "accepted_steps": len(accepts),
        "retry_count": retries,
        "reject_count": rejects,
        "initial_half_width_h_nm": initial_half_h,
        "final_half_width_h_nm": final_half_h,
        "PF_h_displacement_nm": displacement_h,
        "PF_crossing_displacement_nm": displacement_cross,
        "sharp_displacement_nm": sharp_disp,
        "trajectory_error_abs_nm": abs(displacement_h - sharp_disp),
        "trajectory_error_rel": transfer_error,
        "trajectory_allowed_nm": final_allowed,
        "cumulative_transfer_error_rel": transfer_error,
        "final_phase_fraction_error_rel": phase_fraction_error,
        "PF_final_matrix_mean_xB": matrix_mean,
        "sharp_final_matrix_mean_xB": sharp_matrix_mean,
        "matrix_reference_error_abs": matrix_reference_error,
        "thermodynamic_xB_eq": x_eq,
        "equilibrium_composition_residual": equilibrium_residual,
        "growth_law_exponent": growth_law_exponent(
            np.array([float(r["time_s"]) for r in trajectory]),
            np.array([float(r["PF_displacement_h_nm"]) for r in trajectory]),
        ),
        "mass_error_rel_recomputed": mass_error,
        "runtime_mass_error_max": max(runtime_mass, default=math.nan),
        "stefan_storage_residual_rel": abs(stefan) / max(abs(delta_h), 1.0),
        "phase_KKT_max": kkt_max,
        "phase_storage_residual_max": storage_max,
        "phi_min": float(np.min(phi1)),
        "phi_max": float(np.max(phi1)),
        "xB_min": float(np.min(x1)),
        "xB_max": float(np.max(x1)),
        "direction_pass": direction_pass,
        "displacement_ge_5dx": displacement_pass,
        "acceptance_predicate_pass": predicate_pass,
        "energy_work_pass": energy_pass,
        "no_clip_or_projection": no_clip_projection,
        "numerical_hard_gates_pass": numerical_pass,
        "long_time_research_gate_pass": research_gate,
    }
    return metric, trajectory, profile_rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    metrics: list[dict[str, object]] = []
    trajectory: list[dict[str, object]] = []
    profiles: list[dict[str, object]] = []
    for case in sorted((args.matrix_root.resolve() / "cases").iterdir()):
        if (case / "runtime_manifest.json").is_file() and (case / "run/run.log").is_file():
            metric, case_trajectory, case_profiles = analyze_case(case)
            metrics.append(metric)
            trajectory.extend(case_trajectory)
            profiles.extend(case_profiles)
    if not metrics:
        raise RuntimeError("no completed JC4 long-time cases")
    report_root = args.report_root.resolve()
    write_rows(report_root / "jc4_long_time_planar_metrics.csv", metrics)
    write_rows(report_root / "jc4_long_time_trajectory.csv", trajectory)
    write_rows(report_root / "jc4_long_time_profiles.csv", profiles)
    table = "\n".join(
        f"| {row['case_id']} | {float(row['PF_h_displacement_nm']):.6g} | "
        f"{float(row['sharp_displacement_nm']):.6g} | "
        f"{float(row['trajectory_error_rel']):.3%} | "
        f"{row['numerical_hard_gates_pass']} | {row['long_time_research_gate_pass']} |"
        for row in metrics
    )
    all_pass = all(bool(row["long_time_research_gate_pass"]) for row in metrics)
    (report_root / "jc4_long_time_planar_validation.md").write_text(f"""# JC4 long-time planar validation

The CUDA fixed-Ctot candidate is compared at equal physical time and identical
finite-box inventory with the independent one-sided conservative Stefan
ALE--FV reference. The direct-logit Ji--Chen oracle remains the independent
equation/limiting-case check; it is not relabeled as a long-time integrator
after its pure-phase coordinate degeneracy was observed.

| case | PF displacement (nm) | sharp displacement (nm) | trajectory error | numerical gates | research gate |
|---|---:|---:|---:|---|---|
{table}

Overall long-time planar status: **{'PASS' if all_pass else 'FAIL'}**.
""", encoding="utf-8")
    print(f"jc4_long_time_cases_analyzed={len(metrics)}")
    print(f"jc4_long_time_planar_status={'PASS' if all_pass else 'FAIL'}")
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
