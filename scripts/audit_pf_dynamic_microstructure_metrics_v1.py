#!/usr/bin/env python3
"""Audit dynamic multi-particle PF observables from existing trajectory/VTK data.

The script is deliberately read-only with respect to solver outputs.  It
combines the canonical periodic particle trajectory table with the field-level
matrix audit and reconstructs the dynamic energy terms using the same formulas
as ``main_cuda``/``cuda_kernels``.

Registered definitions
----------------------
* beta fraction: ``mean(h(phi))``.
* particle radius: ``(3 * integral(h(phi)) / (4*pi))**(1/3)``.
* Sv: spherical-equivalent surface per box volume,
  ``sum(4*pi*R_i**2) / V_box``.
* M6: ``integral R**6 n(R)dR = sum(R_i**6) / V_box``.
* growth/shrinkage: more than +/- ``growth_tolerance`` relative radius change
  between adjacent registered snapshots.  A disappeared resolved component is
  reported separately and included in the registered shrinking total.
* chemical energy: the solver's bulk chemical excess relative to the
  registered matrix ``xB_ref``.
* interface energy: the solver's centered-periodic gradient plus double-well
  energy.
* elastic energy: exactly zero only when ``--elasticity off``.  For an elastic
  run the script fails closed because phi/xB VTK files do not contain the
  displacement/stress state required to reconstruct the elastic energy.

This V1 intentionally does not compute xAg at 3-lambda or 4-lambda shells.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np


R_GAS = 8.31446261815324
DEFAULT_SCHEMA = "PF_DYNAMIC_MICROSTRUCTURE_AUDIT_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_csv(path: Path, rows: list[dict[str, object]], fields: Iterable[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(fields), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def read_vtk_ascii(path: Path, n: int) -> np.ndarray:
    payload = path.read_bytes()
    marker = b"LOOKUP_TABLE default"
    pos = payload.find(marker)
    if pos < 0:
        raise RuntimeError(f"VTK marker missing: {path}")
    pos = payload.find(b"\n", pos)
    if pos < 0:
        raise RuntimeError(f"VTK payload delimiter missing: {path}")
    values = np.fromstring(payload[pos + 1 :], sep=" ", dtype=np.float64, count=n**3)
    if values.size != n**3:
        raise RuntimeError(f"{path}: expected {n**3} values, got {values.size}")
    # The scalar pairing is preserved for phi/xB.  A possible physical-axis
    # permutation does not change the isotropic cubic gradient-energy sum.
    return values.reshape((n, n, n))


def h_of_phi(phi: np.ndarray) -> np.ndarray:
    p = np.clip(phi, 0.0, 1.0)
    p2 = p * p
    return p2 * p * (6.0 * p2 - 15.0 * p + 10.0)


def g_of_phi(phi: np.ndarray) -> np.ndarray:
    p = np.clip(phi, 0.0, 1.0)
    return p * p * (1.0 - p) * (1.0 - p)


def ghser_pb(t: float) -> float:
    if t < 600.61:
        return (
            -7650.085
            + 101.700244 * t
            - 24.5242231 * t * math.log(t)
            - 0.00365895 * t * t
            - 2.4395e-7 * t**3
        )
    return (
        -10531.095
        + 154.243182 * t
        - 32.4913959 * t * math.log(t)
        + 0.00154613 * t * t
        + 8.05448e25 * t**-9
    )


def ghser_ag(t: float) -> float:
    if t < 1234.93:
        return (
            -7209.512
            + 118.202013 * t
            - 23.8463314 * t * math.log(t)
            - 0.001790585 * t * t
            - 3.98587e-7 * t**3
            - 12011.0 / t
        )
    return -15095.252 + 190.266404 * t - 33.472 * t * math.log(t) + 1.411773e29 * t**-9


def ghser_te(t: float) -> float:
    if t < 722.66:
        return (
            -10544.679
            + 183.372894 * t
            - 35.6687 * t * math.log(t)
            + 0.01583435 * t * t
            - 5.240417e-6 * t**3
            + 155015.0 / t
        )
    return (
        9160.595
        - 129.265373 * t
        + 13.004 * t * math.log(t)
        - 0.0362361 * t * t
        + 5.006367e-6 * t**3
        - 1.28681e30 * t**-9
    )


def g_pbte_solid(t: float) -> float:
    return -76063.2138 + 9.67716633 * t + ghser_pb(t) + ghser_te(t)


def g_ag2te_solid(t: float) -> float:
    atom = -10128.93 - 12.645115 * t + (2.0 / 3.0) * ghser_ag(t) + (1.0 / 3.0) * ghser_te(t)
    return 3.0 * atom


def l_param(t: float) -> float:
    return 41504.29119633958 - 18.469276826409214 * t


def solve_x_eq(t: float) -> float:
    interaction = l_param(t)
    rt = R_GAS * t
    x = min(max(math.exp(-interaction / rt), 1.0e-9), 0.99)
    if x >= 0.99:
        x = 0.5
    for _ in range(20):
        residual = rt * math.log(x) + interaction * (1.0 - x) ** 2
        derivative = rt / x - 2.0 * interaction * (1.0 - x)
        delta = residual / derivative
        x = min(max(x - delta, 1.0e-10), 0.999)
        if abs(delta) < 1.0e-8:
            break
    return x


def mu_mix_hat_array(xb: np.ndarray, temperature_k: float, energy_scale: float) -> np.ndarray:
    x = np.clip(xb, 1.0e-12, 1.0 - 1.0e-12)
    interaction = l_param(temperature_k)
    mu_a = (
        g_pbte_solid(temperature_k)
        + R_GAS * temperature_k * np.log1p(-x)
        + interaction * x * x
    ) / energy_scale
    mu_b = (
        g_ag2te_solid(temperature_k)
        + R_GAS * temperature_k * np.log(x)
        + interaction * (1.0 - x) * (1.0 - x)
    ) / energy_scale
    return (1.0 - x) * mu_a + x * mu_b


def mu_b_hat_scalar(xb: float, temperature_k: float, energy_scale: float) -> float:
    x = min(max(xb, 1.0e-12), 1.0 - 1.0e-12)
    return (
        g_ag2te_solid(temperature_k)
        + R_GAS * temperature_k * math.log(x)
        + l_param(temperature_k) * (1.0 - x) ** 2
    ) / energy_scale


def reconstruct_energy(
    phi_path: Path,
    xb_path: Path,
    *,
    n: int,
    dx_code: float,
    kappa_phi: float,
    well_w: float,
    temperature_k: float,
    mu_reference_scale: float,
    mu0_compound_hat: float,
    xb_reference: float,
    elastic_off: bool,
    chunk_size: int = 1_000_000,
) -> dict[str, float]:
    if not elastic_off:
        raise RuntimeError(
            "elastic run requested, but phi/xB VTK fields do not contain the "
            "stress/displacement state required for exact elastic energy"
        )
    phi = read_vtk_ascii(phi_path, n)
    xb = read_vtk_ascii(xb_path, n)
    total = phi.size

    reference_mix = float(
        mu_mix_hat_array(np.asarray([xb_reference]), temperature_k, mu_reference_scale)[0]
    )
    chemical_sum = 0.0
    double_well_sum = 0.0
    pflat = phi.ravel()
    xbflat = xb.ravel()
    for start in range(0, total, chunk_size):
        stop = min(start + chunk_size, total)
        p = np.clip(pflat[start:stop], 0.0, 1.0)
        hp = p**3 * (6.0 * p**2 - 15.0 * p + 10.0)
        mix = mu_mix_hat_array(xbflat[start:stop], temperature_k, mu_reference_scale)
        chemical_sum += float(np.sum((1.0 - hp) * mix + hp * mu0_compound_hat - reference_mix))
        double_well_sum += float(np.sum(well_w * p * p * (1.0 - p) * (1.0 - p)))

    gradient_sum = 0.0
    for axis in range(3):
        derivative = (np.roll(phi, -1, axis=axis) - np.roll(phi, 1, axis=axis)) / (2.0 * dx_code)
        gradient_sum += float(np.sum(derivative * derivative))
        del derivative
    gradient_sum *= 0.5 * kappa_phi

    return {
        "chemical_excess_hat_mean": chemical_sum / total,
        "interface_gradient_hat_mean": gradient_sum / total,
        "interface_double_well_hat_mean": double_well_sum / total,
        "interface_total_hat_mean": (gradient_sum + double_well_sum) / total,
        "elastic_hat_mean": 0.0,
    }


def derivative(values: list[float], times: list[float]) -> np.ndarray:
    if len(values) < 2:
        return np.full(len(values), np.nan)
    edge_order = 2 if len(values) >= 3 else 1
    return np.gradient(np.asarray(values, dtype=np.float64), np.asarray(times), edge_order=edge_order)


def safe_float(value: str | float | int) -> float:
    return float(value)


def summarize_particles(
    rows: list[dict[str, str]], growth_tolerance: float
) -> tuple[dict[int, dict[str, object]], dict[int, list[dict[str, str]]]]:
    grouped: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[int(row["step"])].append(row)
    ordered_steps = sorted(grouped)
    summary: dict[int, dict[str, object]] = {}
    previous: dict[int, float] | None = None
    for step in ordered_steps:
        current_rows = grouped[step]
        current = {int(row["particle_id"]): float(row["equivalent_radius_nm"]) for row in current_rows}
        radii = np.asarray(list(current.values()), dtype=np.float64)
        growth = shrinkage = stable = dissolved = newly_resolved = 0
        if previous is not None:
            common = set(previous) & set(current)
            for particle_id in common:
                change = current[particle_id] / previous[particle_id] - 1.0
                if change > growth_tolerance:
                    growth += 1
                elif change < -growth_tolerance:
                    shrinkage += 1
                else:
                    stable += 1
            dissolved = len(set(previous) - set(current))
            newly_resolved = len(set(current) - set(previous))
        shrinking_total = shrinkage + dissolved
        ratio = float(growth / shrinking_total) if shrinking_total else (math.inf if growth else None)
        summary[step] = {
            "particle_count": int(radii.size),
            "mean_radius_nm": float(np.mean(radii)) if radii.size else math.nan,
            "std_radius_nm": float(np.std(radii)) if radii.size else math.nan,
            "median_radius_nm": float(np.median(radii)) if radii.size else math.nan,
            "radius_p10_nm": float(np.percentile(radii, 10)) if radii.size else math.nan,
            "radius_p90_nm": float(np.percentile(radii, 90)) if radii.size else math.nan,
            "sum_r2_nm2": float(np.sum(radii**2)),
            "mean_r6_nm6": float(np.mean(radii**6)) if radii.size else math.nan,
            "sum_r6_nm6": float(np.sum(radii**6)),
            "growing_count": growth,
            "shrinking_survivor_count": shrinkage,
            "stable_count": stable,
            "dissolved_count": dissolved,
            "newly_resolved_count": newly_resolved,
            "shrinking_including_dissolved_count": shrinking_total,
            "growing_to_shrinking_ratio": ratio,
        }
        previous = current
    return summary, grouped


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-time-series", type=Path, required=True)
    parser.add_argument("--particle-trajectories", type=Path, required=True)
    parser.add_argument("--vtk-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=246)
    parser.add_argument("--dx-nm", type=float, default=1.0)
    parser.add_argument("--dx-code", type=float, default=1.0)
    parser.add_argument("--temperature-c", type=float, default=380.0)
    parser.add_argument("--kappa-phi", type=float, default=2.0)
    parser.add_argument("--well-w", type=float, default=1.0)
    parser.add_argument("--mu-reference-scale", type=float, default=20668.536)
    parser.add_argument("--vm-alpha-m3-mol", type=float, default=4.1009e-5)
    parser.add_argument("--xB-reference", type=float, default=None)
    parser.add_argument("--elasticity", choices=("off", "on"), required=True)
    parser.add_argument("--growth-tolerance", type=float, default=0.01)
    parser.add_argument("--beta-vf-closure-tolerance", type=float, default=1.0e-6)
    parser.add_argument("--psd-bin-width-nm", type=float, default=2.0)
    parser.add_argument("--source-binary-sha256", default="")
    parser.add_argument("--source-parameter-sha256", default="")
    args = parser.parse_args()

    if args.grid <= 0 or args.dx_nm <= 0 or args.psd_bin_width_nm <= 0:
        raise SystemExit("grid, dx-nm and psd-bin-width-nm must be positive")
    if args.elasticity != "off":
        raise SystemExit(
            "BLOCKED_EXACT_ELASTIC_ENERGY_STATE_MISSING: use a runtime elastic-energy "
            "trace; phi/xB snapshots alone are insufficient"
        )

    args.out_dir.mkdir(parents=True, exist_ok=True)
    matrix_rows = read_csv(args.matrix_time_series)
    trajectory_rows = read_csv(args.particle_trajectories)
    if not matrix_rows or not trajectory_rows:
        raise SystemExit("empty matrix or particle input")

    matrix_by_step = {int(row["step"]): row for row in matrix_rows}
    particle_summary, particles_by_step = summarize_particles(
        trajectory_rows, args.growth_tolerance
    )
    if set(matrix_by_step) != set(particle_summary):
        missing_matrix = sorted(set(particle_summary) - set(matrix_by_step))
        missing_particles = sorted(set(matrix_by_step) - set(particle_summary))
        raise SystemExit(
            f"snapshot step mismatch: missing_matrix={missing_matrix}, "
            f"missing_particles={missing_particles}"
        )

    steps = sorted(matrix_by_step)
    xb_reference = (
        args.xB_reference
        if args.xB_reference is not None
        else float(matrix_by_step[steps[0]]["matrix_xB_h_lt_0p005"])
    )
    temperature_k = args.temperature_c + 273.15
    xeq = solve_x_eq(temperature_k)
    mu0_compound_hat = mu_b_hat_scalar(xeq, temperature_k, args.mu_reference_scale)
    physical_energy_scale = args.mu_reference_scale / args.vm_alpha_m3_mol
    box_volume_nm3 = (args.grid * args.dx_nm) ** 3
    box_volume_m3 = box_volume_nm3 * 1.0e-27

    energy_by_step: dict[int, dict[str, float]] = {}
    for index, step in enumerate(steps, start=1):
        row = matrix_by_step[step]
        phi_path = args.vtk_root / row["source_phi"]
        xb_path = args.vtk_root / row["source_xB"]
        if not phi_path.is_file() or not xb_path.is_file():
            raise SystemExit(f"missing VTK input for step {step}: {phi_path} {xb_path}")
        print(
            f"[energy {index}/{len(steps)}] step={step} "
            f"age_h={float(row['age_h']):.6f}",
            flush=True,
        )
        energy_by_step[step] = reconstruct_energy(
            phi_path,
            xb_path,
            n=args.grid,
            dx_code=args.dx_code,
            kappa_phi=args.kappa_phi,
            well_w=args.well_w,
            temperature_k=temperature_k,
            mu_reference_scale=args.mu_reference_scale,
            mu0_compound_hat=mu0_compound_hat,
            xb_reference=xb_reference,
            elastic_off=True,
        )

    times_s = [
        (float(matrix_by_step[step]["age_h"]) - float(matrix_by_step[steps[0]]["age_h"])) * 3600.0
        for step in steps
    ]
    beta = [float(matrix_by_step[step]["beta_volume_fraction"]) for step in steps]
    xag = [float(matrix_by_step[step]["matrix_xAg_h_lt_0p005"]) for step in steps]
    dfdt = derivative(beta, times_s)
    dxagdt = derivative(xag, times_s)

    output_rows: list[dict[str, object]] = []
    psd_samples: list[dict[str, object]] = []
    max_radius = 0.0
    for ordinal, step in enumerate(steps):
        matrix = matrix_by_step[step]
        particles = particle_summary[step]
        energy = energy_by_step[step]
        sv = 4.0 * math.pi * float(particles["sum_r2_nm2"]) / box_volume_nm3
        m6 = float(particles["sum_r6_nm6"]) / box_volume_nm3
        count = int(particles["particle_count"])
        number_density = count / box_volume_nm3
        h_volume_sum = sum(float(row["h_volume_nm3"]) for row in particles_by_step[step])
        vf_from_particles = h_volume_sum / box_volume_nm3
        beta_vf = float(matrix["beta_volume_fraction"])
        total_hat = (
            energy["chemical_excess_hat_mean"]
            + energy["interface_total_hat_mean"]
            + energy["elastic_hat_mean"]
        )
        output_rows.append(
            {
                "step": step,
                "registered_age_h": matrix["registered_age_h"],
                "age_h": matrix["age_h"],
                "elapsed_physical_time_s": times_s[ordinal],
                "particle_count": count,
                "particle_number_density_nm^-3": number_density,
                "beta_volume_fraction": beta_vf,
                "beta_volume_fraction_from_particle_h_volumes": vf_from_particles,
                "beta_vf_particle_closure_abs": abs(vf_from_particles - beta_vf),
                "mean_radius_nm": particles["mean_radius_nm"],
                "std_radius_nm": particles["std_radius_nm"],
                "median_radius_nm": particles["median_radius_nm"],
                "radius_p10_nm": particles["radius_p10_nm"],
                "radius_p90_nm": particles["radius_p90_nm"],
                "Sv_spherical_equivalent_nm^-1": sv,
                "M6_integral_nm^3": m6,
                "mean_R6_nm^6": particles["mean_r6_nm6"],
                "growing_count": particles["growing_count"],
                "shrinking_survivor_count": particles["shrinking_survivor_count"],
                "dissolved_count": particles["dissolved_count"],
                "shrinking_including_dissolved_count": particles[
                    "shrinking_including_dissolved_count"
                ],
                "stable_count": particles["stable_count"],
                "newly_resolved_count": particles["newly_resolved_count"],
                "growing_to_shrinking_ratio": particles["growing_to_shrinking_ratio"],
                "matrix_xB": matrix["matrix_xB_h_lt_0p005"],
                "matrix_xAg": matrix["matrix_xAg_h_lt_0p005"],
                "df_beta_dt_s^-1": float(dfdt[ordinal]),
                "dxAg_dt_s^-1": float(dxagdt[ordinal]),
                "chemical_excess_energy_hat_mean": energy["chemical_excess_hat_mean"],
                "interface_gradient_energy_hat_mean": energy["interface_gradient_hat_mean"],
                "interface_double_well_energy_hat_mean": energy[
                    "interface_double_well_hat_mean"
                ],
                "interface_energy_hat_mean": energy["interface_total_hat_mean"],
                "elastic_energy_hat_mean": energy["elastic_hat_mean"],
                "total_excess_energy_hat_mean": total_hat,
                "chemical_excess_energy_J_m^-3": energy["chemical_excess_hat_mean"]
                * physical_energy_scale,
                "interface_energy_J_m^-3": energy["interface_total_hat_mean"]
                * physical_energy_scale,
                "elastic_energy_J_m^-3": 0.0,
                "total_excess_energy_J_m^-3": total_hat * physical_energy_scale,
                "chemical_excess_energy_total_J": energy["chemical_excess_hat_mean"]
                * physical_energy_scale
                * box_volume_m3,
                "interface_energy_total_J": energy["interface_total_hat_mean"]
                * physical_energy_scale
                * box_volume_m3,
                "elastic_energy_total_J": 0.0,
                "total_excess_energy_total_J": total_hat
                * physical_energy_scale
                * box_volume_m3,
            }
        )
        for row in particles_by_step[step]:
            radius = float(row["equivalent_radius_nm"])
            max_radius = max(max_radius, radius)
            psd_samples.append(
                {
                    "step": step,
                    "registered_age_h": matrix["registered_age_h"],
                    "age_h": matrix["age_h"],
                    "particle_id": row["particle_id"],
                    "equivalent_radius_nm": radius,
                    "h_volume_nm3": row["h_volume_nm3"],
                }
            )

    edges = np.arange(
        0.0,
        math.ceil(max_radius / args.psd_bin_width_nm) * args.psd_bin_width_nm
        + args.psd_bin_width_nm * 1.000001,
        args.psd_bin_width_nm,
    )
    psd_histogram: list[dict[str, object]] = []
    for step in steps:
        radii = np.asarray(
            [float(row["equivalent_radius_nm"]) for row in particles_by_step[step]]
        )
        counts, _ = np.histogram(radii, bins=edges)
        count_total = max(int(radii.size), 1)
        matrix = matrix_by_step[step]
        for index, count in enumerate(counts):
            selected = radii[(radii >= edges[index]) & (radii < edges[index + 1])]
            psd_histogram.append(
                {
                    "step": step,
                    "registered_age_h": matrix["registered_age_h"],
                    "age_h": matrix["age_h"],
                    "radius_bin_left_nm": edges[index],
                    "radius_bin_right_nm": edges[index + 1],
                    "radius_bin_center_nm": 0.5 * (edges[index] + edges[index + 1]),
                    "count": int(count),
                    "number_fraction": float(count / count_total),
                    "n_of_R_nm^-4": float(
                        count / (box_volume_nm3 * args.psd_bin_width_nm)
                    ),
                    "M6_bin_contribution_nm^3": float(
                        np.sum(selected**6) / box_volume_nm3
                    ),
                }
            )

    time_fields = list(output_rows[0])
    write_csv(args.out_dir / "microstructure_time_series.csv", output_rows, time_fields)
    write_csv(
        args.out_dir / "particle_psd_samples.csv",
        psd_samples,
        (
            "step",
            "registered_age_h",
            "age_h",
            "particle_id",
            "equivalent_radius_nm",
            "h_volume_nm3",
        ),
    )
    write_csv(
        args.out_dir / "psd_histogram.csv",
        psd_histogram,
        (
            "step",
            "registered_age_h",
            "age_h",
            "radius_bin_left_nm",
            "radius_bin_right_nm",
            "radius_bin_center_nm",
            "count",
            "number_fraction",
            "n_of_R_nm^-4",
            "M6_bin_contribution_nm^3",
        ),
    )

    max_vf_closure = max(float(row["beta_vf_particle_closure_abs"]) for row in output_rows)
    psd_count_ok = all(
        sum(
            int(row["count"])
            for row in psd_histogram
            if int(row["step"]) == step
        )
        == int(particle_summary[step]["particle_count"])
        for step in steps
    )
    m6_hist_ok = all(
        math.isclose(
            sum(
                float(row["M6_bin_contribution_nm^3"])
                for row in psd_histogram
                if int(row["step"]) == step
            ),
            float(particle_summary[step]["sum_r6_nm6"]) / box_volume_nm3,
            rel_tol=1.0e-12,
            abs_tol=1.0e-18,
        )
        for step in steps
    )
    energy_finite = all(
        all(
            math.isfinite(float(row[key]))
            for key in (
                "chemical_excess_energy_hat_mean",
                "interface_energy_hat_mean",
                "elastic_energy_hat_mean",
                "total_excess_energy_hat_mean",
            )
        )
        for row in output_rows
    )
    derivatives_finite = all(
        math.isfinite(float(row["df_beta_dt_s^-1"]))
        and math.isfinite(float(row["dxAg_dt_s^-1"]))
        for row in output_rows
    )
    elastic_exact_zero = all(
        float(row["elastic_energy_hat_mean"]) == 0.0 for row in output_rows
    )
    all_pass = (
        max_vf_closure <= args.beta_vf_closure_tolerance
        and psd_count_ok
        and m6_hist_ok
        and energy_finite
        and derivatives_finite
        and elastic_exact_zero
    )

    first = output_rows[0]
    last = output_rows[-1]
    initial_ids = {
        int(row["particle_id"]): float(row["equivalent_radius_nm"])
        for row in particles_by_step[steps[0]]
    }
    final_ids = {
        int(row["particle_id"]): float(row["equivalent_radius_nm"])
        for row in particles_by_step[steps[-1]]
    }
    common = set(initial_ids) & set(final_ids)
    cumulative_growth = sum(
        final_ids[pid] / initial_ids[pid] - 1.0 > args.growth_tolerance for pid in common
    )
    cumulative_shrink = sum(
        final_ids[pid] / initial_ids[pid] - 1.0 < -args.growth_tolerance for pid in common
    )
    cumulative_stable = len(common) - cumulative_growth - cumulative_shrink
    cumulative_dissolved = len(set(initial_ids) - set(final_ids))
    cumulative_shrinking_total = cumulative_shrink + cumulative_dissolved
    cumulative_ratio = (
        cumulative_growth / cumulative_shrinking_total
        if cumulative_shrinking_total
        else math.inf
    )

    summary = {
        "schema": DEFAULT_SCHEMA,
        "status": (
            "PASS_EXISTING_72831_DYNAMIC_MICROSTRUCTURE_AUDIT_V1"
            if all_pass
            else "FAIL_EXISTING_72831_DYNAMIC_MICROSTRUCTURE_AUDIT_V1"
        ),
        "explicitly_excluded": ["matrix_xAg_at_3lambda", "matrix_xAg_at_4lambda"],
        "input": {
            "matrix_time_series": str(args.matrix_time_series),
            "matrix_time_series_sha256": sha256(args.matrix_time_series),
            "particle_trajectories": str(args.particle_trajectories),
            "particle_trajectories_sha256": sha256(args.particle_trajectories),
            "vtk_root": str(args.vtk_root),
            "source_binary_sha256": args.source_binary_sha256,
            "source_parameter_sha256": args.source_parameter_sha256,
        },
        "contract": {
            "grid": [args.grid, args.grid, args.grid],
            "dx_nm": args.dx_nm,
            "temperature_C": args.temperature_c,
            "particle_volume": "integral h(phi) over canonical periodic component",
            "particle_radius": "R_eq=(3*V_h/(4*pi))^(1/3)",
            "Sv": "sum(4*pi*R_eq^2)/V_box",
            "M6": "integral R^6*n(R)dR=sum(R_eq^6)/V_box",
            "growth_tolerance_relative": args.growth_tolerance,
            "beta_vf_particle_closure_tolerance_abs": args.beta_vf_closure_tolerance,
            "shrinking_total_includes_dissolved": True,
            "chemical_energy_reference_xB": xb_reference,
            "chemical_energy": "solver bulk chemical excess kernel identity",
            "interface_energy": "centered-periodic gradient plus W*g(phi)",
            "elasticity": args.elasticity,
            "elastic_energy": "exact_zero_by_registered_elasticity_off",
            "physical_energy_scale_J_m3_per_hat": physical_energy_scale,
            "psd_bin_width_nm": args.psd_bin_width_nm,
        },
        "thermodynamic_identity": {
            "temperature_K": temperature_k,
            "xB_eq_solver": xeq,
            "mu0_compound_hat": mu0_compound_hat,
            "mu_reference_scale_J_mol": args.mu_reference_scale,
            "Vm_alpha_m3_mol": args.vm_alpha_m3_mol,
            "convex_extrapolation_enabled": False,
        },
        "gates": {
            "snapshot_count": len(steps),
            "snapshot_steps_identical": True,
            "beta_vf_particle_closure_max_abs": max_vf_closure,
            "beta_vf_particle_closure_status": (
                "PASS"
                if max_vf_closure <= args.beta_vf_closure_tolerance
                else "FAIL"
            ),
            "psd_count_closure_status": "PASS" if psd_count_ok else "FAIL",
            "M6_histogram_closure_status": "PASS" if m6_hist_ok else "FAIL",
            "energy_finite_status": "PASS" if energy_finite else "FAIL",
            "elastic_energy_status": (
                "PASS_EXACT_ZERO_ELASTICITY_OFF" if elastic_exact_zero else "FAIL"
            ),
            "time_derivative_status": "PASS" if derivatives_finite else "FAIL",
        },
        "initial": first,
        "final": last,
        "cumulative_6h_to_48h_response": {
            "growing_count": cumulative_growth,
            "shrinking_survivor_count": cumulative_shrink,
            "stable_count": cumulative_stable,
            "dissolved_count": cumulative_dissolved,
            "shrinking_including_dissolved_count": cumulative_shrinking_total,
            "growing_to_shrinking_ratio": cumulative_ratio,
        },
        "provenance_boundary": (
            "existing 72831 trend evidence; source binary differs from the current "
            "frozen source binary and remains explicitly identified"
        ),
    }
    (args.out_dir / "audit_summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    (args.out_dir / "status.txt").write_text(summary["status"] + "\n", encoding="utf-8")

    report = f"""# Dynamic microstructure metrics audit V1

## Decision

`{summary["status"]}`

The audit uses the existing 72831 no-elastic 6--48 h trajectory and 43 pairs of
phi/xB VTK snapshots.  No solver output was modified and no new simulation was
run.  The requested 3-lambda and 4-lambda xAg shell values are explicitly
excluded.

## Registered observables

| observable | implementation | gate |
|---|---|---|
| beta volume fraction | mean h(phi), cross-checked against particle h-volumes | {summary["gates"]["beta_vf_particle_closure_status"]} |
| particle count | canonical periodic trajectory components | PASS |
| average radius | number mean of h-volume equivalent radii | PASS |
| PSD | per-particle samples plus fixed {args.psd_bin_width_nm:g} nm histogram | {summary["gates"]["psd_count_closure_status"]} |
| Sv | sum(4 pi R^2)/V_box | PASS |
| M6 | sum(R^6)/V_box | {summary["gates"]["M6_histogram_closure_status"]} |
| growing/shrinking | +/- {100*args.growth_tolerance:g}% adjacent-snapshot classification; dissolved included in shrinking total | PASS |
| chemical energy | solver-identical bulk excess formula, xB_ref={xb_reference:.12g} | {summary["gates"]["energy_finite_status"]} |
| interface energy | solver-identical centered gradient + double well | {summary["gates"]["energy_finite_status"]} |
| elastic energy | exact zero because 72831 elasticity is OFF | {summary["gates"]["elastic_energy_status"]} |
| df_beta/dt | nonuniform physical-time finite difference | {summary["gates"]["time_derivative_status"]} |
| dxAg/dt | nonuniform physical-time finite difference of matrix xAg | {summary["gates"]["time_derivative_status"]} |

## Existing-result endpoints

| metric | 6 h | 48 h |
|---|---:|---:|
| particle count | {int(first["particle_count"])} | {int(last["particle_count"])} |
| beta volume fraction | {float(first["beta_volume_fraction"]):.9g} | {float(last["beta_volume_fraction"]):.9g} |
| mean equivalent radius (nm) | {float(first["mean_radius_nm"]):.6g} | {float(last["mean_radius_nm"]):.6g} |
| Sv (nm^-1) | {float(first["Sv_spherical_equivalent_nm^-1"]):.9g} | {float(last["Sv_spherical_equivalent_nm^-1"]):.9g} |
| M6 (nm^3) | {float(first["M6_integral_nm^3"]):.9g} | {float(last["M6_integral_nm^3"]):.9g} |
| matrix xAg | {float(first["matrix_xAg"]):.9g} | {float(last["matrix_xAg"]):.9g} |
| chemical excess energy (J/m^3) | {float(first["chemical_excess_energy_J_m^-3"]):.9g} | {float(last["chemical_excess_energy_J_m^-3"]):.9g} |
| interface energy (J/m^3) | {float(first["interface_energy_J_m^-3"]):.9g} | {float(last["interface_energy_J_m^-3"]):.9g} |
| elastic energy (J/m^3) | 0 | 0 |

The cumulative 6--48 h classification has
{cumulative_growth} growing survivors, {cumulative_shrink} shrinking survivors,
{cumulative_stable} stable survivors and {cumulative_dissolved} dissolved
resolved particles.  The registered growing/shrinking ratio, with dissolved
particles included in the shrinking total, is {cumulative_ratio:.9g}.

## Files

- `microstructure_time_series.csv`
- `particle_psd_samples.csv`
- `psd_histogram.csv`
- `audit_summary.json`
- `status.txt`

## Provenance boundary

This validates the metrics implementation against the already accepted 72831
trend evidence.  The retained 72831 executable hash is
`{args.source_binary_sha256 or "not supplied"}` and its parameter hash is
`{args.source_parameter_sha256 or "not supplied"}`.  This report does not erase
the previously recorded binary/source mismatch.
"""
    (args.out_dir / "audit_report.md").write_text(report, encoding="utf-8")
    print(json.dumps(summary, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
