#!/usr/bin/env python3
"""Audit frozen FP32 mechanical force spectra without changing the solver."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_case(spec: str) -> tuple[str, Path]:
    if "=" not in spec:
        raise argparse.ArgumentTypeError("case must be LABEL=DUMP_DIR")
    label, path = spec.split("=", 1)
    return label, Path(path)


def wave_numbers(n: int, spacing: float, r2c: bool = False) -> np.ndarray:
    if r2c:
        indices = np.arange(n // 2 + 1, dtype=np.int64)
    else:
        indices = np.arange(n, dtype=np.int64)
        indices = np.where(indices <= n // 2, indices, indices - n)
    return 2.0 * math.pi * indices.astype(np.float64) / (n * spacing)


def analyze(label: str, dump_dir: Path) -> tuple[list[dict[str, object]], dict[str, object]]:
    meta = json.loads((dump_dir / "mechanics_fp32_diagnostics.json").read_text())
    nx, ny, nz = int(meta["Nx"]), int(meta["Ny"]), int(meta["Nz"])
    nzc = int(meta["NzC"])
    dx, dy, dz = float(meta["dx"]), float(meta["dy"]), float(meta["dz"])
    shape = (nx, ny, nzc)
    forces = []
    for component in "xyz":
        path = dump_dir / f"mechanics_force_{component}_k_f32.raw"
        values = np.fromfile(path, dtype=np.complex64)
        if values.size != nx * ny * nzc:
            raise ValueError(f"{path}: expected {nx * ny * nzc} complex values, got {values.size}")
        forces.append(values.reshape(shape).astype(np.complex128))

    stress_names = ("xx", "yy", "zz", "xy", "xz", "yz")
    stress_k = []
    for name in stress_names:
        path = dump_dir / f"mechanics_s{name}_f32.raw"
        values = np.fromfile(path, dtype=np.float32)
        if values.size != nx * ny * nz:
            raise ValueError(f"{path}: expected {nx * ny * nz} real values, got {values.size}")
        stress_k.append(np.fft.rfftn(values.reshape((nx, ny, nz)).astype(np.float64)))

    kx = wave_numbers(nx, dx)[:, None, None]
    ky = wave_numbers(ny, dy)[None, :, None]
    kz = wave_numbers(nz, dz, r2c=True)[None, None, :]
    kmag = np.sqrt(kx * kx + ky * ky + kz * kz)
    cutoff_x = math.pi * (2.0 / 3.0) / dx
    cutoff_y = math.pi * (2.0 / 3.0) / dy
    cutoff_z = math.pi * (2.0 / 3.0) / dz
    retained = (np.abs(kx) < cutoff_x) & (np.abs(ky) < cutoff_y) & (np.abs(kz) < cutoff_z)
    zero = kmag == 0.0
    kmax = float(meta["kmax_dealiased"])
    q = kmag / kmax
    masks = {
        "zero": zero,
        "low": retained & (~zero) & (q <= 0.25),
        "mid": retained & (q > 0.25) & (q <= 0.50),
        "high": retained & (q > 0.50),
        "nyquist_or_dealiased_excluded": ~retained,
    }

    component_power = [np.abs(force) ** 2 for force in forces]
    power = sum(component_power)
    stress_power = (
        np.abs(stress_k[0]) ** 2
        + np.abs(stress_k[1]) ** 2
        + np.abs(stress_k[2]) ** 2
        + 2.0
        * (
            np.abs(stress_k[3]) ** 2
            + np.abs(stress_k[4]) ** 2
            + np.abs(stress_k[5]) ** 2
        )
    )
    operator_scale_power = kmag * kmag * stress_power
    total_power = float(np.sum(power))
    normalization = float(nx * ny * nz)
    solve_index = int(meta.get("mechanics_solve_index", 1))
    rows: list[dict[str, object]] = []
    for bin_name, mask in masks.items():
        bin_power = float(np.sum(power[mask]))
        bin_operator_scale_power = float(np.sum(operator_scale_power[mask]))
        rows.append(
            {
                "case": label,
                "Nx": nx,
                "Ny": ny,
                "Nz": nz,
                "mechanics_solve_index": solve_index,
                "bin": bin_name,
                "mode_count": int(np.count_nonzero(mask)),
                "k_min": float(np.min(kmag[mask])) if np.any(mask) else 0.0,
                "k_max": float(np.max(kmag[mask])) if np.any(mask) else 0.0,
                "spectral_force_l2_over_N": math.sqrt(bin_power) / normalization,
                "spectral_force_linf_over_N": (
                    float(np.max(np.sqrt(power[mask]))) / normalization if np.any(mask) else 0.0
                ),
                "power_fraction": bin_power / total_power if total_power > 0.0 else 0.0,
                "operator_scale_l2_over_N": math.sqrt(bin_operator_scale_power) / normalization,
                "backward_eta_l2_in_bin": (
                    math.sqrt(bin_power / bin_operator_scale_power)
                    if bin_operator_scale_power > 0.0
                    else 0.0
                ),
                "x_power_fraction_within_bin": (
                    float(np.sum(component_power[0][mask])) / bin_power if bin_power > 0.0 else 0.0
                ),
                "y_power_fraction_within_bin": (
                    float(np.sum(component_power[1][mask])) / bin_power if bin_power > 0.0 else 0.0
                ),
                "z_power_fraction_within_bin": (
                    float(np.sum(component_power[2][mask])) / bin_power if bin_power > 0.0 else 0.0
                ),
            }
        )

    retained_power = float(np.sum(power[retained]))
    low_power = float(np.sum(power[masks["low"]]))
    excluded_power = float(np.sum(power[~retained]))
    zero_amp = float(np.sqrt(power[zero][0])) / normalization
    summary = {
        "case": label,
        "grid": [nx, ny, nz],
        "mechanics_solve_index": solve_index,
        "dealias_rule": "abs(k_component) < pi*(2/3)/dx_component",
        "radial_bins": "low: 0<|k|/kmax<=0.25; mid: 0.25<q<=0.5; high: q>0.5",
        "kmax_dealiased": kmax,
        "total_spectral_force_l2_over_N": math.sqrt(total_power) / normalization,
        "zero_mode_amplitude_over_N": zero_amp,
        "low_k_power_fraction_of_retained": low_power / retained_power if retained_power > 0.0 else 0.0,
        "excluded_power_fraction": excluded_power / total_power if total_power > 0.0 else 0.0,
        "runtime_eta_linf": float(meta["eta_Linf"]),
        "runtime_eta_l2": float(meta["eta_L2"]),
    }
    return rows, summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, object]] = []
    summaries = []
    for label, dump_dir in args.case:
        case_rows, case_summary = analyze(label, dump_dir)
        rows.extend(case_rows)
        summaries.append(case_summary)

    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.summary.write_text(json.dumps({"schema": "mechanics_residual_spectrum_v1", "cases": summaries}, indent=2) + "\n")
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
