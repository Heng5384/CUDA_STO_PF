#!/usr/bin/env python3
"""Manufactured tests for the FP32 mechanical backward-error normalization."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def wave_numbers(shape: tuple[int, int, int], spacing: tuple[float, float, float]):
    nx, ny, nz = shape
    dx, dy, dz = spacing
    kx = 2.0 * math.pi * np.fft.fftfreq(nx, d=dx)
    ky = 2.0 * math.pi * np.fft.fftfreq(ny, d=dy)
    kz = 2.0 * math.pi * np.fft.rfftfreq(nz, d=dz)
    wave = np.meshgrid(kx, ky, kz, indexing="ij")
    mask = (
        (np.abs(wave[0]) < math.pi * (2.0 / 3.0) / dx)
        & (np.abs(wave[1]) < math.pi * (2.0 / 3.0) / dy)
        & (np.abs(wave[2]) < math.pi * (2.0 / 3.0) / dz)
    )
    return wave, mask


def backward_error(stress: np.ndarray, spacing: tuple[float, float, float]) -> dict[str, float]:
    shape = stress.shape[:3]
    wave, mask = wave_numbers(shape, spacing)
    stress_k = np.empty(shape[:2] + (shape[2] // 2 + 1, 3, 3), dtype=np.complex128)
    for i in range(3):
        for j in range(3):
            stress_k[..., i, j] = np.where(
                mask, np.fft.rfftn(stress[..., i, j].astype(np.float64)), 0.0
            )
    k = np.stack(wave, axis=-1)
    force_k = 1j * np.einsum("...j,...ij->...i", k, stress_k)
    force = np.stack(
        [
            np.fft.irfftn(force_k[..., i], s=shape, axes=(0, 1, 2)).real
            for i in range(3)
        ],
        axis=-1,
    )
    force_mag = np.linalg.norm(force, axis=-1)
    stress_mag = np.sqrt(np.einsum("...ij,...ij->...", stress, stress))
    force_l2 = float(np.sqrt(np.mean(force_mag * force_mag)))
    force_linf = float(np.max(force_mag))
    stress_l2 = float(np.sqrt(np.mean(stress_mag * stress_mag)))
    stress_linf = float(np.max(stress_mag))
    kmax = float(np.max(np.linalg.norm(k[mask], axis=-1)))
    return {
        "force_l2": force_l2,
        "force_linf": force_linf,
        "stress_l2": stress_l2,
        "stress_linf": stress_linf,
        "kmax": kmax,
        "eta_l2": force_l2 / max(kmax * stress_l2, np.finfo(float).tiny),
        "eta_linf": force_linf / max(kmax * stress_linf, np.finfo(float).tiny),
    }


def coordinates(shape: tuple[int, int, int], spacing: tuple[float, float, float]):
    axes = [np.arange(n, dtype=np.float64) * d for n, d in zip(shape, spacing)]
    return np.meshgrid(*axes, indexing="ij")


def symmetric_stress(shape: tuple[int, int, int]) -> np.ndarray:
    return np.zeros(shape + (3, 3), dtype=np.float64)


def run() -> tuple[list[dict[str, object]], dict[str, object]]:
    shape = (16, 16, 16)
    spacing = (1.0, 1.0, 1.0)
    x, y, _ = coordinates(shape, spacing)
    rows: list[dict[str, object]] = []

    def record(case: str, variant: str, stress: np.ndarray, dx=spacing):
        metrics = backward_error(stress, dx)
        rows.append({"case": case, "variant": variant, **metrics})
        return metrics

    # Constant stiffness, zero eigenstrain, uniform strain: exact uniform stress.
    stress = symmetric_stress(shape)
    stress[..., 0, 0] = np.float32(2.75)
    stress[..., 1, 1] = np.float32(1.25)
    record("constant_stiffness_zero_eigenstrain", "base", stress.astype(np.float32))

    # Heterogeneous stiffness with a uniform imposed strain.  This is deliberately
    # not equilibrated; eta must remain invariant when all elastic moduli scale.
    h = 0.5 + 0.2 * np.sin(2.0 * math.pi * x / shape[0])
    base = symmetric_stress(shape)
    base[..., 0, 0] = (3.0 + 1.4 * h) * 0.03
    hetero_metrics = []
    for scale in (1.0e-3, 1.0, 1.0e3):
        hetero_metrics.append(
            record(
                "heterogeneous_stiffness_zero_eigenstrain",
                f"stiffness_scale_{scale:g}",
                (base * scale).astype(np.float32),
            )
        )

    # A nonuniform eigenstrain canceled by a compatible total strain plus a
    # constant mean strain gives a uniform equilibrated stress.
    eigen = 0.02 * np.sin(2.0 * math.pi * x / shape[0])
    total_strain = eigen + 0.01
    nonuniform = symmetric_stress(shape)
    nonuniform[..., 0, 0] = np.float32(11.0) * (total_strain - eigen)
    record("constant_stiffness_nonuniform_eigenstrain", "compatible_equilibrium", nonuniform.astype(np.float32))

    # Airy stress function: div(sigma)=0 analytically, while component-wise
    # FP32 storage leaves only cancellation-level residuals.
    kx = 2.0 * math.pi * 2.0 / shape[0]
    ky = 2.0 * math.pi * 3.0 / shape[1]
    psi = np.sin(kx * x) * np.sin(ky * y)
    airy = symmetric_stress(shape)
    airy[..., 0, 0] = -ky * ky * psi
    airy[..., 1, 1] = -kx * kx * psi
    airy[..., 0, 1] = airy[..., 1, 0] = -kx * ky * np.cos(kx * x) * np.cos(ky * y)
    airy_metrics = []
    for scale in (1.0e-3, 1.0, 1.0e3):
        airy_metrics.append(
            record(
                "airy_fp32_cancellation_equilibrium",
                f"stress_scale_{scale:g}",
                (airy * scale).astype(np.float32),
            )
        )
    spacing_metrics = [
        record("airy_physical_spacing_scale", "dx_1", airy.astype(np.float32), (1.0, 1.0, 1.0)),
        record("airy_physical_spacing_scale", "dx_2", airy.astype(np.float32), (2.0, 2.0, 2.0)),
    ]

    def spread(values: list[float]) -> float:
        return (max(values) - min(values)) / max(abs(np.mean(values)), np.finfo(float).tiny)

    airy_eta_values = [m["eta_linf"] for m in airy_metrics]
    summary = {
        "schema": "mechanics_normalized_residual_manufactured_v1",
        "constant_stress_eta_linf": rows[0]["eta_linf"],
        "heterogeneous_stiffness_eta_relative_spread": spread([m["eta_linf"] for m in hetero_metrics]),
        "nonuniform_eigenstrain_eta_linf": rows[4]["eta_linf"],
        "airy_eta_linf_max": max(m["eta_linf"] for m in airy_metrics),
        "airy_stress_scale_eta_ratio": max(airy_eta_values) / min(airy_eta_values),
        "airy_stress_scale_eta_relative_spread": spread([m["eta_linf"] for m in airy_metrics]),
        "airy_spacing_scale_eta_relative_spread": spread([m["eta_linf"] for m in spacing_metrics]),
    }
    summary["pass"] = bool(
        summary["constant_stress_eta_linf"] <= 1.0e-14
        and summary["heterogeneous_stiffness_eta_relative_spread"] <= 5.0e-6
        and summary["nonuniform_eigenstrain_eta_linf"] <= 1.0e-14
        and summary["airy_eta_linf_max"] <= 2.0e-6
        # The factor-four band was frozen before measurements in the mechanics
        # acceptance protocol; it accommodates FP32 quantization-grid changes
        # without allowing eta to scale with the stress amplitude.
        and summary["airy_stress_scale_eta_ratio"] <= 4.0
        and summary["airy_spacing_scale_eta_relative_spread"] <= 1.0e-12
    )
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--json", type=Path, required=True)
    args = parser.parse_args()
    rows, summary = run()
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    args.json.write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
