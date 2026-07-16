#!/usr/bin/env python3
"""Kinetic-only traveling-wave oracle for the production beta phase operator."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def h_prime(phi: np.ndarray) -> np.ndarray:
    return 30.0 * phi**2 * (1.0 - phi) ** 2


def g_prime(phi: np.ndarray) -> np.ndarray:
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def crossings(phi: np.ndarray, dx: float) -> list[float]:
    found: list[float] = []
    for i, left in enumerate(phi):
        right = phi[(i + 1) % phi.size]
        if (left - 0.5) * (right - 0.5) < 0.0:
            found.append((i + (0.5 - left) / (right - left)) * dx)
    if len(found) != 2:
        raise RuntimeError(f"expected two phi-half crossings, got {found}")
    return found


def run_case(W: float, kappa: float, Lphi: float, interface_resolution: int,
             driving: float, dt: float, final_time: float) -> dict[str, float]:
    interface_width = math.sqrt(8.0 * kappa / W)
    dx = interface_width / interface_resolution
    n = int(math.ceil(32.0 * interface_width / dx))
    if n % 2:
        n += 1
    length = n * dx
    x = np.arange(n) * dx
    left, right = 0.35 * length, 0.65 * length
    half_width = 0.5 * interface_width
    phi = 0.5 * (np.tanh((x - left) / half_width) -
                 np.tanh((x - right) / half_width))
    initial_crossings = crossings(phi, dx)
    wave = 2.0 * math.pi * np.fft.fftfreq(n, d=dx)
    steps = int(round(final_time / dt))
    dt = final_time / steps
    denominator = 1.0 + dt * Lphi * kappa * wave**2
    min_phi = float(phi.min())
    max_phi = float(phi.max())
    for _ in range(steps):
        explicit = W * g_prime(phi) + driving * h_prime(phi)
        phi = np.fft.ifft(
            (np.fft.fft(phi) - dt * Lphi * np.fft.fft(explicit)) /
            denominator).real
        min_phi = min(min_phi, float(phi.min()))
        max_phi = max(max_phi, float(phi.max()))
        if not np.isfinite(phi).all():
            raise RuntimeError("nonfinite kinetic-only state")
    final_crossings = crossings(phi, dx)
    half_thickness_change = 0.5 * (
        (final_crossings[1] - final_crossings[0]) -
        (initial_crossings[1] - initial_crossings[0]))
    velocity = half_thickness_change / final_time
    s = math.sqrt(W / (2.0 * kappa))
    phi_grad_integral = s / 3.0
    predicted_velocity = -Lphi * driving / phi_grad_integral
    return {
        "interface_resolution": interface_resolution,
        "dx_code": dx,
        "N": n,
        "L_phi_code": Lphi,
        "driving": driving,
        "dt_code": dt,
        "steps": steps,
        "final_time_code": final_time,
        "phi_grad_integral_analytic": phi_grad_integral,
        "phi_half_velocity": velocity,
        "predicted_velocity": predicted_velocity,
        "velocity_error_rel": abs(velocity - predicted_velocity) /
                              max(abs(predicted_velocity), 1.0e-300),
        "phi_min_seen": min_phi,
        "phi_max_seen": max_phi,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--W", type=float, default=1.0)
    parser.add_argument("--kappa", type=float, default=0.045)
    parser.add_argument("--L-phi", type=float, default=5.344088465432383)
    args = parser.parse_args()
    rows: list[dict[str, float]] = []
    for resolution in (4, 6, 8):
        for factor in (0.5, 1.0, 2.0):
            for driving in (-0.004, -0.002, 0.002, 0.004):
                rows.append(run_case(
                    args.W, args.kappa, args.L_phi * factor,
                    resolution, driving, 1.0e-3, 5.0))
    for dt in (2.0e-3, 1.0e-3, 5.0e-4, 2.5e-4):
        rows.append(run_case(
            args.W, args.kappa, args.L_phi, 6, 0.002, dt, 5.0))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    fields = list(rows[0])
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"kinetic_only_cases={len(rows)}")
    print(f"kinetic_only_max_velocity_error_rel="
          f"{max(row['velocity_error_rel'] for row in rows):.17e}")
    print(f"kinetic_only_nonfinite=false")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
