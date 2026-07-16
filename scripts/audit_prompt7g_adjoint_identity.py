#!/usr/bin/env python3
"""Independent host-double mimetic/legacy-hybrid adjoint identity audit."""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np


def h_switch(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def face_difference(field: np.ndarray, axis: int, spacing: float) -> np.ndarray:
    return (np.roll(field, -1, axis=axis) - field) / spacing


def face_divergence(faces: list[np.ndarray], spacing: tuple[float, ...]) -> np.ndarray:
    result = np.zeros_like(faces[0])
    for axis, (face, delta) in enumerate(zip(faces, spacing)):
        result += (face - np.roll(face, 1, axis=axis)) / delta
    return result


def harmonic_face_mobility(mobility: np.ndarray, axis: int) -> np.ndarray:
    neighbor = np.roll(mobility, -1, axis=axis)
    face = np.zeros_like(mobility)
    active = (mobility > 0.0) & (neighbor > 0.0)
    face[active] = (2.0 * mobility[active] * neighbor[active] /
                    (mobility[active] + neighbor[active]))
    return face


def fft_gradient_dealiased(
    field: np.ndarray, spacing: tuple[float, ...]
) -> list[np.ndarray]:
    spectrum = np.fft.fftn(field)
    wavevectors = [2.0 * math.pi * np.fft.fftfreq(n, d=delta)
                   for n, delta in zip(field.shape, spacing)]
    grids = np.meshgrid(*wavevectors, indexing="ij")
    keep = np.ones(field.shape, dtype=bool)
    for grid, delta in zip(grids, spacing):
        keep &= np.abs(grid) < (2.0 / 3.0) * math.pi / delta
    filtered = np.where(keep, spectrum, 0.0)
    return [np.fft.ifftn(1j * grid * filtered).real for grid in grids]


def current_nonadjoint_face_gradient(
    field: np.ndarray, spacing: tuple[float, ...]
) -> list[np.ndarray]:
    cell_gradient = fft_gradient_dealiased(field, spacing)
    return [0.5 * (component + np.roll(component, -1, axis=axis))
            for axis, component in enumerate(cell_gradient)]


@dataclass
class AuditCase:
    name: str
    mu: np.ndarray
    mobility: np.ndarray
    phi: np.ndarray
    x: np.ndarray
    spacing: tuple[float, ...]
    correction_enabled: bool = False


def make_cases() -> list[AuditCase]:
    rng = np.random.default_rng(5384)
    shape = (8, 6, 10)
    spacing = (0.7, 1.1, 1.4)
    index = np.indices(shape, dtype=float)
    phase = 2.0 * math.pi * (index[0] / shape[0] +
                             2.0 * index[1] / shape[1] +
                             index[2] / shape[2])
    phi0 = np.zeros(shape)
    x0 = np.full(shape, 0.03)
    cases = [
        AuditCase("constant_mu_constant_mobility", np.ones(shape),
                  np.ones(shape), phi0, x0, spacing),
        AuditCase("single_fourier_mode", np.sin(phase), np.ones(shape),
                  phi0, x0, spacing),
        AuditCase("random_mu_constant_mobility", rng.normal(size=shape),
                  np.ones(shape), phi0, x0, spacing),
        AuditCase("random_mu_variable_mobility", rng.normal(size=shape),
                  0.1 + rng.random(shape), phi0, x0, spacing),
        AuditCase("anisotropic_spacing", np.cos(phase) + 0.2*np.sin(2*phase),
                  0.2 + rng.random(shape), phi0, x0, (0.3, 0.9, 1.7)),
    ]

    slab_phi = np.zeros(shape)
    slab_phi[2:5, :, :] = 1.0
    slab_mobility = 1.0 - h_switch(slab_phi)
    cases.append(AuditCase("pure_beta_slab", np.sin(phase), slab_mobility,
                           slab_phi, x0, spacing))

    coordinate = index[0] - 0.5 * shape[0]
    smooth_phi = 0.5 * (1.0 - np.tanh(coordinate / 1.2))
    smooth_mobility = (1.0 - h_switch(smooth_phi)) * (0.4 + 0.1*np.cos(phase))
    cases.append(AuditCase("smooth_matrix_interface_beta", np.sin(phase),
                           smooth_mobility, smooth_phi, x0, spacing))

    circular_shape = (64, 2, 64)
    dx = 0.25
    ii, jj, kk = np.indices(circular_shape, dtype=float)
    rr = np.sqrt(((ii - circular_shape[0]/2)*dx)**2 +
                 ((kk - circular_shape[2]/2)*dx)**2)
    circle_phi = 0.5 * (1.0 - np.tanh((rr - 3.0) / 0.3))
    circle_h = h_switch(circle_phi)
    circle_x_eq = np.full(circular_shape, 0.009182741595466306)
    circle_mu_eq = np.log(circle_x_eq / (1.0-circle_x_eq))
    circle_mobility = (1.0-circle_h) * circle_x_eq * (1.0-circle_x_eq)
    cases.append(AuditCase("resolved_circular_equilibrium", circle_mu_eq,
                           circle_mobility, circle_phi, circle_x_eq,
                           (dx, 1.0, dx)))

    x_far, x_surface = 0.0078305391025, 0.009182741595466306
    radial_weight = np.exp(-np.maximum(rr-3.0, 0.0)/2.0)
    circle_x = x_far + (x_surface-x_far)*radial_weight
    circle_mu = np.log(circle_x/(1.0-circle_x))
    circle_mobility = (1.0-circle_h)*circle_x*(1.0-circle_x)
    cases.append(AuditCase("resolved_circular_dissolution_off", circle_mu,
                           circle_mobility, circle_phi, circle_x,
                           (dx, 1.0, dx), False))
    cases.append(AuditCase("resolved_circular_dissolution_on", circle_mu,
                           circle_mobility, circle_phi, circle_x,
                           (dx, 1.0, dx), True))
    return cases


def audit_case(case: AuditCase) -> dict[str, object]:
    volume = math.prod(case.spacing)
    gradients = [face_difference(case.mu, axis, delta)
                 for axis, delta in enumerate(case.spacing)]
    face_mobility = [harmonic_face_mobility(case.mobility, axis)
                     for axis in range(case.mu.ndim)]
    base_flux = [mobility * gradient
                 for mobility, gradient in zip(face_mobility, gradients)]
    correction_flux = [np.zeros_like(case.mu) for _ in range(case.mu.ndim)]
    if case.correction_enabled:
        # An arbitrary deterministic signed interface current is sufficient to
        # verify that its work remains separate from nonnegative base power.
        H = h_switch(case.phi)
        for axis in range(case.mu.ndim):
            correction_flux[axis] = (1.0e-3 * H * (1.0-H) *
                                     np.sin(case.mu + 0.31*axis))
    total_flux = [base + correction for base, correction
                  in zip(base_flux, correction_flux)]
    cdot = face_divergence(total_flux, case.spacing)
    base_dissipation = sum(float(np.sum(mobility*gradient*gradient))*volume
                           for mobility, gradient in zip(face_mobility, gradients))
    correction_work = sum(float(np.sum(correction*gradient))*volume
                          for correction, gradient in zip(correction_flux, gradients))
    cell_power = float(np.sum(case.mu*cdot))*volume
    adjoint_residual = cell_power + base_dissipation + correction_work

    current_gradient = current_nonadjoint_face_gradient(case.mu, case.spacing)
    current_flux = [mobility*gradient for mobility, gradient
                    in zip(face_mobility, current_gradient)]
    current_cdot = face_divergence(current_flux, case.spacing)
    current_cell_power = float(np.sum(case.mu*current_cdot))*volume
    current_face_power = sum(float(np.sum(flux*gradient))*volume
                             for flux, gradient in zip(current_flux, current_gradient))
    current_identity_defect = current_cell_power + current_face_power

    closed_flux_count = 0
    for axis, (mobility, flux) in enumerate(zip(face_mobility, base_flux)):
        closed = mobility == 0.0
        closed_flux_count += int(np.count_nonzero(closed & (flux != 0.0)))
    h = h_switch(case.phi)
    ctot = h + (1.0-h)*case.x
    target = ctot + 1.0e-12*cdot
    lower = h
    upper = np.ones_like(h)
    scale = max(abs(cell_power), abs(base_dissipation), abs(correction_work), 1.0)
    return {
        "case": case.name,
        "shape": "x".join(str(n) for n in case.mu.shape),
        "dx": case.spacing[0], "dy": case.spacing[1], "dz": case.spacing[2],
        "cell_power": cell_power,
        "face_dissipation": base_dissipation,
        "finite_interface_signed_work": correction_work,
        "adjoint_residual": adjoint_residual,
        "adjoint_residual_scaled": abs(adjoint_residual)/scale,
        "global_sum_Cdot": float(np.sum(cdot))*volume,
        "minimum_M_face": min(float(np.min(value)) for value in face_mobility),
        "negative_M_face_count": sum(int(np.count_nonzero(value < 0.0))
                                     for value in face_mobility),
        "closed_face_nonzero_flux_count": closed_flux_count,
        "BE_lower_target_violation_count": int(np.count_nonzero(target < lower-1e-12)),
        "BE_upper_target_violation_count": int(np.count_nonzero(target > upper+1e-12)),
        "current_nonadjoint_cell_power": current_cell_power,
        "current_nonadjoint_face_power": current_face_power,
        "current_nonadjoint_identity_defect": current_identity_defect,
        "status": "PASS" if (
            base_dissipation >= 0.0 and
            abs(adjoint_residual)/scale <= 5.0e-13 and
            abs(float(np.sum(cdot))*volume) <= 5.0e-12 and
            closed_flux_count == 0 and
            not np.any(target < lower-1e-12) and
            not np.any(target > upper+1e-12)
        ) else "FAIL",
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = [audit_case(case) for case in make_cases()]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(f"{row['case']}={row['status']} adj={row['adjoint_residual_scaled']:.3e}")
    return 0 if all(row["status"] == "PASS" for row in rows) else 1


if __name__ == "__main__":
    raise SystemExit(main())
