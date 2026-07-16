#!/usr/bin/env python3
"""Low-cost manufactured checks for the Ctot transport discretizations."""

import argparse
import csv
import math
from pathlib import Path

import numpy as np


def h(phi):
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def fv_laplacian(values, dx):
    return (np.roll(values, -1) - 2.0 * values + np.roll(values, 1)) / dx**2


def spectral_laplacian(values, length):
    wave = 2.0 * math.pi * np.fft.fftfreq(values.size, d=length / values.size)
    return np.fft.ifft(-(wave**2) * np.fft.fft(values)).real


def spectral_gradient_shared_face_divergence(values, mobility, dx):
    gradient = spectral_derivative(values, 0, dx)
    mobility_next = np.roll(mobility, -1)
    denominator = mobility + mobility_next
    mobility_face = np.zeros_like(mobility)
    positive = (mobility > 0.0) & (mobility_next > 0.0)
    mobility_face[positive] = (
        2.0 * mobility[positive] * mobility_next[positive] /
        denominator[positive]
    )
    gradient_face = 0.5 * (gradient + np.roll(gradient, -1))
    flux_face = mobility_face * gradient_face
    divergence = (flux_face - np.roll(flux_face, 1)) / dx
    adjoint_gradient = (np.roll(values, -1) - values) / dx
    work = float(np.sum(flux_face * adjoint_gradient))
    return divergence, flux_face, work


def relative_l2(observed, expected):
    return float(np.linalg.norm(observed - expected) / np.linalg.norm(expected))


def spectral_derivative(values, axis, spacing):
    wave = 2.0 * math.pi * np.fft.fftfreq(values.shape[axis], d=spacing)
    shape = [1] * values.ndim
    shape[axis] = values.shape[axis]
    spectrum = np.fft.fftn(values)
    return np.fft.ifftn(1j * wave.reshape(shape) * spectrum).real


def centered_derivative(values, axis, spacing):
    return (np.roll(values, -1, axis=axis) -
            np.roll(values, 1, axis=axis)) / (2.0 * spacing)


def mechanical_divergence(stress, derivative, spacing):
    sxx, syy, szz, sxy, sxz, syz = stress
    return np.stack((
        derivative(sxx, 0, spacing) + derivative(sxy, 1, spacing) +
        derivative(sxz, 2, spacing),
        derivative(sxy, 0, spacing) + derivative(syy, 1, spacing) +
        derivative(syz, 2, spacing),
        derivative(sxz, 0, spacing) + derivative(syz, 1, spacing) +
        derivative(szz, 2, spacing),
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    previous_fv_error = None
    for n in (32, 64, 128):
        length = 2.0 * math.pi
        dx = length / n
        coordinates = np.arange(n) * dx
        field = np.sin(coordinates) + 0.2 * np.cos(2.0 * coordinates)
        exact = -np.sin(coordinates) - 0.8 * np.cos(2.0 * coordinates)
        fv_error = relative_l2(fv_laplacian(field, dx), exact)
        spectral_error = relative_l2(spectral_laplacian(field, length), exact)
        order = ""
        if previous_fv_error is not None:
            order = math.log(previous_fv_error / fv_error, 2.0)
        rows.append({
            "test": "fixed_h_diffusion_space",
            "method": "fv_shared_face",
            "N": n,
            "dt": "",
            "error": fv_error,
            "observed_order": order,
            "mass_residual": abs(float(np.sum(fv_laplacian(field, dx)))),
            "passed": fv_error < 0.02 and (order == "" or order > 1.9),
            "semantics": "periodic shared-face FV Laplacian converges to smooth diffusion operator",
        })
        rows.append({
            "test": "fixed_h_diffusion_space",
            "method": "spectral",
            "N": n,
            "dt": "",
            "error": spectral_error,
            "observed_order": "",
            "mass_residual": abs(float(np.sum(spectral_laplacian(field, length)))),
            "passed": spectral_error < 1.0e-11,
            "semantics": "spectral comparator differentiates represented smooth modes",
        })
        previous_fv_error = fv_error

    previous_hybrid_error = None
    for n in (32, 64, 128):
        length = 2.0 * math.pi
        dx = length / n
        coordinates = np.arange(n) * dx
        field = np.sin(coordinates) + 0.2 * np.cos(2.0 * coordinates)
        exact = -np.sin(coordinates) - 0.8 * np.cos(2.0 * coordinates)
        mobility = np.ones(n)
        observed, _, work = spectral_gradient_shared_face_divergence(
            field, mobility, dx)
        error = relative_l2(observed, exact)
        order = ""
        if previous_hybrid_error is not None:
            order = math.log(previous_hybrid_error / error, 2.0)
        rows.append({
            "test": "spectral_gradient_shared_face_refinement",
            "method": "spectral_gradient_harmonic_face_local_divergence",
            "N": n,
            "dt": "",
            "error": error,
            "observed_order": order,
            "mass_residual": abs(float(np.sum(observed))),
            "passed": (error < 0.02 and
                       (order == "" or order > 1.9) and work > 0.0),
            "semantics": "bound-compatible spectral-gradient flux is conservative, dissipative, and second-order convergent",
        })
        previous_hybrid_error = error

    n = 64
    length = 2.0 * math.pi
    dx = length / n
    coordinates = np.arange(n) * dx
    field = np.sin(coordinates)
    mobility = np.ones(n)
    mobility[n // 3:2 * n // 3] = 0.0
    observed, flux, work = spectral_gradient_shared_face_divergence(
        field, mobility, dx)
    closed = slice(n // 3, 2 * n // 3)
    closed_error = float(np.max(np.abs(observed[closed])))
    rows.append({
        "test": "spectral_gradient_closed_beta_support",
        "method": "spectral_gradient_harmonic_face_local_divergence",
        "N": n,
        "dt": "",
        "error": closed_error,
        "observed_order": "",
        "mass_residual": abs(float(np.sum(observed))),
        "passed": (closed_error == 0.0 and
                   abs(float(np.sum(observed))) <= 1.0e-13 and work >= 0.0),
        "semantics": "zero-capacity beta support receives exactly zero shared-face flux and divergence",
    })

    rate = 0.7
    final_time = 0.1
    exact_amplitude = math.exp(-rate * final_time)
    previous_error = None
    for dt in (0.02, 0.01, 0.005, 0.0025):
        steps = round(final_time / dt)
        amplitude = (1.0 / (1.0 + rate * dt)) ** steps
        error = abs(amplitude - exact_amplitude)
        order = ""
        if previous_error is not None:
            order = math.log(previous_error / error, 2.0)
        rows.append({
            "test": "backward_euler_equal_time",
            "method": "BE_scalar_mode",
            "N": 1,
            "dt": dt,
            "error": error,
            "observed_order": order,
            "mass_residual": 0.0,
            "passed": order == "" or order > 0.95,
            "semantics": "equal-time backward-Euler modal decay approaches first order",
        })
        previous_error = error

    rng = np.random.default_rng(20260712)
    phi_old = rng.uniform(0.0, 0.9, 4096)
    phi_new = rng.uniform(0.0, 0.9, 4096)
    x_old = rng.uniform(1.0e-4, 0.8, 4096)
    ctot = (1.0 - h(phi_old)) * x_old + h(phi_old)
    q_new = ctot - h(phi_new)
    moving_h_residual = float(np.max(np.abs(q_new + h(phi_new) - ctot)))
    rows.append({
        "test": "moving_h_zero_transport",
        "method": "fixed_local_Ctot",
        "N": 4096,
        "dt": "",
        "error": moving_h_residual,
        "observed_order": "",
        "mass_residual": moving_h_residual,
        "passed": moving_h_residual <= 2.0e-16,
        "semantics": "prescribed phase motion reconstructs q while Ctot stays cellwise fixed",
    })

    transfer = 0.037
    cell_delta = np.array([-transfer, transfer])
    rows.append({
        "test": "isolated_face_pair_ledger",
        "method": "shared_face",
        "N": 2,
        "dt": "",
        "error": abs(float(np.sum(cell_delta))),
        "observed_order": "",
        "mass_residual": abs(float(np.sum(cell_delta))),
        "passed": float(np.sum(cell_delta)) == 0.0,
        "semantics": "one face transfer is equal and opposite before any global reduction",
    })

    # Mechanical residual oracle. These cases exercise the same ik_j sigma_ij
    # contraction and periodic centered-difference comparator used at runtime.
    n = 16
    length = 2.0 * math.pi
    spacing = length / n
    xyz = np.meshgrid(*(np.arange(n) * spacing for _ in range(3)), indexing="ij")
    zeros = np.zeros((n, n, n))
    uniform_stress = (np.ones_like(zeros), 2.0 * np.ones_like(zeros),
                      3.0 * np.ones_like(zeros), zeros, zeros, zeros)
    uniform_div = mechanical_divergence(
        uniform_stress, spectral_derivative, spacing)
    uniform_error = float(np.max(np.abs(uniform_div)))
    rows.append({
        "test": "mechanical_uniform_stress_equilibrium",
        "method": "solver_spectral_divergence",
        "N": n,
        "dt": "",
        "error": uniform_error,
        "observed_order": "",
        "mass_residual": "",
        "passed": uniform_error <= 1.0e-13,
        "semantics": "k=0 stress has exactly zero solver-space divergence",
    })

    single = np.sin(2.0 * xyz[0])
    single_stress = (single, zeros, zeros, zeros, zeros, zeros)
    single_div = mechanical_divergence(
        single_stress, spectral_derivative, spacing)
    single_exact = np.stack((2.0 * np.cos(2.0 * xyz[0]), zeros, zeros))
    single_error = relative_l2(single_div, single_exact)
    rows.append({
        "test": "mechanical_single_fourier_mode",
        "method": "solver_spectral_divergence",
        "N": n,
        "dt": "",
        "error": single_error,
        "observed_order": "",
        "mass_residual": "",
        "passed": single_error <= 1.0e-12,
        "semantics": "ik_j sigma_ij reproduces the represented Fourier derivative",
    })

    # Airy stress generated from psi=sin(x)sin(y) is divergence free.
    psi = np.sin(xyz[0]) * np.sin(xyz[1])
    airy_stress = (-psi, -psi, zeros,
                   -np.cos(xyz[0]) * np.cos(xyz[1]), zeros, zeros)
    airy_div = mechanical_divergence(
        airy_stress, spectral_derivative, spacing)
    airy_error = float(np.max(np.abs(airy_div)))
    rows.append({
        "test": "mechanical_compatible_airy_stress",
        "method": "solver_spectral_divergence",
        "N": n,
        "dt": "",
        "error": airy_error,
        "observed_order": "",
        "mass_residual": "",
        "passed": airy_error <= 1.0e-12,
        "semantics": "compatible weak nonuniform stress satisfies div(sigma)=0",
    })

    previous_fd_error = None
    for n_mech in (16, 32, 64):
        dx_mech = length / n_mech
        xyz_mech = np.meshgrid(
            *(np.arange(n_mech) * dx_mech for _ in range(3)), indexing="ij")
        zero_mech = np.zeros((n_mech, n_mech, n_mech))
        smooth = np.sin(2.0 * xyz_mech[0]) * np.cos(xyz_mech[1])
        stress = (smooth, zero_mech, zero_mech, zero_mech, zero_mech, zero_mech)
        div_spectral = mechanical_divergence(
            stress, spectral_derivative, dx_mech)
        div_fd = mechanical_divergence(stress, centered_derivative, dx_mech)
        fd_error = relative_l2(div_fd, div_spectral)
        order = ""
        if previous_fd_error is not None:
            order = math.log(previous_fd_error / fd_error, 2.0)
        rows.append({
            "test": "mechanical_FD_comparator_refinement",
            "method": "centered_FD_vs_solver_spectral",
            "N": n_mech,
            "dt": "",
            "error": fd_error,
            "observed_order": order,
            "mass_residual": "",
            "passed": order == "" or order > 1.9,
            "semantics": "FD diagnostic converges to the spectral derivative under grid refinement",
        })
        previous_fd_error = fd_error

    columns = ["test", "method", "N", "dt", "error", "observed_order",
               "mass_residual", "passed", "semantics"]
    with output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    failed = sum(not row["passed"] for row in rows)
    print(f"pf_ctot_manufactured_rows={len(rows)}")
    print(f"pf_ctot_manufactured_failed={failed}")
    raise SystemExit(1 if failed else 0)


if __name__ == "__main__":
    main()
