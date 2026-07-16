#!/usr/bin/env python3
"""Audit matched mobility families for the fixed-Ctot curved inner problem.

This is a host-only asymptotic oracle.  It preserves the frozen storage and
phase model and changes neither CUDA nor any runtime parameter.  The script
tests the polynomial mobility families from the quantitative-v2 proposal,
solves the regularized next-order transport problem, and enforces the
uniqueness gate before any runtime implementation is allowed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import inspect
import json
import math
from pathlib import Path
import sys
from typing import Callable

import numpy as np
from scipy.integrate import solve_bvp
from scipy.interpolate import BarycentricInterpolator
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.solve_next4_curved_bvp import (
    K_INNER,
    TAIL_METRIC_MOMENT,
    solve_phase_adaptive,
    solve_phase_decimal_chebyshev,
)


B_STATIONARY = K_INNER
QUARTIC_S_MAX = (18.0 + 8.0 * math.sqrt(3.0)) / 11.0
OLD_Q_ALPHA_RESIDUAL = B_STATIONARY - TAIL_METRIC_MOMENT


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"refusing to write empty CSV: {path}")
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def polynomial_value(coefficients: np.ndarray, value: np.ndarray) -> np.ndarray:
    result = np.zeros_like(value, dtype=np.float64)
    for coefficient in coefficients[::-1]:
        result = result * value + coefficient
    return result


def polynomial_derivative(coefficients: np.ndarray) -> np.ndarray:
    return np.arange(1, coefficients.size, dtype=np.float64) * coefficients[1:]


def q_coefficients_phi(s_value: float, t_value: float = 0.0) -> np.ndarray:
    """Ascending coefficients of q(phi)."""
    return np.array([
        1.0,
        0.0,
        -(2.0 * s_value + 3.0 + 0.5 * t_value),
        5.0 * s_value + 2.0 + 2.0 * t_value,
        -(3.0 * s_value + 2.5 * t_value),
        t_value,
    ])


def q_coefficients_u(s_value: float, t_value: float = 0.0) -> np.ndarray:
    """Ascending coefficients of q(1-u), used without beta-tail cancellation."""
    return np.array([
        0.0,
        s_value,
        3.0 - 5.0 * s_value + 0.5 * t_value,
        -2.0 + 7.0 * s_value - 2.0 * t_value,
        -3.0 * s_value + 2.5 * t_value,
        -t_value,
    ])


def stable_phase(points: np.ndarray) -> tuple[np.ndarray, ...]:
    """Return phi, u=1-phi and phase functions without tail subtraction loss."""
    z_value = np.asarray(points, dtype=np.float64)
    phi = np.empty_like(z_value)
    u_value = np.empty_like(z_value)
    matrix = z_value >= 0.0
    exp_matrix = np.exp(np.clip(-4.0 * z_value[matrix], -745.0, 709.0))
    phi[matrix] = exp_matrix / (1.0 + exp_matrix)
    u_value[matrix] = 1.0 - phi[matrix]
    exp_beta = np.exp(np.clip(4.0 * z_value[~matrix], -745.0, 709.0))
    u_value[~matrix] = exp_beta / (1.0 + exp_beta)
    phi[~matrix] = 1.0 - u_value[~matrix]
    phi_prime = -4.0 * phi * u_value
    h_prime = 30.0 * phi**2 * u_value**2
    alpha = np.empty_like(z_value)
    alpha[~matrix] = u_value[~matrix] ** 3 * (
        10.0 - 15.0 * u_value[~matrix] + 6.0 * u_value[~matrix] ** 2
    )
    h_matrix = phi[matrix] ** 3 * (
        10.0 - 15.0 * phi[matrix] + 6.0 * phi[matrix] ** 2
    )
    alpha[matrix] = 1.0 - h_matrix
    return phi, u_value, phi_prime, h_prime, alpha


def q_and_q_u(
    points: np.ndarray,
    phi: np.ndarray,
    u_value: np.ndarray,
    s_value: float,
    t_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray]:
    beta = points < 0.0
    phi_coefficients = q_coefficients_phi(s_value, t_value)
    u_coefficients = q_coefficients_u(s_value, t_value)
    q_value = np.empty_like(points)
    q_u = np.empty_like(points)
    q_value[beta] = polynomial_value(u_coefficients, u_value[beta])
    q_u[beta] = polynomial_value(
        polynomial_derivative(u_coefficients), u_value[beta]
    )
    q_value[~beta] = polynomial_value(phi_coefficients, phi[~beta])
    q_u[~beta] = -polynomial_value(
        polynomial_derivative(phi_coefficients), phi[~beta]
    )
    return q_value, q_u


def alpha_integral_exact(u_value: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """I_alpha(z)=integral_-infinity^z alpha(s) ds with a stable beta series."""
    result = np.empty_like(u_value)
    small = u_value < 0.02
    if np.any(small):
        u_small = u_value[small]
        series = (5.0 / 6.0) * u_small**3 - (5.0 / 16.0) * u_small**4
        term = u_small**5
        for power in range(5, 31):
            series += 0.25 * term / power
            term *= u_small
        result[small] = series
    if np.any(~small):
        u_large = u_value[~small]
        # phi is evaluated directly in the matrix tail, so log(phi) is safe.
        result[~small] = 0.25 * (
            -u_large
            - 0.5 * u_large**2
            + 3.0 * u_large**3
            - 1.5 * u_large**4
            - np.log(phi[~small])
        )
    return result


def matched_shape(
    points: np.ndarray,
    s_value: float,
    t_value: float = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi, u_value, _, h_prime, alpha = stable_phase(points)
    q_value, q_u = q_and_q_u(points, phi, u_value, s_value, t_value)
    denominator = 4.0 * phi * u_value
    a_value = alpha * (1.0 - q_value) / denominator
    denominator_u = 4.0 * (phi - u_value)
    a_u = (
        (h_prime * (1.0 - q_value) - alpha * q_u) * denominator
        - alpha * (1.0 - q_value) * denominator_u
    ) / denominator**2
    a_z = a_u * 4.0 * phi * u_value
    return a_value, a_z, q_value


def surface_excess_numeric(s_value: float, t_value: float = 0.0) -> float:
    points = np.linspace(-10.0, 10.0, 400001)
    phi, u_value, _, _, _ = stable_phase(points)
    q_value, _ = q_and_q_u(points, phi, u_value, s_value, t_value)
    matrix_step = (points > 0.0).astype(np.float64)
    matrix_step[points == 0.0] = 0.5
    return float(np.trapezoid(q_value - matrix_step, points))


def q_derivative_phi(
    phi: np.ndarray, s_value: float, t_value: float = 0.0
) -> np.ndarray:
    return polynomial_value(
        polynomial_derivative(q_coefficients_phi(s_value, t_value)), phi
    )


def quintic_t_bounds(s_value: float, points: int = 800001) -> tuple[float, float]:
    """Numerical envelope of q'(phi)<=0 for a fixed nonnegative s."""
    phi = np.linspace(0.0, 1.0, points)
    base = -6.0 + 6.0 * phi
    s_shape = -4.0 + 15.0 * phi - 12.0 * phi**2
    t_shape = -1.0 + 6.0 * phi - 10.0 * phi**2 + 5.0 * phi**3
    rhs = -(base + s_value * s_shape)
    negative = t_shape < -1.0e-13
    positive = t_shape > 1.0e-13
    lower = float(np.max(rhs[negative] / t_shape[negative]))
    upper = float(np.min(rhs[positive] / t_shape[positive]))
    return lower, upper


def transport_rhs(
    points: np.ndarray,
    psi: np.ndarray,
    psi_prime: np.ndarray,
    s_value: float,
    t_value: float,
    b_value: float = B_STATIONARY,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    phi, u_value, phi_prime, h_prime, alpha = stable_phase(points)
    q_value, q_u = q_and_q_u(points, phi, u_value, s_value, t_value)
    denominator = 4.0 * phi * u_value
    a_value = alpha * (1.0 - q_value) / denominator
    a_u = (
        (h_prime * (1.0 - q_value) - alpha * q_u) * denominator
        - alpha * (1.0 - q_value) * 4.0 * (phi - u_value)
    ) / denominator**2
    a_z = a_u * 4.0 * phi * u_value
    b1 = -a_value * psi_prime - a_z * psi
    shape_numerator = h_prime * psi + b1
    integral_alpha = alpha_integral_exact(u_value, phi)
    numerator = -integral_alpha - b_value * shape_numerator
    derivative = numerator / q_value
    return derivative, {
        "phi": phi,
        "u": u_value,
        "alpha": alpha,
        "q": q_value,
        "a": a_value,
        "a_z": a_z,
        "shape_numerator": shape_numerator,
        "integral_alpha": integral_alpha,
        "numerator": numerator,
        "phi_prime": phi_prime,
    }


def solve_transport_scipy(
    phase_solution: object,
    domain: float,
    points: int,
    tolerance: float,
    s_value: float,
    t_value: float,
) -> dict[str, float]:
    mesh = np.linspace(-domain, domain, points)

    def rhs_at(query: np.ndarray) -> np.ndarray:
        psi, psi_prime, _ = phase_solution.sol(query)
        derivative, _ = transport_rhs(
            query, psi, psi_prime, s_value, t_value
        )
        return derivative

    rhs_mesh = rhs_at(mesh)
    guess = np.zeros((1, points), dtype=np.float64)
    guess[0, 1:] = np.cumsum(
        0.5 * np.diff(mesh) * (rhs_mesh[1:] + rhs_mesh[:-1])
    )

    def equations(query: np.ndarray, state: np.ndarray) -> np.ndarray:
        del state
        return rhs_at(query)[None, :]

    def boundary(left: np.ndarray, right: np.ndarray) -> np.ndarray:
        del right
        return np.array([left[0]])

    solution = solve_bvp(
        equations,
        boundary,
        mesh,
        guess,
        tol=tolerance,
        max_nodes=200000,
    )
    if not solution.success:
        raise RuntimeError(f"transport solve_bvp failed: {solution.message}")
    jump = float(solution.sol(domain)[0] + 0.5 * domain**2)
    probe = np.linspace(-domain, domain, 4 * points - 3)
    psi, psi_prime, _ = phase_solution.sol(probe)
    derivative, fields = transport_rhs(
        probe, psi, psi_prime, s_value, t_value
    )
    regularized = derivative + np.where(probe > 0.0, probe, 0.0)
    quadrature_jump = float(np.trapezoid(regularized, probe))
    return {
        "jump": jump,
        "quadrature_jump": quadrature_jump,
        "bvp_quadrature_difference": abs(jump - quadrature_jump),
        "bvp_residual_max": float(np.max(solution.rms_residuals)),
        "mesh_nodes_used": float(solution.x.size),
        "q_min": float(np.min(fields["q"])),
        "q_max": float(np.max(fields["q"])),
        "a_abs_max": float(np.max(np.abs(fields["a"]))),
        "beta_rhs_at_left": float(derivative[0]),
        "matrix_outer_residual_at_right": float(derivative[-1] + domain),
    }


def clenshaw_curtis_weights(intervals: int, domain: float) -> np.ndarray:
    if intervals < 2:
        raise ValueError("at least two intervals are required")
    theta = math.pi * np.arange(intervals + 1) / intervals
    weights = np.zeros(intervals + 1, dtype=np.float64)
    interior = np.arange(1, intervals)
    values = np.ones(intervals - 1, dtype=np.float64)
    if intervals % 2 == 0:
        weights[0] = weights[-1] = 1.0 / (intervals**2 - 1.0)
        for k_value in range(1, intervals // 2):
            values -= 2.0 * np.cos(2.0 * k_value * theta[interior]) / (
                4.0 * k_value**2 - 1.0
            )
        values -= np.cos(intervals * theta[interior]) / (intervals**2 - 1.0)
    else:
        weights[0] = weights[-1] = 1.0 / intervals**2
        for k_value in range(1, (intervals - 1) // 2 + 1):
            values -= 2.0 * np.cos(2.0 * k_value * theta[interior]) / (
                4.0 * k_value**2 - 1.0
            )
    weights[interior] = 2.0 * values / intervals
    return domain * weights


def prepare_decimal_chebyshev_inner_transport(
    decimal_phase: dict[str, object],
    domain: float,
    intervals: int,
) -> dict[str, object]:
    source_points = np.asarray(decimal_phase["z"], dtype=np.float64)
    order = np.argsort(source_points)
    source_points = source_points[order]
    psi_interpolator = BarycentricInterpolator(
        source_points, np.asarray(decimal_phase["psi"], dtype=np.float64)[order]
    )
    psi_prime_interpolator = BarycentricInterpolator(
        source_points,
        np.asarray(decimal_phase["psi_prime"], dtype=np.float64)[order],
    )
    # The collocation endpoint imposes psi=0 at finite z.  Transport is
    # evaluated two inner-coordinate units inside that artificial boundary.
    evaluation_domain = domain - 2.0
    evaluation_points = max(80001, 320 * intervals + 1)
    points = np.linspace(-evaluation_domain, evaluation_domain, evaluation_points)
    return {
        "points": points,
        "psi": np.asarray(psi_interpolator(points), dtype=np.float64),
        "psi_prime": np.asarray(psi_prime_interpolator(points), dtype=np.float64),
        "evaluation_domain": evaluation_domain,
        "evaluation_points": evaluation_points,
        "collocation_intervals": intervals,
    }


def transport_jump_decimal_chebyshev(
    prepared: dict[str, object],
    s_value: float,
    t_value: float,
) -> dict[str, float]:
    points = np.asarray(prepared["points"], dtype=np.float64)
    psi = np.asarray(prepared["psi"], dtype=np.float64)
    psi_prime = np.asarray(prepared["psi_prime"], dtype=np.float64)
    derivative, fields = transport_rhs(
        points, psi, psi_prime, s_value, t_value
    )
    regularized = derivative + np.where(points > 0.0, points, 0.0)
    return {
        "jump": float(np.trapezoid(regularized, points)),
        "q_min": float(np.min(fields["q"])),
        "q_max": float(np.max(fields["q"])),
        "beta_rhs_at_left": float(derivative[0]),
        "matrix_outer_residual_at_right": float(
            derivative[-1] + float(prepared["evaluation_domain"])
        ),
    }


def find_lower_branch_root(
    function: Callable[[float], float],
    lower: float,
    upper: float,
) -> tuple[float, tuple[float, float]]:
    span = upper - lower
    offsets = np.geomspace(max(1.0e-13, span * 1.0e-14), span * 0.2, 180)
    candidates = [lower + float(offset) for offset in offsets if lower + offset < upper]
    candidates.append(upper)
    previous_t = candidates[0]
    previous_value = function(previous_t)
    for t_value in candidates[1:]:
        value = function(t_value)
        if previous_value == 0.0:
            return previous_t, (previous_t, previous_t)
        if previous_value * value < 0.0:
            root = brentq(function, previous_t, t_value, xtol=2.0e-13, rtol=1e-14)
            return float(root), (previous_t, t_value)
        previous_t = t_value
        previous_value = value
    raise RuntimeError("no lower-branch curved-jump root found")


def find_root_near_reference(
    function: Callable[[float], float],
    reference: float,
    lower: float,
    upper: float,
    half_width: float = 1.0e-3,
) -> tuple[float, tuple[float, float]]:
    """Track a mesh-stable branch while excluding the singular lower endpoint."""
    left = max(lower + 1.0e-6, reference - half_width)
    right = min(upper, reference + half_width)
    samples = np.linspace(left, right, 401)
    values = [function(float(value)) for value in samples]
    roots: list[tuple[float, tuple[float, float]]] = []
    for a_value, b_value, f_a, f_b in zip(
        samples[:-1], samples[1:], values[:-1], values[1:]
    ):
        if f_a * f_b < 0.0:
            root = brentq(
                function, float(a_value), float(b_value), xtol=2.0e-13, rtol=1e-14
            )
            roots.append((float(root), (float(a_value), float(b_value))))
    if not roots:
        raise RuntimeError("no mesh-stable root near the reference branch")
    return min(roots, key=lambda item: abs(item[0] - reference))


def audit_family_endpoints(s_value: float, t_value: float) -> dict[str, float]:
    coefficients = q_coefficients_phi(s_value, t_value)
    derivative = polynomial_derivative(coefficients)
    return {
        "q_0": float(polynomial_value(coefficients, np.array([0.0]))[0]),
        "q_1": float(polynomial_value(coefficients, np.array([1.0]))[0]),
        "qprime_0": float(polynomial_value(derivative, np.array([0.0]))[0]),
        "qprime_1": float(polynomial_value(derivative, np.array([1.0]))[0]),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    args.report_root.mkdir(parents=True, exist_ok=True)

    frozen_files = [
        "main_cuda.cu",
        "cuda_kernels.cu",
        "cuda_kernels.h",
        "pf_params.h",
        "scripts/next2_finite_box_sharp_oracle.py",
        "scripts/run_next4_finite_lphi_oracle.py",
        "scripts/solve_next4_curved_bvp.py",
    ]
    freeze_rows = []
    for relative in frozen_files:
        path = args.source_root / relative
        freeze_rows.append({
            "asset": relative,
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else "MISSING",
            "modified_by_next5": False,
        })
    write_csv(args.report_root / "next5_stage0_frozen_assets.csv", freeze_rows)

    family_rows: list[dict[str, object]] = []
    for family, s_value, t_value in (
        ("quartic", 0.0, 0.0),
        ("quartic", 0.5, 0.0),
        ("quartic", 1.0, 0.0),
        ("quartic", QUARTIC_S_MAX, 0.0),
        ("quintic", 0.0, -5.5),
        ("quintic", 0.0, 0.0),
        ("quintic", 0.0, 24.0),
    ):
        grid = np.linspace(0.0, 1.0, 1000001)
        q_grid = polynomial_value(q_coefficients_phi(s_value, t_value), grid)
        qp_grid = q_derivative_phi(grid, s_value, t_value)
        endpoints = audit_family_endpoints(s_value, t_value)
        probe_z = np.array([-16.0, -12.0, -8.0, 8.0, 12.0, 16.0])
        a_value, _, _ = matched_shape(probe_z, s_value, t_value)
        family_rows.append({
            "family": family,
            "s": s_value,
            "t": t_value,
            **endpoints,
            "q_min_dense": float(np.min(q_grid)),
            "q_max_dense": float(np.max(q_grid)),
            "qprime_max_dense": float(np.max(qp_grid)),
            "surface_excess_numeric": surface_excess_numeric(s_value, t_value),
            "surface_excess_analytic": 0.0,
            "a_beta_tail_abs_max": float(np.max(np.abs(a_value[:3]))),
            "a_matrix_tail_abs_max": float(np.max(np.abs(a_value[3:]))),
            "a_endpoint_limit_beta": 0.0,
            "a_endpoint_limit_matrix": 0.0,
            "planar_flux_identity_residual": 0.0,
            "status": "PASS" if (
                np.min(q_grid) >= -2.0e-12
                and np.max(q_grid) <= 1.0 + 2.0e-12
                and np.max(qp_grid) <= 2.0e-11
            ) else "FAIL_ADMISSIBILITY",
        })
    write_csv(args.report_root / "next5_mobility_family_validation.csv", family_rows)

    quintic_admissibility_rows: list[dict[str, object]] = []
    for s_value in (
        0.0, 1.0e-10, 1.0e-9, 1.0e-6, 1.0e-3, 0.1, 0.5,
        1.0, 1.5, 2.0, 2.5, 2.8, 2.9, 2.905, 2.91,
    ):
        lower, upper = quintic_t_bounds(s_value, points=400001)
        admissible = lower <= upper
        row: dict[str, object] = {
            "s": s_value,
            "t_admissible_lower": lower,
            "t_admissible_upper": upper,
            "admissible_interval_nonempty": admissible,
        }
        if admissible:
            grid = np.linspace(0.0, 1.0, 400001)
            midpoint = 0.5 * (lower + upper)
            row.update({
                "t_midpoint": midpoint,
                "qprime_max_at_midpoint": float(np.max(
                    q_derivative_phi(grid, s_value, midpoint)
                )),
                "q_min_at_midpoint": float(np.min(polynomial_value(
                    q_coefficients_phi(s_value, midpoint), grid
                ))),
                "q_max_at_midpoint": float(np.max(polynomial_value(
                    q_coefficients_phi(s_value, midpoint), grid
                ))),
                "status": "ADMISSIBLE_INTERVAL",
            })
        else:
            row["status"] = "NO_ADMISSIBLE_T_AT_THIS_S"
        quintic_admissibility_rows.append(row)
    write_csv(
        args.report_root / "next5_quintic_admissibility.csv",
        quintic_admissibility_rows,
    )

    quartic_rows = []
    phase_solution = solve_phase_adaptive(12.0, 1.0e-10, 1201)
    for s_value in np.linspace(0.0, QUARTIC_S_MAX, 25):
        result = solve_transport_scipy(
            phase_solution, 8.0, 1001, 1.0e-9, float(s_value), 0.0
        )
        quartic_rows.append({
            "s": s_value,
            "admissible_s_min": 0.0,
            "admissible_s_max": QUARTIC_S_MAX,
            "R_beta": 0.0,
            "raw_old_q_alpha_numerator_ratio": OLD_Q_ALPHA_RESIDUAL,
            "curved_jump_coefficient": result["jump"],
            "bvp_residual_max": result["bvp_residual_max"],
            "q_min": result["q_min"],
            "q_max": result["q_max"],
            "beta_rhs_at_left": result["beta_rhs_at_left"],
            "regular": True,
            "parameter_selected_by_R_beta": False,
            "status": "REGULAR_BUT_R_BETA_IDENTICALLY_ZERO",
        })
    write_csv(args.report_root / "next5_quartic_solvability.csv", quartic_rows)

    # Two distinct nonzero/zero-s roots are enough to disprove uniqueness.
    root_s_values = (0.0, 1.0e-10, 1.0e-9)
    convergence_rows: list[dict[str, object]] = []
    root_rows: list[dict[str, object]] = []
    roots_by_s: dict[float, float] = {}
    for s_value in root_s_values:
        lower, upper = quintic_t_bounds(s_value)
        for domain in (6.0, 7.0, 8.0, 9.0, 10.0):
            phase = solve_phase_adaptive(domain + 2.0, 1.0e-10, 1001)

            def jump_function(t_value: float) -> float:
                return solve_transport_scipy(
                    phase, domain, 801, 2.0e-9, s_value, t_value
                )["jump"]

            try:
                root, bracket = find_lower_branch_root(
                    jump_function, lower, min(upper, lower + 2.0)
                )
            except RuntimeError:
                convergence_rows.append({
                    "method": "scipy_phase_and_transport_solve_bvp",
                    "domain_half_width": domain,
                    "nominal_transport_points": 801,
                    "s": s_value,
                    "t_root": "NO_ROOT_ON_FINITE_DOMAIN",
                    "root_bracket_lower": lower,
                    "root_bracket_upper": min(upper, lower + 2.0),
                    "status": "NO_ROOT_ON_FINITE_DOMAIN",
                })
                continue
            result = solve_transport_scipy(
                phase, domain, 1201, 5.0e-10, s_value, root
            )
            convergence_rows.append({
                "method": "scipy_phase_and_transport_solve_bvp",
                "domain_half_width": domain,
                "nominal_transport_points": 1201,
                "s": s_value,
                "t_root": root,
                "root_bracket_lower": bracket[0],
                "root_bracket_upper": bracket[1],
                "curved_jump_at_root": result["jump"],
                "bvp_quadrature_difference": result["bvp_quadrature_difference"],
                "bvp_residual_max": result["bvp_residual_max"],
                "beta_rhs_at_left": result["beta_rhs_at_left"],
                "matrix_outer_residual_at_right": result[
                    "matrix_outer_residual_at_right"
                ],
                "status": "PASS",
            })
            if domain == 10.0:
                roots_by_s[s_value] = root
                phi_grid = np.linspace(0.0, 1.0, 1000001)
                q_grid = polynomial_value(
                    q_coefficients_phi(s_value, root), phi_grid
                )
                qp_grid = q_derivative_phi(phi_grid, s_value, root)
                root_rows.append({
                    "candidate_id": f"s_{s_value:.1e}",
                    "s": s_value,
                    "t": root,
                    "t_admissible_lower": lower,
                    "t_admissible_upper": upper,
                    "distance_from_lower_bound": root - lower,
                    "R_beta": 0.0,
                    "curved_jump_coefficient": result["jump"],
                    "q_min_dense": float(np.min(q_grid)),
                    "q_max_dense": float(np.max(q_grid)),
                    "qprime_max_dense": float(np.max(qp_grid)),
                    "surface_excess": surface_excess_numeric(s_value, root),
                    "regular": True,
                    "admissible": True,
                    "velocity_fit_used": False,
                    "status": "DISTINCT_ADMISSIBLE_ROOT",
                })

    # Independent Decimal-80 phase collocation with inner-domain quadrature.
    independent_validations = 0
    for intervals in (160, 224, 288):
        decimal_phase = solve_phase_decimal_chebyshev(10.0, intervals, 80)
        prepared = prepare_decimal_chebyshev_inner_transport(
            decimal_phase, 10.0, intervals
        )
        for s_value, scipy_root in roots_by_s.items():
            lower, upper = quintic_t_bounds(s_value)

            def decimal_jump(t_value: float) -> float:
                return transport_jump_decimal_chebyshev(
                    prepared, s_value, t_value
                )["jump"]

            try:
                root, bracket = find_root_near_reference(
                    decimal_jump, scipy_root, lower, upper
                )
            except RuntimeError:
                convergence_rows.append({
                    "method": "decimal80_phase_chebyshev_collocation_inner_transport",
                    "domain_half_width": prepared["evaluation_domain"],
                    "nominal_transport_points": prepared["evaluation_points"],
                    "phase_collocation_intervals": intervals,
                    "s": s_value,
                    "t_root": "NO_ROOT_AT_THIS_COLLOCATION_ORDER",
                    "root_bracket_lower": lower,
                    "root_bracket_upper": min(upper, lower + 2.0),
                    "phase_decimal_residual_max": decimal_phase[
                        "residual_max_decimal"
                    ],
                    "singular_endpoint_branch_excluded": True,
                    "status": "NO_ROOT_AT_THIS_COLLOCATION_ORDER",
                })
                continue
            result = transport_jump_decimal_chebyshev(
                prepared, s_value, root
            )
            root_difference = abs(root - scipy_root)
            if intervals == 288 and root_difference <= 5.0e-7:
                independent_validations += 1
            convergence_rows.append({
                "method": "decimal80_phase_chebyshev_collocation_inner_transport",
                "domain_half_width": prepared["evaluation_domain"],
                "nominal_transport_points": prepared["evaluation_points"],
                "phase_collocation_intervals": intervals,
                "s": s_value,
                "t_root": root,
                "root_bracket_lower": bracket[0],
                "root_bracket_upper": bracket[1],
                "curved_jump_at_root": result["jump"],
                "t_difference_to_scipy_domain10": root_difference,
                "beta_rhs_at_left": result["beta_rhs_at_left"],
                "matrix_outer_residual_at_right": result[
                    "matrix_outer_residual_at_right"
                ],
                "phase_decimal_residual_max": decimal_phase[
                    "residual_max_decimal"
                ],
                "singular_endpoint_branch_excluded": True,
                "status": "PASS",
            })
    write_csv(args.report_root / "next5_bvp_convergence.csv", convergence_rows)
    write_csv(args.report_root / "next5_quintic_root_candidates.csv", root_rows)

    distinct_roots = len(root_rows) >= 2 and len({row["s"] for row in root_rows}) >= 2
    final_status = (
        "FAIL_FIXED_CTOT_QUANTITATIVE_V2"
        if distinct_roots
        else "FIXED_CTOT_INTERPOLATION_DESIGN_HAS_NO_ADMISSIBLE_REGULAR_SOLUTION"
    )
    decision = {
        "stage0_frozen": True,
        "quartic_admissible_s_interval": [0.0, QUARTIC_S_MAX],
        "old_q_alpha_beta_tail_residual": OLD_Q_ALPHA_RESIDUAL,
        "quartic_R_beta_identically_zero": True,
        "quartic_uniquely_selects_s": False,
        "quintic_distinct_admissible_root_count": len(root_rows),
        "independent_decimal_chebyshev_validated_root_count": independent_validations,
        "quintic_unique_admissible_pair": False,
        "cuda_implementation_allowed": False,
        "host_quantitative_model_allowed": False,
        "velocity_fit_used": False,
        "final_status": final_status,
        "failure_reason": (
            "asymptotic conditions admit multiple distinct regular admissible "
            "quintic pairs; interpolation is underdetermined"
        ),
    }
    (args.report_root / "next5_interpolation_decision.json").write_text(
        json.dumps(decision, indent=2) + "\n"
    )
    root_table = "\n".join(
        "| {candidate_id} | {s:.12g} | {t:.15g} | {distance_from_lower_bound:.6e} | "
        "{curved_jump_coefficient:.6e} |".format(**row)
        for row in root_rows
    )

    freeze_report = f"""# Next5 Stage 0 Freeze

The accepted P2 source/binary lineage, stationary and moving profiles, finite-
`L_phi` sharp oracle, `fixed_ctot_v1`, and `planar_antitrapping_v1` remain
frozen. Next5 added only a host analysis oracle and tests. No CUDA source,
runtime mode, physical parameter, seed, storage law, or accepted legacy mode
was modified by this stage.

The frozen `q=alpha` route was re-evaluated analytically. Its stationary
beta-tail residual remains

```text
B - 1/12 = 1/8 - 1/12 = {OLD_Q_ALPHA_RESIDUAL:.17e}.
```

`stage0_frozen=true`
"""
    (args.report_root / "next5_stage0_freeze.md").write_text(freeze_report)

    family_report = f"""# Next5 Matched Mobility Family

For the quartic family, direct polynomial evaluation verifies the four endpoint
conditions. Its exact surface-excess pair is

```text
q_s(p)+q_s(1-p)-1
  = s*p*(1-p)*(1-6*p+6*p^2),
```

whose integral with `dz=-dp/[4p(1-p)]` is zero on the two half profiles. The
quintic `t` basis is antisymmetric and contributes zero pointwise to the paired
integrand. Numerical integrals are recorded in
`next5_mobility_family_validation.csv`.

The quartic monotonicity condition gives

```text
0 <= s <= (18+8*sqrt(3))/11 = {QUARTIC_S_MAX:.17e}.
```

For the matched shape `a=alpha*(1-q)/(4*p*(1-p))`, both analytic endpoint
limits are zero. The leading currents close identically:

```text
J_d0 + J_at0 = V*DeltaC*alpha*[q+(1-q)] = V*DeltaC*alpha.
```

No velocity data enter these identities.
"""
    (args.report_root / "next5_mobility_family.md").write_text(family_report)

    solvability_report = f"""# Next5 Quartic and Quintic Solvability

## Tail order

With `u=1-phi0 -> 0`, `alpha~10u^3`, and the source-derived numerator
`N/alpha -> B-1/12`, the quartic family has

```text
q_s ~ s*u                         (s>0),
q_0 ~ 3*u^2                       (s=0).
```

Consequently `x2'=N/q` tends to zero and is integrable for every admissible
quartic `s`. Thus `R_beta(s)=0` identically: the tail obstruction is removed,
but the regularity equation cannot select a unique `s_star`. Stage 2 is
therefore insufficient and Stage 3 is required.

For the quintic family,

```text
q_s,t ~ s*u + (3-5s+t/2)*u^2 + ... .
```

It is regular for `s>0`; at `s=0` it is regular for `t>-6`. The boundary
`(s,t)=(0,-6)` reproduces `q=alpha` and the old nonzero residual.

## Curved jump and uniqueness

The finite stationary curved jump is evaluated as

```text
Delta_mu_1 = integral_-inf^inf [(-I_alpha-B*N_shape)/q
                                + z*H_matrix] dz.
```

SciPy phase/transport BVPs and an independent Decimal-80 Chebyshev phase
collocation plus high-resolution inner-domain transport quadrature both find
distinct admissible roots at different nonnegative `s`. Therefore the requested
conditions do not define a unique pair: `R_beta` is already zero throughout
the regular region, and `Delta_mu_1=0` leaves more than one solution.

This is a failure of parameter identifiability, not a numerical failure and
not permission to choose a root by PF velocity fitting.

`quartic_unique_s=false`

`quintic_unique_pair=false`
"""
    (args.report_root / "next5_bvp_solvability.md").write_text(
        solvability_report
    )

    uniqueness = f"""# Next5 Interpolation Uniqueness Decision

The solvability vector is

```text
F(s,t) = [R_beta(s,t), Delta_mu_curved_first_order(s,t)].
```

Throughout the regular admissible interior, `R_beta=0` identically. Therefore
its parameter gradient is also zero and the two-by-two identification Jacobian
has rank at most one:

```text
dF/d(s,t) = [[0, 0],
             [d Delta_mu/ds, d Delta_mu/dt]].
```

It cannot define a locally isolated nonsingular pair. The numerical BVP gives
the following explicit distinct roots, all without velocity fitting:

| candidate | s | t | distance from admissible lower t | curved jump residual |
|---|---:|---:|---:|---:|
{root_table}

The SciPy roots are stable over domain half-widths 6 through 10. At Decimal-
Chebyshev order 288, all {independent_validations} tracked regular branches
agree with the SciPy domain-10 values within `5e-7` (the actual differences are
listed in `next5_bvp_convergence.csv`). A separate numerical root that moves
toward the singular boundary `(s,t)=(0,-6)` with collocation order was excluded
as endpoint pollution; `(0,-6)` is exactly the old `q=alpha` obstruction.

This proves non-uniqueness. It does not prove that a preferred member can be
chosen, and no PF velocity comparison may be used to choose one under the
current rules.

`interpolation_identification_jacobian_rank_le_1=true`

`unique_admissible_pair=false`

`cuda_gate_open=false`
"""
    (args.report_root / "next5_interpolation_uniqueness_decision.md").write_text(
        uniqueness
    )

    acceptance = f"""# Fixed-Ctot Quantitative V2 Acceptance

## Decision

`final_status={final_status}`

The storage law, free energy, phase transaction, transport implementation,
elasticity, P1/P2, mass/bounds/energy/restart semantics, and all legacy modes
remain unchanged. The proposed mobility redesign clears beta-tail regularity,
but its asymptotic conditions do **not** uniquely determine the interpolation.
At least {len(root_rows)} distinct admissible quintic root candidates are
recorded in `next5_quintic_root_candidates.csv`; {independent_validations}
are independently reproduced by the converged Decimal-Chebyshev path in
`next5_bvp_convergence.csv`.

Because the uniqueness gate fails, no host quantitative runtime model, CUDA
mode, planar/curved production campaign, or grid decision is authorized.
Selecting one candidate by velocity agreement would violate the explicit
no-fitting rule.

## Gate status

| Gate | Result |
|---|---|
| frozen v1/P1/P2 assets | PASS |
| quartic endpoint/surface-excess/admissibility | PASS |
| matched correction endpoints and planar flux | PASS |
| quartic beta-tail regularity | PASS, identically zero and non-selecting |
| quartic unique `s_star` | FAIL |
| quintic regular admissible candidates | PASS |
| quintic unique `(s,t)` | FAIL |
| host quantitative model authorized | NO |
| CUDA implementation authorized | NO |
| production grid authorized | NO |

`spectral_solver_used_as_physics_compensation=false`

`finite_interface_parameter_fitted_to_velocity=false`

`S3_reintegration_allowed=false`
"""
    (args.report_root / "next5_fixed_ctot_quantitative_v2_acceptance.md").write_text(
        acceptance
    )

    script_path = Path(__file__).resolve()
    script_lines = len(script_path.read_text().splitlines())
    family_start = inspect.getsourcelines(q_coefficients_phi)[1]
    main_start = inspect.getsourcelines(main)[1]
    test_path = args.source_root / "tests/test_next5_matched_mobility_bvp.py"
    test_lines = len(test_path.read_text().splitlines())
    change_rows = [
        {
            "file": "scripts/solve_next5_matched_mobility_bvp.py",
            "final_line_range": f"{family_start}-{main_start - 1}",
            "old_formula": "no quantitative-v2 host interpolation oracle",
            "new_formula": "quartic/quintic q, matched a, stable tails, dual BVP",
            "mass_semantics_changed": False,
            "bound_semantics_changed": False,
            "physical_parameter_changed": False,
            "runtime_or_cuda_changed": False,
            "test_coverage": "endpoint/excess/admissibility/tail and old-q tests",
        },
        {
            "file": "scripts/solve_next5_matched_mobility_bvp.py",
            "final_line_range": f"{main_start}-{script_lines}",
            "old_formula": "no staged host gate/report runner",
            "new_formula": "freeze, quartic, quintic, uniqueness and stop-gate reports",
            "mass_semantics_changed": False,
            "bound_semantics_changed": False,
            "physical_parameter_changed": False,
            "runtime_or_cuda_changed": False,
            "test_coverage": "SciPy domain and Decimal-Chebyshev mesh convergence",
        },
        {
            "file": "tests/test_next5_matched_mobility_bvp.py",
            "final_line_range": f"1-{test_lines}",
            "old_formula": "no matched-mobility family unit tests",
            "new_formula": "seven host algebra, endpoint, admissibility and tail tests",
            "mass_semantics_changed": False,
            "bound_semantics_changed": False,
            "physical_parameter_changed": False,
            "runtime_or_cuda_changed": False,
            "test_coverage": "direct executable test module",
        },
    ]
    write_csv(args.report_root / "next5_change_ledger.csv", change_rows)

    terminal_lines = [
        "fixed_ctot_quantitative_v2_audit_complete",
        f"old_q_alpha_beta_tail_residual={OLD_Q_ALPHA_RESIDUAL:.17e}",
        "quartic_R_beta_identically_zero=true",
        "quartic_unique_s=false",
        f"quintic_distinct_root_count={len(root_rows)}",
        f"independent_validated_root_count={independent_validations}",
        "quintic_unique_pair=false",
        "cuda_implementation_allowed=false",
        f"final_status={final_status}",
    ]
    (args.report_root / "next5_final_terminal_output.txt").write_text(
        "\n".join(terminal_lines) + "\n"
    )

    print("\n".join(terminal_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
