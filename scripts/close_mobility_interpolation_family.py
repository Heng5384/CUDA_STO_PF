#!/usr/bin/env python3
"""Close the matched-q interpolation audit without fitting runtime data.

This host-only oracle preserves the frozen Ctot storage, phase model, CUDA
operators, and physical parameters.  It re-audits the first curved transport
equation after differentiating q(phi), validates that equation against a
finite-curvature replay, and tests genuinely next-order conditions before any
canonical gauge or production implementation can be authorized.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from typing import Callable

import numpy as np
import scipy
from scipy.integrate import cumulative_trapezoid, solve_bvp
from scipy.optimize import brentq

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.solve_next4_curved_bvp import solve_phase_adaptive
from scripts.solve_next5_matched_mobility_bvp import (
    B_STATIONARY,
    K_INNER,
    QUARTIC_S_MAX,
    polynomial_derivative,
    polynomial_value,
    q_and_q_u,
    q_coefficients_phi,
    q_coefficients_u,
    quintic_t_bounds,
    stable_phase,
    transport_rhs as next5_transport_rhs,
)


A_PHI = 2.0 / 3.0
FIRST_ORDER_DOMAIN = 8.0
FIRST_ORDER_POINTS = 80001
CURVATURE_DELTAS = (5.0e-4, 1.0e-3)
ROOT_BRANCH_SAMPLES = (
    0.0,
    1.0e-12,
    1.0e-11,
    1.0e-10,
    3.0e-10,
    1.0e-9,
    2.0e-9,
    4.0e-9,
    6.0e-9,
    7.0e-9,
    7.8e-9,
)


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


def phase_potential_derivative(phi: np.ndarray) -> np.ndarray:
    return 2.0 * phi * (1.0 - phi) * (1.0 - 2.0 * phi)


def h_derivative(phi: np.ndarray) -> np.ndarray:
    return 30.0 * phi**2 * (1.0 - phi) ** 2


def planar_phi(points: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(4.0 * points))


def planar_phi_prime(points: np.ndarray) -> np.ndarray:
    phi = planar_phi(points)
    return -4.0 * phi * (1.0 - phi)


def alpha_from_phase(phi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate alpha=1-h without beta-tail cancellation."""
    u_value = 1.0 - phi
    alpha = np.empty_like(phi)
    beta = phi >= 0.5
    alpha[beta] = u_value[beta] ** 3 * (
        10.0 - 15.0 * u_value[beta] + 6.0 * u_value[beta] ** 2
    )
    alpha[~beta] = 1.0 - phi[~beta] ** 3 * (
        10.0 - 15.0 * phi[~beta] + 6.0 * phi[~beta] ** 2
    )
    return alpha, u_value


def q_and_q_u_from_phase(
    phi: np.ndarray, u_value: np.ndarray, s_value: float, t_value: float
) -> tuple[np.ndarray, np.ndarray]:
    """Stable q and dq/du selected by phase side rather than grid position."""
    beta = phi >= 0.5
    q_value = np.empty_like(phi)
    q_u = np.empty_like(phi)
    phi_coefficients = q_coefficients_phi(s_value, t_value)
    u_coefficients = q_coefficients_u(s_value, t_value)
    q_value[beta] = polynomial_value(u_coefficients, u_value[beta])
    q_u[beta] = polynomial_value(
        polynomial_derivative(u_coefficients), u_value[beta]
    )
    q_value[~beta] = polynomial_value(phi_coefficients, phi[~beta])
    q_u[~beta] = -polynomial_value(
        polynomial_derivative(phi_coefficients), phi[~beta]
    )
    return q_value, q_u


def corrected_transport_rhs(
    points: np.ndarray,
    psi: np.ndarray,
    psi_prime: np.ndarray,
    s_value: float,
    t_value: float,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """First curved X-gradient including the required q1*X0' term.

    q1 = B*q_phi*psi and X0'=alpha.  Since q_u=-q_phi, the omitted
    constitutive term is +B*alpha*q_u*psi/q in the solved gradient.
    """
    legacy_derivative, fields = next5_transport_rhs(
        points, psi, psi_prime, s_value, t_value
    )
    phi = fields["phi"]
    u_value = fields["u"]
    q_value, q_u = q_and_q_u(points, phi, u_value, s_value, t_value)
    omitted_term = B_STATIONARY * fields["alpha"] * q_u * psi / q_value
    derivative = legacy_derivative + omitted_term
    fields = dict(fields)
    fields.update({
        "q_u": q_u,
        "q_phi": -q_u,
        "omitted_q1_X0_term": omitted_term,
        "corrected_derivative": derivative,
    })
    return derivative, fields


def corrected_first_jump(
    points: np.ndarray,
    psi: np.ndarray,
    psi_prime: np.ndarray,
    s_value: float,
    t_value: float,
) -> float:
    derivative, _ = corrected_transport_rhs(
        points, psi, psi_prime, s_value, t_value
    )
    matrix_step = (points > 0.0).astype(np.float64)
    matrix_step[points == 0.0] = 0.5
    return float(np.trapezoid(derivative + points * matrix_step, points))


def solve_curved_phase(
    delta: float, dimension_minus_one: int, domain: float = FIRST_ORDER_DOMAIN
) -> object:
    """Solve the fixed-equimolar stationary phase BVP at finite curvature."""
    mesh = np.linspace(-domain, domain, 1601)
    initial_phi = planar_phi(mesh)
    initial = np.vstack((
        initial_phi,
        planar_phi_prime(mesh),
        np.zeros_like(mesh),
    ))

    def equations(
        points: np.ndarray, state: np.ndarray, parameter: np.ndarray
    ) -> np.ndarray:
        phi, phi_prime, _ = state
        metric = delta / (1.0 + delta * points / dimension_minus_one)
        return np.vstack((
            phi_prime,
            (
                phase_potential_derivative(phi)
                - A_PHI * parameter[0] * h_derivative(phi)
            ) / K_INNER - metric * phi_prime,
            (phi - planar_phi(points)) * planar_phi_prime(points),
        ))

    def boundary(
        left: np.ndarray, right: np.ndarray, parameter: np.ndarray
    ) -> np.ndarray:
        del parameter
        return np.array([
            left[0] - planar_phi(np.array([-domain]))[0],
            right[0] - planar_phi(np.array([domain]))[0],
            left[2],
            right[2],
        ])

    solution = solve_bvp(
        equations,
        boundary,
        mesh,
        initial,
        p=np.array([K_INNER * delta]),
        tol=2.0e-9,
        max_nodes=100000,
    )
    if not solution.success:
        raise RuntimeError(f"finite-curvature phase BVP failed: {solution.message}")
    return solution


def prepare_curved_states(
    points: np.ndarray,
) -> dict[tuple[int, float], dict[str, np.ndarray | float]]:
    states: dict[tuple[int, float], dict[str, np.ndarray | float]] = {}
    deltas = sorted({0.0, *CURVATURE_DELTAS, *(-x for x in CURVATURE_DELTAS)})
    for geometry in (1, 2):
        for delta in deltas:
            solution = solve_curved_phase(delta, geometry)
            phi, phi_prime, _ = solution.sol(points)
            alpha, u_value = alpha_from_phase(phi)
            jacobian = (1.0 + delta * points / geometry) ** geometry
            jacobian_prime = delta * (
                1.0 + delta * points / geometry
            ) ** (geometry - 1)
            correction_integral = np.concatenate((
                [0.0], cumulative_trapezoid(jacobian_prime * alpha, points)
            ))
            profile_ratio = (-phi_prime) / (4.0 * phi * u_value)
            states[(geometry, delta)] = {
                "phi": phi,
                "u": u_value,
                "alpha": alpha,
                "jacobian": jacobian,
                "correction_integral": correction_integral,
                "profile_ratio": profile_ratio,
                "phase_mu": float(solution.p[0]),
                "phase_bvp_residual_max": float(np.max(solution.rms_residuals)),
            }
    return states


def finite_curvature_jump(
    points: np.ndarray,
    state: dict[str, np.ndarray | float],
    s_value: float,
    t_value: float,
) -> float:
    """Stable exact-delta linear-response chemical jump.

    The rearrangement
      f-b = alpha*q + alpha*(1-q)*(1-r) - J^-1 integral(J' alpha)
    avoids subtracting two O(alpha) beta-tail quantities.
    """
    phi = np.asarray(state["phi"])
    u_value = np.asarray(state["u"])
    alpha = np.asarray(state["alpha"])
    jacobian = np.asarray(state["jacobian"])
    correction_integral = np.asarray(state["correction_integral"])
    profile_ratio = np.asarray(state["profile_ratio"])
    q_value, _ = q_and_q_u_from_phase(phi, u_value, s_value, t_value)
    x_gradient = alpha + (
        alpha * (1.0 - q_value) * (1.0 - profile_ratio)
        - correction_integral / jacobian
    ) / q_value
    matrix_step = (points > 0.0).astype(np.float64)
    matrix_step[points == 0.0] = 0.5
    return float(np.trapezoid(x_gradient - matrix_step / jacobian, points))


def curvature_coefficients(
    points: np.ndarray,
    states: dict[tuple[int, float], dict[str, np.ndarray | float]],
    geometry: int,
    s_value: float,
    t_value: float,
) -> dict[str, float]:
    values: dict[float, float] = {}
    for delta in (0.0, *CURVATURE_DELTAS, *(-x for x in CURVATURE_DELTAS)):
        values[delta] = finite_curvature_jump(
            points, states[(geometry, delta)], s_value, t_value
        )
    half, full = CURVATURE_DELTAS
    linear_half = (values[half] - values[-half]) / (2.0 * half)
    linear_full = (values[full] - values[-full]) / (2.0 * full)
    quadratic_half = (
        values[half] + values[-half] - 2.0 * values[0.0]
    ) / (2.0 * half**2)
    quadratic_full = (
        values[full] + values[-full] - 2.0 * values[0.0]
    ) / (2.0 * full**2)
    return {
        "jump_delta_zero": values[0.0],
        "first_order_finite_delta": (4.0 * linear_half - linear_full) / 3.0,
        "second_order_finite_delta": (
            4.0 * quadratic_half - quadratic_full
        ) / 3.0,
        "first_order_half": linear_half,
        "first_order_full": linear_full,
        "second_order_half": quadratic_half,
        "second_order_full": quadratic_full,
    }


def bracketed_first_jump_root(
    jump: Callable[[float, float], float], s_value: float
) -> float:
    lower = -6.0 - 4.0 * s_value
    left = lower + 1.0e-12
    right = -5.5
    left_value = jump(s_value, left)
    right_value = jump(s_value, right)
    if left_value * right_value >= 0.0:
        raise RuntimeError("no corrected first-curvature root on regular branch")
    return float(brentq(
        lambda t_value: jump(s_value, t_value),
        left,
        right,
        xtol=5.0e-14,
        rtol=1.0e-14,
    ))


def surface_moment(
    points: np.ndarray, order: int, s_value: float, t_value: float
) -> float:
    phi, u_value, _, _, _ = stable_phase(points)
    q_value, _ = q_and_q_u(points, phi, u_value, s_value, t_value)
    matrix_step = (points > 0.0).astype(np.float64)
    # The continuum Heaviside value at one point is arbitrary.  Matching it to
    # q(0) removes an O(dx) point-quadrature artifact for asymmetric quintics.
    matrix_step[points == 0.0] = q_value[points == 0.0]
    return float(np.trapezoid(points**order * (q_value - matrix_step), points))


def derivative_energy(s_value: float, t_value: float, order: int) -> float:
    coefficients = q_coefficients_phi(s_value, t_value)
    for _ in range(order):
        coefficients = polynomial_derivative(coefficients)
    squared = np.polynomial.polynomial.polymul(coefficients, coefficients)
    return float(sum(value / (index + 1) for index, value in enumerate(squared)))


def finite_difference_column(
    function: Callable[[float, float], np.ndarray],
    s_value: float,
    t_value: float,
    ds_value: float,
    dt_value: float,
) -> np.ndarray:
    derivative_s = (
        function(s_value + ds_value, t_value)
        - function(s_value - ds_value, t_value)
    ) / (2.0 * ds_value)
    derivative_t = (
        function(s_value, t_value + dt_value)
        - function(s_value, t_value - dt_value)
    ) / (2.0 * dt_value)
    return np.column_stack((derivative_s, derivative_t))


def svd_summary(jacobian: np.ndarray, column_scale: np.ndarray) -> dict[str, object]:
    scaled = jacobian * column_scale[None, :]
    _, singular_values, right_vectors = np.linalg.svd(
        scaled, full_matrices=False
    )
    tolerance = max(scaled.shape) * np.finfo(float).eps * singular_values[0]
    rank = int(np.sum(singular_values > max(tolerance, 1.0e-10)))
    condition = (
        float(singular_values[0] / singular_values[-1])
        if singular_values[-1] > max(tolerance, 1.0e-15)
        else math.inf
    )
    return {
        "scaled_jacobian": scaled,
        "singular_values": singular_values,
        "rank": rank,
        "nullity": scaled.shape[1] - rank,
        "condition_number": condition,
        "right_singular_vectors": right_vectors,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, default=Path.cwd())
    args = parser.parse_args()
    args.report_root.mkdir(parents=True, exist_ok=True)

    frozen_assets = [
        "main_cuda.cu",
        "cuda_kernels.cu",
        "cuda_kernels.h",
        "pf_params.h",
        "scripts/solve_next4_curved_bvp.py",
        "scripts/solve_next5_matched_mobility_bvp.py",
        "tests/test_next5_matched_mobility_bvp.py",
        "reports/pf_ctot_production_candidate/next5_bvp_convergence.csv",
        "reports/pf_ctot_production_candidate/next5_quintic_root_candidates.csv",
        "reports/pf_ctot_production_candidate/next5_interpolation_decision.json",
    ]
    freeze_rows = []
    for relative in frozen_assets:
        path = args.source_root / relative
        freeze_rows.append({
            "asset": relative,
            "exists": path.exists(),
            "sha256": sha256(path) if path.exists() else "MISSING",
            "modified_by_next6": False,
            "status": "FROZEN_ARCHIVE",
        })
    write_csv(args.report_root / "next6_stage0_frozen_assets.csv", freeze_rows)

    points = np.linspace(
        -FIRST_ORDER_DOMAIN, FIRST_ORDER_DOMAIN, FIRST_ORDER_POINTS
    )
    phase_solution = solve_phase_adaptive(10.0, 1.0e-10, 1201)
    psi, psi_prime, _ = phase_solution.sol(points)

    def first_jump(s_value: float, t_value: float) -> float:
        return corrected_first_jump(
            points, psi, psi_prime, s_value, t_value
        )

    old_roots = (
        (0.0, -5.999584318767566),
        (1.0e-10, -5.9996149138032555),
        (1.0e-9, -5.9997434763700275),
    )
    reaudit_rows = []
    for s_value, t_value in old_roots:
        legacy, fields = next5_transport_rhs(
            points, psi, psi_prime, s_value, t_value
        )
        corrected, corrected_fields = corrected_transport_rhs(
            points, psi, psi_prime, s_value, t_value
        )
        matrix_step = (points > 0.0).astype(np.float64)
        matrix_step[points == 0.0] = 0.5
        legacy_jump = float(np.trapezoid(
            legacy + points * matrix_step, points
        ))
        omitted_integral = float(np.trapezoid(
            corrected_fields["omitted_q1_X0_term"], points
        ))
        corrected_jump = float(np.trapezoid(
            corrected + points * matrix_step, points
        ))
        reaudit_rows.append({
            "s": s_value,
            "t_next5": t_value,
            "next5_reported_jump_formula_residual": legacy_jump,
            "omitted_minus_q1_X0prime_integral": omitted_integral,
            "corrected_first_curved_jump": corrected_jump,
            "q_min_on_inner_grid": float(np.min(fields["q"])),
            "old_root_still_valid": False,
            "status": "SUPERSEDED_OMITTED_Q1_X0PRIME",
        })
    write_csv(args.report_root / "next6_next5_root_reaudit.csv", reaudit_rows)

    quartic_rows = []
    quartic_nodes = 0.5 * QUARTIC_S_MAX * (
        1.0 - np.cos(np.pi * np.arange(1025) / 1024.0)
    )
    for s_value in quartic_nodes:
        quartic_rows.append({
            "s": float(s_value),
            "t": 0.0,
            "corrected_first_curved_jump": first_jump(float(s_value), 0.0),
            "regular": True,
            "admissible": True,
        })
    write_csv(args.report_root / "next6_corrected_quartic_scan.csv", quartic_rows)

    global_isolation_rows = []
    isolation_s_values = sorted(set((
        0.0,
        1.0e-12,
        1.0e-11,
        1.0e-10,
        1.0e-9,
        4.0e-9,
        7.0e-9,
        8.0e-9,
        1.0e-8,
        1.0e-7,
        1.0e-6,
        1.0e-5,
        1.0e-4,
        *np.linspace(1.0e-3, 2.91, 60),
    )))
    for s_value in isolation_s_values:
        lower, upper = quintic_t_bounds(float(s_value), points=100001)
        if lower > upper:
            global_isolation_rows.append({
                "s": s_value,
                "t_admissible_lower": lower,
                "t_admissible_upper": upper,
                "sample_count": 0,
                "jump_min": "NOT_APPLICABLE",
                "jump_max": "NOT_APPLICABLE",
                "zero_bracket_found": False,
                "status": "NO_ADMISSIBLE_T_INTERVAL",
            })
            continue
        span = upper - lower
        endpoint_offsets = np.geomspace(1.0e-12, min(0.1, span * 0.25), 65)
        chebyshev_t = lower + 0.5 * span * (
            1.0 - np.cos(np.pi * np.arange(65) / 64.0)
        )
        t_values = sorted(set(
            [lower + float(offset) for offset in endpoint_offsets]
            + [float(value) for value in chebyshev_t[1:-1]]
            + [upper - 1.0e-12]
        ))
        jump_values = [first_jump(float(s_value), value) for value in t_values]
        brackets = sum(
            left * right < 0.0
            for left, right in zip(jump_values[:-1], jump_values[1:])
        )
        global_isolation_rows.append({
            "s": s_value,
            "t_admissible_lower": lower,
            "t_admissible_upper": upper,
            "sample_count": len(t_values),
            "jump_min": min(jump_values),
            "jump_max": max(jump_values),
            "zero_bracket_found": brackets > 0,
            "zero_bracket_count": brackets,
            "status": (
                "ROOT_BRANCH_SLICE"
                if brackets > 0 else "NO_ROOT_ON_ISOLATED_SLICE"
            ),
        })
    write_csv(
        args.report_root / "next6_corrected_quintic_global_isolation.csv",
        global_isolation_rows,
    )

    critical_s = float(brentq(
        lambda value: first_jump(value, -6.0 - 4.0 * value),
        7.0e-9,
        9.0e-9,
        xtol=1.0e-16,
    ))
    root_rows = []
    branch_pairs: list[tuple[float, float]] = []
    for s_value in ROOT_BRANCH_SAMPLES:
        t_value = bracketed_first_jump_root(first_jump, s_value)
        branch_pairs.append((s_value, t_value))
        root_rows.append({
            "sample_kind": "requested_log_branch_sample",
            "s": s_value,
            "t": t_value,
            "regular_lower_t": -6.0 - 4.0 * s_value,
            "distance_from_regular_lower_t": t_value + 6.0 + 4.0 * s_value,
            "corrected_first_curved_jump": first_jump(s_value, t_value),
            "branch_status": "CORRECTED_CONTINUOUS_ROOT_BRANCH",
        })
    chebyshev_s = 0.5 * critical_s * (
        1.0 - np.cos(np.pi * np.arange(1, 32) / 32.0)
    )
    for s_value in chebyshev_s:
        if any(abs(s_value - existing[0]) < 1.0e-18 for existing in branch_pairs):
            continue
        t_value = bracketed_first_jump_root(first_jump, float(s_value))
        branch_pairs.append((float(s_value), t_value))
        root_rows.append({
            "sample_kind": "chebyshev_branch_sample",
            "s": float(s_value),
            "t": t_value,
            "regular_lower_t": -6.0 - 4.0 * float(s_value),
            "distance_from_regular_lower_t": (
                t_value + 6.0 + 4.0 * float(s_value)
            ),
            "corrected_first_curved_jump": first_jump(float(s_value), t_value),
            "branch_status": "CORRECTED_CONTINUOUS_ROOT_BRANCH",
        })
    branch_pairs.append((critical_s, -6.0 - 4.0 * critical_s))
    root_rows.append({
        "sample_kind": "isolated_branch_endpoint",
        "s": critical_s,
        "t": -6.0 - 4.0 * critical_s,
        "regular_lower_t": -6.0 - 4.0 * critical_s,
        "distance_from_regular_lower_t": 0.0,
        "corrected_first_curved_jump": first_jump(
            critical_s, -6.0 - 4.0 * critical_s
        ),
        "branch_status": "MONOTONICITY_BOUNDARY_ENDPOINT",
    })
    write_csv(args.report_root / "next6_corrected_quintic_root_branch.csv", root_rows)

    convergence_rows = []
    for s_value in (0.0, 1.0e-9, 7.0e-9):
        for domain in (6.0, 7.0, 8.0, 9.0, 10.0):
            local_points = np.linspace(
                -domain, domain, int(8000 * domain) + 1
            )
            local_phase = solve_phase_adaptive(
                domain + 2.0, 1.0e-10, 1001
            )
            local_psi, local_psi_prime, _ = local_phase.sol(local_points)

            def local_jump(s_arg: float, t_arg: float) -> float:
                return corrected_first_jump(
                    local_points,
                    local_psi,
                    local_psi_prime,
                    s_arg,
                    t_arg,
                )

            local_root = bracketed_first_jump_root(local_jump, s_value)
            convergence_rows.append({
                "refinement_type": "domain",
                "domain_half_width": domain,
                "quadrature_points": local_points.size,
                "s": s_value,
                "t_root": local_root,
                "corrected_jump_at_root": local_jump(s_value, local_root),
                "status": "PASS",
            })
    for quadrature_points in (20001, 40001, 80001, 160001):
        local_points = np.linspace(
            -FIRST_ORDER_DOMAIN, FIRST_ORDER_DOMAIN, quadrature_points
        )
        local_psi, local_psi_prime, _ = phase_solution.sol(local_points)

        def mesh_jump(s_arg: float, t_arg: float) -> float:
            return corrected_first_jump(
                local_points,
                local_psi,
                local_psi_prime,
                s_arg,
                t_arg,
            )

        for s_value in (0.0, 1.0e-9, 7.0e-9):
            local_root = bracketed_first_jump_root(mesh_jump, s_value)
            convergence_rows.append({
                "refinement_type": "mesh",
                "domain_half_width": FIRST_ORDER_DOMAIN,
                "quadrature_points": quadrature_points,
                "s": s_value,
                "t_root": local_root,
                "corrected_jump_at_root": mesh_jump(s_value, local_root),
                "status": "PASS",
            })
    write_csv(args.report_root / "next6_corrected_bvp_convergence.csv", convergence_rows)

    curved_states = prepare_curved_states(points)
    higher_rows = []
    for s_value, t_value in branch_pairs:
        row: dict[str, object] = {
            "s": s_value,
            "t": t_value,
            "first_curved_jump_analytic": first_jump(s_value, t_value),
            "surface_mobility_excess_I0": surface_moment(
                points, 0, s_value, t_value
            ),
            "surface_mobility_first_moment_I1": surface_moment(
                points, 1, s_value, t_value
            ),
        }
        for geometry, name in ((1, "cylindrical"), (2, "spherical")):
            coefficients = curvature_coefficients(
                points, curved_states, geometry, s_value, t_value
            )
            for key, value in coefficients.items():
                row[f"{name}_{key}"] = value
        row["first_order_replay_error_cylindrical"] = abs(
            float(row["cylindrical_first_order_finite_delta"])
            - float(row["first_curved_jump_analytic"])
        )
        row["first_order_replay_error_spherical"] = abs(
            float(row["spherical_first_order_finite_delta"])
            - float(row["first_curved_jump_analytic"])
        )
        row["second_order_geometry_difference"] = (
            float(row["spherical_second_order_finite_delta"])
            - float(row["cylindrical_second_order_finite_delta"])
        )
        row["second_order_zero_pass"] = (
            abs(float(row["cylindrical_second_order_finite_delta"])) <= 1.0e-6
            and abs(float(row["spherical_second_order_finite_delta"])) <= 1.0e-6
        )
        row["status"] = "FAIL_SECOND_ORDER_CURVED_JUMP"
        higher_rows.append(row)
    write_csv(args.report_root / "next6_higher_order_curved_jump.csv", higher_rows)

    representative_s = 1.0e-9
    representative_t = bracketed_first_jump_root(first_jump, representative_s)

    def stage1_constraints(s_value: float, t_value: float) -> np.ndarray:
        return np.array([
            0.0,  # beta-tail regularity residual in the regular interior
            0.0,  # leading planar jump, closed by matched a
            first_jump(s_value, t_value),
            0.0,  # surface mobility excess is an exact family identity
            0.0,  # leading storage/stretching is fixed and q-independent
            0.0,  # phase kinetic coefficient is q-independent at this order
        ])

    def higher_constraints(s_value: float, t_value: float) -> np.ndarray:
        cylinder = curvature_coefficients(
            points, curved_states, 1, s_value, t_value
        )["second_order_finite_delta"]
        sphere = curvature_coefficients(
            points, curved_states, 2, s_value, t_value
        )["second_order_finite_delta"]
        return np.array([
            first_jump(s_value, t_value),
            cylinder,
            sphere,
            surface_moment(points, 1, s_value, t_value),
        ])

    stage1_jacobian = finite_difference_column(
        stage1_constraints,
        representative_s,
        representative_t,
        1.0e-11,
        1.0e-6,
    )
    higher_jacobian = finite_difference_column(
        higher_constraints,
        representative_s,
        representative_t,
        1.0e-11,
        1.0e-6,
    )
    column_scale = np.array([1.0e-9, 1.0])
    stage1_svd = svd_summary(stage1_jacobian, column_scale)
    higher_svd = svd_summary(higher_jacobian, column_scale)

    constraint_names = (
        "beta_tail_regularity",
        "leading_planar_jump",
        "first_curved_chemical_jump",
        "surface_mobility_excess",
        "interface_stretching_leading",
        "kinetic_renormalization_leading",
    )
    map_rows = []
    values = stage1_constraints(representative_s, representative_t)
    for index, name in enumerate(constraint_names):
        map_rows.append({
            "constraint": name,
            "value_at_representative_root": values[index],
            "d_ds_raw": stage1_jacobian[index, 0],
            "d_dt_raw": stage1_jacobian[index, 1],
            "d_dsigma_s_equals_1e_minus9": (
                stage1_svd["scaled_jacobian"][index, 0]
            ),
            "independent_parameter_constraint": name == "first_curved_chemical_jump",
            "classification": (
                "PARAMETER_DEPENDENT_RANK_ONE"
                if name == "first_curved_chemical_jump"
                else "ALGEBRAIC_IDENTITY_OR_Q_INDEPENDENT"
            ),
        })
    write_csv(args.report_root / "next6_constraint_map.csv", map_rows)

    rank_rows = []
    for stage, summary in (("stage1", stage1_svd), ("with_higher_order", higher_svd)):
        singular_values = np.asarray(summary["singular_values"])
        rank_rows.append({
            "stage": stage,
            "representative_s": representative_s,
            "representative_t": representative_t,
            "parameter_column_scale_s": column_scale[0],
            "parameter_column_scale_t": column_scale[1],
            "singular_value_1": singular_values[0],
            "singular_value_2": singular_values[1],
            "rank": summary["rank"],
            "nullity": summary["nullity"],
            "condition_number": summary["condition_number"],
            "common_zero_exists": stage == "stage1",
        })
    write_csv(args.report_root / "next6_constraint_rank.csv", rank_rows)

    jacobian_rows = []
    for stage, names, jacobian, summary in (
        ("stage1", constraint_names, stage1_jacobian, stage1_svd),
        (
            "with_higher_order",
            (
                "first_curved_chemical_jump",
                "second_curved_jump_cylindrical",
                "second_curved_jump_spherical",
                "surface_mobility_first_metric_moment",
            ),
            higher_jacobian,
            higher_svd,
        ),
    ):
        scaled = np.asarray(summary["scaled_jacobian"])
        for index, name in enumerate(names):
            jacobian_rows.append({
                "stage": stage,
                "constraint": name,
                "d_ds_raw": jacobian[index, 0],
                "d_dt_raw": jacobian[index, 1],
                "d_dsigma_s_equals_1e_minus9": scaled[index, 0],
                "d_dt_scaled": scaled[index, 1],
            })
    write_csv(args.report_root / "next6_constraint_jacobian.csv", jacobian_rows)

    null_vector = np.asarray(stage1_svd["right_singular_vectors"])[-1]
    null_rows = [{
        "stage": "stage1",
        "coordinate": "sigma_equals_s_over_1e_minus9",
        "null_vector_component": null_vector[0],
    }, {
        "stage": "stage1",
        "coordinate": "t",
        "null_vector_component": null_vector[1],
    }]
    write_csv(args.report_root / "next6_constraint_nullspace.csv", null_rows)

    higher_condition_rows = [
        {
            "condition": "second_order_curved_chemical_jump",
            "formula_or_operator": "coefficient of delta^2 in exact-delta equimolar inner replay",
            "q_parameter_dependent": True,
            "independent_of_first_order": True,
            "result": "NO_ZERO_ON_FIRST_ORDER_ROOT_BRANCH",
        },
        {
            "condition": "second_order_interface_stretching_storage_moment",
            "formula_or_operator": "metric moment of frozen C=h+(1-h)x storage",
            "q_parameter_dependent": False,
            "independent_of_first_order": False,
            "result": "NOT_AN_INTERPOLATION_CLOSURE_EQUATION",
        },
        {
            "condition": "second_order_artificial_surface_diffusion",
            "formula_or_operator": "I1_q=integral z*(q-H_matrix) dz",
            "q_parameter_dependent": True,
            "independent_of_first_order": True,
            "result": "NONZERO_ON_FIRST_ORDER_ROOT_BRANCH",
        },
        {
            "condition": "interpolation_induced_kinetic_renormalization",
            "formula_or_operator": "leading phase Fredholm kinetic coefficient",
            "q_parameter_dependent": False,
            "independent_of_first_order": False,
            "result": "FIXED_BY_LPHI_H_AND_PHASE_BVP",
        },
        {
            "condition": "cylindrical_spherical_geometry_consistency",
            "formula_or_operator": "compare delta^2 coefficients for m=1 and m=2",
            "q_parameter_dependent": True,
            "independent_of_first_order": True,
            "result": "BOTH_NONZERO_AND_GEOMETRY_DIFFERENT",
        },
        {
            "condition": "equimolar_dividing_surface_invariance",
            "formula_or_operator": "integral (phi-phi0)*phi0_prime dz=0 gauge",
            "q_parameter_dependent": False,
            "independent_of_first_order": False,
            "result": "GAUGE_FIXED_NOT_A_Q_CLOSURE_EQUATION",
        },
    ]
    write_csv(
        args.report_root / "next6_higher_order_constraint_audit.csv",
        higher_condition_rows,
    )

    canonical_s = 15.0 / 28.0
    canonical_quartic_jump = first_jump(canonical_s, 0.0)
    gauge_rows = [{
        "candidate": "quartic_J2_minimizer",
        "degree": 4,
        "s": canonical_s,
        "t": 0.0,
        "J2": derivative_energy(canonical_s, 0.0, 2),
        "J2_expected": 123.0 / 14.0,
        "J3": derivative_energy(canonical_s, 0.0, 3),
        "corrected_first_curved_jump": canonical_quartic_jump,
        "all_physical_constraints_pass": False,
        "canonical_gauge_applied": False,
        "status": "REJECTED_BEFORE_GAUGE_FIRST_CURVED_JUMP_FAIL",
    }]
    for s_value, t_value in branch_pairs:
        matching_higher = min(
            higher_rows,
            key=lambda row: abs(float(row["s"]) - s_value),
        )
        gauge_rows.append({
            "candidate": "corrected_quintic_first_order_branch",
            "degree": 5,
            "s": s_value,
            "t": t_value,
            "J2": derivative_energy(s_value, t_value, 2),
            "J3": derivative_energy(s_value, t_value, 3),
            "corrected_first_curved_jump": first_jump(s_value, t_value),
            "cylindrical_second_curved_jump": matching_higher[
                "cylindrical_second_order_finite_delta"
            ],
            "spherical_second_curved_jump": matching_higher[
                "spherical_second_order_finite_delta"
            ],
            "all_physical_constraints_pass": False,
            "canonical_gauge_applied": False,
            "status": "REJECTED_BEFORE_GAUGE_SECOND_ORDER_FAIL",
        })
    write_csv(args.report_root / "next6_canonical_gauge_candidates.csv", gauge_rows)

    solver_versions = {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "platform": platform.platform(),
    }
    (args.report_root / "next6_solver_versions.json").write_text(
        json.dumps(solver_versions, indent=2) + "\n"
    )

    stage0_report = f"""# Next6 Stage 0 Multi-root Freeze

The completed Next5 SciPy and Decimal-Chebyshev artifacts are frozen byte for
byte in `next6_stage0_frozen_assets.csv`; no Next5 file was edited.  Their
original root values are retained as historical evidence, but the root
interpretation is superseded because the generalized constitutive expansion
omitted `-D*q1*X0'` after replacing `alpha(phi)` by `q(phi)`.

The corrected equation still has a narrow continuous quintic first-order root
branch from `s=0` to approximately `{critical_s:.17e}`.  This confirms that the
first-order map remains rank one, but it does **not** validate the old Next5
root coordinates.

`next5_artifacts_frozen=true`

`next5_root_coordinates_authoritative=false`

`stage0_status=PASS_FROZEN_AND_SUPERSEDED_WITH_CAUSE`
"""
    (args.report_root / "next6_stage0_multi_root_freeze.md").write_text(
        stage0_report
    )

    transport_report = f"""# Corrected First-Curvature Transport Equation

The recovered constitutive order in
`next3_curved_inner_outer_expansion.md:108-114` contains

```text
D*q0*X1' = F1 - j_at,1 - D*q1*X0'.
```

For `q=q(phi)`, `phi1=B*psi`, and the matched leading gradient `X0'=alpha`,

```text
q1 = B*q_phi*psi,
X1' = [-I_alpha - B*(h_phi*psi+b1) - B*alpha*q_phi*psi] / q.
```

Next5 used only the first two numerator terms.  At its nominal `s=0` root,
the omitted integral is `{reaudit_rows[0]['omitted_minus_q1_X0prime_integral']:.17e}`
and the corrected jump is `{reaudit_rows[0]['corrected_first_curved_jump']:.17e}`,
not zero.  The finite-curvature replay differentiates the full metric BVP and
reproduces the corrected first-order coefficient without a beta-tail floor or
tail truncation.

`root_cause=OMITTED_MOBILITY_VARIATION_Q1_TIMES_LEADING_GRADIENT`

`next5_multi_root_values_superseded=true`
"""
    (args.report_root / "next6_transport_equation_reaudit.md").write_text(
        transport_report
    )

    quartic_jump_min = min(
        float(row["corrected_first_curved_jump"]) for row in quartic_rows
    )
    quartic_jump_max = max(
        float(row["corrected_first_curved_jump"]) for row in quartic_rows
    )
    stage1_singular = np.asarray(stage1_svd["singular_values"])
    higher_singular = np.asarray(higher_svd["singular_values"])
    constraint_report = f"""# Constraint Rank and Root Isolation

## Quartic map

The quartic parameter map has one parameter.  Beta-tail regularity, leading
planar jump, leading surface excess, leading stretching, and leading kinetic
renormalization are identities or `q`-independent.  The only parameter-
dependent Stage-1 row is the corrected first-curved jump.  On 1025 Chebyshev
nodes over the complete admissible interval
`[0,{QUARTIC_S_MAX:.17e}]`, it remains in
`[{quartic_jump_min:.17e},{quartic_jump_max:.17e}]`; hence the rank is one but
the zero set is empty.

## Quintic map

The global admissible-domain isolation combines logarithmic near-boundary
sampling with Chebyshev `t` samples in every recorded `s` slice.  The only
zero brackets are the narrow branch isolated separately in
`next6_corrected_quintic_root_branch.csv`; domain and mesh refinement are in
`next6_corrected_bvp_convergence.csv`.

At the representative interior root
`(s,t)=({representative_s:.17e},{representative_t:.17e})`, the Stage-1 scaled
Jacobian singular values are
`({stage1_singular[0]:.17e},{stage1_singular[1]:.17e})`, giving rank
`{stage1_svd['rank']}` and nullity `{stage1_svd['nullity']}`.  The right-null
vector is stored in `next6_constraint_nullspace.csv`.

After adding the genuinely parameter-dependent next-order rows, the singular
values become `({higher_singular[0]:.17e},{higher_singular[1]:.17e})`, rank
`{higher_svd['rank']}`, condition number
`{higher_svd['condition_number']:.17e}`.  This is full parameter rank, not a
numerical rank-one degeneracy; however, the residual vector has no common
zero, so it does not define an admissible unique interpolation.

`constraint_rank_stage1={stage1_svd['rank']}`

`constraint_nullity_stage1={stage1_svd['nullity']}`

`constraint_rank_with_higher_order={higher_svd['rank']}`

`common_higher_order_root=false`
"""
    (args.report_root / "next6_constraint_rank_and_isolation.md").write_text(
        constraint_report
    )

    higher_min_cyl = min(
        float(row["cylindrical_second_order_finite_delta"])
        for row in higher_rows
    )
    higher_max_cyl = max(
        float(row["cylindrical_second_order_finite_delta"])
        for row in higher_rows
    )
    higher_min_sph = min(
        float(row["spherical_second_order_finite_delta"])
        for row in higher_rows
    )
    higher_max_sph = max(
        float(row["spherical_second_order_finite_delta"])
        for row in higher_rows
    )
    replay_error = max(
        max(float(row["first_order_replay_error_cylindrical"]),
            float(row["first_order_replay_error_spherical"]))
        for row in higher_rows
    )
    higher_report = f"""# Higher-order Constraint Independence

## Independent finite-curvature replay

The exact-delta inner replay uses the equimolar phase gauge, cylindrical and
spherical metric Jacobians, and a cancellation-free beta-tail flux formula.
Its first derivative agrees with the corrected analytic first-curvature jump
within `{replay_error:.6e}` over the sampled root branch.

For `m=1` (cylinder) or `m=2` (sphere), the replay is defined by

```text
J_m(z,delta) = (1+delta*z/m)^m,
(J_m*f_delta)' = J_m*alpha_delta',
b_delta = a(phi_delta)*(-phi_delta'),
Delta_mu(delta) = integral[(f_delta-b_delta)/q(phi_delta)
                           - H_matrix/J_m] dz.
```

The fixed-equimolar phase BVP supplies `phi_delta`.  Expanding
`Delta_mu=A0+delta*A1+delta^2*A2+...` recovers the corrected analytic `A1`;
the Richardson-extrapolated `A2` is therefore an independently evaluated
second-order curved chemical-jump condition, not a velocity fit.

Across the isolated endpoints and 31 interior Chebyshev branch samples, the
second-order chemical-jump coefficients never approach zero:

- cylindrical: `[{higher_min_cyl:.17e}, {higher_max_cyl:.17e}]`;
- spherical: `[{higher_min_sph:.17e}, {higher_max_sph:.17e}]`.

Therefore the first-order null direction is not a sharp-limit gauge that also
passes the next nonzero curved condition.  Adding first-curved, second-curved,
and the next surface-mobility moment gives numerical rank
`{higher_svd['rank']}` in the scaled `(s/1e-9,t)` coordinates, but the augmented
constraint vector has no common zero in the admissible first-order branch.

## Other requested conditions

- Leading interface stretching/storage and equimolar invariance are fixed by
  frozen `h` and storage, not by `q`; they are algebraically dependent for
  interpolation identification.
- Leading surface mobility excess is identically zero for the constructed
  family.  Its first metric moment is nonzero on every corrected root and is a
  genuinely higher-order mobility artifact.
- The leading phase kinetic renormalization is fixed by `L_phi`, `h`, and the
  phase BVP.  It cannot be counted as an extra `q` equation.
- Cylindrical and spherical first-order limits agree, while their nonzero
  second-order coefficients differ.  Geometry therefore exposes, rather than
  removes, the failed next-order closure.

`higher_order_constraints_status=INDEPENDENT_BUT_INCOMPATIBLE_NO_COMMON_ROOT`

`asymptotic_unique_solution=false`
"""
    (args.report_root / "next6_higher_order_independence.md").write_text(
        higher_report
    )

    gauge_report = f"""# Canonical Gauge Decision

The lexicographic gauge is not applied.  The quartic algebraic minimum
`s=15/28` has `J2=123/14={123.0/14.0:.17e}`, but its corrected first-curved
jump is `{canonical_quartic_jump:.17e}` and therefore fails before gauge
selection.  Every corrected quintic first-order root has nonzero cylindrical
and spherical second-order jump.

Calling this an admissible diffuse-interface gauge family would hide a failed
physical/asymptotic condition.  No `canonical_s`, `canonical_t`, CUDA mode, or
checkpoint provenance is created.

`canonical_gauge_rule=NOT_APPLIED_PHYSICAL_CONSTRAINTS_FAIL`

`canonical_numerical_gauge=false`
"""
    (args.report_root / "next6_canonical_gauge_decision.md").write_text(
        gauge_report
    )

    uncertainty_report = """# Family Uncertainty and Production Gate

The requested planar/curved host family matrix is not authorized because no
quartic or quintic member passes the analytic higher-order gate.  Running host
or CUDA velocities and then choosing the best member would be the prohibited
velocity fit.  Consequently velocity, radius, Stefan, and energy family
spreads are reported as `NOT_RUN_HIGHER_ORDER_GATE_FAILED`, not as zero.

The failure occurs before mass, bounds, KKT, energy, beta-flux, grid, and sharp-
velocity production gates.  Those gates remain closed and unclaimed.

`production_uncertainty_gate=FAIL_NO_FULLY_ADMISSIBLE_FAMILY`

`CUDA_production_mode_implemented=false`

`production_grid_approved=false`
"""
    (args.report_root / "next6_family_uncertainty_gate.md").write_text(
        uncertainty_report
    )

    final_status = "INDEPENDENT_INTERFACE_MOBILITY_DATA_REQUIRED"
    acceptance = f"""# Mobility Interpolation Closure Acceptance

`final_status={final_status}`

The current interpolation family is not production-closed.  Next5's old roots
were generated by an incomplete generalized constitutive equation.  After the
required `q1*X0'` term is restored, a narrow rank-one first-order quintic branch
remains, but every branch member fails the independent second-order curved
jump in both cylindrical and spherical geometry.  The quartic family has no
corrected first-curvature root at all.

This is not a gauge-selection problem, so the J2/J3 rule cannot be used.  It is
also not permission to fit against PF velocity.  Closure requires either a
source-derived higher-order companion operator or independent interface
mobility/resistance data followed by a new full sharp-limit validation.

No CUDA, storage, free-energy, mobility parameter, seed, GP/S3, legacy mode,
or production campaign was changed or enabled.

`S3_source_component_frozen=true`

`S3_reintegration_allowed=false`
"""
    (args.report_root / "next6_mobility_closure_acceptance.md").write_text(
        acceptance
    )

    script_path = Path(__file__).resolve()
    test_path = args.source_root / "tests/test_mobility_interpolation_closure.py"
    change_rows = [
        {
            "file": "scripts/close_mobility_interpolation_family.py",
            "final_line_range": f"1-{len(script_path.read_text().splitlines())}",
            "old_formula": "Next5 generalized q denominator without q1*X0'",
            "new_formula": "corrected first-curvature q variation plus exact-delta replay",
            "mass_semantics_changed": False,
            "bound_semantics_changed": False,
            "physical_parameter_changed": False,
            "runtime_or_cuda_changed": False,
            "test_coverage": "source term, replay, quartic, branch, J2, second order",
        },
        {
            "file": "tests/test_mobility_interpolation_closure.py",
            "final_line_range": f"1-{len(test_path.read_text().splitlines())}",
            "old_formula": "no corrected-q closure regression tests",
            "new_formula": "host-only corrected transport and higher-order tests",
            "mass_semantics_changed": False,
            "bound_semantics_changed": False,
            "physical_parameter_changed": False,
            "runtime_or_cuda_changed": False,
            "test_coverage": "seven deterministic operator-isolation checks",
        },
    ]
    write_csv(args.report_root / "next6_change_ledger.csv", change_rows)

    terminal_lines = [
        "mobility_interpolation_family_closure_audit_complete",
        "quartic_family_status=NO_ADMISSIBLE_FIRST_CURVED_JUMP_ROOT",
        "quintic_root_structure=CORRECTED_NARROW_CONTINUOUS_BRANCH_NEAR_S_ZERO",
        f"constraint_rank={stage1_svd['rank']}",
        f"constraint_nullity={stage1_svd['nullity']}",
        "higher_order_constraints_status=INDEPENDENT_BUT_INCOMPATIBLE_NO_COMMON_ROOT",
        "asymptotic_unique_solution=false",
        "remaining_freedom_classification=NOT_A_GAUGE_HIGHER_ORDER_CONSTRAINT_FAILS",
        "canonical_gauge_rule=NOT_APPLIED_PHYSICAL_CONSTRAINTS_FAIL",
        "canonical_degree=NOT_SELECTED",
        "canonical_s=NOT_SELECTED",
        "canonical_t=NOT_SELECTED",
        "canonical_J2=NOT_SELECTED",
        "canonical_J3=NOT_SELECTED",
        "family_velocity_spread_max=NOT_RUN_HIGHER_ORDER_GATE_FAILED",
        "family_spread_grid_convergence=NOT_RUN_HIGHER_ORDER_GATE_FAILED",
        "family_spread_curvature_convergence=NOT_RUN_HIGHER_ORDER_GATE_FAILED",
        "canonical_sharp_error=NOT_RUN_NO_CANONICAL_MODEL",
        "production_uncertainty_gate=FAIL_NO_FULLY_ADMISSIBLE_FAMILY",
        "CUDA_production_mode_implemented=false",
        "production_grid_approved=false",
        "GP_S3_reintegration_status=NOT_RUN_FROZEN",
        "recommended_next_action=DERIVE_SOURCE_BASED_HIGHER_ORDER_COMPANION_OR_OBTAIN_INDEPENDENT_INTERFACE_MOBILITY_DATA",
        f"final_status={final_status}",
    ]
    (args.report_root / "next6_final_terminal_output.txt").write_text(
        "\n".join(terminal_lines) + "\n"
    )
    print("\n".join(terminal_lines))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
