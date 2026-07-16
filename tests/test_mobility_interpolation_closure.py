#!/usr/bin/env python3
"""Operator-isolation tests for the corrected matched-q closure audit."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.close_mobility_interpolation_family import (
    B_STATIONARY,
    corrected_first_jump,
    corrected_transport_rhs,
    curvature_coefficients,
    derivative_energy,
    finite_curvature_jump,
    prepare_curved_states,
)
from scripts.solve_next4_curved_bvp import solve_phase_adaptive
from scripts.solve_next5_matched_mobility_bvp import (
    q_and_q_u,
    stable_phase,
    transport_rhs as next5_transport_rhs,
)


POINTS = np.linspace(-8.0, 8.0, 40001)
PHASE = solve_phase_adaptive(10.0, 1.0e-10, 1001)
PSI, PSI_PRIME, _ = PHASE.sol(POINTS)


def test_corrected_term_is_exact_q1_x0_constitutive_term() -> None:
    s_value, t_value = 0.0, -5.999584318767566
    _, fields = corrected_transport_rhs(
        POINTS, PSI, PSI_PRIME, s_value, t_value
    )
    phi, u_value, _, _, alpha = stable_phase(POINTS)
    q_value, q_u = q_and_q_u(
        POINTS, phi, u_value, s_value, t_value
    )
    expected = B_STATIONARY * alpha * q_u * PSI / q_value
    assert np.max(np.abs(fields["omitted_q1_X0_term"] - expected)) < 1.0e-14


def test_next5_nominal_root_fails_corrected_jump() -> None:
    jump = corrected_first_jump(
        POINTS, PSI, PSI_PRIME, 0.0, -5.999584318767566
    )
    assert 3.0e-3 < jump < 3.3e-3


def test_quartic_j2_formula_and_minimizer() -> None:
    for s_value in (0.0, 0.25, 15.0 / 28.0, 1.0):
        expected = 4.0 / 5.0 * (
            14.0 * s_value**2 - 15.0 * s_value + 15.0
        )
        assert abs(derivative_energy(s_value, 0.0, 2) - expected) < 2.0e-13
    assert abs(derivative_energy(15.0 / 28.0, 0.0, 2) - 123.0 / 14.0) < 2.0e-13


def test_quartic_j2_minimizer_fails_first_curved_condition() -> None:
    jump = corrected_first_jump(
        POINTS, PSI, PSI_PRIME, 15.0 / 28.0, 0.0
    )
    assert jump < -9.0e-2


def test_finite_curvature_replay_recovers_corrected_first_derivative() -> None:
    states = prepare_curved_states(POINTS)
    s_value, t_value = 0.0, -5.99943857905832
    analytic = corrected_first_jump(
        POINTS, PSI, PSI_PRIME, s_value, t_value
    )
    coefficients = curvature_coefficients(
        POINTS, states, 1, s_value, t_value
    )
    assert abs(coefficients["first_order_finite_delta"] - analytic) < 2.0e-6


def test_second_order_curved_jump_does_not_close_at_first_order_root() -> None:
    states = prepare_curved_states(POINTS)
    s_value, t_value = 0.0, -5.99943857905832
    cylinder = curvature_coefficients(
        POINTS, states, 1, s_value, t_value
    )["second_order_finite_delta"]
    sphere = curvature_coefficients(
        POINTS, states, 2, s_value, t_value
    )["second_order_finite_delta"]
    assert 0.18 < cylinder < 0.22
    assert 0.18 < sphere < 0.22
    assert abs(cylinder - sphere) > 1.0e-3


def test_zero_curvature_replay_has_no_spurious_planar_jump() -> None:
    states = prepare_curved_states(POINTS)
    jump = finite_curvature_jump(
        POINTS, states[(1, 0.0)], 0.0, -5.99943857905832
    )
    assert math.isfinite(jump)
    assert abs(jump) < 1.0e-9


if __name__ == "__main__":
    tests = [value for name, value in globals().items() if name.startswith("test_")]
    for test in tests:
        test()
    print(f"mobility_interpolation_closure_tests_passed={len(tests)}")
