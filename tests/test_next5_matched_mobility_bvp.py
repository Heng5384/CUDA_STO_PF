from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.solve_next5_matched_mobility_bvp import (
    QUARTIC_S_MAX,
    alpha_integral_exact,
    audit_family_endpoints,
    matched_shape,
    polynomial_value,
    q_coefficients_phi,
    q_derivative_phi,
    quintic_t_bounds,
    stable_phase,
    surface_excess_numeric,
)


def test_quartic_endpoint_identities_and_admissibility() -> None:
    grid = np.linspace(0.0, 1.0, 200001)
    for s_value in (0.0, 0.5, 1.0, QUARTIC_S_MAX):
        endpoint = audit_family_endpoints(s_value, 0.0)
        assert endpoint["q_0"] == 1.0
        assert abs(endpoint["q_1"]) < 2.0e-14
        assert endpoint["qprime_0"] == 0.0
        assert abs(endpoint["qprime_1"] + s_value) < 3.0e-14
        q_value = polynomial_value(q_coefficients_phi(s_value), grid)
        assert np.min(q_value) >= -2.0e-14
        assert np.max(q_value) <= 1.0 + 2.0e-14
        assert np.max(q_derivative_phi(grid, s_value)) <= 3.0e-14


def test_quartic_upper_bound_is_sharp() -> None:
    grid = np.linspace(0.0, 1.0, 500001)
    derivative = q_derivative_phi(grid, QUARTIC_S_MAX * (1.0 + 1.0e-5))
    assert np.max(derivative) > 1.0e-7


def test_surface_excess_zero_for_both_families() -> None:
    for s_value, t_value in (
        (0.0, 0.0),
        (0.5, 0.0),
        (1.0, 0.0),
        (QUARTIC_S_MAX, 0.0),
        (0.0, -5.5),
        (1.0e-9, -5.999),
        (0.5, 10.0),
    ):
        assert abs(surface_excess_numeric(s_value, t_value)) < 2.0e-11


def test_matched_shape_has_zero_endpoint_limits() -> None:
    points = np.array([-20.0, -16.0, -12.0, 12.0, 16.0, 20.0])
    for s_value, t_value in ((0.0, 0.0), (1.0, 0.0), (0.0, -5.5)):
        a_value, a_z, q_value = matched_shape(points, s_value, t_value)
        assert np.all(np.isfinite(a_value))
        assert np.all(np.isfinite(a_z))
        assert np.all(np.isfinite(q_value))
        assert max(abs(a_value[0]), abs(a_value[-1])) < 1.0e-24


def test_alpha_integral_has_correct_tail_limits() -> None:
    points = np.array([-12.0, -8.0, 0.0, 8.0, 12.0])
    phi, u_value, _, _, alpha = stable_phase(points)
    integral = alpha_integral_exact(u_value, phi)
    assert abs(integral[0] / alpha[0] - 1.0 / 12.0) < 2.0e-14
    assert abs(integral[-1] - points[-1]) < 2.0e-14
    assert np.all(np.diff(integral) > 0.0)


def test_quintic_admissibility_bounds_at_s_zero() -> None:
    lower, upper = quintic_t_bounds(0.0, points=300001)
    assert abs(lower + 6.0) < 1.0e-12
    assert abs(upper - 24.0) < 1.0e-9
    grid = np.linspace(0.0, 1.0, 200001)
    for t_value in (-6.0, -5.5, 0.0, 24.0):
        derivative = q_derivative_phi(grid, 0.0, t_value)
        assert np.max(derivative) <= 2.0e-13


def test_old_q_alpha_is_quintic_boundary_point() -> None:
    points = np.linspace(-5.0, 5.0, 1001)
    phi, _, _, _, alpha = stable_phase(points)
    q_value = polynomial_value(q_coefficients_phi(0.0, -6.0), phi)
    assert np.max(np.abs(q_value - alpha)) < 2.0e-14
    assert math.isclose((1.0 / 8.0 - 1.0 / 12.0), 1.0 / 24.0)


if __name__ == "__main__":
    tests = [value for name, value in sorted(globals().items())
             if name.startswith("test_") and callable(value)]
    for test in tests:
        test()
    print(f"next5_matched_mobility_tests_passed={len(tests)}")
