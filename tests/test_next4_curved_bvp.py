import unittest

import numpy as np

from scripts.solve_next4_curved_bvp import (
    A_PHI,
    K_INNER,
    TAIL_METRIC_MOMENT,
    alpha_integral_ratio,
    correction_shape_variation,
    forcing,
    interpolate_chebyshev,
    phase_values,
    solve_phase_adaptive,
    solve_phase_decimal_chebyshev,
)


class Next4CurvedBvpTests(unittest.TestCase):
    def test_phase_forcing_satisfies_fredholm_condition(self):
        z = np.linspace(-12.0, 12.0, 200001)
        _, _, phi_prime, _, _, _, _ = phase_values(z)
        inner_product = np.trapezoid(phi_prime * forcing(z), z)
        self.assertLess(abs(inner_product), 2.0e-13)
        self.assertAlmostEqual(
            np.trapezoid(phi_prime * phi_prime, z), A_PHI, places=13
        )

    def test_cubic_beta_capacity_has_one_twelfth_integral_ratio(self):
        z = np.linspace(-8.0, -4.0, 100001)
        ratios = alpha_integral_ratio(z)
        self.assertLess(abs(ratios[-1] - TAIL_METRIC_MOMENT), 5.0e-8)

    def test_adaptive_phase_bvp_closes_gauge_and_tail_shape(self):
        solution = solve_phase_adaptive(8.0, 1.0e-9, 801)
        self.assertLess(abs(float(solution.p[0])), 1.0e-11)
        self.assertLess(abs(float(solution.y[2, -1])), 1.0e-11)
        probes = np.array([-6.0, -5.0, -4.0])
        psi, psi_prime, _ = solution.sol(probes)
        _, ratio = correction_shape_variation(probes, psi, psi_prime)
        self.assertLess(np.max(np.abs(ratio + 1.0)), 1.0e-5)

    def test_decimal_chebyshev_agrees_with_adaptive_phase_solution(self):
        adaptive = solve_phase_adaptive(6.0, 1.0e-9, 801)
        decimal = solve_phase_decimal_chebyshev(6.0, 128, 70)
        points = np.linspace(-4.0, 4.0, 1001)
        difference = np.max(np.abs(
            interpolate_chebyshev(decimal, points) - adaptive.sol(points)[0]
        ))
        self.assertLess(difference, 2.0e-5)

    def test_stationary_transport_tail_obstruction_is_one_twenty_fourth(self):
        self.assertAlmostEqual(
            K_INNER - TAIL_METRIC_MOMENT, 1.0 / 24.0, places=15
        )


if __name__ == "__main__":
    unittest.main()
