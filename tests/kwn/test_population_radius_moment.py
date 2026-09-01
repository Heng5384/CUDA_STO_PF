"""Fixed-pivot and diagnostic cell-integrated moment definitions."""

from __future__ import annotations

import unittest

import numpy as np

from kwn_mvp.populations import Population, PopulationParameters
from kwn_mvp.radius_grid import RadiusGrid


class PopulationRadiusMomentTest(unittest.TestCase):
    """The new audit method must not alter production fixed-pivot observables."""

    def setUp(self) -> None:
        self.grid = RadiusGrid(np.asarray([1.0, 2.0, 4.0], dtype=np.float64))
        parameters = PopulationParameters(
            name="beta",
            x_b=1.0,
            molar_volume_m3_mol=1.0,
            diffusivity_m2_s=0.0,
            gamma_j_m2=0.0,
            xeq_infinity=0.5,
        )
        self.population = Population(
            parameters=parameters,
            grid=self.grid,
            number_density_per_m4=np.asarray([3.0, 5.0], dtype=np.float64),
        )

    def test_fixed_pivot_matches_existing_observable_formulae(self) -> None:
        radii = self.grid.centres_m
        widths = self.grid.widths_m
        density = self.population.number_density_per_m4
        for order in range(4):
            expected = float(np.sum(density * widths * radii**order))
            self.assertEqual(
                self.population.radius_moment(order, quadrature="fixed_pivot"), expected
            )
        self.assertEqual(self.population.number_density_m3(), float(np.sum(density * widths)))
        self.assertAlmostEqual(
            self.population.volume_fraction(),
            (4.0 * np.pi / 3.0) * self.population.radius_moment(3),
            places=12,
        )

    def test_cell_integrated_reconstruction_uses_exact_edges(self) -> None:
        density = self.population.number_density_per_m4
        edges = self.grid.edges_m
        for order in range(4):
            integral = (edges[1:] ** (order + 1) - edges[:-1] ** (order + 1)) / (
                order + 1
            )
            self.assertEqual(
                self.population.radius_moment(order, quadrature="cell_integrated"),
                float(np.sum(density * integral)),
            )
        self.assertEqual(
            self.population.radius_moment(0, quadrature="cell_integrated"),
            self.population.radius_moment(0, quadrature="fixed_pivot"),
        )

    def test_invalid_moment_request_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            self.population.radius_moment(-1)
        with self.assertRaises(ValueError):
            self.population.radius_moment(1.5)
        with self.assertRaises(ValueError):
            self.population.radius_moment(1, quadrature="unknown")


if __name__ == "__main__":
    unittest.main()
