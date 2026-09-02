"""Shared metric, cell-measure, and raw face-Courant regressions."""

from __future__ import annotations

import unittest

import numpy as np

from kwn_mvp.population_metrics import (
    beta_fraction_from_m3,
    close_matrix_from_precipitates,
    metrics_from_discrete_measure,
    metrics_from_piecewise_constant_cells,
    positive_cell_quadrature,
)
from kwn_mvp.solver import KWNSolver, SolverConfig


def _config() -> dict[str, object]:
    return {
        "simulation": {
            "temperature_K": 653.15,
            "max_dt_s": 1.0e6,
            "min_dt_s": 1.0e-18,
            "size_cfl": 0.35,
            "accuracy_radius_cfl": 0.1,
            "population_measure": "cell_integrated",
            "rmax_outflow_relative_tolerance": 1.0,
        },
        "matrix": {
            "molar_volume_m3_mol": 1.0,
            "initial_xB": 0.25,
            "total_b_mol_m3": None,
            "inventory_tolerance_relative": 1.0e-12,
        },
        "radius_grid": {"minimum_m": 1.0, "maximum_m": 4.0, "bins": 2},
        "thermodynamics": {"mode": "approximate_dilute", "planar_reference_xB": 0.1},
        "populations": {
            "g": {
                "xB": 1.0,
                "molar_volume_m3_mol": 1.0,
                "diffusivity_m2_s": 0.0,
                "gamma_j_m2": 0.0,
                "xeq_infinity": 0.1,
                "initial": {"kind": "empty"},
                "nucleation": {"mode": "off"},
            },
            "beta": {
                "xB": 1.0,
                "molar_volume_m3_mol": 1.0,
                "diffusivity_m2_s": 1.0e-3,
                "gamma_j_m2": 0.0,
                "xeq_infinity": 0.1,
                "initial": {
                    "kind": "cell_integrated",
                    "radius_edges_m": [1.0, 2.0, 4.0],
                    "cell_number_density_m3": [1.0e-4, 2.0e-4],
                },
                "nucleation": {"mode": "off"},
            },
        },
    }


class PopulationMetricsTest(unittest.TestCase):
    """A cell measure and its deterministic quadrature share one contract."""

    def test_two_point_quadrature_recovers_piecewise_cell_moments(self) -> None:
        edges = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
        number = np.asarray([3.0, 5.0], dtype=np.float64)
        reference = metrics_from_piecewise_constant_cells(edges, number)
        radii, weights = positive_cell_quadrature(edges, number, 2)
        quadrature = metrics_from_discrete_measure(radii, weights)
        for key in ("M0_m3", "M1_m2", "M2_m", "M3_dimensionless"):
            self.assertAlmostEqual(getattr(quadrature, key), getattr(reference, key), places=13)
        self.assertNotAlmostEqual(reference.Rmean_cubed_m3, reference.mean_R3_m3, places=12)

    def test_one_point_quadrature_preserves_cell_number_and_cubic_moment(self) -> None:
        edges = np.asarray([1.0, 2.0, 4.0], dtype=np.float64)
        number = np.asarray([3.0, 5.0], dtype=np.float64)
        reference = metrics_from_piecewise_constant_cells(edges, number)
        radii, weights = positive_cell_quadrature(edges, number, 1)
        quadrature = metrics_from_discrete_measure(radii, weights)
        self.assertAlmostEqual(quadrature.M0_m3, reference.M0_m3, places=13)
        self.assertAlmostEqual(quadrature.M3_dimensionless, reference.M3_dimensionless, places=13)

    def test_shared_matrix_closure_reconstructs_total_inventory(self) -> None:
        closure = close_matrix_from_precipitates(
            total_b_mol_m3=0.25,
            matrix_molar_volume_m3_mol=1.0,
            precipitate_volume_fraction=beta_fraction_from_m3(0.01),
            precipitate_inventory_mol_m3=beta_fraction_from_m3(0.01),
        )
        self.assertLessEqual(abs(closure.residual_mol_m3), 1.0e-15)
        self.assertGreaterEqual(closure.matrix_xb, 0.0)
        self.assertLessEqual(closure.matrix_xb, 1.0)

    def test_cell_integrated_initialization_and_raw_courant_are_explicit(self) -> None:
        solver = KWNSolver(SolverConfig.from_mapping(_config()))
        beta = solver.population("beta")
        np.testing.assert_array_equal(
            beta.number_density_per_m4 * beta.grid.widths_m,
            np.asarray([1.0e-4, 2.0e-4]),
        )
        diagnostic = solver.advance_one()
        self.assertEqual(diagnostic.timestep_limiter, "accuracy_radius_cfl")
        self.assertLessEqual(diagnostic.radius_courant_max, 0.1 * (1.0 + 1.0e-12))
        self.assertGreaterEqual(diagnostic.radius_courant_face_index, 0)


if __name__ == "__main__":
    unittest.main()
