"""Regression checks for the declared active-population CFL authority."""

from __future__ import annotations

import unittest

import numpy as np

from kwn_mvp.radius_grid import RadiusGrid
from kwn_mvp.solver import SolverConfig

from tests.kwn.test_lower_boundary_contract import (
    _ConstantVelocityEulerian,
    _constant_eulerian,
    _first_cell_initial_definition,
    _mapping,
    _smooth_initial_definition,
)


class _BoundarySpikeEulerian(_ConstantVelocityEulerian):
    """Test-only state with a fast physical Rmin face and slow interior."""

    def __init__(self, *args, boundary_velocity_m_s: float, **kwargs) -> None:
        self.boundary_velocity_m_s = float(boundary_velocity_m_s)
        super().__init__(*args, **kwargs)

    def lower_boundary_growth_velocity(self, population) -> float:
        item = self.population(population) if isinstance(population, str) else population
        return self.boundary_velocity_m_s if item.parameters.name == "beta" else 0.0


class ActiveCFLContractTests(unittest.TestCase):
    """The raw physical face and active-population accuracy policies stay distinct."""

    def test_empty_lower_tail_does_not_impose_raw_boundary_cfl(self) -> None:
        """A remote raw face is telemetry until the declared tail can reach Rmin."""

        solver = _constant_eulerian(
            bins=96,
            initial=_smooth_initial_definition(bins=96),
            dt_fraction_of_smallest_cell=0.1,
        )
        faces = solver.face_velocities("beta", solver.growth_rates()["beta"])
        audit = solver.active_cfl_diagnostics("beta", face_velocity_m_s=faces, dt_s=1.0)
        self.assertGreater(audit.global_max_rate_s_inv, audit.population_active_rate_s_inv)
        self.assertFalse(audit.boundary_face_is_active)
        self.assertEqual(audit.boundary_active_rate_s_inv, 0.0)
        self.assertGreater(audit.boundary_candidate_rate_s_inv, 0.0)

    def test_first_cell_tail_activates_physical_boundary(self) -> None:
        """Once the lower-tail population is in cell zero, Rmin joins the active set."""

        solver = _constant_eulerian(
            bins=64,
            initial=_first_cell_initial_definition(bins=64, number_m3=1.0e18),
            dt_fraction_of_smallest_cell=0.1,
        )
        faces = solver.face_velocities("beta", solver.growth_rates()["beta"])
        audit = solver.active_cfl_diagnostics("beta", face_velocity_m_s=faces, dt_s=1.0)
        self.assertTrue(audit.boundary_face_is_active)
        self.assertEqual(
            audit.boundary_active_rate_s_inv, audit.boundary_candidate_rate_s_inv
        )
        self.assertIn(0, audit.lower_tail_face_indices)

    def test_active_accuracy_cap_does_not_apply_raw_empty_bin_cfl(self) -> None:
        """The accepted solver cap controls M0/M3/tail, not global raw telemetry."""

        mapping = _mapping(
            bins=96,
            beta_initial=_smooth_initial_definition(bins=96),
            max_dt_s=100.0,
        )
        mapping["simulation"]["accuracy_active_radius_cfl"] = 0.05
        solver = _ConstantVelocityEulerian(SolverConfig.from_mapping(mapping))
        diagnostic = solver.advance_one()
        controlled = max(
            diagnostic.active_population_courant,
            diagnostic.lower_tail_active_courant,
            diagnostic.boundary_active_courant,
        )
        self.assertLessEqual(controlled, 0.05 * (1.0 + 1.0e-12))
        self.assertGreater(diagnostic.radius_courant_max, controlled)
        self.assertEqual(diagnostic.timestep_limiter, "accuracy_active_radius_cfl")

    def test_candidate_step_releases_a_boundary_that_becomes_unreachable(self) -> None:
        """A shrink-only active-CFL loop must not retain an obsolete Rmin face.

        The large legacy candidate lets the tail cross a deliberately fast
        physical lower face.  The active cap subdivides before that crossing,
        so the accepted candidate has a normal interior-tail rate rather than
        a fictitious boundary micro-step.
        """

        bins = 64
        grid = RadiusGrid.logarithmic(5.0e-9, 2.0e-8, bins)
        values = np.zeros(bins, dtype=np.float64)
        values[1] = 1.0e18
        initial = {
            "kind": "cell_integrated",
            "radius_edges_m": [float(value) for value in grid.edges_m],
            "cell_number_density_m3": [float(value) for value in values],
        }
        mapping = _mapping(bins=bins, beta_initial=initial, max_dt_s=1.0e6)
        # Keep the production positivity contract: the legacy candidate is
        # still long enough to reach the deliberately fast lower face.
        mapping["simulation"]["size_cfl"] = 0.4
        mapping["simulation"]["accuracy_active_radius_cfl"] = 1.0e-5
        solver = _BoundarySpikeEulerian(
            SolverConfig.from_mapping(mapping),
            constant_velocity_m_s=-1.0e-12,
            boundary_velocity_m_s=-1.0e-8,
        )
        faces = solver.face_velocities("beta", solver.growth_rates()["beta"])
        legacy_audit = solver.active_cfl_diagnostics("beta", face_velocity_m_s=faces, dt_s=1.0)
        self.assertTrue(legacy_audit.boundary_face_is_active)
        false_boundary_dt = 1.0e-5 / legacy_audit.boundary_candidate_rate_s_inv

        diagnostic = solver.advance_one()

        self.assertFalse(diagnostic.boundary_face_is_active)
        self.assertGreater(diagnostic.dt_s, false_boundary_dt * 1.0e3)
        controlled = max(
            diagnostic.active_population_courant,
            diagnostic.lower_tail_active_courant,
            diagnostic.boundary_active_courant,
        )
        self.assertLessEqual(controlled, 1.0e-5 * (1.0 + 1.0e-12))


if __name__ == "__main__":
    unittest.main()
