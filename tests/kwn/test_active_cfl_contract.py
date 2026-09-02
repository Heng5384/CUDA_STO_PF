"""Regression checks for the declared active-population CFL authority."""

from __future__ import annotations

import unittest

from kwn_mvp.solver import SolverConfig

from tests.kwn.test_lower_boundary_contract import (
    _ConstantVelocityEulerian,
    _constant_eulerian,
    _first_cell_initial_definition,
    _mapping,
    _smooth_initial_definition,
)


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


if __name__ == "__main__":
    unittest.main()
