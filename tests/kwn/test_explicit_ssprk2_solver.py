"""Focused contracts for the independent conservative explicit SSPRK2 solver."""

from __future__ import annotations

from dataclasses import replace
import importlib.util
import math
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any

import numpy as np

from kwn_mvp.explicit_solver import ExplicitSSPRK2Solver
from kwn_mvp.lower_boundary import boundary_growth_velocity
from kwn_mvp.radius_grid import RadiusGrid
from kwn_mvp.solver import RadiusGridOverflowError, SolverConfig, SolverStateError


ROOT = Path(__file__).resolve().parents[2]
RMIN_M = 5.0e-9
RMAX_M = 2.0e-8


def _cell_integrated_initial(*, bins: int, numbers_m3: np.ndarray) -> dict[str, Any]:
    grid = RadiusGrid.logarithmic(RMIN_M, RMAX_M, bins)
    values = np.asarray(numbers_m3, dtype=np.float64)
    if values.shape != (bins,):
        raise ValueError("test cell numbers must match bins")
    return {
        "kind": "cell_integrated",
        "radius_edges_m": [float(value) for value in grid.edges_m],
        "cell_number_density_m3": [float(value) for value in values],
    }


def _mapping(*, bins: int, beta_numbers_m3: np.ndarray, max_dt_s: float = 1.0e9) -> dict[str, Any]:
    """Small, closed beta-only state with no physical source term."""

    return {
        "simulation": {
            "temperature_K": 653.15,
            "max_dt_s": max_dt_s,
            "min_dt_s": 1.0e-16,
            "size_cfl": 0.35,
            "cfl_active_inventory_relative_threshold": 1.0e-12,
            "rmax_outflow_relative_tolerance": 1.0e-12,
            "population_measure": "cell_integrated",
        },
        "matrix": {
            "molar_volume_m3_mol": 4.1009e-5,
            "initial_xB": 0.005,
            "total_b_mol_m3": None,
            "inventory_tolerance_relative": 1.0e-12,
        },
        "radius_grid": {"minimum_m": RMIN_M, "maximum_m": RMAX_M, "bins": bins},
        "thermodynamics": {"mode": "approximate_dilute", "planar_reference_xB": 0.006},
        "populations": {
            "g": {
                "xB": 0.02,
                "molar_volume_m3_mol": 4.1009e-5,
                "diffusivity_m2_s": 0.0,
                "gamma_j_m2": 0.0,
                "xeq_infinity": 0.006,
                "initial": {"kind": "empty"},
                "nucleation": {"mode": "off"},
            },
            "beta": {
                "xB": 0.8,
                "molar_volume_m3_mol": 4.1009e-5,
                "diffusivity_m2_s": 1.0e-19,
                "gamma_j_m2": 0.0,
                "xeq_infinity": 0.006,
                "initial": _cell_integrated_initial(bins=bins, numbers_m3=beta_numbers_m3),
                "nucleation": {"mode": "off"},
            },
        },
    }


class _ConstantVelocityExplicit(ExplicitSSPRK2Solver):
    """Test-only transport fixture; production physical growth stays untouched."""

    def __init__(
        self, config: SolverConfig, *, beta_velocity_m_s: float, donor_safety: float = 1.0
    ) -> None:
        self.beta_velocity_m_s = float(beta_velocity_m_s)
        super().__init__(config, donor_safety=donor_safety)

    def _stage_velocity_and_faces(
        self, name: str, density_per_m4: np.ndarray, matrix_xb: float
    ) -> tuple[np.ndarray, np.ndarray]:
        del density_per_m4, matrix_xb
        population = self.populations[name]
        velocity = np.full(
            population.grid.bins,
            self.beta_velocity_m_s if name == "beta" else 0.0,
            dtype=np.float64,
        )
        lower = self.beta_velocity_m_s if name == "beta" else 0.0
        return velocity, self._face_velocities(velocity, lower_boundary_velocity_m_s=lower)


class _TwoSidedDonorExplicit(ExplicitSSPRK2Solver):
    """A three-cell fixture whose central donor leaves through both faces."""

    def _stage_velocity_and_faces(
        self, name: str, density_per_m4: np.ndarray, matrix_xb: float
    ) -> tuple[np.ndarray, np.ndarray]:
        del density_per_m4, matrix_xb
        population = self.populations[name]
        velocity = np.zeros(population.grid.bins, dtype=np.float64)
        faces = np.zeros(population.grid.bins + 1, dtype=np.float64)
        if name == "beta":
            if population.grid.bins != 3:
                raise AssertionError("two-sided donor test requires exactly three cells")
            faces[1] = -2.0e-12
            faces[2] = 3.0e-12
        return velocity, faces


class _EmptyOutflowCellExplicit(ExplicitSSPRK2Solver):
    """A zero-occupied cell still has a frozen forward-Euler donor coefficient."""

    def _stage_velocity_and_faces(
        self, name: str, density_per_m4: np.ndarray, matrix_xb: float
    ) -> tuple[np.ndarray, np.ndarray]:
        del density_per_m4, matrix_xb
        population = self.populations[name]
        velocity = np.zeros(population.grid.bins, dtype=np.float64)
        faces = np.zeros(population.grid.bins + 1, dtype=np.float64)
        if name == "beta":
            faces[0] = -8.0e-12
        return velocity, faces


class _StageOneTighteningExplicit(_ConstantVelocityExplicit):
    """Force exactly one stage-one donor reduction to exercise full rollback."""

    def __init__(self, config: SolverConfig, *, beta_velocity_m_s: float) -> None:
        self.tightened_once = False
        super().__init__(config, beta_velocity_m_s=beta_velocity_m_s)

    def _donor_bound_for_state(self, state, matrix_xb, time_s, *, operator=None):
        result = super()._donor_bound_for_state(
            state, matrix_xb, time_s, operator=operator
        )
        if time_s > self.time_s and not self.tightened_once:
            self.tightened_once = True
            return replace(result, bound_s=0.25 * result.bound_s)
        return result


class _RestartVelocityExplicit(_ConstantVelocityExplicit):
    """Checkpoint-compatible fixed-velocity subclass used only in this test."""

    def __init__(self, config: SolverConfig, donor_safety: float = 1.0) -> None:
        super().__init__(
            config,
            beta_velocity_m_s=-1.0e-12,
            donor_safety=donor_safety,
        )


def _assert_state_equal(test: unittest.TestCase, left: dict[str, np.ndarray], right: dict[str, np.ndarray]) -> None:
    test.assertEqual(set(left), set(right))
    for key in left:
        np.testing.assert_array_equal(left[key], right[key], err_msg=key)


class ExplicitSSPRK2SolverTests(unittest.TestCase):
    """Positivity, physical-boundary, restart, and temporal-reference gates."""

    def test_physical_rmin_face_uses_shared_kernel(self) -> None:
        numbers = np.zeros(16, dtype=np.float64)
        numbers[3] = 1.0e18
        solver = ExplicitSSPRK2Solver(SolverConfig.from_mapping(_mapping(bins=16, beta_numbers_m3=numbers)))
        beta = solver.population("beta")
        _, faces = solver._stage_velocity_and_faces(
            "beta", beta.number_density_per_m4.copy(), solver.matrix_xb
        )
        expected = boundary_growth_velocity(
            radius_m=beta.grid.edges_m[0],
            matrix_xb=solver.matrix_xb,
            parameters=beta.parameters,
            equilibrium_adapter=solver.equilibrium_adapter,
        )
        self.assertEqual(float(faces[0]), expected)

    def test_exact_donor_bound_sums_both_outgoing_faces(self) -> None:
        numbers = np.asarray([0.0, 1.0e18, 0.0], dtype=np.float64)
        solver = _TwoSidedDonorExplicit(SolverConfig.from_mapping(_mapping(bins=3, beta_numbers_m3=numbers)))
        width = solver.config.grid.widths_m[1]
        expected = width / (2.0e-12 + 3.0e-12)
        self.assertTrue(math.isclose(solver.exact_donor_bound_s(), expected, rel_tol=1.0e-14))

    def test_empty_cell_with_outflow_remains_in_full_domain_donor_bound(self) -> None:
        numbers = np.asarray([0.0, 1.0e18, 0.0], dtype=np.float64)
        solver = _EmptyOutflowCellExplicit(
            SolverConfig.from_mapping(_mapping(bins=3, beta_numbers_m3=numbers))
        )
        expected = solver.config.grid.widths_m[0] / 8.0e-12
        self.assertTrue(math.isclose(solver.exact_donor_bound_s(), expected, rel_tol=1.0e-14))

    def test_positive_conservative_step_and_algebraic_matrix_closure(self) -> None:
        numbers = np.zeros(32, dtype=np.float64)
        numbers[0] = 1.0e18
        solver = _ConstantVelocityExplicit(
            SolverConfig.from_mapping(_mapping(bins=32, beta_numbers_m3=numbers)),
            beta_velocity_m_s=-1.0e-12,
        )
        bound = solver.exact_donor_bound_s()
        diagnostic = solver.advance_one(maximum_dt_s=0.25 * bound)
        beta = solver.population("beta")
        self.assertGreater(diagnostic.beta_rmin_number_flux_m3_s, 0.0)
        self.assertGreaterEqual(float(np.min(beta.number_density_per_m4)), 0.0)
        self.assertEqual(diagnostic.roundoff_zeroed_bin_count, 0)
        self.assertLessEqual(diagnostic.inventory.relative_residual, 1.0e-12)
        expected_flux = 1.0e-12 * (numbers[0] / beta.grid.widths_m[0]) * (1.0 - 0.125)
        self.assertTrue(
            math.isclose(
                diagnostic.beta_rmin_number_flux_m3_s, expected_flux, rel_tol=1.0e-13
            )
        )

    def test_stage_one_donor_rejection_restarts_from_full_step_origin(self) -> None:
        numbers = np.zeros(24, dtype=np.float64)
        numbers[0] = 1.0e18
        config = SolverConfig.from_mapping(_mapping(bins=24, beta_numbers_m3=numbers))
        forced = _StageOneTighteningExplicit(config, beta_velocity_m_s=-1.0e-12)
        bound = forced.exact_donor_bound_s()
        diagnostic = forced.advance_one(maximum_dt_s=0.8 * bound)
        self.assertTrue(forced.tightened_once)
        self.assertEqual(diagnostic.rejected_step_count_delta, 1)
        self.assertEqual(forced.rejected_step_count, 1)
        self.assertTrue(math.isclose(diagnostic.dt_s, 0.25 * bound, rel_tol=1.0e-14))

        clean = _ConstantVelocityExplicit(config, beta_velocity_m_s=-1.0e-12)
        clean.advance_one(maximum_dt_s=0.25 * bound)
        forced_state = forced.state_arrays()
        clean_state = clean.state_arrays()
        for key in (
            "g_number_density_per_m4",
            "beta_number_density_per_m4",
            "matrix_xb",
            "time_s",
            "step",
            "donor_safety",
        ):
            np.testing.assert_array_equal(forced_state[key], clean_state[key], err_msg=key)

    def test_rmax_outflow_fails_closed_without_state_mutation(self) -> None:
        numbers = np.zeros(20, dtype=np.float64)
        numbers[-1] = 1.0e18
        solver = _ConstantVelocityExplicit(
            SolverConfig.from_mapping(_mapping(bins=20, beta_numbers_m3=numbers)),
            beta_velocity_m_s=1.0e-12,
        )
        before = solver.state_arrays()
        with self.assertRaises(RadiusGridOverflowError):
            solver.advance_one(maximum_dt_s=0.1 * solver.exact_donor_bound_s())
        _assert_state_equal(self, before, solver.state_arrays())
        self.assertEqual(solver.rejected_step_count, 0)

    def test_checkpoint_restart_is_deterministic(self) -> None:
        bins = 40
        grid = RadiusGrid.logarithmic(RMIN_M, RMAX_M, bins)
        numbers = 1.0e18 * np.exp(-0.5 * ((grid.centres_m - 1.15e-8) / 1.1e-9) ** 2)
        config = SolverConfig.from_mapping(_mapping(bins=bins, beta_numbers_m3=numbers))
        continuous = _RestartVelocityExplicit(config)
        dt_s = 0.15 * continuous.exact_donor_bound_s()
        for _ in range(5):
            continuous.advance_one(maximum_dt_s=dt_s)

        split = _RestartVelocityExplicit(config)
        for _ in range(2):
            split.advance_one(maximum_dt_s=dt_s)
        with tempfile.TemporaryDirectory() as directory:
            checkpoint = Path(directory) / "explicit_restart.npz"
            split.save_checkpoint(checkpoint)
            restarted = _RestartVelocityExplicit.load_checkpoint(config=config, path=checkpoint)
            self.assertIsInstance(restarted, _RestartVelocityExplicit)
            for _ in range(3):
                restarted.advance_one(maximum_dt_s=dt_s)
        _assert_state_equal(self, continuous.state_arrays(), restarted.state_arrays())
        self.assertEqual(continuous.rejected_step_count, restarted.rejected_step_count)

    def test_ssprk2_time_refinement_improves_at_fixed_grid(self) -> None:
        bins = 96
        grid = RadiusGrid.logarithmic(RMIN_M, RMAX_M, bins)
        numbers = 1.0e18 * np.exp(-0.5 * ((grid.centres_m - 1.15e-8) / 1.0e-9) ** 2)
        config = SolverConfig.from_mapping(_mapping(bins=bins, beta_numbers_m3=numbers))

        def evolve(fraction: float) -> np.ndarray:
            solver = _ConstantVelocityExplicit(config, beta_velocity_m_s=1.0e-12)
            base = solver.exact_donor_bound_s()
            target = 8.0 * base
            while solver.time_s < target:
                solver.advance_one(maximum_dt_s=min(fraction * base, target - solver.time_s))
            return solver.population("beta").number_density_per_m4.copy()

        coarse = evolve(0.45)
        medium = evolve(0.225)
        fine = evolve(0.1125)
        reference = evolve(0.028125)
        scale = max(float(np.linalg.norm(reference)), 1.0e-300)
        error_coarse = float(np.linalg.norm(coarse - reference)) / scale
        error_medium = float(np.linalg.norm(medium - reference)) / scale
        error_fine = float(np.linalg.norm(fine - reference)) / scale
        self.assertLess(error_medium, error_coarse)
        self.assertLess(error_fine, error_medium)

    def test_canonical_frozen_first_step_is_honestly_blocked_below_min_dt(self) -> None:
        path = ROOT / "scripts" / "run_kwn_lower_boundary_time_accuracy_v1.py"
        specification = importlib.util.spec_from_file_location("explicit_ssprk2_canonical_fixture", path)
        if specification is None or specification.loader is None:
            self.fail("cannot load frozen canonical context builder")
        module = importlib.util.module_from_spec(specification)
        sys.modules[specification.name] = module
        specification.loader.exec_module(module)
        context = module._build_canonical_context(bins=3200)
        solver = ExplicitSSPRK2Solver(SolverConfig.from_mapping(context.mapping), donor_safety=1.0)
        expected = 6.723883286860197e-20
        self.assertTrue(math.isclose(solver.exact_donor_bound_s(), expected, rel_tol=5.0e-15))
        before = solver.state_arrays()
        with self.assertRaises(SolverStateError):
            solver.advance_one()
        _assert_state_equal(self, before, solver.state_arrays())
        self.assertEqual(solver.rejected_step_count, 0)


if __name__ == "__main__":
    unittest.main()
