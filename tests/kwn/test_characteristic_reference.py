"""CR1--CR9 contracts for the conservative characteristic KWN reference."""

from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import patch

import numpy as np

from kwn_mvp.characteristic_reference import (
    CharacteristicReferenceError,
    CharacteristicReferenceSolver,
)
from kwn_mvp.conservative_remap import (
    conservative_remap_piecewise_constant,
    piecewise_constant_cdf,
)
from kwn_mvp.diagnostics import discrete_wasserstein_distance
from kwn_mvp.ledger import InventorySnapshot
from kwn_mvp.population_metrics import (
    metrics_from_piecewise_constant_cells,
    positive_cell_quadrature,
)
from kwn_mvp.radius_grid import RadiusGrid
from kwn_mvp.solver import RadiusGridOverflowError, SolverConfig


RMIN_M = 5.0e-9
RMAX_M = 5.0e-8


def _cell_initial(*, bins: int, numbers_m3: np.ndarray) -> dict[str, Any]:
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
            "initial_xB": 0.0062,
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
                "initial": _cell_initial(bins=bins, numbers_m3=beta_numbers_m3),
                "nucleation": {"mode": "off"},
            },
        },
    }


def _numbers(*, bins: int) -> np.ndarray:
    value = np.zeros(bins, dtype=np.float64)
    value[bins // 3 : bins // 3 + 5] = np.asarray(
        [1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64
    ) * 1.0e18
    return value


def _interval_mass(edges: np.ndarray, numbers: np.ndarray, lower: float, upper: float) -> float:
    density = numbers / np.diff(edges)
    overlap = np.maximum(0.0, np.minimum(edges[1:], upper) - np.maximum(edges[:-1], lower))
    return float(np.sum(density * overlap, dtype=np.float64))


def _analytic_constant_translation(
    edges: np.ndarray, numbers: np.ndarray, *, shift_m: float
) -> np.ndarray:
    """Independent interval-overlap cell integral for an exact translation."""

    density = numbers / np.diff(edges)
    output = np.zeros_like(numbers)
    for index, (arrival_lower, arrival_upper) in enumerate(zip(edges[:-1], edges[1:])):
        departure_lower = max(float(edges[0]), float(arrival_lower - shift_m))
        departure_upper = min(float(edges[-1]), float(arrival_upper - shift_m))
        if departure_upper <= departure_lower:
            continue
        overlap = np.maximum(
            0.0,
            np.minimum(edges[1:], departure_upper) - np.maximum(edges[:-1], departure_lower),
        )
        output[index] = float(np.sum(density * overlap, dtype=np.float64))
    return output


def _w1_cells(edges: np.ndarray, left: np.ndarray, right: np.ndarray) -> float:
    left_radii, left_weights = positive_cell_quadrature(edges, left, 2)
    right_radii, right_weights = positive_cell_quadrature(edges, right, 2)
    return discrete_wasserstein_distance(left_radii, left_weights, right_radii, right_weights)


class _ConstantVelocityCharacteristic(CharacteristicReferenceSolver):
    def __init__(self, config: SolverConfig, *, velocity_m_s: float, **kwargs: Any) -> None:
        self.velocity_m_s = float(velocity_m_s)
        super().__init__(config, **kwargs)

    def _velocity_at_radii(self, radii_m: np.ndarray, matrix_xb: float) -> np.ndarray:
        del matrix_xb
        return np.full(np.asarray(radii_m).shape, self.velocity_m_s, dtype=np.float64)


class _InverseRadiusCharacteristic(CharacteristicReferenceSolver):
    def __init__(self, config: SolverConfig, *, k_m2_s: float, **kwargs: Any) -> None:
        self.k_m2_s = float(k_m2_s)
        super().__init__(config, **kwargs)

    def _velocity_at_radii(self, radii_m: np.ndarray, matrix_xb: float) -> np.ndarray:
        del matrix_xb
        radii = np.asarray(radii_m, dtype=np.float64)
        return -self.k_m2_s / radii


class _MatrixCoupledCharacteristic(CharacteristicReferenceSolver):
    def _velocity_at_radii(self, radii_m: np.ndarray, matrix_xb: float) -> np.ndarray:
        radii = np.asarray(radii_m, dtype=np.float64)
        return np.full(radii.shape, 2.0e-12 + 1.0e-9 * (float(matrix_xb) - 0.0062), dtype=np.float64)


class _RestartVelocityCharacteristic(CharacteristicReferenceSolver):
    def _velocity_at_radii(self, radii_m: np.ndarray, matrix_xb: float) -> np.ndarray:
        del matrix_xb
        return np.full(np.asarray(radii_m).shape, -1.0e-12, dtype=np.float64)


class CharacteristicReferenceContracts(unittest.TestCase):
    """CR1--CR9: conservative remapping, closure, restart, and no-CFL behavior."""

    @staticmethod
    def _scripted_scalar_closure(
        *,
        closure_map: Any,
        initial_xb: float,
        fixed_point_max_iterations: int = 128,
        assert_state_unchanged: bool = False,
    ) -> Any:
        """Exercise the scalar fixed-point controller with closed trial states."""

        numbers = _numbers(bins=44) * 1.0e2
        solver = CharacteristicReferenceSolver(
            SolverConfig.from_mapping(_mapping(bins=44, beta_numbers_m3=numbers)),
            fixed_point_max_iterations=fixed_point_max_iterations,
        )
        candidate_a = solver._beta_cell_numbers()
        solver.matrix_xb = float(initial_xb)
        snapshot = InventorySnapshot(
            total_mol_m3=1.0,
            matrix_mol_m3=1.0,
            gp_mol_m3=0.0,
            beta_subgrid_mol_m3=0.0,
            beta_resolved_mol_m3=0.0,
            residual_mol_m3=0.0,
            relative_residual=0.0,
            matrix_fraction=1.0,
        )
        state: dict[str, float] = {}

        def remap_once(*args: Any, **kwargs: Any) -> tuple[object, SimpleNamespace]:
            del args
            midpoint = float(kwargs["midpoint_matrix_xb"])
            state["x_guess"] = 2.0 * midpoint - float(initial_xb)
            return object(), SimpleNamespace(cell_number_m3=candidate_a)

        def recover_matrix(candidate: np.ndarray) -> float:
            del candidate
            return float(closure_map(state["x_guess"]))

        before = solver.state_arrays() if assert_state_unchanged else {}
        with (
            patch.object(solver, "_remap_once", side_effect=remap_once),
            patch.object(solver, "_recover_matrix_xb", side_effect=recover_matrix),
            patch.object(solver, "_inventory_snapshot", return_value=snapshot),
        ):
            try:
                return solver._solve_fixed_point(
                    old_cell_number_m3=candidate_a,
                    dt_s=1.0,
                )
            finally:
                if assert_state_unchanged:
                    for key, values in before.items():
                        np.testing.assert_array_equal(values, solver.state_arrays()[key], err_msg=key)

    def test_declared_frozen_grid_edges_are_preserved_exactly(self) -> None:
        """A hash-bound cell measure can carry its exact archived face locations."""

        bins = 40
        numbers = _numbers(bins=bins)
        mapping = _mapping(bins=bins, beta_numbers_m3=numbers)
        edges = RadiusGrid.logarithmic(RMIN_M, RMAX_M, bins).edges_m.copy()
        edges[9] = np.nextafter(edges[9], np.inf)
        mapping["radius_grid"]["edges_m"] = [float(value) for value in edges]
        mapping["populations"]["beta"]["initial"]["radius_edges_m"] = [
            float(value) for value in edges
        ]
        config = SolverConfig.from_mapping(mapping)
        np.testing.assert_array_equal(config.grid.edges_m, edges)

    def test_cr1_zero_velocity_is_bitwise_identity(self) -> None:
        numbers = _numbers(bins=40)
        solver = _ConstantVelocityCharacteristic(
            SolverConfig.from_mapping(_mapping(bins=40, beta_numbers_m3=numbers)), velocity_m_s=0.0
        )
        before = solver.state_arrays()
        diagnostic = solver.advance_one(maximum_dt_s=123.0)
        after = solver.state_arrays()
        np.testing.assert_array_equal(
            before["beta_number_density_per_m4"], after["beta_number_density_per_m4"]
        )
        self.assertEqual(float(before["matrix_xb"][0]), float(after["matrix_xb"][0]))
        self.assertEqual(diagnostic.rmin_number_loss_m3, 0.0)
        self.assertEqual(diagnostic.remap_number_conservation_residual_m3, 0.0)

    def test_cr2_constant_positive_translation_matches_analytic_cell_measure(self) -> None:
        numbers = _numbers(bins=56)
        solver = _ConstantVelocityCharacteristic(
            SolverConfig.from_mapping(_mapping(bins=56, beta_numbers_m3=numbers)), velocity_m_s=1.0e-12
        )
        edges = solver.population("beta").grid.edges_m
        dt_s = 0.1 * float(np.min(np.diff(edges))) / 1.0e-12
        expected = _analytic_constant_translation(edges, numbers, shift_m=1.0e-12 * dt_s)
        before_metrics = metrics_from_piecewise_constant_cells(edges, numbers)
        solver.advance_one(maximum_dt_s=dt_s)
        observed = solver._beta_cell_numbers()
        observed_metrics = metrics_from_piecewise_constant_cells(edges, observed)
        expected_metrics = metrics_from_piecewise_constant_cells(edges, expected)
        np.testing.assert_allclose(observed, expected, rtol=2.0e-13, atol=0.0)
        self.assertAlmostEqual(observed_metrics.M0_m3, before_metrics.M0_m3, places=4)
        for field in ("M0_m3", "M1_m2", "M2_m", "M3_dimensionless"):
            self.assertTrue(
                math.isclose(
                    float(getattr(observed_metrics, field)),
                    float(getattr(expected_metrics, field)),
                    rel_tol=2.0e-13,
                    abs_tol=0.0,
                ),
                msg=field,
            )
        cdf_observed = piecewise_constant_cdf(edges, observed, edges)
        cdf_expected = piecewise_constant_cdf(edges, expected, edges)
        np.testing.assert_allclose(cdf_observed, cdf_expected, rtol=2.0e-13, atol=0.0)
        self.assertLessEqual(_w1_cells(edges, observed, expected), 1.0e-24)

    def test_cr3_constant_negative_translation_has_analytic_rmin_loss(self) -> None:
        numbers = _numbers(bins=48)
        numbers[0] = 5.0e18
        solver = _ConstantVelocityCharacteristic(
            SolverConfig.from_mapping(_mapping(bins=48, beta_numbers_m3=numbers)), velocity_m_s=-1.0e-12
        )
        edges = solver.population("beta").grid.edges_m
        shift = 0.4 * float(np.diff(edges)[0])
        expected_loss = _interval_mass(edges, numbers, float(edges[0]), float(edges[0] + shift))
        diagnostic = solver.advance_one(maximum_dt_s=shift / 1.0e-12)
        observed = solver._beta_cell_numbers()
        self.assertTrue(math.isclose(diagnostic.rmin_number_loss_m3, expected_loss, rel_tol=2.0e-13))
        self.assertTrue(
            math.isclose(float(np.sum(observed)), float(np.sum(numbers)) - expected_loss, rel_tol=2.0e-13)
        )
        self.assertEqual(diagnostic.rmax_number_loss_m3, 0.0)

    def test_cr4_inverse_radius_trace_and_boundary_event_follow_analytic_characteristic(self) -> None:
        numbers = _numbers(bins=48)
        numbers[0] = 2.0e18
        k_m2_s = 2.0e-24
        solver = _InverseRadiusCharacteristic(
            SolverConfig.from_mapping(_mapping(bins=48, beta_numbers_m3=numbers)), k_m2_s=k_m2_s
        )
        edges = solver.population("beta").grid.edges_m
        dt_s = 2.0e4
        trace = solver._trace_faces(dt_s=dt_s, midpoint_matrix_xb=solver.matrix_xb)
        exact_departure = np.sqrt(edges[:-1] ** 2 + 2.0 * k_m2_s * dt_s)
        np.testing.assert_allclose(trace.departure_faces_m[:-1], exact_departure, rtol=2.0e-5, atol=0.0)
        expected_loss = _interval_mass(edges, numbers, float(edges[0]), float(exact_departure[0]))
        diagnostic = solver.advance_one(maximum_dt_s=dt_s)
        self.assertTrue(math.isclose(diagnostic.rmin_number_loss_m3, expected_loss, rel_tol=3.0e-5))
        self.assertLess(metrics_from_piecewise_constant_cells(edges, solver._beta_cell_numbers()).M3_dimensionless,
                        metrics_from_piecewise_constant_cells(edges, numbers).M3_dimensionless)

    def test_cr5_inventory_closure_is_algebraic(self) -> None:
        numbers = _numbers(bins=40)
        solver = _ConstantVelocityCharacteristic(
            SolverConfig.from_mapping(_mapping(bins=40, beta_numbers_m3=numbers)), velocity_m_s=-1.0e-12
        )
        diagnostic = solver.advance_one(maximum_dt_s=10.0)
        self.assertLessEqual(diagnostic.inventory.relative_residual, 1.0e-12)
        relative_remap = abs(diagnostic.remap_number_conservation_residual_m3) / max(
            float(np.sum(numbers)), 1.0e-300
        )
        self.assertLessEqual(relative_remap, 1.0e-12)

    def test_cr6_matrix_coupled_predictor_corrector_closes(self) -> None:
        numbers = _numbers(bins=44) * 1.0e2
        solver = _MatrixCoupledCharacteristic(
            SolverConfig.from_mapping(_mapping(bins=44, beta_numbers_m3=numbers)),
            fixed_point_rtol=1.0e-12,
            fixed_point_atol=1.0e-14,
        )
        dt_s = 20.0
        before = solver._beta_cell_numbers()
        edges = solver.population("beta").grid.edges_m
        x_start = solver.matrix_xb

        # Independent scalar bisection of the documented midpoint closure.
        # It deliberately bypasses the solver's fixed-point loop while using
        # the same frozen algebraic inventory contract.
        def closure_map(x_guess: float) -> float:
            midpoint = 0.5 * (x_start + x_guess)
            velocity = 2.0e-12 + 1.0e-9 * (midpoint - 0.0062)
            departure = np.clip(edges - velocity * dt_s, edges[0], edges[-1])
            remap = conservative_remap_piecewise_constant(edges, before, departure)
            return solver._recover_matrix_xb(remap.cell_number_m3)

        lower = 0.0060
        upper = 0.0062
        lower_residual = closure_map(lower) - lower
        upper_residual = closure_map(upper) - upper
        self.assertGreater(lower_residual, 0.0)
        self.assertLess(upper_residual, 0.0)
        # Orient the brackets explicitly because the map decreases over this
        # interval; the bisection is not allowed to borrow solver iterations.
        for _ in range(80):
            midpoint = 0.5 * (lower + upper)
            residual = closure_map(midpoint) - midpoint
            if residual > 0.0:
                lower = midpoint
            else:
                upper = midpoint
        reference_xb = 0.5 * (lower + upper)

        diagnostic = solver.advance_one(maximum_dt_s=dt_s)
        self.assertGreaterEqual(diagnostic.fixed_point_iterations, 2)
        self.assertEqual(diagnostic.fixed_point_convergence_mode, "DIRECT")
        self.assertLessEqual(diagnostic.fixed_point_xb_residual, 1.0e-12)
        self.assertLessEqual(diagnostic.fixed_point_population_residual, 1.0e-12)
        self.assertLessEqual(diagnostic.inventory.relative_residual, 1.0e-12)
        self.assertTrue(math.isclose(diagnostic.matrix_xb, reference_xb, rel_tol=0.0, abs_tol=1.0e-13))

    def test_exact_two_cycle_triggers_strict_scalar_root_not_cycle_acceptance(self) -> None:
        root = 0.0062
        result = self._scripted_scalar_closure(
            closure_map=lambda x: 2.0 * root - x,
            initial_xb=root + 1.0e-8,
        )
        self.assertEqual(result.convergence_mode, "BRACKETED_SCALAR_ROOT")
        self.assertEqual(result.periodic_cycle_period, 2)
        self.assertGreaterEqual(result.bracketed_root_iterations, 1)
        self.assertLessEqual(result.root_trial_xb_residual, 5.1e-12)
        self.assertLessEqual(result.xb_residual, 5.1e-12)
        self.assertLessEqual(result.root_verification_population_residual, 1.0e-12)
        self.assertEqual(result.inventory.relative_residual, 0.0)

    def test_exact_two_cycle_with_discontinuous_map_fails_closed(self) -> None:
        root = 0.0062
        delta = 2.0e-8
        with self.assertRaises(CharacteristicReferenceError):
            self._scripted_scalar_closure(
                closure_map=lambda x: root + delta if x < root else root - delta,
                initial_xb=root + 4.0e-8,
                assert_state_unchanged=True,
            )

    def test_exact_four_cycle_triggers_strict_scalar_root_not_cycle_acceptance(self) -> None:
        root = 0.0062
        delta = 1.0e-8
        a, b, c, d = root - 4.0 * delta, root + 3.0 * delta, root - 2.0 * delta, root + delta
        periodic_map = {a: b, b: c, c: d, d: a}
        result = self._scripted_scalar_closure(
            closure_map=lambda x: periodic_map.get(x, root),
            initial_xb=a,
        )
        self.assertEqual(result.convergence_mode, "BRACKETED_SCALAR_ROOT")
        self.assertEqual(result.periodic_cycle_period, 4)
        self.assertEqual(result.picard_iterations, 8)
        self.assertGreaterEqual(result.bracketed_root_iterations, 1)
        self.assertLessEqual(result.root_trial_xb_residual, 5.1e-12)
        self.assertLessEqual(result.xb_residual, 5.1e-12)
        self.assertLessEqual(result.root_verification_population_residual, 1.0e-12)
        self.assertEqual(result.inventory.relative_residual, 0.0)

    def test_exact_four_cycle_with_discontinuous_map_fails_closed(self) -> None:
        root = 0.0062
        delta = 2.0e-8
        a, b, c, d = root - 4.0 * delta, root + 3.0 * delta, root - 2.0 * delta, root + delta
        periodic_map = {a: b, b: c, c: d, d: a}
        with self.assertRaises(CharacteristicReferenceError):
            self._scripted_scalar_closure(
                closure_map=lambda x: periodic_map.get(
                    x, root + delta if x < root else root - delta
                ),
                initial_xb=a,
                assert_state_unchanged=True,
            )

    def test_nonexact_slow_picard_converges_directly_with_bounded_128_cap(self) -> None:
        root = 0.0062
        result = self._scripted_scalar_closure(
            closure_map=lambda x: root + 0.9 * (x - root),
            initial_xb=root + 1.0e-6,
        )
        self.assertEqual(result.convergence_mode, "DIRECT")
        self.assertGreater(result.iterations, 64)
        self.assertLessEqual(result.xb_residual, 5.1e-12)
        self.assertEqual(result.bracketed_root_iterations, 0)

    def test_nonexact_unclosed_map_does_not_trigger_generic_bracketing(self) -> None:
        root = 0.0062
        with self.assertRaises(CharacteristicReferenceError):
            self._scripted_scalar_closure(
                closure_map=lambda x: x + 1.0e-8,
                initial_xb=root,
            )

    def test_cr7_restart_matches_continuous_state(self) -> None:
        numbers = _numbers(bins=40)
        config = SolverConfig.from_mapping(_mapping(bins=40, beta_numbers_m3=numbers))
        continuous = _RestartVelocityCharacteristic(config)
        continuous.run_to_time(1.0, maximum_step_s=0.1)
        split = _RestartVelocityCharacteristic(config)
        split.run_to_time(0.5, maximum_step_s=0.1)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "characteristic.npz"
            split.save_checkpoint(checkpoint)
            resumed = _RestartVelocityCharacteristic.load_checkpoint(config=config, path=checkpoint)
            resumed.run_to_time(1.0, maximum_step_s=0.1)
        for key, expected in continuous.state_arrays().items():
            np.testing.assert_array_equal(expected, resumed.state_arrays()[key], err_msg=key)

    def test_cr8_timestep_refinement_converges(self) -> None:
        numbers = _numbers(bins=72) * 1.0e2
        config = SolverConfig.from_mapping(_mapping(bins=72, beta_numbers_m3=numbers))
        target_s = 1.0e3
        values: dict[float, tuple[float, float]] = {}
        timesteps = (target_s, target_s / 2.0, target_s / 4.0, target_s / 8.0, target_s / 16.0, target_s / 32.0)
        for dt_s in timesteps:
            solver = _MatrixCoupledCharacteristic(
                config, fixed_point_rtol=1.0e-12, fixed_point_atol=1.0e-14
            )
            solver.run_to_time(target_s, maximum_step_s=dt_s)
            values[dt_s] = (
                metrics_from_piecewise_constant_cells(
                solver.population("beta").grid.edges_m, solver._beta_cell_numbers()
                ).M3_dimensionless,
                solver.matrix_xb,
            )
        reference = values[target_s / 32.0]
        m3_errors = [
            abs(values[dt_s][0] - reference[0]) / max(abs(reference[0]), 1.0e-300)
            for dt_s in timesteps[:-1]
        ]
        xb_errors = [
            abs(values[dt_s][1] - reference[1]) / max(abs(reference[1]), 1.0e-300)
            for dt_s in timesteps[:-1]
        ]
        # CR1's repeated piecewise-constant projection can introduce a small
        # radius-space plateau, so demand an actual coupled-time reduction
        # rather than an artificial pointwise monotonicity assertion across
        # the whole ladder.  Both registered observables must improve by more
        # than an order of magnitude and the finest tested pair must be well
        # below the coarse-step error.
        self.assertGreater(m3_errors[0], 1.0e-2)
        self.assertGreater(xb_errors[0], 1.0e-2)
        self.assertLess(m3_errors[-1], m3_errors[0] * 0.05)
        self.assertLess(xb_errors[-1], xb_errors[0] * 0.05)
        self.assertLess(m3_errors[-1], 5.0e-4)
        self.assertLess(xb_errors[-1], 5.0e-4)

    def test_cr9_characteristic_has_no_explicit_donor_cfl_dependency(self) -> None:
        numbers = _numbers(bins=56)
        config = SolverConfig.from_mapping(_mapping(bins=56, beta_numbers_m3=numbers))
        velocity = 1.2e-7
        coarse = _ConstantVelocityCharacteristic(config, velocity_m_s=velocity)
        width_min = float(np.min(np.diff(coarse.population("beta").grid.edges_m)))
        self.assertGreater(abs(velocity) * 0.2 / width_min, 100.0)
        coarse.run_to_time(0.2, maximum_step_s=0.2)
        fine = _ConstantVelocityCharacteristic(config, velocity_m_s=velocity)
        fine.run_to_time(0.2, maximum_step_s=0.025)
        self.assertGreaterEqual(float(np.min(coarse.population("beta").number_density_per_m4)), 0.0)
        self.assertLessEqual(coarse.history[-1].inventory.relative_residual, 1.0e-12)
        coarse_m0 = metrics_from_piecewise_constant_cells(
            coarse.population("beta").grid.edges_m, coarse._beta_cell_numbers()
        ).M0_m3
        fine_m0 = metrics_from_piecewise_constant_cells(
            fine.population("beta").grid.edges_m, fine._beta_cell_numbers()
        ).M0_m3
        self.assertTrue(math.isclose(coarse_m0, fine_m0, rel_tol=1.0e-12))
        self.assertLessEqual(
            _w1_cells(coarse.population("beta").grid.edges_m, coarse._beta_cell_numbers(), fine._beta_cell_numbers()),
            float(np.max(np.diff(coarse.population("beta").grid.edges_m))),
        )

    def test_rmax_outflow_fails_closed_before_state_commit(self) -> None:
        numbers = _numbers(bins=32)
        numbers[-1] = 2.0e18
        solver = _ConstantVelocityCharacteristic(
            SolverConfig.from_mapping(_mapping(bins=32, beta_numbers_m3=numbers)), velocity_m_s=1.0e-9
        )
        before = solver.state_arrays()
        with self.assertRaises(RadiusGridOverflowError):
            solver.advance_one(maximum_dt_s=0.1)
        for key, expected in before.items():
            np.testing.assert_array_equal(expected, solver.state_arrays()[key], err_msg=key)


if __name__ == "__main__":
    unittest.main()
