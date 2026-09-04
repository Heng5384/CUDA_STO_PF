"""Focused contracts for ``CHARACTERISTIC_DT_CONTINUATION_V1``.

These tests use a real CR1 state/ledger with a zero-velocity characteristic
fixture.  The scripted wrapper changes only whether a candidate reports the
explicit ordinary-closure failure that the controller is allowed to refine;
it never substitutes a physical model, tolerance, Picard cap, or root solver.
"""

from __future__ import annotations

from dataclasses import replace
import tempfile
import unittest
from pathlib import Path
from typing import Any, Iterable
from unittest.mock import patch

import numpy as np

from kwn_mvp.characteristic_dt_continuation import (
    POLICY_NAME,
    CharacteristicDtContinuation,
    DtContinuationError,
    DtContinuationFatalStepError,
    DtContinuationMinDtError,
    accepted_state_hash,
)
from kwn_mvp.characteristic_reference import (
    CharacteristicReferenceError,
    CharacteristicReferenceSolver,
)
from kwn_mvp.radius_grid import RadiusGrid
from kwn_mvp.solver import SolverConfig
from scripts.run_kwn_characteristic_dt_continuation_v1 import (
    Endpoint,
    _old_root_comparison,
    _select_step245_reference,
)
from tests.kwn.test_characteristic_reference import _SafeguardedRestartCharacteristic


RMIN_M = 5.0e-9
RMAX_M = 5.0e-8


def _numbers(*, bins: int) -> np.ndarray:
    values = np.zeros(bins, dtype=np.float64)
    values[bins // 3 : bins // 3 + 5] = np.asarray(
        [1.0, 2.0, 3.0, 2.0, 1.0], dtype=np.float64
    ) * 1.0e18
    return values


def _mapping(*, bins: int, max_dt_s: float = 4.0) -> dict[str, Any]:
    grid = RadiusGrid.logarithmic(RMIN_M, RMAX_M, bins)
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
                "initial": {
                    "kind": "cell_integrated",
                    "radius_edges_m": [float(value) for value in grid.edges_m],
                    "cell_number_density_m3": [float(value) for value in _numbers(bins=bins)],
                },
                "nucleation": {"mode": "off"},
            },
        },
    }


class _ScriptedContinuationSolver(CharacteristicReferenceSolver):
    """Real CR1 state with a deterministic ordinary-closure event script.

    ``fail`` is the exact retryable ordinary-Picard failure.  All other valid
    successes use the production ordinary/P2/P4 entry point on a zero-velocity
    trace, so inventory, history, checkpoints, and state rollback remain real
    CR1 behaviour.
    """

    def __init__(
        self,
        config: SolverConfig,
        *,
        outcomes: Iterable[str] = (),
        **kwargs: Any,
    ) -> None:
        self._outcomes = list(outcomes)
        self.ordinary_calls: list[float] = []
        self.legacy_calls = 0
        super().__init__(config, **kwargs)

    def _velocity_at_radii(self, radii_m: np.ndarray, matrix_xb: float) -> np.ndarray:
        del matrix_xb
        return np.zeros(np.asarray(radii_m).shape, dtype=np.float64)

    def advance_one(self, maximum_dt_s: float | None = None) -> Any:
        del maximum_dt_s
        self.legacy_calls += 1
        raise AssertionError("dt continuation must never call the legacy advance_one entry point")

    def advance_one_ordinary_or_qualified_cycle(self, maximum_dt_s: float | None = None) -> Any:
        dt_s = float(self.config.max_dt_s if maximum_dt_s is None else maximum_dt_s)
        self.ordinary_calls.append(dt_s)
        outcome = self._outcomes.pop(0) if self._outcomes else "ok"
        if outcome == "fail":
            raise CharacteristicReferenceError(
                "characteristic matrix/population fixed point did not converge after 128 iterations "
                "(scripted ordinary-only trial)"
            )
        if outcome == "fatal":
            raise CharacteristicReferenceError("characteristic trial recovered an invalid matrix composition")
        if outcome == "mutate_fail":
            self.matrix_xb += 1.0e-6
            raise CharacteristicReferenceError(
                "characteristic matrix/population fixed point did not converge after 128 iterations "
                "(scripted mutating trial)"
            )

        diagnostic = super().advance_one_ordinary_or_qualified_cycle(maximum_dt_s=maximum_dt_s)
        if outcome == "p2":
            diagnostic = replace(
                diagnostic,
                fixed_point_convergence_mode="BRACKETED_SCALAR_ROOT",
                fixed_point_periodic_cycle_period=2,
                fixed_point_bracketed_root_iterations=3,
                fixed_point_root_verification_kind="MAP_SUCCESSOR",
            )
        elif outcome == "p4":
            diagnostic = replace(
                diagnostic,
                fixed_point_convergence_mode="BRACKETED_SCALAR_ROOT",
                fixed_point_periodic_cycle_period=4,
                fixed_point_bracketed_root_iterations=4,
                fixed_point_root_verification_kind="MAP_SUCCESSOR",
            )
        elif outcome == "safeguard":
            diagnostic = replace(
                diagnostic,
                fixed_point_convergence_mode="SAFEGUARDED_SCALAR_ROOT_V1",
                fixed_point_periodic_cycle_period=0,
                fixed_point_bracketed_root_iterations=3,
                fixed_point_root_verification_kind="SAME_X_IMMUTABLE_REPLAY",
            )
        elif outcome == "bad_period":
            diagnostic = replace(
                diagnostic,
                fixed_point_convergence_mode="BRACKETED_SCALAR_ROOT",
                fixed_point_periodic_cycle_period=3,
                fixed_point_bracketed_root_iterations=3,
                fixed_point_root_verification_kind="MAP_SUCCESSOR",
            )
        elif outcome != "ok":
            raise AssertionError(f"unknown scripted continuation outcome: {outcome}")
        self.history[-1] = diagnostic
        return diagnostic


class CharacteristicDtContinuationContracts(unittest.TestCase):
    """V1 local reject-and-halve policy, restart, and closure-mode contracts."""

    def setUp(self) -> None:
        self.config = SolverConfig.from_mapping(_mapping(bins=40))

    def _solver(self, *outcomes: str) -> _ScriptedContinuationSolver:
        return _ScriptedContinuationSolver(self.config, outcomes=outcomes)

    def _assert_same_state(
        self,
        expected: CharacteristicReferenceSolver,
        observed: CharacteristicReferenceSolver,
    ) -> None:
        self.assertEqual(accepted_state_hash(expected), accepted_state_hash(observed))
        self.assertEqual(set(expected.state_arrays()), set(observed.state_arrays()))
        for key, values in expected.state_arrays().items():
            np.testing.assert_array_equal(values, observed.state_arrays()[key], err_msg=key)

    def test_rejected_full_trial_is_immutable_before_halving(self) -> None:
        solver = self._solver("fail", "ok", "ok")
        controller = CharacteristicDtContinuation(solver, max_refinement_depth=2)
        initial_hash = accepted_state_hash(solver)
        controller.begin_macro(1.0)

        self.assertIsNone(controller.advance_pending())

        self.assertEqual(accepted_state_hash(solver), initial_hash)
        self.assertEqual(solver.step, 0)
        self.assertEqual(solver.time_s, 0.0)
        self.assertEqual(len(solver.history), 0)
        rejected = [event for event in controller.events if event["event"] == "REJECTED_NONCLOSING_SUBSTEP"]
        self.assertEqual(len(rejected), 1)
        self.assertTrue(rejected[0]["history_unchanged"])
        self.assertEqual(rejected[0]["accepted_state_hash_before_trial"], initial_hash)
        self.assertEqual(rejected[0]["accepted_state_hash_after_rejection"], initial_hash)

    def test_rejected_mutating_trial_fails_closed_and_restores_macro_start(self) -> None:
        solver = self._solver("mutate_fail")
        controller = CharacteristicDtContinuation(solver, max_refinement_depth=1)
        initial_hash = accepted_state_hash(solver)

        with self.assertRaisesRegex(DtContinuationError, "mutated accepted CR1 state"):
            controller.advance_macro(1.0)

        self.assertEqual(accepted_state_hash(solver), initial_hash)
        self.assertIsNone(controller.active_macro)

    def test_nonclosing_full_macro_is_retried_as_two_exact_halves(self) -> None:
        solver = self._solver("fail", "ok", "ok")
        result = CharacteristicDtContinuation(solver, max_refinement_depth=2).advance_macro(1.0)

        self.assertEqual(result.refinement_depth, 1)
        self.assertEqual(result.rejection_count, 1)
        self.assertEqual(result.end_time_s, 1.0)
        self.assertEqual([diagnostic.dt_s for diagnostic in result.accepted_diagnostics], [0.5, 0.5])
        self.assertEqual(solver.ordinary_calls, [1.0, 0.5, 0.5])
        self.assertEqual(solver.legacy_calls, 0)

    def test_next_macro_resets_to_the_nominal_timestep(self) -> None:
        solver = self._solver("fail", "ok", "ok", "ok")
        controller = CharacteristicDtContinuation(solver, max_refinement_depth=2)
        first = controller.advance_macro(1.0)
        second = controller.advance_macro(1.0)

        self.assertEqual(first.refinement_depth, 1)
        self.assertEqual(second.refinement_depth, 0)
        self.assertEqual(solver.ordinary_calls, [1.0, 0.5, 0.5, 1.0])
        self.assertEqual(second.start_time_s, 1.0)
        self.assertEqual(second.end_time_s, 2.0)

    def test_adaptive_m2_endpoint_matches_manual_two_substeps(self) -> None:
        adaptive = self._solver("fail", "ok", "ok")
        adaptive_result = CharacteristicDtContinuation(adaptive, max_refinement_depth=2).advance_macro(1.0)
        manual = self._solver()
        manual.advance_one_ordinary_or_qualified_cycle(maximum_dt_s=0.5)
        manual.advance_one_ordinary_or_qualified_cycle(maximum_dt_s=0.5)

        self.assertEqual(adaptive_result.refinement_depth, 1)
        self._assert_same_state(manual, adaptive)

    def test_adaptive_m4_endpoint_matches_manual_four_substeps(self) -> None:
        adaptive = self._solver("fail", "ok", "fail", "ok", "ok", "ok", "ok")
        adaptive_result = CharacteristicDtContinuation(adaptive, max_refinement_depth=2).advance_macro(1.0)
        manual = self._solver()
        for _ in range(4):
            manual.advance_one_ordinary_or_qualified_cycle(maximum_dt_s=0.25)

        self.assertEqual(adaptive_result.refinement_depth, 2)
        self.assertEqual(adaptive_result.rejection_count, 2)
        self._assert_same_state(manual, adaptive)

    def test_same_script_is_deterministic_in_state_and_event_ledger(self) -> None:
        left = self._solver("fail", "ok", "fail", "ok", "ok", "ok", "ok")
        right = self._solver("fail", "ok", "fail", "ok", "ok", "ok", "ok")
        left_controller = CharacteristicDtContinuation(left, max_refinement_depth=2)
        right_controller = CharacteristicDtContinuation(right, max_refinement_depth=2)

        left_result = left_controller.advance_macro(1.0)
        right_result = right_controller.advance_macro(1.0)

        self.assertEqual(left_result, right_result)
        self.assertEqual(left_controller.events, right_controller.events)
        self._assert_same_state(left, right)

    def test_accepted_leaves_preserve_inventory_and_nonnegative_cells(self) -> None:
        solver = self._solver("fail", "ok", "ok")
        result = CharacteristicDtContinuation(solver, max_refinement_depth=2).advance_macro(1.0)

        self.assertEqual(POLICY_NAME, "CHARACTERISTIC_DT_CONTINUATION_V1")
        self.assertTrue(np.all(np.isfinite(solver._beta_cell_numbers())))
        self.assertTrue(np.all(solver._beta_cell_numbers() >= 0.0))
        for diagnostic in result.accepted_diagnostics:
            self.assertLessEqual(
                diagnostic.inventory.relative_residual,
                solver.config.inventory_tolerance_relative,
            )

    def test_controller_never_calls_legacy_safeguarded_entry_point(self) -> None:
        solver = self._solver("fail", "ok", "ok")
        CharacteristicDtContinuation(solver, max_refinement_depth=2).advance_macro(1.0)

        self.assertEqual(solver.legacy_calls, 0)
        self.assertEqual(len(solver.ordinary_calls), 3)

    def test_ordinary_only_solver_entry_disables_the_actual_final_raw_safeguard(self) -> None:
        """The policy boundary must hold below the controller test double."""

        solver = _SafeguardedRestartCharacteristic(self.config, fixed_point_max_iterations=4)
        before = {key: value.copy() for key, value in solver.state_arrays().items()}
        with patch.object(
            solver,
            "_solve_safeguarded_final_raw_picard_bracket",
            side_effect=AssertionError("ordinary-only path invoked legacy safeguard"),
        ) as safeguard:
            with self.assertRaisesRegex(CharacteristicReferenceError, "did not converge after"):
                solver.advance_one_ordinary_or_qualified_cycle(maximum_dt_s=1.0)
        safeguard.assert_not_called()
        for key, expected in before.items():
            np.testing.assert_array_equal(expected, solver.state_arrays()[key], err_msg=key)

    def test_qualified_exact_p2_cycle_is_accepted(self) -> None:
        solver = self._solver("p2")
        result = CharacteristicDtContinuation(solver).advance_macro(0.5)

        self.assertEqual(result.accepted_diagnostics[0].fixed_point_convergence_mode, "BRACKETED_SCALAR_ROOT")
        self.assertEqual(result.accepted_diagnostics[0].fixed_point_periodic_cycle_period, 2)

    def test_qualified_exact_p4_cycle_is_accepted(self) -> None:
        solver = self._solver("p4")
        result = CharacteristicDtContinuation(solver).advance_macro(0.5)

        self.assertEqual(result.accepted_diagnostics[0].fixed_point_convergence_mode, "BRACKETED_SCALAR_ROOT")
        self.assertEqual(result.accepted_diagnostics[0].fixed_point_periodic_cycle_period, 4)

    def test_legacy_safeguard_diagnostic_is_rejected_without_committing_state(self) -> None:
        solver = self._solver("safeguard")
        controller = CharacteristicDtContinuation(solver)
        initial_hash = accepted_state_hash(solver)

        with self.assertRaisesRegex(DtContinuationError, "forbids closure mode"):
            controller.advance_macro(0.5)

        self.assertEqual(accepted_state_hash(solver), initial_hash)
        self.assertIsNone(controller.active_macro)

    def test_unqualified_scalar_root_period_is_rejected_without_committing_state(self) -> None:
        solver = self._solver("bad_period")
        controller = CharacteristicDtContinuation(solver)
        initial_hash = accepted_state_hash(solver)

        with self.assertRaisesRegex(DtContinuationError, "only exact P2/P4"):
            controller.advance_macro(0.5)

        self.assertEqual(accepted_state_hash(solver), initial_hash)
        self.assertIsNone(controller.active_macro)

    def test_nonclosure_at_registered_minimum_depth_restores_the_macro_start(self) -> None:
        solver = self._solver("fail", "fail")
        controller = CharacteristicDtContinuation(solver, max_refinement_depth=1)
        initial_hash = accepted_state_hash(solver)

        with self.assertRaisesRegex(DtContinuationMinDtError, "reached dt/2"):
            controller.advance_macro(1.0)

        self.assertEqual(accepted_state_hash(solver), initial_hash)
        self.assertIsNone(controller.active_macro)
        self.assertEqual(solver.ordinary_calls, [1.0, 0.5])

    def test_nonclosure_is_the_only_error_eligible_for_halving(self) -> None:
        solver = self._solver("fatal")
        controller = CharacteristicDtContinuation(solver, max_refinement_depth=3)
        initial_hash = accepted_state_hash(solver)

        with self.assertRaisesRegex(DtContinuationFatalStepError, "non-retryable characteristic failure"):
            controller.advance_macro(1.0)

        self.assertEqual(solver.ordinary_calls, [1.0])
        self.assertEqual(accepted_state_hash(solver), initial_hash)
        self.assertIsNone(controller.active_macro)

    def test_deeper_retry_discards_rolled_back_history_and_keeps_effective_leaves(self) -> None:
        solver = self._solver("fail", "ok", "fail", "ok", "ok", "ok", "ok")
        controller = CharacteristicDtContinuation(solver, max_refinement_depth=2)
        result = controller.advance_macro(1.0)

        self.assertEqual(result.refinement_depth, 2)
        self.assertEqual(len(solver.history), 4)
        self.assertEqual([diagnostic.dt_s for diagnostic in solver.history], [0.25] * 4)
        accepted = [event for event in controller.events if event["event"] == "ACCEPTED_SUBSTEP"]
        self.assertEqual(sum(bool(event["rolled_back"]) for event in accepted), 1)
        self.assertEqual(sum(not bool(event["rolled_back"]) for event in accepted), 4)

    def test_macro_start_checkpoint_is_an_exact_before_macro_restart_boundary(self) -> None:
        solver = self._solver()
        controller = CharacteristicDtContinuation(solver)
        baseline = self._solver()
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "macro_start.npz"
            controller.begin_macro(0.5, macro_start_checkpoint=checkpoint)
            restored = _ScriptedContinuationSolver.load_checkpoint(config=self.config, path=checkpoint)

        self._assert_same_state(baseline, restored)
        self.assertIsNotNone(controller.active_macro)

    def test_inside_macro_restart_preserves_pending_queue_and_effective_diagnostics(self) -> None:
        continuous_solver = self._solver("fail", "ok", "ok")
        continuous = CharacteristicDtContinuation(continuous_solver, max_refinement_depth=2)
        with tempfile.TemporaryDirectory() as temporary:
            continuous_start = Path(temporary) / "continuous_start.npz"
            continuous.begin_macro(1.0, macro_start_checkpoint=continuous_start)
            self.assertIsNone(continuous.advance_pending())  # reject full step and schedule halves
            self.assertIsNone(continuous.advance_pending())  # accept first half
            continuous_result = continuous.run_active_macro()

        resumed_solver = self._solver("fail", "ok")
        resumed = CharacteristicDtContinuation(resumed_solver, max_refinement_depth=2)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            start = root / "start.npz"
            checkpoint = root / "inside.npz"
            sidecar = root / "inside.json"
            resumed.begin_macro(1.0, macro_start_checkpoint=start)
            self.assertIsNone(resumed.advance_pending())
            self.assertIsNone(resumed.advance_pending())
            resumed.save_active_restart_bundle(checkpoint_path=checkpoint, sidecar_path=sidecar)
            loaded = CharacteristicDtContinuation.load_active_restart_bundle(
                config=self.config,
                sidecar_path=sidecar,
                solver_class=_ScriptedContinuationSolver,
            )
            resumed_result = loaded.run_active_macro()

        self._assert_same_state(continuous_solver, loaded.solver)
        self.assertEqual(continuous_result.accepted_diagnostics, resumed_result.accepted_diagnostics)
        self.assertEqual(len(resumed_result.accepted_diagnostics), 2)
        self.assertEqual(continuous.events, loaded.events)

    def test_after_macro_checkpoint_restart_preserves_accepted_state(self) -> None:
        solver = self._solver("fail", "ok", "ok")
        CharacteristicDtContinuation(solver, max_refinement_depth=2).advance_macro(1.0)
        with tempfile.TemporaryDirectory() as temporary:
            checkpoint = Path(temporary) / "after_macro.npz"
            solver.save_checkpoint(checkpoint)
            restored = _ScriptedContinuationSolver.load_checkpoint(config=self.config, path=checkpoint)

        self._assert_same_state(solver, restored)

    @staticmethod
    def _endpoint_for_comparison(
        *,
        m: int,
        matrix_xb: float,
        m0: float,
        cells: np.ndarray,
    ) -> Endpoint:
        snapshot = {
            "matrix_xB": matrix_xb,
            "M0_m3": m0,
            "M1_m2": 2.0 * m0,
            "M2_m": 3.0 * m0,
            "M3_dimensionless": 4.0 * m0,
            "Q_beta_mol_m3": 5.0 * m0,
        }
        return Endpoint(
            m=m,
            status="PASS",
            reason=None,
            solver=None,
            rows=(),
            snapshot=snapshot,
            cells=np.asarray(cells, dtype=np.float64),
        )

    def test_reference_selection_uses_only_an_actual_finest_pair(self) -> None:
        self.assertEqual(_select_step245_reference({"m16_vs_m32": {"status": "PASS"}}), 16)
        self.assertEqual(
            _select_step245_reference(
                {"m16_vs_m32": {"status": "FAIL"}, "m32_vs_m64": {"status": "PASS"}}
            ),
            32,
        )
        self.assertIsNone(_select_step245_reference({"m16_vs_m32": {"status": "FAIL"}}))

    def test_old_root_branch_label_uses_refinement_envelope_not_root_residual(self) -> None:
        edges = np.asarray([RMIN_M, 2.0 * RMIN_M, 3.0 * RMIN_M], dtype=np.float64)
        m16 = self._endpoint_for_comparison(m=16, matrix_xb=0.0062, m0=1.001, cells=np.array([1.001, 1.001]))
        m32 = self._endpoint_for_comparison(m=32, matrix_xb=0.0062, m0=1.0, cells=np.array([1.0, 1.0]))
        legacy = Endpoint(
            m=1,
            status="PASS_LEGACY_DIAGNOSTIC",
            reason=None,
            solver=None,
            rows=(),
            snapshot=dict(m32.snapshot or {}),
            cells=np.array([1.0, 1.0]),
        )
        _rows, on_branch = _old_root_comparison(
            legacy=legacy, endpoints={16: m16, 32: m32}, selected_m=16, edges_m=edges
        )
        self.assertEqual(on_branch["status"], "OLD_ROOT_ON_CONTINUATION_BRANCH")
        off_legacy = Endpoint(
            m=1,
            status="PASS_LEGACY_DIAGNOSTIC",
            reason=None,
            solver=None,
            rows=(),
            snapshot={**dict(m32.snapshot or {}), "M0_m3": 1.1},
            cells=np.array([2.0, 0.0]),
        )
        _rows, off_branch = _old_root_comparison(
            legacy=off_legacy, endpoints={16: m16, 32: m32}, selected_m=16, edges_m=edges
        )
        self.assertEqual(off_branch["status"], "OLD_ROOT_OFF_CONTINUATION_BRANCH")

    def test_old_root_comparison_lists_all_successful_refinements_but_uses_finest_envelope(self) -> None:
        edges = np.asarray([RMIN_M, 2.0 * RMIN_M, 3.0 * RMIN_M], dtype=np.float64)
        m8 = self._endpoint_for_comparison(m=8, matrix_xb=0.0062, m0=1.2, cells=np.array([1.2, 1.2]))
        m16 = self._endpoint_for_comparison(m=16, matrix_xb=0.0062, m0=1.1, cells=np.array([1.1, 1.1]))
        m32 = self._endpoint_for_comparison(m=32, matrix_xb=0.0062, m0=1.001, cells=np.array([1.001, 1.001]))
        m64 = self._endpoint_for_comparison(m=64, matrix_xb=0.0062, m0=1.0, cells=np.array([1.0, 1.0]))
        legacy = Endpoint(
            m=1,
            status="PASS_LEGACY_DIAGNOSTIC",
            reason=None,
            solver=None,
            rows=(),
            snapshot={key: 1.01 * float(value) for key, value in dict(m64.snapshot or {}).items()},
            cells=np.array([1.01, 1.01]),
        )

        rows, summary = _old_root_comparison(
            legacy=legacy,
            endpoints={8: m8, 16: m16, 32: m32, 64: m64},
            selected_m=32,
            edges_m=edges,
        )

        comparisons = {str(row["comparison"]) for row in rows}
        self.assertTrue(
            {
                "legacy_root_vs_m8",
                "legacy_root_vs_m16",
                "legacy_root_vs_m32",
                "legacy_root_vs_m64",
            }.issubset(comparisons)
        )
        self.assertEqual(summary["comparison_finest_m"], 64)
        self.assertGreater(
            float(summary["observed_error"]["M0_m3"]),
            float(summary["refinement_envelope"]["M0_m3"]),
        )
        self.assertEqual(summary["status"], "OLD_ROOT_OFF_CONTINUATION_BRANCH")


if __name__ == "__main__":
    unittest.main()
