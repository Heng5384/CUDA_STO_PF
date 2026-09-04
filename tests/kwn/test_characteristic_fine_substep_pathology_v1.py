"""Focused contracts for read-only CR1 fine-substep pathology diagnostics."""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from kwn_mvp.characteristic_pathology import (
    ClosureMapEvaluation,
    _exact_period,
    _trace_classification,
    capture_prestate,
    common_time_comparisons,
    evaluate_closure_map,
    neumaier_sum,
    pairwise_sum,
    precision_reduction_shadow,
    raw_picard_trace,
    repeatability_audit,
    scalar_map_scan,
    solver_state_hash,
    state_summary,
)
from kwn_mvp.radius_grid import RadiusGrid
from kwn_mvp.solver import SolverConfig
from scripts.run_kwn_cr1_fine_substep_pathology_v1 import (
    HASH_BOUND_CONTRACT_PATH,
    FixedRun,
    PathologyWorkflowError,
    RecordingCharacteristicReferenceSolver,
    _accepted_trial_evaluation,
    _formal_endpoint_observation,
    _root_cause,
    _same_prestate_different_dt,
    checkpoint_bound_config,
)


RMIN_M = 5.0e-9
RMAX_M = 5.0e-8


def _mapping(*, bins: int = 24) -> dict[str, object]:
    grid = RadiusGrid.logarithmic(RMIN_M, RMAX_M, bins)
    values = np.zeros(bins, dtype=np.float64)
    values[bins // 3 : bins // 3 + 3] = np.asarray([1.0, 2.0, 1.0]) * 1.0e18
    return {
        "simulation": {
            "temperature_K": 653.15,
            "max_dt_s": 1.0,
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
                    "cell_number_density_m3": [float(value) for value in values],
                },
                "nucleation": {"mode": "off"},
            },
        },
    }


class _ZeroVelocitySolver(RecordingCharacteristicReferenceSolver):
    def _velocity_at_radii(self, radii_m: np.ndarray, matrix_xb: float) -> np.ndarray:
        del matrix_xb
        return np.zeros(np.asarray(radii_m).shape, dtype=np.float64)


class _ConstantVelocitySolver(RecordingCharacteristicReferenceSolver):
    def _velocity_at_radii(self, radii_m: np.ndarray, matrix_xb: float) -> np.ndarray:
        del matrix_xb
        return np.full(np.asarray(radii_m).shape, 1.0e-12, dtype=np.float64)


class CharacteristicFineSubstepPathologyContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SolverConfig.from_mapping(_mapping())

    def _solver(self) -> _ZeroVelocitySolver:
        return _ZeroVelocitySolver(self.config)

    def _moving_solver(self) -> _ConstantVelocitySolver:
        return _ConstantVelocitySolver(self.config)

    def test_read_only_scalar_map_has_no_state_side_effect(self) -> None:
        solver = self._solver()
        prestate = capture_prestate(solver)
        before = solver_state_hash(solver)
        evaluation = evaluate_closure_map(
            solver, prestate=prestate, dt_s=0.25, x_trial=prestate.x_start
        )
        self.assertEqual(solver_state_hash(solver), before)
        self.assertLess(abs(evaluation.signed_f), 1.0e-15)

    def test_same_trial_is_bitwise_repeatable_ten_times(self) -> None:
        solver = self._moving_solver()
        prestate = capture_prestate(solver)
        rows, passed = repeatability_audit(
            solver, prestate=prestate, dt_s=0.25, x_trials=[prestate.x_start], repeats=10
        )
        self.assertTrue(passed)
        self.assertEqual(len(rows), 10)
        self.assertTrue(all(bool(row["bitwise_equal_to_first"]) for row in rows))

    def test_recorded_accepted_leaf_matches_committed_state(self) -> None:
        solver = self._moving_solver()
        old = capture_prestate(solver)
        cursor = solver.trial_cursor()
        diagnostic = solver.advance_one_ordinary_or_qualified_cycle(maximum_dt_s=0.25)
        evaluation = _accepted_trial_evaluation(
            solver, trials=solver.trials_since(cursor), old_cells=old.old_cell_number_m3,
            diagnostic=diagnostic,
        )
        widths = solver.population("beta").grid.widths_m
        np.testing.assert_array_equal(
            (evaluation.cell_number_m3 / widths) * widths, solver._beta_cell_numbers()
        )
        self.assertEqual(evaluation.x_closure, solver.matrix_xb)

    def test_formal_endpoint_ledger_uses_postcommit_state(self) -> None:
        """A one-ulp trial/commit round trip must not false-fail Phase-A."""

        state = {
            "population_array": np.asarray([2.0, 3.0], dtype=np.float64),
            "xB": 0.0062,
            "Q_total_mol_m3": 10.0,
            "Q_beta_mol_m3": 7.0,
            "Q_matrix_mol_m3": 3.0,
            "inventory_residual_mol_m3": 0.0,
            "inventory_relative_residual": 0.0,
            "M0_m3": 5.0,
            "M1_m2": 6.0,
            "M2_m": 7.0,
            "M3_dimensionless": 8.0,
            "Rmean_m": 9.0,
            "Rmean3_m3": 10.0,
            "Sv_m_inv": 11.0,
            "f_beta": 12.0,
        }
        accepted_row = {
            "step": 1,
            "physical_time_s": 0.25,
            "physical_time_h": 0.25 / 3600.0,
            "matrix_xB": 0.006200000000000001,
            "Q_total_mol_m3": 10.000000000000002,
            "Q_beta_mol_m3": 7.000000000000001,
            "Q_matrix_mol_m3": 3.000000000000001,
            "inventory_residual_mol_m3": 1.0e-15,
            "inventory_relative_residual": 1.0e-16,
            "cumulative_number_dissolution_m3": 0.0,
            "cumulative_beta_volume_dissolution": 0.0,
            "cumulative_mol_B_returned_mol_m3": 0.0,
            "state_hash": "postcommit-state-hash",
            "fixed_point_convergence_mode": "DIRECT",
            "fixed_point_periodic_cycle_period": 0,
            "xB_residual": 1.0e-14,
        }
        run = FixedRun(
            m=2,
            dt_s=0.25,
            status="PASS",
            accepted_states=(state,),
            accepted_rows=(accepted_row,),
            failure=None,
            failure_closure_trace=(),
            solver_at_failure_prestate=None,
            prestate_at_failure=None,
        )
        observation = _formal_endpoint_observation(run)
        self.assertEqual(observation["matrix_xB"], state["xB"])
        self.assertEqual(observation["Q_total_mol_m3"], state["Q_total_mol_m3"])
        self.assertEqual(observation["Q_beta_mol_m3"], state["Q_beta_mol_m3"])
        self.assertEqual(observation["Q_matrix_mol_m3"], state["Q_matrix_mol_m3"])
        self.assertEqual(
            observation["inventory_residual_mol_m3"], state["inventory_residual_mol_m3"]
        )

    def test_checkpoint_rebind_accepts_only_the_hash_bound_contract_path_relocation(self) -> None:
        mapping = _mapping()
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            contract_path = root / "runtime_contract.json"
            contract_path.write_text('{"contract":"frozen"}\n', encoding="utf-8")
            contract_sha256 = hashlib.sha256(contract_path.read_bytes()).hexdigest()
            contract_hash = "a" * 64
            mapping["thermodynamics"] = {
                "mode": "approximate_dilute",
                "planar_reference_xB": 0.006,
                "contract_path": str(contract_path),
                "contract_hash": contract_hash,
            }
            semantic_mapping = json.loads(json.dumps(mapping))
            semantic_mapping["thermodynamics"]["contract_path"] = HASH_BOUND_CONTRACT_PATH
            semantic_hash = hashlib.sha256(
                json.dumps(
                    semantic_mapping, ensure_ascii=True, separators=(",", ":"), sort_keys=True
                ).encode("utf-8")
            ).hexdigest()
            archived_mapping = json.loads(json.dumps(mapping))
            archived_mapping["thermodynamics"]["contract_path"] = "/retired/phase_a/contract.json"
            archived_config_hash = SolverConfig.from_mapping(archived_mapping).source_config_hash
            restart = root / "step244.npz"
            np.savez(
                restart,
                metadata_json=np.asarray(json.dumps({
                    "source_config_hash": archived_config_hash,
                    "validation_contract_hash": contract_hash,
                })),
            )
            restart_sha256 = hashlib.sha256(restart.read_bytes()).hexdigest()
            context = SimpleNamespace(mapping=mapping, contract_hash=contract_hash)
            bindings = {
                "EXPECTED_RESTART_CHECKPOINT_SHA256": restart_sha256,
                "EXPECTED_RESTART_SOURCE_CONFIG_HASH": archived_config_hash,
                "EXPECTED_RESTART_VALIDATION_CONTRACT_HASH": contract_hash,
                "EXPECTED_RESTART_CONTRACT_FILE_SHA256": contract_sha256,
                "EXPECTED_RESTART_SEMANTIC_CONFIG_HASH": semantic_hash,
                "FROZEN_RESTART_ARCHIVED_CONTRACT_PATH": "/retired/phase_a/contract.json",
            }
            with patch.multiple("scripts.run_kwn_cr1_fine_substep_pathology_v1", **bindings):
                rebound, binding = checkpoint_bound_config(
                    context=context, restart_checkpoint=restart
                )
                self.assertEqual(rebound.source_config_hash, archived_config_hash)
                self.assertEqual(rebound.validation_contract_path, str(contract_path))
                self.assertTrue(binding["path_only_rebind"])
                self.assertEqual(binding["status"], "PATH_ONLY_CONTRACT_LOCATION_REBIND")

                changed_mapping = json.loads(json.dumps(mapping))
                changed_mapping["matrix"]["initial_xB"] = 0.0063
                with self.assertRaises(PathologyWorkflowError):
                    checkpoint_bound_config(
                        context=SimpleNamespace(mapping=changed_mapping, contract_hash=contract_hash),
                        restart_checkpoint=restart,
                    )

                contract_path.write_text('{"contract":"changed"}\n', encoding="utf-8")
                with self.assertRaises(PathologyWorkflowError):
                    checkpoint_bound_config(context=context, restart_checkpoint=restart)

    def test_common_physical_time_alignment_uses_rational_fraction(self) -> None:
        solver = self._solver()
        state16 = state_summary(solver, fraction=Fraction(1, 16), m=16, substep=1)
        state32 = state_summary(solver, fraction=Fraction(2, 32), m=32, substep=2)
        rows = common_time_comparisons(
            {16: [state16], 32: [state32]}, edges_m=solver.population("beta").grid.edges_m
        )
        self.assertEqual(len(rows), 1)
        self.assertEqual((rows[0]["fraction_numerator"], rows[0]["fraction_denominator"]), (1, 16))
        self.assertEqual(rows[0]["population_L1"], 0.0)

    def test_raw_picard_cycle_detector_covers_p1_through_p32(self) -> None:
        for period in range(1, 33):
            keys = [(str(index),) for index in range(period)] * 2
            self.assertEqual(_exact_period(keys), period)
        rows = [
            {"signed_F": 1.0, "topology_signature": "A"},
            {"signed_F": -1.0, "topology_signature": "A"},
            {"signed_F": 1.0, "topology_signature": "A"},
            {"signed_F": -1.0, "topology_signature": "A"},
        ]
        summary = _trace_classification(rows, [("a",), ("b",), ("a",), ("b",)])
        self.assertEqual(summary["classification"], "P2")

    def test_raw_picard_trace_does_not_commit_or_advance(self) -> None:
        solver = self._solver()
        prestate = capture_prestate(solver)
        before = solver_state_hash(solver)
        rows, summary, _evaluations = raw_picard_trace(
            solver, prestate=prestate, dt_s=0.25, maximum_iterations=8
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(summary["exact_period"], 1)
        self.assertEqual(solver_state_hash(solver), before)
        self.assertEqual(solver.step, 0)

    def test_scalar_scan_is_deterministic_and_single_partition_for_identity(self) -> None:
        solver = self._solver()
        prestate = capture_prestate(solver)
        first_rows, first_transitions, first_summary = scalar_map_scan(
            solver, prestate=prestate, dt_s=0.25, coarse_points=8, label="identity"
        )
        second_rows, second_transitions, second_summary = scalar_map_scan(
            solver, prestate=prestate, dt_s=0.25, coarse_points=8, label="identity"
        )
        self.assertEqual(first_rows, second_rows)
        self.assertEqual(first_transitions, second_transitions)
        self.assertEqual(first_summary, second_summary)
        self.assertEqual(first_summary["NUMBER_OF_TOPOLOGY_PARTITIONS"], 1)

    def test_topology_transition_detector_reports_face_and_source_cells(self) -> None:
        solver = self._solver()
        prestate = capture_prestate(solver)
        edges = solver.population("beta").grid.edges_m
        count = edges.size - 1

        def fake_evaluation(
            _solver: _ZeroVelocitySolver, *, prestate: object, dt_s: float, x_trial: float
        ) -> ClosureMapEvaluation:
            del prestate, dt_s
            right = float(x_trial) >= 0.5
            source = np.zeros(edges.size, dtype=np.int64)
            if right:
                source[1] = 1
            departure = edges.copy()
            cells = np.ones(count, dtype=np.float64)
            cdf = np.cumsum(cells) / float(np.sum(cells))
            return ClosureMapEvaluation(
                x_trial=float(x_trial), x_closure=float(x_trial), signed_f=float(x_trial - 0.5),
                x_tolerance=1.0e-12, midpoint_xb=float(x_trial), cell_number_m3=cells,
                cdf=cdf, cdf_signature="cdf", population_hash="population",
                departure_signature="right" if right else "left",
                remap_topology_signature="right" if right else "left",
                trace_topology_signature="trace", trace_topology_mode="TEST",
                departure_faces_m=departure, source_cell_indices=source,
                M0_m3=1.0, M1_m2=1.0, M2_m=1.0, M3_dimensionless=1.0,
                Rmean_m=1.0, Rmean3_m3=1.0, Sv_m_inv=1.0, f_beta=1.0,
                Q_beta_mol_m3=1.0, Q_matrix_mol_m3=1.0, Q_total_mol_m3=2.0,
                inventory_relative_residual=0.0, cell_measure=0.0,
                critical_radius_m=None, lower_tail_M0_m3=0.0, lower_tail_M3_dimensionless=0.0,
            )

        with patch("kwn_mvp.characteristic_pathology.evaluate_closure_map", side_effect=fake_evaluation):
            _rows, transitions, summary = scalar_map_scan(
                solver, prestate=prestate, dt_s=0.25, coarse_points=4,
                raw_update_intervals=[(0.4, 0.6)], label="synthetic",
            )
        self.assertGreater(summary["NUMBER_OF_TOPOLOGY_PARTITIONS"], 1)
        self.assertTrue(transitions)
        self.assertGreaterEqual(int(transitions[0]["face_index"]), 0)
        self.assertNotEqual(transitions[0]["source_cell_left"], transitions[0]["source_cell_right"])
        self.assertTrue(transitions[0]["local_endpoint_bitwise_repeatable"])
        self.assertFalse(transitions[0]["local_one_sided_F_delta_persistent"])

    def test_unrelated_global_remap_jump_does_not_become_closure_root_cause(self) -> None:
        scan = {"NUMBER_OF_OBSERVED_SIGN_CROSSINGS": 1}
        raw = {"classification": "P4"}
        precision = {"PRECISION_FLOOR_STATUS": "FLOAT64_RESIDUAL_STABLE"}
        base_transition = {
            "map_jump_observed": True,
            "sign_change_observed": False,
            "departure_cell_crossing": True,
            "single_departure_cell_crossing": True,
            "same_trace_topology": True,
            "raw_update_intersects_transition_interval": False,
            "local_one_sided_F_delta_persistent": False,
            "local_endpoint_bitwise_repeatable": False,
            "face_index": 1,
            "source_cell_left": 1,
            "source_cell_right": 2,
            "label": "m32_failure_prestate",
        }
        unrelated = _root_cause(
            prefailure="PREFAILURE_STATES_ASYMPTOTICALLY_ALIGNED",
            m32_scan=scan, m64_scan=scan, m32_raw=raw, m64_raw=raw,
            precision=precision, transitions=[base_transition],
        )
        self.assertEqual(unrelated["PRIMARY_ROOT_CAUSE"], "CLOSURE_SOLVER_PATHOLOGY_ON_SMOOTH_MAP")
        relevant = _root_cause(
            prefailure="PREFAILURE_STATES_ASYMPTOTICALLY_ALIGNED",
            m32_scan=scan, m64_scan=scan, m32_raw=raw, m64_raw=raw,
            precision=precision, transitions=[
                {
                    **base_transition,
                    "raw_update_intersects_transition_interval": True,
                    "local_one_sided_F_delta_persistent": True,
                    "local_endpoint_bitwise_repeatable": True,
                },
                {
                    **base_transition,
                    "label": "m64_failure_prestate",
                    "raw_update_intersects_transition_interval": True,
                    "local_one_sided_F_delta_persistent": True,
                    "local_endpoint_bitwise_repeatable": True,
                },
            ],
        )
        self.assertEqual(relevant["PRIMARY_ROOT_CAUSE"], "CR1_REMAP_NONSMOOTH_MULTIBRANCH_MAP")

    def test_compensated_reduction_recovers_cancellation_fixture(self) -> None:
        values = np.asarray([1.0e16, 1.0, -1.0e16], dtype=np.float64)
        self.assertNotEqual(pairwise_sum(values), 1.0)
        self.assertEqual(neumaier_sum(values), 1.0)

    def test_extended_precision_shadow_is_diagnostic_only(self) -> None:
        solver = self._solver()
        prestate = capture_prestate(solver)
        before = solver_state_hash(solver)
        evaluation = evaluate_closure_map(
            solver, prestate=prestate, dt_s=0.25, x_trial=prestate.x_start
        )
        shadow = precision_reduction_shadow(
            solver, evaluation, previous_cells=prestate.old_cell_number_m3
        )
        self.assertIn("signed_F_longdouble", shadow)
        self.assertTrue(math.isfinite(float(shadow["signed_F_shadow_delta"])))
        self.assertEqual(solver_state_hash(solver), before)

    def test_same_prestate_different_dt_never_advances_the_state(self) -> None:
        solver = self._solver()
        prestate = capture_prestate(solver)
        before = solver_state_hash(solver)
        rows, summary, evidence = _same_prestate_different_dt(solver=solver, prestate=prestate)
        self.assertEqual(len(rows), 4)
        self.assertGreater(len(evidence), 0)
        self.assertIn("trend", summary)
        self.assertTrue(all(bool(row["diagnostic_only_no_state_advance"]) for row in rows))
        self.assertEqual(solver_state_hash(solver), before)


if __name__ == "__main__":
    unittest.main()
