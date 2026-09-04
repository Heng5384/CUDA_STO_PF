"""Focused contracts for the CR1 read-only semigroup diagnostic layer."""

from __future__ import annotations

from fractions import Fraction
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from kwn_mvp.characteristic_dt_continuation import accepted_state_hash
from kwn_mvp.characteristic_reference import CharacteristicReferenceSolver
from kwn_mvp.characteristic_semigroup import (
    PiecewiseLinearMatrixTrajectory,
    classify_semigroup_sequence,
    event_error_localization,
    phi_compose,
    physical_state_distance,
    semigroup_triplet,
    topology_event_proximity_rows,
)
from kwn_mvp.radius_grid import RadiusGrid
from kwn_mvp.solver import SolverConfig
from scripts import run_kwn_cr1_semigroup_audit_v1 as runner


def _mapping(*, bins: int = 16) -> dict[str, object]:
    grid = RadiusGrid.logarithmic(5.0e-9, 5.0e-8, bins)
    values = np.zeros(bins, dtype=np.float64)
    values[bins // 3 : bins // 3 + 3] = np.asarray([1.0, 2.0, 1.0]) * 1.0e18
    return {
        "simulation": {
            "temperature_K": 653.15, "max_dt_s": 1.0, "min_dt_s": 1.0e-16,
            "size_cfl": 0.35, "cfl_active_inventory_relative_threshold": 1.0e-12,
            "rmax_outflow_relative_tolerance": 1.0e-12, "population_measure": "cell_integrated",
        },
        "matrix": {
            "molar_volume_m3_mol": 4.1009e-5, "initial_xB": 0.0062,
            "total_b_mol_m3": None, "inventory_tolerance_relative": 1.0e-12,
        },
        "radius_grid": {"minimum_m": 5.0e-9, "maximum_m": 5.0e-8, "bins": bins},
        "thermodynamics": {"mode": "approximate_dilute", "planar_reference_xB": 0.006},
        "populations": {
            "g": {
                "xB": 0.02, "molar_volume_m3_mol": 4.1009e-5, "diffusivity_m2_s": 0.0,
                "gamma_j_m2": 0.0, "xeq_infinity": 0.006, "initial": {"kind": "empty"},
                "nucleation": {"mode": "off"},
            },
            "beta": {
                "xB": 0.8, "molar_volume_m3_mol": 4.1009e-5, "diffusivity_m2_s": 1.0e-19,
                "gamma_j_m2": 0.0, "xeq_infinity": 0.006,
                "initial": {
                    "kind": "cell_integrated", "radius_edges_m": [float(item) for item in grid.edges_m],
                    "cell_number_density_m3": [float(item) for item in values],
                },
                "nucleation": {"mode": "off"},
            },
        },
    }


def _zero_velocity(_self: object, radii_m: np.ndarray, _matrix_xb: float) -> np.ndarray:
    return np.zeros(np.asarray(radii_m).shape, dtype=np.float64)


def _constant_velocity(_self: object, radii_m: np.ndarray, _matrix_xb: float) -> np.ndarray:
    return np.full(np.asarray(radii_m).shape, 1.0e-12, dtype=np.float64)


def _state(*, source: int = 0, cells: tuple[float, ...] = (1.0, 2.0, 3.0)) -> dict[str, object]:
    array = np.asarray(cells, dtype=np.float64)
    cdf = np.cumsum(array) / float(np.sum(array))
    return {
        "population_array": array, "cdf": cdf, "xB": 0.0062,
        "M0_m3": 6.0, "M1_m2": 7.0, "M2_m": 8.0, "M3_dimensionless": 9.0,
        "Rmean_m": 1.0e-8, "Rmean3_m3": 2.0e-24, "Sv_m_inv": 3.0e8, "f_beta": 1.0e-4,
        "Q_beta_mol_m3": 4.0, "Q_matrix_mol_m3": 5.0, "Q_total_mol_m3": 9.0,
        "departure_faces_m": np.asarray([5.0e-9, 1.0e-8, 2.0e-8, 5.0e-8]),
        "source_cell_indices": np.asarray([0, source, 1, 2], dtype=np.int64),
        "remap_topology_signature": f"topology-{source}",
        "trace_topology_signature": "trace", "departure_signature": "departure",
    }


class CharacteristicSemigroupAuditContracts(unittest.TestCase):
    def setUp(self) -> None:
        self.config = SolverConfig.from_mapping(_mapping())

    def _source(self) -> CharacteristicReferenceSolver:
        return CharacteristicReferenceSolver(self.config)

    @patch.object(CharacteristicReferenceSolver, "_velocity_at_radii", new=_zero_velocity)
    def test_phi_h_is_read_only_and_deterministic(self) -> None:
        source = self._source()
        before_hash = accepted_state_hash(source)
        before_history = tuple(source.history)
        first = phi_compose(source, dt_s=0.25, count=1)
        second = phi_compose(source, dt_s=0.25, count=1)
        self.assertEqual(first.status, "SUCCESS")
        self.assertEqual(second.status, "SUCCESS")
        self.assertEqual(accepted_state_hash(source), before_hash)
        self.assertEqual(tuple(source.history), before_history)
        np.testing.assert_array_equal(first.state["population_array"], second.state["population_array"])
        self.assertEqual(first.state["accepted_state_hash"], second.state["accepted_state_hash"])

    @patch.object(CharacteristicReferenceSolver, "_velocity_at_radii", new=_zero_velocity)
    def test_two_half_step_semigroup_harness(self) -> None:
        triplet = semigroup_triplet(self._source(), h_s=0.25, mode="DYNAMIC")
        self.assertEqual(triplet["status"], "SUCCESS")
        self.assertEqual(float(triplet["E_h"]), 0.0)
        self.assertEqual(triplet["topology_path_h"]["leaf_count"], 1)
        self.assertEqual(triplet["topology_path_h2x2"]["leaf_count"], 2)
        self.assertEqual(triplet["topology_path_h4x4"]["leaf_count"], 4)
        self.assertTrue(triplet["topology_path_h_vs_h2x2"]["comparison_is_not_a_physical_semigroup_gate"])
        summary = classify_semigroup_sequence([
            {"status": "SUCCESS", "E_h": 4.0},
            {"status": "SUCCESS", "E_h": 2.0},
            {"status": "SUCCESS", "E_h": 1.0},
        ])
        self.assertEqual(summary["classification"], "PHYSICAL_SEMIGROUP_CONVERGENCE")

    @patch.object(CharacteristicReferenceSolver, "_velocity_at_radii", new=_zero_velocity)
    def test_frozen_matrix_isolation_keeps_xb_fixed(self) -> None:
        source = self._source()
        result = phi_compose(
            source, dt_s=0.25, count=2, mode="FROZEN_MATRIX", frozen_matrix_xb=source.matrix_xb
        )
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.state["xB"], source.matrix_xb)

    @patch.object(CharacteristicReferenceSolver, "_velocity_at_radii", new=_constant_velocity)
    def test_external_matrix_paths_report_open_inventory_without_nonclosure(self) -> None:
        source = self._source()
        frozen = phi_compose(
            source, dt_s=0.5, count=1, mode="FROZEN_MATRIX", frozen_matrix_xb=source.matrix_xb
        )
        trajectory = PiecewiseLinearMatrixTrajectory(
            np.asarray([0.0, 0.5]), np.asarray([source.matrix_xb, source.matrix_xb])
        )
        prescribed = phi_compose(
            source, dt_s=0.5, count=1, mode="PRESCRIBED_X", prescribed_matrix_trajectory=trajectory
        )
        self.assertEqual(frozen.status, "SUCCESS")
        self.assertEqual(prescribed.status, "SUCCESS")
        self.assertGreater(float(frozen.state["inventory_relative_residual"]), self.config.inventory_tolerance_relative)
        self.assertGreater(float(prescribed.state["inventory_relative_residual"]), self.config.inventory_tolerance_relative)

    @patch.object(CharacteristicReferenceSolver, "_velocity_at_radii", new=_zero_velocity)
    def test_prescribed_x_reader_uses_the_shared_nodes(self) -> None:
        source = self._source()
        trajectory = PiecewiseLinearMatrixTrajectory(
            np.asarray([0.0, 0.25, 0.5]),
            np.asarray([source.matrix_xb, source.matrix_xb, source.matrix_xb]),
        )
        result = phi_compose(
            source, dt_s=0.25, count=2, mode="PRESCRIBED_X", prescribed_matrix_trajectory=trajectory
        )
        self.assertEqual(result.status, "SUCCESS")
        self.assertEqual(result.state["xB"], trajectory.value(0.5))

    def test_physical_state_distance_reports_all_requested_metrics(self) -> None:
        left, right = _state(cells=(1.0, 2.0, 3.0)), _state(cells=(1.0, 2.25, 2.75))
        values = physical_state_distance(left, right, edges_m=np.asarray([5.0e-9, 1.0e-8, 2.0e-8, 5.0e-8]))
        for field in ("xB_error", "M0_m3_error", "M3_dimensionless_error", "population_L1", "PSD_Wasserstein_m", "E_physical_max_relative"):
            self.assertIn(field, values)

    def test_topology_event_proximity_detects_changed_faces(self) -> None:
        states = {"m16": _state(source=0), "m32": _state(source=1), "m64": _state(source=1)}
        rows = topology_event_proximity_rows(states, edges_m=np.asarray([5.0e-9, 1.0e-8, 2.0e-8, 5.0e-8]))
        self.assertEqual({row["face_index"] for row in rows}, {1})
        self.assertTrue(all("eta_abs" in row for row in rows))

    def test_event_boundary_selection_uses_the_explicit_grid(self) -> None:
        states = {"m16": _state(source=0), "m32": _state(source=1), "m64": _state(source=1)}
        boundaries = runner._event_boundaries(
            states, edges_m=np.asarray([5.0e-9, 1.0e-8, 2.0e-8, 5.0e-8])
        )
        self.assertEqual(len(boundaries), 1)
        self.assertEqual(boundaries[0]["face_index"], 1)
        self.assertEqual(boundaries[0]["source_boundary_m"], 1.0e-8)
        self.assertEqual(boundaries[0]["bracketed_boundary_count"], 1)

    def test_event_localization_uses_absolute_measure_error(self) -> None:
        left = _state(source=0, cells=(1.0, 2.0, 3.0))
        right = _state(source=1, cells=(1.25, 1.5, 3.0))
        result = event_error_localization(
            left, right, edges_m=np.asarray([5.0e-9, 1.0e-8, 2.0e-8, 5.0e-8])
        )
        self.assertGreater(result["event_neighborhood_population_L1_fraction"], 0.0)
        self.assertGreaterEqual(result["event_neighborhood_M3_fraction"], 0.0)

    def test_topology_event_bisection_is_diagnostic_only(self) -> None:
        source = SimpleNamespace(time_s=1.0)

        def fake_departure(
            _source: object, *, duration_s: float, face_index: int, mode: str,
            trajectory: object | None,
        ) -> tuple[float | None, str | None]:
            del _source, face_index, mode, trajectory
            return duration_s, None

        with patch.object(runner, "_event_departure", side_effect=fake_departure):
            result = runner._estimate_event_time(
                source, face_index=3, boundary_m=0.0005, mode="DYNAMIC", trajectory=None
            )
        self.assertEqual(result["status"], "EVENT_TIME_ESTIMATED")
        self.assertTrue(result["includes_step244_identity_departure"])
        self.assertTrue(result["diagnostic_only_no_trajectory_change"])

    def test_event_ordering_comparison_uses_existing_accepted_states(self) -> None:
        def accepted(time_s: float, first: int, second: int) -> dict[str, object]:
            return {
                "physical_time_s": time_s,
                "_accepted_evaluation": SimpleNamespace(source_cell_indices=np.asarray([0, first, second])),
            }
        runs = {
            16: SimpleNamespace(accepted_states=(accepted(0.25, 0, 1), accepted(0.5, 0, 2))),
            32: SimpleNamespace(accepted_states=(accepted(0.125, 0, 1), accepted(0.25, 0, 2), accepted(0.375, 0, 3))),
            64: SimpleNamespace(accepted_states=(accepted(0.0625, 0, 1), accepted(0.125, 0, 1), accepted(0.1875, 0, 2), accepted(0.25, 0, 2))),
        }
        rows, summary = runner._event_timeline(
            runs, edges_m=np.asarray([0.0, 0.5, 1.0])
        )
        self.assertTrue(rows)
        self.assertIn("event_order", summary)
        self.assertEqual(summary["accepted_leaf_counts_within_common_horizon"], {"m16": 1, "m32": 2, "m64": 4})
        self.assertTrue(any(row["route"] == "m16" for row in rows))

    def test_event_time_spread_does_not_mix_distinct_faces(self) -> None:
        summary = runner._event_time_spread_summary([
            {"status": "EVENT_TIME_ESTIMATED", "mode": "DYNAMIC", "face_index": 1,
             "source_boundary_m": 1.0, "event_physical_time_s": 3.0},
            {"status": "EVENT_TIME_ESTIMATED", "mode": "FROZEN_MATRIX", "face_index": 1,
             "source_boundary_m": 1.0, "event_physical_time_s": 3.25},
            {"status": "EVENT_TIME_ESTIMATED", "mode": "DYNAMIC", "face_index": 2,
             "source_boundary_m": 2.0, "event_physical_time_s": 9.0},
        ])
        self.assertEqual(summary["maximum_same_event_mode_spread_s"], 0.25)

    def test_cr1_cause_does_not_borrow_dynamic_localization(self) -> None:
        result = runner._root_cause(
            dynamic={"classification": "PHYSICAL_SEMIGROUP_DIVERGENCE"},
            frozen={"classification": "PHYSICAL_SEMIGROUP_STAGNATION"},
            prescribed={"classification": "PHYSICAL_SEMIGROUP_CONVERGENCE"},
            topology_differs_dynamic=True,
            topology_differs_frozen=False,
            local={
                "LOCAL_ROOT_FAMILY": "LOCAL_CONTINUOUS_ROOT_FAMILY_NOT_OBSERVED",
                "DELTA_X_OVER_H_CONVERGENCE": "NOT_ESTABLISHED",
            },
            timeline={"event_order": "SAME_EVENTS_DIFFERENT_DISCRETE_TIMING"},
            localization=[{"event_neighborhood_population_L1_fraction": 1.0}],
        )
        self.assertNotEqual(result["PRIMARY_ROOT_CAUSE"], "CR1_REMAP_SEMIGROUP_DEFECT")

    def test_frozen_topology_evidence_includes_the_finest_path(self) -> None:
        rows = [{
            "status": "SUCCESS",
            "topology_path_h_vs_h2x2": {"different_discrete_topology_regimes": False},
            "topology_path_h2x2_vs_h4x4": {"different_discrete_topology_regimes": True},
        }]
        self.assertTrue(runner._frozen_topology_regimes_differ(rows))

    def test_prescribed_x_failure_after_frozen_convergence_is_not_insufficient(self) -> None:
        result = runner._root_cause(
            dynamic={"classification": "PHYSICAL_SEMIGROUP_STAGNATION"},
            frozen={"classification": "PHYSICAL_SEMIGROUP_CONVERGENCE"},
            prescribed={"classification": "PHYSICAL_SEMIGROUP_DIVERGENCE"},
            topology_differs_dynamic=False,
            topology_differs_frozen=False,
            local={
                "LOCAL_ROOT_FAMILY": "LOCAL_CONTINUOUS_ROOT_FAMILY_NOT_OBSERVED",
                "DELTA_X_OVER_H_CONVERGENCE": "NOT_ESTABLISHED",
            },
            timeline={"event_order": "SAME_EVENTS_DIFFERENT_DISCRETE_TIMING"},
            localization=[],
        )
        self.assertEqual(result["PRIMARY_ROOT_CAUSE"], "MIXED_TIME_REMAP_COUPLING_DEFECT")

    def test_insufficient_semigroup_levels_do_not_become_an_event_or_mixed_defect(self) -> None:
        result = runner._root_cause(
            dynamic={"classification": "INSUFFICIENT_SUCCESSFUL_LEVELS"},
            frozen={"classification": "INSUFFICIENT_SUCCESSFUL_LEVELS"},
            prescribed={"classification": "INSUFFICIENT_SUCCESSFUL_LEVELS"},
            topology_differs_dynamic=True,
            topology_differs_frozen=True,
            local={
                "LOCAL_ROOT_FAMILY": "LOCAL_CONTINUOUS_ROOT_FAMILY_NOT_OBSERVED",
                "DELTA_X_OVER_H_CONVERGENCE": "NOT_ESTABLISHED",
            },
            timeline={"event_order": "DIFFERENT_EVENT_ORDER"},
            localization=[],
        )
        self.assertEqual(result["PRIMARY_ROOT_CAUSE"], "INSUFFICIENT_SEMIGROUP_EVIDENCE")

    def test_baseline_archive_receives_accepted_state_sequences(self) -> None:
        states = {multiplicity: ("state", multiplicity) for multiplicity in (16, 32, 64)}
        runs = {
            multiplicity: SimpleNamespace(accepted_states=states[multiplicity])
            for multiplicity in states
        }
        self.assertEqual(runner._accepted_states_by_m(runs), states)

    def test_semigroup_csv_exposes_changed_face_indices(self) -> None:
        rows = runner._semigroup_csv_rows([{
            "mode": "DYNAMIC", "h_s": 0.25, "status": "SUCCESS",
            "topology_h_vs_two": {
                "topology_equal": False, "changed_face_count": 2,
                "changed_face_indices": np.asarray([3, 7], dtype=np.int64),
            },
        }])
        self.assertEqual(rows[0]["topology_h_vs_two_changed_face_indices_json"], "[3,7]")

    @patch.object(CharacteristicReferenceSolver, "_velocity_at_radii", new=_zero_velocity)
    def test_local_root_family_diagnostic_does_not_accept_a_root(self) -> None:
        source = self._source()
        trajectory = PiecewiseLinearMatrixTrajectory(
            np.asarray([0.0, 0.25]), np.asarray([source.matrix_xb, source.matrix_xb])
        )
        rows, summary = runner._local_root_family(source, trajectory=trajectory)
        self.assertEqual(rows, [])
        self.assertEqual(summary["LOCAL_ROOT_FAMILY"], "NO_NONZERO_OBSERVED_DXDT_SCALE")

    @patch.object(CharacteristicReferenceSolver, "_velocity_at_radii", new=_zero_velocity)
    def test_semigroup_calls_have_no_cumulative_or_history_side_effect(self) -> None:
        source = self._source()
        before = source.state_arrays()
        before_history = tuple(source.history)
        result = semigroup_triplet(source, h_s=0.25, mode="FROZEN_MATRIX", frozen_matrix_xb=source.matrix_xb)
        self.assertEqual(result["status"], "SUCCESS")
        after = source.state_arrays()
        for name in before:
            np.testing.assert_array_equal(before[name], after[name])
        self.assertEqual(tuple(source.history), before_history)


if __name__ == "__main__":
    unittest.main()
