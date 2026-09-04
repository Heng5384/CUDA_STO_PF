"""Contracts for the read-only frozen CR1 semigroup decomposition layer.

These are deliberately small synthetic/operator tests.  They neither load the
PbTe restart nor advance a production solver; their purpose is to make the
three-path diagnostic algebra and topology evidence independently testable.
"""

from __future__ import annotations

import math
import unittest

import numpy as np

from kwn_mvp.conservative_remap import trace_departure_faces_rk2
from kwn_mvp.frozen_semigroup_decomposition import (
    FrozenSemigroupDecomposition,
    FrozenSemigroupDecompositionError,
    additive_residual_metrics,
    canonical_table_half_flow_query,
    changed_face_indices,
    decompose_frozen_cr1,
    defect_metrics,
    face_flow_rows,
    operator_audit,
    projection_connectivity_rows,
    search_zero_event_control,
    support_expansion_rows,
    support_from_faces,
    support_pairing_metrics,
    synthetic_single_event_control,
)


def _synthetic_edges(*, bins: int = 16) -> np.ndarray:
    return np.geomspace(5.0e-9, 5.0e-8, bins + 1, dtype=np.float64)


def _synthetic_cells(edges: np.ndarray) -> np.ndarray:
    centres = np.sqrt(edges[:-1] * edges[1:])
    log_centres = np.log(centres)
    smooth = np.exp(-0.5 * ((log_centres - float(np.mean(log_centres))) / 0.45) ** 2)
    return np.asarray(smooth / np.sum(smooth, dtype=np.float64) * 1.0e18, dtype=np.float64)


def _nonlinear_velocity(radii_m: np.ndarray) -> np.ndarray:
    """The smooth autonomous velocity used by the module's synthetic control."""

    lower, upper = 5.0e-9, 5.0e-8
    normalized = (np.asarray(radii_m, dtype=np.float64) - lower) / (upper - lower)
    return 3.0e-10 * (0.15 + normalized * normalized)


def _zero_velocity(radii_m: np.ndarray) -> np.ndarray:
    return np.zeros(np.asarray(radii_m).shape, dtype=np.float64)


class FrozenSemigroupDecompositionContracts(unittest.TestCase):
    """A/B/C, support, control, and sparse-operator contracts."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.base_edges = _synthetic_edges()
        cls.base_initial = _synthetic_cells(cls.base_edges)
        cls.base_h_s = 0.2
        cls.base_decomposition = decompose_frozen_cr1(
            edges_m=cls.base_edges,
            initial_cells=cls.base_initial,
            h_s=cls.base_h_s,
            velocity_m_s=_nonlinear_velocity,
        )

    @classmethod
    def _event_case(cls) -> tuple[FrozenSemigroupDecomposition, dict[str, object], np.ndarray, np.ndarray]:
        """Load the deterministic synthetic topology control only when needed."""

        cached = getattr(cls, "_event_case_cache", None)
        if cached is None:
            decomposition, summary = synthetic_single_event_control()
            if decomposition is None:
                raise AssertionError(f"synthetic CR1 event control was unavailable: {summary}")
            edges = np.asarray(decomposition.direct.trace.arrival_faces_m, dtype=np.float64)
            cached = (decomposition, summary, edges, _synthetic_cells(edges))
            cls._event_case_cache = cached
        return cached

    def _base_repeat(self) -> FrozenSemigroupDecomposition:
        return decompose_frozen_cr1(
            edges_m=self.base_edges,
            initial_cells=self.base_initial,
            h_s=self.base_h_s,
            velocity_m_s=_nonlinear_velocity,
        )

    def _event_repeat(self) -> FrozenSemigroupDecomposition:
        _decomposition, summary, edges, initial = self._event_case()
        return decompose_frozen_cr1(
            edges_m=edges,
            initial_cells=initial,
            h_s=float(summary["selected_h_s"]),
            velocity_m_s=_nonlinear_velocity,
        )

    @staticmethod
    def _assert_path_equal(
        left: FrozenSemigroupDecomposition,
        right: FrozenSemigroupDecomposition,
        attribute: str,
    ) -> None:
        first = getattr(left, attribute)
        second = getattr(right, attribute)
        np.testing.assert_array_equal(first.cells, second.cells)
        np.testing.assert_array_equal(first.trace.arrival_faces_m, second.trace.arrival_faces_m)
        np.testing.assert_array_equal(first.trace.departure_faces_m, second.trace.departure_faces_m)
        np.testing.assert_array_equal(
            first.partition.source_cell_indices, second.partition.source_cell_indices
        )

    def test_direct_flow_diagnostic_is_deterministic(self) -> None:
        first, second = self._base_repeat(), self._base_repeat()
        self._assert_path_equal(first, second, "direct")

    def test_composed_flow_diagnostic_is_deterministic(self) -> None:
        first, second = self._base_repeat(), self._base_repeat()
        self._assert_path_equal(first, second, "composed_flow")

    def test_sequential_half_diagnostic_is_deterministic(self) -> None:
        first, second = self._base_repeat(), self._base_repeat()
        self._assert_path_equal(first, second, "sequential")
        self.assertIsNotNone(first.sequential.intermediate_cells)
        np.testing.assert_array_equal(
            first.sequential.intermediate_cells, second.sequential.intermediate_cells
        )

    def test_additive_defect_identity_closes_at_float64_roundoff(self) -> None:
        decomposition = self.base_decomposition
        np.testing.assert_allclose(
            decomposition.d_total,
            decomposition.d_trace + decomposition.d_remap,
            rtol=0.0,
            atol=0.0,
        )
        metrics = additive_residual_metrics(
            decomposition.additive_residual,
            reference_cells=self.base_initial,
        )
        # The three subtract operations have an exact algebraic cancellation
        # in this implementation.  The scale check makes any future nonzero
        # residual accountable as float64 arithmetic rather than a solver
        # tolerance change.
        float64_bound = 64.0 * np.finfo(np.float64).eps
        self.assertLessEqual(float(metrics["L1_relative"]), float64_bound)
        self.assertLessEqual(float(metrics["Linf_relative"]), float64_bound)

    def test_composed_flow_uses_the_physical_half_departure_not_offset_addition(self) -> None:
        decomposition = self.base_decomposition
        # Path B makes one final remap, so its saved trace is the composed
        # trace itself.  The independently retained first-half trace belongs
        # to Path C, and is the same frozen physical map used to build B.
        first_half = decomposition.sequential.first_half_trace
        self.assertIsNotNone(first_half)
        np.testing.assert_array_equal(
            decomposition.composed_flow.trace.midpoint_faces_m,
            first_half.departure_faces_m,
        )

        points = np.asarray(first_half.departure_faces_m, dtype=np.float64)
        continued, audit = canonical_table_half_flow_query(
            canonical_edges_m=self.base_edges,
            query_points_m=points,
            half_dt_s=0.5 * float(decomposition.h_s),
            velocity_m_s=_nonlinear_velocity,
            canonical_public_trace=first_half,
        )
        np.testing.assert_array_equal(
            decomposition.composed_flow.trace.departure_faces_m,
            continued,
        )
        self.assertTrue(bool(audit["canonical_edge_bitwise_parity"]))
        self.assertTrue(bool(audit["no_augmented_trace_mesh"]))

        naive_double_offset = 2.0 * points - self.base_edges
        interior = slice(2, -2)
        self.assertGreater(
            float(
                np.max(
                    np.abs(
                        decomposition.composed_flow.trace.departure_faces_m[interior]
                        - naive_double_offset[interior]
                    )
                )
            ),
            0.0,
        )

    def test_face_flow_rows_expose_all_faces_and_geometric_changed_sets(self) -> None:
        decomposition, _summary, edges, _initial = self._event_case()
        rows = face_flow_rows(decomposition, edges_m=edges)
        self.assertEqual(len(rows), edges.size)
        required = {
            "arrival_face_radius_m",
            "direct_departure_radius_m",
            "composed_departure_radius_m",
            "delta_departure_over_local_cell_width",
            "direct_source_cell_index",
            "composed_source_cell_index",
            "source_cell_changed",
            "S_near_boundary",
        }
        self.assertTrue(required.issubset(rows[0]))
        observed_changed = {
            int(row["face_index"]) for row in rows if bool(row["source_cell_changed"])
        }
        expected_changed = {int(face) for face in changed_face_indices(decomposition)}
        self.assertEqual(observed_changed, expected_changed)
        self.assertTrue(expected_changed)
        self.assertTrue(
            all(math.isfinite(float(row["near_boundary_threshold_m"])) for row in rows)
        )

    def test_projection_connectivity_uses_geometry_not_population_thresholds(self) -> None:
        decomposition, summary, edges, initial = self._event_case()
        rows, changed_cells = projection_connectivity_rows(decomposition)
        self.assertTrue(rows)
        self.assertGreater(changed_cells.size, 0)
        required = {
            "final_destination_cell",
            "old_source_cell",
            "intermediate_cells_json",
            "composed_flow_direct_weight",
            "sequential_weight",
            "S_projection_changed_cell",
        }
        self.assertTrue(required.issubset(rows[0]))
        self.assertEqual(
            {int(row["final_destination_cell"]) for row in rows if bool(row["S_projection_changed_cell"])},
            {int(cell) for cell in changed_cells},
        )

        # The support classification must be invariant to the population
        # values: alter the smooth population without changing grid, h, or G.
        alternate = np.roll(initial, 3).copy()
        alternate *= 1.37
        alternate_decomposition = decompose_frozen_cr1(
            edges_m=edges,
            initial_cells=alternate,
            h_s=float(summary["selected_h_s"]),
            velocity_m_s=_nonlinear_velocity,
        )
        _alternate_rows, alternate_changed = projection_connectivity_rows(alternate_decomposition)
        np.testing.assert_array_equal(changed_cells, alternate_changed)

    def test_changed_support_pairing_reports_exact_and_all_required_halos(self) -> None:
        decomposition, _summary, edges, _initial = self._event_case()
        flow_faces = changed_face_indices(decomposition)
        exact_support = support_from_faces(
            flow_faces, cell_count=decomposition.d_trace.size, halo=0
        )
        halo_one = support_from_faces(flow_faces, cell_count=decomposition.d_trace.size, halo=1)
        halo_two = support_from_faces(flow_faces, cell_count=decomposition.d_trace.size, halo=2)
        self.assertTrue(np.all(exact_support <= halo_one))
        self.assertTrue(np.all(halo_one <= halo_two))

        trace_pairing = support_pairing_metrics(
            decomposition.d_trace, edges_m=edges, support=exact_support
        )
        self.assertIn("support_population_L1_fraction", trace_pairing)
        self.assertIn("support_absolute_weighted_M3_fraction", trace_pairing)

        rows = support_expansion_rows(
            decomposition.d_trace,
            edges_m=edges,
            seed_cells=np.flatnonzero(exact_support),
            kind="D_trace",
            h_s=float(decomposition.h_s),
        )
        self.assertEqual([int(row["halo_cells"]) for row in rows], [0, 1, 2])

        _connectivity, projection_cells = projection_connectivity_rows(decomposition)
        remap_support = np.zeros(decomposition.d_remap.size, dtype=bool)
        remap_support[projection_cells] = True
        remap_pairing = support_pairing_metrics(
            decomposition.d_remap, edges_m=edges, support=remap_support
        )
        self.assertIn("complement_population_Linf_abs", remap_pairing)
        self.assertIn("support_absolute_weighted_M0", remap_pairing)

    def test_zero_event_control_is_found_by_dyadic_search_and_runs_all_paths(self) -> None:
        _decomposition, summary, edges, initial = self._event_case()
        control, rows = search_zero_event_control(
            edges_m=edges,
            initial_cells=initial,
            velocity_m_s=_nonlinear_velocity,
            starting_h_s=float(summary["selected_h_s"]),
            maximum_halvings=32,
        )
        self.assertTrue(rows)
        self.assertGreater(int(rows[0]["changed_face_count"]), 0)
        self.assertIsNotNone(control)
        assert control is not None
        self.assertEqual(int(changed_face_indices(control).size), 0)
        self.assertEqual(control.direct.label, "DIRECT")
        self.assertEqual(control.composed_flow.label, "COMPOSED_FLOW_SINGLE_REMAP")
        self.assertEqual(control.sequential.label, "SEQUENTIAL_HALF_STEPS")

    def test_synthetic_single_event_control_is_nonphysical_and_topology_selected(self) -> None:
        decomposition, summary, _edges, _initial = self._event_case()
        self.assertEqual(summary["status"], "SYNTHETIC_SINGLE_OR_FEW_EVENT_FOUND")
        self.assertFalse(bool(summary["physical_material_state_used"]))
        self.assertGreaterEqual(int(summary["changed_face_count"]), 1)
        self.assertLessEqual(int(summary["changed_face_count"]), 4)
        self.assertEqual(
            int(changed_face_indices(decomposition).size),
            int(summary["changed_face_count"]),
        )

    def test_sparse_operator_identity_reproduces_state_and_decomposition(self) -> None:
        decomposition = self.base_decomposition
        audit = operator_audit(decomposition, initial_cells=self.base_initial)
        scale = max(float(np.sum(np.abs(self.base_initial), dtype=np.float64)), 1.0)
        float64_l1_bound = 4096.0 * np.finfo(np.float64).eps * scale
        for field in (
            "P_h_state_residual",
            "P_compflow_state_residual",
            "P_sequential_state_residual",
            "trace_operator_state_residual",
            "remap_operator_state_residual",
        ):
            self.assertLessEqual(float(audit[field]["L1_abs"]), float64_l1_bound, msg=field)
        for field in (
            "operator_trace_1_norm",
            "operator_trace_infinity_norm",
            "operator_remap_1_norm",
            "operator_remap_infinity_norm",
        ):
            self.assertTrue(math.isfinite(float(audit[field])), msg=field)
            self.assertGreaterEqual(float(audit[field]), 0.0, msg=field)

    def test_signed_and_absolute_defect_metrics_do_not_cancel(self) -> None:
        edges = np.asarray([1.0, 2.0, 4.0, 8.0], dtype=np.float64)
        left = np.asarray([4.0, 1.0, 3.0], dtype=np.float64)
        right = np.asarray([1.0, 4.0, 3.0], dtype=np.float64)
        metrics = defect_metrics(left, right, edges_m=edges)
        self.assertEqual(float(metrics["signed_M0"]), 0.0)
        self.assertGreater(float(metrics["absolute_weighted_M0"]), 0.0)
        self.assertGreater(float(metrics["population_L1_abs"]), 0.0)
        self.assertGreaterEqual(float(metrics["CDF_max_error"]), 0.0)

    def test_diagnostic_has_no_side_effect_on_caller_owned_population(self) -> None:
        edges = _synthetic_edges()
        cells = _synthetic_cells(edges)
        before = cells.copy()
        first = decompose_frozen_cr1(
            edges_m=edges, initial_cells=cells, h_s=0.2, velocity_m_s=_zero_velocity
        )
        second = decompose_frozen_cr1(
            edges_m=edges, initial_cells=cells, h_s=0.2, velocity_m_s=_zero_velocity
        )
        np.testing.assert_array_equal(cells, before)
        np.testing.assert_array_equal(first.direct.cells, before)
        np.testing.assert_array_equal(first.direct.cells, second.direct.cells)
        self.assertFalse(np.shares_memory(first.direct.cells, cells))
        self.assertFalse(np.shares_memory(first.composed_flow.cells, cells))
        self.assertFalse(np.shares_memory(first.sequential.cells, cells))
        self.assertEqual(first.composed_query_audit["mode"], "PUBLIC_ZERO_MOBILITY_IDENTITY")

    def test_canonical_table_rejects_a_stationary_query_without_a_new_tail_path(self) -> None:
        edges = np.asarray([1.0, 1.5, 2.0, 3.0, 4.0], dtype=np.float64)

        def stationary_velocity(radii_m: np.ndarray) -> np.ndarray:
            return np.asarray(radii_m, dtype=np.float64) - 2.0

        half_trace = trace_departure_faces_rk2(
            edges,
            dt_s=0.1,
            velocity_m_s=stationary_velocity,
            lower_radius_m=float(edges[0]),
            upper_radius_m=float(edges[-1]),
        )
        with self.assertRaisesRegex(
            FrozenSemigroupDecompositionError,
            "B_QUERY_ON_STATIONARY_RADIUS_IS_NOT_ADMISSIBLE",
        ):
            canonical_table_half_flow_query(
                canonical_edges_m=edges,
                query_points_m=np.asarray([2.0], dtype=np.float64),
                half_dt_s=0.1,
                velocity_m_s=stationary_velocity,
                canonical_public_trace=half_trace,
            )


if __name__ == "__main__":
    unittest.main()
