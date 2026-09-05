"""Focused contracts for the read-only production trace audit harness."""

from __future__ import annotations

from pathlib import Path
import unittest

import numpy as np

from kwn_mvp.frozen_exact_flow_reference import FrozenAutonomousExactFlow, FrozenAutonomousGrowthLaw
from kwn_mvp.populations import PopulationParameters
from kwn_mvp.production_trace_audit import (
    ProductionAutonomousTOFTable,
    ProductionTraceAuditConfig,
    ProductionTraceAuditError,
    QualifiedAutonomousTOFTraceKernelV1,
    public_production_trace_probe,
)
from kwn_mvp.thermo_adapter import DiluteEquilibriumAdapter
from scripts.run_kwn_production_trace_audit_v1 import _single_step_scaling_rows


def _law(*, gamma: float = 0.0) -> FrozenAutonomousGrowthLaw:
    return FrozenAutonomousGrowthLaw(
        parameters=PopulationParameters(
            name="beta",
            x_b=1.0,
            molar_volume_m3_mol=1.0,
            diffusivity_m2_s=0.2,
            gamma_j_m2=gamma,
            xeq_infinity=0.2,
            nucleation={"mode": "off"},
        ),
        equilibrium_adapter=DiluteEquilibriumAdapter(temperature_k=1.0),
        matrix_xb=0.6,
        lower_radius_m=1.0,
        upper_radius_m=4.0,
    )


class ProductionTraceAuditContracts(unittest.TestCase):
    """The audit table must reproduce, never replace, the public tracer."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.edges = np.asarray([1.0, 1.25, 1.6, 2.1, 2.8, 4.0], dtype=np.float64)
        cls.law = _law()
        cls.flow = FrozenAutonomousExactFlow(cls.law, cls.edges)

    def test_default_table_bitwise_reproduces_public_trace_and_is_deterministic(self) -> None:
        table = ProductionAutonomousTOFTable(edges_m=self.edges, velocity_m_s=self.law.velocity)
        public = table.require_default_public_parity(0.1)
        queried = table.query_departure(self.edges, 0.1)
        np.testing.assert_array_equal(queried.radius_m, public.departure_faces_m)
        repeated = table.query_departure(self.edges, 0.1)
        np.testing.assert_array_equal(repeated.radius_m, queried.radius_m)
        self.assertTrue(
            set(queried.status.tolist()).issubset(
                {"INTERIOR", "NO_INFLOW_LOWER", "NO_INFLOW_UPPER", "STATIONARY_CRITICAL"}
            )
        )

    def test_exact_tau_and_bracketed_variants_are_read_only_component_probes(self) -> None:
        production = ProductionAutonomousTOFTable(
            edges_m=self.edges,
            velocity_m_s=self.law.velocity,
            exact_flow=self.flow,
        )
        production.require_default_public_parity(0.1)
        bracketed = ProductionAutonomousTOFTable(
            edges_m=self.edges,
            velocity_m_s=self.law.velocity,
            exact_flow=self.flow,
            config=ProductionTraceAuditConfig(inversion_mode="BRACKETED_TABLE"),
        )
        exact_tau = ProductionAutonomousTOFTable(
            edges_m=self.edges,
            velocity_m_s=self.law.velocity,
            exact_flow=self.flow,
            config=ProductionTraceAuditConfig(table_kind="EXACT_TAU", inversion_mode="BRACKETED_TABLE"),
        )
        for table in (production, bracketed, exact_tau):
            result = table.query_departure(self.edges, 0.1)
            self.assertTrue(np.all(np.isfinite(result.radius_m)))
            self.assertEqual(result.radius_m.shape, self.edges.shape)
        tau_rows = production.tau_rows(include_midpoints=True)
        self.assertGreater(len(tau_rows), len(self.edges))
        self.assertTrue(all(float(row["absolute_error_s"]) >= 0.0 for row in tau_rows))

    def test_public_probe_reports_backward_statuses(self) -> None:
        trace, statuses = public_production_trace_probe(
            arrival_faces_m=self.edges,
            duration_s=0.1,
            velocity_m_s=self.law.velocity,
            lower_radius_m=float(self.edges[0]),
            upper_radius_m=float(self.edges[-1]),
        )
        self.assertEqual(trace.departure_faces_m.shape, self.edges.shape)
        self.assertEqual(statuses.shape, self.edges.shape)
        self.assertTrue(set(statuses.tolist()).issubset({"INTERIOR", "NO_INFLOW_LOWER", "NO_INFLOW_UPPER", "STATIONARY_CRITICAL"}))

    def test_qualified_wrapper_rejects_dynamic_xb(self) -> None:
        kernel = QualifiedAutonomousTOFTraceKernelV1()
        with self.assertRaisesRegex(ProductionTraceAuditError, "rejects dynamic xB"):
            kernel.trace(
                arrival_faces_m=self.edges,
                duration_s=0.1,
                velocity_m_s=self.law.velocity,
                lower_radius_m=float(self.edges[0]),
                upper_radius_m=float(self.edges[-1]),
                matrix_xb_constant_over_interval=False,
            )
        trace = kernel.trace(
            arrival_faces_m=self.edges,
            duration_s=0.1,
            velocity_m_s=self.law.velocity,
            lower_radius_m=float(self.edges[0]),
            upper_radius_m=float(self.edges[-1]),
            matrix_xb_constant_over_interval=True,
        )
        self.assertEqual(trace.departure_faces_m.shape, self.edges.shape)

    def test_invalid_auxiliary_table_configuration_fails_closed(self) -> None:
        with self.assertRaises(ProductionTraceAuditError):
            ProductionTraceAuditConfig(subcells_per_cell=0)
        with self.assertRaises(ProductionTraceAuditError):
            ProductionTraceAuditConfig(subcells_per_cell=1.5)  # type: ignore[arg-type]

    def test_exact_tau_representation_requires_the_independent_flow(self) -> None:
        with self.assertRaisesRegex(ProductionTraceAuditError, "requires the independent frozen flow"):
            ProductionAutonomousTOFTable(
                edges_m=self.edges,
                velocity_m_s=self.law.velocity,
                config=ProductionTraceAuditConfig(table_kind="EXACT_TAU"),
            )

    def test_tau_rows_include_node_and_midpoint_samples(self) -> None:
        table = ProductionAutonomousTOFTable(
            edges_m=self.edges,
            velocity_m_s=self.law.velocity,
            exact_flow=self.flow,
        )
        rows = table.tau_rows(include_midpoints=True)
        kinds = {str(row["sample_kind"]) for row in rows}
        self.assertEqual(kinds, {"NODE", "MIDPOINT"})
        self.assertTrue(all(float(row["tau_production_anchor_normalized_s"]) >= 0.0 for row in rows))
        self.assertTrue(all(float(row["tau_exact_anchor_normalized_s"]) >= 0.0 for row in rows))

    def test_table_query_does_not_mutate_caller_points(self) -> None:
        points = self.edges.copy()
        before = points.copy()
        table = ProductionAutonomousTOFTable(edges_m=self.edges, velocity_m_s=self.law.velocity)
        table.query_departure(points, 0.1)
        np.testing.assert_array_equal(points, before)

    def test_bracketed_table_inversion_is_deterministic(self) -> None:
        table = ProductionAutonomousTOFTable(
            edges_m=self.edges,
            velocity_m_s=self.law.velocity,
            config=ProductionTraceAuditConfig(inversion_mode="BRACKETED_TABLE"),
        )
        run = table.runs[0]
        target = float(0.5 * (run.cumulative_time_s[1] + run.cumulative_time_s[2]))
        first = table.invert(run, target)
        second = table.invert(run, target)
        self.assertEqual(first, second)
        self.assertGreaterEqual(first.iteration_count, 1)

    def test_exact_tau_bracketed_inverse_preserves_a_table_knot(self) -> None:
        table = ProductionAutonomousTOFTable(
            edges_m=self.edges,
            velocity_m_s=self.law.velocity,
            exact_flow=self.flow,
            config=ProductionTraceAuditConfig(table_kind="EXACT_TAU", inversion_mode="BRACKETED_TABLE"),
        )
        run = table.runs[0]
        index = min(3, run.coordinates_m.size - 1)
        inverted = table.invert(run, float(run.cumulative_time_s[index]))
        self.assertEqual(inverted.radius_m, float(run.coordinates_m[index]))
        self.assertEqual(inverted.iteration_count, 0)

    def test_production_fixed_inverse_reports_its_fixed_six_iterations(self) -> None:
        table = ProductionAutonomousTOFTable(edges_m=self.edges, velocity_m_s=self.law.velocity)
        run = table.runs[0]
        target = float(0.5 * (run.cumulative_time_s[1] + run.cumulative_time_s[2]))
        inverted = table.invert(run, target)
        self.assertEqual(inverted.iteration_count, 6)
        self.assertGreater(inverted.bracket_width_m, 0.0)

    def test_multiple_durations_retain_public_trace_parity(self) -> None:
        table = ProductionAutonomousTOFTable(edges_m=self.edges, velocity_m_s=self.law.velocity)
        for duration in (0.025, 0.05, 0.1):
            public = table.require_default_public_parity(duration)
            queried = table.query_departure(self.edges, duration)
            np.testing.assert_array_equal(queried.radius_m, public.departure_faces_m)

    def test_qualified_wrapper_matches_public_autonomous_trace(self) -> None:
        public, _status = public_production_trace_probe(
            arrival_faces_m=self.edges,
            duration_s=0.1,
            velocity_m_s=self.law.velocity,
            lower_radius_m=float(self.edges[0]),
            upper_radius_m=float(self.edges[-1]),
        )
        qualified = QualifiedAutonomousTOFTraceKernelV1().trace(
            arrival_faces_m=self.edges,
            duration_s=0.1,
            velocity_m_s=self.law.velocity,
            lower_radius_m=float(self.edges[0]),
            upper_radius_m=float(self.edges[-1]),
            matrix_xb_constant_over_interval=True,
        )
        np.testing.assert_array_equal(qualified.departure_faces_m, public.departure_faces_m)

    def test_audit_module_does_not_import_or_call_cr1_remap(self) -> None:
        source = Path(__file__).resolve().parents[2] / "src" / "kwn_mvp" / "production_trace_audit.py"
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("conservative_remap_piecewise_constant", text)
        self.assertNotIn("phi_compose", text)

    def test_single_step_scaling_accepts_exact_zero_error_pairs(self) -> None:
        rows = [
            {
                "face_index": 0,
                "h_s": h_s,
                "absolute_radius_error_m": 0.0,
                "arrival_radius_m": 1.0,
            }
            for h_s in (1.0 / 128.0, 1.0 / 256.0, 1.0 / 512.0, 1.0 / 1024.0)
        ]
        scaling, summary = _single_step_scaling_rows(rows)
        self.assertEqual(summary["face_class_counts"], {"TRACE_SINGLE_STEP_FIXED_FLOOR": 1})
        self.assertEqual(len(scaling), 3)
        self.assertTrue(all(row["observed_order"] is None for row in scaling))


if __name__ == "__main__":
    unittest.main()
