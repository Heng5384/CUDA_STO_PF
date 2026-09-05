"""Focused contracts for the read-only production trace audit harness."""

from __future__ import annotations

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


if __name__ == "__main__":
    unittest.main()
