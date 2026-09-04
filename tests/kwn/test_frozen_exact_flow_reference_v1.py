"""Unit contracts for the diagnostic-only frozen exact-flow reference."""

from __future__ import annotations

import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kwn_mvp.frozen_exact_flow_reference import (
    FrozenAutonomousExactFlow,
    FrozenAutonomousGrowthLaw,
    FrozenExactFlowReferenceError,
    PiecewiseConstantCumulativeMeasure,
    PiecewiseLinearCumulativeMeasure,
    classify_refinement_against_reference,
    measure_error_metrics,
    observed_orders,
)
from kwn_mvp.populations import PopulationParameters
from kwn_mvp.thermo_adapter import DiluteEquilibriumAdapter
from scripts.run_kwn_frozen_exact_flow_reference_v1 import _formal_u0_semantic_replay


def _law(*, matrix_xb: float, diffusivity: float = 0.2) -> FrozenAutonomousGrowthLaw:
    parameters = PopulationParameters(
        name="beta",
        x_b=1.0,
        molar_volume_m3_mol=1.0,
        diffusivity_m2_s=diffusivity,
        gamma_j_m2=0.0,
        xeq_infinity=0.2,
        nucleation={"mode": "off"},
    )
    return FrozenAutonomousGrowthLaw(
        parameters=parameters,
        equilibrium_adapter=DiluteEquilibriumAdapter(temperature_k=1.0),
        matrix_xb=matrix_xb,
        lower_radius_m=1.0,
        upper_radius_m=4.0,
    )


class FrozenExactFlowReferenceContracts(unittest.TestCase):
    """Small deterministic tests; no restart, CR1 trace, or remap is used."""

    @classmethod
    def setUpClass(cls) -> None:
        cls.edges = np.asarray([1.0, 1.25, 1.6, 2.1, 2.8, 4.0], dtype=np.float64)
        cls.cells = np.asarray([1.0, 3.0, 5.0, 2.0, 1.0], dtype=np.float64)
        cls.growing_flow = FrozenAutonomousExactFlow(_law(matrix_xb=0.6), cls.edges)

    def test_exact_flow_semigroup_and_inverse_roundtrip(self) -> None:
        semigroup = self.growing_flow.semigroup_check(self.edges, 0.1)
        self.assertEqual(semigroup["status_mismatch_count"], 0)
        self.assertLess(float(semigroup["max_relative_radius_error"]), 1.0e-12)
        roundtrip = self.growing_flow.roundtrip_check(self.edges, 0.1)
        self.assertEqual(roundtrip["status_mismatch_count"], 0)
        self.assertGreater(int(roundtrip["tested_count"]), 0)
        self.assertLess(float(roundtrip["max_relative_radius_error"]), 1.0e-12)

    def test_rmin_is_an_absorbing_event_not_an_extrapolation(self) -> None:
        shrinking = FrozenAutonomousExactFlow(_law(matrix_xb=0.1, diffusivity=0.2), self.edges)
        event = shrinking.evaluate(1.05, 3.0)
        self.assertEqual(event.status, "ABSORBED_RMIN")
        self.assertEqual(event.radius_m, 1.0)
        self.assertIsNotNone(event.event_time_s)
        self.assertGreater(float(event.event_time_s), 0.0)
        self.assertLess(float(event.event_time_s), 3.0)

    def test_ref_pc_t0_identity_cdf_and_number_balance(self) -> None:
        measure = PiecewiseConstantCumulativeMeasure(self.edges, self.cells)
        state = measure.pushforward(self.growing_flow, 0.0)
        np.testing.assert_array_equal(state.cell_number_m3, self.cells)
        np.testing.assert_array_equal(state.departure_faces_m, self.edges)
        np.testing.assert_array_equal(measure.cdf(self.edges), np.concatenate(([0.0], np.cumsum(self.cells))))
        self.assertEqual(state.conservation_residual_m3, 0.0)
        advanced = measure.pushforward(self.growing_flow, 0.1)
        self.assertGreaterEqual(advanced.lower_number_loss_m3, 0.0)
        self.assertGreaterEqual(advanced.upper_number_loss_m3, 0.0)
        self.assertLess(abs(advanced.conservation_residual_m3), 1.0e-12)

    def test_ref_pl_is_conservative_and_diagnostic_only(self) -> None:
        pc = PiecewiseConstantCumulativeMeasure(self.edges, self.cells)
        pl = PiecewiseLinearCumulativeMeasure(self.edges, self.cells)
        np.testing.assert_allclose(pl.cdf(self.edges), pc.cdf(self.edges), rtol=0.0, atol=1.0e-14)
        advanced = pl.pushforward(self.growing_flow, 0.1)
        self.assertEqual(advanced.representation, "REF_PL_DIAGNOSTIC_ONLY")
        self.assertLess(abs(advanced.conservation_residual_m3), 1.0e-12)

    def test_error_metrics_are_independent_and_complete(self) -> None:
        shifted = np.asarray([0.5, 3.5, 4.5, 2.0, 1.5], dtype=np.float64)
        metrics = measure_error_metrics(shifted, self.cells, edges_m=self.edges)
        for key in (
            "population_L1_abs",
            "population_relative_L1",
            "population_Linf_abs",
            "CDF_max_error",
            "Wasserstein_m",
            "signed_M0",
            "absolute_weighted_M3",
        ):
            self.assertIn(key, metrics)
        self.assertGreater(float(metrics["Wasserstein_m"]), 0.0)

    def test_observed_order_and_reference_only_classification(self) -> None:
        h = [1.0, 0.5, 0.25, 0.125, 0.0625]
        first = [0.8 * value for value in h]
        subfirst = [math.sqrt(value) for value in h]
        plateau = [1.0, 0.5, 0.25, 0.25, 0.25]
        order = observed_orders(h, first, metric="synthetic")
        self.assertAlmostEqual(float(order["late_level_median_order"]), 1.0, places=12)
        self.assertEqual(
            classify_refinement_against_reference(h, first)["classification"],
            "CR1_ASYMPTOTICALLY_CONSISTENT_FIRST_ORDER_LIKE",
        )
        self.assertEqual(
            classify_refinement_against_reference(h, subfirst)["classification"],
            "CR1_ASYMPTOTICALLY_CONSISTENT_SUBFIRST_ORDER",
        )
        self.assertEqual(
            classify_refinement_against_reference(h, plateau)["classification"],
            "CR1_ERROR_STAGNATION_AGAINST_EXACT_REFERENCE",
        )

    def test_mpmath_shadow_fails_closed_or_agrees(self) -> None:
        if importlib.util.find_spec("mpmath") is None:
            with self.assertRaises(FrozenExactFlowReferenceError):
                self.growing_flow.mpmath_tau(1.25, 1.6)
            return
        scipy_value = self.growing_flow._quad_tau(1.25, 1.6)
        shadow = self.growing_flow.mpmath_tau(1.25, 1.6, dps=80)
        self.assertLess(abs(scipy_value - shadow), 1.0e-10)

    def test_reference_module_has_no_cr1_trace_or_remap_import(self) -> None:
        source = Path(__file__).resolve().parents[2] / "src" / "kwn_mvp" / "frozen_exact_flow_reference.py"
        text = source.read_text(encoding="utf-8")
        self.assertNotIn("trace_departure_faces_rk2", text)
        self.assertNotIn("conservative_remap_piecewise_constant", text)

    def test_formal_u0_replay_allows_only_path_rebind_metadata(self) -> None:
        arrays = {
            "radius_edges_m": self.edges,
            "cell_number_m3": self.cells,
            "matrix_xb": np.asarray([0.6], dtype=np.float64),
        }
        formal = {
            "schema": "TEST",
            "checkpoint_binding": {
                "checkpoint_sha256": "bound",
                "semantic_config_hash": "semantic",
                "runtime_contract_path": "/old/source/contract.json",
                "runtime_source_config_hash_before_rebind": "old-path-hash",
            },
        }
        rebound = {
            "schema": "TEST",
            "checkpoint_binding": {
                "checkpoint_sha256": "bound",
                "semantic_config_hash": "semantic",
                "runtime_contract_path": "/new/source/contract.json",
                "runtime_source_config_hash_before_rebind": "new-path-hash",
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "formal_u0.npz"
            np.savez(path, **arrays, metadata_json=np.asarray(json.dumps(formal, sort_keys=True)))
            result = _formal_u0_semantic_replay(
                path,
                reconstructed_arrays=arrays,
                reconstructed_metadata=rebound,
            )
        self.assertEqual(result["status"], "PASS_FORMAL_U0_BITWISE_ARRAY_AND_SEMANTIC_METADATA_REPLAY")


if __name__ == "__main__":
    unittest.main()
