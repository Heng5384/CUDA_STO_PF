#!/usr/bin/env python3
"""Unit contracts for long transport-gate trajectory acceptance."""

from pathlib import Path
import sys
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))

from analyze_transport_residual_gate_long_window import (  # noqa: E402
    boundary_consecutive_fallback,
    continuation_hard_pass,
    failed_cell_attempt_counts,
    retry_efficiency_pass,
)


class TransportGateLongAnalysisTest(unittest.TestCase):
    def test_fallback_sequence_is_joined_across_restart(self):
        self.assertEqual(
            boundary_consecutive_fallback(10, [8, 9, 10], [1, 2, 5]),
            5,
        )
        self.assertEqual(
            boundary_consecutive_fallback(10, [8, 9], [1, 2]),
            2,
        )

    def test_failed_cell_count_uses_worst_cell_per_attempt(self):
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder) / "ctot_failed_cell_state.csv"
            path.write_text(
                "physical_step,attempt_id,idx,residual\n"
                "4,1,10,1e-9\n"
                "4,1,11,2e-9\n"
                "5,1,11,3e-9\n",
                encoding="utf-8",
            )
            self.assertEqual(failed_cell_attempt_counts(Path(folder))[11], 2)

    def test_retry_gate_rejects_persistence_and_excess_depth(self):
        summary = {
            "bounded_retry_summary": {
                "macro_hard_rejects": "0",
                "retry_fraction": "0.001",
                "fallback_fraction": "0.001",
                "reject_trial_wall_fraction": "0.01",
                "max_accepted_subcycle_depth": "2",
                "accepted_iteration_p99": "40",
                "nonlinear_iteration_budget": "500",
            }
        }
        self.assertTrue(retry_efficiency_pass(summary, [], 2))
        self.assertFalse(retry_efficiency_pass(summary, [244], 2))
        self.assertFalse(retry_efficiency_pass(summary, [], 3))

    def test_reference_hard_gate_requires_transactional_time_closure(self):
        summary = {
            "acceptance_hard_pass": True,
            "energy_pass": True,
            "rollback_pass": True,
            "max_mass_error": 1.0e-12,
            "trajectory_meta": {
                "observed_time_code": 4.6875,
                "schema": "CTOT_TRANSPORT_GATE_TRAJECTORY_V2_TRANSACTIONAL",
            },
        }
        self.assertTrue(continuation_hard_pass(summary, 4.6875))
        summary["trajectory_meta"]["observed_time_code"] = 4.68
        self.assertFalse(continuation_hard_pass(summary, 4.6875))

    def test_long_fp64_time_accumulation_uses_roundoff_bound(self):
        summary = {
            "acceptance_hard_pass": True,
            "energy_pass": True,
            "rollback_pass": True,
            "max_mass_error": 1.0e-12,
            "trajectory_meta": {
                "accepted_rows": 25497,
                "observed_time_code": 4.6874999999979385,
                "schema": "CTOT_TRANSPORT_GATE_TRAJECTORY_V2_TRANSACTIONAL",
            },
        }
        self.assertTrue(continuation_hard_pass(summary, 4.6875))
        summary["trajectory_meta"]["observed_time_code"] += 1.0e-4
        self.assertFalse(continuation_hard_pass(summary, 4.6875))


if __name__ == "__main__":
    unittest.main()
