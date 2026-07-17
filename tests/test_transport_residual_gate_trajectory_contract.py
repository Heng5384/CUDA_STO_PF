#!/usr/bin/env python3
"""Source contract for the default-off transport-gate trajectory observer."""

from pathlib import Path
import re
import unittest


ROOT = Path(__file__).resolve().parents[1]
MAIN = (ROOT / "main_cuda.cu").read_text(encoding="utf-8")
PARAMS = (ROOT / "pf_params.h").read_text(encoding="utf-8")


class TransportGateTrajectoryContractTest(unittest.TestCase):
    def test_observer_is_default_off_and_runtime_selectable(self):
        self.assertIn(
            "int ctot_transport_gate_trajectory_diagnostics;", PARAMS
        )
        self.assertIn(
            "P->ctot_transport_gate_trajectory_diagnostics = 0;", MAIN
        )
        self.assertIn(
            'TRY_SET_INT("ctot_transport_gate_trajectory_diagnostics"', MAIN
        )

    def test_cold_residual_is_captured_at_method_context(self):
        capture = MAIN.index("ctot_transport_gate_pending_residual.data()")
        method_eval = MAIN.rfind(
            "evaluate_ctot_transport_residual_at_phi(", 0, capture
        )
        final_split_eval = MAIN.find(
            "final_transport_res_inf", capture
        )
        self.assertGreater(method_eval, 0)
        self.assertGreater(final_split_eval, capture)

    def test_only_accepted_trials_are_accumulated(self):
        reject_gate = MAIN.index(
            "if (!transport_ok || !converged || !energy_audit_pass)"
        )
        accepted_row = MAIN.index(
            "CtotTransportGateTrajectoryRow trajectory_row =", reject_gate
        )
        history_commit = MAIN.index(
            "Atomic accepted-history commit", accepted_row
        )
        self.assertLess(reject_gate, accepted_row)
        self.assertLess(accepted_row, history_commit)

    def test_event_substeps_follow_macro_transaction_commit(self):
        self.assertIn("ctot_transport_gate_event_rows", MAIN)
        self.assertIn("ctot_transport_gate_event_defect", MAIN)
        self.assertIn("ctot_transport_gate_reset_event_transaction();", MAIN)
        ready = MAIN.index("CTOT_BDF2_EVENT_MACRO_READY")
        commit = MAIN.index(
            "ctot_transport_gate_commit_event_transaction();", ready
        )
        history_commit = MAIN.index("Atomic accepted-history commit", commit)
        self.assertLess(ready, commit)
        self.assertLess(commit, history_commit)

    def test_observer_does_not_feed_device_or_solver(self):
        observer_symbols = set(
            re.findall(r"ctot_transport_gate_[A-Za-z0-9_]+", MAIN)
        )
        forbidden = [
            symbol
            for symbol in observer_symbols
            if re.search(
                rf"launch_[A-Za-z0-9_]+\([^;]*\b{re.escape(symbol)}\b",
                MAIN,
                flags=re.DOTALL,
            )
        ]
        self.assertEqual(forbidden, [])
        self.assertNotIn("cudaMemcpyHostToDevice));\n                "
                         "ctot_transport_gate", MAIN)

    def test_output_definitions_are_explicit(self):
        self.assertIn(
            "D_i=sum_macro_committed_dt_try_times_method_cold_R_i", MAIN
        )
        self.assertIn(
            "R_C=Delta_C-dt_transport_divJ_at_method_phase_context", MAIN
        )


if __name__ == "__main__":
    unittest.main()
