import unittest

from scripts.analyze_transport_defect_gate_v2 import (
    registered_qoi_pass,
    v1_status,
)


class TransportDefectGateV2AnalysisTest(unittest.TestCase):
    def test_preregistered_local_peak_failure_is_preserved_verbatim(self):
        row = {
            "long_window_status": "FAIL",
            "local_defect_rate_growth_pass": "False",
            "long_hard_gates_pass": "True",
            "long_retry_efficiency_gate_pass": "True",
            "all_registered_windows_accuracy_pass": "True",
            "signed_residual_bias_growth_pass": "True",
            "failure_reasons": "LOCAL_DEFECT_RATE_GROWTH_GATE",
        }
        self.assertEqual(
            v1_status(row),
            "FAIL_PREREGISTERED_LOCAL_PEAK_DEFECT_RATE_GATE",
        )

    def test_existing_qoi_limits_are_not_relaxed(self):
        passing = {
            "Ctot_increment_relative_L2_error": 0.02,
            "phi_increment_relative_L2_error": 0.02,
            "cumulative_transfer_relative_error": 0.03,
            "hvolume_increment_relative_error": 0.03,
            "phi_half_interface_error_dx": 0.5,
            "matrix_profile_capacity_weighted_relative_error": 0.05,
            "far_field_relative_error": 0.02,
            "interface_direction_same": True,
        }
        self.assertTrue(registered_qoi_pass(passing))
        failing = dict(passing)
        failing["far_field_relative_error"] = 0.0200001
        self.assertFalse(registered_qoi_pass(failing))
        failing = dict(passing)
        failing["phi_increment_relative_L2_error"] = 0.0200001
        self.assertFalse(registered_qoi_pass(failing))


if __name__ == "__main__":
    unittest.main()
