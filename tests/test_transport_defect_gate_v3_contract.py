import unittest

from scripts.transport_defect_gate_v3_contract import (
    evaluate_case,
    metric_record_pass,
    sign_coherence_interpretation,
)


def row(window=6, **updates):
    value = {
        "record_type": "WINDOW",
        "window_index": window,
        "eta_global": 1.0e-8,
        "eta_interface": 1.0e-8,
        "eta_cell_max": 3.0e-4,
        "beta_global": 1.0e-8,
        "beta_interface": 1.0e-8,
        "beta_interface_excess": 1.0e-8,
    }
    value.update(updates)
    return value


class TransportDefectGateV3ContractTests(unittest.TestCase):
    def test_frozen_limits_pass(self):
        result = evaluate_case(
            [row()],
            holdout_window_index=6,
            require_excess=True,
            qoi_pass=True,
            numerical_pass=True,
        )
        self.assertEqual(
            result.status, "PASS_PHYSICALLY_SCALED_SIGNED_DEFECT_GATE_V3"
        )

    def test_interface_signed_excess_is_hard_gate(self):
        self.assertFalse(
            metric_record_pass(
                row(beta_interface_excess=1.0001e-4), require_excess=True
            )
        )

    def test_sign_coherence_fraction_is_not_a_hard_gate(self):
        candidate = row()
        candidate["interface_sign_coherence_diagnostic"] = 1.0
        self.assertTrue(metric_record_pass(candidate, require_excess=True))

    def test_tiny_interface_defect_makes_coherence_nonmaterial(self):
        self.assertEqual(
            sign_coherence_interpretation(9.9e-7),
            "SIGN_COHERENCE_NOT_MATERIALLY_INTERPRETABLE",
        )

    def test_holdout_must_be_present(self):
        result = evaluate_case(
            [row(window=5)],
            holdout_window_index=6,
            require_excess=True,
            qoi_pass=True,
            numerical_pass=True,
        )
        self.assertFalse(result.holdout_pass)


if __name__ == "__main__":
    unittest.main()
