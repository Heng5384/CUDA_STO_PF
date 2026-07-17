import unittest

from scripts.transport_defect_gate_v2_contract import (
    evaluate_case,
    peak_localization_class,
    strict_reference_peak_floor,
    three_consecutive_growth_failure,
)


def row(index, **overrides):
    base = {
        "window_index": index,
        "eta_global": 1.0e-8,
        "eta_interface": 1.0e-7,
        "eta_cell_max": 1.0e-7,
        "b_signed": 1.0e-6,
        "b_interface": 1.0e-6,
    }
    base.update(overrides)
    return base


class TransportDefectGateV2ContractTest(unittest.TestCase):
    def test_v1_peak_failure_can_coexist_with_v2_warning_pass(self):
        windows = [row(index) for index in range(1, 6)]
        result = evaluate_case(
            windows,
            row(0),
            g_d_floor=3.93,
            qoi_pass=True,
            numerical_contract_pass=True,
            holdout_window_index=5,
        )
        self.assertTrue(result.normalized_hard_gates_pass)
        self.assertEqual(result.peak_localization_class,
                         "PEAK_LOCALIZATION_INCREASED")
        self.assertEqual(result.v2_status,
                         "PASS_V2_WITH_LOCAL_PEAK_WARNING")

    def test_interface_defect_failure_is_material_v2_failure(self):
        windows = [row(1), row(2, eta_interface=1.01e-3), row(3), row(4)]
        result = evaluate_case(
            windows,
            row(0),
            g_d_floor=1.0,
            qoi_pass=True,
            numerical_contract_pass=True,
            holdout_window_index=4,
        )
        self.assertFalse(result.interface_defect_pass)
        self.assertEqual(result.v2_status,
                         "FAIL_V2_MATERIAL_TRANSPORT_DEFECT")

    def test_signed_growth_requires_ratio_absolute_level_and_three_increases(self):
        self.assertTrue(
            three_consecutive_growth_failure([1e-5, 3e-5, 7e-5, 2e-4])
        )
        self.assertFalse(
            three_consecutive_growth_failure([1e-8, 3e-8, 7e-8, 2e-7])
        )
        self.assertFalse(
            three_consecutive_growth_failure([1e-5, 3e-5, 2e-5, 2e-4])
        )

    def test_signed_growth_uses_full_series_first_and_final_windows(self):
        self.assertFalse(
            three_consecutive_growth_failure([2e-5, 4e-5, 8e-5, 1.1e-4, 7e-5])
        )
        self.assertFalse(
            three_consecutive_growth_failure([8e-5, 2e-5, 4e-5, 8e-5, 1.1e-4])
        )
        self.assertTrue(
            three_consecutive_growth_failure([2e-5, 4e-5, 8e-5, 9e-5, 1.1e-4])
        )

    def test_strict_reference_floor_uses_p95_or_roundoff(self):
        values = strict_reference_peak_floor([1.0, 2.0, 3.0, 4.0], 0.1, 2.0)
        self.assertAlmostEqual(values["r_peak_ref_median"], 2.5)
        self.assertAlmostEqual(values["r_peak_ref_p95"], 3.85)
        self.assertEqual(values["r_peak_floor"], 5.0)

    def test_peak_classes_are_diagnostic_only(self):
        self.assertEqual(peak_localization_class(1.0),
                         "PEAK_LOCALIZATION_STABLE")
        self.assertEqual(peak_localization_class(3.93),
                         "PEAK_LOCALIZATION_INCREASED")
        self.assertEqual(peak_localization_class(7.0),
                         "PEAK_LOCALIZATION_WARNING")
        self.assertEqual(peak_localization_class(11.0),
                         "STRONG_PEAK_LOCALIZATION_WARNING")


if __name__ == "__main__":
    unittest.main()
