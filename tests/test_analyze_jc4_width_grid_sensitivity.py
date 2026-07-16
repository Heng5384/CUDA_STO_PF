import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class AnalyzeJc4WidthGridSensitivityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(ROOT))
        from scripts import analyze_jc4_width_grid_sensitivity as module
        cls.module = module

    @staticmethod
    def metric(displacement=10.0, half=20.0, numerical=True):
        return {
            "case_id": "case",
            "PF_h_displacement_nm": displacement,
            "final_half_width_h_nm": half,
            "dx_nm": 1.0,
            "numerical_hard_gates_pass": numerical,
        }

    @staticmethod
    def trajectory(scale=1.0):
        return [
            {"time_s": 0.0, "PF_displacement_h_nm": 0.0},
            {"time_s": 1.0, "PF_displacement_h_nm": 5.0 * scale},
            {"time_s": 2.0, "PF_displacement_h_nm": 10.0 * scale},
        ]

    def test_equal_observable_case_passes(self):
        result = self.module.compare_to_baseline(
            self.metric(), self.trajectory(), self.metric(), self.trajectory()
        )
        self.assertTrue(result["width_grid_sensitivity_gate_pass"])
        self.assertEqual(result["trajectory_variation_rel"], 0.0)

    def test_transfer_or_hard_gate_failure_is_not_hidden(self):
        result = self.module.compare_to_baseline(
            self.metric(displacement=11.2, half=21.2, numerical=False),
            self.trajectory(1.12), self.metric(), self.trajectory(),
        )
        self.assertFalse(result["width_grid_sensitivity_gate_pass"])
        self.assertGreater(result["cumulative_transfer_variation_rel"], 0.10)


if __name__ == "__main__":
    unittest.main()
