import unittest

from scripts.analyze_lie_be_v2_step655_ladder import (
    choose_selected,
    observed_order,
)


class AnalyzeLieBeV2Step655LadderTests(unittest.TestCase):
    def test_observed_first_order_from_halving_error(self):
        self.assertAlmostEqual(observed_order(0.02, 0.01), 1.0)

    def test_selection_is_highest_throughput_eligible_case(self):
        rows = [
            {"divisor": 1, "time_error_gate_pass": False,
             "throughput_code_time_per_wall_s": 10.0},
            {"divisor": 2, "time_error_gate_pass": True,
             "throughput_code_time_per_wall_s": 4.0},
            {"divisor": 4, "time_error_gate_pass": True,
             "throughput_code_time_per_wall_s": 2.0},
        ]
        self.assertEqual(choose_selected(rows), 2)
        self.assertIsNone(choose_selected(rows[:1]))


if __name__ == "__main__":
    unittest.main()
