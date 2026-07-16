import pathlib
import unittest


ROOT = pathlib.Path(__file__).resolve().parents[1]


class AnalyzeJc4PlanarEngineeringTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import sys
        sys.path.insert(0, str(ROOT))
        from scripts import analyze_jc4_planar_engineering as module
        cls.module = module

    def test_h_switch_endpoints(self):
        import numpy as np
        values = self.module.h_switch(np.array([0.0, 1.0]))
        np.testing.assert_array_equal(values, np.array([0.0, 1.0]))

    def test_throughput_classification_is_fail_closed(self):
        rows = self.module.throughput_rows([
            {
                "direction": "growth", "wall_time_yz1_s": 50.0,
                "accepted_steps_yz1": 200,
            },
            {
                "direction": "dissolution", "wall_time_yz1_s": 70.0,
                "accepted_steps_yz1": 200,
            },
        ])
        by_direction = {row["direction"]: row for row in rows}
        self.assertEqual(
            by_direction["dissolution"]["projected_status"],
            "BLOCKED_LONG_TIME_THROUGHPUT",
        )
        self.assertEqual(by_direction["dissolution"]["research_gate_steps"], 640000)


if __name__ == "__main__":
    unittest.main()
