import pathlib
import sys
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.run_T400_longtime_bdf2_matrix import (  # noqa: E402
    crossing_half_width,
    crossings,
    h,
    rewrite_params,
)


class T400LongtimeBdf2RunnerTests(unittest.TestCase):
    def test_periodic_two_interface_crossing_and_h_volume(self):
        n = 512
        dx = 1.0
        coordinate = (np.arange(n, dtype=np.float64) + 0.5) * dx
        center = 0.5 * n * dx
        half_width = 16.0
        phi = 0.5 * (
            np.tanh((coordinate - (center - half_width)) / 2.0)
            - np.tanh((coordinate - (center + half_width)) / 2.0)
        )
        self.assertEqual(len(crossings(phi, dx)), 2)
        self.assertAlmostEqual(crossing_half_width(phi, dx), half_width, places=12)
        self.assertAlmostEqual(0.5 * float(np.sum(h(phi))) * dx, half_width, places=12)

    def test_parameter_rewrite_preserves_unrelated_values_and_appends_new(self):
        source = "dt=1e-3\nD_alpha=400\n# comment\n"
        updated = rewrite_params(source, {"dt": "2e-4", "ctot_step_max_retries": "0"})
        self.assertIn("dt=2e-4\n", updated)
        self.assertIn("D_alpha=400\n", updated)
        self.assertIn("ctot_step_max_retries=0\n", updated)
        self.assertNotIn("dt=1e-3", updated)


if __name__ == "__main__":
    unittest.main()
