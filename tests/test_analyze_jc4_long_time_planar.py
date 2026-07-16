import pathlib
import sys
import unittest

import numpy as np


ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.analyze_jc4_long_time_planar import (  # noqa: E402
    crossings,
    growth_law_exponent,
    half_width_from_crossings,
    interpolate_sharp_state,
)


class AnalyzeJc4LongTimePlanarTests(unittest.TestCase):
    def test_periodic_slab_crossings_and_half_width(self):
        phi = np.zeros(32)
        phi[8:24] = 1.0
        points = crossings(phi, 1.0)
        self.assertEqual(points, [7.5, 23.5])
        self.assertEqual(half_width_from_crossings(phi, 1.0), 8.0)

    def test_growth_law_exponent(self):
        time = np.linspace(1.0, 100.0, 50)
        displacement = 2.0 * np.sqrt(time)
        self.assertAlmostEqual(growth_law_exponent(time, displacement), 0.5, places=12)

    def test_sharp_state_is_interpolated_at_actual_pf_time(self):
        rows = [
            {"time_s": "0", "radius_nm": "10", "displacement_nm": "0", "matrix_mean_xB": "0.05"},
            {"time_s": "10", "radius_nm": "12", "displacement_nm": "2", "matrix_mean_xB": "0.04"},
        ]
        state = interpolate_sharp_state(rows, 2.5)
        self.assertAlmostEqual(state["radius_nm"], 10.5)
        self.assertAlmostEqual(state["displacement_nm"], 0.5)
        self.assertAlmostEqual(state["matrix_mean_xB"], 0.0475)

    def test_sharp_interpolation_rejects_time_outside_reference(self):
        rows = [
            {"time_s": "0", "radius_nm": "10", "displacement_nm": "0", "matrix_mean_xB": "0.05"},
            {"time_s": "1", "radius_nm": "11", "displacement_nm": "1", "matrix_mean_xB": "0.04"},
        ]
        with self.assertRaises(ValueError):
            interpolate_sharp_state(rows, 2.0)


if __name__ == "__main__":
    unittest.main()
