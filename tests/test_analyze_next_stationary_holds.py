import importlib.util
from pathlib import Path
import unittest

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "analyze_next_stationary_holds.py"
SPEC = importlib.util.spec_from_file_location("stationary_hold_analysis", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class StationaryHoldAnalysisTests(unittest.TestCase):
    def test_h_stable_endpoints_and_symmetry(self):
        values = np.linspace(0.0, 1.0, 257)
        observed = MODULE.h_stable(values)
        self.assertEqual(float(observed[0]), 0.0)
        self.assertEqual(float(observed[-1]), 1.0)
        np.testing.assert_allclose(observed, 1.0 - observed[::-1],
                                   rtol=0.0, atol=2.0e-15)

    def test_cylindrical_radius_uses_y_average(self):
        n = 65
        coords = np.arange(n) - n // 2
        xx, zz = np.meshgrid(coords, coords, indexing="ij")
        radius = 10.0
        plane = (np.sqrt(xx * xx + zz * zz) <= radius).astype(np.float64)
        field = np.repeat(plane[:, None, :], 3, axis=1)
        h_radius, half_radius = MODULE.radii(field, 1.0)
        expected = np.sqrt(float(np.sum(plane)) / np.pi)
        self.assertAlmostEqual(h_radius, expected, places=14)
        self.assertAlmostEqual(half_radius, 10.5, places=14)


if __name__ == "__main__":
    unittest.main()
