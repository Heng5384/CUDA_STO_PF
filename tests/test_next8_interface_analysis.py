import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location(
    "next8_analysis", ROOT / "scripts" / "analyze_next8_interface_response.py"
)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


class Next8InterfaceAnalysisTests(unittest.TestCase):
    def test_linear_velocity_and_equimolar_mapping(self):
        t = np.arange(10.0)
        x = 2.5e-9 * t + 4.0e-9
        slope, rms = MODULE.linear_slope(t, x)
        self.assertAlmostEqual(slope, 2.5e-9, places=20)
        self.assertLess(rms, 1.0e-20)
        z = np.linspace(0.0, 10.0, 1001)
        c = np.where(z <= 4.0, 1.0, 0.2)
        position = MODULE.equimolar_position(z, c, 0.2, 1.0)
        self.assertAlmostEqual(position, 4.0, delta=0.01)

    def test_crossing_inventory_and_stefan_flux_consistency(self):
        crossing = MODULE.crossing_flux(3.2e-18, 2.0e-18, 4.0, 1)
        inventory = MODULE.inventory_flux(3.2e-18, 2.0e-18, 4.0, 1)
        self.assertAlmostEqual(crossing, inventory)
        self.assertAlmostEqual(MODULE.stefan_flux(2.0e-9, 10.0, 2.0), 1.6e-8)

    def test_block_covariance_uses_blocks_not_timesteps(self):
        rng = np.random.default_rng(4)
        v = rng.normal(size=100)
        j = 0.4 * v + rng.normal(scale=0.2, size=100)
        cov, nblock = MODULE.block_mean_covariance(v, j, 10)
        self.assertEqual(nblock, 10)
        self.assertEqual(cov.shape, (2, 2))
        self.assertGreater(cov[0, 1], 0.0)
        self.assertGreater(np.linalg.det(cov), 0.0)

    def test_linearity_A_validation_and_nested_predictions(self):
        self.assertAlmostEqual(MODULE.linearity_deviation(4.0, 2.0), 0.0)
        result = MODULE.validate_A_I(6.0, 3.0, 0.1, 2.0)
        self.assertTrue(result["expected_inside_95CI"])
        self.assertEqual(MODULE.predict_diagonal(4.0, 3.0, 2.0, 1.5), (2.0, 2.0))
        v, j = MODULE.predict_reciprocal(4.0, 3.0, 2.0, 0.25, 1.5)
        np.testing.assert_allclose(np.array([[2.0, 0.25], [0.25, 1.5]]) @ np.array([v, j]), [4.0, 3.0])

    def test_invalid_inputs_fail_closed(self):
        with self.assertRaises(ValueError):
            MODULE.block_mean_covariance([1.0], [1.0], 1)
        with self.assertRaises(ValueError):
            MODULE.predict_reciprocal(1.0, 1.0, 1.0, 2.0, 1.0)
        with self.assertRaises(ValueError):
            MODULE.inventory_flux(1.0, 0.0, 1.0)


if __name__ == "__main__":
    unittest.main()
