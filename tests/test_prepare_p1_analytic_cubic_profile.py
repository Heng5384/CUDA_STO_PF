import importlib.util
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "prepare_p1_analytic_cubic_profile.py"
SPEC = importlib.util.spec_from_file_location("analytic_profile", SCRIPT)
module = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(module)


class P1AnalyticCubicProfileTests(unittest.TestCase):
    def test_fields_match_p1_formula_and_storage(self):
        phi, x_b, ctot, x_b_eq = module.build_fields(32)
        self.assertEqual(phi.shape, (32, 32, 32))
        self.assertAlmostEqual(phi[16, 16, 16], 0.5 * (1.0 + np.tanh(10.0)))
        self.assertAlmostEqual(phi[19, 16, 16], 0.5)
        self.assertAlmostEqual(x_b[19, 16, 16], 0.5 * (0.007830539083594734 + x_b_eq))
        alpha = module.alpha_stable(phi)
        np.testing.assert_array_equal(ctot, (1.0 - alpha) + alpha * x_b)
        self.assertTrue(np.all(ctot >= 1.0 - alpha))
        self.assertTrue(np.all(ctot <= 1.0))

    def test_metadata_matches_frozen_discretization(self):
        data = module.metadata(64)
        self.assertEqual(data["dx_nm"], 1.0)
        self.assertEqual(data["interface_width_nm"], 0.6)
        self.assertEqual(data["authoritative_state"], "Ctot")
        self.assertFalse(data["physical_benchmark_evidence"])


if __name__ == "__main__":
    unittest.main()
