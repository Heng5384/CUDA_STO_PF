import pathlib
import sys
import unittest

import numpy as np

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from analyze_prompt7f_circular_benchmark import (  # noqa: E402
    circular_control_volume_ledger,
)
from prepare_prompt7f_circular_benchmark import even_grid_at_least  # noqa: E402
from prepare_prompt7f_circular_benchmark import exp1_positive  # noqa: E402


class Prompt7fCircularBenchmarkTests(unittest.TestCase):
    def test_analyzer_uses_runtime_correction_enable_flag(self):
        source = (ROOT / "scripts/analyze_prompt7f_circular_benchmark.py").read_text()
        self.assertIn(
            'params.get("ctot_finite_interface_antitrapping_enabled", "0")',
            source,
        )
        self.assertIn("if correction_enabled else 0.0", source)

    def test_even_grid_preserves_or_exceeds_domain(self):
        self.assertEqual(even_grid_at_least(16.0, 0.1), 160)
        self.assertEqual(even_grid_at_least(16.0, 0.075), 214)
        self.assertEqual(even_grid_at_least(16.0, 0.06), 268)
        self.assertEqual(even_grid_at_least(16.0, 0.05), 320)

    def test_control_volume_ledger_closes_storage_identity(self):
        phi0 = np.zeros((8, 2, 8))
        phi1 = phi0.copy()
        phi1[3:5, :, 3:5] = 0.5
        c0 = np.full_like(phi0, 0.1)
        c1 = c0.copy()
        ledger = circular_control_volume_ledger(
            phi0, c0, phi1, c1, 1.0, 0.25, 3.0
        )
        self.assertAlmostEqual(
            ledger["Hdot"] + ledger["Qdot"], ledger["Cdot"], places=14
        )
        self.assertAlmostEqual(ledger["closure"], 0.0, places=14)

    def test_exp1_quadrature_matches_radial_oracle(self):
        value = float(exp1_positive(np.array([0.5625]))[0])
        self.assertAlmostEqual(value, 0.4904787689328628, places=11)


if __name__ == "__main__":
    unittest.main()
