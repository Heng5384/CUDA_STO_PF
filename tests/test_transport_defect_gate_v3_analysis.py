import unittest

import numpy as np

from scripts import analyze_transport_defect_gate_v3 as analysis
from scripts import run_transport_defect_gate_v3_sixth_window as runner


class TransportDefectGateV3AnalysisTests(unittest.TestCase):
    def test_periodic_slab_crossings_are_directional(self):
        phi = np.zeros(16)
        phi[4:12] = 1.0
        left, right = analysis.slab_crossings(phi)
        self.assertEqual(left, 3.5)
        self.assertEqual(right, 11.5)

    def test_raw_metrics_use_physical_A_normalization(self):
        D = np.array([1.0e-8, -0.5e-8, 0.0, 0.0])
        A = np.array([0.5, 0.5, 0.0, 0.0])
        interface = np.array([True, True, False, False])
        result = analysis.raw_metrics(D, A, interface)
        self.assertAlmostEqual(result["beta_global"], 0.5e-8)
        self.assertAlmostEqual(result["beta_interface_own_mask"], 0.5e-8)
        self.assertAlmostEqual(result["eta_global"], 1.5e-8)

    def test_displacement_classification_is_separate_from_v3_gate(self):
        self.assertEqual(
            analysis.displacement_class(3.0), "MODERATE_INTERFACE_MOTION"
        )
        self.assertEqual(
            analysis.displacement_class(8.0),
            "PASS_8NM_LONG_DISPLACEMENT_COVERAGE",
        )

    def test_runner_is_limited_to_two_exact_cases(self):
        self.assertEqual(set(runner.CASES), {"strict", "candidate"})
        self.assertEqual(runner.CASES["strict"]["steps"], 16000)
        self.assertEqual(runner.CASES["candidate"]["steps"], 2000)
        self.assertEqual(
            runner.EXPECTED_BINARY_SHA256,
            "7d76e43e262a99bc452efc008a1ea0e767745f423e508555eb13aed15fffc16d",
        )


if __name__ == "__main__":
    unittest.main()
