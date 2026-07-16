import unittest

from scripts.analyze_next2_curved_matrix import minimum_contiguous_ratio


class AnalyzeNext2CurvedMatrixTests(unittest.TestCase):
    def test_contiguous_threshold_requires_all_larger_radii(self):
        errors = {5: 0.1, 8: 0.04, 10: 0.06, 15: 0.03, 20: 0.02}
        self.assertEqual(minimum_contiguous_ratio(errors, 0.05), "15")
        self.assertEqual(minimum_contiguous_ratio(errors, 0.01),
                         "NOT_DEMONSTRATED")


if __name__ == "__main__":
    unittest.main()
