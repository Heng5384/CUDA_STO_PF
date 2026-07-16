import unittest

import numpy as np

from scripts.prepare_next2_moving_curved_benchmark import h_stable


class PrepareNext2MovingCurvedTests(unittest.TestCase):
    def test_stable_h_is_bounded_and_symmetric(self):
        phi = np.linspace(0.0, 1.0, 1001)
        values = h_stable(phi)
        self.assertGreaterEqual(float(values.min()), 0.0)
        self.assertLessEqual(float(values.max()), 1.0)
        self.assertLess(float(np.max(np.abs(values + values[::-1] - 1.0))),
                        5.0e-15)


if __name__ == "__main__":
    unittest.main()
