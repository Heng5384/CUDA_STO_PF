import unittest

from pathlib import Path


class Next2SharpConvergenceSourceTests(unittest.TestCase):
    def test_reference_is_independent_and_refined(self):
        text = (Path(__file__).parents[1] / "scripts"
                / "run_next2_sharp_convergence.py").read_text()
        self.assertIn("time_refined_1024", text)
        self.assertIn("no_PF_velocity_fit", text)
        self.assertNotIn("correction coefficient", text.lower())


if __name__ == "__main__":
    unittest.main()
