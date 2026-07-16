import unittest

from pathlib import Path


class Next2MovingCurvedRunnerTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.text = (Path(__file__).parents[1] / "scripts"
                    / "run_next2_moving_curved_workstation.sh").read_text()

    def test_runner_pins_p2_binary_and_refuses_busy_gpu(self):
        self.assertIn("7a6cf5a03c58f160", self.text)
        self.assertIn("refusing to displace it", self.text)

    def test_runner_has_no_retry_or_dt_rewrite(self):
        self.assertNotIn("--dt", self.text)
        self.assertNotIn("ctot_step_max_retries=", self.text)
        self.assertIn("retries -ne 0", self.text)


if __name__ == "__main__":
    unittest.main()
