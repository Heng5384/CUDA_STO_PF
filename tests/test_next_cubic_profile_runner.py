import re
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RUNNER = ROOT / "scripts" / "run_next_cubic_profile_workstation.sh"


class NextCubicProfileRunnerTests(unittest.TestCase):
    def test_runner_freezes_p1_dt_and_profiler_contract(self):
        text = RUNNER.read_text()
        self.assertIn("echo 'dt=6.25e-6'", text)
        self.assertIn("echo 'dt_code=6.25e-6'", text)
        self.assertIn("--cuda-event-trace=false", text)
        self.assertIn("CTOT_MIMETIC_BE_ACCEPT", text)
        self.assertNotIn("CTOT_FV_BE_ACCEPT", text)
        self.assertIn("ctot_step_max_retries=0", text)
        self.assertIn("ctot_automatic_dt_growth=0", text)

    def test_runner_contains_only_allowed_cubic_matrix(self):
        text = RUNNER.read_text()
        self.assertIn("run_case cubic32_elastic_off 32 5 0", text)
        self.assertIn("run_case cubic32_elastic_on 32 3 1", text)
        self.assertIn("run_case cubic64_elastic_off 64 3 0", text)
        self.assertIn("run_case cubic64_elastic_on 64 1 1", text)
        run_sizes = {
            int(match.group(1))
            for match in re.finditer(r"^run_case\s+\S+\s+(\d+)\s+", text, re.MULTILINE)
        }
        self.assertEqual(run_sizes, {32, 64})


if __name__ == "__main__":
    unittest.main()
