import json
import os
import subprocess
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


class Next8RunnerTests(unittest.TestCase):
    def test_manifest_is_symbolic_and_unarmed(self):
        data = json.loads((ROOT / "examples" / "next8_T400_pilot_manifest.json").read_text())
        self.assertFalse(data["backend_eligible"])
        self.assertFalse(data["execution_authorized"])
        self.assertIsNone(data["delta_phase"])
        self.assertIsNone(data["delta_diff"])
        self.assertEqual(len(data["cases"]), 12)

    def test_runner_refuses_missing_backend_without_starting_process(self):
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / "run_next8_T400_pilot.sh")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            env={key: value for key, value in os.environ.items() if key != "SLURM_JOB_ID"},
        )
        self.assertEqual(result.returncode, 78)
        self.assertIn("next8_pilot_refused=no_independent_backend", result.stdout)
        self.assertIn("pilot_cases_completed=0", result.stdout)

    def test_runner_refuses_cluster_environment_first(self):
        env = dict(os.environ)
        env["SLURM_JOB_ID"] = "synthetic-test-only"
        result = subprocess.run(
            ["bash", str(ROOT / "scripts" / "run_next8_T400_pilot.sh")],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=False,
            env=env,
        )
        self.assertEqual(result.returncode, 77)
        self.assertIn("cluster_environment_detected", result.stderr)


if __name__ == "__main__":
    unittest.main()
