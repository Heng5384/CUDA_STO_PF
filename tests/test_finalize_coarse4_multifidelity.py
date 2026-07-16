import pathlib
import tempfile
import unittest

from scripts.finalize_coarse4_multifidelity import (
    calibration_binary_provenance,
    classify_candidates,
    parse_stage1_log,
)


class FinalizeCoarse4MultifidelityTests(unittest.TestCase):
    def test_candidate_classification_fails_closed(self):
        scores = [
            {
                "phase_candidate": "finite_lphi", "a_M": "0.25",
                "objective": "0.35", "calibration_pass": "False",
                "flux_error_max": "0.13", "inventory_error_max": "0.78",
                "trajectory_error_max": "0.82",
            },
            {
                "phase_candidate": "quasi_equilibrium", "a_M": "0.0",
                "objective": "inf", "calibration_pass": "False",
                "flux_error_max": "inf", "inventory_error_max": "inf",
                "trajectory_error_max": "inf",
            },
        ]
        scan = [{
            "phase_candidate": "quasi_equilibrium", "direction": "growth",
            "failure_reason": "PHASE_KKT_FAIL_CLOSED_AT_FIRST_STEP",
        }]
        result = classify_candidates(scores, scan)
        self.assertEqual(
            result["finite_lphi"]["status"],
            "FAIL_NO_SINGLE_AM_MEETS_FLUX_INVENTORY_TRAJECTORY",
        )
        self.assertEqual(
            result["quasi_equilibrium"]["status"],
            "FAIL_PHASE_KKT_GROWTH_ALL_AM",
        )
        self.assertIsNone(result["finite_lphi"]["selected"])
        self.assertIsNone(result["quasi_equilibrium"]["selected"])

    def test_stage1_log_requires_two_clean_accepts(self):
        accepted = (
            "CTOT_MIMETIC_BE_ACCEPT step={step} mass_error=0 "
            "projection_mass=0 clip_count=0\n"
        )
        with tempfile.TemporaryDirectory() as directory:
            path = pathlib.Path(directory) / "run.log"
            path.write_text(accepted.format(step=1) + accepted.format(step=2))
            row = parse_stage1_log("finite_lphi_r0p90_elastic0", path)
            self.assertEqual(row["status"], "PASS")
            path.write_text(accepted.format(step=1))
            row = parse_stage1_log("finite_lphi_r0p90_elastic0", path)
            self.assertEqual(row["status"], "FAIL")

    def test_binary_rebuilds_require_identical_source_provenance(self):
        statuses = [
            {"binary_sha256": "old", "source_hashes": {"main": "same"}},
            {"binary_sha256": "new", "source_hashes": {"main": "same"}},
            {"binary_sha256": "new", "source_hashes": {"main": "same"}},
        ]
        result = calibration_binary_provenance(statuses, {"main": "same"})
        self.assertEqual(result["counts"], {"new": 2, "old": 1})
        self.assertEqual(result["current"], "new")
        self.assertTrue(result["all_runtime_source_hashes_match"])
        statuses[-1]["source_hashes"]["main"] = "drifted"
        result = calibration_binary_provenance(statuses, {"main": "same"})
        self.assertFalse(result["all_runtime_source_hashes_match"])


if __name__ == "__main__":
    unittest.main()
