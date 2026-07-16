import math
import unittest

from scripts.calibrate_coarse4_interface import relative_error, score_candidate_records


class CalibrateCoarse4InterfaceTests(unittest.TestCase):
    def setUp(self):
        self.weights = {
            "matrix_flux": 0.4,
            "beta_inventory": 0.4,
            "trajectory": 0.2,
        }

    @staticmethod
    def row(direction, flux=0.01, inventory=0.02, trajectory=0.04, hard=True):
        return {
            "direction": direction,
            "hard_gates_pass": hard,
            "matrix_flux_error_rel": flux,
            "beta_inventory_error_rel": inventory,
            "trajectory_error_rel": trajectory,
        }

    def test_relative_error_uses_reference(self):
        self.assertAlmostEqual(relative_error(9.0, 10.0), 0.1)

    def test_both_directions_pass_frozen_gates(self):
        score = score_candidate_records(
            [self.row("growth"), self.row("dissolution")], self.weights
        )
        self.assertTrue(score["calibration_pass"])
        self.assertTrue(math.isfinite(score["objective"]))

    def test_inventory_gate_is_not_hidden_by_weighted_objective(self):
        score = score_candidate_records(
            [self.row("growth", inventory=0.051), self.row("dissolution")],
            self.weights,
        )
        self.assertFalse(score["calibration_pass"])

    def test_missing_direction_and_hard_failure_fail_closed(self):
        missing = score_candidate_records([self.row("growth")], self.weights)
        failed = score_candidate_records(
            [self.row("growth", hard=False), self.row("dissolution")], self.weights
        )
        self.assertFalse(missing["calibration_pass"])
        self.assertFalse(failed["calibration_pass"])
        self.assertTrue(math.isinf(failed["objective"]))


if __name__ == "__main__":
    unittest.main()
