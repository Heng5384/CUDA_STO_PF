"""Focused contracts for the post-diagnosis local closure-regression driver."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from types import SimpleNamespace
import tempfile
import unittest

from scripts import run_kwn_characteristic_closure_v2 as driver


class CharacteristicClosureRegressionV2Tests(unittest.TestCase):
    def test_full_formal_replay_comparison_checks_every_step(self) -> None:
        replay = [
            {
                "step": step,
                "time_s": step * driver.DT_S,
                "dt_s": driver.DT_S,
                "fixed_point_iterations": 2,
                "fixed_point_picard_iterations": 2,
                "fixed_point_xb_residual": 1.0e-13,
                "fixed_point_population_residual": 0.0,
                "fixed_point_cell_measure_residual": 0.0,
                "fixed_point_convergence_rate": math.nan,
                "fixed_point_convergence_mode": "DIRECT",
                "inventory_relative_residual": 0.0,
                "rmin_number_loss_m3": 0.0,
                "rmin_mol_b_loss_mol_m3": 0.0,
                "remap_number_conservation_residual_m3": 0.0,
            }
            for step in range(1, driver.LAST_ACCEPTED_STEP + 1)
        ]
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "formal.csv"
            fieldnames = ["policy", *replay[0].keys()]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for row in replay:
                    writer.writerow({"policy": "CR1_dt_0.015625s", **row})
            compared = driver._compare_full_formal_replay_trace(replay, formal_trace_csv=path)
            self.assertEqual(compared["status"], "PASS_EXACT_FORMAL_CR1_TRACE_1_TO_244_MATCH")

            replay[-1]["fixed_point_xb_residual"] = 2.0e-13
            mismatch = driver._compare_full_formal_replay_trace(replay, formal_trace_csv=path)
            self.assertEqual(mismatch["status"], "FAIL")
            self.assertEqual(mismatch["mismatches"][0]["step"], driver.LAST_ACCEPTED_STEP)

    def test_step245_acceptance_uses_the_exact_trial_tolerance(self) -> None:
        solver = SimpleNamespace(_population_convergence_rtol=1.0e-9, fixed_point_max_iterations=128)
        row = {
            "fixed_point_convergence_mode": "SAFEGUARDED_SCALAR_ROOT_V1",
            "fixed_point_periodic_cycle_period": 0,
            "fixed_point_picard_iterations": 128,
            "fixed_point_bracketed_root_iterations": 1,
            "fixed_point_bracket_left_xb": 0.006199999999,
            "fixed_point_bracket_right_xb": 0.006200000001,
            "fixed_point_bracket_left_signed_residual": 1.0e-11,
            "fixed_point_bracket_right_signed_residual": -1.0e-11,
            "fixed_point_root_verification_kind": "SAME_X_IMMUTABLE_REPLAY",
            "fixed_point_xb_residual": 4.0e-12,
            "fixed_point_xb_tolerance": 5.0e-12,
            "fixed_point_root_verification_population_residual": 0.0,
            "fixed_point_cell_measure_residual": 0.0,
            "inventory_relative_residual": 0.0,
            "nonfinite_cell_count": 0,
            "negative_cell_count": 0,
            "dt_s": driver.DT_S,
            "matrix_xB": 0.0062,
        }
        accepted = driver._step245_acceptance(row=row, solver=solver)
        self.assertEqual(accepted["status"], "PASS_STEP245_ACCEPTANCE")
        self.assertEqual(accepted["root_tolerance"], 5.0e-12)

        row["fixed_point_xb_residual"] = 5.1e-12
        rejected = driver._step245_acceptance(row=row, solver=solver)
        self.assertEqual(rejected["status"], "FAIL_STEP245_ACCEPTANCE")
        self.assertFalse(rejected["checks"]["original_xB_tolerance"])

    def test_local_window_requires_the_single_expected_safeguarded_step(self) -> None:
        rows = [
            {
                "step": step,
                "dt_s": driver.DT_S,
                "negative_cell_count": 0,
                "nonfinite_cell_count": 0,
                "inventory_relative_residual": 0.0,
                "fixed_point_convergence_mode": (
                    "SAFEGUARDED_SCALAR_ROOT_V1" if step == driver.STEP_245 else "DIRECT"
                ),
                "fixed_point_periodic_cycle_period": 0,
                "fixed_point_xb_residual": 0.0,
            }
            for step in range(driver.LOCAL_WINDOW_FIRST_STEP, driver.LOCAL_WINDOW_LAST_STEP + 1)
        ]
        passed = driver._local_window_gate(rows)
        self.assertEqual(passed["status"], "PASS_CHARACTERISTIC_LOCAL_CLOSURE_REGRESSION")

        rows[-1]["fixed_point_convergence_mode"] = "SAFEGUARDED_SCALAR_ROOT_V1"
        failed = driver._local_window_gate(rows)
        self.assertEqual(failed["status"], "FAIL_CHARACTERISTIC_LOCAL_CLOSURE_REGRESSION")
        self.assertFalse(failed["checks"]["safeguarded_root_only_at_step245"])


if __name__ == "__main__":
    unittest.main()
