"""Regression coverage for adaptive CR1 reference-ladder closure."""

from __future__ import annotations

import unittest
from unittest.mock import patch

import numpy as np

from scripts import run_kwn_characteristic_reference_v1 as driver


class CharacteristicTimeRefinementTests(unittest.TestCase):
    """A reference gate must not impose an arbitrary one-refinement ceiling."""

    def test_refines_until_the_finest_adjacent_pair_meets_the_gate(self) -> None:
        # The two first added levels remain outside the 0.25% gate, while the
        # third finest adjacent pair closes it. This catches a regression to
        # the old one-extra-refinement behavior without running KWN physics.
        cumulative_by_dt_s = {
            1.0: 0.99,
            0.5: 0.99,
            0.25: 0.99,
            0.125: 1.01,
            0.0625: 1.005,
            0.03125: 1.002,
            0.015625: 1.001,
        }
        called_dt_s: list[float] = []

        def fake_run(
            context: object,
            *,
            policy: str,
            dt_s: float,
            target_times_h: object,
            max_steps: int,
            max_wall_s: float,
        ) -> driver.RunResult:
            del context, target_times_h, max_steps, max_wall_s
            called_dt_s.append(dt_s)
            snapshots = []
            measures = []
            for time_h in driver.SHORT_TIMES_H:
                row = {metric: 1.0 for metric in driver.PRIMARY_METRICS}
                row["time_h"] = time_h
                row["time_s"] = time_h * 3600.0
                row["cumulative_number_dissolution_m3"] = (
                    0.0 if time_h == 0.0 else cumulative_by_dt_s[dt_s]
                )
                snapshots.append(row)
                measures.append((np.asarray([1.0]), np.asarray([1.0])))
            return driver.RunResult(
                name=policy,
                status="PASS_CHARACTERISTIC_RUN",
                reason=None,
                snapshots=snapshots,
                measures=measures,
                trace_rows=[],
                accepted_steps=1,
                runtime_s=0.0,
                max_inventory_relative_residual=0.0,
                max_fixed_point_residual=0.0,
            )

        with patch.object(driver, "_run_characteristic", side_effect=fake_run):
            result = driver._characteristic_self_convergence(
                object(),
                dt_policy={"levels_s": [1.0, 0.5, 0.25, 0.125]},
                max_steps=100,
                max_wall_s=10.0,
            )

        self.assertEqual(result["status"], "PASS_CHARACTERISTIC_SELF_CONVERGENCE")
        self.assertEqual(result["additional_refinement_count"], 3)
        self.assertEqual(result["policy"]["dt_s"], 0.03125)
        self.assertEqual(called_dt_s, [1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625])

    def test_resource_limited_refinement_is_blocked_not_failed(self) -> None:
        called_dt_s: list[float] = []

        def fake_run(
            context: object,
            *,
            policy: str,
            dt_s: float,
            target_times_h: object,
            max_steps: int,
            max_wall_s: float,
        ) -> driver.RunResult:
            del context, target_times_h, max_steps, max_wall_s
            called_dt_s.append(dt_s)
            if dt_s == 0.0625:
                return driver.RunResult(
                    name=policy,
                    status="INCOMPLETE_CHARACTERISTIC_RUN",
                    reason="WorkflowError: characteristic max_steps=100 reached before 0.1 h",
                    snapshots=[],
                    measures=[],
                    trace_rows=[],
                    accepted_steps=100,
                    runtime_s=0.0,
                    max_inventory_relative_residual=0.0,
                    max_fixed_point_residual=0.0,
                )
            snapshots = []
            measures = []
            for time_h in driver.SHORT_TIMES_H:
                row = {metric: 1.0 for metric in driver.PRIMARY_METRICS}
                row["time_h"] = time_h
                row["time_s"] = time_h * 3600.0
                row["cumulative_number_dissolution_m3"] = 0.0 if time_h == 0.0 else (1.01 if dt_s == 0.125 else 0.99)
                snapshots.append(row)
                measures.append((np.asarray([1.0]), np.asarray([1.0])))
            return driver.RunResult(
                name=policy,
                status="PASS_CHARACTERISTIC_RUN",
                reason=None,
                snapshots=snapshots,
                measures=measures,
                trace_rows=[],
                accepted_steps=1,
                runtime_s=0.0,
                max_inventory_relative_residual=0.0,
                max_fixed_point_residual=0.0,
            )

        with patch.object(driver, "_run_characteristic", side_effect=fake_run):
            result = driver._characteristic_self_convergence(
                object(),
                dt_policy={"levels_s": [1.0, 0.5, 0.25, 0.125]},
                max_steps=100,
                max_wall_s=10.0,
            )

        self.assertEqual(result["status"], "BLOCKED_TIME_REFERENCE_NOT_CLOSED")
        self.assertEqual(result["attempted_levels_s"], [1.0, 0.5, 0.25, 0.125, 0.0625])
        self.assertEqual(called_dt_s, [1.0, 0.5, 0.25, 0.125, 0.0625])

    def test_final_status_preserves_a_numerical_convergence_failure(self) -> None:
        final = driver._final_record(
            launch={"git_branch_at_launch": driver.REQUIRED_BRANCH, "git_head_at_launch": "c" * 40},
            blocker={
                "status": "BLOCKED_EXACT_DONOR_BOUND_REFERENCE",
                "strict_donor_dt_s": 1.0e-20,
                "frozen_min_dt_s": 1.0e-12,
                "theoretical_steps_to_0p1h": 1.0e20,
            },
            inherited_tests={
                "status": "PASS",
                "test_count": 52,
                "strict_ssprk2_unit_tests": {"status": "PASS", "test_count": 9},
            },
            boundary={"status": "PASS_LOWER_BOUNDARY_OPERATOR_PARITY"},
            current_cap4={"status": "PASS_CURRENT_VS_CAP4_REPRODUCTION", "difference": {}},
            characteristic_tests={"status": "PASS_CHARACTERISTIC_REFERENCE_UNIT_TESTS"},
            convergence={"status": "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS"},
            restart={"status": "BLOCKED_PREREQUISITE_GATE"},
            parity={"status": "BLOCKED_PREREQUISITE_GATE"},
            frozen={"status": "BLOCKED_PREREQUISITE_GATE"},
            ladder={"status": "BLOCKED_PREREQUISITE_GATE"},
            one_hour={"status": "BLOCKED_PREREQUISITE_GATE"},
            choice={},
            qualification={"status": "BLOCKED_PREREQUISITE_GATE"},
            authority={},
            final_crosscheck={"status": "BLOCKED_PREREQUISITE_GATE"},
            beta={"status": "BLOCKED_PREREQUISITE_GATE"},
        )
        self.assertEqual(final["STATUS"], "FAIL_CHARACTERISTIC_REFERENCE_NUMERICS")


if __name__ == "__main__":
    unittest.main()
