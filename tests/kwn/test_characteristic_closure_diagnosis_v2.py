"""Focused regression coverage for the non-mutating CR1 closure diagnosis."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

import numpy as np

from kwn_mvp.solver import SolverConfig
from scripts.diagnose_kwn_characteristic_closure_v2 import (
    ImmutableStep245Map,
    _classify,
    _compare_formal_step244_trace,
)
from tests.kwn.test_characteristic_reference import (
    _ConstantVelocityCharacteristic,
    _mapping,
    _numbers,
)


class CharacteristicClosureDiagnosisV2Tests(unittest.TestCase):
    def test_map_evaluation_is_side_effect_free(self) -> None:
        """A diagnostic trial must not alter the accepted state it reads."""

        solver = _ConstantVelocityCharacteristic(
            SolverConfig.from_mapping(_mapping(bins=40, beta_numbers_m3=_numbers(bins=40))),
            velocity_m_s=-1.0e-12,
        )
        before = {key: value.copy() for key, value in solver.state_arrays().items()}
        closure_map = ImmutableStep245Map(
            solver,
            old_cell_number_m3=solver._beta_cell_numbers(),
            dt_s=1.0,
        )
        evaluation = closure_map.evaluate(solver.matrix_xb, phase="unit")
        self.assertIsNotNone(evaluation.trial)
        self.assertEqual(evaluation.state_hash_before, evaluation.state_hash_after)
        for key, expected in before.items():
            np.testing.assert_array_equal(expected, solver.state_arrays()[key], err_msg=key)

    def test_unique_noncontractive_root_is_not_mislabelled_as_a_cycle(self) -> None:
        root = {
            "root_found": True,
            "root_xB": 0.0062,
            "root_residual": 0.0,
            "root_tolerance": 1.0e-12,
            "verification_residual": 0.0,
            "verification_population_residual": 1.0e-12,
            "population_tolerance": 1.0e-9,
            "signature_changed_within_bracket": False,
            "stop_reason": "residual_tolerance_reached",
        }
        result = _classify(
            raw_rows=[{"abs_F": 1.0e-8, "xB_tolerance": 1.0e-12}],
            cycles=[],
            contraction={"last_64": {"median_q_F": 1.01}},
            roots=[root],
            brackets=[],
        )
        self.assertEqual(result["step245_classification"], "STEP245_NONCONTRACTIVE_BUT_UNIQUE_SCALAR_ROOT")
        self.assertTrue(result["scalar_root_exists"])
        self.assertEqual(result["number_of_admissible_roots"], 1)

    def test_formal_trace_comparison_requires_exactly_equal_replay_values(self) -> None:
        observed = {
            "step": 244,
            "time_s": 3.8125,
            "dt_s": 0.015625,
            "fixed_point_iterations": 2,
            "fixed_point_picard_iterations": 2,
            "fixed_point_xb_residual": 6.0e-13,
            "fixed_point_population_residual": 2.4e-11,
            "fixed_point_cell_measure_residual": 2.6e-9,
            "fixed_point_convergence_rate": 5.0e-5,
            "fixed_point_convergence_mode": "DIRECT",
            "inventory_relative_residual": 0.0,
            "rmin_number_loss_m3": 2.0e-16,
            "rmin_mol_b_loss_mol_m3": 2.0e-39,
            "remap_number_conservation_residual_m3": 0.0,
        }
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "formal.csv"
            row = {"policy": "CR1_dt_0.015625s", **{key: str(value) for key, value in observed.items()}}
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                writer.writeheader()
                writer.writerow(row)
            result = _compare_formal_step244_trace([observed], formal_trace_csv=path)
        self.assertEqual(result["status"], "PASS_EXACT_FORMAL_TRACE_MATCH")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
