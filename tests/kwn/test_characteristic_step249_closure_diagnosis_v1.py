"""Unit contracts for the step-249 read-only closure diagnosis."""

from __future__ import annotations

import math
from types import SimpleNamespace
import unittest

import numpy as np

from kwn_mvp.solver import SolverConfig
from scripts.run_kwn_step249_closure_diagnosis_v1 import (
    Evaluation,
    ImmutableStepMap,
    RAW_PICARD_CAP,
    _arrays_equal,
    _cycle_metrics,
    _natural_brackets,
    _raw_picard,
    _rows_equal_with_nan,
    _stage_one_status,
)
from tests.kwn.test_characteristic_reference import (
    _ConstantVelocityCharacteristic,
    _mapping,
    _numbers,
)


class _PairSolver:
    """Tiny branch-comparison stand-in for adjacent-pair contract tests."""

    def _same_cdf_source_partition(self, left: object, right: object) -> bool:
        return bool(left.cdf == right.cdf)

    def _same_trace_topology(self, left: object, right: object) -> bool:
        return bool(left.topology == right.topology)


def _evaluation(iteration: int, residual: float, *, cdf: str = "A", topology: str = "A") -> tuple[dict[str, object], Evaluation]:
    x = 0.006 + iteration * 1.0e-15
    cells = np.array([1.0, 2.0], dtype=np.float64)
    trial = SimpleNamespace(
        signed_xb_residual=residual,
        cdf=cdf,
        topology=topology,
        cell_number_m3=cells,
    )
    evaluation = Evaluation(
        evaluation_id=iteration,
        phase="unit",
        x_guess=x,
        trial=trial,
        state_hash_before="state248",
        state_hash_after="state248",
        error_type=None,
        error_message=None,
    )
    row: dict[str, object] = {
        "iteration": iteration,
        "F_signed": residual,
        "abs_F": abs(residual),
        "cdf_partition_signature": cdf,
        "departure_cell_index_signature_hash": cdf,
        "remap_topology_hash": topology,
        "population_L1_change": 0.0,
        "population_Linf_change": 0.0,
        "nonfinite_cell_count": 0,
    }
    return row, evaluation


class CharacteristicStep249DiagnosisV1Tests(unittest.TestCase):
    def test_immutable_step_map_keeps_accepted_state_and_ledger_unchanged(self) -> None:
        solver = _ConstantVelocityCharacteristic(
            SolverConfig.from_mapping(_mapping(bins=40, beta_numbers_m3=_numbers(bins=40))),
            velocity_m_s=-1.0e-12,
        )
        arrays_before = solver.state_arrays()
        ledger_before = solver.ledger.snapshot(
            matrix_xb=solver.matrix_xb,
            populations=solver.population_list(),
            beta_resolved_fraction=1.0,
        )
        map_ = ImmutableStepMap(
            solver,
            old_cell_number_m3=solver._beta_cell_numbers(),
            dt_s=1.0,
            state_label="unit_state248",
        )
        result = map_.evaluate(solver.matrix_xb, phase="unit")
        self.assertIsNotNone(result.trial)
        self.assertEqual(result.state_hash_before, result.state_hash_after)
        self.assertTrue(_arrays_equal(arrays_before, solver.state_arrays()))
        self.assertEqual(
            ledger_before,
            solver.ledger.snapshot(
                matrix_xb=solver.matrix_xb,
                populations=solver.population_list(),
                beta_resolved_fraction=1.0,
            ),
        )

    def test_raw_picard_is_uncommitted_and_records_binary_iterations(self) -> None:
        solver = _ConstantVelocityCharacteristic(
            SolverConfig.from_mapping(_mapping(bins=40, beta_numbers_m3=_numbers(bins=40))),
            velocity_m_s=-1.0e-12,
        )
        before = solver.state_arrays()
        map_ = ImmutableStepMap(
            solver,
            old_cell_number_m3=solver._beta_cell_numbers(),
            dt_s=1.0,
            state_label="unit_state248",
        )
        rows, evaluations = _raw_picard(
            map_, edges_m=solver.population("beta").grid.edges_m, iterations=8, phase="unit_raw"
        )
        self.assertEqual(len(rows), 8)
        self.assertEqual(len(evaluations), 8)
        self.assertTrue(all(bool(row["state_unchanged"]) for row in rows))
        self.assertTrue(all("x_guess_hex" in row and "F_hex" in row for row in rows))
        self.assertTrue(_arrays_equal(before, solver.state_arrays()))

    def test_final_detector_uses_only_127_128_and_never_promotes_earlier_pair(self) -> None:
        rows: list[dict[str, object]] = []
        evaluations: list[Evaluation] = []
        for iteration in range(1, RAW_PICARD_CAP + 1):
            residual = 1.0
            if iteration == 100:
                residual = -1.0
            elif iteration == 101:
                residual = 1.0
            row, evaluation = _evaluation(iteration, residual)
            rows.append(row)
            evaluations.append(evaluation)
        final, all_pairs = _natural_brackets(rows, evaluations, _PairSolver())
        self.assertEqual((final["left_iteration"], final["right_iteration"]), (127, 128))
        self.assertFalse(bool(final["strict_sign_change"]))
        earlier = [pair for pair in all_pairs if pair.get("left_iteration") == 100]
        self.assertEqual(len(earlier), 1)
        self.assertTrue(bool(earlier[0]["eligible"]))
        self.assertFalse(bool(final["eligible"]))

    def test_final_pair_requires_cdf_departure_and_topology_agreement(self) -> None:
        rows: list[dict[str, object]] = []
        evaluations: list[Evaluation] = []
        for iteration in range(1, RAW_PICARD_CAP + 1):
            residual = -1.0 if iteration == 127 else 1.0
            cdf = "A" if iteration != 128 else "B"
            row, evaluation = _evaluation(iteration, residual, cdf=cdf)
            rows.append(row)
            evaluations.append(evaluation)
        final, _pairs = _natural_brackets(rows, evaluations, _PairSolver())
        self.assertTrue(bool(final["strict_sign_change"]))
        self.assertFalse(bool(final["same_cdf_partition"]))
        self.assertFalse(bool(final["eligible"]))

    def test_cycle_metrics_checks_all_periods_one_through_sixteen(self) -> None:
        rows: list[dict[str, object]] = []
        evaluations: list[Evaluation] = []
        for iteration in range(1, RAW_PICARD_CAP + 1):
            row, evaluation = _evaluation(iteration, 1.0)
            rows.append(row)
            evaluations.append(evaluation)
        metrics = _cycle_metrics(rows, evaluations)
        periods = {int(row["period"]) for row in metrics if int(row["sample_size"]) == RAW_PICARD_CAP}
        self.assertEqual(periods, set(range(1, 17)))

    def test_nan_equal_telemetry_is_deterministic(self) -> None:
        first = [{"mode": "SAFEGUARDED", "q": math.nan, "x": 1.0}]
        second = [{"mode": "SAFEGUARDED", "q": math.nan, "x": 1.0}]
        self.assertTrue(_rows_equal_with_nan(first, second))
        self.assertFalse(_rows_equal_with_nan(first, [{"mode": "SAFEGUARDED", "q": 0.0, "x": 1.0}]))

    def test_generalization_gate_needs_reproduction_final_bracket_and_same_class(self) -> None:
        step249 = {
            "final_pair": {"strict_sign_change": True, "same_cdf_partition": True, "same_remap_topology": True},
            "local_map": {"status": "LOCAL_BRACKET_CONTINUOUS_SINGLE_CROSSING"},
        }
        reproduction = {"status": "PASS_STEP249_FAILURE_REPRODUCTION"}
        status, gate = _stage_one_status(
            reproduction=reproduction,
            step249=step249,
            failure_class="SAME_FAILURE_CLASS",
            cycle=[],
        )
        self.assertEqual(status, "STEP249_SAME_CLASS_NATURAL_BRACKET_DIAGNOSIS_ONLY")
        self.assertTrue(bool(gate["eligible"]))

    def test_no_final_bracket_is_not_overstated_as_noncontractive(self) -> None:
        status, gate = _stage_one_status(
            reproduction={"status": "PASS_STEP249_FAILURE_REPRODUCTION"},
            step249={
                "asymptotic_classification": "OSCILLATORY_APPROACH",
                "final_pair": {"strict_sign_change": False},
                "local_map": {},
            },
            failure_class="DISTINCT_FAILURE_CLASS",
            cycle=[],
        )
        self.assertEqual(status, "STEP249_NO_NATURAL_ADJACENT_BRACKET")
        self.assertFalse(bool(gate["eligible"]))


if __name__ == "__main__":
    unittest.main()
