"""Focused regression coverage for the non-mutating CR1 closure diagnosis."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from kwn_mvp.solver import SolverConfig
from scripts.diagnose_kwn_characteristic_closure_v2 import (
    Evaluation,
    EvaluationBudget,
    ImmutableStep245Map,
    MAX_POST_RAW_MAP_EVALUATIONS,
    _classify,
    _cycle_analysis,
    _compare_formal_step244_trace,
    _group_contiguous_interval_records,
    _raw_picard,
    _replay_to_step244,
    _scalar_scan,
    _same_cdf_source_partition,
    _solve_brackets,
    _trace_topology_for_trial,
    _trace_topology_pair_evidence,
)
from tests.kwn.test_characteristic_reference import (
    _ConstantVelocityCharacteristic,
    _mapping,
    _numbers,
)


class _SyntheticClosureMap:
    """Small deterministic map used to exercise scan-control decisions only."""

    def __init__(self, residual_and_branch: object) -> None:
        self._residual_and_branch = residual_and_branch
        self._next_id = 1
        self._cache: dict[str, Evaluation] = {}
        self.evaluations: list[Evaluation] = []
        self.solver = SimpleNamespace(
            _population_convergence_rtol=1.0e-9,
            _population_observable_residual=lambda _candidate, _previous: 0.0,
        )

    def evaluate(self, x_guess: float, *, phase: str, reuse: bool = False) -> Evaluation:
        value = float(x_guess)
        key = value.hex()
        if reuse and key in self._cache:
            cached = self._cache[key]
            return Evaluation(
                evaluation_id=cached.evaluation_id,
                phase=f"{phase}:cached",
                x_guess=cached.x_guess,
                trial=cached.trial,
                state_hash_before=cached.state_hash_before,
                state_hash_after=cached.state_hash_after,
                error_type=cached.error_type,
                error_message=cached.error_message,
            )
        residual, branch = self._residual_and_branch(value)
        departure_faces = np.array(
            [1.25 if branch == "A" else 2.25, 1.75, 2.25], dtype=np.float64
        )
        trial = SimpleNamespace(
            matrix_xb=value + float(residual),
            signed_xb_residual=float(residual),
            xb_tolerance=1.0e-12,
            midpoint_matrix_xb=value,
            cell_number_m3=np.array([1.0, 1.0], dtype=np.float64),
            inventory=SimpleNamespace(
                total_mol_m3=1.0,
                beta_resolved_mol_m3=0.5,
                matrix_mol_m3=0.5,
                gp_mol_m3=0.0,
                relative_residual=0.0,
                residual_mol_m3=0.0,
                matrix_fraction=1.0,
            ),
            trace=SimpleNamespace(
                arrival_faces_m=np.array([1.0, 2.0, 3.0], dtype=np.float64),
                departure_faces_m=departure_faces,
                midpoint_faces_m=np.array([1.0, 2.0, 3.0], dtype=np.float64),
                lower_no_inflow_face_count=0,
                upper_no_inflow_face_count=0,
            ),
            remap=SimpleNamespace(
                lower_number_loss_m3=0.0,
                upper_number_loss_m3=0.0,
                conservation_residual_m3=0.0,
            ),
        )
        evaluation = Evaluation(
            evaluation_id=self._next_id,
            phase=phase,
            x_guess=value,
            trial=trial,
            state_hash_before="state244",
            state_hash_after="state244",
            error_type=None,
            error_message=None,
        )
        self._next_id += 1
        self.evaluations.append(evaluation)
        self._cache[key] = evaluation
        return evaluation


class CharacteristicClosureDiagnosisV2Tests(unittest.TestCase):
    @staticmethod
    def _valid_root(xb: float = 0.0062, **overrides: object) -> dict[str, object]:
        root: dict[str, object] = {
            "root_found": True,
            "root_xB": xb,
            "root_residual": 0.0,
            "root_tolerance": 1.0e-12,
            "verification_residual": 0.0,
            "verification_tolerance": 1.0e-12,
            "verification_population_residual": 0.0,
            "population_tolerance": 1.0e-9,
            "candidate_inventory_relative_residual": 0.0,
            "verification_inventory_relative_residual": 0.0,
            "inventory_relative_tolerance": 1.0e-10,
            "candidate_nonnegative_cells": True,
            "verification_nonnegative_cells": True,
            "bracket_kind": "SIGN_CHANGE",
            "root_authority_signature_continuous": True,
            "root_interval_left_x": xb - 4.0e-12,
            "root_interval_right_x": xb + 4.0e-12,
            "root_location_tolerance": 8.0e-12,
        }
        root.update(overrides)
        return root

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

    def test_trace_topology_audit_is_read_only_on_a_real_characteristic_trial(self) -> None:
        """The topology key mirrors a real trace without touching state 244."""

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
        first = closure_map.evaluate(solver.matrix_xb, phase="unit_topology")
        repeat = closure_map.evaluate(solver.matrix_xb, phase="unit_topology_repeat")
        evidence = _trace_topology_pair_evidence(
            closure_map,
            edges_m=solver.population("beta").grid.edges_m,
            left=first,
            right=repeat,
        )
        diagnostic_topology = _trace_topology_for_trial(
            closure_map,
            edges_m=solver.population("beta").grid.edges_m,
            evaluation=first,
        )
        self.assertIsNotNone(first.trial)
        runtime_topology = solver._trace_topology(first.trial)
        self.assertIsNotNone(runtime_topology)
        assert runtime_topology is not None
        self.assertEqual(diagnostic_topology.mode, runtime_topology.mode)
        np.testing.assert_array_equal(diagnostic_topology.node_sign, runtime_topology.node_sign)
        np.testing.assert_array_equal(
            diagnostic_topology.gauss_left_sign, runtime_topology.gauss_left_sign
        )
        np.testing.assert_array_equal(
            diagnostic_topology.gauss_right_sign, runtime_topology.gauss_right_sign
        )
        np.testing.assert_array_equal(
            diagnostic_topology.valid_interval, runtime_topology.valid_interval
        )
        np.testing.assert_array_equal(
            diagnostic_topology.arrival_face_run_id, runtime_topology.arrival_face_run_id
        )
        self.assertEqual(
            diagnostic_topology.lower_no_inflow_face_count,
            runtime_topology.lower_no_inflow_face_count,
        )
        self.assertEqual(
            diagnostic_topology.upper_no_inflow_face_count,
            runtime_topology.upper_no_inflow_face_count,
        )
        self.assertEqual(
            diagnostic_topology.identity_departure_map,
            runtime_topology.identity_departure_map,
        )
        self.assertEqual(evidence["trace_topology_status"], "PASS_SAME_TRACE_TOPOLOGY")
        self.assertEqual(evidence["left_trace_topology"]["trace_topology_mode"], "CONSTANT_TRANSLATION")
        for key, expected in before.items():
            np.testing.assert_array_equal(expected, solver.state_arrays()[key], err_msg=key)

    def test_targeted_scalar_scan_is_bounded_and_never_certifies_root_count(self) -> None:
        """The real immutable-map path uses coarse plus bounded local probes, not dense expansion."""

        solver = _ConstantVelocityCharacteristic(
            SolverConfig.from_mapping(_mapping(bins=40, beta_numbers_m3=_numbers(bins=40))),
            velocity_m_s=-1.0e-12,
        )
        closure_map = ImmutableStep245Map(
            solver,
            old_cell_number_m3=solver._beta_cell_numbers(),
            dt_s=1.0,
        )
        raw_rows, cells, departures = _raw_picard(
            closure_map,
            edges_m=solver.population("beta").grid.edges_m,
            iterations=16,
        )
        cycles = _cycle_analysis(raw_rows, cells, departures)
        scalar_rows, brackets, _refined_rows, audit, budget = _scalar_scan(
            closure_map,
            edges_m=solver.population("beta").grid.edges_m,
            interval={"x_min": 0.0, "x_max": 0.02},
            raw_rows=raw_rows,
            cycles=cycles,
            coarse_points=257,
            refinement_depth=8,
        )
        self.assertGreaterEqual(len(scalar_rows), 257)
        self.assertEqual(audit["scan_strategy"], "COARSE_257_PLUS_BOUNDED_TARGETED_REFINEMENT_V1")
        self.assertFalse(audit["root_count_certified"])
        self.assertLessEqual(budget.new_evaluation_count, MAX_POST_RAW_MAP_EVALUATIONS)
        self.assertLessEqual(
            audit["evaluation_budget"]["planned_worst_case_post_raw_map_evaluations"],
            MAX_POST_RAW_MAP_EVALUATIONS,
        )
        self.assertTrue(
            all(
                item.get("left_signature") == item.get("right_signature")
                for item in brackets
                if item.get("bracket_kind") == "SIGN_CHANGE"
            )
        )

    def test_cross_signature_sign_change_is_not_a_root_authority(self) -> None:
        """A sign flip across remap branches is transition evidence, never a bisection bracket."""

        closure_map = _SyntheticClosureMap(
            lambda value: (value - 0.502, "A" if value < 0.501 else "B")
        )
        _rows, brackets, _refined_rows, audit, _budget = _scalar_scan(
            closure_map,
            edges_m=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            interval={"x_min": 0.0, "x_max": 1.0},
            raw_rows=[{"x_guess": 0.9, "error_type": ""}],
            cycles=[],
            coarse_points=257,
            refinement_depth=8,
        )
        self.assertGreater(audit["cross_signature_sign_change_count"], 0)
        self.assertTrue(
            any(
                item.get("bracket_kind") == "SIGN_CHANGE_ACROSS_CDF_SOURCE_PARTITION"
                for item in brackets
            )
        )
        self.assertTrue(
            all(
                item.get("left_signature") == item.get("right_signature")
                for item in brackets
                if item.get("bracket_kind") == "SIGN_CHANGE"
            )
        )

    def test_cdf_partition_requires_endpoint_masks_and_no_inflow_counts(self) -> None:
        """The diagnosis must consume the runtime's full six-part CDF key."""

        edges = np.array([1.0, 2.0, 3.0], dtype=np.float64)

        def evaluation(*, departure: np.ndarray, lower_no_inflow: int, identifier: int) -> Evaluation:
            return Evaluation(
                evaluation_id=identifier,
                phase="unit_cdf_partition",
                x_guess=0.5,
                trial=SimpleNamespace(
                    trace=SimpleNamespace(
                        arrival_faces_m=edges.copy(),
                        departure_faces_m=departure,
                        midpoint_faces_m=edges.copy(),
                        lower_no_inflow_face_count=lower_no_inflow,
                        upper_no_inflow_face_count=0,
                    )
                ),
                state_hash_before="state244",
                state_hash_after="state244",
                error_type=None,
                error_message=None,
            )

        at_lower_endpoint = evaluation(
            departure=np.array([1.0, 1.5, 2.5], dtype=np.float64),
            lower_no_inflow=1,
            identifier=1,
        )
        just_inside_lower = evaluation(
            departure=np.array([np.nextafter(1.0, np.inf), 1.5, 2.5], dtype=np.float64),
            lower_no_inflow=0,
            identifier=2,
        )
        self.assertFalse(
            _same_cdf_source_partition(at_lower_endpoint, just_inside_lower, edges_m=edges)
        )

    def test_narrow_same_branch_sign_bracket_samples_midpoint_and_repeats_same_x(self) -> None:
        """A noncontractive root uses same-x replay, not an extra Picard step."""

        root_x = 0.5 + 5.0e-14
        # T(x) = root_x - 10 (x - root_x), so F(x) = -11 (x - root_x).
        # The first midpoint is tolerance-valid but not exact; one extra
        # Picard step multiplies |F| by ten and fails the scalar tolerance.
        closure_map = _SyntheticClosureMap(lambda value: (-11.0 * (value - root_x), "A"))
        left = 0.5 - 2.0e-12
        right = 0.5 + 2.0e-12
        budget = EvaluationBudget(
            closure_map=closure_map,
            maximum_new_evaluations=32,
            start_evaluation_count=0,
        )
        roots, _audit = _solve_brackets(
            closure_map,
            edges_m=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            brackets=[
                {
                    "bracket_kind": "SIGN_CHANGE",
                    "reason": "unit",
                    "left_x": left,
                    "right_x": right,
                    "sign_change": True,
                    "root_search_selected": True,
                }
            ],
            budget=budget,
        )
        self.assertEqual(len(roots), 1)
        root = roots[0]
        self.assertTrue(root["root_found"])
        self.assertEqual(root["stop_reason"], "midpoint_scalar_tolerance")
        self.assertEqual(root["verification_kind"], "SAME_X_REPEATABILITY")
        self.assertTrue(root["same_x_repeatable"])
        self.assertLessEqual(abs(float(root["root_residual"])), float(root["root_tolerance"]))
        old_style = closure_map.evaluate(
            float(root["root_xB"]) + float(root["root_residual"]),
            phase="unit_old_picard_style_probe",
            reuse=False,
        )
        self.assertIsNotNone(old_style.trial)
        self.assertGreater(
            abs(float(old_style.trial.signed_xb_residual)),
            float(old_style.trial.xb_tolerance),
        )
        self.assertGreaterEqual(budget.new_evaluation_count, 4)

    def test_point_candidate_is_not_root_authority_without_a_sign_bracket(self) -> None:
        """A tolerance-valid scan point cannot authorize the safeguarded path alone."""

        closure_map = _SyntheticClosureMap(lambda value: (value - 0.5, "A"))
        budget = EvaluationBudget(
            closure_map=closure_map,
            maximum_new_evaluations=16,
            start_evaluation_count=0,
        )
        roots, _audit = _solve_brackets(
            closure_map,
            edges_m=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            brackets=[
                {
                    "bracket_kind": "POINT_CANDIDATE",
                    "reason": "unit",
                    "left_x": 0.5,
                    "right_x": 0.5,
                    "root_search_selected": True,
                }
            ],
            budget=budget,
        )
        self.assertEqual(len(roots), 1)
        root = roots[0]
        self.assertTrue(root["root_candidate_within_scalar_tolerance"])
        self.assertTrue(root["same_x_repeatable"])
        self.assertFalse(root["root_authority_signature_continuous"])
        self.assertFalse(root["root_found"])

    def test_trace_topology_transition_fails_closed_as_a_p0_bracket(self) -> None:
        """A real trace-branch change cannot be hidden by equal CDF indices."""

        closure_map = _SyntheticClosureMap(lambda value: (value - 0.5, "A"))
        closure_map.solver._velocity_at_radii = lambda radii, midpoint: np.full_like(
            radii, -1.0 if midpoint < 0.5 else 1.0
        )
        closure_map.solver.state_arrays = lambda: {}
        budget = EvaluationBudget(
            closure_map=closure_map,
            maximum_new_evaluations=16,
            start_evaluation_count=0,
        )
        roots, audit = _solve_brackets(
            closure_map,
            edges_m=np.array([1.0, 2.0, 3.0], dtype=np.float64),
            brackets=[
                {
                    "bracket_kind": "SIGN_CHANGE",
                    "reason": "unit",
                    "left_x": 0.49,
                    "right_x": 0.51,
                    "sign_change": True,
                    "root_search_selected": True,
                }
            ],
            budget=budget,
        )
        self.assertEqual(roots, [])
        self.assertTrue(
            any(
                row.get("status")
                == "TRACE_TOPOLOGY_TRANSITION_WITHIN_ROOT_BRACKET_UNRESOLVED"
                for row in audit
            )
        )
        classification = _classify(
            raw_rows=[{"abs_F": 1.0e-8, "xB_tolerance": 1.0e-12}],
            cycles=[],
            contraction={"last_64": {"median_q_F": 1.01}},
            roots=[],
            brackets=audit,
        )
        self.assertEqual(
            classification["step245_classification"],
            "STEP245_DISCONTINUOUS_REMAP_ROOT_UNRESOLVED",
        )

    def test_source_cell_cdf_kink_does_not_by_itself_classify_a_discontinuous_remap(self) -> None:
        """A source-index hash change is not a proof that the CDF map jumps."""

        result = _classify(
            raw_rows=[{"abs_F": 1.0e-8, "xB_tolerance": 1.0e-12}],
            cycles=[],
            contraction={"last_64": {"median_q_F": 1.01}},
            roots=[],
            brackets=[
                {
                    "bracket_kind": "SIGNATURE_TRANSITION",
                    "status": "MULTIPLE_SOURCE_CELL_CDF_KINKS_OBSERVED",
                }
            ],
        )
        self.assertEqual(result["step245_classification"], "OTHER_WITH_EXPLICIT_EVIDENCE")
        self.assertFalse(result["discontinuity_evidence"])
        self.assertTrue(result["source_cell_signature_transition_observed"])

    def test_nonroot_trace_topology_observation_is_not_a_root_discontinuity(self) -> None:
        """An unrelated topology transition cannot veto a separate root bracket."""

        result = _classify(
            raw_rows=[{"abs_F": 1.0e-8, "xB_tolerance": 1.0e-12}],
            cycles=[],
            contraction={"last_64": {"median_q_F": 1.01}},
            roots=[],
            brackets=[
                {
                    "bracket_kind": "SIGNATURE_TRANSITION",
                    "status": "TRACE_TOPOLOGY_TRANSITION_OBSERVED_NONROOT",
                    "sign_change": False,
                }
            ],
        )
        self.assertEqual(result["step245_classification"], "OTHER_WITH_EXPLICIT_EVIDENCE")
        self.assertFalse(result["discontinuity_evidence"])

    def test_tangent_targeting_skips_a_cross_branch_local_minimum(self) -> None:
        """Only a three-point same-branch minimum enters the tangent-target refinement plan."""

        edges = np.array([1.0, 2.0, 3.0], dtype=np.float64)
        same_branch = _SyntheticClosureMap(lambda value: ((value - 0.33) ** 2 + 1.0e-8, "A"))
        _rows, _brackets, _refined, same_audit, _budget = _scalar_scan(
            same_branch,
            edges_m=edges,
            interval={"x_min": 0.0, "x_max": 1.0},
            raw_rows=[{"x_guess": 0.9, "error_type": ""}],
            cycles=[],
            coarse_points=257,
            refinement_depth=8,
        )
        self.assertTrue(
            any(
                "COARSE_LOCAL_ABS_F_MINIMUM" in item["reasons"]
                for item in same_audit["targeted_regions"]
            )
        )
        cross_branch = _SyntheticClosureMap(
            lambda value: ((value - 0.33) ** 2 + 1.0e-8, "A" if value < 0.33 else "B")
        )
        _rows, _brackets, _refined, cross_audit, _budget = _scalar_scan(
            cross_branch,
            edges_m=edges,
            interval={"x_min": 0.0, "x_max": 1.0},
            raw_rows=[{"x_guess": 0.9, "error_type": ""}],
            cycles=[],
            coarse_points=257,
            refinement_depth=8,
        )
        self.assertFalse(
            any(
                "COARSE_LOCAL_ABS_F_MINIMUM" in item["reasons"]
                for item in cross_audit["targeted_regions"]
            )
        )

    def test_finite_scan_root_does_not_claim_unique_noncontractive_root(self) -> None:
        root = self._valid_root(verification_population_residual=1.0e-12)
        result = _classify(
            raw_rows=[{"abs_F": 1.0e-8, "xB_tolerance": 1.0e-12}],
            cycles=[],
            contraction={"last_64": {"median_q_F": 1.01}},
            roots=[root],
            brackets=[],
        )
        self.assertEqual(result["step245_classification"], "OTHER_WITH_EXPLICIT_EVIDENCE")
        self.assertTrue(result["scalar_root_exists"])
        self.assertFalse(result["root_count_certified"])
        self.assertIsNone(result["number_of_admissible_roots"])
        self.assertEqual(result["scalar_root_uniqueness_status"], "UNRESOLVED_NO_INTERVAL_OR_ANALYTIC_ENCLOSURE")

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

    def test_replay_trace_carries_every_formal_comparison_field(self) -> None:
        """The replay schema must feed the strict formal step-244 comparator."""

        diagnostic = SimpleNamespace(
            step=1,
            time_s=0.015625,
            dt_s=0.015625,
            matrix_xb=0.012,
            fixed_point_iterations=2,
            fixed_point_picard_iterations=2,
            fixed_point_xb_residual=1.0e-13,
            fixed_point_xb_tolerance=1.0e-12,
            fixed_point_population_residual=2.0e-13,
            fixed_point_cell_measure_residual=3.0e-13,
            fixed_point_convergence_rate=4.0e-5,
            fixed_point_convergence_mode="DIRECT",
            inventory=SimpleNamespace(relative_residual=0.0),
            rmin_number_loss_m3=0.0,
            rmin_mol_b_loss_mol_m3=0.0,
            remap_number_conservation_residual_m3=0.0,
        )

        class FakeSolver:
            def __init__(self, _config: object) -> None:
                self._diagnostic = diagnostic

            def advance_one(self, *, maximum_dt_s: float) -> SimpleNamespace:
                self._diagnostic.time_s = maximum_dt_s
                return self._diagnostic

        context = SimpleNamespace(contract_hash="d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333", mapping={})
        with patch("scripts.diagnose_kwn_characteristic_closure_v2.build_frozen_canonical_context", return_value=context), \
             patch("scripts.diagnose_kwn_characteristic_closure_v2.CharacteristicReferenceSolver", FakeSolver), \
             patch("scripts.diagnose_kwn_characteristic_closure_v2.SolverConfig.from_mapping", return_value=object()):
            _solver, _context, trace = _replay_to_step244(dt_s=0.015625, accepted_step=1)
        self.assertEqual(trace[0]["fixed_point_convergence_rate"], 4.0e-5)

    def test_cycle_analysis_uses_minimum_primitive_period(self) -> None:
        """A P1 fixed point must never be reported as a P2/P4 detector miss."""

        rows = [
            {"x_guess": 0.01, "x_closure": 0.01, "F_signed": 0.0, "departure_signature_hash": "same"}
            for _ in range(17)
        ]
        cells = [np.array([1.0, 2.0], dtype=np.float64) for _ in rows]
        departures = [np.array([1.0, 2.0], dtype=np.float64) for _ in rows]
        cycles = _cycle_analysis(rows, cells, departures)
        final = [row for row in cycles if row["iteration"] == len(rows)]
        primitive = [row["period"] for row in final if row["primitive_exact_period"]]
        self.assertEqual(primitive, [1])
        result = _classify(
            raw_rows=[
                {
                    **rows[-1],
                    "abs_F": 0.0,
                    "xB_tolerance": 1.0e-12,
                    "unchanged_from_state244": True,
                }
            ],
            cycles=cycles,
            contraction={"last_64": {"median_q_F": 0.0}},
            roots=[self._valid_root(0.01)],
            brackets=[],
        )
        self.assertEqual(result["step245_classification"], "OTHER_WITH_EXPLICIT_EVIDENCE")
        self.assertEqual(result["picard_regime"], "DIRECT_CONVERGED")

    def test_cycle_analysis_marks_distinct_opposite_sign_p2_as_primitive(self) -> None:
        """Only a genuine two-phase orbit is eligible for the legacy P2 path."""

        x_values = (0.01, 0.02) * 9
        rows = [
            {
                "x_guess": value,
                "x_closure": 0.02 if value == 0.01 else 0.01,
                "F_signed": 0.01 if value == 0.01 else -0.01,
                "departure_signature_hash": "A" if value == 0.01 else "B",
            }
            for value in x_values
        ]
        cells = [np.array([value, 1.0], dtype=np.float64) for value in x_values]
        departures = [np.array([value], dtype=np.float64) for value in x_values]
        cycles = _cycle_analysis(rows, cells, departures)
        final = [row for row in cycles if row["iteration"] == len(rows)]
        primitive = [row for row in final if row["primitive_exact_period"]]
        self.assertEqual([row["period"] for row in primitive], [2])
        self.assertTrue(primitive[0]["phase_values_distinct"])
        self.assertTrue(primitive[0]["cyclic_adjacent_residual_sign_change"])

    def test_cycle_analysis_marks_distinct_opposite_sign_p4_as_primitive(self) -> None:
        """A genuine P4 remains distinct from a P1/P2 recurrence."""

        x_values = (0.01, 0.02, 0.04, 0.03) * 5
        closures = (0.02, 0.04, 0.03, 0.01) * 5
        signs = (0.01, -0.01, 0.01, -0.01) * 5
        rows = [
            {
                "x_guess": x_value,
                "x_closure": closure,
                "F_signed": sign,
                "departure_signature_hash": x_value.hex(),
            }
            for x_value, closure, sign in zip(x_values, closures, signs)
        ]
        cells = [np.array([value, 1.0], dtype=np.float64) for value in x_values]
        departures = [np.array([value], dtype=np.float64) for value in x_values]
        cycles = _cycle_analysis(rows, cells, departures)
        final = [row for row in cycles if row["iteration"] == len(rows)]
        primitive = [row for row in final if row["primitive_exact_period"]]
        self.assertEqual([row["period"] for row in primitive], [4])
        self.assertTrue(primitive[0]["phase_values_distinct"])
        self.assertTrue(primitive[0]["cyclic_adjacent_residual_sign_change"])
        result = _classify(
            raw_rows=[{"abs_F": 0.01, "xB_tolerance": 1.0e-12}],
            cycles=cycles,
            contraction={"last_64": {"median_q_F": 1.0}},
            roots=[self._valid_root()],
            brackets=[],
        )
        self.assertEqual(result["step245_classification"], "STEP245_P2_OR_P4_DETECTION_FAILURE")
        self.assertIsNone(result["number_of_admissible_roots"])

    def test_scalar_residual_alone_does_not_mark_raw_picard_direct(self) -> None:
        """The original direct contract also requires M0--M3 convergence."""

        result = _classify(
            raw_rows=[
                {
                    "abs_F": 0.0,
                    "xB_tolerance": 1.0e-12,
                    "population_observable_residual_to_previous": 1.0e-5,
                    "population_convergence_tolerance": 1.0e-9,
                }
            ],
            cycles=[],
            contraction={"last_64": {"median_q_F": 1.1}},
            roots=[self._valid_root()],
            brackets=[],
        )
        self.assertFalse(result["raw_picard_direct_convergence_observed"])
        self.assertEqual(result["step245_classification"], "OTHER_WITH_EXPLICIT_EVIDENCE")

    def test_overlapping_root_intervals_are_not_multiple_roots(self) -> None:
        """Residual-tolerance representatives are merged by their final brackets."""

        first = self._valid_root(
            0.01,
            root_interval_left_x=0.009999999996,
            root_interval_right_x=0.010000000004,
        )
        second = self._valid_root(
            0.010000000006,
            root_interval_left_x=0.010000000002,
            root_interval_right_x=0.010000000010,
        )
        result = _classify(
            raw_rows=[{"abs_F": 1.0e-8, "xB_tolerance": 1.0e-12}],
            cycles=[],
            contraction={"last_64": {"median_q_F": 1.1}},
            roots=[first, second],
            brackets=[],
        )
        self.assertEqual(result["observed_admissible_root_clusters"], 1)
        self.assertIsNone(result["number_of_admissible_roots"])
        self.assertNotEqual(result["step245_classification"], "STEP245_MULTIPLE_ADMISSIBLE_ROOTS")

    def test_unresolved_scan_never_claims_no_root(self) -> None:
        """A nonmonotone/same-sign scan is diagnostic evidence, not a no-root proof."""

        result = _classify(
            raw_rows=[{"abs_F": 1.0e-8, "xB_tolerance": 1.0e-12}],
            cycles=[],
            contraction={"last_64": {"median_q_F": 1.1}},
            roots=[],
            brackets=[],
            scan_audit={"scan_based_root_count_certified": True},
        )
        self.assertEqual(result["step245_classification"], "OTHER_WITH_EXPLICIT_EVIDENCE")
        self.assertIsNone(result["number_of_admissible_roots"])
        self.assertIsNone(result["scalar_root_exists"])
        self.assertFalse(result["observed_scalar_root_exists"])

    def test_two_point_only_observations_do_not_claim_multiple_roots(self) -> None:
        """Tolerance-valid points need disjoint same-branch brackets for a multiple-root claim."""

        first = self._valid_root(
            0.01,
            bracket_kind="POINT_CANDIDATE",
            root_interval_left_x=0.01,
            root_interval_right_x=0.01,
        )
        second = self._valid_root(
            0.02,
            bracket_kind="POINT_CANDIDATE",
            root_interval_left_x=0.02,
            root_interval_right_x=0.02,
        )
        result = _classify(
            raw_rows=[{"abs_F": 1.0e-8, "xB_tolerance": 1.0e-12}],
            cycles=[],
            contraction={"last_64": {"median_q_F": 1.1}},
            roots=[first, second],
            brackets=[],
        )
        self.assertEqual(result["observed_admissible_root_clusters"], 2)
        self.assertEqual(result["observed_authoritative_sign_bracket_root_clusters"], 0)
        self.assertEqual(result["step245_classification"], "OTHER_WITH_EXPLICIT_EVIDENCE")

    def test_two_disjoint_same_branch_sign_roots_prove_multiple_observations(self) -> None:
        """Two verified, disjoint authoritative sign brackets establish multiplicity."""

        first = self._valid_root(
            0.01,
            root_interval_left_x=0.00999999999,
            root_interval_right_x=0.01000000001,
        )
        second = self._valid_root(
            0.02,
            root_interval_left_x=0.01999999999,
            root_interval_right_x=0.02000000001,
        )
        result = _classify(
            raw_rows=[{"abs_F": 1.0e-8, "xB_tolerance": 1.0e-12}],
            cycles=[],
            contraction={"last_64": {"median_q_F": 1.1}},
            roots=[first, second],
            brackets=[],
        )
        self.assertEqual(result["step245_classification"], "STEP245_MULTIPLE_ADMISSIBLE_ROOTS")
        self.assertEqual(result["observed_admissible_root_count_lower_bound"], 2)
        self.assertIsNone(result["number_of_admissible_roots"])

    def test_contiguous_signature_intervals_share_one_target_region(self) -> None:
        """Nested coarse/refined observations must not create a false second transition."""

        groups = _group_contiguous_interval_records(
            [
                {"left_x": 0.0, "right_x": 1.0},
                {"left_x": 0.25, "right_x": 0.5},
                {"left_x": 2.0, "right_x": 3.0},
            ]
        )
        self.assertEqual(len(groups), 2)
        self.assertEqual(len(groups[0]["records"]), 2)
        self.assertEqual((groups[0]["left_x"], groups[0]["right_x"]), (0.0, 1.0))

    def test_signature_only_jump_is_not_mislabelled_no_root(self) -> None:
        """A confirmed remap jump has its own fail-closed classification."""

        result = _classify(
            raw_rows=[{"abs_F": 1.0e-8, "xB_tolerance": 1.0e-12}],
            cycles=[],
            contraction={"last_64": {"median_q_F": 1.1}},
            roots=[],
            brackets=[{"bracket_kind": "SIGNATURE_TRANSITION", "status": "DISCONTINUITY_CONFIRMED"}],
            scan_audit={"scan_based_root_count_certified": False},
        )
        self.assertEqual(result["step245_classification"], "STEP245_DISCONTINUOUS_REMAP_ROOT_UNRESOLVED")

    def test_stagnating_contraction_cannot_authorize_picard_extension(self) -> None:
        """A q close to one must expose its impractical predicted iteration count."""

        result = _classify(
            raw_rows=[{"abs_F": 1.0e-8, "xB_tolerance": 1.0e-12}],
            cycles=[],
            contraction={
                "last_64": {
                    "median_q_F": 0.999999,
                    "monotone_residual_nonincreasing": True,
                    "predicted_remaining_iterations_to_xB_tolerance": 9_000_000,
                    "extended_picard_within_reasonable_budget": False,
                }
            },
            roots=[self._valid_root()],
            brackets=[],
        )
        self.assertEqual(result["picard_regime"], "STAGNATING_CONTRACTION")
        self.assertFalse(result["extended_picard_reasonable"])
        self.assertEqual(result["step245_classification"], "OTHER_WITH_EXPLICIT_EVIDENCE")


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
