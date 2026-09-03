#!/usr/bin/env python3
"""Diagnose the uncommitted CR1 scalar closure at the formal step-245 gate.

This program is deliberately not a new solver.  It replays the frozen
canonical CR1 trajectory through its last accepted step and then repeatedly
evaluates the existing ``T(x_next) - x_next`` map from that immutable state.
No trial is committed, no checkpoint is advanced, and no cumulative boundary
ledger is updated during the diagnosis.

The output is an evidence package for deciding whether a minimal closure
repair is justified.  In particular, this script does not loosen tolerances,
change physical inputs, clamp cells, alter the remap, or turn a root-search
trial into an accepted macro step.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.characteristic_reference import (  # noqa: E402
    CharacteristicReferenceError,
    CharacteristicReferenceSolver,
)
from kwn_mvp.population_metrics import cell_moments_from_piecewise_constant_cells  # noqa: E402
from kwn_mvp.solver import RadiusGridOverflowError, SolverConfig  # noqa: E402
from scripts.frozen_canonical_smooth_population_v1 import (  # noqa: E402
    build_frozen_canonical_context,
    frozen_canonical_snapshot_provenance,
)


TASK_NAME = "kwn_characteristic_closure_v2"
BASE_COMMIT = "2ee679437421c750ccf2f2173d208df73bf28475"
FROZEN_CONTRACT_HASH = "d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333"
FORMAL_FAILURE_JOB = "90321"
FORMAL_FAILURE_REPORT = (
    "/data/home/luozhiheng/tmp/"
    "kwn_characteristic_reference_v1_2ee679437421_20260902T202810Z/"
    "formal_all_14d_v4/reports/kwn_characteristic_reference_v1/"
    "15_final_acceptance_report.md"
)
FORMAL_FAILURE_TRACE = (
    "/data/home/luozhiheng/tmp/"
    "kwn_characteristic_reference_v1_2ee679437421_20260902T202810Z/"
    "formal_all_14d_v4/outputs/kwn_characteristic_reference_v1/"
    "characteristic_fixed_point_trace.csv"
)
DT_S = 0.015625
LAST_ACCEPTED_STEP = 244
FAILED_STEP = 245
MAX_RAW_PICARD_ITERATIONS = 1024
COARSE_SCAN_POINTS = 257
ROOT_MAX_ITERATIONS = 48
MACHINE_EPS_FACTOR = 64.0
MAX_REASONABLE_EXTENDED_PICARD_ITERATIONS = 1024
# A closure trial retains a full conservative-remap state.  The raw Picard
# trace already deliberately retains 1024 such trials.  The post-Picard audit
# therefore has a hard, recorded map-evaluation budget rather than blindly
# densifying every coarse interval or signature change.  Reaching this limit
# is evidence of incomplete coverage, never evidence of no/one root.
MAX_POST_RAW_MAP_EVALUATIONS = 1024
MAX_LOCAL_REFINEMENT_DEPTH = 3
MAX_TARGETED_LOCAL_REGIONS = 12
MAX_SIGN_CHANGE_ROOT_BRACKETS = 8
MAX_POINT_ROOT_CANDIDATES = 8
MAX_SIGNATURE_TRANSITION_REGIONS = 8
SIGNATURE_TRANSITION_MAX_ITERATIONS = 24
RAW_ATTRACTOR_TAIL_POINTS = 16
SIGNATURE_FACE_PREVIEW = 16
PLANNED_MAX_POST_RAW_MAP_EVALUATIONS = (
    COARSE_SCAN_POINTS
    + RAW_ATTRACTOR_TAIL_POINTS
    + MAX_TARGETED_LOCAL_REGIONS * ((1 << MAX_LOCAL_REFINEMENT_DEPTH) + 1)
    + MAX_SIGNATURE_TRANSITION_REGIONS * SIGNATURE_TRANSITION_MAX_ITERATIONS
    + MAX_SIGN_CHANGE_ROOT_BRACKETS * ROOT_MAX_ITERATIONS
    + MAX_SIGN_CHANGE_ROOT_BRACKETS
    + MAX_POINT_ROOT_CANDIDATES
)
PLANNED_MAX_TOTAL_RETAINED_MAP_EVALUATIONS = (
    MAX_RAW_PICARD_ITERATIONS + PLANNED_MAX_POST_RAW_MAP_EVALUATIONS
)


class DiagnosticError(RuntimeError):
    """Raised when the evidence package cannot retain its frozen contract."""


@dataclass(frozen=True)
class Evaluation:
    """One side-effect-free evaluation of the unchanged step-245 map."""

    evaluation_id: int
    phase: str
    x_guess: float
    trial: Any | None
    state_hash_before: str
    state_hash_after: str
    error_type: str | None
    error_message: str | None


@dataclass
class EvaluationBudget:
    """Bound post-Picard trial retention without making a coverage claim.

    ``ImmutableStep245Map`` retains a complete trial for provenance and
    immutability checks.  This object shares one deterministic budget between
    targeted scan, signature localization, and root verification.  Cached
    evaluations do not consume the budget; a requested uncached evaluation
    after exhaustion returns ``None`` and is explicitly reported downstream.
    """

    closure_map: "ImmutableStep245Map"
    maximum_new_evaluations: int
    start_evaluation_count: int
    exhausted: bool = False
    exhausted_phases: set[str] | None = None

    def __post_init__(self) -> None:
        if self.maximum_new_evaluations < 0:
            raise DiagnosticError("post-Picard evaluation budget must be non-negative")
        if self.exhausted_phases is None:
            self.exhausted_phases = set()

    @property
    def new_evaluation_count(self) -> int:
        return len(self.closure_map.evaluations) - self.start_evaluation_count

    @property
    def remaining(self) -> int:
        return max(0, self.maximum_new_evaluations - self.new_evaluation_count)

    def evaluate(self, value: float, *, phase: str, reuse: bool = True) -> Evaluation | None:
        key = float(value).hex()
        cached = key in self.closure_map._cache
        # A same-x reproducibility check deliberately bypasses the map cache:
        # it must retain a second, independently executed immutable trial.
        # It therefore consumes the shared diagnostic budget even though the
        # binary64 input value is already known.
        requires_new_evaluation = not cached or not reuse
        if requires_new_evaluation and self.new_evaluation_count >= self.maximum_new_evaluations:
            self.exhausted = True
            assert self.exhausted_phases is not None
            self.exhausted_phases.add(phase)
            return None
        return self.closure_map.evaluate(float(value), phase=phase, reuse=reuse)

    def audit(self) -> dict[str, Any]:
        return {
            "post_raw_maximum_new_map_evaluations": self.maximum_new_evaluations,
            "post_raw_new_map_evaluations": self.new_evaluation_count,
            "post_raw_remaining_map_evaluations": self.remaining,
            "planned_worst_case_post_raw_map_evaluations": PLANNED_MAX_POST_RAW_MAP_EVALUATIONS,
            "planned_worst_case_total_retained_map_evaluations": PLANNED_MAX_TOTAL_RETAINED_MAP_EVALUATIONS,
            "evaluation_budget_exhausted": self.exhausted,
            "evaluation_budget_exhausted_phases": sorted(self.exhausted_phases or ()),
            "retention_rationale": (
                "full conservative-remap trials are retained for read-only provenance; "
                "a bounded targeted audit prevents an unbounded dense scan from exhausting "
                "the declared CPU job resources"
            ),
        }


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(value)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, title: str, body: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fallback_fields: Sequence[str]) -> None:
    materialized = [dict(_json_safe(row)) for row in rows]
    fieldnames: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    if not fieldnames:
        fieldnames = list(fallback_fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _array_hash(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _float_bits(value: float) -> tuple[str, str]:
    array = np.asarray([float(value)], dtype=np.float64)
    bits = int(array.view(np.uint64)[0])
    return float(value).hex(), f"0x{bits:016x}"


def _git_value(*args: str) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(ROOT), *args], text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "NOT_AVAILABLE"


def _state_signature(solver: CharacteristicReferenceSolver) -> str:
    return _array_hash({key: np.asarray(value) for key, value in solver.state_arrays().items()})


def _moments(edges_m: np.ndarray, cell_number_m3: np.ndarray) -> dict[str, float]:
    cell = cell_moments_from_piecewise_constant_cells(edges_m, cell_number_m3)
    return {f"M{order}": float(np.sum(cell[order], dtype=np.float64)) for order in range(4)}


def _departure_source_indices(edges_m: np.ndarray, departure_faces_m: np.ndarray) -> np.ndarray:
    indices = np.searchsorted(edges_m, departure_faces_m, side="right") - 1
    return np.clip(indices, 0, edges_m.size - 2).astype(np.int64, copy=False)


def _signature_hash(indices: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(indices, dtype=np.int64).tobytes()).hexdigest()


def _trial_row(
    evaluation: Evaluation,
    *,
    edges_m: np.ndarray,
    previous_trial: Any | None,
) -> dict[str, Any]:
    """Convert one evaluation into the fixed scalar-map audit schema."""

    row: dict[str, Any] = {
        "evaluation_id": evaluation.evaluation_id,
        "phase": evaluation.phase,
        "x_guess": evaluation.x_guess,
        "state_hash_before": evaluation.state_hash_before,
        "state_hash_after": evaluation.state_hash_after,
        "state_unchanged": evaluation.state_hash_before == evaluation.state_hash_after,
        "error_type": evaluation.error_type or "",
        "error_message": evaluation.error_message or "",
    }
    x_hex, x_bits = _float_bits(evaluation.x_guess)
    row["x_guess_hex"] = x_hex
    row["x_guess_bits"] = x_bits
    trial = evaluation.trial
    if trial is None:
        return row

    closure_hex, closure_bits = _float_bits(trial.matrix_xb)
    residual_hex, residual_bits = _float_bits(trial.signed_xb_residual)
    candidate = np.asarray(trial.cell_number_m3, dtype=np.float64)
    indices = _departure_source_indices(edges_m, trial.trace.departure_faces_m)
    previous_candidate = None if previous_trial is None else np.asarray(previous_trial.cell_number_m3, dtype=np.float64)
    moments = _moments(edges_m, candidate)
    if previous_candidate is None:
        l1_relative = math.nan
        linf = math.nan
        population_residual = math.inf
        moment_residuals = {f"M{order}_relative_to_previous": math.inf for order in range(4)}
        changed = 0
        first_changed = -1
        largest_changed = -1
        lower_tail_l1 = math.nan
        changed_departure_faces = 0
    else:
        difference = candidate - previous_candidate
        absolute = np.abs(difference)
        denominator = max(
            float(np.sum(np.abs(candidate), dtype=np.float64)),
            float(np.sum(np.abs(previous_candidate), dtype=np.float64)),
            1.0e-300,
        )
        l1_relative = float(np.sum(absolute, dtype=np.float64) / denominator)
        linf = float(np.max(absolute))
        changed_mask = candidate != previous_candidate
        changed = int(np.count_nonzero(changed_mask))
        first_changed = int(np.flatnonzero(changed_mask)[0]) if changed else -1
        largest_changed = int(np.argmax(absolute)) if changed else -1
        lower_tail_l1 = float(np.sum(absolute[: min(64, absolute.size)], dtype=np.float64))
        previous_indices = _departure_source_indices(edges_m, previous_trial.trace.departure_faces_m)
        changed_departure_faces = int(np.count_nonzero(indices != previous_indices))
        previous_moments = _moments(edges_m, previous_candidate)
        moment_residuals = {
            f"M{order}_relative_to_previous": abs(moments[f"M{order}"] - previous_moments[f"M{order}"])
            / max(abs(moments[f"M{order}"]), abs(previous_moments[f"M{order}"]), 1.0e-300)
            for order in range(4)
        }
        population_residual = max(moment_residuals.values())
    row.update(
        {
            "x_closure": float(trial.matrix_xb),
            "x_closure_hex": closure_hex,
            "x_closure_bits": closure_bits,
            "F_signed": float(trial.signed_xb_residual),
            "F_hex": residual_hex,
            "F_bits": residual_bits,
            "abs_F": abs(float(trial.signed_xb_residual)),
            "xB_tolerance": float(trial.xb_tolerance),
            "midpoint_xB": float(trial.midpoint_matrix_xb),
            "Q_total_mol_m3": float(trial.inventory.total_mol_m3),
            "Q_beta_mol_m3": float(trial.inventory.beta_resolved_mol_m3),
            "Q_matrix_mol_m3": float(trial.inventory.matrix_mol_m3),
            "Q_GP_mol_m3": float(trial.inventory.gp_mol_m3),
            "inventory_relative_residual": float(trial.inventory.relative_residual),
            "inventory_residual_mol_m3": float(trial.inventory.residual_mol_m3),
            "matrix_fraction": float(trial.inventory.matrix_fraction),
            "M0": moments["M0"],
            "M1": moments["M1"],
            "M2": moments["M2"],
            "M3": moments["M3"],
            "population_observable_residual_to_previous": population_residual,
            "cell_measure_relative_residual_to_previous": l1_relative,
            **moment_residuals,
            "cell_state_hash": _array_hash({"cell_number_m3": candidate}),
            "remap_state_hash": _array_hash(
                {
                    "cell_number_m3": candidate,
                    "departure_faces_m": np.asarray(trial.trace.departure_faces_m, dtype=np.float64),
                }
            ),
            "departure_signature_hash": _signature_hash(indices),
            "minimum_departure_radius_m": float(np.min(trial.trace.departure_faces_m)),
            "maximum_departure_radius_m": float(np.max(trial.trace.departure_faces_m)),
            "lower_no_inflow_face_count": int(trial.trace.lower_no_inflow_face_count),
            "upper_no_inflow_face_count": int(trial.trace.upper_no_inflow_face_count),
            "rmin_number_loss_m3": float(trial.remap.lower_number_loss_m3),
            "rmax_number_loss_m3": float(trial.remap.upper_number_loss_m3),
            "remap_number_conservation_residual_m3": float(trial.remap.conservation_residual_m3),
            "cell_L1_relative_to_previous": l1_relative,
            "cell_Linf_to_previous": linf,
            "changed_cell_count": changed,
            "first_changed_cell_index": first_changed,
            "largest_change_cell_index": largest_changed,
            "lower_tail_first64_cell_L1": lower_tail_l1,
            "departure_faces_changing_source_cell": changed_departure_faces,
        }
    )
    return row


class ImmutableStep245Map:
    """Evaluate the existing CR1 closure map from one immutable accepted state."""

    def __init__(
        self,
        solver: CharacteristicReferenceSolver,
        *,
        old_cell_number_m3: np.ndarray,
        dt_s: float,
    ) -> None:
        self.solver = solver
        self.old_cell_number_m3 = np.asarray(old_cell_number_m3, dtype=np.float64).copy()
        self.dt_s = float(dt_s)
        self.x_start = float(solver.matrix_xb)
        self.initial_state_hash = _state_signature(solver)
        self._next_id = 1
        self.evaluations: list[Evaluation] = []
        self._cache: dict[str, Evaluation] = {}

    def evaluate(self, x_guess: float, *, phase: str, reuse: bool = False) -> Evaluation:
        value = float(x_guess)
        if not math.isfinite(value):
            raise DiagnosticError("closure-map x_guess must be finite")
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
        before = _state_signature(self.solver)
        if before != self.initial_state_hash:
            raise DiagnosticError("state_244 changed before a diagnostic closure-map evaluation")
        trial = None
        error_type = error_message = None
        try:
            trial = self.solver._evaluate_closure_trial(
                old_cell_number_m3=self.old_cell_number_m3,
                dt_s=self.dt_s,
                x_start=self.x_start,
                x_guess=value,
            )
        except (CharacteristicReferenceError, RadiusGridOverflowError, ValueError, FloatingPointError) as error:
            error_type = type(error).__name__
            error_message = str(error)
        after = _state_signature(self.solver)
        if after != before or after != self.initial_state_hash:
            raise DiagnosticError("closure-map evaluation mutated state_244")
        result = Evaluation(
            evaluation_id=self._next_id,
            phase=phase,
            x_guess=value,
            trial=trial,
            state_hash_before=before,
            state_hash_after=after,
            error_type=error_type,
            error_message=error_message,
        )
        self._next_id += 1
        self.evaluations.append(result)
        self._cache[key] = result
        return result


def _replay_to_step244(*, dt_s: float, accepted_step: int) -> tuple[CharacteristicReferenceSolver, Any, list[dict[str, Any]]]:
    """Replay the frozen CR1 trajectory, stopping exactly before step 245."""

    context = build_frozen_canonical_context()
    if context.contract_hash != FROZEN_CONTRACT_HASH:
        raise DiagnosticError("frozen canonical context has an unexpected validation-contract hash")
    solver = CharacteristicReferenceSolver(SolverConfig.from_mapping(context.mapping))
    trace: list[dict[str, Any]] = []
    for expected_step in range(1, accepted_step + 1):
        diagnostic = solver.advance_one(maximum_dt_s=dt_s)
        if diagnostic.step != expected_step:
            raise DiagnosticError(
                f"replay returned step={diagnostic.step}, expected accepted step={expected_step}"
            )
        trace.append(
            {
                "step": int(diagnostic.step),
                "time_s": float(diagnostic.time_s),
                "dt_s": float(diagnostic.dt_s),
                "matrix_xB": float(diagnostic.matrix_xb),
                "fixed_point_iterations": int(diagnostic.fixed_point_iterations),
                "fixed_point_picard_iterations": int(diagnostic.fixed_point_picard_iterations),
                "fixed_point_xb_residual": float(diagnostic.fixed_point_xb_residual),
                "fixed_point_population_residual": float(diagnostic.fixed_point_population_residual),
                "fixed_point_cell_measure_residual": float(diagnostic.fixed_point_cell_measure_residual),
                "fixed_point_convergence_rate": float(diagnostic.fixed_point_convergence_rate),
                "fixed_point_convergence_mode": diagnostic.fixed_point_convergence_mode,
                "inventory_relative_residual": float(diagnostic.inventory.relative_residual),
                "rmin_number_loss_m3": float(diagnostic.rmin_number_loss_m3),
                "rmin_mol_b_loss_mol_m3": float(diagnostic.rmin_mol_b_loss_mol_m3),
                "remap_number_conservation_residual_m3": float(
                    diagnostic.remap_number_conservation_residual_m3
                ),
            }
        )
    return solver, context, trace


def _compare_formal_step244_trace(
    replay_trace: Sequence[Mapping[str, Any]], *, formal_trace_csv: Path | None
) -> dict[str, Any]:
    """Compare the available formal step-244 trace evidence exactly.

    The v1 formal run did not persist a step-244 state archive.  Its accepted
    diagnostic row is nevertheless immutable evidence.  This comparison is
    intentionally exact binary64 equality for values emitted by the same
    validated cluster interpreter; no numerical tolerance is used to turn a
    different replay into a pass.
    """

    observed = replay_trace[-1] if replay_trace else {}
    result: dict[str, Any] = {
        "formal_state_244_archive": "NOT_AVAILABLE_IN_V1_RUN_ROOT",
        "formal_trace_csv": str(formal_trace_csv) if formal_trace_csv is not None else "NOT_REQUESTED",
        "observed_step": observed.get("step"),
        "observed_time_s": observed.get("time_s"),
    }
    if not observed or int(observed.get("step", -1)) != LAST_ACCEPTED_STEP:
        result.update({"status": "FAIL", "reason": "local replay did not reach formal accepted step 244"})
        return result
    if formal_trace_csv is None:
        result.update({"status": "PASS_NO_FORMAL_TRACE_REQUESTED", "reason": "formal state archive is unavailable"})
        return result
    if not formal_trace_csv.is_file():
        result.update({"status": "FAIL", "reason": "requested formal trace CSV is unreadable"})
        return result
    with formal_trace_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    candidates = [
        row
        for row in rows
        if row.get("policy") == "CR1_dt_0.015625s"
        and row.get("step") == str(LAST_ACCEPTED_STEP)
        and row.get("time_s") == str(LAST_ACCEPTED_STEP * DT_S)
    ]
    if len(candidates) != 1:
        result.update({"status": "FAIL", "reason": f"expected one formal CR1 dt=0.015625 step-244 row, found {len(candidates)}"})
        return result
    formal = candidates[0]
    fields = (
        "step",
        "time_s",
        "dt_s",
        "fixed_point_iterations",
        "fixed_point_picard_iterations",
        "fixed_point_xb_residual",
        "fixed_point_population_residual",
        "fixed_point_cell_measure_residual",
        "fixed_point_convergence_rate",
        "fixed_point_convergence_mode",
        "inventory_relative_residual",
        "rmin_number_loss_m3",
        "rmin_mol_b_loss_mol_m3",
        "remap_number_conservation_residual_m3",
    )
    mismatches: dict[str, dict[str, Any]] = {}
    for field in fields:
        expected = formal[field]
        actual = observed.get(field)
        if field == "fixed_point_convergence_mode":
            equal = str(actual) == expected
        elif field in {"step", "fixed_point_iterations", "fixed_point_picard_iterations"}:
            equal = int(actual) == int(expected)
        else:
            equal = float(actual) == float(expected)
        if not equal:
            mismatches[field] = {"formal": expected, "replay": actual}
    result.update(
        {
            "formal_row": formal,
            "compared_fields": list(fields),
            "mismatches": mismatches,
            "status": "PASS_EXACT_FORMAL_TRACE_MATCH" if not mismatches else "FAIL",
            "reason": "" if not mismatches else "step-244 replay differs from formal accepted trace",
        }
    )
    return result


def _save_state244(
    path: Path,
    *,
    solver: CharacteristicReferenceSolver,
    context: Any,
    replay_trace: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Write the complete accepted state plus the requested closure provenance."""

    beta_cells = solver._beta_cell_numbers()
    inventory = solver.ledger.snapshot(
        matrix_xb=solver.matrix_xb,
        populations=solver.population_list(),
        beta_resolved_fraction=1.0,
    )
    moments = _moments(context.edges_m, beta_cells)
    arrays = {key: np.asarray(value) for key, value in solver.state_arrays().items()}
    arrays.update(
        {
            "cell_edges_m": np.asarray(context.edges_m, dtype=np.float64),
            "cell_number_m3": np.asarray(beta_cells, dtype=np.float64),
            "xB_matrix": np.asarray([solver.matrix_xb], dtype=np.float64),
            "Q_total_mol_m3": np.asarray([inventory.total_mol_m3], dtype=np.float64),
            "Q_beta_mol_m3": np.asarray([inventory.beta_resolved_mol_m3], dtype=np.float64),
            "Q_matrix_mol_m3": np.asarray([inventory.matrix_mol_m3], dtype=np.float64),
            "M0": np.asarray([moments["M0"]], dtype=np.float64),
            "M1": np.asarray([moments["M1"]], dtype=np.float64),
            "M2": np.asarray([moments["M2"]], dtype=np.float64),
            "M3": np.asarray([moments["M3"]], dtype=np.float64),
            "previous_midpoint_matrix_xB": np.asarray(
                [solver.history[-1].midpoint_matrix_xb if solver.history else solver.matrix_xb], dtype=np.float64
            ),
            "state_metadata_json": np.asarray(
                json.dumps(
                    _json_safe(
                        {
                            "schema_version": "KWN_CHARACTERISTIC_CLOSURE_V2_STATE_244",
                            "step": int(solver.step),
                            "time_s": float(solver.time_s),
                            "validation_contract_hash": context.contract_hash,
                            "canonical_state_hash": context.canonical_hash,
                            "fixture_hash": context.fixture_hash,
                            "replay_last_row": dict(replay_trace[-1]) if replay_trace else {},
                        }
                    ),
                    sort_keys=True,
                )
            ),
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return {
        "state_244_file": str(path),
        "state_244_sha256": _sha256_file(path),
        "state_244_array_hash": _state_signature(solver),
        "step": int(solver.step),
        "time_s": float(solver.time_s),
        "matrix_xB": float(solver.matrix_xb),
        "Q_total_mol_m3": float(inventory.total_mol_m3),
        "Q_beta_mol_m3": float(inventory.beta_resolved_mol_m3),
        "Q_matrix_mol_m3": float(inventory.matrix_mol_m3),
        **moments,
    }


def _physical_x_interval(solver: CharacteristicReferenceSolver) -> dict[str, float]:
    """Derive a non-arbitrary scalar scan interval from frozen inventory bounds.

    With GP disabled and a fixed beta composition, the total B inventory bounds
    beta volume fraction.  That in turn bounds the smallest possible matrix
    fraction and therefore the maximum physically possible matrix xB.  The
    result is intersected with the frozen solver/thermodynamic [0, 1] domain.
    """

    beta = solver.population("beta")
    total = float(solver.ledger.total_b_mol_m3)
    beta_max_fraction = min(
        1.0,
        total * float(beta.parameters.molar_volume_m3_mol) / float(beta.parameters.x_b),
    )
    matrix_fraction_min = max(1.0 - beta_max_fraction, np.finfo(np.float64).tiny)
    inventory_upper = total * float(solver.ledger.matrix_molar_volume_m3_mol) / matrix_fraction_min
    x_min = 0.0
    x_max = min(1.0, inventory_upper)
    if not x_max > x_min:
        raise DiagnosticError("frozen inventory bounds produced an empty physical xB interval")
    return {
        "x_min": x_min,
        "x_max": x_max,
        "total_b_mol_m3": total,
        "beta_xB": float(beta.parameters.x_b),
        "beta_molar_volume_m3_mol": float(beta.parameters.molar_volume_m3_mol),
        "matrix_molar_volume_m3_mol": float(solver.ledger.matrix_molar_volume_m3_mol),
        "beta_volume_fraction_upper_bound": beta_max_fraction,
        "matrix_fraction_lower_bound": matrix_fraction_min,
        "inventory_xB_upper_bound": inventory_upper,
        "frozen_thermodynamic_xB_min": 0.0,
        "frozen_thermodynamic_xB_max": 1.0,
    }


def _raw_picard(
    closure_map: ImmutableStep245Map,
    *,
    edges_m: np.ndarray,
    iterations: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray], list[np.ndarray]]:
    """Reproduce raw Picard exactly, without invoking period-root acceptance."""

    x_guess = closure_map.x_start
    previous_trial = None
    previous_x_guess: float | None = None
    rows: list[dict[str, Any]] = []
    cell_states: list[np.ndarray] = []
    departure_faces: list[np.ndarray] = []
    for iteration in range(1, iterations + 1):
        evaluation = closure_map.evaluate(x_guess, phase="raw_picard")
        row = _trial_row(evaluation, edges_m=edges_m, previous_trial=previous_trial)
        row["iteration"] = iteration
        row["delta_x"] = math.nan if previous_x_guess is None else x_guess - previous_x_guess
        if evaluation.trial is None:
            row["raw_picard_terminal"] = True
            rows.append(row)
            break
        row["raw_picard_terminal"] = False
        row["population_convergence_tolerance"] = float(closure_map.solver._population_convergence_rtol)
        row["unchanged_from_state244"] = bool(
            np.array_equal(evaluation.trial.cell_number_m3, closure_map.old_cell_number_m3)
            and evaluation.trial.matrix_xb == closure_map.x_start
        )
        rows.append(row)
        cell_states.append(np.asarray(evaluation.trial.cell_number_m3, dtype=np.float64).copy())
        departure_faces.append(np.asarray(evaluation.trial.trace.departure_faces_m, dtype=np.float64).copy())
        previous_trial = evaluation.trial
        previous_x_guess = x_guess
        x_guess = float(evaluation.trial.matrix_xb)
    for index, row in enumerate(rows):
        if index == 0 or not math.isfinite(float(row.get("abs_F", math.nan))):
            row["q_F"] = math.nan
        else:
            prior = float(rows[index - 1].get("abs_F", math.nan))
            current = float(row.get("abs_F", math.nan))
            row["q_F"] = current / prior if prior > 0.0 and math.isfinite(current) else math.nan
        if index < 2:
            row["q_delta"] = math.nan
        else:
            prior_delta = abs(float(rows[index - 1].get("delta_x", math.nan)))
            current_delta = abs(float(row.get("delta_x", math.nan)))
            row["q_delta"] = current_delta / prior_delta if prior_delta > 0.0 and math.isfinite(current_delta) else math.nan
    return rows, cell_states, departure_faces


def _cycle_analysis(
    rows: Sequence[Mapping[str, Any]], cell_states: Sequence[np.ndarray], departure_faces: Sequence[np.ndarray]) -> list[dict[str, Any]]:
    """Audit exact, machine-scale, and drifting periods one through sixteen."""

    results: list[dict[str, Any]] = []
    if len(cell_states) != len(rows) or len(departure_faces) != len(rows):
        # A failed raw evaluation can only occur at the terminal point.  The
        # cycle report remains meaningful for all successful prefix entries.
        successful_rows = [row for row in rows if row.get("error_type", "") == ""]
    else:
        successful_rows = list(rows)
    count = min(len(successful_rows), len(cell_states), len(departure_faces))
    for k in range(count):
        comparisons: list[dict[str, Any]] = []
        for period in range(1, 17):
            if k < period:
                continue
            current = successful_rows[k]
            prior = successful_rows[k - period]
            current_state = cell_states[k]
            prior_state = cell_states[k - period]
            current_departure = departure_faces[k]
            prior_departure = departure_faces[k - period]
            scalar_distance = abs(float(current["x_guess"]) - float(prior["x_guess"]))
            denominator = max(
                float(np.sum(np.abs(current_state), dtype=np.float64)),
                float(np.sum(np.abs(prior_state), dtype=np.float64)),
                1.0e-300,
            )
            state_l1 = float(np.sum(np.abs(current_state - prior_state), dtype=np.float64) / denominator)
            state_linf = float(np.max(np.abs(current_state - prior_state)))
            scalar_scale = max(abs(float(current["x_guess"])), abs(float(prior["x_guess"])), 1.0)
            state_scale = max(float(np.max(np.abs(current_state))), float(np.max(np.abs(prior_state))), 1.0e-300)
            machine_scalar = MACHINE_EPS_FACTOR * np.finfo(np.float64).eps * scalar_scale
            machine_state = MACHINE_EPS_FACTOR * np.finfo(np.float64).eps * state_scale
            exact = (
                float(current["x_guess"]) == float(prior["x_guess"])
                and float(current.get("x_closure", math.nan)) == float(prior.get("x_closure", math.nan))
                and np.array_equal(current_state, prior_state)
                and np.array_equal(current_departure, prior_departure)
            )
            signature_equal = str(current.get("departure_signature_hash")) == str(prior.get("departure_signature_hash"))
            machine = scalar_distance <= machine_scalar and state_linf <= machine_state and signature_equal
            comparisons.append(
                {
                    "iteration": k + 1,
                    "period": period,
                    "d_x": scalar_distance,
                    "d_state_L1_relative": state_l1,
                    "d_state_Linf": state_linf,
                    "d_signature_equal": signature_equal,
                    "exact_bitwise_period": exact,
                    "machine_precision_period": machine,
                    "slowly_drifting_near_cycle": (
                        not exact and signature_equal and state_l1 <= 1.0e-9 and scalar_distance <= 1.0e-11
                    ),
                    "machine_scalar_tolerance": machine_scalar,
                    "machine_state_tolerance": machine_state,
                }
            )
        exact_periods = [int(row["period"]) for row in comparisons if row["exact_bitwise_period"]]
        machine_periods = [int(row["period"]) for row in comparisons if row["machine_precision_period"]]
        primitive_exact = min(exact_periods) if exact_periods else None
        primitive_machine = min(machine_periods) if machine_periods else None
        for row in comparisons:
            period = int(row["period"])
            phase_rows = successful_rows[k - period + 1 : k + 1]
            phase_values = [float(item["x_guess"]) for item in phase_rows]
            phase_bits = {value.hex() for value in phase_values}
            phase_residuals = [float(item.get("F_signed", math.nan)) for item in phase_rows]
            cyclic_pairs = zip(phase_residuals, phase_residuals[1:] + phase_residuals[:1])
            row["primitive_exact_period"] = bool(
                row["exact_bitwise_period"] and period == primitive_exact
            )
            row["primitive_machine_precision_period"] = bool(
                row["machine_precision_period"] and period == primitive_machine
            )
            row["phase_values_distinct"] = len(phase_bits) == period
            row["cyclic_adjacent_residual_sign_change"] = any(
                math.isfinite(left)
                and math.isfinite(right)
                and left * right < 0.0
                for left, right in cyclic_pairs
            )
            results.append(row)
    return results


def _window_contraction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for width in (16, 32, 64, 128):
        sample = [float(row["q_F"]) for row in rows[-width:] if math.isfinite(float(row.get("q_F", math.nan)))]
        delta = [float(row["q_delta"]) for row in rows[-width:] if math.isfinite(float(row.get("q_delta", math.nan)))]
        residuals = [
            float(row["abs_F"])
            for row in rows[-width:]
            if math.isfinite(float(row.get("abs_F", math.nan)))
        ]
        final_residual = residuals[-1] if residuals else math.nan
        final_tolerance = float(rows[-1].get("xB_tolerance", math.nan)) if rows else math.nan
        median_q = float(np.median(sample)) if sample else math.nan
        monotone_nonincreasing = bool(
            len(residuals) >= 2
            and all(right <= left for left, right in zip(residuals, residuals[1:]))
        )
        predicted_remaining: int | None
        if math.isfinite(final_residual) and math.isfinite(final_tolerance) and final_residual <= final_tolerance:
            predicted_remaining = 0
        elif (
            math.isfinite(final_residual)
            and math.isfinite(final_tolerance)
            and final_residual > 0.0
            and final_tolerance > 0.0
            and math.isfinite(median_q)
            and 0.0 < median_q < 1.0
        ):
            predicted_remaining = max(
                0,
                int(math.ceil(math.log(final_tolerance / final_residual) / math.log(median_q))),
            )
        else:
            predicted_remaining = None
        summary[f"last_{width}"] = {
            "count_q_F": len(sample),
            "median_q_F": median_q,
            "max_q_F": float(np.max(sample)) if sample else math.nan,
            "min_q_F": float(np.min(sample)) if sample else math.nan,
            "count_q_delta": len(delta),
            "median_q_delta": float(np.median(delta)) if delta else math.nan,
            "max_q_delta": float(np.max(delta)) if delta else math.nan,
            "monotone_residual_nonincreasing": monotone_nonincreasing,
            "final_abs_F": final_residual,
            "final_xB_tolerance": final_tolerance,
            "predicted_remaining_iterations_to_xB_tolerance": predicted_remaining,
            "predicted_total_iterations_to_xB_tolerance": (
                len(rows) + predicted_remaining if predicted_remaining is not None else None
            ),
            "extended_picard_within_reasonable_budget": bool(
                monotone_nonincreasing
                and predicted_remaining is not None
                and predicted_remaining <= MAX_REASONABLE_EXTENDED_PICARD_ITERATIONS
            ),
        }
    return summary


def _group_contiguous_interval_records(
    records: Sequence[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    """Group overlapping/touching sampled intervals without inventing continuity."""

    ordered = sorted(
        (dict(record) for record in records),
        key=lambda record: (float(record["left_x"]), float(record["right_x"])),
    )
    groups: list[dict[str, Any]] = []
    for record in ordered:
        left = float(record["left_x"])
        right = float(record["right_x"])
        if right < left:
            raise DiagnosticError("interval record has descending endpoints")
        if groups and left <= np.nextafter(float(groups[-1]["right_x"]), math.inf):
            group = groups[-1]
            group["records"].append(record)
            group["right_x"] = max(float(group["right_x"]), right)
        else:
            groups.append({"left_x": left, "right_x": right, "records": [record]})
    return groups


def _signature_transition_evidence(
    left: Evaluation,
    right: Evaluation,
    *,
    edges_m: np.ndarray,
) -> dict[str, Any]:
    """Locate a sampled remap-signature transition at face/cell granularity."""

    evidence: dict[str, Any] = {
        "remap_branch": "departure_source_cell_index",
        "changed_departure_face_count": 0,
        "first_changed_departure_face_index": -1,
        "last_changed_departure_face_index": -1,
        "changed_departure_faces_preview": [],
    }
    if left.trial is None or right.trial is None:
        evidence["transition_face_localization_status"] = "INVALID_ENDPOINT"
        return evidence
    left_indices = _departure_source_indices(edges_m, left.trial.trace.departure_faces_m)
    right_indices = _departure_source_indices(edges_m, right.trial.trace.departure_faces_m)
    if left_indices.shape != right_indices.shape:
        evidence["transition_face_localization_status"] = "SOURCE_INDEX_SHAPE_MISMATCH"
        return evidence
    changed = np.flatnonzero(left_indices != right_indices)
    evidence["changed_departure_face_count"] = int(changed.size)
    if not changed.size:
        evidence["transition_face_localization_status"] = "SIGNATURE_HASH_WITHOUT_SOURCE_INDEX_DIFFERENCE"
        return evidence
    evidence["first_changed_departure_face_index"] = int(changed[0])
    evidence["last_changed_departure_face_index"] = int(changed[-1])
    preview: list[dict[str, Any]] = []
    for face_index in changed[:SIGNATURE_FACE_PREVIEW]:
        index = int(face_index)
        left_cell = int(left_indices[index])
        right_cell = int(right_indices[index])
        preview.append(
            {
                "departure_face_index": index,
                "left_source_cell_index": left_cell,
                "right_source_cell_index": right_cell,
                "left_departure_radius_m": float(left.trial.trace.departure_faces_m[index]),
                "right_departure_radius_m": float(right.trial.trace.departure_faces_m[index]),
                "left_source_cell_left_edge_m": float(edges_m[left_cell]),
                "left_source_cell_right_edge_m": float(edges_m[left_cell + 1]),
                "right_source_cell_left_edge_m": float(edges_m[right_cell]),
                "right_source_cell_right_edge_m": float(edges_m[right_cell + 1]),
            }
        )
    evidence["changed_departure_faces_preview"] = preview
    evidence["transition_face_localization_status"] = "FACE_AND_SOURCE_CELL_LOCATED"
    return evidence


def _scalar_scan(
    closure_map: ImmutableStep245Map,
    *,
    edges_m: np.ndarray,
    interval: Mapping[str, float],
    raw_rows: Sequence[Mapping[str, Any]],
    cycles: Sequence[Mapping[str, Any]],
    coarse_points: int,
    refinement_depth: int,
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    dict[str, Any],
    EvaluationBudget,
]:
    """Run a bounded, targeted audit of the immutable step-245 scalar map.

    A finite sampled map can discover roots, multiple disjoint roots, branch
    switches, or a failure to evaluate.  It cannot prove the absence or
    uniqueness of roots without an interval/analytic enclosure.  The audit
    records that distinction explicitly and leaves those global claims open.
    """

    if coarse_points != COARSE_SCAN_POINTS:
        raise DiagnosticError(f"coarse scalar-map scan must use exactly {COARSE_SCAN_POINTS} points")
    if refinement_depth < 0:
        raise DiagnosticError("refinement depth must be non-negative")
    x_min = float(interval["x_min"])
    x_max = float(interval["x_max"])
    if not x_max > x_min:
        raise DiagnosticError("scalar-map scan interval must be nonempty")

    budget = EvaluationBudget(
        closure_map=closure_map,
        maximum_new_evaluations=MAX_POST_RAW_MAP_EVALUATIONS,
        start_evaluation_count=len(closure_map.evaluations),
    )
    scan_cache: dict[str, Evaluation] = {}
    refined_rows: list[dict[str, Any]] = []
    invalid_intervals: list[dict[str, Any]] = []
    cross_signature_sign_changes: list[dict[str, Any]] = []
    sign_candidates: dict[tuple[str, str], dict[str, Any]] = {}
    point_candidates: dict[str, dict[str, Any]] = {}
    signature_candidates: dict[tuple[str, str], dict[str, Any]] = {}

    def evaluate(value: float, phase: str) -> Evaluation | None:
        item = budget.evaluate(float(value), phase=phase)
        if item is not None:
            scan_cache.setdefault(float(value).hex(), item)
        return item

    def row_for(item: Evaluation) -> dict[str, Any]:
        return _trial_row(item, edges_m=edges_m, previous_trial=None)

    def add_point_candidate(item: Evaluation | None, *, reason: str, priority: int) -> None:
        if item is None or item.trial is None:
            return
        key = item.x_guess.hex()
        row = row_for(item)
        existing = point_candidates.get(key)
        if existing is not None:
            existing["candidate_reasons"].append(reason)
            existing["candidate_priority"] = min(int(existing["candidate_priority"]), priority)
            return
        point_candidates[key] = {
            "bracket_kind": "POINT_CANDIDATE",
            "reason": reason,
            "candidate_reasons": [reason],
            "candidate_priority": priority,
            "left_x": item.x_guess,
            "right_x": item.x_guess,
            "left_F": row.get("F_signed", math.nan),
            "right_F": row.get("F_signed", math.nan),
            "left_signature": row.get("departure_signature_hash", ""),
            "right_signature": row.get("departure_signature_hash", ""),
            "final_width": 0.0,
            "sign_change": False,
            "status": "POINT_RESIDUAL_CANDIDATE",
        }

    def add_sign_candidate(
        left: Evaluation | None,
        right: Evaluation | None,
        *,
        reason: str,
        priority: int,
    ) -> None:
        if left is None or right is None or left.trial is None or right.trial is None:
            return
        if right.x_guess < left.x_guess:
            left, right = right, left
        if not right.x_guess > left.x_guess:
            return
        key = (left.x_guess.hex(), right.x_guess.hex())
        existing = sign_candidates.get(key)
        if existing is not None:
            existing["candidate_reasons"].append(reason)
            existing["candidate_priority"] = min(int(existing["candidate_priority"]), priority)
            return
        left_row = row_for(left)
        right_row = row_for(right)
        sign_candidates[key] = {
            "bracket_kind": "SIGN_CHANGE",
            "reason": reason,
            "candidate_reasons": [reason],
            "candidate_priority": priority,
            "left_x": left.x_guess,
            "right_x": right.x_guess,
            "left_F": float(left.trial.signed_xb_residual),
            "right_F": float(right.trial.signed_xb_residual),
            "left_signature": left_row.get("departure_signature_hash", ""),
            "right_signature": right_row.get("departure_signature_hash", ""),
            "final_width": right.x_guess - left.x_guess,
            "sign_change": True,
            "status": "SIGN_CHANGE_CANDIDATE",
        }

    def add_signature_candidate(
        left: Evaluation | None,
        right: Evaluation | None,
        *,
        reason: str,
        priority: int,
    ) -> None:
        if left is None or right is None or left.trial is None or right.trial is None:
            return
        if right.x_guess < left.x_guess:
            left, right = right, left
        left_row = row_for(left)
        right_row = row_for(right)
        if left_row.get("departure_signature_hash") == right_row.get("departure_signature_hash"):
            return
        key = (left.x_guess.hex(), right.x_guess.hex())
        existing = signature_candidates.get(key)
        if existing is not None:
            existing["candidate_reasons"].append(reason)
            existing["candidate_priority"] = min(int(existing["candidate_priority"]), priority)
            return
        signature_candidates[key] = {
            "left_x": left.x_guess,
            "right_x": right.x_guess,
            "candidate_reasons": [reason],
            "candidate_priority": priority,
            "_left": left,
            "_right": right,
        }

    def inspect_sequence(
        evaluations: Sequence[Evaluation],
        *,
        reason_prefix: str,
        sign_priority: int,
        point_priority: int,
        signature_priority: int,
    ) -> None:
        for item in evaluations:
            if item.trial is not None and abs(float(item.trial.signed_xb_residual)) <= float(item.trial.xb_tolerance):
                add_point_candidate(
                    item,
                    reason=f"{reason_prefix}_RESIDUAL_TOLERANCE",
                    priority=point_priority,
                )
        for left, right in zip(evaluations, evaluations[1:]):
            if left.trial is None or right.trial is None:
                invalid_intervals.append(
                    {
                        "bracket_kind": "INVALID_INTERVAL",
                        "reason": f"{reason_prefix}_INVALID_TRIAL",
                        "left_x": left.x_guess,
                        "right_x": right.x_guess,
                        "status": "INVALID_ENDPOINT",
                        "sign_change": False,
                    }
                )
                continue
            left_f = float(left.trial.signed_xb_residual)
            right_f = float(right.trial.signed_xb_residual)
            left_signature = row_for(left).get("departure_signature_hash", "")
            right_signature = row_for(right).get("departure_signature_hash", "")
            sign_change = (left_f < 0.0 < right_f) or (right_f < 0.0 < left_f)
            if sign_change:
                if left_signature == right_signature:
                    add_sign_candidate(
                        left,
                        right,
                        reason=f"{reason_prefix}_F_SIGN_CHANGE",
                        priority=sign_priority,
                    )
                else:
                    cross_signature_sign_changes.append(
                        {
                            "bracket_kind": "SIGN_CHANGE_ACROSS_SIGNATURE_TRANSITION",
                            "reason": f"{reason_prefix}_F_SIGN_CHANGE",
                            "left_x": left.x_guess,
                            "right_x": right.x_guess,
                            "left_F": left_f,
                            "right_F": right_f,
                            "left_signature": left_signature,
                            "right_signature": right_signature,
                            "final_width": right.x_guess - left.x_guess,
                            "sign_change": True,
                            "root_search_selected": False,
                            "status": "SIGN_CHANGE_ACROSS_SIGNATURE_TRANSITION_UNRESOLVED",
                        }
                    )
            if left_signature != right_signature:
                add_signature_candidate(
                    left,
                    right,
                    reason=f"{reason_prefix}_DEPARTURE_SIGNATURE_CHANGE",
                    priority=min(signature_priority, 1) if sign_change else signature_priority,
                )

    coarse_values = np.linspace(x_min, x_max, coarse_points, dtype=np.float64)
    coarse_evaluations: list[Evaluation] = []
    for value in coarse_values:
        item = evaluate(float(value), "scalar_coarse_scan")
        if item is None:
            raise DiagnosticError("post-Picard budget cannot cover the required coarse scalar scan")
        coarse_evaluations.append(item)
    coarse_evaluations.sort(key=lambda item: item.x_guess)
    inspect_sequence(
        coarse_evaluations,
        reason_prefix="COARSE_SCAN",
        sign_priority=0,
        point_priority=1,
        signature_priority=2,
    )

    coarse_spacing = (x_max - x_min) / float(coarse_points - 1)
    successful_raw_values = [
        min(x_max, max(x_min, float(row["x_guess"])))
        for row in raw_rows
        if not row.get("error_type", "")
        and isinstance(row.get("x_guess"), (int, float))
        and math.isfinite(float(row["x_guess"]))
    ]
    tail_values = successful_raw_values[-RAW_ATTRACTOR_TAIL_POINTS:]
    target_specs: list[dict[str, Any]] = []

    def add_target_spec(values: Sequence[float], *, reason: str, priority: int) -> None:
        if not values:
            return
        left = max(x_min, min(values) - 0.5 * coarse_spacing)
        right = min(x_max, max(values) + 0.5 * coarse_spacing)
        if right < left:
            return
        key = (left.hex(), right.hex())
        for existing in target_specs:
            if existing["key"] == key:
                existing["reasons"].append(reason)
                existing["priority"] = min(int(existing["priority"]), priority)
                return
        target_specs.append(
            {
                "key": key,
                "left_x": left,
                "right_x": right,
                "reasons": [reason],
                "priority": priority,
            }
        )

    add_target_spec(tail_values, reason="RAW_PICARD_ATTRACTOR_SPAN", priority=0)
    final_cycle_iteration = max((int(row.get("iteration", 0)) for row in cycles), default=0)
    primitive_periods = sorted(
        {
            int(row["period"])
            for row in cycles
            if int(row.get("iteration", 0)) == final_cycle_iteration
            and bool(row.get("primitive_exact_period"))
        }
    )
    cycle_phase_values: list[float] = []
    if primitive_periods:
        period = primitive_periods[0]
        cycle_phase_values = successful_raw_values[-period:]
        add_target_spec(cycle_phase_values, reason="EXACT_CYCLE_PHASE_SPAN", priority=1)
    raw_or_cycle_probe_values = sorted({float(item) for item in tail_values + cycle_phase_values})
    raw_or_cycle_evaluations: list[Evaluation] = []
    for value in raw_or_cycle_probe_values:
        probe_phase = (
            "exact_cycle_phase_probe" if value in set(cycle_phase_values) else "raw_picard_attractor_probe"
        )
        item = evaluate(value, probe_phase)
        if item is not None:
            raw_or_cycle_evaluations.append(item)
    if raw_or_cycle_evaluations:
        raw_or_cycle_evaluations.sort(key=lambda item: item.x_guess)
        inspect_sequence(
            raw_or_cycle_evaluations,
            reason_prefix="RAW_TAIL_OR_EXACT_CYCLE",
            sign_priority=0,
            point_priority=0,
            signature_priority=0,
        )

    local_minima: list[tuple[float, float]] = []
    for left, middle, right in zip(
        coarse_evaluations,
        coarse_evaluations[1:],
        coarse_evaluations[2:],
    ):
        if left.trial is None or middle.trial is None or right.trial is None:
            continue
        if len(
            {
                row_for(left).get("departure_signature_hash", ""),
                row_for(middle).get("departure_signature_hash", ""),
                row_for(right).get("departure_signature_hash", ""),
            }
        ) != 1:
            continue
        middle_abs = abs(float(middle.trial.signed_xb_residual))
        if (
            middle_abs <= abs(float(left.trial.signed_xb_residual))
            and middle_abs <= abs(float(right.trial.signed_xb_residual))
        ):
            local_minima.append((middle_abs, middle.x_guess))
    for _residual, value in sorted(local_minima, key=lambda item: (item[0], item[1])):
        add_target_spec([value], reason="COARSE_LOCAL_ABS_F_MINIMUM", priority=2)
    target_specs.sort(key=lambda item: (int(item["priority"]), float(item["left_x"]), float(item["right_x"])))
    selected_target_specs = target_specs[:MAX_TARGETED_LOCAL_REGIONS]
    omitted_target_specs = target_specs[MAX_TARGETED_LOCAL_REGIONS:]
    subdivisions = 1 << min(int(refinement_depth), MAX_LOCAL_REFINEMENT_DEPTH)
    targeted_region_audit: list[dict[str, Any]] = []
    for spec in selected_target_specs:
        before_count = budget.new_evaluation_count
        values = np.linspace(
            float(spec["left_x"]),
            float(spec["right_x"]),
            subdivisions + 1,
            dtype=np.float64,
        )
        region_evaluations: list[Evaluation] = []
        status = "TARGETED_REFINEMENT_COMPLETE"
        for value in values:
            item = evaluate(float(value), "targeted_scalar_refinement")
            if item is None:
                status = "TARGETED_REFINEMENT_EVALUATION_BUDGET_EXHAUSTED"
                break
            region_evaluations.append(item)
            refined_rows.append(row_for(item))
        if region_evaluations:
            region_evaluations.sort(key=lambda item: item.x_guess)
            inspect_sequence(
                region_evaluations,
                reason_prefix="TARGETED_REFINEMENT",
                sign_priority=1,
                point_priority=1,
                signature_priority=1,
            )
        targeted_region_audit.append(
            {
                "left_x": float(spec["left_x"]),
                "right_x": float(spec["right_x"]),
                "reasons": list(spec["reasons"]),
                "status": status,
                "planned_lattice_points": int(subdivisions + 1),
                "evaluated_lattice_points": len(region_evaluations),
                "new_map_evaluations": budget.new_evaluation_count - before_count,
            }
        )
    for spec in omitted_target_specs:
        targeted_region_audit.append(
            {
                "left_x": float(spec["left_x"]),
                "right_x": float(spec["right_x"]),
                "reasons": list(spec["reasons"]),
                "status": "TARGETED_REFINEMENT_UNEXAMINED_REGION_BUDGET",
                "planned_lattice_points": int(subdivisions + 1),
                "evaluated_lattice_points": 0,
                "new_map_evaluations": 0,
            }
        )

    for raw in raw_rows:
        value = raw.get("x_guess")
        residual = raw.get("abs_F")
        tolerance = raw.get("xB_tolerance")
        if (
            isinstance(value, (float, int))
            and isinstance(residual, (float, int))
            and isinstance(tolerance, (float, int))
            and math.isfinite(float(value))
            and float(residual) <= float(tolerance)
        ):
            add_point_candidate(
                evaluate(float(value), "raw_picard_root_probe"),
                reason="RAW_PICARD_RESIDUAL_TOLERANCE",
                priority=0,
            )

    signature_groups = _group_contiguous_interval_records(list(signature_candidates.values()))
    signature_groups.sort(
        key=lambda group: (
            min(int(record["candidate_priority"]) for record in group["records"]),
            float(group["left_x"]),
            float(group["right_x"]),
        )
    )
    signature_outputs: list[dict[str, Any]] = []
    selected_signature_groups = signature_groups[:MAX_SIGNATURE_TRANSITION_REGIONS]
    omitted_signature_groups = signature_groups[MAX_SIGNATURE_TRANSITION_REGIONS:]
    for transition_id, group in enumerate(selected_signature_groups, start=1):
        def signature_pair(record: Mapping[str, Any]) -> tuple[str, str]:
            return (
                str(row_for(record["_left"]).get("departure_signature_hash", "")),
                str(row_for(record["_right"]).get("departure_signature_hash", "")),
            )

        signature_pairs_in_group = sorted({signature_pair(record) for record in group["records"]})
        representative = min(
            group["records"],
            key=lambda record: float(record["right_x"]) - float(record["left_x"]),
        )
        left = representative["_left"]
        right = representative["_right"]
        initial_left = left
        initial_right = right
        multiple_pairs = len(signature_pairs_in_group) > 1
        status = (
            "MULTIPLE_SOURCE_CELL_CDF_KINKS_OBSERVED"
            if multiple_pairs
            else "SOURCE_CELL_CDF_KINK_OBSERVED"
        )
        transition_iterations = 0
        if not multiple_pairs:
            for iteration in range(1, SIGNATURE_TRANSITION_MAX_ITERATIONS + 1):
                transition_iterations = iteration
                midpoint = 0.5 * (left.x_guess + right.x_guess)
                if midpoint == left.x_guess or midpoint == right.x_guess:
                    break
                middle = evaluate(midpoint, "signature_transition_refinement")
                if middle is None:
                    status = "SIGNATURE_TRANSITION_EVALUATION_BUDGET_EXHAUSTED"
                    break
                middle_row = row_for(middle)
                refined_rows.append(middle_row)
                if middle.trial is None:
                    status = "SIGNATURE_TRANSITION_INVALID_MIDPOINT"
                    break
                left_signature = row_for(left).get("departure_signature_hash", "")
                middle_signature = middle_row.get("departure_signature_hash", "")
                right_signature = row_for(right).get("departure_signature_hash", "")
                if left_signature != middle_signature and middle_signature != right_signature:
                    status = "MULTIPLE_SOURCE_CELL_CDF_KINKS_OBSERVED"
                    break
                if left_signature != middle_signature:
                    right = middle
                elif middle_signature != right_signature:
                    left = middle
                else:
                    status = "SOURCE_CELL_SIGNATURE_TRANSITION_NOT_RETAINED"
                    break
        if left.trial is not None and right.trial is not None:
            left_row = row_for(left)
            right_row = row_for(right)
            residual_jump = abs(float(left.trial.signed_xb_residual) - float(right.trial.signed_xb_residual))
            jump_tolerance = 8.0 * max(float(left.trial.xb_tolerance), float(right.trial.xb_tolerance))
            # A residual difference over a finite x interval measures a
            # slope as well as any possible jump.  It is retained below as
            # provenance, but cannot label a CDF source-cell crossing a
            # discontinuity.
            if (
                float(left.trial.signed_xb_residual) < 0.0 < float(right.trial.signed_xb_residual)
                or float(right.trial.signed_xb_residual) < 0.0 < float(left.trial.signed_xb_residual)
            ):
                cross_signature_sign_changes.append(
                    {
                        "bracket_kind": "SIGN_CHANGE_ACROSS_SIGNATURE_TRANSITION",
                        "reason": "SIGNATURE_LOCALIZATION_F_SIGN_CHANGE",
                        "left_x": left.x_guess,
                        "right_x": right.x_guess,
                        "left_F": float(left.trial.signed_xb_residual),
                        "right_F": float(right.trial.signed_xb_residual),
                        "left_signature": left_row.get("departure_signature_hash", ""),
                        "right_signature": right_row.get("departure_signature_hash", ""),
                        "final_width": right.x_guess - left.x_guess,
                        "sign_change": True,
                        "root_search_selected": False,
                        "status": "SIGN_CHANGE_ACROSS_SIGNATURE_TRANSITION_UNRESOLVED",
                    }
                )
            transition_evidence = _signature_transition_evidence(left, right, edges_m=edges_m)
        else:
            left_row = row_for(left)
            right_row = row_for(right)
            residual_jump = math.nan
            jump_tolerance = math.nan
            transition_evidence = _signature_transition_evidence(left, right, edges_m=edges_m)
        signature_outputs.append(
            {
                "bracket_kind": "SIGNATURE_TRANSITION",
                "reason": ";".join(
                    sorted({reason for record in group["records"] for reason in record["candidate_reasons"]})
                ),
                "transition_id": transition_id,
                "coarse_or_targeted_pair_count": len(group["records"]),
                "distinct_signature_pair_count": len(signature_pairs_in_group),
                "signature_pairs_observed": [list(pair) for pair in signature_pairs_in_group],
                "initial_left_x": initial_left.x_guess,
                "initial_right_x": initial_right.x_guess,
                "left_x": left.x_guess,
                "right_x": right.x_guess,
                "left_F": left_row.get("F_signed", math.nan),
                "right_F": right_row.get("F_signed", math.nan),
                "left_signature": left_row.get("departure_signature_hash", ""),
                "right_signature": right_row.get("departure_signature_hash", ""),
                "final_width": right.x_guess - left.x_guess,
                "residual_jump": residual_jump,
                "jump_tolerance": jump_tolerance,
                "transition_iterations": transition_iterations,
                "sign_change": (
                    left.trial is not None
                    and right.trial is not None
                    and (
                        float(left.trial.signed_xb_residual) < 0.0 < float(right.trial.signed_xb_residual)
                        or float(right.trial.signed_xb_residual) < 0.0 < float(left.trial.signed_xb_residual)
                    )
                ),
                "continuity_theory_status": (
                    "PIECEWISE_CONSTANT_CDF_CONTINUOUS_AT_SOURCE_CELL_EDGE;"
                    "TRACE_LIMIT_NOT_ESTABLISHED_BY_SIGNATURE_HASH"
                ),
                "status": status,
                **transition_evidence,
            }
        )
    for group in omitted_signature_groups:
        representative = min(
            group["records"],
            key=lambda record: float(record["right_x"]) - float(record["left_x"]),
        )
        left = representative["_left"]
        right = representative["_right"]
        signature_outputs.append(
            {
                "bracket_kind": "SIGNATURE_TRANSITION",
                "reason": ";".join(
                    sorted({reason for record in group["records"] for reason in record["candidate_reasons"]})
                ),
                "transition_id": None,
                "coarse_or_targeted_pair_count": len(group["records"]),
                "initial_left_x": left.x_guess,
                "initial_right_x": right.x_guess,
                "left_x": left.x_guess,
                "right_x": right.x_guess,
                "final_width": right.x_guess - left.x_guess,
                "transition_iterations": 0,
                "continuity_theory_status": (
                    "PIECEWISE_CONSTANT_CDF_CONTINUOUS_AT_SOURCE_CELL_EDGE;"
                    "TRACE_LIMIT_NOT_ESTABLISHED_BY_SIGNATURE_HASH"
                ),
                "status": "SOURCE_CELL_CDF_KINK_REGION_UNEXAMINED_BUDGET",
                **_signature_transition_evidence(left, right, edges_m=edges_m),
            }
        )

    def select_root_candidates(
        candidates: Sequence[dict[str, Any]], *, maximum: int, selected_status: str, omitted_status: str
    ) -> list[dict[str, Any]]:
        selected: list[dict[str, Any]] = []
        ordered = sorted(
            candidates,
            key=lambda record: (
                int(record["candidate_priority"]),
                float(record["left_x"]),
                float(record["right_x"]),
            ),
        )
        for index, candidate in enumerate(ordered):
            output = dict(candidate)
            output["reason"] = ";".join(sorted(set(output["candidate_reasons"])))
            output["root_search_selected"] = index < maximum
            output["status"] = selected_status if index < maximum else omitted_status
            del output["candidate_priority"]
            selected.append(output)
        return selected

    sign_outputs = select_root_candidates(
        list(sign_candidates.values()),
        maximum=MAX_SIGN_CHANGE_ROOT_BRACKETS,
        selected_status="SELECTED_FOR_ROOT_SEARCH",
        omitted_status="SIGN_CHANGE_UNEXAMINED_ROOT_BRACKET_BUDGET",
    )
    point_outputs = select_root_candidates(
        list(point_candidates.values()),
        maximum=MAX_POINT_ROOT_CANDIDATES,
        selected_status="SELECTED_FOR_ROOT_SEARCH",
        omitted_status="POINT_CANDIDATE_UNEXAMINED_ROOT_BRACKET_BUDGET",
    )
    root_brackets = (
        invalid_intervals
        + sign_outputs
        + point_outputs
        + cross_signature_sign_changes
        + signature_outputs
    )

    scalar_evaluations = sorted(scan_cache.values(), key=lambda item: item.x_guess)
    scalar_rows = [row_for(item) for item in scalar_evaluations]
    branches: list[list[dict[str, Any]]] = []
    current_branch: list[dict[str, Any]] = []
    invalid_scalar_trial = False
    for row in scalar_rows:
        if row.get("error_type", ""):
            invalid_scalar_trial = True
            if current_branch:
                branches.append(current_branch)
                current_branch = []
            continue
        if current_branch and row.get("departure_signature_hash") != current_branch[-1].get("departure_signature_hash"):
            branches.append(current_branch)
            current_branch = []
        current_branch.append(row)
    if current_branch:
        branches.append(current_branch)
    branch_monotonicity: list[dict[str, Any]] = []
    for branch in branches:
        residuals = [float(row["F_signed"]) for row in branch]
        nondecreasing = all(right >= left for left, right in zip(residuals, residuals[1:]))
        nonincreasing = all(right <= left for left, right in zip(residuals, residuals[1:]))
        branch_monotonicity.append(
            {
                "point_count": len(branch),
                "left_x": branch[0]["x_guess"],
                "right_x": branch[-1]["x_guess"],
                "nondecreasing": nondecreasing,
                "nonincreasing": nonincreasing,
                "monotonic_on_finite_sample_only": nondecreasing or nonincreasing,
            }
        )

    unexamined_sign_brackets = sum(not bool(row["root_search_selected"]) for row in sign_outputs)
    unexamined_point_candidates = sum(not bool(row["root_search_selected"]) for row in point_outputs)
    unexamined_target_regions = len(omitted_target_specs) + sum(
        item["status"] == "TARGETED_REFINEMENT_EVALUATION_BUDGET_EXHAUSTED"
        for item in targeted_region_audit
    )
    transition_statuses = [str(row["status"]) for row in signature_outputs]
    scan_audit = {
        "scan_strategy": "COARSE_257_PLUS_BOUNDED_TARGETED_REFINEMENT_V1",
        "coarse_scan_points": len(coarse_evaluations),
        "targeted_refinement_depth_requested": int(refinement_depth),
        "targeted_refinement_depth_applied": min(int(refinement_depth), MAX_LOCAL_REFINEMENT_DEPTH),
        "targeted_refinement_lattice_points_per_region": int(subdivisions + 1),
        "targeted_region_candidates": len(target_specs),
        "targeted_region_budget": MAX_TARGETED_LOCAL_REGIONS,
        "targeted_regions": targeted_region_audit,
        "unexamined_target_regions": int(unexamined_target_regions),
        "coarse_local_abs_f_minimum_candidates": len(local_minima),
        "raw_picard_attractor_tail_points": len(tail_values),
        "primitive_exact_cycle_periods": primitive_periods,
        "exact_cycle_phase_probe_count": len(cycle_phase_values),
        "invalid_scalar_trial": invalid_scalar_trial,
        "sign_change_bracket_candidates": len(sign_outputs),
        "sign_change_root_bracket_budget": MAX_SIGN_CHANGE_ROOT_BRACKETS,
        "unexamined_sign_change_brackets": int(unexamined_sign_brackets),
        "cross_signature_sign_change_count": len(cross_signature_sign_changes),
        "point_root_candidates": len(point_outputs),
        "point_root_candidate_budget": MAX_POINT_ROOT_CANDIDATES,
        "unexamined_point_root_candidates": int(unexamined_point_candidates),
        "signature_transition_region_candidates": len(signature_groups),
        "signature_transition_region_budget": MAX_SIGNATURE_TRANSITION_REGIONS,
        "unexamined_signature_transition_regions": len(omitted_signature_groups),
        "signature_transition_statuses": transition_statuses,
        "finite_scan_branch_monotonicity_observed": branch_monotonicity,
        "finite_audit_complete_under_declared_budget": bool(
            not budget.exhausted
            and not unexamined_target_regions
            and not unexamined_sign_brackets
            and not unexamined_point_candidates
            and not omitted_signature_groups
        ),
        "root_count_certified": False,
        "scan_based_root_count_certified": False,
        "root_count_certification_status": "UNRESOLVED_NO_INTERVAL_OR_ANALYTIC_ENCLOSURE",
        "root_count_certification_reason": (
            "finite sampled branch behavior can discover evidence but cannot prove no or unique "
            "physical scalar root; no interval/analytic enclosure is implemented in this diagnostic"
        ),
        "evaluation_budget": budget.audit(),
    }
    return scalar_rows, root_brackets, refined_rows, scan_audit, budget


def _solve_brackets(
    closure_map: ImmutableStep245Map,
    *,
    edges_m: np.ndarray,
    brackets: Sequence[Mapping[str, Any]],
    budget: EvaluationBudget,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Verify selected point candidates and same-branch sign brackets read-only."""

    solutions: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()

    def location_tolerance(*items: Evaluation) -> float:
        tolerances = [
            float(item.trial.xb_tolerance)
            for item in items
            if item.trial is not None and math.isfinite(float(item.trial.xb_tolerance))
        ]
        scale = max((abs(item.x_guess) for item in items), default=1.0)
        return max(8.0 * max(tolerances, default=0.0), MACHINE_EPS_FACTOR * np.finfo(np.float64).eps * scale)

    def verified_solution(
        *,
        bracket_id: int,
        bracket: Mapping[str, Any],
        candidate: Evaluation | None,
        left: Evaluation,
        right: Evaluation,
        stop: str,
        signature_changed: bool,
        signature_authoritative: bool,
    ) -> dict[str, Any]:
        root_row = _trial_row(candidate, edges_m=edges_m, previous_trial=None) if candidate is not None else {}
        verification: Evaluation | None = None
        verify_row: dict[str, Any] = {}
        population_residual = math.nan
        candidate_inventory_residual = math.nan
        verification_inventory_residual = math.nan
        candidate_nonnegative = False
        verification_nonnegative = False
        same_x_repeatable = False
        accepted = False
        verification_status = "NOT_ATTEMPTED"
        candidate_scalar_valid = bool(
            candidate is not None
            and candidate.trial is not None
            and abs(float(candidate.trial.signed_xb_residual)) <= float(candidate.trial.xb_tolerance)
        )
        if candidate_scalar_valid and candidate is not None and candidate.trial is not None:
            candidate_inventory_residual = float(candidate.trial.inventory.relative_residual)
            candidate_nonnegative = bool(np.all(np.asarray(candidate.trial.cell_number_m3) >= 0.0))
            # A safeguarded scalar root closes F(x)=T(x)-x at the root trial
            # itself.  Re-evaluating F(T(x)) would instead demand that the
            # raw Picard map be contractive at the root, which is neither the
            # scalar equation nor a valid criterion for a bracketed root.
            # Repeat the *same* binary64 input from state 244 instead.
            verification = budget.evaluate(
                float(candidate.x_guess), phase="root_same_x_repeatability", reuse=False
            )
            if verification is None:
                verification_status = "EVALUATION_BUDGET_EXHAUSTED"
            else:
                verification_status = "SAME_X_REPEAT_COMPLETED"
                verify_row = _trial_row(verification, edges_m=edges_m, previous_trial=None)
            if verification is not None and verification.trial is not None:
                verification_inventory_residual = float(verification.trial.inventory.relative_residual)
                verification_nonnegative = bool(
                    np.all(np.asarray(verification.trial.cell_number_m3) >= 0.0)
                )
                population_residual = closure_map.solver._population_observable_residual(
                    verification.trial.cell_number_m3, candidate.trial.cell_number_m3
                )
                same_x_repeatable = bool(
                    verification.x_guess == candidate.x_guess
                    and verification.trial.matrix_xb == candidate.trial.matrix_xb
                    and np.array_equal(
                        np.asarray(verification.trial.cell_number_m3),
                        np.asarray(candidate.trial.cell_number_m3),
                    )
                    and np.array_equal(
                        np.asarray(verification.trial.trace.departure_faces_m),
                        np.asarray(candidate.trial.trace.departure_faces_m),
                    )
                    and verification.trial.trace.lower_no_inflow_face_count
                    == candidate.trial.trace.lower_no_inflow_face_count
                    and verification.trial.trace.upper_no_inflow_face_count
                    == candidate.trial.trace.upper_no_inflow_face_count
                    and verification.trial.inventory.total_mol_m3 == candidate.trial.inventory.total_mol_m3
                    and verification.trial.inventory.matrix_mol_m3 == candidate.trial.inventory.matrix_mol_m3
                    and verification.trial.inventory.beta_resolved_mol_m3
                    == candidate.trial.inventory.beta_resolved_mol_m3
                )
                accepted = bool(
                    signature_authoritative
                    and same_x_repeatable
                    and abs(float(verification.trial.signed_xb_residual))
                    <= float(verification.trial.xb_tolerance)
                    and population_residual <= float(closure_map.solver._population_convergence_rtol)
                    and candidate_inventory_residual <= 1.0e-10
                    and verification_inventory_residual <= 1.0e-10
                    and candidate_nonnegative
                    and verification_nonnegative
                )
            elif verification is not None:
                verification_status = "INVALID_VERIFICATION_TRIAL"
        elif candidate is not None:
            verification_status = "NOT_ATTEMPTED_SCALAR_TOLERANCE_NOT_MET"
        interval_left = min(left.x_guess, right.x_guess)
        interval_right = max(left.x_guess, right.x_guess)
        return {
            "bracket_id": bracket_id,
            "bracket_kind": bracket.get("bracket_kind", "SIGN_CHANGE"),
            "scan_reason": bracket.get("reason", ""),
            "initial_left_x": float(bracket.get("left_x", left.x_guess)),
            "initial_right_x": float(bracket.get("right_x", right.x_guess)),
            "final_left_x": interval_left,
            "final_right_x": interval_right,
            "final_width": interval_right - interval_left,
            "root_interval_left_x": interval_left,
            "root_interval_right_x": interval_right,
            "root_location_tolerance": location_tolerance(left, right, *( [candidate] if candidate is not None else [] )),
            "stop_reason": stop,
            "signature_changed_within_bracket": signature_changed,
            "root_authority_signature_continuous": signature_authoritative,
            "root_found": accepted,
            "root_candidate_within_scalar_tolerance": candidate_scalar_valid,
            "root_candidate_within_final_interval": bool(
                candidate is not None and interval_left <= candidate.x_guess <= interval_right
            ),
            "root_xB": root_row.get("x_guess", math.nan),
            "root_residual": root_row.get("F_signed", math.nan),
            "root_tolerance": root_row.get("xB_tolerance", math.nan),
            "verification_residual": verify_row.get("F_signed", math.nan),
            "verification_tolerance": verify_row.get("xB_tolerance", math.nan),
            "verification_kind": "SAME_X_REPEATABILITY",
            "same_x_repeatable": same_x_repeatable,
            "verification_population_residual": population_residual,
            "population_tolerance": float(closure_map.solver._population_convergence_rtol),
            "candidate_inventory_relative_residual": candidate_inventory_residual,
            "verification_inventory_relative_residual": verification_inventory_residual,
            "inventory_relative_tolerance": 1.0e-10,
            "candidate_nonnegative_cells": candidate_nonnegative,
            "verification_nonnegative_cells": verification_nonnegative,
            "verification_status": verification_status,
            "verification_signature": verify_row.get("departure_signature_hash", ""),
            "root_signature": root_row.get("departure_signature_hash", ""),
        }

    for bracket_id, bracket in enumerate(brackets, start=1):
        kind = str(bracket.get("bracket_kind", "SIGN_CHANGE"))
        if kind not in {"SIGN_CHANGE", "POINT_CANDIDATE"}:
            continue
        if not bool(bracket.get("root_search_selected", True)):
            continue
        left_x = float(bracket["left_x"])
        right_x = float(bracket["right_x"])
        key = (kind, *sorted((left_x.hex(), right_x.hex())))
        if key in seen:
            continue
        seen.add(key)
        left = budget.evaluate(left_x, phase="root_endpoint")
        right = budget.evaluate(right_x, phase="root_endpoint")
        if left is None or right is None:
            audit_rows.append(
                {
                    "bracket_id": bracket_id,
                    "bracket_kind": kind,
                    "root_search_status": "ROOT_SEARCH_EVALUATION_BUDGET_EXHAUSTED_AT_ENDPOINT",
                    "left_x": left_x,
                    "right_x": right_x,
                }
            )
            continue
        if left.trial is None or right.trial is None:
            audit_rows.append(
                {
                    "bracket_id": bracket_id,
                    "bracket_kind": kind,
                    "root_search_status": "ROOT_SEARCH_INVALID_ENDPOINT",
                    "left_x": left_x,
                    "right_x": right_x,
                }
            )
            continue
        if kind == "POINT_CANDIDATE":
            # A residual-tolerance point is useful scan evidence, and its
            # same-x repeatability is still recorded below.  It is not,
            # however, authority for a safeguarded closure: that path is
            # deliberately restricted to a real, same-CDF-partition
            # sign-changing bracket from this immutable step.
            solutions.append(
                verified_solution(
                    bracket_id=bracket_id,
                    bracket=bracket,
                    candidate=left,
                    left=left,
                    right=right,
                    stop="point_residual_candidate",
                    signature_changed=False,
                    signature_authoritative=False,
                )
            )
            continue
        if not bool(bracket.get("sign_change")):
            continue
        initial_signature = _trial_row(left, edges_m=edges_m, previous_trial=None).get(
            "departure_signature_hash", ""
        )
        right_signature = _trial_row(right, edges_m=edges_m, previous_trial=None).get(
            "departure_signature_hash", ""
        )
        if initial_signature != right_signature:
            audit_rows.append(
                {
                    "bracket_id": bracket_id,
                    "bracket_kind": kind,
                    "root_search_status": "SIGN_CHANGE_ACROSS_SIGNATURE_TRANSITION_UNRESOLVED",
                    "left_x": left.x_guess,
                    "right_x": right.x_guess,
                    "left_signature": initial_signature,
                    "right_signature": right_signature,
                }
            )
            continue
        stop = "root_location_iteration_cap"
        signature_changed = False
        candidate: Evaluation | None = None
        if abs(float(left.trial.signed_xb_residual)) <= float(left.trial.xb_tolerance):
            candidate = left
            right = left
            stop = "left_endpoint_scalar_tolerance"
        elif abs(float(right.trial.signed_xb_residual)) <= float(right.trial.xb_tolerance):
            candidate = right
            left = right
            stop = "right_endpoint_scalar_tolerance"
        for iteration in range(1, ROOT_MAX_ITERATIONS + 1):
            if candidate is not None:
                break
            # Location width is provenance, not an acceptance surrogate.  A
            # bracket narrower than the scalar tolerance can still have no
            # endpoint satisfying F(x), so it must receive at least the
            # ordinary midpoint residual test before a root search stops.
            midpoint = 0.5 * (left.x_guess + right.x_guess)
            if midpoint == left.x_guess or midpoint == right.x_guess:
                stop = "binary64_endpoint_coalescence"
                break
            middle = budget.evaluate(midpoint, phase="root_bisection")
            if middle is None:
                stop = "root_bisection_evaluation_budget_exhausted"
                audit_rows.append(
                    {
                        "bracket_id": bracket_id,
                        "root_iteration": iteration,
                        "root_search_status": "ROOT_SEARCH_EVALUATION_BUDGET_EXHAUSTED",
                    }
                )
                break
            row = _trial_row(middle, edges_m=edges_m, previous_trial=None)
            row.update({"bracket_id": bracket_id, "root_iteration": iteration})
            audit_rows.append(row)
            if middle.trial is None:
                stop = "invalid_midpoint"
                break
            signature_changed = row.get("departure_signature_hash") != initial_signature
            if signature_changed:
                stop = "signature_transition_within_sign_bracket"
                break
            middle_f = float(middle.trial.signed_xb_residual)
            if abs(middle_f) <= float(middle.trial.xb_tolerance):
                candidate = middle
                left = right = middle
                stop = "midpoint_scalar_tolerance"
                break
            if float(left.trial.signed_xb_residual) * middle_f < 0.0:
                right = middle
            else:
                left = middle
        if candidate is None:
            candidate = min(
                (left, right),
                key=lambda item: abs(float(item.trial.signed_xb_residual)),
            )
        solutions.append(
            verified_solution(
                bracket_id=bracket_id,
                bracket=bracket,
                candidate=candidate,
                left=left,
                right=right,
                stop=stop,
                signature_changed=signature_changed,
                signature_authoritative=not signature_changed,
            )
        )
    return solutions, audit_rows


def _cluster_valid_roots(valid_roots: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Merge only roots whose resolved bisection intervals overlap in tolerance.

    A scalar-residual acceptance point is not itself a location certificate:
    overlapping brackets can legitimately return values many ulps apart.  The
    final sign-bracket interval is therefore the identity evidence, rather
    than an arbitrary ulp comparison of two tolerance-valid trial values.
    """

    ordered: list[dict[str, Any]] = []
    for row in valid_roots:
        value = float(row["root_xB"])
        left = float(row.get("root_interval_left_x", value))
        right = float(row.get("root_interval_right_x", value))
        left, right = min(left, right), max(left, right)
        resolution = max(
            float(row.get("root_location_tolerance", 0.0)),
            8.0 * float(row.get("root_tolerance", 0.0)),
            MACHINE_EPS_FACTOR * np.finfo(np.float64).eps * max(abs(value), 1.0),
        )
        ordered.append(
            {
                "row": row,
                "value": value,
                "left": left,
                "right": right,
                "resolution": resolution,
                "abs_residual": abs(float(row["root_residual"])),
            }
        )
    ordered.sort(key=lambda item: (item["left"], item["right"], item["value"]))
    clusters: list[dict[str, Any]] = []
    for item in ordered:
        if not clusters or item["left"] > clusters[-1]["right"] + max(item["resolution"], clusters[-1]["resolution"]):
            clusters.append(
                {
                    "left": item["left"],
                    "right": item["right"],
                    "resolution": item["resolution"],
                    "members": [item],
                }
            )
            continue
        cluster = clusters[-1]
        cluster["left"] = min(float(cluster["left"]), item["left"])
        cluster["right"] = max(float(cluster["right"]), item["right"])
        cluster["resolution"] = max(float(cluster["resolution"]), item["resolution"])
        cluster["members"].append(item)
    for cluster in clusters:
        representative = min(cluster["members"], key=lambda item: item["abs_residual"])
        cluster["root_xB"] = representative["value"]
        cluster["member_count"] = len(cluster["members"])
        cluster["member_bracket_ids"] = [item["row"].get("bracket_id") for item in cluster["members"]]
        cluster["member_bracket_kinds"] = sorted(
            {str(item["row"].get("bracket_kind", "")) for item in cluster["members"]}
        )
        cluster["has_authoritative_sign_bracket"] = any(
            str(item["row"].get("bracket_kind", "")) == "SIGN_CHANGE"
            and bool(item["row"].get("root_authority_signature_continuous", False))
            for item in cluster["members"]
        )
        del cluster["members"]
    return clusters


def _classify(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    cycles: Sequence[Mapping[str, Any]],
    contraction: Mapping[str, Any],
    roots: Sequence[Mapping[str, Any]],
    brackets: Sequence[Mapping[str, Any]],
    scan_audit: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Choose one required root-cause classification from recorded evidence."""

    final_cycle_iteration = max((int(row.get("iteration", 0)) for row in cycles), default=0)
    exact_cycles = [
        row
        for row in cycles
        if int(row.get("iteration", 0)) == final_cycle_iteration
        and bool(row.get("primitive_exact_period"))
    ]
    exact_periods = sorted({int(row["period"]) for row in exact_cycles})
    machine_cycles = [
        row
        for row in cycles
        if int(row.get("iteration", 0)) == final_cycle_iteration
        and bool(row.get("primitive_machine_precision_period"))
    ]
    valid_roots = [
        row
        for row in roots
        if bool(row.get("root_found"))
        and math.isfinite(float(row.get("root_residual", math.nan)))
        and abs(float(row["root_residual"])) <= float(row.get("root_tolerance", -1.0))
        and math.isfinite(float(row.get("verification_residual", math.nan)))
        and abs(float(row["verification_residual"]))
        <= float(row.get("verification_tolerance", row.get("root_tolerance", -1.0)))
        and math.isfinite(float(row.get("verification_population_residual", math.nan)))
        and float(row["verification_population_residual"])
        <= float(row.get("population_tolerance", math.inf))
    ]
    root_clusters = _cluster_valid_roots(valid_roots)
    root_values = [float(cluster["root_xB"]) for cluster in root_clusters]
    authoritative_sign_root_clusters = [
        cluster for cluster in root_clusters if bool(cluster.get("has_authoritative_sign_bracket", False))
    ]
    scan = dict(scan_audit or {})
    # This diagnostic deliberately has no interval/analytic enclosure.  No
    # finite-scan flag, including a stale caller-supplied one, may turn its
    # sampled observations into a no-root/unique-root certification.
    root_count_certified = False
    # A source-cell index is an implementation detail of the continuous
    # piecewise-constant CDF: at an internal edge the left limit and the
    # ``searchsorted(..., side='right')`` value are the same cumulative
    # measure.  Its hash may therefore change at a benign CDF kink.  Neither
    # that event nor a finite-difference residual across a sampled interval
    # is evidence of a discontinuous CR1 remap.  Only a separately established
    # trace-topology/limit failure may set this P0 blocker.
    discontinuity_statuses = {
        "DISCONTINUITY_CONFIRMED",
        "TRACE_TOPOLOGY_TRANSITION_UNRESOLVED",
        "TRACE_LIMIT_FAILURE_CONFIRMED",
    }
    discontinuous = any(
        str(bracket.get("status", "")) in discontinuity_statuses
        for bracket in brackets
        if str(bracket.get("bracket_kind", "")) == "SIGNATURE_TRANSITION"
    )
    source_cell_signature_transition_observed = any(
        str(bracket.get("bracket_kind", "")) in {
            "SIGNATURE_TRANSITION",
            "SIGN_CHANGE_ACROSS_SIGNATURE_TRANSITION",
        }
        for bracket in brackets
    )
    final_raw = raw_rows[-1] if raw_rows else {}
    direct = bool(
        math.isfinite(float(final_raw.get("abs_F", math.nan)))
        and float(final_raw.get("abs_F", math.nan)) <= float(final_raw.get("xB_tolerance", -1.0))
        and (
            bool(final_raw.get("unchanged_from_state244", False))
            or (
                math.isfinite(float(final_raw.get("population_observable_residual_to_previous", math.nan)))
                and float(final_raw.get("population_observable_residual_to_previous", math.inf))
                <= float(final_raw.get("population_convergence_tolerance", -1.0))
            )
        )
    )
    tail_contraction = contraction.get("last_64", {})
    median_q = float(tail_contraction.get("median_q_F", math.nan))
    monotone_contraction = bool(tail_contraction.get("monotone_residual_nonincreasing", False))
    predicted_remaining = tail_contraction.get("predicted_remaining_iterations_to_xB_tolerance")
    extended_picard_reasonable = bool(tail_contraction.get("extended_picard_within_reasonable_budget", False))
    final_residual = float(raw_rows[-1].get("abs_F", math.nan)) if raw_rows else math.nan
    final_tolerance = float(raw_rows[-1].get("xB_tolerance", math.nan)) if raw_rows else math.nan
    if len(authoritative_sign_root_clusters) > 1:
        classification = "STEP245_MULTIPLE_ADMISSIBLE_ROOTS"
    elif discontinuous:
        classification = "STEP245_DISCONTINUOUS_REMAP_ROOT_UNRESOLVED"
    elif direct:
        # A direct raw-Picard convergence observation establishes a closure on
        # that trajectory, but this finite audit still does not establish that
        # it is the only physical root in the full interval.
        classification = "OTHER_WITH_EXPLICIT_EVIDENCE"
    elif exact_periods and root_clusters:
        period = exact_periods[0]
        detector_cycle = next(row for row in exact_cycles if int(row["period"]) == period)
        classification = (
            "STEP245_P2_OR_P4_DETECTION_FAILURE"
            if period in (2, 4)
            and bool(detector_cycle.get("phase_values_distinct"))
            and bool(detector_cycle.get("cyclic_adjacent_residual_sign_change"))
            else "OTHER_WITH_EXPLICIT_EVIDENCE"
        )
    elif machine_cycles:
        classification = "STEP245_FLOATING_POINT_STAGNATION"
    else:
        classification = "OTHER_WITH_EXPLICIT_EVIDENCE"
    if direct:
        picard_regime = "DIRECT_CONVERGED"
    elif exact_periods:
        picard_regime = "PERIODIC_ORBIT"
    elif monotone_contraction and math.isfinite(median_q) and 0.0 < median_q < 1.0:
        picard_regime = (
            "MONOTONE_CONTRACTION"
            if extended_picard_reasonable
            else "STAGNATING_CONTRACTION"
        )
    elif math.isfinite(median_q) and median_q >= 1.0:
        picard_regime = "OSCILLATORY_OR_NONCONTRACTIVE"
    else:
        picard_regime = "OTHER_WITH_EXPLICIT_EVIDENCE"
    return {
        "step245_classification": classification,
        "raw_picard_final_residual": final_residual,
        "raw_picard_final_tolerance": final_tolerance,
        "raw_picard_direct_convergence_observed": direct,
        "exact_periods": exact_periods,
        "final_cycle_iteration": final_cycle_iteration,
        "machine_precision_period_rows": len(machine_cycles),
        "scalar_root_exists": True if root_clusters else None,
        "observed_scalar_root_exists": bool(root_clusters),
        "scalar_root_existence_status": (
            "OBSERVED_ADMISSIBLE_ROOT" if root_clusters else "UNRESOLVED_NO_INTERVAL_OR_ANALYTIC_ENCLOSURE"
        ),
        "root_count_certified": root_count_certified,
        "number_of_admissible_roots": None,
        "observed_admissible_root_clusters": len(root_clusters),
        "observed_authoritative_sign_bracket_root_clusters": len(authoritative_sign_root_clusters),
        "observed_admissible_root_count_lower_bound": len(authoritative_sign_root_clusters),
        "scalar_root_uniqueness_status": "UNRESOLVED_NO_INTERVAL_OR_ANALYTIC_ENCLOSURE",
        "admissible_root_xB": root_values,
        "root_clusters": root_clusters,
        "discontinuity_evidence": discontinuous,
        "source_cell_signature_transition_observed": source_cell_signature_transition_observed,
        "asymptotic_contraction_factor": median_q,
        "picard_regime": picard_regime,
        "predicted_remaining_iterations_to_xB_tolerance": predicted_remaining,
        "extended_picard_reasonable": extended_picard_reasonable,
        "scalar_scan_audit": scan,
        "root_search_bracket_count": len(roots),
    }


def _render_reports(
    report_root: Path,
    *,
    replay: Mapping[str, Any],
    raw_rows: Sequence[Mapping[str, Any]],
    cycles: Sequence[Mapping[str, Any]],
    interval: Mapping[str, Any],
    brackets: Sequence[Mapping[str, Any]],
    roots: Sequence[Mapping[str, Any]],
    classification: Mapping[str, Any],
    provenance: Mapping[str, Any],
) -> None:
    """Write the user-specified v2 report skeleton without inventing later gates."""

    _write_markdown(
        report_root / "00_step244_replay.md",
        "Step-244 deterministic replay",
        "```json\n" + json.dumps(_json_safe(replay), indent=2, sort_keys=True) + "\n```\n",
    )
    _write_markdown(
        report_root / "01_step245_raw_picard.md",
        "Step-245 raw Picard trace",
        "Raw Picard was evaluated only through the immutable state-244 closure map. "
        "No trial was accepted.\n\n```json\n"
        + json.dumps(
            _json_safe(
                {
                    "iterations_recorded": len(raw_rows),
                    "iteration_128": raw_rows[127] if len(raw_rows) >= 128 else "NOT_REACHED",
                    "iteration_256": raw_rows[255] if len(raw_rows) >= 256 else "NOT_REACHED",
                    "iteration_512": raw_rows[511] if len(raw_rows) >= 512 else "NOT_REACHED",
                    "iteration_1024": raw_rows[1023] if len(raw_rows) >= 1024 else "NOT_REACHED",
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n```\n",
    )
    _write_markdown(
        report_root / "02_step245_cycle_analysis.md",
        "Step-245 cycle analysis",
        "```json\n"
        + json.dumps(
            _json_safe(
                {
                    "all_exact_matching_periods": sorted({int(row["period"]) for row in cycles if row.get("exact_bitwise_period")}),
                    "primitive_exact_periods": sorted({int(row["period"]) for row in cycles if row.get("primitive_exact_period")}),
                    "all_machine_precision_matching_periods": sorted({int(row["period"]) for row in cycles if row.get("machine_precision_period")}),
                    "primitive_machine_precision_periods": sorted({int(row["period"]) for row in cycles if row.get("primitive_machine_precision_period")}),
                    "cycle_rows": len(cycles),
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n```\n",
    )
    _write_markdown(
        report_root / "03_step245_scalar_map.md",
        "Step-245 scalar map",
        "The scan interval is derived from frozen total inventory, the beta composition/volume bound, "
        "and the solver's [0,1] thermodynamic composition domain.\n\n```json\n"
        + json.dumps(
            _json_safe(
                {
                    "interval": interval,
                    "brackets": list(brackets),
                    "roots": list(roots),
                    "scan_audit": classification.get("scalar_scan_audit", {}),
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n```\n",
    )
    _write_markdown(
        report_root / "04_step245_root_classification.md",
        "Step-245 root classification",
        "```json\n" + json.dumps(_json_safe(classification), indent=2, sort_keys=True) + "\n```\n",
    )
    blocked = "BLOCKED_BY_STEP245_DIAGNOSIS_PENDING_MINIMAL_REPAIR_DECISION"
    for filename, title in (
        ("05_minimal_closure_repair.md", "Minimal closure repair"),
        ("06_step245_acceptance.md", "Step-245 acceptance"),
        ("07_local_240_260_regression.md", "Local 240--260 regression"),
        ("08_characteristic_self_convergence.md", "Characteristic self-convergence"),
        ("09_cohort_characteristic_parity.md", "Cohort--characteristic parity"),
        ("10_implicit_reference_adjudication.md", "Implicit reference adjudication"),
    ):
        _write_markdown(report_root / filename, title, f"Status: `{blocked}`.\n")
    _write_markdown(
        report_root / "11_model_boundary.md",
        "Model boundary",
        "This evidence package changes no PF/CUDA source, physical parameter, canonical measure, "
        "thermodynamic mapping, growth kernel, remap order, GP path, or strict SSPRK2 donor-bound policy.\n",
    )
    _write_markdown(
        report_root / "12_final_acceptance_report.md",
        "Final acceptance report",
        "```json\n"
        + json.dumps(
            _json_safe(
                {
                    "STATUS": "DIAGNOSIS_COMPLETE_MINIMAL_REPAIR_NOT_YET_APPLIED",
                    "STEP245_CLASSIFICATION": classification.get("step245_classification"),
                    "TIME_REFERENCE_V2": "NOT_ASSIGNED",
                    "IMPLICIT_TIME_INACCURACY_PROVEN": False,
                    "PF_SOURCE_MODIFIED": False,
                    "CUDA_RERUN": False,
                    "PHYSICAL_RETUNING": False,
                    "GP_RELEASE_RUN": False,
                    "P0_BLOCKERS": [classification.get("step245_classification")],
                }
            ),
            indent=2,
            sort_keys=True,
        )
        + "\n```\n",
    )
    _write_markdown(
        report_root / "13_reproduction_commands.md",
        "Reproduction commands",
        "```bash\n"
        "PYTHONPATH=src python scripts/diagnose_kwn_characteristic_closure_v2.py diagnose \\\n"
        "  --output-root outputs/kwn_characteristic_closure_v2 \\\n"
        "  --report-root reports/kwn_characteristic_closure_v2\n"
        "```\n\n"
        + "```json\n"
        + json.dumps(_json_safe(provenance), indent=2, sort_keys=True)
        + "\n```\n",
    )


def _diagnose(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = arguments.output_root.resolve()
    report_root = arguments.report_root.resolve()
    if output_root.exists() or report_root.exists():
        raise DiagnosticError("diagnostic output/report roots must not pre-exist")
    if float(arguments.dt_s) != DT_S:
        raise DiagnosticError(f"diagnosis must preserve frozen dt_s={DT_S}")
    if int(arguments.accepted_step) != LAST_ACCEPTED_STEP:
        raise DiagnosticError(f"diagnosis must stop at frozen accepted step={LAST_ACCEPTED_STEP}")
    if int(arguments.raw_picard_iterations) != MAX_RAW_PICARD_ITERATIONS:
        raise DiagnosticError("raw diagnostic must use exactly 1024 Picard iterations")
    if int(arguments.coarse_scan_points) != COARSE_SCAN_POINTS:
        raise DiagnosticError(f"coarse scalar-map scan must use exactly {COARSE_SCAN_POINTS} points")
    started = time.monotonic()
    output_root.mkdir(parents=True, exist_ok=False)
    report_root.mkdir(parents=True, exist_ok=False)
    replay_trace: list[dict[str, Any]] = []
    try:
        solver, context, replay_trace = _replay_to_step244(
            dt_s=float(arguments.dt_s), accepted_step=int(arguments.accepted_step)
        )
    except Exception as error:
        failure = {
            "STATUS": "FAIL_STEP244_DETERMINISTIC_REPLAY",
            "reason": f"{type(error).__name__}: {error}",
            "expected_step": LAST_ACCEPTED_STEP,
            "replay_rows": replay_trace,
        }
        _write_json(output_root / "analysis_provenance.json", failure)
        _write_markdown(report_root / "00_step244_replay.md", "Step-244 deterministic replay", "```json\n" + json.dumps(failure, indent=2) + "\n```\n")
        return failure
    if solver.step != LAST_ACCEPTED_STEP or solver.time_s != LAST_ACCEPTED_STEP * DT_S:
        raise DiagnosticError("replay did not end at the exact frozen state-244 time")

    state_record = _save_state244(
        output_root / "state_244.npz", solver=solver, context=context, replay_trace=replay_trace
    )
    formal_replay_comparison = _compare_formal_step244_trace(
        replay_trace, formal_trace_csv=arguments.formal_trace_csv
    )
    if formal_replay_comparison.get("status") != "PASS_EXACT_FORMAL_TRACE_MATCH":
        failure = {
            "STATUS": "FAIL_STEP244_DETERMINISTIC_REPLAY",
            "state_244": state_record,
            "formal_replay_comparison": formal_replay_comparison,
        }
        _write_json(output_root / "analysis_provenance.json", failure)
        _write_markdown(
            report_root / "00_step244_replay.md",
            "Step-244 deterministic replay",
            "```json\n" + json.dumps(_json_safe(failure), indent=2, sort_keys=True) + "\n```\n",
        )
        return failure
    closure_map = ImmutableStep245Map(
        solver, old_cell_number_m3=solver._beta_cell_numbers(), dt_s=float(arguments.dt_s)
    )
    raw_rows, cell_states, departure_faces = _raw_picard(
        closure_map,
        edges_m=context.edges_m,
        iterations=int(arguments.raw_picard_iterations),
    )
    if len(raw_rows) < MAX_RAW_PICARD_ITERATIONS:
        raise DiagnosticError("raw Picard trace terminated before the required 1024 diagnostic iterations")
    cycles = _cycle_analysis(raw_rows, cell_states, departure_faces)
    contraction = _window_contraction(raw_rows)
    interval = _physical_x_interval(solver)
    scalar_rows, brackets, _refined_rows, scalar_scan_audit, evaluation_budget = _scalar_scan(
        closure_map,
        edges_m=context.edges_m,
        interval=interval,
        raw_rows=raw_rows,
        cycles=cycles,
        coarse_points=int(arguments.coarse_scan_points),
        refinement_depth=int(arguments.refinement_depth),
    )
    roots, root_audit_rows = _solve_brackets(
        closure_map,
        edges_m=context.edges_m,
        brackets=brackets,
        budget=evaluation_budget,
    )
    scalar_scan_audit["evaluation_budget"] = evaluation_budget.audit()
    scalar_scan_audit["root_search_completed_within_shared_budget"] = not evaluation_budget.exhausted
    scalar_scan_audit["finite_audit_complete_under_declared_budget"] = bool(
        scalar_scan_audit["finite_audit_complete_under_declared_budget"]
        and not evaluation_budget.exhausted
    )
    classification = _classify(
        raw_rows=raw_rows,
        cycles=cycles,
        contraction=contraction,
        roots=roots,
        brackets=brackets,
        scan_audit=scalar_scan_audit,
    )

    np.savez_compressed(
        output_root / "step245_picard_cells.npz",
        cell_number_m3=np.asarray(cell_states, dtype=np.float64),
        departure_faces_m=np.asarray(departure_faces, dtype=np.float64),
        edges_m=np.asarray(context.edges_m, dtype=np.float64),
    )
    _write_csv(output_root / "step245_picard_trace.csv", raw_rows, fallback_fields=("iteration", "x_guess", "F_signed"))
    _write_csv(
        output_root / "step245_departure_signature.csv",
        [
            {
                "iteration": row.get("iteration"),
                "x_guess": row.get("x_guess"),
                "departure_signature_hash": row.get("departure_signature_hash"),
                "remap_state_hash": row.get("remap_state_hash"),
                "minimum_departure_radius_m": row.get("minimum_departure_radius_m"),
                "maximum_departure_radius_m": row.get("maximum_departure_radius_m"),
                "departure_faces_changing_source_cell": row.get("departure_faces_changing_source_cell"),
            }
            for row in raw_rows
        ],
        fallback_fields=("iteration", "departure_signature_hash"),
    )
    _write_csv(output_root / "step245_cycle_metrics.csv", cycles, fallback_fields=("iteration", "period", "d_x"))
    _write_csv(output_root / "step245_scalar_map.csv", scalar_rows, fallback_fields=("evaluation_id", "x_guess", "F_signed"))
    _write_csv(output_root / "step245_root_brackets.csv", list(brackets) + root_audit_rows, fallback_fields=("bracket_id", "left_x", "right_x"))
    _write_json(
        output_root / "step245_root_solution.json",
        {
            "classification": classification,
            "roots": roots,
            "contraction": contraction,
            "interval": interval,
            "scalar_scan_audit": scalar_scan_audit,
        },
    )
    provenance = {
        "schema_version": "KWN_CHARACTERISTIC_NONLINEAR_CLOSURE_DIAGNOSIS_V2",
        "task_name": TASK_NAME,
        "base_commit": BASE_COMMIT,
        "formal_failure_job": FORMAL_FAILURE_JOB,
        "formal_failure_report": FORMAL_FAILURE_REPORT,
        "git_branch": _git_value("branch", "--show-current"),
        "git_commit": _git_value("rev-parse", "HEAD"),
        "git_status": _git_value("status", "--short"),
        "validation_contract_hash": context.contract_hash,
        "canonical_snapshot": frozen_canonical_snapshot_provenance(),
        "dt_s": float(arguments.dt_s),
        "accepted_step": LAST_ACCEPTED_STEP,
        "candidate_step": FAILED_STEP,
        "state_244": state_record,
        "formal_step244_replay_comparison": formal_replay_comparison,
        "closure_map_initial_state_hash": closure_map.initial_state_hash,
        "closure_map_evaluation_count": len(closure_map.evaluations),
        "scalar_scan_audit": scalar_scan_audit,
        "all_closure_map_evaluations_left_state_unchanged": all(
            item.state_hash_before == item.state_hash_after == closure_map.initial_state_hash
            for item in closure_map.evaluations
        ),
        "wall_seconds": time.monotonic() - started,
    }
    _write_json(output_root / "analysis_provenance.json", provenance)
    _render_reports(
        report_root,
        replay={
            "STATUS": "PASS_STEP244_DETERMINISTIC_REPLAY",
            **state_record,
            "trace_rows": len(replay_trace),
            "formal_replay_comparison": formal_replay_comparison,
        },
        raw_rows=raw_rows,
        cycles=cycles,
        interval=interval,
        brackets=brackets,
        roots=roots,
        classification={**classification, "contraction": contraction},
        provenance=provenance,
    )
    return {
        "STATUS": "PASS_STEP245_DIAGNOSIS_COMPLETE",
        "STEP244_REPLAY": "PASS",
        "STEP244_STATE_HASH": state_record["state_244_sha256"],
        "STEP245_CLASSIFICATION": classification["step245_classification"],
        "SCALAR_ROOT_EXISTS": classification["scalar_root_exists"],
        "NUMBER_OF_ADMISSIBLE_ROOTS": classification["number_of_admissible_roots"],
        "P0_BLOCKERS": [classification["step245_classification"]],
        "KEY_REPORTS": [
            str(report_root / "00_step244_replay.md"),
            str(report_root / "01_step245_raw_picard.md"),
            str(report_root / "04_step245_root_classification.md"),
        ],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("diagnose",))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / TASK_NAME)
    parser.add_argument("--report-root", type=Path, default=ROOT / "reports" / TASK_NAME)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument("--accepted-step", type=int, default=LAST_ACCEPTED_STEP)
    parser.add_argument(
        "--raw-picard-iterations",
        type=int,
        choices=(MAX_RAW_PICARD_ITERATIONS,),
        default=MAX_RAW_PICARD_ITERATIONS,
    )
    parser.add_argument(
        "--coarse-scan-points",
        type=int,
        choices=(COARSE_SCAN_POINTS,),
        default=COARSE_SCAN_POINTS,
    )
    parser.add_argument("--refinement-depth", type=int, default=8)
    parser.add_argument("--formal-trace-csv", type=Path, default=Path(FORMAL_FAILURE_TRACE))
    arguments = parser.parse_args()
    try:
        final = _diagnose(arguments)
    except Exception as error:
        print(
            json.dumps(
                {
                    "STATUS": "FAIL_STEP245_DIAGNOSTIC_EXECUTION",
                    "reason": f"{type(error).__name__}: {error}",
                },
                sort_keys=True,
            ),
            file=sys.stderr,
        )
        return 2
    print(json.dumps(_json_safe(final), sort_keys=True))
    return 0 if final.get("STATUS") == "PASS_STEP245_DIAGNOSIS_COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
