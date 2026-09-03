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
ROOT_MAX_ITERATIONS = 96
MACHINE_EPS_FACTOR = 64.0


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
    if previous_candidate is None:
        l1_relative = math.nan
        linf = math.nan
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
    moments = _moments(edges_m, candidate)
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
            results.append(
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
    return results


def _window_contraction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    summary: dict[str, Any] = {}
    for width in (16, 32, 64, 128):
        sample = [float(row["q_F"]) for row in rows[-width:] if math.isfinite(float(row.get("q_F", math.nan)))]
        delta = [float(row["q_delta"]) for row in rows[-width:] if math.isfinite(float(row.get("q_delta", math.nan)))]
        summary[f"last_{width}"] = {
            "count_q_F": len(sample),
            "median_q_F": float(np.median(sample)) if sample else math.nan,
            "max_q_F": float(np.max(sample)) if sample else math.nan,
            "min_q_F": float(np.min(sample)) if sample else math.nan,
            "count_q_delta": len(delta),
            "median_q_delta": float(np.median(delta)) if delta else math.nan,
            "max_q_delta": float(np.max(delta)) if delta else math.nan,
        }
    return summary


def _scalar_scan(
    closure_map: ImmutableStep245Map,
    *,
    edges_m: np.ndarray,
    interval: Mapping[str, float],
    raw_rows: Sequence[Mapping[str, Any]],
    coarse_points: int,
    refinement_depth: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    """Scan the physical interval and make deterministic local refinements."""

    if coarse_points < 128:
        raise DiagnosticError("coarse scalar-map scan must contain at least 128 points")
    values = np.linspace(float(interval["x_min"]), float(interval["x_max"]), coarse_points, dtype=np.float64)
    evaluations = [closure_map.evaluate(float(value), phase="scalar_scan") for value in values]
    rows = [_trial_row(item, edges_m=edges_m, previous_trial=None) for item in evaluations]
    candidate_intervals: set[tuple[float, float, str]] = set()
    # Only adjacent points in the original coarse grid may form a bracket.
    # Joining two valid values across an invalid trial would create a false
    # sign change through an unphysical map segment.
    for (left_eval, left_row), (right_eval, right_row) in zip(
        zip(evaluations, rows), zip(evaluations[1:], rows[1:])
    ):
        if left_eval.trial is None or right_eval.trial is None:
            continue
        left_x = float(left_eval.x_guess)
        right_x = float(right_eval.x_guess)
        left_f = float(left_row["F_signed"])
        right_f = float(right_row["F_signed"])
        if left_f == 0.0 or right_f == 0.0 or left_f * right_f < 0.0:
            candidate_intervals.add((left_x, right_x, "F_sign_change"))
        if left_row["departure_signature_hash"] != right_row["departure_signature_hash"]:
            candidate_intervals.add((left_x, right_x, "departure_signature_change"))

    raw_values = [
        float(value)
        for row in raw_rows
        for value in (row.get("x_guess"), row.get("x_closure"))
        if isinstance(value, (float, int)) and math.isfinite(float(value))
    ]
    if raw_values:
        left = max(float(interval["x_min"]), min(raw_values))
        right = min(float(interval["x_max"]), max(raw_values))
        if right > left:
            candidate_intervals.add((left, right, "raw_picard_attractor_span"))

    refined_rows: list[dict[str, Any]] = []
    root_brackets: list[dict[str, Any]] = []
    cache: dict[str, Evaluation] = {item.x_guess.hex(): item for item in evaluations}

    def evaluate(value: float, phase: str) -> Evaluation:
        key = float(value).hex()
        if key in cache:
            cached = cache[key]
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
        item = closure_map.evaluate(float(value), phase=phase)
        cache[key] = item
        return item

    for left0, right0, reason in sorted(candidate_intervals):
        left = evaluate(left0, "scalar_refine_endpoint")
        right = evaluate(right0, "scalar_refine_endpoint")
        if left.trial is None or right.trial is None:
            root_brackets.append(
                {
                    "reason": reason,
                    "left_x": left0,
                    "right_x": right0,
                    "status": "INVALID_ENDPOINT",
                }
            )
            continue
        for depth in range(refinement_depth):
            midpoint = 0.5 * (left.x_guess + right.x_guess)
            if midpoint == left.x_guess or midpoint == right.x_guess:
                break
            middle = evaluate(midpoint, "scalar_refine")
            refined_rows.append(_trial_row(middle, edges_m=edges_m, previous_trial=None))
            if middle.trial is None:
                break
            left_f = float(left.trial.signed_xb_residual)
            middle_f = float(middle.trial.signed_xb_residual)
            right_f = float(right.trial.signed_xb_residual)
            if middle_f == 0.0:
                left = right = middle
                break
            if left_f * middle_f <= 0.0:
                right = middle
            elif middle_f * right_f <= 0.0:
                left = middle
            else:
                # Signature-only refinement is centered deterministically.
                if abs(left_f) <= abs(right_f):
                    right = middle
                else:
                    left = middle
        left_row = _trial_row(left, edges_m=edges_m, previous_trial=None)
        right_row = _trial_row(right, edges_m=edges_m, previous_trial=None)
        root_brackets.append(
            {
                "reason": reason,
                "left_x": left.x_guess,
                "right_x": right.x_guess,
                "left_F": left_row.get("F_signed", math.nan),
                "right_F": right_row.get("F_signed", math.nan),
                "left_signature": left_row.get("departure_signature_hash", ""),
                "right_signature": right_row.get("departure_signature_hash", ""),
                "final_width": right.x_guess - left.x_guess,
                "sign_change": (
                    left.trial is not None
                    and right.trial is not None
                    and float(left.trial.signed_xb_residual) * float(right.trial.signed_xb_residual) <= 0.0
                ),
                "status": "REFINED",
            }
        )
    all_rows = rows + refined_rows
    return all_rows, root_brackets, refined_rows


def _solve_brackets(
    closure_map: ImmutableStep245Map,
    *,
    edges_m: np.ndarray,
    brackets: Sequence[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Bisection only on actual sign-changing brackets; never commit a trial."""

    solutions: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    seen: set[tuple[str, str]] = set()
    for bracket_id, bracket in enumerate(brackets, start=1):
        if not bracket.get("sign_change"):
            continue
        left_x = float(bracket["left_x"])
        right_x = float(bracket["right_x"])
        key = tuple(sorted((left_x.hex(), right_x.hex())))
        if key in seen:
            continue
        seen.add(key)
        left = closure_map.evaluate(left_x, phase="root_endpoint", reuse=True)
        right = closure_map.evaluate(right_x, phase="root_endpoint", reuse=True)
        if left.trial is None or right.trial is None:
            continue
        initial_width = right.x_guess - left.x_guess
        root: Evaluation | None = None
        verification: Evaluation | None = None
        stop = "iteration_cap"
        signature_changed = False
        endpoint_root = next(
            (
                candidate
                for candidate in (left, right)
                if candidate.trial is not None
                and abs(float(candidate.trial.signed_xb_residual)) <= float(candidate.trial.xb_tolerance)
            ),
            None,
        )
        if endpoint_root is not None:
            root = endpoint_root
            verification = closure_map.evaluate(
                float(endpoint_root.trial.matrix_xb), phase="root_map_verification", reuse=True
            )
            stop = "endpoint_residual_tolerance_reached"
        for iteration in range(1, ROOT_MAX_ITERATIONS + 1):
            if root is not None:
                break
            midpoint = 0.5 * (left.x_guess + right.x_guess)
            if midpoint == left.x_guess or midpoint == right.x_guess:
                stop = "binary64_endpoint_coalescence"
                break
            middle = closure_map.evaluate(midpoint, phase="root_bisection", reuse=True)
            row = _trial_row(middle, edges_m=edges_m, previous_trial=None)
            row.update({"bracket_id": bracket_id, "root_iteration": iteration})
            audit_rows.append(row)
            if middle.trial is None:
                stop = "invalid_midpoint"
                break
            signature_changed = signature_changed or (
                _trial_row(left, edges_m=edges_m, previous_trial=None).get("departure_signature_hash")
                != row.get("departure_signature_hash")
                or _trial_row(right, edges_m=edges_m, previous_trial=None).get("departure_signature_hash")
                != row.get("departure_signature_hash")
            )
            if abs(float(middle.trial.signed_xb_residual)) <= float(middle.trial.xb_tolerance):
                root = middle
                verification = closure_map.evaluate(
                    float(middle.trial.matrix_xb), phase="root_map_verification", reuse=True
                )
                stop = "residual_tolerance_reached"
                break
            if float(left.trial.signed_xb_residual) * float(middle.trial.signed_xb_residual) <= 0.0:
                right = middle
            else:
                left = middle
        root_row = _trial_row(root, edges_m=edges_m, previous_trial=None) if root is not None else {}
        verify_row = _trial_row(verification, edges_m=edges_m, previous_trial=None) if verification is not None else {}
        population_residual = math.nan
        if root is not None and verification is not None and root.trial is not None and verification.trial is not None:
            population_residual = closure_map.solver._population_observable_residual(
                verification.trial.cell_number_m3, root.trial.cell_number_m3
            )
        solutions.append(
            {
                "bracket_id": bracket_id,
                "initial_left_x": left_x,
                "initial_right_x": right_x,
                "initial_width": initial_width,
                "final_left_x": left.x_guess,
                "final_right_x": right.x_guess,
                "final_width": right.x_guess - left.x_guess,
                "stop_reason": stop,
                "signature_changed_within_bracket": signature_changed,
                "root_found": root is not None,
                "root_xB": root_row.get("x_guess", math.nan),
                "root_residual": root_row.get("F_signed", math.nan),
                "root_tolerance": root_row.get("xB_tolerance", math.nan),
                "verification_residual": verify_row.get("F_signed", math.nan),
                "verification_population_residual": population_residual,
                "population_tolerance": float(closure_map.solver._population_convergence_rtol),
                "verification_signature": verify_row.get("departure_signature_hash", ""),
                "root_signature": root_row.get("departure_signature_hash", ""),
            }
        )
    return solutions, audit_rows


def _classify(
    *,
    raw_rows: Sequence[Mapping[str, Any]],
    cycles: Sequence[Mapping[str, Any]],
    contraction: Mapping[str, Any],
    roots: Sequence[Mapping[str, Any]],
    brackets: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Choose one required root-cause classification from recorded evidence."""

    exact_cycles = [row for row in cycles if bool(row.get("exact_bitwise_period"))]
    exact_periods = sorted({int(row["period"]) for row in exact_cycles})
    machine_cycles = [row for row in cycles if bool(row.get("machine_precision_period"))]
    valid_roots = [
        row
        for row in roots
        if bool(row.get("root_found"))
        and math.isfinite(float(row.get("root_residual", math.nan)))
        and abs(float(row["root_residual"])) <= float(row.get("root_tolerance", -1.0))
        and math.isfinite(float(row.get("verification_residual", math.nan)))
        and abs(float(row["verification_residual"])) <= float(row.get("root_tolerance", -1.0))
        and math.isfinite(float(row.get("verification_population_residual", math.nan)))
        and float(row["verification_population_residual"])
        <= float(row.get("population_tolerance", math.inf))
    ]
    unique_root_values: list[float] = []
    for row in valid_roots:
        value = float(row["root_xB"])
        if not any(abs(value - prior) <= 8.0 * np.finfo(np.float64).eps * max(abs(value), abs(prior), 1.0) for prior in unique_root_values):
            unique_root_values.append(value)
    # A departure-cell signature transition is a diagnostic refinement trigger,
    # not by itself a discontinuity: a piecewise-constant CDF remap can change
    # source cells while retaining a continuous scalar residual.  It becomes a
    # blocker only when a sign bracket cannot be closed at binary64 resolution.
    discontinuous = any(
        bool(row.get("signature_changed_within_bracket"))
        and not bool(row.get("root_found"))
        and str(row.get("stop_reason"))
        in {"binary64_endpoint_coalescence", "iteration_cap", "invalid_midpoint"}
        for row in roots
    )
    direct = [
        row
        for row in raw_rows
        if math.isfinite(float(row.get("abs_F", math.nan)))
        and float(row["abs_F"]) <= float(row.get("xB_tolerance", -1.0))
    ]
    median_q = float(contraction.get("last_64", {}).get("median_q_F", math.nan))
    final_residual = float(raw_rows[-1].get("abs_F", math.nan)) if raw_rows else math.nan
    final_tolerance = float(raw_rows[-1].get("xB_tolerance", math.nan)) if raw_rows else math.nan
    if len(unique_root_values) > 1:
        classification = "STEP245_MULTIPLE_ADMISSIBLE_ROOTS"
    elif discontinuous and not unique_root_values:
        classification = "STEP245_DISCONTINUOUS_REMAP_ROOT_UNRESOLVED"
    elif not unique_root_values:
        classification = "STEP245_NO_ADMISSIBLE_ROOT_AT_CURRENT_DT"
    elif exact_periods:
        classification = (
            "STEP245_P2_OR_P4_DETECTION_FAILURE"
            if any(period in (2, 4) for period in exact_periods)
            else "STEP245_HIGHER_PERIOD_PICARD_BUT_UNIQUE_SCALAR_ROOT"
        )
    elif direct:
        classification = "STEP245_SLOW_SINGLE_FIXED_POINT"
    elif machine_cycles:
        classification = "STEP245_FLOATING_POINT_STAGNATION"
    elif math.isfinite(median_q) and median_q < 1.0:
        classification = "STEP245_SLOW_SINGLE_FIXED_POINT"
    else:
        classification = "STEP245_NONCONTRACTIVE_BUT_UNIQUE_SCALAR_ROOT"
    return {
        "step245_classification": classification,
        "raw_picard_final_residual": final_residual,
        "raw_picard_final_tolerance": final_tolerance,
        "raw_picard_direct_convergence_observed": bool(direct),
        "exact_periods": exact_periods,
        "machine_precision_period_rows": len(machine_cycles),
        "scalar_root_exists": bool(unique_root_values),
        "number_of_admissible_roots": len(unique_root_values),
        "admissible_root_xB": unique_root_values,
        "discontinuity_evidence": discontinuous,
        "asymptotic_contraction_factor": median_q,
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
                    "exact_periods": sorted({int(row["period"]) for row in cycles if row.get("exact_bitwise_period")}),
                    "machine_precision_periods": sorted({int(row["period"]) for row in cycles if row.get("machine_precision_period")}),
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
        + json.dumps(_json_safe({"interval": interval, "brackets": list(brackets), "roots": list(roots)}), indent=2, sort_keys=True)
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
    if int(arguments.raw_picard_iterations) < MAX_RAW_PICARD_ITERATIONS:
        raise DiagnosticError("raw diagnostic must retain the required 1024 Picard iterations")
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
    scalar_rows, brackets, _refined_rows = _scalar_scan(
        closure_map,
        edges_m=context.edges_m,
        interval=interval,
        raw_rows=raw_rows,
        coarse_points=int(arguments.coarse_scan_points),
        refinement_depth=int(arguments.refinement_depth),
    )
    roots, root_audit_rows = _solve_brackets(closure_map, edges_m=context.edges_m, brackets=brackets)
    classification = _classify(
        raw_rows=raw_rows,
        cycles=cycles,
        contraction=contraction,
        roots=roots,
        brackets=brackets,
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
        {"classification": classification, "roots": roots, "contraction": contraction, "interval": interval},
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
    parser.add_argument("--raw-picard-iterations", type=int, default=MAX_RAW_PICARD_ITERATIONS)
    parser.add_argument("--coarse-scan-points", type=int, default=COARSE_SCAN_POINTS)
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
