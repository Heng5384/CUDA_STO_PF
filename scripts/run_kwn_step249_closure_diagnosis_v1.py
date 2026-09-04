#!/usr/bin/env python3
"""Read-only diagnosis of the frozen CR1 step-249 nonlinear closure map.

This is deliberately a diagnostic program, not a new controller.  It first
replays the frozen accepted trajectory through step 248 twice, then evaluates
only ``T(x) - x`` from an immutable copy of that accepted state.  It never
calls ``advance_one`` for candidate step 249, never changes a timestep, and
never invokes a scalar-root safeguard.  A later, separately committed phase
may use this evidence to decide whether a narrow generic controller is legal.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
import hashlib
import json
import math
from pathlib import Path
import platform
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
from kwn_mvp.conservative_remap import (  # noqa: E402
    CharacteristicRemapPartition,
    CharacteristicTraceTopology,
)
from kwn_mvp.population_metrics import cell_moments_from_piecewise_constant_cells  # noqa: E402
from kwn_mvp.solver import RadiusGridOverflowError, SolverConfig  # noqa: E402
from scripts.frozen_canonical_smooth_population_v1 import (  # noqa: E402
    build_frozen_canonical_context,
    frozen_canonical_snapshot_provenance,
)


TASK_NAME = "kwn_step249_closure_diagnosis_v1"
REQUIRED_BRANCH = "codex/kwn-step249-closure-diagnosis-v1"
BASE_COMMIT = "512cc0a54cea75e23306d699a4224489226e812f"
FROZEN_START_COMMIT = "9269e07a2d0fafbf35be950b858373fb71b54e4b"
FROZEN_CONTRACT_HASH = "d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333"
FORMAL_TRACE = Path(
    "/data/home/luozhiheng/tmp/"
    "kwn_characteristic_reference_v1_2ee679437421_20260902T202810Z/"
    "formal_all_14d_v4/outputs/kwn_characteristic_reference_v1/"
    "characteristic_fixed_point_trace.csv"
)
DT_S = 0.015625
FORMAL_LAST_ACCEPTED_STEP = 244
STEP_245 = 245
STEP_248 = 248
STEP_249 = 249
RAW_PICARD_CAP = 128
RAW_DIAGNOSTIC_CAP = 1024
LOCAL_MAP_POINTS = 64
REPLAY_COUNT = 2
MACHINE_EPS_FACTOR = 64.0
EXPECTED_FAILURE = {
    "xb_residual": 1.177e-11,
    "population_residual": 4.831e-10,
    "cell_measure": 2.557e-7,
}


class Step249DiagnosisError(RuntimeError):
    """A concrete failure of the frozen diagnostic contract."""


@dataclass(frozen=True)
class Evaluation:
    """One side-effect-free evaluation of an immutable CR1 scalar map."""

    evaluation_id: int
    phase: str
    x_guess: float
    trial: Any | None
    state_hash_before: str
    state_hash_after: str
    error_type: str | None
    error_message: str | None


class ImmutableStepMap:
    """Evaluate CR1 closure trials from one immutable accepted macro state."""

    def __init__(
        self,
        solver: CharacteristicReferenceSolver,
        *,
        old_cell_number_m3: np.ndarray,
        dt_s: float,
        state_label: str,
    ) -> None:
        self.solver = solver
        self.old_cell_number_m3 = np.asarray(old_cell_number_m3, dtype=np.float64).copy()
        self.dt_s = float(dt_s)
        self.x_start = float(solver.matrix_xb)
        self.state_label = str(state_label)
        self.initial_state_hash = _state_hash(solver)
        self._next_id = 1
        self.evaluations: list[Evaluation] = []
        self._cache: dict[str, Evaluation] = {}

    def evaluate(self, x_guess: float, *, phase: str, reuse: bool = False) -> Evaluation:
        """Run exactly one non-committing closure trial from the accepted state."""

        value = float(x_guess)
        if not math.isfinite(value):
            raise Step249DiagnosisError("closure-map x guess must be finite")
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
        before = _state_hash(self.solver)
        if before != self.initial_state_hash:
            raise Step249DiagnosisError(
                f"{self.state_label} changed before a read-only map evaluation"
            )
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
        after = _state_hash(self.solver)
        if before != after or after != self.initial_state_hash:
            raise Step249DiagnosisError(
                f"{self.state_label} was changed by a supposedly read-only map evaluation"
            )
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


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    return value


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(value)), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, title: str, value: Mapping[str, Any] | str) -> None:
    if isinstance(value, str):
        body = value.rstrip()
    else:
        body = "```json\n" + json.dumps(_json_safe(dict(value)), indent=2, sort_keys=True) + "\n```"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fallback_fields: Sequence[str]) -> None:
    materialized = [dict(_json_safe(row)) for row in rows]
    fields: list[str] = []
    for row in materialized:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = list(fallback_fields)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(materialized)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash_arrays(arrays: Mapping[str, np.ndarray]) -> str:
    digest = hashlib.sha256()
    for key in sorted(arrays):
        array = np.ascontiguousarray(np.asarray(arrays[key]))
        digest.update(key.encode("utf-8"))
        digest.update(str(array.dtype).encode("ascii"))
        digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
        digest.update(array.tobytes())
    return digest.hexdigest()


def _state_hash(solver: CharacteristicReferenceSolver) -> str:
    return _hash_arrays({key: np.asarray(value) for key, value in solver.state_arrays().items()})


def _float_encoding(value: float) -> tuple[str, str]:
    number = float(value)
    bits = int(np.asarray([number], dtype=np.float64).view(np.uint64)[0])
    return number.hex(), f"0x{bits:016x}"


def _moments(edges_m: np.ndarray, cells: np.ndarray) -> dict[str, float]:
    values = cell_moments_from_piecewise_constant_cells(edges_m, cells)
    return {f"M{order}": float(np.sum(values[order], dtype=np.float64)) for order in range(4)}


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *arguments], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _source_identity() -> dict[str, Any]:
    source = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status": _git("status", "--short"),
        "base_commit": _git("rev-parse", BASE_COMMIT),
        "frozen_start_commit": _git("rev-parse", FROZEN_START_COMMIT),
    }
    for label, commit in (("base_is_ancestor", BASE_COMMIT), ("frozen_start_is_ancestor", FROZEN_START_COMMIT)):
        source[label] = subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
    if source["git_branch"] != REQUIRED_BRANCH:
        raise Step249DiagnosisError(f"diagnosis requires branch {REQUIRED_BRANCH}")
    if source["git_status"]:
        raise Step249DiagnosisError("diagnosis requires a clean source tree")
    if not source["base_is_ancestor"] or not source["frozen_start_is_ancestor"]:
        raise Step249DiagnosisError("source does not descend from the frozen step-249 start")
    return source


def _runtime_provenance() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }


def _accepted_row(solver: CharacteristicReferenceSolver, diagnostic: Any, edges_m: np.ndarray) -> dict[str, Any]:
    cells = np.asarray(solver._beta_cell_numbers(), dtype=np.float64)
    return {
        "step": int(diagnostic.step),
        "time_s": float(diagnostic.time_s),
        "dt_s": float(diagnostic.dt_s),
        "matrix_xB": float(diagnostic.matrix_xb),
        "fixed_point_iterations": int(diagnostic.fixed_point_iterations),
        "fixed_point_picard_iterations": int(diagnostic.fixed_point_picard_iterations),
        "fixed_point_xb_residual": float(diagnostic.fixed_point_xb_residual),
        "fixed_point_xb_tolerance": float(diagnostic.fixed_point_xb_tolerance),
        "fixed_point_population_residual": float(diagnostic.fixed_point_population_residual),
        "fixed_point_cell_measure_residual": float(diagnostic.fixed_point_cell_measure_residual),
        "fixed_point_convergence_rate": float(diagnostic.fixed_point_convergence_rate),
        "fixed_point_convergence_mode": str(diagnostic.fixed_point_convergence_mode),
        "fixed_point_periodic_cycle_period": int(diagnostic.fixed_point_periodic_cycle_period),
        "fixed_point_bracketed_root_iterations": int(diagnostic.fixed_point_bracketed_root_iterations),
        "fixed_point_bracket_initial_width": float(diagnostic.fixed_point_bracket_initial_width),
        "fixed_point_bracket_final_width": float(diagnostic.fixed_point_bracket_final_width),
        "fixed_point_bracket_left_xb": float(diagnostic.fixed_point_bracket_left_xb),
        "fixed_point_bracket_right_xb": float(diagnostic.fixed_point_bracket_right_xb),
        "fixed_point_bracket_left_signed_residual": float(diagnostic.fixed_point_bracket_left_signed_residual),
        "fixed_point_bracket_right_signed_residual": float(diagnostic.fixed_point_bracket_right_signed_residual),
        "fixed_point_root_verification_kind": str(diagnostic.fixed_point_root_verification_kind),
        "inventory_relative_residual": float(diagnostic.inventory.relative_residual),
        "Q_total_mol_m3": float(diagnostic.inventory.total_mol_m3),
        "Q_beta_mol_m3": float(diagnostic.inventory.beta_resolved_mol_m3),
        "Q_matrix_mol_m3": float(diagnostic.inventory.matrix_mol_m3),
        "rmin_number_loss_m3": float(diagnostic.rmin_number_loss_m3),
        "rmin_mol_b_loss_mol_m3": float(diagnostic.rmin_mol_b_loss_mol_m3),
        "remap_number_conservation_residual_m3": float(diagnostic.remap_number_conservation_residual_m3),
        "state_array_hash": _state_hash(solver),
        "nonfinite_cell_count": int(np.count_nonzero(~np.isfinite(cells))),
        "negative_cell_count": int(np.count_nonzero(cells < 0.0)),
        **_moments(edges_m, cells),
    }


def _formal_row(row: Mapping[str, Any]) -> dict[str, Any]:
    fields = (
        "step", "time_s", "dt_s", "fixed_point_iterations", "fixed_point_picard_iterations",
        "fixed_point_xb_residual", "fixed_point_population_residual",
        "fixed_point_cell_measure_residual", "fixed_point_convergence_rate",
        "fixed_point_convergence_mode", "inventory_relative_residual", "rmin_number_loss_m3",
        "rmin_mol_b_loss_mol_m3", "remap_number_conservation_residual_m3",
    )
    return {field: row[field] for field in fields}


def _compare_formal_trace(rows: Sequence[Mapping[str, Any]], trace_path: Path) -> dict[str, Any]:
    if not trace_path.is_file():
        return {"status": "FAIL", "reason": "formal trace CSV is unreadable", "path": str(trace_path)}
    if len(rows) != FORMAL_LAST_ACCEPTED_STEP:
        return {"status": "FAIL", "reason": "replay did not produce all 244 accepted rows"}
    with trace_path.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    formal = {
        int(row["step"]): row
        for row in source_rows
        if row.get("policy") == "CR1_dt_0.015625s" and row.get("step", "").isdigit()
        and 1 <= int(row["step"]) <= FORMAL_LAST_ACCEPTED_STEP
    }
    missing = [step for step in range(1, FORMAL_LAST_ACCEPTED_STEP + 1) if step not in formal]
    if missing:
        return {"status": "FAIL", "reason": f"formal trace misses steps {missing[:8]}"}
    fields = tuple(_formal_row(rows[0]).keys())
    mismatches: list[dict[str, Any]] = []
    for step, observed in enumerate(rows, start=1):
        expected = formal[step]
        for field in fields:
            if field == "fixed_point_convergence_mode":
                equal = str(observed[field]) == expected[field]
            elif field in {"step", "fixed_point_iterations", "fixed_point_picard_iterations"}:
                equal = int(observed[field]) == int(expected[field])
            else:
                left = float(observed[field])
                right = float(expected[field])
                equal = left == right or (math.isnan(left) and math.isnan(right))
            if not equal:
                mismatches.append({"step": step, "field": field, "formal": expected[field], "replay": observed[field]})
    return {
        "status": "PASS_EXACT_FORMAL_CR1_TRACE_1_TO_244_MATCH" if not mismatches else "FAIL",
        "formal_state_244_archive": "NOT_AVAILABLE_IN_V1_RUN_ROOT",
        "trace_csv": str(trace_path),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:32],
    }


def _replay_through_step248(context: Any) -> tuple[CharacteristicReferenceSolver, list[dict[str, Any]], list[dict[str, Any]]]:
    solver = CharacteristicReferenceSolver(SolverConfig.from_mapping(context.mapping))
    formal_rows: list[dict[str, Any]] = []
    closure_rows: list[dict[str, Any]] = []
    for expected_step in range(1, STEP_248 + 1):
        diagnostic = solver.advance_one(maximum_dt_s=DT_S)
        if diagnostic.step != expected_step:
            raise Step249DiagnosisError(f"replay committed step {diagnostic.step}, expected {expected_step}")
        row = _accepted_row(solver, diagnostic, context.edges_m)
        if expected_step <= FORMAL_LAST_ACCEPTED_STEP:
            formal_rows.append(_formal_row(row))
        else:
            closure_rows.append(row)
    return solver, formal_rows, closure_rows


def _arrays_equal(first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]) -> bool:
    return set(first) == set(second) and all(np.array_equal(first[key], second[key]) for key in first)


def _rows_equal_with_nan(first: Sequence[Mapping[str, Any]], second: Sequence[Mapping[str, Any]]) -> bool:
    """Compare accepted telemetry without treating two NaNs as a difference."""

    if len(first) != len(second):
        return False
    for left, right in zip(first, second):
        if set(left) != set(right):
            return False
        for key in left:
            left_value = left[key]
            right_value = right[key]
            if isinstance(left_value, float) and isinstance(right_value, float):
                if left_value == right_value or (math.isnan(left_value) and math.isnan(right_value)):
                    continue
                return False
            if left_value != right_value:
                return False
    return True


def _save_state248(path: Path, *, solver: CharacteristicReferenceSolver, context: Any, history: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    cells = np.asarray(solver._beta_cell_numbers(), dtype=np.float64)
    inventory = solver.ledger.snapshot(
        matrix_xb=solver.matrix_xb,
        populations=solver.population_list(),
        beta_resolved_fraction=1.0,
    )
    moments = _moments(context.edges_m, cells)
    arrays = {key: np.asarray(value) for key, value in solver.state_arrays().items()}
    arrays.update({
        "cell_edges_m": np.asarray(context.edges_m, dtype=np.float64),
        "cell_number_m3": cells,
        "xB_matrix": np.asarray([solver.matrix_xb], dtype=np.float64),
        "Q_total_mol_m3": np.asarray([inventory.total_mol_m3], dtype=np.float64),
        "Q_beta_mol_m3": np.asarray([inventory.beta_resolved_mol_m3], dtype=np.float64),
        "Q_matrix_mol_m3": np.asarray([inventory.matrix_mol_m3], dtype=np.float64),
        "M0": np.asarray([moments["M0"]], dtype=np.float64),
        "M1": np.asarray([moments["M1"]], dtype=np.float64),
        "M2": np.asarray([moments["M2"]], dtype=np.float64),
        "M3": np.asarray([moments["M3"]], dtype=np.float64),
        "closure_history_245_248_json": np.asarray(json.dumps(_json_safe(list(history)), sort_keys=True)),
        "state_metadata_json": np.asarray(json.dumps(_json_safe({
            "schema_version": "KWN_STEP249_CLOSURE_DIAGNOSIS_V1_STATE_248",
            "step": int(solver.step), "time_s": float(solver.time_s),
            "validation_contract_hash": context.contract_hash,
            "canonical_state_hash": context.canonical_hash,
            "fixture_hash": context.fixture_hash,
        }), sort_keys=True)),
    })
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return {
        "file": str(path), "sha256": _sha256_file(path), "state_array_hash": _state_hash(solver),
        "step": int(solver.step), "time_s": float(solver.time_s), "xB": float(solver.matrix_xb),
        "Q_total_mol_m3": float(inventory.total_mol_m3), "Q_beta_mol_m3": float(inventory.beta_resolved_mol_m3),
        "Q_matrix_mol_m3": float(inventory.matrix_mol_m3), **moments,
    }


def _partition_signature(partition: CharacteristicRemapPartition | None) -> tuple[str, str]:
    if partition is None:
        return "", ""
    source_hash = hashlib.sha256(
        np.ascontiguousarray(partition.source_cell_indices, dtype=np.int64).tobytes()
    ).hexdigest()
    signature = _hash_arrays({
        "source_cell_indices": partition.source_cell_indices,
        "lower_endpoint_mask": partition.lower_endpoint_mask,
        "upper_endpoint_mask": partition.upper_endpoint_mask,
        "identity_departure_map": np.asarray([partition.identity_departure_map], dtype=np.bool_),
        "lower_no_inflow_face_count": np.asarray([partition.lower_no_inflow_face_count], dtype=np.int64),
        "upper_no_inflow_face_count": np.asarray([partition.upper_no_inflow_face_count], dtype=np.int64),
    })
    return signature, source_hash


def _topology_signature(topology: CharacteristicTraceTopology | None) -> str:
    if topology is None:
        return ""
    payload = _hash_arrays({
        "node_sign": topology.node_sign,
        "gauss_left_sign": topology.gauss_left_sign,
        "gauss_right_sign": topology.gauss_right_sign,
        "valid_interval": topology.valid_interval,
        "arrival_face_run_id": topology.arrival_face_run_id,
        "lower_no_inflow_face_count": np.asarray([topology.lower_no_inflow_face_count], dtype=np.int64),
        "upper_no_inflow_face_count": np.asarray([topology.upper_no_inflow_face_count], dtype=np.int64),
        "identity_departure_map": np.asarray([topology.identity_departure_map], dtype=np.bool_),
    })
    return hashlib.sha256((topology.mode + ":" + payload).encode("utf-8")).hexdigest()


def _trial_record(
    evaluation: Evaluation,
    *,
    solver: CharacteristicReferenceSolver,
    edges_m: np.ndarray,
    previous: Evaluation | None,
) -> dict[str, Any]:
    row: dict[str, Any] = {
        "evaluation_id": evaluation.evaluation_id, "phase": evaluation.phase,
        "x_guess": evaluation.x_guess, "state_hash_before": evaluation.state_hash_before,
        "state_hash_after": evaluation.state_hash_after,
        "state_unchanged": evaluation.state_hash_before == evaluation.state_hash_after,
        "error_type": evaluation.error_type or "", "error_message": evaluation.error_message or "",
    }
    row["x_guess_hex"], row["x_guess_bits"] = _float_encoding(evaluation.x_guess)
    trial = evaluation.trial
    if trial is None:
        return row
    previous_trial = None if previous is None else previous.trial
    cells = np.asarray(trial.cell_number_m3, dtype=np.float64)
    moments = _moments(edges_m, cells)
    tail_cells = cells[: min(64, cells.size)]
    tail_edges = edges_m[: tail_cells.size + 1]
    tail_moments = _moments(tail_edges, tail_cells)
    partition = solver._cdf_source_partition(trial)
    topology = solver._trace_topology(trial)
    cdf_signature, departure_signature = _partition_signature(partition)
    topology_hash = _topology_signature(topology)
    closure_hex, closure_bits = _float_encoding(trial.matrix_xb)
    residual_hex, residual_bits = _float_encoding(trial.signed_xb_residual)
    population_hash = _hash_arrays({"cell_number_m3": cells})
    if previous_trial is None:
        population_l1 = math.nan
        population_linf = math.nan
        population_residual = math.inf
        cell_measure = math.inf
        changed_faces = 0
        first_changed_face = -1
        last_changed_face = -1
    else:
        prior_cells = np.asarray(previous_trial.cell_number_m3, dtype=np.float64)
        difference = cells - prior_cells
        absolute = np.abs(difference)
        denominator = max(float(np.sum(np.abs(cells))), float(np.sum(np.abs(prior_cells))), 1.0e-300)
        population_l1 = float(np.sum(absolute) / denominator)
        population_linf = float(np.max(absolute))
        population_residual = solver._population_observable_residual(cells, prior_cells)
        cell_measure = solver._cell_measure_relative_residual(cells, prior_cells)
        prior_partition = solver._cdf_source_partition(previous_trial)
        if partition is None or prior_partition is None:
            changed_faces = -1
            first_changed_face = -1
            last_changed_face = -1
        else:
            changed = partition.source_cell_indices != prior_partition.source_cell_indices
            positions = np.flatnonzero(changed)
            changed_faces = int(positions.size)
            first_changed_face = int(positions[0]) if positions.size else -1
            last_changed_face = int(positions[-1]) if positions.size else -1
    row.update({
        "x_closure": float(trial.matrix_xb), "x_closure_hex": closure_hex, "x_closure_bits": closure_bits,
        "F_signed": float(trial.signed_xb_residual), "F_hex": residual_hex, "F_bits": residual_bits,
        "abs_F": abs(float(trial.signed_xb_residual)), "xB_tolerance": float(trial.xb_tolerance),
        "midpoint_xB": float(trial.midpoint_matrix_xb),
        "Q_total_mol_m3": float(trial.inventory.total_mol_m3),
        "Q_beta_mol_m3": float(trial.inventory.beta_resolved_mol_m3),
        "Q_matrix_mol_m3": float(trial.inventory.matrix_mol_m3),
        "inventory_relative_residual": float(trial.inventory.relative_residual),
        "population_L1_change": population_l1, "population_Linf_change": population_linf,
        "population_M0_M3_residual": population_residual, "cell_measure": cell_measure,
        "lower_tail_M0": tail_moments["M0"], "lower_tail_M3": tail_moments["M3"],
        "minimum_departure_radius_m": float(np.min(trial.trace.departure_faces_m)),
        "maximum_departure_radius_m": float(np.max(trial.trace.departure_faces_m)),
        "faces_departure_source_cell_changed": changed_faces,
        "first_changed_face": first_changed_face, "last_changed_face": last_changed_face,
        "cdf_partition_signature": cdf_signature,
        "departure_cell_index_signature_hash": departure_signature,
        "remap_topology_hash": topology_hash,
        "trial_population_hash": population_hash,
        "lower_no_inflow_face_count": int(trial.trace.lower_no_inflow_face_count),
        "upper_no_inflow_face_count": int(trial.trace.upper_no_inflow_face_count),
        "nonfinite_cell_count": int(np.count_nonzero(~np.isfinite(cells))),
        "negative_cell_count": int(np.count_nonzero(cells < 0.0)),
        **moments,
    })
    return row


def _raw_picard(closure_map: ImmutableStepMap, *, edges_m: np.ndarray, iterations: int, phase: str) -> tuple[list[dict[str, Any]], list[Evaluation]]:
    """Run raw, unrelaxed Picard only; no candidate is committed."""

    x_guess = closure_map.x_start
    previous: Evaluation | None = None
    previous_x: float | None = None
    rows: list[dict[str, Any]] = []
    evaluations: list[Evaluation] = []
    for iteration in range(1, iterations + 1):
        evaluation = closure_map.evaluate(x_guess, phase=phase)
        row = _trial_record(evaluation, solver=closure_map.solver, edges_m=edges_m, previous=previous)
        row["iteration"] = iteration
        row["delta_guess"] = math.nan if previous_x is None else x_guess - previous_x
        rows.append(row)
        evaluations.append(evaluation)
        if evaluation.trial is None:
            row["terminal_error"] = True
            break
        row["terminal_error"] = False
        previous = evaluation
        previous_x = x_guess
        x_guess = float(evaluation.trial.matrix_xb)
    for index, row in enumerate(rows):
        if index == 0:
            row["q_F"] = math.nan
            row["q_x"] = math.nan
            continue
        prior_f = float(rows[index - 1].get("abs_F", math.nan))
        current_f = float(row.get("abs_F", math.nan))
        row["q_F"] = current_f / prior_f if prior_f > 0.0 and math.isfinite(current_f) else math.nan
        if index < 2:
            row["q_x"] = math.nan
        else:
            prior_x = abs(float(rows[index - 1].get("delta_guess", math.nan)))
            current_x = abs(float(row.get("delta_guess", math.nan)))
            row["q_x"] = current_x / prior_x if prior_x > 0.0 and math.isfinite(current_x) else math.nan
    return rows, evaluations


def _save_raw_states(path: Path, evaluations: Sequence[Evaluation]) -> dict[str, Any]:
    successful = [item for item in evaluations if item.trial is not None]
    arrays: dict[str, np.ndarray] = {
        "iteration": np.asarray([index + 1 for index in range(len(successful))], dtype=np.int64),
        "x_guess": np.asarray([item.x_guess for item in successful], dtype=np.float64),
        "x_closure": np.asarray([item.trial.matrix_xb for item in successful], dtype=np.float64),
        "F_signed": np.asarray([item.trial.signed_xb_residual for item in successful], dtype=np.float64),
    }
    if successful:
        arrays["cell_number_m3"] = np.stack([item.trial.cell_number_m3 for item in successful])
        arrays["departure_faces_m"] = np.stack([item.trial.trace.departure_faces_m for item in successful])
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return {"file": str(path), "sha256": _sha256_file(path), "successful_iterations": len(successful)}


def _window_contraction(rows: Sequence[Mapping[str, Any]], count: int) -> dict[str, Any]:
    tail = list(rows[-count:])
    q_f = [float(row["q_F"]) for row in tail if math.isfinite(float(row.get("q_F", math.nan)))]
    q_x = [float(row["q_x"]) for row in tail if math.isfinite(float(row.get("q_x", math.nan)))]
    signs = [math.copysign(1.0, float(row["F_signed"])) for row in tail if float(row.get("F_signed", 0.0)) != 0.0]
    sign_flips = sum(left != right for left, right in zip(signs, signs[1:]))
    return {
        "window": len(tail), "q_F_mean": float(np.mean(q_f)) if q_f else math.nan,
        "q_F_median": float(np.median(q_f)) if q_f else math.nan,
        "q_F_min": float(np.min(q_f)) if q_f else math.nan,
        "q_F_max": float(np.max(q_f)) if q_f else math.nan,
        "q_x_mean": float(np.mean(q_x)) if q_x else math.nan,
        "q_x_median": float(np.median(q_x)) if q_x else math.nan,
        "q_x_min": float(np.min(q_x)) if q_x else math.nan,
        "q_x_max": float(np.max(q_x)) if q_x else math.nan,
        "sign_flip_count": sign_flips,
    }


def _cycle_metrics(rows: Sequence[Mapping[str, Any]], evaluations: Sequence[Evaluation]) -> list[dict[str, Any]]:
    metrics: list[dict[str, Any]] = []
    for sample_size in (RAW_PICARD_CAP, 256, 512, RAW_DIAGNOSTIC_CAP):
        if len(evaluations) < sample_size:
            continue
        current_index = sample_size - 1
        for period in range(1, 17):
            if current_index < period:
                continue
            current = evaluations[current_index]
            prior = evaluations[current_index - period]
            if current.trial is None or prior.trial is None:
                continue
            current_cells = np.asarray(current.trial.cell_number_m3)
            prior_cells = np.asarray(prior.trial.cell_number_m3)
            denominator = max(float(np.sum(np.abs(current_cells))), float(np.sum(np.abs(prior_cells))), 1.0e-300)
            population_error = float(np.sum(np.abs(current_cells - prior_cells)) / denominator)
            x_error = abs(current.x_guess - prior.x_guess)
            x_scale = max(abs(current.x_guess), abs(prior.x_guess), np.finfo(np.float64).tiny)
            machine_x = MACHINE_EPS_FACTOR * np.finfo(np.float64).eps * x_scale
            machine_population = MACHINE_EPS_FACTOR * np.finfo(np.float64).eps
            cdf_match = str(rows[current_index].get("cdf_partition_signature", "")) == str(rows[current_index - period].get("cdf_partition_signature", ""))
            topology_match = str(rows[current_index].get("remap_topology_hash", "")) == str(rows[current_index - period].get("remap_topology_hash", ""))
            exact = current.x_guess == prior.x_guess and np.array_equal(current_cells, prior_cells) and cdf_match and topology_match
            machine = x_error <= machine_x and population_error <= machine_population and cdf_match and topology_match
            metrics.append({
                "sample_size": sample_size, "period": period, "x_period_error": x_error,
                "population_period_error": population_error, "cdf_partition_signature_match": cdf_match,
                "remap_topology_signature_match": topology_match, "exact_period": exact,
                "machine_period": machine, "machine_x_tolerance": machine_x,
                "machine_population_tolerance": machine_population,
            })
    return metrics


def _pair_evidence(
    left_row: Mapping[str, Any], left: Evaluation, right_row: Mapping[str, Any], right: Evaluation,
    *, solver: CharacteristicReferenceSolver, pair_kind: str, window: int | None = None,
) -> dict[str, Any]:
    if left.trial is None or right.trial is None:
        return {"pair_kind": pair_kind, "window": window, "eligible": False, "reason": "trial_error"}
    left_f = float(left.trial.signed_xb_residual)
    right_f = float(right.trial.signed_xb_residual)
    strict_sign = math.isfinite(left_f) and math.isfinite(right_f) and left_f * right_f < 0.0
    physical = all(math.isfinite(value) and 0.0 <= value <= 1.0 for value in (left.x_guess, right.x_guess)) and left.x_guess != right.x_guess
    same_cdf = solver._same_cdf_source_partition(left.trial, right.trial)
    same_departure = str(left_row.get("departure_cell_index_signature_hash", "")) == str(right_row.get("departure_cell_index_signature_hash", ""))
    same_topology = solver._same_trace_topology(left.trial, right.trial)
    population_continuity = all(
        math.isfinite(float(value))
        for value in (right_row.get("population_L1_change", math.nan), right_row.get("population_Linf_change", math.nan))
    ) and int(left_row.get("nonfinite_cell_count", -1)) == 0 and int(right_row.get("nonfinite_cell_count", -1)) == 0
    eligible = bool(strict_sign and physical and same_cdf and same_departure and same_topology and population_continuity)
    return {
        "pair_kind": pair_kind, "window": window, "left_iteration": int(left_row["iteration"]),
        "right_iteration": int(right_row["iteration"]), "left_x": left.x_guess,
        "right_x": right.x_guess, "left_F": left_f, "right_F": right_f,
        "left_x_hex": left.x_guess.hex(), "right_x_hex": right.x_guess.hex(),
        "left_F_hex": left_f.hex(), "right_F_hex": right_f.hex(),
        "strict_sign_change": strict_sign, "finite_physical_endpoints": physical,
        "same_cdf_partition": same_cdf, "same_departure_signature": same_departure,
        "same_remap_topology": same_topology,
        "known_branch_boundary_inside_or_at_endpoint": not (same_cdf and same_topology),
        "population_state_change_observed_continuous": population_continuity,
        "eligible": eligible,
    }


def _natural_brackets(rows: Sequence[Mapping[str, Any]], evaluations: Sequence[Evaluation], solver: CharacteristicReferenceSolver) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if len(rows) < RAW_PICARD_CAP or len(evaluations) < RAW_PICARD_CAP:
        raise Step249DiagnosisError("raw Picard did not reach frozen 128-iteration cap")
    final = _pair_evidence(rows[126], evaluations[126], rows[127], evaluations[127], solver=solver, pair_kind="FINAL_127_128")
    all_pairs = [final]
    if not final.get("eligible", False):
        seen: set[tuple[int, int]] = {(127, 128)}
        for window in (8, 16, 32):
            first = RAW_PICARD_CAP - window
            for index in range(first, RAW_PICARD_CAP - 1):
                key = (index + 1, index + 2)
                if key in seen:
                    continue
                seen.add(key)
                all_pairs.append(_pair_evidence(rows[index], evaluations[index], rows[index + 1], evaluations[index + 1], solver=solver, pair_kind="ORDERED_RAW_ADJACENT_AUDIT", window=window))
    return final, all_pairs


def _local_map(closure_map: ImmutableStepMap, *, edges_m: np.ndarray, bracket: Mapping[str, Any]) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not bool(bracket.get("eligible", False)):
        return {"status": "NOT_RUN_NO_ELIGIBLE_FINAL_NATURAL_BRACKET", "point_count": 0}, []
    low = float(min(float(bracket["left_x"]), float(bracket["right_x"])))
    high = float(max(float(bracket["left_x"]), float(bracket["right_x"])))
    points = np.linspace(low, high, LOCAL_MAP_POINTS, dtype=np.float64)
    rows: list[dict[str, Any]] = []
    evaluations: list[Evaluation] = []
    previous: Evaluation | None = None
    for index, value in enumerate(points):
        evaluation = closure_map.evaluate(float(value), phase="step249_local_map", reuse=True)
        row = _trial_record(evaluation, solver=closure_map.solver, edges_m=edges_m, previous=previous)
        row["local_index"] = index
        row["inside_natural_bracket"] = low <= float(value) <= high
        rows.append(row)
        evaluations.append(evaluation)
        previous = evaluation
    reference = evaluations[0]
    same_cdf = all(item.trial is not None and reference.trial is not None and closure_map.solver._same_cdf_source_partition(reference.trial, item.trial) for item in evaluations)
    same_topology = all(item.trial is not None and reference.trial is not None and closure_map.solver._same_trace_topology(reference.trial, item.trial) for item in evaluations)
    finite = all(item.trial is not None and math.isfinite(float(item.trial.signed_xb_residual)) for item in evaluations)
    signs = [math.copysign(1.0, float(item.trial.signed_xb_residual)) if item.trial is not None and item.trial.signed_xb_residual != 0.0 else 0.0 for item in evaluations]
    compact = [sign for sign in signs if sign != 0.0]
    sign_crossings = sum(left != right for left, right in zip(compact, compact[1:]))
    exact_zero_points = sum(sign == 0.0 for sign in signs)
    crossing_count = sign_crossings if exact_zero_points == 0 else max(1, sign_crossings)
    all_inside = all(bool(row["inside_natural_bracket"]) for row in rows)
    if not same_cdf or not same_topology:
        status = "LOCAL_BRACKET_TOPOLOGY_SWITCH"
    elif not finite:
        status = "LOCAL_BRACKET_DISCONTINUOUS"
    elif crossing_count > 1:
        status = "LOCAL_BRACKET_MULTIPLE_CROSSINGS"
    elif crossing_count == 1:
        status = "LOCAL_BRACKET_CONTINUOUS_SINGLE_CROSSING"
    else:
        status = "LOCAL_BRACKET_DISCONTINUOUS"
    return {
        "status": status, "point_count": LOCAL_MAP_POINTS, "low": low, "high": high,
        "all_points_inside_natural_bracket": all_inside, "all_points_same_cdf_partition": same_cdf,
        "all_points_same_remap_topology": same_topology, "all_residuals_finite": finite,
        "observed_sign_crossing_count": crossing_count, "exact_zero_point_count": exact_zero_points,
        "observed_sample_continuity": bool(same_cdf and same_topology and finite),
    }, rows


def _asymptotic_classification(rows: Sequence[Mapping[str, Any]], cycle: Sequence[Mapping[str, Any]]) -> tuple[str, list[dict[str, Any]]]:
    windows = [_window_contraction(rows, size) for size in (16, 32, 64, 128)]
    largest_sample = max((int(row["sample_size"]) for row in cycle), default=0)
    periodic = any(
        bool(row["exact_period"]) or bool(row["machine_period"])
        for row in cycle
        if int(row["sample_size"]) == largest_sample
    )
    topology_switches = sum(
        str(left.get("remap_topology_hash", "")) != str(right.get("remap_topology_hash", ""))
        for left, right in zip(rows[-128:], rows[-127:])
    )
    tail = windows[-1]
    if periodic:
        classification = "PERIODIC_ORBIT"
    elif topology_switches:
        classification = "TOPOLOGY_SWITCHING"
    elif math.isfinite(float(tail["q_F_median"])) and float(tail["q_F_median"]) >= 1.0:
        classification = "NONCONTRACTIVE"
    elif int(tail["sign_flip_count"]) > 0:
        classification = "OSCILLATORY_APPROACH"
    elif math.isfinite(float(tail["q_F_median"])) and float(tail["q_F_median"]) < 1.0:
        classification = "MONOTONE_SLOW_CONTRACTION"
    else:
        classification = "OTHER_EXPLICIT"
    for row in windows:
        row["topology_switch_count_in_last128"] = topology_switches
    return classification, windows


def _failure_reproduction(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    if len(rows) < RAW_PICARD_CAP:
        return {"status": "FAIL_STEP249_FAILURE_REPRODUCTION", "reason": "fewer than 128 raw iterations"}
    row = rows[RAW_PICARD_CAP - 1]
    actual = {
        "xb_residual": float(row.get("abs_F", math.nan)),
        "population_residual": float(row.get("population_M0_M3_residual", math.nan)),
        "cell_measure": float(row.get("cell_measure", math.nan)),
    }
    # This is a diagnostic reproduction band for values previously published
    # only to four significant figures.  It is not, and cannot alter, any
    # frozen solver acceptance tolerance.
    checks = {key: math.isclose(actual[key], expected, rel_tol=5.0e-3, abs_tol=0.0) for key, expected in EXPECTED_FAILURE.items()}
    return {
        "status": "PASS_STEP249_FAILURE_REPRODUCTION" if all(checks.values()) else "FAIL_STEP249_FAILURE_REPRODUCTION",
        "iteration": RAW_PICARD_CAP, "expected_reporting_values": EXPECTED_FAILURE,
        "actual": actual, "reporting_band_relative": 5.0e-3, "checks": checks,
    }


def _step245_probe(context: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """Read only the step-245 raw map for structural comparison, never acceptance."""

    solver = CharacteristicReferenceSolver(SolverConfig.from_mapping(context.mapping))
    for expected_step in range(1, FORMAL_LAST_ACCEPTED_STEP + 1):
        diagnostic = solver.advance_one(maximum_dt_s=DT_S)
        if diagnostic.step != expected_step:
            raise Step249DiagnosisError("step-245 comparison could not reproduce step 244")
    closure_map = ImmutableStepMap(
        solver, old_cell_number_m3=solver._beta_cell_numbers(), dt_s=DT_S, state_label="state244_step245_comparison"
    )
    rows, evaluations = _raw_picard(closure_map, edges_m=context.edges_m, iterations=RAW_PICARD_CAP, phase="step245_raw_comparison")
    final, _all = _natural_brackets(rows, evaluations, solver)
    cycle = _cycle_metrics(rows, evaluations)
    classification, windows = _asymptotic_classification(rows, cycle)
    local, _local_rows = _local_map(closure_map, edges_m=context.edges_m, bracket=final)
    return {
        "raw_final_residual": float(rows[-1].get("abs_F", math.nan)), "final_pair": final,
        "asymptotic_classification": classification, "contraction": windows,
        "local_map": local, "state_unchanged": all(bool(row.get("state_unchanged")) for row in rows),
    }, rows


def _comparison(step245: Mapping[str, Any], step249: Mapping[str, Any]) -> tuple[str, list[dict[str, Any]]]:
    def fields(label: str, payload: Mapping[str, Any]) -> dict[str, Any]:
        pair = payload["final_pair"]
        local = payload["local_map"]
        tail = payload["contraction"][-1] if payload["contraction"] else {}
        return {
            "step": label, "raw_final_residual": payload["raw_final_residual"],
            "asymptotic_classification": payload["asymptotic_classification"],
            "strict_sign_change": pair.get("strict_sign_change"), "same_cdf_partition": pair.get("same_cdf_partition"),
            "same_remap_topology": pair.get("same_remap_topology"), "bracket_width": abs(float(pair.get("right_x", math.nan)) - float(pair.get("left_x", math.nan))) if "right_x" in pair else math.nan,
            "local_map_status": local.get("status"), "local_crossing_count": local.get("observed_sign_crossing_count"),
            "q_F_median_last128": tail.get("q_F_median"), "q_x_median_last128": tail.get("q_x_median"),
        }
    rows = [fields("245", step245), fields("249", step249)]
    same = (
        step245["final_pair"].get("eligible") is True
        and step249["final_pair"].get("eligible") is True
        and step245["local_map"].get("status") == "LOCAL_BRACKET_CONTINUOUS_SINGLE_CROSSING"
        and step249["local_map"].get("status") == "LOCAL_BRACKET_CONTINUOUS_SINGLE_CROSSING"
        and step245["asymptotic_classification"] in {"NONCONTRACTIVE", "OSCILLATORY_APPROACH"}
        and step249["asymptotic_classification"] in {"NONCONTRACTIVE", "OSCILLATORY_APPROACH"}
    )
    return ("SAME_FAILURE_CLASS" if same else "DISTINCT_FAILURE_CLASS"), rows


def _stage_one_status(
    *, reproduction: Mapping[str, Any], step249: Mapping[str, Any], failure_class: str,
    cycle: Sequence[Mapping[str, Any]],
) -> tuple[str, dict[str, Any]]:
    periodic = [row for row in cycle if bool(row["exact_period"]) or bool(row["machine_period"])]
    pair = step249["final_pair"]
    local = step249["local_map"]
    if reproduction["status"] != "PASS_STEP249_FAILURE_REPRODUCTION":
        return "FAIL_STEP249_FAILURE_REPRODUCTION", {"eligible": False, "reason": "raw failure mismatch"}
    if periodic:
        return "STEP249_PERIODIC_ORBIT", {"eligible": False, "reason": "P1_P16 periodic evidence", "periods": periodic}
    if not pair.get("strict_sign_change", False):
        status = (
            "STEP249_NONCONTRACTIVE_NO_BRACKET"
            if step249.get("asymptotic_classification") == "NONCONTRACTIVE"
            else "STEP249_NO_NATURAL_ADJACENT_BRACKET"
        )
        return status, {"eligible": False, "reason": "final raw pair lacks strict sign change"}
    if not pair.get("same_cdf_partition", False) or not pair.get("same_remap_topology", False):
        return "STEP249_TOPOLOGY_SWITCHING", {"eligible": False, "reason": "final pair crosses partition/topology"}
    if local.get("status") == "LOCAL_BRACKET_MULTIPLE_CROSSINGS":
        return "STEP249_LOCAL_MULTIPLE_ROOTS", {"eligible": False, "reason": "local sampled multiple crossings"}
    if local.get("status") != "LOCAL_BRACKET_CONTINUOUS_SINGLE_CROSSING":
        return "STEP249_NATURAL_BRACKET_DIFFERENT_CLASS", {"eligible": False, "reason": str(local.get("status"))}
    if failure_class != "SAME_FAILURE_CLASS":
        return "STEP249_NATURAL_BRACKET_DIFFERENT_CLASS", {"eligible": False, "reason": "structural comparison differs from step245"}
    return "STEP249_SAME_CLASS_NATURAL_BRACKET_DIAGNOSIS_ONLY", {
        "eligible": True,
        "rule": "NATURAL_ADJACENT_BRACKET_FALLBACK_V1",
        "reason": "final natural same-topology bracket with sampled single crossing matches step245 class",
    }


def _blocked(value: str) -> dict[str, Any]:
    return {"status": "BLOCKED_NOT_RUN_PHASE1_READ_ONLY_DIAGNOSIS", "reason": value}


def _write_reports(
    report_root: Path, *, replay: Mapping[str, Any], reproduction: Mapping[str, Any], raw: Mapping[str, Any],
    cycles: Mapping[str, Any], brackets: Mapping[str, Any], local_map: Mapping[str, Any], comparison: Mapping[str, Any],
    gate: Mapping[str, Any], final: Mapping[str, Any], command: str,
) -> None:
    documents: tuple[tuple[str, str, Mapping[str, Any] | str], ...] = (
        ("00_step248_replay.md", "Step-248 deterministic replay", replay),
        ("01_step249_failure_reproduction.md", "Step-249 failure reproduction", reproduction),
        ("02_step249_raw_picard.md", "Step-249 raw Picard", raw),
        ("03_step249_cycle_analysis.md", "Step-249 P1–P16 cycle analysis", cycles),
        ("04_step249_natural_bracket.md", "Step-249 natural adjacent bracket", brackets),
        ("05_step249_local_scalar_map.md", "Step-249 local scalar map", local_map),
        ("06_step245_step249_comparison.md", "Step-245 / step-249 structural comparison", comparison),
        ("07_generalization_gate.md", "Natural-adjacent-bracket generalization gate", gate),
        ("08_generic_safeguard_implementation.md", "Generic safeguard implementation", _blocked("implementation is forbidden until this independent read-only gate is reviewed")),
        ("09_step249_acceptance.md", "Step-249 tentative acceptance", _blocked("no step249 candidate was accepted in the diagnostic phase")),
        ("10_repeatability_restart.md", "Repeatability and restart", _blocked("requires a separately authorized generic implementation")),
        ("11_local_240_260_regression.md", "Local 240–260 regression", _blocked("requires repeatability/restart gate first")),
        ("12_closure_frequency_audit.md", "Closure frequency audit", _blocked("no generic fallback was executed")),
        ("13_final_acceptance_report.md", "Final acceptance report", final),
        ("14_reproduction_commands.md", "Reproduction commands", "```bash\n" + command + "\n```"),
    )
    for filename, title, content in documents:
        _write_markdown(report_root / filename, title, content)


def _workflow(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = arguments.output_root.resolve()
    report_root = arguments.report_root.resolve()
    if output_root.exists() or report_root.exists():
        raise Step249DiagnosisError("output and report roots must not already exist")
    if float(arguments.dt_s) != DT_S:
        raise Step249DiagnosisError(f"diagnosis must retain frozen dt_s={DT_S}")
    started = time.monotonic()
    source = _source_identity()
    context = build_frozen_canonical_context()
    if context.contract_hash != FROZEN_CONTRACT_HASH:
        raise Step249DiagnosisError("frozen canonical context hash differs")
    output_root.mkdir(parents=True, exist_ok=False)
    report_root.mkdir(parents=True, exist_ok=False)
    final: dict[str, Any]
    try:
        replay_runs: list[dict[str, Any]] = []
        first_solver: CharacteristicReferenceSolver | None = None
        first_formal: list[dict[str, Any]] = []
        first_history: list[dict[str, Any]] = []
        for repeat in range(1, REPLAY_COUNT + 1):
            solver, formal_rows, history = _replay_through_step248(context)
            record = {
                "repeat": repeat, "state_hash": _state_hash(solver), "state_arrays": solver.state_arrays(),
                "xB": float(solver.matrix_xb), "history": history,
                "moments": _moments(context.edges_m, solver._beta_cell_numbers()),
            }
            replay_runs.append(record)
            if first_solver is None:
                first_solver, first_formal, first_history = solver, formal_rows, history
        assert first_solver is not None
        formal_replay = _compare_formal_trace(first_formal, arguments.formal_trace_csv)
        states_equal = _arrays_equal(replay_runs[0]["state_arrays"], replay_runs[1]["state_arrays"])
        history_equal = _rows_equal_with_nan(replay_runs[0]["history"], replay_runs[1]["history"])
        expected_modes = ["SAFEGUARDED_SCALAR_ROOT_V1", "DIRECT", "DIRECT", "DIRECT"]
        observed_modes = [str(row["fixed_point_convergence_mode"]) for row in first_history]
        state248 = _save_state248(output_root / "state_248.npz", solver=first_solver, context=context, history=first_history)
        checkpoint248 = output_root / "checkpoint_step248.npz"
        first_solver.save_checkpoint(checkpoint248)
        replay = {
            "status": "PASS_STEP248_DETERMINISTIC_REPLAY" if states_equal and history_equal and observed_modes == expected_modes and formal_replay["status"] == "PASS_EXACT_FORMAL_CR1_TRACE_1_TO_244_MATCH" else "FAIL_STEP248_DETERMINISTIC_REPLAY",
            "replay_count": REPLAY_COUNT, "formal_replay": formal_replay,
            "expected_modes_245_248": expected_modes, "observed_modes_245_248": observed_modes,
            "state_arrays_equal_bitwise": states_equal, "closure_history_equal": history_equal,
            "run_hashes": [run["state_hash"] for run in replay_runs], "state248": state248,
            "checkpoint_step248": {"file": str(checkpoint248), "sha256": _sha256_file(checkpoint248)},
        }
        if replay["status"] != "PASS_STEP248_DETERMINISTIC_REPLAY":
            raise Step249DiagnosisError("deterministic frozen replay through step248 failed")

        step249_map = ImmutableStepMap(first_solver, old_cell_number_m3=first_solver._beta_cell_numbers(), dt_s=DT_S, state_label="state248")
        raw_rows, raw_evaluations = _raw_picard(step249_map, edges_m=context.edges_m, iterations=RAW_DIAGNOSTIC_CAP, phase="step249_raw_picard")
        _write_csv(output_root / "step249_raw_picard.csv", raw_rows, fallback_fields=("iteration", "x_guess", "F_signed"))
        raw_states = _save_raw_states(output_root / "step249_raw_picard_states.npz", raw_evaluations)
        _write_csv(output_root / "step249_topology_signatures.csv", raw_rows, fallback_fields=("iteration", "cdf_partition_signature", "remap_topology_hash"))
        reproduction = _failure_reproduction(raw_rows)
        if reproduction["status"] != "PASS_STEP249_FAILURE_REPRODUCTION":
            raise Step249DiagnosisError("frozen step249 raw failure did not reproduce")
        cycle_rows = _cycle_metrics(raw_rows, raw_evaluations)
        _write_csv(output_root / "step249_cycle_metrics.csv", cycle_rows, fallback_fields=("sample_size", "period", "x_period_error"))
        asymptotic, contraction = _asymptotic_classification(raw_rows, cycle_rows)
        final_pair, pair_rows = _natural_brackets(raw_rows, raw_evaluations, first_solver)
        _write_csv(output_root / "step249_natural_brackets.csv", pair_rows, fallback_fields=("pair_kind", "left_iteration", "right_iteration", "eligible"))
        local_summary, local_rows = _local_map(step249_map, edges_m=context.edges_m, bracket=final_pair)
        _write_csv(output_root / "step249_local_scalar_map.csv", local_rows, fallback_fields=("local_index", "x_guess", "F_signed"))
        step249_summary = {
            "raw_final_residual": float(raw_rows[RAW_PICARD_CAP - 1]["abs_F"]), "final_pair": final_pair,
            "asymptotic_classification": asymptotic, "contraction": contraction, "local_map": local_summary,
        }
        step245_summary, _step245_rows = _step245_probe(context)
        failure_class, comparison_rows = _comparison(step245_summary, step249_summary)
        _write_csv(output_root / "step245_vs_step249_closure_comparison.csv", comparison_rows, fallback_fields=("step", "asymptotic_classification"))
        status, gate = _stage_one_status(reproduction=reproduction, step249=step249_summary, failure_class=failure_class, cycle=cycle_rows)
        _write_json(output_root / "generalization_gate.json", gate)
        _write_json(output_root / "step249_root_solution.json", {
            "status": "NOT_RUN_READ_ONLY_DIAGNOSIS_ONLY", "reason": "no root search or candidate acceptance is permitted in phase one",
            "natural_final_pair": final_pair,
        })
        _write_csv(output_root / "fallback_repeatability.csv", [], fallback_fields=("step", "repeat_index", "status"))
        _write_csv(output_root / "restart_comparison.csv", [], fallback_fields=("restart_case", "status"))
        _write_csv(output_root / "local_240_260_trace.csv", [], fallback_fields=("step", "status"))
        _write_csv(output_root / "fallback_frequency.csv", [], fallback_fields=("step", "status"))
        final = {
            "STATUS": status, "BRANCH": source["git_branch"], "COMMIT": source["git_commit"],
            "SOURCE_CLEAN": True, "TESTS": "NOT_RUN_IN_CLUSTER_READ_ONLY_DIAGNOSIS_STAGE",
            "STEP248_REPLAY": replay["status"], "STEP248_STATE_HASH": state248["state_array_hash"],
            "STEP249_FAILURE_REPRODUCED": reproduction["status"],
            "STEP249_PICARD_128_RESIDUAL": raw_rows[127]["abs_F"],
            "STEP249_PICARD_256_RESIDUAL": raw_rows[255]["abs_F"],
            "STEP249_PICARD_512_RESIDUAL": raw_rows[511]["abs_F"],
            "STEP249_ASYMPTOTIC_CONTRACTION": {"classification": asymptotic, "windows": contraction},
            "STEP249_PERIOD": [row for row in cycle_rows if bool(row["exact_period"]) or bool(row["machine_period"])],
            "STEP249_TOPOLOGY_SWITCHING": not bool(final_pair.get("same_remap_topology", False)),
            "NATURAL_ADJACENT_BRACKET": "PASS" if final_pair.get("eligible") else "FAIL",
            "BRACKET_ITERATIONS": [final_pair.get("left_iteration"), final_pair.get("right_iteration")],
            "BRACKET_X_LOW": final_pair.get("left_x"), "BRACKET_X_HIGH": final_pair.get("right_x"),
            "BRACKET_F_LOW": final_pair.get("left_F"), "BRACKET_F_HIGH": final_pair.get("right_F"),
            "SAME_CDF_PARTITION": final_pair.get("same_cdf_partition"), "SAME_REMAP_TOPOLOGY": final_pair.get("same_remap_topology"),
            "LOCAL_MAP_CONTINUITY": local_summary.get("observed_sample_continuity"), "LOCAL_CROSSING_COUNT": local_summary.get("observed_sign_crossing_count"),
            "STEP245_VS_STEP249_CLASS": failure_class, "GENERALIZATION_GATE": gate,
            "GENERIC_FALLBACK_IMPLEMENTED": False, "FALLBACK_RULE": "NOT_IMPLEMENTED_PHASE1",
            "GLOBAL_SCAN_USED": False, "HISTORICAL_BRACKET_USED": False, "TOLERANCE_CHANGED": False,
            "STEP249_ACCEPTED": False, "STEP249_CLOSURE_MODE": "NOT_RUN_READ_ONLY_DIAGNOSIS",
            "STEP249_ROOT": "NOT_ATTEMPTED", "STEP249_ROOT_RESIDUAL": "NOT_ATTEMPTED", "STEP249_INVENTORY_RESIDUAL": "NOT_ATTEMPTED",
            "STEP245_REPEATABILITY": "NOT_RUN", "STEP249_REPEATABILITY": "NOT_RUN", "RESTART_GATE": "NOT_RUN",
            "LOCAL_240_260_GATE": "NOT_RUN", "FALLBACK_STEPS_240_260": [], "FALLBACK_FREQUENCY": "NOT_RUN",
            "TIME_REFERENCE_V2": "NOT_ASSIGNED", "IMPLICIT_TIME_INACCURACY_PROVEN": False,
            "CR1_TIME_LADDER_RUN": False, "COHORT_PARITY_RUN": False,
            "PF_SOURCE_MODIFIED": False, "CUDA_RERUN": False, "PHYSICAL_RETUNING": False, "GP_RELEASE_RUN": False,
            "TOP_5_FINDINGS": [
                "Step248 is a repeated, immutable accepted initial state for the step249 map diagnosis.",
                "Step249 was evaluated only through T(x)-x from state248; no candidate was committed.",
                "Raw Picard was extended only as read-only diagnostic evidence through iteration 1024.",
                "Only chronological adjacent raw pairs were audited; no global scan or historical bracket was used.",
                "Global root count remains unresolved without an interval or analytic enclosure.",
            ],
            "P0_BLOCKERS": ["No time-reference work is allowed before a separate local closure regression passes."],
            "NEXT_ACTION": "Review the frozen diagnostic gate; implement a generic fallback only if GENERALIZATION_GATE.eligible is true.",
            "LOCAL_GP_RELEASE_AUTHORIZED": False,
            "KEY_REPORTS": [str(report_root / "13_final_acceptance_report.md")],
            "state248": state248, "replay": replay, "reproduction": reproduction,
            "raw_states": raw_states, "step249": step249_summary, "step245": step245_summary,
            "comparison": comparison_rows, "source": source, "runtime": _runtime_provenance(),
            "validation_contract_hash": context.contract_hash, "canonical_snapshot": frozen_canonical_snapshot_provenance(),
            "wall_seconds": time.monotonic() - started,
        }
    except Exception as error:
        final = {
            "STATUS": "FAIL_STEP248_DETERMINISTIC_REPLAY" if "step248" in str(error).lower() else "FAIL_STEP249_FAILURE_REPRODUCTION",
            "reason": f"{type(error).__name__}: {error}", "BRANCH": source.get("git_branch"),
            "COMMIT": source.get("git_commit"), "SOURCE_CLEAN": True,
            "TIME_REFERENCE_V2": "NOT_ASSIGNED", "IMPLICIT_TIME_INACCURACY_PROVEN": False,
            "PF_SOURCE_MODIFIED": False, "CUDA_RERUN": False, "PHYSICAL_RETUNING": False, "GP_RELEASE_RUN": False,
            "GLOBAL_SCAN_USED": False, "HISTORICAL_BRACKET_USED": False, "TOLERANCE_CHANGED": False,
            "wall_seconds": time.monotonic() - started,
        }
    _write_json(output_root / "analysis_provenance.json", final)
    _write_reports(
        report_root, replay=final.get("replay", _blocked("replay failed")), reproduction=final.get("reproduction", _blocked("reproduction failed")),
        raw={"raw_states": final.get("raw_states", _blocked("raw trace unavailable")), "step249": final.get("step249", {})},
        cycles={"status": final.get("STATUS"), "periods": final.get("STEP249_PERIOD", [])},
        brackets={"natural_final_pair": final.get("step249", {}).get("final_pair", {})},
        local_map=final.get("step249", {}).get("local_map", _blocked("local map unavailable")),
        comparison={"class": final.get("STEP245_VS_STEP249_CLASS", "NOT_AVAILABLE"), "rows": final.get("comparison", [])},
        gate=final.get("GENERALIZATION_GATE", _blocked("gate unavailable")), final=final,
        command=" ".join(sys.argv),
    )
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("diagnose",))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / TASK_NAME)
    parser.add_argument("--report-root", type=Path, default=ROOT / "reports" / TASK_NAME)
    parser.add_argument("--formal-trace-csv", type=Path, default=FORMAL_TRACE)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    arguments = parser.parse_args()
    try:
        result = _workflow(arguments)
    except Exception as error:
        print(json.dumps({"STATUS": "FAIL_STEP249_FAILURE_REPRODUCTION", "reason": f"{type(error).__name__}: {error}"}, indent=2))
        return 2
    print(json.dumps(_json_safe(result), indent=2, sort_keys=True))
    return 0 if result["STATUS"] == "STEP249_SAME_CLASS_NATURAL_BRACKET_DIAGNOSIS_ONLY" else 2


if __name__ == "__main__":
    raise SystemExit(main())
