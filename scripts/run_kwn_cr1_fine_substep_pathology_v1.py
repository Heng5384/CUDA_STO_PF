#!/usr/bin/env python3
"""Diagnostic-only audit of the CR1 fine-substep closure failures at step 245.

The runner starts from the frozen accepted step-244 checkpoint.  It recreates
the fixed m=2,4,8,16,32,64 Phase-A paths and then evaluates raw nonlinear maps
from immutable pre-failure states.  It never accepts a scan point, changes a
tolerance, advances m=128, invokes a generic root selector, or changes PF/CUDA
or physical inputs.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import csv
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
import platform
import re
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.characteristic_dt_continuation import accepted_state_hash  # noqa: E402
from kwn_mvp.characteristic_pathology import (  # noqa: E402
    ClosureMapEvaluation,
    DiagnosticPrestate,
    capture_prestate,
    common_time_comparisons,
    evaluate_closure_map,
    evaluation_from_recorded_trial,
    evaluation_row,
    precision_reduction_shadow,
    raw_picard_trace,
    repeatability_audit,
    scalar_map_scan,
    state_summary,
)
from kwn_mvp.characteristic_reference import (  # noqa: E402
    CharacteristicReferenceError,
    CharacteristicReferenceSolver,
    CharacteristicStepDiagnostics,
)
from kwn_mvp.solver import RadiusGridOverflowError, SolverConfig  # noqa: E402
from scripts.frozen_canonical_smooth_population_v1 import (  # noqa: E402
    build_frozen_canonical_context,
    frozen_canonical_snapshot_provenance,
)


TASK_NAME = "kwn_cr1_fine_substep_pathology_v1"
REQUIRED_BRANCH = "codex/kwn-cr1-fine-substep-pathology-v1"
BASE_COMMIT = "1ab2c6130e7c5edfe5b0c4439a2844b7cc9878b7"
V1_CLOSURE_COMMIT = "2ee679437421c750ccf2f2173d208df73bf28475"
FROZEN_START_COMMIT = "9269e07a2d0fafbf35be950b858373fb71b54e4b"
DT0_S = 0.015625
RESTART_STEP = 244
RESTART_STATE_HASH = "51c243b61f4278fdf8f5bae9afe50672f79a6e0be57260b9c7cc0fdb46dca4c2"
EXPECTED_FORMAL_TRACE_SHA256 = "e63e546ccd31d93d105b2bf40e3b5abb0fee9e9015f06eace45610a5654dad65"
EXPECTED_FORMAL_PHASE_A_SHA256 = "a91d1940fe471b7f2ebfde5402e177b06fa7b55056bcacdce93519c668755703"
EXPECTED_FIXTURE_INPUT_SET_SHA256 = "c97de66ec1a89c00f4badd2d3546e9f744ae9dfd10fdadc10cb9941de10da9ae"
EXPECTED_RESTART_CHECKPOINT_SHA256 = "c0bc8dd946769550d780763d5a73446b929bac15c5bf7a04895a6288a2314770"
SUBSTEP_MULTIPLICITIES = (2, 4, 8, 16, 32, 64)
REPORT_TITLES = {
    "00_phase_a_reproduction.md": "Phase-A reproduction",
    "01_common_time_state_alignment.md": "Common physical-time state alignment",
    "02_m32_failure_replay.md": "m32 failure replay",
    "03_m64_failure_replay.md": "m64 failure replay",
    "04_failure_closure_sequences.md": "Failure closure sequences",
    "05_scalar_map_topology.md": "Scalar-map and topology audit",
    "06_m32_m64_comparison.md": "m32 versus m64 closure-map comparison",
    "07_same_state_different_dt.md": "Same-prestate different-dt audit",
    "08_precision_conditioning.md": "Precision and conditioning audit",
    "09_cr1_remap_nonsmoothness.md": "CR1 remap non-smoothness audit",
    "10_root_cause_classification.md": "Root-cause classification",
    "11_next_method_decision.md": "Next-method decision",
    "12_final_acceptance_report.md": "Final diagnostic acceptance report",
    "13_reproduction_commands.md": "Reproduction commands",
}


class PathologyWorkflowError(RuntimeError):
    """A fail-closed orchestration error, never a request for a root repair."""


class RecordingCharacteristicReferenceSolver(CharacteristicReferenceSolver):
    """CR1 solver that retains only uncommitted trial telemetry for an audit."""

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._recorded_trials: list[Any] = []
        super().__init__(*args, **kwargs)

    def _evaluate_closure_trial(self, **kwargs: Any) -> Any:
        trial = super()._evaluate_closure_trial(**kwargs)
        self._recorded_trials.append(trial)
        return trial

    def trial_cursor(self) -> int:
        return len(self._recorded_trials)

    def trials_since(self, cursor: int) -> tuple[Any, ...]:
        return tuple(self._recorded_trials[int(cursor) :])


@dataclass(frozen=True)
class FixedRun:
    m: int
    dt_s: float
    status: str
    accepted_states: tuple[dict[str, Any], ...]
    accepted_rows: tuple[dict[str, Any], ...]
    failure: Mapping[str, Any] | None
    failure_closure_trace: tuple[dict[str, Any], ...]
    solver_at_failure_prestate: RecordingCharacteristicReferenceSolver | None
    prestate_at_failure: DiagnosticPrestate | None


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items() if not str(key).startswith("_")}
    if isinstance(value, (tuple, list)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return {"array_shape": list(value.shape), "array_dtype": str(value.dtype)}
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, title: str, payload: Mapping[str, Any] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        body = payload.rstrip()
    else:
        body = "```json\n" + json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True) + "\n```"
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
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore", lineterminator="\n")
        writer.writeheader()
        writer.writerows(materialized)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *arguments], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _source_identity(*, require_clean: bool) -> dict[str, Any]:
    source = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status": _git("status", "--short"),
        "base_commit": BASE_COMMIT,
        "v1_closure_commit": V1_CLOSURE_COMMIT,
        "frozen_start_commit": FROZEN_START_COMMIT,
    }
    for label, commit in (
        ("base_is_ancestor", BASE_COMMIT),
        ("v1_closure_is_ancestor", V1_CLOSURE_COMMIT),
        ("frozen_start_is_ancestor", FROZEN_START_COMMIT),
    ):
        source[label] = subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
            stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=False,
        ).returncode == 0
    if source["git_branch"] != REQUIRED_BRANCH:
        raise PathologyWorkflowError(f"task requires branch {REQUIRED_BRANCH}")
    if require_clean and source["git_status"]:
        raise PathologyWorkflowError("formal pathology run requires a clean source tree")
    if not all(bool(source[name]) for name in (
        "base_is_ancestor", "v1_closure_is_ancestor", "frozen_start_is_ancestor"
    )):
        raise PathologyWorkflowError("source no longer descends from the frozen diagnostic ancestors")
    return source


def _runtime_provenance() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }


def _parse_failure_message(message: str) -> dict[str, Any]:
    fields: dict[str, Any] = {"error_message": message}
    for label, pattern in (
        ("xB_residual", r"(?:xb_residual|scalar_residual)=([0-9.eE+-]+)"),
        ("xB_tolerance", r"(?:xb_tolerance|scalar_tolerance|verification_tolerance)=([0-9.eE+-]+)"),
        ("verification_xB_residual", r"verification_residual=([0-9.eE+-]+)"),
        ("population_residual", r"population_residual=([0-9.eE+-]+)"),
        ("cell_measure", r"cell_measure_residual=([0-9.eE+-]+)"),
    ):
        match = re.search(pattern, message)
        if match is not None:
            fields[label] = float(match.group(1))
    fields["closure_path"] = (
        "ALLOWED_P4_EXACT_CYCLE_PATH" if "period-4" in message else "ORDINARY_PICARD"
    )
    return fields


def _recorded_trial_key(evaluation: ClosureMapEvaluation) -> tuple[str, ...]:
    return (
        evaluation.x_trial.hex(),
        evaluation.x_closure.hex(),
        evaluation.population_hash,
        evaluation.cdf_signature,
        evaluation.remap_topology_signature,
    )


def _first_exact_p4_window(evaluations: Sequence[ClosureMapEvaluation]) -> tuple[int, int] | None:
    """Locate the first exact raw-P4 observation in an original trial record."""

    keys: list[tuple[str, ...]] = []
    for stop, evaluation in enumerate(evaluations, start=1):
        keys.append(_recorded_trial_key(evaluation))
        if stop >= 8 and keys[-4:] == keys[-8:-4]:
            return (stop - 7, stop)
    return None


def _recorded_failure_trace(
    solver: RecordingCharacteristicReferenceSolver,
    *,
    trials: Sequence[Any],
    prestate: DiagnosticPrestate,
    closure_path: str,
) -> tuple[tuple[dict[str, Any], ...], dict[str, Any]]:
    """Serialize the exact rejected-path trials without re-running a closure.

    For a rejected P4 bracket, the production path never evaluates a map
    successor when the scalar trial misses the frozen scalar gate.  That fact
    is represented explicitly rather than manufacturing a verification
    residual after the exception.
    """

    evaluations = tuple(
        evaluation_from_recorded_trial(
            solver, trial=trial, old_cell_number_m3=prestate.old_cell_number_m3
        )
        for trial in trials
    )
    if not evaluations:
        raise PathologyWorkflowError("rejected closure produced no recorded trial telemetry")
    p4_window = _first_exact_p4_window(evaluations) if closure_path == "ALLOWED_P4_EXACT_CYCLE_PATH" else None
    raw_stop = len(evaluations) if p4_window is None else p4_window[1]
    rows: list[dict[str, Any]] = []
    previous: ClosureMapEvaluation | None = None
    for index, evaluation in enumerate(evaluations, start=1):
        row = evaluation_row(evaluation, iteration=index, previous=previous, label="original_rejected_closure")
        row["original_closure_iteration"] = index
        if p4_window is not None and p4_window[0] <= index <= p4_window[1]:
            row["record_type"] = "ORIGINAL_RAW_PICARD_P4_CYCLE_OBSERVATION"
            row["p4_cycle_phase"] = 1 + (index - p4_window[0]) % 4
        elif index <= raw_stop:
            row["record_type"] = "ORIGINAL_RAW_PICARD"
            row["p4_cycle_phase"] = ""
        else:
            row["record_type"] = "ORIGINAL_P4_BRACKET_TRIAL"
            row["p4_cycle_phase"] = ""
        rows.append(row)
        previous = evaluation
    original_verification_attempted = any(
        abs(left.signed_f) <= left.x_tolerance and right.x_trial == left.x_closure
        for left, right in zip(evaluations, evaluations[1:])
    )
    final_row = rows[-1]
    return tuple(rows), {
        "recorded_failure_trial_count": len(rows),
        "original_raw_picard_trial_count": raw_stop,
        "original_p4_cycle_window": None if p4_window is None else {
            "first_iteration": p4_window[0], "last_iteration": p4_window[1],
        },
        "original_p4_bracket_trial_count": 0 if p4_window is None else len(rows) - raw_stop,
        "original_path_verification_attempted": original_verification_attempted,
        "last_recorded_xB_residual": abs(float(final_row["signed_F"])),
        "last_recorded_xB_tolerance": float(final_row["x_tolerance"]),
        "last_recorded_population_change_from_previous": final_row["population_L1_from_previous"],
        "last_recorded_cell_measure_from_previous": final_row["cell_measure_from_previous"],
        "population_cell_measure_semantics": (
            "original_direct_picard_previous_iterate" if p4_window is None
            else "P4_bracket_trial_to_previous_bracket_trial; map-successor verification was not run unless original_path_verification_attempted is true"
        ),
    }


def _validate_accepted_diagnostic(
    solver: CharacteristicReferenceSolver, diagnostic: CharacteristicStepDiagnostics
) -> None:
    mode = str(diagnostic.fixed_point_convergence_mode)
    period = int(diagnostic.fixed_point_periodic_cycle_period)
    allowed = mode in {"DIRECT", "IDENTITY"} and period == 0
    allowed = allowed or (
        mode == "BRACKETED_SCALAR_ROOT" and period in {2, 4}
        and int(diagnostic.fixed_point_bracketed_root_iterations) > 0
    )
    if not allowed:
        raise PathologyWorkflowError(f"baseline used a forbidden closure mode {mode}/P{period}")
    if float(diagnostic.fixed_point_xb_residual) > float(diagnostic.fixed_point_xb_tolerance):
        raise PathologyWorkflowError("an accepted baseline substep failed its frozen scalar tolerance")
    if float(diagnostic.fixed_point_population_residual) > float(solver._population_convergence_rtol):
        raise PathologyWorkflowError("an accepted baseline substep failed population closure")
    if not math.isfinite(float(diagnostic.fixed_point_cell_measure_residual)):
        raise PathologyWorkflowError("an accepted baseline substep has non-finite cell measure")
    if float(diagnostic.inventory.relative_residual) > float(solver.config.inventory_tolerance_relative):
        raise PathologyWorkflowError("an accepted baseline substep failed inventory closure")


def _accepted_trial_evaluation(
    solver: RecordingCharacteristicReferenceSolver,
    *,
    trials: Sequence[Any],
    old_cells: np.ndarray,
    diagnostic: CharacteristicStepDiagnostics,
) -> ClosureMapEvaluation:
    current_cells = np.asarray(solver._beta_cell_numbers(), dtype=np.float64)
    widths = np.asarray(solver.population("beta").grid.widths_m, dtype=np.float64)
    # A CR1 commit stores density then later reconstructs cell numbers through
    # ``(N / width) * width``.  That documented binary64 round trip can differ
    # by one ulp from the uncommitted remap array.  Match the exact committed
    # representation, rather than weakening the match with a tolerance.
    matches = [
        trial for trial in trials
        if trial.matrix_xb == float(solver.matrix_xb)
        and trial.midpoint_matrix_xb == float(diagnostic.midpoint_matrix_xb)
        and np.array_equal(
            (np.asarray(trial.cell_number_m3, dtype=np.float64) / widths) * widths,
            current_cells,
        )
    ]
    if not matches:
        raise PathologyWorkflowError("accepted substep has no matching immutable closure trial telemetry")
    evaluations = [
        evaluation_from_recorded_trial(solver, trial=trial, old_cell_number_m3=old_cells)
        for trial in matches
    ]
    signatures = {
        (item.departure_signature, item.remap_topology_signature, item.population_hash, item.x_trial.hex())
        for item in evaluations
    }
    if len(signatures) != 1:
        raise PathologyWorkflowError("accepted substep has ambiguous final closure-trial telemetry")
    return evaluations[-1]


def _run_fixed_substeps(
    *,
    config: SolverConfig,
    restart_checkpoint: Path,
    m: int,
) -> FixedRun:
    """Run only the frozen ordinary/P2/P4 baseline path and retain each leaf."""

    solver = RecordingCharacteristicReferenceSolver.load_checkpoint(
        config=config, path=restart_checkpoint
    )
    if accepted_state_hash(solver) != RESTART_STATE_HASH or int(solver.step) != RESTART_STEP:
        raise PathologyWorkflowError("loaded checkpoint is not the frozen step-244 restart state")
    dt_s = DT0_S / float(m)
    states: list[dict[str, Any]] = []
    rows: list[dict[str, Any]] = []
    for substep in range(1, int(m) + 1):
        before_prestate = capture_prestate(solver)
        before_hash = accepted_state_hash(solver)
        before_history = tuple(solver.history)
        cursor = solver.trial_cursor()
        try:
            diagnostic = solver.advance_one_ordinary_or_qualified_cycle(maximum_dt_s=dt_s)
        except (CharacteristicReferenceError, RadiusGridOverflowError, ValueError, FloatingPointError) as error:
            immutable = (
                accepted_state_hash(solver) == before_hash and tuple(solver.history) == before_history
            )
            if not immutable:
                raise PathologyWorkflowError(
                    f"rejected m{m} substep {substep} mutated accepted state"
                ) from error
            parsed = _parse_failure_message(str(error))
            failure_trace, failure_trial_summary = _recorded_failure_trace(
                solver,
                trials=solver.trials_since(cursor),
                prestate=before_prestate,
                closure_path=str(parsed["closure_path"]),
            )
            failure = {
                "m": int(m),
                "dt_s": dt_s,
                "failed_substep": int(substep),
                "prestate_physical_time_s": float(solver.time_s),
                "candidate_end_physical_time_s": float(solver.time_s + dt_s),
                "state_before_failure_hash": before_hash,
                "state_immutable_after_rejection": immutable,
                "error_type": type(error).__name__,
                **parsed,
                **failure_trial_summary,
            }
            return FixedRun(
                m=m, dt_s=dt_s, status="NONCLOSING_SUBSTEP" if m != 1 else "FULL_STEP_NOT_ADMISSIBLE",
                accepted_states=tuple(states), accepted_rows=tuple(rows), failure=failure,
                failure_closure_trace=failure_trace,
                solver_at_failure_prestate=solver, prestate_at_failure=before_prestate,
            )
        _validate_accepted_diagnostic(solver, diagnostic)
        evaluation = _accepted_trial_evaluation(
            solver, trials=solver.trials_since(cursor), old_cells=before_prestate.old_cell_number_m3,
            diagnostic=diagnostic,
        )
        state = state_summary(
            solver, fraction=Fraction(substep, m), m=m, substep=substep, evaluation=evaluation
        )
        state["_accepted_evaluation"] = evaluation
        states.append(state)
        rows.append({
            "m": int(m), "substep": int(substep), "step": int(diagnostic.step),
            "dt_s": dt_s,
            "physical_time_s": float(diagnostic.time_s),
            "physical_time_h": float(diagnostic.time_s / 3600.0),
            "matrix_xB": float(diagnostic.matrix_xb),
            "midpoint_matrix_xB": float(diagnostic.midpoint_matrix_xb),
            "fixed_point_convergence_mode": str(diagnostic.fixed_point_convergence_mode),
            "fixed_point_periodic_cycle_period": int(diagnostic.fixed_point_periodic_cycle_period),
            "fixed_point_iterations": int(diagnostic.fixed_point_iterations),
            "fixed_point_picard_iterations": int(diagnostic.fixed_point_picard_iterations),
            "fixed_point_bracketed_root_iterations": int(diagnostic.fixed_point_bracketed_root_iterations),
            "xB_residual": float(diagnostic.fixed_point_xb_residual),
            "xB_tolerance": float(diagnostic.fixed_point_xb_tolerance),
            "population_residual": float(diagnostic.fixed_point_population_residual),
            "cell_measure": float(diagnostic.fixed_point_cell_measure_residual),
            "convergence_rate": float(diagnostic.fixed_point_convergence_rate),
            "root_verification_kind": str(diagnostic.fixed_point_root_verification_kind),
            "root_trial_xB_residual": float(diagnostic.fixed_point_root_trial_xb_residual),
            "root_verification_population_residual": float(diagnostic.fixed_point_root_verification_population_residual),
            "bracket_initial_width": float(diagnostic.fixed_point_bracket_initial_width),
            "bracket_final_width": float(diagnostic.fixed_point_bracket_final_width),
            "bracket_left_xB": float(diagnostic.fixed_point_bracket_left_xb),
            "bracket_right_xB": float(diagnostic.fixed_point_bracket_right_xb),
            "bracket_left_signed_F": float(diagnostic.fixed_point_bracket_left_signed_residual),
            "bracket_right_signed_F": float(diagnostic.fixed_point_bracket_right_signed_residual),
            "inventory_residual_mol_m3": float(diagnostic.inventory.residual_mol_m3),
            "inventory_relative_residual": float(diagnostic.inventory.relative_residual),
            "Q_total_mol_m3": float(diagnostic.inventory.total_mol_m3),
            "Q_beta_mol_m3": float(diagnostic.inventory.beta_resolved_mol_m3),
            "Q_matrix_mol_m3": float(diagnostic.inventory.matrix_mol_m3),
            "rmin_number_loss_m3": float(diagnostic.rmin_number_loss_m3),
            "rmin_mol_b_loss_mol_m3": float(diagnostic.rmin_mol_b_loss_mol_m3),
            "remap_number_conservation_residual_m3": float(diagnostic.remap_number_conservation_residual_m3),
            "lower_no_inflow_face_count": int(diagnostic.lower_no_inflow_face_count),
            "upper_no_inflow_face_count": int(diagnostic.upper_no_inflow_face_count),
            "cumulative_number_dissolution_m3": float(solver.cumulative_number_dissolution_m3),
            "cumulative_beta_volume_dissolution": float(solver.cumulative_beta_volume_dissolution),
            "cumulative_mol_B_returned_mol_m3": float(solver.cumulative_mol_b_returned_mol_m3),
            "state_hash": accepted_state_hash(solver),
            "population_array_hash": state["population_array_hash"],
            "departure_signature": evaluation.departure_signature,
            "remap_topology_signature": evaluation.remap_topology_signature,
        })
    return FixedRun(
        m=m, dt_s=dt_s, status="PASS", accepted_states=tuple(states), accepted_rows=tuple(rows),
        failure=None, failure_closure_trace=(), solver_at_failure_prestate=None, prestate_at_failure=None,
    )


def _save_accepted_states(
    *, output_root: Path, states_by_m: Mapping[int, Sequence[Mapping[str, Any]]], edges_m: np.ndarray
) -> tuple[list[dict[str, Any]], str]:
    arrays: dict[str, Any] = {"radius_edges_m": np.asarray(edges_m, dtype=np.float64)}
    index_rows: list[dict[str, Any]] = []
    record = 0
    for m in sorted(states_by_m):
        for state in states_by_m[m]:
            evaluation = state.get("_accepted_evaluation")
            prefix = f"record_{record:04d}_m{m}_s{int(state['substep']):03d}"
            arrays[f"{prefix}_population_cell_number_m3"] = np.asarray(state["population_array"], dtype=np.float64)
            arrays[f"{prefix}_cdf"] = np.asarray(state["cdf"], dtype=np.float64)
            if isinstance(evaluation, ClosureMapEvaluation):
                arrays[f"{prefix}_departure_faces_m"] = np.asarray(evaluation.departure_faces_m, dtype=np.float64)
                arrays[f"{prefix}_source_cell_indices"] = np.asarray(evaluation.source_cell_indices, dtype=np.int64)
            index_rows.append({
                **{key: value for key, value in state.items() if not key.startswith("_") and key not in {"population_array", "cdf"}},
                "record_id": record,
                "npz_population_key": f"{prefix}_population_cell_number_m3",
                "npz_cdf_key": f"{prefix}_cdf",
                "npz_departure_key": f"{prefix}_departure_faces_m" if isinstance(evaluation, ClosureMapEvaluation) else "",
                "npz_source_indices_key": f"{prefix}_source_cell_indices" if isinstance(evaluation, ClosureMapEvaluation) else "",
            })
            record += 1
    path = output_root / "accepted_substep_states.npz"
    np.savez_compressed(path, **arrays)
    _write_csv(output_root / "accepted_substep_index.csv", index_rows, ("record_id", "m", "substep"))
    return index_rows, _sha256_file(path)


def _save_prestate(
    *, solver: RecordingCharacteristicReferenceSolver, prestate: DiagnosticPrestate, path: Path
) -> dict[str, Any]:
    if path.exists():
        raise PathologyWorkflowError(f"refusing to overwrite prestate checkpoint {path}")
    solver.save_checkpoint(path)
    return {
        "checkpoint": str(path),
        "checkpoint_sha256": _sha256_file(path),
        "accepted_state_hash": accepted_state_hash(solver),
        "diagnostic_state_hash": prestate.state_hash,
        "step": int(solver.step),
        "time_s": float(solver.time_s),
        "matrix_xB": float(solver.matrix_xb),
    }


def _load_formal_phase_a_ladder(path: Path) -> dict[int, dict[str, str]]:
    """Load the hash-bound Phase-A endpoint/failure authority table."""

    if not path.is_file():
        raise PathologyWorkflowError("frozen Phase-A ladder CSV is unavailable")
    if _sha256_file(path) != EXPECTED_FORMAL_PHASE_A_SHA256:
        raise PathologyWorkflowError("frozen Phase-A ladder checksum differs")
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_m = {
        int(row["m"]): dict(row)
        for row in rows
        if row.get("m", "").isdigit() and int(row["m"]) in SUBSTEP_MULTIPLICITIES
    }
    missing = [m for m in SUBSTEP_MULTIPLICITIES if m not in by_m]
    if missing:
        raise PathologyWorkflowError(f"frozen Phase-A ladder misses m={missing}")
    return by_m


def _failure_reason_for_phase_a_comparison(failure: Mapping[str, Any], *, m: int) -> str:
    return (
        f"substep={int(failure['failed_substep'])}/{m}; "
        f"state_immutable_after_rejection={bool(failure['state_immutable_after_rejection'])}; "
        f"{failure['error_type']}: {failure['error_message']}"
    )


def _formal_population_array_hash(cells: np.ndarray) -> str:
    """Match the frozen continuation ladder's raw-array hash convention."""

    return hashlib.sha256(
        np.ascontiguousarray(np.asarray(cells, dtype=np.float64)).tobytes()
    ).hexdigest()


def _formal_endpoint_observation(run: FixedRun) -> dict[str, Any]:
    """Map a fixed-substep replay to every non-empty formal ladder field."""

    if run.failure is not None:
        return {
            "m": int(run.m),
            "dt_s": float(run.dt_s),
            "status": run.status,
            "reason": _failure_reason_for_phase_a_comparison(run.failure, m=run.m),
        }
    if not run.accepted_rows or not run.accepted_states:
        raise PathologyWorkflowError(f"successful m{run.m} replay has no endpoint record")
    row = run.accepted_rows[-1]
    state = run.accepted_states[-1]
    modes = [str(item["fixed_point_convergence_mode"]) for item in run.accepted_rows]
    periods = [int(item["fixed_point_periodic_cycle_period"]) for item in run.accepted_rows]
    cells = np.asarray(state["population_array"], dtype=np.float64)
    maximum_residual = max(float(item["xB_residual"]) for item in run.accepted_rows)
    return {
        "m": int(run.m),
        "dt_s": float(run.dt_s),
        "status": run.status,
        "reason": "",
        "policy": f"STEP245_FIXED_M{run.m}",
        "step": int(row["step"]),
        "time_s": float(row["physical_time_s"]),
        "time_h": float(row["physical_time_h"]),
        "matrix_xB": float(state["xB"]),
        # The formal ladder snapshots the state after CR1 commits its density
        # representation.  Its N/width*width round trip can differ by one ulp
        # from the uncommitted trial held by CharacteristicStepDiagnostics, so
        # bind every ledger field to the same post-commit state here.
        "Q_total_mol_m3": float(state["Q_total_mol_m3"]),
        "Q_beta_mol_m3": float(state["Q_beta_mol_m3"]),
        "Q_matrix_mol_m3": float(state["Q_matrix_mol_m3"]),
        "inventory_residual_mol_m3": float(state["inventory_residual_mol_m3"]),
        "inventory_relative_residual": float(state["inventory_relative_residual"]),
        "cumulative_number_dissolution_m3": float(row["cumulative_number_dissolution_m3"]),
        "cumulative_beta_volume_dissolution": float(row["cumulative_beta_volume_dissolution"]),
        "cumulative_mol_B_returned_mol_m3": float(row["cumulative_mol_B_returned_mol_m3"]),
        "state_array_hash": str(row["state_hash"]),
        "population_array_hash": _formal_population_array_hash(cells),
        "nonfinite_cell_count": int(np.count_nonzero(~np.isfinite(cells))),
        "negative_cell_count": int(np.count_nonzero(cells < 0.0)),
        "M0_m3": float(state["M0_m3"]),
        "M1_m2": float(state["M1_m2"]),
        "M2_m": float(state["M2_m"]),
        "M3_dimensionless": float(state["M3_dimensionless"]),
        "Rmean_m": float(state["Rmean_m"]),
        "Rmean3_m3": float(state["Rmean3_m3"]),
        "mean_R3_m3": float(state["M3_dimensionless"]) / max(float(state["M0_m3"]), 1.0e-300),
        "Sv_m_inv": float(state["Sv_m_inv"]),
        "f_beta": float(state["f_beta"]),
        "direct_picard_closure_count": sum(mode == "DIRECT" for mode in modes),
        "p2_closure_count": sum(period == 2 for period in periods),
        "p4_closure_count": sum(period == 4 for period in periods),
        "rejected_trial_step_count": 0,
        "maximum_picard_residual": maximum_residual,
        "maximum_accepted_closure_residual": maximum_residual,
    }


def _same_formal_value(expected: str, observed: Any) -> bool:
    if isinstance(observed, str):
        return observed == expected
    if isinstance(observed, (int, np.integer)):
        return int(expected) == int(observed)
    try:
        left, right = float(expected), float(observed)
    except (TypeError, ValueError):
        return False
    return left == right or (math.isnan(left) and math.isnan(right))


def _compare_phase_a_ladder(
    runs: Mapping[int, FixedRun], formal: Mapping[int, Mapping[str, str]]
) -> dict[str, Any]:
    comparisons: dict[str, Any] = {}
    all_equal = True
    for m in SUBSTEP_MULTIPLICITIES:
        expected = formal[m]
        observed = _formal_endpoint_observation(runs[m])
        fields = [field for field, value in expected.items() if value != ""]
        mismatches = [
            {"field": field, "formal": expected[field], "replay": observed.get(field, "MISSING")}
            for field in fields
            if field not in observed or not _same_formal_value(expected[field], observed[field])
        ]
        comparisons[f"m{m}"] = {
            "status": "PASS" if not mismatches else "FAIL",
            "field_count": len(fields),
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:32],
        }
        all_equal = all_equal and not mismatches
    return {"status": "PASS" if all_equal else "FAIL", "by_m": comparisons}


def _baseline_requirements(
    runs: Mapping[int, FixedRun], *, formal_phase_a: Mapping[int, Mapping[str, str]]
) -> dict[str, Any]:
    conditions = {f"m{m}": runs[m].status for m in SUBSTEP_MULTIPLICITIES}
    m32, m64 = runs[32], runs[64]
    formal_comparison = _compare_phase_a_ladder(runs, formal_phase_a)
    valid = (
        all(runs[m].status == "PASS" for m in (2, 4, 8, 16))
        and m32.status == "NONCLOSING_SUBSTEP" and m32.failure is not None
        and int(m32.failure["failed_substep"]) == 6
        and m32.failure.get("closure_path") == "ALLOWED_P4_EXACT_CYCLE_PATH"
        and m64.status == "NONCLOSING_SUBSTEP" and m64.failure is not None
        and int(m64.failure["failed_substep"]) == 5
        and m64.failure.get("closure_path") == "ORDINARY_PICARD"
        and formal_comparison["status"] == "PASS"
    )
    return {
        "status": "PASS_FINE_SUBSTEP_BASELINE_REPRODUCTION" if valid else "FAIL_FINE_SUBSTEP_BASELINE_REPRODUCTION",
        "runs": conditions,
        "formal_phase_a_ladder_comparison": formal_comparison,
    }


def _exact_diagnostic_equal(left: Any, right: Any) -> bool:
    if isinstance(left, np.ndarray) or isinstance(right, np.ndarray):
        return isinstance(left, np.ndarray) and isinstance(right, np.ndarray) and np.array_equal(left, right)
    if isinstance(left, Mapping) or isinstance(right, Mapping):
        if not isinstance(left, Mapping) or not isinstance(right, Mapping) or set(left) != set(right):
            return False
        return all(_exact_diagnostic_equal(left[key], right[key]) for key in left)
    if isinstance(left, (list, tuple)) or isinstance(right, (list, tuple)):
        return (
            isinstance(left, (list, tuple)) and isinstance(right, (list, tuple))
            and len(left) == len(right)
            and all(_exact_diagnostic_equal(a, b) for a, b in zip(left, right))
        )
    if isinstance(left, (float, np.floating)) or isinstance(right, (float, np.floating)):
        return float(left) == float(right) or (math.isnan(float(left)) and math.isnan(float(right)))
    return left == right


def _accepted_leaf_replay_comparison(first: FixedRun, repeated: FixedRun) -> dict[str, Any]:
    if len(first.accepted_rows) != len(repeated.accepted_rows) or len(first.accepted_states) != len(repeated.accepted_states):
        return {"status": "FAIL", "reason": "accepted leaf count differs"}
    mismatch: dict[str, Any] | None = None
    for index, (left_row, right_row, left_state, right_state) in enumerate(
        zip(first.accepted_rows, repeated.accepted_rows, first.accepted_states, repeated.accepted_states), start=1
    ):
        if not _exact_diagnostic_equal(left_row, right_row):
            mismatch = {"accepted_leaf": index, "kind": "telemetry"}
            break
        left_public_state = {key: value for key, value in left_state.items() if key != "_accepted_evaluation"}
        right_public_state = {key: value for key, value in right_state.items() if key != "_accepted_evaluation"}
        if not _exact_diagnostic_equal(left_public_state, right_public_state):
            mismatch = {"accepted_leaf": index, "kind": "accepted_state"}
            break
    return {
        "status": "PASS" if mismatch is None else "FAIL",
        "accepted_leaf_count": len(first.accepted_rows),
        "first_mismatch": mismatch,
    }


def _repeat_failure_prestates(
    *, config: SolverConfig, restart_checkpoint: Path, first: Mapping[int, FixedRun]
) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for m in (32, 64):
        repeated = _run_fixed_substeps(config=config, restart_checkpoint=restart_checkpoint, m=m)
        original = first[m]
        if original.failure is None or repeated.failure is None:
            raise PathologyWorkflowError(f"m{m} failure was not reproducible on its second generation")
        fields = ("failed_substep", "closure_path", "state_before_failure_hash", "error_message")
        equal = all(original.failure.get(name) == repeated.failure.get(name) for name in fields)
        leaf_comparison = _accepted_leaf_replay_comparison(original, repeated)
        equal = equal and leaf_comparison["status"] == "PASS"
        payload[f"m{m}"] = {
            "repeat_status": "PASS" if equal else "FAIL",
            "first_prestate_hash": original.failure.get("state_before_failure_hash"),
            "second_prestate_hash": repeated.failure.get("state_before_failure_hash"),
            "same_failure_location_and_path": equal,
            "accepted_leaf_replay_comparison": leaf_comparison,
        }
        if not equal:
            raise PathologyWorkflowError(f"m{m} pre-failure state is not deterministically reproducible")
    return payload


def _failure_replay(
    *, config: SolverConfig, restart_checkpoint: Path, baseline: FixedRun
) -> dict[str, Any]:
    replay = _run_fixed_substeps(config=config, restart_checkpoint=restart_checkpoint, m=baseline.m)
    if baseline.failure is None or replay.failure is None:
        raise PathologyWorkflowError(f"m{baseline.m} closure replay did not reproduce a failure")
    fields = ("failed_substep", "closure_path", "state_before_failure_hash", "error_message")
    equal = all(baseline.failure.get(field) == replay.failure.get(field) for field in fields)
    if not equal:
        raise PathologyWorkflowError(f"m{baseline.m} closure failure replay differs")
    return {
        "status": "PASS_FINE_SUBSTEP_CLOSURE_REPRODUCTION",
        "m": baseline.m,
        "closure_path": baseline.failure["closure_path"],
        "failed_substep": baseline.failure["failed_substep"],
        "state_immutable_after_rejection": baseline.failure["state_immutable_after_rejection"],
        "failure": dict(baseline.failure),
    }


def _failure_replay_from_saved_prestate(
    *, config: SolverConfig, prestate_checkpoint: Path, baseline: FixedRun
) -> dict[str, Any]:
    """Invoke the original closure policy once from the archived failure state."""

    if baseline.failure is None:
        raise PathologyWorkflowError("cannot replay a missing baseline failure")
    solver = CharacteristicReferenceSolver.load_checkpoint(config=config, path=prestate_checkpoint)
    before_hash = accepted_state_hash(solver)
    before_history = tuple(solver.history)
    if before_hash != baseline.failure["state_before_failure_hash"]:
        raise PathologyWorkflowError("saved failure prestate hash differs from the baseline prestate")
    try:
        solver.advance_one_ordinary_or_qualified_cycle(maximum_dt_s=baseline.dt_s)
    except (CharacteristicReferenceError, RadiusGridOverflowError, ValueError, FloatingPointError) as error:
        immutable = accepted_state_hash(solver) == before_hash and tuple(solver.history) == before_history
        parsed = _parse_failure_message(str(error))
        fields = ("closure_path", "error_message", "xB_residual", "xB_tolerance", "population_residual", "cell_measure")
        equal = immutable and all(
            parsed.get(field) == baseline.failure.get(field)
            for field in fields
            if field in parsed or field in baseline.failure
        )
        if not equal:
            raise PathologyWorkflowError(
                f"saved m{baseline.m} prestate did not reproduce the original closure failure"
            )
        return {
            "status": "PASS_FINE_SUBSTEP_CLOSURE_REPRODUCTION",
            "m": baseline.m,
            "replay_input": str(prestate_checkpoint),
            "closure_path": parsed["closure_path"],
            "state_immutable_after_rejection": immutable,
            "failure": parsed,
        }
    raise PathologyWorkflowError(
        f"saved m{baseline.m} prestate unexpectedly accepted a closure trial"
    )


def _common_time_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    at_one_sixteenth = [
        row for row in rows
        if int(row.get("fraction_numerator", -1)) == 1 and int(row.get("fraction_denominator", -1)) == 16
    ]
    error_fields = (
        "xB", "M0_m3", "M1_m2", "M2_m", "M3_dimensionless", "Rmean_m",
        "Rmean3_m3", "Sv_m_inv", "f_beta", "Q_beta_mol_m3", "Q_matrix_mol_m3",
        "Q_total_mol_m3", "lower_tail_M0_m3", "lower_tail_M3_dimensionless",
        "population_L1", "population_Linf", "PSD_Wasserstein_m", "CDF_difference",
    )
    errors: dict[str, float | None] = {}
    topologies: dict[str, Any] = {}
    for left, right in ((16, 32), (16, 64), (32, 64)):
        label = f"m{left}_m{right}"
        match = next((row for row in at_one_sixteenth if int(row["left_m"]) == left and int(row["right_m"]) == right), None)
        errors[label] = None if match is None else max(
            float(match[name]) for name in error_fields
        )
        topologies[label] = None if match is None else bool(match["topology_equal"])
    pair_rows = {
        (int(row["left_m"]), int(row["right_m"])): row for row in at_one_sixteenth
    }
    coarse = pair_rows.get((16, 32))
    fine = pair_rows.get((32, 64))
    direction: dict[str, Any] = {}
    if coarse is not None and fine is not None:
        for field in error_fields:
            coarse_value, fine_value = float(coarse[field]), float(fine[field])
            direction[field] = {
                "m16_m32": coarse_value,
                "m32_m64": fine_value,
                "nonincreasing_under_refinement": fine_value <= coarse_value,
            }
    refinement_nonincreasing = bool(direction) and all(
        bool(item["nonincreasing_under_refinement"]) for item in direction.values()
    )
    return {
        "rows_at_1_16": at_one_sixteenth,
        "maximum_errors_at_1_16": errors,
        "topology_at_1_16": topologies,
        "refinement_direction_at_1_16": direction,
        "all_selected_metrics_nonincreasing_at_1_16": refinement_nonincreasing,
    }


def _classify_prefailure_path(
    rows: Sequence[Mapping[str, Any]], common: Mapping[str, Any]
) -> str:
    available = [row for row in rows if bool(row.get("topology_equal"))]
    topology_equal = len(available) == len(rows) and bool(rows)
    errors = [float(value) for value in common["maximum_errors_at_1_16"].values() if value is not None]
    # The pre-registered 0.25% continuous-observable scale is used only to
    # classify path proximity; it is not a new acceptance threshold.
    aligned = (
        topology_equal
        and bool(errors)
        and max(errors) < 0.0025
        and bool(common["all_selected_metrics_nonincreasing_at_1_16"])
    )
    if not topology_equal:
        return "PREFAILURE_TOPOLOGY_BRANCH_DIVERGENCE"
    if aligned:
        return "PREFAILURE_STATES_ASYMPTOTICALLY_ALIGNED"
    return "PREFAILURE_PATH_DIVERGENCE_DETECTED"


def _scan_crossing_intervals(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    ordered = sorted(
        (row for row in rows if row.get("status") == "OK"),
        key=lambda row: float(row["x_trial"]),
    )
    intervals: list[dict[str, Any]] = []
    for left, right in zip(ordered, ordered[1:]):
        left_f, right_f = float(left["signed_F"]), float(right["signed_F"])
        if left_f == 0.0 or right_f == 0.0 or left_f * right_f < 0.0:
            intervals.append({
                "x_left": float(left["x_trial"]),
                "x_right": float(right["x_trial"]),
                "x_midpoint": 0.5 * (float(left["x_trial"]) + float(right["x_trial"])),
                "F_left": left_f,
                "F_right": right_f,
                "same_topology": left["topology_signature"] == right["topology_signature"],
                "left_topology": left["topology_signature"],
                "right_topology": right["topology_signature"],
            })
    return intervals


def _orbit_crossing_relation(
    trace_rows: Sequence[Mapping[str, Any]], crossings: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    updates = _raw_update_intervals(trace_rows)
    hit = [
        crossing for crossing in crossings
        if any(
            max(min(left, right), min(float(crossing["x_left"]), float(crossing["x_right"])))
            <= min(max(left, right), max(float(crossing["x_left"]), float(crossing["x_right"])))
            for left, right in updates
        )
    ]
    return {
        "raw_update_count": len(updates),
        "raw_update_intersects_local_crossing": bool(hit),
        "intersected_crossing_count": len(hit),
        "intersected_crossings": hit,
    }


def _m32_m64_map_comparison(
    *,
    m32_scan_rows: Sequence[Mapping[str, Any]],
    m64_scan_rows: Sequence[Mapping[str, Any]],
    m32_scan: Mapping[str, Any],
    m64_scan: Mapping[str, Any],
    m32_trace: Sequence[Mapping[str, Any]],
    m64_trace: Sequence[Mapping[str, Any]],
    m32_prestate_state: Mapping[str, Any],
    m64_prestate_state: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Compare map geometry and actual raw orbit/crossing relations."""

    m32_by_x = {float(row["x_trial"]): row for row in m32_scan_rows if row.get("status") == "OK"}
    m64_by_x = {float(row["x_trial"]): row for row in m64_scan_rows if row.get("status") == "OK"}
    common_x = sorted(set(m32_by_x).intersection(m64_by_x))
    f_differences = [
        abs(float(m32_by_x[x]["signed_F"]) - float(m64_by_x[x]["signed_F"]))
        for x in common_x
    ]
    f_scale = max(
        [abs(float(m32_by_x[x]["signed_F"])) for x in common_x]
        + [abs(float(m64_by_x[x]["signed_F"])) for x in common_x]
        + [1.0e-300]
    )
    same_topology_points = sum(
        m32_by_x[x]["topology_signature"] == m64_by_x[x]["topology_signature"] for x in common_x
    )
    m32_crossings = _scan_crossing_intervals(m32_scan_rows)
    m64_crossings = _scan_crossing_intervals(m64_scan_rows)
    m32_orbit = _orbit_crossing_relation(m32_trace, m32_crossings)
    m64_orbit = _orbit_crossing_relation(m64_trace, m64_crossings)
    crossing_count_delta = len(m64_crossings) - len(m32_crossings)
    m32_sign_changes = sum(
        float(left["signed_F"]) * float(right["signed_F"]) < 0.0
        for left, right in zip(m32_trace, m32_trace[1:])
    )
    m64_sign_changes = sum(
        float(left["signed_F"]) * float(right["signed_F"]) < 0.0
        for left, right in zip(m64_trace, m64_trace[1:])
    )
    if crossing_count_delta > 0:
        refinement_behavior = "MORE_MULTIPLY_CROSSED"
    elif m64_sign_changes > m32_sign_changes:
        refinement_behavior = "MORE_OSCILLATORY"
    elif (
        int(m64_scan["NUMBER_OF_TOPOLOGY_PARTITIONS"])
        < int(m32_scan["NUMBER_OF_TOPOLOGY_PARTITIONS"])
        and len(m64_crossings) <= len(m32_crossings)
    ):
        refinement_behavior = "SMOOTHER_BY_OBSERVED_PARTITION_AND_CROSSING_COUNTS"
    else:
        refinement_behavior = "UNCHANGED_OR_UNRESOLVED_FROM_OBSERVED_SAMPLES"
    prestate_topology_equal = (
        m32_prestate_state["remap_topology_signature"]
        == m64_prestate_state["remap_topology_signature"]
    )
    summary = {
        "failure_prestates_have_same_topology": prestate_topology_equal,
        "common_scalar_sample_count": len(common_x),
        "same_topology_common_sample_count": same_topology_points,
        "same_topology_common_sample_fraction": (
            same_topology_points / len(common_x) if common_x else None
        ),
        "F_Linf_difference": max(f_differences, default=None),
        "F_Linf_relative_to_observed_scale": (
            max(f_differences, default=0.0) / f_scale if common_x else None
        ),
        "m32_crossing_count": len(m32_crossings),
        "m64_crossing_count": len(m64_crossings),
        "m32_crossings": m32_crossings,
        "m64_crossings": m64_crossings,
        "m32_p4_orbit_relation": m32_orbit,
        "m64_raw_orbit_relation": m64_orbit,
        "m32_raw_sign_change_count": m32_sign_changes,
        "m64_raw_sign_change_count": m64_sign_changes,
        "refinement_behavior": refinement_behavior,
    }
    rows = [
        {"record_type": "summary", "feature": key, "m32": value, "m64": ""}
        for key, value in summary.items()
        if key not in {"m32_crossings", "m64_crossings", "m32_p4_orbit_relation", "m64_raw_orbit_relation"}
    ]
    rows.extend(
        {"record_type": "m32_crossing", "feature": "observed_crossing", "m32": item, "m64": ""}
        for item in m32_crossings
    )
    rows.extend(
        {"record_type": "m64_crossing", "feature": "observed_crossing", "m32": "", "m64": item}
        for item in m64_crossings
    )
    rows.append({"record_type": "orbit_relation", "feature": "m32_p4", "m32": m32_orbit, "m64": ""})
    rows.append({"record_type": "orbit_relation", "feature": "m64_raw", "m32": "", "m64": m64_orbit})
    return rows, summary


def _milestones(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    payload: dict[str, Any] = {}
    for count in (128, 256, 512, 1024):
        if len(rows) >= count:
            item = rows[count - 1]
            payload[str(count)] = {"absF": item.get("absF"), "signed_F": item.get("signed_F"), "x_trial": item.get("x_trial")}
        else:
            payload[str(count)] = "NOT_REACHED"
    return payload


def _probe_x(rows: Sequence[Mapping[str, Any]]) -> list[float]:
    valid = [row for row in rows if math.isfinite(float(row.get("absF", math.nan)))]
    if not valid:
        return []
    selected = [valid[0], valid[-1], min(valid, key=lambda item: float(item["absF"]))]
    answer: list[float] = []
    for item in selected:
        x = float(item["x_trial"])
        if x not in answer:
            answer.append(x)
    return answer


def _raw_update_intervals(rows: Sequence[Mapping[str, Any]]) -> list[tuple[float, float]]:
    """Return exact raw-Picard update segments for orbit/transition linkage."""

    intervals: list[tuple[float, float]] = []
    for row in rows:
        try:
            left, right = float(row["x_trial"]), float(row["x_closure"])
        except (KeyError, TypeError, ValueError):
            continue
        if math.isfinite(left) and math.isfinite(right):
            intervals.append((left, right))
    return intervals


def _assert_raw_trace_matches_original_rejection(
    *, original: Sequence[Mapping[str, Any]], diagnostic: Sequence[Mapping[str, Any]], label: str
) -> None:
    original_raw = [
        row for row in original
        if str(row.get("record_type", "")).startswith("ORIGINAL_RAW_PICARD")
        or row.get("record_type") == "ORIGINAL_RAW_PICARD"
    ]
    if len(diagnostic) < len(original_raw):
        raise PathologyWorkflowError(f"{label} diagnostic raw trace is shorter than original rejected path")
    fields = (
        "x_trial_hex", "x_closure_hex", "signed_F_hex", "population_hash",
        "CDF_signature", "departure_signature", "topology_signature",
    )
    for index, (expected, observed) in enumerate(zip(original_raw, diagnostic), start=1):
        if any(expected.get(field) != observed.get(field) for field in fields):
            raise PathologyWorkflowError(
                f"{label} diagnostic raw trace differs from its original rejected closure at iteration {index}"
            )


def _same_prestate_different_dt(
    *, solver: CharacteristicReferenceSolver, prestate: DiagnosticPrestate
) -> tuple[list[dict[str, Any]], dict[str, Any], list[dict[str, Any]]]:
    rows: list[dict[str, Any]] = []
    evidence_rows: list[dict[str, Any]] = []
    for divisor in (16, 32, 64, 128):
        dt_s = DT0_S / float(divisor)
        trace_rows, trace_summary, trace_evaluations = raw_picard_trace(
            solver, prestate=prestate, dt_s=dt_s, maximum_iterations=128,
            stop_after_first_exact_period=False, label=f"same_prestate_dt0_over_{divisor}",
        )
        scan_rows, transitions, scan_summary = scalar_map_scan(
            solver, prestate=prestate, dt_s=dt_s, coarse_points=256,
            anchor_x=_probe_x(trace_rows),
            raw_update_intervals=_raw_update_intervals(trace_rows),
            label=f"same_prestate_dt0_over_{divisor}",
        )
        del trace_evaluations
        evidence_rows.extend(
            {"record_type": "raw_picard", "divisor": divisor, **item}
            for item in trace_rows
        )
        evidence_rows.extend(
            {"record_type": "scalar_map", "divisor": divisor, **item}
            for item in scan_rows
        )
        evidence_rows.extend(
            {"record_type": "topology_transition", "divisor": divisor, **item}
            for item in transitions
        )
        rows.append({
            "record_type": "summary",
            "candidate_dt_s": dt_s,
            "divisor": divisor,
            "diagnostic_only_no_state_advance": True,
            "raw_picard_classification": trace_summary["classification"],
            "raw_exact_period": trace_summary["exact_period"],
            "raw_final_absF": trace_summary["final_absF"],
            "raw_initial_absF": trace_summary["initial_absF"],
            "raw_topology_signature_count": trace_summary["topology_signature_count"],
            "observed_sign_crossings": scan_summary["NUMBER_OF_OBSERVED_SIGN_CROSSINGS"],
            "topology_partitions": scan_summary["NUMBER_OF_TOPOLOGY_PARTITIONS"],
        })
    finite = [float(row["raw_final_absF"]) for row in rows if row["raw_final_absF"] is not None]
    trend = "UNRESOLVED"
    if len(finite) == 4:
        if finite[-1] < finite[0]:
            trend = "SMALLER_DT_IMPROVES_RAW_CLOSURE_FROM_SAME_PRESTATE"
        elif finite[-1] > finite[0]:
            trend = "SMALLER_DT_WORSENS_RAW_CLOSURE_FROM_SAME_PRESTATE"
        else:
            trend = "SMALLER_DT_RAW_CLOSURE_UNCHANGED_FROM_SAME_PRESTATE"
    return rows, {
        "canonical_fraction": "1/16", "trend": trend,
        "evidence_row_count": len(evidence_rows),
    }, evidence_rows


def _precision_audit(
    *, cases: Sequence[tuple[str, CharacteristicReferenceSolver, DiagnosticPrestate, float, Sequence[Mapping[str, Any]]]]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    bitwise = True
    max_reduction_relative = 0.0
    max_reduction_absolute = 0.0
    max_shadow = 0.0
    tolerance_scale = 0.0
    for label, solver, prestate, dt_s, trace_rows in cases:
        probes = _probe_x(trace_rows)
        repeat_rows, repeat_ok = repeatability_audit(
            solver, prestate=prestate, dt_s=dt_s, x_trials=probes, repeats=10
        )
        for row in repeat_rows:
            rows.append({"record_type": "repeatability", "case": label, **row})
        bitwise = bitwise and repeat_ok
        for x_trial in probes:
            evaluation = evaluate_closure_map(
                solver, prestate=prestate, dt_s=dt_s, x_trial=x_trial
            )
            shadow = precision_reduction_shadow(
                solver, evaluation, previous_cells=prestate.old_cell_number_m3
            )
            shadow["record_type"] = "reduction_shadow"
            shadow["case"] = label
            rows.append(shadow)
            tolerance_scale = max(tolerance_scale, float(evaluation.x_tolerance))
            max_shadow = max(max_shadow, float(shadow["signed_F_shadow_delta"]))
            max_reduction_absolute = max(max_reduction_absolute, *(
                float(shadow.get(name, 0.0)) for name in (
                    "M0_max_reduction_delta", "M1_max_reduction_delta", "M2_max_reduction_delta",
                    "M3_max_reduction_delta", "Q_beta_max_reduction_delta", "cell_measure_max_reduction_delta",
                )
            ))
            max_reduction_relative = max(max_reduction_relative, *(
                float(shadow.get(name, 0.0)) for name in (
                    "M0_max_reduction_relative_delta", "M1_max_reduction_relative_delta",
                    "M2_max_reduction_relative_delta", "M3_max_reduction_relative_delta",
                    "Q_beta_max_reduction_relative_delta", "cell_measure_max_reduction_relative_delta",
                )
            ))
    if not bitwise:
        status = "PRECISION_FLOOR_SUSPECTED"
    elif max_shadow > 0.1 * max(tolerance_scale, 1.0e-300):
        status = "INVERSE_MAPPING_CONDITIONING_SENSITIVE"
    elif max_reduction_relative > 1.0e-10:
        status = "REDUCTION_ORDER_SENSITIVE"
    else:
        status = "FLOAT64_RESIDUAL_STABLE"
    return rows, {
        "FLOAT64_REPEATABILITY": "BITWISE_DETERMINISTIC" if bitwise else "NOT_BITWISE_DETERMINISTIC",
        "COMPENSATED_REDUCTION_EFFECT": {
            "maximum_absolute_native_unit_delta": max_reduction_absolute,
            "maximum_relative_delta": max_reduction_relative,
        },
        "EXTENDED_PRECISION_EFFECT": max_shadow,
        "PRECISION_FLOOR_STATUS": status,
        "tolerance_scale_for_diagnostic_comparison": tolerance_scale,
    }


def _root_cause(
    *, prefailure: str, m32_scan: Mapping[str, Any], m64_scan: Mapping[str, Any],
    m32_raw: Mapping[str, Any], m64_raw: Mapping[str, Any], precision: Mapping[str, Any],
    transitions: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    transition_count = len(transitions)
    jump_count = sum(bool(row.get("map_jump_observed")) for row in transitions)
    relevant = [
        row for row in transitions
        if bool(row.get("departure_cell_crossing"))
        and bool(row.get("single_departure_cell_crossing"))
        and bool(row.get("same_trace_topology"))
        and bool(row.get("raw_update_intersects_transition_interval"))
        and bool(row.get("local_one_sided_F_delta_persistent"))
        and row.get("local_endpoint_bitwise_repeatable") is True
    ]
    relevant_labels = {str(row.get("label")) for row in relevant}
    relevant_boundary_keys = {
        (
            str(row.get("label")), int(row.get("face_index", -1)),
            str(row.get("source_cell_left")), str(row.get("source_cell_right")),
        )
        for row in relevant
    }
    closure_relevant_jump_count = len(relevant_boundary_keys)
    # The scan covers the complete physical [0, 1] interval.  A non-smooth
    # event elsewhere remains reportable, but only a one-face, locally
    # persistent, bitwise-repeatable departure crossing intersected by the
    # actual failure orbit is eligible to explain a rejected closure.
    remap_evidence = len(relevant_labels) == 2
    partial_remap_evidence = len(relevant_labels) == 1
    precision_status = str(precision["PRECISION_FLOOR_STATUS"])
    secondary: str | None = None
    if prefailure == "PREFAILURE_TOPOLOGY_BRANCH_DIVERGENCE":
        primary = "PREFAILURE_TRAJECTORY_BRANCH_DIVERGENCE"
    elif precision_status in {"REDUCTION_ORDER_SENSITIVE", "INVERSE_MAPPING_CONDITIONING_SENSITIVE", "PRECISION_FLOOR_SUSPECTED"}:
        primary = "FLOAT64_CLOSURE_CONDITIONING_LIMIT"
        if remap_evidence:
            secondary = "CR1_REMAP_NONSMOOTH_MULTIBRANCH_MAP"
    elif remap_evidence:
        primary = "CR1_REMAP_NONSMOOTH_MULTIBRANCH_MAP"
    else:
        crossings = int(m32_scan["NUMBER_OF_OBSERVED_SIGN_CROSSINGS"]) + int(m64_scan["NUMBER_OF_OBSERVED_SIGN_CROSSINGS"])
        stable_path = prefailure == "PREFAILURE_STATES_ASYMPTOTICALLY_ALIGNED"
        if stable_path and crossings > 0:
            primary = "CLOSURE_SOLVER_PATHOLOGY_ON_SMOOTH_MAP"
        elif stable_path:
            primary = "NO_STABLE_ADMISSIBLE_CLOSURE_OBSERVED"
        else:
            primary = "MIXED_PATHOLOGY_WITH_EXPLICIT_EVIDENCE"
            secondary = "PREFAILURE_TRAJECTORY_BRANCH_DIVERGENCE"
        if partial_remap_evidence and secondary is None:
            secondary = "CR1_REMAP_NONSMOOTH_MULTIBRANCH_MAP"
    status_map = {
        "CLOSURE_SOLVER_PATHOLOGY_ON_SMOOTH_MAP": "DIAG_CLOSURE_SOLVER_PATHOLOGY",
        "CR1_REMAP_NONSMOOTH_MULTIBRANCH_MAP": "DIAG_CR1_REMAP_NONSMOOTHNESS",
        "PREFAILURE_TRAJECTORY_BRANCH_DIVERGENCE": "DIAG_PREFAILURE_BRANCH_DIVERGENCE",
        "FLOAT64_CLOSURE_CONDITIONING_LIMIT": "DIAG_FLOAT64_CONDITIONING",
        "NO_STABLE_ADMISSIBLE_CLOSURE_OBSERVED": "DIAG_NO_STABLE_CLOSURE",
        "MIXED_PATHOLOGY_WITH_EXPLICIT_EVIDENCE": "DIAG_MIXED_CLOSURE_PATHOLOGY",
    }
    return {
        "PRIMARY_ROOT_CAUSE": primary,
        "SECONDARY_ROOT_CAUSE": secondary,
        "TOP_LEVEL_STATUS": status_map[primary],
        "evidence": {
            "prefailure_path": prefailure,
            "m32_raw_behavior": m32_raw["classification"],
            "m64_raw_behavior": m64_raw["classification"],
            "m32_observed_crossings": m32_scan["NUMBER_OF_OBSERVED_SIGN_CROSSINGS"],
            "m64_observed_crossings": m64_scan["NUMBER_OF_OBSERVED_SIGN_CROSSINGS"],
            "topology_transition_count": transition_count,
            "global_coarse_delta_outlier_count": jump_count,
            "closure_relevant_jump_correlated_transition_count": closure_relevant_jump_count,
            "nonclosure_relevant_transition_count": transition_count - len(relevant),
            "closure_relevant_labels": sorted(relevant_labels),
            "common_m32_m64_remap_evidence": remap_evidence,
            "partial_single_failure_remap_evidence": partial_remap_evidence,
            "precision_status": precision_status,
        },
    }


def _next_method(primary: str) -> str:
    choices = {
        "CLOSURE_SOLVER_PATHOLOGY_ON_SMOOTH_MAP": "SCALAR_ROOT_FIRST_WITH_TEMPORAL_CONTINUATION_CANDIDATE_ONLY",
        "CR1_REMAP_NONSMOOTH_MULTIBRANCH_MAP": "CR2_MONOTONE_PIECEWISE_LINEAR_CONSERVATIVE_REMAP_CANDIDATE_ONLY",
        "PREFAILURE_TRAJECTORY_BRANCH_DIVERGENCE": "REVISIT_CONTINUATION_AND_TIME_DISCRETIZATION_DEFINITION",
        "FLOAT64_CLOSURE_CONDITIONING_LIMIT": "AUDIT_REDUCTIONS_AND_MATRIX_INVERSE_BEFORE_ANY_TOLERANCE_CHANGE",
        "NO_STABLE_ADMISSIBLE_CLOSURE_OBSERVED": "FORMALLY_REASSESS_CHARACTERISTIC_REFERENCE_CR1_ROUTE",
        "MIXED_PATHOLOGY_WITH_EXPLICIT_EVIDENCE": "RESOLVE_THE_DOCUMENTED_PRIMARY_NUMERICAL_PATHOLOGY_BEFORE_ANY_METHOD_UPGRADE",
    }
    return choices[primary]


def _write_reports(
    *, report_root: Path, payloads: Mapping[str, Mapping[str, Any] | str]
) -> None:
    for filename, payload in payloads.items():
        _write_markdown(report_root / filename, REPORT_TITLES[filename], payload)


def _run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    restart_checkpoint = Path(args.restart_checkpoint)
    formal_trace = Path(args.formal_trace_csv)
    formal_phase_a = Path(args.formal_phase_a_csv)
    if output_root.exists() or report_root.exists():
        raise PathologyWorkflowError("refusing to overwrite existing pathology output/report roots")
    if not restart_checkpoint.is_file():
        raise PathologyWorkflowError("frozen step-244 restart checkpoint is unavailable")
    if not formal_trace.is_file():
        raise PathologyWorkflowError("frozen formal trace is unavailable")
    if not formal_phase_a.is_file():
        raise PathologyWorkflowError("frozen Phase-A ladder is unavailable")
    output_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    (output_root / "figures").mkdir()
    started = time.monotonic()
    source = _source_identity(require_clean=not bool(args.allow_dirty_source))
    if _sha256_file(formal_trace) != EXPECTED_FORMAL_TRACE_SHA256:
        raise PathologyWorkflowError("formal trace checksum differs from the frozen Phase-A record")
    formal_phase_a_rows = _load_formal_phase_a_ladder(formal_phase_a)
    if _sha256_file(restart_checkpoint) != EXPECTED_RESTART_CHECKPOINT_SHA256:
        raise PathologyWorkflowError("restart checkpoint checksum differs from the frozen Phase-A record")
    context = build_frozen_canonical_context()
    config = SolverConfig.from_mapping(context.mapping)
    loaded = RecordingCharacteristicReferenceSolver.load_checkpoint(config=config, path=restart_checkpoint)
    if accepted_state_hash(loaded) != RESTART_STATE_HASH or int(loaded.step) != RESTART_STEP:
        raise PathologyWorkflowError("restart checkpoint does not bind the frozen state-244 hash")

    runs = {m: _run_fixed_substeps(config=config, restart_checkpoint=restart_checkpoint, m=m) for m in SUBSTEP_MULTIPLICITIES}
    baseline = _baseline_requirements(runs, formal_phase_a=formal_phase_a_rows)
    if baseline["status"] != "PASS_FINE_SUBSTEP_BASELINE_REPRODUCTION":
        _write_reports(report_root=report_root, payloads={
            "00_phase_a_reproduction.md": {**baseline, "stop_reason": "baseline failure location/path differs"},
        })
        fields = {"STATUS": "FAIL_FINE_SUBSTEP_BASELINE_REPRODUCTION", "BRANCH": source["git_branch"], "COMMIT": source["git_commit"]}
        _write_markdown(report_root / "12_final_acceptance_report.md", REPORT_TITLES["12_final_acceptance_report.md"], fields)
        # A failed reproduction is still a scientific artifact.  Preserve the
        # frozen inputs and canonical-contract identity needed to distinguish
        # a genuine Phase-A mismatch from an accidental environment change.
        _write_json(output_root / "analysis_provenance.json", {
            "task_name": TASK_NAME,
            "top_status": fields["STATUS"],
            "source": source,
            "runtime": _runtime_provenance(),
            "frozen_canonical_snapshot": frozen_canonical_snapshot_provenance(),
            "formal_trace_csv": str(formal_trace),
            "formal_trace_sha256": _sha256_file(formal_trace),
            "formal_phase_a_csv": str(formal_phase_a),
            "formal_phase_a_sha256": _sha256_file(formal_phase_a),
            "restart_checkpoint": str(restart_checkpoint),
            "restart_checkpoint_sha256": _sha256_file(restart_checkpoint),
            "restart_state_hash": RESTART_STATE_HASH,
            "validation_contract_hash": context.contract_hash,
            "fixture_hash": context.fixture_hash,
            "fixture_input_set_sha256": EXPECTED_FIXTURE_INPUT_SET_SHA256,
            "test_status": str(args.test_status),
            "baseline": baseline,
            "stop_reason": "baseline failure location/path differs",
            "runner_sha256": _sha256_file(Path(__file__)),
            "pathology_module_sha256": _sha256_file(ROOT / "src/kwn_mvp/characteristic_pathology.py"),
            "characteristic_solver_sha256": _sha256_file(ROOT / "src/kwn_mvp/characteristic_reference.py"),
            "diagnostic_only": True,
            "scalar_root_fallback_used": False,
            "generic_root_selection_used": False,
            "time_reference_v2": "NOT_ASSIGNED",
        })
        return 2, fields

    repeated = _repeat_failure_prestates(config=config, restart_checkpoint=restart_checkpoint, first=runs)
    m32, m64 = runs[32], runs[64]
    if m32.solver_at_failure_prestate is None or m32.prestate_at_failure is None or m32.failure is None:
        raise PathologyWorkflowError("m32 baseline did not preserve its pre-failure state")
    if m64.solver_at_failure_prestate is None or m64.prestate_at_failure is None or m64.failure is None:
        raise PathologyWorkflowError("m64 baseline did not preserve its pre-failure state")
    m32_prestate = _save_prestate(
        solver=m32.solver_at_failure_prestate, prestate=m32.prestate_at_failure,
        path=output_root / "m32_failure_prestate.npz",
    )
    m64_prestate = _save_prestate(
        solver=m64.solver_at_failure_prestate, prestate=m64.prestate_at_failure,
        path=output_root / "m64_failure_prestate.npz",
    )
    m32_replay = _failure_replay_from_saved_prestate(
        config=config, prestate_checkpoint=output_root / "m32_failure_prestate.npz", baseline=m32
    )
    m64_replay = _failure_replay_from_saved_prestate(
        config=config, prestate_checkpoint=output_root / "m64_failure_prestate.npz", baseline=m64
    )
    # All high-volume raw-map work uses ordinary solvers freshly loaded from
    # the saved immutable prestate.  The recording subclass is restricted to
    # the fixed-substep baseline, so diagnostic scans cannot accumulate trial
    # history or accidentally affect accepted-step telemetry.
    m32_diagnostic_solver = CharacteristicReferenceSolver.load_checkpoint(
        config=config, path=output_root / "m32_failure_prestate.npz"
    )
    m64_diagnostic_solver = CharacteristicReferenceSolver.load_checkpoint(
        config=config, path=output_root / "m64_failure_prestate.npz"
    )
    m32_diagnostic_prestate = capture_prestate(m32_diagnostic_solver)
    m64_diagnostic_prestate = capture_prestate(m64_diagnostic_solver)
    if (
        accepted_state_hash(m32_diagnostic_solver) != m32_prestate["accepted_state_hash"]
        or accepted_state_hash(m64_diagnostic_solver) != m64_prestate["accepted_state_hash"]
    ):
        raise PathologyWorkflowError("saved failure prestates do not reload bitwise")

    states_by_m = {m: runs[m].accepted_states for m in (16, 32, 64)}
    accepted_index, accepted_archive_sha = _save_accepted_states(
        output_root=output_root, states_by_m=states_by_m, edges_m=np.asarray(context.edges_m, dtype=np.float64)
    )
    common_rows = common_time_comparisons(states_by_m, edges_m=np.asarray(context.edges_m, dtype=np.float64))
    _write_csv(output_root / "common_physical_time_state_comparison.csv", common_rows, ("left_m", "right_m", "fraction"))
    common = _common_time_summary(common_rows)
    prefailure = _classify_prefailure_path(common_rows, common)

    m32_trace, m32_raw, m32_evaluations = raw_picard_trace(
        m32_diagnostic_solver, prestate=m32_diagnostic_prestate, dt_s=m32.dt_s,
        maximum_iterations=1024, stop_after_first_exact_period=True, label="m32_failure_prestate_raw_picard",
    )
    # The serialized trace carries all evidence needed downstream.  Drop the
    # full trial arrays before the independent m64 trace to keep this
    # diagnostic comfortably within its declared memory budget.
    del m32_evaluations
    m64_trace, m64_raw, m64_evaluations = raw_picard_trace(
        m64_diagnostic_solver, prestate=m64_diagnostic_prestate, dt_s=m64.dt_s,
        maximum_iterations=1024, stop_after_first_exact_period=False, label="m64_failure_prestate_raw_picard",
    )
    del m64_evaluations
    _assert_raw_trace_matches_original_rejection(
        original=m32.failure_closure_trace, diagnostic=m32_trace, label="m32"
    )
    _assert_raw_trace_matches_original_rejection(
        original=m64.failure_closure_trace, diagnostic=m64_trace, label="m64"
    )
    m32_output_trace = [dict(row) for row in m32.failure_closure_trace]
    original_m64_count = len(m64.failure_closure_trace)
    m64_output_trace = [dict(row) for row in m64.failure_closure_trace]
    m64_output_trace.extend({
        **row, "record_type": "DIAGNOSTIC_RAW_PICARD_EXTENSION",
    } for row in m64_trace[original_m64_count:])
    _write_csv(output_root / "m32_failure_closure_trace.csv", m32_output_trace, ("iteration", "x_trial", "signed_F"))
    _write_csv(output_root / "m64_failure_closure_trace.csv", m64_output_trace, ("iteration", "x_trial", "signed_F"))
    m32_scan_rows, m32_transitions, m32_scan = scalar_map_scan(
        m32_diagnostic_solver, prestate=m32_diagnostic_prestate, dt_s=m32.dt_s,
        coarse_points=256, anchor_x=_probe_x(m32_trace),
        raw_update_intervals=_raw_update_intervals(m32_trace),
        label="m32_failure_prestate",
    )
    m64_scan_rows, m64_transitions, m64_scan = scalar_map_scan(
        m64_diagnostic_solver, prestate=m64_diagnostic_prestate, dt_s=m64.dt_s,
        coarse_points=256, anchor_x=_probe_x(m64_trace),
        raw_update_intervals=_raw_update_intervals(m64_trace),
        label="m64_failure_prestate",
    )
    _write_csv(output_root / "m32_scalar_map.csv", m32_scan_rows, ("x_trial", "status", "signed_F"))
    _write_csv(output_root / "m64_scalar_map.csv", m64_scan_rows, ("x_trial", "status", "signed_F"))
    topology_transitions = [*m32_transitions, *m64_transitions]
    _write_csv(output_root / "topology_transition_map.csv", topology_transitions, ("label", "x_left", "x_right"))

    comparison_rows, comparison_summary = _m32_m64_map_comparison(
        m32_scan_rows=m32_scan_rows, m64_scan_rows=m64_scan_rows,
        m32_scan=m32_scan, m64_scan=m64_scan, m32_trace=m32_trace, m64_trace=m64_trace,
        m32_prestate_state=runs[32].accepted_states[-1],
        m64_prestate_state=runs[64].accepted_states[-1],
    )
    _write_csv(output_root / "m32_vs_m64_scalar_map_comparison.csv", comparison_rows, ("feature", "m32", "m64"))

    canonical = next(
        (state for state in runs[64].accepted_states if int(state["substep"]) == 4), None
    )
    if canonical is None:
        raise PathologyWorkflowError("m64 did not preserve the required 4/64 common prestate")
    # m64 remains exactly at its rejected-substep prestate, which is the same
    # accepted state reached after its fourth leaf.
    common_prestate = capture_prestate(m64_diagnostic_solver)
    if common_prestate.state_hash != m64_diagnostic_prestate.state_hash:
        raise PathologyWorkflowError("canonical m64 1/16 state no longer matches its frozen prestate")
    same_dt_rows, same_dt_summary, same_dt_evidence_rows = _same_prestate_different_dt(
        solver=m64_diagnostic_solver, prestate=common_prestate
    )
    _write_csv(
        output_root / "same_prestate_dt_audit.csv",
        [*same_dt_rows, *same_dt_evidence_rows],
        ("record_type", "divisor", "candidate_dt_s"),
    )

    precision_rows, precision = _precision_audit(cases=(
        ("m32", m32_diagnostic_solver, m32_diagnostic_prestate, m32.dt_s, m32_trace),
        ("m64", m64_diagnostic_solver, m64_diagnostic_prestate, m64.dt_s, m64_trace),
    ))
    _write_csv(output_root / "precision_conditioning_audit.csv", precision_rows, ("record_type", "case"))

    root_cause = _root_cause(
        prefailure=prefailure, m32_scan=m32_scan, m64_scan=m64_scan,
        m32_raw=m32_raw, m64_raw=m64_raw, precision=precision, transitions=topology_transitions,
    )
    root_cause["M16_AUTHORITY_STATUS"] = "CONVERGED_DIAGNOSTIC_ENDPOINT_NOT_DT_TO_ZERO_AUTHORITY"
    root_cause["TIME_REFERENCE_V2"] = "NOT_ASSIGNED"
    _write_json(output_root / "root_cause.json", root_cause)

    recommended = _next_method(str(root_cause["PRIMARY_ROOT_CAUSE"]))
    topology_jump_count = sum(bool(row.get("map_jump_observed")) for row in topology_transitions)
    persistent_local_deltas = [
        row for row in topology_transitions
        if bool(row.get("local_one_sided_F_delta_persistent"))
    ]
    all_persistent_local_deltas_are_single_cell_crossings = bool(persistent_local_deltas) and all(
        bool(row.get("departure_cell_crossing"))
        and bool(row.get("single_departure_cell_crossing"))
        for row in persistent_local_deltas
    )
    final_fields = {
        "STATUS": root_cause["TOP_LEVEL_STATUS"],
        "BRANCH": source["git_branch"],
        "COMMIT": source["git_commit"],
        "SOURCE_CLEAN": not bool(source["git_status"]),
        "TESTS": str(args.test_status),
        "PHASE_A_REPRODUCED": baseline["status"],
        "M16_STATUS": "CONVERGED_DIAGNOSTIC_ENDPOINT",
        "M16_PHASE_A_STATUS": runs[16].status,
        "M32_FAILURE_REPRODUCED": m32_replay["status"],
        "M64_FAILURE_REPRODUCED": m64_replay["status"],
        "COMMON_TIME_1_16": "AVAILABLE",
        "M16_M32_STATE_ERROR": common["maximum_errors_at_1_16"]["m16_m32"],
        "M16_M64_STATE_ERROR": common["maximum_errors_at_1_16"]["m16_m64"],
        "M32_M64_STATE_ERROR": common["maximum_errors_at_1_16"]["m32_m64"],
        "COMMON_TIME_TOPOLOGY": common["topology_at_1_16"],
        "PREFAILURE_PATH_CLASSIFICATION": prefailure,
        "M32_PRESTATE_HASH": m32_prestate["accepted_state_hash"],
        "M64_PRESTATE_HASH": m64_prestate["accepted_state_hash"],
        "M32_CLOSURE_BEHAVIOR": m32_raw["classification"],
        "M32_PERIOD": m32_raw["exact_period"],
        "M32_OBSERVED_CROSSINGS": m32_scan["NUMBER_OF_OBSERVED_SIGN_CROSSINGS"],
        "M32_TOPOLOGY_PARTITIONS": m32_scan["NUMBER_OF_TOPOLOGY_PARTITIONS"],
        "M64_CLOSURE_BEHAVIOR": m64_raw["classification"],
        "M64_PERIOD": m64_raw["exact_period"],
        "M64_OBSERVED_CROSSINGS": m64_scan["NUMBER_OF_OBSERVED_SIGN_CROSSINGS"],
        "M64_TOPOLOGY_PARTITIONS": m64_scan["NUMBER_OF_TOPOLOGY_PARTITIONS"],
        "M32_M64_FAILURE_CLASS": comparison_summary,
        "SAME_PRESTATE_DT16": same_dt_rows[0],
        "SAME_PRESTATE_DT32": same_dt_rows[1],
        "SAME_PRESTATE_DT64": same_dt_rows[2],
        "SAME_PRESTATE_DT128": same_dt_rows[3],
        "SMALLER_DT_CLOSURE_TREND": same_dt_summary["trend"],
        **precision,
        "CR1_TOPOLOGY_CROSSINGS": len(topology_transitions),
        "F_JUMPS_CORRELATED_WITH_CELL_CROSSINGS": all_persistent_local_deltas_are_single_cell_crossings,
        "CLOSURE_RELEVANT_F_JUMP_COUNT": root_cause["evidence"]["closure_relevant_jump_correlated_transition_count"],
        "PRIMARY_ROOT_CAUSE": root_cause["PRIMARY_ROOT_CAUSE"],
        "SECONDARY_ROOT_CAUSE": root_cause["SECONDARY_ROOT_CAUSE"],
        "M16_AUTHORITY_STATUS": "CONVERGED_DIAGNOSTIC_ENDPOINT_NOT_DT_TO_ZERO_AUTHORITY",
        "TIME_REFERENCE_V2": "NOT_ASSIGNED",
        "RECOMMENDED_NEXT_METHOD": recommended,
        "CR2_AUTHORIZED": False,
        "ROOT_SOLVER_REDESIGN_AUTHORIZED": False,
        "PF_SOURCE_MODIFIED": False,
        "CUDA_RERUN": False,
        "PHYSICAL_RETUNING": False,
        "GP_RELEASE_RUN": False,
        "TOP_5_FINDINGS": [
            "m16 remains a converged diagnostic endpoint, not a dt-to-zero authority.",
            "m32 and m64 failures were recreated from the frozen step-244 state without accepting a new root.",
            f"Pre-failure path classification: {prefailure}.",
            f"Primary diagnostic root cause: {root_cause['PRIMARY_ROOT_CAUSE']}.",
            "TIME_REFERENCE_V2 remains NOT_ASSIGNED; no CR1 ladder, cohort, implicit, PF/CUDA, or GP phase ran.",
        ],
        "P0_BLOCKERS": ["FINE_REFINEMENT_ENDPOINT_NOT_AVAILABLE", root_cause["TOP_LEVEL_STATUS"]],
        "NEXT_ACTION": "Await explicit authorization before implementing any recommended method change.",
        "KEY_REPORTS": [f"reports/{TASK_NAME}/{name}" for name in (
            "00_phase_a_reproduction.md", "05_scalar_map_topology.md", "10_root_cause_classification.md", "12_final_acceptance_report.md"
        )],
    }
    report_payloads: dict[str, Mapping[str, Any] | str] = {
        "00_phase_a_reproduction.md": {
            **baseline, "restart_checkpoint": str(restart_checkpoint), "restart_checkpoint_sha256": _sha256_file(restart_checkpoint),
            "restart_state_hash": RESTART_STATE_HASH, "formal_trace_sha256": _sha256_file(formal_trace),
            "formal_phase_a_csv": str(formal_phase_a), "formal_phase_a_sha256": _sha256_file(formal_phase_a),
            "accepted_rows": {str(m): list(runs[m].accepted_rows) for m in SUBSTEP_MULTIPLICITIES},
            "failure_records": {"m32": m32.failure, "m64": m64.failure}, "repeat_prestates": repeated,
        },
        "01_common_time_state_alignment.md": {"summary": common, "classification": prefailure, "comparison_row_count": len(common_rows)},
        "02_m32_failure_replay.md": {**m32_replay, "prestate": m32_prestate},
        "03_m64_failure_replay.md": {**m64_replay, "prestate": m64_prestate},
        "04_failure_closure_sequences.md": {
            "m32": {
                **m32_raw, "raw_rows": len(m32_trace), "original_path_rows": len(m32_output_trace),
                "original_p4_path": m32.failure,
            },
            "m64": {
                **m64_raw, "raw_rows": len(m64_trace), "original_path_rows": original_m64_count,
                "milestones": _milestones(m64_trace), "original_ordinary_path": m64.failure,
            },
        },
        "05_scalar_map_topology.md": {"m32": m32_scan, "m64": m64_scan, "transition_count": len(topology_transitions), "accepted_new_root": False},
        "06_m32_m64_comparison.md": {**comparison_summary, "comparison_rows": comparison_rows, "m32_failure_path": m32.failure["closure_path"], "m64_failure_path": m64.failure["closure_path"]},
        "07_same_state_different_dt.md": {"canonical_state": {key: value for key, value in canonical.items() if not key.startswith("_") and key not in {"population_array", "cdf"}}, "summary": same_dt_summary, "rows": same_dt_rows, "m128_production": False},
        "08_precision_conditioning.md": precision,
        "09_cr1_remap_nonsmoothness.md": {
            "transition_count": len(topology_transitions),
            "map_jump_count": topology_jump_count,
            "persistent_local_delta_count": len(persistent_local_deltas),
            "closure_relevant_map_jump_count": root_cause["evidence"]["closure_relevant_jump_correlated_transition_count"],
            "all_persistent_local_deltas_are_single_cell_crossings": all_persistent_local_deltas_are_single_cell_crossings,
            "root_cause_B_criterion": "both m32 and m64 must each show a single departure-cell crossing with unchanged trace topology, a locally persistent one-sided F delta under fixed topology narrowing, bitwise-repeatable endpoints, and intersection with that failure's actual raw-Picard update interval; global transitions outside the orbit are recorded but never assigned as the closure root cause",
        },
        "10_root_cause_classification.md": root_cause,
        "11_next_method_decision.md": {"RECOMMENDED_NEXT_METHOD": recommended, "implementation_this_run": False, "CR2_AUTHORIZED": False, "ROOT_SOLVER_REDESIGN_AUTHORIZED": False},
        "12_final_acceptance_report.md": final_fields,
        "13_reproduction_commands.md": {
            "command": f"PYTHONPATH=src {sys.executable} scripts/run_kwn_cr1_fine_substep_pathology_v1.py diagnose --restart-checkpoint {restart_checkpoint} --formal-trace-csv {formal_trace} --formal-phase-a-csv {formal_phase_a} --output-root <new-output-root> --report-root <new-report-root>",
            "frozen_restart_step": RESTART_STEP,
            "frozen_restart_state_hash": RESTART_STATE_HASH,
            "no_root_acceptance_or_policy_change": True,
        },
    }
    _write_reports(report_root=report_root, payloads=report_payloads)
    provenance = {
        "task_name": TASK_NAME, "top_status": final_fields["STATUS"], "source": source,
        "runtime": _runtime_provenance(), "frozen_canonical_snapshot": frozen_canonical_snapshot_provenance(),
        "formal_trace_csv": str(formal_trace), "formal_trace_sha256": _sha256_file(formal_trace),
        "formal_phase_a_csv": str(formal_phase_a), "formal_phase_a_sha256": _sha256_file(formal_phase_a),
        "restart_checkpoint": str(restart_checkpoint), "restart_checkpoint_sha256": _sha256_file(restart_checkpoint),
        "restart_state_hash": RESTART_STATE_HASH, "accepted_substep_archive_sha256": accepted_archive_sha,
        "validation_contract_hash": context.contract_hash, "fixture_hash": context.fixture_hash,
        "fixture_input_set_sha256": EXPECTED_FIXTURE_INPUT_SET_SHA256,
        "accepted_substep_count": len(accepted_index), "runner_sha256": _sha256_file(Path(__file__)),
        "pathology_module_sha256": _sha256_file(ROOT / "src/kwn_mvp/characteristic_pathology.py"),
        "characteristic_solver_sha256": _sha256_file(ROOT / "src/kwn_mvp/characteristic_reference.py"),
        "test_status": str(args.test_status), "baseline": baseline,
        "runtime_s": time.monotonic() - started,
        "scalar_root_fallback_used": False, "generic_root_selection_used": False,
        "time_reference_v2": "NOT_ASSIGNED", "pf_source_modified": False, "cuda_rerun": False,
        "physical_retuning": False, "gp_release": False,
    }
    _write_json(output_root / "analysis_provenance.json", provenance)
    print(json.dumps(_json_safe(final_fields), indent=2, sort_keys=True))
    return 0, final_fields


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    command = subparsers.add_parser("diagnose")
    command.add_argument("--restart-checkpoint", type=Path, required=True)
    command.add_argument("--formal-trace-csv", type=Path, required=True)
    command.add_argument("--formal-phase-a-csv", type=Path, required=True)
    command.add_argument("--output-root", type=Path, required=True)
    command.add_argument("--report-root", type=Path, required=True)
    command.add_argument("--allow-dirty-source", action="store_true")
    command.add_argument("--test-status", default="NOT_RUN_BY_PATHOLOGY_RUNNER")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "diagnose":
        raise AssertionError("unreachable command")
    try:
        code, _fields = _run(args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
