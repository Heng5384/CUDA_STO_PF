#!/usr/bin/env python3
"""Execute the narrow post-diagnosis CR1 closure-regression gate.

This runner is intentionally smaller than the time-reference workflow.  It
replays the frozen trajectory to the formally accepted state 244, requires the
new controller to close only step 245, repeats that accepted step from the
same immutable checkpoint, verifies a restart across it, and continues the
registered ``dt=0.015625 s`` trajectory through step 260.  It neither changes
the timestep, retries a rejected step, scans for other roots, nor invokes
PF/CUDA.

Passing this program is evidence only for the local nonlinear-closure gate.
It does not establish characteristic self-convergence, cohort parity, an
independent time reference, or implicit time inaccuracy.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import platform
import subprocess
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.characteristic_reference import (  # noqa: E402
    FIXED_POINT_ROOT_INVENTORY_RELATIVE_TOLERANCE,
    CharacteristicReferenceSolver,
)
from kwn_mvp.solver import SolverConfig  # noqa: E402
from scripts.diagnose_kwn_characteristic_closure_v2 import (  # noqa: E402
    DT_S,
    FORMAL_FAILURE_TRACE,
    FROZEN_CONTRACT_HASH,
    LAST_ACCEPTED_STEP,
    _compare_formal_step244_trace,
    _json_safe,
    _moments,
    _save_state244,
    _sha256_file,
    _state_signature,
    _write_csv,
    _write_json,
    _write_markdown,
)
from scripts.frozen_canonical_smooth_population_v1 import (  # noqa: E402
    build_frozen_canonical_context,
    frozen_canonical_snapshot_provenance,
)


TASK_NAME = "kwn_characteristic_closure_v2"
REQUIRED_BRANCH = "codex/kwn-characteristic-closure-v2"
FROZEN_START_COMMIT = "9269e07a2d0fafbf35be950b858373fb71b54e4b"
FROZEN_V1_CLOSURE_COMMIT = "2ee679437421c750ccf2f2173d208df73bf28475"
STEP_245 = LAST_ACCEPTED_STEP + 1
LOCAL_WINDOW_FIRST_STEP = 240
LOCAL_WINDOW_LAST_STEP = 260
REPEAT_COUNT = 10


class ClosureRegressionError(RuntimeError):
    """Raise a concrete, fail-closed local-closure gate result."""


def _git(*arguments: str) -> str:
    return subprocess.check_output(
        ["git", "-C", str(ROOT), *arguments], text=True, stderr=subprocess.DEVNULL
    ).strip()


def _source_identity() -> dict[str, Any]:
    """Require the isolated, clean descendant before numerical work starts."""

    identity = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status": _git("status", "--short"),
        "frozen_start_commit": _git("rev-parse", FROZEN_START_COMMIT),
    }
    frozen_start_ancestry = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", FROZEN_START_COMMIT, "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    frozen_v1_closure_ancestry = subprocess.run(
        ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", FROZEN_V1_CLOSURE_COMMIT, "HEAD"],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        check=False,
    ).returncode == 0
    identity["frozen_start_is_ancestor"] = frozen_start_ancestry
    identity["frozen_v1_closure_commit"] = FROZEN_V1_CLOSURE_COMMIT
    identity["frozen_v1_closure_is_ancestor"] = frozen_v1_closure_ancestry
    if identity["git_branch"] != REQUIRED_BRANCH:
        raise ClosureRegressionError(
            f"local closure regression requires branch {REQUIRED_BRANCH!r}"
        )
    if identity["git_status"]:
        raise ClosureRegressionError("local closure regression requires a clean source tree")
    if not frozen_start_ancestry:
        raise ClosureRegressionError("source does not descend from frozen characteristic-reference start")
    if not frozen_v1_closure_ancestry:
        raise ClosureRegressionError("source does not descend from frozen v1 closure commit")
    return identity


def _runtime_context() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }


def _diagnostic_row(
    solver: CharacteristicReferenceSolver,
    diagnostic: Any,
    *,
    edges_m: np.ndarray,
) -> dict[str, Any]:
    """Persist all acceptance facts for one committed physical macro step."""

    cells = solver._beta_cell_numbers()
    return {
        "step": int(diagnostic.step),
        "time_s": float(diagnostic.time_s),
        "dt_s": float(diagnostic.dt_s),
        "matrix_xB": float(diagnostic.matrix_xb),
        "midpoint_matrix_xB": float(diagnostic.midpoint_matrix_xb),
        "fixed_point_iterations": int(diagnostic.fixed_point_iterations),
        "fixed_point_picard_iterations": int(diagnostic.fixed_point_picard_iterations),
        "fixed_point_xb_residual": float(diagnostic.fixed_point_xb_residual),
        "fixed_point_xb_tolerance": float(diagnostic.fixed_point_xb_tolerance),
        "fixed_point_population_residual": float(diagnostic.fixed_point_population_residual),
        "fixed_point_cell_measure_residual": float(diagnostic.fixed_point_cell_measure_residual),
        "fixed_point_convergence_rate": float(diagnostic.fixed_point_convergence_rate),
        "fixed_point_convergence_mode": str(diagnostic.fixed_point_convergence_mode),
        "fixed_point_periodic_cycle_period": int(diagnostic.fixed_point_periodic_cycle_period),
        "fixed_point_bracketed_root_iterations": int(
            diagnostic.fixed_point_bracketed_root_iterations
        ),
        "fixed_point_bracket_initial_width": float(diagnostic.fixed_point_bracket_initial_width),
        "fixed_point_bracket_final_width": float(diagnostic.fixed_point_bracket_final_width),
        "fixed_point_bracket_left_xb": float(diagnostic.fixed_point_bracket_left_xb),
        "fixed_point_bracket_right_xb": float(diagnostic.fixed_point_bracket_right_xb),
        "fixed_point_bracket_left_signed_residual": float(
            diagnostic.fixed_point_bracket_left_signed_residual
        ),
        "fixed_point_bracket_right_signed_residual": float(
            diagnostic.fixed_point_bracket_right_signed_residual
        ),
        "fixed_point_root_trial_xb_residual": float(
            diagnostic.fixed_point_root_trial_xb_residual
        ),
        "fixed_point_root_verification_population_residual": float(
            diagnostic.fixed_point_root_verification_population_residual
        ),
        "fixed_point_root_verification_kind": str(
            diagnostic.fixed_point_root_verification_kind
        ),
        "inventory_relative_residual": float(diagnostic.inventory.relative_residual),
        "rmin_number_loss_m3": float(diagnostic.rmin_number_loss_m3),
        "rmin_mol_b_loss_mol_m3": float(diagnostic.rmin_mol_b_loss_mol_m3),
        "rmax_number_loss_m3": float(diagnostic.rmax_number_loss_m3),
        "remap_number_conservation_residual_m3": float(
            diagnostic.remap_number_conservation_residual_m3
        ),
        "lower_no_inflow_face_count": int(diagnostic.lower_no_inflow_face_count),
        "upper_no_inflow_face_count": int(diagnostic.upper_no_inflow_face_count),
        "nonfinite_cell_count": int(np.count_nonzero(~np.isfinite(cells))),
        "negative_cell_count": int(np.count_nonzero(cells < 0.0)),
        "state_array_hash": _state_signature(solver),
        **_moments(edges_m, cells),
    }


def _formal_trace_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """Match the exact frozen-v1 row schema used by the step-244 comparator."""

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
    return {field: row[field] for field in fields}


def _compare_full_formal_replay_trace(
    replay_rows: Sequence[Mapping[str, Any]], *, formal_trace_csv: Path
) -> dict[str, Any]:
    """Require bit-exact telemetry equality for every committed step 1--244.

    The v1 run has no archived full state at step 244.  Its complete CR1
    telemetry trace is available, however, so this check is deliberately
    stronger than a last-row comparison while remaining honest about the
    unavailable formal state archive.
    """

    if not formal_trace_csv.is_file():
        return {"status": "FAIL", "reason": "formal trace CSV is unreadable"}
    fields = tuple(_formal_trace_row(replay_rows[0]).keys()) if replay_rows else ()
    if len(replay_rows) != LAST_ACCEPTED_STEP:
        return {
            "status": "FAIL",
            "reason": f"replay has {len(replay_rows)} rows, expected {LAST_ACCEPTED_STEP}",
        }
    with formal_trace_csv.open(newline="", encoding="utf-8") as handle:
        source_rows = list(csv.DictReader(handle))
    formal_by_step: dict[int, Mapping[str, str]] = {}
    for row in source_rows:
        if row.get("policy") != "CR1_dt_0.015625s":
            continue
        try:
            step = int(row.get("step", ""))
        except ValueError:
            continue
        if 1 <= step <= LAST_ACCEPTED_STEP:
            if step in formal_by_step:
                return {"status": "FAIL", "reason": f"duplicate formal row for step {step}"}
            formal_by_step[step] = row
    missing = [step for step in range(1, LAST_ACCEPTED_STEP + 1) if step not in formal_by_step]
    if missing:
        return {"status": "FAIL", "reason": f"formal trace misses replay steps: {missing[:8]}"}

    mismatches: list[dict[str, Any]] = []
    for expected_step, observed in enumerate(replay_rows, start=1):
        formal = formal_by_step[expected_step]
        for field in fields:
            if field == "fixed_point_convergence_mode":
                equal = str(observed[field]) == formal[field]
            elif field in {"step", "fixed_point_iterations", "fixed_point_picard_iterations"}:
                equal = int(observed[field]) == int(formal[field])
            else:
                observed_value = float(observed[field])
                formal_value = float(formal[field])
                equal = (
                    observed_value == formal_value
                    or (math.isnan(observed_value) and math.isnan(formal_value))
                )
            if not equal:
                mismatches.append(
                    {
                        "step": expected_step,
                        "field": field,
                        "formal": formal[field],
                        "replay": observed[field],
                    }
                )
    return {
        "status": "PASS_EXACT_FORMAL_CR1_TRACE_1_TO_244_MATCH"
        if not mismatches
        else "FAIL",
        "formal_state_244_archive": "NOT_AVAILABLE_IN_V1_RUN_ROOT",
        "formal_trace_csv": str(formal_trace_csv),
        "compared_steps": [1, LAST_ACCEPTED_STEP],
        "compared_fields": list(fields),
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:32],
    }


def _arrays_equal(first: Mapping[str, np.ndarray], second: Mapping[str, np.ndarray]) -> bool:
    return set(first) == set(second) and all(
        np.array_equal(first[key], second[key]) for key in first
    )


def _replay_to_step(
    *,
    context: Any,
    final_step: int,
    capture_from_step: int | None = None,
) -> tuple[CharacteristicReferenceSolver, list[dict[str, Any]], list[dict[str, Any]]]:
    """Replay only registered fixed-dt CR1 steps from the canonical state."""

    solver = CharacteristicReferenceSolver(SolverConfig.from_mapping(context.mapping))
    formal_rows: list[dict[str, Any]] = []
    captured_rows: list[dict[str, Any]] = []
    for expected_step in range(1, final_step + 1):
        diagnostic = solver.advance_one(maximum_dt_s=DT_S)
        if diagnostic.step != expected_step:
            raise ClosureRegressionError(
                f"committed step is {diagnostic.step}, expected {expected_step}"
            )
        row = _diagnostic_row(solver, diagnostic, edges_m=context.edges_m)
        if expected_step <= LAST_ACCEPTED_STEP:
            formal_rows.append(_formal_trace_row(row))
        if capture_from_step is not None and expected_step >= capture_from_step:
            captured_rows.append(row)
    return solver, formal_rows, captured_rows


def _step245_acceptance(
    *,
    row: Mapping[str, Any],
    solver: CharacteristicReferenceSolver,
) -> dict[str, Any]:
    """Apply only the frozen, already-implemented scalar-root acceptance tests."""

    mode = str(row["fixed_point_convergence_mode"])
    residual = float(row["fixed_point_xb_residual"])
    matrix_xb = float(row["matrix_xB"])
    exact_tolerance = float(row["fixed_point_xb_tolerance"])
    left_xb = float(row["fixed_point_bracket_left_xb"])
    right_xb = float(row["fixed_point_bracket_right_xb"])
    left_residual = float(row["fixed_point_bracket_left_signed_residual"])
    right_residual = float(row["fixed_point_bracket_right_signed_residual"])
    checks = {
        "safeguarded_mode": mode == "SAFEGUARDED_SCALAR_ROOT_V1",
        "nonperiodic_safeguarded_root": int(row["fixed_point_periodic_cycle_period"]) == 0,
        "frozen_raw_picard_cap": int(row["fixed_point_picard_iterations"])
        == solver.fixed_point_max_iterations,
        "root_bisection_attempted": int(row["fixed_point_bracketed_root_iterations"]) > 0,
        "finite_ordered_bracket": all(
            math.isfinite(value)
            for value in (left_xb, right_xb, left_residual, right_residual)
        )
        and left_xb < right_xb,
        "strict_sign_changing_bracket": (
            left_residual < 0.0 < right_residual
            or right_residual < 0.0 < left_residual
        ),
        "same_x_immutable_replay": row["fixed_point_root_verification_kind"]
        == "SAME_X_IMMUTABLE_REPLAY",
        "original_xB_tolerance": residual <= exact_tolerance,
        "M0_M3_same_x_verification": float(
            row["fixed_point_root_verification_population_residual"]
        )
        <= solver._population_convergence_rtol,
        "cell_measure_same_x_verification": float(
            row["fixed_point_cell_measure_residual"]
        )
        <= solver._population_convergence_rtol,
        "inventory_relative_residual": float(row["inventory_relative_residual"])
        <= FIXED_POINT_ROOT_INVENTORY_RELATIVE_TOLERANCE,
        "finite_nonnegative_cells": int(row["nonfinite_cell_count"]) == 0
        and int(row["negative_cell_count"]) == 0,
        "committed_fixed_dt": float(row["dt_s"]) == DT_S,
    }
    return {
        "status": "PASS_STEP245_ACCEPTANCE" if all(checks.values()) else "FAIL_STEP245_ACCEPTANCE",
        "checks": checks,
        "root_xB": matrix_xb,
        "root_residual": residual,
        "root_tolerance": exact_tolerance,
        "root_bracket": {
            "left_xB": left_xb,
            "right_xB": right_xb,
            "left_signed_residual": left_residual,
            "right_signed_residual": right_residual,
        },
        "inventory_relative_residual": float(row["inventory_relative_residual"]),
        "root_method": mode,
        "root_verification_kind": row["fixed_point_root_verification_kind"],
        "number_of_admissible_roots": "UNRESOLVED_NO_INTERVAL_OR_ANALYTIC_ENCLOSURE",
        "step_rejection_required": "NOT_IMPLEMENTED_OR_TRIGGERED_BY_THIS_NARROW_FINAL_RAW_BRACKET_POLICY",
    }


def _repeat_step245(
    *,
    context: Any,
    state244_checkpoint: Path,
) -> dict[str, Any]:
    """Independently accept step 245 ten times from one exact checkpoint."""

    config = SolverConfig.from_mapping(context.mapping)
    rows: list[dict[str, Any]] = []
    reference_arrays: dict[str, np.ndarray] | None = None
    reference_hash = ""
    reference_mode = ""
    reference_verification_kind = ""
    reference_xb_residual = math.nan
    for repetition in range(1, REPEAT_COUNT + 1):
        solver = CharacteristicReferenceSolver.load_checkpoint(config=config, path=state244_checkpoint)
        diagnostic = solver.advance_one(maximum_dt_s=DT_S)
        row = _diagnostic_row(solver, diagnostic, edges_m=context.edges_m)
        arrays = solver.state_arrays()
        if reference_arrays is None:
            reference_arrays = arrays
            reference_hash = _state_signature(solver)
            reference_mode = str(row["fixed_point_convergence_mode"])
            reference_verification_kind = str(row["fixed_point_root_verification_kind"])
            reference_xb_residual = float(row["fixed_point_xb_residual"])
        row["repeat_index"] = repetition
        row["matches_repeat_1_bitwise"] = _arrays_equal(reference_arrays, arrays)
        row["matches_repeat_1_closure_fields"] = bool(
            row["fixed_point_convergence_mode"] == reference_mode
            and row["fixed_point_root_verification_kind"] == reference_verification_kind
            and float(row["fixed_point_xb_residual"]) == reference_xb_residual
        )
        rows.append(row)
    return {
        "status": "PASS_STEP245_BITWISE_REPEATABILITY"
        if all(
            bool(row["matches_repeat_1_bitwise"])
            and bool(row["matches_repeat_1_closure_fields"])
            for row in rows
        )
        else "FAIL_STEP245_REPEATABILITY",
        "repeat_count": REPEAT_COUNT,
        "reference_state_array_hash": reference_hash,
        "rows": rows,
    }


def _restart_across_step245(
    *,
    context: Any,
    state244_checkpoint: Path,
    output_root: Path,
) -> dict[str, Any]:
    """Compare continuous 244→260 to a checkpoint made after root-closed 245."""

    config = SolverConfig.from_mapping(context.mapping)
    continuous = CharacteristicReferenceSolver.load_checkpoint(config=config, path=state244_checkpoint)
    continuous_rows: list[dict[str, Any]] = []
    for expected_step in range(STEP_245, LOCAL_WINDOW_LAST_STEP + 1):
        diagnostic = continuous.advance_one(maximum_dt_s=DT_S)
        if diagnostic.step != expected_step:
            raise ClosureRegressionError("continuous restart path advanced an unexpected step")
        continuous_rows.append(_diagnostic_row(continuous, diagnostic, edges_m=context.edges_m))
    split = CharacteristicReferenceSolver.load_checkpoint(config=config, path=state244_checkpoint)
    split_245 = split.advance_one(maximum_dt_s=DT_S)
    checkpoint_245 = output_root / "state_245_safeguarded_root_checkpoint.npz"
    split.save_checkpoint(checkpoint_245)
    resumed = CharacteristicReferenceSolver.load_checkpoint(config=config, path=checkpoint_245)
    split_rows = [_diagnostic_row(split, split_245, edges_m=context.edges_m)]
    for expected_step in range(STEP_245 + 1, LOCAL_WINDOW_LAST_STEP + 1):
        diagnostic = resumed.advance_one(maximum_dt_s=DT_S)
        if diagnostic.step != expected_step:
            raise ClosureRegressionError("resumed restart path advanced an unexpected step")
        split_rows.append(_diagnostic_row(resumed, diagnostic, edges_m=context.edges_m))
    arrays_equal = _arrays_equal(continuous.state_arrays(), resumed.state_arrays())
    trace_fields = (
        "step",
        "time_s",
        "dt_s",
        "matrix_xB",
        "fixed_point_convergence_mode",
        "fixed_point_periodic_cycle_period",
        "fixed_point_xb_residual",
        "fixed_point_root_verification_kind",
        "state_array_hash",
    )
    diagnostics_equal = len(continuous_rows) == len(split_rows) and all(
        all(left[field] == right[field] for field in trace_fields)
        for left, right in zip(continuous_rows, split_rows)
    )
    return {
        "status": "PASS_CHARACTERISTIC_RESTART_ACROSS_SAFEGUARDED_ROOT"
        if arrays_equal and diagnostics_equal
        else "FAIL_CHARACTERISTIC_RESTART_ACROSS_SAFEGUARDED_ROOT",
        "checkpoint_245": str(checkpoint_245),
        "continuous_step245_mode": continuous_rows[0]["fixed_point_convergence_mode"],
        "split_step245_mode": split_245.fixed_point_convergence_mode,
        "continuous_state260_hash": _state_signature(continuous),
        "resumed_state260_hash": _state_signature(resumed),
        "compared_steps": [STEP_245, LOCAL_WINDOW_LAST_STEP],
        "arrays_equal_bitwise": arrays_equal,
        "trace_fields_equal": diagnostics_equal,
    }


def _local_window_gate(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    expected_steps = list(range(LOCAL_WINDOW_FIRST_STEP, LOCAL_WINDOW_LAST_STEP + 1))
    observed_steps = [int(row["step"]) for row in rows]
    allowed_modes = {
        "DIRECT",
        "IDENTITY",
        "BRACKETED_SCALAR_ROOT",
        "SAFEGUARDED_SCALAR_ROOT_V1",
    }
    checks = {
        "complete_registered_window": observed_steps == expected_steps,
        "fixed_dt_throughout": all(float(row["dt_s"]) == DT_S for row in rows),
        "finite_nonnegative_cells": all(
            int(row["nonfinite_cell_count"]) == 0 and int(row["negative_cell_count"]) == 0
            for row in rows
        ),
        "finite_inventory": all(
            math.isfinite(float(row["inventory_relative_residual"])) for row in rows
        ),
        "no_unknown_closure_mode": all(
            str(row["fixed_point_convergence_mode"]) in allowed_modes for row in rows
        ),
        "safeguarded_root_only_at_step245": [
            int(row["step"])
            for row in rows
            if row["fixed_point_convergence_mode"] == "SAFEGUARDED_SCALAR_ROOT_V1"
        ]
        == [STEP_245],
        "no_step_rejection_policy": True,
    }
    periodic_events = [
        {
            "step": int(row["step"]),
            "period": int(row["fixed_point_periodic_cycle_period"]),
            "mode": str(row["fixed_point_convergence_mode"]),
        }
        for row in rows
        if int(row["fixed_point_periodic_cycle_period"]) > 0
    ]
    checks["no_periodic_cycle_in_local_window"] = not periodic_events
    return {
        "status": "PASS_CHARACTERISTIC_LOCAL_CLOSURE_REGRESSION"
        if all(checks.values())
        else "FAIL_CHARACTERISTIC_LOCAL_CLOSURE_REGRESSION",
        "checks": checks,
        "periodic_events": periodic_events,
        "safeguarded_root_steps": [
            int(row["step"])
            for row in rows
            if row["fixed_point_convergence_mode"] == "SAFEGUARDED_SCALAR_ROOT_V1"
        ],
        "maximum_inventory_relative_residual": max(
            float(row["inventory_relative_residual"]) for row in rows
        ),
        "maximum_xB_residual": max(float(row["fixed_point_xb_residual"]) for row in rows),
    }


def _write_reports(
    *,
    report_root: Path,
    output_root: Path,
    final: Mapping[str, Any],
    state244: Mapping[str, Any],
    formal_replay: Mapping[str, Any],
    acceptance: Mapping[str, Any],
    repeatability: Mapping[str, Any],
    restart: Mapping[str, Any],
    local_gate: Mapping[str, Any],
    command: str,
) -> None:
    payload = lambda value: "```json\n" + json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n```\n"
    _write_markdown(
        report_root / "05_minimal_closure_repair.md",
        "Minimal closure repair",
        "The controller is restricted to the final adjacent raw-Picard strict-sign pair after the frozen cap, "
        "requires one unchanged CDF/remap partition and time-of-flight topology, and verifies the same binary64 "
        "root trial without applying a successor Picard map.\n\n" + payload(final["closure_contract"]),
    )
    _write_markdown(
        report_root / "06_step245_acceptance.md",
        "Step-245 acceptance",
        payload({"state_244": state244, "formal_step244_replay": formal_replay, "acceptance": acceptance, "repeatability": repeatability, "restart": restart}),
    )
    _write_markdown(
        report_root / "07_local_240_260_regression.md",
        "Local 240–260 regression",
        payload({"local_gate": local_gate, "trace_csv": str(output_root / "local_240_260_trace.csv")}),
    )
    for filename, title, status in (
        ("08_characteristic_self_convergence.md", "Characteristic self-convergence", "BLOCKED_BY_LOCAL_CLOSURE_GATE_OR_NOT_RUN"),
        ("09_cohort_characteristic_parity.md", "Cohort–characteristic parity", "BLOCKED_BY_CHARACTERISTIC_SELF_CONVERGENCE"),
        ("10_implicit_reference_adjudication.md", "Implicit reference adjudication", "BLOCKED_NO_TIME_REFERENCE_V2"),
    ):
        _write_markdown(report_root / filename, title, f"Status: `{status}`.\n")
    _write_markdown(
        report_root / "11_model_boundary.md",
        "Model boundary",
        "This KWN-only regression modifies no PF/CUDA source, launches no CUDA workload, changes no physical "
        "parameter/canonical measure/thermodynamic or growth law, performs no clamp, and adds no GP path. "
        "The strict full-domain donor-bound SSPRK2 reference remains blocked.\n",
    )
    _write_markdown(report_root / "12_final_acceptance_report.md", "Final acceptance report", payload(final))
    _write_markdown(
        report_root / "13_reproduction_commands.md",
        "Reproduction commands",
        "```bash\n" + command + "\n```\n",
    )


def _workflow(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = arguments.output_root.resolve()
    report_root = arguments.report_root.resolve()
    if output_root.exists() or report_root.exists():
        raise ClosureRegressionError("output and report roots must not already exist")
    if float(arguments.dt_s) != DT_S:
        raise ClosureRegressionError(f"closure regression must preserve frozen dt_s={DT_S}")
    started = time.monotonic()
    source = _source_identity()
    context = build_frozen_canonical_context()
    if context.contract_hash != FROZEN_CONTRACT_HASH:
        raise ClosureRegressionError("frozen canonical context has an unexpected validation-contract hash")
    output_root.mkdir(parents=True, exist_ok=False)
    report_root.mkdir(parents=True, exist_ok=False)
    solver: CharacteristicReferenceSolver | None = None
    local_rows: list[dict[str, Any]] = []
    state244: Mapping[str, Any] | None = None
    formal_replay: Mapping[str, Any] | None = None
    try:
        state244_solver, state244_formal_rows, local_rows = _replay_to_step(
            context=context,
            final_step=LAST_ACCEPTED_STEP,
            capture_from_step=LOCAL_WINDOW_FIRST_STEP,
        )
        state244 = _save_state244(
            output_root / "state_244.npz",
            solver=state244_solver,
            context=context,
            replay_trace=state244_formal_rows,
        )
        state244_checkpoint = output_root / "checkpoint_step244.npz"
        state244_solver.save_checkpoint(state244_checkpoint)
        restored_state244 = CharacteristicReferenceSolver.load_checkpoint(
            config=SolverConfig.from_mapping(context.mapping), path=state244_checkpoint
        )
        checkpoint_state_matches_archive = _arrays_equal(
            state244_solver.state_arrays(), restored_state244.state_arrays()
        )
        if not checkpoint_state_matches_archive:
            raise ClosureRegressionError("restartable step-244 checkpoint differs from its frozen replay state")
        state244_checkpoint_record = {
            "checkpoint_file": str(state244_checkpoint),
            "checkpoint_sha256": _sha256_file(state244_checkpoint),
            "checkpoint_state_array_hash": _state_signature(restored_state244),
            "archive_state_array_hash": state244["state_244_array_hash"],
            "matches_archive_state_arrays_bitwise": checkpoint_state_matches_archive,
        }
        formal_step244_replay = _compare_formal_step244_trace(
            state244_formal_rows, formal_trace_csv=arguments.formal_trace_csv
        )
        formal_full_replay = _compare_full_formal_replay_trace(
            state244_formal_rows, formal_trace_csv=arguments.formal_trace_csv
        )
        formal_replay = {
            "last_step_244": formal_step244_replay,
            "full_steps_1_to_244": formal_full_replay,
        }
        if (
            formal_step244_replay.get("status") != "PASS_EXACT_FORMAL_TRACE_MATCH"
            or formal_full_replay.get("status") != "PASS_EXACT_FORMAL_CR1_TRACE_1_TO_244_MATCH"
        ):
            raise ClosureRegressionError("step-244 replay differs from the frozen formal trace")
        solver = CharacteristicReferenceSolver.load_checkpoint(
            config=SolverConfig.from_mapping(context.mapping), path=state244_checkpoint
        )
        for expected_step in range(STEP_245, LOCAL_WINDOW_LAST_STEP + 1):
            diagnostic = solver.advance_one(maximum_dt_s=DT_S)
            if diagnostic.step != expected_step:
                raise ClosureRegressionError(
                    f"committed step is {diagnostic.step}, expected {expected_step}"
                )
            local_rows.append(_diagnostic_row(solver, diagnostic, edges_m=context.edges_m))
        row245 = next((row for row in local_rows if int(row["step"]) == STEP_245), None)
        if row245 is None:
            raise ClosureRegressionError("local window omitted candidate step 245")
        acceptance = _step245_acceptance(row=row245, solver=solver)
        repeatability = _repeat_step245(context=context, state244_checkpoint=state244_checkpoint)
        restart = _restart_across_step245(
            context=context, state244_checkpoint=state244_checkpoint, output_root=output_root
        )
        local_gate = _local_window_gate(local_rows)
        _write_csv(
            output_root / "local_240_260_trace.csv",
            local_rows,
            fallback_fields=("step", "time_s", "fixed_point_convergence_mode"),
        )
        _write_csv(
            output_root / "step245_repeatability.csv",
            repeatability["rows"],
            fallback_fields=("repeat_index", "state_array_hash", "matches_repeat_1_bitwise"),
        )
        _write_json(output_root / "restart_across_step245.json", restart)
        closure_contract = {
            "mode": "SAFEGUARDED_SCALAR_ROOT_V1",
            "trigger": "FINAL_ADJACENT_RAW_PICARD_SAME_REMAP_BRANCH_STRICT_SIGN_BRACKET_ONLY",
            "verification": "SAME_X_IMMUTABLE_REPLAY",
            "no_global_scan_or_history_bracket": True,
            "no_picard_cap_expansion": True,
            "no_timestep_rejection_or_halving": True,
            "strict_ssprk2_status": "BLOCKED_EXACT_DONOR_BOUND_REFERENCE",
        }
        passed = (
            acceptance["status"] == "PASS_STEP245_ACCEPTANCE"
            and repeatability["status"] == "PASS_STEP245_BITWISE_REPEATABILITY"
            and restart["status"] == "PASS_CHARACTERISTIC_RESTART_ACROSS_SAFEGUARDED_ROOT"
            and local_gate["status"] == "PASS_CHARACTERISTIC_LOCAL_CLOSURE_REGRESSION"
        )
        final = {
            "STATUS": "PASS_CHARACTERISTIC_LOCAL_CLOSURE_REGRESSION"
            if passed
            else "FAIL_CHARACTERISTIC_LOCAL_CLOSURE_REGRESSION",
            "STEP244_REPLAY": "PASS_EXACT_FORMAL_CR1_TRACE_1_TO_244_MATCH_STATE_ARCHIVE_UNAVAILABLE",
            "STEP244_STATE_HASH": state244["state_244_sha256"],
            "STEP244_RESTARTABLE_CHECKPOINT": state244_checkpoint_record,
            "STEP245_CLASSIFICATION": "OTHER_WITH_EXPLICIT_EVIDENCE",
            "STEP245_CLASSIFICATION_EVIDENCE": "NONCONTRACTIVE_RAW_PICARD_WITH_OBSERVED_SAME_BRANCH_ROOT",
            "SCALAR_ROOT_EXISTS": "OBSERVED_NATURAL_FINAL_RAW_ADJACENT_SAME_BRANCH_BRACKET_ONLY",
            "NUMBER_OF_ADMISSIBLE_ROOTS": "UNRESOLVED_NO_INTERVAL_OR_ANALYTIC_ENCLOSURE",
            "STEP_REJECTION_REQUIRED": acceptance["step_rejection_required"],
            "STEP245_ACCEPTANCE": acceptance["status"],
            "STEP245_INVENTORY_RESIDUAL": acceptance["inventory_relative_residual"],
            "STEP245_REPEATABILITY": repeatability["status"],
            "CHARACTERISTIC_RESTART": restart["status"],
            "LOCAL_240_260_GATE": local_gate["status"],
            "CHARACTERISTIC_SELF_CONVERGENCE": "NOT_RUN",
            "TIME_REFERENCE_V2": "NOT_ASSIGNED",
            "IMPLICIT_TIME_INACCURACY_PROVEN": False,
            "PF_SOURCE_MODIFIED": False,
            "CUDA_RERUN": False,
            "PHYSICAL_RETUNING": False,
            "GP_RELEASE_RUN": False,
            "closure_contract": closure_contract,
            "source": source,
            "runtime": _runtime_context(),
            "validation_contract_hash": context.contract_hash,
            "canonical_snapshot": frozen_canonical_snapshot_provenance(),
            "formal_step244_replay": formal_replay,
            "step244_restartable_checkpoint": state244_checkpoint_record,
            "step245_acceptance": acceptance,
            "step245_repeatability": {
                key: value for key, value in repeatability.items() if key != "rows"
            },
            "restart": restart,
            "local_window": local_gate,
            "wall_seconds": time.monotonic() - started,
        }
        _write_json(output_root / "analysis_provenance.json", final)
        _write_reports(
            report_root=report_root,
            output_root=output_root,
            final=final,
            state244={**state244, "restartable_checkpoint": state244_checkpoint_record},
            formal_replay=formal_replay,
            acceptance=acceptance,
            repeatability={key: value for key, value in repeatability.items() if key != "rows"},
            restart=restart,
            local_gate=local_gate,
            command=" ".join(sys.argv),
        )
        return final
    except Exception as error:
        failed_candidate_step = int(solver.step) + 1 if solver is not None else None
        failure_status = (
            "FAIL_STEP244_DETERMINISTIC_REPLAY"
            if "step-244" in str(error)
            else "FAIL_CHARACTERISTIC_LOCAL_CLOSURE_REGRESSION"
        )
        failure = {
            "STATUS": failure_status,
            "reason": f"{type(error).__name__}: {error}",
            "last_committed_step": int(solver.step) if solver is not None else None,
            "failed_candidate_step": failed_candidate_step,
            "partial_local_window_steps": [int(row["step"]) for row in local_rows],
            "state_244": state244,
            "formal_step244_replay": formal_replay,
            "source": source,
            "runtime": _runtime_context(),
            "validation_contract_hash": context.contract_hash,
            "TIME_REFERENCE_V2": "NOT_ASSIGNED",
            "IMPLICIT_TIME_INACCURACY_PROVEN": False,
            "PF_SOURCE_MODIFIED": False,
            "CUDA_RERUN": False,
            "PHYSICAL_RETUNING": False,
            "GP_RELEASE_RUN": False,
        }
        if local_rows:
            _write_csv(
                output_root / "local_240_260_trace.csv",
                local_rows,
                fallback_fields=("step", "time_s", "fixed_point_convergence_mode"),
            )
        _write_json(output_root / "analysis_provenance.json", failure)
        _write_markdown(
            report_root / "07_local_240_260_regression.md",
            "Local 240–260 regression",
            "```json\n" + json.dumps(_json_safe(failure), indent=2, sort_keys=True) + "\n```\n",
        )
        _write_markdown(
            report_root / "12_final_acceptance_report.md",
            "Final acceptance report",
            "```json\n" + json.dumps(_json_safe(failure), indent=2, sort_keys=True) + "\n```\n",
        )
        return failure


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("closure-regression",))
    parser.add_argument("--output-root", type=Path, default=ROOT / "outputs" / TASK_NAME)
    parser.add_argument("--report-root", type=Path, default=ROOT / "reports" / TASK_NAME)
    parser.add_argument("--dt-s", type=float, default=DT_S)
    parser.add_argument("--formal-trace-csv", type=Path, default=Path(FORMAL_FAILURE_TRACE))
    arguments = parser.parse_args()
    try:
        final = _workflow(arguments)
    except Exception as error:
        print(
            json.dumps(
                {
                    "STATUS": "FAIL_CHARACTERISTIC_LOCAL_CLOSURE_REGRESSION",
                    "reason": f"{type(error).__name__}: {error}",
                },
                indent=2,
                sort_keys=True,
            )
        )
        return 2
    print(json.dumps(_json_safe(final), indent=2, sort_keys=True))
    return 0 if final["STATUS"] == "PASS_CHARACTERISTIC_LOCAL_CLOSURE_REGRESSION" else 2


if __name__ == "__main__":
    raise SystemExit(main())
