#!/usr/bin/env python3
"""KWN CR1 reject-and-halve characteristic continuation, phase A.

This runner intentionally separates three authority stages.  ``local-window``
is the only executable production stage in this revision: it reconstructs the
uncontested step-244 state, establishes a fixed substep reference for step
245, and advances the 245--260 physical window with
``CHARACTERISTIC_DT_CONTINUATION_V1``.  It never calls the legacy final raw
Picard safeguard on an authority candidate.  The CR1 nominal time ladder and
cohort parity remain separately gated commands; this program does not chain
them automatically.

The legacy safeguarded step-245 calculation is run, when requested by the
local-window audit, only as an explicitly labelled diagnostic comparison.  It
is not used to seed, repair, or accept the continuation trajectory.
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
import tempfile
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.characteristic_dt_continuation import (  # noqa: E402
    POLICY_NAME,
    CharacteristicDtContinuation,
    DtContinuationError,
    DtContinuationFatalStepError,
    DtContinuationMinDtError,
    accepted_state_hash,
)
from kwn_mvp.characteristic_reference import (  # noqa: E402
    CharacteristicReferenceError,
    CharacteristicReferenceSolver,
    CharacteristicStepDiagnostics,
)
from kwn_mvp.diagnostics import discrete_wasserstein_distance  # noqa: E402
from kwn_mvp.population_metrics import (  # noqa: E402
    PopulationMetrics,
    metrics_from_piecewise_constant_cells,
    positive_cell_quadrature,
)
from kwn_mvp.solver import RadiusGridOverflowError, SolverConfig  # noqa: E402
from scripts.frozen_canonical_smooth_population_v1 import (  # noqa: E402
    build_frozen_canonical_context,
    frozen_canonical_snapshot_provenance,
)


TASK_NAME = "kwn_characteristic_dt_continuation_v1"
REQUIRED_BRANCH = "codex/kwn-characteristic-dt-continuation-v1"
CONTINUATION_START_COMMIT = "e372a6e6b3401db697bf6a9d1a8a5d70fb27ec8a"
V1_CLOSURE_COMMIT = "2ee679437421c750ccf2f2173d208df73bf28475"
FROZEN_START_COMMIT = "9269e07a2d0fafbf35be950b858373fb71b54e4b"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / TASK_NAME
DEFAULT_REPORT_ROOT = ROOT / "reports" / TASK_NAME
DEFAULT_FORMAL_TRACE = Path(
    "/data/home/luozhiheng/tmp/"
    "kwn_characteristic_reference_v1_2ee679437421_20260902T202810Z/"
    "formal_all_14d_v4/outputs/kwn_characteristic_reference_v1/"
    "characteristic_fixed_point_trace.csv"
)

DT0_S = 0.015625
FORMAL_LAST_ACCEPTED_STEP = 244
STEP245 = 245
WINDOW_LAST_OLD_STEP = 260
STEP245_LADDER = (1, 2, 4, 8, 16, 32)
OPTIONAL_STEP245_M = 64
CONTINUOUS_METRIC_GATE = 0.0025
OLD_ROOT_XB = 0.0062176557756607005
LOCAL_ALLOWED_MODES = {"DIRECT", "IDENTITY", "BRACKETED_SCALAR_ROOT"}
LOCAL_ALLOWED_PERIODS = {2, 4}
PRIMARY_CONTINUOUS_METRICS = (
    "M0_m3",
    "M1_m2",
    "M2_m",
    "M3_dimensionless",
    "Rmean_m",
    "Rmean3_m3",
    "Sv_m_inv",
    "f_beta",
    "matrix_xB",
)

REPORT_TITLES = {
    "00_authority_boundary.md": "Authority boundary",
    "01_restart_state_selection.md": "Uncontested restart-state selection",
    "02_dt_continuation_contract.md": "Characteristic dt-continuation contract",
    "03_step245_substep_ladder.md": "Step-245 fixed substep ladder",
    "04_step245_endpoint_convergence.md": "Step-245 endpoint convergence",
    "05_old_root_continuation_comparison.md": "Legacy step-245 root comparison",
    "06_local_240_260_continuation.md": "Local physical 240--260 continuation",
    "07_refinement_frequency.md": "Local refinement frequency",
    "08_restart_qualification.md": "Continuation restart qualification",
    "09_cr1_nominal_timestep_ladder.md": "CR1 nominal timestep ladder",
    "10_characteristic_self_convergence.md": "Characteristic self-convergence",
    "11_cohort_characteristic_parity.md": "Cohort--characteristic parity",
    "12_model_role_boundary.md": "Model-role boundary",
    "13_final_acceptance_report.md": "Final acceptance report",
    "14_reproduction_commands.md": "Reproduction commands",
}


class ContinuationWorkflowError(RuntimeError):
    """A fail-closed orchestration error, never a request for a root hack."""


@dataclass(frozen=True)
class Endpoint:
    """A fully accepted fixed-substep endpoint and its retained state."""

    m: int
    status: str
    reason: str | None
    solver: CharacteristicReferenceSolver | None
    rows: tuple[dict[str, Any], ...]
    snapshot: dict[str, Any] | None
    cells: np.ndarray | None


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
        json.dumps(_json_safe(dict(value)), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_markdown(path: Path, title: str, value: Mapping[str, Any] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(value, str):
        body = value.rstrip()
    else:
        body = "```json\n" + json.dumps(_json_safe(dict(value)), indent=2, sort_keys=True) + "\n```"
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


def _source_identity(*, require_clean: bool = True) -> dict[str, Any]:
    source = {
        "git_commit": _git("rev-parse", "HEAD"),
        "git_branch": _git("branch", "--show-current"),
        "git_status": _git("status", "--short"),
        "continuation_start_commit": CONTINUATION_START_COMMIT,
        "v1_closure_commit": V1_CLOSURE_COMMIT,
        "frozen_start_commit": FROZEN_START_COMMIT,
    }
    for label, commit in (
        ("continuation_start_is_ancestor", CONTINUATION_START_COMMIT),
        ("v1_closure_is_ancestor", V1_CLOSURE_COMMIT),
        ("frozen_start_is_ancestor", FROZEN_START_COMMIT),
    ):
        source[label] = subprocess.run(
            ["git", "-C", str(ROOT), "merge-base", "--is-ancestor", commit, "HEAD"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
        ).returncode == 0
    if source["git_branch"] != REQUIRED_BRANCH:
        raise ContinuationWorkflowError(f"task requires branch {REQUIRED_BRANCH}")
    if require_clean and source["git_status"]:
        raise ContinuationWorkflowError("task requires a clean source tree")
    if not all(bool(source[key]) for key in (
        "continuation_start_is_ancestor", "v1_closure_is_ancestor", "frozen_start_is_ancestor"
    )):
        raise ContinuationWorkflowError("source does not descend from all frozen continuation ancestors")
    return source


def _runtime_provenance() -> dict[str, Any]:
    return {
        "hostname": platform.node(),
        "platform": platform.platform(),
        "python_executable": sys.executable,
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }


def _metrics_dict(metrics: PopulationMetrics) -> dict[str, float]:
    return {
        "M0_m3": float(metrics.M0_m3),
        "M1_m2": float(metrics.M1_m2),
        "M2_m": float(metrics.M2_m),
        "M3_dimensionless": float(metrics.M3_dimensionless),
        "Rmean_m": float(metrics.Rmean_number_m),
        "Rmean3_m3": float(metrics.Rmean_cubed_m3),
        "mean_R3_m3": float(metrics.mean_R3_m3),
        "Sv_m_inv": float(metrics.Sv_m_inv),
        "f_beta": float(metrics.f_beta),
    }


def _snapshot(solver: CharacteristicReferenceSolver, *, policy: str) -> tuple[dict[str, Any], np.ndarray]:
    cells = np.asarray(solver._beta_cell_numbers(), dtype=np.float64)
    metrics = metrics_from_piecewise_constant_cells(solver.population("beta").grid.edges_m, cells)
    inventory = solver.ledger.snapshot(
        matrix_xb=solver.matrix_xb,
        populations=solver.population_list(),
        beta_resolved_fraction=1.0,
    )
    return ({
        "policy": policy,
        "step": int(solver.step),
        "time_s": float(solver.time_s),
        "time_h": float(solver.time_s / 3600.0),
        "matrix_xB": float(solver.matrix_xb),
        "Q_total_mol_m3": float(inventory.total_mol_m3),
        "Q_beta_mol_m3": float(inventory.beta_resolved_mol_m3),
        "Q_matrix_mol_m3": float(inventory.matrix_mol_m3),
        "inventory_residual_mol_m3": float(inventory.residual_mol_m3),
        "inventory_relative_residual": float(inventory.relative_residual),
        "cumulative_number_dissolution_m3": float(solver.cumulative_number_dissolution_m3),
        "cumulative_beta_volume_dissolution": float(solver.cumulative_beta_volume_dissolution),
        "cumulative_mol_B_returned_mol_m3": float(solver.cumulative_mol_b_returned_mol_m3),
        "state_array_hash": accepted_state_hash(solver),
        "population_array_hash": _array_hash(cells),
        "nonfinite_cell_count": int(np.count_nonzero(~np.isfinite(cells))),
        "negative_cell_count": int(np.count_nonzero(cells < 0.0)),
        **_metrics_dict(metrics),
    }, cells)


def _array_hash(array: np.ndarray) -> str:
    values = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
    return hashlib.sha256(values.tobytes()).hexdigest()


def _diagnostic_row(
    solver: CharacteristicReferenceSolver,
    diagnostic: CharacteristicStepDiagnostics,
    *,
    policy: str,
    macro_old_step: int | None = None,
    refinement_depth: int | None = None,
) -> dict[str, Any]:
    snapshot, _cells = _snapshot(solver, policy=policy)
    return {
        "policy": policy,
        "macro_old_step": "" if macro_old_step is None else int(macro_old_step),
        "refinement_depth": "" if refinement_depth is None else int(refinement_depth),
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
        "fixed_point_root_verification_kind": str(diagnostic.fixed_point_root_verification_kind),
        "inventory_relative_residual": float(diagnostic.inventory.relative_residual),
        "Q_total_mol_m3": float(diagnostic.inventory.total_mol_m3),
        "Q_beta_mol_m3": float(diagnostic.inventory.beta_resolved_mol_m3),
        "Q_matrix_mol_m3": float(diagnostic.inventory.matrix_mol_m3),
        "rmin_number_loss_m3": float(diagnostic.rmin_number_loss_m3),
        "rmin_mol_b_loss_mol_m3": float(diagnostic.rmin_mol_b_loss_mol_m3),
        "remap_number_conservation_residual_m3": float(diagnostic.remap_number_conservation_residual_m3),
        "state_array_hash": snapshot["state_array_hash"],
        "population_array_hash": snapshot["population_array_hash"],
        "nonfinite_cell_count": snapshot["nonfinite_cell_count"],
        "negative_cell_count": snapshot["negative_cell_count"],
        "M0": snapshot["M0_m3"],
        "M1": snapshot["M1_m2"],
        "M2": snapshot["M2_m"],
        "M3": snapshot["M3_dimensionless"],
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
        return {"status": "FAIL_STEP244_DETERMINISTIC_REPLAY", "reason": "formal trace CSV is unreadable"}
    if len(rows) != FORMAL_LAST_ACCEPTED_STEP:
        return {
            "status": "FAIL_STEP244_DETERMINISTIC_REPLAY",
            "reason": f"replay produced {len(rows)} rows instead of 244",
        }
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
        return {
            "status": "FAIL_STEP244_DETERMINISTIC_REPLAY",
            "reason": f"formal trace misses steps {missing[:8]}",
        }
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
                left, right = float(observed[field]), float(expected[field])
                equal = left == right or (math.isnan(left) and math.isnan(right))
            if not equal:
                mismatches.append({"step": step, "field": field, "formal": expected[field], "replay": observed[field]})
    return {
        "status": "PASS_STEP244_DETERMINISTIC_REPLAY" if not mismatches else "FAIL_STEP244_DETERMINISTIC_REPLAY",
        "formal_trace_csv": str(trace_path),
        "formal_trace_sha256": _sha256_file(trace_path),
        "formal_state_244_archive": "NOT_AVAILABLE_IN_V1_RUN_ROOT",
        "mismatch_count": len(mismatches),
        "mismatches": mismatches[:32],
    }


def _allowed_diagnostic(diagnostic: CharacteristicStepDiagnostics) -> bool:
    mode = str(diagnostic.fixed_point_convergence_mode)
    period = int(diagnostic.fixed_point_periodic_cycle_period)
    if mode not in LOCAL_ALLOWED_MODES:
        return False
    if mode == "BRACKETED_SCALAR_ROOT":
        return period in LOCAL_ALLOWED_PERIODS and int(diagnostic.fixed_point_bracketed_root_iterations) > 0
    return period == 0 and int(diagnostic.fixed_point_bracketed_root_iterations) == 0


def _validate_authority_diagnostic(diagnostic: CharacteristicStepDiagnostics) -> None:
    if not _allowed_diagnostic(diagnostic):
        raise ContinuationWorkflowError(
            "ordinary/P2/P4-only authority closure was violated by "
            f"{diagnostic.fixed_point_convergence_mode}/P{diagnostic.fixed_point_periodic_cycle_period}"
        )
    for value, label in (
        (diagnostic.fixed_point_xb_residual, "xB residual"),
        (diagnostic.fixed_point_xb_tolerance, "xB tolerance"),
        (diagnostic.fixed_point_population_residual, "population residual"),
        (diagnostic.fixed_point_cell_measure_residual, "cell-measure residual"),
        (diagnostic.inventory.relative_residual, "inventory residual"),
    ):
        if not math.isfinite(float(value)):
            raise ContinuationWorkflowError(f"accepted authority diagnostic has non-finite {label}")
    if float(diagnostic.fixed_point_xb_residual) > float(diagnostic.fixed_point_xb_tolerance):
        raise ContinuationWorkflowError("accepted authority diagnostic violates its frozen xB tolerance")


def _replay_to_step244(
    context: Any,
    *,
    formal_trace_csv: Path,
    checkpoint_path: Path,
) -> tuple[CharacteristicReferenceSolver, list[dict[str, Any]], dict[str, Any]]:
    solver = CharacteristicReferenceSolver(SolverConfig.from_mapping(context.mapping))
    rows: list[dict[str, Any]] = []
    for expected_step in range(1, FORMAL_LAST_ACCEPTED_STEP + 1):
        diagnostic = solver.advance_one_ordinary_or_qualified_cycle(maximum_dt_s=DT0_S)
        if diagnostic.step != expected_step:
            raise ContinuationWorkflowError(
                f"formal replay committed step {diagnostic.step}, expected {expected_step}"
            )
        _validate_authority_diagnostic(diagnostic)
        rows.append(_formal_row(_diagnostic_row(solver, diagnostic, policy="CR1_dt_0.015625s")))
    formal = _compare_formal_trace(rows, formal_trace_csv)
    if formal["status"] != "PASS_STEP244_DETERMINISTIC_REPLAY":
        raise ContinuationWorkflowError(str(formal.get("reason", "formal replay mismatch")))
    if checkpoint_path.exists():
        raise ContinuationWorkflowError("refusing to overwrite step-244 continuation checkpoint")
    solver.save_checkpoint(checkpoint_path)
    formal.update({
        "restart_checkpoint": str(checkpoint_path),
        "restart_checkpoint_sha256": _sha256_file(checkpoint_path),
        "continuation_restart_step": int(solver.step),
        "continuation_restart_state_hash": accepted_state_hash(solver),
    })
    return solver, rows, formal


def _endpoint_from_checkpoint(
    *,
    config: SolverConfig,
    checkpoint_path: Path,
    m: int,
) -> Endpoint:
    if m <= 0:
        raise ValueError("substep multiplicity must be positive")
    solver = CharacteristicReferenceSolver.load_checkpoint(config=config, path=checkpoint_path)
    rows: list[dict[str, Any]] = []
    dt_s = DT0_S / float(m)
    for substep in range(1, m + 1):
        before_hash = accepted_state_hash(solver)
        before_history = tuple(solver.history)
        try:
            diagnostic = solver.advance_one_ordinary_or_qualified_cycle(maximum_dt_s=dt_s)
        except (CharacteristicReferenceError, RadiusGridOverflowError, ValueError, FloatingPointError) as error:
            unchanged = accepted_state_hash(solver) == before_hash and tuple(solver.history) == before_history
            if not unchanged:
                raise ContinuationWorkflowError(
                    "a rejected fixed-substep trial mutated the immutable accepted state "
                    f"at substep={substep}/{m}"
                ) from error
            status = "FULL_STEP_NOT_ADMISSIBLE" if m == 1 else "NONCLOSING_SUBSTEP"
            return Endpoint(
                m=m,
                status=status,
                reason=(
                    f"substep={substep}/{m}; state_immutable_after_rejection={unchanged}; "
                    f"{type(error).__name__}: {error}"
                ),
                solver=None,
                rows=tuple(rows),
                snapshot=None,
                cells=None,
            )
        _validate_authority_diagnostic(diagnostic)
        rows.append(_diagnostic_row(solver, diagnostic, policy=f"STEP245_FIXED_M{m}"))
    snapshot, cells = _snapshot(solver, policy=f"STEP245_FIXED_M{m}")
    return Endpoint(m=m, status="PASS", reason=None, solver=solver, rows=tuple(rows), snapshot=snapshot, cells=cells)


def _relative_or_absolute(left: float, right: float) -> float:
    if left == 0.0 and right == 0.0:
        return 0.0
    if right == 0.0:
        return abs(left)
    return abs(left - right) / max(abs(right), 1.0e-300)


def _state_distance(
    left: Endpoint,
    right: Endpoint,
    *,
    edges_m: np.ndarray,
) -> dict[str, float]:
    if left.cells is None or right.cells is None:
        return {"population_normalized_L1": math.inf, "population_normalized_Linf": math.inf, "PSD_Wasserstein_m": math.inf}
    difference = np.asarray(left.cells, dtype=np.float64) - np.asarray(right.cells, dtype=np.float64)
    denominator = max(float(np.sum(np.abs(right.cells), dtype=np.float64)), 1.0e-300)
    l1 = float(np.sum(np.abs(difference), dtype=np.float64) / denominator)
    linf = float(np.max(np.abs(difference)) / max(float(np.max(np.abs(right.cells))), 1.0e-300))
    left_radii, left_weights = positive_cell_quadrature(edges_m, left.cells, 2)
    right_radii, right_weights = positive_cell_quadrature(edges_m, right.cells, 2)
    w1 = discrete_wasserstein_distance(left_radii, left_weights, right_radii, right_weights)
    return {
        "population_normalized_L1": l1,
        "population_normalized_Linf": linf,
        "PSD_Wasserstein_m": w1,
    }


def _endpoint_pair(
    left: Endpoint,
    right: Endpoint,
    *,
    edges_m: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    label = f"m{left.m}_vs_m{right.m}"
    if left.status != "PASS" or right.status != "PASS" or left.snapshot is None or right.snapshot is None:
        return ([{
            "comparison": label,
            "status": "UNAVAILABLE",
            "reason": f"left={left.status}; right={right.status}",
        }], {"status": "UNAVAILABLE", "maximum_primary_error": math.inf})
    rows: list[dict[str, Any]] = []
    maximum = 0.0
    for metric in PRIMARY_CONTINUOUS_METRICS:
        error = _relative_or_absolute(float(left.snapshot[metric]), float(right.snapshot[metric]))
        maximum = max(maximum, error)
        rows.append({
            "comparison": label,
            "metric": metric,
            "left_value": left.snapshot[metric],
            "right_value": right.snapshot[metric],
            "relative_or_absolute_error": error,
            "gate": CONTINUOUS_METRIC_GATE,
            "pass": error <= CONTINUOUS_METRIC_GATE,
        })
    distance = _state_distance(left, right, edges_m=edges_m)
    rows.append({"comparison": label, "metric": "population_normalized_L1", "left_value": distance["population_normalized_L1"], "right_value": 0.0, "relative_or_absolute_error": distance["population_normalized_L1"], "gate": "REPORT_ONLY", "pass": True})
    rows.append({"comparison": label, "metric": "population_normalized_Linf", "left_value": distance["population_normalized_Linf"], "right_value": 0.0, "relative_or_absolute_error": distance["population_normalized_Linf"], "gate": "REPORT_ONLY", "pass": True})
    rows.append({"comparison": label, "metric": "PSD_Wasserstein_m", "left_value": distance["PSD_Wasserstein_m"], "right_value": 0.0, "relative_or_absolute_error": distance["PSD_Wasserstein_m"], "gate": "REPORT_ONLY", "pass": True})
    return rows, {
        "status": "PASS" if maximum <= CONTINUOUS_METRIC_GATE else "FAIL",
        "maximum_primary_error": maximum,
        **distance,
    }


def _select_step245_reference(pair_summaries: Mapping[str, Mapping[str, Any]]) -> int | None:
    """Choose only the coarsest endpoint justified by an actual finest pair.

    There is no score, extrapolation, or root residual shortcut here.  The
    registered m16--m32 pair makes m16 admissible; only an actual m32--m64
    pair can make m32 admissible when the first pair has not converged.
    """

    if pair_summaries.get("m16_vs_m32", {}).get("status") == "PASS":
        return 16
    if pair_summaries.get("m32_vs_m64", {}).get("status") == "PASS":
        return 32
    return None


def _run_step245_ladder(
    *,
    context: Any,
    config: SolverConfig,
    checkpoint_path: Path,
    output_root: Path,
) -> tuple[dict[int, Endpoint], list[dict[str, Any]], dict[str, Any], int | None]:
    endpoints: dict[int, Endpoint] = {}
    for m in STEP245_LADDER:
        endpoints[m] = _endpoint_from_checkpoint(config=config, checkpoint_path=checkpoint_path, m=m)
    comparison_rows: list[dict[str, Any]] = []
    pair_summaries: dict[str, Any] = {}
    for left_m, right_m in zip((2, 4, 8, 16), (4, 8, 16, 32)):
        rows, summary = _endpoint_pair(endpoints[left_m], endpoints[right_m], edges_m=context.edges_m)
        comparison_rows.extend(rows)
        pair_summaries[f"m{left_m}_vs_m{right_m}"] = summary
    need_m64 = pair_summaries["m16_vs_m32"]["status"] != "PASS"
    if need_m64:
        endpoints[OPTIONAL_STEP245_M] = _endpoint_from_checkpoint(
            config=config, checkpoint_path=checkpoint_path, m=OPTIONAL_STEP245_M
        )
        rows, summary = _endpoint_pair(endpoints[32], endpoints[OPTIONAL_STEP245_M], edges_m=context.edges_m)
        comparison_rows.extend(rows)
        pair_summaries["m32_vs_m64"] = summary
    selected_m = _select_step245_reference(pair_summaries)

    ladder_rows: list[dict[str, Any]] = []
    for m in (*STEP245_LADDER, *((OPTIONAL_STEP245_M,) if need_m64 else ())):
        endpoint = endpoints[m]
        row: dict[str, Any] = {"m": m, "dt_s": DT0_S / m, "status": endpoint.status, "reason": endpoint.reason or ""}
        if endpoint.snapshot is not None:
            row.update(endpoint.snapshot)
            row.update({
                "direct_picard_closure_count": sum(item.get("fixed_point_convergence_mode") == "DIRECT" for item in endpoint.rows),
                "p2_closure_count": sum(item.get("fixed_point_periodic_cycle_period") == 2 for item in endpoint.rows),
                "p4_closure_count": sum(item.get("fixed_point_periodic_cycle_period") == 4 for item in endpoint.rows),
                "rejected_trial_step_count": 0,
                "maximum_picard_residual": max((float(item["fixed_point_xb_residual"]) for item in endpoint.rows), default=0.0),
                "maximum_accepted_closure_residual": max((float(item["fixed_point_xb_residual"]) for item in endpoint.rows), default=0.0),
            })
        ladder_rows.append(row)
    _write_csv(output_root / "step245_substep_ladder.csv", ladder_rows, ("m", "dt_s", "status", "reason"))
    _write_csv(output_root / "step245_endpoint_convergence.csv", comparison_rows, ("comparison", "status"))
    status = (
        "PASS_STEP245_CONTINUATION_CONVERGENCE"
        if selected_m is not None
        else "FAIL_DT_CONTINUATION_NONCONVERGENT_ENDPOINT"
    )
    summary = {
        "status": status,
        "substep_multiplicities": list(endpoints),
        "m64_run": need_m64,
        "pair_summaries": pair_summaries,
        "selected_m": selected_m,
        "selected_dt_s": None if selected_m is None else DT0_S / selected_m,
        "selected_state_hash": None if selected_m is None else endpoints[selected_m].snapshot["state_array_hash"],
    }
    return endpoints, comparison_rows, summary, selected_m


def _legacy_root_diagnostic(
    *,
    config: SolverConfig,
    checkpoint_path: Path,
) -> Endpoint:
    """Reproduce the old full macro only for comparison, never authority."""

    solver = CharacteristicReferenceSolver.load_checkpoint(config=config, path=checkpoint_path)
    try:
        diagnostic = solver.advance_one(maximum_dt_s=DT0_S)
    except (CharacteristicReferenceError, RadiusGridOverflowError, ValueError, FloatingPointError) as error:
        return Endpoint(1, "LEGACY_ROOT_DIAGNOSTIC_FAILED", f"{type(error).__name__}: {error}", None, (), None, None)
    row = _diagnostic_row(solver, diagnostic, policy="LEGACY_DIAGNOSTIC_TRAJECTORY")
    snapshot, cells = _snapshot(solver, policy="LEGACY_DIAGNOSTIC_TRAJECTORY")
    return Endpoint(1, "PASS_LEGACY_DIAGNOSTIC", None, solver, (row,), snapshot, cells)


def _old_root_comparison(
    *,
    legacy: Endpoint,
    endpoints: Mapping[int, Endpoint],
    selected_m: int | None,
    edges_m: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if legacy.status != "PASS_LEGACY_DIAGNOSTIC" or legacy.snapshot is None:
        return rows, {
            "status": "OLD_ROOT_DIFFERENCE_UNRESOLVED",
            "reason": "legacy diagnostic is unavailable",
        }

    metric_names = ("matrix_xB", "M0_m3", "M1_m2", "M2_m", "M3_dimensionless", "Q_beta_mol_m3")
    report_multiplicities = tuple(m for m in (8, 16, 32, 64) if m in endpoints)
    finest_m: int | None = 64 if 64 in endpoints else 32
    prior_m: int | None = 32 if finest_m == 64 else 16
    envelope: dict[str, float] = {}
    observed: dict[str, float] = {}
    classification_reason = ""
    if selected_m is None:
        finest_m = None
        prior_m = None
        classification_reason = "a converged continuation endpoint is unavailable"
    else:
        assert finest_m is not None and prior_m is not None
        finest = endpoints.get(finest_m)
        prior = endpoints.get(prior_m)
        if (
            finest is None
            or prior is None
            or finest.status != "PASS"
            or prior.status != "PASS"
            or finest.snapshot is None
            or prior.snapshot is None
        ):
            classification_reason = "refinement envelope is unavailable"
            finest_m = None
            prior_m = None
        else:
            for metric in metric_names:
                envelope[metric] = _relative_or_absolute(float(prior.snapshot[metric]), float(finest.snapshot[metric]))
                observed[metric] = _relative_or_absolute(float(legacy.snapshot[metric]), float(finest.snapshot[metric]))
            legacy_distance = _state_distance(legacy, finest, edges_m=edges_m)
            refinement_distance = _state_distance(prior, finest, edges_m=edges_m)
            for metric in ("population_normalized_L1", "PSD_Wasserstein_m"):
                observed[metric] = legacy_distance[metric]
                envelope[metric] = refinement_distance[metric]

    for m in report_multiplicities:
        endpoint = endpoints[m]
        comparison = f"legacy_root_vs_m{m}"
        if endpoint.status != "PASS" or endpoint.snapshot is None:
            rows.append({
                "comparison": comparison,
                "status": "UNAVAILABLE",
                "reason": f"continuation endpoint status={endpoint.status}",
            })
            continue
        for metric in metric_names:
            error = _relative_or_absolute(float(legacy.snapshot[metric]), float(endpoint.snapshot[metric]))
            rows.append({
                "comparison": comparison,
                "metric": metric,
                "legacy_value": legacy.snapshot[metric],
                "continuation_value": endpoint.snapshot[metric],
                "observed_error": error,
                "classification_reference": m == finest_m,
                "actual_refinement_envelope": envelope.get(metric) if m == finest_m else None,
                "within_envelope": None if m != finest_m or metric not in envelope else error <= envelope[metric],
            })
        distance = _state_distance(legacy, endpoint, edges_m=edges_m)
        for metric in ("population_normalized_L1", "PSD_Wasserstein_m"):
            error = distance[metric]
            rows.append({
                "comparison": comparison,
                "metric": metric,
                "legacy_value": error,
                "continuation_value": 0.0,
                "observed_error": error,
                "classification_reference": m == finest_m,
                "actual_refinement_envelope": envelope.get(metric) if m == finest_m else None,
                "within_envelope": None if m != finest_m or metric not in envelope else error <= envelope[metric],
            })

    if finest_m is None:
        status = "OLD_ROOT_DIFFERENCE_UNRESOLVED"
    else:
        status = "OLD_ROOT_ON_CONTINUATION_BRANCH" if all(
            float(observed[key]) <= float(envelope[key]) for key in observed
        ) else "OLD_ROOT_OFF_CONTINUATION_BRANCH"
    return rows, {
        "status": status,
        "legacy_root_xB": float(legacy.snapshot["matrix_xB"]),
        "published_legacy_root_xB": OLD_ROOT_XB,
        "legacy_root_xB_difference_from_published": abs(float(legacy.snapshot["matrix_xB"]) - OLD_ROOT_XB),
        "comparison_finest_m": finest_m,
        "comparison_prior_m": prior_m,
        "reported_comparison_multiplicities": list(report_multiplicities),
        "reason": classification_reason,
        "refinement_envelope": envelope,
        "observed_error": observed,
    }


def _check_state_equal(left: CharacteristicReferenceSolver, right: CharacteristicReferenceSolver) -> bool:
    return accepted_state_hash(left) == accepted_state_hash(right) and all(
        np.array_equal(values, right.state_arrays()[key])
        for key, values in left.state_arrays().items()
    )


def _controller_window(
    *,
    config: SolverConfig,
    selected_checkpoint: Path,
    start_old_step: int = STEP245 + 1,
    end_old_step: int = WINDOW_LAST_OLD_STEP,
) -> tuple[CharacteristicReferenceSolver, list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    solver = CharacteristicReferenceSolver.load_checkpoint(config=config, path=selected_checkpoint)
    controller = CharacteristicDtContinuation(solver)
    macro_rows: list[dict[str, Any]] = []
    for old_step in range(start_old_step, end_old_step + 1):
        event_offset = len(controller.events)
        try:
            result = controller.advance_macro(DT0_S)
        except DtContinuationMinDtError as error:
            return solver, macro_rows, controller.events, {
                "status": "FAIL_DT_CONTINUATION_NONCLOSING_SUBSTEP",
                "detail_status": "FAIL_DT_CONTINUATION_MIN_DT",
                "old_step": old_step,
                "old_step249_failure_reappears": old_step == 249,
                "physical_time_s": float(solver.time_s),
                "reason": str(error),
            }
        except (DtContinuationFatalStepError, DtContinuationError) as error:
            return solver, macro_rows, controller.events, {
                "status": "FAIL_CHARACTERISTIC_LOCAL_WELL_POSEDNESS",
                "old_step": old_step,
                "old_step249_failure_reappears": old_step == 249,
                "physical_time_s": float(solver.time_s),
                "reason": f"{type(error).__name__}: {error}",
            }
        new_events = controller.events[event_offset:]
        modes = [item.fixed_point_convergence_mode for item in result.accepted_diagnostics]
        periods = [item.fixed_point_periodic_cycle_period for item in result.accepted_diagnostics]
        if any(mode == "SAFEGUARDED_SCALAR_ROOT_V1" for mode in modes) or any(
            mode == "BRACKETED_SCALAR_ROOT" and period not in LOCAL_ALLOWED_PERIODS
            for mode, period in zip(modes, periods)
        ):
            return solver, macro_rows, controller.events, {
                "status": "FAIL_CHARACTERISTIC_LOCAL_WELL_POSEDNESS",
                "old_step": old_step,
                "old_step249_failure_reappears": old_step == 249,
                "physical_time_s": float(solver.time_s),
                "reason": "forbidden scalar-root closure appeared in controller event stream",
            }
        snapshot, _cells = _snapshot(solver, policy=POLICY_NAME)
        macro_rows.append({
            "old_step_equivalent": old_step,
            "physical_macro_start_s": result.start_time_s,
            "physical_macro_end_s": result.end_time_s,
            "nominal_dt_s": result.nominal_dt_s,
            "refinement_depth": result.refinement_depth,
            "actual_substep_count": len(result.accepted_diagnostics),
            "smallest_dt_s": min(item.dt_s for item in result.accepted_diagnostics),
            "rejection_count": result.rejection_count,
            "closure_modes": ";".join(modes),
            "closure_periods": ";".join(str(item) for item in periods),
            "max_accepted_xb_residual": max(item.fixed_point_xb_residual for item in result.accepted_diagnostics),
            "max_inventory_relative_residual": max(item.inventory.relative_residual for item in result.accepted_diagnostics),
            "event_count": len(new_events),
            "state_array_hash": result.state_hash,
            **snapshot,
        })
    return solver, macro_rows, controller.events, {
        "status": "PASS_CHARACTERISTIC_DT_CONTINUATION_LOCAL_WINDOW",
        "detail_status": "PASS",
        "old_step": end_old_step,
        "old_step249_failure_reappears": False,
        "physical_time_s": float(solver.time_s),
        "reason": "",
    }


def _restart_qualification(
    *,
    config: SolverConfig,
    state244_checkpoint: Path,
    selected_m: int,
    selected_checkpoint: Path,
    selected_endpoint: Endpoint,
    output_root: Path,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Exercise restart before, inside, and after the refined macro policy."""

    rows: list[dict[str, Any]] = []
    before = _endpoint_from_checkpoint(config=config, checkpoint_path=state244_checkpoint, m=selected_m)
    before_equal = (
        before.status == "PASS"
        and selected_endpoint.solver is not None
        and before.solver is not None
        and _check_state_equal(before.solver, selected_endpoint.solver)
    )
    rows.append({"case": "before_problem_interval", "status": "PASS" if before_equal else "FAIL", "detail": "fixed substep replay from immutable step-244 checkpoint"})

    continuous = CharacteristicDtContinuation(
        CharacteristicReferenceSolver.load_checkpoint(config=config, path=state244_checkpoint)
    )
    continuous_start = output_root / "restart_inside_continuous_macro_start.npz"
    continuous.begin_macro(DT0_S, macro_start_checkpoint=continuous_start)
    while not any(event.get("event") == "ACCEPTED_SUBSTEP" for event in continuous.events):
        if continuous.advance_pending() is not None:
            raise ContinuationWorkflowError("step-245 adaptive restart probe did not enter a refined macro")
    continuous_result = continuous.run_active_macro()

    split = CharacteristicDtContinuation(
        CharacteristicReferenceSolver.load_checkpoint(config=config, path=state244_checkpoint)
    )
    split_start = output_root / "restart_inside_split_macro_start.npz"
    split.begin_macro(DT0_S, macro_start_checkpoint=split_start)
    while not any(event.get("event") == "ACCEPTED_SUBSTEP" for event in split.events):
        if split.advance_pending() is not None:
            raise ContinuationWorkflowError("step-245 split restart probe did not enter a refined macro")
    inside_checkpoint = output_root / "restart_inside_refined_macro.npz"
    inside_sidecar = output_root / "restart_inside_refined_macro.json"
    split.save_active_restart_bundle(checkpoint_path=inside_checkpoint, sidecar_path=inside_sidecar)
    resumed = CharacteristicDtContinuation.load_active_restart_bundle(config=config, sidecar_path=inside_sidecar)
    resumed_result = resumed.run_active_macro()
    inside_equal = _check_state_equal(continuous.solver, resumed.solver) and (
        continuous_result.accepted_diagnostics == resumed_result.accepted_diagnostics
        and continuous.events == resumed.events
    )
    rows.append({"case": "inside_refined_macro", "status": "PASS" if inside_equal else "FAIL", "detail": "pending substep queue, state, events, and diagnostics"})

    direct_after = CharacteristicDtContinuation(
        CharacteristicReferenceSolver.load_checkpoint(config=config, path=selected_checkpoint)
    ).advance_macro(DT0_S)
    checkpoint_after = CharacteristicReferenceSolver.load_checkpoint(config=config, path=selected_checkpoint)
    after_controller = CharacteristicDtContinuation(checkpoint_after)
    after_result = after_controller.advance_macro(DT0_S)
    after_equal = direct_after == after_result
    rows.append({"case": "after_refined_interval", "status": "PASS" if after_equal else "FAIL", "detail": "same post-step245 nominal macro sequence"})
    status = "PASS_DT_CONTINUATION_RESTART" if all(row["status"] == "PASS" for row in rows) else "FAIL_DT_CONTINUATION_RESTART"
    return rows, {
        "status": status,
        "inside_continuous_refinement_depth": continuous_result.refinement_depth,
        "inside_resumed_refinement_depth": resumed_result.refinement_depth,
        "inside_restart_checkpoint": str(inside_checkpoint),
        "inside_restart_sidecar": str(inside_sidecar),
    }


def _frequency_summary(macro_rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    total = len(macro_rows)
    refined = [row for row in macro_rows if int(row.get("refinement_depth", 0)) > 0]
    depths: dict[str, int] = {}
    for row in macro_rows:
        key = str(int(row.get("refinement_depth", 0)))
        depths[key] = depths.get(key, 0) + 1
    max_depth = max((int(row.get("refinement_depth", 0)) for row in macro_rows), default=0)
    extra = [int(row.get("actual_substep_count", 1)) - 1 for row in macro_rows]
    if not refined:
        interpretation = "LOCAL_NONLINEAR_STIFFNESS_MANAGEABLE: no post-step245 macro interval required refinement."
    else:
        interpretation = (
            "LOCAL_REFINEMENT_DISTRIBUTION_REPORTED_WITHOUT_POSTHOC_PERCENT_THRESHOLD; "
            "inspect the full depth histogram and physical locations before judging primary Picard robustness."
        )
    return {
        "total_macro_intervals": total,
        "macro_intervals_requiring_refinement": len(refined),
        "refinement_fraction": 0.0 if total == 0 else len(refined) / total,
        "refinement_depth_histogram": depths,
        "maximum_refinement_depth": max_depth,
        "mean_extra_substeps": 0.0 if not extra else float(np.mean(np.asarray(extra, dtype=np.float64))),
        "minimum_effective_dt_s": min((float(row["smallest_dt_s"]) for row in macro_rows), default=math.nan),
        "worst_physical_time_s": None if not refined else max(float(row["physical_macro_start_s"]) for row in refined),
        "primary_closure_robustness": interpretation,
    }


def _write_required_placeholders(output_root: Path, report_root: Path, *, top_status: str, reason: str) -> None:
    """Keep downstream phases visibly gated, never silently omitted."""

    placeholders = {
        "cr1_timestep_ladder.csv": ("status", "reason"),
        "characteristic_self_convergence.csv": ("status", "reason"),
        "cohort_characteristic_parity.csv": ("status", "reason"),
    }
    for name, fields in placeholders.items():
        _write_csv(output_root / name, [{"status": "BLOCKED_PREREQUISITE_GATE", "reason": reason}], fields)
    _write_json(output_root / "time_reference_v2.json", {
        "status": "NOT_ASSIGNED_BLOCKED_PREREQUISITE_GATE",
        "upstream_top_status": top_status,
        "reason": reason,
    })
    for name in ("09_cr1_nominal_timestep_ladder.md", "10_characteristic_self_convergence.md", "11_cohort_characteristic_parity.md"):
        _write_markdown(report_root / name, REPORT_TITLES[name], {
            "status": "BLOCKED_PREREQUISITE_GATE",
            "upstream_top_status": top_status,
            "reason": reason,
            "separate_job_required": True,
        })


def _final_fields(
    *,
    source: Mapping[str, Any],
    test_status: str,
    formal: Mapping[str, Any] | None,
    ladder: Mapping[str, Any] | None,
    old_root: Mapping[str, Any] | None,
    local: Mapping[str, Any] | None,
    frequency: Mapping[str, Any] | None,
    restart: Mapping[str, Any] | None,
    top_status: str,
    next_action: str,
) -> dict[str, Any]:
    return {
        "STATUS": top_status,
        "BRANCH": source.get("git_branch", ""),
        "COMMIT": source.get("git_commit", ""),
        "SOURCE_CLEAN": not bool(source.get("git_status", "")),
        "TESTS": test_status,
        "CONTINUATION_RESTART_STEP": None if formal is None else formal.get("continuation_restart_step"),
        "CONTINUATION_RESTART_STATE_HASH": None if formal is None else formal.get("continuation_restart_state_hash"),
        "SCALAR_ROOT_FALLBACK_USED": False,
        "GENERIC_ROOT_SELECTION_USED": False,
        "STEP245_FULL_STEP_STATUS": None if ladder is None else "FULL_STEP_NOT_ADMISSIBLE",
        "STEP245_SUBSTEP_M2": None if ladder is None else None,
        "STEP245_SUBSTEP_M4": None if ladder is None else None,
        "STEP245_SUBSTEP_M8": None if ladder is None else None,
        "STEP245_SUBSTEP_M16": None if ladder is None else None,
        "STEP245_SUBSTEP_M32": None if ladder is None else None,
        "STEP245_SUBSTEP_M64": None if ladder is None else None,
        "STEP245_CONTINUATION_CONVERGENCE": None if ladder is None else ladder.get("status"),
        "STEP245_CONTINUATION_M": None if ladder is None else ladder.get("selected_m"),
        "STEP245_CONTINUATION_STATE_HASH": None if ladder is None else ladder.get("selected_state_hash"),
        "OLD_STEP245_ROOT_BRANCH_STATUS": None if old_root is None else old_root.get("status"),
        "OLD_ROOT_XB_ERROR": None if old_root is None else old_root.get("observed_error", {}).get("matrix_xB"),
        "OLD_ROOT_POPULATION_ERROR": None if old_root is None else old_root.get("observed_error", {}).get("population_normalized_L1"),
        "LOCAL_WINDOW_GATE": None if local is None else local.get("status"),
        "OLD_STEP249_FAILURE_REAPPEARS": None if local is None else bool(local.get("old_step249_failure_reappears", False)),
        "NEW_FAILURE_PHYSICAL_TIME": (
            None
            if local is None or local.get("status") == "PASS_CHARACTERISTIC_DT_CONTINUATION_LOCAL_WINDOW"
            else local.get("physical_time_s")
        ),
        "REFINED_MACRO_INTERVALS": None if frequency is None else frequency.get("macro_intervals_requiring_refinement"),
        "REFINEMENT_FRACTION": None if frequency is None else frequency.get("refinement_fraction"),
        "MAX_REFINEMENT_DEPTH": None if frequency is None else frequency.get("maximum_refinement_depth"),
        "MIN_EFFECTIVE_DT": None if frequency is None else frequency.get("minimum_effective_dt_s"),
        "PRIMARY_CLOSURE_ROBUSTNESS": None if frequency is None else frequency.get("primary_closure_robustness"),
        "RESTART_GATE": None if restart is None else restart.get("status"),
        "CR1_TIME_LADDER_RUN": False,
        "CR1_NOMINAL_DTS": [0.125, 0.0625, 0.03125, 0.015625, 0.0078125],
        "CR1_EFFECTIVE_DT_AUDIT": "BLOCKED_PREREQUISITE_GATE",
        "CHARACTERISTIC_SELF_CONVERGENCE": "BLOCKED_PREREQUISITE_GATE",
        "CHARACTERISTIC_REFERENCE_POLICY_V2": "NOT_ASSIGNED",
        "COHORT_CHARACTERISTIC_PARITY": "BLOCKED_PREREQUISITE_GATE",
        "COHORT_CHARACTERISTIC_MAX_ERROR": None,
        "TIME_REFERENCE_V2": "NOT_ASSIGNED",
        "IMPLICIT_TIME_INACCURACY_PROVEN": False,
        "IMPLICIT_LADDER_RUN": False,
        "PF_SOURCE_MODIFIED": False,
        "CUDA_RERUN": False,
        "PHYSICAL_RETUNING": False,
        "GP_RELEASE_RUN": False,
        "TOP_5_FINDINGS": [
            "Legacy full-step scalar-root selection is excluded from the new authority trajectory.",
            "The continuation candidate restarts from the formally replayed, uncontested step-244 state.",
            "All authority candidates use direct Picard or qualified exact P2/P4 only.",
            "CR1 time ladder and cohort parity are separate downstream jobs, not automatic follow-ons.",
            "PF/CUDA, implicit adjudication, beta-PF comparison, and GP release remain out of scope.",
        ],
        "P0_BLOCKERS": [] if top_status == "PASS_CHARACTERISTIC_DT_CONTINUATION_LOCAL_WINDOW" else [top_status],
        "NEXT_ACTION": next_action,
        "LOCAL_GP_RELEASE_AUTHORIZED": False,
        "KEY_REPORTS": [f"reports/{TASK_NAME}/13_final_acceptance_report.md"],
    }


def _run_local_window(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    if output_root.exists() or report_root.exists():
        raise ContinuationWorkflowError("refusing to overwrite an existing task output or report root")
    output_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    (output_root / "figures").mkdir()
    started = time.monotonic()
    source: dict[str, Any] = {}
    formal: dict[str, Any] | None = None
    ladder_summary: dict[str, Any] | None = None
    root_summary: dict[str, Any] | None = None
    local_summary: dict[str, Any] | None = None
    frequency: dict[str, Any] | None = None
    restart_summary: dict[str, Any] | None = None
    top_status = "FAIL_CHARACTERISTIC_LOCAL_WELL_POSEDNESS"
    next_action = "Inspect the phase-A report; do not enter CR1 ladder or cohort parity."
    test_status = str(args.test_status)
    provenance: dict[str, Any] = {}
    try:
        source = _source_identity(require_clean=not bool(args.allow_dirty_source))
        context = build_frozen_canonical_context()
        config = SolverConfig.from_mapping(context.mapping)
        state244_checkpoint = output_root / "continuation_restart_state.npz"
        state244_solver, replay_rows, formal = _replay_to_step244(
            context,
            formal_trace_csv=Path(args.formal_trace_csv),
            checkpoint_path=state244_checkpoint,
        )
        _write_csv(output_root / "formal_replay_1_244.csv", replay_rows, tuple(replay_rows[0]) if replay_rows else ("step",))
        state244_snapshot, _ = _snapshot(state244_solver, policy="STEP244_UNCONTESTED_FORMAL_REPLAY")
        _write_json(output_root / "restart_state_selection.json", {**formal, **state244_snapshot})

        endpoints, convergence_rows, ladder_summary, selected_m = _run_step245_ladder(
            context=context,
            config=config,
            checkpoint_path=state244_checkpoint,
            output_root=output_root,
        )
        legacy = _legacy_root_diagnostic(config=config, checkpoint_path=state244_checkpoint)
        root_rows, root_summary = _old_root_comparison(
            legacy=legacy,
            endpoints=endpoints,
            selected_m=selected_m,
            edges_m=context.edges_m,
        )
        _write_csv(output_root / "old_root_vs_continuation.csv", root_rows, ("comparison", "metric", "observed_error"))
        if selected_m is None:
            top_status = "FAIL_DT_CONTINUATION_NONCONVERGENT_ENDPOINT"
            next_action = "Stop: step-245 endpoint did not enter the registered 0.25% refinement regime."
            raise ContinuationWorkflowError("step-245 continuation endpoint did not converge")
        selected = endpoints[selected_m]
        if selected.solver is None or selected.snapshot is None:
            raise ContinuationWorkflowError("selected continuation endpoint has no accepted solver state")
        selected_checkpoint = output_root / "step245_continuation_reference.npz"
        selected.solver.save_checkpoint(selected_checkpoint)
        ladder_summary.update({
            "selected_checkpoint": str(selected_checkpoint),
            "selected_checkpoint_sha256": _sha256_file(selected_checkpoint),
        })

        restart_rows, restart_summary = _restart_qualification(
            config=config,
            state244_checkpoint=state244_checkpoint,
            selected_m=selected_m,
            selected_checkpoint=selected_checkpoint,
            selected_endpoint=selected,
            output_root=output_root,
        )
        _write_csv(output_root / "restart_comparison.csv", restart_rows, ("case", "status", "detail"))
        if restart_summary["status"] != "PASS_DT_CONTINUATION_RESTART":
            top_status = "FAIL_CHARACTERISTIC_LOCAL_WELL_POSEDNESS"
            next_action = "Stop: the continuation restart contract failed."
            raise ContinuationWorkflowError("continuation restart qualification failed")

        local_solver, local_rows, events, local_summary = _controller_window(
            config=config,
            selected_checkpoint=selected_checkpoint,
        )
        _write_csv(output_root / "local_window_trajectory.csv", local_rows, ("old_step_equivalent", "status"))
        _write_csv(output_root / "refinement_events.csv", events, ("event", "macro_index"))
        frequency = _frequency_summary(local_rows)
        histogram_rows = [
            {"refinement_depth": int(depth), "macro_interval_count": count}
            for depth, count in sorted(frequency["refinement_depth_histogram"].items(), key=lambda item: int(item[0]))
        ]
        _write_csv(output_root / "refinement_histogram.csv", histogram_rows, ("refinement_depth", "macro_interval_count"))
        if local_summary["status"] != "PASS_CHARACTERISTIC_DT_CONTINUATION_LOCAL_WINDOW":
            top_status = str(local_summary["status"])
            next_action = "Stop: local dt continuation did not close the physical window without a root fallback."
            raise ContinuationWorkflowError(str(local_summary.get("reason", "local continuation failed")))

        replay_solver, replay_rows_local, replay_events, replay_summary = _controller_window(
            config=config,
            selected_checkpoint=selected_checkpoint,
        )
        deterministic = (
            local_summary == replay_summary
            and local_rows == replay_rows_local
            and events == replay_events
            and _check_state_equal(local_solver, replay_solver)
        )
        local_summary["deterministic_replay"] = deterministic
        if not deterministic:
            top_status = "FAIL_CHARACTERISTIC_LOCAL_WELL_POSEDNESS"
            next_action = "Stop: repeated continuation did not produce the same accepted trajectory."
            raise ContinuationWorkflowError("local continuation deterministic replay differs")
        top_status = "PASS_CHARACTERISTIC_DT_CONTINUATION_LOCAL_WINDOW"
        next_action = "Submit the separately gated CR1 nominal timestep ladder; do not run cohort parity until self-convergence passes."
    except Exception as error:
        if top_status == "FAIL_CHARACTERISTIC_LOCAL_WELL_POSEDNESS":
            top_status = "FAIL_CHARACTERISTIC_LOCAL_WELL_POSEDNESS"
        provenance["error"] = f"{type(error).__name__}: {error}"
    finally:
        if not source:
            try:
                source = _source_identity(require_clean=False)
            except Exception:
                source = {"git_branch": "UNAVAILABLE", "git_commit": "UNAVAILABLE", "git_status": "UNAVAILABLE"}
        fields = _final_fields(
            source=source,
            test_status=test_status,
            formal=formal,
            ladder=ladder_summary,
            old_root=root_summary,
            local=local_summary,
            frequency=frequency,
            restart=restart_summary,
            top_status=top_status,
            next_action=next_action,
        )
        if ladder_summary is not None:
            endpoints_for_fields = locals().get("endpoints", {})
            for m in (*STEP245_LADDER, OPTIONAL_STEP245_M):
                endpoint = endpoints_for_fields.get(m)
                fields[f"STEP245_SUBSTEP_M{m}"] = None if endpoint is None else endpoint.status
            fields["STEP245_FULL_STEP_STATUS"] = fields["STEP245_SUBSTEP_M1"]
        _write_required_placeholders(output_root, report_root, top_status=top_status, reason=next_action)
        report_payloads: dict[str, Mapping[str, Any] | str] = {
            "00_authority_boundary.md": {
                "policy": POLICY_NAME,
                "legacy_step245": "LEGACY_DIAGNOSTIC_TRAJECTORY",
                "legacy_step249": "STEP249_NO_NATURAL_ADJACENT_BRACKET (historical diagnostic only)",
                "generic_scalar_root": "FORBIDDEN",
                "physical_or_PF_changes": "FORBIDDEN_AND_NOT_PERFORMED",
            },
            "01_restart_state_selection.md": formal or {"status": "UNAVAILABLE"},
            "02_dt_continuation_contract.md": {
                "policy": POLICY_NAME,
                "ordinary_or_qualified_P2_P4_only": True,
                "reject_whole_macro_then_uniform_halve": True,
                "next_macro_retries_nominal_dt": True,
                "max_refinement_depth": 6,
                "generic_root_selection_used": False,
            },
            "03_step245_substep_ladder.md": ladder_summary or {"status": "UNAVAILABLE"},
            "04_step245_endpoint_convergence.md": ladder_summary or {"status": "UNAVAILABLE"},
            "05_old_root_continuation_comparison.md": root_summary or {"status": "UNAVAILABLE"},
            "06_local_240_260_continuation.md": local_summary or {"status": "UNAVAILABLE"},
            "07_refinement_frequency.md": frequency or {"status": "UNAVAILABLE"},
            "08_restart_qualification.md": restart_summary or {"status": "UNAVAILABLE"},
            "12_model_role_boundary.md": {
                "PF_source_modified": False,
                "CUDA_rerun": False,
                "implicit_time_accuracy_adjudication": "NOT_RUN",
                "Eulerian_authority_V2": "NOT_RUN",
                "beta_PF_comparison": "NOT_RUN",
                "GP_release": "NOT_RUN",
            },
            "13_final_acceptance_report.md": fields,
            "14_reproduction_commands.md": {
                "phase_A": f"PYTHONPATH=src {sys.executable} scripts/run_kwn_characteristic_dt_continuation_v1.py local-window --output-root <new-output-root> --report-root <new-report-root> --formal-trace-csv <frozen-trace>",
                "phase_B": "A separate cr1-ladder job is required only after Phase A PASS.",
                "phase_C": "A separate cohort-parity job is required only after Phase B PASS.",
            },
        }
        for name, payload in report_payloads.items():
            _write_markdown(report_root / name, REPORT_TITLES[name], payload)
        provenance.update({
            "task_name": TASK_NAME,
            "top_status": top_status,
            "source": source,
            "runtime": _runtime_provenance(),
            "frozen_canonical_snapshot": frozen_canonical_snapshot_provenance(),
            "formal": formal,
            "step245_ladder": ladder_summary,
            "old_root_comparison": root_summary,
            "local_window": local_summary,
            "refinement_frequency": frequency,
            "restart": restart_summary,
            "runner_sha256": _sha256_file(Path(__file__)),
            "characteristic_solver_sha256": _sha256_file(ROOT / "src/kwn_mvp/characteristic_reference.py"),
            "continuation_controller_sha256": _sha256_file(ROOT / "src/kwn_mvp/characteristic_dt_continuation.py"),
            "runtime_s": time.monotonic() - started,
        })
        _write_json(output_root / "analysis_provenance.json", provenance)
        print(json.dumps(_json_safe(fields), indent=2, sort_keys=True))
    return (0 if top_status == "PASS_CHARACTERISTIC_DT_CONTINUATION_LOCAL_WINDOW" else 2), fields


def _run_downstream_placeholder(args: argparse.Namespace, *, phase: str) -> tuple[int, dict[str, Any]]:
    """Fail closed until a separately implemented, separately submitted phase exists."""

    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    if output_root.exists() or report_root.exists():
        raise ContinuationWorkflowError("refusing to overwrite an existing task output or report root")
    output_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    source = _source_identity(require_clean=not bool(args.allow_dirty_source))
    message = (
        f"{phase} is a separately gated stage.  This revision performs only the local-window job; "
        "it will not infer or auto-run a downstream authority phase."
    )
    _write_required_placeholders(output_root, report_root, top_status="BLOCKED_PREREQUISITE_GATE", reason=message)
    fields = _final_fields(
        source=source,
        test_status="NOT_RUN",
        formal=None,
        ladder=None,
        old_root=None,
        local=None,
        frequency=None,
        restart=None,
        top_status="BLOCKED_PREREQUISITE_GATE",
        next_action=message,
    )
    _write_markdown(report_root / "13_final_acceptance_report.md", REPORT_TITLES["13_final_acceptance_report.md"], fields)
    _write_json(output_root / "analysis_provenance.json", {
        "task_name": TASK_NAME,
        "phase": phase,
        "top_status": "BLOCKED_PREREQUISITE_GATE",
        "source": source,
        "reason": message,
    })
    print(json.dumps(_json_safe(fields), indent=2, sort_keys=True))
    return 2, fields


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    for name in ("local-window", "cr1-ladder", "cohort-parity"):
        command = subparsers.add_parser(name)
        command.add_argument("--output-root", type=Path, required=True)
        command.add_argument("--report-root", type=Path, required=True)
        command.add_argument("--formal-trace-csv", type=Path, default=DEFAULT_FORMAL_TRACE)
        command.add_argument(
            "--allow-dirty-source",
            action="store_true",
            help="local development only; a production sbatch always requires a clean source tree",
        )
        command.add_argument(
            "--test-status",
            default="NOT_RUN_BY_PHASE_A_RUNNER",
            help="externally verified regression-test status recorded in the final provenance",
        )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "local-window":
        code, _fields = _run_local_window(args)
    else:
        code, _fields = _run_downstream_placeholder(args, phase=str(args.command))
    return code


if __name__ == "__main__":
    raise SystemExit(main())
