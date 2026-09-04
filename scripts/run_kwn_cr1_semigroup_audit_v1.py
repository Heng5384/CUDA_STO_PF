#!/usr/bin/env python3
"""Diagnostic-only CR1 semigroup, topology-event, and time-centering audit.

The audit starts from the frozen accepted step-244 checkpoint and runs no
production continuation.  Every candidate one-step map is evaluated on an
ephemeral in-memory clone.  It therefore cannot accept a new scalar root,
change the global trajectory, write a checkpoint, or establish a time
reference.  The only scalar closure used by the dynamic operator is the
already-frozen ordinary/P2/P4 CR1 candidate closure.
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
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

from kwn_mvp.characteristic_pathology import (  # noqa: E402
    capture_prestate,
    evaluate_closure_map,
)
from kwn_mvp.characteristic_reference import (  # noqa: E402
    CharacteristicReferenceSolver,
)
from kwn_mvp.characteristic_semigroup import (  # noqa: E402
    PiecewiseLinearMatrixTrajectory,
    classify_semigroup_sequence,
    event_error_localization,
    phi_compose,
    physical_state_distance,
    semigroup_triplet,
    topology_comparison,
    topology_event_proximity_rows,
)
from scripts.frozen_canonical_smooth_population_v1 import (  # noqa: E402
    build_frozen_canonical_context,
    frozen_canonical_snapshot_provenance,
)
from scripts import run_kwn_cr1_fine_substep_pathology_v1 as prior  # noqa: E402


TASK_NAME = "kwn_cr1_semigroup_audit_v1"
REQUIRED_BRANCH = "codex/kwn-cr1-semigroup-audit-v1"
BASE_COMMIT = "d64504481a03b290d1e02663e08333a75cfa5a7d"
DT0_S = 0.015625
RESTART_STEP = 244
RESTART_STATE_HASH = "51c243b61f4278fdf8f5bae9afe50672f79a6e0be57260b9c7cc0fdb46dca4c2"
EXPECTED_SAME_PRESTATE_SHA256 = "8caa31185510a55fce0e74e9b7a95a645bd653ec1d3a7f4c3cd7f58582e1cdc3"
SEMIGROUP_DIVISORS = (2, 4, 8, 16)
LOCAL_SCAN_POINTS = 65
EVENT_SCAN_SUBINTERVALS = 16
EVENT_BISECTION_MAX_ITERATIONS = 64
EVENT_BOUNDARY_SELECTION = "ONE_NEAREST_COMMON_TIME_BOUNDARY_PER_CHANGED_FACE_TIE_LOWER_RADIUS"

REPORT_TITLES = {
    "00_baseline.md": "Phase-A baseline reproduction",
    "01_one_step_operator_contract.md": "Read-only characteristic one-step operator contract",
    "02_dynamic_semigroup.md": "Dynamic-matrix semigroup audit",
    "03_frozen_matrix_semigroup.md": "Frozen-matrix semigroup audit",
    "04_prescribed_x_semigroup.md": "Prescribed-x bridge semigroup audit",
    "05_time_centering_local_consistency.md": "Matrix-coupled time-centering local consistency",
    "06_topology_event_proximity.md": "CR1 topology-event proximity",
    "07_topology_event_timeline.md": "Topology-event timeline and ordering",
    "08_event_localized_state_error.md": "Event-localized physical-state error",
    "09_semigroup_root_cause.md": "Semigroup root-cause classification",
    "10_method_decision.md": "Method-selection gate",
    "11_final_acceptance_report.md": "Final diagnostic acceptance report",
    "12_reproduction_commands.md": "Reproduction commands",
}


class SemigroupWorkflowError(RuntimeError):
    """A fail-closed orchestration error, never a request for a repair."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, Fraction):
        return {"numerator": value.numerator, "denominator": value.denominator}
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.ndarray):
        return {
            "array_shape": list(value.shape),
            "array_dtype": str(value.dtype),
            "array_sha256": hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest(),
        }
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


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


def _git(command: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *command], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return completed.stdout.strip()


def _source_identity(*, require_clean: bool) -> dict[str, Any]:
    try:
        branch = _git(("branch", "--show-current"))
        commit = _git(("rev-parse", "HEAD"))
        status = _git(("status", "--short"))
        base_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", BASE_COMMIT, "HEAD"],
            cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        ).returncode == 0
    except subprocess.CalledProcessError as error:
        raise SemigroupWorkflowError("could not establish source identity") from error
    if branch != REQUIRED_BRANCH:
        raise SemigroupWorkflowError(f"source branch differs from required {REQUIRED_BRANCH}")
    if not base_ancestor:
        raise SemigroupWorkflowError("source does not descend from frozen d645044 baseline")
    if require_clean and status:
        raise SemigroupWorkflowError("source tree is dirty")
    return {
        "git_branch": branch,
        "git_commit": commit,
        "git_status": status,
        "base_commit": BASE_COMMIT,
        "base_ancestor": base_ancestor,
    }


def _runtime_provenance() -> dict[str, Any]:
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "numpy": np.__version__,
    }


def _implementation_provenance() -> dict[str, str]:
    """Bind each diagnostic result to the code that produced it."""

    files = {
        "semigroup_runner_sha256": Path(__file__).resolve(),
        "semigroup_operator_sha256": ROOT / "src" / "kwn_mvp" / "characteristic_semigroup.py",
        "production_characteristic_reference_sha256": (
            ROOT / "src" / "kwn_mvp" / "characteristic_reference.py"
        ),
        "pathology_replay_helper_sha256": (
            ROOT / "scripts" / "run_kwn_cr1_fine_substep_pathology_v1.py"
        ),
        "frozen_canonical_helper_sha256": (
            ROOT / "scripts" / "frozen_canonical_smooth_population_v1.py"
        ),
        "sbatch_wrapper_sha256": ROOT / "jobs" / "run_kwn_cr1_semigroup_audit_v1_cpu.sbatch",
    }
    return {label: _sha256_file(path) for label, path in files.items()}


def _formal_reproduction(
    runs: Mapping[int, prior.FixedRun], formal: Mapping[int, Mapping[str, str]]
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    all_ok = True
    for multiplicity in (16, 32, 64):
        expected = formal[multiplicity]
        observed = prior._formal_endpoint_observation(runs[multiplicity])
        fields = [field for field, value in expected.items() if value != ""]
        mismatches = [
            {"field": field, "formal": expected[field], "replay": observed.get(field, "MISSING")}
            for field in fields
            if field not in observed or not prior._same_formal_value(expected[field], observed[field])
        ]
        results[f"m{multiplicity}"] = {
            "status": "PASS" if not mismatches else "FAIL",
            "mismatch_count": len(mismatches),
            "mismatches": mismatches[:32],
        }
        all_ok = all_ok and not mismatches
    m16, m32, m64 = runs[16], runs[32], runs[64]
    expected_paths = (
        m16.status == "PASS"
        and m32.status == "NONCLOSING_SUBSTEP"
        and m32.failure is not None
        and int(m32.failure["failed_substep"]) == 6
        and m32.failure.get("closure_path") == "ALLOWED_P4_EXACT_CYCLE_PATH"
        and m64.status == "NONCLOSING_SUBSTEP"
        and m64.failure is not None
        and int(m64.failure["failed_substep"]) == 5
        and m64.failure.get("closure_path") == "ORDINARY_PICARD"
    )
    return {
        "status": "PASS_SEMIGROUP_BASELINE_REPRODUCTION" if all_ok and expected_paths else "FAIL_SEMIGROUP_BASELINE_REPRODUCTION",
        "formal_endpoint_comparison": results,
        "path_conditions_match": expected_paths,
        "runs": {f"m{item}": runs[item].status for item in (16, 32, 64)},
    }


def _baseline_state(state: Mapping[str, Any]) -> dict[str, Any]:
    evaluation = state.get("_accepted_evaluation")
    if evaluation is None:
        raise SemigroupWorkflowError("baseline accepted state lacks topology telemetry")
    return {
        "population_array": np.asarray(state["population_array"], dtype=np.float64).copy(),
        "cdf": np.asarray(state["cdf"], dtype=np.float64).copy(),
        "xB": float(state["xB"]),
        "M0_m3": float(state["M0_m3"]),
        "M1_m2": float(state["M1_m2"]),
        "M2_m": float(state["M2_m"]),
        "M3_dimensionless": float(state["M3_dimensionless"]),
        "Rmean_m": float(state["Rmean_m"]),
        "Rmean3_m3": float(state["Rmean3_m3"]),
        "Sv_m_inv": float(state["Sv_m_inv"]),
        "f_beta": float(state["f_beta"]),
        "Q_beta_mol_m3": float(state["Q_beta_mol_m3"]),
        "Q_matrix_mol_m3": float(state["Q_matrix_mol_m3"]),
        "Q_total_mol_m3": float(state["Q_total_mol_m3"]),
        "inventory_relative_residual": float(state["inventory_relative_residual"]),
        "time_s": float(state["physical_time_s"]),
        "step": int(state["substep"]),
        "accepted_state_hash": str(state["state_hash"]),
        "diagnostic_state_hash": str(state["state_hash"]),
        "departure_faces_m": np.asarray(evaluation.departure_faces_m, dtype=np.float64).copy(),
        "source_cell_indices": np.asarray(evaluation.source_cell_indices, dtype=np.int64).copy(),
        "remap_topology_signature": str(evaluation.remap_topology_signature),
        "trace_topology_signature": str(evaluation.trace_topology_signature),
        "departure_signature": str(evaluation.departure_signature),
    }


def _accepted_states_by_m(
    runs: Mapping[int, prior.FixedRun],
) -> dict[int, Sequence[Mapping[str, Any]]]:
    """Extract leaf states, rather than passing FixedRun containers to an archiver."""

    return {
        multiplicity: runs[multiplicity].accepted_states
        for multiplicity in (16, 32, 64)
    }


def _semigroup_rows(
    source: CharacteristicReferenceSolver,
    *,
    mode: str,
    trajectory: PiecewiseLinearMatrixTrajectory | None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    frozen_x = float(source.matrix_xb) if mode == "FROZEN_MATRIX" else None
    rows = [
        semigroup_triplet(
            source,
            h_s=DT0_S / divisor,
            mode=mode,
            frozen_matrix_xb=frozen_x,
            prescribed_matrix_trajectory=trajectory,
        )
        for divisor in SEMIGROUP_DIVISORS
    ]
    classification = classify_semigroup_sequence(rows)
    return rows, classification


def _semigroup_csv_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    flattened: list[dict[str, Any]] = []
    for row in rows:
        result = {
            key: value
            for key, value in row.items()
            if key not in {
                "h_vs_two", "two_vs_four", "h_vs_four", "topology_h_vs_two",
                "topology_two_vs_four", "topology_h_vs_four", "Phi_h_state",
                "Phi_h2x2_state", "Phi_h4x4_state", "topology_path_h",
                "topology_path_h2x2", "topology_path_h4x4",
                "topology_path_h_vs_h2x2", "topology_path_h2x2_vs_h4x4",
            }
        }
        for prefix in ("topology_h_vs_two", "topology_two_vs_four", "topology_h_vs_four"):
            topology = row.get(prefix)
            if isinstance(topology, Mapping):
                result[f"{prefix}_different"] = not bool(topology.get("topology_equal"))
                result[f"{prefix}_changed_face_count"] = int(topology.get("changed_face_count", 0))
                result[f"{prefix}_changed_face_indices_json"] = json.dumps(
                    [int(face) for face in np.asarray(topology.get("changed_face_indices", ()), dtype=np.int64)],
                    separators=(",", ","),
                )
                result[f"{prefix}_semantics"] = "FINAL_LEAF_ONLY_NOT_A_SEMIGROUP_TOPOLOGY_GATE"
        for prefix in ("topology_path_h_vs_h2x2", "topology_path_h2x2_vs_h4x4"):
            topology = row.get(prefix)
            if isinstance(topology, Mapping):
                result[f"{prefix}_different_discrete_regimes"] = bool(
                    topology.get("different_discrete_topology_regimes")
                )
        flattened.append(result)
    return flattened


def _semigroup_topology_path_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Write explicit discrete partitions for the otherwise hash-heavy reports."""

    output: list[dict[str, Any]] = []
    paths = (
        ("Phi_h", "topology_path_h"),
        ("Phi_h2x2", "topology_path_h2x2"),
        ("Phi_h4x4", "topology_path_h4x4"),
    )
    for row in rows:
        if row.get("status") != "SUCCESS":
            continue
        for path_label, field in paths:
            summary = row[field]
            signatures = list(summary["leaf_topology_signatures"])
            departure_signatures = list(summary["leaf_departure_signatures"])
            partitions = list(summary["leaf_source_cell_indices"])
            for leaf_index, (signature, departure_signature, partition) in enumerate(
                zip(signatures, departure_signatures, partitions), start=1
            ):
                output.append({
                    "mode": row["mode"],
                    "h_s": row["h_s"],
                    "path": path_label,
                    "leaf_index": leaf_index,
                    "remap_topology_signature": str(signature),
                    "departure_signature": str(departure_signature),
                    "source_cell_indices_json": json.dumps(
                        [int(value) for value in np.asarray(partition, dtype=np.int64)],
                        separators=(",", ","),
                    ),
                })
    return output


def _semigroup_error_rows(rows: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for row in rows:
        for comparison_name in ("h_vs_two", "two_vs_four", "h_vs_four"):
            values = row.get(comparison_name)
            if not isinstance(values, Mapping):
                continue
            output.append({
                "mode": row["mode"], "h_s": row["h_s"], "comparison": comparison_name,
                **values,
            })
    return output


def _local_root_family(
    source: CharacteristicReferenceSolver,
    *,
    trajectory: PiecewiseLinearMatrixTrajectory,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Observe, but never accept, near-state scalar crossings at decreasing h."""

    rates = np.abs(np.diff(trajectory.xb) / np.diff(trajectory.times_s))
    c_scale = float(np.max(rates)) if rates.size else 0.0
    prestate = capture_prestate(source)
    rows: list[dict[str, Any]] = []
    level_summary: list[dict[str, Any]] = []
    if not math.isfinite(c_scale) or c_scale <= 0.0:
        return rows, {
            "LOCAL_ROOT_FAMILY": "NO_NONZERO_OBSERVED_DXDT_SCALE",
            "DELTA_X_OVER_H_CONVERGENCE": "NOT_ASSESSABLE",
            "C_from_m16_dxdt_s_inv": c_scale,
            "levels": level_summary,
        }
    for divisor in SEMIGROUP_DIVISORS:
        h = DT0_S / divisor
        lower = max(0.0, float(prestate.x_start) - c_scale * h)
        upper = min(1.0, float(prestate.x_start) + c_scale * h)
        x_values = np.linspace(lower, upper, LOCAL_SCAN_POINTS, dtype=np.float64)
        evaluations: list[dict[str, Any]] = []
        for index, x_trial in enumerate(x_values):
            try:
                evaluation = evaluate_closure_map(
                    source, prestate=prestate, dt_s=h, x_trial=float(x_trial)
                )
            except Exception as error:  # read-only map failures are evidence, not a repair trigger
                entry = {
                    "record_type": "MAP_ERROR", "h_s": h, "divisor": divisor, "sample": index,
                    "x_trial": float(x_trial), "error_type": type(error).__name__, "error": str(error),
                }
                rows.append(entry)
                evaluations.append(entry)
                continue
            entry = {
                "record_type": "MAP", "h_s": h, "divisor": divisor, "sample": index,
                "x_trial": float(x_trial), "x_trial_hex": float(x_trial).hex(),
                "signed_F": float(evaluation.signed_f), "signed_F_hex": float(evaluation.signed_f).hex(),
                "x_closure": float(evaluation.x_closure),
                "distance_from_xn": abs(float(x_trial) - float(prestate.x_start)),
                "distance_over_h": abs(float(x_trial) - float(prestate.x_start)) / h,
                "topology_signature": str(evaluation.remap_topology_signature),
                "departure_signature": str(evaluation.departure_signature),
            }
            rows.append(entry)
            evaluations.append(entry)
        crossings: list[dict[str, Any]] = []
        valid = [entry for entry in evaluations if entry.get("record_type") == "MAP"]
        for left, right in zip(valid, valid[1:]):
            f_left, f_right = float(left["signed_F"]), float(right["signed_F"])
            if f_left == 0.0 or f_right == 0.0 or f_left * f_right < 0.0:
                midpoint = 0.5 * (float(left["x_trial"]) + float(right["x_trial"]))
                crossing = {
                    "record_type": "OBSERVED_CROSSING", "h_s": h, "divisor": divisor,
                    "x_left": float(left["x_trial"]), "x_right": float(right["x_trial"]),
                    "F_left": f_left, "F_right": f_right, "x_midpoint": midpoint,
                    "distance_from_xn": abs(midpoint - float(prestate.x_start)),
                    "delta_x_over_h": (midpoint - float(prestate.x_start)) / h,
                    "same_topology": left["topology_signature"] == right["topology_signature"],
                    "left_topology": left["topology_signature"], "right_topology": right["topology_signature"],
                }
                crossings.append(crossing)
                rows.append(crossing)
        admissible = [item for item in crossings if bool(item["same_topology"])]
        closest = min(admissible, key=lambda item: abs(float(item["delta_x_over_h"]))) if admissible else None
        level_summary.append({
            "h_s": h, "divisor": divisor, "scan_lower_xB": lower, "scan_upper_xB": upper,
            "observed_crossing_count": len(crossings), "same_topology_crossing_count": len(admissible),
            "closest_admissible_crossing": closest,
        })
    selected = [item["closest_admissible_crossing"] for item in level_summary]
    all_local = all(item is not None for item in selected)
    ratios = [float(item["delta_x_over_h"]) for item in selected if item is not None]
    ratio_differences = [
        abs(right - left) for left, right in zip(ratios, ratios[1:])
    ]
    finite_ratios = all(math.isfinite(value) for value in ratios)
    cauchy_like = (
        all_local
        and finite_ratios
        and len(ratio_differences) >= 2
        and all(right <= left for left, right in zip(ratio_differences, ratio_differences[1:]))
    )
    crossing_counts = [int(item["observed_crossing_count"]) for item in level_summary]
    crossings_densify = (
        len(crossing_counts) >= 2
        and all(right >= left for left, right in zip(crossing_counts, crossing_counts[1:]))
        and any(right > left for left, right in zip(crossing_counts, crossing_counts[1:]))
    )
    return rows, {
        "LOCAL_ROOT_FAMILY": (
            "LOCAL_CONTINUOUS_ROOT_FAMILY_OBSERVED" if all_local else "LOCAL_CONTINUOUS_ROOT_FAMILY_NOT_OBSERVED"
        ),
        "DELTA_X_OVER_H_CONVERGENCE": (
            "OBSERVED_CAUCHY_LIKE_TREND" if cauchy_like
            else ("FINITE_BUT_NO_CONVERGENCE_TREND" if all_local and finite_ratios else "NOT_ESTABLISHED")
        ),
        "C_from_m16_dxdt_s_inv": c_scale,
        "delta_x_over_h_sequence": ratios,
        "delta_x_over_h_successive_absolute_differences": ratio_differences,
        "observed_crossing_counts": crossing_counts,
        "crossings_densify_as_h_decreases": crossings_densify,
        "levels": level_summary,
        "diagnostic_only_no_root_accepted": True,
    }


def _event_boundaries(
    states: Mapping[str, Mapping[str, Any]], *, edges_m: np.ndarray
) -> list[dict[str, Any]]:
    """Choose one reproducible primary event boundary for every changed face.

    All boundaries bracketed by the three common-time departures are retained
    as a count, but event-time bisection is deliberately limited to the one
    boundary nearest to any observed departure for that face.  This covers
    every changed face without turning the diagnostic into a path-dependent
    many-boundary production sweep.
    """

    faces: set[int] = set()
    for left, right in (("m16", "m32"), ("m16", "m64"), ("m32", "m64")):
        faces.update(int(face) for face in topology_comparison(states[left], states[right])["changed_face_indices"])
    output: list[dict[str, Any]] = []
    for face in sorted(faces):
        departures = [float(np.asarray(state["departure_faces_m"])[face]) for state in states.values()]
        lower, upper = min(departures), max(departures)
        candidates = [
            float(boundary)
            for boundary in np.asarray(edges_m, dtype=np.float64)
            if lower <= float(boundary) <= upper
        ]
        if not candidates:
            continue
        minimum_distance, selected = min(
            (
                min(abs(departure - boundary) for departure in departures),
                boundary,
            )
            for boundary in candidates
        )
        output.append({
            "face_index": face,
            "source_boundary_m": selected,
            "bracketed_boundary_count": len(candidates),
            "selection_rule": EVENT_BOUNDARY_SELECTION,
            "minimum_common_time_departure_distance_m": minimum_distance,
        })
    return output


def _event_departure(
    source: CharacteristicReferenceSolver,
    *,
    duration_s: float,
    face_index: int,
    mode: str,
    trajectory: PiecewiseLinearMatrixTrajectory | None,
) -> tuple[float | None, str | None]:
    """Read a face departure from a successful disposable one-step map."""

    if duration_s < 0.0:
        return None, "NEGATIVE_DURATION_NOT_A_TOPOLOGY_EVENT"
    if duration_s == 0.0:
        edges = np.asarray(source.population("beta").grid.edges_m, dtype=np.float64)
        if face_index < 0 or face_index >= edges.size:
            return None, "FACE_INDEX_OUT_OF_RANGE"
        return float(edges[face_index]), None
    result = phi_compose(
        source,
        dt_s=duration_s,
        count=1,
        mode=mode,
        frozen_matrix_xb=float(source.matrix_xb) if mode == "FROZEN_MATRIX" else None,
        prescribed_matrix_trajectory=trajectory,
    )
    state = result.state
    if result.status != "SUCCESS" or not isinstance(state, Mapping):
        return None, str(result.error_message or "NONCLOSING_EVENT_PROBE")
    departure = np.asarray(state["departure_faces_m"], dtype=np.float64)
    if face_index < 0 or face_index >= departure.size:
        return None, "FACE_INDEX_OUT_OF_RANGE"
    return float(departure[face_index]), None


def _estimate_event_time(
    source: CharacteristicReferenceSolver,
    *,
    face_index: int,
    boundary_m: float,
    mode: str,
    trajectory: PiecewiseLinearMatrixTrajectory | None,
) -> dict[str, Any]:
    """Bracket and bisect a departure-face event without selecting an xB root."""

    horizon = DT0_S / 16.0
    samples: list[tuple[float, float]] = []
    for index in range(0, EVENT_SCAN_SUBINTERVALS + 1):
        duration = horizon * index / EVENT_SCAN_SUBINTERVALS
        departure, error = _event_departure(
            source, duration_s=duration, face_index=face_index, mode=mode, trajectory=trajectory
        )
        if departure is None:
            return {
                "mode": mode, "face_index": face_index, "source_boundary_m": boundary_m,
                "status": "NONCLOSING_EVENT_PROBE", "error": error,
            }
        samples.append((duration, departure - boundary_m))
    bracket: tuple[float, float, float, float] | None = None
    for (left_t, left_f), (right_t, right_f) in zip(samples, samples[1:]):
        if left_f == 0.0:
            bracket = (left_t, left_t, left_f, left_f)
            break
        if right_f == 0.0 or left_f * right_f < 0.0:
            bracket = (left_t, right_t, left_f, right_f)
            break
    if bracket is None:
        return {
            "mode": mode, "face_index": face_index, "source_boundary_m": boundary_m,
            "status": "NO_NATURAL_DURATION_BRACKET", "probe_count": len(samples),
            "minimum_abs_departure_minus_boundary_m": min(abs(item[1]) for item in samples),
        }
    left_t, right_t, left_f, right_f = bracket
    iterations = 0
    while left_t != right_t and iterations < EVENT_BISECTION_MAX_ITERATIONS:
        midpoint = 0.5 * (left_t + right_t)
        if midpoint == left_t or midpoint == right_t:
            break
        departure, error = _event_departure(
            source, duration_s=midpoint, face_index=face_index, mode=mode, trajectory=trajectory
        )
        if departure is None:
            return {
                "mode": mode, "face_index": face_index, "source_boundary_m": boundary_m,
                "status": "NONCLOSING_DURING_EVENT_BISECTION", "error": error,
            }
        value = departure - boundary_m
        iterations += 1
        if value == 0.0:
            left_t = right_t = midpoint
            left_f = right_f = value
            break
        if left_f * value < 0.0:
            right_t, right_f = midpoint, value
        else:
            left_t, left_f = midpoint, value
    duration = 0.5 * (left_t + right_t)
    exact_root = left_f == 0.0 or right_f == 0.0
    return {
        "mode": mode, "face_index": face_index, "source_boundary_m": boundary_m,
        "status": (
            "EVENT_TIME_ESTIMATED"
            if exact_root else "EVENT_TIME_BRACKETED_TOPOLOGY_TRANSITION"
        ),
        "event_duration_from_step244_s": duration,
        "event_physical_time_s": float(source.time_s) + duration,
        "bracket_left_duration_s": left_t, "bracket_right_duration_s": right_t,
        "bracket_left_signed_distance_m": left_f, "bracket_right_signed_distance_m": right_f,
        "bisection_iterations": iterations,
        "includes_step244_identity_departure": True,
        "diagnostic_only_no_trajectory_change": True,
    }


def _event_timeline(
    runs: Mapping[int, prior.FixedRun], *, edges_m: np.ndarray
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Record discrete topology changes along each already accepted Phase-A path."""

    rows: list[dict[str, Any]] = []
    sequences: dict[str, list[tuple[int, int, int]]] = {}
    endpoints = [
        float(run.accepted_states[-1]["physical_time_s"])
        for run in runs.values() if run.accepted_states
    ]
    if len(endpoints) != 3:
        raise SemigroupWorkflowError("each event-timeline route requires one accepted state")
    common_end_time_s = min(endpoints)
    time_tolerance = 8.0 * np.finfo(np.float64).eps * max(abs(common_end_time_s), 1.0)
    edges = np.asarray(edges_m, dtype=np.float64)
    if edges.ndim != 1 or edges.size < 3 or np.any(np.diff(edges) <= 0.0):
        raise SemigroupWorkflowError("event timeline requires a valid frozen CR1 edge grid")
    identity_source = np.minimum(
        np.arange(edges.size, dtype=np.int64), edges.size - 2
    )
    coverage: dict[str, int] = {}
    for multiplicity in (16, 32, 64):
        route = f"m{multiplicity}"
        previous = identity_source.copy()
        sequence: list[tuple[int, int, int]] = []
        states = [
            state for state in runs[multiplicity].accepted_states
            if float(state["physical_time_s"]) <= common_end_time_s + time_tolerance
        ]
        coverage[route] = len(states)
        for state_index, state in enumerate(states):
            evaluation = state.get("_accepted_evaluation")
            if evaluation is None:
                raise SemigroupWorkflowError("accepted timeline state lacks topology evaluation")
            current = np.asarray(evaluation.source_cell_indices, dtype=np.int64)
            if current.shape != previous.shape:
                raise SemigroupWorkflowError("accepted topology does not match the frozen step-244 face grid")
            for face in np.flatnonzero(previous != current):
                event = (int(face), int(previous[face]), int(current[face]))
                sequence.append(event)
                rows.append({
                    "route": route, "physical_time_s": float(state["physical_time_s"]),
                    "face_index": event[0], "old_source_cell": event[1], "new_source_cell": event[2],
                    "event_kind": (
                        "STEP244_IDENTITY_TO_FIRST_LEAF"
                        if state_index == 0 else "SOURCE_CELL_CHANGE"
                    ),
                })
            previous = current
        sequences[route] = sequence
    set_values = {route: set(values) for route, values in sequences.items()}
    if len({tuple(values) for values in sequences.values()}) == 1:
        event_order = "SAME_EVENTS_DIFFERENT_DISCRETE_TIMING"
        event_set = "SAME_EVENT_SET"
    elif len({frozenset(values) for values in set_values.values()}) == 1:
        event_order = "DIFFERENT_EVENT_ORDER"
        event_set = "SAME_EVENT_SET"
    elif all(values for values in sequences.values()):
        event_order = "DIFFERENT_EVENT_SET"
        event_set = "DIFFERENT_EVENT_SET"
    else:
        event_order = "INSUFFICIENT_EVENT_SEQUENCE_COVERAGE"
        event_set = "INSUFFICIENT_EVENT_SET_COVERAGE"
    return rows, {
        "event_set": event_set,
        "event_order": event_order,
        "common_end_time_s": common_end_time_s,
        "accepted_leaf_counts_within_common_horizon": coverage,
        "initial_partition": "STEP244_IDENTITY_CDF_SOURCE_PARTITION",
        "sequences": {route: [list(item) for item in values] for route, values in sequences.items()},
    }


def _root_cause(
    *,
    dynamic: Mapping[str, Any],
    frozen: Mapping[str, Any],
    prescribed: Mapping[str, Any],
    topology_differs_dynamic: bool,
    topology_differs_frozen: bool,
    local: Mapping[str, Any],
    timeline: Mapping[str, Any],
    localization: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    dynamic_status = str(dynamic["classification"])
    frozen_status = str(frozen["classification"])
    prescribed_status = str(prescribed["classification"])
    local_family = str(local["LOCAL_ROOT_FAMILY"])
    local_ratio_trend = str(local["DELTA_X_OVER_H_CONVERGENCE"])
    localized = max(
        (float(row.get("event_neighborhood_population_L1_fraction", 0.0)) for row in localization),
        default=0.0,
    )
    if dynamic_status == "PHYSICAL_SEMIGROUP_CONVERGENCE" and frozen_status == "PHYSICAL_SEMIGROUP_CONVERGENCE":
        primary = "TOPOLOGY_DIVERGENCE_PHYSICALLY_BENIGN"
        secondary = "DISCRETE_TOPOLOGY_SENSITIVITY" if topology_differs_dynamic or topology_differs_frozen else None
    elif (
        frozen_status == "PHYSICAL_SEMIGROUP_CONVERGENCE"
        and prescribed_status == "PHYSICAL_SEMIGROUP_CONVERGENCE"
        and dynamic_status in {"PHYSICAL_SEMIGROUP_DIVERGENCE", "PHYSICAL_SEMIGROUP_STAGNATION"}
        and local_family == "LOCAL_CONTINUOUS_ROOT_FAMILY_OBSERVED"
        and local_ratio_trend == "OBSERVED_CAUCHY_LIKE_TREND"
    ):
        primary = "DYNAMIC_MATRIX_TIME_COUPLING_INCONSISTENCY"
        secondary = "LOCAL_TIME_CENTERING_GEOMETRY"
    elif (
        frozen_status in {"PHYSICAL_SEMIGROUP_DIVERGENCE", "PHYSICAL_SEMIGROUP_STAGNATION"}
        and topology_differs_frozen
    ):
        primary = "CR1_REMAP_SEMIGROUP_DEFECT"
        secondary = "TOPOLOGY_EVENT_CORRELATION"
    elif (
        str(timeline["event_order"]) == "DIFFERENT_EVENT_ORDER"
        and dynamic_status != "PHYSICAL_SEMIGROUP_CONVERGENCE"
    ):
        primary = "EVENT_ORDERING_TIME_DISCRETIZATION_DEFECT"
        secondary = "TOPOLOGY_EVENT_ORDER"
    elif (
        dynamic_status != "PHYSICAL_SEMIGROUP_CONVERGENCE"
        and frozen_status != "PHYSICAL_SEMIGROUP_CONVERGENCE"
        and prescribed_status != "PHYSICAL_SEMIGROUP_CONVERGENCE"
    ):
        primary = "MIXED_TIME_REMAP_COUPLING_DEFECT"
        secondary = "MULTIPLE_NONCONVERGENT_DIAGNOSTIC_PATHS"
    else:
        primary = "INSUFFICIENT_SEMIGROUP_EVIDENCE"
        secondary = None
    statuses = {
        "TOPOLOGY_DIVERGENCE_PHYSICALLY_BENIGN": "DIAG_TOPOLOGY_DIVERGENCE_PHYSICALLY_BENIGN",
        "DYNAMIC_MATRIX_TIME_COUPLING_INCONSISTENCY": "DIAG_DYNAMIC_MATRIX_TIME_COUPLING_INCONSISTENCY",
        "CR1_REMAP_SEMIGROUP_DEFECT": "DIAG_CR1_REMAP_SEMIGROUP_DEFECT",
        "EVENT_ORDERING_TIME_DISCRETIZATION_DEFECT": "DIAG_EVENT_ORDERING_TIME_DISCRETIZATION_DEFECT",
        "MIXED_TIME_REMAP_COUPLING_DEFECT": "DIAG_MIXED_TIME_REMAP_COUPLING_DEFECT",
        "INSUFFICIENT_SEMIGROUP_EVIDENCE": "DIAG_INSUFFICIENT_SEMIGROUP_EVIDENCE",
    }
    return {
        "PRIMARY_ROOT_CAUSE": primary,
        "SECONDARY_ROOT_CAUSE": secondary,
        "TOP_LEVEL_STATUS": statuses[primary],
        "evidence": {
            "dynamic_semigroup": dynamic_status,
            "frozen_semigroup": frozen_status,
            "prescribed_x_semigroup": prescribed_status,
            "topology_differs_dynamic": topology_differs_dynamic,
            "topology_differs_frozen": topology_differs_frozen,
            "local_root_family": local_family,
            "local_delta_x_over_h_trend": local_ratio_trend,
            "event_order": timeline["event_order"],
            "maximum_dynamic_common_time_event_neighborhood_population_fraction": localized,
        },
    }


def _event_time_spread_summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compare modes only for the same face and selected source boundary."""

    usable = {
        "EVENT_TIME_ESTIMATED",
        "EVENT_TIME_BRACKETED_TOPOLOGY_TRANSITION",
    }
    grouped: dict[tuple[int, str], list[Mapping[str, Any]]] = {}
    for row in rows:
        if row.get("status") not in usable:
            continue
        key = (int(row["face_index"]), float(row["source_boundary_m"]).hex())
        grouped.setdefault(key, []).append(row)
    per_event: list[dict[str, Any]] = []
    spreads: list[float] = []
    for (face, boundary_hex), group in sorted(grouped.items()):
        times = [float(item["event_physical_time_s"]) for item in group]
        spread = max(times) - min(times) if len(times) >= 2 else None
        if spread is not None:
            spreads.append(spread)
        per_event.append({
            "face_index": face,
            "source_boundary_m": float.fromhex(boundary_hex),
            "mode_count": len(group),
            "modes": sorted(str(item["mode"]) for item in group),
            "event_time_spread_across_modes_s": spread,
            "statuses": sorted(str(item["status"]) for item in group),
        })
    return {
        "per_event": per_event,
        "events_with_two_or_more_modes": len(spreads),
        "maximum_same_event_mode_spread_s": max(spreads) if spreads else None,
        "median_same_event_mode_spread_s": float(np.median(spreads)) if spreads else None,
        "semantics": "only same face plus same selected source boundary are compared across modes",
    }


def _method_decision(root: Mapping[str, Any]) -> dict[str, Any]:
    cause = str(root["PRIMARY_ROOT_CAUSE"])
    if cause == "TOPOLOGY_DIVERGENCE_PHYSICALLY_BENIGN":
        method = "REVISIT_CLOSURE_PRIMARY_SOLVER_WITH_PHYSICAL_STATE_CONVERGENCE_AUTHORITY"
        flags = (False, False, False)
    elif cause == "DYNAMIC_MATRIX_TIME_COUPLING_INCONSISTENCY":
        method = "COUPLED_CHARACTERISTIC_TIME_INTEGRATOR_V2"
        flags = (False, True, False)
    elif cause == "CR1_REMAP_SEMIGROUP_DEFECT":
        method = "CR2_MONOTONE_LINEAR_CONSERVATIVE_REMAP_PROTOTYPE"
        flags = (True, False, False)
    elif cause == "EVENT_ORDERING_TIME_DISCRETIZATION_DEFECT":
        method = "EVENT_ALIGNED_CHARACTERISTIC_STEPPING_PROTOTYPE"
        flags = (False, False, True)
    else:
        method = "NO_METHOD_CHANGE_AUTHORIZED_PENDING_FURTHER_DIAGNOSTIC_EVIDENCE"
        flags = (False, False, False)
    return {
        "RECOMMENDED_NEXT_METHOD": method,
        "CR2_AUTHORIZED": flags[0],
        "COUPLED_TIME_V2_AUTHORIZED": flags[1],
        "EVENT_ALIGNED_STEPPING_AUTHORIZED": flags[2],
        "implementation_this_run": False,
    }


def _reinterpret_same_prestate(
    path: Path, *, dynamic_summary: Mapping[str, Any]
) -> dict[str, Any]:
    """Reuse the prior read-only same-state/dt evidence without recomputing it."""

    with path.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        fields = tuple(reader.fieldnames or ())
    dt_values = sorted({row.get("candidate_dt_s", "") for row in rows if row.get("candidate_dt_s", "")})
    topology_values = sorted({
        row.get("topology_signature", "") for row in rows if row.get("topology_signature", "")
    })
    raw_picard_classes = sorted({
        row.get("raw_picard_classification", "")
        for row in rows if row.get("raw_picard_classification", "")
    })
    return {
        "source_csv": str(path), "source_sha256": _sha256_file(path), "row_count": len(rows),
        "candidate_dt_s_values": dt_values, "topology_signature_count": len(topology_values),
        "raw_picard_classifications": raw_picard_classes,
        "available_raw_fields": [field for field in fields if field.startswith("raw_")],
        "classification": "INSUFFICIENT_SAME_PRESTATE_PHYSICAL_INCREMENT_EVIDENCE",
        "new_dynamic_semigroup_context": str(dynamic_summary["classification"]),
        "interpretation_scope": (
            "The archived same-prestate table contains raw closure-map telemetry, not accepted "
            "physical increments divided by h; it cannot independently establish physical "
            "consistency or time-discretization inconsistency."
        ),
        "recomputed": False,
    }


def _write_reports(report_root: Path, payloads: Mapping[str, Mapping[str, Any] | str]) -> None:
    for filename, payload in payloads.items():
        _write_markdown(report_root / filename, REPORT_TITLES[filename], payload)


def _run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    restart_checkpoint = Path(args.restart_checkpoint)
    formal_trace = Path(args.formal_trace_csv)
    formal_phase_a = Path(args.formal_phase_a_csv)
    same_prestate = Path(args.same_prestate_csv)
    if output_root.exists() or report_root.exists():
        raise SemigroupWorkflowError("refusing to overwrite existing semigroup output/report roots")
    for path, label in (
        (restart_checkpoint, "frozen step-244 restart checkpoint"),
        (formal_trace, "formal trace"),
        (formal_phase_a, "formal Phase-A ladder"),
        (same_prestate, "same-prestate different-dt evidence"),
    ):
        if not path.is_file():
            raise SemigroupWorkflowError(f"{label} is unavailable")
    if _sha256_file(same_prestate) != EXPECTED_SAME_PRESTATE_SHA256:
        raise SemigroupWorkflowError("same-prestate different-dt evidence differs from the frozen pathology record")
    # Check the clean source identity before creating any caller-selected
    # report/output path: a path inside the worktree must never make this
    # runner fail its own SOURCE_CLEAN contract.
    source_identity = _source_identity(require_clean=not bool(args.allow_dirty_source))
    output_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    (output_root / "figures").mkdir()
    started = time.monotonic()
    if _sha256_file(formal_trace) != prior.EXPECTED_FORMAL_TRACE_SHA256:
        raise SemigroupWorkflowError("formal trace checksum differs from frozen record")
    formal = prior._load_formal_phase_a_ladder(formal_phase_a)
    context = build_frozen_canonical_context()
    config, checkpoint_binding = prior.checkpoint_bound_config(
        context=context, restart_checkpoint=restart_checkpoint
    )
    source = CharacteristicReferenceSolver.load_checkpoint(config=config, path=restart_checkpoint)
    if int(source.step) != RESTART_STEP or prior.accepted_state_hash(source) != RESTART_STATE_HASH:
        raise SemigroupWorkflowError("restart checkpoint does not bind frozen step-244 state")

    runs = {
        multiplicity: prior._run_fixed_substeps(
            config=config, restart_checkpoint=restart_checkpoint, m=multiplicity
        )
        for multiplicity in (16, 32, 64)
    }
    baseline = _formal_reproduction(runs, formal)
    if baseline["status"] != "PASS_SEMIGROUP_BASELINE_REPRODUCTION":
        final = {
            "STATUS": "FAIL_SEMIGROUP_BASELINE_REPRODUCTION",
            "BRANCH": source_identity["git_branch"], "COMMIT": source_identity["git_commit"],
            "SOURCE_CLEAN": not bool(source_identity["git_status"]), "TESTS": str(args.test_status),
            "TIME_REFERENCE_V2": "NOT_ASSIGNED", "M16_AUTHORITY": "NOT_AUTHORITY",
            "NEXT_ACTION": "Stop: frozen Phase-A reproduction differs before semigroup diagnosis.",
        }
        _write_reports(report_root, {
            "00_baseline.md": baseline,
            "11_final_acceptance_report.md": final,
            "12_reproduction_commands.md": {"command": "See submitted job provenance; baseline failed closed."},
        })
        _write_json(output_root / "analysis_provenance.json", {
            "task_name": TASK_NAME, "top_status": final["STATUS"], "source": source_identity,
            "baseline": baseline, "test_status": str(args.test_status),
            "diagnostic_only": True, "time_reference_v2": "NOT_ASSIGNED",
            "frozen_fixture_hash": str(context.fixture_hash),
            "frozen_contract_hash": str(context.contract_hash),
            "implementation": _implementation_provenance(),
        })
        return 2, final

    # Preserve every accepted Phase-A state required by the task; no later
    # audit relies on a silently regenerated endpoint.
    accepted_states_by_m = _accepted_states_by_m(runs)
    archive_index, archive_sha = prior._save_accepted_states(
        output_root=output_root,
        states_by_m=accepted_states_by_m,
        edges_m=np.asarray(context.edges_m, dtype=np.float64),
    )
    baseline_rows = [row for run in runs.values() for row in run.accepted_rows]
    _write_csv(output_root / "baseline_accepted_substeps.csv", baseline_rows, ("m", "substep"))

    schedule_times = np.asarray(
        [float(source.time_s)] + [float(item["physical_time_s"]) for item in runs[16].accepted_states],
        dtype=np.float64,
    )
    schedule_xb = np.asarray(
        [float(source.matrix_xb)] + [float(item["xB"]) for item in runs[16].accepted_states],
        dtype=np.float64,
    )
    trajectory = PiecewiseLinearMatrixTrajectory(schedule_times, schedule_xb)
    dynamic_rows, dynamic_summary = _semigroup_rows(source, mode="DYNAMIC", trajectory=None)
    frozen_rows, frozen_summary = _semigroup_rows(source, mode="FROZEN_MATRIX", trajectory=None)
    prescribed_rows, prescribed_summary = _semigroup_rows(source, mode="PRESCRIBED_X", trajectory=trajectory)
    same_prestate_reinterpretation = _reinterpret_same_prestate(
        same_prestate, dynamic_summary=dynamic_summary
    )
    _write_csv(output_root / "dynamic_semigroup.csv", _semigroup_csv_rows(dynamic_rows), ("mode", "h_s", "status"))
    _write_csv(output_root / "frozen_matrix_semigroup.csv", _semigroup_csv_rows(frozen_rows), ("mode", "h_s", "status"))
    _write_csv(output_root / "prescribed_x_semigroup.csv", _semigroup_csv_rows(prescribed_rows), ("mode", "h_s", "status"))
    _write_csv(
        output_root / "semigroup_topology_paths.csv",
        [
            *_semigroup_topology_path_rows(dynamic_rows),
            *_semigroup_topology_path_rows(frozen_rows),
            *_semigroup_topology_path_rows(prescribed_rows),
        ],
        ("mode", "h_s", "path", "leaf_index", "remap_topology_signature"),
    )
    _write_csv(
        output_root / "semigroup_state_errors.csv",
        [*_semigroup_error_rows(dynamic_rows), *_semigroup_error_rows(frozen_rows), *_semigroup_error_rows(prescribed_rows)],
        ("mode", "h_s", "comparison", "E_physical_max_relative"),
    )

    local_rows, local_summary = _local_root_family(source, trajectory=trajectory)
    _write_csv(output_root / "local_root_family.csv", local_rows, ("record_type", "h_s", "x_trial"))

    common_states = {
        "m16": _baseline_state(runs[16].accepted_states[0]),
        "m32": _baseline_state(runs[32].accepted_states[1]),
        "m64": _baseline_state(runs[64].accepted_states[3]),
    }
    proximity_rows = topology_event_proximity_rows(common_states, edges_m=np.asarray(context.edges_m, dtype=np.float64))
    _write_csv(output_root / "topology_event_proximity.csv", proximity_rows, ("route", "face_index", "eta_abs"))
    event_boundaries = _event_boundaries(common_states, edges_m=np.asarray(context.edges_m, dtype=np.float64))
    event_time_rows = [
        {
            **boundary,
            **_estimate_event_time(
                source,
                face_index=int(boundary["face_index"]),
                boundary_m=float(boundary["source_boundary_m"]),
                mode=mode,
                trajectory=trajectory if mode == "PRESCRIBED_X" else None,
            ),
        }
        for boundary in event_boundaries
        for mode in ("FROZEN_MATRIX", "PRESCRIBED_X", "DYNAMIC")
    ]
    _write_csv(output_root / "topology_event_times.csv", event_time_rows, ("mode", "face_index", "status"))
    event_time_spread = _event_time_spread_summary(event_time_rows)
    timeline_rows, timeline_summary = _event_timeline(
        runs, edges_m=np.asarray(context.edges_m, dtype=np.float64)
    )
    _write_csv(output_root / "topology_event_ordering.csv", timeline_rows, ("route", "physical_time_s", "face_index"))

    localization_rows: list[dict[str, Any]] = []
    for left, right in (("m16", "m32"), ("m16", "m64"), ("m32", "m64")):
        row = event_error_localization(
            common_states[left], common_states[right], edges_m=np.asarray(context.edges_m, dtype=np.float64)
        )
        distances = physical_state_distance(
            common_states[left], common_states[right], edges_m=np.asarray(context.edges_m, dtype=np.float64)
        )
        localization_rows.append({"left_route": left, "right_route": right, **row, **distances})
    _write_csv(output_root / "event_error_localization.csv", localization_rows, ("left_route", "right_route"))

    # Dynamic topology divergence is evaluated at the genuinely shared
    # physical time 1/16, using m16/m32/m64 accepted states.  Do not infer it
    # from the final leaf of paths with different leaf sizes.
    common_topology = {
        f"{left}_{right}": topology_comparison(common_states[left], common_states[right])
        for left, right in (("m16", "m32"), ("m16", "m64"), ("m32", "m64"))
    }
    topology_differs_dynamic = any(
        not bool(item["topology_equal"]) for item in common_topology.values()
    )
    # Frozen paths have no independent m16/m32/m64 trajectory.  Record only
    # the nontrivial difference in their *sets* of CR1 regimes; leaf count is
    # explicitly not an equality gate.
    topology_differs_frozen = any(
        row.get("status") == "SUCCESS"
        and bool(row["topology_path_h_vs_h2x2"]["different_discrete_topology_regimes"])
        for row in frozen_rows
    )
    root = _root_cause(
        dynamic=dynamic_summary, frozen=frozen_summary, prescribed=prescribed_summary,
        topology_differs_dynamic=topology_differs_dynamic,
        topology_differs_frozen=topology_differs_frozen, local=local_summary,
        timeline=timeline_summary, localization=localization_rows,
    )
    decision = _method_decision(root)
    _write_json(output_root / "method_decision.json", {**root, **decision})

    dynamic_e = list(dynamic_summary.get("E_values", []))
    frozen_e = list(frozen_summary.get("E_values", []))
    prescribed_e = list(prescribed_summary.get("E_values", []))
    eta_values = [float(row["eta_abs"]) for row in proximity_rows]
    final = {
        "STATUS": root["TOP_LEVEL_STATUS"],
        "BRANCH": source_identity["git_branch"], "COMMIT": source_identity["git_commit"],
        "SOURCE_CLEAN": not bool(source_identity["git_status"]), "TESTS": str(args.test_status),
        "PHI_OPERATOR_DEFINED": "READ_ONLY_CLONE_ORDINARY_OR_QUALIFIED_P2_P4_ONLY",
        "DYNAMIC_SEMIGROUP": dynamic_summary["classification"],
        "DYNAMIC_E_H": dynamic_e[0] if len(dynamic_e) > 0 else None,
        "DYNAMIC_E_H2": dynamic_e[1] if len(dynamic_e) > 1 else None,
        "DYNAMIC_E_H4": dynamic_e[2] if len(dynamic_e) > 2 else None,
        "DYNAMIC_OBSERVED_RATIO": dynamic_summary.get("observed_ratios"),
        "FROZEN_MATRIX_SEMIGROUP": frozen_summary["classification"],
        "FROZEN_E_H": frozen_e[0] if len(frozen_e) > 0 else None,
        "FROZEN_E_H2": frozen_e[1] if len(frozen_e) > 1 else None,
        "FROZEN_E_H4": frozen_e[2] if len(frozen_e) > 2 else None,
        "PRESCRIBED_X_SEMIGROUP": prescribed_summary["classification"],
        "PRESCRIBED_E_H": prescribed_e[0] if len(prescribed_e) > 0 else None,
        "PRESCRIBED_E_H2": prescribed_e[1] if len(prescribed_e) > 1 else None,
        "TOPOLOGY_DIFFERS_DYNAMIC": topology_differs_dynamic,
        "TOPOLOGY_DIFFERS_FROZEN": topology_differs_frozen,
        "PHYSICAL_STATE_CONVERGES_DESPITE_TOPOLOGY": (
            dynamic_summary["classification"] == "PHYSICAL_SEMIGROUP_CONVERGENCE"
            and (topology_differs_dynamic or topology_differs_frozen)
        ),
        "LOCAL_ROOT_FAMILY": local_summary["LOCAL_ROOT_FAMILY"],
        "DELTA_X_OVER_H_CONVERGENCE": local_summary["DELTA_X_OVER_H_CONVERGENCE"],
        "TOPOLOGY_EVENT_FACES": len({int(row["face_index"]) for row in proximity_rows}),
        "EVENT_PROXIMITY_MIN": min(eta_values) if eta_values else None,
        "EVENT_PROXIMITY_MEDIAN": float(np.median(eta_values)) if eta_values else None,
        "EVENT_SET": timeline_summary["event_set"], "EVENT_ORDER": timeline_summary["event_order"],
        "EVENT_TIME_SPREAD": event_time_spread["maximum_same_event_mode_spread_s"],
        "ERROR_FRACTION_NEAR_EVENT_CELLS": max(
            (float(row["event_neighborhood_population_L1_fraction"]) for row in localization_rows), default=None
        ),
        "PRIMARY_ROOT_CAUSE": root["PRIMARY_ROOT_CAUSE"], "SECONDARY_ROOT_CAUSE": root["SECONDARY_ROOT_CAUSE"],
        "SAME_PRESTATE_DIFFERENT_DT": same_prestate_reinterpretation["classification"],
        "TIME_REFERENCE_V2": "NOT_ASSIGNED", "M16_AUTHORITY": "NOT_AUTHORITY",
        **decision,
        "PF_SOURCE_MODIFIED": False, "CUDA_RERUN": False, "PHYSICAL_RETUNING": False, "GP_RELEASE_RUN": False,
        "TOP_5_FINDINGS": [
            f"Dynamic physical semigroup classification: {dynamic_summary['classification']}.",
            f"Frozen-matrix physical semigroup classification: {frozen_summary['classification']}.",
            f"Prescribed-x bridge classification: {prescribed_summary['classification']}.",
            f"Topology event-order evidence: {timeline_summary['event_order']}.",
            f"Primary root-cause classification: {root['PRIMARY_ROOT_CAUSE']}.",
        ],
        "P0_BLOCKERS": ["TIME_REFERENCE_V2_NOT_ASSIGNED", root["TOP_LEVEL_STATUS"]],
        "NEXT_ACTION": "Await explicit authorization before implementing the selected next-method prototype.",
        "KEY_REPORTS": [
            f"reports/{TASK_NAME}/02_dynamic_semigroup.md",
            f"reports/{TASK_NAME}/03_frozen_matrix_semigroup.md",
            f"reports/{TASK_NAME}/09_semigroup_root_cause.md",
            f"reports/{TASK_NAME}/11_final_acceptance_report.md",
        ],
    }
    reports = {
        "00_baseline.md": {**baseline, "accepted_state_archive_sha256": archive_sha, "accepted_state_count": len(archive_index)},
        "01_one_step_operator_contract.md": {
            "Phi_h": "ephemeral in-memory clone; ordinary direct/P2/P4 production closure only",
            "no_global_state_mutation": True, "no_checkpoint_write": True,
            "generic_root_fallback": False, "dynamic_time_centering": "0.5*(x_n+x_nplus1)",
            "frozen_matrix_isolation": "population-to-xB feedback removed only",
            "prescribed_x_bridge": "shared m16 piecewise-linear xB(t), same endpoint midpoint rule",
        },
        "02_dynamic_semigroup.md": {
            "summary": dynamic_summary, "levels": dynamic_rows,
            "explicit_topology_path_output": "outputs/kwn_cr1_semigroup_audit_v1/semigroup_topology_paths.csv",
        },
        "03_frozen_matrix_semigroup.md": {
            "summary": frozen_summary, "levels": frozen_rows,
            "explicit_topology_path_output": "outputs/kwn_cr1_semigroup_audit_v1/semigroup_topology_paths.csv",
        },
        "04_prescribed_x_semigroup.md": {
            "summary": prescribed_summary, "trajectory_times_s": schedule_times, "trajectory_xB": schedule_xb,
            "levels": prescribed_rows,
            "explicit_topology_path_output": "outputs/kwn_cr1_semigroup_audit_v1/semigroup_topology_paths.csv",
        },
        "05_time_centering_local_consistency.md": local_summary,
        "06_topology_event_proximity.md": {
            "row_count": len(proximity_rows),
            "common_time_topology": common_topology,
            "event_boundaries": event_boundaries,
            "event_time_rows": event_time_rows,
            "event_time_spread": event_time_spread,
        },
        "07_topology_event_timeline.md": {**timeline_summary, "timeline_rows": timeline_rows},
        "08_event_localized_state_error.md": {"rows": localization_rows, "actual_fraction_not_a_pass_threshold": True},
        "09_semigroup_root_cause.md": {**root, "same_prestate_different_dt_reinterpretation": same_prestate_reinterpretation},
        "10_method_decision.md": decision,
        "11_final_acceptance_report.md": final,
        "12_reproduction_commands.md": {
            "command": f"PYTHONPATH=src {sys.executable} scripts/run_kwn_cr1_semigroup_audit_v1.py audit --restart-checkpoint {restart_checkpoint} --formal-trace-csv {formal_trace} --formal-phase-a-csv {formal_phase_a} --same-prestate-csv {same_prestate} --output-root <new-output-root> --report-root <new-report-root>",
            "no_time_reference_assignment": True, "no_cr2_or_event_aligned_implementation": True,
        },
    }
    _write_reports(report_root, reports)
    provenance = {
        "task_name": TASK_NAME, "top_status": final["STATUS"], "source": source_identity,
        "runtime": _runtime_provenance(), "frozen_canonical_snapshot": frozen_canonical_snapshot_provenance(),
        "frozen_fixture_hash": str(context.fixture_hash),
        "frozen_contract_hash": str(context.contract_hash),
        "implementation": _implementation_provenance(),
        "restart_checkpoint": str(restart_checkpoint), "restart_checkpoint_sha256": _sha256_file(restart_checkpoint),
        "restart_state_hash": RESTART_STATE_HASH, "formal_trace_csv": str(formal_trace),
        "formal_trace_sha256": _sha256_file(formal_trace), "formal_phase_a_csv": str(formal_phase_a),
        "formal_phase_a_sha256": _sha256_file(formal_phase_a), "same_prestate_csv": str(same_prestate),
        "same_prestate_sha256": _sha256_file(same_prestate), "checkpoint_config_binding": checkpoint_binding,
        "accepted_state_archive_sha256": archive_sha, "test_status": str(args.test_status),
        "runtime_s": time.monotonic() - started, "diagnostic_only": True,
        "generic_root_selection_used": False, "scalar_root_fallback_used": False,
        "time_reference_v2": "NOT_ASSIGNED", "pf_source_modified": False,
        "cuda_rerun": False, "physical_retuning": False, "gp_release": False,
    }
    _write_json(output_root / "analysis_provenance.json", provenance)
    print(json.dumps(_json_safe(final), indent=2, sort_keys=True))
    return 0, final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    audit = commands.add_parser("audit")
    audit.add_argument("--restart-checkpoint", type=Path, required=True)
    audit.add_argument("--formal-trace-csv", type=Path, required=True)
    audit.add_argument("--formal-phase-a-csv", type=Path, required=True)
    audit.add_argument("--same-prestate-csv", type=Path, required=True)
    audit.add_argument("--output-root", type=Path, required=True)
    audit.add_argument("--report-root", type=Path, required=True)
    audit.add_argument("--allow-dirty-source", action="store_true")
    audit.add_argument("--test-status", default="NOT_RUN_BY_SEMIGROUP_RUNNER")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    try:
        code, _ = _run(args)
    except Exception as error:
        print(f"{type(error).__name__}: {error}", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
