#!/usr/bin/env python3
"""Frozen production characteristic-trace audit and qualification runner.

This orchestration layer is deliberately narrow.  It reconstructs a frozen
step-244 state, audits the *actual* production backward characteristic map,
and compares it with the independently qualified exact flow.  It never opens
a nonlinear closure, changes CR1 remapping, runs PF/CUDA, or assigns a dynamic
time reference.
"""

from __future__ import annotations

import argparse
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
from numpy.typing import NDArray


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.characteristic_semigroup import phi_compose  # noqa: E402
from kwn_mvp.conservative_remap import (  # noqa: E402
    _TRACE_SUBCELLS_PER_CELL,
    characteristic_remap_partition,
    characteristic_trace_topology,
    conservative_remap_piecewise_constant,
)
from kwn_mvp.frozen_exact_flow_reference import (  # noqa: E402
    FrozenAutonomousExactFlow,
    FrozenAutonomousGrowthLaw,
    FrozenExactFlowReferenceError,
    PiecewiseConstantCumulativeMeasure,
    measure_error_metrics,
    observed_orders,
)
from kwn_mvp.production_trace_audit import (  # noqa: E402
    ProductionAutonomousTOFTable,
    ProductionTraceAuditConfig,
    ProductionTraceAuditError,
    public_production_trace_probe,
)
from scripts import run_kwn_frozen_exact_flow_reference_v1 as exact_prior  # noqa: E402
from scripts import run_kwn_frozen_semigroup_decomposition_v1 as frozen_prior  # noqa: E402


TASK_NAME = "kwn_production_trace_audit_v1"
REQUIRED_BRANCH = "codex/kwn-production-trace-audit-v1"
FROZEN_ANCESTOR = "7759c83ca4da7fd930e0d67dfbd15788095702cd"
EXPECTED_FROZEN_U0_HASH = "7de7d098e1ee4a44e291d4771854fd8dfb8a05cdb5e404ae176d22d7bc88c98a"
EXPECTED_RESTART_SHA256 = "c0bc8dd946769550d780763d5a73446b929bac15c5bf7a04895a6288a2314770"
FINAL_HORIZON_S = 1.0 / 128.0
H_LADDER_S = tuple(1.0 / float(2**power) for power in range(7, 13))
LEGACY_SUBCELLS_PER_CELL = 16
REFERENCE_SEMIGROUP_RELATIVE = 1.3374806576082248e-12
REFERENCE_TAU_SHADOW_RELATIVE = 9.187923568012652e-15
REFERENCE_ROUNDTRIP_RELATIVE = 0.0
REFERENCE_FLOOR_RELATIVE = max(
    REFERENCE_SEMIGROUP_RELATIVE,
    REFERENCE_TAU_SHADOW_RELATIVE,
    REFERENCE_ROUNDTRIP_RELATIVE,
)
TRACE_ENVELOPE_RELATIVE = 10.0 * REFERENCE_FLOOR_RELATIVE

REPORT_TITLES = {
    "00_baseline_reproduction.md": "Baseline reproduction",
    "01_production_trace_architecture.md": "Production trace architecture",
    "02_growth_kernel_parity.md": "Shared growth-kernel parity",
    "03_all_face_trace_vs_exact.md": "All-face production trace versus exact flow",
    "04_region_conditioning.md": "Trace error by physical region and conditioning",
    "05_tau_table_audit.md": "Production Tau-table audit",
    "06_interpolation_inversion_audit.md": "Interpolation and inversion audit",
    "07_branch_critical_radius_audit.md": "Branch and critical-radius audit",
    "08_single_step_h_scaling.md": "Single-step trace h scaling",
    "09_repeated_trace_accumulation.md": "Repeated trace-only composition",
    "10_component_ablation.md": "Trace component ablation",
    "11_trace_root_cause.md": "Trace root-cause decision",
    "12_minimal_trace_fix.md": "Minimal trace fix",
    "13_trace_kernel_qualification.md": "Frozen trace-kernel qualification",
    "14_corrected_cr1_vs_exact.md": "Corrected CR1 versus exact pushforward",
    "15_model_boundary.md": "Frozen-model boundary",
    "16_final_acceptance_report.md": "Final trace-audit acceptance report",
    "17_reproduction_commands.md": "Reproduction commands",
}


class ProductionTraceWorkflowError(RuntimeError):
    """A fail-closed error in the frozen trace audit."""


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.ndarray):
        return [_json_safe(item) for item in value.tolist()]
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _csv_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _csv_safe(value.item())
    if isinstance(value, (Mapping, list, tuple, np.ndarray)):
        return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]], fallback_fields: Sequence[str]) -> None:
    materialized = [{str(key): _csv_safe(value) for key, value in dict(row).items()} for row in rows]
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


def _write_report(path: Path, title: str, payload: Mapping[str, Any] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if isinstance(payload, str):
        body = payload.rstrip()
    else:
        body = "```json\n" + json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True) + "\n```"
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")


def _write_reports(report_root: Path, payloads: Mapping[str, Mapping[str, Any] | str]) -> None:
    unknown = set(payloads).difference(REPORT_TITLES)
    if unknown:
        raise ProductionTraceWorkflowError(f"unrecognized trace report names: {sorted(unknown)}")
    for name, title in REPORT_TITLES.items():
        _write_report(report_root / name, title, payloads.get(name, {"status": "NOT_REACHED"}))


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(arguments: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *arguments], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return completed.stdout.strip()


def _source_identity() -> dict[str, Any]:
    try:
        branch = _git(("branch", "--show-current"))
        commit = _git(("rev-parse", "HEAD"))
        status = _git(("status", "--short"))
        ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", FROZEN_ANCESTOR, "HEAD"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).returncode == 0
    except subprocess.CalledProcessError as error:
        raise ProductionTraceWorkflowError("could not resolve trace-audit source identity") from error
    if branch != REQUIRED_BRANCH:
        raise ProductionTraceWorkflowError(f"source branch differs from required {REQUIRED_BRANCH}")
    if not ancestor:
        raise ProductionTraceWorkflowError("trace-audit source does not descend from frozen exact-flow commit")
    if status:
        raise ProductionTraceWorkflowError("trace-audit source tree is dirty")
    return {
        "git_branch": branch,
        "git_commit": commit,
        "git_status": status,
        "frozen_ancestor": FROZEN_ANCESTOR,
        "frozen_ancestor_is_ancestor": ancestor,
    }


def _state_arrays_equal(first: Mapping[str, NDArray[np.generic]], second: Mapping[str, NDArray[np.generic]]) -> bool:
    return set(first) == set(second) and all(np.array_equal(np.asarray(first[key]), np.asarray(second[key])) for key in first)


def _load_json(path: Path, *, purpose: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as error:
        raise ProductionTraceWorkflowError(f"could not read {purpose}") from error
    if not isinstance(value, dict):
        raise ProductionTraceWorkflowError(f"{purpose} is not a JSON object")
    return value


def _baseline_reproduction(baseline_root: Path) -> dict[str, Any]:
    root = baseline_root.resolve()
    status_path = root / "status.txt"
    decision_path = root / "outputs" / "kwn_frozen_exact_flow_reference_v1" / "method_decision.json"
    cr1_path = root / "outputs" / "kwn_frozen_exact_flow_reference_v1" / "cr1_vs_exact_refinement.csv"
    projection_path = root / "outputs" / "kwn_frozen_exact_flow_reference_v1" / "exact_flow_cr1_remap.csv"
    for path, purpose in ((status_path, "baseline status"), (decision_path, "baseline decision"), (cr1_path, "baseline CR1 ladder"), (projection_path, "baseline projection ladder")):
        if not path.is_file():
            raise ProductionTraceWorkflowError(f"baseline reproduction input is unavailable: {purpose}")
    status = status_path.read_text(encoding="utf-8").strip()
    if status != "COMPLETED_KWN_FROZEN_EXACT_FLOW_REFERENCE_V1":
        raise ProductionTraceWorkflowError("baseline exact-flow runner did not complete")
    decision = _load_json(decision_path, purpose="baseline method decision")
    if decision.get("STATUS") != "DIAG_TRACE_INTEGRATOR_ERROR_SUPPORTED":
        raise ProductionTraceWorkflowError("baseline exact-flow decision differs from the authorized trace-audit state")
    if decision.get("CR2_AUTHORIZED") is not False or decision.get("TRACE_INTEGRATOR_AUDIT_AUTHORIZED") is not True:
        raise ProductionTraceWorkflowError("baseline authorization gates differ from the frozen formal result")
    try:
        with cr1_path.open(newline="", encoding="utf-8") as handle:
            cr1_rows = list(csv.DictReader(handle))
        with projection_path.open(newline="", encoding="utf-8") as handle:
            projection_rows = list(csv.DictReader(handle))
    except OSError as error:
        raise ProductionTraceWorkflowError("could not parse baseline refinement artifacts") from error
    if len(cr1_rows) != len(H_LADDER_S) or len(projection_rows) != len(H_LADDER_S):
        raise ProductionTraceWorkflowError("baseline refinement ladders are incomplete")
    baseline_cr1 = [float(row["population_relative_L1"]) for row in cr1_rows]
    baseline_projection = [float(row["population_relative_L1"]) for row in projection_rows]
    expected_cr1 = [
        1.542236538331858e-09,
        2.115415842013939e-09,
        2.11346717971721e-09,
        2.4339632230402895e-09,
        5.566689004070315e-09,
        1.1496305331149472e-08,
    ]
    expected_projection = [
        0.0,
        1.2715508480789759e-12,
        1.9112734643276343e-12,
        2.314976984674558e-12,
        2.6961972582011648e-12,
        3.3189087009327836e-12,
    ]
    if not np.allclose(baseline_cr1, expected_cr1, rtol=1.0e-13, atol=0.0) or not np.allclose(
        baseline_projection, expected_projection, rtol=1.0e-13, atol=0.0
    ):
        raise ProductionTraceWorkflowError("baseline refinement numbers differ from the frozen formal report")
    return {
        "status": "PASS_TRACE_AUDIT_BASELINE_REPRODUCTION",
        "baseline_root": str(root),
        "baseline_status": status,
        "method_decision_sha256": _sha256(decision_path),
        "cr1_refinement_sha256": _sha256(cr1_path),
        "projection_refinement_sha256": _sha256(projection_path),
        "decision": decision,
        "cr1_relative_l1": baseline_cr1,
        "projection_relative_l1": baseline_projection,
    }


def _load_frozen_state(restart_checkpoint: Path, output_root: Path) -> tuple[Any, dict[str, Any], dict[str, NDArray[np.generic]], dict[str, Any]]:
    if not restart_checkpoint.is_file():
        raise ProductionTraceWorkflowError("step-244 restart checkpoint is unavailable")
    if _sha256(restart_checkpoint) != EXPECTED_RESTART_SHA256:
        raise ProductionTraceWorkflowError("step-244 restart checkpoint hash differs")
    try:
        source, baseline, arrays, metadata = frozen_prior._load_exact_u0(restart_checkpoint, output_root)
    except Exception as error:  # the inherited loader owns its specific fail-closed contract
        raise ProductionTraceWorkflowError("could not reconstruct the frozen U0 state") from error
    baseline["frozen_u0_content_hash"] = EXPECTED_FROZEN_U0_HASH
    return source, baseline, arrays, metadata


def _production_velocity(source: Any, frozen_xb: float) -> Any:
    def velocity(radii_m: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(source._velocity_at_radii(np.asarray(radii_m, dtype=np.float64), frozen_xb), dtype=np.float64)

    return velocity


def _extended_parity_radii(edges_m: NDArray[np.float64], law: FrozenAutonomousGrowthLaw) -> NDArray[np.float64]:
    values: list[float] = [*map(float, edges_m), *map(float, np.sqrt(edges_m[:-1] * edges_m[1:]))]
    lower, upper = float(edges_m[0]), float(edges_m[-1])
    values.extend((float(np.nextafter(lower, upper)), float(np.nextafter(upper, lower))))
    critical = law.critical_radius_m
    if critical is not None and lower < critical < upper:
        values.extend((float(np.nextafter(critical, lower)), float(np.nextafter(critical, upper))))
    array = np.asarray(sorted(set(values)), dtype=np.float64)
    return array[(array >= lower) & (array <= upper)]


def _growth_parity_rows(
    *, source: Any, law: FrozenAutonomousGrowthLaw, edges_m: NDArray[np.float64], frozen_xb: float
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    radii = _extended_parity_radii(edges_m, law)
    production = np.asarray(source._velocity_at_radii(radii, frozen_xb), dtype=np.float64)
    shared = law.velocity(radii)
    delta = production - shared
    critical = law.critical_radius_m
    rows: list[dict[str, Any]] = []
    for radius, left, right, difference in zip(radii, production, shared, delta):
        rows.append(
            {
                "radius_m": float(radius),
                "production_growth_m_s": float(left),
                "shared_growth_m_s": float(right),
                "absolute_difference_m_s": abs(float(difference)),
                "bitwise_equal": bool(float(left) == float(right)),
                "sign_equal": bool(np.sign(left) == np.sign(right)),
                "branch": law.branch_name(float(radius)),
                "critical_radius_m": critical,
            }
        )
    g_value = bool(np.array_equal(production, shared))
    sign = bool(np.array_equal(np.sign(production), np.sign(shared)))
    critical_parity = all(row["sign_equal"] for row in rows if row["branch"] != "stationary_critical")
    return rows, {
        "G_VALUE_PARITY": g_value,
        "G_SIGN_PARITY": sign,
        "CRITICAL_RADIUS_PARITY": critical_parity,
        "sample_count": int(radii.size),
        "max_absolute_difference_m_s": float(np.max(np.abs(delta))) if delta.size else 0.0,
        "critical_radius_m": critical,
    }


def _source_cell_indices(edges_m: NDArray[np.float64], departure_faces_m: NDArray[np.float64]) -> NDArray[np.int64]:
    indices = np.searchsorted(edges_m, departure_faces_m, side="right") - 1
    return np.clip(indices, 0, edges_m.size - 2).astype(np.int64)


def _local_face_widths(edges_m: NDArray[np.float64]) -> NDArray[np.float64]:
    widths = np.diff(edges_m)
    result = np.empty(edges_m.shape, dtype=np.float64)
    result[0] = widths[0]
    result[-1] = widths[-1]
    result[1:-1] = np.minimum(widths[:-1], widths[1:])
    return result


def _trace_face_rows(
    *,
    source: Any,
    flow: FrozenAutonomousExactFlow,
    edges_m: NDArray[np.float64],
    frozen_xb: float,
) -> tuple[list[dict[str, Any]], dict[float, dict[str, Any]], dict[float, NDArray[np.float64]], dict[float, NDArray[np.float64]]]:
    velocity = _production_velocity(source, frozen_xb)
    local_width = _local_face_widths(edges_m)
    rows: list[dict[str, Any]] = []
    summaries: dict[float, dict[str, Any]] = {}
    production_departures: dict[float, NDArray[np.float64]] = {}
    exact_departures: dict[float, NDArray[np.float64]] = {}
    for h_s in H_LADDER_S:
        trace, production_status = public_production_trace_probe(
            arrival_faces_m=edges_m,
            duration_s=float(h_s),
            velocity_m_s=velocity,
            lower_radius_m=float(edges_m[0]),
            upper_radius_m=float(edges_m[-1]),
        )
        exact = flow.departure_faces(edges_m, float(h_s))
        production_departures[float(h_s)] = trace.departure_faces_m.copy()
        exact_departures[float(h_s)] = exact.radius_m.copy()
        delta = np.asarray(trace.departure_faces_m - exact.radius_m, dtype=np.float64)
        scale = np.maximum(np.maximum(np.abs(trace.departure_faces_m), np.abs(exact.radius_m)), 1.0e-300)
        prod_source = _source_cell_indices(edges_m, trace.departure_faces_m)
        exact_source = _source_cell_indices(edges_m, exact.radius_m)
        status_mismatch = production_status != exact.status
        for index, (arrival, production, reference, difference, relative, width, source_index, reference_index, prod_status, exact_status) in enumerate(
            zip(
                edges_m,
                trace.departure_faces_m,
                exact.radius_m,
                delta,
                np.abs(delta) / scale,
                local_width,
                prod_source,
                exact_source,
                production_status,
                exact.status,
            )
        ):
            rows.append(
                {
                    "h_s": float(h_s),
                    "face_index": int(index),
                    "arrival_radius_m": float(arrival),
                    "production_departure_radius_m": float(production),
                    "exact_departure_radius_m": float(reference),
                    "delta_radius_m": float(difference),
                    "absolute_radius_error_m": abs(float(difference)),
                    "relative_radius_error": float(relative),
                    "local_cell_width_m": float(width),
                    "error_over_local_cell_width": abs(float(difference)) / max(float(width), 1.0e-300),
                    "production_source_cell_index": int(source_index),
                    "exact_source_cell_index": int(reference_index),
                    "source_cell_mismatch": bool(source_index != reference_index),
                    "production_status": str(prod_status),
                    "exact_status": str(exact_status),
                    "status_mismatch": bool(prod_status != exact_status),
                }
            )
        summaries[float(h_s)] = {
            "h_s": float(h_s),
            "face_count": int(edges_m.size),
            "max_absolute_radius_error_m": float(np.max(np.abs(delta))),
            "rms_absolute_radius_error_m": float(math.sqrt(float(np.mean(np.square(delta))))),
            "median_absolute_radius_error_m": float(np.median(np.abs(delta))),
            "p95_absolute_radius_error_m": float(np.quantile(np.abs(delta), 0.95)),
            "p99_absolute_radius_error_m": float(np.quantile(np.abs(delta), 0.99)),
            "max_relative_radius_error": float(np.max(np.abs(delta) / scale)),
            "source_cell_mismatch_count": int(np.count_nonzero(prod_source != exact_source)),
            "status_mismatch_count": int(np.count_nonzero(status_mismatch)),
            "production_lower_no_inflow_faces": int(trace.lower_no_inflow_face_count),
            "production_upper_no_inflow_faces": int(trace.upper_no_inflow_face_count),
        }
    return rows, summaries, production_departures, exact_departures


def _region_labels(
    *, edges_m: NDArray[np.float64], velocity_m_s: NDArray[np.float64]
) -> tuple[NDArray[np.str_], NDArray[np.float64], dict[str, Any]]:
    """Use flight-time conditioning, rather than an arbitrary nm window."""

    speed = np.abs(np.asarray(velocity_m_s, dtype=np.float64))
    conditioning = 1.0 / np.maximum(speed, 1.0e-300)
    interior = np.arange(edges_m.size)
    nonboundary = interior[(interior > 1) & (interior < edges_m.size - 2)]
    if nonboundary.size == 0:
        raise ProductionTraceWorkflowError("frozen grid has no nonboundary faces for conditioning audit")
    threshold = float(np.quantile(conditioning[nonboundary], 0.95))
    labels = np.empty(edges_m.shape, dtype="<U40")
    for index, value in enumerate(velocity_m_s):
        if index <= 1:
            labels[index] = "NEAR_RMIN_GRID_LOCAL"
        elif index >= edges_m.size - 2:
            labels[index] = "NEAR_RMAX_GRID_LOCAL"
        elif conditioning[index] >= threshold:
            labels[index] = "NEAR_CRITICAL_CONDITIONING_Q95"
        elif value < 0.0:
            labels[index] = "SHRINKING_BULK"
        else:
            labels[index] = "GROWING_BULK"
    return labels, conditioning, {
        "rule": "near-critical faces are the top 5% of nonboundary |1/G|; Rmin/Rmax neighborhoods are the adjacent resolved grid faces",
        "conditioning_quantity": "abs_dTau_dR_s_per_m=1/abs(G)",
        "near_critical_conditioning_threshold_s_per_m": threshold,
        "near_rmin_face_indices": [0, 1],
        "near_rmax_face_indices": [int(edges_m.size - 2), int(edges_m.size - 1)],
    }


def _region_rows(
    *, face_rows: Sequence[Mapping[str, Any]], edges_m: NDArray[np.float64], velocity: NDArray[np.float64]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    labels, conditioning, rule = _region_labels(edges_m=edges_m, velocity_m_s=velocity)
    grouped: dict[tuple[float, str], list[Mapping[str, Any]]] = {}
    for row in face_rows:
        label = str(labels[int(row["face_index"])])
        grouped.setdefault((float(row["h_s"]), label), []).append(row)
    rows: list[dict[str, Any]] = []
    for (h_s, label), values in sorted(grouped.items()):
        errors = np.asarray([float(value["absolute_radius_error_m"]) for value in values], dtype=np.float64)
        relative = np.asarray([float(value["relative_radius_error"]) for value in values], dtype=np.float64)
        indices = np.asarray([int(value["face_index"]) for value in values], dtype=np.int64)
        rows.append(
            {
                "h_s": h_s,
                "region": label,
                "face_count": int(len(values)),
                "max_absolute_radius_error_m": float(np.max(errors)),
                "rms_absolute_radius_error_m": float(math.sqrt(float(np.mean(np.square(errors))))),
                "max_relative_radius_error": float(np.max(relative)),
                "conditioning_min_s_per_m": float(np.min(conditioning[indices])),
                "conditioning_median_s_per_m": float(np.median(conditioning[indices])),
                "conditioning_max_s_per_m": float(np.max(conditioning[indices])),
            }
        )
    return rows, rule


def _table_resolution_rows(
    *,
    source: Any,
    flow: FrozenAutonomousExactFlow,
    edges_m: NDArray[np.float64],
    frozen_xb: float,
) -> tuple[list[dict[str, Any]], dict[int, dict[float, NDArray[np.float64]]]]:
    velocity = _production_velocity(source, frozen_xb)
    rows: list[dict[str, Any]] = []
    departures: dict[int, dict[float, NDArray[np.float64]]] = {}
    for resolution in (16, 32, 64, 128):
        table = ProductionAutonomousTOFTable(
            edges_m=edges_m,
            velocity_m_s=velocity,
            exact_flow=flow,
            config=ProductionTraceAuditConfig(subcells_per_cell=resolution),
        )
        departures[resolution] = {}
        for h_s in H_LADDER_S:
            query = table.query_departure(edges_m, float(h_s))
            exact = flow.departure_faces(edges_m, float(h_s))
            delta = np.asarray(query.radius_m - exact.radius_m, dtype=np.float64)
            scale = np.maximum(np.maximum(np.abs(query.radius_m), np.abs(exact.radius_m)), 1.0e-300)
            departures[resolution][float(h_s)] = query.radius_m.copy()
            rows.append(
                {
                    "trace_auxiliary_subcells_per_population_cell": resolution,
                    "h_s": float(h_s),
                    "max_absolute_radius_error_m": float(np.max(np.abs(delta))),
                    "rms_absolute_radius_error_m": float(math.sqrt(float(np.mean(np.square(delta))))),
                    "max_relative_radius_error": float(np.max(np.abs(delta) / scale)),
                    "status_mismatch_count": int(np.count_nonzero(query.status != exact.status)),
                    "inversion_mode": "PRODUCTION_FIXED_6",
                    "table_kind": "PRODUCTION_GL2",
                }
            )
    return rows, departures


def _inversion_audit_rows(
    *,
    source: Any,
    flow: FrozenAutonomousExactFlow,
    edges_m: NDArray[np.float64],
    frozen_xb: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    velocity = _production_velocity(source, frozen_xb)
    production = ProductionAutonomousTOFTable(edges_m=edges_m, velocity_m_s=velocity, exact_flow=flow)
    bracketed = ProductionAutonomousTOFTable(
        edges_m=edges_m,
        velocity_m_s=velocity,
        exact_flow=flow,
        config=ProductionTraceAuditConfig(inversion_mode="BRACKETED_TABLE"),
    )
    exact_table = ProductionAutonomousTOFTable(
        edges_m=edges_m,
        velocity_m_s=velocity,
        exact_flow=flow,
        config=ProductionTraceAuditConfig(table_kind="EXACT_TAU", inversion_mode="BRACKETED_TABLE"),
    )
    rows: list[dict[str, Any]] = []
    max_production_error = 0.0
    max_bracketed_error = 0.0
    for prod_run, bracket_run, exact_run in zip(production.runs, bracketed.runs, exact_table.runs):
        if not (
            prod_run.run_id == bracket_run.run_id == exact_run.run_id
            and np.array_equal(prod_run.coordinates_m, bracket_run.coordinates_m)
            and np.array_equal(prod_run.coordinates_m, exact_run.coordinates_m)
        ):
            raise ProductionTraceWorkflowError("inversion audit table runs do not remain aligned")
        sample_indices = np.unique(
            np.concatenate(
                (
                    np.asarray([0, 1, prod_run.coordinates_m.size - 2, prod_run.coordinates_m.size - 1], dtype=np.int64),
                    np.linspace(0, prod_run.coordinates_m.size - 1, num=min(65, prod_run.coordinates_m.size), dtype=np.int64),
                )
            )
        )
        for index in sample_indices:
            expected_radius = float(prod_run.coordinates_m[int(index)])
            target_production = float(prod_run.cumulative_time_s[int(index)])
            target_exact = float(exact_run.cumulative_time_s[int(index)])
            prod_same = production.invert(prod_run, target_production)
            bracket_same = bracketed.invert(bracket_run, target_production)
            # An independently exact target can legitimately fall just
            # outside a finite production table due to that table's own
            # quadrature bias.  This is itself an inversion/table diagnostic,
            # not a reason to silently clamp it or abort the complete audit.
            try:
                prod_exact_target = production.invert(prod_run, target_exact)
                exact_target_status = "SUCCESS"
            except ProductionTraceAuditError:
                prod_exact_target = None
                exact_target_status = "OUT_OF_PRODUCTION_TABLE_RANGE"
            exact_bracket = exact_table.invert(exact_run, target_exact)
            max_production_error = max(max_production_error, abs(prod_same.radius_m - expected_radius))
            if prod_exact_target is not None:
                max_production_error = max(max_production_error, abs(prod_exact_target.radius_m - expected_radius))
            max_bracketed_error = max(max_bracketed_error, abs(bracket_same.radius_m - expected_radius), abs(exact_bracket.radius_m - expected_radius))
            rows.append(
                {
                    "run_id": int(prod_run.run_id),
                    "branch_sign": int(np.sign(prod_run.sign)),
                    "sample_index": int(index),
                    "expected_radius_m": expected_radius,
                    "production_table_target_s": target_production,
                    "exact_tau_target_s": target_exact,
                    "production_fixed_same_table_radius_m": prod_same.radius_m,
                    "production_fixed_same_table_error_m": abs(prod_same.radius_m - expected_radius),
                    "production_fixed_exact_target_status": exact_target_status,
                    "production_fixed_exact_target_radius_m": None if prod_exact_target is None else prod_exact_target.radius_m,
                    "production_fixed_exact_target_error_m": None if prod_exact_target is None else abs(prod_exact_target.radius_m - expected_radius),
                    "bracketed_same_table_radius_m": bracket_same.radius_m,
                    "bracketed_same_table_error_m": abs(bracket_same.radius_m - expected_radius),
                    "exact_tau_bracketed_radius_m": exact_bracket.radius_m,
                    "exact_tau_bracketed_error_m": abs(exact_bracket.radius_m - expected_radius),
                    "production_fixed_iteration_count": prod_same.iteration_count,
                    "production_fixed_bracket_width_m": prod_same.bracket_width_m,
                    "production_fixed_residual_s": prod_same.residual_s,
                    "bracketed_iteration_count": bracket_same.iteration_count,
                    "bracketed_bracket_width_m": bracket_same.bracket_width_m,
                    "bracketed_residual_s": bracket_same.residual_s,
                }
            )
    return rows, {
        "production_fixed_inverse_max_radius_error_m": max_production_error,
        "bracketed_inverse_max_radius_error_m": max_bracketed_error,
        "production_inversion_has_fixed_iterations": True,
        "production_inversion_iteration_count": 6,
    }


def _component_ablation_rows(
    *,
    source: Any,
    flow: FrozenAutonomousExactFlow,
    edges_m: NDArray[np.float64],
    frozen_xb: float,
) -> list[dict[str, Any]]:
    velocity = _production_velocity(source, frozen_xb)
    modes: list[tuple[str, ProductionAutonomousTOFTable | None]] = [
        ("TRACE_MODE_0_FULL_PRODUCTION", None),
        (
            "TRACE_MODE_2_PRODUCTION_GL2_TABLE_PLUS_BRACKETED_TABLE_INVERSION",
            ProductionAutonomousTOFTable(
                edges_m=edges_m,
                velocity_m_s=velocity,
                exact_flow=flow,
                config=ProductionTraceAuditConfig(inversion_mode="BRACKETED_TABLE"),
            ),
        ),
        (
            "TRACE_MODE_3_EXACT_TAU_TABLE_NODES_PLUS_PRODUCTION_STYLE_LOCAL_INVERSION",
            ProductionAutonomousTOFTable(
                edges_m=edges_m,
                velocity_m_s=velocity,
                exact_flow=flow,
                config=ProductionTraceAuditConfig(table_kind="EXACT_TAU", inversion_mode="PRODUCTION_FIXED_6"),
            ),
        ),
        (
            "TRACE_MODE_4_EXACT_TAU_PLUS_BRACKETED_INVERSION",
            ProductionAutonomousTOFTable(
                edges_m=edges_m,
                velocity_m_s=velocity,
                exact_flow=flow,
                config=ProductionTraceAuditConfig(table_kind="EXACT_TAU", inversion_mode="BRACKETED_TABLE"),
            ),
        ),
    ]
    rows: list[dict[str, Any]] = []
    for h_s in H_LADDER_S:
        exact = flow.departure_faces(edges_m, float(h_s))
        public, public_status = public_production_trace_probe(
            arrival_faces_m=edges_m,
            duration_s=float(h_s),
            velocity_m_s=velocity,
            lower_radius_m=float(edges_m[0]),
            upper_radius_m=float(edges_m[-1]),
        )
        mode_arrays: list[tuple[str, NDArray[np.float64], NDArray[np.str_]]] = [
            ("TRACE_MODE_0_FULL_PRODUCTION", public.departure_faces_m, public_status),
            ("TRACE_MODE_1_PRODUCTION_G_PLUS_EXACT_FLOW", exact.radius_m, exact.status),
            ("TRACE_MODE_5_EXACT_REFERENCE", exact.radius_m, exact.status),
        ]
        for label, table in modes:
            if table is not None:
                query = table.query_departure(edges_m, float(h_s))
                mode_arrays.append((label, query.radius_m, query.status))
        for label, departures, statuses in mode_arrays:
            delta = np.asarray(departures - exact.radius_m, dtype=np.float64)
            scale = np.maximum(np.maximum(np.abs(departures), np.abs(exact.radius_m)), 1.0e-300)
            rows.append(
                {
                    "h_s": float(h_s),
                    "trace_mode": label,
                    "max_absolute_radius_error_m": float(np.max(np.abs(delta))),
                    "rms_absolute_radius_error_m": float(math.sqrt(float(np.mean(np.square(delta))))),
                    "max_relative_radius_error": float(np.max(np.abs(delta) / scale)),
                    "status_mismatch_count": int(np.count_nonzero(statuses != exact.status)),
                }
            )
    return rows


def _single_step_scaling_rows(face_rows: Sequence[Mapping[str, Any]]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    by_face: dict[int, list[Mapping[str, Any]]] = {}
    for row in face_rows:
        by_face.setdefault(int(row["face_index"]), []).append(row)
    rows: list[dict[str, Any]] = []
    classes: dict[str, int] = {}
    for face_index, values in sorted(by_face.items()):
        ordered = sorted(values, key=lambda row: float(row["h_s"]), reverse=True)
        h_values = [float(row["h_s"]) for row in ordered]
        errors = [float(row["absolute_radius_error_m"]) for row in ordered]
        order = observed_orders(h_values, errors, metric="single_face_absolute_radius_error")
        finite = [float(row["observed_order"]) for row in order["pair_rows"] if math.isfinite(float(row["observed_order"]))]
        late = float(np.median(finite[-2:])) if finite else math.nan
        if not finite:
            classification = "TRACE_SINGLE_STEP_FIXED_FLOOR"
        elif late >= 0.75:
            classification = "TRACE_SINGLE_STEP_CONVERGENT"
        elif max(errors[-2:]) <= 1.25 * min(errors[-2:]):
            classification = "TRACE_SINGLE_STEP_FIXED_FLOOR"
        else:
            classification = "TRACE_SINGLE_STEP_NONMONOTONE"
        classes[classification] = classes.get(classification, 0) + 1
        for pair in order["pair_rows"]:
            rows.append(
                {
                    "face_index": face_index,
                    "arrival_radius_m": float(ordered[0]["arrival_radius_m"]),
                    "classification": classification,
                    "late_median_observed_order": late,
                    **pair,
                }
            )
    return rows, {"face_class_counts": classes, "face_count": len(by_face)}


def _representative_radii(
    *, edges_m: NDArray[np.float64], velocity_m_s: NDArray[np.float64]
) -> tuple[list[tuple[str, float]], dict[str, Any]]:
    labels, conditioning, rule = _region_labels(edges_m=edges_m, velocity_m_s=velocity_m_s)
    choices: list[tuple[str, float]] = []
    for wanted in ("SHRINKING_BULK", "GROWING_BULK", "NEAR_CRITICAL_CONDITIONING_Q95", "NEAR_RMIN_GRID_LOCAL"):
        indices = np.flatnonzero(labels == wanted)
        if indices.size == 0:
            continue
        if wanted == "NEAR_CRITICAL_CONDITIONING_Q95":
            index = int(indices[np.argmax(conditioning[indices])])
        elif wanted == "NEAR_RMIN_GRID_LOCAL":
            index = int(indices[-1])
        else:
            index = int(indices[indices.size // 2])
        choices.append((wanted, float(edges_m[index])))
    if len(choices) < 3:
        raise ProductionTraceWorkflowError("could not select sufficient representative frozen trace radii")
    return choices, rule


def _repeated_trace_rows(
    *,
    source: Any,
    flow: FrozenAutonomousExactFlow,
    edges_m: NDArray[np.float64],
    frozen_xb: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    velocity = _production_velocity(source, frozen_xb)
    velocity_edges = velocity(edges_m)
    radii, rule = _representative_radii(edges_m=edges_m, velocity_m_s=velocity_edges)
    table = ProductionAutonomousTOFTable(edges_m=edges_m, velocity_m_s=velocity, exact_flow=flow)
    rows: list[dict[str, Any]] = []
    classifications: dict[str, str] = {}
    for label, radius in radii:
        per_radius: list[dict[str, Any]] = []
        exact = flow.evaluate(radius, -FINAL_HORIZON_S)
        for h_s in H_LADDER_S:
            count = int(round(FINAL_HORIZON_S / float(h_s)))
            if not math.isclose(count * float(h_s), FINAL_HORIZON_S, rel_tol=0.0, abs_tol=1.0e-18):
                raise ProductionTraceWorkflowError("registered trace h does not divide the frozen horizon")
            current = float(radius)
            first_error = math.nan
            first_status = "NOT_RUN"
            for step in range(count):
                query = table.query_departure(np.asarray([current], dtype=np.float64), float(h_s))
                if step == 0:
                    first_exact = flow.evaluate(current, -float(h_s))
                    first_error = abs(float(query.radius_m[0]) - float(first_exact.radius_m))
                    first_status = str(query.status[0])
                current = float(query.radius_m[0])
            accumulated = abs(current - float(exact.radius_m))
            row = {
                "representative_region": label,
                "initial_radius_m": radius,
                "h_s": float(h_s),
                "step_count": count,
                "single_step_error_m": first_error,
                "single_step_status": first_status,
                "production_repeated_departure_radius_m": current,
                "exact_backward_departure_radius_m": float(exact.radius_m),
                "exact_backward_status": str(exact.status),
                "final_accumulated_error_m": accumulated,
                "accumulated_over_single_step": accumulated / max(first_error, 1.0e-300),
            }
            per_radius.append(row)
            rows.append(row)
        errors = [float(row["final_accumulated_error_m"]) for row in per_radius]
        if errors[-1] > 0.8 * errors[0]:
            classifications[label] = "TRACE_REPEATED_NONCONVERGENT_OR_GENERATOR_BIAS"
        else:
            classifications[label] = "TRACE_REPEATED_CONVERGENT"
    return rows, {
        "direction": "backward departure composition; exact comparator is Flow_exact(R0, -H), matching the production departure map",
        "horizon_s": FINAL_HORIZON_S,
        "representative_selection": rule,
        "per_region_classification": classifications,
    }


def _cr1_rows_against_exact(
    *,
    source: Any,
    edges_m: NDArray[np.float64],
    u0_cells: NDArray[np.float64],
    exact_reference: Any,
    frozen_xb: float,
    exact_departures: Mapping[float, NDArray[np.float64]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    initial_arrays = {key: np.asarray(value).copy() for key, value in source.state_arrays().items()}
    initial_history = tuple(source.history)
    cr1_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    for h_s in H_LADDER_S:
        count = int(round(FINAL_HORIZON_S / float(h_s)))
        if not math.isclose(count * float(h_s), FINAL_HORIZON_S, rel_tol=0.0, abs_tol=1.0e-18):
            raise ProductionTraceWorkflowError("registered CR1 h does not divide the frozen horizon")
        result = phi_compose(
            source,
            dt_s=float(h_s),
            count=count,
            mode="FROZEN_MATRIX",
            frozen_matrix_xb=float(frozen_xb),
        )
        if result.status != "SUCCESS" or result.state is None or result.solver is None:
            raise ProductionTraceWorkflowError(f"frozen production CR1 did not close at h={h_s:.17g}: {result.error_message}")
        if not _state_arrays_equal(initial_arrays, source.state_arrays()) or tuple(source.history) != initial_history:
            raise ProductionTraceWorkflowError("frozen CR1 audit modified the accepted U0 source")
        production_cells = np.asarray(result.state["population_array"], dtype=np.float64)
        metrics = measure_error_metrics(production_cells, exact_reference.cell_number_m3, edges_m=edges_m)
        history = tuple(result.solver.history[-count:])
        cr1_rows.append(
            {
                "comparison": "CURRENT_PRODUCTION_CR1_VS_REF_PC",
                "h_s": float(h_s),
                "substep_count": count,
                **metrics,
                "cr1_state_hash": str(result.state["accepted_state_hash"]),
                "rmin_number_loss_m3": float(sum(float(item.rmin_number_loss_m3) for item in history)),
                "rmax_number_loss_m3": float(sum(float(item.rmax_number_loss_m3) for item in history)),
                "fixed_point_modes_json": [str(item.fixed_point_convergence_mode) for item in history],
            }
        )
        current = u0_cells.copy()
        departure = np.asarray(exact_departures[float(h_s)], dtype=np.float64)
        conservation = 0.0
        for _ in range(count):
            remap = conservative_remap_piecewise_constant(edges_m, current, departure)
            current = np.asarray(remap.cell_number_m3, dtype=np.float64)
            conservation += float(remap.conservation_residual_m3)
        projection_rows.append(
            {
                "comparison": "EXACT_FLOW_PLUS_SAME_CR1_REMAP_VS_REF_PC",
                "h_s": float(h_s),
                "substep_count": count,
                **measure_error_metrics(current, exact_reference.cell_number_m3, edges_m=edges_m),
                "remap_cumulative_conservation_residual_m3": conservation,
            }
        )
    return cr1_rows, projection_rows


def _trace_induced_rows(
    *,
    cr1_rows: Sequence[Mapping[str, Any]], projection_rows: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    if len(cr1_rows) != len(projection_rows):
        raise ProductionTraceWorkflowError("CR1 and projection ladders cannot be paired")
    rows: list[dict[str, Any]] = []
    for full, projection in zip(cr1_rows, projection_rows):
        if float(full["h_s"]) != float(projection["h_s"]):
            raise ProductionTraceWorkflowError("CR1 and projection h ladders are misaligned")
        rows.append(
            {
                "h_s": float(full["h_s"]),
                "full_vs_exact_relative_L1": float(full["population_relative_L1"]),
                "projection_only_vs_exact_relative_L1": float(projection["population_relative_L1"]),
                "trace_induced_relative_L1_proxy": abs(
                    float(full["population_relative_L1"]) - float(projection["population_relative_L1"])
                ),
                "full_vs_exact_L1_abs": float(full["population_L1_abs"]),
                "projection_only_vs_exact_L1_abs": float(projection["population_L1_abs"]),
            }
        )
    return rows


def _branch_rows(
    *,
    source: Any,
    flow: FrozenAutonomousExactFlow,
    edges_m: NDArray[np.float64],
    frozen_xb: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    velocity = _production_velocity(source, frozen_xb)
    rows: list[dict[str, Any]] = []
    mismatch_count = 0
    for h_s in H_LADDER_S:
        trace, statuses = public_production_trace_probe(
            arrival_faces_m=edges_m,
            duration_s=float(h_s),
            velocity_m_s=velocity,
            lower_radius_m=float(edges_m[0]),
            upper_radius_m=float(edges_m[-1]),
        )
        topology = characteristic_trace_topology(edges_m, trace=trace, velocity_m_s=velocity)
        exact = flow.departure_faces(edges_m, float(h_s))
        for index, radius in enumerate(edges_m):
            branch = flow.law.branch_name(float(radius))
            production_branch = "stationary_critical" if statuses[index] == "STATIONARY_CRITICAL" else (
                "growing" if float(velocity(np.asarray([radius], dtype=np.float64))[0]) > 0.0 else "shrinking"
            )
            mismatch = production_branch != branch or statuses[index] != exact.status[index]
            mismatch_count += int(mismatch)
            rows.append(
                {
                    "h_s": float(h_s),
                    "face_index": int(index),
                    "arrival_radius_m": float(radius),
                    "production_branch": production_branch,
                    "exact_branch": branch,
                    "production_backward_status": str(statuses[index]),
                    "exact_backward_status": str(exact.status[index]),
                    "branch_or_status_mismatch": bool(mismatch),
                    "topology_mode": topology.mode,
                    "topology_run_id": int(topology.arrival_face_run_id[index]),
                }
            )
    return rows, {"branch_or_status_mismatch_count": mismatch_count, "topology_checked": True}


def _root_cause(
    *,
    table_resolution_rows: Sequence[Mapping[str, Any]],
    component_rows: Sequence[Mapping[str, Any]],
    inversion: Mapping[str, Any],
    repeated: Mapping[str, Any],
) -> dict[str, Any]:
    def metric(rows: Sequence[Mapping[str, Any]], *, resolution: int | None = None, mode: str | None = None) -> float:
        matches = [
            row
            for row in rows
            if float(row["h_s"]) == H_LADDER_S[0]
            and (resolution is None or int(row["trace_auxiliary_subcells_per_population_cell"]) == resolution)
            and (mode is None or str(row["trace_mode"]) == mode)
        ]
        if len(matches) != 1:
            raise ProductionTraceWorkflowError("root-cause component metrics are not uniquely registered at h=1/128")
        return float(matches[0]["max_relative_radius_error"])

    legacy = metric(table_resolution_rows, resolution=16)
    resolution32 = metric(table_resolution_rows, resolution=32)
    resolution64 = metric(table_resolution_rows, resolution=64)
    resolution128 = metric(table_resolution_rows, resolution=128)
    bracketed = metric(component_rows, mode="TRACE_MODE_2_PRODUCTION_GL2_TABLE_PLUS_BRACKETED_TABLE_INVERSION")
    exact_tau = metric(component_rows, mode="TRACE_MODE_4_EXACT_TAU_PLUS_BRACKETED_INVERSION")
    resolution_gain = legacy / max(resolution64, 1.0e-300)
    inversion_gain = legacy / max(bracketed, 1.0e-300)
    accumulation_classes = dict(repeated.get("per_region_classification", {}))
    accumulation_supported = any("NONCONVERGENT" in value for value in accumulation_classes.values())
    if resolution64 <= TRACE_ENVELOPE_RELATIVE and resolution_gain >= 8.0 and inversion_gain < 4.0:
        primary = "TRACE_TOF_TABLE_RESOLUTION_FLOOR"
        status = "DIAG_TRACE_TOF_TABLE_RESOLUTION_FLOOR"
        fix_resolution = 64
        secondary = "TRACE_FIXED_PER_UNIT_TIME_GENERATOR_BIAS" if accumulation_supported else "NONE"
    elif inversion_gain >= 8.0 and bracketed <= TRACE_ENVELOPE_RELATIVE:
        primary = "TRACE_INVERSION_ERROR"
        status = "DIAG_TRACE_INTERPOLATION_OR_INVERSION_ERROR"
        fix_resolution = None
        secondary = "FIXED_SIX_ITERATION_NO_TERMINATION_CRITERION"
    elif resolution64 <= TRACE_ENVELOPE_RELATIVE and inversion_gain >= 4.0:
        primary = "MIXED_TRACE_NUMERICAL_ERROR"
        status = "DIAG_MIXED_TRACE_NUMERICAL_ERROR"
        fix_resolution = 64
        secondary = "TABLE_AND_INVERSION"
    elif legacy > TRACE_ENVELOPE_RELATIVE and exact_tau <= TRACE_ENVELOPE_RELATIVE:
        primary = "TRACE_TAU_INTERPOLATION_ERROR"
        status = "DIAG_TRACE_INTERPOLATION_OR_INVERSION_ERROR"
        fix_resolution = None
        secondary = "TABLE_REPRESENTATION"
    else:
        primary = "TRACE_ROOT_CAUSE_NOT_ISOLATED"
        status = "DIAG_TRACE_ROOT_CAUSE_NOT_ISOLATED"
        fix_resolution = None
        secondary = "NO_SINGLE_COMPONENT_MET_THE_DIRECT_CAUSAL_GATE"
    return {
        "STATUS": status,
        "PRIMARY_ROOT_CAUSE": primary,
        "SECONDARY_ROOT_CAUSE": secondary,
        "legacy_h128_max_relative_error": legacy,
        "resolution32_h128_max_relative_error": resolution32,
        "resolution64_h128_max_relative_error": resolution64,
        "resolution128_h128_max_relative_error": resolution128,
        "bracketed_same_table_h128_max_relative_error": bracketed,
        "exact_tau_bracketed_h128_max_relative_error": exact_tau,
        "table_resolution_gain_16_to_64": resolution_gain,
        "inversion_gain_fixed_to_bracketed": inversion_gain,
        "production_fixed_inverse_max_radius_error_m": inversion["production_fixed_inverse_max_radius_error_m"],
        "bracketed_inverse_max_radius_error_m": inversion["bracketed_inverse_max_radius_error_m"],
        "repeated_accumulation_supported": accumulation_supported,
        "directly_supported_minimal_fix_auxiliary_subcells": fix_resolution,
        "trace_reference_envelope_relative": TRACE_ENVELOPE_RELATIVE,
    }


def _architecture_report() -> dict[str, Any]:
    return {
        "arrival_to_departure_path": [
            "CharacteristicReferenceSolver._trace_faces",
            "trace_departure_faces_rk2",
            "same-sign branch classification on log auxiliary submesh",
            "two-node Gauss-Legendre |dR/G| table per same-sign run",
            "linear table bracket plus six fixed Gauss/Newton/bisection iterations",
            "Rmin/Rmax no-inflow or stationary-tail handling",
            "departure faces supplied unchanged to CR1 CDF remap",
        ],
        "production_source": {
            "growth_kernel": "src/kwn_mvp/characteristic_reference.py: CharacteristicReferenceSolver._velocity_at_radii",
            "trace_entry": "src/kwn_mvp/conservative_remap.py: trace_departure_faces_rk2",
            "auxiliary_table": "_log_subdivided_faces, _gauss_time_of_flight, np.cumsum",
            "inversion": "_invert_time_of_flight: fixed six iterations with no residual termination",
            "physical_branch_tail": "_stationary_root and _stationary_tail_from_endpoint",
            "remap": "conservative_remap_piecewise_constant",
        },
        "precision": "binary64 throughout production trace; no persistent cross-step cache or quantization",
        "production_default_auxiliary_subcells_per_population_cell": int(_TRACE_SUBCELLS_PER_CELL),
        "interpolation": "linear lookup only to seed an inverse; local GL2 re-integration is used in each fixed inverse iteration",
        "reference_difference": "independent frozen exact flow uses SciPy adaptive quadrature and residual-qualified safeguarded inverse; it does not import production trace or CR1 remap",
    }


def _run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    if output_root.exists() or report_root.exists():
        raise ProductionTraceWorkflowError("refusing to overwrite trace-audit output or report roots")
    source_identity = _source_identity()
    baseline_reproduction = _baseline_reproduction(Path(args.baseline_root))
    try:
        mpmath = exact_prior._mpmath_preflight(Path(args.mpmath_vendor_root))
    except Exception as error:
        raise ProductionTraceWorkflowError("required isolated mpmath shadow is unavailable") from error
    output_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    (output_root / "figures").mkdir()
    started = time.monotonic()
    source, frozen_baseline, u0_arrays, u0_metadata = _load_frozen_state(Path(args.restart_checkpoint), output_root)
    before_arrays = {key: np.asarray(value).copy() for key, value in source.state_arrays().items()}
    before_history = tuple(source.history)
    edges = np.asarray(u0_arrays["radius_edges_m"], dtype=np.float64)
    u0_cells = np.asarray(u0_arrays["cell_number_m3"], dtype=np.float64)
    frozen_xb = float(u0_arrays["matrix_xb"][0])
    if not np.array_equal(u0_cells, np.asarray(source._beta_cell_numbers(), dtype=np.float64)):
        raise ProductionTraceWorkflowError("reconstructed beta measure differs from frozen U0")
    beta = source.population("beta")
    law = FrozenAutonomousGrowthLaw(
        parameters=beta.parameters,
        equilibrium_adapter=source.equilibrium_adapter,
        matrix_xb=frozen_xb,
        lower_radius_m=float(edges[0]),
        upper_radius_m=float(edges[-1]),
    )
    growth_rows, growth_summary = _growth_parity_rows(
        source=source, law=law, edges_m=edges, frozen_xb=frozen_xb
    )
    if not (growth_summary["G_VALUE_PARITY"] and growth_summary["G_SIGN_PARITY"] and growth_summary["CRITICAL_RADIUS_PARITY"]):
        raise ProductionTraceWorkflowError("DIAG_PRODUCTION_TRACE_GROWTH_KERNEL_MISMATCH")
    flow = FrozenAutonomousExactFlow(law, edges)
    measure = PiecewiseConstantCumulativeMeasure(edges, u0_cells)
    reference_validation, exact_reference, shadow_rows = exact_prior._reference_validation(
        flow=flow, measure=measure, u0_cells=u0_cells, edges_m=edges
    )
    semigroup = flow.semigroup_check(edges, FINAL_HORIZON_S)
    roundtrip = flow.roundtrip_check(edges, FINAL_HORIZON_S)
    if int(semigroup["status_mismatch_count"]) != 0 or int(roundtrip["status_mismatch_count"]) != 0:
        raise ProductionTraceWorkflowError("independent exact-flow reference validation did not close")
    # The read-only default-table parity check is a hard contract before any
    # table-resolution or inversion component result can be interpreted.
    default_table = ProductionAutonomousTOFTable(
        edges_m=edges, velocity_m_s=_production_velocity(source, frozen_xb), exact_flow=flow
    )
    for h_s in H_LADDER_S:
        default_table.require_default_public_parity(float(h_s))
    face_rows, face_summaries, _production_departures, exact_departures = _trace_face_rows(
        source=source, flow=flow, edges_m=edges, frozen_xb=frozen_xb
    )
    velocity_edges = _production_velocity(source, frozen_xb)(edges)
    region_rows, conditioning_rule = _region_rows(face_rows=face_rows, edges_m=edges, velocity=velocity_edges)
    tau_rows = default_table.tau_rows(include_midpoints=True)
    resolution_rows, _resolution_departures = _table_resolution_rows(
        source=source, flow=flow, edges_m=edges, frozen_xb=frozen_xb
    )
    inversion_rows, inversion_summary = _inversion_audit_rows(
        source=source, flow=flow, edges_m=edges, frozen_xb=frozen_xb
    )
    branch_rows, branch_summary = _branch_rows(
        source=source, flow=flow, edges_m=edges, frozen_xb=frozen_xb
    )
    scaling_rows, scaling_summary = _single_step_scaling_rows(face_rows)
    repeated_rows, repeated_summary = _repeated_trace_rows(
        source=source, flow=flow, edges_m=edges, frozen_xb=frozen_xb
    )
    component_rows = _component_ablation_rows(
        source=source, flow=flow, edges_m=edges, frozen_xb=frozen_xb
    )
    root_cause = _root_cause(
        table_resolution_rows=resolution_rows,
        component_rows=component_rows,
        inversion=inversion_summary,
        repeated=repeated_summary,
    )
    if not _state_arrays_equal(before_arrays, source.state_arrays()) or tuple(source.history) != before_history:
        raise ProductionTraceWorkflowError("read-only trace audit modified accepted frozen U0")

    current_default_summary = face_summaries[H_LADDER_S[0]]
    qualified_trace = False
    cr1_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    trace_induced: list[dict[str, Any]] = []
    post_fix_repeated: dict[str, Any] = {"status": "NOT_RUN_IN_DIAGNOSIS_PHASE"}
    final_status = root_cause["STATUS"]
    fix_description = "NO_FIX_IMPLEMENTED_IN_DIAGNOSIS_PHASE"
    if args.phase == "qualify":
        all_face_pass = all(
            int(summary["status_mismatch_count"]) == 0
            and float(summary["max_relative_radius_error"]) <= TRACE_ENVELOPE_RELATIVE
            for summary in face_summaries.values()
        )
        repeat_pass = all("NONCONVERGENT" not in value for value in repeated_summary["per_region_classification"].values())
        cr1_rows, projection_rows = _cr1_rows_against_exact(
            source=source,
            edges_m=edges,
            u0_cells=u0_cells,
            exact_reference=exact_reference,
            frozen_xb=frozen_xb,
            exact_departures=exact_departures,
        )
        trace_induced = _trace_induced_rows(cr1_rows=cr1_rows, projection_rows=projection_rows)
        projection_scale = max(float(row["population_relative_L1"]) for row in projection_rows)
        full_scale = max(float(row["population_relative_L1"]) for row in cr1_rows)
        same_order = full_scale <= 10.0 * max(projection_scale, 1.0e-300)
        qualified_trace = all_face_pass and repeat_pass
        post_fix_repeated = {"status": "PASS_TRACE_REPEATED_COMPOSITION" if repeat_pass else "FAIL_TRACE_REPEATED_COMPOSITION", **repeated_summary}
        fix_description = (
            f"Current production auxiliary TOF table uses {_TRACE_SUBCELLS_PER_CELL} subcells per population cell; "
            "the frozen qualification reuses the unchanged CR1 remap, grid, thermodynamics, and growth kernel."
        )
        if qualified_trace and same_order:
            final_status = "PASS_PRODUCTION_TRACE_KERNEL_FROZEN_QUALIFICATION"
        elif qualified_trace:
            final_status = "DIAG_TRACE_FIX_INSUFFICIENT_OTHER_OPERATOR_ERROR"
        else:
            final_status = root_cause["STATUS"]

    _write_json(output_root / "baseline_reproduction.json", baseline_reproduction)
    _write_json(output_root / "production_trace_architecture.json", _architecture_report())
    _write_csv(output_root / "growth_kernel_parity.csv", growth_rows, ("radius_m", "bitwise_equal"))
    _write_csv(output_root / "trace_face_error_all.csv", face_rows, ("h_s", "face_index", "relative_radius_error"))
    _write_csv(output_root / "trace_region_error.csv", region_rows, ("h_s", "region", "max_relative_radius_error"))
    _write_csv(output_root / "tau_production_vs_exact.csv", tau_rows, ("radius_m", "relative_error"))
    _write_csv(output_root / "tau_table_resolution.csv", resolution_rows, ("trace_auxiliary_subcells_per_population_cell", "h_s"))
    _write_csv(output_root / "trace_interpolation_audit.csv", component_rows, ("trace_mode", "h_s"))
    _write_csv(output_root / "trace_inversion_audit.csv", inversion_rows, ("run_id", "sample_index"))
    _write_csv(output_root / "trace_branch_classification.csv", branch_rows, ("h_s", "face_index"))
    _write_csv(output_root / "trace_h_scaling.csv", scaling_rows, ("face_index", "h_coarse_s"))
    _write_csv(output_root / "repeated_trace_accumulation.csv", repeated_rows, ("representative_region", "h_s"))
    _write_csv(output_root / "trace_component_ablation.csv", component_rows, ("trace_mode", "h_s"))
    _write_json(output_root / "trace_root_cause.json", root_cause)
    _write_csv(output_root / "qualified_trace_face_error.csv", [face_summaries[h] for h in H_LADDER_S], ("h_s", "max_relative_radius_error"))
    _write_csv(output_root / "corrected_cr1_vs_exact.csv", cr1_rows, ("h_s", "population_relative_L1"))
    _write_csv(output_root / "trace_projection_comparator.csv", trace_induced, ("h_s", "trace_induced_relative_L1_proxy"))
    analysis_provenance = {
        "task_name": TASK_NAME,
        "phase": args.phase,
        "top_status": final_status,
        "source": source_identity,
        "baseline_reproduction": baseline_reproduction,
        "frozen_u0": frozen_baseline,
        "u0_metadata": u0_metadata,
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "mpmath": mpmath,
        "reference_validation": reference_validation,
        "semigroup": semigroup,
        "roundtrip": roundtrip,
        "trace_reference_floor_relative": REFERENCE_FLOOR_RELATIVE,
        "trace_qualification_envelope_relative": TRACE_ENVELOPE_RELATIVE,
        "dynamic_time_reference": "NOT_ASSIGNED",
        "cr2_authorized": False,
        "pf_source_modified": False,
        "cuda_rerun": False,
        "physical_retuning": False,
        "gp_release": False,
        "runtime_s": time.monotonic() - started,
    }
    _write_json(output_root / "analysis_provenance.json", analysis_provenance)
    reports = {
        "00_baseline_reproduction.md": baseline_reproduction,
        "01_production_trace_architecture.md": _architecture_report(),
        "02_growth_kernel_parity.md": growth_summary,
        "03_all_face_trace_vs_exact.md": {"summaries": [face_summaries[h] for h in H_LADDER_S], "reference_envelope_relative": TRACE_ENVELOPE_RELATIVE},
        "04_region_conditioning.md": {"conditioning_rule": conditioning_rule, "rows": region_rows},
        "05_tau_table_audit.md": {"table_representation": "ephemeral per-trace anchor-normalized same-branch GL2 cumulative flight time", "row_count": len(tau_rows), "csv": f"outputs/{TASK_NAME}/tau_production_vs_exact.csv"},
        "06_interpolation_inversion_audit.md": {"inversion_summary": inversion_summary, "component_csv": f"outputs/{TASK_NAME}/trace_component_ablation.csv"},
        "07_branch_critical_radius_audit.md": branch_summary,
        "08_single_step_h_scaling.md": scaling_summary,
        "09_repeated_trace_accumulation.md": repeated_summary,
        "10_component_ablation.md": {"rows": component_rows},
        "11_trace_root_cause.md": root_cause,
        "12_minimal_trace_fix.md": {"phase": args.phase, "trace_fix_implemented": args.phase == "qualify", "description": fix_description, "cr1_remap_modified": False},
        "13_trace_kernel_qualification.md": {"qualified": qualified_trace, "reference_floor_relative": REFERENCE_FLOOR_RELATIVE, "qualification_envelope_relative": TRACE_ENVELOPE_RELATIVE, "face_summaries": [face_summaries[h] for h in H_LADDER_S], "repeated": post_fix_repeated},
        "14_corrected_cr1_vs_exact.md": {"current_cr1_rows": cr1_rows, "projection_rows": projection_rows, "trace_induced_rows": trace_induced},
        "15_model_boundary.md": {
            "FROZEN_TRACE_KERNEL_V1": "QUALIFIED" if qualified_trace else "NOT_QUALIFIED",
            "TIME_REFERENCE_V2": "NOT_ASSIGNED",
            "CR2_AUTHORIZED": False,
            "DYNAMIC_TRACE_SUPPORTED": False,
            "statement": "This frozen autonomous trace audit neither establishes nor approximates a dynamic nonlinear matrix-population time reference.",
        },
        "17_reproduction_commands.md": {
            "phase": args.phase,
            "command": f"PYTHONPATH=<vendor>:src {sys.executable} scripts/run_kwn_production_trace_audit_v1.py audit --phase {args.phase} --baseline-root <baseline> --restart-checkpoint <step244> --mpmath-vendor-root <vendor> --output-root <output> --report-root <reports> --test-status <tests>",
        },
    }
    final = {
        "STATUS": final_status,
        "BRANCH": source_identity["git_branch"],
        "COMMIT": source_identity["git_commit"],
        "SOURCE_CLEAN": True,
        "TESTS": str(args.test_status),
        "FROZEN_U0_HASH": EXPECTED_FROZEN_U0_HASH,
        "BASELINE_EXACT_REFERENCE": baseline_reproduction["status"],
        "BASELINE_CR1_REPRODUCED": True,
        "PRODUCTION_TRACE_ARCHITECTURE": "fixed log auxiliary TOF table + GL2 + six-iteration production inversion",
        "G_PARITY": growth_summary["G_VALUE_PARITY"],
        "CRITICAL_RADIUS_PARITY": growth_summary["CRITICAL_RADIUS_PARITY"],
        "BRANCH_STATUS_PARITY": branch_summary["branch_or_status_mismatch_count"] == 0,
        **{f"TRACE_H{int(round(1.0 / h))}_MAX_REL_ERROR": face_summaries[h]["max_relative_radius_error"] for h in H_LADDER_S},
        "TRACE_RMS_ERRORS": {str(int(round(1.0 / h))): face_summaries[h]["rms_absolute_radius_error_m"] for h in H_LADDER_S},
        "SOURCE_CELL_MISMATCHES": {str(int(round(1.0 / h))): face_summaries[h]["source_cell_mismatch_count"] for h in H_LADDER_S},
        "STATUS_MISMATCHES": {str(int(round(1.0 / h))): face_summaries[h]["status_mismatch_count"] for h in H_LADDER_S},
        "ERROR_BY_REGION": region_rows,
        "CRITICAL_REGION_ERROR": [row for row in region_rows if row["region"] == "NEAR_CRITICAL_CONDITIONING_Q95"],
        "PRODUCTION_TAU_ERROR": max(float(row["relative_error"]) for row in tau_rows),
        "TABLE_RESOLUTION_EFFECT": root_cause["table_resolution_gain_16_to_64"],
        "INTERPOLATION_EFFECT": root_cause["exact_tau_bracketed_h128_max_relative_error"],
        "INVERSION_EFFECT": root_cause["inversion_gain_fixed_to_bracketed"],
        "SINGLE_STEP_TRACE_ORDER": scaling_summary,
        "TRACE_ERROR_FLOOR": "NOT_A_FIXED_RADIUS_FLOOR" if root_cause["repeated_accumulation_supported"] else "NOT_SUPPORTED",
        "REPEATED_TRACE_ACCUMULATION": repeated_summary,
        "ACCUMULATION_SCALING": repeated_summary["per_region_classification"],
        "COMPONENT_ABLATION_RESULT": root_cause,
        "PRIMARY_ROOT_CAUSE": root_cause["PRIMARY_ROOT_CAUSE"],
        "SECONDARY_ROOT_CAUSE": root_cause["SECONDARY_ROOT_CAUSE"],
        "TRACE_FIX_IMPLEMENTED": args.phase == "qualify",
        "TRACE_FIX_DESCRIPTION": fix_description,
        "CR1_REMAP_MODIFIED": False,
        "QUALIFIED_TRACE_KERNEL": qualified_trace,
        "TRACE_REFERENCE_ENVELOPE": TRACE_ENVELOPE_RELATIVE,
        "POST_FIX_MAX_TRACE_ERROR": max(float(summary["max_relative_radius_error"]) for summary in face_summaries.values()) if args.phase == "qualify" else None,
        "POST_FIX_REPEATED_COMPOSITION": post_fix_repeated,
        **{f"POST_FIX_CR1_H{int(round(1.0 / float(row['h_s'])))}_ERROR": float(row["population_relative_L1"]) for row in cr1_rows},
        "PROJECTION_ONLY_ERRORS": [{"h_s": row["h_s"], "population_relative_L1": row["population_relative_L1"]} for row in projection_rows],
        "TRACE_INDUCED_ERRORS": trace_induced,
        "FROZEN_TRACE_KERNEL_V1": "QUALIFIED" if qualified_trace else "NOT_QUALIFIED",
        "TIME_REFERENCE_V2": "NOT_ASSIGNED",
        "CR2_AUTHORIZED": False,
        "DYNAMIC_TRACE_SUPPORTED": False,
        "PF_SOURCE_MODIFIED": False,
        "CUDA_RERUN": False,
        "PHYSICAL_RETUNING": False,
        "GP_RELEASE_RUN": False,
        "TOP_5_FINDINGS": [
            f"Production default auxiliary table uses {_TRACE_SUBCELLS_PER_CELL} subcells per population cell.",
            f"Legacy h=1/128 trace error is {root_cause['legacy_h128_max_relative_error']:.6e} relative.",
            f"The reference-derived trace envelope is {TRACE_ENVELOPE_RELATIVE:.6e} relative.",
            f"Primary root-cause classification: {root_cause['PRIMARY_ROOT_CAUSE']}.",
            "CR2 remains unauthorized and dynamic TIME_REFERENCE_V2 remains unassigned.",
        ],
        "P0_BLOCKERS": [] if final_status == "PASS_PRODUCTION_TRACE_KERNEL_FROZEN_QUALIFICATION" else [final_status],
        "NEXT_ACTION": "Stop at frozen trace qualification; do not enter dynamic nonlinear time-reference work without a separate authorization.",
        "KEY_REPORTS": [
            f"reports/{TASK_NAME}/03_all_face_trace_vs_exact.md",
            f"reports/{TASK_NAME}/10_component_ablation.md",
            f"reports/{TASK_NAME}/11_trace_root_cause.md",
            f"reports/{TASK_NAME}/16_final_acceptance_report.md",
        ],
    }
    reports["16_final_acceptance_report.md"] = final
    _write_reports(report_root, reports)
    print(json.dumps(_json_safe(final), indent=2, sort_keys=True))
    return 0, final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="run frozen production-trace audit")
    audit.add_argument("--phase", choices=("diagnose", "qualify"), required=True)
    audit.add_argument("--baseline-root", required=True)
    audit.add_argument("--restart-checkpoint", required=True)
    audit.add_argument("--mpmath-vendor-root", required=True)
    audit.add_argument("--output-root", required=True)
    audit.add_argument("--report-root", required=True)
    audit.add_argument("--test-status", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        code, _final = _run(args)
    except (ProductionTraceWorkflowError, ProductionTraceAuditError, FrozenExactFlowReferenceError, OSError, ValueError) as error:
        print(json.dumps({"STATUS": "FAIL_TRACE_AUDIT_BASELINE_REPRODUCTION", "error_type": type(error).__name__, "error": str(error)}, indent=2), file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
