#!/usr/bin/env python3
"""Frozen-autonomous exact-flow reference and CR1 consistency audit.

This is a read-only diagnostic harness.  It loads the exact accepted step-244
state, freezes xB, and compares disposable CR1 trajectories with an
independent time-of-flight pushforward.  It is not a production CR1 change, a
CR2 prototype, a nonlinear closure experiment, or a dynamic time reference.
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
    conservative_remap_piecewise_constant,
    trace_departure_faces_rk2,
)
from kwn_mvp.frozen_exact_flow_reference import (  # noqa: E402
    ExactPushforwardState,
    FrozenAutonomousExactFlow,
    FrozenAutonomousGrowthLaw,
    FrozenExactFlowReferenceError,
    PiecewiseConstantCumulativeMeasure,
    PiecewiseLinearCumulativeMeasure,
    classify_refinement_against_reference,
    measure_error_metrics,
    observed_orders,
)
from kwn_mvp.frozen_semigroup_decomposition import decompose_frozen_cr1  # noqa: E402
from kwn_mvp.population_metrics import metrics_from_piecewise_constant_cells  # noqa: E402
from scripts import run_kwn_frozen_semigroup_decomposition_v1 as prior  # noqa: E402


TASK_NAME = "kwn_frozen_exact_flow_reference_v1"
REQUIRED_BRANCH = "codex/kwn-frozen-exact-flow-reference-v1"
FROZEN_PARENT_COMMIT = "ad7f9b66a2e750bdf80a0c101d7ad7c167297c5d"
EXPECTED_FROZEN_U0_HASH = "7de7d098e1ee4a44e291d4771854fd8dfb8a05cdb5e404ae176d22d7bc88c98a"
EXPECTED_RESTART_SHA256 = "c0bc8dd946769550d780763d5a73446b929bac15c5bf7a04895a6288a2314770"
EXPECTED_FORMAL_TRACE_SHA256 = "e63e546ccd31d93d105b2bf40e3b5abb0fee9e9015f06eace45610a5654dad65"
EXPECTED_FORMAL_PHASE_A_SHA256 = "a91d1940fe471b7f2ebfde5402e177b06fa7b55056bcacdce93519c668755703"
EXPECTED_FROZEN_SEMIGROUP_SHA256 = "2eed7069411a21b5d23de6d7749f88af3c93729790ebd884e8df956c2393dc1c"
EXPECTED_FORMAL_FROZEN_U0_SHA256 = "fd777c36018027c4b499c2a1878f6b5b8d68d239866c98e0e1c762c9b3a6dd8b"
EXPECTED_THREE_PATH_STATES_SHA256 = "3f51d0e1b6ef9e1406a4355dce44bb16cbde40f3aaa5d8a30d8aa06fe83dd552"
EXPECTED_H_REFINEMENT_DECOMPOSITION_SHA256 = "06a1b65de7e559fa44bb03211ba5f0cf27c948895220fd01da4812fb4da9eee5"
EXPECTED_MPMATH_VERSION = "1.3.0"
EXPECTED_MPMATH_SOURCE_MANIFEST_SHA256 = "f08fb116da8ea63f88f14b873bfa0f79ae100ce631945d06f7432b60f79f4cf3"
FINAL_HORIZON_S = 1.0 / 128.0
H_LADDER_S = tuple(1.0 / float(2**power) for power in range(7, 13))
OLD_SEMIGROUP_H_S = tuple(1.0 / float(2**power) for power in range(7, 11))
# Tau is a branch-local time coordinate, whose absolute magnitude varies by
# many orders across the physical radius domain.  Its independent quadrature
# shadow is therefore qualified by relative integral precision; radius-map
# accuracy is separately fail-closed by the exact-flow semigroup and inverse
# roundtrip checks below.
TAU_SHADOW_RELATIVE_LIMIT = 1.0e-12
ALLOWED_TOP_LEVEL_STATUSES = {
    "DIAG_CR1_ASYMPTOTICALLY_CONSISTENT",
    "DIAG_TRACE_INTEGRATOR_ERROR_SUPPORTED",
    "DIAG_CR1_PROJECTION_ERROR_SUPPORTED",
    "DIAG_INITIAL_MEASURE_REPRESENTATION_LIMIT",
    "DIAG_FROZEN_EXACT_REFERENCE_FAILED",
    "DIAG_INSUFFICIENT_ASYMPTOTIC_RANGE",
}
REPORT_TITLES = {
    "00_baseline.md": "Frozen baseline and input provenance",
    "01_autonomous_growth_law.md": "Frozen autonomous growth law",
    "02_time_of_flight_reference.md": "Independent time-of-flight coordinate",
    "03_exact_flow_validation.md": "Exact-flow semigroup and roundtrip validation",
    "04_exact_pushforward_reference.md": "Direct REF-PC exact-flow pushforward",
    "05_cr1_vs_exact_refinement.md": "Frozen CR1 versus exact pushforward refinement",
    "06_observed_order.md": "Observed-order analysis",
    "07_trace_vs_exact.md": "Production trace versus exact flow",
    "08_projection_only_error.md": "Exact-flow plus CR1-remap isolation",
    "09_old_semigroup_reinterpretation.md": "Prior A/B/C semigroup results against exact flow",
    "10_representation_sensitivity.md": "REF-PC versus REF-PL sensitivity",
    "11_method_decision.md": "Fail-closed method decision",
    "12_final_acceptance_report.md": "Final frozen exact-flow acceptance report",
    "13_reproduction_commands.md": "Reproduction commands",
}


class FrozenExactFlowWorkflowError(RuntimeError):
    """A fail-closed error in this diagnostic-only orchestration layer."""


_PATH_DEPENDENT_CHECKPOINT_BINDING_FIELDS = {
    # A clean bundle clone necessarily changes these two location-derived
    # values.  All byte/semantic physics and checkpoint fields remain bound
    # below; accepting any other metadata difference would be a provenance
    # failure rather than a harmless path-only rebind.
    "runtime_contract_path",
    "runtime_source_config_hash_before_rebind",
}


def _json_safe(value: Any) -> Any:
    if isinstance(value, np.generic):
        return _json_safe(value.item())
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, float):
        return value if math.isfinite(value) else str(value)
    if isinstance(value, np.ndarray):
        array = np.ascontiguousarray(value)
        return {
            "array_shape": list(array.shape),
            "array_dtype": str(array.dtype),
            "array_sha256": hashlib.sha256(array.tobytes()).hexdigest(),
        }
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


def _write_markdown(path: Path, title: str, payload: Mapping[str, Any] | str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = payload.rstrip() if isinstance(payload, str) else "```json\n" + json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True) + "\n```"
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")


def _write_reports(report_root: Path, payloads: Mapping[str, Mapping[str, Any] | str]) -> None:
    unknown = set(payloads).difference(REPORT_TITLES)
    if unknown:
        raise FrozenExactFlowWorkflowError(f"unknown report names: {sorted(unknown)}")
    for name, title in REPORT_TITLES.items():
        _write_markdown(report_root / name, title, payloads.get(name, {"status": "NOT_REACHED"}))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _git(arguments: Sequence[str]) -> str:
    completed = subprocess.run(["git", *arguments], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return completed.stdout.strip()


def _source_identity() -> dict[str, Any]:
    try:
        branch = _git(("branch", "--show-current"))
        commit = _git(("rev-parse", "HEAD"))
        status = _git(("status", "--short"))
        parent_is_ancestor = subprocess.run(
            ["git", "merge-base", "--is-ancestor", FROZEN_PARENT_COMMIT, "HEAD"],
            cwd=ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        ).returncode == 0
    except subprocess.CalledProcessError as error:
        raise FrozenExactFlowWorkflowError("could not establish source identity") from error
    if branch != REQUIRED_BRANCH:
        raise FrozenExactFlowWorkflowError(f"source branch differs from required {REQUIRED_BRANCH}")
    if not parent_is_ancestor:
        raise FrozenExactFlowWorkflowError("source does not descend from the frozen parent")
    if status:
        raise FrozenExactFlowWorkflowError("source tree is dirty")
    return {
        "git_branch": branch,
        "git_commit": commit,
        "git_status": status,
        "frozen_parent_commit": FROZEN_PARENT_COMMIT,
        "frozen_parent_is_ancestor": parent_is_ancestor,
    }


def _state_arrays_equal(first: Mapping[str, NDArray[np.generic]], second: Mapping[str, NDArray[np.generic]]) -> bool:
    return set(first) == set(second) and all(np.array_equal(np.asarray(first[key]), np.asarray(second[key])) for key in first)


def _require_file(path: Path, *, expected_sha256: str, purpose: str) -> dict[str, str]:
    if not path.is_file():
        raise FrozenExactFlowWorkflowError(f"{purpose} is unavailable: {path}")
    observed = _sha256_file(path)
    if observed != expected_sha256:
        raise FrozenExactFlowWorkflowError(f"{purpose} SHA-256 differs: {observed}")
    return {"path": str(path), "sha256": observed, "expected_sha256": expected_sha256, "purpose": purpose}


def _inherited_inputs(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    return {
        "restart_checkpoint": _require_file(Path(args.restart_checkpoint), expected_sha256=EXPECTED_RESTART_SHA256, purpose="exact step-244 restart"),
        "formal_trace": _require_file(Path(args.formal_trace_csv), expected_sha256=EXPECTED_FORMAL_TRACE_SHA256, purpose="formal characteristic trace provenance"),
        "formal_phase_a": _require_file(Path(args.formal_phase_a_csv), expected_sha256=EXPECTED_FORMAL_PHASE_A_SHA256, purpose="formal Phase-A ladder provenance"),
        "formal_frozen_semigroup": _require_file(Path(args.frozen_semigroup_csv), expected_sha256=EXPECTED_FROZEN_SEMIGROUP_SHA256, purpose="prior frozen semigroup provenance"),
        "formal_frozen_u0": _require_file(Path(args.formal_frozen_u0), expected_sha256=EXPECTED_FORMAL_FROZEN_U0_SHA256, purpose="formal frozen U0 artifact"),
        "formal_three_path_states": _require_file(Path(args.formal_three_path_states), expected_sha256=EXPECTED_THREE_PATH_STATES_SHA256, purpose="formal A/B/C state artifact"),
        "formal_h_refinement": _require_file(Path(args.formal_h_refinement_csv), expected_sha256=EXPECTED_H_REFINEMENT_DECOMPOSITION_SHA256, purpose="formal semigroup h-refinement artifact"),
    }


def _mpmath_preflight(vendor_root: Path) -> dict[str, Any]:
    try:
        import mpmath  # type: ignore[import-not-found]
    except ImportError as error:
        raise FrozenExactFlowWorkflowError("mpmath high-precision shadow is unavailable") from error
    path = Path(str(mpmath.__file__)).resolve()
    root = vendor_root.resolve()
    if root not in path.parents:
        raise FrozenExactFlowWorkflowError("mpmath was not imported from the isolated task-local vendor root")
    if str(mpmath.__version__) != EXPECTED_MPMATH_VERSION:
        raise FrozenExactFlowWorkflowError("mpmath version differs from the pinned high-precision shadow")
    return {
        "mpmath_version": str(mpmath.__version__),
        "mpmath_file": str(path),
        "vendor_root": str(root),
        "expected_source_manifest_sha256": EXPECTED_MPMATH_SOURCE_MANIFEST_SHA256,
        "import_isolated_from_runtime": True,
    }


def _formal_u0_semantic_replay(
    path: Path,
    *,
    reconstructed_arrays: Mapping[str, NDArray[np.generic]],
    reconstructed_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    """Bind a path-rebound U0 to the byte-verified formal U0 artifact.

    The historical ``FROZEN_U0_HASH`` was produced before the task-local
    contract path was rebound.  Its original helper hashed a Python mapping's
    insertion order, so it is intentionally retained as a *formal declared
    identity* rather than recomputed from a location-dependent metadata
    serialisation.  This function establishes the stronger fact needed here:
    every U0 array is bitwise equal and every non-path checkpoint field is
    equal to the verified formal artifact.
    """

    try:
        with np.load(path, allow_pickle=False) as archive:
            if "metadata_json" not in archive.files:
                raise FrozenExactFlowWorkflowError("formal frozen U0 lacks metadata_json")
            formal_arrays = {
                key: np.asarray(archive[key]).copy()
                for key in archive.files
                if key != "metadata_json"
            }
            formal_metadata = json.loads(str(archive["metadata_json"]))
    except (OSError, ValueError, json.JSONDecodeError) as error:
        raise FrozenExactFlowWorkflowError("could not load verified formal frozen U0") from error
    if set(formal_arrays) != set(reconstructed_arrays):
        raise FrozenExactFlowWorkflowError("formal and reconstructed frozen U0 array keys differ")
    unequal = [
        key
        for key in sorted(formal_arrays)
        if not np.array_equal(np.asarray(formal_arrays[key]), np.asarray(reconstructed_arrays[key]))
    ]
    if unequal:
        raise FrozenExactFlowWorkflowError(f"formal and reconstructed frozen U0 arrays differ: {unequal}")
    if not isinstance(formal_metadata, Mapping):
        raise FrozenExactFlowWorkflowError("formal frozen U0 metadata is not a mapping")
    formal_top = {key: value for key, value in formal_metadata.items() if key != "checkpoint_binding"}
    reconstructed_top = {key: value for key, value in reconstructed_metadata.items() if key != "checkpoint_binding"}
    if formal_top != reconstructed_top:
        raise FrozenExactFlowWorkflowError("formal and reconstructed U0 non-checkpoint metadata differ")
    formal_binding = formal_metadata.get("checkpoint_binding")
    reconstructed_binding = reconstructed_metadata.get("checkpoint_binding")
    if not isinstance(formal_binding, Mapping) or not isinstance(reconstructed_binding, Mapping):
        raise FrozenExactFlowWorkflowError("frozen U0 checkpoint-binding metadata is invalid")
    formal_semantic = {
        key: value for key, value in formal_binding.items() if key not in _PATH_DEPENDENT_CHECKPOINT_BINDING_FIELDS
    }
    reconstructed_semantic = {
        key: value for key, value in reconstructed_binding.items() if key not in _PATH_DEPENDENT_CHECKPOINT_BINDING_FIELDS
    }
    if formal_semantic != reconstructed_semantic:
        raise FrozenExactFlowWorkflowError("formal and reconstructed U0 semantic checkpoint binding differs")
    return {
        "status": "PASS_FORMAL_U0_BITWISE_ARRAY_AND_SEMANTIC_METADATA_REPLAY",
        "formal_declared_frozen_u0_hash": EXPECTED_FROZEN_U0_HASH,
        "array_key_count": len(formal_arrays),
        "path_dependent_rebind_fields": sorted(_PATH_DEPENDENT_CHECKPOINT_BINDING_FIELDS),
        "formal_runtime_contract_path": formal_binding.get("runtime_contract_path"),
        "reconstructed_runtime_contract_path": reconstructed_binding.get("runtime_contract_path"),
    }


def _frozen_observation(source: Any, *, edges_m: NDArray[np.float64], cells: NDArray[np.float64], frozen_xb: float) -> dict[str, float]:
    metrics = metrics_from_piecewise_constant_cells(edges_m, cells)
    beta = source._beta_population_from_numbers(np.asarray(cells, dtype=np.float64))
    gp = source.population("g")
    matrix_fraction = 1.0 - gp.volume_fraction() - beta.volume_fraction()
    if not math.isfinite(matrix_fraction) or matrix_fraction < 0.0:
        raise FrozenExactFlowWorkflowError("frozen open-inventory observation is invalid")
    q_beta = float(beta.b_inventory_mol_m3())
    q_gp = float(gp.b_inventory_mol_m3())
    q_matrix = float(matrix_fraction * frozen_xb / source.ledger.matrix_molar_volume_m3_mol)
    q_total = float(source.ledger.total_b_mol_m3)
    q_reconstructed = q_beta + q_gp + q_matrix
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
        "Q_beta_mol_m3": q_beta,
        "Q_gp_mol_m3": q_gp,
        "Q_matrix_mol_m3": q_matrix,
        "Q_total_mol_m3": q_total,
        "Q_reconstructed_mol_m3": q_reconstructed,
        "inventory_relative_residual": abs(q_reconstructed - q_total) / max(abs(q_total), 1.0e-300),
    }


_OBSERVABLE_FIELDS = (
    "M0_m3",
    "M1_m2",
    "M2_m",
    "M3_dimensionless",
    "Rmean_m",
    "Rmean3_m3",
    "mean_R3_m3",
    "Sv_m_inv",
    "f_beta",
    "Q_beta_mol_m3",
    "Q_matrix_mol_m3",
    "Q_total_mol_m3",
)


def _comparison_row(
    *,
    h_s: float,
    substep_count: int,
    left_cells: NDArray[np.float64],
    right_cells: NDArray[np.float64],
    edges_m: NDArray[np.float64],
    source: Any,
    frozen_xb: float,
    left_lower_loss_m3: float | None = None,
    left_upper_loss_m3: float | None = None,
    right_lower_loss_m3: float | None = None,
    right_upper_loss_m3: float | None = None,
    label: str = "CR1_VS_REF_PC",
) -> dict[str, Any]:
    metrics = measure_error_metrics(left_cells, right_cells, edges_m=edges_m)
    left = _frozen_observation(source, edges_m=edges_m, cells=left_cells, frozen_xb=frozen_xb)
    right = _frozen_observation(source, edges_m=edges_m, cells=right_cells, frozen_xb=frozen_xb)
    row: dict[str, Any] = {
        "comparison": label,
        "h_s": float(h_s),
        "substep_count": int(substep_count),
        **metrics,
        "left_lower_cumulative_number_loss_m3": left_lower_loss_m3,
        "left_upper_cumulative_number_loss_m3": left_upper_loss_m3,
        "right_lower_number_loss_m3": right_lower_loss_m3,
        "right_upper_number_loss_m3": right_upper_loss_m3,
    }
    for field in _OBSERVABLE_FIELDS:
        signed = float(left[field] - right[field])
        row[f"left_{field}"] = float(left[field])
        row[f"reference_{field}"] = float(right[field])
        row[f"signed_{field}"] = signed
        row[f"absolute_{field}"] = abs(signed)
        row[f"relative_{field}"] = abs(signed) / max(abs(float(right[field])), 1.0e-300)
    return row


def _source_cell_indices(edges_m: NDArray[np.float64], departure_faces_m: NDArray[np.float64]) -> NDArray[np.int64]:
    indices = np.searchsorted(edges_m, departure_faces_m, side="right") - 1
    return np.clip(indices, 0, edges_m.size - 2).astype(np.int64)


def _trace_comparison(
    *,
    source: Any,
    flow: FrozenAutonomousExactFlow,
    edges_m: NDArray[np.float64],
    frozen_xb: float,
    h_values: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[float, NDArray[np.float64]]]:
    rows: list[dict[str, Any]] = []
    exact_departures: dict[float, NDArray[np.float64]] = {}

    def production_velocity(radii_m: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(source._velocity_at_radii(np.asarray(radii_m, dtype=np.float64), frozen_xb), dtype=np.float64)

    for h_s in h_values:
        production = trace_departure_faces_rk2(
            edges_m,
            dt_s=float(h_s),
            velocity_m_s=production_velocity,
            lower_radius_m=float(edges_m[0]),
            upper_radius_m=float(edges_m[-1]),
        )
        exact = flow.departure_faces(edges_m, float(h_s))
        exact_departures[float(h_s)] = exact.radius_m.copy()
        delta = np.asarray(production.departure_faces_m - exact.radius_m, dtype=np.float64)
        scale = np.maximum(np.maximum(np.abs(production.departure_faces_m), np.abs(exact.radius_m)), 1.0e-300)
        production_source = _source_cell_indices(edges_m, production.departure_faces_m)
        exact_source = _source_cell_indices(edges_m, exact.radius_m)
        rows.append(
            {
                "h_s": float(h_s),
                "face_count": int(edges_m.size),
                "max_departure_error_m": float(np.max(np.abs(delta))),
                "rms_departure_error_m": float(math.sqrt(float(np.mean(np.square(delta))))),
                "max_relative_departure_error": float(np.max(np.abs(delta) / scale)),
                "source_cell_difference_count": int(np.count_nonzero(production_source != exact_source)),
                "source_cell_difference_fraction": float(np.mean(production_source != exact_source)),
                "production_lower_no_inflow_faces": int(production.lower_no_inflow_face_count),
                "production_upper_no_inflow_faces": int(production.upper_no_inflow_face_count),
                "exact_statuses_json": {str(key): int(value) for key, value in zip(*np.unique(exact.status, return_counts=True))},
            }
        )
    return rows, exact_departures


def _projection_only_rows(
    *,
    source: Any,
    edges_m: NDArray[np.float64],
    u0_cells: NDArray[np.float64],
    exact_reference: ExactPushforwardState,
    exact_departures: Mapping[float, NDArray[np.float64]],
    frozen_xb: float,
    h_values: Sequence[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for h_s in h_values:
        count = int(round(FINAL_HORIZON_S / float(h_s)))
        if not math.isclose(count * float(h_s), FINAL_HORIZON_S, rel_tol=0.0, abs_tol=1.0e-18):
            raise FrozenExactFlowWorkflowError("registered projection h does not divide the fixed final horizon")
        current = np.asarray(u0_cells, dtype=np.float64).copy()
        lower_loss = 0.0
        upper_loss = 0.0
        residual = 0.0
        departure = np.asarray(exact_departures[float(h_s)], dtype=np.float64)
        for _ in range(count):
            remap = conservative_remap_piecewise_constant(edges_m, current, departure)
            current = np.asarray(remap.cell_number_m3, dtype=np.float64)
            lower_loss += float(remap.lower_number_loss_m3)
            upper_loss += float(remap.upper_number_loss_m3)
            residual += float(remap.conservation_residual_m3)
        row = _comparison_row(
            h_s=float(h_s),
            substep_count=count,
            left_cells=current,
            right_cells=exact_reference.cell_number_m3,
            edges_m=edges_m,
            source=source,
            frozen_xb=frozen_xb,
            left_lower_loss_m3=lower_loss,
            left_upper_loss_m3=upper_loss,
            right_lower_loss_m3=exact_reference.lower_number_loss_m3,
            right_upper_loss_m3=exact_reference.upper_number_loss_m3,
            label="EXACT_FLOW_CR1_REMAP_VS_REF_PC",
        )
        row["remap_cumulative_conservation_residual_m3"] = residual
        rows.append(row)
    return rows


def _cr1_rows(
    *,
    source: Any,
    edges_m: NDArray[np.float64],
    exact_reference: ExactPushforwardState,
    frozen_xb: float,
    h_values: Sequence[float],
) -> list[dict[str, Any]]:
    initial_arrays = {key: np.asarray(value).copy() for key, value in source.state_arrays().items()}
    initial_history = tuple(source.history)
    rows: list[dict[str, Any]] = []
    for h_s in h_values:
        count = int(round(FINAL_HORIZON_S / float(h_s)))
        result = phi_compose(
            source,
            dt_s=float(h_s),
            count=count,
            mode="FROZEN_MATRIX",
            frozen_matrix_xb=float(frozen_xb),
        )
        if result.status != "SUCCESS" or result.state is None or result.solver is None:
            raise FrozenExactFlowWorkflowError(
                f"frozen disposable CR1 trajectory did not close for h={h_s:.17g}: {result.error_message}"
            )
        if not _state_arrays_equal(initial_arrays, source.state_arrays()) or tuple(source.history) != initial_history:
            raise FrozenExactFlowWorkflowError("a disposable frozen CR1 refinement mutated the accepted U0 source")
        cells = np.asarray(result.state["population_array"], dtype=np.float64)
        history = tuple(result.solver.history[-count:])
        if len(history) != count:
            raise FrozenExactFlowWorkflowError("frozen CR1 result did not retain one diagnostic per requested substep")
        lower_loss = float(sum(float(item.rmin_number_loss_m3) for item in history))
        upper_loss = float(sum(float(item.rmax_number_loss_m3) for item in history))
        row = _comparison_row(
            h_s=float(h_s),
            substep_count=count,
            left_cells=cells,
            right_cells=exact_reference.cell_number_m3,
            edges_m=edges_m,
            source=source,
            frozen_xb=frozen_xb,
            left_lower_loss_m3=lower_loss,
            left_upper_loss_m3=upper_loss,
            right_lower_loss_m3=exact_reference.lower_number_loss_m3,
            right_upper_loss_m3=exact_reference.upper_number_loss_m3,
        )
        row["cr1_state_hash"] = str(result.state["accepted_state_hash"])
        row["cr1_fixed_point_modes_json"] = [str(item.fixed_point_convergence_mode) for item in history]
        rows.append(row)
    return rows


def _formal_three_path_cells(path: Path, h_values: Sequence[float]) -> list[tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]]:
    try:
        with np.load(path, allow_pickle=False) as archive:
            keys = set(archive.files)
            result: list[tuple[float, NDArray[np.float64], NDArray[np.float64], NDArray[np.float64]]] = []
            for index, h_s in enumerate(h_values):
                label = f"level_{index:02d}_h_{float(h_s):.17g}".replace(".", "p").replace("-", "m")
                required = (f"{label}_U_A", f"{label}_U_B", f"{label}_U_C")
                if not all(key in keys for key in required):
                    raise FrozenExactFlowWorkflowError(f"formal three-path NPZ lacks expected level {label}")
                result.append((
                    float(h_s),
                    np.asarray(archive[required[0]], dtype=np.float64).copy(),
                    np.asarray(archive[required[1]], dtype=np.float64).copy(),
                    np.asarray(archive[required[2]], dtype=np.float64).copy(),
                ))
    except (OSError, ValueError, KeyError) as error:
        raise FrozenExactFlowWorkflowError("could not load formal A/B/C state artifact") from error
    return result


def _old_semigroup_rows(
    *,
    formal_path: Path,
    measure: PiecewiseConstantCumulativeMeasure,
    flow: FrozenAutonomousExactFlow,
    source: Any,
    edges_m: NDArray[np.float64],
    frozen_xb: float,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for h_s, direct, composed, sequential in _formal_three_path_cells(formal_path, OLD_SEMIGROUP_H_S):
        reference = measure.pushforward(flow, h_s)
        for label, cells in (("A_DIRECT", direct), ("B_COMPOSED_HALF_FLOW_SINGLE_REMAP", composed), ("C_SEQUENTIAL_HALF_REMAPS", sequential)):
            row = _comparison_row(
                h_s=h_s,
                substep_count=1 if label != "C_SEQUENTIAL_HALF_REMAPS" else 2,
                left_cells=cells,
                right_cells=reference.cell_number_m3,
                edges_m=edges_m,
                source=source,
                frozen_xb=frozen_xb,
                right_lower_loss_m3=reference.lower_number_loss_m3,
                right_upper_loss_m3=reference.upper_number_loss_m3,
                label=f"OLD_{label}_VS_EXACT_REF_PC_AT_H",
            )
            row["old_path"] = label
            rows.append(row)
    return rows


def _orders_for_rows(rows: Sequence[Mapping[str, Any]], *, prefix: str = "") -> tuple[list[dict[str, Any]], dict[str, Any]]:
    h_values = [float(row["h_s"]) for row in rows]
    metrics = [
        "population_L1_abs",
        "population_relative_L1",
        "population_Linf_abs",
        "CDF_max_error",
        "Wasserstein_m",
        "absolute_weighted_M0",
        "absolute_weighted_M1",
        "absolute_weighted_M2",
        "absolute_weighted_M3",
        *[f"absolute_{field}" for field in _OBSERVABLE_FIELDS],
    ]
    output_rows: list[dict[str, Any]] = []
    summaries: dict[str, Any] = {}
    for metric in metrics:
        values = [float(row[metric]) for row in rows]
        result = observed_orders(h_values, values, metric=f"{prefix}{metric}")
        summaries[metric] = {key: value for key, value in result.items() if key != "pair_rows"}
        output_rows.extend(result["pair_rows"])
    return output_rows, summaries


def _reference_validation(
    *,
    flow: FrozenAutonomousExactFlow,
    measure: PiecewiseConstantCumulativeMeasure,
    u0_cells: NDArray[np.float64],
    edges_m: NDArray[np.float64],
) -> tuple[dict[str, Any], ExactPushforwardState, list[dict[str, Any]]]:
    t0 = measure.pushforward(flow, 0.0)
    if not np.array_equal(t0.cell_number_m3, u0_cells) or not np.array_equal(t0.departure_faces_m, edges_m):
        raise FrozenExactFlowWorkflowError("REF-PC failed its exact t=0 initial-measure identity")
    expected_cdf = np.concatenate((np.asarray([0.0]), np.cumsum(u0_cells, dtype=np.float64)))
    if not np.array_equal(measure.cdf(edges_m), expected_cdf):
        raise FrozenExactFlowWorkflowError("REF-PC CDF does not reproduce initial cell measures")
    t0_metrics = t0.metrics(edges_m)
    u0_metrics = metrics_from_piecewise_constant_cells(edges_m, u0_cells)
    for field in ("M0_m3", "M1_m2", "M2_m", "M3_dimensionless"):
        if getattr(t0_metrics, field) != getattr(u0_metrics, field):
            raise FrozenExactFlowWorkflowError("REF-PC t=0 moments do not reproduce U0 exactly")
    exact_final = measure.pushforward(flow, FINAL_HORIZON_S)
    scale = max(measure.total_number_m3, 1.0e-300)
    conservation_limit = 4096.0 * np.finfo(np.float64).eps * scale
    if abs(exact_final.conservation_residual_m3) > conservation_limit:
        raise FrozenExactFlowWorkflowError("exact pushforward number balance exceeds its float64 reduction budget")
    shadow_rows = flow.tau_shadow_rows(sample_count_per_branch=5, dps=80)
    max_shadow = max(float(row["absolute_discrepancy_s"]) for row in shadow_rows)
    max_shadow_relative = max(float(row["relative_discrepancy"]) for row in shadow_rows)
    # Do not turn a large, physically valid Tau interval into a reference
    # failure merely because its absolute representation is in seconds.  The
    # independently evaluated 80-dps integral instead must agree to a
    # scale-free precision that is materially tighter than the CR1 error
    # regime; physical map closure is checked at radius level immediately
    # after this validation.
    if not math.isfinite(max_shadow) or not math.isfinite(max_shadow_relative) or max_shadow_relative > TAU_SHADOW_RELATIVE_LIMIT:
        raise FrozenExactFlowWorkflowError("SciPy versus mpmath Tau shadow discrepancy is too large for the frozen reference")
    return {
        "t0_identity": "PASS_BITWISE_CELL_MEASURE_AND_CDF",
        "t0_metrics": t0_metrics.as_dict(),
        "reference_final_number_balance_residual_m3": exact_final.conservation_residual_m3,
        "reference_number_balance_limit_m3": conservation_limit,
        "tau_shadow_max_absolute_discrepancy_s": max_shadow,
        "tau_shadow_max_relative_discrepancy": max_shadow_relative,
        "tau_shadow_relative_limit": TAU_SHADOW_RELATIVE_LIMIT,
        "tau_shadow_precision_status": "PASS_SCIPY_QUAD_VS_MPMATH_RELATIVE_PRECISION",
        "tau_shadow_row_count": len(shadow_rows),
    }, exact_final, shadow_rows


def _decision(
    *,
    reference_valid: bool,
    cr1_classification: Mapping[str, Any],
    projection_classification: Mapping[str, Any],
    trace_rows: Sequence[Mapping[str, Any]],
    semigroup: Mapping[str, Any],
    representation_row: Mapping[str, Any],
    cr1_rows: Sequence[Mapping[str, Any]],
    projection_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    if not reference_valid:
        return {
            "STATUS": "DIAG_FROZEN_EXACT_REFERENCE_FAILED",
            "PRIMARY_ROOT_CAUSE": "EXACT_FLOW_REFERENCE_VALIDATION_FAILED",
            "SECONDARY_ROOT_CAUSE": None,
            "CR2_AUTHORIZED": False,
            "TRACE_INTEGRATOR_AUDIT_AUTHORIZED": False,
        }
    trace_floor = max(float(semigroup["max_relative_radius_error"]), 32.0 * np.finfo(np.float64).eps)
    trace_max = max(float(row["max_relative_departure_error"]) for row in trace_rows)
    trace_validated = trace_max <= 1024.0 * trace_floor
    cr1_primary = str(cr1_classification["classification"])
    projection_primary = str(projection_classification["classification"])
    cr1_late = float(cr1_rows[-1]["population_relative_L1"])
    projection_late = float(projection_rows[-1]["population_relative_L1"])
    representation_l1 = float(representation_row["population_relative_L1"])
    trace_dominates = (not trace_validated) and cr1_late > 3.0 * max(projection_late, 1.0e-300)
    representation_dominates = representation_l1 > max(cr1_late, projection_late)
    convergent = cr1_primary in {
        "CR1_ASYMPTOTICALLY_CONSISTENT_FIRST_ORDER_LIKE",
        "CR1_ASYMPTOTICALLY_CONSISTENT_SUBFIRST_ORDER",
        "CR1_CONVERGENT_BUT_PREASYMPTOTIC",
    }
    projection_bad = projection_primary in {
        "CR1_ERROR_STAGNATION_AGAINST_EXACT_REFERENCE",
        "CR1_NONMONOTONE_NONCONVERGENT",
    }
    if trace_dominates:
        status = "DIAG_TRACE_INTEGRATOR_ERROR_SUPPORTED"
        primary = "FINITE_H_PRODUCTION_TRACE_ERROR_DOMINATES_EXACT_FLOW_REMAPPING_COMPARATOR"
        secondary = cr1_primary
        cr2 = False
        trace_authorized = True
    elif projection_bad and trace_validated:
        status = "DIAG_CR1_PROJECTION_ERROR_SUPPORTED"
        primary = "EXACT_FLOW_PLUS_CR1_REMAP_DOES_NOT_CONVERGE_AGAINST_REF_PC"
        secondary = cr1_primary
        cr2 = "ELIGIBLE_FOR_SEPARATELY_GATED_CR2_PROTOTYPE"
        trace_authorized = False
    elif representation_dominates:
        status = "DIAG_INITIAL_MEASURE_REPRESENTATION_LIMIT"
        primary = "REF_PC_REF_PL_INITIAL_MEASURE_SENSITIVITY_EXCEEDS_LATE_TEMPORAL_ERROR"
        secondary = cr1_primary
        cr2 = False
        trace_authorized = False
    elif convergent:
        status = "DIAG_CR1_ASYMPTOTICALLY_CONSISTENT"
        primary = "CR1_CONVERGES_TO_INDEPENDENT_FROZEN_REF_PC"
        secondary = "TRACE_GEOMETRY_VALIDATED" if trace_validated else "FINITE_H_TRACE_ERROR_QUANTIFIED_NOT_DOMINANT"
        cr2 = False
        trace_authorized = False
    else:
        status = "DIAG_INSUFFICIENT_ASYMPTOTIC_RANGE"
        primary = "EXACT_REFERENCE_EXISTS_BUT_REGISTERED_REFINEMENT_DOES_NOT_YET_SUPPORT_A_METHOD_DECISION"
        secondary = cr1_primary
        cr2 = False
        trace_authorized = False
    if status not in ALLOWED_TOP_LEVEL_STATUSES:
        raise FrozenExactFlowWorkflowError("forbidden frozen exact-flow top-level status")
    return {
        "STATUS": status,
        "PRIMARY_ROOT_CAUSE": primary,
        "SECONDARY_ROOT_CAUSE": secondary,
        "CR2_AUTHORIZED": cr2,
        "TRACE_INTEGRATOR_AUDIT_AUTHORIZED": trace_authorized,
        "TRACE_VS_EXACT": "TRACE_GEOMETRY_VALIDATED" if trace_validated else "TRACE_TIME_INTEGRATION_ERROR_QUANTIFIED",
        "trace_reference_floor_relative": trace_floor,
        "trace_max_relative_departure_error": trace_max,
        "trace_dominates": trace_dominates,
        "representation_dominates": representation_dominates,
    }


def _run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    if output_root.exists() or report_root.exists():
        raise FrozenExactFlowWorkflowError("refusing to overwrite existing output or report roots")
    source_identity = _source_identity()
    inherited = _inherited_inputs(args)
    mpmath = _mpmath_preflight(Path(args.mpmath_vendor_root))
    output_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    (output_root / "figures").mkdir()
    started = time.monotonic()

    source, baseline, u0_arrays, u0_metadata = prior._load_exact_u0(Path(args.restart_checkpoint), output_root)
    formal_u0_replay = _formal_u0_semantic_replay(
        Path(args.formal_frozen_u0),
        reconstructed_arrays=u0_arrays,
        reconstructed_metadata=u0_metadata,
    )
    baseline["reconstructed_path_sensitive_content_hash"] = baseline["frozen_u0_content_hash"]
    baseline["frozen_u0_content_hash"] = EXPECTED_FROZEN_U0_HASH
    baseline["formal_u0_semantic_replay"] = formal_u0_replay
    baseline["inherited_provenance_inputs"] = inherited
    before_arrays = {key: np.asarray(value).copy() for key, value in source.state_arrays().items()}
    before_history = tuple(source.history)
    edges = np.asarray(u0_arrays["radius_edges_m"], dtype=np.float64)
    u0_cells = np.asarray(u0_arrays["cell_number_m3"], dtype=np.float64)
    frozen_xb = float(u0_arrays["matrix_xb"][0])
    if not np.array_equal(u0_cells, np.asarray(source._beta_cell_numbers(), dtype=np.float64)):
        raise FrozenExactFlowWorkflowError("source beta cell measure differs from the frozen U0 artifact")

    beta = source.population("beta")
    law = FrozenAutonomousGrowthLaw(
        parameters=beta.parameters,
        equilibrium_adapter=source.equilibrium_adapter,
        matrix_xb=frozen_xb,
        lower_radius_m=float(edges[0]),
        upper_radius_m=float(edges[-1]),
    )
    parity_radii = np.unique(np.concatenate((edges, np.sqrt(edges[:-1] * edges[1:]))))
    shared_velocity = np.asarray(source._velocity_at_radii(parity_radii, frozen_xb), dtype=np.float64)
    independent_velocity = law.velocity(parity_radii)
    if not np.array_equal(shared_velocity, independent_velocity):
        difference = float(np.max(np.abs(shared_velocity - independent_velocity)))
        raise FrozenExactFlowWorkflowError(f"independent law does not bitwise reproduce shared growth kernel: {difference:.17g}")
    flow = FrozenAutonomousExactFlow(law, edges)
    measure = PiecewiseConstantCumulativeMeasure(edges, u0_cells)
    reference_validation, exact_reference, shadow_rows = _reference_validation(
        flow=flow, measure=measure, u0_cells=u0_cells, edges_m=edges
    )
    semigroup = flow.semigroup_check(edges, FINAL_HORIZON_S)
    roundtrip = flow.roundtrip_check(edges, FINAL_HORIZON_S)
    semigroup_limit = 8192.0 * np.finfo(np.float64).eps
    if int(semigroup["status_mismatch_count"]) != 0 or float(semigroup["max_relative_radius_error"]) > semigroup_limit:
        raise FrozenExactFlowWorkflowError("exact-flow semigroup self-check did not close at declared reference precision")
    if int(roundtrip["status_mismatch_count"]) != 0 or roundtrip["max_relative_radius_error"] is None or float(roundtrip["max_relative_radius_error"]) > semigroup_limit:
        raise FrozenExactFlowWorkflowError("exact-flow inverse roundtrip did not close at declared reference precision")

    law_rows = [dict(row) for row in flow.growth_law_rows]
    tau_by_radius: dict[float, float] = {float(row["radius_m"]): float(row["tau_s"]) for row in flow.tau_table_rows()}
    crossing_by_radius = {float(row["radius_m"]): float(row["t_to_Rmin_s"]) for row in flow.rmin_crossing_rows(edges)}
    for row in law_rows:
        radius = float(row["radius_m"])
        row["tau_s"] = tau_by_radius.get(radius)
        row["t_to_Rmin_s"] = crossing_by_radius.get(radius)
    _write_csv(output_root / "frozen_exact_flow_table.csv", law_rows, ("radius_m", "growth_velocity_m_s", "branch"))
    tau_rows = flow.tau_table_rows()
    tau_rows.extend({"shadow": True, **row} for row in shadow_rows)
    _write_csv(output_root / "tau_reference.csv", tau_rows, ("radius_m", "tau_s", "branch"))
    _write_csv(output_root / "exact_flow_semigroup.csv", [semigroup, {"check": "inverse_roundtrip", **roundtrip}], ("max_radius_error_m",))

    pl_measure = PiecewiseLinearCumulativeMeasure(edges, u0_cells)
    pl_reference = pl_measure.pushforward(flow, FINAL_HORIZON_S)
    representation_row = _comparison_row(
        h_s=FINAL_HORIZON_S,
        substep_count=1,
        left_cells=pl_reference.cell_number_m3,
        right_cells=exact_reference.cell_number_m3,
        edges_m=edges,
        source=source,
        frozen_xb=frozen_xb,
        left_lower_loss_m3=pl_reference.lower_number_loss_m3,
        left_upper_loss_m3=pl_reference.upper_number_loss_m3,
        right_lower_loss_m3=exact_reference.lower_number_loss_m3,
        right_upper_loss_m3=exact_reference.upper_number_loss_m3,
        label="REF_PL_DIAGNOSTIC_VS_REF_PC_AUTHORITY",
    )
    _write_csv(output_root / "ref_pc_vs_ref_pl.csv", [representation_row], ("comparison", "population_relative_L1"))
    np.savez(
        output_root / "exact_pushforward_states.npz",
        radius_edges_m=edges,
        u0_cell_number_m3=u0_cells,
        ref_pc_final_cell_number_m3=exact_reference.cell_number_m3,
        ref_pc_final_departure_faces_m=exact_reference.departure_faces_m,
        ref_pl_final_cell_number_m3=pl_reference.cell_number_m3,
        ref_pl_final_departure_faces_m=pl_reference.departure_faces_m,
        frozen_xb=np.asarray([frozen_xb], dtype=np.float64),
        final_horizon_s=np.asarray([FINAL_HORIZON_S], dtype=np.float64),
    )

    trace_rows, exact_departures = _trace_comparison(
        source=source, flow=flow, edges_m=edges, frozen_xb=frozen_xb, h_values=H_LADDER_S
    )
    trace_order = observed_orders(
        H_LADDER_S, [float(row["max_relative_departure_error"]) for row in trace_rows], metric="trace_max_relative_departure_error"
    )
    _write_csv(output_root / "trace_vs_exact.csv", trace_rows, ("h_s", "max_departure_error_m"))

    projection_rows = _projection_only_rows(
        source=source,
        edges_m=edges,
        u0_cells=u0_cells,
        exact_reference=exact_reference,
        exact_departures=exact_departures,
        frozen_xb=frozen_xb,
        h_values=H_LADDER_S,
    )
    _write_csv(output_root / "exact_flow_cr1_remap.csv", projection_rows, ("h_s", "population_relative_L1"))
    cr1_rows = _cr1_rows(
        source=source, edges_m=edges, exact_reference=exact_reference, frozen_xb=frozen_xb, h_values=H_LADDER_S
    )
    _write_csv(output_root / "cr1_vs_exact_refinement.csv", cr1_rows, ("h_s", "population_relative_L1"))
    if not _state_arrays_equal(before_arrays, source.state_arrays()) or tuple(source.history) != before_history:
        raise FrozenExactFlowWorkflowError("frozen diagnostic modified the accepted source state")

    cr1_order_rows, cr1_orders = _orders_for_rows(cr1_rows, prefix="CR1_")
    projection_order_rows, projection_orders = _orders_for_rows(projection_rows, prefix="EXACT_FLOW_CR1_REMAP_")
    trace_order_rows = trace_order["pair_rows"]
    _write_csv(output_root / "observed_order.csv", [*cr1_order_rows, *projection_order_rows, *trace_order_rows], ("metric", "h_coarse_s", "observed_order"))
    cr1_classification = classify_refinement_against_reference(
        H_LADDER_S, [float(row["population_relative_L1"]) for row in cr1_rows]
    )
    projection_classification = classify_refinement_against_reference(
        H_LADDER_S, [float(row["population_relative_L1"]) for row in projection_rows]
    )

    old_rows = _old_semigroup_rows(
        formal_path=Path(args.formal_three_path_states),
        measure=measure,
        flow=flow,
        source=source,
        edges_m=edges,
        frozen_xb=frozen_xb,
    )
    _write_csv(output_root / "old_semigroup_vs_exact.csv", old_rows, ("h_s", "old_path", "population_relative_L1"))

    decision = _decision(
        reference_valid=True,
        cr1_classification=cr1_classification,
        projection_classification=projection_classification,
        trace_rows=trace_rows,
        semigroup=semigroup,
        representation_row=representation_row,
        cr1_rows=cr1_rows,
        projection_rows=projection_rows,
    )
    _write_json(output_root / "method_decision.json", decision)
    old_h128 = [row for row in old_rows if float(row["h_s"]) == OLD_SEMIGROUP_H_S[0]]
    old_h128_by_path = {str(row["old_path"]): float(row["population_relative_L1"]) for row in old_h128}
    final = {
        "STATUS": decision["STATUS"],
        "BRANCH": source_identity["git_branch"],
        "COMMIT": source_identity["git_commit"],
        "SOURCE_CLEAN": True,
        "TESTS": str(args.test_status),
        "FROZEN_U0_HASH": baseline["frozen_u0_content_hash"],
        "EXACT_FLOW_REFERENCE": "FROZEN_AUTONOMOUS_TIME_OF_FLIGHT_SCIPY_QUAD_PLUS_MPMATH_SHADOW",
        "EXACT_FLOW_SEMIGROUP_ERROR": semigroup,
        "TAU_QUADRATURE_ERROR": {
            "scipy_quad_reported_max_s": flow.quadrature_error_max_s,
            "mpmath_shadow_max_absolute_discrepancy_s": reference_validation["tau_shadow_max_absolute_discrepancy_s"],
            "mpmath_shadow_max_relative_discrepancy": reference_validation["tau_shadow_max_relative_discrepancy"],
            "mpmath_shadow_relative_limit": reference_validation["tau_shadow_relative_limit"],
            "shadow_precision_status": reference_validation["tau_shadow_precision_status"],
        },
        "INVERSE_FLOW_ROUNDTRIP_ERROR": roundtrip,
        "EXACT_PUSHFORWARD_REFERENCE": "REF_PC_DIRECT_U0_TO_T_NO_REPEATED_REMAP",
        "T0_IDENTITY": reference_validation["t0_identity"],
        "REFERENCE_CONSERVATION": {
            "residual_m3": exact_reference.conservation_residual_m3,
            "limit_m3": reference_validation["reference_number_balance_limit_m3"],
        },
        **{f"CR1_H{int(round(1.0 / float(row['h_s'])))}_ERROR": float(row["population_relative_L1"]) for row in cr1_rows},
        "CR1_OBSERVED_ORDER": cr1_orders["population_relative_L1"],
        "CR1_ASYMPTOTIC_CLASS": cr1_classification,
        "TRACE_VS_EXACT": decision["TRACE_VS_EXACT"],
        "TRACE_OBSERVED_ORDER": {key: value for key, value in trace_order.items() if key != "pair_rows"},
        "EXACT_FLOW_CR1_REMAP_ERROR": [
            {"h_s": row["h_s"], "population_relative_L1": row["population_relative_L1"]} for row in projection_rows
        ],
        "PROJECTION_OBSERVED_ORDER": projection_orders["population_relative_L1"],
        "OLD_TOTAL_DEFECT_VS_EXACT": old_h128_by_path.get("C_SEQUENTIAL_HALF_REMAPS"),
        "OLD_TRACE_DEFECT_VS_EXACT": old_h128_by_path.get("B_COMPOSED_HALF_FLOW_SINGLE_REMAP"),
        "OLD_REMAP_DEFECT_VS_EXACT": old_h128_by_path.get("A_DIRECT"),
        "REF_PC_VS_REF_PL": {key: representation_row[key] for key in ("population_relative_L1", "CDF_max_error", "Wasserstein_m")},
        "PRIMARY_ROOT_CAUSE": decision["PRIMARY_ROOT_CAUSE"],
        "SECONDARY_ROOT_CAUSE": decision["SECONDARY_ROOT_CAUSE"],
        "CR2_AUTHORIZED": decision["CR2_AUTHORIZED"],
        "TRACE_INTEGRATOR_AUDIT_AUTHORIZED": decision["TRACE_INTEGRATOR_AUDIT_AUTHORIZED"],
        "FROZEN_TIME_REFERENCE_V1": "ESTABLISHED_FOR_FROZEN_AUTONOMOUS_GROWTH_ONLY",
        "TIME_REFERENCE_V2": "NOT_ASSIGNED",
        "M16_AUTHORITY": "NOT_AUTHORITY",
        "PF_SOURCE_MODIFIED": False,
        "CUDA_RERUN": False,
        "PHYSICAL_RETUNING": False,
        "GP_RELEASE_RUN": False,
        "TOP_5_FINDINGS": [
            "REF-PC is a direct U0-to-final-time exact-flow pushforward; it never repeats CR1 projection.",
            f"CR1 exact-reference classification: {cr1_classification['classification']}.",
            f"Projection-only exact-flow classification: {projection_classification['classification']}.",
            f"Trace comparison: {decision['TRACE_VS_EXACT']}.",
            "Frozen reference is not a dynamic nonlinear TIME_REFERENCE_V2 authority.",
        ],
        "P0_BLOCKERS": ["TIME_REFERENCE_V2_NOT_ASSIGNED", decision["STATUS"]],
        "NEXT_ACTION": (
            "Proceed only with the separately authorized production trace-integrator audit; CR2 remains unauthorized."
            if decision["TRACE_INTEGRATOR_AUDIT_AUTHORIZED"]
            else "Return to separately designed dynamic nonlinear closure/time-reference work."
            if decision["CR2_AUTHORIZED"] is False
            else "Await explicit separately gated CR2 prototype authorization."
        ),
        "KEY_REPORTS": [
            f"reports/{TASK_NAME}/04_exact_pushforward_reference.md",
            f"reports/{TASK_NAME}/05_cr1_vs_exact_refinement.md",
            f"reports/{TASK_NAME}/08_projection_only_error.md",
            f"reports/{TASK_NAME}/12_final_acceptance_report.md",
        ],
    }
    reports = {
        "00_baseline.md": {**baseline, "source_identity": source_identity, "mpmath_preflight": mpmath},
        "01_autonomous_growth_law.md": {
            "shared_kernel": "growth_rate_m_s",
            "frozen_matrix_xB": frozen_xb,
            "critical_radius_m": law.critical_radius_m,
            "growth_kernel_bitwise_parity": True,
            "full_domain_table": f"outputs/{TASK_NAME}/frozen_exact_flow_table.csv",
        },
        "02_time_of_flight_reference.md": {
            "method": "SciPy adaptive Gauss-Kronrod Tau plus independent mpmath 80-dps shadow",
            "no_production_trace_or_time_of_flight_helper": True,
            "quadrature": reference_validation,
            "tau_table": f"outputs/{TASK_NAME}/tau_reference.csv",
        },
        "03_exact_flow_validation.md": {"semigroup": semigroup, "roundtrip": roundtrip, "reference_precision_limit_relative": semigroup_limit},
        "04_exact_pushforward_reference.md": {
            "authority": "REF_PC",
            "definition": "one exact backward face map from U0 at each target time; no timestep-by-timestep remap",
            "t0_identity": reference_validation["t0_identity"],
            "reference_final": {
                "lower_loss_m3": exact_reference.lower_number_loss_m3,
                "upper_loss_m3": exact_reference.upper_number_loss_m3,
                "conservation_residual_m3": exact_reference.conservation_residual_m3,
            },
        },
        "05_cr1_vs_exact_refinement.md": {"final_horizon_s": FINAL_HORIZON_S, "rows": cr1_rows, "classification": cr1_classification},
        "06_observed_order.md": {"cr1_orders": cr1_orders, "projection_orders": projection_orders, "trace_order": trace_order},
        "07_trace_vs_exact.md": {"rows": trace_rows, "interpretation": decision["TRACE_VS_EXACT"]},
        "08_projection_only_error.md": {"rows": projection_rows, "classification": projection_classification},
        "09_old_semigroup_reinterpretation.md": {
            "rule": "A/B/C are each compared with exact U0-to-h REF-PC; pairwise A-B/B-C differences are not absolute error",
            "rows": old_rows,
        },
        "10_representation_sensitivity.md": {"authority": "REF_PC", "diagnostic_only": "REF_PL", "comparison": representation_row},
        "11_method_decision.md": decision,
        "12_final_acceptance_report.md": final,
        "13_reproduction_commands.md": {
            "final_horizon_s": FINAL_HORIZON_S,
            "registered_h_s": list(H_LADDER_S),
            "command": (
                f"PYTHONPATH=<vendor>:src {sys.executable} scripts/run_kwn_frozen_exact_flow_reference_v1.py audit "
                "--restart-checkpoint <step244-restart> --formal-trace-csv <formal-trace> "
                "--formal-phase-a-csv <phase-a> --frozen-semigroup-csv <old-semigroup> "
                "--formal-frozen-u0 <formal-u0> --formal-three-path-states <three-path-npz> "
                "--formal-h-refinement-csv <h-refinement-csv> --mpmath-vendor-root <vendor> "
                "--output-root <new-output> --report-root <new-report> --test-status <tests>"
            ),
            "diagnostic_only": True,
            "time_reference_v2": "NOT_ASSIGNED",
        },
    }
    _write_reports(report_root, reports)
    provenance = {
        "task_name": TASK_NAME,
        "top_status": final["STATUS"],
        "source": source_identity,
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "mpmath": mpmath,
        "baseline": baseline,
        "u0_metadata": u0_metadata,
        "inherited_inputs": inherited,
        "registered_h_s": list(H_LADDER_S),
        "final_horizon_s": FINAL_HORIZON_S,
        "time_reference_v2": "NOT_ASSIGNED",
        "diagnostic_only": True,
        "pf_source_modified": False,
        "cuda_rerun": False,
        "physical_retuning": False,
        "gp_release": False,
        "runtime_s": time.monotonic() - started,
    }
    _write_json(output_root / "analysis_provenance.json", provenance)
    print(json.dumps(_json_safe(final), indent=2, sort_keys=True))
    return 0, final


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    audit = subparsers.add_parser("audit", help="run the frozen exact-flow diagnostic")
    audit.add_argument("--restart-checkpoint", required=True)
    audit.add_argument("--formal-trace-csv", required=True)
    audit.add_argument("--formal-phase-a-csv", required=True)
    audit.add_argument("--frozen-semigroup-csv", required=True)
    audit.add_argument("--formal-frozen-u0", required=True)
    audit.add_argument("--formal-three-path-states", required=True)
    audit.add_argument("--formal-h-refinement-csv", required=True)
    audit.add_argument("--mpmath-vendor-root", required=True)
    audit.add_argument("--output-root", required=True)
    audit.add_argument("--report-root", required=True)
    audit.add_argument("--test-status", required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        code, _final = _run(args)
    except (FrozenExactFlowWorkflowError, FrozenExactFlowReferenceError, OSError, ValueError) as error:
        print(json.dumps({"STATUS": "DIAG_FROZEN_EXACT_REFERENCE_FAILED", "error_type": type(error).__name__, "error": str(error)}, indent=2), file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
