#!/usr/bin/env python3
"""Frozen-only CR1 semigroup operator-error decomposition and causality audit.

This runner consumes the exact accepted step-244 CR1 restart used by the
previous frozen-matrix semigroup audit.  It never advances that accepted
trajectory, changes a closure/root policy, or evaluates dynamic/prescribed-x
paths.  All A/B/C maps are disposable, autonomous, frozen-xB diagnostics:

* A = one current CR1 face flow followed by one CR1 remap;
* B = two physical half flows composed before one final remap; and
* C = two ordinary half trace/remap operations.

The implementation belongs to an audit harness only.  It is not a CR2
prototype, a time-reference qualification, or a production solver.
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

from kwn_mvp.characteristic_dt_continuation import accepted_state_hash  # noqa: E402
from kwn_mvp.characteristic_reference import CharacteristicReferenceSolver  # noqa: E402
from kwn_mvp.characteristic_semigroup import phi_compose  # noqa: E402
from kwn_mvp.frozen_semigroup_decomposition import (  # noqa: E402
    FrozenSemigroupDecomposition,
    FrozenSemigroupDecompositionError,
    additive_residual_metrics,
    changed_face_indices,
    decompose_frozen_cr1,
    defect_cosine,
    defect_metrics,
    face_flow_rows,
    operator_audit,
    projection_connectivity_rows,
    ratio,
    search_zero_event_control,
    support_expansion_rows,
    support_from_faces,
    support_pairing_metrics,
    synthetic_single_event_control,
)
from kwn_mvp.population_metrics import metrics_from_piecewise_constant_cells  # noqa: E402
from scripts import run_kwn_cr1_fine_substep_pathology_v1 as prior  # noqa: E402
from scripts.frozen_canonical_smooth_population_v1 import (  # noqa: E402
    build_frozen_canonical_context,
    frozen_canonical_snapshot_provenance,
)


TASK_NAME = "kwn_frozen_semigroup_decomposition_v1"
REQUIRED_BRANCH = "codex/kwn-frozen-semigroup-decomposition-v1"
FROZEN_PARENT_COMMIT = "52055befdb5093c348617defeaf03b149cba3c18"
DT0_S = 0.015625
SEMIGROUP_DIVISORS = (2, 4, 8, 16)
RESTART_STEP = 244
RESTART_STATE_HASH = "51c243b61f4278fdf8f5bae9afe50672f79a6e0be57260b9c7cc0fdb46dca4c2"
EXPECTED_FORMAL_TRACE_SHA256 = "e63e546ccd31d93d105b2bf40e3b5abb0fee9e9015f06eace45610a5654dad65"
EXPECTED_FORMAL_PHASE_A_SHA256 = "a91d1940fe471b7f2ebfde5402e177b06fa7b55056bcacdce93519c668755703"
EXPECTED_FROZEN_SEMIGROUP_SHA256 = "2eed7069411a21b5d23de6d7749f88af3c93729790ebd884e8df956c2393dc1c"
INHERITED_TEST_STATUS = "INHERITED_187_PASS_NOT_RERUN"
ADDITIVE_MACHINE_EPS_MULTIPLIER = 1024.0
OPERATOR_MACHINE_EPS_MULTIPLIER = 4096.0
PRODUCTION_PARITY_MACHINE_EPS_MULTIPLIER = 4096.0
# This is a diagnostic classification threshold, fixed before inspecting the
# zero-event result.  It is deliberately expressed in the normalized physical
# population metric used by the control, rather than in an absolute-count
# threshold that would depend on the frozen population normalization.
ZERO_EVENT_MACHINE_RELATIVE_L1_EPS_MULTIPLIER = 4096.0
NEAR_ORTHOGONAL_COSINE = 0.10
CAUSAL_DOMINANCE_RATIO = 3.0

ALLOWED_TOP_LEVEL_STATUSES = {
    "DIAG_CR1_INTERMEDIATE_REMAP_DEFECT_SUPPORTED",
    "DIAG_CHARACTERISTIC_FLOW_COMPOSITION_DEFECT_SUPPORTED",
    "DIAG_MIXED_FLOW_AND_REMAP_DEFECT",
    "DIAG_TOPOLOGY_EVENT_LINK_NOT_SUPPORTED",
    "DIAG_SEMIGROUP_DECOMPOSITION_HARNESS_FAILURE",
    "DIAG_INSUFFICIENT_FROZEN_OPERATOR_EVIDENCE",
}

REPORT_TITLES = {
    "00_baseline.md": "Frozen baseline and U0 repeat-load preflight",
    "01_frozen_operator_contract.md": "Frozen autonomous CR1 operator contract",
    "02_three_path_definition.md": "A/B/C three-path definition",
    "03_additive_error_decomposition.md": "Strict additive error decomposition",
    "04_face_flow_composition.md": "Characteristic face-flow composition",
    "05_intermediate_projection_connectivity.md": "Intermediate CR1 projection connectivity",
    "06_frozen_error_changed_face_pairing.md": "Frozen error and changed-support pairing",
    "07_support_expansion.md": "Exact support and halo expansion",
    "08_sparse_operator_audit.md": "Sparse CR1 operator audit",
    "09_zero_event_control.md": "Algorithmic zero-event control",
    "10_single_event_control.md": "Synthetic single-event control",
    "11_h_refinement.md": "Frozen h-refinement decomposition",
    "12_causal_classification.md": "Fail-closed causal classification",
    "13_method_decision.md": "Method decision boundary",
    "14_final_acceptance_report.md": "Final acceptance report",
    "15_reproduction_commands.md": "Reproduction commands",
}


class FrozenSemigroupWorkflowError(RuntimeError):
    """Fail-closed orchestration error for this diagnostic-only workflow."""


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
    if isinstance(value, np.ndarray):
        return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    if isinstance(value, (Mapping, list, tuple)):
        return json.dumps(_json_safe(value), sort_keys=True, separators=(",", ":"))
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


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
    if isinstance(payload, str):
        body = payload.rstrip()
    else:
        body = "```json\n" + json.dumps(_json_safe(dict(payload)), indent=2, sort_keys=True) + "\n```"
    path.write_text(f"# {title}\n\n{body}\n", encoding="utf-8")


def _write_reports(report_root: Path, payloads: Mapping[str, Mapping[str, Any] | str]) -> None:
    unknown = set(payloads).difference(REPORT_TITLES)
    if unknown:
        raise FrozenSemigroupWorkflowError(f"unknown report name(s): {sorted(unknown)}")
    for name, title in REPORT_TITLES.items():
        _write_markdown(report_root / name, title, payloads.get(name, {"status": "NOT_REACHED"}))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _content_hash(parts: Mapping[str, Any]) -> str:
    """Hash scalar and ndarray content without depending on NPZ ZIP timestamps."""

    digest = hashlib.sha256()
    for key in sorted(parts):
        value = parts[key]
        digest.update(str(key).encode("utf-8"))
        digest.update(b"\0")
        if isinstance(value, np.ndarray):
            array = np.ascontiguousarray(value)
            digest.update(str(array.dtype).encode("ascii"))
            digest.update(repr(tuple(array.shape)).encode("ascii"))
            digest.update(array.tobytes())
        elif isinstance(value, (float, np.floating)):
            digest.update(float(value).hex().encode("ascii"))
        elif isinstance(value, (int, np.integer, bool, np.bool_)):
            digest.update(repr(int(value)).encode("ascii"))
        else:
            digest.update(str(value).encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()


def _git(command: Sequence[str]) -> str:
    completed = subprocess.run(
        ["git", *command], cwd=ROOT, check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
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
        raise FrozenSemigroupWorkflowError("could not establish source identity") from error
    if branch != REQUIRED_BRANCH:
        raise FrozenSemigroupWorkflowError(f"source branch differs from required {REQUIRED_BRANCH}")
    if not parent_is_ancestor:
        raise FrozenSemigroupWorkflowError("source does not descend from required frozen parent")
    if status:
        raise FrozenSemigroupWorkflowError("source tree is dirty")
    return {
        "git_branch": branch,
        "git_commit": commit,
        "git_status": status,
        "frozen_parent_commit": FROZEN_PARENT_COMMIT,
        "frozen_parent_is_ancestor": parent_is_ancestor,
    }


def _implementation_provenance() -> dict[str, str]:
    files = {
        "runner_sha256": Path(__file__).resolve(),
        "frozen_decomposition_module_sha256": ROOT / "src" / "kwn_mvp" / "frozen_semigroup_decomposition.py",
        "production_characteristic_reference_sha256": ROOT / "src" / "kwn_mvp" / "characteristic_reference.py",
        "conservative_remap_sha256": ROOT / "src" / "kwn_mvp" / "conservative_remap.py",
        "prior_semigroup_runner_sha256": ROOT / "scripts" / "run_kwn_cr1_semigroup_audit_v1.py",
        "pathology_replay_helper_sha256": ROOT / "scripts" / "run_kwn_cr1_fine_substep_pathology_v1.py",
        "frozen_canonical_helper_sha256": ROOT / "scripts" / "frozen_canonical_smooth_population_v1.py",
    }
    return {label: _sha256_file(path) for label, path in files.items()}


def _validate_inherited_evidence_inputs(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    """Bind read-only prior evidence by the exact archived file identities.

    The files are inherited provenance only.  This frozen A/B/C runner does
    not parse them to reconstruct a state or re-open dynamic/prescribed-x
    branches; their hashes prevent a changed historical artifact from being
    silently associated with this diagnostic run.
    """

    expected = {
        "formal_trace_csv": (Path(args.formal_trace_csv), EXPECTED_FORMAL_TRACE_SHA256),
        "formal_phase_a_csv": (Path(args.formal_phase_a_csv), EXPECTED_FORMAL_PHASE_A_SHA256),
        "frozen_semigroup_csv": (Path(args.frozen_semigroup_csv), EXPECTED_FROZEN_SEMIGROUP_SHA256),
    }
    verified: dict[str, dict[str, str]] = {}
    for label, (path, expected_sha256) in expected.items():
        if not path.is_file():
            raise FrozenSemigroupWorkflowError(f"required inherited evidence is unavailable: {label}")
        observed_sha256 = _sha256_file(path)
        if observed_sha256 != expected_sha256:
            raise FrozenSemigroupWorkflowError(
                f"required inherited evidence hash differs for {label}: {observed_sha256}"
            )
        verified[label] = {
            "path": str(path),
            "sha256": observed_sha256,
            "expected_sha256": expected_sha256,
            "use": "READ_ONLY_INHERITED_PROVENANCE_NOT_DIAGNOSTIC_INPUT_STATE",
        }
    return verified


def _state_arrays_equal(
    first: Mapping[str, NDArray[np.generic]], second: Mapping[str, NDArray[np.generic]]
) -> bool:
    return set(first) == set(second) and all(
        np.array_equal(np.asarray(first[key]), np.asarray(second[key])) for key in first
    )


def _open_frozen_inventory(
    source: CharacteristicReferenceSolver, cells: NDArray[np.float64], frozen_xb: float
) -> dict[str, float]:
    """Report held-xB inventory without invoking a production closure change."""

    beta = source._beta_population_from_numbers(np.asarray(cells, dtype=np.float64))
    gp = source.population("g")
    matrix_fraction = 1.0 - gp.volume_fraction() - beta.volume_fraction()
    if not math.isfinite(matrix_fraction) or matrix_fraction < 0.0:
        raise FrozenSemigroupWorkflowError("frozen diagnostic produced an invalid matrix fraction")
    q_beta = float(beta.b_inventory_mol_m3())
    q_gp = float(gp.b_inventory_mol_m3())
    q_matrix = float(matrix_fraction * frozen_xb / source.ledger.matrix_molar_volume_m3_mol)
    q_reconstructed = q_beta + q_gp + q_matrix
    q_total = float(source.ledger.total_b_mol_m3)
    residual = q_reconstructed - q_total
    return {
        "Q_beta_mol_m3": q_beta,
        "Q_gp_mol_m3": q_gp,
        "Q_matrix_mol_m3": q_matrix,
        "Q_total_mol_m3": q_total,
        "Q_reconstructed_mol_m3": q_reconstructed,
        "inventory_relative_residual": abs(residual) / max(abs(q_total), 1.0e-300),
    }


def _state_observation(
    source: CharacteristicReferenceSolver,
    *,
    edges_m: NDArray[np.float64],
    cells: NDArray[np.float64],
    frozen_xb: float,
) -> dict[str, float]:
    metrics = metrics_from_piecewise_constant_cells(edges_m, cells)
    return {
        "M0_m3": float(metrics.M0_m3),
        "M1_m2": float(metrics.M1_m2),
        "M2_m": float(metrics.M2_m),
        "M3_dimensionless": float(metrics.M3_dimensionless),
        "Rmean_m": float(metrics.Rmean_number_m),
        "Rmean3_m3": float(metrics.Rmean_cubed_m3),
        "Sv_m_inv": float(metrics.Sv_m_inv),
        "f_beta": float(metrics.f_beta),
        **_open_frozen_inventory(source, cells, frozen_xb),
    }


def _observable_difference(left: Mapping[str, float], right: Mapping[str, float]) -> dict[str, float]:
    result: dict[str, float] = {}
    for field in (
        "M0_m3",
        "M1_m2",
        "M2_m",
        "M3_dimensionless",
        "Rmean_m",
        "Rmean3_m3",
        "Sv_m_inv",
        "f_beta",
        "Q_beta_mol_m3",
        "Q_matrix_mol_m3",
        "Q_total_mol_m3",
    ):
        signed = float(left[field] - right[field])
        result[f"signed_{field}"] = signed
        result[f"absolute_{field}"] = abs(signed)
    return result


def _u0_arrays_and_metadata(
    source: CharacteristicReferenceSolver,
    *,
    checkpoint_binding: Mapping[str, Any],
) -> tuple[dict[str, NDArray[np.generic]], dict[str, Any], str]:
    edges = np.asarray(source.population("beta").grid.edges_m, dtype=np.float64).copy()
    cells = np.asarray(source._beta_cell_numbers(), dtype=np.float64).copy()
    frozen_xb = float(source.matrix_xb)
    observation = _state_observation(source, edges_m=edges, cells=cells, frozen_xb=frozen_xb)
    arrays: dict[str, NDArray[np.generic]] = {
        "radius_edges_m": edges,
        "cell_number_m3": cells,
        "matrix_xb": np.asarray([frozen_xb], dtype=np.float64),
        "step": np.asarray([int(source.step)], dtype=np.int64),
        "time_s": np.asarray([float(source.time_s)], dtype=np.float64),
        "M0_m3": np.asarray([observation["M0_m3"]], dtype=np.float64),
        "M1_m2": np.asarray([observation["M1_m2"]], dtype=np.float64),
        "M2_m": np.asarray([observation["M2_m"]], dtype=np.float64),
        "M3_dimensionless": np.asarray([observation["M3_dimensionless"]], dtype=np.float64),
        "Rmin_m": np.asarray([float(edges[0])], dtype=np.float64),
        "Rmax_m": np.asarray([float(edges[-1])], dtype=np.float64),
        "Q_beta_mol_m3": np.asarray([observation["Q_beta_mol_m3"]], dtype=np.float64),
        "Q_matrix_mol_m3": np.asarray([observation["Q_matrix_mol_m3"]], dtype=np.float64),
        "Q_total_mol_m3": np.asarray([observation["Q_total_mol_m3"]], dtype=np.float64),
    }
    metadata = {
        "schema": "KWN_FROZEN_SEMIGROUP_U0_V1",
        "accepted_state_hash": accepted_state_hash(source),
        "source_contract_hash": source.contract_hash,
        "source_config_hash": source.config.source_config_hash,
        "checkpoint_binding": dict(checkpoint_binding),
        "matrix_inventory_mode": "FROZEN_XB_OPEN_INVENTORY_DIAGNOSTIC",
    }
    digest_parts: dict[str, Any] = {**arrays, **metadata}
    return arrays, metadata, _content_hash(digest_parts)


def _save_and_repeat_load_u0(
    path: Path,
    *,
    arrays: Mapping[str, NDArray[np.generic]],
    metadata: Mapping[str, Any],
    expected_hash: str,
) -> dict[str, Any]:
    materialized = {key: np.asarray(value).copy() for key, value in arrays.items()}
    materialized["metadata_json"] = np.asarray(json.dumps(dict(metadata), sort_keys=True))
    np.savez(path, **materialized)
    loaded: list[dict[str, NDArray[np.generic]]] = []
    for _ in range(2):
        with np.load(path, allow_pickle=False) as archive:
            loaded.append({key: np.asarray(archive[key]).copy() for key in archive.files})
    if not _state_arrays_equal(loaded[0], loaded[1]):
        raise FrozenSemigroupWorkflowError("frozen_u0 repeat loads are not bitwise identical")
    if set(loaded[0]) != set(materialized) or not _state_arrays_equal(loaded[0], materialized):
        raise FrozenSemigroupWorkflowError("frozen_u0 repeat load differs from written state")
    rehash_parts: dict[str, Any] = {key: value for key, value in loaded[0].items() if key != "metadata_json"}
    rehash_parts.update(dict(metadata))
    observed_hash = _content_hash(rehash_parts)
    if observed_hash != expected_hash:
        raise FrozenSemigroupWorkflowError("frozen_u0 content hash differs after repeat load")
    return {
        "status": "PASS_BITWISE_REPEAT_LOAD",
        "frozen_u0_content_hash": observed_hash,
        "frozen_u0_file_sha256": _sha256_file(path),
        "repeat_load_count": 2,
        "array_key_count": len(materialized),
    }


def _load_exact_u0(restart_checkpoint: Path, output_root: Path) -> tuple[
    CharacteristicReferenceSolver,
    dict[str, Any],
    dict[str, NDArray[np.generic]],
    dict[str, Any],
]:
    if not restart_checkpoint.is_file():
        raise FrozenSemigroupWorkflowError("frozen step-244 restart checkpoint is unavailable")
    context = build_frozen_canonical_context()
    config, checkpoint_binding = prior.checkpoint_bound_config(
        context=context, restart_checkpoint=restart_checkpoint
    )
    first = CharacteristicReferenceSolver.load_checkpoint(config=config, path=restart_checkpoint)
    second = CharacteristicReferenceSolver.load_checkpoint(config=config, path=restart_checkpoint)
    context_edges = np.asarray(context.edges_m, dtype=np.float64)
    for solver in (first, second):
        if int(solver.step) != RESTART_STEP or accepted_state_hash(solver) != RESTART_STATE_HASH:
            raise FrozenSemigroupWorkflowError("restart checkpoint does not bind the exact frozen step-244 U0")
        if not np.array_equal(
            np.asarray(solver.population("beta").grid.edges_m, dtype=np.float64), context_edges
        ):
            raise FrozenSemigroupWorkflowError(
                "restart beta grid differs from the registered frozen canonical context grid"
            )
    if not _state_arrays_equal(first.state_arrays(), second.state_arrays()):
        raise FrozenSemigroupWorkflowError("checkpoint repeat loads are not bitwise identical")
    arrays, metadata, u0_hash = _u0_arrays_and_metadata(first, checkpoint_binding=checkpoint_binding)
    u0_repeat = _save_and_repeat_load_u0(
        output_root / "frozen_u0.npz", arrays=arrays, metadata=metadata, expected_hash=u0_hash
    )
    baseline = {
        "status": "PASS_FROZEN_U0_PRECHECK",
        "restart_checkpoint": str(restart_checkpoint),
        "restart_checkpoint_sha256": _sha256_file(restart_checkpoint),
        "restart_step": int(first.step),
        "restart_state_hash": accepted_state_hash(first),
        "frozen_u0_content_hash": u0_hash,
        "frozen_xB": float(first.matrix_xb),
        "checkpoint_repeat_load": "PASS_BITWISE",
        "frozen_u0_repeat_load": u0_repeat,
        "checkpoint_config_binding": checkpoint_binding,
        "frozen_canonical_context": frozen_canonical_snapshot_provenance(),
    }
    return first, baseline, arrays, metadata


def _state_parity(
    source: CharacteristicReferenceSolver,
    *,
    h_s: float,
    frozen_xb: float,
    source_cells: NDArray[np.float64],
    decomposition: FrozenSemigroupDecomposition,
) -> dict[str, Any]:
    """Verify A/C against the unchanged frozen CR1 disposable operator.

    ``phi_compose`` is the retained production-compatible frozen diagnostic.
    Exact bitwise identity is recorded, while pass/fail uses the explicitly
    declared float64-scale budget: the diagnostic module may perform equivalent
    reductions in a different, deterministic order.
    """

    direct = phi_compose(
        source, dt_s=float(h_s), count=1, mode="FROZEN_MATRIX", frozen_matrix_xb=float(frozen_xb)
    )
    sequential = phi_compose(
        source, dt_s=0.5 * float(h_s), count=2, mode="FROZEN_MATRIX", frozen_matrix_xb=float(frozen_xb)
    )
    scale_l1 = max(float(np.sum(np.abs(source_cells), dtype=np.float64)), 1.0e-300)
    scale_linf = max(float(np.max(np.abs(source_cells))), 1.0e-300)
    budget_l1 = PRODUCTION_PARITY_MACHINE_EPS_MULTIPLIER * np.finfo(np.float64).eps * scale_l1
    budget_linf = PRODUCTION_PARITY_MACHINE_EPS_MULTIPLIER * np.finfo(np.float64).eps * scale_linf
    if direct.status != "SUCCESS" or sequential.status != "SUCCESS" or direct.state is None or sequential.state is None:
        return {
            "status": "FAIL_PRODUCTION_PARITY_OPERATOR_NONCLOSING",
            "direct_status": direct.status,
            "sequential_status": sequential.status,
            "direct_error": direct.error_message,
            "sequential_error": sequential.error_message,
            "direct_equal": False,
            "sequential_equal": False,
            "direct_within_declared_float64_scale": False,
            "sequential_within_declared_float64_scale": False,
            "L1_abs_budget": budget_l1,
            "Linf_abs_budget": budget_linf,
        }
    direct_cells = np.asarray(direct.state["population_array"], dtype=np.float64)
    sequential_cells = np.asarray(sequential.state["population_array"], dtype=np.float64)
    if direct_cells.shape != source_cells.shape or sequential_cells.shape != source_cells.shape:
        return {
            "status": "FAIL_PRODUCTION_PARITY_SHAPE",
            "direct_status": direct.status,
            "sequential_status": sequential.status,
            "direct_equal": False,
            "sequential_equal": False,
            "direct_within_declared_float64_scale": False,
            "sequential_within_declared_float64_scale": False,
            "L1_abs_budget": budget_l1,
            "Linf_abs_budget": budget_linf,
        }
    direct_delta = np.asarray(direct_cells - decomposition.direct.cells, dtype=np.float64)
    sequential_delta = np.asarray(sequential_cells - decomposition.sequential.cells, dtype=np.float64)
    direct_l1 = float(np.sum(np.abs(direct_delta), dtype=np.float64))
    direct_linf = float(np.max(np.abs(direct_delta))) if direct_delta.size else 0.0
    sequential_l1 = float(np.sum(np.abs(sequential_delta), dtype=np.float64))
    sequential_linf = float(np.max(np.abs(sequential_delta))) if sequential_delta.size else 0.0
    direct_within = bool(direct_l1 <= budget_l1 and direct_linf <= budget_linf)
    sequential_within = bool(sequential_l1 <= budget_l1 and sequential_linf <= budget_linf)
    direct_equal = bool(np.array_equal(direct_cells, decomposition.direct.cells))
    sequential_equal = bool(np.array_equal(sequential_cells, decomposition.sequential.cells))
    return {
        "status": "PASS_PRODUCTION_PARITY" if direct_within and sequential_within else "FAIL_PRODUCTION_PARITY",
        "direct_status": direct.status,
        "sequential_status": sequential.status,
        "direct_equal": direct_equal,
        "sequential_equal": sequential_equal,
        "direct_within_declared_float64_scale": direct_within,
        "sequential_within_declared_float64_scale": sequential_within,
        "direct_L1_abs": direct_l1,
        "direct_Linf_abs": direct_linf,
        "sequential_L1_abs": sequential_l1,
        "sequential_Linf_abs": sequential_linf,
        "L1_abs_budget": budget_l1,
        "Linf_abs_budget": budget_linf,
        "budget_multiplier_float64_eps": PRODUCTION_PARITY_MACHINE_EPS_MULTIPLIER,
    }


def _machine_pass(metrics: Mapping[str, float], *, multiplier: float) -> tuple[bool, dict[str, float]]:
    l1_budget = float(multiplier * np.finfo(np.float64).eps)
    linf_budget = float(multiplier * np.finfo(np.float64).eps)
    return bool(float(metrics["L1_relative"]) <= l1_budget and float(metrics["Linf_relative"]) <= linf_budget), {
        "L1_relative_budget": l1_budget,
        "Linf_relative_budget": linf_budget,
    }


def _operator_machine_pass(
    audit: Mapping[str, Any], *, source_cells: NDArray[np.float64]
) -> tuple[bool, dict[str, float]]:
    scale_l1 = max(float(np.sum(np.abs(source_cells), dtype=np.float64)), 1.0e-300)
    scale_linf = max(float(np.max(np.abs(source_cells))), 1.0e-300)
    budget_l1 = OPERATOR_MACHINE_EPS_MULTIPLIER * np.finfo(np.float64).eps * scale_l1
    budget_linf = OPERATOR_MACHINE_EPS_MULTIPLIER * np.finfo(np.float64).eps * scale_linf
    residuals = [
        audit["P_h_state_residual"],
        audit["P_compflow_state_residual"],
        audit["P_sequential_state_residual"],
        audit["trace_operator_state_residual"],
        audit["remap_operator_state_residual"],
    ]
    passed = all(
        float(item["L1_abs"]) <= budget_l1 and float(item["Linf_abs"]) <= budget_linf
        for item in residuals
    )
    return bool(passed), {"L1_abs_budget": budget_l1, "Linf_abs_budget": budget_linf}


def _cosine_relation(value: float | None) -> str:
    if value is None:
        return "ZERO_DEFECT_VECTOR"
    if abs(value) <= NEAR_ORTHOGONAL_COSINE:
        return "NEARLY_ORTHOGONAL"
    return "REINFORCE" if value > 0.0 else "PARTIALLY_CANCEL"


def _trend(values: Sequence[float]) -> str:
    if not values or any(not math.isfinite(float(value)) for value in values):
        return "INSUFFICIENT_OR_NONFINITE"
    if all(float(right) < float(left) for left, right in zip(values, values[1:])):
        return "MONOTONE_DECREASING"
    if all(float(right) <= float(left) for left, right in zip(values, values[1:])):
        return "NONINCREASING"
    if all(float(right) >= float(left) for left, right in zip(values, values[1:])):
        return "NONDECREASING"
    return "NONMONOTONE_OR_STAGNANT"


def _support_is_enriched(pairing: Mapping[str, Any]) -> bool:
    """Use only geometry/support coverage, never a population-selection threshold."""

    total_cells = int(pairing["support_cell_count"]) + int(pairing["complement_cell_count"])
    if total_cells <= 0 or float(pairing["total_population_L1_abs"]) == 0.0:
        return False
    geometric_fraction = float(pairing["support_cell_count"]) / float(total_cells)
    return float(pairing["support_population_L1_fraction"]) > geometric_fraction


def _zero_event_causal_outcome(zero_control: Mapping[str, Any]) -> dict[str, Any]:
    """Classify the no-changed-face control without selecting a method.

    A zero-event topology does not by itself demonstrate that the flow-map
    defect vanished.  The frozen relative-L1 threshold is predeclared here so
    a later remap/CR2 conclusion cannot silently ignore material flow error
    that persists with no changed source-cell face.
    """

    threshold = (
        ZERO_EVENT_MACHINE_RELATIVE_L1_EPS_MULTIPLIER * np.finfo(np.float64).eps
    )
    if str(zero_control.get("status")) != "ZERO_EVENT_CONTROL_FOUND":
        return {
            "status": "ZERO_EVENT_CAUSAL_OUTCOME_UNAVAILABLE",
            "machine_relative_L1_threshold": threshold,
            "trace_population_relative_L1": None,
            "remap_population_relative_L1": None,
            "trace_machine_small": False,
            "remap_machine_small": False,
            "interpretation": "NO_CAUSAL_INFERENCE_WITHOUT_A_NO_CHANGED_FACE_CONTROL",
        }
    trace_metrics = zero_control.get("trace_metrics")
    remap_metrics = zero_control.get("remap_metrics")
    if not isinstance(trace_metrics, Mapping) or not isinstance(remap_metrics, Mapping):
        return {
            "status": "ZERO_EVENT_CAUSAL_OUTCOME_INVALID_METRICS",
            "machine_relative_L1_threshold": threshold,
            "trace_population_relative_L1": None,
            "remap_population_relative_L1": None,
            "trace_machine_small": False,
            "remap_machine_small": False,
            "interpretation": "NO_CAUSAL_INFERENCE_WITHOUT_FINITE_CONTROL_METRICS",
        }
    try:
        trace_relative = float(trace_metrics["population_relative_L1"])
        remap_relative = float(remap_metrics["population_relative_L1"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "ZERO_EVENT_CAUSAL_OUTCOME_INVALID_METRICS",
            "machine_relative_L1_threshold": threshold,
            "trace_population_relative_L1": None,
            "remap_population_relative_L1": None,
            "trace_machine_small": False,
            "remap_machine_small": False,
            "interpretation": "NO_CAUSAL_INFERENCE_WITHOUT_FINITE_CONTROL_METRICS",
        }
    if not math.isfinite(trace_relative) or not math.isfinite(remap_relative):
        return {
            "status": "ZERO_EVENT_CAUSAL_OUTCOME_INVALID_METRICS",
            "machine_relative_L1_threshold": threshold,
            "trace_population_relative_L1": trace_relative,
            "remap_population_relative_L1": remap_relative,
            "trace_machine_small": False,
            "remap_machine_small": False,
            "interpretation": "NO_CAUSAL_INFERENCE_WITHOUT_FINITE_CONTROL_METRICS",
        }
    trace_machine_small = trace_relative <= threshold
    remap_machine_small = remap_relative <= threshold
    if trace_machine_small and remap_machine_small:
        status = "ZERO_EVENT_TRACE_AND_REMAP_MACHINE_SMALL"
        interpretation = "NO_CHANGED_FACE_CONTROL_IS_MACHINE_SMALL_FOR_BOTH_DEFECTS"
    elif trace_machine_small:
        status = "ZERO_EVENT_TRACE_MACHINE_SMALL_REMAP_MATERIAL"
        interpretation = "INTERMEDIATE_REMAP_EFFECT_PERSISTS_WITHOUT_CHANGED_SOURCE_FACE"
    elif remap_machine_small:
        status = "ZERO_EVENT_TRACE_MATERIAL_REMAP_MACHINE_SMALL"
        interpretation = "FLOW_COMPOSITION_EFFECT_PERSISTS_WITHOUT_CHANGED_SOURCE_FACE"
    else:
        status = "ZERO_EVENT_TRACE_AND_REMAP_MATERIAL"
        interpretation = "BOTH_DEFECTS_PERSIST_WITHOUT_CHANGED_SOURCE_FACE"
    return {
        "status": status,
        "machine_relative_L1_threshold": threshold,
        "trace_population_relative_L1": trace_relative,
        "remap_population_relative_L1": remap_relative,
        "trace_machine_small": trace_machine_small,
        "remap_machine_small": remap_machine_small,
        "interpretation": interpretation,
    }


def _classification(
    rows: Sequence[Mapping[str, Any]],
    *,
    zero_control: Mapping[str, Any],
    synthetic_control: Mapping[str, Any],
) -> dict[str, Any]:
    """Apply transparent, deliberately fail-closed causal decision predicates."""

    zero_outcome = _zero_event_causal_outcome(zero_control)
    if not rows:
        return {
            "TOP_LEVEL_STATUS": "DIAG_INSUFFICIENT_FROZEN_OPERATOR_EVIDENCE",
            "PRIMARY_ROOT_CAUSE": "NO_SUCCESSFUL_FROZEN_THREE_PATH_LEVEL",
            "SECONDARY_ROOT_CAUSE": None,
            "gates": {
                "successful_h_levels": False,
                "zero_event_causal_outcome": zero_outcome,
            },
        }
    additive_ok = all(bool(row["additive_machine_pass"]) for row in rows)
    parity_ok = all(bool(row["production_parity_pass"]) for row in rows)
    operator_ok = all(bool(row["operator_state_parity_pass"]) for row in rows)
    if not additive_ok or not parity_ok or not operator_ok:
        return {
            "TOP_LEVEL_STATUS": "DIAG_SEMIGROUP_DECOMPOSITION_HARNESS_FAILURE",
            "PRIMARY_ROOT_CAUSE": "ADDITIVE_OR_OPERATOR_HARNESS_NONCLOSURE",
            "SECONDARY_ROOT_CAUSE": None,
            "gates": {
                "additive_machine_pass": additive_ok,
                "production_parity_pass": parity_ok,
                "operator_state_parity_pass": operator_ok,
                "zero_event_causal_outcome": zero_outcome,
            },
        }
    zero_found = str(zero_control.get("status")) == "ZERO_EVENT_CONTROL_FOUND"
    zero_trace_machine_small = bool(zero_outcome["trace_machine_small"])
    zero_remap_machine_small = bool(zero_outcome["remap_machine_small"])
    synthetic_found = str(synthetic_control.get("status")) == "SYNTHETIC_SINGLE_OR_FEW_EVENT_FOUND"
    synthetic_remap_pairing = bool(synthetic_control.get("remap_projection_support_enriched", False))
    nonzero_flow_rows = [row for row in rows if float(row["trace_population_L1_abs"]) > 0.0]
    nonzero_remap_rows = [row for row in rows if float(row["remap_population_L1_abs"]) > 0.0]
    nonzero_joint_rows = [
        row
        for row in rows
        if float(row["trace_population_L1_abs"]) > 0.0 or float(row["remap_population_L1_abs"]) > 0.0
    ]
    # Exactly zero defect levels are neutral to a localization predicate: there
    # is no physical error for a changed-support relation to explain.  Every
    # registered *nonzero* level must nevertheless pass its direct pairing.
    flow_pairing = bool(nonzero_flow_rows) and all(
        bool(row["flow_support_enriched"]) for row in nonzero_flow_rows
    )
    remap_pairing = bool(nonzero_remap_rows) and all(
        bool(row["projection_support_enriched"]) for row in nonzero_remap_rows
    )
    flow_nonzero = bool(nonzero_flow_rows)
    remap_nonzero = bool(nonzero_remap_rows)
    if (flow_nonzero and not flow_pairing) or (remap_nonzero and not remap_pairing):
        return {
            "TOP_LEVEL_STATUS": "DIAG_TOPOLOGY_EVENT_LINK_NOT_SUPPORTED",
            "PRIMARY_ROOT_CAUSE": "FROZEN_PHYSICAL_ERROR_NOT_ENRICHED_ON_ITS_DECLARED_CHANGED_SUPPORT",
            "SECONDARY_ROOT_CAUSE": None,
            "gates": {
                "flow_support_enriched": flow_pairing,
                "projection_support_enriched": remap_pairing,
                "support_pairing_rule": "ALL_REGISTERED_NONZERO_LEVELS",
                "zero_defect_levels_neutral": True,
                "zero_event_control_found": zero_found,
                "zero_event_causal_outcome": zero_outcome,
                "synthetic_control_found": synthetic_found,
                "synthetic_remap_projection_support_enriched": synthetic_remap_pairing,
            },
        }
    if not zero_found or not synthetic_found:
        return {
            "TOP_LEVEL_STATUS": "DIAG_INSUFFICIENT_FROZEN_OPERATOR_EVIDENCE",
            "PRIMARY_ROOT_CAUSE": "REQUIRED_ZERO_OR_SYNTHETIC_CONTROL_UNAVAILABLE",
            "SECONDARY_ROOT_CAUSE": None,
            "gates": {
                "zero_event_control_found": zero_found,
                "zero_event_causal_outcome": zero_outcome,
                "synthetic_control_found": synthetic_found,
                "flow_support_enriched": flow_pairing,
                "projection_support_enriched": remap_pairing,
                "synthetic_remap_projection_support_enriched": synthetic_remap_pairing,
                "support_pairing_rule": "ALL_REGISTERED_NONZERO_LEVELS",
                "zero_defect_levels_neutral": True,
            },
        }
    trace_dominant = bool(nonzero_joint_rows) and all(
        float(row["trace_population_L1_abs"])
        > CAUSAL_DOMINANCE_RATIO * float(row["remap_population_L1_abs"])
        for row in nonzero_joint_rows
    )
    remap_dominant = bool(nonzero_joint_rows) and all(
        float(row["remap_population_L1_abs"])
        > CAUSAL_DOMINANCE_RATIO * float(row["trace_population_L1_abs"])
        for row in nonzero_joint_rows
    )
    common_gates = {
        "flow_support_enriched_all_registered_nonzero_levels": flow_pairing,
        "projection_support_enriched_all_registered_nonzero_levels": remap_pairing,
        "support_pairing_rule": "ALL_REGISTERED_NONZERO_LEVELS",
        "zero_defect_levels_neutral": True,
        "causal_dominance_ratio": CAUSAL_DOMINANCE_RATIO,
        "zero_event_control_found": zero_found,
        "zero_event_causal_outcome": zero_outcome,
        "zero_event_trace_machine_small": zero_trace_machine_small,
        "zero_event_remap_machine_small": zero_remap_machine_small,
        "synthetic_control_found": synthetic_found,
        "synthetic_remap_projection_support_enriched": synthetic_remap_pairing,
    }
    if remap_dominant and remap_pairing and synthetic_remap_pairing and not zero_trace_machine_small:
        return {
            "TOP_LEVEL_STATUS": "DIAG_TOPOLOGY_EVENT_LINK_NOT_SUPPORTED",
            "PRIMARY_ROOT_CAUSE": "MATERIAL_FLOW_DEFECT_PERSISTS_WITHOUT_CHANGED_SOURCE_FACE",
            "SECONDARY_ROOT_CAUSE": "ONE_SIDED_INTERMEDIATE_REMAP_OR_CR2_CLAIM_BLOCKED",
            "gates": {
                **common_gates,
                "remap_dominant_at_all_registered_nonzero_levels": True,
                "zero_event_trace_machine_small_required_for_one_sided_remap_claim": False,
            },
        }
    if remap_dominant and remap_pairing and synthetic_remap_pairing and zero_trace_machine_small:
        return {
            "TOP_LEVEL_STATUS": "DIAG_CR1_INTERMEDIATE_REMAP_DEFECT_SUPPORTED",
            "PRIMARY_ROOT_CAUSE": "INTERMEDIATE_CR1_REMAP_PROJECTION",
            "SECONDARY_ROOT_CAUSE": "CHARACTERISTIC_FLOW_COMPOSITION_SUBORDINATE",
            "gates": {
                **common_gates,
                "remap_dominant_at_all_registered_nonzero_levels": True,
                "zero_event_trace_machine_small_required_for_one_sided_remap_claim": True,
            },
        }
    if trace_dominant and flow_pairing:
        return {
            "TOP_LEVEL_STATUS": "DIAG_CHARACTERISTIC_FLOW_COMPOSITION_DEFECT_SUPPORTED",
            "PRIMARY_ROOT_CAUSE": "CHARACTERISTIC_FLOW_MAP_COMPOSITION",
            "SECONDARY_ROOT_CAUSE": "INTERMEDIATE_CR1_REMAP_PROJECTION_SUBORDINATE",
            "gates": {**common_gates, "trace_dominant_at_all_registered_nonzero_levels": True},
        }
    if flow_nonzero and remap_nonzero and flow_pairing and remap_pairing:
        return {
            "TOP_LEVEL_STATUS": "DIAG_MIXED_FLOW_AND_REMAP_DEFECT",
            "PRIMARY_ROOT_CAUSE": "MIXED_CHARACTERISTIC_FLOW_AND_INTERMEDIATE_REMAP",
            "SECONDARY_ROOT_CAUSE": None,
            "gates": common_gates,
        }
    return {
        "TOP_LEVEL_STATUS": "DIAG_INSUFFICIENT_FROZEN_OPERATOR_EVIDENCE",
        "PRIMARY_ROOT_CAUSE": "FROZEN_DECOMPOSITION_DOES_NOT_MEET_A_SINGLE_FAIL_CLOSED_CAUSAL_PATTERN",
        "SECONDARY_ROOT_CAUSE": None,
        "gates": common_gates,
    }


def _method_decision(classification: Mapping[str, Any]) -> dict[str, Any]:
    status = str(classification["TOP_LEVEL_STATUS"])
    if status == "DIAG_CR1_INTERMEDIATE_REMAP_DEFECT_SUPPORTED":
        return {
            "CR2_AUTHORIZED": "ELIGIBLE_FOR_SEPARATELY_GATED_CR2_PROTOTYPE",
            "TRACE_INTEGRATOR_AUDIT_AUTHORIZED": False,
            "METHOD_CHANGE_AUTHORIZATION": "NO_IMPLEMENTATION_IN_THIS_TASK",
        }
    if status == "DIAG_CHARACTERISTIC_FLOW_COMPOSITION_DEFECT_SUPPORTED":
        return {
            "CR2_AUTHORIZED": False,
            "TRACE_INTEGRATOR_AUDIT_AUTHORIZED": "CHARACTERISTIC_TRACE_INTEGRATOR_AUDIT",
            "METHOD_CHANGE_AUTHORIZATION": "NO_IMPLEMENTATION_IN_THIS_TASK",
        }
    if status == "DIAG_MIXED_FLOW_AND_REMAP_DEFECT":
        return {
            "CR2_AUTHORIZED": False,
            "TRACE_INTEGRATOR_AUDIT_AUTHORIZED": False,
            "METHOD_CHANGE_AUTHORIZATION": "MANUAL_MAGNITUDE_AND_SCALING_REVIEW_REQUIRED",
        }
    return {
        "CR2_AUTHORIZED": False,
        "TRACE_INTEGRATOR_AUDIT_AUTHORIZED": False,
        "METHOD_CHANGE_AUTHORIZATION": "NO_METHOD_CHANGE_AUTHORIZED",
    }


def _state_npz_arrays(
    source_arrays: Mapping[str, NDArray[np.generic]],
    decompositions: Sequence[Mapping[str, Any]],
) -> dict[str, NDArray[np.generic]]:
    result = {f"u0_{key}": np.asarray(value).copy() for key, value in source_arrays.items()}
    for index, item in enumerate(decompositions):
        label = f"level_{index:02d}_h_{float(item['h_s']):.17g}".replace(".", "p").replace("-", "m")
        decomposition = item["decomposition"]
        if not isinstance(decomposition, FrozenSemigroupDecomposition):
            continue
        result.update({
            f"{label}_U_A": decomposition.direct.cells,
            f"{label}_U_B": decomposition.composed_flow.cells,
            f"{label}_U_C": decomposition.sequential.cells,
            f"{label}_D_total": decomposition.d_total,
            f"{label}_D_trace": decomposition.d_trace,
            f"{label}_D_remap": decomposition.d_remap,
            f"{label}_additive_residual": decomposition.additive_residual,
            f"{label}_direct_departure_faces_m": decomposition.direct.trace.departure_faces_m,
            f"{label}_composed_departure_faces_m": decomposition.composed_flow.trace.departure_faces_m,
            f"{label}_first_half_departure_faces_m": decomposition.sequential.first_half_trace.departure_faces_m,
            f"{label}_second_half_grid_departure_faces_m": decomposition.sequential.second_half_trace.departure_faces_m,
            f"{label}_sequential_intermediate_cells": decomposition.sequential.intermediate_cells,
        })
    return result


def _run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    restart_checkpoint = Path(args.restart_checkpoint)
    if output_root.exists() or report_root.exists():
        raise FrozenSemigroupWorkflowError("refusing to overwrite existing output/report roots")
    source_identity = _source_identity()
    inherited_evidence_inputs = _validate_inherited_evidence_inputs(args)
    output_root.mkdir(parents=True)
    report_root.mkdir(parents=True)
    (output_root / "figures").mkdir()
    started = time.monotonic()

    source, baseline, u0_arrays, u0_metadata = _load_exact_u0(restart_checkpoint, output_root)
    baseline["inherited_evidence_inputs"] = inherited_evidence_inputs
    before_source_arrays = {key: np.asarray(value).copy() for key, value in source.state_arrays().items()}
    before_source_history = tuple(source.history)
    edges = np.asarray(u0_arrays["radius_edges_m"], dtype=np.float64)
    u0_cells = np.asarray(u0_arrays["cell_number_m3"], dtype=np.float64)
    frozen_xb = float(u0_arrays["matrix_xb"][0])
    source_cells = np.asarray(source._beta_cell_numbers(), dtype=np.float64).copy()
    if not np.array_equal(u0_cells, source_cells):
        raise FrozenSemigroupWorkflowError(
            "frozen U0 cell numbers differ from the source beta-cell population"
        )

    def velocity(radii_m: NDArray[np.float64]) -> NDArray[np.float64]:
        return np.asarray(source._velocity_at_radii(np.asarray(radii_m, dtype=np.float64), frozen_xb), dtype=np.float64)

    levels: list[dict[str, Any]] = []
    additive_rows: list[dict[str, Any]] = []
    face_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    pairing_rows: list[dict[str, Any]] = []
    expansion_rows: list[dict[str, Any]] = []
    operator_rows: list[dict[str, Any]] = []
    all_successful = True
    for divisor in SEMIGROUP_DIVISORS:
        h_s = DT0_S / float(divisor)
        try:
            decomposition = decompose_frozen_cr1(
                edges_m=edges, initial_cells=u0_cells, h_s=h_s, velocity_m_s=velocity
            )
            parity = _state_parity(
                source,
                h_s=h_s,
                frozen_xb=frozen_xb,
                source_cells=source_cells,
                decomposition=decomposition,
            )
            additive = additive_residual_metrics(
                decomposition.additive_residual, reference_cells=u0_cells
            )
            additive_pass, additive_budget = _machine_pass(
                additive, multiplier=ADDITIVE_MACHINE_EPS_MULTIPLIER
            )
            audit = operator_audit(decomposition, initial_cells=u0_cells)
            operator_pass, operator_budget = _operator_machine_pass(audit, source_cells=u0_cells)
            direct_observation = _state_observation(
                source, edges_m=edges, cells=decomposition.direct.cells, frozen_xb=frozen_xb
            )
            composed_observation = _state_observation(
                source, edges_m=edges, cells=decomposition.composed_flow.cells, frozen_xb=frozen_xb
            )
            sequential_observation = _state_observation(
                source, edges_m=edges, cells=decomposition.sequential.cells, frozen_xb=frozen_xb
            )
            total_metrics = defect_metrics(decomposition.sequential.cells, decomposition.direct.cells, edges_m=edges)
            trace_metrics = defect_metrics(decomposition.composed_flow.cells, decomposition.direct.cells, edges_m=edges)
            remap_metrics = defect_metrics(decomposition.sequential.cells, decomposition.composed_flow.cells, edges_m=edges)
            changed_faces = changed_face_indices(decomposition)
            flow_support = support_from_faces(changed_faces, cell_count=u0_cells.size, halo=0)
            projection_for_h, projection_changed_cells = projection_connectivity_rows(decomposition)
            projection_rows.extend(projection_for_h)
            trace_pairing = support_pairing_metrics(
                decomposition.d_trace, edges_m=edges, support=flow_support
            )
            projection_support = np.zeros(u0_cells.size, dtype=bool)
            projection_support[projection_changed_cells] = True
            remap_pairing = support_pairing_metrics(
                decomposition.d_remap, edges_m=edges, support=projection_support
            )
            pairing_rows.extend([
                {
                    "h_s": h_s,
                    "defect": "D_trace",
                    "support_kind": "S_flow_changed_destination_support",
                    "changed_faces_json": json.dumps([int(value) for value in changed_faces]),
                    "support_cells_json": json.dumps([int(value) for value in np.flatnonzero(flow_support)]),
                    **trace_pairing,
                },
                {
                    "h_s": h_s,
                    "defect": "D_remap",
                    "support_kind": "S_projection_changed_cells",
                    "changed_faces_json": "[]",
                    "support_cells_json": json.dumps([int(value) for value in projection_changed_cells]),
                    **remap_pairing,
                },
            ])
            expansion_rows.extend(support_expansion_rows(
                decomposition.d_trace,
                edges_m=edges,
                seed_cells=np.flatnonzero(flow_support),
                kind="D_trace_S_flow_changed_destination_support",
                h_s=h_s,
            ))
            expansion_rows.extend(support_expansion_rows(
                decomposition.d_remap,
                edges_m=edges,
                seed_cells=projection_changed_cells,
                kind="D_remap_S_projection_changed_cells",
                h_s=h_s,
            ))
            faces = face_flow_rows(decomposition, edges_m=edges)
            face_rows.extend(faces)
            cosine = defect_cosine(decomposition.d_trace, decomposition.d_remap)
            trace_l1 = float(trace_metrics["population_L1_abs"])
            remap_l1 = float(remap_metrics["population_L1_abs"])
            total_l1 = float(total_metrics["population_L1_abs"])
            level = {
                "h_s": h_s,
                "divisor": int(divisor),
                "status": "SUCCESS",
                "decomposition": decomposition,
                "additive": additive,
                "additive_budget": additive_budget,
                "additive_machine_pass": additive_pass,
                "parity": parity,
                "production_parity_pass": str(parity["status"]) == "PASS_PRODUCTION_PARITY",
                "operator_audit": audit,
                "operator_budget": operator_budget,
                "operator_state_parity_pass": operator_pass,
                "direct_observation": direct_observation,
                "composed_observation": composed_observation,
                "sequential_observation": sequential_observation,
                "total_metrics": total_metrics,
                "trace_metrics": trace_metrics,
                "remap_metrics": remap_metrics,
                "trace_population_L1_abs": trace_l1,
                "remap_population_L1_abs": remap_l1,
                "total_population_L1_abs": total_l1,
                "trace_to_total_norm_ratio": ratio(trace_l1, total_l1),
                "remap_to_total_norm_ratio": ratio(remap_l1, total_l1),
                "trace_to_remap_ratio": ratio(trace_l1, remap_l1),
                "trace_remap_cosine": cosine,
                "trace_remap_relation": _cosine_relation(cosine),
                "changed_faces": changed_faces,
                "flow_changed_face_count": int(changed_faces.size),
                "flow_support": flow_support,
                "projection_changed_cells": projection_changed_cells,
                "projection_changed_cell_count": int(projection_changed_cells.size),
                "trace_pairing": trace_pairing,
                "remap_pairing": remap_pairing,
                "flow_support_enriched": _support_is_enriched(trace_pairing),
                "projection_support_enriched": _support_is_enriched(remap_pairing),
            }
            levels.append(level)
            additive_rows.append({
                "h_s": h_s,
                "divisor": int(divisor),
                "status": "SUCCESS",
                "additive_machine_pass": additive_pass,
                "production_parity_pass": level["production_parity_pass"],
                "operator_state_parity_pass": operator_pass,
                "flow_changed_face_count": int(changed_faces.size),
                "projection_changed_cell_count": int(projection_changed_cells.size),
                "trace_to_total_norm_ratio": level["trace_to_total_norm_ratio"],
                "remap_to_total_norm_ratio": level["remap_to_total_norm_ratio"],
                "trace_to_remap_ratio": level["trace_to_remap_ratio"],
                "trace_remap_cosine": cosine,
                "trace_remap_relation": level["trace_remap_relation"],
                **{f"additive_{key}": value for key, value in additive.items()},
                **{f"additive_budget_{key}": value for key, value in additive_budget.items()},
                **{f"total_{key}": value for key, value in total_metrics.items()},
                **{f"trace_{key}": value for key, value in trace_metrics.items()},
                **{f"remap_{key}": value for key, value in remap_metrics.items()},
                **{f"total_observable_{key}": value for key, value in _observable_difference(sequential_observation, direct_observation).items()},
                **{f"trace_observable_{key}": value for key, value in _observable_difference(composed_observation, direct_observation).items()},
                **{f"remap_observable_{key}": value for key, value in _observable_difference(sequential_observation, composed_observation).items()},
            })
            operator_rows.append({
                "h_s": h_s,
                "divisor": int(divisor),
                "operator_state_parity_pass": operator_pass,
                **operator_budget,
                **audit,
            })
        except (FrozenSemigroupDecompositionError, FrozenSemigroupWorkflowError, ValueError, FloatingPointError) as error:
            all_successful = False
            levels.append({"h_s": h_s, "divisor": int(divisor), "status": "FAILURE", "error": f"{type(error).__name__}: {error}"})
            additive_rows.append({"h_s": h_s, "divisor": int(divisor), "status": "FAILURE", "error": f"{type(error).__name__}: {error}"})

    if not _state_arrays_equal(before_source_arrays, source.state_arrays()) or tuple(source.history) != before_source_history:
        raise FrozenSemigroupWorkflowError("frozen decomposition diagnostic mutated the accepted step-244 source")

    successful_levels = [item for item in levels if item["status"] == "SUCCESS"]
    _write_csv(output_root / "additive_error_decomposition.csv", additive_rows, ("h_s", "status"))
    _write_csv(output_root / "face_flow_semigroup_error.csv", face_rows, ("h_s", "face_index"))
    _write_csv(output_root / "projection_connectivity_difference.csv", projection_rows, ("h_s", "final_destination_cell"))
    _write_csv(output_root / "frozen_error_event_pairing.csv", pairing_rows, ("h_s", "defect"))
    _write_csv(output_root / "frozen_error_support_expansion.csv", expansion_rows, ("h_s", "defect", "halo_cells"))
    _write_csv(output_root / "operator_norms.csv", operator_rows, ("h_s", "operator_state_parity_pass"))
    np.savez(output_root / "three_path_states.npz", **_state_npz_arrays(u0_arrays, successful_levels))

    zero_rows: list[dict[str, Any]] = []
    zero_summary: dict[str, Any]
    if successful_levels:
        minimum_h = min(float(item["h_s"]) for item in successful_levels)
        zero_decomposition, zero_rows_raw = search_zero_event_control(
            edges_m=edges,
            initial_cells=u0_cells,
            velocity_m_s=velocity,
            starting_h_s=minimum_h,
        )
        zero_rows.extend(dict(row) for row in zero_rows_raw)
        if zero_decomposition is None:
            zero_summary = {"status": "ZERO_EVENT_CONTROL_NOT_FOUND", "starting_h_s": minimum_h}
        else:
            zero_trace = defect_metrics(
                zero_decomposition.composed_flow.cells, zero_decomposition.direct.cells, edges_m=edges
            )
            zero_remap = defect_metrics(
                zero_decomposition.sequential.cells, zero_decomposition.composed_flow.cells, edges_m=edges
            )
            zero_summary = {
                "status": "ZERO_EVENT_CONTROL_FOUND",
                "h_s": float(zero_decomposition.h_s),
                "changed_face_count": int(changed_face_indices(zero_decomposition).size),
                "trace_metrics": zero_trace,
                "remap_metrics": zero_remap,
            }
            zero_rows.append({
                "h_s": float(zero_decomposition.h_s),
                "halving": "SELECTED",
                "status": "ZERO_EVENT_CONTROL_FOUND",
                "changed_face_count": 0,
                **{f"trace_{key}": value for key, value in zero_trace.items()},
                **{f"remap_{key}": value for key, value in zero_remap.items()},
            })
    else:
        zero_summary = {"status": "ZERO_EVENT_CONTROL_NOT_ATTEMPTED_NO_SUCCESSFUL_REGISTERED_LEVEL"}
    zero_summary["causal_outcome"] = _zero_event_causal_outcome(zero_summary)
    _write_csv(output_root / "zero_event_control.csv", zero_rows, ("h_s", "status", "changed_face_count"))

    synthetic_rows: list[dict[str, Any]] = []
    synthetic_decomposition, synthetic_summary = synthetic_single_event_control()
    if synthetic_decomposition is not None:
        synthetic_faces = changed_face_indices(synthetic_decomposition)
        synthetic_flow_support = support_from_faces(
            synthetic_faces, cell_count=synthetic_decomposition.direct.cells.size, halo=0
        )
        synthetic_projection_rows, synthetic_projection_cells = projection_connectivity_rows(synthetic_decomposition)
        synthetic_trace = defect_metrics(
            synthetic_decomposition.composed_flow.cells, synthetic_decomposition.direct.cells,
            edges_m=synthetic_decomposition.direct.trace.arrival_faces_m,
        )
        synthetic_remap = defect_metrics(
            synthetic_decomposition.sequential.cells, synthetic_decomposition.composed_flow.cells,
            edges_m=synthetic_decomposition.direct.trace.arrival_faces_m,
        )
        synthetic_pair = support_pairing_metrics(
            synthetic_decomposition.d_remap,
            edges_m=synthetic_decomposition.direct.trace.arrival_faces_m,
            support=np.isin(np.arange(synthetic_decomposition.direct.cells.size), synthetic_projection_cells),
        )
        synthetic_summary.update({
            "control_provenance_scope": (
                "DIAGNOSTIC_CR1_OPERATOR_CONTROL_ONLY; "
                "NOT_PBTE_MATERIAL_STATE_EVIDENCE"
            ),
            "physical_material_state_used": bool(
                synthetic_summary.get("physical_material_state_used", False)
            ),
            "trace_metrics": synthetic_trace,
            "remap_metrics": synthetic_remap,
            "flow_changed_face_count": int(synthetic_faces.size),
            "projection_changed_cell_count": int(synthetic_projection_cells.size),
            "flow_support_cell_count": int(np.count_nonzero(synthetic_flow_support)),
            "remap_projection_support_enriched": _support_is_enriched(synthetic_pair),
            "projection_connectivity_row_count": len(synthetic_projection_rows),
        })
        synthetic_rows.append({
            **synthetic_summary,
            **{f"trace_{key}": value for key, value in synthetic_trace.items()},
            **{f"remap_{key}": value for key, value in synthetic_remap.items()},
            **{f"remap_pairing_{key}": value for key, value in synthetic_pair.items()},
        })
    else:
        synthetic_rows.append(dict(synthetic_summary))
    _write_csv(output_root / "single_event_control.csv", synthetic_rows, ("status", "selected_h_s", "changed_face_count"))

    refinement_rows: list[dict[str, Any]] = []
    for item in levels:
        if item["status"] != "SUCCESS":
            refinement_rows.append({"h_s": item["h_s"], "divisor": item["divisor"], "status": item["status"], "error": item["error"]})
            continue
        refinement_rows.append({
            "h_s": item["h_s"],
            "divisor": item["divisor"],
            "status": "SUCCESS",
            "total_population_L1_abs": item["total_population_L1_abs"],
            "trace_population_L1_abs": item["trace_population_L1_abs"],
            "remap_population_L1_abs": item["remap_population_L1_abs"],
            "flow_changed_face_count": item["flow_changed_face_count"],
            "projection_changed_cell_count": item["projection_changed_cell_count"],
            "trace_to_remap_ratio": item["trace_to_remap_ratio"],
            "trace_remap_cosine": item["trace_remap_cosine"],
        })
    _write_csv(output_root / "h_refinement_decomposition.csv", refinement_rows, ("h_s", "status"))

    if all_successful and len(successful_levels) == len(SEMIGROUP_DIVISORS):
        classification = _classification(
            successful_levels, zero_control=zero_summary, synthetic_control=synthetic_summary
        )
    else:
        classification = {
            "TOP_LEVEL_STATUS": "DIAG_INSUFFICIENT_FROZEN_OPERATOR_EVIDENCE",
            "PRIMARY_ROOT_CAUSE": "ONE_OR_MORE_REGISTERED_FROZEN_THREE_PATH_LEVELS_FAILED",
            "SECONDARY_ROOT_CAUSE": None,
            "gates": {"all_registered_h_levels_successful": False},
        }
    if classification["TOP_LEVEL_STATUS"] not in ALLOWED_TOP_LEVEL_STATUSES:
        raise FrozenSemigroupWorkflowError("classification produced a forbidden top-level status")
    decision = _method_decision(classification)
    _write_json(output_root / "causal_classification.json", classification)
    _write_json(output_root / "method_decision.json", decision)

    trace_trend = _trend([float(item["trace_population_L1_abs"]) for item in successful_levels])
    remap_trend = _trend([float(item["remap_population_L1_abs"]) for item in successful_levels])
    first = successful_levels[0] if successful_levels else None
    first_three = successful_levels[:3]
    face_delta = [abs(float(row["delta_departure_radius_m"])) for row in face_rows]
    final = {
        "STATUS": classification["TOP_LEVEL_STATUS"],
        "BRANCH": source_identity["git_branch"],
        "COMMIT": source_identity["git_commit"],
        "SOURCE_CLEAN": True,
        "TESTS": str(args.test_status),
        "FROZEN_U0_HASH": baseline["frozen_u0_content_hash"],
        "THREE_PATHS_DEFINED": "A_DIRECT__B_COMPOSED_FLOW_SINGLE_REMAP__C_SEQUENTIAL_HALF_REMAPS",
        "ADDITIVE_DECOMPOSITION": "CELLWISE_D_TOTAL_EQUALS_D_TRACE_PLUS_D_REMAP",
        "DECOMPOSITION_L1_RESIDUAL": None if first is None else first["additive"]["L1_abs"],
        "DECOMPOSITION_LINF_RESIDUAL": None if first is None else first["additive"]["Linf_abs"],
        "TOTAL_DEFECT_H": None if len(first_three) < 1 else first_three[0]["total_population_L1_abs"],
        "TRACE_DEFECT_H": None if len(first_three) < 1 else first_three[0]["trace_population_L1_abs"],
        "REMAP_DEFECT_H": None if len(first_three) < 1 else first_three[0]["remap_population_L1_abs"],
        "TOTAL_DEFECT_H2": None if len(first_three) < 2 else first_three[1]["total_population_L1_abs"],
        "TRACE_DEFECT_H2": None if len(first_three) < 2 else first_three[1]["trace_population_L1_abs"],
        "REMAP_DEFECT_H2": None if len(first_three) < 2 else first_three[1]["remap_population_L1_abs"],
        "TOTAL_DEFECT_H4": None if len(first_three) < 3 else first_three[2]["total_population_L1_abs"],
        "TRACE_DEFECT_H4": None if len(first_three) < 3 else first_three[2]["trace_population_L1_abs"],
        "REMAP_DEFECT_H4": None if len(first_three) < 3 else first_three[2]["remap_population_L1_abs"],
        "TRACE_TO_REMAP_RATIO_H": None if len(first_three) < 1 else first_three[0]["trace_to_remap_ratio"],
        "TRACE_TO_REMAP_RATIO_H2": None if len(first_three) < 2 else first_three[1]["trace_to_remap_ratio"],
        "TRACE_TO_REMAP_RATIO_H4": None if len(first_three) < 3 else first_three[2]["trace_to_remap_ratio"],
        "TRACE_REMAP_COSINE_H": None if len(first_three) < 1 else first_three[0]["trace_remap_cosine"],
        "TRACE_REMAP_COSINE_H2": None if len(first_three) < 2 else first_three[1]["trace_remap_cosine"],
        "TRACE_REMAP_COSINE_H4": None if len(first_three) < 3 else first_three[2]["trace_remap_cosine"],
        "FACE_MAP_MAX_ERROR": max(face_delta) if face_delta else None,
        "FACE_MAP_RMS_ERROR": math.sqrt(float(np.mean(np.square(face_delta)))) if face_delta else None,
        "FLOW_CHANGED_FACES": sum(int(item["flow_changed_face_count"]) for item in successful_levels),
        "PROJECTION_CHANGED_CELLS": sum(int(item["projection_changed_cell_count"]) for item in successful_levels),
        "TRACE_ERROR_FRACTION_ON_FLOW_CHANGED_SUPPORT": None if first is None else first["trace_pairing"]["support_population_L1_fraction"],
        "REMAP_ERROR_FRACTION_ON_PROJECTION_CHANGED_SUPPORT": None if first is None else first["remap_pairing"]["support_population_L1_fraction"],
        "HALO0_ERROR_FRACTION": next((row["support_population_L1_fraction"] for row in expansion_rows if row["defect"] == "D_trace_S_flow_changed_destination_support" and row["halo_cells"] == 0), None),
        "HALO1_ERROR_FRACTION": next((row["support_population_L1_fraction"] for row in expansion_rows if row["defect"] == "D_trace_S_flow_changed_destination_support" and row["halo_cells"] == 1), None),
        "HALO2_ERROR_FRACTION": next((row["support_population_L1_fraction"] for row in expansion_rows if row["defect"] == "D_trace_S_flow_changed_destination_support" and row["halo_cells"] == 2), None),
        "SPARSE_OPERATOR_AUDIT": "PASS_MACHINE_PRECISION" if all(bool(item["operator_state_parity_pass"]) for item in successful_levels) else "FAIL_OR_INCOMPLETE",
        "OPERATOR_TRACE_NORM": None if first is None else first["operator_audit"]["operator_trace_1_norm"],
        "OPERATOR_REMAP_NORM": None if first is None else first["operator_audit"]["operator_remap_1_norm"],
        "ZERO_EVENT_CONTROL": zero_summary["status"],
        "ZERO_EVENT_CAUSAL_OUTCOME": zero_summary["causal_outcome"]["status"],
        "ZERO_EVENT_MACHINE_RELATIVE_L1_THRESHOLD": zero_summary["causal_outcome"][
            "machine_relative_L1_threshold"
        ],
        "ZERO_EVENT_TRACE_MACHINE_SMALL": zero_summary["causal_outcome"][
            "trace_machine_small"
        ],
        "ZERO_EVENT_REMAP_MACHINE_SMALL": zero_summary["causal_outcome"][
            "remap_machine_small"
        ],
        "ZERO_EVENT_TRACE_DEFECT": zero_summary.get("trace_metrics", {}).get("population_L1_abs"),
        "ZERO_EVENT_REMAP_DEFECT": zero_summary.get("remap_metrics", {}).get("population_L1_abs"),
        "SINGLE_EVENT_CONTROL": synthetic_summary.get("status"),
        "SINGLE_EVENT_CAUSALITY": synthetic_summary.get("remap_projection_support_enriched", False),
        "SINGLE_EVENT_CONTROL_SCOPE": synthetic_summary.get("control_provenance_scope"),
        "H_REFINEMENT_TRACE_TREND": trace_trend,
        "H_REFINEMENT_REMAP_TREND": remap_trend,
        "PRIMARY_ROOT_CAUSE": classification["PRIMARY_ROOT_CAUSE"],
        "SECONDARY_ROOT_CAUSE": classification["SECONDARY_ROOT_CAUSE"],
        **decision,
        "TIME_REFERENCE_V2": "NOT_ASSIGNED",
        "M16_AUTHORITY": "NOT_AUTHORITY",
        "PF_SOURCE_MODIFIED": False,
        "CUDA_RERUN": False,
        "PHYSICAL_RETUNING": False,
        "GP_RELEASE_RUN": False,
        "TOP_5_FINDINGS": [
            f"Frozen A/B/C successful levels: {len(successful_levels)}/{len(SEMIGROUP_DIVISORS)}.",
            f"Additive decomposition machine-closure: {all(bool(item['additive_machine_pass']) for item in successful_levels)}.",
            f"Trace h-refinement trend: {trace_trend}.",
            f"Remap h-refinement trend: {remap_trend}.",
            f"Frozen causal classification: {classification['TOP_LEVEL_STATUS']}.",
        ],
        "P0_BLOCKERS": ["TIME_REFERENCE_V2_NOT_ASSIGNED", classification["TOP_LEVEL_STATUS"]],
        "NEXT_ACTION": "Await explicit authorization before any separately gated method prototype or time-reference work.",
        "KEY_REPORTS": [
            f"reports/{TASK_NAME}/03_additive_error_decomposition.md",
            f"reports/{TASK_NAME}/05_intermediate_projection_connectivity.md",
            f"reports/{TASK_NAME}/12_causal_classification.md",
            f"reports/{TASK_NAME}/14_final_acceptance_report.md",
        ],
    }

    reports = {
        "00_baseline.md": {**baseline, "source_identity": source_identity},
        "01_frozen_operator_contract.md": {
            "matrix_xB": frozen_xb,
            "autonomous_velocity": "G(R)=shared_beta_growth_rate(R,xB0,T)",
            "dynamic_matrix_path_run": False,
            "prescribed_x_path_run": False,
            "production_solver_modified": False,
            "checkpoint_or_trajectory_write": False,
        },
        "02_three_path_definition.md": {
            "A": "R[F_h] U0",
            "B": "R[F_h_over_2 composed with F_h_over_2] U0; one final remap only",
            "C": "R[F_h_over_2] R[F_h_over_2] U0",
            "B_intermediate_remap": False,
            "B_physical_radius_continuation": True,
            "registered_h_s": [DT0_S / float(divisor) for divisor in SEMIGROUP_DIVISORS],
        },
        "03_additive_error_decomposition.md": {"rows": additive_rows},
        "04_face_flow_composition.md": {
            "face_row_count": len(face_rows),
            "output": f"outputs/{TASK_NAME}/face_flow_semigroup_error.csv",
            "S_flow_changed": "direct_source_cell_index != composed_source_cell_index",
            "S_near_boundary": "fixed binary64-plus-geometric radius diagnostic only; no population threshold",
        },
        "05_intermediate_projection_connectivity.md": {
            "row_count": len(projection_rows),
            "output": f"outputs/{TASK_NAME}/projection_connectivity_difference.csv",
            "S_projection_changed_cells": "destination rows with source-support or declared round-off-resolved coefficient change",
        },
        "06_frozen_error_changed_face_pairing.md": {"rows": pairing_rows},
        "07_support_expansion.md": {"rows": expansion_rows},
        "08_sparse_operator_audit.md": {"rows": operator_rows},
        "09_zero_event_control.md": {"summary": zero_summary, "search_rows": zero_rows},
        "10_single_event_control.md": {"summary": synthetic_summary, "rows": synthetic_rows},
        "11_h_refinement.md": {"trace_trend": trace_trend, "remap_trend": remap_trend, "rows": refinement_rows},
        "12_causal_classification.md": classification,
        "13_method_decision.md": decision,
        "14_final_acceptance_report.md": final,
        "15_reproduction_commands.md": {
            "command": (
                f"PYTHONPATH=src {sys.executable} scripts/run_kwn_frozen_semigroup_decomposition_v1.py audit "
                "--restart-checkpoint <exact-step-244-checkpoint> "
                "--formal-trace-csv <exact-formal-trace-csv> "
                "--formal-phase-a-csv <exact-formal-phase-a-csv> "
                "--frozen-semigroup-csv <exact-frozen-semigroup-csv> "
                f"--output-root <new-output-root> --report-root <new-report-root> --test-status {INHERITED_TEST_STATUS}"
            ),
            "frozen_only": True,
            "no_dynamic_or_prescribed_x": True,
            "no_method_prototype": True,
            "time_reference_v2": "NOT_ASSIGNED",
        },
    }
    _write_reports(report_root, reports)
    provenance = {
        "task_name": TASK_NAME,
        "top_status": final["STATUS"],
        "source": source_identity,
        "runtime": {"python": sys.version, "platform": platform.platform(), "numpy": np.__version__},
        "implementation": _implementation_provenance(),
        "baseline": baseline,
        "inherited_evidence_inputs": inherited_evidence_inputs,
        "u0_metadata": u0_metadata,
        "registered_h_s": [DT0_S / float(divisor) for divisor in SEMIGROUP_DIVISORS],
        "successful_h_level_count": len(successful_levels),
        "test_status": str(args.test_status),
        "runtime_s": time.monotonic() - started,
        "diagnostic_only": True,
        "dynamic_matrix_path_run": False,
        "prescribed_x_path_run": False,
        "root_selection_used": False,
        "time_reference_v2": "NOT_ASSIGNED",
        "pf_source_modified": False,
        "cuda_rerun": False,
        "physical_retuning": False,
        "gp_release": False,
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
    audit.add_argument("--frozen-semigroup-csv", type=Path, required=True)
    audit.add_argument("--output-root", type=Path, required=True)
    audit.add_argument("--report-root", type=Path, required=True)
    audit.add_argument("--test-status", default=INHERITED_TEST_STATUS)
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
