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
import ast
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
from kwn_mvp.characteristic_dt_continuation import accepted_state_hash  # noqa: E402
from kwn_mvp.characteristic_reference import (  # noqa: E402
    FIXED_POINT_CLOSURE,
    FIXED_POINT_PERIODIC_CYCLE_PERIODS,
    FIXED_POINT_PERIODIC_ROOT_TRIGGER,
    FIXED_POINT_SAFEGUARDED_ROOT_TRIGGER,
    FIXED_POINT_SCALAR_ROOT_MAX_ITERATIONS,
    REMAP_ORDER,
    TRACE_INTEGRATOR,
    CharacteristicReferenceError,
    CharacteristicReferenceSolver,
)
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
from scripts import run_kwn_cr1_fine_substep_pathology_v1 as legacy_checkpoint  # noqa: E402
from scripts.frozen_canonical_smooth_population_v1 import (  # noqa: E402
    build_frozen_canonical_context,
    frozen_canonical_snapshot_provenance,
)


TASK_NAME = "kwn_production_trace_audit_v1"
REQUIRED_BRANCH = "codex/kwn-production-trace-audit-v1"
FROZEN_ANCESTOR = "7759c83ca4da7fd930e0d67dfbd15788095702cd"
EXPECTED_FROZEN_U0_HASH = "7de7d098e1ee4a44e291d4771854fd8dfb8a05cdb5e404ae176d22d7bc88c98a"
EXPECTED_RESTART_SHA256 = "c0bc8dd946769550d780763d5a73446b929bac15c5bf7a04895a6288a2314770"
LEGACY_RESTART_STEP = 244
LEGACY_RESTART_STATE_HASH = "51c243b61f4278fdf8f5bae9afe50672f79a6e0be57260b9c7cc0fdb46dca4c2"
LEGACY_SOLVER_VERSION = "kwn_conservative_characteristic_remap_cr1_gl2_bracket_v5"
LEGACY_TRACE_INTEGRATOR = "AUTONOMOUS_RADIUS_GAUSS_LEGENDRE_2_BACKWARD_V1"
LEGACY_CHECKPOINT_ARRAY_KEYS = frozenset(
    {
        "g_number_density_per_m4",
        "beta_number_density_per_m4",
        "matrix_xb",
        "time_s",
        "step",
        "cumulative_number_dissolution_m3",
        "cumulative_beta_volume_dissolution",
        "cumulative_mol_b_returned_mol_m3",
        "metadata_json",
    }
)
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
# This is the existing CR1 acceptance scale used by CR5; the trace audit does
# not relax it or introduce a new population-conservation tolerance.
REMAP_CONSERVATION_RELATIVE_TOLERANCE = 1.0e-12
_REMAP_GUARD_SYMBOLS = (
    "characteristic_remap_partition",
    "same_characteristic_remap_partition",
    "piecewise_constant_cdf",
    "conservative_remap_piecewise_constant",
)

# The V1 diagnosis was completed and frozen before the V2 repair.  These are
# evidence values from its hash-pinned remote report, retained here only so a
# V2 qualification does not pretend that rerunning a repaired kernel is a
# second independent reconstruction of the removed V1 fixed-six path.
HISTORICAL_V1_TRACE_ROOT_CAUSE: dict[str, Any] = {
    "STATUS": "DIAG_TRACE_INTERPOLATION_OR_INVERSION_ERROR",
    "PRIMARY_ROOT_CAUSE": "TRACE_INVERSION_ERROR",
    "SECONDARY_ROOT_CAUSE": "FIXED_SIX_ITERATION_NO_TERMINATION_CRITERION",
    "legacy_h128_max_relative_error": 1.1248904585492794e-08,
    "resolution32_h128_max_relative_error": 1.1344280911503236e-08,
    "resolution64_h128_max_relative_error": 1.132527624980762e-08,
    "resolution128_h128_max_relative_error": 1.1171999175201331e-08,
    "bracketed_same_table_h128_max_relative_error": 3.820913359031391e-13,
    "exact_tau_bracketed_h128_max_relative_error": 3.820913359031391e-13,
    "table_resolution_gain_16_to_64": 0.9932565296748392,
    "inversion_gain_production_kernel_to_bracketed": 29440.35503685018,
    "production_kernel_inverse_max_radius_error_m": 9.021913788141991e-10,
    "bracketed_inverse_max_radius_error_m": 0.0,
    "repeated_accumulation_supported": True,
    "directly_supported_minimal_fix_auxiliary_subcells": None,
    "trace_reference_envelope_relative": TRACE_ENVELOPE_RELATIVE,
    "historical_evidence": {
        "source_commit": "9fecdc1d67469ae88d9765d63675e775de6f06bd",
        "run_root": (
            "/data/home/luozhiheng/tmp/"
            "kwn_production_trace_audit_v1_7759c83ca4da_20260905T050939Z/diagnosis_retry2"
        ),
        "report": "reports/kwn_production_trace_audit_v1/16_final_acceptance_report.md",
        "method": "V1 production GL2 table plus fixed six local inverse iterations",
    },
}

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


def _named_definition_ast(source_text: str, symbol: str) -> str:
    """Return a location-independent AST for one top-level remap definition."""

    try:
        tree = ast.parse(source_text)
    except SyntaxError as error:
        raise ProductionTraceWorkflowError("could not parse CR1 remap source for frozen comparison") from error
    candidates = [
        node
        for node in tree.body
        if isinstance(node, (ast.ClassDef, ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == symbol
    ]
    if len(candidates) != 1:
        raise ProductionTraceWorkflowError(f"could not identify frozen CR1 remap symbol {symbol}")
    return ast.dump(candidates[0], annotate_fields=True, include_attributes=False)


def _cr1_remap_unchanged_guard() -> dict[str, Any]:
    """Prove that the frozen CR1 CDF/remap implementation was not edited.

    The trace lives in the same module as the remap, so a file-level diff
    would be meaningless.  Compare the four actual remap definitions to the
    required frozen ancestor and permit no semantic edit to them.
    """

    relative_path = "src/kwn_mvp/conservative_remap.py"
    current_path = ROOT / relative_path
    try:
        frozen_source = _git(("show", f"{FROZEN_ANCESTOR}:{relative_path}"))
        current_source = current_path.read_text(encoding="utf-8")
    except (OSError, subprocess.CalledProcessError) as error:
        raise ProductionTraceWorkflowError("could not load frozen CR1 remap for unchanged-operator guard") from error
    symbols = {
        symbol: _named_definition_ast(current_source, symbol) == _named_definition_ast(frozen_source, symbol)
        for symbol in _REMAP_GUARD_SYMBOLS
    }
    return {
        "frozen_ancestor": FROZEN_ANCESTOR,
        "source_path": relative_path,
        "checked_symbols": list(_REMAP_GUARD_SYMBOLS),
        "symbol_ast_matches_frozen": symbols,
        "CR1_REMAP_IMPLEMENTATION_UNCHANGED": all(symbols.values()),
        "scope": "CR1 CDF, partition, and piecewise-constant remap definitions only; production trace inversion is audited separately",
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


def _legacy_checkpoint_options(metadata: Mapping[str, Any], *, config: Any) -> dict[str, Any]:
    """Validate the one V1 checkpoint contract without relaxing its loader.

    This is intentionally narrower than a legacy compatibility layer: the
    caller has already pinned one file by SHA-256, and this accepts exactly
    its V1 CR1 metadata solely to materialize a disposable frozen U0 under
    the qualified V2 trace kernel.
    """

    expected_keys = {
        "fixed_point_atol",
        "fixed_point_closure",
        "fixed_point_max_iterations",
        "fixed_point_periodic_scalar_root_requires_map_successor_population_check",
        "fixed_point_periodic_scalar_root_trigger",
        "fixed_point_rtol",
        "fixed_point_safeguarded_scalar_root_requires_same_cdf_source_partition",
        "fixed_point_safeguarded_scalar_root_requires_same_trace_topology",
        "fixed_point_safeguarded_scalar_root_trigger",
        "fixed_point_safeguarded_scalar_root_verification",
        "fixed_point_scalar_root_cycle_periods",
        "fixed_point_scalar_root_max_iterations",
        "fixed_point_scalar_root_requires_original_xb_tolerance",
        "remap_order",
        "solver_version",
        "source_config_hash",
        "temperature_K",
        "total_b_mol_m3",
        "trace_integrator",
        "under_relaxation",
        "validation_contract_hash",
    }
    if set(metadata) != expected_keys:
        raise ProductionTraceWorkflowError("pinned legacy checkpoint metadata schema differs")
    if metadata.get("solver_version") != LEGACY_SOLVER_VERSION:
        raise ProductionTraceWorkflowError("pinned checkpoint does not carry the registered V1 CR1 solver")
    if metadata.get("trace_integrator") != LEGACY_TRACE_INTEGRATOR:
        raise ProductionTraceWorkflowError("pinned checkpoint does not carry the registered V1 trace kernel")
    if metadata.get("remap_order") != REMAP_ORDER:
        raise ProductionTraceWorkflowError("pinned checkpoint remap contract differs")
    if metadata.get("fixed_point_closure") != FIXED_POINT_CLOSURE:
        raise ProductionTraceWorkflowError("pinned checkpoint closure contract differs")
    if int(metadata.get("fixed_point_scalar_root_max_iterations", -1)) != FIXED_POINT_SCALAR_ROOT_MAX_ITERATIONS:
        raise ProductionTraceWorkflowError("pinned checkpoint scalar-root iteration cap differs")
    if metadata.get("fixed_point_periodic_scalar_root_trigger") != FIXED_POINT_PERIODIC_ROOT_TRIGGER:
        raise ProductionTraceWorkflowError("pinned checkpoint periodic scalar-root trigger differs")
    if metadata.get("fixed_point_safeguarded_scalar_root_trigger") != FIXED_POINT_SAFEGUARDED_ROOT_TRIGGER:
        raise ProductionTraceWorkflowError("pinned checkpoint safeguarded scalar-root trigger differs")
    if metadata.get("fixed_point_scalar_root_cycle_periods") != list(FIXED_POINT_PERIODIC_CYCLE_PERIODS):
        raise ProductionTraceWorkflowError("pinned checkpoint scalar-root cycle contract differs")
    if metadata.get("fixed_point_scalar_root_requires_original_xb_tolerance") is not True:
        raise ProductionTraceWorkflowError("pinned checkpoint scalar-root tolerance contract differs")
    if metadata.get("fixed_point_periodic_scalar_root_requires_map_successor_population_check") is not True:
        raise ProductionTraceWorkflowError("pinned checkpoint periodic population contract differs")
    if metadata.get("fixed_point_safeguarded_scalar_root_requires_same_cdf_source_partition") is not True:
        raise ProductionTraceWorkflowError("pinned checkpoint CDF partition contract differs")
    if metadata.get("fixed_point_safeguarded_scalar_root_requires_same_trace_topology") is not True:
        raise ProductionTraceWorkflowError("pinned checkpoint trace topology contract differs")
    if metadata.get("fixed_point_safeguarded_scalar_root_verification") != "SAME_X_IMMUTABLE_REPLAY":
        raise ProductionTraceWorkflowError("pinned checkpoint scalar-root verification contract differs")
    if metadata.get("source_config_hash") != config.source_config_hash:
        raise ProductionTraceWorkflowError("pinned checkpoint config binding differs")
    if metadata.get("validation_contract_hash") != config.validation_contract_hash:
        raise ProductionTraceWorkflowError("pinned checkpoint validation-contract binding differs")
    if float(metadata.get("temperature_K", math.nan)) != float(config.temperature_k):
        raise ProductionTraceWorkflowError("pinned checkpoint temperature differs")
    options = {
        "fixed_point_rtol": float(metadata["fixed_point_rtol"]),
        "fixed_point_atol": float(metadata["fixed_point_atol"]),
        "fixed_point_max_iterations": int(metadata["fixed_point_max_iterations"]),
        "under_relaxation": float(metadata["under_relaxation"]),
    }
    if not math.isfinite(float(metadata.get("total_b_mol_m3", math.nan))):
        raise ProductionTraceWorkflowError("pinned checkpoint total inventory is invalid")
    return options


def _materialize_pinned_legacy_step244(
    *, restart_checkpoint: Path, config: Any
) -> tuple[CharacteristicReferenceSolver, dict[str, Any]]:
    """Raw-read exactly one historical V1 state into a current V2 solver.

    This is not a checkpoint loader and must never be made general-purpose.
    Normal ``CharacteristicReferenceSolver.load_checkpoint`` remains the
    only restart path and deliberately rejects the V1 trace identity.
    """

    try:
        with np.load(restart_checkpoint, allow_pickle=False) as archive:
            if frozenset(archive.files) != LEGACY_CHECKPOINT_ARRAY_KEYS:
                raise ProductionTraceWorkflowError("pinned legacy checkpoint array schema differs")
            metadata = json.loads(str(archive["metadata_json"].item()))
            if not isinstance(metadata, dict):
                raise ProductionTraceWorkflowError("pinned legacy checkpoint metadata is not an object")
            options = _legacy_checkpoint_options(metadata, config=config)
            float_names = (
                "g_number_density_per_m4",
                "beta_number_density_per_m4",
                "matrix_xb",
                "time_s",
                "cumulative_number_dissolution_m3",
                "cumulative_beta_volume_dissolution",
                "cumulative_mol_b_returned_mol_m3",
            )
            for name in float_names:
                if archive[name].dtype != np.dtype(np.float64):
                    raise ProductionTraceWorkflowError(f"pinned legacy checkpoint {name} dtype differs")
            if archive["step"].dtype != np.dtype(np.int64):
                raise ProductionTraceWorkflowError("pinned legacy checkpoint step dtype differs")
            values = {name: np.asarray(archive[name]).copy() for name in LEGACY_CHECKPOINT_ARRAY_KEYS - {"metadata_json"}}
    except (OSError, KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise ProductionTraceWorkflowError("could not raw-read the pinned legacy checkpoint") from error

    source = CharacteristicReferenceSolver(config, **options)
    if float(metadata["total_b_mol_m3"]) != float(source.ledger.total_b_mol_m3):
        raise ProductionTraceWorkflowError("pinned legacy checkpoint total inventory differs from its config")
    if metadata["validation_contract_hash"] != source.contract_hash:
        raise ProductionTraceWorkflowError("pinned legacy checkpoint contract differs from its configured ledger")
    expected_density_shape = source.population("beta").number_density_per_m4.shape
    g_density = np.asarray(values["g_number_density_per_m4"], dtype=np.float64)
    beta_density = np.asarray(values["beta_number_density_per_m4"], dtype=np.float64)
    scalars = {
        "matrix_xb": np.asarray(values["matrix_xb"], dtype=np.float64),
        "time_s": np.asarray(values["time_s"], dtype=np.float64),
        "step": np.asarray(values["step"], dtype=np.int64),
        "cumulative_number_dissolution_m3": np.asarray(values["cumulative_number_dissolution_m3"], dtype=np.float64),
        "cumulative_beta_volume_dissolution": np.asarray(values["cumulative_beta_volume_dissolution"], dtype=np.float64),
        "cumulative_mol_b_returned_mol_m3": np.asarray(values["cumulative_mol_b_returned_mol_m3"], dtype=np.float64),
    }
    if g_density.shape != expected_density_shape or beta_density.shape != expected_density_shape:
        raise ProductionTraceWorkflowError("pinned legacy checkpoint density shape differs from the frozen grid")
    if any(value.shape != (1,) for value in scalars.values()):
        raise ProductionTraceWorkflowError("pinned legacy checkpoint scalar shape differs")
    if not np.all(np.isfinite(g_density)) or not np.all(np.isfinite(beta_density)) or np.any(g_density < 0.0) or np.any(beta_density < 0.0):
        raise ProductionTraceWorkflowError("pinned legacy checkpoint density state is invalid")
    matrix_xb = float(scalars["matrix_xb"][0])
    time_s = float(scalars["time_s"][0])
    step = int(scalars["step"][0])
    cumulative = tuple(
        float(scalars[name][0])
        for name in (
            "cumulative_number_dissolution_m3",
            "cumulative_beta_volume_dissolution",
            "cumulative_mol_b_returned_mol_m3",
        )
    )
    if not math.isfinite(matrix_xb) or not 0.0 <= matrix_xb <= 1.0:
        raise ProductionTraceWorkflowError("pinned legacy checkpoint matrix composition is invalid")
    if not math.isfinite(time_s) or time_s < 0.0 or step != LEGACY_RESTART_STEP:
        raise ProductionTraceWorkflowError("pinned legacy checkpoint time/step differs from step 244")
    if any(not math.isfinite(value) or value < 0.0 for value in cumulative):
        raise ProductionTraceWorkflowError("pinned legacy checkpoint boundary inventory is invalid")
    source.population("g").number_density_per_m4[:] = g_density
    source.population("beta").number_density_per_m4[:] = beta_density
    source.matrix_xb = matrix_xb
    source.time_s = time_s
    source.step = step
    source.cumulative_number_dissolution_m3 = cumulative[0]
    source.cumulative_beta_volume_dissolution = cumulative[1]
    source.cumulative_mol_b_returned_mol_m3 = cumulative[2]
    source.history = []
    try:
        source._assert_beta_only_non_nucleating_scope()
        source.ledger.snapshot(matrix_xb=source.matrix_xb, populations=source.population_list())
    except Exception as error:
        raise ProductionTraceWorkflowError("pinned legacy checkpoint inventory does not close") from error
    if accepted_state_hash(source) != LEGACY_RESTART_STATE_HASH:
        raise ProductionTraceWorkflowError("pinned legacy checkpoint accepted-state hash differs")
    return source, metadata


def _load_frozen_state(restart_checkpoint: Path, output_root: Path) -> tuple[Any, dict[str, Any], dict[str, NDArray[np.generic]], dict[str, Any]]:
    if not restart_checkpoint.is_file():
        raise ProductionTraceWorkflowError("step-244 restart checkpoint is unavailable")
    if _sha256(restart_checkpoint) != EXPECTED_RESTART_SHA256:
        raise ProductionTraceWorkflowError("step-244 restart checkpoint hash differs")
    try:
        context = build_frozen_canonical_context()
        config, checkpoint_binding = legacy_checkpoint.checkpoint_bound_config(
            context=context, restart_checkpoint=restart_checkpoint
        )
    except Exception as error:
        raise ProductionTraceWorkflowError("could not bind the pinned legacy restart configuration") from error
    try:
        CharacteristicReferenceSolver.load_checkpoint(config=config, path=restart_checkpoint)
    except CharacteristicReferenceError as error:
        if "checkpoint solver version differs" not in str(error):
            raise ProductionTraceWorkflowError("normal V2 loader rejected legacy input for an unexpected reason") from error
    else:
        raise ProductionTraceWorkflowError("normal V2 checkpoint loader unexpectedly accepted a V1 trace identity")
    first, metadata = _materialize_pinned_legacy_step244(restart_checkpoint=restart_checkpoint, config=config)
    second, repeated_metadata = _materialize_pinned_legacy_step244(restart_checkpoint=restart_checkpoint, config=config)
    if metadata != repeated_metadata or not _state_arrays_equal(first.state_arrays(), second.state_arrays()):
        raise ProductionTraceWorkflowError("pinned legacy checkpoint materialization is not bitwise repeatable")
    try:
        arrays, u0_metadata, u0_hash = frozen_prior._u0_arrays_and_metadata(
            first, checkpoint_binding=checkpoint_binding
        )
        u0_repeat = frozen_prior._save_and_repeat_load_u0(
            output_root / "frozen_u0.npz", arrays=arrays, metadata=u0_metadata, expected_hash=u0_hash
        )
    except Exception as error:
        raise ProductionTraceWorkflowError("could not materialize the pinned frozen U0 artifact") from error
    baseline = {
        "status": "PASS_PINNED_LEGACY_V1_U0_MATERIALIZATION",
        "restart_checkpoint": str(restart_checkpoint),
        "restart_checkpoint_sha256": EXPECTED_RESTART_SHA256,
        "restart_step": int(first.step),
        "restart_state_hash": accepted_state_hash(first),
        # The formal historical U0 content digest embeds its staging-tree
        # contract path.  Its physical identity is instead pinned above by
        # the restart SHA and accepted-state hash; record both digests rather
        # than falsely demanding a path-dependent byte identity.
        "frozen_u0_content_hash": EXPECTED_FROZEN_U0_HASH,
        "materialized_frozen_u0_content_hash": u0_hash,
        "frozen_xB": float(first.matrix_xb),
        "normal_loader_legacy_checkpoint": "REJECTED_BY_STRICT_V2_LOADER",
        "checkpoint_repeat_materialization": "PASS_BITWISE",
        "frozen_u0_repeat_load": u0_repeat,
        "checkpoint_config_binding": checkpoint_binding,
        "frozen_canonical_context": frozen_canonical_snapshot_provenance(),
        "legacy_input_solver_version": LEGACY_SOLVER_VERSION,
        "legacy_input_trace_integrator": LEGACY_TRACE_INTEGRATOR,
        "qualified_solver_version": CharacteristicReferenceSolver.solver_version,
        "qualified_trace_integrator": TRACE_INTEGRATOR,
        "scope": "PINNED_LEGACY_U0_FOR_FROZEN_DISPOSABLE_TRACE_AND_PHI_COMPOSE_ONLY",
    }
    return first, baseline, arrays, u0_metadata


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
                    "inversion_mode": "PRODUCTION_KERNEL",
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
                    "production_kernel_same_table_radius_m": prod_same.radius_m,
                    "production_kernel_same_table_error_m": abs(prod_same.radius_m - expected_radius),
                    "production_kernel_exact_target_status": exact_target_status,
                    "production_kernel_exact_target_radius_m": None if prod_exact_target is None else prod_exact_target.radius_m,
                    "production_kernel_exact_target_error_m": None if prod_exact_target is None else abs(prod_exact_target.radius_m - expected_radius),
                    "bracketed_same_table_radius_m": bracket_same.radius_m,
                    "bracketed_same_table_error_m": abs(bracket_same.radius_m - expected_radius),
                    "exact_tau_bracketed_radius_m": exact_bracket.radius_m,
                    "exact_tau_bracketed_error_m": abs(exact_bracket.radius_m - expected_radius),
                    "production_kernel_iteration_count": prod_same.iteration_count,
                    "production_kernel_bracket_width_m": prod_same.bracket_width_m,
                    "production_kernel_residual_s": prod_same.residual_s,
                    "bracketed_iteration_count": bracket_same.iteration_count,
                    "bracketed_bracket_width_m": bracket_same.bracket_width_m,
                    "bracketed_residual_s": bracket_same.residual_s,
                }
            )
    return rows, {
        "production_kernel_inverse_max_radius_error_m": max_production_error,
        "bracketed_inverse_max_radius_error_m": max_bracketed_error,
        "production_kernel_iteration_telemetry": "NOT_EXPOSED_BY_PUBLIC_TRACE_KERNEL",
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
            "TRACE_MODE_3_EXACT_TAU_TABLE_NODES_PLUS_PRODUCTION_KERNEL_INVERSION",
            ProductionAutonomousTOFTable(
                edges_m=edges_m,
                velocity_m_s=velocity,
                exact_flow=flow,
                config=ProductionTraceAuditConfig(table_kind="EXACT_TAU", inversion_mode="PRODUCTION_KERNEL"),
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
        finite = [
            float(value)
            for row in order["pair_rows"]
            for value in (row["observed_order"],)
            if value is not None and math.isfinite(float(value))
        ]
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
    required_regions = (
        "SHRINKING_BULK",
        "GROWING_BULK",
        "NEAR_CRITICAL_CONDITIONING_Q95",
        "NEAR_RMIN_GRID_LOCAL",
    )
    choices: list[tuple[str, float]] = []
    missing: list[str] = []
    for wanted in required_regions:
        indices = np.flatnonzero(labels == wanted)
        if indices.size == 0:
            missing.append(wanted)
            continue
        if wanted == "NEAR_CRITICAL_CONDITIONING_Q95":
            index = int(indices[np.argmax(conditioning[indices])])
        elif wanted == "NEAR_RMIN_GRID_LOCAL":
            index = int(indices[-1])
        else:
            index = int(indices[indices.size // 2])
        choices.append((wanted, float(edges_m[index])))
    if missing:
        raise ProductionTraceWorkflowError(
            f"could not select every required representative frozen trace region: {missing}"
        )
    return choices, {**rule, "required_repeated_trace_regions": list(required_regions)}


def _classify_repeated_accumulation(
    rows: Sequence[Mapping[str, Any]],
) -> tuple[str, list[dict[str, Any]], bool]:
    """Classify every adjacent refinement pair without endpoint shortcuts."""

    ordered = sorted(rows, key=lambda row: float(row["h_s"]), reverse=True)
    if [float(row["h_s"]) for row in ordered] != list(H_LADDER_S):
        raise ProductionTraceWorkflowError("repeated trace rows do not carry the complete registered h ladder")
    relative = [float(row["final_accumulated_relative_error"]) for row in ordered]
    if not all(math.isfinite(value) and value >= 0.0 for value in relative):
        raise ProductionTraceWorkflowError("repeated trace accumulation produced an invalid relative error")
    pair_rows: list[dict[str, Any]] = []
    n_like_pair = False
    exponents: list[float] = []
    for coarse, fine, coarse_relative, fine_relative in zip(ordered[:-1], ordered[1:], relative[:-1], relative[1:]):
        h_ratio = float(coarse["h_s"]) / float(fine["h_s"])
        if h_ratio <= 1.0:
            raise ProductionTraceWorkflowError("repeated trace h ladder is not strictly refined")
        if coarse_relative == 0.0:
            exponent = math.inf if fine_relative > 0.0 else math.nan
        elif fine_relative == 0.0:
            exponent = -math.inf
        else:
            exponent = math.log(fine_relative / coarse_relative) / math.log(h_ratio)
        above_reference = fine_relative > TRACE_ENVELOPE_RELATIVE
        is_n_like = above_reference and exponent >= 0.75
        n_like_pair = n_like_pair or is_n_like
        if math.isfinite(exponent):
            exponents.append(exponent)
        pair_rows.append(
            {
                "h_coarse_s": float(coarse["h_s"]),
                "h_fine_s": float(fine["h_s"]),
                "coarse_final_relative_error": coarse_relative,
                "fine_final_relative_error": fine_relative,
                "accumulation_scaling_exponent": exponent,
                "fine_above_reference_envelope": above_reference,
                "n_like_or_faster_accumulation": is_n_like,
            }
        )
    if max(relative) <= TRACE_ENVELOPE_RELATIVE:
        classification = "TRACE_REPEATED_OTHER_REFERENCE_ENVELOPE"
    elif n_like_pair:
        classification = "TRACE_REPEATED_SCALES_AS_N_OR_FASTER"
    elif exponents and all(0.25 <= value < 0.75 for value in exponents):
        classification = "TRACE_REPEATED_SCALES_AS_SQRT_N"
    elif all(fine <= coarse for coarse, fine in zip(relative[:-1], relative[1:])):
        classification = "TRACE_REPEATED_CONVERGENT"
    else:
        classification = "TRACE_REPEATED_OTHER"
    return classification, pair_rows, n_like_pair


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
    scaling_rows: list[dict[str, Any]] = []
    n_like_regions: list[str] = []
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
            accumulated_relative = accumulated / max(abs(float(exact.radius_m)), 1.0e-300)
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
                "final_accumulated_relative_error": accumulated_relative,
                "accumulated_over_single_step": accumulated / max(first_error, 1.0e-300),
            }
            per_radius.append(row)
            rows.append(row)
        classification, pairs, n_like = _classify_repeated_accumulation(per_radius)
        classifications[label] = classification
        for row in per_radius:
            row["accumulation_classification"] = classification
            row["n_like_or_faster_accumulation"] = n_like
        if n_like:
            n_like_regions.append(label)
        scaling_rows.extend({"representative_region": label, **pair} for pair in pairs)
    return rows, {
        "direction": "backward departure composition; exact comparator is Flow_exact(R0, -H), matching the production departure map",
        "horizon_s": FINAL_HORIZON_S,
        "representative_selection": rule,
        "per_region_classification": classifications,
        "adjacent_refinement_scaling": scaling_rows,
        "n_like_or_faster_accumulation_regions": n_like_regions,
        "PASS_TRACE_REPEATED_COMPOSITION": not n_like_regions,
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
    initial_number_m3 = max(float(np.sum(u0_cells, dtype=np.float64)), 1.0e-300)
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
        cumulative_residual = float(sum(float(item.remap_number_conservation_residual_m3) for item in history))
        running_old_number = initial_number_m3
        step_relative_residuals: list[float] = []
        step_old_number_m3: list[float] = []
        max_abs_residual = 0.0
        for item in history:
            residual = float(item.remap_number_conservation_residual_m3)
            lower_loss = float(item.rmin_number_loss_m3)
            upper_loss = float(item.rmax_number_loss_m3)
            if not all(math.isfinite(value) for value in (residual, lower_loss, upper_loss, running_old_number)):
                raise ProductionTraceWorkflowError("CR1 history carries a non-finite remap conservation value")
            if lower_loss < 0.0 or upper_loss < 0.0 or running_old_number <= 0.0:
                raise ProductionTraceWorkflowError("CR1 history carries an invalid remap conservation scale")
            step_old_number_m3.append(running_old_number)
            step_relative_residuals.append(abs(residual) / running_old_number)
            max_abs_residual = max(max_abs_residual, abs(residual))
            running_old_number = running_old_number - lower_loss - upper_loss - residual
        if not math.isfinite(running_old_number) or running_old_number < 0.0:
            raise ProductionTraceWorkflowError("CR1 remap history lost its physical conservation scale")
        cr1_rows.append(
            {
                "comparison": "CURRENT_PRODUCTION_CR1_VS_REF_PC",
                "h_s": float(h_s),
                "substep_count": count,
                **metrics,
                "cr1_state_hash": str(result.state["accepted_state_hash"]),
                "rmin_number_loss_m3": float(sum(float(item.rmin_number_loss_m3) for item in history)),
                "rmax_number_loss_m3": float(sum(float(item.rmax_number_loss_m3) for item in history)),
                "remap_cumulative_conservation_residual_m3": cumulative_residual,
                "remap_cumulative_conservation_relative": abs(cumulative_residual) / initial_number_m3,
                "remap_max_abs_conservation_residual_m3": max_abs_residual,
                "remap_max_abs_conservation_relative": max(step_relative_residuals, default=0.0),
                "remap_step_old_number_m3_json": step_old_number_m3,
                "remap_step_conservation_relative_json": step_relative_residuals,
                "fixed_point_modes_json": [str(item.fixed_point_convergence_mode) for item in history],
            }
        )
        current = u0_cells.copy()
        departure = np.asarray(exact_departures[float(h_s)], dtype=np.float64)
        conservation = 0.0
        max_abs_conservation = 0.0
        max_relative_conservation = 0.0
        step_old_number_m3: list[float] = []
        step_relative_residuals: list[float] = []
        for _ in range(count):
            remap = conservative_remap_piecewise_constant(edges_m, current, departure)
            current = np.asarray(remap.cell_number_m3, dtype=np.float64)
            conservation += float(remap.conservation_residual_m3)
            max_abs_conservation = max(max_abs_conservation, abs(float(remap.conservation_residual_m3)))
            old_number = max(float(remap.old_number_m3), 1.0e-300)
            relative_residual = abs(float(remap.conservation_residual_m3)) / old_number
            step_old_number_m3.append(old_number)
            step_relative_residuals.append(relative_residual)
            max_relative_conservation = max(max_relative_conservation, relative_residual)
        projection_rows.append(
            {
                "comparison": "EXACT_FLOW_PLUS_SAME_CR1_REMAP_VS_REF_PC",
                "h_s": float(h_s),
                "substep_count": count,
                **measure_error_metrics(current, exact_reference.cell_number_m3, edges_m=edges_m),
                "remap_cumulative_conservation_residual_m3": conservation,
                "remap_cumulative_conservation_relative": abs(conservation) / initial_number_m3,
                "remap_max_abs_conservation_residual_m3": max_abs_conservation,
                "remap_max_abs_conservation_relative": max_relative_conservation,
                "remap_step_old_number_m3_json": step_old_number_m3,
                "remap_step_conservation_relative_json": step_relative_residuals,
            }
        )
        trace_induced_metrics = measure_error_metrics(production_cells, current, edges_m=edges_m)
        cr1_rows[-1].update(
            {
                f"trace_induced_{key}": value
                for key, value in trace_induced_metrics.items()
            }
        )
    return cr1_rows, projection_rows


def _conservation_unchanged_summary(
    *,
    cr1_rows: Sequence[Mapping[str, Any]],
    projection_rows: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Apply the existing CR1 remap-conservation contract to both paths."""

    if len(cr1_rows) != len(H_LADDER_S) or len(projection_rows) != len(H_LADDER_S):
        raise ProductionTraceWorkflowError("conservation gate requires the complete paired CR1 ladder")
    current_relative = [float(row["remap_max_abs_conservation_relative"]) for row in cr1_rows]
    projection_relative = [float(row["remap_max_abs_conservation_relative"]) for row in projection_rows]
    all_relative = current_relative + projection_relative
    if not all(math.isfinite(value) for value in all_relative):
        raise ProductionTraceWorkflowError("CR1 conservation gate received a non-finite cumulative residual")
    return {
        "criterion": "every CR1 remap substep must meet the existing CR5 remap-conservation relative tolerance; signed cumulative residual is diagnostic only",
        "relative_tolerance": REMAP_CONSERVATION_RELATIVE_TOLERANCE,
        "current_cr1_max_abs_substep_relative_residual": max(current_relative),
        "projection_only_max_abs_substep_relative_residual": max(projection_relative),
        "current_cr1_per_h_max_abs_substep_relative_residual": {
            str(int(round(1.0 / float(row["h_s"])))): float(row["remap_max_abs_conservation_relative"])
            for row in cr1_rows
        },
        "projection_only_per_h_max_abs_substep_relative_residual": {
            str(int(round(1.0 / float(row["h_s"])))): float(row["remap_max_abs_conservation_relative"])
            for row in projection_rows
        },
        "CONSERVATION_UNCHANGED": max(all_relative) <= REMAP_CONSERVATION_RELATIVE_TOLERANCE,
    }


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
                "trace_induced_relative_L1": float(full["trace_induced_population_relative_L1"]),
                "full_vs_exact_L1_abs": float(full["population_L1_abs"]),
                "projection_only_vs_exact_L1_abs": float(projection["population_L1_abs"]),
                "trace_induced_L1_abs": float(full["trace_induced_population_L1_abs"]),
                "trace_induced_relative_Linf": float(full["trace_induced_population_relative_Linf"]),
                "trace_induced_CDF_max_error": float(full["trace_induced_CDF_max_error"]),
                "trace_induced_Wasserstein_m": float(full["trace_induced_Wasserstein_m"]),
            }
        )
    return rows


def _paired_cr1_comparator(
    *,
    trace_induced_rows: Sequence[Mapping[str, Any]],
    baseline_cr1_relative_l1: Sequence[float],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Gate full and trace-induced error against the paired h comparator.

    The h=1/128 projection comparator is exactly zero in the frozen report,
    so the already-qualified independent reference floor supplies its only
    meaningful binary64 scale.  A maximum from another h level must never
    mask a failed paired comparison.
    """

    if len(trace_induced_rows) != len(H_LADDER_S) or len(baseline_cr1_relative_l1) != len(H_LADDER_S):
        raise ProductionTraceWorkflowError("paired CR1 comparator requires complete current and frozen h ladders")
    by_h = {float(row["h_s"]): row for row in trace_induced_rows}
    if len(by_h) != len(H_LADDER_S) or set(by_h) != set(H_LADDER_S):
        raise ProductionTraceWorkflowError("paired CR1 comparator h ladder differs from the frozen registration")
    rows: list[dict[str, Any]] = []
    for h_s, historical_full in zip(H_LADDER_S, baseline_cr1_relative_l1):
        row = by_h[float(h_s)]
        full = float(row["full_vs_exact_relative_L1"])
        projection = float(row["projection_only_vs_exact_relative_L1"])
        trace = float(row["trace_induced_relative_L1"])
        historical = float(historical_full)
        if not all(math.isfinite(value) and value >= 0.0 for value in (full, projection, trace, historical)):
            raise ProductionTraceWorkflowError("paired CR1 comparator received an invalid relative error")
        scale = max(projection, REFERENCE_FLOOR_RELATIVE)
        full_ratio = full / scale
        trace_ratio = trace / scale
        same_order = full_ratio <= 10.0 and trace_ratio <= 10.0
        rows.append(
            {
                **row,
                "paired_projection_comparator_scale": scale,
                "full_to_paired_projection_ratio": full_ratio,
                "trace_induced_to_paired_projection_ratio": trace_ratio,
                "historical_trace_dominated_full_relative_L1": historical,
                "trace_induced_reduction_factor_vs_historical": (
                    math.inf if trace == 0.0 and historical > 0.0 else historical / max(trace, 1.0e-300)
                ),
                "trace_induced_below_historical": trace < historical,
                "same_order_at_h": same_order,
            }
        )
    all_h_same_order = all(bool(row["same_order_at_h"]) for row in rows)
    all_trace_below_historical = all(bool(row["trace_induced_below_historical"]) for row in rows)
    all_trace_reduced_one_order = all(
        float(row["trace_induced_reduction_factor_vs_historical"]) >= 10.0 for row in rows
    )
    adjacent_trace_induced_refinement = [
        {
            "h_coarse_s": float(coarse["h_s"]),
            "h_fine_s": float(fine["h_s"]),
            "trace_induced_fine_to_coarse_ratio": (
                math.inf
                if float(coarse["trace_induced_relative_L1"]) == 0.0
                and float(fine["trace_induced_relative_L1"]) > 0.0
                else float(fine["trace_induced_relative_L1"])
                / max(float(coarse["trace_induced_relative_L1"]), 1.0e-300)
            ),
        }
        for coarse, fine in zip(rows[:-1], rows[1:])
    ]
    return rows, {
        "criterion": "at every h, full and direct trace-induced relative-L1 are at most 10 times max(paired projection-only relative-L1, independent reference floor)",
        "reference_floor_relative": REFERENCE_FLOOR_RELATIVE,
        "ALL_H_SAME_ORDER": all_h_same_order,
        "ALL_H_TRACE_INDUCED_BELOW_HISTORICAL": all_trace_below_historical,
        "ALL_H_TRACE_INDUCED_REDUCED_BY_AT_LEAST_ONE_ORDER": all_trace_reduced_one_order,
        "TRACE_INDUCED_H_WISE_BOUND": (
            "BOUNDED_BY_PAIRED_PROJECTION_COMPARATOR_AT_EVERY_H"
            if all_h_same_order
            else "NOT_BOUNDED_BY_PAIRED_PROJECTION_COMPARATOR_AT_EVERY_H"
        ),
        "adjacent_trace_induced_refinement": adjacent_trace_induced_refinement,
        "per_h": rows,
    }


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
    accumulation_supported = bool(repeated.get("n_like_or_faster_accumulation_regions", ()))
    if resolution64 <= TRACE_ENVELOPE_RELATIVE and resolution_gain >= 8.0 and inversion_gain < 4.0:
        primary = "TRACE_TOF_TABLE_RESOLUTION_FLOOR"
        status = "DIAG_TRACE_TOF_TABLE_RESOLUTION_FLOOR"
        fix_resolution = 64
        secondary = "TRACE_FIXED_PER_UNIT_TIME_GENERATOR_BIAS" if accumulation_supported else "NONE"
    elif inversion_gain >= 8.0 and bracketed <= TRACE_ENVELOPE_RELATIVE:
        primary = "TRACE_INVERSION_ERROR"
        status = "DIAG_TRACE_INTERPOLATION_OR_INVERSION_ERROR"
        fix_resolution = None
        secondary = "INVERSION_TERMINATION_OR_LOCAL_RESIDUAL_ERROR"
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
        "inversion_gain_production_kernel_to_bracketed": inversion_gain,
        "production_kernel_inverse_max_radius_error_m": inversion["production_kernel_inverse_max_radius_error_m"],
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
            "exact table-knot return plus local GL2 Newton/binary64 bracket inversion",
            "Rmin/Rmax no-inflow or stationary-tail handling",
            "departure faces supplied unchanged to CR1 CDF remap",
        ],
        "production_source": {
            "growth_kernel": "src/kwn_mvp/characteristic_reference.py: CharacteristicReferenceSolver._velocity_at_radii",
            "trace_entry": "src/kwn_mvp/conservative_remap.py: trace_departure_faces_rk2",
            "auxiliary_table": "_log_subdivided_faces, _gauss_time_of_flight, np.cumsum",
            "inversion": "_invert_time_of_flight: exact knot fast path; local residual with binary64 bracket closure",
            "physical_branch_tail": "_stationary_root and _stationary_tail_from_endpoint",
            "remap": "conservative_remap_piecewise_constant",
        },
        "precision": "binary64 throughout production trace; no persistent cross-step cache or quantization",
        "production_default_auxiliary_subcells_per_population_cell": int(_TRACE_SUBCELLS_PER_CELL),
        "interpolation": "linear lookup only selects a local table interval; local GL2 re-integration closes its binary64 inverse",
        "reference_difference": "independent frozen exact flow uses SciPy adaptive quadrature and residual-qualified safeguarded inverse; it does not import production trace or CR1 remap",
    }


def _run(args: argparse.Namespace) -> tuple[int, dict[str, Any]]:
    output_root = Path(args.output_root)
    report_root = Path(args.report_root)
    if output_root.exists() or report_root.exists():
        raise ProductionTraceWorkflowError("refusing to overwrite trace-audit output or report roots")
    source_identity = _source_identity()
    remap_guard = _cr1_remap_unchanged_guard()
    if not remap_guard["CR1_REMAP_IMPLEMENTATION_UNCHANGED"]:
        raise ProductionTraceWorkflowError("CR1 remap implementation differs from the frozen ancestor")
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
    current_component_reaudit = _root_cause(
        table_resolution_rows=resolution_rows,
        component_rows=component_rows,
        inversion=inversion_summary,
        repeated=repeated_summary,
    )
    root_cause = (
        {
            **HISTORICAL_V1_TRACE_ROOT_CAUSE,
            "post_fix_current_kernel_component_reaudit": current_component_reaudit,
        }
        if args.phase == "qualify"
        else current_component_reaudit
    )
    if not _state_arrays_equal(before_arrays, source.state_arrays()) or tuple(source.history) != before_history:
        raise ProductionTraceWorkflowError("read-only trace audit modified accepted frozen U0")

    current_default_summary = face_summaries[H_LADDER_S[0]]
    qualified_trace = False
    cr1_rows: list[dict[str, Any]] = []
    projection_rows: list[dict[str, Any]] = []
    trace_induced: list[dict[str, Any]] = []
    post_fix_repeated: dict[str, Any] = {"status": "NOT_RUN_IN_DIAGNOSIS_PHASE"}
    trace_kernel_face_and_composition_pass = False
    cr1_projection_comparator_same_order: bool | None = None
    cr1_comparator_summary: dict[str, Any] = {
        "ALL_H_SAME_ORDER": None,
        "ALL_H_TRACE_INDUCED_REDUCED_BY_AT_LEAST_ONE_ORDER": None,
        "status": "NOT_RUN_IN_DIAGNOSIS_PHASE",
    }
    conservation_summary: dict[str, Any] = {"CONSERVATION_UNCHANGED": None, "status": "NOT_RUN_IN_DIAGNOSIS_PHASE"}
    final_status = root_cause["STATUS"]
    fix_description = "NO_FIX_IMPLEMENTED_IN_DIAGNOSIS_PHASE"
    if args.phase == "qualify":
        all_face_pass = all(
            int(summary["status_mismatch_count"]) == 0
            and float(summary["max_relative_radius_error"]) <= TRACE_ENVELOPE_RELATIVE
            for summary in face_summaries.values()
        )
        cr1_rows, projection_rows = _cr1_rows_against_exact(
            source=source,
            edges_m=edges,
            u0_cells=u0_cells,
            exact_reference=exact_reference,
            frozen_xb=frozen_xb,
            exact_departures=exact_departures,
        )
        trace_induced = _trace_induced_rows(cr1_rows=cr1_rows, projection_rows=projection_rows)
        trace_induced, cr1_comparator_summary = _paired_cr1_comparator(
            trace_induced_rows=trace_induced,
            baseline_cr1_relative_l1=baseline_reproduction["cr1_relative_l1"],
        )
        conservation_summary = _conservation_unchanged_summary(cr1_rows=cr1_rows, projection_rows=projection_rows)
        same_order = bool(cr1_comparator_summary["ALL_H_SAME_ORDER"])
        trace_induced_reduced = bool(
            cr1_comparator_summary["ALL_H_TRACE_INDUCED_REDUCED_BY_AT_LEAST_ONE_ORDER"]
        )
        branch_pass = branch_summary["branch_or_status_mismatch_count"] == 0
        repeat_pass = bool(repeated_summary["PASS_TRACE_REPEATED_COMPOSITION"])
        trace_kernel_face_and_composition_pass = all_face_pass and branch_pass and repeat_pass
        cr1_projection_comparator_same_order = same_order
        # The frozen trace kernel is not fully qualified until the unchanged
        # CR1 remap also lands at the projection-only comparator scale.
        qualified_trace = (
            trace_kernel_face_and_composition_pass
            and same_order
            and trace_induced_reduced
            and bool(conservation_summary["CONSERVATION_UNCHANGED"])
        )
        post_fix_repeated = {"status": "PASS_TRACE_REPEATED_COMPOSITION" if repeat_pass else "FAIL_TRACE_REPEATED_COMPOSITION", **repeated_summary}
        fix_description = (
            f"Current production auxiliary TOF table uses {_TRACE_SUBCELLS_PER_CELL} subcells per population cell; "
            "V2 returns exact table knots and otherwise closes the same local GL2 interval at binary64 bracket resolution; "
            "the frozen qualification reuses the unchanged CR1 remap, grid, thermodynamics, and growth kernel."
        )
        if qualified_trace:
            final_status = "PASS_PRODUCTION_TRACE_KERNEL_FROZEN_QUALIFICATION"
        elif trace_kernel_face_and_composition_pass:
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
    _write_csv(
        output_root / "repeated_trace_refinement_scaling.csv",
        repeated_summary["adjacent_refinement_scaling"],
        ("representative_region", "h_coarse_s"),
    )
    _write_csv(output_root / "trace_component_ablation.csv", component_rows, ("trace_mode", "h_s"))
    _write_json(output_root / "trace_root_cause.json", root_cause)
    _write_csv(output_root / "qualified_trace_face_error.csv", [face_summaries[h] for h in H_LADDER_S], ("h_s", "max_relative_radius_error"))
    _write_csv(output_root / "corrected_cr1_vs_exact.csv", cr1_rows, ("h_s", "population_relative_L1"))
    _write_csv(output_root / "trace_projection_comparator.csv", trace_induced, ("h_s", "trace_induced_relative_L1"))
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
        "cr1_remap_unchanged_guard": remap_guard,
        "conservation_unchanged": conservation_summary,
        "cr1_paired_projection_comparator": cr1_comparator_summary,
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
        "12_minimal_trace_fix.md": {
            "phase": args.phase,
            "trace_fix_implemented": args.phase == "qualify",
            "description": fix_description,
            "cr1_remap_modified": False,
            "cr1_remap_unchanged_guard": remap_guard,
        },
        "13_trace_kernel_qualification.md": {
            "qualified": qualified_trace,
            "reference_floor_relative": REFERENCE_FLOOR_RELATIVE,
            "qualification_envelope_relative": TRACE_ENVELOPE_RELATIVE,
            "absolute_radius_gate": "NOT_REQUIRED: every frozen departure radius is positive and carries a defined relative scale",
            "face_summaries": [face_summaries[h] for h in H_LADDER_S],
            "branch_status_parity": branch_summary["branch_or_status_mismatch_count"] == 0,
            "repeated": post_fix_repeated,
            "cr1_projection_comparator_same_order": cr1_projection_comparator_same_order,
            "cr1_paired_projection_comparator": cr1_comparator_summary,
            "conservation_unchanged": conservation_summary,
            "cr1_remap_unchanged_guard": remap_guard,
        },
        "14_corrected_cr1_vs_exact.md": {
            "current_cr1_rows": cr1_rows,
            "projection_rows": projection_rows,
            "trace_induced_rows": trace_induced,
            "cr1_paired_projection_comparator": cr1_comparator_summary,
            "conservation_unchanged": conservation_summary,
        },
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
        "PRODUCTION_TRACE_ARCHITECTURE": "fixed log auxiliary TOF table + GL2 + binary64-bracketed production inversion",
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
        "INVERSION_EFFECT": root_cause["inversion_gain_production_kernel_to_bracketed"],
        "SINGLE_STEP_TRACE_ORDER": scaling_summary,
        "TRACE_ERROR_FLOOR": (
            "HISTORICAL_V1_FIXED_PER_STEP_INVERSION_FLOOR_WITH_STEP_COUNT_ACCUMULATION"
            if root_cause["repeated_accumulation_supported"]
            else "NOT_SUPPORTED_BY_THE_HISTORICAL_V1_DIAGNOSIS"
        ),
        "REPEATED_TRACE_ACCUMULATION": repeated_summary,
        "ACCUMULATION_SCALING": repeated_summary["per_region_classification"],
        "COMPONENT_ABLATION_RESULT": root_cause,
        "PRIMARY_ROOT_CAUSE": root_cause["PRIMARY_ROOT_CAUSE"],
        "SECONDARY_ROOT_CAUSE": root_cause["SECONDARY_ROOT_CAUSE"],
        "TRACE_FIX_IMPLEMENTED": args.phase == "qualify",
        "TRACE_FIX_DESCRIPTION": fix_description,
        "CR1_REMAP_MODIFIED": not bool(remap_guard["CR1_REMAP_IMPLEMENTATION_UNCHANGED"]),
        "CR1_REMAP_UNCHANGED_GUARD": remap_guard,
        "CONSERVATION_UNCHANGED": conservation_summary,
        "QUALIFIED_TRACE_KERNEL": qualified_trace,
        "CR1_PROJECTION_COMPARATOR_SAME_ORDER": cr1_projection_comparator_same_order,
        "TRACE_INDUCED_REDUCED_BY_AT_LEAST_ONE_ORDER": cr1_comparator_summary[
            "ALL_H_TRACE_INDUCED_REDUCED_BY_AT_LEAST_ONE_ORDER"
        ],
        "CR1_PAIRED_PROJECTION_COMPARATOR": cr1_comparator_summary,
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
