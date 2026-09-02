#!/usr/bin/env python3
"""Gate the KWN physical-Rmin and active-CFL qualification workflow.

This runner is intentionally KWN-only.  It never invokes PF/CUDA, modifies a
validation contract, or retunes material parameters.  It records the old
boundary failure as immutable baseline evidence, then permits later stages
only after the physical lower-face/operator tests pass.

The default ``all`` command runs the finite lower-boundary gates and, only on
success, the first-significant-lower-tail implicit active-CFL ladder.  A full canonical cohort
crosscheck is deliberately opt-in because it is a long CPU calculation; the
runner writes a fail-closed provenance-backed stub until that calculation is
explicitly authorized with ``--allow-long-cohort``.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import dataclass
import hashlib
import importlib
import importlib.util
import io
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time
import traceback
from typing import Any, Callable, Iterable, Mapping, Sequence
import unittest

import numpy as np
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.cohort_solver import Cohort, CohortSolver  # noqa: E402
from kwn_mvp.diagnostics import discrete_wasserstein_distance  # noqa: E402
from kwn_mvp.lower_boundary import (  # noqa: E402
    boundary_growth_velocity,
    boundary_inventory_diagnostic,
    boundary_number_flux_diagnostic,
    boundary_radius,
    particle_inventory_at_radius,
)
from kwn_mvp.population_metrics import (  # noqa: E402
    beta_fraction_from_m3,
    beta_inventory_from_m3,
    cell_moments_from_piecewise_constant_cells,
    close_matrix_from_precipitates,
    metrics_from_discrete_measure,
    metrics_from_piecewise_constant_cells,
    positive_cell_quadrature,
)
from kwn_mvp.populations import Population  # noqa: E402
from kwn_mvp.radius_grid import RadiusGrid  # noqa: E402
from kwn_mvp.solver import KWNSolver, SolverConfig  # noqa: E402


TASK_NAME = "kwn_lower_boundary_time_accuracy_v1"
DEFAULT_OUTPUT_ROOT = ROOT / "outputs" / TASK_NAME
DEFAULT_REPORT_ROOT = ROOT / "reports" / TASK_NAME
BASELINE_COMMIT = "78b6745767607dcdbcf069ce4498744fbef612df"
EXPECTED_CONTRACT_HASH = "d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333"
TWO_PERCENT = 0.02
SUPPORT_FRACTION = 0.999999
LOWER_TAIL_QUANTILES = (1.0e-8, 1.0e-6)
SMOOTH_TIMES_H = (0.0, 0.1, 0.39317699499770825, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0)
# Every policy uses this same horizon.  It covers the task contract's
# first-significant-lower-tail-dissolution alternative while ensuring that a
# frozen-state boundary audit cannot be substituted for a source-owned,
# self-consistent active-CFL step decision.
FIRST_SIGNIFICANT_BOUNDARY_TIME_H = (0.0, 0.1)
IMPLICIT_ACTIVE_CFL_CAPS = (4.0, 2.0, 1.0, 0.5, 0.25, 0.125)
REQUIRED_REPORTS = {
    "00_baseline_boundary_reproduction.md": "Baseline boundary reproduction",
    "01_physical_lower_boundary_contract.md": "Physical lower-boundary contract",
    "02_boundary_unit_tests.md": "Boundary unit tests",
    "03_analytic_boundary_benchmarks.md": "Analytic lower-boundary benchmarks",
    "04_boundary_operator_parity.md": "Physical lower-boundary operator parity",
    "05_courant_definition_and_audit.md": "Courant definition and audit",
    "06_implicit_time_accuracy.md": "Implicit time-accuracy ladder",
    "07_explicit_reference_diagnostic.md": "Explicit reference diagnostic",
    "08_transport_scheme_decision.md": "Transport-scheme decision",
    "09_cohort_eulerian_crosscheck.md": "Cohort--Eulerian crosscheck",
    "10_eulerian_authority_v2.md": "Eulerian authority v2",
    "11_beta_only_cohort_pf_comparison.md": "Beta-only cohort--PF comparison",
    "12_model_role_boundary.md": "Model-role boundary",
    "13_final_acceptance_report.md": "Final acceptance",
    "14_reproduction_commands.md": "Reproduction commands",
}


@dataclass(frozen=True)
class CanonicalContext:
    """One explicit canonical smooth measure and its immutable construction."""

    solver: KWNSolver
    mapping: dict[str, Any]
    edges_m: np.ndarray
    cell_number_m3: np.ndarray
    contract_hash: str
    fixture_hash: str
    source_initial_psd_hash: str
    canonical_hash: str
    median_radius_m: float
    log_sigma: float
    cell_number_roundtrip_relative_error: float


@dataclass
class EulerianRun:
    """Accepted-state telemetry for one policy/horizon calculation."""

    policy: str
    requested_active_cfl: float | None
    status: str
    reason: str | None
    snapshots: list[dict[str, Any]]
    cfl_rows: list[dict[str, Any]]
    lower_tail_rows: list[dict[str, Any]]
    accepted_steps: int
    rejected_steps: int
    runtime_s: float
    cumulative_number_m3: float
    cumulative_beta_volume: float
    cumulative_mol_b_mol_m3: float
    feasibility: dict[str, Any]


class WorkflowError(RuntimeError):
    """Raised for a gate that cannot be honestly evaluated."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        digest.update(np.ascontiguousarray(np.asarray(array, dtype=np.float64)).tobytes())
    return digest.hexdigest()


def _json_safe(value: Any) -> Any:
    """Convert numpy scalars and non-finite diagnostics for JSON evidence."""

    if isinstance(value, np.generic):
        return _json_safe(value.item())
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


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], *, fallback_fields: Sequence[str] = ()) -> None:
    """Write a stable, inspectable CSV even when a gate is intentionally blocked."""

    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    if not fields:
        fields = list(fallback_fields) or ["status", "reason"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(row.get(key, "")) for key in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(_json_safe(value), sort_keys=True)
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return str(value)
    return value


def _write_report(report_root: Path, name: str, title: str, body: str) -> None:
    path = report_root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def _write_unfilled_reports(report_root: Path, completed: Mapping[str, str]) -> None:
    """Ensure every required report has a truthful gate-state placeholder."""

    for name, title in REQUIRED_REPORTS.items():
        if name not in completed:
            _write_report(
                report_root,
                name,
                title,
                "Status: `BLOCKED_OR_NOT_RUN`.  This artifact is a required status stub; "
                "the preceding numerical gate has not supplied admissible evidence yet.",
            )


def _relative_error(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(abs(float(right)), 1.0e-300)


def _absolute_or_relative_error(left: float, right: float) -> float:
    if left == 0.0 and right == 0.0:
        return 0.0
    return _relative_error(left, right)


def _git_text(*args: str) -> str:
    process = subprocess.run(
        ["git", *args], cwd=ROOT, text=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
    )
    return process.stdout.strip()


def _launch_context() -> dict[str, Any]:
    """Capture source/binary provenance without treating KWN edits as PF edits."""

    changed = [item for item in _git_text("diff", "--name-only", BASELINE_COMMIT).splitlines() if item]
    physical_paths = {
        "src/kwn_mvp/growth.py",
        "src/kwn_mvp/thermo_adapter.py",
        "src/kwn_mvp/composition_mapping.py",
    }
    pf_prefixes = ("cuda/", "src/pf/", "src/phase_field/", "phasefield/")
    physical_retuning = [
        path for path in changed
        if path.startswith(("contracts/", "configs/", "data/")) or path in physical_paths
    ]
    pf_changes = [
        path for path in changed
        if path.startswith(pf_prefixes) or Path(path).suffix.lower() in {".cu", ".cuh", ".cuf"}
    ]
    return {
        "baseline_commit": BASELINE_COMMIT,
        "baseline_commit_resolved": _git_text("rev-parse", BASELINE_COMMIT),
        "git_head_at_launch": _git_text("rev-parse", "HEAD"),
        "git_branch_at_launch": _git_text("branch", "--show-current"),
        "git_status_at_launch": _git_text("status", "--short"),
        "source_paths_changed_from_baseline": changed,
        "pf_source_paths_changed_from_baseline": pf_changes,
        "pf_source_modified": bool(pf_changes),
        "cuda_rerun": False,
        "physical_input_paths_modified": physical_retuning,
        "physical_retuning": bool(physical_retuning),
        "local_gp_release": False,
    }


def _load_radius_audit_module() -> Any:
    path = ROOT / "scripts" / "run_kwn_radius_grid_convergence_v1.py"
    specification = importlib.util.spec_from_file_location("radius_grid_audit_lower_boundary_v1", path)
    if specification is None or specification.loader is None:
        raise WorkflowError(f"cannot load {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _lognormal_cell_moment(
    edges_m: np.ndarray, *, median_m: float, log_sigma: float, order: int
) -> np.ndarray:
    mu = math.log(float(median_m))
    sigma = float(log_sigma)
    shift = float(order) * sigma**2
    prefactor = math.exp(float(order) * mu + 0.5 * float(order * order) * sigma**2)
    z = (np.log(np.asarray(edges_m, dtype=np.float64)) - mu - shift) / sigma
    return prefactor * np.diff(ndtr(z))


def _build_canonical_context(*, bins: int = 3200) -> CanonicalContext:
    """Build the frozen smooth measure directly as a cell-integrated state."""

    audit = _load_radius_audit_module()
    source_solver, construction, contract, fixture = audit._build_smooth_solver(bins=bins)
    if str(contract.contract_hash) != EXPECTED_CONTRACT_HASH:
        raise WorkflowError(
            "canonical source validation-contract hash differs from the frozen lower-boundary task contract"
        )
    beta = source_solver.population("beta")
    benchmark = construction["smooth_psd_benchmark"]
    edges = beta.grid.edges_m.copy()
    median = float(benchmark["median_radius_m"])
    sigma = float(benchmark["log_sigma"])
    raw_number = _lognormal_cell_moment(edges, median_m=median, log_sigma=sigma, order=0)
    raw_moments = cell_moments_from_piecewise_constant_cells(edges, raw_number)
    target_m3 = float(benchmark["same_fixed_pivot_M3_as_fixture"])
    scale = target_m3 / float(np.sum(raw_moments[3]))
    cell_number = raw_number * scale
    mapping = deepcopy(construction["config_mapping"])
    mapping["simulation"]["population_measure"] = "cell_integrated"
    # This runner applies active-CFL caps at the caller level.  It must never
    # accidentally re-enable the legacy raw-empty-bin accuracy limiter.
    mapping["simulation"]["accuracy_radius_cfl"] = None
    mapping["simulation"]["accuracy_active_radius_cfl"] = None
    mapping["populations"]["beta"]["initial"] = {
        "kind": "cell_integrated",
        "radius_edges_m": [float(item) for item in edges],
        "cell_number_density_m3": [float(item) for item in cell_number],
    }
    solver = KWNSolver(SolverConfig.from_mapping(mapping))
    constructed = solver.population("beta")
    observed = constructed.number_density_per_m4 * constructed.grid.widths_m
    # Dividing a binary64 cell number by its width and multiplying it back is
    # not bitwise identity for every cell.  This is representation round-off,
    # not a second initial measure; retain a machine-scale quantitative gate.
    canonical_roundtrip_error = float(
        np.max(np.abs(observed - cell_number) / np.maximum(np.abs(cell_number), 1.0e-300))
    )
    if canonical_roundtrip_error > 2.0e-15:
        raise WorkflowError(
            f"canonical cell-integrated round-trip differs by {canonical_roundtrip_error:.3e}"
        )
    canonical_hash = _array_hash(edges, cell_number, cell_moments_from_piecewise_constant_cells(edges, cell_number))
    return CanonicalContext(
        solver=solver,
        mapping=mapping,
        edges_m=edges,
        cell_number_m3=cell_number,
        contract_hash=str(contract.contract_hash),
        fixture_hash=str(fixture.fixture_hash),
        source_initial_psd_hash=str(construction["initial_psd_source_hash"]),
        canonical_hash=canonical_hash,
        median_radius_m=median,
        log_sigma=sigma,
        cell_number_roundtrip_relative_error=canonical_roundtrip_error,
    )


def _baseline_reproduction(output_root: Path) -> dict[str, Any]:
    """Verify the pre-change trace instead of rerunning its historical source."""

    trace = output_root / "baseline_boundary_trace.csv"
    state = output_root / "baseline_boundary_state.npz"
    provenance_path = output_root / "baseline_boundary_trace_provenance.json"
    missing = [str(path) for path in (trace, state, provenance_path) if not path.is_file()]
    if missing:
        return {
            "status": "FAIL_BASELINE_REPRODUCTION",
            "reason": "missing immutable pre-change boundary trace artifact",
            "missing": missing,
        }
    try:
        provenance = json.loads(provenance_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        return {"status": "FAIL_BASELINE_REPRODUCTION", "reason": f"invalid provenance: {error}"}
    checks = {
        "historical_status": provenance.get("baseline_status") == "FAIL_LOWER_BOUNDARY_OPERATOR_PARITY",
        "source_commit": provenance.get("source_commit") == BASELINE_COMMIT,
        "contract": provenance.get("validation_contract_hash") == EXPECTED_CONTRACT_HASH,
        "trace_sha256": provenance.get("trace_csv_sha256") == _sha256_file(trace),
        "state_sha256": provenance.get("state_npz_sha256") == _sha256_file(state),
        "physical_rmin_binary64": isinstance(provenance.get("physical_rmin_hex"), str),
    }
    return {
        "schema_version": "KWN_LOWER_BOUNDARY_BASELINE_REPRODUCTION_V1",
        "status": "PASS_BASELINE_REPRODUCTION" if all(checks.values()) else "FAIL_BASELINE_REPRODUCTION",
        "checks": checks,
        "trace": str(trace),
        "state": str(state),
        "provenance": provenance,
    }


def _boundary_contract(context: CanonicalContext, baseline: Mapping[str, Any]) -> dict[str, Any]:
    beta = context.solver.population("beta")
    rmin = boundary_radius(beta.grid)
    baseline_hex = None
    if isinstance(baseline.get("provenance"), Mapping):
        baseline_hex = baseline["provenance"].get("physical_rmin_hex")
    return {
        "schema_version": "KWN_PHYSICAL_LOWER_BOUNDARY_CONTRACT_V1",
        "validation_contract_hash": context.contract_hash,
        "radius_domain_m": [float(beta.grid.edges_m[0]), float(beta.grid.edges_m[-1])],
        "R_boundary_m": rmin,
        "R_boundary_binary64_hex": float(rmin).hex(),
        "baseline_R_boundary_binary64_hex": baseline_hex,
        "lower_face_location": "Rmin_exact_grid_edge",
        "boundary_growth_kernel": "shared_growth_rate_beta",
        "boundary_growth_velocity_expression": "growth_rate_beta(R=Rmin, x_alpha, T, validation_contract)",
        "outward_radius_direction": "negative_R",
        "J_N_out_positive": True,
        "outflow_rule": "G_boundary_lt_0",
        "external_inflow_rule": "zero_when_G_boundary_ge_0_without_subgrid_source",
        "upwind_reconstruction": "piecewise_constant_first_resolved_cell_donor",
        "ghost_population": "forbidden",
        "particle_inventory_price_radius": "Rmin_exact_grid_edge",
        "inventory_authority": "Q_total = Q_matrix + Q_beta_resolved",
        "matrix_closure": "algebraic_from_current_population_only",
        "boundary_diagnostics_are_matrix_sources": False,
        "GP_release": "not_authorized",
    }


def _flatten_suite(suite: unittest.TestSuite) -> list[unittest.TestCase]:
    result: list[unittest.TestCase] = []
    for item in suite:
        if isinstance(item, unittest.TestSuite):
            result.extend(_flatten_suite(item))
        else:
            result.append(item)
    return result


def _boundary_unit_tests() -> dict[str, Any]:
    """Run B1--B7 in-process so the CSV identifies the exact failing gate."""

    module_name = "tests.kwn.test_lower_boundary_contract"
    try:
        module = importlib.import_module(module_name)
        suite = unittest.defaultTestLoader.loadTestsFromModule(module)
    except Exception:
        return {
            "status": "FAIL_LOWER_BOUNDARY_UNIT_TESTS",
            "reason": "cannot import B1--B7 boundary test module",
            "traceback": traceback.format_exc(),
            "rows": [
                {"gate": f"B{index}", "status": "NOT_RUN", "reason": "test module import failed"}
                for index in range(1, 8)
            ],
        }
    tests = _flatten_suite(suite)
    stream = io.StringIO()
    result = unittest.TextTestRunner(stream=stream, verbosity=2).run(suite)
    failures = {item.id(): detail for item, detail in result.failures}
    errors = {item.id(): detail for item, detail in result.errors}
    skipped = {item.id(): detail for item, detail in result.skipped}
    expected: dict[str, list[str]] = {f"B{index}": [] for index in range(1, 8)}
    for test in tests:
        identifier = test.id()
        match = re.search(r"(?:^|[._])b([1-7])(?:[_]|$)", identifier.lower())
        if match:
            expected[f"B{match.group(1)}"].append(identifier)
    rows: list[dict[str, Any]] = []
    all_pass = True
    for gate, identifiers in expected.items():
        if not identifiers:
            all_pass = False
            rows.append({"gate": gate, "test_id": "", "status": "MISSING", "reason": "no B-named unittest found"})
            continue
        for identifier in identifiers:
            if identifier in failures:
                status, detail = "FAIL", failures[identifier]
            elif identifier in errors:
                status, detail = "ERROR", errors[identifier]
            elif identifier in skipped:
                status, detail = "SKIPPED", skipped[identifier]
            else:
                status, detail = "PASS", ""
            all_pass &= status == "PASS"
            rows.append({"gate": gate, "test_id": identifier, "status": status, "reason": detail[-4000:]})
    return {
        "status": "PASS_LOWER_BOUNDARY_UNIT_TESTS" if all_pass else "FAIL_LOWER_BOUNDARY_UNIT_TESTS",
        "module": module_name,
        "test_count": result.testsRun,
        "runner_output": stream.getvalue()[-12000:],
        "rows": rows,
    }


def _minimal_contiguous_support(weights: np.ndarray, fraction: float = SUPPORT_FRACTION) -> tuple[int, int] | None:
    """Smallest log-radius interval carrying the declared conserved fraction."""

    values = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(values))
    if total <= 0.0:
        return None
    target = fraction * total
    right = 0
    running = 0.0
    best: tuple[int, int, int] | None = None
    for left in range(values.size):
        while right < values.size and running < target:
            running += float(values[right])
            right += 1
        if running >= target:
            candidate = (right - left, left, right - 1)
            if best is None or candidate < best:
                best = candidate
        running -= float(values[left])
    return None if best is None else (best[1], best[2])


def _quantile_interval(weights: np.ndarray, quantiles: tuple[float, float]) -> tuple[int, int] | None:
    values = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(values))
    if total <= 0.0:
        return None
    cdf = np.cumsum(values) / total
    lower = int(np.searchsorted(cdf, quantiles[0], side="left"))
    upper = int(np.searchsorted(cdf, quantiles[1], side="left"))
    lower = min(max(lower, 0), values.size - 1)
    upper = min(max(upper, lower), values.size - 1)
    return lower, upper


def _face_cfls(face_velocity_m_s: np.ndarray, widths_m: np.ndarray, dt_s: float) -> np.ndarray:
    faces = np.asarray(face_velocity_m_s, dtype=np.float64)
    widths = np.asarray(widths_m, dtype=np.float64)
    values = np.empty(faces.size, dtype=np.float64)
    values[0] = abs(faces[0]) * dt_s / widths[0]
    values[-1] = abs(faces[-1]) * dt_s / widths[-1]
    if widths.size > 1:
        values[1:-1] = abs(faces[1:-1]) * dt_s / np.minimum(widths[:-1], widths[1:])
    return values


def _support_face_cfl(face_cfl: np.ndarray, support: tuple[int, int] | None) -> float:
    if support is None:
        return 0.0
    lower, upper = support
    return float(np.max(face_cfl[lower : upper + 2]))


def _cfl_audit(solver: KWNSolver, *, dt_s: float) -> dict[str, Any]:
    """Compute raw and active CFLs from the *actual* physical face operator."""

    beta = solver.population("beta")
    cell_number = beta.number_density_per_m4 * beta.grid.widths_m
    cell_moments = cell_moments_from_piecewise_constant_cells(beta.grid.edges_m, cell_number)
    velocities = solver.growth_rates()["beta"]
    faces = solver.face_velocities(beta, velocities)
    # The solver owns the active-CFL contract.  Reusing it avoids a runner
    # side approximation accidentally changing the support tie-break,
    # reachable-face closure, or the physical-Rmin activation criterion.
    if hasattr(solver, "active_cfl_diagnostics"):
        active = solver.active_cfl_diagnostics(
            beta, face_velocity_m_s=faces, dt_s=dt_s
        )
        m0_lo, m0_hi = int(active.m0_support_i_lo), int(active.m0_support_i_hi)
        m3_lo, m3_hi = int(active.m3_support_i_lo), int(active.m3_support_i_hi)
        tail_lo, tail_hi = int(active.lower_tail_i_lo), int(active.lower_tail_i_hi)
        total_m0 = float(np.sum(cell_moments[0]))
        total_m3 = float(np.sum(cell_moments[3]))
        first_m0 = 0.0 if total_m0 == 0.0 else float(cell_moments[0, 0] / total_m0)
        first_m3 = 0.0 if total_m3 == 0.0 else float(cell_moments[3, 0] / total_m3)
        rates, _ = solver._face_operator_rates(faces, beta.grid.widths_m)
        global_index = int(np.argmax(rates))
        return {
            "dt_s": float(dt_s),
            "global_max_cfl": float(active.global_max_rate_s_inv * dt_s),
            "global_face_index": global_index,
            "global_face_velocity_m_s": float(faces[global_index]),
            "active_M0_cfl": float(active.m0_rate_s_inv * dt_s),
            "active_M3_cfl": float(active.m3_rate_s_inv * dt_s),
            "population_active_cfl": float(active.population_active_rate_s_inv * dt_s),
            "lower_tail_active_cfl": float(active.lower_tail_rate_s_inv * dt_s),
            "boundary_candidate_cfl": float(active.boundary_candidate_rate_s_inv * dt_s),
            "boundary_active_cfl": float(active.boundary_active_rate_s_inv * dt_s),
            "boundary_active": bool(active.boundary_face_is_active),
            "boundary_flux_always_executed": True,
            "M0_support_low_cell": m0_lo,
            "M0_support_high_cell": m0_hi,
            "M0_support_low_radius_m": "" if m0_lo < 0 else float(beta.grid.edges_m[m0_lo]),
            "M0_support_high_radius_m": "" if m0_hi < 0 else float(beta.grid.edges_m[m0_hi + 1]),
            "M3_support_low_cell": m3_lo,
            "M3_support_high_cell": m3_hi,
            "M3_support_low_radius_m": "" if m3_lo < 0 else float(beta.grid.edges_m[m3_lo]),
            "M3_support_high_radius_m": "" if m3_hi < 0 else float(beta.grid.edges_m[m3_hi + 1]),
            "lower_tail_low_cell": tail_lo,
            "lower_tail_high_cell": tail_hi,
            "lower_tail_low_radius_m": "" if tail_lo < 0 else float(beta.grid.edges_m[tail_lo]),
            "lower_tail_high_radius_m": "" if tail_hi < 0 else float(beta.grid.edges_m[tail_hi + 1]),
            "first_cell_M0_fraction": first_m0,
            "first_cell_M3_fraction": first_m3,
            "lower_boundary_radius_m": float(beta.grid.edges_m[0]),
            "lower_boundary_velocity_m_s": float(faces[0]),
            "active_definition": "solver_active_cfl_diagnostics: smallest_contiguous_99.9999_percent_M0_or_M3_support_plus_reachable_faces",
            "lower_tail_definition": "solver_active_cfl_diagnostics: M0_quantile_band_[1e-8,1e-6]_plus_reachable_faces",
            "boundary_active_definition": "solver_active_cfl_diagnostics: quantile_tail_characteristic_reaches_Rmin",
            "M0_covered_fraction": float(active.m0_covered_fraction),
            "M3_covered_fraction": float(active.m3_covered_fraction),
            "lower_tail_M0_fraction": float(active.lower_tail_m0_fraction),
        }
    # Compatibility fallback for an older in-progress source revision.  The
    # final task gate records the solver-owned implementation above.
    face_cfl = _face_cfls(faces, beta.grid.widths_m, dt_s)
    m0_support = _minimal_contiguous_support(cell_moments[0])
    m3_support = _minimal_contiguous_support(cell_moments[3])
    tail_support = _quantile_interval(cell_moments[0], LOWER_TAIL_QUANTILES)
    # The boundary does not become an *active-population accuracy limiter*
    # merely because a lognormal has a nonzero underflow-scale first bin.  It
    # activates when the declared quantile tail reaches the first cell or can
    # reach Rmin during this frozen-operator step.  Flux remains active in
    # the FV update regardless of this diagnostic flag.
    rmin = float(beta.grid.edges_m[0])
    boundary_active = False
    if tail_support is not None:
        lo, hi = tail_support
        tail_indices = np.arange(lo, hi + 1)
        can_reach = (velocities[tail_indices] < 0.0) & (
            beta.grid.centres_m[tail_indices] - rmin <= -velocities[tail_indices] * dt_s
        )
        boundary_active = lo == 0 or bool(np.any(can_reach))
    global_index = int(np.argmax(face_cfl))
    m0_lo, m0_hi = (-1, -1) if m0_support is None else m0_support
    m3_lo, m3_hi = (-1, -1) if m3_support is None else m3_support
    tail_lo, tail_hi = (-1, -1) if tail_support is None else tail_support
    total_m0 = float(np.sum(cell_moments[0]))
    total_m3 = float(np.sum(cell_moments[3]))
    first_m0 = 0.0 if total_m0 == 0.0 else float(cell_moments[0, 0] / total_m0)
    first_m3 = 0.0 if total_m3 == 0.0 else float(cell_moments[3, 0] / total_m3)
    return {
        "dt_s": float(dt_s),
        "global_max_cfl": float(face_cfl[global_index]),
        "global_face_index": global_index,
        "global_face_velocity_m_s": float(faces[global_index]),
        "active_M0_cfl": _support_face_cfl(face_cfl, m0_support),
        "active_M3_cfl": _support_face_cfl(face_cfl, m3_support),
        "population_active_cfl": max(
            _support_face_cfl(face_cfl, m0_support), _support_face_cfl(face_cfl, m3_support)
        ),
        "lower_tail_active_cfl": _support_face_cfl(face_cfl, tail_support),
        "boundary_active_cfl": float(face_cfl[0]) if boundary_active else 0.0,
        "boundary_active": boundary_active,
        "boundary_flux_always_executed": True,
        "M0_support_low_cell": m0_lo,
        "M0_support_high_cell": m0_hi,
        "M0_support_low_radius_m": "" if m0_lo < 0 else float(beta.grid.edges_m[m0_lo]),
        "M0_support_high_radius_m": "" if m0_hi < 0 else float(beta.grid.edges_m[m0_hi + 1]),
        "M3_support_low_cell": m3_lo,
        "M3_support_high_cell": m3_hi,
        "M3_support_low_radius_m": "" if m3_lo < 0 else float(beta.grid.edges_m[m3_lo]),
        "M3_support_high_radius_m": "" if m3_hi < 0 else float(beta.grid.edges_m[m3_hi + 1]),
        "lower_tail_low_cell": tail_lo,
        "lower_tail_high_cell": tail_hi,
        "lower_tail_low_radius_m": "" if tail_lo < 0 else float(beta.grid.edges_m[tail_lo]),
        "lower_tail_high_radius_m": "" if tail_hi < 0 else float(beta.grid.edges_m[tail_hi + 1]),
        "first_cell_M0_fraction": first_m0,
        "first_cell_M3_fraction": first_m3,
        "lower_boundary_radius_m": rmin,
        "lower_boundary_velocity_m_s": float(faces[0]),
        "active_definition": "smallest_contiguous_99.9999_percent_M0_or_M3_support_plus_incident_faces",
        "lower_tail_definition": "M0_quantile_band_[1e-8,1e-6]_plus_incident_faces",
        "boundary_active_definition": "quantile_tail_touches_or_reaches_Rmin_during_frozen_step",
    }


def _critical_radius_m(solver: KWNSolver) -> float | None:
    beta = solver.population("beta")
    lower, upper = (float(beta.grid.edges_m[0]), float(beta.grid.edges_m[-1]))

    def residual(radius_m: float) -> float:
        equilibrium = solver.equilibrium_adapter.equilibrium_xb(
            np.asarray([radius_m], dtype=np.float64), beta.parameters
        )
        return solver.matrix_xb - float(equilibrium[0])

    low, high = residual(lower), residual(upper)
    if low == 0.0:
        return lower
    if high == 0.0:
        return upper
    if low * high > 0.0:
        return None
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        value = residual(middle)
        if value == 0.0:
            return middle
        if low * value < 0.0:
            upper, high = middle, value
        else:
            lower, low = middle, value
    return 0.5 * (lower + upper)


def _eulerian_snapshot(
    solver: KWNSolver,
    *,
    policy: str,
    cumulative_number_m3: float,
    cumulative_beta_volume: float,
    cumulative_mol_b_mol_m3: float,
) -> dict[str, Any]:
    beta = solver.population("beta")
    cell_number = beta.number_density_per_m4 * beta.grid.widths_m
    metrics = metrics_from_piecewise_constant_cells(beta.grid.edges_m, cell_number)
    ledger = solver.ledger.snapshot(
        matrix_xb=solver.matrix_xb,
        populations=solver.population_list(),
        beta_resolved_fraction=1.0,
    )
    return {
        "policy": policy,
        "time_s": float(solver.time_s),
        "time_h": float(solver.time_s / 3600.0),
        "accepted_steps": int(solver.step),
        **metrics.as_dict(),
        "Rmean_m": metrics.Rmean_number_m,
        "Rmean3_m3": metrics.Rmean_cubed_m3,
        "matrix_xB": float(solver.matrix_xb),
        "beta_inventory_mol_m3": float(ledger.beta_resolved_mol_m3),
        "matrix_inventory_mol_m3": float(ledger.matrix_mol_m3),
        "total_inventory_mol_m3": float(ledger.total_mol_m3),
        "inventory_residual_mol_m3": float(ledger.residual_mol_m3),
        "inventory_relative_residual": float(ledger.relative_residual),
        "critical_radius_m": _critical_radius_m(solver),
        "cumulative_number_dissolution_m3": cumulative_number_m3,
        "cumulative_beta_volume_dissolution": cumulative_beta_volume,
        "cumulative_mol_B_returned_mol_m3": cumulative_mol_b_mol_m3,
        "first_cell_number_m3": float(cell_number[0]),
    }


def _active_cfl_rate(solver: KWNSolver) -> float:
    audit = _cfl_audit(solver, dt_s=1.0)
    return max(
        float(audit["population_active_cfl"]),
        float(audit["lower_tail_active_cfl"]),
        float(audit["boundary_active_cfl"]),
    )


def _advance_eulerian(
    solver: KWNSolver,
    *,
    policy: str,
    target_times_h: Sequence[float],
    active_cfl_cap: float | None,
    max_steps: int,
    max_wall_s: float,
) -> EulerianRun:
    """Advance with a caller-controlled active CFL, never a raw-empty-bin cap."""

    snapshots: list[dict[str, Any]] = []
    cfl_rows: list[dict[str, Any]] = []
    lower_tail_rows: list[dict[str, Any]] = []
    cumulative_number = 0.0
    cumulative_volume = 0.0
    cumulative_mol_b = 0.0
    started = time.monotonic()
    try:
        targets = tuple(float(item) * 3600.0 for item in target_times_h)
        if not targets or targets[0] != 0.0:
            raise WorkflowError("Eulerian diagnostic schedule must begin at t=0")
        snapshots.append(
            _eulerian_snapshot(
                solver,
                policy=policy,
                cumulative_number_m3=cumulative_number,
                cumulative_beta_volume=cumulative_volume,
                cumulative_mol_b_mol_m3=cumulative_mol_b,
            )
        )
        for target_s in targets[1:]:
            if target_s < solver.time_s:
                raise WorkflowError("Eulerian output schedule is not monotonic")
            while solver.time_s < target_s:
                if time.monotonic() - started >= max_wall_s:
                    raise WorkflowError(
                        f"diagnostic wall budget max_wall_s={max_wall_s:g} exhausted before {target_s / 3600.0:g} h"
                    )
                if solver.step >= max_steps:
                    raise WorkflowError(
                        f"policy reached max_steps={max_steps} before {target_s / 3600.0:g} h"
                    )
                remaining = target_s - solver.time_s
                # In the final solver, ``accuracy_active_radius_cfl`` is the
                # authoritative fixed-point limiter.  It includes the M0/M3
                # reachable support, lower-tail faces, and physical boundary
                # whenever the tail can reach Rmin.  The compatibility branch
                # retains the same max() rule rather than capping M0/M3 only.
                maximum_dt = remaining
                if active_cfl_cap is not None and getattr(
                    solver.config, "accuracy_active_radius_cfl", None
                ) is None:
                    rate = _active_cfl_rate(solver)
                    if rate > 0.0:
                        maximum_dt = min(maximum_dt, active_cfl_cap / rate)
                before = _cfl_audit(solver, dt_s=0.0)
                diagnostic = solver.advance_one(maximum_dt_s=maximum_dt)
                if hasattr(diagnostic, "active_population_courant"):
                    # StepDiagnostics is calculated by the accepted
                    # start-state fixed point, which is the only honest
                    # telemetry for a dynamic support mask.
                    before.update({
                        "dt_s": float(diagnostic.dt_s),
                        "global_max_cfl": float(diagnostic.radius_courant_max),
                        "active_M0_cfl": float(diagnostic.active_m0_courant),
                        "active_M3_cfl": float(diagnostic.active_m3_courant),
                        "population_active_cfl": float(diagnostic.active_population_courant),
                        "lower_tail_active_cfl": float(diagnostic.lower_tail_active_courant),
                        "boundary_candidate_cfl": float(diagnostic.boundary_candidate_courant),
                        "boundary_active_cfl": float(diagnostic.boundary_active_courant),
                        "boundary_active": bool(diagnostic.boundary_face_is_active),
                        "M0_support_low_cell": int(diagnostic.active_m0_support_i_lo),
                        "M0_support_high_cell": int(diagnostic.active_m0_support_i_hi),
                        "M3_support_low_cell": int(diagnostic.active_m3_support_i_lo),
                        "M3_support_high_cell": int(diagnostic.active_m3_support_i_hi),
                        "lower_tail_low_cell": int(diagnostic.lower_tail_i_lo),
                        "lower_tail_high_cell": int(diagnostic.lower_tail_i_hi),
                        "lower_tail_M0_fraction": float(diagnostic.lower_tail_m0_fraction),
                    })
                else:
                    # Older source revisions have no accepted-step active
                    # telemetry.  The direct audit still includes lower tail
                    # and boundary terms, all linear in the accepted dt.
                    before = _cfl_audit(solver, dt_s=float(diagnostic.dt_s))
                cumulative_number += float(getattr(diagnostic, "beta_rmin_number_flux_m3_s", 0.0)) * diagnostic.dt_s
                cumulative_volume += float(getattr(diagnostic, "beta_rmin_volume_flux_s", 0.0)) * diagnostic.dt_s
                cumulative_mol_b += float(getattr(diagnostic, "beta_rmin_mol_b_flux_mol_m3_s", 0.0)) * diagnostic.dt_s
                row = {
                    "policy": policy,
                    "step": int(diagnostic.step),
                    "time_start_s": float(diagnostic.time_s - diagnostic.dt_s),
                    "time_end_s": float(diagnostic.time_s),
                    "time_end_h": float(diagnostic.time_s / 3600.0),
                    "requested_active_cfl_cap": "UNBOUNDED" if active_cfl_cap is None else active_cfl_cap,
                    "solver_timestep_limiter": str(getattr(diagnostic, "timestep_limiter", "unknown")),
                    "legacy_size_cfl": float(getattr(diagnostic, "size_cfl", 0.0)),
                    "raw_face_cfl_reported_by_solver": float(getattr(diagnostic, "radius_courant_max", 0.0)),
                    "cumulative_number_dissolution_m3": cumulative_number,
                    "cumulative_beta_volume_dissolution": cumulative_volume,
                    "cumulative_mol_B_returned_mol_m3": cumulative_mol_b,
                    **before,
                }
                cfl_rows.append(row)
                lower_tail_rows.append({
                    key: row[key] for key in (
                        "policy", "step", "time_start_s", "time_end_s", "time_end_h", "dt_s",
                        "lower_tail_active_cfl", "boundary_active_cfl", "boundary_active",
                        "lower_tail_low_cell", "lower_tail_high_cell", "lower_tail_low_radius_m",
                        "lower_tail_high_radius_m", "first_cell_M0_fraction", "first_cell_M3_fraction",
                        "cumulative_number_dissolution_m3", "cumulative_beta_volume_dissolution",
                        "cumulative_mol_B_returned_mol_m3",
                    )
                })
            snapshots.append(
                _eulerian_snapshot(
                    solver,
                    policy=policy,
                    cumulative_number_m3=cumulative_number,
                    cumulative_beta_volume=cumulative_volume,
                    cumulative_mol_b_mol_m3=cumulative_mol_b,
                )
            )
    except Exception as error:
        # Preserve the actually accepted terminal state when a bounded
        # diagnostic stops between scheduled output times.  Without this
        # snapshot the initial t=0 row would overwrite the run-level step
        # count in the ladder report, making real partial progress look like
        # an unattempted policy.
        if not snapshots or float(snapshots[-1]["time_s"]) != float(solver.time_s):
            partial = _eulerian_snapshot(
                solver,
                policy=policy,
                cumulative_number_m3=cumulative_number,
                cumulative_beta_volume=cumulative_volume,
                cumulative_mol_b_mol_m3=cumulative_mol_b,
            )
            partial["snapshot_status"] = "INCOMPLETE_TERMINAL"
            snapshots.append(partial)
        active = _cfl_audit(solver, dt_s=1.0)
        required_rate = max(
            float(active["population_active_cfl"]),
            float(active["lower_tail_active_cfl"]),
            float(active["boundary_active_cfl"]),
        )
        audit_implied_dt = (
            float("inf") if active_cfl_cap is None or required_rate == 0.0
            else float(active_cfl_cap / required_rate)
        )
        error_text = str(error)
        min_dt_match = re.search(r"CFL-limited step\s+([0-9.+-eE]+)\s+s\s+is below configured min_dt", error_text)
        solver_limited_dt = None if min_dt_match is None else float(min_dt_match.group(1))
        # A wall-clock diagnostic budget, accepted-state audit, or static
        # ``cap / rate`` calculation is not evidence that the source-owned
        # active-CFL fixed point is infeasible: reducing a candidate step can
        # make the Rmin tail unreachable and release the boundary limiter.
        # Reserve ``BRUTE_FORCE`` exclusively for a min-dt rejection emitted
        # by the actual candidate-step chooser.
        infeasible = active_cfl_cap is not None and min_dt_match is not None
        feasibility = {
            "terminal_audit_active_rate_s_inv": required_rate,
            "terminal_audit_implied_candidate_dt_s": audit_implied_dt,
            "configured_min_dt_s": float(solver.config.min_dt_s),
            "solver_reported_cfl_limited_dt_s": solver_limited_dt,
            "boundary_active": bool(active["boundary_active"]),
            # This is a frozen-state rate audit at dt=1 s.  It is telemetry
            # only, never a substitute for an accepted candidate-step
            # decision or a universal feasibility bound.
            "terminal_cfl_audit": active,
            "static_audit_is_not_a_feasibility_bound": True,
        }
        return EulerianRun(
            policy=policy,
            requested_active_cfl=active_cfl_cap,
            status=(
                "BRUTE_FORCE_CFL_NOT_PRACTICAL"
                if infeasible
                else "INCOMPLETE_DIAGNOSTIC_BUDGET"
            ),
            reason=f"{type(error).__name__}: {error}; feasibility={json.dumps(_json_safe(feasibility), sort_keys=True)}",
            snapshots=snapshots,
            cfl_rows=cfl_rows,
            lower_tail_rows=lower_tail_rows,
            accepted_steps=len(cfl_rows),
            rejected_steps=0,
            runtime_s=time.monotonic() - started,
            cumulative_number_m3=cumulative_number,
            cumulative_beta_volume=cumulative_volume,
            cumulative_mol_b_mol_m3=cumulative_mol_b,
            feasibility=feasibility,
        )
    terminal_audit = _cfl_audit(solver, dt_s=1.0)
    return EulerianRun(
        policy=policy,
        requested_active_cfl=active_cfl_cap,
        status="PASS_EULERIAN_SHORT_RUN",
        reason=None,
        snapshots=snapshots,
        cfl_rows=cfl_rows,
        lower_tail_rows=lower_tail_rows,
        accepted_steps=len(cfl_rows),
        rejected_steps=0,
        runtime_s=time.monotonic() - started,
        cumulative_number_m3=cumulative_number,
        cumulative_beta_volume=cumulative_volume,
        cumulative_mol_b_mol_m3=cumulative_mol_b,
        feasibility={
            "terminal_audit_active_rate_s_inv": max(
                float(terminal_audit["population_active_cfl"]),
                float(terminal_audit["lower_tail_active_cfl"]),
                float(terminal_audit["boundary_active_cfl"]),
            ),
            "terminal_audit_implied_candidate_dt_s": (
                None
                if active_cfl_cap is None
                else float(active_cfl_cap) / max(
                    float(terminal_audit["population_active_cfl"]),
                    float(terminal_audit["lower_tail_active_cfl"]),
                    float(terminal_audit["boundary_active_cfl"]),
                )
            ),
            "configured_min_dt_s": float(solver.config.min_dt_s),
            "boundary_active": bool(terminal_audit["boundary_active"]),
            "terminal_cfl_audit": terminal_audit,
            "static_audit_is_not_a_feasibility_bound": True,
        },
    )


def _build_canonical_solver_from_context(
    context: CanonicalContext, *, active_cfl_cap: float | None = None
) -> KWNSolver:
    mapping = deepcopy(context.mapping)
    mapping["simulation"]["accuracy_active_radius_cfl"] = active_cfl_cap
    return KWNSolver(SolverConfig.from_mapping(mapping))


def _cohort_from_context(context: CanonicalContext, *, points_per_cell: int) -> CohortSolver:
    """Create a full-M3-preserving discrete measure; no radius-bin projection."""

    radii, weights = positive_cell_quadrature(context.edges_m, context.cell_number_m3, points_per_cell)
    cohorts = [
        Cohort(initial_id=f"C{index:05d}", radius_m=float(radius), weight_m3=float(weight))
        for index, (radius, weight) in enumerate(zip(radii, weights))
    ]
    return CohortSolver.from_kwn_solver(
        kwn_solver=_build_canonical_solver_from_context(context),
        cohorts=cohorts,
        rtol=1.0e-10,
        atol_m=1.0e-18,
        method="DOP853",
    )


def _physical_boundary_parity(context: CanonicalContext) -> dict[str, Any]:
    """Compare helper, FV face and cohort physical quantities at exact Rmin."""

    solver = _build_canonical_solver_from_context(context)
    beta = solver.population("beta")
    rmin = boundary_radius(beta.grid)
    velocity = solver.growth_rates()["beta"]
    face_velocity = solver.face_velocities(beta, velocity)
    helper_velocity = boundary_growth_velocity(
        radius_m=rmin,
        matrix_xb=solver.matrix_xb,
        parameters=beta.parameters,
        equilibrium_adapter=solver.equilibrium_adapter,
    )
    euler_velocity = solver.lower_boundary_growth_velocity("beta")
    density = float(beta.number_density_per_m4[0])
    helper_number_flux = boundary_number_flux_diagnostic(helper_velocity, density)
    actual_signed_flux = float(
        solver._upwind_face_fluxes(
            beta.number_density_per_m4, velocity, face_velocity_m_s=face_velocity
        )[0]
    )
    euler_number_flux = max(-actual_signed_flux, 0.0)
    inventory = particle_inventory_at_radius(
        rmin,
        x_b=beta.parameters.x_b,
        molar_volume_m3_mol=beta.parameters.molar_volume_m3_mol,
    )
    helper_inventory_flux = boundary_inventory_diagnostic(helper_number_flux, inventory)
    try:
        cohort = _cohort_from_context(context, points_per_cell=2)
        if not hasattr(cohort, "boundary_growth_velocity"):
            raise WorkflowError("CohortSolver lacks public boundary_growth_velocity() shared-helper path")
        cohort_velocity = float(cohort.boundary_growth_velocity())
        if not hasattr(cohort, "boundary_particle_inventory"):
            raise WorkflowError("CohortSolver lacks public boundary_particle_inventory() shared-helper path")
        cohort_inventory = cohort.boundary_particle_inventory()
        first_weights = sum(
            item.weight_m3 for item in cohort.cohorts
            if item.active and rmin < item.radius_m < float(beta.grid.edges_m[1])
        )
        cohort_density = first_weights / float(beta.grid.widths_m[0])
        cohort_number_flux = boundary_number_flux_diagnostic(cohort_velocity, cohort_density)
        cohort_inventory_flux = boundary_inventory_diagnostic(cohort_number_flux, cohort_inventory)
        cohort_error: str | None = None
    except Exception as error:
        cohort_velocity = float("nan")
        cohort_density = float("nan")
        cohort_number_flux = float("nan")
        cohort_inventory = None
        cohort_inventory_flux = None
        cohort_error = f"{type(error).__name__}: {error}"
    fields: list[tuple[str, float, float]] = [
        ("growth_velocity_eulerian_vs_helper", euler_velocity, helper_velocity),
        ("growth_velocity_fv_face_vs_helper", float(face_velocity[0]), helper_velocity),
        ("number_flux_fv_vs_helper", euler_number_flux, helper_number_flux),
    ]
    if math.isfinite(cohort_velocity):
        fields.extend(
            [
                ("growth_velocity_cohort_vs_helper", cohort_velocity, helper_velocity),
                ("number_flux_cohort_vs_helper", cohort_number_flux, helper_number_flux),
                ("cohort_first_cell_density_vs_eulerian", cohort_density, density),
                ("particle_volume_cohort_vs_helper", float(cohort_inventory.volume_m3), inventory.volume_m3),
                ("particle_beta_moles_cohort_vs_helper", float(cohort_inventory.beta_moles_mol), inventory.beta_moles_mol),
                ("particle_B_moles_cohort_vs_helper", float(cohort_inventory.b_moles_mol), inventory.b_moles_mol),
                ("volume_flux_cohort_vs_helper", float(cohort_inventory_flux.beta_volume_flux_out_s), helper_inventory_flux.beta_volume_flux_out_s),
                ("mol_B_flux_cohort_vs_helper", float(cohort_inventory_flux.b_mol_flux_out_mol_m3_s), helper_inventory_flux.b_mol_flux_out_mol_m3_s),
            ]
        )
    rows = [
        {
            "record_type": "physical_boundary_parity",
            "quantity": name,
            "left": left,
            "right": right,
            "relative_error": _relative_error(left, right),
            "pass_1e-12": _relative_error(left, right) <= 1.0e-12,
            "Rmin_m": rmin,
            "Rmin_binary64_hex": float(rmin).hex(),
        }
        for name, left, right in fields
    ]
    all_pass = cohort_error is None and all(bool(row["pass_1e-12"]) for row in rows)
    return {
        "status": "PASS_PHYSICAL_LOWER_BOUNDARY_PARITY" if all_pass else "FAIL_PHYSICAL_LOWER_BOUNDARY_PARITY",
        "reason": cohort_error,
        "rows": rows,
        "rmin_m": rmin,
        "rmin_hex": float(rmin).hex(),
        "growth_velocity_m_s": {
            "helper": helper_velocity,
            "eulerian": euler_velocity,
            "fv_lower_face": float(face_velocity[0]),
            "cohort": cohort_velocity,
        },
        "number_flux_m3_s": {"helper": helper_number_flux, "fv": euler_number_flux, "cohort": cohort_number_flux},
        "inventory": {
            "helper_volume_m3": inventory.volume_m3,
            "helper_beta_moles_mol": inventory.beta_moles_mol,
            "helper_B_moles_mol": inventory.b_moles_mol,
        },
    }


def _build_analytic_transport_context(context: CanonicalContext, *, bins: int) -> tuple[KWNSolver, np.ndarray]:
    """Remesh the same analytic smooth law without retuning physical inputs."""

    mapping = deepcopy(context.mapping)
    grid_data = mapping["radius_grid"]
    grid = RadiusGrid.logarithmic(float(grid_data["minimum_m"]), float(grid_data["maximum_m"]), bins)
    raw = _lognormal_cell_moment(
        grid.edges_m, median_m=context.median_radius_m, log_sigma=context.log_sigma, order=0
    )
    raw_m3 = float(np.sum(cell_moments_from_piecewise_constant_cells(grid.edges_m, raw)[3]))
    reference_m3 = float(np.sum(cell_moments_from_piecewise_constant_cells(context.edges_m, context.cell_number_m3)[3]))
    number = raw * reference_m3 / raw_m3
    mapping["radius_grid"]["bins"] = bins
    mapping["populations"]["beta"]["initial"] = {
        "kind": "cell_integrated",
        "radius_edges_m": [float(item) for item in grid.edges_m],
        "cell_number_density_m3": [float(item) for item in number],
    }
    solver = KWNSolver(SolverConfig.from_mapping(mapping))
    return solver, number


def _analytic_survival(
    edges_m: np.ndarray,
    cell_number_m3: np.ndarray,
    *,
    lower_initial_radius_m: float,
    transform: Callable[[np.ndarray], np.ndarray],
) -> tuple[float, float, float]:
    """Exact M0 and high-order M3 for a piecewise-constant initial density."""

    edges = np.asarray(edges_m, dtype=np.float64)
    number = np.asarray(cell_number_m3, dtype=np.float64)
    widths = np.diff(edges)
    density = number / widths
    lower = float(lower_initial_radius_m)
    surviving_m0 = 0.0
    transformed_m3 = 0.0
    nodes, weights = np.polynomial.legendre.leggauss(24)
    for index, item_density in enumerate(density):
        lo = max(float(edges[index]), lower)
        hi = float(edges[index + 1])
        if hi <= lo:
            continue
        surviving_m0 += float(item_density) * (hi - lo)
        points = 0.5 * (hi + lo) + 0.5 * (hi - lo) * nodes
        transformed_m3 += float(item_density) * 0.5 * (hi - lo) * float(
            np.sum(weights * transform(points) ** 3)
        )
    if lower <= edges[0]:
        flux_density = float(density[0])
    elif lower >= edges[-1]:
        flux_density = 0.0
    else:
        index = min(int(np.searchsorted(edges, lower, side="right") - 1), density.size - 1)
        flux_density = float(density[index])
    return surviving_m0, transformed_m3, flux_density


class _AnalyticBoundaryCohort(CohortSolver):
    """Test-only characteristic control sharing the production event path.

    The benchmark prescribes a known velocity law, while every event,
    physical-Rmin definition, and algebraic matrix closure remains the
    production :class:`CohortSolver` implementation.  It is intentionally
    confined to this diagnostic runner and never changes the frozen material
    growth law used by scientific KWN trajectories.
    """

    def __init__(self, *args: Any, velocity_law: Callable[[np.ndarray], np.ndarray], **kwargs: Any) -> None:
        self._velocity_law = velocity_law
        super().__init__(*args, **kwargs)

    def _velocity(self, radii_m: np.ndarray) -> np.ndarray:
        values = np.asarray(self._velocity_law(np.asarray(radii_m, dtype=np.float64)), dtype=np.float64)
        if values.shape != np.asarray(radii_m).shape or not np.all(np.isfinite(values)):
            raise WorkflowError("analytic cohort velocity law returned invalid values")
        return values

    def growth_rates(self) -> np.ndarray:
        return self._velocity(np.asarray([item.radius_m for item in self._active_cohorts()], dtype=np.float64))

    def _rhs(self, _time_s: float, radii_m: np.ndarray) -> np.ndarray:
        return self._velocity(np.asarray(radii_m, dtype=np.float64))

    def boundary_growth_velocity(self) -> float:
        return float(self._velocity(np.asarray([self.r_diss_m], dtype=np.float64))[0])


def _analytic_cohort_control(
    solver: KWNSolver,
    *,
    cell_number_m3: np.ndarray,
    velocity_law: Callable[[np.ndarray], np.ndarray],
) -> CohortSolver:
    """Discretise the same cell measure for an event-aware analytic control."""

    beta = solver.population("beta")
    radii, weights = positive_cell_quadrature(beta.grid.edges_m, cell_number_m3, points_per_cell=2)
    cohorts = [
        Cohort(initial_id=f"analytic_{index:05d}", radius_m=float(radius), weight_m3=float(weight))
        for index, (radius, weight) in enumerate(zip(radii, weights))
    ]
    return _AnalyticBoundaryCohort(
        cohorts=cohorts,
        beta_parameters=beta.parameters,
        matrix_molar_volume_m3_mol=solver.config.matrix_molar_volume_m3_mol,
        total_b_mol_m3=solver.ledger.total_b_mol_m3,
        equilibrium_adapter=solver.equilibrium_adapter,
        temperature_k=solver.config.temperature_k,
        r_diss_m=boundary_radius(beta.grid),
        inventory_tolerance_relative=solver.config.inventory_tolerance_relative,
        rtol=1.0e-11,
        atol_m=1.0e-18,
        method="DOP853",
        contract_hash=solver.contract_hash,
        source_config_hash=solver.config.source_config_hash,
        velocity_law=velocity_law,
    )


def _analytic_matrix_closure(solver: KWNSolver, *, m3_dimensionless: float):
    """Apply the one-authority matrix closure to an analytic M3 value."""

    beta = solver.population("beta")
    fraction = beta_fraction_from_m3(m3_dimensionless)
    inventory = beta_inventory_from_m3(
        m3_dimensionless,
        x_b=beta.parameters.x_b,
        molar_volume_m3_mol=beta.parameters.molar_volume_m3_mol,
    )
    return close_matrix_from_precipitates(
        total_b_mol_m3=solver.ledger.total_b_mol_m3,
        matrix_molar_volume_m3_mol=solver.config.matrix_molar_volume_m3_mol,
        precipitate_volume_fraction=fraction,
        precipitate_inventory_mol_m3=inventory,
    )


def _run_analytic_case(
    context: CanonicalContext,
    *,
    case: str,
    bins: int,
    cfl_cap: float,
) -> dict[str, Any]:
    """Run one constant-G or -K/R absorbing-boundary control at a known CFL."""

    solver, cell_number = _build_analytic_transport_context(context, bins=bins)
    beta = solver.population("beta")
    population = beta.copy()
    rmin = float(beta.grid.edges_m[0])
    target_radius = float(context.median_radius_m)
    if target_radius <= rmin:
        raise WorkflowError("canonical median must lie above Rmin for analytic boundary benchmark")
    if case == "constant_negative_velocity":
        speed = 1.0e-6
        total_s = (target_radius - rmin) / speed
        centre_velocity = np.full(beta.grid.bins, -speed, dtype=np.float64)
        face_velocity = np.full(beta.grid.bins + 1, -speed, dtype=np.float64)
        cohort_velocity_law = lambda radii: np.full(np.asarray(radii).shape, -speed, dtype=np.float64)
        event_time_from_radius = lambda radius: (float(radius) - rmin) / speed
        lower_initial = rmin + speed * total_s
        transform = lambda radii: radii - speed * total_s
        analytic_flux = speed
    elif case == "minus_K_over_R":
        k_value = 1.0e-15
        total_s = (target_radius**2 - rmin**2) / (2.0 * k_value)
        centre_velocity = -k_value / beta.grid.centres_m
        face_velocity = -k_value / beta.grid.edges_m
        cohort_velocity_law = lambda radii: -k_value / np.asarray(radii, dtype=np.float64)
        event_time_from_radius = lambda radius: (float(radius) ** 2 - rmin**2) / (2.0 * k_value)
        lower_initial = math.sqrt(rmin**2 + 2.0 * k_value * total_s)
        transform = lambda radii: np.sqrt(np.maximum(radii**2 - 2.0 * k_value * total_s, 0.0))
        analytic_flux = k_value / lower_initial
    else:
        raise WorkflowError(f"unknown analytic transport case {case!r}")
    rate, _, _, _, _ = solver._raw_face_operator_rate(
        centre_velocity, beta.grid.widths_m, face_velocity_m_s=face_velocity
    )
    steps = max(1, int(math.ceil(total_s * rate / cfl_cap)))
    dt_s = total_s / steps
    cumulative_loss = 0.0
    last_flux = 0.0
    for _ in range(steps):
        flux, _, _, _ = solver._advect_population(
            population, centre_velocity, dt_s, face_velocity_m_s=face_velocity
        )
        cumulative_loss += flux * dt_s
        last_flux = flux
    analytic_m0, analytic_m3, initial_density_at_crossing = _analytic_survival(
        beta.grid.edges_m,
        cell_number,
        lower_initial_radius_m=lower_initial,
        transform=transform,
    )
    eulerian = metrics_from_piecewise_constant_cells(
        beta.grid.edges_m, population.number_density_per_m4 * beta.grid.widths_m
    )
    initial_m0 = float(np.sum(cell_number))
    analytic_loss = initial_m0 - analytic_m0
    analytic_matrix = _analytic_matrix_closure(solver, m3_dimensionless=analytic_m3)
    eulerian_matrix = _analytic_matrix_closure(
        solver, m3_dimensionless=eulerian.M3_dimensionless
    )
    cohort = _analytic_cohort_control(
        solver, cell_number_m3=cell_number, velocity_law=cohort_velocity_law
    )
    cohort.advance_to(total_s)
    cohort_snapshot = cohort.snapshot()
    cohort_loss = float(cohort_snapshot.cumulative_number_dissolved_m3)
    dissolved = [item for item in cohort.cohorts if not item.active]
    event_time_errors = [
        abs(float(item.dissolution_time_s) - event_time_from_radius(float(item.initial_radius_m)))
        for item in dissolved
        if item.dissolution_time_s is not None and item.initial_radius_m is not None
    ]
    event_residuals = [
        abs(float(item.boundary_event_residual_m))
        for item in dissolved
        if item.boundary_event_residual_m is not None
    ]
    return {
        "record_type": "analytic_boundary_control",
        "case": case,
        "bins": bins,
        "target_face_CFL": cfl_cap,
        "actual_global_face_CFL": rate * dt_s,
        "steps": steps,
        "dt_s": dt_s,
        "total_time_s": total_s,
        "Rmin_m": rmin,
        "crossing_initial_radius_m": lower_initial,
        "analytic_M0_m3": analytic_m0,
        "eulerian_M0_m3": eulerian.M0_m3,
        "M0_error": _relative_error(eulerian.M0_m3, analytic_m0),
        "analytic_M3_dimensionless": analytic_m3,
        "eulerian_M3_dimensionless": eulerian.M3_dimensionless,
        "M3_error": _relative_error(eulerian.M3_dimensionless, analytic_m3),
        "analytic_cumulative_number_loss_m3": analytic_loss,
        "eulerian_cumulative_number_loss_m3": cumulative_loss,
        "cumulative_number_loss_error": _relative_error(cumulative_loss, analytic_loss),
        "analytic_boundary_number_flux_m3_s": analytic_flux * initial_density_at_crossing,
        "eulerian_boundary_number_flux_m3_s": last_flux,
        "boundary_number_flux_error": _relative_error(last_flux, analytic_flux * initial_density_at_crossing),
        "analytic_matrix_inventory_mol_m3": analytic_matrix.matrix_inventory_mol_m3,
        "eulerian_matrix_inventory_mol_m3": eulerian_matrix.matrix_inventory_mol_m3,
        "matrix_inventory_error": _relative_error(
            eulerian_matrix.matrix_inventory_mol_m3,
            analytic_matrix.matrix_inventory_mol_m3,
        ),
        "cohort_M0_m3": cohort_snapshot.M0_m3,
        "cohort_M0_error": _relative_error(cohort_snapshot.M0_m3, analytic_m0),
        "cohort_M3_dimensionless": cohort_snapshot.M3_dimensionless,
        "cohort_M3_error": _relative_error(
            cohort_snapshot.M3_dimensionless, analytic_m3
        ),
        "cohort_cumulative_number_loss_m3": cohort_loss,
        "cohort_cumulative_number_loss_error": _relative_error(cohort_loss, analytic_loss),
        "cohort_dissolved_event_count": len(dissolved),
        "cohort_event_time_max_abs_error_s": max(event_time_errors, default=0.0),
        "cohort_event_radius_residual_max_abs_m": max(event_residuals, default=0.0),
        "Eulerian_vs_cohort_M0_error": _relative_error(
            eulerian.M0_m3, cohort_snapshot.M0_m3
        ),
        "Eulerian_vs_cohort_M3_error": _relative_error(
            eulerian.M3_dimensionless, cohort_snapshot.M3_dimensionless
        ),
        "Eulerian_vs_cohort_matrix_inventory_error": _relative_error(
            eulerian_matrix.matrix_inventory_mol_m3,
            cohort_snapshot.matrix_inventory_mol_m3,
        ),
        "cohort_inventory_relative_residual": cohort_snapshot.inventory_relative_residual,
    }


def _monotone_nonincreasing(values: Sequence[float], *, tolerance: float = 1.0e-12) -> bool:
    return all(
        float(right) <= float(left) * (1.0 + tolerance) + tolerance
        for left, right in zip(values, values[1:])
    )


def _analytic_benchmarks(context: CanonicalContext, *, fast: bool) -> dict[str, Any]:
    """B4-style radius/time refinement for both required absorbing laws."""

    # This is an operator/convergence gate, not the later 3200-bin 2%
    # smooth-population accuracy gate.  The compact ladder keeps the required
    # constant-G and -K/R tests executable while exposing the first-order
    # spatial error instead of hiding it behind an unaffordable fine mesh.
    spatial_bins = (32, 64, 128) if fast else (64, 128, 256)
    spatial_cfl = 0.5 if fast else 0.25
    temporal_caps = (1.0, 0.5, 0.25) if fast else (0.5, 0.25, 0.125)
    rows: list[dict[str, Any]] = []
    for case in ("constant_negative_velocity", "minus_K_over_R"):
        for bins in spatial_bins:
            row = _run_analytic_case(context, case=case, bins=bins, cfl_cap=spatial_cfl)
            row["refinement_axis"] = "radius"
            rows.append(row)
        for cap in temporal_caps:
            row = _run_analytic_case(context, case=case, bins=spatial_bins[1], cfl_cap=cap)
            row["refinement_axis"] = "time"
            rows.append(row)
    metrics = (
        "M0_error",
        "M3_error",
        "cumulative_number_loss_error",
        "boundary_number_flux_error",
        "matrix_inventory_error",
    )
    monotonic = True
    fine_errors: list[float] = []
    for case in ("constant_negative_velocity", "minus_K_over_R"):
        spatial = sorted(
            (row for row in rows if row["case"] == case and row["refinement_axis"] == "radius"),
            key=lambda item: int(item["bins"]),
        )
        temporal = sorted(
            (row for row in rows if row["case"] == case and row["refinement_axis"] == "time"),
            key=lambda item: float(item["target_face_CFL"]), reverse=True,
        )
        monotonic &= all(_monotone_nonincreasing([float(row[key]) for row in spatial]) for key in metrics)
        monotonic &= all(_monotone_nonincreasing([float(row[key]) for row in temporal]) for key in metrics)
        fine_errors.extend(float(spatial[-1][key]) for key in metrics)
        fine_errors.extend(float(temporal[-1][key]) for key in metrics)
    max_fine = max(fine_errors, default=float("inf"))
    # A coarse first-order Eulerian control cannot honestly be forced through
    # the later 2% canonical crosscheck threshold.  Its actual gate is the
    # specified monotonic radius/time convergence and finite conservative
    # errors.  The full 2% decision remains in the later cohort comparison.
    finite = all(
        math.isfinite(float(row[key]))
        for row in rows
        for key in (
            *metrics,
            "cohort_M0_error",
            "cohort_M3_error",
            "cohort_cumulative_number_loss_error",
            "cohort_inventory_relative_residual",
            "cohort_event_time_max_abs_error_s",
            "cohort_event_radius_residual_max_abs_m",
        )
    )
    cohort_closure = all(float(row["cohort_inventory_relative_residual"]) <= 1.0e-10 for row in rows)
    passed = monotonic and finite and cohort_closure
    return {
        "status": "PASS_BOUNDARY_ANALYTIC_BENCHMARKS" if passed else "FAIL_BOUNDARY_ANALYTIC_BENCHMARKS",
        "rows": rows,
        "radius_refinement_bins": spatial_bins,
        "time_refinement_caps": temporal_caps,
        "monotonic_convergence": monotonic,
        "maximum_finest_error": max_fine,
        "finest_within_two_percent": max_fine <= TWO_PERCENT,
        "cohort_inventory_closed": cohort_closure,
        "gate_criterion": "finite_and_monotonic_Eulerian_radius_and_time_refinement_plus_cohort_event_inventory_closure; 2_percent_reserved_for_full_canonical_cohort_crosscheck",
    }


def _lower_boundary_operator_gate(
    *,
    baseline: Mapping[str, Any],
    contract: Mapping[str, Any],
    tests: Mapping[str, Any],
    analytic: Mapping[str, Any],
    parity: Mapping[str, Any],
) -> dict[str, Any]:
    checks = {
        "baseline": baseline.get("status") == "PASS_BASELINE_REPRODUCTION",
        "contract_Rmin_binary64": contract.get("R_boundary_binary64_hex") == contract.get("baseline_R_boundary_binary64_hex"),
        "boundary_unit_tests": tests.get("status") == "PASS_LOWER_BOUNDARY_UNIT_TESTS",
        "analytic_absorbing_benchmarks": analytic.get("status") == "PASS_BOUNDARY_ANALYTIC_BENCHMARKS",
        "physical_quantity_parity": parity.get("status") == "PASS_PHYSICAL_LOWER_BOUNDARY_PARITY",
    }
    return {
        "status": "PASS_LOWER_BOUNDARY_OPERATOR_PARITY" if all(checks.values()) else "FAIL_LOWER_BOUNDARY_OPERATOR_PARITY",
        "checks": checks,
    }


def _implicit_accuracy_ladder(
    context: CanonicalContext,
    *,
    max_steps: int,
    max_wall_s: float,
) -> dict[str, Any]:
    """Run actual, self-consistent policies over one common short horizon."""

    runs: list[EulerianRun] = []
    policies: list[tuple[str, float | None]] = [("current_implicit_policy", None)]
    policies.extend(
        (f"implicit_active_CFL_lte_{cap:g}", float(cap))
        for cap in IMPLICIT_ACTIVE_CFL_CAPS
    )
    unattempted_caps: list[float] = []
    for index, (policy, cap) in enumerate(policies):
        run = _advance_eulerian(
            _build_canonical_solver_from_context(context, active_cfl_cap=cap),
            policy=policy,
            target_times_h=FIRST_SIGNIFICANT_BOUNDARY_TIME_H,
            active_cfl_cap=cap,
            max_steps=max_steps,
            max_wall_s=max_wall_s,
        )
        runs.append(run)
        if run.status == "BRUTE_FORCE_CFL_NOT_PRACTICAL":
            # This is the only allowed early stop: the source-owned
            # candidate-step chooser actually rejected a step below min_dt.
            # Do not synthesize feasibility conclusions for tighter caps;
            # their support masks may change with their own candidate dt.
            unattempted_caps = [
                float(later_cap)
                for _, later_cap in policies[index + 1:]
                if later_cap is not None
            ]
            break
    completed = [item for item in runs if item.status == "PASS_EULERIAN_SHORT_RUN" and item.snapshots]
    finest = next((item for item in reversed(completed) if item.requested_active_cfl is not None), None)
    cap_runs = [item for item in runs if item.requested_active_cfl is not None]
    all_caps_completed = (
        len(cap_runs) == len(IMPLICIT_ACTIVE_CFL_CAPS)
        and all(item.status == "PASS_EULERIAN_SHORT_RUN" and item.snapshots for item in cap_runs)
    )
    rows: list[dict[str, Any]] = []
    cfl_rows = [row for item in runs for row in item.cfl_rows]
    tail_rows = [row for item in runs for row in item.lower_tail_rows]
    primary = (
        "N_m0_m3", "Rmean_number_m", "Rmean_cubed_m3", "Sv_m_inv", "f_beta", "matrix_xB",
        "cumulative_number_dissolution_m3", "cumulative_mol_B_returned_mol_m3",
    )
    error_map: dict[str, dict[str, float]] = {}
    for item in runs:
        row: dict[str, Any] = {
            "policy": item.policy,
            "requested_active_cfl": "UNBOUNDED" if item.requested_active_cfl is None else item.requested_active_cfl,
            "status": item.status,
            "evaluation_mode": (
                "accepted_short_trajectory"
                if item.status == "PASS_EULERIAN_SHORT_RUN"
                else (
                    "incomplete_partial_trajectory"
                    if item.cfl_rows
                    else (
                        "solver_reported_min_dt_rejection"
                        if item.status == "BRUTE_FORCE_CFL_NOT_PRACTICAL"
                        else "incomplete_diagnostic_budget"
                    )
                )
            ),
            "reason": item.reason or "",
            "accepted_steps": item.accepted_steps,
            "rejected_steps": item.rejected_steps,
            "runtime_s": item.runtime_s,
            # The support is dynamic, so this short diagnostic does not make
            # a credible 48 h runtime forecast.
            "estimated_48h_steps": "NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN",
            "estimated_48h_runtime_s": "NOT_PROJECTED_FROM_UNQUALIFIED_SHORT_RUN",
            "terminal_audit_active_rate_s_inv": item.feasibility.get("terminal_audit_active_rate_s_inv"),
            "terminal_audit_implied_candidate_dt_s": item.feasibility.get(
                "terminal_audit_implied_candidate_dt_s"
            ),
            "solver_reported_cfl_limited_dt_s": item.feasibility.get("solver_reported_cfl_limited_dt_s"),
            "terminal_audit_boundary_active": item.feasibility.get("boundary_active"),
            "terminal_audit_is_not_a_feasibility_bound": item.feasibility.get(
                "static_audit_is_not_a_feasibility_bound", False
            ),
        }
        if item.snapshots:
            final = item.snapshots[-1]
            row.update({key: final.get(key, "") for key in final})
        if finest is not None and item.status == "PASS_EULERIAN_SHORT_RUN":
            reference = finest.snapshots[-1]
            errors = {
                key: _absolute_or_relative_error(float(item.snapshots[-1][key]), float(reference[key]))
                for key in primary
            }
            error_map[item.policy] = errors
            row.update({f"{key}_vs_finest_implicit_error": value for key, value in errors.items()})
        rows.append(row)
    convergence = False
    measurable = False
    candidate_passing: str | None = None
    if finest is not None:
        numeric = [item for item in runs if item.policy in error_map and item.requested_active_cfl is not None]
        ordered = sorted(numeric, key=lambda item: float(item.requested_active_cfl), reverse=True)
        convergence = all_caps_completed and all(
            _monotone_nonincreasing([error_map[item.policy][metric] for item in ordered])
            for metric in primary
        )
        current_errors = error_map.get("current_implicit_policy", {})
        measurable = bool(current_errors) and max(current_errors.values()) > 1.0e-4
        passing = [
            item for item in ordered
            if max(error_map[item.policy].values(), default=float("inf")) <= TWO_PERCENT
        ]
        if convergence and passing:
            # Largest allowable cap is the cheapest candidate; it remains a
            # candidate until the cohort reference and original 2% gate run.
            candidate_passing = passing[0].policy
    brute_force = [item for item in runs if item.status == "BRUTE_FORCE_CFL_NOT_PRACTICAL"]
    if brute_force:
        status = "BRUTE_FORCE_CFL_NOT_PRACTICAL"
    elif any(item.status == "INCOMPLETE_DIAGNOSTIC_BUDGET" for item in runs):
        status = "INCOMPLETE_DIAGNOSTIC_BUDGET"
    elif not all_caps_completed or not completed:
        status = "INCOMPLETE_DIAGNOSTIC_BUDGET"
    elif candidate_passing is None:
        status = "FAIL_EULERIAN_TIME_ACCURACY"
    else:
        status = "INCOMPLETE_NO_COHORT_REFERENCE"
    if brute_force:
        reference_limit = (
            "The source-owned active-CFL chooser emitted a below-min_dt rejection; "
            "no tighter policy is inferred from a frozen-state boundary audit."
        )
    elif status == "INCOMPLETE_DIAGNOSTIC_BUDGET":
        reference_limit = (
            "At least one actual policy exhausted its diagnostic budget.  No self-convergence, "
            "2% cohort, or production-cost conclusion is claimed."
        )
    elif status == "FAIL_EULERIAN_TIME_ACCURACY":
        reference_limit = (
            "All requested policies completed the common horizon, but the active-CFL self-convergence "
            "criterion did not identify a cohort-eligible candidate."
        )
    else:
        reference_limit = (
            "All requested policies completed the common first-significant-lower-tail horizon; "
            "the selected candidate remains provisional until the exact canonical cohort 2% crosscheck."
        )
    return {
        "status": status,
        "runs": runs,
        "rows": rows,
        "cfl_rows": cfl_rows,
        "lower_tail_rows": tail_rows,
        "finest_policy": None if finest is None else finest.policy,
        "temporal_self_convergence_monotonic": convergence,
        "measurable_difference_vs_finest_implicit": measurable,
        "candidate_active_cfl": candidate_passing,
        "all_requested_caps_completed": all_caps_completed,
        "brute_force_policies": [item.policy for item in brute_force],
        "brute_force_feasibility": {item.policy: item.feasibility for item in brute_force},
        "unattempted_active_cfl_caps_after_actual_brute_force": unattempted_caps,
        "reference_limit": reference_limit,
    }


def _cohort_snapshot_row(cohort: CohortSolver, *, policy: str) -> dict[str, Any]:
    snapshot = cohort.snapshot()
    value = snapshot.as_dict()
    value.update({
        "policy": policy,
        "time_h": float(snapshot.time_s / 3600.0),
        "Rmean_number_m": float(getattr(snapshot, "Rmean_number_m", snapshot.Rmean_m)),
        "Rmean_cubed_m3": float(getattr(snapshot, "Rmean_cubed_m3", snapshot.Rmean3_m3)),
        "mean_R3_m3": float(getattr(snapshot, "mean_R3_m3", snapshot.Rmean3_m3)),
        "cumulative_number_dissolution_m3": float(getattr(snapshot, "cumulative_number_dissolved_m3", 0.0)),
        "cumulative_beta_volume_dissolution": float(getattr(snapshot, "cumulative_beta_volume_dissolved", 0.0)),
        "cumulative_mol_B_returned_mol_m3": float(getattr(snapshot, "cumulative_mol_B_returned_mol_m3", 0.0)),
    })
    return value


def _cohort_eulerian_crosscheck(
    context: CanonicalContext,
    *,
    active_cfl_cap: float,
    points_per_cell: int,
    max_steps: int,
    max_wall_s: float,
) -> dict[str, Any]:
    """Optional exact long comparator; it is not silently replaced by a proxy."""

    policy = f"implicit_active_CFL_lte_{active_cfl_cap:g}"
    eulerian = _advance_eulerian(
        _build_canonical_solver_from_context(context, active_cfl_cap=active_cfl_cap),
        policy=policy,
        target_times_h=SMOOTH_TIMES_H,
        active_cfl_cap=active_cfl_cap,
        max_steps=max_steps,
        max_wall_s=max_wall_s,
    )
    if eulerian.status != "PASS_EULERIAN_SHORT_RUN":
        return {
            "status": "FAIL_EULERIAN_TIME_ACCURACY",
            "reason": eulerian.reason,
            "rows": [],
            "eulerian": eulerian,
        }
    started = time.monotonic()
    try:
        cohort = _cohort_from_context(context, points_per_cell=points_per_cell)
        cohort_rows: dict[float, dict[str, Any]] = {0.0: _cohort_snapshot_row(cohort, policy=policy)}
        for time_h in SMOOTH_TIMES_H[1:]:
            cohort.advance_to(float(time_h) * 3600.0)
            cohort_rows[time_h] = _cohort_snapshot_row(cohort, policy=policy)
    except Exception as error:
        return {
            "status": "FAIL_EULERIAN_TIME_ACCURACY",
            "reason": f"cohort comparator failed: {type(error).__name__}: {error}",
            "rows": [],
            "eulerian": eulerian,
            "runtime_s": time.monotonic() - started,
        }
    euler_rows = {round(float(row["time_h"]), 12): row for row in eulerian.snapshots}
    metric_pairs = (
        ("N_m0_m3", "N_m0_m3"), ("M0_m3", "M0_m3"), ("M1_m2", "M1_m2"),
        ("M2_m", "M2_m"), ("M3_dimensionless", "M3_dimensionless"),
        ("Rmean_m", "Rmean_number_m"), ("Rmean3_m3", "Rmean_cubed_m3"),
        ("mean_R3_m3", "mean_R3_m3"), ("Sv_m_inv", "Sv_m_inv"), ("f_beta", "f_beta"),
        ("matrix_xB", "matrix_xB"),
        ("cumulative_number_dissolution_m3", "cumulative_number_dissolution_m3"),
        ("cumulative_beta_volume_dissolution", "cumulative_beta_volume_dissolution"),
        ("cumulative_mol_B_returned_mol_m3", "cumulative_mol_B_returned_mol_m3"),
    )
    rows: list[dict[str, Any]] = []
    maximum: dict[str, float] = {}
    for time_h in SMOOTH_TIMES_H:
        euler = euler_rows.get(round(time_h, 12))
        cohort_row = cohort_rows.get(time_h)
        if euler is None or cohort_row is None:
            raise WorkflowError(f"missing crosscheck output at {time_h:g} h")
        for label, cohort_key in metric_pairs:
            error = _absolute_or_relative_error(float(euler[label]), float(cohort_row[cohort_key]))
            maximum[label] = max(maximum.get(label, 0.0), error)
            rows.append({
                "record_type": "metric",
                "time_h": time_h,
                "metric": label,
                "eulerian_value": euler[label],
                "cohort_value": cohort_row[cohort_key],
                "relative_error": error,
                "gate": TWO_PERCENT,
                "pass": error <= TWO_PERCENT,
            })
        beta = _build_canonical_solver_from_context(context).population("beta")
        # Cohort active atoms are a direct characteristic measure; Eulerian
        # is converted by positive quadrature solely for a distribution metric.
        euler_solver = None
        # Avoid reconstructing a false state from the scalar snapshot.  The
        # full PSD W1 is provided only by a dedicated future state archive.
        rows.append({
            "record_type": "distribution",
            "time_h": time_h,
            "metric": "PSD_Wasserstein_distance_m",
            "eulerian_value": "NOT_ARCHIVED_IN_SCALAR_CROSSCHECK",
            "cohort_value": "NOT_ARCHIVED_IN_SCALAR_CROSSCHECK",
            "relative_error": "NOT_EVALUATED",
            "gate": TWO_PERCENT,
            "pass": "NOT_EVALUATED",
        })
        del beta, euler_solver
    passed = bool(maximum) and all(value <= TWO_PERCENT for value in maximum.values())
    return {
        "status": "PASS_COHORT_EULERIAN_SHARED_OPERATOR_PARITY" if passed else "FAIL_EULERIAN_TIME_ACCURACY",
        "rows": rows,
        "maximum_errors": maximum,
        "eulerian": eulerian,
        "cohort_points_per_cell": points_per_cell,
        "runtime_s": time.monotonic() - started,
        "limitation": "scalar crosscheck records all required scalar metrics; PSD W1 requires a future accepted-state PSD archive",
    }


def _blocked_crosscheck(reason: str) -> dict[str, Any]:
    return {
        "status": "BLOCKED_PREREQUISITE_GATE",
        "reason": reason,
        "rows": [{"status": "BLOCKED_PREREQUISITE_GATE", "reason": reason}],
    }


def _authority_stub(crosscheck: Mapping[str, Any], context: CanonicalContext | None) -> dict[str, Any]:
    if crosscheck.get("status") != "PASS_COHORT_EULERIAN_SHARED_OPERATOR_PARITY":
        return {
            "schema_version": "EULERIAN_SMOOTH_AUTHORITY_GRID_V2",
            "status": "BLOCKED_COHORT_EULERIAN_SHARED_OPERATOR_PARITY",
            "authority_grid": None,
            "solver": "CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE",
            "reason": "800/1600/3200 v2 authority is not inherited before the exact shared-operator crosscheck passes.",
            "validation_contract_hash": None if context is None else context.contract_hash,
        }
    return {
        "schema_version": "EULERIAN_SMOOTH_AUTHORITY_GRID_V2",
        "status": "NOT_RUN_GRID_LADDER_REQUIRED",
        "authority_grid": None,
        "solver": "CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE",
        "reason": "Crosscheck passed; separately run 800/1600/3200 v2 active-CFL ladder before selecting authority.",
        "validation_contract_hash": None if context is None else context.contract_hash,
    }


def _beta_pf_stub(crosscheck: Mapping[str, Any]) -> dict[str, Any]:
    if crosscheck.get("status") != "PASS_COHORT_EULERIAN_SHARED_OPERATOR_PARITY":
        reason = "Frozen PF evidence is intentionally not read for a direction conclusion before KWN shared-operator parity passes."
        return {"status": "BLOCKED_PREREQUISITE_GATE", "reason": reason}
    return {
        "status": "NOT_RUN_FROZEN_PF_COMPARISON_REQUIRED",
        "reason": "No CUDA/PF was rerun.  A separately invoked frozen Case A/B comparison may now use the discrete-cohort solver only.",
    }


def _explicit_stub(
    lower_gate: Mapping[str, Any],
    implicit: Mapping[str, Any],
    context: CanonicalContext | None,
) -> dict[str, Any]:
    if lower_gate.get("status") != "PASS_LOWER_BOUNDARY_OPERATOR_PARITY":
        return {
            "status": "NOT_AUTHORIZED_BEFORE_LOWER_BOUNDARY_OPERATOR_PARITY",
            "reason": "Diagnostic explicit SSPRK2 is forbidden until the physical-Rmin operator gate passes.",
        }
    if context is None:
        return {
            "status": "BLOCKED_MISSING_CANONICAL_CONTEXT",
            "reason": "No canonical Eulerian state is available for the whole-domain donor-bound audit.",
        }
    solver = _build_canonical_solver_from_context(context)
    audit = _cfl_audit(solver, dt_s=1.0)
    donor_rate = float(audit["global_max_cfl"])
    donor_dt = 1.0 / donor_rate if donor_rate > 0.0 else float("inf")
    return {
        "status": "BRUTE_FORCE_DONOR_BOUND_NOT_PRACTICAL",
        "reason": (
            "A whole-domain donor-bound SSPRK2 reference would require dt <= "
            f"{donor_dt:.17e} s from the actual physical-Rmin lower face at the canonical initial state, "
            f"or at least {48.0 * 3600.0 / donor_dt:.17e} accepted steps for 48 h.  "
            "It is therefore not implemented or relabelled as an active-only explicit proxy."
        ),
        "reference_solver": "EXPLICIT_UPWIND_DONOR_BOUND_SSPRK2",
        "same_physical_Rmin_contract": True,
        "whole_domain_donor_bound_rate_s_inv": donor_rate,
        "whole_domain_donor_bound_dt_s": donor_dt,
        "whole_domain_donor_bound_projected_48h_steps": 48.0 * 3600.0 / donor_dt,
        "initial_global_face_index": audit.get("global_face_index"),
        "initial_lower_boundary_velocity_m_s": audit.get("lower_boundary_velocity_m_s"),
        "implicit_ladder_status": implicit.get("status"),
        "actual_implicit_brute_force_policies": implicit.get("brute_force_policies", []),
    }


def _top_status(
    *, lower: Mapping[str, Any], implicit: Mapping[str, Any], crosscheck: Mapping[str, Any], authority: Mapping[str, Any], beta_pf: Mapping[str, Any]
) -> str:
    if lower.get("status") != "PASS_LOWER_BOUNDARY_OPERATOR_PARITY":
        return "FAIL_LOWER_BOUNDARY_OPERATOR_PARITY"
    if crosscheck.get("status") == "PASS_COHORT_EULERIAN_SHARED_OPERATOR_PARITY":
        if authority.get("status") != "PASS_EULERIAN_SMOOTH_POPULATION_AUTHORITY_V2":
            return "BLOCKED_PRODUCTION_TRANSPORT_SCHEME_UPGRADE"
        if beta_pf.get("status") == "FAIL_BETA_ONLY_DIRECTION":
            return "FAIL_BETA_ONLY_DIRECTION"
        if beta_pf.get("status") == "PASS_BETA_ONLY_DIRECTION":
            return "PASS_KWN_PF_ONE_WAY_STORAGE_COUPLING_V1"
    if implicit.get("status") == "BRUTE_FORCE_CFL_NOT_PRACTICAL":
        return "BLOCKED_PRODUCTION_TRANSPORT_SCHEME_UPGRADE"
    if implicit.get("status") == "FAIL_EULERIAN_TIME_ACCURACY" or crosscheck.get("status") == "FAIL_EULERIAN_TIME_ACCURACY":
        return "FAIL_EULERIAN_TIME_ACCURACY"
    # Fail closed: no unqualified Eulerian policy may be presented as an
    # authority or a shared-operator pass.
    return "FAIL_EULERIAN_TIME_ACCURACY"


def _report_body_table(rows: Sequence[Mapping[str, Any]], fields: Sequence[str], limit: int = 20) -> str:
    if not rows:
        return "No rows were produced."
    header = "| " + " | ".join(fields) + " |\n|" + "|".join("---" for _ in fields) + "|\n"
    body = []
    for row in rows[:limit]:
        body.append("| " + " | ".join(str(_csv_value(row.get(field, ""))) for field in fields) + " |")
    suffix = "" if len(rows) <= limit else f"\n\nOnly the first {limit} rows are shown; the complete CSV is authoritative."
    return header + "\n".join(body) + suffix


def _write_stage_reports(
    *,
    report_root: Path,
    output_root: Path,
    baseline: Mapping[str, Any],
    contract: Mapping[str, Any] | None,
    tests: Mapping[str, Any] | None,
    analytic: Mapping[str, Any] | None,
    parity: Mapping[str, Any] | None,
    lower: Mapping[str, Any] | None,
    implicit: Mapping[str, Any] | None,
    explicit: Mapping[str, Any] | None,
    crosscheck: Mapping[str, Any] | None,
    authority: Mapping[str, Any] | None,
    beta_pf: Mapping[str, Any] | None,
    final: Mapping[str, Any],
    command: str,
) -> None:
    completed: dict[str, str] = {}
    baseline_provenance = baseline.get("provenance", {}) if isinstance(baseline, Mapping) else {}
    _write_report(
        report_root,
        "00_baseline_boundary_reproduction.md",
        REQUIRED_REPORTS["00_baseline_boundary_reproduction.md"],
        f"Status: `{baseline.get('status')}`.  The trace is a pre-source-change artifact from "
        f"`{baseline_provenance.get('source_commit', 'unknown')}` and records historical "
        f"`{baseline_provenance.get('baseline_status', 'unknown')}`.\n\n"
        f"Trace: `{baseline.get('trace', '')}`.  State: `{baseline.get('state', '')}`.\n\n"
        f"Checks: `{json.dumps(_json_safe(baseline.get('checks', {})), sort_keys=True)}`.",
    )
    completed["00_baseline_boundary_reproduction.md"] = "done"
    if contract is not None:
        _write_report(
            report_root,
            "01_physical_lower_boundary_contract.md",
            REQUIRED_REPORTS["01_physical_lower_boundary_contract.md"],
            "The only resolved lower boundary is the exact binary64 grid edge `Rmin`; it is not a cell centre, "
            "occupied bin, or extrapolation point.  A negative growth velocity is outward, a nonnegative velocity "
            "has zero external inflow, and boundary diagnostics never update matrix composition.\n\n"
            "```json\n" + json.dumps(_json_safe(contract), indent=2, sort_keys=True) + "\n```",
        )
        completed["01_physical_lower_boundary_contract.md"] = "done"
    if tests is not None:
        _write_report(
            report_root,
            "02_boundary_unit_tests.md",
            REQUIRED_REPORTS["02_boundary_unit_tests.md"],
            f"Status: `{tests.get('status')}`.\n\n" + _report_body_table(
                tests.get("rows", []), ("gate", "test_id", "status", "reason")
            ),
        )
        completed["02_boundary_unit_tests.md"] = "done"
    if analytic is not None:
        _write_report(
            report_root,
            "03_analytic_boundary_benchmarks.md",
            REQUIRED_REPORTS["03_analytic_boundary_benchmarks.md"],
            f"Status: `{analytic.get('status')}`; monotonic refinement: `{analytic.get('monotonic_convergence')}`; "
            f"maximum finest error: `{analytic.get('maximum_finest_error')}`; finest within 2%: "
            f"`{analytic.get('finest_within_two_percent')}`; cohort event inventory closed: "
            f"`{analytic.get('cohort_inventory_closed')}`.  The analytic gate criterion is "
            f"`{analytic.get('gate_criterion', '')}`.\n\n" + _report_body_table(
                analytic.get("rows", []),
                (
                    "case", "refinement_axis", "bins", "target_face_CFL",
                    "M0_error", "M3_error", "matrix_inventory_error",
                    "cumulative_number_loss_error", "boundary_number_flux_error",
                    "cohort_M0_error", "cohort_M3_error",
                    "Eulerian_vs_cohort_M0_error", "Eulerian_vs_cohort_M3_error",
                    "cohort_event_time_max_abs_error_s",
                    "cohort_event_radius_residual_max_abs_m",
                    "cohort_inventory_relative_residual",
                ),
            ),
        )
        completed["03_analytic_boundary_benchmarks.md"] = "done"
    if parity is not None and lower is not None:
        _write_report(
            report_root,
            "04_boundary_operator_parity.md",
            REQUIRED_REPORTS["04_boundary_operator_parity.md"],
            f"Physical-quantity status: `{parity.get('status')}`.  Composite lower-boundary gate: `{lower.get('status')}`.\n\n"
            + _report_body_table(parity.get("rows", []), ("quantity", "left", "right", "relative_error", "pass_1e-12"))
            + "\n\nThe Eulerian lower face is accepted only when its actual finite-volume face velocity, number flux, "
            "inventory price, and cohort event/operator path all agree with the shared Rmin helper.",
        )
        completed["04_boundary_operator_parity.md"] = "done"
    if implicit is not None:
        cfl_rows = implicit.get("cfl_rows", [])
        first = cfl_rows[0] if cfl_rows else {}
        _write_report(
            report_root,
            "05_courant_definition_and_audit.md",
            REQUIRED_REPORTS["05_courant_definition_and_audit.md"],
            "`GLOBAL_MAX_CFL` measures every actual FV face.  `ACTIVE_M0_CFL` and `ACTIVE_M3_CFL` use the smallest "
            "contiguous 99.9999% moment supports plus incident faces.  `LOWER_TAIL_ACTIVE_CFL` uses the M0 CDF band "
            "[1e-8, 1e-6].  `BOUNDARY_ACTIVE_CFL` is only active when that declared tail touches or reaches Rmin; "
            "the physical flux is nevertheless always executed.\n\n"
            f"First recorded step: global `{first.get('global_max_cfl', 'NOT_RUN')}`, M0 `{first.get('active_M0_cfl', 'NOT_RUN')}`, "
            f"M3 `{first.get('active_M3_cfl', 'NOT_RUN')}`, lower-tail `{first.get('lower_tail_active_cfl', 'NOT_RUN')}`, "
            f"boundary `{first.get('boundary_active_cfl', 'NOT_RUN')}`.\n\n"
            "Full per-step telemetry is in `courant_audit.csv` and `lower_tail_diagnostics.csv`.",
        )
        completed["05_courant_definition_and_audit.md"] = "done"
        _write_report(
            report_root,
            "06_implicit_time_accuracy.md",
            REQUIRED_REPORTS["06_implicit_time_accuracy.md"],
            f"Status: `{implicit.get('status')}`.  The reference limit is `{implicit.get('reference_limit', '')}`.  "
            "The current policy and every attempted cap use the same 0--0.1 h first-significant-lower-tail horizon; "
            "each accepted timestep comes from the solver's self-consistent candidate-step decision.  A frozen dt=1 s "
            "boundary audit is reported only as telemetry and never promoted to a universal cap feasibility bound.  "
            f"All requested caps completed: `{implicit.get('all_requested_caps_completed')}`.  "
            f"Candidate cheapest cap: `{implicit.get('candidate_active_cfl')}`.  Measurable current-vs-finest implicit "
            f"difference: `{implicit.get('measurable_difference_vs_finest_implicit')}`.  "
            f"Actual solver-reported brute-force policies: `{implicit.get('brute_force_policies', [])}`.  "
            f"Unattempted caps after an actual brute-force rejection: `{implicit.get('unattempted_active_cfl_caps_after_actual_brute_force', [])}`.\n\n"
            + _report_body_table(implicit.get("rows", []), ("policy", "requested_active_cfl", "evaluation_mode", "status", "accepted_steps", "runtime_s", "estimated_48h_steps", "estimated_48h_runtime_s", "terminal_audit_active_rate_s_inv", "terminal_audit_implied_candidate_dt_s", "solver_reported_cfl_limited_dt_s", "terminal_audit_boundary_active", "terminal_audit_is_not_a_feasibility_bound")),
        )
        completed["06_implicit_time_accuracy.md"] = "done"
    if explicit is not None:
        _write_report(
            report_root,
            "07_explicit_reference_diagnostic.md",
            REQUIRED_REPORTS["07_explicit_reference_diagnostic.md"],
            f"Status: `{explicit.get('status')}`.  {explicit.get('reason', '')}\n\n"
            "```json\n" + json.dumps(_json_safe(explicit), indent=2, sort_keys=True) + "\n```",
        )
        completed["07_explicit_reference_diagnostic.md"] = "done"
        _write_report(
            report_root,
            "08_transport_scheme_decision.md",
            REQUIRED_REPORTS["08_transport_scheme_decision.md"],
            "No production transport replacement is selected from an implicit-stability observation.  A donor-bound "
            "SSPRK2 reference must be whole-domain positivity-safe; an active-only cap is not presented as such a reference. "
            "The actual active-CFL ladder and donor-bound reference record are below.  A frozen boundary-state rate is not "
            "a universal feasibility bound and does not by itself establish backward-Euler temporal diffusion.  The next "
            "action depends on the actual ladder result: complete its diagnostic budget when incomplete, or evaluate a "
            "conservative characteristic or semi-Lagrangian remap only after a genuine source-reported min-dt block.\n\n```json\n"
            + json.dumps(_json_safe({
                "implicit_ladder_status": implicit.get("status"),
                "actual_implicit_brute_force_policies": implicit.get("brute_force_policies", []),
                "explicit_reference": explicit,
            }), indent=2, sort_keys=True)
            + "\n```",
        )
        completed["08_transport_scheme_decision.md"] = "done"
    if crosscheck is not None:
        _write_report(
            report_root,
            "09_cohort_eulerian_crosscheck.md",
            REQUIRED_REPORTS["09_cohort_eulerian_crosscheck.md"],
            f"Status: `{crosscheck.get('status')}`.  {crosscheck.get('reason', crosscheck.get('limitation', ''))}\n\n"
            + _report_body_table(crosscheck.get("rows", []), ("record_type", "time_h", "metric", "relative_error", "pass")),
        )
        completed["09_cohort_eulerian_crosscheck.md"] = "done"
    if authority is not None:
        _write_report(
            report_root,
            "10_eulerian_authority_v2.md",
            REQUIRED_REPORTS["10_eulerian_authority_v2.md"],
            f"Status: `{authority.get('status')}`.  {authority.get('reason', '')}",
        )
        completed["10_eulerian_authority_v2.md"] = "done"
    if beta_pf is not None:
        _write_report(
            report_root,
            "11_beta_only_cohort_pf_comparison.md",
            REQUIRED_REPORTS["11_beta_only_cohort_pf_comparison.md"],
            f"Status: `{beta_pf.get('status')}`.  {beta_pf.get('reason', '')}",
        )
        completed["11_beta_only_cohort_pf_comparison.md"] = "done"
    _write_report(
        report_root,
        "12_model_role_boundary.md",
        REQUIRED_REPORTS["12_model_role_boundary.md"],
        "Eulerian KWN is the smooth population-balance representation; the discrete cohort solver is the event-aware "
        "characteristic comparator.  Frozen PF/CUDA evidence is neither rerun nor made an authority before the KWN gates close. "
        "No local GP release, GP→beta source, physical retuning, or online coupling was performed.",
    )
    completed["12_model_role_boundary.md"] = "done"
    _write_report(
        report_root,
        "13_final_acceptance_report.md",
        REQUIRED_REPORTS["13_final_acceptance_report.md"],
        f"Top-level status: `{final.get('STATUS')}`.\n\n"
        "```json\n" + json.dumps(_json_safe(final), indent=2, sort_keys=True) + "\n```",
    )
    completed["13_final_acceptance_report.md"] = "done"
    _write_report(
        report_root,
        "14_reproduction_commands.md",
        REQUIRED_REPORTS["14_reproduction_commands.md"],
        "```bash\n"
        "# Finite lower-boundary gates and common 0--0.1 h active-CFL ladder\n"
        f"PYTHONPATH=src python3 scripts/run_kwn_lower_boundary_time_accuracy_v1.py all\n\n"
        "# Explicitly authorize the expensive exact canonical cohort path\n"
        f"PYTHONPATH=src python3 scripts/run_kwn_lower_boundary_time_accuracy_v1.py crosscheck --allow-long-cohort --cohort-points-per-cell 2\n"
        "```\n\n"
        f"Executed command: `{command}`.  This runner does not invoke CUDA/PF.",
    )
    completed["14_reproduction_commands.md"] = "done"
    _write_unfilled_reports(report_root, completed)


def _final_summary(
    *,
    launch: Mapping[str, Any],
    baseline: Mapping[str, Any],
    contract: Mapping[str, Any] | None,
    parity: Mapping[str, Any] | None,
    lower: Mapping[str, Any] | None,
    implicit: Mapping[str, Any] | None,
    explicit: Mapping[str, Any] | None,
    crosscheck: Mapping[str, Any] | None,
    authority: Mapping[str, Any] | None,
    beta_pf: Mapping[str, Any] | None,
) -> dict[str, Any]:
    first_cfl = (implicit or {}).get("cfl_rows", [{}])[0] if (implicit or {}).get("cfl_rows") else {}
    rows = (implicit or {}).get("rows", [])
    candidate = (implicit or {}).get("candidate_active_cfl")
    estimated = next(
        (row.get("estimated_48h_runtime_s") for row in rows if row.get("policy") == candidate),
        "NOT_PROJECTED_FROM_SHORT_DIAGNOSTIC",
    )
    runs = (implicit or {}).get("runs", [])
    terminal_audit = runs[0].feasibility.get("terminal_cfl_audit", {}) if runs else {}
    accepted_boundary_rows = [
        row for row in (implicit or {}).get("cfl_rows", [])
        if bool(row.get("boundary_active", False))
    ]
    capped_boundary_rows = [
        row for row in accepted_boundary_rows
        if row.get("policy") != "current_implicit_policy"
    ]
    boundary_active_summary: dict[str, Any] = {
        "first_accepted_step_cfl": first_cfl.get("boundary_active_cfl", "NOT_RUN"),
        "all_policies_accepted_boundary_active_step_count": len(accepted_boundary_rows),
        "all_policies_max_accepted_boundary_active_cfl": max(
            (float(row.get("boundary_active_cfl", 0.0)) for row in accepted_boundary_rows),
            default=0.0,
        ),
        "capped_policies_accepted_boundary_active_step_count": len(capped_boundary_rows),
        "capped_policies_max_accepted_boundary_active_cfl": max(
            (float(row.get("boundary_active_cfl", 0.0)) for row in capped_boundary_rows),
            default=0.0,
        ),
        "terminal_dt_1_s_audit_boundary_cfl": terminal_audit.get("boundary_active_cfl", "NOT_RUN"),
        "terminal_dt_1_s_audit_boundary_active": terminal_audit.get("boundary_active", "NOT_RUN"),
    }
    lower_status = (lower or {}).get("status", "FAIL_LOWER_BOUNDARY_OPERATOR_PARITY")
    cross_status = (crosscheck or {}).get("status", "BLOCKED_PREREQUISITE_GATE")
    implicit_status = (implicit or {}).get("status", "NOT_RUN")
    top = _top_status(
        lower={"status": lower_status},
        implicit=implicit or {"status": "FAIL_EULERIAN_TIME_ACCURACY"},
        crosscheck={"status": cross_status},
        authority=authority or {},
        beta_pf=beta_pf or {},
    )
    baseline_provenance = baseline.get("provenance", {}) if isinstance(baseline, Mapping) else {}
    root_cause = (
        "LOWER_BOUNDARY_OPERATOR_MISMATCH_RESOLVED"
        if lower_status == "PASS_LOWER_BOUNDARY_OPERATOR_PARITY"
        else "EULERIAN_LOWER_FACE_PREVIOUSLY_USED_G_FIRST_CELL_CENTER_WHILE_COHORT_EVENT_USED_PHYSICAL_RMIN"
    )
    if lower_status != "PASS_LOWER_BOUNDARY_OPERATOR_PARITY":
        secondary_root_cause = "NOT_REACHED_BEFORE_LOWER_BOUNDARY_GATE"
    elif implicit_status == "BRUTE_FORCE_CFL_NOT_PRACTICAL":
        secondary_root_cause = "SOURCE_REPORTED_ACTIVE_CFL_MIN_DT_LIMIT"
    elif implicit_status == "INCOMPLETE_DIAGNOSTIC_BUDGET":
        secondary_root_cause = "IMPLICIT_TIME_ACCURACY_UNQUALIFIED_DIAGNOSTIC_BUDGET"
    elif implicit_status == "FAIL_EULERIAN_TIME_ACCURACY":
        secondary_root_cause = "IMPLICIT_ACTIVE_CFL_SELF_CONVERGENCE_NOT_QUALIFIED"
    else:
        secondary_root_cause = "IMPLICIT_ACTIVE_CFL_CANDIDATE_PENDING_EXACT_COHORT_CROSSCHECK"
    if implicit_status == "BRUTE_FORCE_CFL_NOT_PRACTICAL":
        next_action = (
            "Preserve the source-reported min-dt evidence and evaluate a conservative characteristic or "
            "semi-Lagrangian remap with the same physical-Rmin contract before the cohort crosscheck, authority ladder, "
            "PF comparison, or local GP release."
        )
    elif implicit_status == "INCOMPLETE_DIAGNOSTIC_BUDGET":
        next_action = (
            "Continue the actual shared-horizon active-CFL ladder with a sufficient diagnostic budget; do not infer "
            "a transport-scheme block or run the cohort crosscheck, authority ladder, PF comparison, or local GP release yet."
        )
    elif implicit_status == "FAIL_EULERIAN_TIME_ACCURACY":
        next_action = (
            "Inspect the completed self-convergence evidence before selecting a different accuracy policy or a transport "
            "scheme upgrade; do not run the cohort crosscheck, authority ladder, PF comparison, or local GP release yet."
        )
    else:
        next_action = (
            "Run the explicitly authorized exact canonical cohort--Eulerian 2% crosscheck for the provisional active-CFL "
            "candidate before the authority ladder, PF comparison, or local GP release."
        )
    p0_blockers = [
        f"IMPLICIT_ACCURACY_LADDER_{implicit_status}",
        cross_status,
        "No time-qualified Eulerian solver/policy exists for the retained 2% cohort gate or v2 authority.",
        "No frozen beta-only PF direction comparison before cohort--Eulerian shared-operator parity passes.",
    ]
    if implicit_status == "BRUTE_FORCE_CFL_NOT_PRACTICAL":
        p0_blockers.insert(1, "BRUTE_FORCE_CFL_NOT_PRACTICAL_FROM_ACTUAL_SOLVER_MIN_DT_REJECTION")
    return {
        "STATUS": top,
        "BRANCH": launch.get("git_branch_at_launch"),
        "COMMIT": launch.get("git_head_at_launch"),
        "BASELINE_REPRODUCED": baseline.get("status"),
        "VALIDATION_CONTRACT_HASH": None if contract is None else contract.get("validation_contract_hash"),
        "PHYSICAL_RMIN": None if contract is None else contract.get("R_boundary_m"),
        "BOUNDARY_GROWTH_VELOCITY_PARITY": None if parity is None else parity.get("status"),
        "BOUNDARY_INVENTORY_PARITY": None if parity is None else parity.get("status"),
        "BOUNDARY_NUMBER_FLUX_PARITY": None if parity is None else parity.get("status"),
        "BOUNDARY_ANALYTIC_BENCHMARK": (lower or {}).get("checks", {}).get("analytic_absorbing_benchmarks", False),
        "LOWER_BOUNDARY_OPERATOR_PARITY": lower_status,
        "GLOBAL_MAX_CFL": first_cfl.get("global_max_cfl", "NOT_RUN"),
        "ACTIVE_M0_CFL": first_cfl.get("active_M0_cfl", "NOT_RUN"),
        "ACTIVE_M3_CFL": first_cfl.get("active_M3_cfl", "NOT_RUN"),
        "LOWER_TAIL_ACTIVE_CFL": first_cfl.get("lower_tail_active_cfl", "NOT_RUN"),
        "BOUNDARY_ACTIVE_CFL": boundary_active_summary,
        "PRIMARY_ROOT_CAUSE": root_cause,
        "SECONDARY_ROOT_CAUSE": secondary_root_cause,
        "IMPLICIT_ACCURACY_LADDER": implicit_status,
        "PASSING_ACTIVE_CFL": candidate or "NOT_ASSIGNED_UNTIL_COHORT_2_PERCENT_CROSSCHECK",
        "ESTIMATED_48H_COST": estimated,
        "DIAGNOSTIC_EXPLICIT": None if explicit is None else explicit.get("status"),
        "EXPLICIT_VS_COHORT": "NOT_RUN_WHOLE_DOMAIN_DONOR_BOUND_NOT_PRACTICAL",
        "IMPLICIT_VS_COHORT": cross_status,
        "FINAL_EULERIAN_SOLVER": "CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE_UNQUALIFIED",
        "EULERIAN_AUTHORITY_GRID_V2": None if authority is None else authority.get("authority_grid"),
        "EULERIAN_AUTHORITY_CONFIG_HASH": "NOT_ASSIGNED",
        "EULERIAN_RESTART": "BOUNDARY_UNIT_B7_ONLY" if (lower or {}).get("checks", {}).get("boundary_unit_tests") else "NOT_RUN",
        "EULERIAN_MAX_RESIDUAL": "NOT_RUN_FINAL_AUTHORITY",
        "COHORT_EULERIAN_CROSSCHECK": cross_status,
        "N_M0_ERROR": (crosscheck or {}).get("maximum_errors", {}).get("N_m0_m3", "NOT_EVALUATED"),
        "RMEAN_ERROR": (crosscheck or {}).get("maximum_errors", {}).get("Rmean_m", "NOT_EVALUATED"),
        "RMEAN3_ERROR": (crosscheck or {}).get("maximum_errors", {}).get("Rmean3_m3", "NOT_EVALUATED"),
        "SV_ERROR": (crosscheck or {}).get("maximum_errors", {}).get("Sv_m_inv", "NOT_EVALUATED"),
        "FBETA_ERROR": (crosscheck or {}).get("maximum_errors", {}).get("f_beta", "NOT_EVALUATED"),
        "XMATRIX_ERROR": (crosscheck or {}).get("maximum_errors", {}).get("matrix_xB", "NOT_EVALUATED"),
        "CUMULATIVE_NUMBER_DISSOLUTION_ERROR": (crosscheck or {}).get("maximum_errors", {}).get("cumulative_number_dissolution_m3", "NOT_EVALUATED"),
        "CUMULATIVE_MOL_B_ERROR": (crosscheck or {}).get("maximum_errors", {}).get("cumulative_mol_B_returned_mol_m3", "NOT_EVALUATED"),
        "BETA_INITIAL_STATE_IDENTITY": "NOT_RUN_BEFORE_COHORT_EULERIAN_SHARED_OPERATOR_PARITY",
        "BETA_ONLY_DIRECTION": (beta_pf or {}).get("status", "BLOCKED_PREREQUISITE_GATE"),
        "BETA_ONLY_TIMESCALE": "NOT_RUN",
        "MEAN_FIELD_PF_GAP": "NOT_RUN",
        "PF_SOURCE_MODIFIED": launch.get("pf_source_modified"),
        "CUDA_RERUN": False,
        "PHYSICAL_RETUNING": launch.get("physical_retuning"),
        "LEGACY_SIX_PARTICLE_EULERIAN_P5": "FAIL_RETAINED",
        "HISTORICAL_AUTHORITY": "HISTORICAL_SMOOTH_3200_ONLY_V2_NOT_INHERITED",
        "TOP_5_FINDINGS": [
            "The immutable pre-change trace reproduced the old lower-boundary parity failure.",
            "Rmin is frozen as one exact binary64 grid edge for helper, FV face, and cohort event paths.",
            "Raw global CFL and active-population CFL are reported separately; a negligible tail is not silently made an accuracy limiter.",
            "Boundary flux diagnostics are not an additional matrix source; matrix composition is algebraically closed from current beta inventory.",
            "Each active-CFL policy is advanced with the source-owned self-consistent candidate step; a frozen boundary audit is telemetry, not a universal feasibility bound.",
        ],
        "P0_BLOCKERS": p0_blockers,
        "NEXT_ACTION": next_action,
        "LOCAL_GP_RELEASE_AUTHORIZED": "LOCAL_GP_RELEASE_NOT_AUTHORIZED",
        "KEY_REPORTS": [str(DEFAULT_REPORT_ROOT / name) for name in (
            "00_baseline_boundary_reproduction.md", "01_physical_lower_boundary_contract.md",
            "04_boundary_operator_parity.md", "05_courant_definition_and_audit.md",
            "06_implicit_time_accuracy.md", "09_cohort_eulerian_crosscheck.md",
            "13_final_acceptance_report.md",
        )],
        "baseline_historical_Rmin_hex": baseline_provenance.get("physical_rmin_hex"),
    }


def _write_outputs(
    *,
    output_root: Path,
    baseline: Mapping[str, Any],
    contract: Mapping[str, Any] | None,
    tests: Mapping[str, Any] | None,
    analytic: Mapping[str, Any] | None,
    parity: Mapping[str, Any] | None,
    implicit: Mapping[str, Any] | None,
    explicit: Mapping[str, Any] | None,
    crosscheck: Mapping[str, Any] | None,
    authority: Mapping[str, Any] | None,
    beta_pf: Mapping[str, Any] | None,
    launch: Mapping[str, Any],
    final: Mapping[str, Any],
    context: CanonicalContext | None,
) -> None:
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / "figures").mkdir(parents=True, exist_ok=True)
    (output_root / "figures" / "README.md").write_text(
        "# Plot-ready KWN lower-boundary evidence\n\nThe gate evidence is the CSV/JSON data in the parent directory; no figure substitutes for a numerical gate.\n",
        encoding="utf-8",
    )
    if contract is not None:
        _write_json(output_root / "lower_boundary_contract.json", contract)
    _write_csv(output_root / "boundary_unit_tests.csv", [] if tests is None else tests.get("rows", []))
    _write_csv(output_root / "analytic_boundary_benchmarks.csv", [] if analytic is None else analytic.get("rows", []))
    _write_csv(output_root / "boundary_operator_parity.csv", [] if parity is None else parity.get("rows", []))
    _write_csv(output_root / "courant_audit.csv", [] if implicit is None else implicit.get("cfl_rows", []))
    _write_csv(output_root / "implicit_accuracy_ladder.csv", [] if implicit is None else implicit.get("rows", []))
    _write_csv(
        output_root / "explicit_reference_comparison.csv",
        [] if explicit is None else [explicit],
    )
    _write_csv(output_root / "lower_tail_diagnostics.csv", [] if implicit is None else implicit.get("lower_tail_rows", []))
    _write_csv(output_root / "final_cohort_eulerian_crosscheck.csv", [] if crosscheck is None else crosscheck.get("rows", []))
    _write_json(output_root / "eulerian_authority_v2.json", authority or {"status": "NOT_RUN"})
    _write_csv(output_root / "beta_only_cohort_pf_comparison.csv", [] if beta_pf is None else [beta_pf])
    source_paths = (
        "scripts/run_kwn_lower_boundary_time_accuracy_v1.py",
        "src/kwn_mvp/lower_boundary.py",
        "src/kwn_mvp/solver.py",
        "src/kwn_mvp/cohort_solver.py",
        "src/kwn_mvp/population_metrics.py",
        "src/kwn_mvp/ledger.py",
    )
    provenance = {
        "schema_version": "KWN_LOWER_BOUNDARY_TIME_ACCURACY_ANALYSIS_PROVENANCE_V1",
        "launch": launch,
        "binary": {"python_executable": sys.executable, "python_version": sys.version},
        "baseline": baseline,
        "source_sha256": {path: _sha256_file(ROOT / path) for path in source_paths if (ROOT / path).is_file()},
        "configuration": None if context is None else {
            "canonical_hash": context.canonical_hash,
            "canonical_mapping_hash": _canonical_hash(context.mapping),
            "validation_contract_hash": context.contract_hash,
            "fixture_hash": context.fixture_hash,
            "source_initial_psd_hash": context.source_initial_psd_hash,
            "cell_number_roundtrip_relative_error": context.cell_number_roundtrip_relative_error,
        },
        "forbidden_actions": {
            "pf_source_modified": launch.get("pf_source_modified"),
            "cuda_rerun": False,
            "physical_retuning": launch.get("physical_retuning"),
            "local_gp_release": False,
        },
        "final_status": final.get("STATUS"),
    }
    _write_json(output_root / "analysis_provenance.json", provenance)
    _write_json(output_root / "final_acceptance.json", final)


def _run_workflow(arguments: argparse.Namespace) -> dict[str, Any]:
    output_root = arguments.output_root.resolve()
    report_root = arguments.report_root.resolve()
    command = str(arguments.command)
    launch = _launch_context()
    baseline = _baseline_reproduction(output_root)
    context: CanonicalContext | None = None
    contract: dict[str, Any] | None = None
    tests: dict[str, Any] | None = None
    analytic: dict[str, Any] | None = None
    parity: dict[str, Any] | None = None
    lower: dict[str, Any] | None = None
    implicit: dict[str, Any] | None = None
    explicit: dict[str, Any] | None = None
    crosscheck: dict[str, Any] | None = None
    authority: dict[str, Any] | None = None
    beta_pf: dict[str, Any] | None = None
    finite_commands = {"all", "boundary", "analytic", "implicit", "crosscheck"}
    try:
        if command in finite_commands:
            context = _build_canonical_context()
            contract = _boundary_contract(context, baseline)
        if command in {"all", "boundary", "analytic", "implicit", "crosscheck"}:
            tests = _boundary_unit_tests()
            parity = _physical_boundary_parity(context) if context is not None else None
        if command in {"all", "analytic", "implicit", "crosscheck"}:
            analytic = _analytic_benchmarks(context, fast=arguments.analytic_fast) if context is not None else None
        if all(item is not None for item in (contract, tests, analytic, parity)):
            lower = _lower_boundary_operator_gate(
                baseline=baseline, contract=contract, tests=tests, analytic=analytic, parity=parity
            )
        else:
            lower = {"status": "FAIL_LOWER_BOUNDARY_OPERATOR_PARITY", "checks": {"incomplete_command": False}}
        if lower["status"] == "PASS_LOWER_BOUNDARY_OPERATOR_PARITY" and command in {"all", "implicit", "crosscheck"}:
            implicit = _implicit_accuracy_ladder(
                context, max_steps=arguments.max_steps, max_wall_s=arguments.max_wall_s
            )
        else:
            implicit = {
                "status": "BLOCKED_LOWER_BOUNDARY_OPERATOR_PARITY",
                "rows": [{"status": "BLOCKED_LOWER_BOUNDARY_OPERATOR_PARITY", "reason": lower["status"]}],
                "cfl_rows": [],
                "lower_tail_rows": [],
                "candidate_active_cfl": None,
            }
        explicit = _explicit_stub(lower, implicit, context)
        if command == "crosscheck":
            if lower["status"] != "PASS_LOWER_BOUNDARY_OPERATOR_PARITY":
                crosscheck = _blocked_crosscheck("physical lower-boundary operator parity did not pass")
            elif implicit.get("status") != "INCOMPLETE_NO_COHORT_REFERENCE":
                crosscheck = _blocked_crosscheck(
                    "no time-qualified Eulerian policy exists: "
                    f"implicit ladder status is {implicit.get('status')}"
                )
            elif not arguments.allow_long_cohort:
                crosscheck = _blocked_crosscheck("rerun with --allow-long-cohort; no proxy replaces the exact canonical cohort comparator")
            else:
                cap = arguments.crosscheck_active_cfl
                crosscheck = _cohort_eulerian_crosscheck(
                    context,
                    active_cfl_cap=cap,
                    points_per_cell=arguments.cohort_points_per_cell,
                    max_steps=arguments.max_steps,
                    max_wall_s=arguments.max_wall_s,
                )
        else:
            crosscheck_reason = (
                "no time-qualified Eulerian policy exists: "
                f"implicit ladder status is {implicit.get('status')}"
                if implicit.get("status") != "INCOMPLETE_NO_COHORT_REFERENCE"
                else "full canonical cohort--Eulerian crosscheck is not part of this invocation; use the explicit crosscheck command after lower parity and time-accuracy qualification."
            )
            crosscheck = _blocked_crosscheck(
                crosscheck_reason
            )
        authority = _authority_stub(crosscheck, context)
        beta_pf = _beta_pf_stub(crosscheck)
    except Exception as error:
        trace = traceback.format_exc()
        lower = lower or {"status": "FAIL_LOWER_BOUNDARY_OPERATOR_PARITY", "checks": {"runner_exception": False}}
        implicit = implicit or {"status": "FAIL_EULERIAN_TIME_ACCURACY", "rows": [], "cfl_rows": [], "lower_tail_rows": []}
        explicit = explicit or {"status": "NOT_RUN", "reason": f"runner exception: {type(error).__name__}: {error}"}
        crosscheck = crosscheck or _blocked_crosscheck(f"runner exception: {type(error).__name__}: {error}")
        authority = _authority_stub(crosscheck, context)
        beta_pf = _beta_pf_stub(crosscheck)
        if tests is None:
            tests = {"status": "NOT_RUN", "rows": []}
        if analytic is None:
            analytic = {"status": "NOT_RUN", "rows": []}
        if parity is None:
            parity = {"status": "NOT_RUN", "rows": []}
        if contract is None and context is not None:
            contract = _boundary_contract(context, baseline)
        launch = {**launch, "runner_exception": trace}
    final = _final_summary(
        launch=launch,
        baseline=baseline,
        contract=contract,
        parity=parity,
        lower=lower,
        implicit=implicit,
        explicit=explicit,
        crosscheck=crosscheck,
        authority=authority,
        beta_pf=beta_pf,
    )
    _write_outputs(
        output_root=output_root,
        baseline=baseline,
        contract=contract,
        tests=tests,
        analytic=analytic,
        parity=parity,
        implicit=implicit,
        explicit=explicit,
        crosscheck=crosscheck,
        authority=authority,
        beta_pf=beta_pf,
        launch=launch,
        final=final,
        context=context,
    )
    _write_stage_reports(
        report_root=report_root,
        output_root=output_root,
        baseline=baseline,
        contract=contract,
        tests=tests,
        analytic=analytic,
        parity=parity,
        lower=lower,
        implicit=implicit,
        explicit=explicit,
        crosscheck=crosscheck,
        authority=authority,
        beta_pf=beta_pf,
        final=final,
        command=" ".join(sys.argv),
    )
    return final


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=("all", "boundary", "analytic", "implicit", "crosscheck"),
        help="gate stage to execute; crosscheck requires explicit long-run authorization",
    )
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--report-root", type=Path, default=DEFAULT_REPORT_ROOT)
    parser.add_argument(
        "--max-steps", type=int, default=250000,
        help="hard accepted-step ceiling for a numerical diagnostic; failure is recorded rather than silently running longer",
    )
    parser.add_argument(
        "--max-wall-s", type=float, default=20.0,
        help="per-policy wall-time ceiling for a diagnostic; budget exhaustion is recorded without inferring a feasibility bound",
    )
    parser.add_argument(
        "--analytic-fast", action="store_true", default=True,
        help="use the default compact 32/64/128 analytic refinement ladder",
    )
    parser.add_argument(
        "--analytic-full", action="store_false", dest="analytic_fast",
        help="use the longer 64/128/256 analytic refinement ladder",
    )
    parser.add_argument(
        "--allow-long-cohort", action="store_true",
        help="authorize the full canonical discrete-cohort trajectory for the crosscheck command",
    )
    parser.add_argument("--cohort-points-per-cell", type=int, default=2)
    parser.add_argument("--crosscheck-active-cfl", type=float, default=0.125)
    arguments = parser.parse_args()
    if arguments.max_steps <= 0:
        parser.error("--max-steps must be positive")
    if arguments.max_wall_s <= 0.0:
        parser.error("--max-wall-s must be positive")
    if arguments.cohort_points_per_cell <= 0:
        parser.error("--cohort-points-per-cell must be positive")
    if arguments.crosscheck_active_cfl <= 0.0:
        parser.error("--crosscheck-active-cfl must be positive")
    final = _run_workflow(arguments)
    print(json.dumps(_json_safe(final), indent=2, sort_keys=True))
    return 0 if str(final["STATUS"]).startswith("PASS_") else 2


if __name__ == "__main__":
    raise SystemExit(main())
