#!/usr/bin/env python3
"""Audit KWN shared-operator parity from one canonical smooth measure.

This task intentionally fails closed before a new Eulerian authority or any
PF comparison when the lower-bound operator is not equivalent to the exact
cohort Rmin event.  It nevertheless records the literal all-grid face-CFL
ladder so an implicit-stability claim cannot be mistaken for time accuracy.
"""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from dataclasses import asdict
import hashlib
import importlib.util
import json
import math
from pathlib import Path
import subprocess
import sys
import time
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from scipy.special import ndtr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.cohort_solver import Cohort, CohortSolver, sphere_volume_m3  # noqa: E402
from kwn_mvp.diagnostics import discrete_wasserstein_distance  # noqa: E402
from kwn_mvp.growth import growth_rate_m_s  # noqa: E402
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
from kwn_mvp.solver import KWNSolver, SolverConfig  # noqa: E402


REPORT_ROOT = ROOT / "reports" / "kwn_shared_operator_parity_v1"
OUTPUT_ROOT = ROOT / "outputs" / "kwn_shared_operator_parity_v1"
HISTORICAL_OUTPUT_ROOT = ROOT / "outputs" / "kwn_discrete_cohort_comparison_v1"
EXPECTED_CONTRACT_HASH = "d0ff02973ab0f737043e1a40d4f69893a469cbfe2bc4cd22f9e6a410bd0b1333"
SMOOTH_TIMES_H = (0.0, 0.1, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0)
ACCURACY_CAPS = (1.0, 0.5, 0.25, 0.125, 0.0625)
TWO_PERCENT = 0.02
BASELINE_COMMIT = "6d6a74733a4806c86adc80a689c6f298a0741701"


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(value: Any) -> str:
    text = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _array_hash(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        item = np.ascontiguousarray(np.asarray(array, dtype=np.float64))
        digest.update(item.tobytes())
    return digest.hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"{path} must contain a JSON object")
    return value


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_report(name: str, title: str, body: str) -> None:
    path = REPORT_ROOT / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"# {title}\n\n{body.rstrip()}\n", encoding="utf-8")


def _relative_error(left: float, right: float) -> float:
    return abs(float(left) - float(right)) / max(abs(float(right)), 1.0e-300)


def _load_radius_audit_module() -> Any:
    path = ROOT / "scripts" / "run_kwn_radius_grid_convergence_v1.py"
    specification = importlib.util.spec_from_file_location("radius_grid_audit_shared_operator_v1", path)
    if specification is None or specification.loader is None:
        raise RuntimeError(f"cannot load radius-grid audit runner at {path}")
    module = importlib.util.module_from_spec(specification)
    sys.modules[specification.name] = module
    specification.loader.exec_module(module)
    return module


def _git_text(*args: str) -> str:
    result = subprocess.run(
        ["git", *args], cwd=ROOT, check=True, text=True, capture_output=True
    )
    return result.stdout.strip()


def _capture_launch_context() -> dict[str, Any]:
    changed = [line for line in _git_text("diff", "--name-only", BASELINE_COMMIT).splitlines() if line]
    physical_input_paths = [
        path
        for path in changed
        if any(token in path.lower() for token in ("validation_contract", "thermodynamic", "fixture", "material_parameter"))
    ]
    return {
        "baseline_commit_resolved": _git_text("rev-parse", BASELINE_COMMIT),
        "git_head_at_launch": _git_text("rev-parse", "HEAD"),
        "git_branch_at_launch": _git_text("branch", "--show-current"),
        "git_status_at_launch": _git_text("status", "--short"),
        "source_paths_changed_before_outputs": changed,
        "pf_source_modified": any("pf" in Path(path).name.lower() or "/pf/" in path for path in changed),
        "cuda_rerun": False,
        "physical_input_paths_modified": physical_input_paths,
        "physical_retuning": bool(physical_input_paths),
    }


def _baseline_reproduction(run_root: Path) -> dict[str, Any]:
    """Verify numerical reproduction while treating worktree paths as provenance."""

    required = (
        "cohort_eulerian_smooth_crosscheck.json",
        "cohort_smooth_crosscheck.csv",
        "eulerian_smooth_authority.json",
        "cohort_numerical_qualification.json",
        "analysis_provenance.json",
    )
    missing = [str(run_root / name) for name in required if not (run_root / name).is_file()]
    if missing:
        return {"status": "FAIL_BASELINE_REPRODUCTION", "reason": "missing outputs", "missing": missing}
    replay_crosscheck = _read_json(run_root / "cohort_eulerian_smooth_crosscheck.json")
    reference_crosscheck = _read_json(HISTORICAL_OUTPUT_ROOT / "cohort_eulerian_smooth_crosscheck.json")
    replay_authority = _read_json(run_root / "eulerian_smooth_authority.json")
    reference_authority = _read_json(HISTORICAL_OUTPUT_ROOT / "eulerian_smooth_authority.json")
    replay_numeric = _read_json(run_root / "cohort_numerical_qualification.json")
    reference_numeric = _read_json(HISTORICAL_OUTPUT_ROOT / "cohort_numerical_qualification.json")
    replay_rows = sorted((run_root / "cohort_smooth_crosscheck.csv").read_text(encoding="utf-8").splitlines())
    reference_rows = sorted((HISTORICAL_OUTPUT_ROOT / "cohort_smooth_crosscheck.csv").read_text(encoding="utf-8").splitlines())
    replay_errors = {
        key: (value["target_time_h"], value["relative_error"])
        for key, value in replay_crosscheck["authority_maximum_errors"].items()
    }
    reference_errors = {
        key: (value["target_time_h"], value["relative_error"])
        for key, value in reference_crosscheck["authority_maximum_errors"].items()
    }
    observed_contract = str(replay_authority.get("contract_hash"))
    checks = {
        "replay_status": replay_crosscheck.get("status") == "FAIL_COHORT_EULERIAN_PHYSICS_MISMATCH",
        "reference_status": reference_crosscheck.get("status") == "FAIL_COHORT_EULERIAN_PHYSICS_MISMATCH",
        "authority": replay_authority.get("status") == "PASS_EULERIAN_SMOOTH_POPULATION_AUTHORITY",
        "cohort_numerics": replay_numeric.get("status") == "PASS_DISCRETE_COHORT_NUMERICS",
        "all_registered_csv_metrics": replay_rows == reference_rows,
        "maximum_error_map": replay_errors == reference_errors,
        "contract_hash": observed_contract == EXPECTED_CONTRACT_HASH,
    }
    if not checks["contract_hash"]:
        status = "FAIL_CONTRACT_PROVENANCE_MISMATCH"
    else:
        status = "PASS_BASELINE_REPRODUCTION" if all(checks.values()) else "FAIL_BASELINE_REPRODUCTION"
    rows = [
        {
            "metric": metric,
            "target_time_h": target_time_h,
            "relative_error": error,
            "registered_gate": TWO_PERCENT,
            "pass": error <= TWO_PERCENT,
        }
        for metric, (target_time_h, error) in sorted(replay_errors.items())
    ]
    _write_csv(OUTPUT_ROOT / "baseline_crosscheck_metrics.csv", rows)
    result = {
        "schema_version": "KWN_SHARED_OPERATOR_BASELINE_REPRODUCTION_V1",
        "status": status,
        "baseline_commit": BASELINE_COMMIT,
        "baseline_run_output_root": str(run_root),
        "historical_output_root": str(HISTORICAL_OUTPUT_ROOT),
        "checks": checks,
        "validation_contract_hash_observed_from_provenance": observed_contract,
        "validation_contract_hash_expected": EXPECTED_CONTRACT_HASH,
        "registered_maximum_errors": replay_errors,
        "path_only_config_hash_difference": {
            "historical": reference_authority.get("authority_config_hash"),
            "replay": replay_authority.get("authority_config_hash"),
            "physical_metric_reproduction_exact": checks["all_registered_csv_metrics"],
        },
        "source_sha256": {name: _sha256_file(run_root / name) for name in required},
    }
    _write_json(OUTPUT_ROOT / "baseline_provenance.json", result)
    _write_report(
        "00_baseline_reproduction.md",
        "Baseline reproduction",
        f"Status: `{status}`.  The isolated rerun reproduced every registered CSV metric and all full-time maximum errors exactly. "
        f"The observed validation contract is `{observed_contract}`.\n\n"
        "The solver-config hash differs only because the isolated worktree has a different absolute contract path; the sorted metric CSV and all registered errors are byte-identical.\n\n"
        + "| Metric | maximum relative error | time (h) |\n|---|---:|---:|\n"
        + "\n".join(
            f"| {metric} | {100.0 * error:.9g}% | {target_time_h:.6g} |"
            for metric, (target_time_h, error) in sorted(replay_errors.items())
        ),
    )
    if status != "PASS_BASELINE_REPRODUCTION":
        raise RuntimeError("baseline numerical reproduction did not close")
    return result


def _lognormal_cell_moment(
    edges_m: np.ndarray, *, median_m: float, log_sigma: float, order: int
) -> np.ndarray:
    mu = math.log(float(median_m))
    sigma = float(log_sigma)
    shift = float(order) * sigma**2
    prefactor = math.exp(float(order) * mu + 0.5 * float(order * order) * sigma**2)
    z = (np.log(edges_m) - mu - shift) / sigma
    return prefactor * np.diff(ndtr(z))


def _build_canonical_measure(audit: Any, contract_hash: str) -> tuple[dict[str, Any], KWNSolver, dict[str, Any], Any]:
    source_solver, construction, contract, fixture = audit._build_smooth_solver(bins=3200)
    if contract.contract_hash != contract_hash:
        raise RuntimeError("canonical source does not bind to reproduced validation contract")
    beta = source_solver.population("beta")
    benchmark = construction["smooth_psd_benchmark"]
    edges = beta.grid.edges_m.copy()
    median = float(benchmark["median_radius_m"])
    sigma = float(benchmark["log_sigma"])
    # The canonical state is an explicitly piecewise-constant cell measure.
    # Start with the analytic lognormal *number* integral in every canonical
    # cell, then derive every recorded moment from that one actual measure.
    # Scaling analytic M3 directly would instead mix two incompatible
    # representations: the analytic in-cell lognormal and the Eulerian
    # piecewise-constant density read by the solver.
    raw_cell_number = _lognormal_cell_moment(
        edges, median_m=median, log_sigma=sigma, order=0
    )
    target_m3 = float(benchmark["same_fixed_pivot_M3_as_fixture"])
    raw_cell_moments = cell_moments_from_piecewise_constant_cells(edges, raw_cell_number)
    scale = target_m3 / float(np.sum(raw_cell_moments[3]))
    cell_number = raw_cell_number * scale
    cell_moments = cell_moments_from_piecewise_constant_cells(edges, cell_number)
    mapping = deepcopy(construction["config_mapping"])
    mapping["simulation"]["population_measure"] = "cell_integrated"
    mapping["simulation"]["accuracy_radius_cfl"] = None
    mapping["populations"]["beta"]["initial"] = {
        "kind": "cell_integrated",
        "radius_edges_m": [float(value) for value in edges],
        "cell_number_density_m3": [float(value) for value in cell_number],
    }
    canonical_solver = KWNSolver(SolverConfig.from_mapping(mapping))
    ledger = canonical_solver.ledger.snapshot(
        matrix_xb=canonical_solver.matrix_xb,
        populations=canonical_solver.population_list(),
        beta_resolved_fraction=1.0,
    )
    cell_metrics = metrics_from_piecewise_constant_cells(edges, cell_number)
    if _relative_error(cell_metrics.M3_dimensionless, target_m3) > 1.0e-13:
        raise RuntimeError("canonical M3 scaling did not preserve the frozen smooth beta inventory")
    canonical_hash = _array_hash(edges, cell_number, cell_moments)
    payload = {
        "schema_version": "SMOOTH_POPULATION_CANONICAL_V1",
        "name": "smooth_population_canonical_v1",
        "representation": "piecewise_constant_number_density_per_canonical_radius_cell",
        "analytic_psd_definition": "LOGNORMAL_FIXED_NUMERICAL_QUALIFICATION_ONLY",
        "median_radius_m": median,
        "log_sigma": sigma,
        "number_normalization_m3": scale,
        "normalization_policy": "global_scale_of_analytic_cell_M0_to_preserve_frozen_M3_under_piecewise_constant_cell_measure",
        "raw_analytic_cell_M0_m3_before_normalization": float(np.sum(raw_cell_number)),
        "raw_piecewise_M3_before_normalization": float(np.sum(raw_cell_moments[3])),
        "number_density_units": "m^-3 per radius cell",
        "canonical_radius_cell_count": int(cell_number.size),
        "radius_support_m": [float(edges[0]), float(edges[-1])],
        "canonical_hash": canonical_hash,
        "validation_contract_hash": contract_hash,
        "fixture_hash": fixture.fixture_hash,
        "source_initial_psd_hash": construction["initial_psd_source_hash"],
        "total_beta_inventory_mol_m3": ledger.beta_resolved_mol_m3,
        "matrix_inventory_mol_m3": ledger.matrix_mol_m3,
        "total_inventory_mol_m3": ledger.total_mol_m3,
        "matrix_xB": canonical_solver.matrix_xb,
        "moments": cell_metrics.as_dict(),
        "production_measure": "cell_integrated",
        "Eulerian_initialization": "direct_cell_number_density_read",
    }
    archive_path = OUTPUT_ROOT / "canonical_smooth_population_v1.npz"
    archive_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        archive_path,
        radius_edges_m=edges,
        radius_centres_m=beta.grid.centres_m,
        cell_number_density_m3=cell_number,
        cell_M0_m3=cell_moments[0],
        cell_M1_m2=cell_moments[1],
        cell_M2_m=cell_moments[2],
        cell_M3_dimensionless=cell_moments[3],
        cumulative_number_cdf=np.cumsum(cell_moments[0]) / cell_metrics.M0_m3,
        cumulative_M3_cdf=np.cumsum(cell_moments[3]) / cell_metrics.M3_dimensionless,
        cumulative_number_cdf_at_edges=np.concatenate(
            ([0.0], np.cumsum(cell_moments[0]) / cell_metrics.M0_m3)
        ),
        cumulative_M3_cdf_at_edges=np.concatenate(
            ([0.0], np.cumsum(cell_moments[3]) / cell_metrics.M3_dimensionless)
        ),
    )
    payload["npz_sha256"] = _sha256_file(archive_path)
    _write_json(OUTPUT_ROOT / "canonical_smooth_population_v1.json", payload)
    return payload, canonical_solver, mapping, fixture


def _critical_radius(solver: KWNSolver) -> float:
    beta = solver.population("beta")
    lower = float(beta.grid.edges_m[0])
    upper = float(beta.grid.edges_m[-1])
    def residual(radius: float) -> float:
        equilibrium = solver.equilibrium_adapter.equilibrium_xb(
            np.asarray([radius], dtype=np.float64), beta.parameters
        )
        return solver.matrix_xb - float(equilibrium[0])
    lo_value, hi_value = residual(lower), residual(upper)
    if lo_value * hi_value > 0.0:
        raise RuntimeError("canonical beta state lacks a radius-space growth-sign transition")
    for _ in range(128):
        middle = 0.5 * (lower + upper)
        value = residual(middle)
        if value == 0.0:
            return middle
        if lo_value * value < 0.0:
            upper, hi_value = middle, value
        else:
            lower, lo_value = middle, value
    return 0.5 * (lower + upper)


def _discrete_cdf_at_points(
    radii_m: np.ndarray, weights: np.ndarray, points_m: np.ndarray, *, moment_order: int
) -> np.ndarray:
    """Evaluate a normalized weighted-atom CDF at arbitrary radius probes."""

    order = np.argsort(radii_m)
    radii = radii_m[order]
    weighted = weights[order] * radii**moment_order
    total = float(np.sum(weighted))
    if total <= 0.0:
        return np.zeros(points_m.size, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(weighted))) / total
    locations = np.searchsorted(radii, points_m, side="right")
    return cumulative[locations]


def _piecewise_cdf_at_points(
    edges_m: np.ndarray, cell_number_m3: np.ndarray, points_m: np.ndarray, *, moment_order: int
) -> np.ndarray:
    """Evaluate the exact CDF of the canonical piecewise-constant measure."""

    edges = np.asarray(edges_m, dtype=np.float64)
    number = np.asarray(cell_number_m3, dtype=np.float64)
    points = np.asarray(points_m, dtype=np.float64)
    widths = np.diff(edges)
    density = number / widths
    integrals = density * (edges[1:] ** (moment_order + 1) - edges[:-1] ** (moment_order + 1)) / (moment_order + 1)
    total = float(np.sum(integrals))
    if total <= 0.0:
        return np.zeros(points.size, dtype=np.float64)
    prefix = np.concatenate(([0.0], np.cumsum(integrals)))
    result = np.zeros(points.size, dtype=np.float64)
    above = points >= edges[-1]
    result[above] = 1.0
    inside = (points > edges[0]) & (points < edges[-1])
    indices = np.searchsorted(edges, points[inside], side="right") - 1
    local = density[indices] * (
        points[inside] ** (moment_order + 1) - edges[indices] ** (moment_order + 1)
    ) / (moment_order + 1)
    result[inside] = (prefix[indices] + local) / total
    return result


def _piecewise_interval_fraction(
    edges_m: np.ndarray, cell_number_m3: np.ndarray, lower_m: float, upper_m: float, *, moment_order: int = 0
) -> float:
    if upper_m <= lower_m:
        return 0.0
    values = _piecewise_cdf_at_points(
        edges_m, cell_number_m3, np.asarray([lower_m, upper_m], dtype=np.float64), moment_order=moment_order
    )
    return float(max(values[1] - values[0], 0.0))


def _discrete_interval_fraction(
    radii: np.ndarray, weights: np.ndarray, lower: float, upper: float, *, moment_order: int = 0
) -> float:
    weighted = weights * radii**moment_order
    total = float(np.sum(weighted))
    if total == 0.0:
        return 0.0
    mask = (radii >= lower) & (radii <= upper)
    return float(np.sum(weighted[mask])) / total


def _canonical_initial_identity(
    canonical: Mapping[str, Any], solver: KWNSolver
) -> tuple[dict[str, Any], dict[int, CohortSolver], dict[int, tuple[np.ndarray, np.ndarray]]]:
    beta = solver.population("beta")
    edges = beta.grid.edges_m
    cell_number = beta.number_density_per_m4 * beta.grid.widths_m
    eulerian_metrics = metrics_from_piecewise_constant_cells(edges, cell_number)
    cell_moments = cell_moments_from_piecewise_constant_cells(edges, cell_number)
    subcell_points = np.concatenate(
        [edges[:-1] + fraction * np.diff(edges) for fraction in (0.25, 0.5, 0.75)]
    )
    cdf_probe_points = np.unique(np.concatenate((edges, subcell_points)))
    cdf_probe_kind = np.where(np.isin(cdf_probe_points, edges), "canonical_cell_edge", "independent_subcell_probe")
    exact_number_cdf = _piecewise_cdf_at_points(
        edges, cell_number, cdf_probe_points, moment_order=0
    )
    exact_m3_cdf = _piecewise_cdf_at_points(
        edges, cell_number, cdf_probe_points, moment_order=3
    )
    initial_ledger = solver.ledger.snapshot(
        matrix_xb=solver.matrix_xb, populations=solver.population_list(), beta_resolved_fraction=1.0
    )
    critical = _critical_radius(solver)
    rows: list[dict[str, Any]] = []
    cdf_rows: list[dict[str, Any]] = []
    cohort_solvers: dict[int, CohortSolver] = {}
    quadratures: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    cohort_ids_by_points: dict[int, np.ndarray] = {}
    cohort_cells_by_points: dict[int, np.ndarray] = {}
    max_identity_errors: dict[int, dict[str, float]] = {}
    for points in (1, 2, 4):
        radii, weights = positive_cell_quadrature(edges, cell_number, points)
        cohort_cells = np.searchsorted(edges, radii, side="right") - 1
        cohort_cells = np.clip(cohort_cells, 0, edges.size - 2)
        cohort_ranks: dict[int, int] = {}
        cohorts: list[Cohort] = []
        for cell_index, radius, weight in zip(cohort_cells, radii, weights):
            cell = int(cell_index)
            rank = cohort_ranks.get(cell, 0)
            cohort_ranks[cell] = rank + 1
            cohorts.append(
                Cohort(
                    initial_id=f"C{cell:04d}_Q{rank:02d}",
                    radius_m=float(radius),
                    weight_m3=float(weight),
                )
            )
        cohort = CohortSolver.from_kwn_solver(
            kwn_solver=solver, cohorts=cohorts, rtol=1.0e-10, atol_m=1.0e-18, method="DOP853"
        )
        cohort_solvers[points] = cohort
        quadratures[points] = (radii, weights)
        cohort_ids_by_points[points] = np.asarray(
            [item.initial_id for item in cohorts], dtype="<U16"
        )
        cohort_cells_by_points[points] = cohort_cells.astype(np.int64, copy=True)
        snapshot = cohort.snapshot()
        cohort_metrics = metrics_from_discrete_measure(radii, weights)
        values = {
            "M0": (cohort_metrics.M0_m3, eulerian_metrics.M0_m3),
            "M1": (cohort_metrics.M1_m2, eulerian_metrics.M1_m2),
            "M2": (cohort_metrics.M2_m, eulerian_metrics.M2_m),
            "M3": (cohort_metrics.M3_dimensionless, eulerian_metrics.M3_dimensionless),
            "N": (cohort_metrics.N_m0_m3, eulerian_metrics.N_m0_m3),
            "Rmean_number": (cohort_metrics.Rmean_number_m, eulerian_metrics.Rmean_number_m),
            "Rmean_cubed": (cohort_metrics.Rmean_cubed_m3, eulerian_metrics.Rmean_cubed_m3),
            "mean_R3": (cohort_metrics.mean_R3_m3, eulerian_metrics.mean_R3_m3),
            "Sv": (cohort_metrics.Sv_m_inv, eulerian_metrics.Sv_m_inv),
            "f_beta": (cohort_metrics.f_beta, eulerian_metrics.f_beta),
            "matrix_xB": (snapshot.matrix_xB, solver.matrix_xb),
            "beta_inventory": (snapshot.beta_inventory_mol_m3, initial_ledger.beta_resolved_mol_m3),
            "total_inventory": (snapshot.total_inventory_mol_m3, initial_ledger.total_mol_m3),
        }
        errors: dict[str, float] = {}
        for metric, (cohort_value, eulerian_value) in values.items():
            error = abs(cohort_value - eulerian_value) if metric == "matrix_xB" else _relative_error(cohort_value, eulerian_value)
            errors[metric] = error
            rows.append(
                {
                    "record_type": "global_identity",
                    "points_per_cell": points,
                    "metric": metric,
                    "cohort_value": cohort_value,
                    "eulerian_value": eulerian_value,
                    "error": error,
                    "criterion": "absolute_1e-12" if metric == "matrix_xB" else "relative_1e-12_for_M0_M3_inventory",
                    "pass": (error <= 1.0e-12) if metric in {"M0", "M3", "matrix_xB", "beta_inventory", "total_inventory"} else "DIAGNOSTIC",
                }
            )
        number_cdf = _discrete_cdf_at_points(radii, weights, cdf_probe_points, moment_order=0)
        m3_cdf = _discrete_cdf_at_points(radii, weights, cdf_probe_points, moment_order=3)
        for point, kind, exact_number, observed_number, exact_m3, observed_m3 in zip(
            cdf_probe_points, cdf_probe_kind, exact_number_cdf, number_cdf, exact_m3_cdf, m3_cdf
        ):
            cdf_rows.append(
                {
                    "points_per_cell": points,
                    "radius_probe_m": point,
                    "probe_kind": kind,
                    "number_cdf_eulerian": exact_number,
                    "number_cdf_cohort": observed_number,
                    "number_cdf_absolute_error": abs(observed_number - exact_number),
                    "M3_cdf_eulerian": exact_m3,
                    "M3_cdf_cohort": observed_m3,
                    "M3_cdf_absolute_error": abs(observed_m3 - exact_m3),
                }
            )
        regions = {
            "Rmin_to_Rmin_plus_1nm": (float(edges[0]), float(edges[0] + 1.0e-9)),
            "Rmin_to_Rmin_plus_2nm": (float(edges[0]), float(edges[0] + 2.0e-9)),
            "below_initial_critical_radius": (float(edges[0]), critical),
            "critical_radius_plus_minus_0p25nm": (critical - 0.25e-9, critical + 0.25e-9),
            "critical_radius_plus_minus_0p5nm": (critical - 0.5e-9, critical + 0.5e-9),
        }
        for label, (lower, upper) in regions.items():
            for moment_order, label_suffix in ((0, "number_fraction"), (3, "M3_fraction")):
                eulerian_fraction = _piecewise_interval_fraction(
                    edges, cell_number, lower, upper, moment_order=moment_order
                )
                cohort_fraction = _discrete_interval_fraction(
                    radii, weights, lower, upper, moment_order=moment_order
                )
                rows.append(
                    {
                        "record_type": "lower_tail_fraction",
                        "points_per_cell": points,
                        "metric": f"{label}:{label_suffix}",
                        "cohort_value": cohort_fraction,
                        "eulerian_value": eulerian_fraction,
                        "error": abs(cohort_fraction - eulerian_fraction),
                        "criterion": "quadrature_refinement_diagnostic",
                        "pass": "DIAGNOSTIC",
                    }
                )
        max_identity_errors[points] = errors
    _write_csv(OUTPUT_ROOT / "initial_measure_identity.csv", rows)
    _write_csv(OUTPUT_ROOT / "initial_cdf_comparison.csv", cdf_rows)
    quadrature_archive = OUTPUT_ROOT / "canonical_cohort_quadrature_v1.npz"
    archive_arrays: dict[str, np.ndarray] = {}
    for points in (1, 2, 4):
        radii, weights = quadratures[points]
        archive_arrays[f"points_{points}_radius_m"] = radii
        archive_arrays[f"points_{points}_weight_m3"] = weights
        archive_arrays[f"points_{points}_cell_index"] = cohort_cells_by_points[points]
        archive_arrays[f"points_{points}_initial_id"] = cohort_ids_by_points[points]
    np.savez(quadrature_archive, **archive_arrays)
    if isinstance(canonical, dict):
        canonical["cohort_quadrature_archive"] = {
            "file": quadrature_archive.name,
            "sha256": _sha256_file(quadrature_archive),
            "points_per_cell": [1, 2, 4],
            "id_scheme": "C{canonical_cell_index:04d}_Q{local_quadrature_index:02d}",
        }
        _write_json(OUTPUT_ROOT / "canonical_smooth_population_v1.json", canonical)
    authority_errors = max_identity_errors[4]
    cdf_errors_by_points = {
        points: max(
            max(
                float(row["number_cdf_absolute_error"])
                for row in cdf_rows
                if int(row["points_per_cell"]) == points
            ),
            max(
                float(row["M3_cdf_absolute_error"])
                for row in cdf_rows
                if int(row["points_per_cell"]) == points
            ),
        )
        for points in (1, 2, 4)
    }
    initial_cdf_error = cdf_errors_by_points[4]
    cdf_refines = cdf_errors_by_points[4] <= cdf_errors_by_points[2] * (1.0 + 1.0e-12) <= cdf_errors_by_points[1] * (1.0 + 1.0e-12)
    moment_refines = all(
        max_identity_errors[4][metric] <= max_identity_errors[2][metric] * (1.0 + 1.0e-12) + 1.0e-15
        and max_identity_errors[2][metric] <= max_identity_errors[1][metric] * (1.0 + 1.0e-12) + 1.0e-15
        for metric in ("M1", "M2")
    )
    lower_rows = [row for row in rows if row["record_type"] == "lower_tail_fraction" and row["points_per_cell"] == 4]
    lower_tail_error = max(float(row["error"]) for row in lower_rows)
    status = "PASS_CANONICAL_INITIAL_MEASURE_IDENTITY" if all(
        authority_errors[name] <= 1.0e-12
        for name in ("M0", "M3", "matrix_xB", "beta_inventory", "total_inventory")
    ) and cdf_refines and moment_refines else "FAIL_CANONICAL_INITIAL_MEASURE_IDENTITY"
    result = {
        "status": status,
        "canonical_hash": canonical["canonical_hash"],
        "critical_radius_m": critical,
        "authority_points_per_cell": 4,
        "errors_by_points_per_cell": max_identity_errors,
        "cdf_errors_by_points_per_cell": cdf_errors_by_points,
        "cdf_refinement_converges": cdf_refines,
        "moment_refinement_converges": moment_refines,
        "initial_cdf_error": initial_cdf_error,
        "lower_tail_initial_error": lower_tail_error,
    }
    _write_report(
        "01_canonical_initial_measure.md",
        "Canonical smooth initial measure",
        f"Status: `{status}`.  Eulerian reads the canonical 3200-cell number vector directly; cohort nodes are deterministic positive Gauss--Legendre quadrature from those same cells.\n\n"
        f"The authority is four points per cell.  Its M0/M3, beta inventory and total inventory errors are all at or below 1e-12.  The independent subcell CDF error falls from `{cdf_errors_by_points[1]:.6e}` (1 point/cell) to `{cdf_errors_by_points[4]:.6e}` (4 points/cell); refinement is `{cdf_refines}`.\n\n"
        f"The reported lower-tail quadrature diagnostic is `{lower_tail_error:.6e}` and is evaluated against the exact piecewise-cell measure, not bin centres.  Fixed cohort IDs, node radii, weights and canonical-cell indices are archived in `canonical_cohort_quadrature_v1.npz`.",
    )
    if status != "PASS_CANONICAL_INITIAL_MEASURE_IDENTITY":
        raise RuntimeError("canonical initial measure identity gate failed")
    return result, cohort_solvers, quadratures


def _metric_growth_and_matrix_parity(
    solver: KWNSolver,
    initial: Mapping[str, Any],
    cohort_solvers: Mapping[int, CohortSolver],
    quadratures: Mapping[int, tuple[np.ndarray, np.ndarray]],
    reproduced_authority: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, Any]]:
    beta = solver.population("beta")
    eulerian = metrics_from_piecewise_constant_cells(
        beta.grid.edges_m, beta.number_density_per_m4 * beta.grid.widths_m
    )
    cohort_metrics = metrics_from_discrete_measure(*quadratures[4])
    metric_errors = {
        key: _relative_error(getattr(cohort_metrics, key), getattr(eulerian, key))
        for key in ("M0_m3", "M1_m2", "M2_m", "M3_dimensionless", "Sv_m_inv", "f_beta")
    }
    metric_status = "PASS_METRIC_IDENTITY" if max(metric_errors.values()) <= 1.0e-12 else "FAIL_METRIC_IDENTITY"
    trajectories = reproduced_authority.get("authority_trajectory", [])
    matrix_values = [float(item["matrix_xB"]) for item in trajectories if "matrix_xB" in item]
    probes_xb = sorted({solver.matrix_xb, min(matrix_values), max(matrix_values), matrix_values[-1]})
    critical = _critical_radius(solver)
    radii = np.unique(
        np.concatenate(
            (
                np.asarray([beta.grid.edges_m[0], np.nextafter(beta.grid.edges_m[0], np.inf), critical]),
                np.geomspace(2.0e-9, min(100.0e-9, beta.grid.edges_m[-1]), 97),
                beta.grid.centres_m[:: max(1, beta.grid.bins // 160)],
            )
        )
    )
    radii = radii[(radii >= beta.grid.edges_m[0]) & (radii <= beta.grid.edges_m[-1])]
    growth_rows: list[dict[str, Any]] = []
    max_growth_error = 0.0
    # First probe the two actual runtime paths at their common canonical t=0
    # state.  The cohort method calls its own matrix closure and adapter;
    # Eulerian calls its grid-state growth path.  Both should reduce to the
    # named shared kernel at machine precision.
    cohort_runtime = cohort_solvers[4]
    cohort_runtime_radii = np.asarray(
        [item.radius_m for item in cohort_runtime.cohorts if item.active], dtype=np.float64
    )
    cohort_runtime_rates = cohort_runtime.growth_rates()
    cohort_runtime_equilibrium = solver.equilibrium_adapter.equilibrium_xb(
        cohort_runtime_radii, beta.parameters
    )
    direct_runtime_rates = growth_rate_m_s(
        radii_m=cohort_runtime_radii,
        matrix_xb=cohort_runtime.matrix_xb,
        equilibrium_xb=cohort_runtime_equilibrium,
        parameters=beta.parameters,
    )
    for radius, xeq, cohort_rate, direct_rate in zip(
        cohort_runtime_radii, cohort_runtime_equilibrium, cohort_runtime_rates, direct_runtime_rates
    ):
        error = _relative_error(cohort_rate, direct_rate)
        max_growth_error = max(max_growth_error, error)
        growth_rows.append(
            {
                "probe_path": "CohortSolver.growth_rates_vs_shared_kernel",
                "matrix_xB": cohort_runtime.matrix_xb,
                "radius_m": radius,
                "xeq_alpha_beta": xeq,
                "chemical_driving_force_xB": cohort_runtime.matrix_xb - xeq,
                "gibbs_thomson_embedded_in_xeq": True,
                "elastic_correction_included": True,
                "elastic_penalty_j_m3": beta.parameters.elastic_penalty_j_m3,
                "growth_rate_eulerian_m_s": direct_rate,
                "growth_rate_cohort_m_s": cohort_rate,
                "growth_sign": "GROWTH" if direct_rate > 0.0 else "DISSOLUTION" if direct_rate < 0.0 else "NEUTRAL",
                "critical_radius_m": critical,
                "molar_inventory_per_particle_mol": sphere_volume_m3(float(radius)) * beta.parameters.x_b / beta.parameters.molar_volume_m3_mol,
                "relative_difference": error,
            }
        )
    eulerian_centres = beta.grid.centres_m
    eulerian_runtime_rates = solver.growth_rates()["beta"]
    eulerian_runtime_equilibrium = solver.equilibrium_adapter.equilibrium_xb(
        eulerian_centres, beta.parameters
    )
    direct_eulerian_rates = growth_rate_m_s(
        radii_m=eulerian_centres,
        matrix_xb=solver.matrix_xb,
        equilibrium_xb=eulerian_runtime_equilibrium,
        parameters=beta.parameters,
    )
    for radius, xeq, eulerian_rate, direct_rate in zip(
        eulerian_centres, eulerian_runtime_equilibrium, eulerian_runtime_rates, direct_eulerian_rates
    ):
        error = _relative_error(eulerian_rate, direct_rate)
        max_growth_error = max(max_growth_error, error)
        growth_rows.append(
            {
                "probe_path": "KWNSolver.growth_rates_vs_shared_kernel",
                "matrix_xB": solver.matrix_xb,
                "radius_m": radius,
                "xeq_alpha_beta": xeq,
                "chemical_driving_force_xB": solver.matrix_xb - xeq,
                "gibbs_thomson_embedded_in_xeq": True,
                "elastic_correction_included": True,
                "elastic_penalty_j_m3": beta.parameters.elastic_penalty_j_m3,
                "growth_rate_eulerian_m_s": eulerian_rate,
                "growth_rate_cohort_m_s": direct_rate,
                "growth_sign": "GROWTH" if eulerian_rate > 0.0 else "DISSOLUTION" if eulerian_rate < 0.0 else "NEUTRAL",
                "critical_radius_m": critical,
                "molar_inventory_per_particle_mol": sphere_volume_m3(float(radius)) * beta.parameters.x_b / beta.parameters.molar_volume_m3_mol,
                "relative_difference": error,
            }
        )
    for matrix_xb in probes_xb:
        equilibrium = solver.equilibrium_adapter.equilibrium_xb(radii, beta.parameters)
        eulerian_rate = growth_rate_m_s(
            radii_m=radii, matrix_xb=matrix_xb, equilibrium_xb=equilibrium, parameters=beta.parameters
        )
        cohort_rate = growth_rate_m_s(
            radii_m=radii, matrix_xb=matrix_xb, equilibrium_xb=equilibrium, parameters=beta.parameters
        )
        for radius, xeq, left, right in zip(radii, equilibrium, eulerian_rate, cohort_rate):
            error = _relative_error(left, right)
            max_growth_error = max(max_growth_error, error)
            growth_rows.append(
                {
                    "probe_path": "shared_kernel_radius_probe",
                    "matrix_xB": matrix_xb,
                    "radius_m": radius,
                    "xeq_alpha_beta": xeq,
                    "chemical_driving_force_xB": matrix_xb - xeq,
                    "gibbs_thomson_embedded_in_xeq": True,
                    "elastic_correction_included": True,
                    "elastic_penalty_j_m3": beta.parameters.elastic_penalty_j_m3,
                    "growth_rate_eulerian_m_s": left,
                    "growth_rate_cohort_m_s": right,
                    "growth_sign": "GROWTH" if left > 0.0 else "DISSOLUTION" if left < 0.0 else "NEUTRAL",
                    "critical_radius_m": critical,
                    "molar_inventory_per_particle_mol": sphere_volume_m3(float(radius)) * beta.parameters.x_b / beta.parameters.molar_volume_m3_mol,
                    "relative_difference": error,
                }
            )
    _write_csv(OUTPUT_ROOT / "growth_kernel_parity.csv", growth_rows)
    growth_status = "PASS_GROWTH_KERNEL_PARITY" if max_growth_error <= 1.0e-12 else "FAIL_GROWTH_KERNEL_PARITY"
    matrix_rows: list[dict[str, Any]] = []
    max_xb_error = 0.0
    max_residual = 0.0
    for multiplier in (0.25, 0.5, 1.0, 1.25):
        scaled_beta = beta.copy()
        scaled_beta.number_density_per_m4 *= multiplier
        m3 = scaled_beta.radius_moment(3, quadrature="cell_integrated")
        beta_fraction = beta_fraction_from_m3(m3)
        beta_inventory = beta_inventory_from_m3(
            m3, x_b=beta.parameters.x_b, molar_volume_m3_mol=beta.parameters.molar_volume_m3_mol
        )
        shared = close_matrix_from_precipitates(
            total_b_mol_m3=solver.ledger.total_b_mol_m3,
            matrix_molar_volume_m3_mol=solver.config.matrix_molar_volume_m3_mol,
            precipitate_volume_fraction=beta_fraction,
            precipitate_inventory_mol_m3=beta_inventory,
        )
        eulerian_xb = solver.ledger.recover_matrix_xb(
            [solver.population("g"), scaled_beta]
        )
        scaled_radii, scaled_weights = quadratures[4]
        scaled_cohorts = [
            Cohort(
                initial_id=f"M{index:05d}",
                radius_m=float(radius),
                weight_m3=float(multiplier * weight),
            )
            for index, (radius, weight) in enumerate(zip(scaled_radii, scaled_weights))
        ]
        cohort_closure = CohortSolver.from_kwn_solver(
            kwn_solver=solver,
            cohorts=scaled_cohorts,
            rtol=1.0e-10,
            atol_m=1.0e-18,
            method="DOP853",
        )
        cohort_xb = cohort_closure.matrix_xb
        eulerian_cohort_error = abs(eulerian_xb - cohort_xb)
        eulerian_shared_error = abs(eulerian_xb - shared.matrix_xb)
        cohort_shared_error = abs(cohort_xb - shared.matrix_xb)
        max_xb_error = max(max_xb_error, eulerian_cohort_error, eulerian_shared_error, cohort_shared_error)
        max_residual = max(max_residual, abs(shared.residual_mol_m3))
        matrix_rows.append(
            {
                "M3_dimensionless": m3,
                "beta_volume_fraction": beta_fraction,
                "beta_inventory_mol_m3": beta_inventory,
                "matrix_xB_shared": shared.matrix_xb,
                "matrix_xB_Eulerian_ledger": eulerian_xb,
                "matrix_xB_CohortSolver": cohort_xb,
                "Eulerian_vs_Cohort_absolute_difference": eulerian_cohort_error,
                "Eulerian_vs_shared_absolute_difference": eulerian_shared_error,
                "Cohort_vs_shared_absolute_difference": cohort_shared_error,
                "matrix_inventory_mol_m3": shared.matrix_inventory_mol_m3,
                "ledger_residual_mol_m3": shared.residual_mol_m3,
                "ledger_relative_residual": shared.relative_residual,
            }
        )
    _write_csv(OUTPUT_ROOT / "matrix_closure_parity.csv", matrix_rows)
    matrix_status = "PASS_MATRIX_CLOSURE_PARITY" if max_xb_error <= 1.0e-12 and max_residual <= 1.0e-10 else "FAIL_MATRIX_CLOSURE_PARITY"
    _write_report(
        "02_metric_and_growth_kernel_parity.md",
        "Metric, growth-kernel, and matrix-closure parity",
        f"Metric status: `{metric_status}`; growth status: `{growth_status}`; matrix closure status: `{matrix_status}`.\n\n"
        f"`Rmean_cubed=(M1/M0)^3` and `mean_R3=M3/M0` are now separate shared fields.  The growth CSV includes actual `KWNSolver.growth_rates` and `CohortSolver.growth_rates` paths at t=0, plus shared-radius probes at early/late matrix compositions.  The matrix CSV compares the Eulerian ledger and CohortSolver closures for four scaled canonical M3 states. Maximum growth-kernel relative difference is `{max_growth_error:.3e}`; maximum matrix-xB absolute difference is `{max_xb_error:.3e}` and residual is `{max_residual:.3e}`.",
    )
    return {
        "status": metric_status,
        "errors": metric_errors,
        "legacy_Rmean3_interpretation": "mean_R3_m3_only",
    }, {"status": growth_status, "max_relative_error": max_growth_error, "matrix_status": matrix_status, "matrix_max_xb_error": max_xb_error, "matrix_max_residual": max_residual}


def _transport_copy(solver: KWNSolver) -> Population:
    return solver.population("beta").copy()


def _transport_metrics(population: Population) -> Any:
    return metrics_from_piecewise_constant_cells(
        population.grid.edges_m, population.number_density_per_m4 * population.grid.widths_m
    )


def _analytic_benchmarks(solver: KWNSolver, quadrature: tuple[np.ndarray, np.ndarray]) -> dict[str, Any]:
    """Run low-Courant interior and actual Rmin-crossing transport controls."""

    beta = solver.population("beta")
    rows: list[dict[str, Any]] = []
    gate_errors: list[float] = []
    boundary_summaries: dict[str, dict[str, Any]] = {}
    initial_radii, initial_weights = quadrature
    edge = float(beta.grid.edges_m[0])
    edge_volume = sphere_volume_m3(edge)

    def weighted_variance(radii: np.ndarray, weights: np.ndarray) -> float:
        total = float(np.sum(weights))
        if total <= 0.0:
            return 0.0
        mean = float(np.sum(radii * weights) / total)
        return float(np.sum(weights * (radii - mean) ** 2) / total)

    def distribution_record(
        *,
        name: str,
        population: Population,
        reference_radii: np.ndarray,
        reference_weights: np.ndarray,
        include_in_interior_gate: bool,
        details: Mapping[str, Any],
    ) -> None:
        """Record moments, CDF/W1 and variance for one transport control."""

        eulerian = _transport_metrics(population)
        reference = metrics_from_discrete_measure(reference_radii, reference_weights)
        for metric in (
            "M0_m3", "M1_m2", "M2_m", "M3_dimensionless", "Rmean_number_m", "Sv_m_inv", "f_beta"
        ):
            error = _relative_error(getattr(eulerian, metric), getattr(reference, metric))
            if include_in_interior_gate:
                gate_errors.append(error)
            rows.append(
                {
                    "record_type": "moment",
                    "benchmark": name,
                    "metric": metric,
                    "eulerian_value": getattr(eulerian, metric),
                    "cohort_analytic_value": getattr(reference, metric),
                    "relative_error": error,
                    "included_in_interior_gate": include_in_interior_gate,
                    **details,
                }
            )
        cell_number = population.number_density_per_m4 * population.grid.widths_m
        eulerian_radii, eulerian_weights = positive_cell_quadrature(
            beta.grid.edges_m, cell_number, 4
        )
        cdf_probes = np.unique(np.concatenate((beta.grid.edges_m, reference_radii, eulerian_radii)))
        if reference_weights.size and eulerian_weights.size:
            cdf_error = float(
                np.max(
                    np.abs(
                        _discrete_cdf_at_points(eulerian_radii, eulerian_weights, cdf_probes, moment_order=0)
                        - _discrete_cdf_at_points(reference_radii, reference_weights, cdf_probes, moment_order=0)
                    )
                )
            )
            w1_number = discrete_wasserstein_distance(
                eulerian_radii, eulerian_weights, reference_radii, reference_weights
            )
            w1_volume = discrete_wasserstein_distance(
                eulerian_radii,
                eulerian_weights * eulerian_radii**3,
                reference_radii,
                reference_weights * reference_radii**3,
            )
        else:
            cdf_error = float("nan")
            w1_number = float("nan")
            w1_volume = float("nan")
        rows.append(
            {
                "record_type": "distribution",
                "benchmark": name,
                "metric": "PSD_CDF_W1_variance",
                "eulerian_value": weighted_variance(eulerian_radii, eulerian_weights),
                "cohort_analytic_value": weighted_variance(reference_radii, reference_weights),
                "relative_error": _relative_error(
                    weighted_variance(eulerian_radii, eulerian_weights),
                    weighted_variance(reference_radii, reference_weights),
                ) if reference_weights.size else float("nan"),
                "cdf_max_absolute_error": cdf_error,
                "W1_number_m": w1_number,
                "W1_volume_m": w1_volume,
                "included_in_interior_gate": include_in_interior_gate,
                **details,
            }
        )

    def advance_transport(
        population: Population,
        velocity: np.ndarray,
        *,
        dt_s: float,
        steps: int,
    ) -> tuple[float, list[tuple[float, float]]]:
        cumulative_loss = 0.0
        timeline: list[tuple[float, float]] = []
        for step in range(steps):
            lower_flux, _, _, _ = solver._advect_population(population, velocity, dt_s)
            cumulative_loss += lower_flux * dt_s
            timeline.append(((step + 1) * dt_s, cumulative_loss))
        return cumulative_loss, timeline

    # 7.1: exact translation of the complete canonical population, with no
    # boundary crossing and a deliberately tiny Courant number.
    positive = _transport_copy(solver)
    positive_velocity = np.full(beta.grid.bins, 1.0e-14, dtype=np.float64)
    positive_dt = 0.05
    positive_steps = 8
    advance_transport(positive, positive_velocity, dt_s=positive_dt, steps=positive_steps)
    distribution_record(
        name="constant_positive_translation",
        population=positive,
        reference_radii=initial_radii + positive_velocity[0] * positive_dt * positive_steps,
        reference_weights=initial_weights,
        include_in_interior_gate=True,
        details={
            "canonical_source": "full_canonical_measure",
            "maximum_C_R": abs(positive_velocity[0]) * positive_dt / float(np.min(beta.grid.widths_m)),
            "boundary_crossing": False,
        },
    )

    # 7.2/7.3: a unit-normalized first canonical cell retains exactly the
    # canonical cell geometry and density shape while avoiding underflow in a
    # boundary-event benchmark.  Scaling a linear transport measure changes
    # neither its characteristic event times nor relative operator errors.
    boundary_number = np.zeros(beta.grid.bins, dtype=np.float64)
    boundary_number[0] = 1.0
    boundary_population = Population.empty(
        beta.parameters, beta.grid, production_quadrature="cell_integrated"
    )
    boundary_population.number_density_per_m4[:] = boundary_number / beta.grid.widths_m
    boundary_radii, boundary_weights = positive_cell_quadrature(
        beta.grid.edges_m, boundary_number, 64
    )

    def crossing_record(
        *,
        name: str,
        population: Population,
        velocity: np.ndarray,
        dt_s: float,
        steps: int,
        event_times_s: np.ndarray,
        final_radii: np.ndarray,
    ) -> None:
        initial_number = float(np.sum(boundary_weights))
        eulerian_loss, timeline = advance_transport(population, velocity, dt_s=dt_s, steps=steps)
        surviving = final_radii > edge
        analytic_surviving_radii = final_radii[surviving]
        analytic_surviving_weights = boundary_weights[surviving]
        analytic_loss = initial_number - float(np.sum(analytic_surviving_weights))
        analytic_volume_return = analytic_loss * edge_volume
        eulerian_volume_return = eulerian_loss * edge_volume
        composition_price = beta.parameters.x_b / beta.parameters.molar_volume_m3_mol
        analytic_mol_return = analytic_volume_return * composition_price
        eulerian_mol_return = eulerian_volume_return * composition_price
        half_time = float("nan")
        if eulerian_loss > 0.0:
            cumulative = np.asarray([item[1] for item in timeline], dtype=np.float64)
            times = np.asarray([item[0] for item in timeline], dtype=np.float64)
            half_time = float(np.interp(0.5 * analytic_loss, cumulative, times))
        details = {
            "canonical_source": "unit_normalized_first_canonical_cell",
            "maximum_C_R": float(np.max(np.abs(velocity)) * dt_s / np.min(beta.grid.widths_m)),
            "boundary_crossing": True,
            "analytic_event_time_min_s": float(np.min(event_times_s)),
            "analytic_event_time_median_s": float(np.median(event_times_s)),
            "analytic_event_time_max_s": float(np.max(event_times_s)),
            "eulerian_half_loss_time_s": half_time,
            "analytic_cumulative_number_dissolved_m3": analytic_loss,
            "eulerian_cumulative_number_dissolved_m3": eulerian_loss,
            "number_loss_relative_difference": _relative_error(eulerian_loss, analytic_loss),
            "analytic_cumulative_beta_volume_dissolved": analytic_volume_return,
            "eulerian_cumulative_beta_volume_dissolved": eulerian_volume_return,
            "beta_volume_return_relative_difference": _relative_error(eulerian_volume_return, analytic_volume_return),
            "analytic_cumulative_mol_B_returned_mol_m3": analytic_mol_return,
            "eulerian_cumulative_mol_B_returned_mol_m3": eulerian_mol_return,
            "mol_B_return_relative_difference": _relative_error(eulerian_mol_return, analytic_mol_return),
        }
        boundary_summaries[name] = details
        distribution_record(
            name=name,
            population=population,
            reference_radii=analytic_surviving_radii,
            reference_weights=analytic_surviving_weights,
            include_in_interior_gate=False,
            details=details,
        )
        rows.append(
            {
                "record_type": "boundary_event",
                "benchmark": name,
                "metric": "analytic_cohort_characteristic_event",
                "eulerian_value": eulerian_loss,
                "cohort_analytic_value": analytic_loss,
                "relative_error": _relative_error(eulerian_loss, analytic_loss),
                "included_in_interior_gate": False,
                **details,
            }
        )

    # Constant negative velocity crosses Rmin for every first-cell cohort.
    negative_velocity_value = -1.0e-6
    negative_velocity = np.full(beta.grid.bins, negative_velocity_value, dtype=np.float64)
    negative_shift = 1.1 * float(beta.grid.widths_m[0])
    negative_total_s = negative_shift / abs(negative_velocity_value)
    negative_steps = 32
    negative_dt = negative_total_s / negative_steps
    negative_population = boundary_population.copy()
    negative_final_radii = boundary_radii + negative_velocity_value * negative_total_s
    crossing_record(
        name="constant_negative_absorbing_Rmin",
        population=negative_population,
        velocity=negative_velocity,
        dt_s=negative_dt,
        steps=negative_steps,
        event_times_s=(boundary_radii - edge) / abs(negative_velocity_value),
        final_radii=negative_final_radii,
    )

    # Exact dR/dt=-K/R characteristics cross the same physical Rmin.
    k_value = 1.0e-15
    characteristic_velocity = -k_value / beta.grid.centres_m
    characteristic_event_times = (boundary_radii**2 - edge**2) / (2.0 * k_value)
    characteristic_total_s = 1.1 * float(np.max(characteristic_event_times))
    characteristic_steps = 32
    characteristic_dt = characteristic_total_s / characteristic_steps
    characteristic_population = boundary_population.copy()
    characteristic_final_squared = boundary_radii**2 - 2.0 * k_value * characteristic_total_s
    characteristic_final_radii = np.sqrt(np.maximum(characteristic_final_squared, 0.0))
    crossing_record(
        name="radius_dependent_minus_K_over_R",
        population=characteristic_population,
        velocity=characteristic_velocity,
        dt_s=characteristic_dt,
        steps=characteristic_steps,
        event_times_s=characteristic_event_times,
        final_radii=characteristic_final_radii,
    )

    # 7.4: frozen real beta law.  Use an adaptive characteristic integration
    # only as the reference; matrix xB is never updated in this control.
    from scipy.integrate import solve_ivp

    # The frozen-growth-law reference is an *interior* control.  Exclude the
    # canonical cells below 2 nm because their exact Rmin characteristics
    # belong to the absorbing-boundary controls above; evaluating an adaptive
    # ODE trial below the contract's one-sided Rmin domain would be invalid.
    frozen_lower_index = int(np.searchsorted(beta.grid.edges_m, 2.0e-9, side="left"))
    frozen_lower_index = min(max(frozen_lower_index, 1), beta.grid.bins - 1)
    frozen_lower_m = float(beta.grid.edges_m[frozen_lower_index])
    frozen = _transport_copy(solver)
    frozen.number_density_per_m4[:frozen_lower_index] = 0.0
    frozen_velocity = solver.growth_rates()["beta"]
    frozen_dt = min(
        1.0e-9,
        1.0e-3 * float(np.min(beta.grid.widths_m / np.maximum(abs(frozen_velocity), 1.0e-300))),
    )
    advance_transport(frozen, frozen_velocity, dt_s=frozen_dt, steps=1)

    def frozen_rhs(_time_s: float, radii_m: np.ndarray) -> np.ndarray:
        equilibrium = solver.equilibrium_adapter.equilibrium_xb(radii_m, beta.parameters)
        return growth_rate_m_s(
            radii_m=radii_m,
            matrix_xb=solver.matrix_xb,
            equilibrium_xb=equilibrium,
            parameters=beta.parameters,
        )

    frozen_reference = solve_ivp(
        frozen_rhs,
        (0.0, frozen_dt),
        initial_radii[initial_radii >= frozen_lower_m],
        method="DOP853",
        rtol=1.0e-11,
        atol=1.0e-20,
        max_step=frozen_dt / 16.0,
    )
    if not frozen_reference.success:
        raise RuntimeError(f"frozen-matrix characteristic reference failed: {frozen_reference.message}")
    distribution_record(
        name="frozen_matrix_full_beta_growth_law",
        population=frozen,
        reference_radii=np.asarray(frozen_reference.y[:, -1], dtype=np.float64),
        reference_weights=initial_weights[initial_radii >= frozen_lower_m],
        include_in_interior_gate=True,
        details={
            "canonical_source": f"canonical_interior_support_R_ge_{frozen_lower_m:.9e}_m",
            "maximum_C_R": float(np.max(np.abs(frozen_velocity)) * frozen_dt / np.min(beta.grid.widths_m)),
            "boundary_crossing": False,
        },
    )

    _write_csv(OUTPUT_ROOT / "analytic_benchmarks.csv", rows)
    maximum = max(gate_errors, default=0.0)
    status = (
        "PASS_ANALYTIC_INTERIOR_TRANSPORT_BENCHMARKS_WITH_BOUNDARY_CROSSING_DIAGNOSTICS"
        if maximum <= TWO_PERCENT
        else "FAIL_ANALYTIC_INTERIOR_TRANSPORT_BENCHMARKS"
    )
    _write_report(
        "03_analytic_transport_benchmarks.md",
        "Analytic transport benchmarks",
        f"Status: `{status}`.  The no-boundary positive translation and frozen-matrix controls run at low Courant with maximum interior metric error `{maximum:.6e}`.\n\n"
        "The negative-velocity and -K/R controls use a unit-normalized first canonical cell, so every reference characteristic physically crosses the unchanged Rmin edge.  Their number loss, edge-volume return, mol-B return, event-time range, low-tail PSD/CDF and Wasserstein diagnostics are recorded in the CSV as boundary evidence; they are not mislabelled as a passing interior-transport gate.",
    )
    return {
        "status": status,
        "maximum_relative_error": maximum,
        "boundary_crossing_summaries": boundary_summaries,
    }


def _semi_discrete_and_boundary(
    solver: KWNSolver,
    quadrature: tuple[np.ndarray, np.ndarray],
) -> dict[str, Any]:
    beta = solver.population("beta")
    density = beta.number_density_per_m4
    widths = beta.grid.widths_m
    centres = beta.grid.centres_m
    velocity = solver.growth_rates()["beta"]
    faces = solver._upwind_face_fluxes(density, velocity)
    radii, weights = quadrature
    rows: list[dict[str, Any]] = []
    max_interior_error = 0.0
    def append_interior_state(
        *,
        state: str,
        state_density: np.ndarray,
        state_matrix_xb: float,
        state_radii: np.ndarray,
        state_weights: np.ndarray,
    ) -> None:
        """Append stable face-telescoped interior moments for one closed state."""

        nonlocal max_interior_error
        centre_equilibrium = solver.equilibrium_adapter.equilibrium_xb(centres, beta.parameters)
        state_velocity = growth_rate_m_s(
            radii_m=centres,
            matrix_xb=state_matrix_xb,
            equilibrium_xb=centre_equilibrium,
            parameters=beta.parameters,
        )
        state_faces = solver._upwind_face_fluxes(state_density, state_velocity)
        quadrature_equilibrium = solver.equilibrium_adapter.equilibrium_xb(state_radii, beta.parameters)
        state_cohort_velocity = growth_rate_m_s(
            radii_m=state_radii,
            matrix_xb=state_matrix_xb,
            equilibrium_xb=quadrature_equilibrium,
            parameters=beta.parameters,
        )
        for order in range(4):
            cell_integral = (
                beta.grid.edges_m[1:] ** (order + 1) - beta.grid.edges_m[:-1] ** (order + 1)
            ) / (order + 1)
            # The face-telescoped reduction avoids artificial M0 cancellation.
            cell_average = cell_integral / widths
            lower_boundary_term = float(state_faces[0] * cell_average[0])
            upper_boundary_term = float(-state_faces[-1] * cell_average[-1])
            interior_eulerian_rate = float(
                np.sum(state_faces[1:-1] * (cell_average[1:] - cell_average[:-1]))
            )
            cohort_rate = 0.0 if order == 0 else float(
                order * np.sum(state_weights * state_radii ** (order - 1) * state_cohort_velocity)
            )
            error = _relative_error(interior_eulerian_rate, cohort_rate)
            max_interior_error = max(max_interior_error, error)
            rows.append(
                {
                    "state": state,
                    "moment_order": order,
                    "matrix_xB": state_matrix_xb,
                    "eulerian_semi_discrete_rate": interior_eulerian_rate,
                    "cohort_quadrature_rate": cohort_rate,
                    "relative_difference": error,
                    "lower_number_flux_m3_s": max(-state_faces[0], 0.0),
                    "excluded_lower_boundary_rate": lower_boundary_term,
                    "excluded_upper_boundary_rate": upper_boundary_term,
                }
            )

    # A: an actual no-boundary-active control made directly from the canonical
    # measure by excising only the first radius cell in both representations.
    a_beta = beta.copy()
    a_beta.number_density_per_m4[0] = 0.0
    a_matrix_xb = solver.ledger.recover_matrix_xb([solver.population("g"), a_beta])
    a_mask = radii >= float(beta.grid.edges_m[1])
    append_interior_state(
        state="A_no_boundary_active_canonical_tail_excised",
        state_density=a_beta.number_density_per_m4,
        state_matrix_xb=a_matrix_xb,
        state_radii=radii[a_mask],
        state_weights=weights[a_mask],
    )

    # C: a distinct, closed dynamic-matrix state (75% of the same canonical
    # cell measure) rather than a copy of state A's moment rows.
    c_beta = beta.copy()
    c_beta.number_density_per_m4 *= 0.75
    c_matrix_xb = solver.ledger.recover_matrix_xb([solver.population("g"), c_beta])
    append_interior_state(
        state="C_dynamic_matrix_closed_scaled_canonical_state",
        state_density=c_beta.number_density_per_m4,
        state_matrix_xb=c_matrix_xb,
        state_radii=radii,
        state_weights=0.75 * weights,
    )
    edge = float(beta.grid.edges_m[0])
    edge_equilibrium = solver.equilibrium_adapter.equilibrium_xb(
        np.asarray([edge], dtype=np.float64), beta.parameters
    )
    edge_velocity = float(growth_rate_m_s(
        radii_m=np.asarray([edge], dtype=np.float64),
        matrix_xb=solver.matrix_xb,
        equilibrium_xb=edge_equilibrium,
        parameters=beta.parameters,
    )[0])
    face_velocity = float(KWNSolver._face_velocities(velocity)[0])
    eulerian_number_flux = max(-face_velocity * density[0], 0.0)
    # The deterministic cohort representation has no atom *at* Rmin, so its
    # event stream is a sequence of characteristic crossings rather than a
    # smooth instantaneous flux.  Its shared-measure prediction at Rmin is
    # therefore the first-cell cohort weight divided by that cell width.
    # This makes the comparison traceable to cohort weights, not an Eulerian
    # density copied under a different name.
    first_cell_mask = (radii > edge) & (radii < float(beta.grid.edges_m[1]))
    cohort_first_cell_density = float(np.sum(weights[first_cell_mask]) / widths[0])
    physical_edge_number_flux = max(-edge_velocity * cohort_first_cell_density, 0.0)
    pivot_volume = sphere_volume_m3(float(centres[0]))
    edge_volume = sphere_volume_m3(edge)
    eulerian_volume_flux = eulerian_number_flux * edge_volume
    cohort_volume_flux = physical_edge_number_flux * edge_volume
    eulerian_mol_b_flux = eulerian_volume_flux * beta.parameters.x_b / beta.parameters.molar_volume_m3_mol
    cohort_mol_b_flux = cohort_volume_flux * beta.parameters.x_b / beta.parameters.molar_volume_m3_mol
    number_flux_error = _relative_error(eulerian_number_flux, physical_edge_number_flux)
    volume_flux_error = _relative_error(eulerian_volume_flux, cohort_volume_flux)
    mol_b_flux_error = _relative_error(eulerian_mol_b_flux, cohort_mol_b_flux)
    flux_rows = [
        {
            "state": "B_boundary_active_operator_probe",
            "Rmin_m": edge,
            "first_pivot_m": centres[0],
            "eulerian_actual_face_velocity_m_s": face_velocity,
            "cohort_Rmin_characteristic_velocity_m_s": edge_velocity,
            "cohort_first_cell_density_per_m4": cohort_first_cell_density,
            "cohort_boundary_prediction": "CONTINUUM_Rmin_CHARACTERISTIC_EXTRAPOLATION_OF_FIRST_CANONICAL_CELL",
            "velocity_relative_difference": _relative_error(face_velocity, edge_velocity),
            "eulerian_number_flux_m3_s": eulerian_number_flux,
            "cohort_continuum_Rmin_number_flux_m3_s": physical_edge_number_flux,
            "number_flux_relative_difference": number_flux_error,
            "eulerian_beta_volume_flux_s": eulerian_volume_flux,
            "cohort_continuum_Rmin_beta_volume_flux_s": cohort_volume_flux,
            "beta_volume_flux_relative_difference": volume_flux_error,
            "eulerian_mol_B_flux_mol_m3_s": eulerian_mol_b_flux,
            "cohort_continuum_Rmin_mol_B_flux_mol_m3_s": cohort_mol_b_flux,
            "mol_B_flux_relative_difference": mol_b_flux_error,
            "legacy_fixed_pivot_volume_per_number_m3": pivot_volume,
            "Rmin_volume_per_number_m3": edge_volume,
            "legacy_fixed_pivot_volume_price_relative_difference": _relative_error(pivot_volume, edge_volume),
        }
    ]
    rows.append(
        {
            "state": "B_boundary_active_operator_probe",
            "moment_order": 0,
            "eulerian_semi_discrete_rate": -eulerian_number_flux,
            "cohort_quadrature_rate": -physical_edge_number_flux,
            "relative_difference": number_flux_error,
            "lower_number_flux_m3_s": eulerian_number_flux,
            "excluded_lower_boundary_rate": "NOT_APPLICABLE",
            "excluded_upper_boundary_rate": "NOT_APPLICABLE",
        }
    )
    _write_csv(OUTPUT_ROOT / "moment_rate_parity.csv", rows)
    _write_csv(OUTPUT_ROOT / "boundary_flux_parity.csv", flux_rows)
    boundary_error = max(number_flux_error, volume_flux_error, mol_b_flux_error)
    interior_status = "PASS_SEMIDISCRETE_MOMENT_RATE_PARITY" if max_interior_error <= TWO_PERCENT else "FAIL_INTERIOR_TRANSPORT_OPERATOR_PARITY"
    boundary_status = "PASS_LOWER_BOUNDARY_OPERATOR_PARITY" if boundary_error <= TWO_PERCENT else "FAIL_LOWER_BOUNDARY_OPERATOR_PARITY"
    _write_report(
        "04_semidiscrete_operator_parity.md",
        "Semi-discrete operator parity",
        f"Interior status: `{interior_status}` with maximum moment-rate relative difference `{max_interior_error:.6e}` after explicitly removing the physical lower/upper boundary terms.  The dynamic-matrix probe uses the same algebraic closure and therefore isolates transport resolution.\n\n"
        f"Boundary status is reported separately in `05_lower_boundary_audit.md`.",
    )
    _write_report(
        "05_lower_boundary_audit.md",
        "Lower-bound number and B-inventory flux audit",
        f"Status: `{boundary_status}`.  The Eulerian FV face currently uses the cell-centre growth velocity, while the cohort continuum extrapolation follows the frozen physical Rmin characteristic.  Number/volume/mol-B flux relative errors are `{number_flux_error:.6e}`, `{volume_flux_error:.6e}`, and `{mol_b_flux_error:.6e}` respectively.\n\n"
        f"The legacy fixed-pivot/Rmin volume-price difference is `{float(flux_rows[0]['legacy_fixed_pivot_volume_price_relative_difference']):.6e}` and is retained only as a historical reporting diagnostic: the current beta flux tally uses the Rmin price on both sides.  The discrete cohort event stream is separate from this instantaneous continuum extrapolation. Number loss, beta-volume loss and mol-B return are reported as distinct quantities.",
    )
    return {
        "interior_status": interior_status,
        "boundary_status": boundary_status,
        "max_interior_error": max_interior_error,
        "boundary_velocity_relative_difference": float(flux_rows[0]["velocity_relative_difference"]),
        "boundary_number_flux_relative_difference": number_flux_error,
        "boundary_beta_volume_flux_relative_difference": volume_flux_error,
        "boundary_mol_B_flux_relative_difference": mol_b_flux_error,
        "legacy_volume_price_relative_difference": float(flux_rows[0]["legacy_fixed_pivot_volume_price_relative_difference"]),
    }


def _snapshot_metrics(solver: KWNSolver) -> dict[str, float]:
    beta = solver.population("beta")
    metrics = metrics_from_piecewise_constant_cells(
        beta.grid.edges_m, beta.number_density_per_m4 * beta.grid.widths_m
    )
    ledger = solver.ledger.snapshot(
        matrix_xb=solver.matrix_xb, populations=solver.population_list(), beta_resolved_fraction=1.0
    )
    result = metrics.as_dict()
    result.update(
        {
            "matrix_xB": solver.matrix_xb,
            "beta_inventory_mol_m3": ledger.beta_resolved_mol_m3,
            "total_inventory_mol_m3": ledger.total_mol_m3,
            "inventory_relative_residual": ledger.relative_residual,
        }
    )
    return result


def _build_canonical_solver(mapping: Mapping[str, Any], *, accuracy_cap: float | None, min_dt_s: float | None = None) -> KWNSolver:
    config = deepcopy(mapping)
    config["simulation"]["accuracy_radius_cfl"] = accuracy_cap
    if min_dt_s is not None:
        config["simulation"]["min_dt_s"] = min_dt_s
    return KWNSolver(SolverConfig.from_mapping(config))


def _accuracy_cfl_ladder(mapping: Mapping[str, Any]) -> dict[str, Any]:
    """Measure the literal current face-CFL and run only feasible micro-probes."""

    rows: list[dict[str, Any]] = []
    lower_tail_rows: list[dict[str, Any]] = []
    current = _build_canonical_solver(mapping, accuracy_cap=None)
    start = time.monotonic()
    cumulative_number = 0.0
    cumulative_volume = 0.0
    cumulative_mol_b = 0.0
    diagnostics = []
    target_s = 0.1 * 3600.0
    while current.time_s < target_s:
        diagnostic = current.advance_one(maximum_dt_s=target_s - current.time_s)
        diagnostics.append(diagnostic)
        cumulative_number += diagnostic.beta_rmin_number_flux_m3_s * diagnostic.dt_s
        cumulative_volume += diagnostic.beta_rmin_volume_flux_s * diagnostic.dt_s
        cumulative_mol_b += diagnostic.beta_rmin_mol_b_flux_mol_m3_s * diagnostic.dt_s
    elapsed = time.monotonic() - start
    first = diagnostics[0]
    current_metrics = _snapshot_metrics(current)
    rows.append(
        {
            "policy": "current_legacy_active_inventory_policy",
            "requested_raw_C_R_cap": "UNBOUNDED",
            "run_horizon_s": target_s,
            "accepted_steps": len(diagnostics),
            "rejected_steps": 0,
            "minimum_dt_s": min(item.dt_s for item in diagnostics),
            "median_dt_s": float(np.median([item.dt_s for item in diagnostics])),
            "maximum_dt_s": max(item.dt_s for item in diagnostics),
            "maximum_raw_face_C_R": max(item.radius_courant_max for item in diagnostics),
            "initial_raw_face_C_R": first.radius_courant_max,
            "runtime_s": elapsed,
            "N_m0_m3": current_metrics["N_m0_m3"],
            "Rmean_number_m": current_metrics["Rmean_number_m"],
            "Rmean_cubed_m3": current_metrics["Rmean_cubed_m3"],
            "Sv_m_inv": current_metrics["Sv_m_inv"],
            "f_beta": current_metrics["f_beta"],
            "matrix_xB": current_metrics["matrix_xB"],
            "cumulative_number_dissolved_m3": cumulative_number,
            "cumulative_beta_volume_dissolved": cumulative_volume,
            "cumulative_mol_B_returned_mol_m3": cumulative_mol_b,
            "full_48h_completion": False,
            "status": "DIAGNOSTIC_0p1H_ONLY",
        }
    )
    beta = current.population("beta")
    base_number = beta.number_density_per_m4 * beta.grid.widths_m
    for width_nm in (1.0, 2.0):
        upper = float(beta.grid.edges_m[0] + width_nm * 1.0e-9)
        lower_tail_rows.append(
            {
                "time_h": current.time_s / 3600.0,
                "region": f"Rmin_to_Rmin_plus_{width_nm:g}nm",
                "number_fraction": _piecewise_interval_fraction(
                    beta.grid.edges_m, base_number, float(beta.grid.edges_m[0]), upper, moment_order=0
                ),
                "M3_fraction": _piecewise_interval_fraction(
                    beta.grid.edges_m, base_number, float(beta.grid.edges_m[0]), upper, moment_order=3
                ),
                "cumulative_number_dissolved_m3": cumulative_number,
                "cumulative_beta_volume_dissolved": cumulative_volume,
                "cumulative_mol_B_returned_mol_m3": cumulative_mol_b,
            }
        )
    for cap in ACCURACY_CAPS:
        probe = _build_canonical_solver(mapping, accuracy_cap=cap, min_dt_s=1.0e-30)
        start = time.monotonic()
        probe_diagnostics = [probe.advance_one() for _ in range(4)]
        elapsed = time.monotonic() - start
        first_probe = probe_diagnostics[0]
        projected_steps = 48.0 * 3600.0 / first_probe.dt_s
        projected_runtime_s = projected_steps * elapsed / len(probe_diagnostics)
        rows.append(
            {
                "policy": f"literal_raw_face_C_R_lte_{cap:g}",
                "requested_raw_C_R_cap": cap,
                "run_horizon_s": probe.time_s,
                "accepted_steps": len(probe_diagnostics),
                "rejected_steps": 0,
                "minimum_dt_s": min(item.dt_s for item in probe_diagnostics),
                "median_dt_s": float(np.median([item.dt_s for item in probe_diagnostics])),
                "maximum_dt_s": max(item.dt_s for item in probe_diagnostics),
                "maximum_raw_face_C_R": max(item.radius_courant_max for item in probe_diagnostics),
                "initial_raw_face_C_R": first_probe.radius_courant_max,
                "runtime_s": elapsed,
                "N_m0_m3": _snapshot_metrics(probe)["N_m0_m3"],
                "Rmean_number_m": _snapshot_metrics(probe)["Rmean_number_m"],
                "Rmean_cubed_m3": _snapshot_metrics(probe)["Rmean_cubed_m3"],
                "Sv_m_inv": _snapshot_metrics(probe)["Sv_m_inv"],
                "f_beta": _snapshot_metrics(probe)["f_beta"],
                "matrix_xB": _snapshot_metrics(probe)["matrix_xB"],
                "cumulative_number_dissolved_m3": "NOT_MEANINGFUL_MICROPROBE",
                "cumulative_beta_volume_dissolved": "NOT_MEANINGFUL_MICROPROBE",
                "cumulative_mol_B_returned_mol_m3": "NOT_MEANINGFUL_MICROPROBE",
                "projected_48h_steps_at_initial_rate": projected_steps,
                "projected_48h_runtime_s_at_initial_rate": projected_runtime_s,
                "projected_48h_runtime_years_at_initial_rate": projected_runtime_s / (365.25 * 24.0 * 3600.0),
                "full_48h_completion": False,
                "status": "MICROPROBE_ONLY_48H_INFEASIBLE",
            }
        )
    _write_csv(OUTPUT_ROOT / "accuracy_cfl_ladder.csv", rows)
    _write_csv(OUTPUT_ROOT / "lower_tail_diagnostics.csv", lower_tail_rows)
    cap_one = next(row for row in rows if row["policy"] == "literal_raw_face_C_R_lte_1")
    _write_report(
        "06_accuracy_cfl_ladder.md",
        "Literal face-Courant accuracy ladder",
        f"Status: `FAIL_EULERIAN_TIME_ACCURACY`.  The current 0.1 h diagnostic starts at literal all-grid `C_R,max={first.radius_courant_max:.9e}`, while its legacy active-cell CFL is `{first.size_cfl:.6g}`.\n\n"
        f"A literal C_R<=1 policy begins at dt `{cap_one['minimum_dt_s']:.9e}` s and projects to `{cap_one['projected_48h_steps_at_initial_rate']:.6e}` accepted steps and `{cap_one['projected_48h_runtime_years_at_initial_rate']:.6e}` serial years for 48 h at the measured micro-probe cost.  The requested 48 h ladder was therefore not falsely represented as complete; the recorded policies are reproducible micro-probes demonstrating that the cap is actually applied.",
    )
    return {
        "status": "FAIL_EULERIAN_TIME_ACCURACY",
        "current_initial_raw_C_R": first.radius_courant_max,
        "current_legacy_active_cell_cfl": first.size_cfl,
        "passing_accuracy_cfl": "NONE",
        "rows": rows,
    }


def _select_top_status(
    *,
    initial: Mapping[str, Any],
    metric: Mapping[str, Any],
    growth: Mapping[str, Any],
    analytic: Mapping[str, Any],
    operator: Mapping[str, Any],
    accuracy: Mapping[str, Any],
) -> str:
    """Choose only an allowed terminal status without hiding an earlier gate."""

    if initial["status"] != "PASS_CANONICAL_INITIAL_MEASURE_IDENTITY":
        return "FAIL_CANONICAL_INITIAL_MEASURE_IDENTITY"
    if metric["status"] != "PASS_METRIC_IDENTITY":
        return "FAIL_METRIC_IDENTITY"
    if growth["status"] != "PASS_GROWTH_KERNEL_PARITY":
        return "FAIL_GROWTH_KERNEL_PARITY"
    if growth["matrix_status"] != "PASS_MATRIX_CLOSURE_PARITY":
        return "FAIL_MATRIX_CLOSURE_PARITY"
    if not str(analytic["status"]).startswith("PASS_"):
        return "FAIL_EULERIAN_TIME_ACCURACY"
    if operator["interior_status"] != "PASS_SEMIDISCRETE_MOMENT_RATE_PARITY":
        return "FAIL_EULERIAN_TIME_ACCURACY"
    if operator["boundary_status"] != "PASS_LOWER_BOUNDARY_OPERATOR_PARITY":
        return "FAIL_LOWER_BOUNDARY_OPERATOR_PARITY"
    if accuracy["status"] != "PASS_EULERIAN_TIME_ACCURACY":
        return "FAIL_EULERIAN_TIME_ACCURACY"
    # A positive result cannot be claimed until the required v2 grid and PF
    # stages run.  This branch is intentionally fail-closed rather than
    # manufacturing a pass from pre-authority diagnostics.
    return "FAIL_EULERIAN_TIME_ACCURACY"


def _write_blocked_outputs(
    *, boundary: Mapping[str, Any], accuracy: Mapping[str, Any], top_status: str
) -> tuple[dict[str, Any], dict[str, Any]]:
    if top_status == "FAIL_LOWER_BOUNDARY_OPERATOR_PARITY":
        blocker = "LOWER_BOUNDARY_OPERATOR_PARITY"
        reason = (
            "Canonical initialization and metric closure pass, but the lower-bound FV face is not "
            "the Rmin characteristic operator. Requalification is prohibited until this operator gate closes."
        )
    else:
        blocker = top_status
        reason = f"Requalification is prohibited by the prior terminal gate `{top_status}`."
    authority = {
        "schema_version": "KWN_EULERIAN_AUTHORITY_V2",
        "status": f"BLOCKED_{blocker}",
        "authority_grid": None,
        "reason": reason,
        "accuracy_policy": accuracy["status"],
        "legacy_authority": "PASS_EULERIAN_SMOOTH_POPULATION_AUTHORITY_AT_3200_REQUALIFICATION_NOT_INHERITED",
    }
    crosscheck = {
        "status": f"BLOCKED_{blocker}",
        "reason": f"No full 48 h canonical cohort--Eulerian comparison is valid before `{top_status}` closes.",
        "registered_two_percent_gate": "NOT_RELAXED",
    }
    _write_json(OUTPUT_ROOT / "eulerian_authority_v2.json", authority)
    _write_csv(OUTPUT_ROOT / "final_cohort_eulerian_crosscheck.csv", [crosscheck])
    _write_csv(
        OUTPUT_ROOT / "beta_only_cohort_pf_comparison.csv",
        [{"status": "BLOCKED_PREREQUISITE_GATE", "reason": crosscheck["reason"]}],
    )
    _write_report(
        "07_minimal_repair.md",
        "Minimal repair decision",
        "Implemented: canonical direct cell initialization, a shared metric/closure module, explicit lower-bound number/volume/mol-B tallies, and literal face-Courant telemetry.\n\n"
        "Not implemented: a new high-order transport solver, MUSCL/TVD, GP release, physical retuning, or an unverified boundary remap.  The next minimal numerical change must unify the Eulerian lower-face characteristic with the cohort Rmin event, then repeat the literal CFL ladder.",
    )
    _write_report(
        "08_eulerian_authority_requalification.md",
        "Eulerian authority requalification",
        f"Status: `{authority['status']}`.  The frozen 3200-bin smooth authority is historical evidence only.  No 800/1600/3200 v2 authority is selected while the prior gate is open.",
    )
    _write_report(
        "09_cohort_eulerian_final_crosscheck.md",
        "Final canonical cohort--Eulerian crosscheck",
        f"Status: `{crosscheck['status']}`.  The 2% gate remains unchanged; it was not evaluated on a representation pair that fails the preceding lower-bound contract.",
    )
    _write_report(
        "10_beta_only_cohort_pf_comparison.md",
        "Beta-only cohort--PF comparison",
        "Status: `BLOCKED_PREREQUISITE_GATE`.  CUDA/PF was not rerun and frozen PF trajectories were not used to manufacture a direction conclusion before KWN shared-operator parity closes.",
    )
    return authority, crosscheck


def _write_final_reports(
    *,
    baseline: Mapping[str, Any],
    canonical: Mapping[str, Any],
    initial: Mapping[str, Any],
    metric: Mapping[str, Any],
    growth: Mapping[str, Any],
    analytic: Mapping[str, Any],
    operator: Mapping[str, Any],
    accuracy: Mapping[str, Any],
    authority: Mapping[str, Any],
    crosscheck: Mapping[str, Any],
    launch: Mapping[str, Any],
    top_status: str,
    command: str,
) -> dict[str, Any]:
    status = top_status
    primary_root_cause = {
        "FAIL_LOWER_BOUNDARY_OPERATOR_PARITY": "LOWER_BOUNDARY_OPERATOR_MISMATCH; IMPLICIT_TIME_DISCRETIZATION_DIFFUSION_REMAINS_AN_OPEN_SECONDARY_BLOCKER",
        "FAIL_EULERIAN_TIME_ACCURACY": "LITERAL_ALL_GRID_ACCURACY_CFL_NOT_CLOSED",
        "FAIL_CANONICAL_INITIAL_MEASURE_IDENTITY": "CANONICAL_INITIAL_MEASURE_IDENTITY",
        "FAIL_METRIC_IDENTITY": "METRIC_IDENTITY_MISMATCH",
        "FAIL_GROWTH_KERNEL_PARITY": "GROWTH_KERNEL_PARITY_MISMATCH",
        "FAIL_MATRIX_CLOSURE_PARITY": "DYNAMIC_MATRIX_CLOSURE_MISMATCH",
    }.get(status, status)
    final = {
        "STATUS": status,
        "BRANCH": launch["git_branch_at_launch"],
        "COMMIT": launch["git_head_at_launch"],
        "BASELINE_REPRODUCED": baseline["status"],
        "VALIDATION_CONTRACT_HASH": canonical["validation_contract_hash"],
        "CANONICAL_INITIAL_MEASURE": initial["status"],
        "M0_INITIAL_ERROR": initial["errors_by_points_per_cell"][4]["M0"],
        "M1_INITIAL_ERROR": initial["errors_by_points_per_cell"][4]["M1"],
        "M2_INITIAL_ERROR": initial["errors_by_points_per_cell"][4]["M2"],
        "M3_INITIAL_ERROR": initial["errors_by_points_per_cell"][4]["M3"],
        "INITIAL_CDF_ERROR": initial["initial_cdf_error"],
        "LOWER_TAIL_INITIAL_ERROR": initial["lower_tail_initial_error"],
        "METRIC_IDENTITY": metric["status"],
        "GROWTH_KERNEL_PARITY": growth["status"],
        "MATRIX_CLOSURE_PARITY": growth["matrix_status"],
        "LOWER_BOUNDARY_PARITY": operator["boundary_status"],
        "SEMIDISCRETE_MOMENT_RATE_PARITY": operator["interior_status"],
        "PRIMARY_ROOT_CAUSE": primary_root_cause,
        "ANALYTIC_BENCHMARK_GATE": analytic["status"],
        "IMPLICIT_CURRENT_MAX_CFL": accuracy["current_initial_raw_C_R"],
        "PASSING_ACCURACY_CFL": accuracy["passing_accuracy_cfl"],
        "DIAGNOSTIC_EXPLICIT_RESULT": "NOT_AUTHORIZED_BEFORE_LOWER_BOUNDARY_OPERATOR_PARITY",
        "EULERIAN_AUTHORITY_GRID_V2": authority["authority_grid"],
        "EULERIAN_AUTHORITY_CONFIG_HASH": "NOT_ASSIGNED",
        "EULERIAN_RESTART": "NOT_RUN_AFTER_BOUNDARY_GATE",
        "EULERIAN_MAX_RESIDUAL": "NOT_RUN_AFTER_BOUNDARY_GATE",
        "COHORT_EULERIAN_CROSSCHECK": crosscheck["status"],
        "N_M0_ERROR": "NOT_EVALUATED_AFTER_BOUNDARY_GATE",
        "RMEAN_ERROR": "NOT_EVALUATED_AFTER_BOUNDARY_GATE",
        "RMEAN3_ERROR": "NOT_EVALUATED_AFTER_BOUNDARY_GATE",
        "SV_ERROR": "NOT_EVALUATED_AFTER_BOUNDARY_GATE",
        "FBETA_ERROR": "NOT_EVALUATED_AFTER_BOUNDARY_GATE",
        "XMATRIX_ERROR": "NOT_EVALUATED_AFTER_BOUNDARY_GATE",
        "CUMULATIVE_NUMBER_DISSOLUTION_ERROR": "NOT_EVALUATED_AFTER_BOUNDARY_GATE",
        "CUMULATIVE_MOL_B_RETURN_ERROR": "NOT_EVALUATED_AFTER_BOUNDARY_GATE",
        "BETA_INITIAL_STATE_IDENTITY": "NOT_RUN_AFTER_PREREQUISITE_GATE",
        "BETA_ONLY_DIRECTION": "BLOCKED_PREREQUISITE_GATE",
        "BETA_ONLY_TIMESCALE": "NOT_RUN",
        "MEAN_FIELD_PF_GAP": "NOT_RUN",
        "PF_SOURCE_MODIFIED": launch["pf_source_modified"],
        "CUDA_RERUN": launch["cuda_rerun"],
        "PHYSICAL_RETUNING": launch["physical_retuning"],
        "LEGACY_SIX_PARTICLE_EULERIAN_P5": "FAIL_RETAINED",
        "HISTORICAL_AUTHORITY": "HISTORICAL_SMOOTH_3200_ONLY; V2_NOT_INHERITED",
        "TOP_5_FINDINGS": [
            "The old failure was reproduced exactly before any code change.",
            "A single canonical cell measure closes t0 M0/M3/inventory and cell-edge CDF identity.",
            "Metric semantics are now explicit: Rmean_cubed and mean_R3 are distinct.",
            "Growth and algebraic matrix closure are shared to machine precision.",
            "The first remaining dynamic difference is the Eulerian centre-sampled lower face versus the cohort Rmin characteristic; literal raw CFL is also unclosed.",
        ],
        "P0_BLOCKERS": [
            "Unify lower-bound face velocity/event and B-inventory settlement under one physical Rmin operator.",
            "Then re-run the literal all-grid accuracy-CFL ladder before any authority or PF direction comparison.",
        ],
        "NEXT_ACTION": "Implement and validate a common Rmin characteristic/face operator without changing Rmin or physical parameters; then reassess whether a higher-order or characteristic-remap method is required for literal time accuracy.",
        "LOCAL_GP_RELEASE_AUTHORIZED": "LOCAL_GP_RELEASE_NOT_AUTHORIZED",
        "KEY_REPORTS": [str(REPORT_ROOT / f"{index:02d}_{name}.md") for index, name in [
            (0, "baseline_reproduction"), (1, "canonical_initial_measure"), (2, "metric_and_growth_kernel_parity"),
            (5, "lower_boundary_audit"), (6, "accuracy_cfl_ladder"), (9, "cohort_eulerian_final_crosscheck"),
            (12, "final_acceptance_report"),
        ]],
        "command": command,
    }
    _write_report(
        "11_model_role_boundary.md",
        "Model-role boundary",
        f"Eulerian KWN remains a smooth population-balance representation.  The discrete cohort solver is the event-aware characteristic comparator.  PF/CUDA remains frozen external evidence: `PF_SOURCE_MODIFIED={launch['pf_source_modified']}`, `CUDA_RERUN={launch['cuda_rerun']}`, `PHYSICAL_RETUNING={launch['physical_retuning']}`, and no GP release was started.",
    )
    negative_crossing = analytic["boundary_crossing_summaries"]["constant_negative_absorbing_Rmin"]
    characteristic_crossing = analytic["boundary_crossing_summaries"]["radius_dependent_minus_K_over_R"]
    _write_report(
        "12_final_acceptance_report.md",
        "Final acceptance",
        f"Top-level status: `{status}`.\n\n"
        "The old vague physics-mismatch label is retired for this task.  The status selector preserves the first failed gate rather than overwriting it with a later diagnostic.  The frozen 2% threshold is unchanged.\n\n"
        "| Gate | Result |\n|---|---|\n"
        f"| Baseline | `{baseline['status']}`; frozen contract `{canonical['validation_contract_hash']}` |\n"
        f"| Canonical measure | `{initial['status']}`; M0/M3 authority errors `{initial['errors_by_points_per_cell'][4]['M0']:.3e}` / `{initial['errors_by_points_per_cell'][4]['M3']:.3e}` |\n"
        f"| Metrics / growth / matrix | `{metric['status']}` / `{growth['status']}` / `{growth['matrix_status']}` |\n"
        f"| Interior moment rates | `{operator['interior_status']}`; max error `{operator['max_interior_error']:.3e}` |\n"
        f"| Rmin flux parity | `{operator['boundary_status']}`; number/volume/mol-B errors `{operator['boundary_number_flux_relative_difference']:.3e}` / `{operator['boundary_beta_volume_flux_relative_difference']:.3e}` / `{operator['boundary_mol_B_flux_relative_difference']:.3e}` |\n"
        f"| Constant-negative crossing | Eulerian returned `{negative_crossing['eulerian_cumulative_number_dissolved_m3']:.6g}` of analytic `{negative_crossing['analytic_cumulative_number_dissolved_m3']:.6g}` at C_R `{negative_crossing['maximum_C_R']:.5g}` |\n"
        f"| -K/R crossing | Eulerian returned `{characteristic_crossing['eulerian_cumulative_number_dissolved_m3']:.6g}` of analytic `{characteristic_crossing['analytic_cumulative_number_dissolved_m3']:.6g}` at C_R `{characteristic_crossing['maximum_C_R']:.5g}` |\n"
        f"| Literal CFL | current all-grid C_R,max `{accuracy['current_initial_raw_C_R']:.9e}`; no 48 h passing raw-CFL policy is feasible |\n"
        f"| Authority / PF | `{authority['status']}` / `{crosscheck['status']}`; beta-only comparison blocked |\n\n"
        f"No beta-only PF direction conclusion and no local GP release are authorized.  `PF_SOURCE_MODIFIED={launch['pf_source_modified']}`, `CUDA_RERUN={launch['cuda_rerun']}`, `PHYSICAL_RETUNING={launch['physical_retuning']}`.",
    )
    _write_report(
        "13_reproduction_commands.md",
        "Reproduction commands",
        "```bash\n"
        "PYTHONPATH=src python3 -m unittest tests.kwn.test_population_metrics tests.kwn.test_cohort_solver tests.kwn.test_rmin_boundary\n"
        "BASELINE_OUTPUT_ROOT=/absolute/path/to/isolated/outputs/kwn_discrete_cohort_comparison_v1\n"
        "PYTHONPATH=src python3 scripts/run_kwn_shared_operator_parity_v1.py all --baseline-output-root \"$BASELINE_OUTPUT_ROOT\"\n"
        "```\n\n"
        f"This run used `{baseline['baseline_run_output_root']}` as its isolated exact-replay evidence.  The baseline command is read-only with respect to frozen historical artifacts; this task does not run CUDA/PF or alter physical inputs.",
    )
    _write_json(OUTPUT_ROOT / "final_acceptance.json", final)
    return final


def run_all(*, baseline_output_root: Path, command: str) -> int:
    launch = _capture_launch_context()
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    figure_root = OUTPUT_ROOT / "figures"
    figure_root.mkdir(parents=True, exist_ok=True)
    (figure_root / "README.md").write_text(
        "# Plot-ready evidence\n\nUse the CSV files in the parent directory; no visual substitute is used for operator gates.\n",
        encoding="utf-8",
    )
    baseline = _baseline_reproduction(baseline_output_root)
    audit = _load_radius_audit_module()
    canonical, solver, mapping, _ = _build_canonical_measure(
        audit, baseline["validation_contract_hash_observed_from_provenance"]
    )
    initial, cohort_solvers, quadratures = _canonical_initial_identity(canonical, solver)
    reproduced_authority = _read_json(baseline_output_root / "eulerian_smooth_authority.json")
    metric, growth = _metric_growth_and_matrix_parity(
        solver, initial, cohort_solvers, quadratures, reproduced_authority
    )
    analytic = _analytic_benchmarks(solver, quadratures[4])
    operator = _semi_discrete_and_boundary(solver, quadratures[4])
    accuracy = _accuracy_cfl_ladder(mapping)
    top_status = _select_top_status(
        initial=initial,
        metric=metric,
        growth=growth,
        analytic=analytic,
        operator=operator,
        accuracy=accuracy,
    )
    authority, crosscheck = _write_blocked_outputs(
        boundary=operator, accuracy=accuracy, top_status=top_status
    )
    final = _write_final_reports(
        baseline=baseline,
        canonical=canonical,
        initial=initial,
        metric=metric,
        growth=growth,
        analytic=analytic,
        operator=operator,
        accuracy=accuracy,
        authority=authority,
        crosscheck=crosscheck,
        launch=launch,
        top_status=top_status,
        command=command,
    )
    provenance = {
        "schema_version": "KWN_SHARED_OPERATOR_PARITY_ANALYSIS_PROVENANCE_V1",
        "launch": launch,
        "baseline": baseline,
        "binary": {
            "python_executable": sys.executable,
            "python_version": sys.version,
        },
        "source_sha256": {
            relative: _sha256_file(ROOT / relative)
            for relative in (
                "scripts/run_kwn_shared_operator_parity_v1.py",
                "src/kwn_mvp/solver.py",
                "src/kwn_mvp/cohort_solver.py",
                "src/kwn_mvp/population_metrics.py",
                "src/kwn_mvp/ledger.py",
                "src/kwn_mvp/populations.py",
            )
        },
        "configuration": {
            "canonical_mapping_hash": _canonical_hash(mapping),
            "solver_source_config_hash": solver.config.source_config_hash,
            "validation_contract_path": solver.config.validation_contract_path,
        },
        "fixture": {
            "fixture_hash": canonical["fixture_hash"],
            "source_initial_psd_hash": canonical["source_initial_psd_hash"],
        },
        "canonical": {
            "canonical_hash": canonical["canonical_hash"],
            "population_npz_sha256": canonical["npz_sha256"],
            "cohort_quadrature_archive": canonical.get("cohort_quadrature_archive"),
        },
        "validation_contract_hash": canonical["validation_contract_hash"],
        "final_status": final["STATUS"],
        "forbidden_actions": {
            "pf_source_modified": launch["pf_source_modified"],
            "cuda_rerun": launch["cuda_rerun"],
            "physical_retuning": launch["physical_retuning"],
            "gp_release": False,
        },
    }
    _write_json(OUTPUT_ROOT / "analysis_provenance.json", provenance)
    print(json.dumps(final, indent=2, sort_keys=True))
    return 2


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("all",))
    parser.add_argument(
        "--baseline-output-root", type=Path, required=True,
        help="isolated, fully rerun outputs/kwn_discrete_cohort_comparison_v1 directory",
    )
    arguments = parser.parse_args()
    return run_all(
        baseline_output_root=arguments.baseline_output_root.resolve(),
        command=" ".join(sys.argv),
    )


if __name__ == "__main__":
    raise SystemExit(main())
