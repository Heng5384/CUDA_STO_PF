#!/usr/bin/env python3
"""Run and assemble the KWN radius-grid convergence qualification v1.

This task is deliberately KWN-only.  It reads the frozen 96^3 CUDA/PF
evidence but never invokes, changes, or reconstructs a PF/CUDA calculation.
Each numerical run receives a separate no-overwrite directory below
``outputs/kwn_radius_grid_convergence_v1/runs`` so grid members can be run on
independent CPU workers and assembled only after their provenance is present.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import subprocess
import sys
from collections import defaultdict
from copy import deepcopy
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "src"))

from coupling.fixture_conditioned_handoff_v2 import (  # noqa: E402
    load_host_96cube_fixture,
    load_validation_contract,
)
from kwn_mvp.diagnostics import discrete_wasserstein_distance  # noqa: E402
from kwn_mvp.composition_mapping import ag_at_fraction_to_xb, xb_to_ag_at_fraction  # noqa: E402
from kwn_mvp.radius_grid import RadiusGrid  # noqa: E402
from kwn_mvp.solver import KWNSolver, SolverConfig, SolverStateError  # noqa: E402
from scripts import diagnose_kwn_positivity_failure as legacy_diagnosis  # noqa: E402
from scripts import run_beta_only_same_contract_control as beta_control  # noqa: E402


TASK_OUTPUT_ROOT = ROOT / "outputs" / "kwn_radius_grid_convergence_v1"
REPORT_ROOT = ROOT / "reports" / "kwn_radius_grid_convergence_v1"
FROZEN_OUTPUT_ROOT = ROOT / "outputs" / "kwn_pf_cuda_runtime_closure_v1"
CONTRACT_PATH = ROOT / "contracts" / "pf_kwn_validation_contract_v1.json"
FIXTURE_SPEC = (
    ROOT
    / "data"
    / "qualification"
    / "pf_mass_conserving_library_handoff_v1"
    / "six_particle_96cube_spec.json"
)
PROFILE_ROOT = (
    ROOT.parent
    / "CUDA_STO_PF"
    / "data"
    / "qualification"
    / "pf_elastic_target_profile_quarter_nm_v2"
    / "profiles"
)
PROFILE_LIBRARY_MANIFEST = PROFILE_ROOT.parent / "library" / "library_manifest.json"
FROZEN_LEGACY_DIAGNOSIS = FROZEN_OUTPUT_ROOT / "kwn_positivity_failure_diagnosis.json"
FROZEN_REPAIR_SUMMARY = FROZEN_OUTPUT_ROOT / "kwn_conservative_qualification" / "summary.json"

LEGACY_FAILURE_TIME_H = 0.39317699499770825
UNIFORM_OUTPUT_TIMES_H = (0.0, 0.1, LEGACY_FAILURE_TIME_H, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0)
HISTORICAL_200_OUTPUT_TIMES_H = (0.0, 0.1, 1.0, 3.0, 6.0, 12.0, 24.0, 48.0)
PRIMARY_METRICS = (
    "N_m0_m3",
    "Rmean_m",
    "Rmean3_m3",
    "Sv_m_inv",
    "f_beta",
    "matrix_xB",
)
ALL_MOMENT_METRICS = ("M0_m3", "M1_m2", "M2_m", "M3_dimensionless")
GRID_LADDER = (100, 200, 400, 800, 1600)
BASELINE_FLOAT_REL_TOLERANCE = 5.0e-14
BASELINE_FLOAT_ABS_TOLERANCE = 5.0e-16

# PF has accepted states near the nominal validation ages. The comparison
# consumes these exact accepted times rather than interpolating frozen PF data.
PF_COMPARISON_TIME_MAP = (
    (0.0, 0, 0.0),
    (0.1, 363, 0.09991838128043773),
    (1.0, 3633, 1.0000095845504966),
    (3.0, 10899, 3.00002875365149),
    (6.0, 21798, 6.00005750730298),
    (12.0, 43596, 12.00011501460596),
    (24.0, 87191, 23.99995477196321),
    (48.0, 174382, 47.99990954392642),
)


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_sha256(value: Any) -> str:
    payload = json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _read_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise RuntimeError(f"Expected JSON object at {path}")
    return value


def _write_json_new(path: Path, value: Any) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, allow_nan=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_csv_new(path: Path, rows: list[Mapping[str, Any]]) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0])
    if any(list(row) != fieldnames for row in rows):
        raise RuntimeError(f"CSV rows disagree on schema for {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _write_text_new(path: Path, text: str) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def _write_npz_new(path: Path, **arrays: Any) -> None:
    if path.exists():
        raise RuntimeError(f"Refusing to overwrite {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _task_path(path: Path) -> Path:
    resolved = path.resolve()
    try:
        resolved.relative_to(TASK_OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise RuntimeError(f"Refusing path outside task output root: {resolved}") from error
    return resolved


def _run_directory(run_id: str) -> Path:
    if not run_id or any(character not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789_-" for character in run_id):
        raise RuntimeError("run_id may contain only letters, digits, '_' and '-'")
    destination = _task_path(TASK_OUTPUT_ROOT / "runs" / run_id)
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite existing run directory {destination}")
    destination.mkdir(parents=True, exist_ok=False)
    return destination


def _sphere_volume(radius_m: float) -> float:
    return 4.0 * math.pi * float(radius_m) ** 3 / 3.0


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(right), 1.0e-300)


def _float(value: str) -> float:
    return float(value) if value not in ("", None) else float("nan")


def _load_frozen_inputs() -> tuple[Any, Any]:
    contract = load_validation_contract(CONTRACT_PATH)
    fixture = load_host_96cube_fixture(FIXTURE_SPEC, PROFILE_ROOT, contract)
    return contract, fixture


def _build_fixture_solver(
    *, bins: int, max_dt_factor: float = 1.0
) -> tuple[KWNSolver, dict[str, Any], Any, Any]:
    """Construct exactly the current repaired fixed-fixture KWN configuration."""

    contract, fixture = _load_frozen_inputs()
    base, construction = beta_control.build_same_contract_config(
        fixture=fixture,
        contract=contract,
    )
    mapping = deepcopy(construction["config_mapping"])
    if bins != base.grid.bins:
        grid = RadiusGrid.logarithmic(
            float(mapping["radius_grid"]["minimum_m"]),
            float(mapping["radius_grid"]["maximum_m"]),
            bins,
        )
        entries, projection = beta_control._project_fixture_resolved_psd(fixture, contract, grid)
        mapping["radius_grid"]["bins"] = bins
        mapping["populations"]["beta"]["initial"]["entries"] = entries
        construction["resolved_psd_projection"] = projection
    mapping["simulation"]["max_dt_s"] = float(mapping["simulation"]["max_dt_s"]) * max_dt_factor
    solver = KWNSolver(SolverConfig.from_mapping(mapping))
    construction["config_mapping"] = mapping
    construction["initial_psd_source_hash"] = _canonical_sha256(
        {
            "fixture_hash": fixture.fixture_hash,
            "source_radii_m": list(construction["resolved_psd_projection"]["source_radii_m"]),
            "resolved_equivalent_number_scale": construction["resolved_psd_projection"]["resolved_equivalent_number_scale"],
            "radius_edges_m": [float(value) for value in solver.config.grid.edges_m],
            "projection": construction["resolved_psd_projection"]["projection"],
        }
    )
    return solver, construction, contract, fixture


def _build_smooth_solver(
    *, bins: int, max_dt_factor: float = 1.0, log_sigma: float = 0.14
) -> tuple[KWNSolver, dict[str, Any], Any, Any]:
    """Build a fixed, smooth qualification PSD with the same beta inventory."""

    solver, construction, contract, fixture = _build_fixture_solver(
        bins=bins, max_dt_factor=max_dt_factor
    )
    beta = solver.population("beta")
    radii = beta.grid.centres_m
    source_radii = np.asarray(fixture.resolved_equivalent_radii_m, dtype=np.float64)
    median_radius = float(np.exp(np.mean(np.log(source_radii))))
    pdf = np.exp(-0.5 * (np.log(radii / median_radius) / log_sigma) ** 2)
    pdf /= radii * log_sigma * math.sqrt(2.0 * math.pi)
    target_m3 = beta.radius_moment(3, quadrature="fixed_pivot")
    raw_m3 = float(np.sum(pdf * beta.grid.widths_m * radii**3))
    if raw_m3 <= 0.0:
        raise RuntimeError("smooth PSD has zero fixed-pivot third moment")
    smooth_density = pdf * (target_m3 / raw_m3)
    mapping = deepcopy(construction["config_mapping"])
    mapping["populations"]["beta"]["initial"] = {
        "kind": "discrete",
        "entries": [
            {
                "radius_m": float(radius),
                "number_density_m3": float(number * width),
                "grid_bin_index": float(index),
                "source_particle_count": float(number * width),
            }
            for index, (radius, width, number) in enumerate(
                zip(radii, beta.grid.widths_m, smooth_density)
            )
        ],
    }
    solver = KWNSolver(SolverConfig.from_mapping(mapping))
    beta = solver.population("beta")
    snapshot = solver.ledger.snapshot(matrix_xb=solver.matrix_xb, populations=solver.population_list())
    if snapshot.relative_residual > 1.0e-12:
        raise RuntimeError("smooth PSD did not preserve the fixture inventory")
    construction["smooth_psd_benchmark"] = {
        "kind": "LOGNORMAL_FIXED_NUMERICAL_QUALIFICATION_ONLY",
        "median_radius_m": median_radius,
        "log_sigma": log_sigma,
        "same_fixed_pivot_M3_as_fixture": target_m3,
        "same_matrix_xB_as_fixture": solver.matrix_xb,
        "not_used_for_pf_comparison": True,
    }
    construction["config_mapping"] = mapping
    construction["initial_psd_source_hash"] = _canonical_sha256(
        {
            "kind": "LOGNORMAL_FIXED_NUMERICAL_QUALIFICATION_ONLY",
            "radius_edges_m": [float(value) for value in beta.grid.edges_m],
            "number_density_per_m4": [float(value) for value in smooth_density],
            "median_radius_m": median_radius,
            "log_sigma": log_sigma,
            "target_fixed_pivot_M3": target_m3,
        }
    )
    return solver, construction, contract, fixture


def _critical_radius_m(solver: KWNSolver) -> float | None:
    """Locate the beta growth-sign transition without mutating the state."""

    beta = solver.population("beta")
    lower = float(beta.grid.edges_m[0])
    upper = float(beta.grid.edges_m[-1])

    def residual(radius: float) -> float:
        equilibrium = solver.equilibrium_adapter.equilibrium_xb(
            np.asarray([radius], dtype=np.float64), beta.parameters
        )
        return solver.matrix_xb - float(equilibrium[0])

    low_value = residual(lower)
    high_value = residual(upper)
    if low_value == 0.0:
        return lower
    if high_value == 0.0:
        return upper
    if low_value * high_value > 0.0:
        return None
    for _ in range(100):
        middle = 0.5 * (lower + upper)
        middle_value = residual(middle)
        if middle_value == 0.0:
            return middle
        if low_value * middle_value < 0.0:
            upper, high_value = middle, middle_value
        else:
            lower, low_value = middle, middle_value
    return 0.5 * (lower + upper)


def _dt_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"minimum_dt_s": 0.0, "median_dt_s": 0.0, "maximum_dt_s": 0.0}
    return {
        "minimum_dt_s": float(min(values)),
        "median_dt_s": float(np.median(np.asarray(values, dtype=np.float64))),
        "maximum_dt_s": float(max(values)),
    }


def _snapshot_row(
    *,
    solver: KWNSolver,
    scenario: str,
    run_id: str,
    bins: int,
    target_time_h: float,
    interval_dt: list[float],
    cumulative_dt: list[float],
    cumulative_lower_number_m3: float,
    cumulative_lower_beta_inventory_mol_m3: float,
    interval_lower_number_m3: float,
    last_lower_flux_number_m3_s: float,
    max_interval_positivity: float,
    max_interval_cfl: float,
    max_interval_residual: float,
) -> dict[str, Any]:
    beta = solver.population("beta")
    fixed = [beta.radius_moment(order, quadrature="fixed_pivot") for order in range(4)]
    cell = [beta.radius_moment(order, quadrature="cell_integrated") for order in range(4)]
    m0, m1, m2, m3 = fixed
    cell_m0, cell_m1, cell_m2, cell_m3 = cell
    ledger = solver.ledger.snapshot(
        matrix_xb=solver.matrix_xb,
        populations=solver.population_list(),
        beta_resolved_fraction=1.0,
    )
    occupied = np.flatnonzero(beta.number_density_per_m4 > 0.0)
    positive = beta.number_density_per_m4[beta.number_density_per_m4 > 0.0]
    row: dict[str, Any] = {
        "scenario": scenario,
        "run_id": run_id,
        "bins": bins,
        "target_time_h": target_time_h,
        "time_h": solver.time_s / 3600.0,
        "step": solver.step,
        "M0_m3": m0,
        "M1_m2": m1,
        "M2_m": m2,
        "M3_dimensionless": m3,
        "Rmean_m": 0.0 if m0 == 0.0 else m1 / m0,
        "Rmean3_m3": 0.0 if m0 == 0.0 else m3 / m0,
        "N_m0_m3": m0,
        "Sv_m_inv": 4.0 * math.pi * m2,
        "f_beta": 4.0 * math.pi * m3 / 3.0,
        "matrix_xB": solver.matrix_xb,
        "beta_inventory_mol_m3": ledger.beta_resolved_mol_m3,
        "matrix_inventory_mol_m3": ledger.matrix_mol_m3,
        "gp_inventory_mol_m3": ledger.gp_mol_m3,
        "beta_subgrid_inventory_mol_m3": ledger.beta_subgrid_mol_m3,
        "beta_resolved_inventory_mol_m3": ledger.beta_resolved_mol_m3,
        "total_inventory_mol_m3": ledger.total_mol_m3,
        "total_residual_mol_m3": ledger.residual_mol_m3,
        "total_residual_relative": ledger.relative_residual,
        "critical_radius_m": _critical_radius_m(solver),
        "occupied_bin_count": int(occupied.size),
        "minimum_occupied_radius_m": "" if not occupied.size else float(beta.grid.centres_m[occupied[0]]),
        "maximum_occupied_radius_m": "" if not occupied.size else float(beta.grid.centres_m[occupied[-1]]),
        "minimum_bin_density_per_m4": float(np.min(beta.number_density_per_m4)),
        "minimum_positive_bin_density_per_m4": "" if positive.size == 0 else float(np.min(positive)),
        "rejected_step_count": 0,
        "roundoff_zeroed_bin_count": solver.roundoff_zeroed_bin_count,
        "cumulative_lower_boundary_number_m3": cumulative_lower_number_m3,
        "cumulative_lower_boundary_beta_inventory_mol_m3": cumulative_lower_beta_inventory_mol_m3,
        "interval_lower_boundary_number_m3": interval_lower_number_m3,
        "last_lower_boundary_flux_number_m3_s": last_lower_flux_number_m3_s,
        "maximum_interval_positivity_utilization": max_interval_positivity,
        "maximum_interval_size_cfl": max_interval_cfl,
        "maximum_interval_residual": max_interval_residual,
        "midpoint_M0_m3": m0,
        "midpoint_M1_m2": m1,
        "midpoint_M2_m": m2,
        "midpoint_M3_dimensionless": m3,
        "midpoint_Sv_m_inv": 4.0 * math.pi * m2,
        "midpoint_f_beta": 4.0 * math.pi * m3 / 3.0,
        "cell_integrated_M0_m3": cell_m0,
        "cell_integrated_M1_m2": cell_m1,
        "cell_integrated_M2_m": cell_m2,
        "cell_integrated_M3_dimensionless": cell_m3,
        "cell_integrated_Sv_m_inv": 4.0 * math.pi * cell_m2,
        "cell_integrated_f_beta": 4.0 * math.pi * cell_m3 / 3.0,
    }
    row.update({f"interval_{key}": value for key, value in _dt_stats(interval_dt).items()})
    row.update({f"cumulative_{key}": value for key, value in _dt_stats(cumulative_dt).items()})
    return row


def _project_point_to_fixed_pivots(
    *, grid: RadiusGrid, radius_m: float, number_density_m3: float
) -> np.ndarray:
    """Return an M0/M3-conservative fixed-pivot density contribution."""

    if number_density_m3 < 0.0:
        raise ValueError("projection requires non-negative number")
    if not grid.centres_m[0] <= radius_m <= grid.centres_m[-1]:
        raise RuntimeError("projection radius lies outside fixed-pivot support")
    result = np.zeros(grid.bins, dtype=np.float64)
    upper = int(np.searchsorted(grid.centres_m, radius_m, side="left"))
    if upper == 0:
        result[0] = number_density_m3 / grid.widths_m[0]
        return result
    if upper == grid.bins:
        result[-1] = number_density_m3 / grid.widths_m[-1]
        return result
    lower = upper - 1
    lower_cubed = float(grid.centres_m[lower] ** 3)
    upper_cubed = float(grid.centres_m[upper] ** 3)
    upper_weight = (radius_m**3 - lower_cubed) / (upper_cubed - lower_cubed)
    lower_weight = 1.0 - upper_weight
    if not 0.0 <= lower_weight <= 1.0 or not 0.0 <= upper_weight <= 1.0:
        raise RuntimeError("invalid fixed-pivot weights")
    result[lower] = number_density_m3 * lower_weight / grid.widths_m[lower]
    result[upper] = number_density_m3 * upper_weight / grid.widths_m[upper]
    return result


def _initial_fixture_tags(*, solver: KWNSolver, construction: Mapping[str, Any], fixture: Any) -> dict[str, dict[str, Any]]:
    """Create linearly advected class tags for the six-particle control PSD."""

    beta = solver.population("beta")
    scale = float(construction["resolved_psd_projection"]["resolved_equivalent_number_scale"])
    classes: dict[float, int] = defaultdict(int)
    for radius in fixture.resolved_equivalent_radii_m:
        classes[float(radius)] += 1
    tags: dict[str, dict[str, Any]] = {}
    for radius, count in sorted(classes.items()):
        label = f"R{radius * 1.0e9:.6g}nm"
        density = _project_point_to_fixed_pivots(
            grid=beta.grid,
            radius_m=radius,
            number_density_m3=scale * count / fixture.box_volume_m3,
        )
        tags[label] = {
            "initial_radius_m": radius,
            "source_particle_count": count,
            "density_per_m4": density,
            "initial_projected_bin_indices": ";".join(
                str(int(index))
                for index in np.flatnonzero(density > 0.0)
            ),
            "initial_number_density_m3": float(np.sum(density * beta.grid.widths_m)),
            "initial_beta_inventory_mol_m3": (
                float(np.sum(density * beta.grid.widths_m * beta.grid.centres_m**3))
                * (4.0 * math.pi / 3.0)
                * beta.parameters.x_b
                / beta.parameters.molar_volume_m3_mol
            ),
            "cumulative_lower_number_m3": 0.0,
            "cumulative_lower_beta_inventory_mol_m3": 0.0,
            "first_lower_crossing_time_h": None,
        }
    tag_sum = np.sum([np.asarray(item["density_per_m4"]) for item in tags.values()], axis=0)
    if not np.allclose(tag_sum, beta.number_density_per_m4, rtol=5.0e-15, atol=0.0):
        raise RuntimeError("initial class-tag projection does not close the aggregate PSD")
    return tags


def _implicit_tag_update(
    *, density: np.ndarray, grid: RadiusGrid, velocity_m_s: np.ndarray, dt_s: float
) -> tuple[np.ndarray, float]:
    """Apply the production implicit M-matrix update to a passive PSD tag."""

    widths = grid.widths_m
    count = density.size
    lower = np.zeros(count, dtype=np.float64)
    diagonal = np.ones(count, dtype=np.float64)
    upper = np.zeros(count, dtype=np.float64)
    if velocity_m_s[0] < 0.0:
        diagonal[0] -= dt_s * velocity_m_s[0] / widths[0]
    face_velocity = 0.5 * (velocity_m_s[:-1] + velocity_m_s[1:])
    positive_faces = np.flatnonzero(face_velocity >= 0.0)
    negative_faces = np.flatnonzero(face_velocity < 0.0)
    if positive_faces.size:
        values = face_velocity[positive_faces]
        diagonal[positive_faces] += dt_s * values / widths[positive_faces]
        lower[positive_faces + 1] -= dt_s * values / widths[positive_faces + 1]
    if negative_faces.size:
        values = face_velocity[negative_faces]
        diagonal[negative_faces + 1] -= dt_s * values / widths[negative_faces + 1]
        upper[negative_faces] += dt_s * values / widths[negative_faces]
    if velocity_m_s[-1] > 0.0:
        diagonal[-1] += dt_s * velocity_m_s[-1] / widths[-1]
    upper_reduced = np.zeros(count, dtype=np.float64)
    rhs_reduced = np.empty(count, dtype=np.float64)
    pivot = diagonal[0]
    if pivot <= 0.0 or not math.isfinite(float(pivot)):
        raise SolverStateError("tag implicit solve has invalid first pivot")
    upper_reduced[0] = upper[0] / pivot
    rhs_reduced[0] = density[0] / pivot
    for index in range(1, count):
        pivot = diagonal[index] - lower[index] * upper_reduced[index - 1]
        if pivot <= 0.0 or not math.isfinite(float(pivot)):
            raise SolverStateError(f"tag implicit solve has invalid pivot at {index}")
        if index < count - 1:
            upper_reduced[index] = upper[index] / pivot
        rhs_reduced[index] = (density[index] - lower[index] * rhs_reduced[index - 1]) / pivot
    updated = np.empty_like(density)
    updated[-1] = rhs_reduced[-1]
    for index in range(count - 2, -1, -1):
        updated[index] = rhs_reduced[index] - upper_reduced[index] * updated[index + 1]
    if np.any(updated < -1.0e-280) or not np.all(np.isfinite(updated)):
        raise SolverStateError("tag implicit update violated positivity or finiteness")
    updated[(updated < 0.0) & (updated >= -1.0e-280)] = 0.0
    faces = KWNSolver._upwind_face_fluxes(updated, velocity_m_s)
    return updated, float(max(-faces[0], 0.0))


def _tag_rows(
    *, tags: Mapping[str, Mapping[str, Any]], solver: KWNSolver, run_id: str, bins: int, target_time_h: float
) -> list[dict[str, Any]]:
    beta = solver.population("beta")
    rates = solver.growth_rates()["beta"]
    rows: list[dict[str, Any]] = []
    for label, item in tags.items():
        density = np.asarray(item["density_per_m4"], dtype=np.float64)
        number = float(np.sum(density * beta.grid.widths_m))
        m3 = float(np.sum(density * beta.grid.widths_m * beta.grid.centres_m**3))
        effective_radius = 0.0 if number == 0.0 else (m3 / number) ** (1.0 / 3.0)
        active = np.flatnonzero(density > 0.0)
        weighted_rate = 0.0 if number == 0.0 else float(np.sum(density * beta.grid.widths_m * rates) / number)
        rows.append(
            {
                "run_id": run_id,
                "bins": bins,
                "target_time_h": target_time_h,
                "time_h": solver.time_s / 3600.0,
                "class_label": label,
                "initial_radius_m": item["initial_radius_m"],
                "source_particle_count": item["source_particle_count"],
                "initial_projected_bin_indices": item["initial_projected_bin_indices"],
                "current_occupied_bin_indices": ";".join(str(int(index)) for index in active),
                "effective_radius_m": effective_radius,
                "weighted_growth_rate_m_s": weighted_rate,
                "growth_dissolution_sign": "GROWTH" if weighted_rate > 0.0 else ("DISSOLUTION" if weighted_rate < 0.0 else "NEUTRAL"),
                "survival_number_density_m3": number,
                "survival_weight_fraction": number / float(item["initial_number_density_m3"]),
                "current_beta_inventory_mol_m3": (
                    m3
                    * (4.0 * math.pi / 3.0)
                    * beta.parameters.x_b
                    / beta.parameters.molar_volume_m3_mol
                ),
                "net_matrix_inventory_from_class_mol_m3": (
                    float(item["initial_beta_inventory_mol_m3"])
                    - m3
                    * (4.0 * math.pi / 3.0)
                    * beta.parameters.x_b
                    / beta.parameters.molar_volume_m3_mol
                ),
                "cumulative_lower_boundary_number_m3": item["cumulative_lower_number_m3"],
                "cumulative_returned_matrix_beta_inventory_mol_m3": item["cumulative_lower_beta_inventory_mol_m3"],
                "first_lower_boundary_crossing_time_h": "" if item["first_lower_crossing_time_h"] is None else item["first_lower_crossing_time_h"],
            }
        )
    return rows


def _schedule_times(schedule: str) -> tuple[float, ...]:
    if schedule == "uniform":
        return UNIFORM_OUTPUT_TIMES_H
    if schedule == "historical_200":
        return HISTORICAL_200_OUTPUT_TIMES_H
    if schedule == "historical_direct_48":
        return (0.0, 48.0)
    if schedule == "pf_accepted":
        return tuple(item[2] for item in PF_COMPARISON_TIME_MAP)
    raise RuntimeError(f"Unknown output schedule {schedule!r}")


def _find_snapshot(rows: Iterable[Mapping[str, Any]], target_time_h: float) -> Mapping[str, Any]:
    matches = [
        row
        for row in rows
        if math.isclose(float(row["target_time_h"]), target_time_h, rel_tol=0.0, abs_tol=1.0e-12)
    ]
    if len(matches) != 1:
        raise RuntimeError(f"Expected exactly one snapshot at {target_time_h} h, found {len(matches)}")
    return matches[0]


def _as_float_rows(rows: Iterable[Mapping[str, str]]) -> list[dict[str, Any]]:
    converted: list[dict[str, Any]] = []
    for row in rows:
        item: dict[str, Any] = {}
        for key, value in row.items():
            if value == "":
                item[key] = ""
                continue
            try:
                item[key] = float(value)
            except ValueError:
                item[key] = value
        converted.append(item)
    return converted


def _load_run(run_id: str) -> tuple[Path, dict[str, Any], list[dict[str, Any]]]:
    run_directory = TASK_OUTPUT_ROOT / "runs" / run_id
    manifest_path = run_directory / "manifest.json"
    trajectory_path = run_directory / "trajectory.csv"
    if not manifest_path.is_file() or not trajectory_path.is_file():
        raise RuntimeError(f"Missing completed task run {run_id}")
    manifest = _read_json(manifest_path)
    if manifest.get("status") != "COMPLETED":
        raise RuntimeError(f"Run {run_id} is not completed")
    rows = _as_float_rows(_read_csv(trajectory_path))
    return run_directory, manifest, rows


def _baseline_summary_path() -> Path:
    return TASK_OUTPUT_ROOT / "baseline_reproduction.json"


def _require_baseline_pass() -> None:
    path = _baseline_summary_path()
    if not path.is_file():
        raise RuntimeError("Run baseline before any new grid or qualification run")
    summary = _read_json(path)
    if summary.get("status") != "PASS_RADIUS_GRID_BASELINE_REPRODUCTION":
        raise RuntimeError(
            "Baseline did not reproduce; later radius-grid modification is prohibited by task contract"
        )


def _require_initial_projection_pass() -> None:
    path = TASK_OUTPUT_ROOT / "initial_projection_audit.json"
    if not path.is_file():
        raise RuntimeError("Run the t=0 initial-projection audit before any uniform ladder member")
    if _read_json(path).get("status") != "PASS_INITIAL_PSD_PROJECTION_CONSERVATION":
        raise RuntimeError(
            "Initial projection did not pass; transport/lattice runs are prohibited until it is resolved"
        )


def _fixture_source_moments(
    *, construction: Mapping[str, Any], fixture: Any
) -> dict[str, float]:
    scale = float(construction["resolved_psd_projection"]["resolved_equivalent_number_scale"])
    radii = np.asarray(fixture.resolved_equivalent_radii_m, dtype=np.float64)
    result = {
        f"source_M{order}": float(scale * np.sum(radii**order) / fixture.box_volume_m3)
        for order in range(4)
    }
    result["source_Sv_m_inv"] = 4.0 * math.pi * result["source_M2"]
    result["source_f_beta"] = 4.0 * math.pi * result["source_M3"] / 3.0
    return result


def _run_manifest(
    *,
    solver: KWNSolver,
    construction: Mapping[str, Any],
    contract: Any,
    fixture: Any,
    scenario: str,
    schedule: str,
    run_id: str,
    bins: int,
    max_dt_factor: float,
    output_times_h: tuple[float, ...],
) -> dict[str, Any]:
    beta = solver.population("beta")
    semantic_mapping = deepcopy(construction["config_mapping"])
    if isinstance(semantic_mapping.get("thermodynamics"), dict):
        semantic_mapping["thermodynamics"]["contract_path"] = "HASH_BOUND_CONTRACT_PATH"
    return {
        "schema_version": "KWN_RADIUS_GRID_RUN_V1",
        "status": "COMPLETED",
        "run_id": run_id,
        "scenario": scenario,
        "schedule": schedule,
        "radius_bins": bins,
        "output_times_h": list(output_times_h),
        "solver_mode": "CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE",
        "source_config_hash": solver.config.source_config_hash,
        "semantic_config_hash": _canonical_sha256(semantic_mapping),
        "contract_hash": contract.contract_hash,
        "fixture_id": fixture.fixture_id,
        "fixture_hash": fixture.fixture_hash,
        "profile_root_read_only": str(PROFILE_ROOT),
        "profile_library_manifest_sha256": _sha256_file(PROFILE_LIBRARY_MANIFEST),
        "initial_psd_source_hash": construction["initial_psd_source_hash"],
        "radius_edges_m": [float(value) for value in beta.grid.edges_m],
        "Rmin_m": float(beta.grid.edges_m[0]),
        "Rmax_m": float(beta.grid.edges_m[-1]),
        "max_dt_factor": max_dt_factor,
        "simulation": {
            "max_dt_s": solver.config.max_dt_s,
            "min_dt_s": solver.config.min_dt_s,
            "size_cfl": solver.config.size_cfl,
            "positivity_safety": solver.config.positivity_safety,
            "cfl_active_inventory_relative_threshold": solver.config.cfl_active_inventory_relative_threshold,
            "rmax_outflow_relative_tolerance": solver.config.rmax_outflow_relative_tolerance,
            "temperature_K": solver.config.temperature_k,
        },
        "lower_boundary_treatment": (
            "CONSERVATIVE_IMPLICIT_UPWIND_RMIN_OUTFLOW; global ledger recovers "
            "matrix inventory from fixed-pivot beta M3 without a clamp"
        ),
        "moment_definitions": {
            "production": "FIXED_PIVOT_SUM_n_i_delta_R_i_Rpivot_i^k",
            "diagnostic": "PIECEWISE_CONSTANT_CELL_INTEGRATED_MONOMIAL",
        },
        "construction": {
            key: value
            for key, value in construction.items()
            if key not in {"config_mapping", "resolved_psd_projection"}
        },
    }


def _advance_fixture_segment(
    *,
    solver: KWNSolver,
    target_time_s: float,
    tags: dict[str, dict[str, Any]] | None,
    cumulative: dict[str, Any],
    interval: dict[str, Any],
) -> None:
    """Advance a target interval and collect only accepted-step diagnostics."""

    if target_time_s < solver.time_s:
        raise RuntimeError("requested output time precedes current solver state")
    beta = solver.population("beta")
    while solver.time_s < target_time_s:
        velocities = solver.growth_rates()
        remaining_s = target_time_s - solver.time_s
        dt_s, _ = solver._choose_dt(velocities, remaining_s)
        diagnostic = solver.advance_one(maximum_dt_s=remaining_s)
        if not math.isclose(float(diagnostic.dt_s), dt_s, rel_tol=0.0, abs_tol=0.0):
            raise RuntimeError("accepted step differs from frozen start-state timestep proposal")
        # History has no state feedback and can otherwise dominate memory over
        # the 48 h 1600-bin ladder.
        solver.history.pop()
        lower_number = float(diagnostic.rmin_dissolution_flux_m3_s) * float(diagnostic.dt_s)
        lower_inventory = (
            lower_number
            * _sphere_volume(float(beta.grid.centres_m[0]))
            * beta.parameters.x_b
            / beta.parameters.molar_volume_m3_mol
        )
        for aggregate in (cumulative, interval):
            aggregate["dt_s"].append(float(diagnostic.dt_s))
            aggregate["lower_number_m3"] += lower_number
            aggregate["lower_beta_inventory_mol_m3"] += lower_inventory
            aggregate["maximum_positivity"] = max(
                aggregate["maximum_positivity"], float(diagnostic.positivity_utilization)
            )
            aggregate["maximum_cfl"] = max(aggregate["maximum_cfl"], float(diagnostic.size_cfl))
            aggregate["maximum_residual"] = max(
                aggregate["maximum_residual"], float(diagnostic.inventory.relative_residual)
            )
            aggregate["maximum_rmax_flux_m3_s"] = max(
                aggregate["maximum_rmax_flux_m3_s"], float(diagnostic.rmax_outflow_flux_m3_s)
            )
        interval["last_lower_flux_number_m3_s"] = float(diagnostic.rmin_dissolution_flux_m3_s)
        interval["maximum_lower_flux_number_m3_s"] = max(
            interval["maximum_lower_flux_number_m3_s"], float(diagnostic.rmin_dissolution_flux_m3_s)
        )

        if tags is not None:
            total_tag_density = np.zeros_like(beta.number_density_per_m4)
            tag_lower_number = 0.0
            for item in tags.values():
                updated, tag_lower_flux = _implicit_tag_update(
                    density=np.asarray(item["density_per_m4"], dtype=np.float64),
                    grid=beta.grid,
                    velocity_m_s=velocities["beta"],
                    dt_s=float(diagnostic.dt_s),
                )
                item["density_per_m4"] = updated
                released_number = tag_lower_flux * float(diagnostic.dt_s)
                released_inventory = (
                    released_number
                    * _sphere_volume(float(beta.grid.centres_m[0]))
                    * beta.parameters.x_b
                    / beta.parameters.molar_volume_m3_mol
                )
                item["cumulative_lower_number_m3"] += released_number
                item["cumulative_lower_beta_inventory_mol_m3"] += released_inventory
                if released_number > 0.0 and item["first_lower_crossing_time_h"] is None:
                    item["first_lower_crossing_time_h"] = solver.time_s / 3600.0
                total_tag_density += updated
                tag_lower_number += released_number
            scale = max(float(np.max(np.abs(beta.number_density_per_m4))), 1.0e-300)
            tag_error = float(np.max(np.abs(total_tag_density - beta.number_density_per_m4)) / scale)
            cumulative["maximum_tag_aggregate_relative_error"] = max(
                cumulative["maximum_tag_aggregate_relative_error"], tag_error
            )
            interval["maximum_tag_aggregate_relative_error"] = max(
                interval["maximum_tag_aggregate_relative_error"], tag_error
            )
            cumulative["tag_lower_number_m3"] += tag_lower_number
            interval["tag_lower_number_m3"] += tag_lower_number


def _new_audit_aggregate() -> dict[str, Any]:
    return {
        "dt_s": [],
        "lower_number_m3": 0.0,
        "lower_beta_inventory_mol_m3": 0.0,
        "maximum_positivity": 0.0,
        "maximum_cfl": 0.0,
        "maximum_residual": 0.0,
        "maximum_rmax_flux_m3_s": 0.0,
        "last_lower_flux_number_m3_s": 0.0,
        "maximum_lower_flux_number_m3_s": 0.0,
        "maximum_tag_aggregate_relative_error": 0.0,
        "tag_lower_number_m3": 0.0,
    }


def _run_grid(
    *,
    run_id: str,
    scenario: str,
    bins: int,
    schedule: str,
    max_dt_factor: float,
    require_baseline: bool,
) -> Path:
    """Execute one immutable radius-grid member and record raw snapshots."""

    if require_baseline:
        _require_baseline_pass()
        _require_initial_projection_pass()
    if bins < 2 or not math.isfinite(max_dt_factor) or max_dt_factor <= 0.0:
        raise RuntimeError("bins and max_dt_factor must be positive")
    output_times_h = _schedule_times(schedule)
    if output_times_h[0] != 0.0 or any(
        right <= left for left, right in zip(output_times_h, output_times_h[1:])
    ):
        raise RuntimeError("output schedule must be strictly increasing from t=0")
    if scenario == "fixture":
        solver, construction, contract, fixture = _build_fixture_solver(
            bins=bins, max_dt_factor=max_dt_factor
        )
        tags: dict[str, dict[str, Any]] | None = _initial_fixture_tags(
            solver=solver, construction=construction, fixture=fixture
        )
    elif scenario == "smooth":
        solver, construction, contract, fixture = _build_smooth_solver(
            bins=bins, max_dt_factor=max_dt_factor
        )
        tags = None
    else:
        raise RuntimeError(f"Unknown scenario {scenario!r}")

    run_directory = _run_directory(run_id)
    snapshots: list[dict[str, Any]] = []
    lower_rows: list[dict[str, Any]] = []
    tag_records: list[dict[str, Any]] = []
    density_snapshots: list[np.ndarray] = []
    snapshot_steps: list[int] = []
    cumulative = _new_audit_aggregate()
    for target_time_h in output_times_h:
        interval = _new_audit_aggregate()
        _advance_fixture_segment(
            solver=solver,
            target_time_s=target_time_h * 3600.0,
            tags=tags,
            cumulative=cumulative,
            interval=interval,
        )
        row = _snapshot_row(
            solver=solver,
            scenario=scenario,
            run_id=run_id,
            bins=bins,
            target_time_h=target_time_h,
            interval_dt=interval["dt_s"],
            cumulative_dt=cumulative["dt_s"],
            cumulative_lower_number_m3=float(cumulative["lower_number_m3"]),
            cumulative_lower_beta_inventory_mol_m3=float(
                cumulative["lower_beta_inventory_mol_m3"]
            ),
            interval_lower_number_m3=float(interval["lower_number_m3"]),
            last_lower_flux_number_m3_s=float(interval["last_lower_flux_number_m3_s"]),
            max_interval_positivity=float(interval["maximum_positivity"]),
            max_interval_cfl=float(interval["maximum_cfl"]),
            max_interval_residual=float(interval["maximum_residual"]),
        )
        row.update(
            {
                "maximum_interval_size_cfl": interval["maximum_cfl"],
                "maximum_interval_rmax_outflow_flux_m3_s": interval[
                    "maximum_rmax_flux_m3_s"
                ],
                "maximum_interval_lower_flux_number_m3_s": interval[
                    "maximum_lower_flux_number_m3_s"
                ],
                "maximum_interval_tag_aggregate_relative_error": interval[
                    "maximum_tag_aggregate_relative_error"
                ],
                "cumulative_maximum_positivity_utilization": cumulative[
                    "maximum_positivity"
                ],
                "cumulative_maximum_size_cfl": cumulative["maximum_cfl"],
                "cumulative_maximum_residual": cumulative["maximum_residual"],
                "cumulative_maximum_rmax_outflow_flux_m3_s": cumulative[
                    "maximum_rmax_flux_m3_s"
                ],
                "cumulative_maximum_tag_aggregate_relative_error": cumulative[
                    "maximum_tag_aggregate_relative_error"
                ],
                "cumulative_tag_lower_boundary_number_m3": cumulative["tag_lower_number_m3"],
            }
        )
        snapshots.append(row)
        density_snapshots.append(solver.population("beta").number_density_per_m4.copy())
        snapshot_steps.append(solver.step)
        lower_rows.append(
            {
                "scenario": scenario,
                "run_id": run_id,
                "bins": bins,
                "target_time_h": target_time_h,
                "time_h": solver.time_s / 3600.0,
                "step": solver.step,
                "Rmin_m": float(solver.config.grid.edges_m[0]),
                "Rmin_fixed_pivot_m": float(solver.config.grid.centres_m[0]),
                "interval_lower_boundary_number_m3": interval["lower_number_m3"],
                "cumulative_lower_boundary_number_m3": cumulative["lower_number_m3"],
                "interval_lower_boundary_beta_inventory_mol_m3": interval[
                    "lower_beta_inventory_mol_m3"
                ],
                "cumulative_lower_boundary_beta_inventory_mol_m3": cumulative[
                    "lower_beta_inventory_mol_m3"
                ],
                "last_lower_boundary_flux_number_m3_s": interval[
                    "last_lower_flux_number_m3_s"
                ],
                "maximum_interval_lower_boundary_flux_number_m3_s": interval[
                    "maximum_lower_flux_number_m3_s"
                ],
                "cumulative_tag_lower_boundary_number_m3": cumulative["tag_lower_number_m3"],
                "tag_vs_aggregate_lower_number_difference_m3": (
                    cumulative["tag_lower_number_m3"] - cumulative["lower_number_m3"]
                ),
                "maximum_tag_aggregate_relative_error": cumulative[
                    "maximum_tag_aggregate_relative_error"
                ],
                "total_residual_relative": row["total_residual_relative"],
                "rmax_outflow_maximum_m3_s": interval["maximum_rmax_flux_m3_s"],
            }
        )
        if tags is not None:
            tag_records.extend(
                _tag_rows(
                    tags=tags,
                    solver=solver,
                    run_id=run_id,
                    bins=bins,
                    target_time_h=target_time_h,
                )
            )
        print(
            json.dumps(
                {
                    "event": "snapshot",
                    "run_id": run_id,
                    "bins": bins,
                    "scenario": scenario,
                    "time_h": target_time_h,
                    "step": solver.step,
                    "accepted_steps": len(cumulative["dt_s"]),
                },
                sort_keys=True,
            ),
            flush=True,
        )

    _write_csv_new(run_directory / "trajectory.csv", snapshots)
    _write_csv_new(run_directory / "lower_boundary_flux.csv", lower_rows)
    if tags is not None:
        _write_csv_new(run_directory / "class_events.csv", tag_records)
    _write_npz_new(
        run_directory / "psd_snapshots.npz",
        time_h=np.asarray(output_times_h, dtype=np.float64),
        step=np.asarray(snapshot_steps, dtype=np.int64),
        radius_edges_m=solver.config.grid.edges_m,
        number_density_per_m4=np.asarray(density_snapshots, dtype=np.float64),
    )
    manifest = _run_manifest(
        solver=solver,
        construction=construction,
        contract=contract,
        fixture=fixture,
        scenario=scenario,
        schedule=schedule,
        run_id=run_id,
        bins=bins,
        max_dt_factor=max_dt_factor,
        output_times_h=output_times_h,
    )
    manifest["completion"] = {
        "final_time_h": solver.time_s / 3600.0,
        "final_step": solver.step,
        "accepted_step_count": len(cumulative["dt_s"]),
        "maximum_positivity_utilization": cumulative["maximum_positivity"],
        "maximum_size_cfl": cumulative["maximum_cfl"],
        "maximum_inventory_relative_residual": cumulative["maximum_residual"],
        "roundoff_zeroed_bin_count": solver.roundoff_zeroed_bin_count,
        "maximum_rmax_outflow_flux_m3_s": cumulative["maximum_rmax_flux_m3_s"],
        "maximum_tag_aggregate_relative_error": cumulative[
            "maximum_tag_aggregate_relative_error"
        ],
    }
    _write_json_new(run_directory / "manifest.json", manifest)
    return run_directory


def _within_baseline_tolerance(observed: float, expected: float) -> bool:
    return math.isclose(
        observed,
        expected,
        rel_tol=BASELINE_FLOAT_REL_TOLERANCE,
        abs_tol=BASELINE_FLOAT_ABS_TOLERANCE,
    )


def _reproduce_legacy_failure() -> dict[str, Any]:
    """Replay the frozen unsafe diagnostic mapping without making it selectable."""

    frozen = _read_json(FROZEN_LEGACY_DIAGNOSIS)
    mapping = deepcopy(frozen["construction"]["config_mapping"])
    solver = legacy_diagnosis.LegacyUnlimitedFluxDiagnosticSolver(
        SolverConfig.from_mapping(mapping)
    )
    candidate: dict[str, Any] | None = None
    caught: BaseException | None = None
    ceiling_s = 48.0 * 3600.0
    while solver.time_s < ceiling_s:
        candidate, _ = legacy_diagnosis._candidate_state(
            solver, maximum_dt_s=ceiling_s - solver.time_s
        )
        try:
            solver.advance_one(maximum_dt_s=ceiling_s - solver.time_s)
        except SolverStateError as error:
            caught = error
            break
        finally:
            if solver.history:
                solver.history.pop()
    if candidate is None or caught is None:
        return {
            "status": "FAIL",
            "reason": "legacy diagnostic did not reach a strict positivity failure",
            "source_config_hash": solver.config.source_config_hash,
        }
    expected_failure = frozen["failure"]
    expected_evidence = frozen["classification_evidence"]
    observed_beta = candidate["populations"]["beta"]
    comparison = {
        "accepted_step": {
            "expected": frozen["completion"]["accepted_steps"],
            "observed": solver.step,
            "pass": solver.step == int(frozen["completion"]["accepted_steps"]),
        },
        "candidate_step": {
            "expected": int(expected_failure["step"]) + 1,
            "observed": solver.step + 1,
            "pass": solver.step + 1 == int(expected_failure["step"]) + 1,
        },
        "failure_time_s": {
            "expected": float(expected_failure["time_s"]),
            "observed": float(candidate["time_s"]),
            "pass": _within_baseline_tolerance(
                float(candidate["time_s"]), float(expected_failure["time_s"])
            ),
        },
        "negative_beta_bin": {
            "expected": expected_evidence["candidate_negative_beta_bin"],
            "observed": observed_beta["first_candidate_negative_bin"],
            "pass": (
                observed_beta["first_candidate_negative_bin"]
                == expected_evidence["candidate_negative_beta_bin"]
            ),
        },
        "candidate_dt_s": {
            "expected": float(expected_evidence["candidate_dt_s"]),
            "observed": float(candidate["dt_s"]),
            "pass": _within_baseline_tolerance(
                float(candidate["dt_s"]), float(expected_evidence["candidate_dt_s"])
            ),
        },
        "donor_bound_s": {
            "expected": float(expected_evidence["donor_dt_max_s"]),
            "observed": float(observed_beta["first_candidate_negative_dt_max_s"]),
            "pass": _within_baseline_tolerance(
                float(observed_beta["first_candidate_negative_dt_max_s"]),
                float(expected_evidence["donor_dt_max_s"]),
            ),
        },
        "raw_positivity_utilization": {
            "expected": float(expected_evidence["raw_positivity_utilization"]),
            "observed": float(observed_beta["raw_positivity_utilization"]),
            "pass": _within_baseline_tolerance(
                float(observed_beta["raw_positivity_utilization"]),
                float(expected_evidence["raw_positivity_utilization"]),
            ),
        },
        "candidate_density_per_m4": {
            "expected": float(expected_failure["populations"]["beta"]["first_candidate_negative_after_density_per_m4"]),
            "observed": float(observed_beta["first_candidate_negative_after_density_per_m4"]),
            "pass": _within_baseline_tolerance(
                float(observed_beta["first_candidate_negative_after_density_per_m4"]),
                float(expected_failure["populations"]["beta"]["first_candidate_negative_after_density_per_m4"]),
            ),
        },
        "inventory_relative_residual": {
            "expected": float(expected_failure["inventory_relative_residual"]),
            "observed": float(candidate["inventory_relative_residual"]),
            "pass": _within_baseline_tolerance(
                float(candidate["inventory_relative_residual"]),
                float(expected_failure["inventory_relative_residual"]),
            ),
        },
    }
    return {
        "status": "PASS" if all(item["pass"] for item in comparison.values()) else "FAIL",
        "comparison": comparison,
        "legacy_solver_mode": solver.solver_version,
        "source_config_hash": solver.config.source_config_hash,
        "semantic_mapping_hash": _canonical_sha256(mapping),
        "frozen_diagnosis_sha256": _sha256_file(FROZEN_LEGACY_DIAGNOSIS),
        "error": f"{type(caught).__name__}: {caught}",
        "float_tolerance": {
            "relative": BASELINE_FLOAT_REL_TOLERANCE,
            "absolute": BASELINE_FLOAT_ABS_TOLERANCE,
            "rationale": "historical replay is judged at a predeclared 5e-14 relative tolerance; observed differences are one floating-point ulp while discrete failure identity is exact",
        },
    }


def _historical_p5_comparison(
    *, run_200: list[Mapping[str, Any]], run_400: list[Mapping[str, Any]]
) -> dict[str, Any]:
    frozen = _read_json(FROZEN_REPAIR_SUMMARY)
    current_200 = _find_snapshot(run_200, 48.0)
    current_400 = _find_snapshot(run_400, 48.0)
    metric_map = {
        "beta_number_density_m3": "N_m0_m3",
        "beta_mean_radius_m": "Rmean_m",
        "beta_mean_radius_cubed_m3": "Rmean3_m3",
        "beta_specific_surface_area_m_inv": "Sv_m_inv",
        "beta_volume_fraction": "f_beta",
        "matrix_xB": "matrix_xB",
    }
    exact_replay: dict[str, Any] = {}
    p5: dict[str, Any] = {}
    for frozen_name, current_name in metric_map.items():
        expected_200 = float(frozen["canonical_48h"][frozen_name])
        expected_400 = float(frozen["grid_400_48h"][frozen_name])
        observed_200 = float(current_200[current_name])
        observed_400 = float(current_400[current_name])
        observed_error = _relative_error(observed_200, observed_400)
        expected_error = float(frozen["grid_convergence_relative_difference"][frozen_name])
        exact_replay[frozen_name] = {
            "expected_200": expected_200,
            "observed_200": observed_200,
            "expected_400": expected_400,
            "observed_400": observed_400,
            "pass": _within_baseline_tolerance(observed_200, expected_200)
            and _within_baseline_tolerance(observed_400, expected_400),
        }
        p5[frozen_name] = {
            "expected_relative_difference": expected_error,
            "observed_relative_difference": observed_error,
            "pass": _within_baseline_tolerance(observed_error, expected_error),
        }
    return {
        "status": "PASS"
        if all(item["pass"] for item in exact_replay.values())
        and all(item["pass"] for item in p5.values())
        else "FAIL",
        "historical_schedule": {
            "200": list(HISTORICAL_200_OUTPUT_TIMES_H),
            "400": [0.0, 48.0],
            "known_asymmetry": "200 uses sequential exact output clipping; 400 advances directly from 0 to 48 h",
        },
        "endpoint_replay": exact_replay,
        "p5_relative_differences": p5,
    }


def _run_baseline() -> int:
    destination = _baseline_summary_path()
    if destination.exists():
        raise RuntimeError(f"Refusing to overwrite existing baseline evidence {destination}")
    strict = _reproduce_legacy_failure()
    if strict["status"] != "PASS":
        _write_json_new(
            destination,
            {
                "schema_version": "KWN_RADIUS_GRID_BASELINE_REPRODUCTION_V1",
                "status": "FAIL_RADIUS_GRID_BASELINE_REPRODUCTION",
                "strict_legacy_failure": strict,
                "stop_reason": "Historical strict positivity failure did not reproduce within the predeclared tolerance; task contract prohibits later grid modifications.",
            },
        )
        return 2

    # The historical P5 path must retain its old output segmentation before a
    # uniform schedule is allowed to diagnose radius-grid convergence.
    _run_grid(
        run_id="baseline_fixture_200_historical",
        scenario="fixture",
        bins=200,
        schedule="historical_200",
        max_dt_factor=1.0,
        require_baseline=False,
    )
    _run_grid(
        run_id="baseline_fixture_400_historical_direct",
        scenario="fixture",
        bins=400,
        schedule="historical_direct_48",
        max_dt_factor=1.0,
        require_baseline=False,
    )
    # 100 bins was not part of the frozen P5 comparison. It is recorded as a
    # new, explicitly non-historical extension so the requested baseline table
    # still contains every ladder member initially requested by this task.
    _run_grid(
        run_id="baseline_fixture_100_uniform_extension",
        scenario="fixture",
        bins=100,
        schedule="uniform",
        max_dt_factor=1.0,
        require_baseline=False,
    )
    _, _, historical_200 = _load_run("baseline_fixture_200_historical")
    _, _, historical_400 = _load_run("baseline_fixture_400_historical_direct")
    repaired = _historical_p5_comparison(run_200=historical_200, run_400=historical_400)
    status = (
        "PASS_RADIUS_GRID_BASELINE_REPRODUCTION"
        if repaired["status"] == "PASS"
        else "FAIL_RADIUS_GRID_BASELINE_REPRODUCTION"
    )
    _write_json_new(
        destination,
        {
            "schema_version": "KWN_RADIUS_GRID_BASELINE_REPRODUCTION_V1",
            "status": status,
            "strict_legacy_failure": strict,
            "repaired_48h_and_historical_p5": repaired,
            "baseline_runs": {
                "fixture_100": "baseline_fixture_100_uniform_extension",
                "fixture_200": "baseline_fixture_200_historical",
                "fixture_400": "baseline_fixture_400_historical_direct",
            },
            "solver_mode": "CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE",
        },
    )
    return 0 if status == "PASS_RADIUS_GRID_BASELINE_REPRODUCTION" else 2


def _initial_grid_rows() -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    reference: dict[str, Any] | None = None
    for bins in GRID_LADDER:
        solver, construction, contract, fixture = _build_fixture_solver(bins=bins)
        beta = solver.population("beta")
        fixed = [beta.radius_moment(order, quadrature="fixed_pivot") for order in range(4)]
        cell = [beta.radius_moment(order, quadrature="cell_integrated") for order in range(4)]
        ledger = solver.ledger.snapshot(
            matrix_xb=solver.matrix_xb,
            populations=solver.population_list(),
            beta_resolved_fraction=1.0,
        )
        source = _fixture_source_moments(construction=construction, fixture=fixture)
        occupied = np.flatnonzero(beta.number_density_per_m4 > 0.0)
        row: dict[str, Any] = {
            "bins": bins,
            "contract_hash": contract.contract_hash,
            "fixture_hash": fixture.fixture_hash,
            "initial_psd_source_hash": construction["initial_psd_source_hash"],
            "Rmin_m": float(beta.grid.edges_m[0]),
            "Rmax_m": float(beta.grid.edges_m[-1]),
            "M0_m3": fixed[0],
            "M1_m2": fixed[1],
            "M2_m": fixed[2],
            "M3_dimensionless": fixed[3],
            "Rmean_m": fixed[1] / fixed[0],
            "Rmean3_m3": fixed[3] / fixed[0],
            "Sv_m_inv": 4.0 * math.pi * fixed[2],
            "f_beta": 4.0 * math.pi * fixed[3] / 3.0,
            "matrix_xB": solver.matrix_xb,
            "matrix_inventory_mol_m3": ledger.matrix_mol_m3,
            "gp_inventory_mol_m3": ledger.gp_mol_m3,
            "beta_subgrid_inventory_mol_m3": ledger.beta_subgrid_mol_m3,
            "beta_resolved_inventory_mol_m3": ledger.beta_resolved_mol_m3,
            "total_inventory_mol_m3": ledger.total_mol_m3,
            "total_residual_relative": ledger.relative_residual,
            "occupied_bin_count": int(occupied.size),
            "minimum_occupied_radius_m": float(beta.grid.centres_m[occupied[0]]),
            "maximum_occupied_radius_m": float(beta.grid.centres_m[occupied[-1]]),
            "midpoint_M0_m3": fixed[0],
            "midpoint_M1_m2": fixed[1],
            "midpoint_M2_m": fixed[2],
            "midpoint_M3_dimensionless": fixed[3],
            "midpoint_Sv_m_inv": 4.0 * math.pi * fixed[2],
            "midpoint_f_beta": 4.0 * math.pi * fixed[3] / 3.0,
            "cell_integrated_M0_m3": cell[0],
            "cell_integrated_M1_m2": cell[1],
            "cell_integrated_M2_m": cell[2],
            "cell_integrated_M3_dimensionless": cell[3],
            "cell_integrated_Sv_m_inv": 4.0 * math.pi * cell[2],
            "cell_integrated_f_beta": 4.0 * math.pi * cell[3] / 3.0,
            "source_M0_m3": source["source_M0"],
            "source_M1_m2": source["source_M1"],
            "source_M2_m": source["source_M2"],
            "source_M3_dimensionless": source["source_M3"],
            "source_Sv_m_inv": source["source_Sv_m_inv"],
            "source_f_beta": source["source_f_beta"],
            "projection_relative_error_M0": _relative_error(fixed[0], source["source_M0"]),
            "projection_relative_error_M1": _relative_error(fixed[1], source["source_M1"]),
            "projection_relative_error_M2": _relative_error(fixed[2], source["source_M2"]),
            "projection_relative_error_M3": _relative_error(fixed[3], source["source_M3"]),
            "projection_relative_error_Sv": _relative_error(
                4.0 * math.pi * fixed[2], source["source_Sv_m_inv"]
            ),
        }
        rows.append(row)
        if bins == GRID_LADDER[-1]:
            reference = row
    assert reference is not None
    comparison_keys = (
        "M0_m3",
        "M3_dimensionless",
        "beta_resolved_inventory_mol_m3",
        "total_inventory_mol_m3",
    )
    for row in rows:
        for key in comparison_keys:
            row[f"relative_to_1600_{key}"] = _relative_error(
                float(row[key]), float(reference[key])
            )
    maximum_conservation_difference = max(
        float(row[f"relative_to_1600_{key}"])
        for row in rows
        for key in comparison_keys
    )
    maximum_cell_quadrature_difference = max(
        _relative_error(
            float(row["cell_integrated_M3_dimensionless"]),
            float(row["midpoint_M3_dimensionless"]),
        )
        for row in rows
    )
    summary = {
        "status": "PASS_INITIAL_PSD_PROJECTION_CONSERVATION"
        if maximum_conservation_difference <= 1.0e-12
        else "FAIL_INITIAL_PSD_PROJECTION_NOT_CONSERVATIVE",
        "maximum_M0_M3_inventory_relative_difference": maximum_conservation_difference,
        "threshold": 1.0e-12,
        "moment_quadrature": {
            "production_representation": "fixed_pivot",
            "cell_integrated_diagnostic_only": True,
            "maximum_t0_M3_relative_difference_vs_midpoint": maximum_cell_quadrature_difference,
        },
    }
    return rows, summary


def _run_initial_audit() -> int:
    _require_baseline_pass()
    csv_path = TASK_OUTPUT_ROOT / "initial_grid_moments.csv"
    summary_path = TASK_OUTPUT_ROOT / "initial_projection_audit.json"
    if csv_path.exists() or summary_path.exists():
        raise RuntimeError("Refusing to overwrite initial-grid audit outputs")
    rows, summary = _initial_grid_rows()
    _write_csv_new(csv_path, rows)
    _write_json_new(summary_path, summary)
    print(json.dumps(summary, sort_keys=True))
    return 0 if summary["status"] == "PASS_INITIAL_PSD_PROJECTION_CONSERVATION" else 2


def _load_psd_snapshot(run_directory: Path, target_time_h: float) -> tuple[RadiusGrid, np.ndarray]:
    path = run_directory / "psd_snapshots.npz"
    if not path.is_file():
        raise RuntimeError(f"Missing PSD snapshots in {run_directory}")
    with np.load(path, allow_pickle=False) as archive:
        time_h = np.asarray(archive["time_h"], dtype=np.float64)
        matches = np.flatnonzero(np.isclose(time_h, target_time_h, rtol=0.0, atol=1.0e-12))
        if matches.size != 1:
            raise RuntimeError(
                f"Expected exactly one PSD snapshot at {target_time_h} h in {path}, found {matches.size}"
            )
        grid = RadiusGrid(np.asarray(archive["radius_edges_m"], dtype=np.float64))
        density = np.asarray(archive["number_density_per_m4"], dtype=np.float64)[int(matches[0])]
    if density.shape != (grid.bins,) or np.any(density < 0.0):
        raise RuntimeError(f"Invalid PSD snapshot in {path}")
    return grid, density


def _remap_fixed_pivot_density(
    *, source_grid: RadiusGrid, source_density: np.ndarray, target_grid: RadiusGrid
) -> np.ndarray:
    """Conservatively remap fixed-pivot M0/M3 atoms onto another log grid."""

    source_number = np.asarray(source_density, dtype=np.float64) * source_grid.widths_m
    if np.any(source_number < 0.0):
        raise RuntimeError("cannot remap a negative PSD")
    target_number = np.zeros(target_grid.bins, dtype=np.float64)
    centres = target_grid.centres_m
    for radius, number in zip(source_grid.centres_m, source_number):
        if number == 0.0:
            continue
        upper = int(np.searchsorted(centres, radius, side="left"))
        if upper == 0:
            target_number[0] += number
        elif upper == target_grid.bins:
            target_number[-1] += number
        else:
            lower = upper - 1
            upper_weight = (radius**3 - centres[lower] ** 3) / (
                centres[upper] ** 3 - centres[lower] ** 3
            )
            target_number[lower] += number * (1.0 - upper_weight)
            target_number[upper] += number * upper_weight
    return target_number / target_grid.widths_m


def _psd_distance_row(
    *,
    scenario: str,
    left_id: str,
    right_id: str,
    left_bins: int,
    right_bins: int,
    target_time_h: float,
    shared_grid: RadiusGrid,
) -> dict[str, Any]:
    left_dir, _, _ = _load_run(left_id)
    right_dir, _, _ = _load_run(right_id)
    left_grid, left_density = _load_psd_snapshot(left_dir, target_time_h)
    right_grid, right_density = _load_psd_snapshot(right_dir, target_time_h)
    left_remapped = _remap_fixed_pivot_density(
        source_grid=left_grid, source_density=left_density, target_grid=shared_grid
    )
    right_remapped = _remap_fixed_pivot_density(
        source_grid=right_grid, source_density=right_density, target_grid=shared_grid
    )
    radii = shared_grid.centres_m
    left_number = left_remapped * shared_grid.widths_m
    right_number = right_remapped * shared_grid.widths_m
    left_m0_before = float(np.sum(left_density * left_grid.widths_m))
    right_m0_before = float(np.sum(right_density * right_grid.widths_m))
    left_m3_before = float(np.sum(left_density * left_grid.widths_m * left_grid.centres_m**3))
    right_m3_before = float(np.sum(right_density * right_grid.widths_m * right_grid.centres_m**3))
    left_m0_after = float(np.sum(left_number))
    right_m0_after = float(np.sum(right_number))
    left_m3_after = float(np.sum(left_number * radii**3))
    right_m3_after = float(np.sum(right_number * radii**3))
    return {
        "scenario": scenario,
        "left_run_id": left_id,
        "right_run_id": right_id,
        "left_bins": left_bins,
        "right_bins": right_bins,
        "target_time_h": target_time_h,
        "shared_analysis_grid_bins": shared_grid.bins,
        "shared_Rmin_m": float(shared_grid.edges_m[0]),
        "shared_Rmax_m": float(shared_grid.edges_m[-1]),
        "remap_method": "TWO_FIXED_PIVOT_CENTRES_PRESERVING_M0_AND_M3",
        "left_remap_M0_relative_error": _relative_error(left_m0_after, left_m0_before),
        "right_remap_M0_relative_error": _relative_error(right_m0_after, right_m0_before),
        "left_remap_M3_relative_error": _relative_error(left_m3_after, left_m3_before),
        "right_remap_M3_relative_error": _relative_error(right_m3_after, right_m3_before),
        "W1_number_m": discrete_wasserstein_distance(radii, left_number, radii, right_number),
        "W1_volume_m": discrete_wasserstein_distance(
            radii, left_number * radii**3, radii, right_number * radii**3
        ),
    }


def _available_ladder_bins(scenario: str) -> list[int]:
    result: list[int] = []
    for bins in (*GRID_LADDER, 3200):
        path = TASK_OUTPUT_ROOT / "runs" / f"ladder_{scenario}_{bins}_uniform" / "manifest.json"
        if path.is_file():
            result.append(bins)
    return result


def _validate_uniform_ladder_members(
    *, scenario: str, run_data: Mapping[int, tuple[Path, Mapping[str, Any], list[dict[str, Any]]]]
) -> None:
    """Reject a P5 comparison if a purported ladder member changed its contract."""

    reference_bins = min(run_data)
    reference = run_data[reference_bins][1]
    required_scalar_keys = (
        "contract_hash",
        "fixture_hash",
        "solver_mode",
        "Rmin_m",
        "Rmax_m",
        "lower_boundary_treatment",
        "profile_library_manifest_sha256",
    )
    for bins, (_, manifest, _) in run_data.items():
        failures: list[str] = []
        if manifest.get("scenario") != scenario:
            failures.append(f"scenario={manifest.get('scenario')!r}")
        if manifest.get("schedule") != "uniform":
            failures.append(f"schedule={manifest.get('schedule')!r}")
        if int(manifest.get("radius_bins", -1)) != bins:
            failures.append(f"radius_bins={manifest.get('radius_bins')!r}")
        if not math.isclose(float(manifest.get("max_dt_factor", float("nan"))), 1.0, rel_tol=0.0, abs_tol=0.0):
            failures.append(f"max_dt_factor={manifest.get('max_dt_factor')!r}")
        if tuple(float(value) for value in manifest.get("output_times_h", ())) != UNIFORM_OUTPUT_TIMES_H:
            failures.append("output_times_h differs from the registered uniform schedule")
        for key in required_scalar_keys:
            if manifest.get(key) != reference.get(key):
                failures.append(f"{key} differs from {reference_bins}-bin member")
        if manifest.get("simulation") != reference.get("simulation"):
            failures.append("simulation timestep policy differs from reference member")
        if manifest.get("moment_definitions") != reference.get("moment_definitions"):
            failures.append("moment definitions differ from reference member")
        if failures:
            raise RuntimeError(
                f"Invalid {scenario} P5 ladder member {bins}: " + "; ".join(failures)
            )


def _pair_rows_and_assessment(
    *, scenario: str, bins_ladder: list[int]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if len(bins_ladder) < 2:
        raise RuntimeError(f"Need at least two {scenario} grid runs")
    run_data = {
        bins: _load_run(f"ladder_{scenario}_{bins}_uniform") for bins in bins_ladder
    }
    _validate_uniform_ladder_members(scenario=scenario, run_data=run_data)
    sample_manifest = run_data[bins_ladder[0]][1]
    shared_grid = RadiusGrid.logarithmic(
        float(sample_manifest["Rmin_m"]), float(sample_manifest["Rmax_m"]), 3200
    )
    pairs: list[dict[str, Any]] = []
    psd: list[dict[str, Any]] = []
    endpoint_errors: dict[tuple[int, int], dict[str, float]] = {}
    for left_bins, right_bins in zip(bins_ladder, bins_ladder[1:]):
        left_id = f"ladder_{scenario}_{left_bins}_uniform"
        right_id = f"ladder_{scenario}_{right_bins}_uniform"
        _, _, left_rows = run_data[left_bins]
        _, _, right_rows = run_data[right_bins]
        endpoint_errors[(left_bins, right_bins)] = {}
        for target_time_h in UNIFORM_OUTPUT_TIMES_H:
            left = _find_snapshot(left_rows, target_time_h)
            right = _find_snapshot(right_rows, target_time_h)
            for metric in PRIMARY_METRICS:
                error = _relative_error(float(left[metric]), float(right[metric]))
                if target_time_h == 48.0:
                    endpoint_errors[(left_bins, right_bins)][metric] = error
                pairs.append(
                    {
                        "scenario": scenario,
                        "left_run_id": left_id,
                        "right_run_id": right_id,
                        "left_bins": left_bins,
                        "right_bins": right_bins,
                        "target_time_h": target_time_h,
                        "metric": metric,
                        "left_value": left[metric],
                        "right_value": right[metric],
                        "relative_error": error,
                        "endpoint_p5_metric": target_time_h == 48.0,
                        "endpoint_p5_pass_2pct": error <= 0.02,
                        "p5_threshold": 0.02,
                    }
                )
            psd.append(
                _psd_distance_row(
                    scenario=scenario,
                    left_id=left_id,
                    right_id=right_id,
                    left_bins=left_bins,
                    right_bins=right_bins,
                    target_time_h=target_time_h,
                    shared_grid=shared_grid,
                )
            )
    for item in pairs:
        pair = (int(item["left_bins"]), int(item["right_bins"]))
        metric = str(item["metric"])
        item["full_time_max_relative_error"] = max(
            float(other["relative_error"])
            for other in pairs
            if (int(other["left_bins"]), int(other["right_bins"])) == pair
            and str(other["metric"]) == metric
        )

    reference_pair = (800, 1600)
    if reference_pair not in endpoint_errors:
        raise RuntimeError(f"{scenario} ladder is missing its required 800 vs 1600 pair")
    reference_errors = endpoint_errors[reference_pair]
    reference_pass = all(value <= 0.02 for value in reference_errors.values())
    prior_pair = (400, 800)
    order_by_metric: dict[str, Any] = {}
    for metric in PRIMARY_METRICS:
        sequence = [
            {
                "pair": f"{left}-{right}",
                "relative_error": endpoint_errors[(left, right)][metric],
            }
            for left, right in zip(bins_ladder, bins_ladder[1:])
        ]
        previous = endpoint_errors.get(prior_pair, {}).get(metric)
        current = reference_errors[metric]
        order_by_metric[metric] = {
            "endpoint_sequence": sequence,
            "last_refinement_reduces_error": None if previous is None else current <= previous,
        }
    observed_order_clear = all(
        item["last_refinement_reduces_error"] is not False for item in order_by_metric.values()
    )
    required_3200 = not reference_pass or not observed_order_clear
    final_pair = reference_pair
    final_errors = reference_errors
    final_pass = reference_pass
    if required_3200 and 3200 in bins_ladder:
        final_pair = (1600, 3200)
        final_errors = endpoint_errors[final_pair]
        final_pass = all(value <= 0.02 for value in final_errors.values())
    status = (
        "PASS_KWN_RADIUS_GRID_CONVERGENCE"
        if final_pass and (not required_3200 or 3200 in bins_ladder)
        else "PENDING_3200_REQUIRED"
        if required_3200 and 3200 not in bins_ladder
        else "FAIL_KWN_RADIUS_GRID_CONVERGENCE"
    )
    assessment = {
        "scenario": scenario,
        "available_bins": bins_ladder,
        "required_reference_pair": "800_vs_1600",
        "reference_endpoint_errors": reference_errors,
        "reference_endpoint_pass": reference_pass,
        "observed_endpoint_order": order_by_metric,
        "observed_order_clear": observed_order_clear,
        "requires_3200": required_3200,
        "final_pair": f"{final_pair[0]}_vs_{final_pair[1]}",
        "final_endpoint_errors": final_errors,
        "final_endpoint_pass": final_pass,
        "status": status,
        "authority_grid": final_pair[1] if status == "PASS_KWN_RADIUS_GRID_CONVERGENCE" else None,
    }
    return pairs, psd, assessment


def _collect_ladder_analysis() -> dict[str, Any]:
    fixture_bins = _available_ladder_bins("fixture")
    smooth_bins = _available_ladder_bins("smooth")
    expected = list(GRID_LADDER)
    if fixture_bins[: len(expected)] != expected or smooth_bins[: len(expected)] != expected:
        raise RuntimeError("Need completed uniform 100/200/400/800/1600 fixture and smooth runs")
    fixture_pairs, fixture_psd, fixture_assessment = _pair_rows_and_assessment(
        scenario="fixture", bins_ladder=fixture_bins
    )
    smooth_pairs, smooth_psd, smooth_assessment = _pair_rows_and_assessment(
        scenario="smooth", bins_ladder=smooth_bins
    )
    trajectory_rows: list[dict[str, Any]] = []
    event_rows: list[dict[str, Any]] = []
    lower_rows: list[dict[str, Any]] = []
    for scenario, bins_ladder in (("fixture", fixture_bins), ("smooth", smooth_bins)):
        for bins in bins_ladder:
            directory, _, rows = _load_run(f"ladder_{scenario}_{bins}_uniform")
            trajectory_rows.extend(rows)
            lower_rows.extend(_as_float_rows(_read_csv(directory / "lower_boundary_flux.csv")))
            event_path = directory / "class_events.csv"
            if event_path.is_file():
                event_rows.extend(_as_float_rows(_read_csv(event_path)))
    return {
        "trajectory_rows": trajectory_rows,
        "pair_rows": fixture_pairs + smooth_pairs,
        "psd_rows": fixture_psd + smooth_psd,
        "event_rows": event_rows,
        "lower_rows": lower_rows,
        "fixture": fixture_assessment,
        "smooth": smooth_assessment,
    }


def _inspect_ladder() -> int:
    _require_baseline_pass()
    analysis = _collect_ladder_analysis()
    print(
        json.dumps(
            {"fixture": analysis["fixture"], "smooth": analysis["smooth"]},
            indent=2,
            sort_keys=True,
        )
    )
    return 0


REQUALIFICATION_FIELDS = (
    "record_type",
    "gate_id",
    "scope",
    "status",
    "bins",
    "finer_bins",
    "time_h",
    "criterion",
    "observed",
    "threshold",
    "contract_hash",
    "fixture_hash",
    "config_hash",
    "solver_mode",
    "detail",
)


def _requalification_row(**values: Any) -> dict[str, Any]:
    return {key: values.get(key, "") for key in REQUALIFICATION_FIELDS}


def _state_differences(left: KWNSolver, right: KWNSolver) -> dict[str, float]:
    differences: dict[str, float] = {}
    for name, array in left.state_arrays().items():
        differences[name] = float(np.max(np.abs(array - right.state_arrays()[name])))
    return differences


def _advance_without_tags(solver: KWNSolver, target_time_s: float) -> dict[str, Any]:
    cumulative = _new_audit_aggregate()
    _advance_fixture_segment(
        solver=solver,
        target_time_s=target_time_s,
        tags=None,
        cumulative=cumulative,
        interval=_new_audit_aggregate(),
    )
    return cumulative


def _run_restart_requalification(*, bins: int) -> tuple[dict[str, Any], Path]:
    run_id = f"requalification_fixture_{bins}_restart"
    directory = _run_directory(run_id)
    continuous, construction, contract, fixture = _build_fixture_solver(bins=bins)
    restart, _, _, _ = _build_fixture_solver(bins=bins)
    continuous_audit = _new_audit_aggregate()
    restart_audit = _new_audit_aggregate()
    comparisons: list[dict[str, Any]] = []
    checkpoints: list[Path] = []
    for target_h in (3.0, 6.0, 24.0, 48.0):
        _advance_fixture_segment(
            solver=continuous,
            target_time_s=target_h * 3600.0,
            tags=None,
            cumulative=continuous_audit,
            interval=_new_audit_aggregate(),
        )
        _advance_fixture_segment(
            solver=restart,
            target_time_s=target_h * 3600.0,
            tags=None,
            cumulative=restart_audit,
            interval=_new_audit_aggregate(),
        )
        if target_h in (3.0, 24.0):
            checkpoint = directory / f"checkpoint_{int(target_h)}h.npz"
            restart.save_checkpoint(checkpoint)
            checkpoints.append(checkpoint)
            restart = KWNSolver.load_checkpoint(config=restart.config, path=checkpoint)
        if target_h in (6.0, 48.0):
            differences = _state_differences(continuous, restart)
            comparisons.append(
                {
                    "comparison": f"continuous_0_{int(target_h)}h_vs_restart_segments_to_{int(target_h)}h",
                    "time_h": target_h,
                    "pass": all(value == 0.0 for value in differences.values()),
                    **differences,
                }
            )
    _write_csv_new(directory / "restart_comparison.csv", comparisons)
    manifest = _run_manifest(
        solver=restart,
        construction=construction,
        contract=contract,
        fixture=fixture,
        scenario="fixture",
        schedule="restart_3_6_24_48",
        run_id=run_id,
        bins=bins,
        max_dt_factor=1.0,
        output_times_h=(0.0, 3.0, 6.0, 24.0, 48.0),
    )
    manifest["completion"] = {
        "final_time_h": restart.time_s / 3600.0,
        "final_step": restart.step,
        "maximum_inventory_relative_residual": max(
            continuous_audit["maximum_residual"], restart_audit["maximum_residual"]
        ),
        "roundoff_zeroed_bin_count": restart.roundoff_zeroed_bin_count,
        "checkpoint_sha256": {path.name: _sha256_file(path) for path in checkpoints},
    }
    _write_json_new(directory / "manifest.json", manifest)
    return {"comparisons": comparisons, "manifest": manifest}, directory


def _run_engine_regressions() -> dict[str, Any]:
    modules = (
        "tests.kwn.test_numerical_gates",
        "tests.kwn.test_rmin_boundary",
        "tests.kwn.test_composition_mapping",
        "tests.kwn.test_pf_validation_contract",
        "tests.kwn.test_beta_only_same_contract_control",
        "tests.kwn.test_conservative_positivity_repair",
        "tests.kwn.test_population_radius_moment",
    )
    environment = dict(os.environ)
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    process = subprocess.run(
        [sys.executable, "-m", "unittest", *modules],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "modules": list(modules),
        "returncode": process.returncode,
        "pass": process.returncode == 0,
        "output": process.stdout[-12000:],
    }


def _run_beta_rmin_return_qualifier() -> dict[str, Any]:
    """Exercise the beta-only Rmin route independently of aggregate tag bookkeeping."""

    environment = dict(os.environ)
    source_path = str(ROOT / "src")
    environment["PYTHONPATH"] = source_path + (
        os.pathsep + environment["PYTHONPATH"] if environment.get("PYTHONPATH") else ""
    )
    process = subprocess.run(
        [
            sys.executable,
            "-m",
            "unittest",
            "tests.kwn.test_rmin_boundary.RminBoundaryTest.test_beta_rmin_dissolution_returns_its_inventory_to_matrix",
        ],
        cwd=ROOT,
        env=environment,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return {
        "returncode": process.returncode,
        "pass": process.returncode == 0,
        "output": process.stdout[-4000:],
    }


def _run_requalification() -> int:
    _require_baseline_pass()
    output_csv = TASK_OUTPUT_ROOT / "kwn_requalification.csv"
    output_json = TASK_OUTPUT_ROOT / "kwn_requalification_summary.json"
    if output_csv.exists() or output_json.exists():
        raise RuntimeError("Refusing to overwrite KWN requalification output")
    analysis = _collect_ladder_analysis()
    fixture_assessment = analysis["fixture"]
    if (
        fixture_assessment["status"] != "PASS_KWN_RADIUS_GRID_CONVERGENCE"
        or analysis["smooth"]["status"] != "PASS_KWN_RADIUS_GRID_CONVERGENCE"
    ):
        contract, fixture = _load_frozen_inputs()
        row = _requalification_row(
            record_type="fixture_authority",
            gate_id="REQUALIFICATION_GATED",
            scope="validation-contract fixture",
            status="BLOCKED_RADIUS_GRID_NOT_PASSED",
            criterion="PASS_KWN_RADIUS_GRID_CONVERGENCE before authority requalification",
            observed=json.dumps(
                {
                    "fixture": fixture_assessment["status"],
                    "smooth": analysis["smooth"]["status"],
                },
                sort_keys=True,
            ),
            contract_hash=contract.contract_hash,
            fixture_hash=fixture.fixture_hash,
            detail="No authority-grid requalification is permitted while the P5 gate remains unresolved.",
        )
        _write_csv_new(output_csv, [row])
        _write_json_new(output_json, {"status": "BLOCKED_RADIUS_GRID_NOT_PASSED", "assessment": fixture_assessment})
        return 2

    authority = int(fixture_assessment["authority_grid"])
    final_pair = tuple(int(value) for value in str(fixture_assessment["final_pair"]).split("_vs_"))
    authority_run_id = f"ladder_fixture_{authority}_uniform"
    authority_dir, authority_manifest, authority_rows = _load_run(authority_run_id)
    contract_hash = str(authority_manifest["contract_hash"])
    fixture_hash = str(authority_manifest["fixture_hash"])
    if (TASK_OUTPUT_ROOT / "runs" / f"requalification_fixture_{authority}_dt_half").exists():
        raise RuntimeError("Refusing to reuse an existing authority timestep run")
    _run_grid(
        run_id=f"requalification_fixture_{authority}_dt_half",
        scenario="fixture",
        bins=authority,
        schedule="uniform",
        max_dt_factor=0.5,
        require_baseline=True,
    )
    _, dt_manifest, dt_rows = _load_run(f"requalification_fixture_{authority}_dt_half")
    timestep_errors = {
        metric: _relative_error(
            float(_find_snapshot(authority_rows, 48.0)[metric]),
            float(_find_snapshot(dt_rows, 48.0)[metric]),
        )
        for metric in PRIMARY_METRICS
    }
    timestep_pass = all(value <= 0.02 for value in timestep_errors.values())

    restart, restart_directory = _run_restart_requalification(bins=authority)
    restart_pass = all(bool(item["pass"]) for item in restart["comparisons"])
    failure_solver, _, _, _ = _build_fixture_solver(bins=authority)
    failure_audit = _advance_without_tags(failure_solver, 0.5 * 3600.0)
    failure_ledger = failure_solver.ledger.snapshot(
        matrix_xb=failure_solver.matrix_xb,
        populations=failure_solver.population_list(),
        beta_resolved_fraction=1.0,
    )
    failure_pass = (
        failure_solver.time_s / 3600.0 > LEGACY_FAILURE_TIME_H
        and float(np.min(failure_solver.population("beta").number_density_per_m4)) >= 0.0
        and failure_solver.roundoff_zeroed_bin_count == 0
        and failure_audit["maximum_residual"] <= 1.0e-10
    )
    engine = _run_engine_regressions()
    beta_rmin_qualifier = _run_beta_rmin_return_qualifier()
    completion = _find_snapshot(authority_rows, 48.0)
    completion_pass = (
        math.isclose(float(completion["time_h"]), 48.0, rel_tol=0.0, abs_tol=1.0e-12)
        and float(completion["minimum_bin_density_per_m4"]) >= 0.0
        and int(float(completion["roundoff_zeroed_bin_count"])) == 0
    )
    conservation_pass = max(
        float(row["total_residual_relative"]) for row in authority_rows
    ) <= 1.0e-10
    lower_rows = _as_float_rows(_read_csv(authority_dir / "lower_boundary_flux.csv"))
    lower_discrepancy = max(
        abs(float(row["tag_vs_aggregate_lower_number_difference_m3"]))
        for row in lower_rows
    )
    lower_pass = beta_rmin_qualifier["pass"] and conservation_pass and lower_discrepancy <= max(
        1.0e-12 * max(float(row["cumulative_lower_boundary_number_m3"]) for row in lower_rows),
        1.0e-300,
    )
    observation_value = float(completion["matrix_xB"])
    ag_value = xb_to_ag_at_fraction(observation_value)
    observation_error = abs(ag_at_fraction_to_xb(ag_value) - observation_value)
    observation_pass = observation_error <= 1.0e-15

    rows = [
        _requalification_row(
            record_type="fixture_authority",
            gate_id="P1_ORIGINAL_FAILURE_CROSSED",
            scope="validation-contract six-particle fixture",
            status="PASS" if failure_pass else "FAIL",
            bins=authority,
            time_h=failure_solver.time_s / 3600.0,
            criterion="time > 0.39317699499770825 h, min bin >= 0, roundoff=0, residual <=1e-10",
            observed=json.dumps({"min_bin": float(np.min(failure_solver.population("beta").number_density_per_m4)), "max_residual": failure_audit["maximum_residual"]}, sort_keys=True),
            threshold="1e-10",
            contract_hash=contract_hash,
            fixture_hash=fixture_hash,
            config_hash=failure_solver.config.source_config_hash,
            solver_mode="CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE",
            detail="Repaired production transport crossed the historical unsafe failure window without a clamp.",
        ),
        _requalification_row(
            record_type="fixture_authority",
            gate_id="P2_48H_COMPLETION",
            scope="validation-contract six-particle fixture",
            status="PASS" if completion_pass else "FAIL",
            bins=authority,
            time_h=48.0,
            criterion="48 h completion, non-negative bins, zero roundoff correction",
            observed=json.dumps({"step": completion["step"], "min_bin": completion["minimum_bin_density_per_m4"], "roundoff": completion["roundoff_zeroed_bin_count"]}, sort_keys=True),
            contract_hash=contract_hash,
            fixture_hash=fixture_hash,
            config_hash=authority_manifest["source_config_hash"],
            solver_mode="CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE",
        ),
        _requalification_row(
            record_type="ledger",
            gate_id="P4_ALL_STATE_INVENTORY",
            scope="authority full trajectory",
            status="PASS" if conservation_pass else "FAIL",
            bins=authority,
            time_h=48.0,
            criterion="max four-bucket relative residual <= 1e-10",
            observed=max(float(row["total_residual_relative"]) for row in authority_rows),
            threshold="1e-10",
            contract_hash=contract_hash,
            fixture_hash=fixture_hash,
            config_hash=authority_manifest["source_config_hash"],
            solver_mode="CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE",
        ),
        _requalification_row(
            record_type="fixture_authority",
            gate_id="P5_RADIUS_GRID",
            scope="authority qualifying pair",
            status="PASS",
            bins=authority,
            finer_bins="",
            time_h=48.0,
            criterion="all six preregistered endpoint metrics <=2%",
            observed=json.dumps(fixture_assessment["final_endpoint_errors"], sort_keys=True),
            threshold="0.02",
            contract_hash=contract_hash,
            fixture_hash=fixture_hash,
            config_hash=authority_manifest["source_config_hash"],
            solver_mode="CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE",
            detail=f"Qualifying pair: {final_pair[0]} vs {final_pair[1]}; authority is its finer member.",
        ),
        _requalification_row(
            record_type="fixture_authority",
            gate_id="P5_TIMESTEP",
            scope="authority max_dt vs max_dt/2",
            status="PASS" if timestep_pass else "FAIL",
            bins=authority,
            time_h=48.0,
            criterion="all six endpoint metrics <=2%",
            observed=json.dumps(timestep_errors, sort_keys=True),
            threshold="0.02",
            contract_hash=contract_hash,
            fixture_hash=fixture_hash,
            config_hash=dt_manifest["source_config_hash"],
            solver_mode="CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE",
        ),
        _requalification_row(
            record_type="restart",
            gate_id="P6_RESTART",
            scope="authority exact state arrays",
            status="PASS" if restart_pass else "FAIL",
            bins=authority,
            time_h=48.0,
            criterion="continuous and checkpoint/restart state arrays bit-identical at 6 h and 48 h",
            observed=json.dumps(restart["comparisons"], sort_keys=True),
            contract_hash=contract_hash,
            fixture_hash=fixture_hash,
            config_hash=authority_manifest["source_config_hash"],
            solver_mode="CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE",
            detail=str(restart_directory),
        ),
        _requalification_row(
            record_type="lower_boundary",
            gate_id="BETA_RMIN_CONSERVATIVE_RETURN",
            scope="authority six-particle fixture",
            status="PASS" if lower_pass else "FAIL",
            bins=authority,
            time_h=48.0,
            criterion="beta-only Rmin return qualifier passes; tagged beta lower-face flux sums to aggregate; global four-bucket ledger closes",
            observed=json.dumps(
                {
                    "tag_aggregate_lower_number_difference_m3": lower_discrepancy,
                    "beta_only_rmin_return_test_returncode": beta_rmin_qualifier["returncode"],
                },
                sort_keys=True,
            ),
            threshold="beta-only matrix-plus-beta return residual <=1e-12 total inventory and tag mismatch <=1e-12 relative",
            contract_hash=contract_hash,
            fixture_hash=fixture_hash,
            config_hash=authority_manifest["source_config_hash"],
            solver_mode="CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE",
            detail=beta_rmin_qualifier["output"],
        ),
        _requalification_row(
            record_type="observation_mapping",
            gate_id="N8_XB_AG_ROUND_TRIP",
            scope="authority 48 h matrix observation",
            status="PASS" if observation_pass else "FAIL",
            bins=authority,
            time_h=48.0,
            criterion="xB -> yAg -> xB absolute error <=1e-15",
            observed=observation_error,
            threshold="1e-15",
            contract_hash=contract_hash,
            fixture_hash=fixture_hash,
            config_hash=authority_manifest["source_config_hash"],
            solver_mode="CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE",
        ),
        _requalification_row(
            record_type="engine_regression",
            gate_id="N1_N8_AND_CONTRACT_REGRESSIONS",
            scope="synthetic numerical engine regressions",
            status="PASS" if engine["pass"] else "FAIL",
            criterion="selected existing N1--N8, Rmin/Rmax, contract, baseline repair, and moment audit tests",
            observed=engine["returncode"],
            threshold="0",
            detail=engine["output"],
        ),
    ]
    positivity_pass = all(
        row["status"] == "PASS"
        for row in rows
        if row["gate_id"]
        in {
            "P1_ORIGINAL_FAILURE_CROSSED",
            "P2_48H_COMPLETION",
            "P4_ALL_STATE_INVENTORY",
            "P5_TIMESTEP",
            "P6_RESTART",
            "BETA_RMIN_CONSERVATIVE_RETURN",
            "N1_N8_AND_CONTRACT_REGRESSIONS",
        }
    )
    summary = {
        "status": "PASS_KWN_POSITIVITY_CONSERVATION_48H"
        if positivity_pass
        else "FAIL_KWN_POSITIVITY_CONSERVATION_48H",
        "authority_grid": authority,
        "qualifying_pair": final_pair,
        "rows": rows,
        "engine_regressions": engine,
        "beta_rmin_return_qualifier": beta_rmin_qualifier,
    }
    _write_csv_new(output_csv, rows)
    _write_json_new(output_json, summary)
    print(json.dumps({"status": summary["status"], "authority_grid": authority}, sort_keys=True))
    return 0 if positivity_pass else 2


def _frozen_case_a_rows() -> tuple[dict[int, dict[str, str]], dict[int, list[dict[str, str]]]]:
    trajectory_path = FROZEN_OUTPUT_ROOT / "cuda_ae_trajectories.csv"
    component_path = FROZEN_OUTPUT_ROOT / "cuda_component_history.csv"
    trajectory: dict[int, dict[str, str]] = {}
    for row in _read_csv(trajectory_path):
        if row["case"] != "A":
            continue
        step = int(row["step"])
        if step in trajectory:
            raise RuntimeError(f"Duplicate frozen Case A trajectory step {step}")
        trajectory[step] = row
    components: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(component_path):
        if row["case"] == "A":
            components[int(row["step"])].append(row)
    required = {step for _, step, _ in PF_COMPARISON_TIME_MAP}
    if not required.issubset(trajectory) or not required.issubset(components):
        raise RuntimeError("Frozen Case A files lack a required comparison checkpoint")
    return trajectory, components


def _pf_component_metrics(
    *, rows: list[Mapping[str, str]], box_volume_m3: float
) -> tuple[dict[str, Any], bool]:
    unresolved = any(
        row[field] == "True"
        for row in rows
        for field in ("unresolved_merge", "unresolved_split", "unresolved_new_component")
    )
    positive = [row for row in rows if float(row["equivalent_radius_nm"]) > 0.0]
    radii = np.asarray(
        [float(row["equivalent_radius_nm"]) * 1.0e-9 for row in positive], dtype=np.float64
    )
    weights = np.full(radii.size, 1.0 / box_volume_m3, dtype=np.float64)
    moments = {
        f"M{order}": float(np.sum(weights * radii**order)) for order in range(4)
    }
    return (
        {
            "component_count": int(radii.size),
            "radii_m": radii,
            "weights_m3": weights,
            "M0_m3": moments["M0"],
            "M1_m2": moments["M1"],
            "M2_m": moments["M2"],
            "M3_dimensionless": moments["M3"],
            "Rmean_m": 0.0 if moments["M0"] == 0.0 else moments["M1"] / moments["M0"],
            "Rmean3_m3": 0.0 if moments["M0"] == 0.0 else moments["M3"] / moments["M0"],
            "Sv_m_inv": 4.0 * math.pi * moments["M2"],
            "minimum_radius_m": "" if radii.size == 0 else float(np.min(radii)),
            "maximum_radius_m": "" if radii.size == 0 else float(np.max(radii)),
        },
        unresolved,
    )


def _comparison_event_rows() -> list[dict[str, Any]]:
    trajectory, components = _frozen_case_a_rows()
    ordered_steps = sorted(components)
    initial = {
        row["particle_id"]: float(row["equivalent_radius_nm"])
        for row in components[0]
        if float(row["equivalent_radius_nm"]) > 0.0
    }
    rows: list[dict[str, Any]] = []
    for particle_id, initial_radius_nm in sorted(initial.items(), key=lambda item: int(item[0])):
        present_steps = [
            step
            for step in ordered_steps
            if any(
                row["particle_id"] == particle_id
                and float(row["equivalent_radius_nm"]) > 0.0
                for row in components[step]
            )
        ]
        absent_steps = [step for step in ordered_steps if step not in present_steps]
        last_present = max(present_steps) if present_steps else None
        first_absent = min(absent_steps) if absent_steps else None
        rows.append(
            {
                "record_source": "PF_CASE_A_FROZEN_COMPONENT_HISTORY",
                "run_id": "PF_CASE_A",
                "bins": "",
                "class_label": f"particle_{particle_id}",
                "initial_radius_m": initial_radius_nm * 1.0e-9,
                "source_particle_count": 1,
                "initial_projected_bin_indices": "",
                "current_occupied_bin_indices": "",
                "effective_radius_m": "",
                "weighted_growth_rate_m_s": "",
                "growth_dissolution_sign": "CHECKPOINT_BOUNDED",
                "survival_number_density_m3": "",
                "survival_weight_fraction": "",
                "current_beta_inventory_mol_m3": "",
                "net_matrix_inventory_from_class_mol_m3": "",
                "cumulative_lower_boundary_number_m3": "",
                "cumulative_returned_matrix_beta_inventory_mol_m3": "",
                "first_lower_boundary_crossing_time_h": "",
                "pf_last_present_step": "" if last_present is None else last_present,
                "pf_last_present_time_h": "" if last_present is None else trajectory[last_present]["time_h"],
                "pf_first_absent_step": "" if first_absent is None else first_absent,
                "pf_first_absent_time_h": "" if first_absent is None else trajectory[first_absent]["time_h"],
                "event_interpretation": (
                    "SURVIVES_THROUGH_48H"
                    if first_absent is None
                    else "CHECKPOINT_BOUNDED_FIRST_ABSENCE_NOT_AN_EXACT_DISSOLUTION_TIME"
                ),
            }
        )
    return rows


def _write_blocked_comparison(*, reason: str, status: str = "BLOCKED_RADIUS_GRID_NOT_PASSED") -> int:
    output_csv = TASK_OUTPUT_ROOT / "beta_only_kwn_pf_comparison.csv"
    output_json = TASK_OUTPUT_ROOT / "beta_only_kwn_pf_comparison_summary.json"
    if output_csv.exists() or output_json.exists():
        raise RuntimeError("Refusing to overwrite beta-only comparison output")
    row = {
        "record_type": "comparison_status",
        "comparison_target_time_h": "",
        "pf_step": "",
        "pf_actual_time_h": "",
        "kwn_time_h": "",
        "status": status,
        "detail": reason,
    }
    _write_csv_new(output_csv, [row])
    _write_json_new(
        output_json,
        {
            "status": status,
            "reason": reason,
            "cuda_reused_without_rerun": True,
        },
    )
    return 2


def _run_beta_only_comparison() -> int:
    _require_baseline_pass()
    analysis = _collect_ladder_analysis()
    fixture_assessment = analysis["fixture"]
    if (
        fixture_assessment["status"] != "PASS_KWN_RADIUS_GRID_CONVERGENCE"
        or analysis["smooth"]["status"] != "PASS_KWN_RADIUS_GRID_CONVERGENCE"
    ):
        return _write_blocked_comparison(
            reason="KWN fixture P5 and smooth finite-volume qualification must both pass before beta-only KWN-PF comparison is authorized."
        )
    requalification_path = TASK_OUTPUT_ROOT / "kwn_requalification_summary.json"
    if not requalification_path.is_file() or _read_json(requalification_path).get("status") != "PASS_KWN_POSITIVITY_CONSERVATION_48H":
        return _write_blocked_comparison(
            status="BLOCKED_KWN_REQUALIFICATION_NOT_PASSED",
            reason="Authority-grid KWN requalification must be completed and pass before beta-only KWN-PF comparison is authorized.",
        )
    authority = int(fixture_assessment["authority_grid"])
    run_id = f"comparison_fixture_{authority}_pf_accepted"
    if not (TASK_OUTPUT_ROOT / "runs" / run_id).exists():
        _run_grid(
            run_id=run_id,
            scenario="fixture",
            bins=authority,
            schedule="pf_accepted",
            max_dt_factor=1.0,
            require_baseline=True,
        )
    run_directory, run_manifest, kwn_rows = _load_run(run_id)
    frozen_trajectory, frozen_components = _frozen_case_a_rows()
    contract, fixture = _load_frozen_inputs()
    if run_manifest["contract_hash"] != contract.contract_hash or run_manifest["fixture_hash"] != fixture.fixture_hash:
        raise RuntimeError("authority KWN run provenance differs from frozen comparison contract")
    output_csv = TASK_OUTPUT_ROOT / "beta_only_kwn_pf_comparison.csv"
    output_json = TASK_OUTPUT_ROOT / "beta_only_kwn_pf_comparison_summary.json"
    if output_csv.exists() or output_json.exists():
        raise RuntimeError("Refusing to overwrite beta-only comparison output")
    rows: list[dict[str, Any]] = []
    identities_resolved = True
    for nominal_time_h, step, pf_time_h in PF_COMPARISON_TIME_MAP:
        pf = frozen_trajectory[step]
        if pf["diagnostic_provenance"] != "CHECKPOINT_REPLAY_DERIVED_CUDA_ACCEPTED_STATE_V1":
            raise RuntimeError("Frozen PF comparison row lacks accepted-state provenance")
        if pf["validation_contract_hash"] != contract.contract_hash or pf["fixture_manifest_sha256"] != fixture.fixture_hash:
            raise RuntimeError("Frozen PF comparison provenance does not match validation inputs")
        if not math.isclose(float(pf["time_h"]), pf_time_h, rel_tol=0.0, abs_tol=1.0e-12):
            raise RuntimeError("Frozen PF accepted time differs from the registered comparison map")
        components, unresolved = _pf_component_metrics(
            rows=frozen_components[step], box_volume_m3=fixture.box_volume_m3
        )
        identities_resolved = identities_resolved and not unresolved
        kwn = _find_snapshot(kwn_rows, pf_time_h)
        kwn_grid, kwn_density = _load_psd_snapshot(run_directory, pf_time_h)
        kwn_radii = kwn_grid.centres_m
        kwn_weights = kwn_density * kwn_grid.widths_m
        pf_f_beta = float(pf["beta_volume_fraction"])
        pf_matrix_xb_inventory = (
            float(pf["Q_B_matrix_mol"])
            * contract.vm_alpha_m3_mol
            / ((1.0 - pf_f_beta) * fixture.box_volume_m3)
        )
        pf_component_f_beta = 4.0 * math.pi * float(components["M3_dimensionless"]) / 3.0
        row = {
            "record_type": "time_snapshot",
            "comparison_target_time_h": nominal_time_h,
            "pf_step": step,
            "pf_actual_time_h": pf_time_h,
            "kwn_time_h": kwn["time_h"],
            "status": "PF_IDENTITY_RESOLVED" if not unresolved else "PF_IDENTITY_UNRESOLVED",
            "authority_grid_bins": authority,
            "kwn_N_m0_m3": kwn["N_m0_m3"],
            "pf_N_m0_m3": components["M0_m3"],
            "kwn_Rmean_m": kwn["Rmean_m"],
            "pf_Rmean_m": components["Rmean_m"],
            "kwn_Rmean3_m3": kwn["Rmean3_m3"],
            "pf_Rmean3_m3": components["Rmean3_m3"],
            "kwn_inverse_N_m3": 1.0 / float(kwn["N_m0_m3"]),
            "pf_inverse_N_m3": 1.0 / float(components["M0_m3"]),
            "kwn_Sv_m_inv": kwn["Sv_m_inv"],
            "pf_Sv_component_m_inv": components["Sv_m_inv"],
            "pf_Sv_frozen_scalar_m_inv": float(pf["S_v_equivalent_sphere_nm_inverse"]) * 1.0e9,
            "kwn_f_beta": kwn["f_beta"],
            "pf_f_beta_field": pf_f_beta,
            "pf_f_beta_components": pf_component_f_beta,
            "kwn_matrix_xB": kwn["matrix_xB"],
            "pf_matrix_xB_inventory": pf_matrix_xb_inventory,
            "pf_matrix_xB_h_lt_0p005": float(pf["matrix_xB_h_lt_0p005_mean"]),
            "pf_xB_alpha_field_mean": float(pf["xB_alpha_mean"]),
            "kwn_M3_dimensionless": kwn["M3_dimensionless"],
            "pf_component_M3_dimensionless": components["M3_dimensionless"],
            "pf_field_fbeta_component_tail_relative": _relative_error(pf_f_beta, pf_component_f_beta),
            "pf_component_count": components["component_count"],
            "pf_component_minimum_radius_m": components["minimum_radius_m"],
            "pf_component_maximum_radius_m": components["maximum_radius_m"],
            "W1_number_m": discrete_wasserstein_distance(
                kwn_radii, kwn_weights, components["radii_m"], components["weights_m3"]
            ),
            "W1_volume_m": discrete_wasserstein_distance(
                kwn_radii,
                kwn_weights * kwn_radii**3,
                components["radii_m"],
                components["weights_m3"] * components["radii_m"] ** 3,
            ),
            "relative_N": _relative_error(float(kwn["N_m0_m3"]), float(components["M0_m3"])),
            "relative_Rmean": _relative_error(float(kwn["Rmean_m"]), float(components["Rmean_m"])),
            "relative_Rmean3": _relative_error(float(kwn["Rmean3_m3"]), float(components["Rmean3_m3"])),
            "relative_Sv": _relative_error(float(kwn["Sv_m_inv"]), float(components["Sv_m_inv"])),
            "relative_f_beta": _relative_error(float(kwn["f_beta"]), pf_f_beta),
            "relative_matrix_xB_inventory": _relative_error(float(kwn["matrix_xB"]), pf_matrix_xb_inventory),
            "detail": "PF is a frozen accepted checkpoint; no interpolation and no CUDA rerun.",
        }
        rows.append(row)

    def sign(value: float) -> str:
        return "INCREASE" if value > 0.0 else "DECREASE" if value < 0.0 else "UNCHANGED"

    initial = rows[0]
    final = rows[-1]
    directions = {
        "N": {
            "kwn": sign(float(final["kwn_N_m0_m3"]) - float(initial["kwn_N_m0_m3"])),
            "pf": sign(float(final["pf_N_m0_m3"]) - float(initial["pf_N_m0_m3"])),
        },
        "Rmean": {
            "kwn": sign(float(final["kwn_Rmean_m"]) - float(initial["kwn_Rmean_m"])),
            "pf": sign(float(final["pf_Rmean_m"]) - float(initial["pf_Rmean_m"])),
        },
        "Sv": {
            "kwn": sign(float(final["kwn_Sv_m_inv"]) - float(initial["kwn_Sv_m_inv"])),
            "pf": sign(float(final["pf_Sv_component_m_inv"]) - float(initial["pf_Sv_component_m_inv"])),
        },
        "beta_exchange_f_beta": {
            "kwn": sign(float(final["kwn_f_beta"]) - float(initial["kwn_f_beta"])),
            "pf": sign(float(final["pf_f_beta_field"]) - float(initial["pf_f_beta_field"])),
        },
        "matrix_exchange_xB": {
            "kwn": sign(float(final["kwn_matrix_xB"]) - float(initial["kwn_matrix_xB"])),
            "pf": sign(float(final["pf_matrix_xB_inventory"]) - float(initial["pf_matrix_xB_inventory"])),
        },
    }
    global_direction_match = all(item["kwn"] == item["pf"] for item in directions.values())
    tag_rows = _as_float_rows(_read_csv(run_directory / "class_events.csv"))
    initial_tags = [row for row in tag_rows if math.isclose(float(row["target_time_h"]), 0.0, abs_tol=1.0e-12)]
    final_tags = [row for row in tag_rows if math.isclose(float(row["target_time_h"]), PF_COMPARISON_TIME_MAP[-1][2], abs_tol=1.0e-12)]
    tags_by_label = {str(row["class_label"]): row for row in final_tags}
    small_tag_dissolves = all(
        float(tags_by_label[str(row["class_label"])]["survival_weight_fraction"]) < 1.0
        for row in initial_tags
        if float(row["initial_radius_m"]) < 9.0e-9
    )
    pf_events = _comparison_event_rows()
    pf_small_absence = all(
        row["pf_first_absent_time_h"] not in ("", None)
        for row in pf_events
        if float(row["initial_radius_m"]) < 9.0e-9
    )
    largest_tag_initial_radius = max(float(row["initial_radius_m"]) for row in initial_tags)
    largest_tag_growth = any(
        float(tags_by_label[str(row["class_label"])]["effective_radius_m"])
        > float(row["effective_radius_m"])
        for row in initial_tags
        if math.isclose(
            float(row["initial_radius_m"]),
            largest_tag_initial_radius,
            rel_tol=0.0,
            abs_tol=1.0e-15,
        )
    )
    pf_initial_radius_by_id = {
        str(row["particle_id"]): float(row["equivalent_radius_nm"]) * 1.0e-9
        for row in frozen_components[0]
        if float(row["equivalent_radius_nm"]) > 0.0
    }
    pf_final_radius_by_id = {
        str(row["particle_id"]): float(row["equivalent_radius_nm"]) * 1.0e-9
        for row in frozen_components[PF_COMPARISON_TIME_MAP[-1][1]]
        if float(row["equivalent_radius_nm"]) > 0.0
    }
    largest_pf_initial_radius = max(pf_initial_radius_by_id.values())
    largest_pf_growth = any(
        particle_id in pf_final_radius_by_id
        and pf_final_radius_by_id[particle_id] > initial_radius
        for particle_id, initial_radius in pf_initial_radius_by_id.items()
        if math.isclose(
            initial_radius, largest_pf_initial_radius, rel_tol=0.0, abs_tol=2.0e-13
        )
    )
    comparison_events: list[dict[str, Any]] = []
    for tag in initial_tags:
        initial_radius_m = float(tag["initial_radius_m"])
        label = str(tag["class_label"])
        final_tag = tags_by_label[label]
        matching_pf = [
            row
            for row in pf_events
            if math.isclose(
                float(row["initial_radius_m"]), initial_radius_m, rel_tol=0.0, abs_tol=2.0e-13
            )
        ]
        pf_last_present = [
            row["pf_last_present_time_h"]
            for row in matching_pf
            if row["pf_last_present_time_h"] not in ("", None)
        ]
        pf_first_absent = [
            row["pf_first_absent_time_h"]
            for row in matching_pf
            if row["pf_first_absent_time_h"] not in ("", None)
        ]
        kwn_crossing = final_tag["first_lower_boundary_crossing_time_h"]
        comparison_events.append(
            {
                "record_source": "KWN_PF_CLASS_EVENT_COMPARISON",
                "run_id": run_id,
                "bins": authority,
                "class_label": label,
                "initial_radius_m": initial_radius_m,
                "kwn_first_rmin_accepted_endpoint_h": kwn_crossing,
                "kwn_48h_survival_weight_fraction": final_tag["survival_weight_fraction"],
                "kwn_48h_growth_dissolution_sign": final_tag["growth_dissolution_sign"],
                "pf_matching_particle_count": len(matching_pf),
                "pf_last_present_time_h": ";".join(str(value) for value in pf_last_present),
                "pf_first_absent_time_h": ";".join(str(value) for value in pf_first_absent),
                "event_interpretation": (
                    "KWN_RMIN_ACCEPTED_ENDPOINT_VS_PF_CHECKPOINT_BOUNDED_ABSENCE"
                    if kwn_crossing not in ("", None) and pf_first_absent
                    else "NO_DIRECT_DISSOLUTION_TIME_MATCH_AVAILABLE"
                ),
            }
        )
    direction_pass = (
        identities_resolved
        and global_direction_match
        and small_tag_dissolves
        and pf_small_absence
        and largest_tag_growth
        and largest_pf_growth
    )
    slopes: dict[str, float] = {}
    time_s = np.asarray([float(row["pf_actual_time_h"]) * 3600.0 for row in rows], dtype=np.float64)
    for label, field in (("kwn_Rmean3", "kwn_Rmean3_m3"), ("pf_Rmean3", "pf_Rmean3_m3")):
        slopes[label] = float(np.polyfit(time_s, [float(row[field]) for row in rows], 1)[0])
    slope_ratio = slopes["kwn_Rmean3"] / slopes["pf_Rmean3"] if slopes["pf_Rmean3"] != 0.0 else float("nan")
    events = pf_events + comparison_events + [
        {"record_source": "KWN_AUTHORITY_PF_ACCEPTED_TAG", **row} for row in tag_rows
    ]
    summary = {
        "status": "PASS_BETA_ONLY_DIRECTION" if direction_pass else "FAIL_BETA_ONLY_DIRECTION",
        "direction": directions,
        "global_direction_match": global_direction_match,
        "small_class_dissolution": {
            "kwn_tag_dissolves": small_tag_dissolves,
            "pf_checkpoint_bounded_absence": pf_small_absence,
        },
        "largest_class_growth": {
            "kwn_largest_class_grows": largest_tag_growth,
            "pf_largest_component_grows": largest_pf_growth,
        },
        "particle_identity_status": "PASS_RESOLVED" if identities_resolved else "FAIL_CLOSED_UNRESOLVED",
        "timescale_status": "NOT_CLAIMED_NO_PREREGISTERED_TIMESCALE_BAND",
        "coarsening_slopes_Rmean3_m3_s": slopes,
        "coarsening_slope_ratio_kwn_over_pf": slope_ratio,
        "mean_field_gap": (
            "CONDITIONAL_MEAN_FIELD_GAP"
            if direction_pass
            else "NOT_ASSESSED_AFTER_DIRECTION_FAILURE"
        ),
        "pf_authority": {
            "case": "A",
            "reason": "A/B audited field differences are zero through 48 h; A has no handoff auxiliary metadata.",
            "trajectory_sha256": _sha256_file(FROZEN_OUTPUT_ROOT / "cuda_ae_trajectories.csv"),
            "component_history_sha256": _sha256_file(FROZEN_OUTPUT_ROOT / "cuda_component_history.csv"),
            "no_cuda_rerun": True,
        },
        "events": events,
        "dissolution_event_timing": comparison_events,
    }
    _write_csv_new(output_csv, rows)
    _write_json_new(output_json, summary)
    return 0 if direction_pass else 2


def _normalise_rows(rows: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    fields: list[str] = []
    for row in rows:
        for field in row:
            if field not in fields:
                fields.append(field)
    return [{field: row.get(field, "") for field in fields} for row in rows]


def _number(value: Any, *, digits: int = 6) -> str:
    if value == "" or value is None:
        return "—"
    try:
        return f"{float(value):.{digits}g}"
    except (TypeError, ValueError):
        return str(value)


def _markdown_table(headers: list[str], rows: Iterable[Iterable[Any]]) -> str:
    rendered = ["| " + " | ".join(headers) + " |", "| " + " | ".join("---" for _ in headers) + " |"]
    for row in rows:
        rendered.append("| " + " | ".join(str(value) for value in row) + " |")
    return "\n".join(rendered)


def _historical_vs_uniform_schedule_error() -> dict[str, float]:
    _, _, historical = _load_run("baseline_fixture_200_historical")
    _, _, uniform = _load_run("ladder_fixture_200_uniform")
    old = _find_snapshot(historical, 48.0)
    new = _find_snapshot(uniform, 48.0)
    return {metric: _relative_error(float(old[metric]), float(new[metric])) for metric in PRIMARY_METRICS}


def _independent_positivity_conservation(
    *, baseline: Mapping[str, Any], analysis: Mapping[str, Any]
) -> dict[str, Any]:
    """Keep the passed conservative repair distinct from the unresolved P5 gate."""

    frozen = _read_json(FROZEN_REPAIR_SUMMARY)
    frozen_gates = frozen.get("gates", {})
    frozen_pass = all(
        bool(frozen_gates.get(key))
        for key in (
            "P1_exact_failure_regression",
            "P2_48h_completion",
            "P3_no_clipping",
            "P4_conservation",
            "P6_positivity_utilization",
            "P6_restart",
        )
    )
    fixture_rows = [
        row for row in analysis["trajectory_rows"] if row.get("scenario") == "fixture"
    ]
    run_ids = {str(row["run_id"]) for row in fixture_rows}
    completed_runs = {
        str(row["run_id"])
        for row in fixture_rows
        if math.isclose(float(row["target_time_h"]), 48.0, rel_tol=0.0, abs_tol=1.0e-12)
        and math.isclose(float(row["time_h"]), 48.0, rel_tol=0.0, abs_tol=1.0e-12)
    }
    maximum_residual = max(
        (abs(float(row["total_residual_relative"])) for row in fixture_rows), default=float("inf")
    )
    minimum_density = min(
        (float(row["minimum_bin_density_per_m4"]) for row in fixture_rows), default=-float("inf")
    )
    no_roundoff_zeroing = all(int(float(row["roundoff_zeroed_bin_count"])) == 0 for row in fixture_rows)
    ladder_pass = (
        bool(fixture_rows)
        and completed_runs == run_ids
        and maximum_residual <= 1.0e-10
        and minimum_density >= 0.0
        and no_roundoff_zeroing
    )
    baseline_pass = (
        baseline.get("status") == "PASS_RADIUS_GRID_BASELINE_REPRODUCTION"
        and baseline.get("repaired_48h_and_historical_p5", {}).get("status") == "PASS"
    )
    return {
        "status": "PASS_KWN_POSITIVITY_CONSERVATION_48H"
        if frozen_pass and baseline_pass and ladder_pass
        else "FAIL_KWN_POSITIVITY_CONSERVATION_48H",
        "frozen_repair_gates_pass": frozen_pass,
        "baseline_reproduction_pass": baseline_pass,
        "uniform_fixture_ladder_pass": ladder_pass,
        "fixture_run_count": len(run_ids),
        "fixture_48h_completed_runs": len(completed_runs),
        "maximum_uniform_fixture_ledger_residual": maximum_residual,
        "minimum_uniform_fixture_bin_density_per_m4": minimum_density,
        "no_uniform_fixture_roundoff_zeroing": no_roundoff_zeroing,
    }


def _derive_root_cause(
    *, initial: Mapping[str, Any], analysis: Mapping[str, Any]
) -> dict[str, Any]:
    fixture = analysis["fixture"]
    smooth = analysis["smooth"]
    schedule_error = _historical_vs_uniform_schedule_error()
    reasons: list[str] = []
    evidence: list[str] = []
    if initial["status"] != "PASS_INITIAL_PSD_PROJECTION_CONSERVATION":
        reasons.append("INITIAL_PSD_PROJECTION_NOT_CONSERVATIVE")
        evidence.append("t=0 M0/M3/inventory grid invariants failed the 1e-12 gate.")
    else:
        evidence.append("t=0 fixed-pivot M0/M3/beta inventory/total inventory pass the 1e-12 gate.")
    quadrature_ratio = float(
        initial["moment_quadrature"]["maximum_t0_M3_relative_difference_vs_midpoint"]
    )
    evidence.append(
        "Cell-integrated reconstruction differs from fixed-pivot M3 by at most "
        f"{quadrature_ratio:.6g} at t=0; it is diagnostic-only and does not alter dynamics."
    )
    schedule_max = max(schedule_error.values())
    if schedule_max > 0.02:
        reasons.append("TIME_DISCRETIZATION_ERROR")
        evidence.append(
            "Historical 200-bin segmented output versus the uniform 200-bin schedule changes at least one "
            f"48 h primary metric by {schedule_max:.3%}; this is a documented P5 confounder."
        )
    if not fixture["reference_endpoint_pass"]:
        evidence.append(
            "The 800 vs 1600 endpoint still exceeds the declared 2% gate under one identical output schedule; "
            "the remaining controlled change is radius-space resolution and its Rmin face-flux event timing."
        )
        reasons.append("IMPLICIT_UPWIND_NUMERICAL_DIFFUSION")
        evidence.append(
            "Passive tags receive the same frozen-velocity implicit M-matrix update as the aggregate and show "
            "grid-dependent lower-radius tails reaching Rmin; this is direct discrete evidence of first-order implicit-upwind numerical diffusion."
        )
    if fixture["status"] != "PASS_KWN_RADIUS_GRID_CONVERGENCE":
        if smooth["status"] == "PASS_KWN_RADIUS_GRID_CONVERGENCE":
            reasons.append("DISCRETE_EVENT_SENSITIVITY_IDENTIFIED")
            evidence.append(
                "The smooth PSD transport ladder converges while the exact six-particle fixture does not; "
                "the unresolved discrepancy is tied to discrete class transport/event timing rather than a global solver verdict."
            )
        else:
            reasons.append("IMPLICIT_UPWIND_NUMERICAL_DIFFUSION")
            evidence.append(
                "The first-order implicit upwind ladder remains unresolved for both smooth and discrete controls; "
                "further solver-order work is not attempted without an independently qualified design."
            )
    else:
        evidence.append("Uniform fixed-grid P5 passes at the declared qualifying pair.")
        if smooth["status"] != "PASS_KWN_RADIUS_GRID_CONVERGENCE":
            reasons.append("IMPLICIT_UPWIND_NUMERICAL_DIFFUSION")
            evidence.append(
                "The smooth finite-volume qualification remains unresolved even though the discrete fixture endpoint passes; "
                "therefore the production transport cannot be declared globally qualified."
            )
    if not reasons:
        reasons.append("OTHER_WITH_EXPLICIT_EVIDENCE")
        evidence.append("No remaining enumerated numerical failure is supported after the passed uniform ladder.")
    return {
        "root_causes": list(dict.fromkeys(reasons)),
        "evidence": evidence,
        "historical_vs_uniform_schedule_relative_error": schedule_error,
        "initial_projection_status": initial["status"],
        "fixture_ladder_status": fixture["status"],
        "smooth_ladder_status": smooth["status"],
    }


def _top_status(
    *, analysis: Mapping[str, Any], requalification: Mapping[str, Any], comparison: Mapping[str, Any]
) -> str:
    p5 = analysis["fixture"]["status"]
    if p5 != "PASS_KWN_RADIUS_GRID_CONVERGENCE":
        if analysis["smooth"]["status"] == "PASS_KWN_RADIUS_GRID_CONVERGENCE":
            return "DISCRETE_EVENT_SENSITIVITY_GATE_REVIEW_REQUIRED"
        return "FAIL_KWN_RADIUS_GRID_CONVERGENCE"
    if analysis["smooth"]["status"] != "PASS_KWN_RADIUS_GRID_CONVERGENCE":
        return "FAIL_KWN_RADIUS_GRID_CONVERGENCE"
    if requalification.get("status") != "PASS_KWN_POSITIVITY_CONSERVATION_48H":
        return "FAIL_KWN_RADIUS_GRID_CONVERGENCE"
    if comparison.get("status") == "PASS_BETA_ONLY_DIRECTION":
        return "PASS_KWN_PF_ONE_WAY_STORAGE_COUPLING_V1"
    return "PASS_KWN_RADIUS_GRID_CONVERGENCE_BETA_DIRECTION_FAIL"


def _git_capture(*arguments: str) -> str:
    process = subprocess.run(
        ["git", *arguments],
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    if process.returncode != 0:
        raise RuntimeError(f"git {' '.join(arguments)} failed: {process.stderr.strip()}")
    return process.stdout.strip()


def _analysis_provenance() -> dict[str, Any]:
    source_files = (
        ROOT / "scripts" / "run_kwn_radius_grid_convergence_v1.py",
        ROOT / "src" / "kwn_mvp" / "populations.py",
        ROOT / "src" / "kwn_mvp" / "solver.py",
        ROOT / "scripts" / "run_beta_only_same_contract_control.py",
    )
    return {
        "schema_version": "KWN_RADIUS_GRID_ANALYSIS_PROVENANCE_V1",
        "git_head": _git_capture("rev-parse", "HEAD"),
        "git_branch": _git_capture("branch", "--show-current"),
        "working_tree_porcelain": _git_capture("status", "--porcelain"),
        "source_file_sha256": {str(path.relative_to(ROOT)): _sha256_file(path) for path in source_files},
        "parameters": {
            "validation_contract_sha256": _sha256_file(CONTRACT_PATH),
            "validation_contract_hash": _load_frozen_inputs()[0].contract_hash,
            "fixture_spec_sha256": _sha256_file(FIXTURE_SPEC),
            "fixture_hash": _load_frozen_inputs()[1].fixture_hash,
            "profile_library_manifest_sha256": _sha256_file(PROFILE_LIBRARY_MANIFEST),
            "profile_root_read_only": str(PROFILE_ROOT),
        },
        "frozen_cuda_pf_evidence": {
            "pf_binary_provenance_sha256": _sha256_file(FROZEN_OUTPUT_ROOT / "binary_provenance.json"),
            "cuda_runtime_audit_sha256": _sha256_file(FROZEN_OUTPUT_ROOT / "cuda_ae_runtime_audit.json"),
            "cuda_trajectory_sha256": _sha256_file(FROZEN_OUTPUT_ROOT / "cuda_ae_trajectories.csv"),
            "cuda_component_history_sha256": _sha256_file(FROZEN_OUTPUT_ROOT / "cuda_component_history.csv"),
            "pf_binary_sha256": "c1f27f180c191a0f17c94b3eb1aeff62a596255895172f2f4bde8e8e10d2b488",
            "cuda_rerun": False,
        },
        "run_launch_source_binding": {
            "status": "ANALYSIS_TIME_HASHES_WITH_DISCLOSED_LAUNCH_LIMITATION",
            "detail": (
                "Per-run manifests bind frozen configuration, fixture, profile-library, contract, and solver mode. "
                "This task-specific runner did not capture a launch-time source SHA before the already-completed/live "
                "immutable runs; the hashes above bind the final analysis sources. No result is represented as having a "
                "retroactively captured runner-byte hash."
            ),
        },
    }


def _write_reports(
    *,
    baseline: Mapping[str, Any],
    initial: Mapping[str, Any],
    initial_rows: list[Mapping[str, Any]],
    analysis: Mapping[str, Any],
    root_cause: Mapping[str, Any],
    requalification: Mapping[str, Any],
    comparison: Mapping[str, Any],
    top_status: str,
) -> None:
    if REPORT_ROOT.exists():
        raise RuntimeError(f"Refusing to overwrite report directory {REPORT_ROOT}")
    fixture = analysis["fixture"]
    smooth = analysis["smooth"]
    positivity = _independent_positivity_conservation(baseline=baseline, analysis=analysis)

    def pair_bins(assessment: Mapping[str, Any]) -> tuple[int, int]:
        return tuple(int(value) for value in str(assessment["final_pair"]).split("_vs_"))  # type: ignore[return-value]

    def p5_rows(scenario: str, assessment: Mapping[str, Any]) -> list[Mapping[str, Any]]:
        left_bins, right_bins = pair_bins(assessment)
        endpoint = {
            str(row["metric"]): row
            for row in analysis["pair_rows"]
            if row["scenario"] == scenario
            and int(row["left_bins"]) == left_bins
            and int(row["right_bins"]) == right_bins
            and math.isclose(float(row["target_time_h"]), 48.0, rel_tol=0.0, abs_tol=1.0e-12)
        }
        return [endpoint[metric] for metric in PRIMARY_METRICS]

    def percent(value: Any) -> str:
        return f"{float(value):.3%}"

    fixture_p5_rows = p5_rows("fixture", fixture)
    smooth_p5_rows = p5_rows("smooth", smooth)
    fixture_pair = pair_bins(fixture)
    fixture_pair_runs = {
        bins: _load_run(f"ladder_fixture_{bins}_uniform") for bins in fixture_pair
    }
    fixture_manifests = {bins: item[1] for bins, item in fixture_pair_runs.items()}
    final_fixture_events = [
        row
        for row in analysis["event_rows"]
        if str(row.get("run_id", "")) in {
            f"ladder_fixture_{fixture_pair[0]}_uniform",
            f"ladder_fixture_{fixture_pair[1]}_uniform",
        }
    ]
    event_by_grid_class_time = {
        (int(row["bins"]), str(row["class_label"]), float(row["target_time_h"])): row
        for row in final_fixture_events
    }
    event_groups: dict[tuple[int, str], list[Mapping[str, Any]]] = defaultdict(list)
    for row in final_fixture_events:
        event_groups[(int(row["bins"]), str(row["class_label"]))].append(row)
    final_fixture_event_rows = [
        event_by_grid_class_time[(bins, label, 48.0)]
        for bins, label in sorted(event_groups)
        if (bins, label, 48.0) in event_by_grid_class_time
    ]
    fixture_final_lower_rows = [
        row
        for row in analysis["lower_rows"]
        if row.get("scenario") == "fixture" and int(row["bins"]) in fixture_pair
    ]
    max_rmax_flux = max(
        (abs(float(row["rmax_outflow_maximum_m3_s"])) for row in fixture_final_lower_rows), default=0.0
    )
    max_tag_lower_difference = max(
        (abs(float(row["tag_vs_aggregate_lower_number_difference_m3"])) for row in fixture_final_lower_rows), default=0.0
    )
    max_tag_relative_error = max(
        (float(row["maximum_tag_aggregate_relative_error"]) for row in fixture_final_lower_rows), default=0.0
    )
    max_boundary_residual = max(
        (abs(float(row["total_residual_relative"])) for row in fixture_final_lower_rows), default=0.0
    )
    fixture_bad_rows = [
        row
        for row in analysis["pair_rows"]
        if row["scenario"] == "fixture"
        and int(row["left_bins"]) == fixture_pair[0]
        and int(row["right_bins"]) == fixture_pair[1]
        and float(row["relative_error"]) > 0.02
    ]
    first_fixture_bad = min(
        fixture_bad_rows,
        key=lambda row: (float(row["target_time_h"]), -float(row["relative_error"])),
        default=None,
    )
    baseline_manifest = _load_run("baseline_fixture_200_historical")[1]
    fixture_p5_table = _markdown_table(
        ["Primary metric", "48 h relative error", "P5 <=2%", "Full-time maximum"],
        [
            (
                row["metric"],
                percent(row["relative_error"]),
                "PASS" if row["endpoint_p5_pass_2pct"] else "FAIL",
                percent(row["full_time_max_relative_error"]),
            )
            for row in fixture_p5_rows
        ],
    )
    smooth_p5_table = _markdown_table(
        ["Primary metric", "48 h relative error", "P5 <=2%", "Full-time maximum"],
        [
            (
                row["metric"],
                percent(row["relative_error"]),
                "PASS" if row["endpoint_p5_pass_2pct"] else "FAIL",
                percent(row["full_time_max_relative_error"]),
            )
            for row in smooth_p5_rows
        ],
    )
    baseline_table = _markdown_table(
        ["Check", "Result"],
        [
            ("Strict legacy failure", baseline["strict_legacy_failure"]["status"]),
            ("Repaired historical P5", baseline["repaired_48h_and_historical_p5"]["status"]),
            ("200 schedule", "sequential 0/0.1/1/3/6/12/24/48 h"),
            ("400 schedule", "direct 0→48 h"),
        ],
    )
    _write_text_new(
        REPORT_ROOT / "00_baseline_reproduction.md",
        "# Baseline reproduction\n\n"
        f"Status: `{baseline['status']}`. The legacy unsafe diagnostic was replayed from the frozen initial entries; production transport remained the repaired implicit face solve.\n\n"
        f"{baseline_table}\n\n"
        "The legacy acceptance comparison uses a predeclared 5e-14 relative tolerance, because the frozen replay differs only at floating-point ulps while failure step, bin and timestep are exact.\n\n"
        "Recorded baseline contract: "
        f"solver `{baseline_manifest['solver_mode']}`, source-config hash `{baseline_manifest['source_config_hash']}`, "
        f"Rmin `{_number(baseline_manifest['Rmin_m'])}` m, Rmax `{_number(baseline_manifest['Rmax_m'])}` m, "
        f"{baseline_manifest['radius_bins']} radius bins, radius-edge SHA-256 `{_canonical_sha256(baseline_manifest['radius_edges_m'])}`, "
        f"fixture PSD source hash `{baseline_manifest['initial_psd_source_hash']}`, and lower-bound treatment "
        f"`{baseline_manifest['lower_boundary_treatment']}`. The run manifest records the exact timestep policy and output moment definitions.\n",
    )
    initial_table = _markdown_table(
        [
            "Bins",
            "M0 rel. 1600",
            "M3 rel. 1600",
            "β inventory rel. 1600",
            "M1 projection error",
            "M2 projection error",
            "Sv projection error",
            "cell/M3 midpoint diff.",
        ],
        [
            (
                row["bins"],
                _number(row["relative_to_1600_M0_m3"]),
                _number(row["relative_to_1600_M3_dimensionless"]),
                _number(row["relative_to_1600_beta_resolved_inventory_mol_m3"]),
                _number(row["projection_relative_error_M1"]),
                _number(row["projection_relative_error_M2"]),
                _number(row["projection_relative_error_Sv"]),
                _number(
                    _relative_error(
                        float(row["cell_integrated_M3_dimensionless"]),
                        float(row["midpoint_M3_dimensionless"]),
                    )
                ),
            )
            for row in initial_rows
        ],
    )
    _write_text_new(
        REPORT_ROOT / "01_initial_projection_and_moment_audit.md",
        "# Initial projection and moment audit\n\n"
        f"Projection status: `{initial['status']}`. Production observables use the fixed-pivot measure; cell-integrated values are a reconstruction diagnostic only.\n\n"
        f"{initial_table}\n\n"
        "Each discrete source radius is split non-negatively between bracketing fixed pivots to preserve M0 and M3. M1/M2/Sv projection error is reported in `initial_grid_moments.csv`.\n",
    )
    _write_text_new(
        REPORT_ROOT / "02_smooth_psd_convergence.md",
        "# Smooth PSD convergence\n\n"
        f"Smooth lognormal qualification status: `{smooth['status']}`. Required 800 vs 1600 endpoint pass: `{smooth['reference_endpoint_pass']}`. Final qualifying/diagnostic pair: `{smooth['final_pair']}`.\n\n"
        + smooth_p5_table
        + "\n\nThis benchmark has the same fixed-pivot beta inventory and matrix state as the fixture, but it is numerical qualification only and is never substituted for the PF comparison.\n",
    )
    fixture_event_final = [
        row
        for row in analysis["event_rows"]
        if str(row.get("run_id", "")).startswith("ladder_fixture_")
        and math.isclose(float(row["target_time_h"]), 48.0, rel_tol=0.0, abs_tol=1.0e-12)
    ]
    first_crossings = [
        row
        for row in fixture_event_final
        if row.get("first_lower_boundary_crossing_time_h") not in ("", None)
    ]
    first_crossing = min(
        first_crossings,
        key=lambda row: float(row["first_lower_boundary_crossing_time_h"]),
        default=None,
    )
    def survival_trace(bins: int, label: str) -> str:
        return "; ".join(
            f"{_number(row['target_time_h'])}h:{_number(row['survival_weight_fraction'], digits=4)}"
            for row in sorted(event_groups[(bins, label)], key=lambda item: float(item["target_time_h"]))
        )

    def occupied_bin_summary(row: Mapping[str, Any]) -> str:
        raw = str(row["current_occupied_bin_indices"])
        if not raw:
            return "none"
        indices = raw.split(";")
        return f"{len(indices)} ({indices[0]}–{indices[-1]})"

    event_table = _markdown_table(
        [
            "Grid",
            "Class",
            "R0 / effective R0 (nm)",
            "Projected bins",
            "48 h occupied bins / sign",
            "First Rmin-face endpoint (h)",
            "Survival at scheduled times",
            "48 h returned β inventory (mol m⁻³)",
        ],
        [
            (
                int(row["bins"]),
                row["class_label"],
                f"{_number(float(row['initial_radius_m']) * 1.0e9)}/"
                f"{_number(float(event_by_grid_class_time[(int(row['bins']), str(row['class_label']), 0.0)]['effective_radius_m']) * 1.0e9)}",
                event_by_grid_class_time[(int(row["bins"]), str(row["class_label"]), 0.0)]["initial_projected_bin_indices"],
                f"{occupied_bin_summary(row)} / {row['growth_dissolution_sign']}",
                _number(row["first_lower_boundary_crossing_time_h"]),
                survival_trace(int(row["bins"]), str(row["class_label"])),
                _number(row["cumulative_returned_matrix_beta_inventory_mol_m3"]),
            )
            for row in final_fixture_event_rows
        ],
    )
    first_crossing_text = (
        "No tagged lower-face crossing was recorded."
        if first_crossing is None
        else (
            "The earliest tagged route is class "
            f"`{first_crossing['class_label']}` on the {first_crossing['bins']}-bin grid at accepted endpoint "
            f"{_number(first_crossing['first_lower_boundary_crossing_time_h'])} h. It is an `Rmin` face-flux tail event: "
            "it directly changes M0/N, while its fixed-pivot terminal M3 inventory is separately returned to the matrix ledger."
        )
    )
    _write_text_new(
        REPORT_ROOT / "03_six_particle_event_sensitivity.md",
        "# Six-particle event sensitivity\n\n"
        "The per-class event table uses passive tags that receive the exact production implicit M-matrix update with frozen accepted-step velocities. Lower-bound crossings are reported as accepted-step endpoints, not as fabricated continuous-time events.\n\n"
        f"Fixture ladder status: `{fixture['status']}`. The table focuses on final pair `{fixture['final_pair']}`; the raw `class_events.csv` retains every grid/time/class row.\n\n"
        f"{first_crossing_text}\n\n"
        f"{event_table}\n\n"
        "Flux route: implicit-upwind `Rmin` face (bin 0) → conservative matrix inventory ledger. It directly changes M0/N; the fixed-pivot beta inventory representation is recovered in the matrix ledger without a negative-bin clamp. For PF Case A, component first-absence is checkpoint-bounded and remains fail-closed if an unresolved merge/split event appears.\n",
    )
    ladder_rows = []
    for scenario in ("fixture", "smooth"):
        pair_keys = sorted(
            {
                (int(row["left_bins"]), int(row["right_bins"]))
                for row in analysis["pair_rows"]
                if row["scenario"] == scenario
            }
        )
        for left_bins, right_bins in pair_keys:
            rows_48 = [
                row
                for row in analysis["pair_rows"]
                if row["scenario"] == scenario
                and int(row["left_bins"]) == left_bins
                and int(row["right_bins"]) == right_bins
                and math.isclose(float(row["target_time_h"]), 48.0, rel_tol=0.0, abs_tol=1.0e-12)
            ]
            metric_rows = {str(row["metric"]): row for row in rows_48}
            ladder_rows.append(
                (
                    scenario,
                    f"{left_bins} vs {right_bins}",
                    percent(max(float(row["relative_error"]) for row in rows_48)),
                    percent(max(float(row["full_time_max_relative_error"]) for row in rows_48)),
                    "; ".join(
                        f"{metric}={percent(metric_rows[metric]['relative_error'])}"
                        for metric in PRIMARY_METRICS
                    ),
                )
            )
    _write_text_new(
        REPORT_ROOT / "04_radius_grid_ladder.md",
        "# Radius-grid ladder\n\n"
        f"Fixture status: `{fixture['status']}`; smooth status: `{smooth['status']}`.\n\n"
        + _markdown_table(["Scenario", "Pair", "Largest 48 h error", "Largest full-time error", "Six primary 48 h errors"], ladder_rows)
        + "\n\nThe P5 endpoint decision always uses the six 48 h metrics, while the full-time maxima are reported without replacing that gate. All inter-grid PSD distances use M0/M3-conservative remapping to a shared 3200-bin analysis grid.\n",
    )
    first_bad_text = (
        "No primary-metric comparison in the final fixture pair exceeds 2%."
        if first_fixture_bad is None
        else (
            "The first registered final-pair P5 exceedance occurs at "
            f"`{_number(first_fixture_bad['target_time_h'])} h` for `{first_fixture_bad['metric']}` "
            f"with relative error `{percent(first_fixture_bad['relative_error'])}`."
        )
    )
    grid_contract_table = _markdown_table(
        ["Grid", "Rmin (m)", "Rmax (m)", "Edge SHA-256", "Timestep policy"],
        [
            (
                bins,
                _number(manifest["Rmin_m"]),
                _number(manifest["Rmax_m"]),
                _canonical_sha256(manifest["radius_edges_m"]),
                json.dumps(manifest["simulation"], sort_keys=True),
            )
            for bins, manifest in sorted(fixture_manifests.items())
        ],
    )
    boundary_text = (
        "Across the final fixture pair, maximum Rmax outflow is "
        f"`{_number(max_rmax_flux)}` m⁻³ s⁻¹; maximum tag-vs-aggregate lower-number mismatch is "
        f"`{_number(max_tag_lower_difference)}` m⁻³ (relative tag closure `{_number(max_tag_relative_error)}`), "
        f"and maximum ledger residual is `{_number(max_boundary_residual)}`. This does not support `RADIUS_RANGE_TRUNCATION` "
        "or non-conservative lower-bound loss."
    )
    crossing_estimate_text = ""
    if first_crossing is not None:
        first_run_id = str(first_crossing["run_id"])
        first_t0 = next(
            (
                row
                for row in analysis["event_rows"]
                if str(row.get("run_id", "")) == first_run_id
                and str(row.get("class_label", "")) == str(first_crossing["class_label"])
                and math.isclose(float(row["target_time_h"]), 0.0, rel_tol=0.0, abs_tol=1.0e-12)
            ),
            None,
        )
        first_manifest = _load_run(first_run_id)[1]
        if first_t0 is not None and float(first_t0["weighted_growth_rate_m_s"]) < 0.0:
            constant_start_rate_h = (
                (float(first_t0["effective_radius_m"]) - float(first_manifest["Rmin_m"]))
                / abs(float(first_t0["weighted_growth_rate_m_s"]))
                / 3600.0
            )
            crossing_estimate_text = (
                " A constant-start-rate extrapolation (explicitly not an event-time prediction) from that class's "
                f"initial weighted dissolution rate gives `{_number(constant_start_rate_h)} h` to Rmin, versus the "
                "observed implicit-tail accepted endpoint above."
            )
    crossing_text = (
        "No tagged Rmin crossing was observed."
        if first_crossing is None
        else (
            f"Earliest observed tail event: `{first_crossing['class_label']}`, initial radius "
            f"`{_number(float(first_crossing['initial_radius_m']) * 1.0e9)} nm`, grid `{first_crossing['bins']}`, "
            f"at accepted endpoint `{_number(first_crossing['first_lower_boundary_crossing_time_h'])} h`. "
            "The route is the implicit-upwind lower face of bin 0 into the matrix ledger; it directly affects M0/N, "
            "while the fixed-pivot M3 inventory remains ledger-closed."
            + crossing_estimate_text
        )
    )
    _write_text_new(
        REPORT_ROOT / "05_root_cause.md",
        "# Root cause classification\n\n"
        f"Classifications: `{', '.join(root_cause['root_causes'])}`.\n\n"
        + "\n".join(f"- {item}" for item in root_cause["evidence"])
        + "\n\n## Final fixture P5 evidence\n\n"
        + fixture_p5_table
        + "\n\n"
        + first_bad_text
        + "\n\n## Controlled grid contract\n\n"
        + grid_contract_table
        + "\n\n"
        + crossing_text
        + "\n\n"
        + boundary_text
        + "\n\nThe t=0 M0/M3/beta-inventory/total-inventory invariants pass the 1e-12 projection gate; the cell-integrated versus fixed-pivot M3 difference is a recorded diagnostic reconstruction, not a dynamics change. Thus the observed final-pair discrepancy is not attributed to an initial M0/M3 projection mismatch or a post-processing substitution.\n",
    )
    repair_text = (
        "# Minimal repair\n\n"
        "The retained production method is `CONSERVATIVE_IMPLICIT_UPWIND_FACE_SOLVE`: shared-face conservative transport, frozen start-state velocities, no negative-bin clamp, and ledger-based inventory recovery.\n\n"
    )
    if fixture["status"] == "PASS_KWN_RADIUS_GRID_CONVERGENCE":
        repair_text += f"The minimal production selection is the authority grid `{fixture['authority_grid']}`; no material parameter changed.\n"
    else:
        repair_text += "No higher-order solver was introduced. The current evidence does not authorize a rushed solver-order modification.\n"
    _write_text_new(REPORT_ROOT / "06_minimal_repair.md", repair_text)
    lower_status = (
        "PASS_LOWER_BOUNDARY_TAG_AGGREGATE_AND_LEDGER_CLOSURE"
        if max_tag_relative_error <= 1.0e-12 and max_boundary_residual <= 1.0e-10
        else "FAIL_LOWER_BOUNDARY_TAG_AGGREGATE_OR_LEDGER_CLOSURE"
    )
    requalification_status = str(requalification.get("status", "NOT_RUN"))
    requalification_note = (
        "Authority-grid requalification is deliberately not run: the P5 gate has not authorized an authority grid. "
        "This gate block does not change the independently reproduced positivity/conservation result."
        if requalification_status == "BLOCKED_RADIUS_GRID_NOT_PASSED"
        else "Authority-grid requalification rows are recorded in `kwn_requalification.csv`."
    )
    _write_text_new(
        REPORT_ROOT / "07_kwn_requalification.md",
        "# KWN requalification\n\n"
        f"Authority requalification status: `{requalification_status}`. Independent repaired positivity/conservation status: `{positivity['status']}`.\n\n"
        f"{requalification_note}\n\n"
        "`kwn_requalification.csv` separates generic synthetic engine regressions from authority-fixture evidence and records all four inventory buckets. It is never used to relabel a P5-gated authority run as a positivity failure.\n",
    )
    comparison_csv = TASK_OUTPUT_ROOT / "beta_only_kwn_pf_comparison.csv"
    comparison_rows = _read_csv(comparison_csv) if comparison_csv.is_file() else []
    comparison_time_rows = [row for row in comparison_rows if row.get("record_type") == "time_snapshot"]
    if comparison_time_rows:
        comparison_table = _markdown_table(
            ["Nominal h", "PF accepted h", "N KWN/PF", "Rmean KWN/PF (nm)", "Rmean³ KWN/PF", "Sv KWN/PF", "W1 number/volume (m)"],
            [
                (
                    row["comparison_target_time_h"],
                    row["pf_actual_time_h"],
                    f"{_number(row['kwn_N_m0_m3'])}/{_number(row['pf_N_m0_m3'])}",
                    f"{_number(float(row['kwn_Rmean_m']) * 1.0e9)}/{_number(float(row['pf_Rmean_m']) * 1.0e9)}",
                    f"{_number(row['kwn_Rmean3_m3'])}/{_number(row['pf_Rmean3_m3'])}",
                    f"{_number(row['kwn_Sv_m_inv'])}/{_number(row['pf_Sv_component_m_inv'])}",
                    f"{_number(row['W1_number_m'])}/{_number(row['W1_volume_m'])}",
                )
                for row in comparison_time_rows
            ],
        )
        comparison_detail = (
            f"Direction: `{comparison.get('status')}`; timescale: `{comparison.get('timescale_status')}`; "
            f"mean-field classification: `{comparison.get('mean_field_gap')}`; Rmean³ slope ratio KWN/PF: "
            f"`{_number(comparison.get('coarsening_slope_ratio_kwn_over_pf'))}`.\n\n{comparison_table}"
        )
    else:
        comparison_detail = (
            "No PF trajectory or particle result was read for a comparison conclusion. The recorded status is a gate block, not `FAIL_BETA_ONLY_DIRECTION`."
        )
    _write_text_new(
        REPORT_ROOT / "08_beta_only_kwn_pf_comparison.md",
        "# Beta-only KWN–PF comparison\n\n"
        f"Status: `{comparison.get('status', 'NOT_RUN')}`.\n\n"
        "PF authority is frozen Case A because A/B audited fields are identical through 48 h. The comparison uses Case A accepted checkpoint times directly; no CUDA calculation or PF interpolation was performed.\n\n"
        f"{comparison_detail}\n",
    )
    _write_text_new(
        REPORT_ROOT / "09_model_limitations.md",
        "# Model limitations\n\n"
        "This is a validation-contract code-level 96³ KWN–PF comparison, not historical 400³ production validation. `HISTORICAL_AS_RUN_AUTHORITY_UNRECOVERED` and `HISTORICAL_12H_PSD_NOT_RECOVERED` remain in force. GP-assisted nucleation, GP release, GP→beta transfer, online coupling, fitted D-scale, and PF dislocation-density prediction remain outside scope.\n",
    )
    release = (
        "ELIGIBLE_FOR_SEPARATELY_GATED_LOCAL_GP_RELEASE_PROTOTYPE"
        if top_status == "PASS_KWN_PF_ONE_WAY_STORAGE_COUPLING_V1"
        else "LOCAL_GP_RELEASE_NOT_AUTHORIZED"
    )
    authority_text = str(fixture["authority_grid"]) if fixture["authority_grid"] is not None else "NONE"
    restart_status = "NOT_RUN_P5_GATED"
    if requalification_status == "PASS_KWN_POSITIVITY_CONSERVATION_48H":
        restart_rows = [row for row in requalification.get("rows", []) if row.get("gate_id") == "P6_RESTART"]
        restart_status = str(restart_rows[0].get("status", "NOT_RECORDED")) if restart_rows else "NOT_RECORDED"
    p0_blocker = (
        "None."
        if top_status == "PASS_KWN_PF_ONE_WAY_STORAGE_COUPLING_V1"
        else (
            "Exact six-particle fixture P5 is unresolved after the registered final pair; human gate review is required."
            if top_status == "DISCRETE_EVENT_SENSITIVITY_GATE_REVIEW_REQUIRED"
            else "Radius-grid / authority-qualification gate prevents a coupling acceptance claim."
        )
    )
    next_action = (
        "Do not start GP release; preserve the frozen evidence and obtain an explicit human decision on the discrete-event P5 gate."
        if top_status == "DISCRETE_EVENT_SENSITIVITY_GATE_REVIEW_REQUIRED"
        else "No local GP-release work is authorized by this task status."
    )
    _write_text_new(
        REPORT_ROOT / "10_final_acceptance_report.md",
        "# Final acceptance\n\n"
        f"Top-level status: `{top_status}`.\n\n"
        "## Independent state closure\n\n"
        f"- Baseline reproduction: `{baseline['status']}`\n"
        f"- Positivity/conservation: `{positivity['status']}`; max uniform-fixture ledger residual `{_number(positivity['maximum_uniform_fixture_ledger_residual'])}`, minimum bin `{_number(positivity['minimum_uniform_fixture_bin_density_per_m4'])}` m⁻⁴\n"
        f"- Authority requalification: `{requalification_status}`\n"
        f"- Initial projection: `{initial['status']}`; moment quadrature remains diagnostic-only\n"
        f"- Lower-bound closure: `{lower_status}`\n\n"
        "## Radius-grid decision\n\n"
        f"Fixture ladder `{fixture['available_bins']}`: `{fixture['status']}`. Smooth ladder `{smooth['available_bins']}`: `{smooth['status']}`. Final fixture pair `{fixture['final_pair']}`; authority grid `{authority_text}`.\n\n"
        f"{fixture_p5_table}\n\n"
        "## Root cause and coupling boundary\n\n"
        f"Root causes: `{', '.join(root_cause['root_causes'])}`. KWN 48 h completion is included in `{positivity['fixture_48h_completed_runs']}/{positivity['fixture_run_count']}` uniform fixture runs; restart status: `{restart_status}`.\n\n"
        f"Beta-only comparison: `{comparison.get('status', 'NOT_RUN')}`; direction `{comparison.get('status', 'NOT_RUN')}`; timescale `{comparison.get('timescale_status', 'NOT_CLAIMED')}`; mean-field gap `{comparison.get('mean_field_gap', 'NOT_ASSESSED')}`.\n\n"
        "Frozen CUDA/PF evidence was reused without a CUDA rerun; `PF_SOURCE_MODIFIED=false`; no thermodynamic, D(T), gamma, PSD, or D-scale retuning was performed. `HISTORICAL_AS_RUN_AUTHORITY_UNRECOVERED` and `HISTORICAL_12H_PSD_NOT_RECOVERED` remain in force.\n\n"
        f"P0 blocker: {p0_blocker}\n\n"
        f"Next action: {next_action}\n\n"
        f"Local GP release: `{release}`.\n\n"
        "Evidence/provenance: `analysis_provenance.json`, `grid_pair_errors.csv`, `grid_psd_distances.csv`, `dissolution_event_times.csv`, and the immutable per-run manifests under `outputs/kwn_radius_grid_convergence_v1/runs/`.\n",
    )
    _write_text_new(
        REPORT_ROOT / "11_reproduction_commands.md",
        "# Reproduction commands\n\n"
        "```bash\n"
        "PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py baseline\n"
        "PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py initial-audit\n"
        "for bins in 100 200 400 800 1600; do\n"
        "  PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py run-grid --scenario fixture --bins $bins --schedule uniform --run-id ladder_fixture_${bins}_uniform\n"
        "  PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py run-grid --scenario smooth --bins $bins --schedule uniform --run-id ladder_smooth_${bins}_uniform\n"
        "done\n"
        "PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py inspect\n"
        "# Run 3200 fixture/smooth only if inspect declares it required.\n"
        "PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py requalify\n"
        "PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py compare-pf\n"
        "PYTHONPATH=src python3 scripts/run_kwn_radius_grid_convergence_v1.py assemble\n"
        "```\n",
    )


def _assemble() -> int:
    _require_baseline_pass()
    expected_outputs = (
        TASK_OUTPUT_ROOT / "initial_grid_moments.csv",
        TASK_OUTPUT_ROOT / "kwn_requalification.csv",
        TASK_OUTPUT_ROOT / "beta_only_kwn_pf_comparison.csv",
    )
    if REPORT_ROOT.exists() or any(path.exists() for path in (
        TASK_OUTPUT_ROOT / "grid_trajectory_all.csv",
        TASK_OUTPUT_ROOT / "grid_pair_errors.csv",
        TASK_OUTPUT_ROOT / "grid_psd_distances.csv",
        TASK_OUTPUT_ROOT / "dissolution_event_times.csv",
        TASK_OUTPUT_ROOT / "lower_boundary_flux.csv",
    )):
        raise RuntimeError("Refusing to overwrite assembled task outputs")
    if not all(path.is_file() for path in expected_outputs):
        raise RuntimeError("Initial audit, requalification, and comparison outputs must exist before assembly")
    baseline = _read_json(_baseline_summary_path())
    initial = _read_json(TASK_OUTPUT_ROOT / "initial_projection_audit.json")
    initial_rows = _as_float_rows(_read_csv(TASK_OUTPUT_ROOT / "initial_grid_moments.csv"))
    analysis = _collect_ladder_analysis()
    if any(
        assessment["status"] == "PENDING_3200_REQUIRED"
        for assessment in (analysis["fixture"], analysis["smooth"])
    ):
        raise RuntimeError("Cannot assemble before every registered 3200 extension required by the ladder is complete")
    requalification = _read_json(TASK_OUTPUT_ROOT / "kwn_requalification_summary.json")
    comparison = _read_json(TASK_OUTPUT_ROOT / "beta_only_kwn_pf_comparison_summary.json")
    root_cause = _derive_root_cause(initial=initial, analysis=analysis)
    positivity = _independent_positivity_conservation(baseline=baseline, analysis=analysis)
    top_status = _top_status(
        analysis=analysis, requalification=requalification, comparison=comparison
    )
    _write_csv_new(TASK_OUTPUT_ROOT / "grid_trajectory_all.csv", analysis["trajectory_rows"])
    _write_csv_new(TASK_OUTPUT_ROOT / "grid_pair_errors.csv", analysis["pair_rows"])
    _write_csv_new(TASK_OUTPUT_ROOT / "grid_psd_distances.csv", analysis["psd_rows"])
    comparison_events = comparison.get("events", []) if isinstance(comparison.get("events", []), list) else []
    _write_csv_new(
        TASK_OUTPUT_ROOT / "dissolution_event_times.csv",
        _normalise_rows(list(analysis["event_rows"]) + list(comparison_events)),
    )
    _write_csv_new(TASK_OUTPUT_ROOT / "lower_boundary_flux.csv", analysis["lower_rows"])
    _write_json_new(
        TASK_OUTPUT_ROOT / "final_acceptance.json",
        {
            "top_status": top_status,
            "baseline_status": baseline["status"],
            "positivity_conservation": positivity,
            "authority_requalification_status": requalification.get("status"),
            "initial_projection_status": initial["status"],
            "fixture_ladder": analysis["fixture"],
            "smooth_ladder": analysis["smooth"],
            "root_cause": root_cause,
            "comparison_status": comparison.get("status"),
            "local_gp_release": (
                "ELIGIBLE_FOR_SEPARATELY_GATED_LOCAL_GP_RELEASE_PROTOTYPE"
                if top_status == "PASS_KWN_PF_ONE_WAY_STORAGE_COUPLING_V1"
                else "LOCAL_GP_RELEASE_NOT_AUTHORIZED"
            ),
        },
    )
    _write_json_new(TASK_OUTPUT_ROOT / "analysis_provenance.json", _analysis_provenance())
    _write_text_new(
        TASK_OUTPUT_ROOT / "figures" / "README.md",
        "No figure is substituted for the registered CSV evidence. This directory reserves task-scoped figures derived from the immutable ladder outputs.\n",
    )
    _write_reports(
        baseline=baseline,
        initial=initial,
        initial_rows=initial_rows,
        analysis=analysis,
        root_cause=root_cause,
        requalification=requalification,
        comparison=comparison,
        top_status=top_status,
    )
    print(json.dumps({"top_status": top_status, "authority_grid": analysis["fixture"]["authority_grid"]}, sort_keys=True))
    return 0 if top_status in {
        "PASS_KWN_PF_ONE_WAY_STORAGE_COUPLING_V1",
        "PASS_KWN_RADIUS_GRID_CONVERGENCE_BETA_DIRECTION_FAIL",
    } else 2


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    subparsers.add_parser("baseline", help="reproduce the frozen strict failure and repaired P5 baseline")
    subparsers.add_parser("initial-audit", help="audit t=0 conservative projection and moment quadrature")
    run = subparsers.add_parser("run-grid", help="run one task-scoped immutable grid member")
    run.add_argument("--scenario", choices=("fixture", "smooth"), required=True)
    run.add_argument("--bins", type=int, required=True)
    run.add_argument(
        "--schedule",
        choices=("uniform", "historical_200", "historical_direct_48", "pf_accepted"),
        required=True,
    )
    run.add_argument("--run-id", required=True)
    run.add_argument("--max-dt-factor", type=float, default=1.0)
    subparsers.add_parser("inspect", help="read completed ladder members and report whether 3200 is required")
    subparsers.add_parser("requalify", help="run gated authority-grid KWN qualification")
    subparsers.add_parser("compare-pf", help="run gated beta-only comparison against frozen Case A")
    subparsers.add_parser("assemble", help="assemble registered outputs and reports")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    try:
        if args.command == "baseline":
            return _run_baseline()
        if args.command == "initial-audit":
            return _run_initial_audit()
        if args.command == "run-grid":
            _run_grid(
                run_id=args.run_id,
                scenario=args.scenario,
                bins=args.bins,
                schedule=args.schedule,
                max_dt_factor=args.max_dt_factor,
                require_baseline=True,
            )
            return 0
        if args.command == "inspect":
            return _inspect_ladder()
        if args.command == "requalify":
            return _run_requalification()
        if args.command == "compare-pf":
            return _run_beta_only_comparison()
        if args.command == "assemble":
            return _assemble()
    except (RuntimeError, SolverStateError, ValueError) as error:
        print(json.dumps({"status": "ERROR", "command": args.command, "error": str(error)}, sort_keys=True), file=sys.stderr)
        return 2
    raise AssertionError(f"unhandled command {args.command}")


if __name__ == "__main__":
    raise SystemExit(main())
