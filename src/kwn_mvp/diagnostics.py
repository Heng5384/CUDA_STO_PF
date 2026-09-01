"""Observable calculations and portable CSV diagnostics for KWN trajectories."""

from __future__ import annotations

import csv
from dataclasses import asdict
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence

import numpy as np

from .populations import Population
from .solver import KWNSolver, StepDiagnostics
from .units import m_to_nm, seconds_to_hours


def population_observables(population: Population) -> Dict[str, float]:
    """Return standard SI observables for one population."""

    return {
        "number_density_m3": population.number_density_m3(),
        "mean_radius_m": population.mean_radius_m(),
        "mean_radius_nm": m_to_nm(population.mean_radius_m()),
        "mean_radius_cubed_m3": population.mean_radius_cubed_m3(),
        "specific_surface_area_m_inv": population.specific_surface_area_m_inv(),
        "volume_fraction": population.volume_fraction(),
        "B_inventory_mol_m3": population.b_inventory_mol_m3(),
    }


def solver_observables(solver: KWNSolver) -> Dict[str, float]:
    """Return all current population, matrix, and ledger observables."""

    result: Dict[str, float] = {
        "time_s": solver.time_s,
        "time_h": seconds_to_hours(solver.time_s),
        "matrix_xB": solver.matrix_xb,
    }
    for name in ("g", "beta"):
        for key, value in population_observables(solver.population(name)).items():
            result[f"{name}_{key}"] = value
    item = solver.ledger.snapshot(matrix_xb=solver.matrix_xb, populations=solver.population_list())
    result.update(
        {
            "C_B_total_mol_m3": item.total_mol_m3,
            "C_B_matrix_mol_m3": item.matrix_mol_m3,
            "C_B_GP_mol_m3": item.gp_mol_m3,
            "C_B_beta_mol_m3": item.beta_subgrid_mol_m3 + item.beta_resolved_mol_m3,
            "inventory_residual_mol_m3": item.residual_mol_m3,
            "inventory_relative_residual": item.relative_residual,
        }
    )
    return result


def records_from_history(solver: KWNSolver, history: Iterable[StepDiagnostics]) -> List[Dict[str, float]]:
    """Convert accepted-step diagnostics to row dictionaries with current observable names."""

    rows: List[Dict[str, float]] = []
    # History is primarily a step-level mass/CFL trace.  Use the values stored
    # in diagnostics rather than retroactively labelling current population data
    # as historical microstructure.
    for item in history:
        row = {
            "step": float(item.step),
            "time_s": item.time_s,
            "time_h": seconds_to_hours(item.time_s),
            "dt_s": item.dt_s,
            "size_cfl": item.size_cfl,
            "positivity_utilization": item.positivity_utilization,
            "roundoff_zeroed_bin_count": float(item.roundoff_zeroed_bin_count),
            "matrix_xB": item.matrix_xb,
            "C_B_total_mol_m3": item.inventory.total_mol_m3,
            "C_B_matrix_mol_m3": item.inventory.matrix_mol_m3,
            "C_B_GP_mol_m3": item.inventory.gp_mol_m3,
            "C_B_beta_subgrid_mol_m3": item.inventory.beta_subgrid_mol_m3,
            "C_B_beta_resolved_mol_m3": item.inventory.beta_resolved_mol_m3,
            "inventory_relative_residual": item.inventory.relative_residual,
            "gp_nucleation_rate_m3_s": item.gp_nucleation_rate_m3_s,
            "beta_nucleation_rate_m3_s": item.beta_nucleation_rate_m3_s,
            "rmin_dissolution_flux_m3_s": item.rmin_dissolution_flux_m3_s,
            "rmax_outflow_flux_m3_s": item.rmax_outflow_flux_m3_s,
        }
        rows.append(row)
    return rows


def write_csv(path: str | Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Write deterministic CSV output without hidden column selection."""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        destination.write_text("", encoding="utf-8")
        return
    keys = list(rows[0].keys())
    for row in rows:
        if list(row.keys()) != keys:
            raise ValueError("All CSV rows must use the same ordered fields")
    with destination.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def discrete_wasserstein_distance(
    radii_a_m: np.ndarray,
    weights_a_m3: np.ndarray,
    radii_b_m: np.ndarray,
    weights_b_m3: np.ndarray,
) -> float:
    """Compute W1 distance between two non-negative discrete radius distributions.

    The result is in metres.  This direct CDF implementation avoids an
    undeclared SciPy dependency.
    """

    a_r = np.asarray(radii_a_m, dtype=np.float64)
    a_w = np.asarray(weights_a_m3, dtype=np.float64)
    b_r = np.asarray(radii_b_m, dtype=np.float64)
    b_w = np.asarray(weights_b_m3, dtype=np.float64)
    if np.any(a_w < 0.0) or np.any(b_w < 0.0):
        raise ValueError("Wasserstein weights must be non-negative")
    if a_w.sum() <= 0.0 or b_w.sum() <= 0.0:
        return float("nan")
    points = np.unique(np.concatenate([a_r, b_r]))
    a_order = np.argsort(a_r)
    b_order = np.argsort(b_r)
    a_r, a_w = a_r[a_order], a_w[a_order] / a_w.sum()
    b_r, b_w = b_r[b_order], b_w[b_order] / b_w.sum()
    cdf_a = np.searchsorted(a_r, points, side="right")
    cdf_b = np.searchsorted(b_r, points, side="right")
    mass_a = np.concatenate([[0.0], np.cumsum(a_w)])[cdf_a]
    mass_b = np.concatenate([[0.0], np.cumsum(b_w)])[cdf_b]
    return float(np.sum(np.abs(mass_a[:-1] - mass_b[:-1]) * np.diff(points)))
