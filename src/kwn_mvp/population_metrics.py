"""Shared population moments and pseudo-binary inventory closure helpers.

The KWN finite-volume state stores a cell-integrated number population, while
the cohort comparator stores weighted radius atoms.  This module gives both
representations one explicit set of moment and inventory definitions.  It is
deliberately free of solver state so the same functions can be used for
initial-measure identity checks and production post-processing.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math

import numpy as np
from numpy.typing import NDArray


class PopulationMetricError(ValueError):
    """Raised when a measure or an inventory closure is not physically valid."""


@dataclass(frozen=True)
class PopulationMetrics:
    """Unambiguous spherical-population moments and derived observables."""

    M0_m3: float
    M1_m2: float
    M2_m: float
    M3_dimensionless: float
    N_m0_m3: float
    Rmean_number_m: float
    Rmean_cubed_m3: float
    mean_R3_m3: float
    Sv_m_inv: float
    f_beta: float

    def as_dict(self) -> dict[str, float]:
        """Return a stable, CSV/JSON-friendly mapping."""

        return {key: float(value) for key, value in asdict(self).items()}


@dataclass(frozen=True)
class MatrixClosure:
    """Algebraic pseudo-binary matrix closure for prescribed precipitates."""

    matrix_xb: float
    matrix_fraction: float
    matrix_inventory_mol_m3: float
    precipitate_inventory_mol_m3: float
    total_inventory_mol_m3: float
    residual_mol_m3: float
    relative_residual: float

    def as_dict(self) -> dict[str, float]:
        """Return a stable, CSV/JSON-friendly mapping."""

        return {key: float(value) for key, value in asdict(self).items()}


def _as_positive_radii(radii_m: NDArray[np.float64] | np.ndarray) -> NDArray[np.float64]:
    radii = np.asarray(radii_m, dtype=np.float64)
    if radii.ndim != 1 or not np.all(np.isfinite(radii)) or np.any(radii <= 0.0):
        raise PopulationMetricError("radii must be a finite, one-dimensional positive array")
    return radii


def _as_nonnegative_weights(
    weights_m3: NDArray[np.float64] | np.ndarray, expected_size: int
) -> NDArray[np.float64]:
    weights = np.asarray(weights_m3, dtype=np.float64)
    if weights.ndim != 1 or weights.size != expected_size:
        raise PopulationMetricError("measure weights must be one-dimensional and match radii")
    if not np.all(np.isfinite(weights)) or np.any(weights < 0.0):
        raise PopulationMetricError("measure weights must be finite and non-negative")
    return weights


def _metrics_from_moments(moments: NDArray[np.float64]) -> PopulationMetrics:
    m0, m1, m2, m3 = (float(value) for value in moments)
    if m0 < 0.0 or m1 < 0.0 or m2 < 0.0 or m3 < 0.0:
        raise PopulationMetricError("radius moments must be non-negative")
    rmean = 0.0 if m0 == 0.0 else m1 / m0
    mean_r3 = 0.0 if m0 == 0.0 else m3 / m0
    return PopulationMetrics(
        M0_m3=m0,
        M1_m2=m1,
        M2_m=m2,
        M3_dimensionless=m3,
        N_m0_m3=m0,
        Rmean_number_m=rmean,
        Rmean_cubed_m3=rmean**3,
        mean_R3_m3=mean_r3,
        Sv_m_inv=4.0 * math.pi * m2,
        f_beta=4.0 * math.pi * m3 / 3.0,
    )


def metrics_from_discrete_measure(
    radii_m: NDArray[np.float64] | np.ndarray,
    number_weights_m3: NDArray[np.float64] | np.ndarray,
) -> PopulationMetrics:
    """Return moments of weighted radius atoms.

    ``number_weights_m3`` is the number density associated with every radius
    atom, not a normalized probability weight.
    """

    radii = _as_positive_radii(radii_m)
    weights = _as_nonnegative_weights(number_weights_m3, radii.size)
    moments = np.asarray(
        [np.sum(weights * radii**order, dtype=np.float64) for order in range(4)],
        dtype=np.float64,
    )
    return _metrics_from_moments(moments)


def cell_moments_from_piecewise_constant_cells(
    edges_m: NDArray[np.float64] | np.ndarray,
    cell_number_m3: NDArray[np.float64] | np.ndarray,
) -> NDArray[np.float64]:
    """Return per-cell M0--M3 for a density constant within each radius cell."""

    edges = _as_positive_radii(edges_m)
    if edges.size < 3 or np.any(np.diff(edges) <= 0.0):
        raise PopulationMetricError("cell edges must be strictly increasing with at least two cells")
    number = _as_nonnegative_weights(cell_number_m3, edges.size - 1)
    widths = np.diff(edges)
    density = number / widths
    values = np.empty((4, number.size), dtype=np.float64)
    for order in range(4):
        integral = (edges[1:] ** (order + 1) - edges[:-1] ** (order + 1)) / (order + 1)
        values[order] = density * integral
    return values


def metrics_from_piecewise_constant_cells(
    edges_m: NDArray[np.float64] | np.ndarray,
    cell_number_m3: NDArray[np.float64] | np.ndarray,
) -> PopulationMetrics:
    """Return exact M0--M3 for a piecewise-constant cell density measure."""

    return _metrics_from_moments(np.sum(cell_moments_from_piecewise_constant_cells(edges_m, cell_number_m3), axis=1))


def positive_cell_quadrature(
    edges_m: NDArray[np.float64] | np.ndarray,
    cell_number_m3: NDArray[np.float64] | np.ndarray,
    points_per_cell: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Create deterministic positive Gauss--Legendre atoms from cell numbers.

    The nodes and weights integrate a piecewise-constant cell density.  A
    one-point rule uses the cell's exact cubic-moment pivot, so it preserves
    each-cell M0 and M3.  Two or more Gauss--Legendre points preserve each
    cell's M0--M3 to binary64 round-off.  Zero-number cells are omitted
    rather than emitting zero-weight cohorts.
    """

    if isinstance(points_per_cell, bool) or int(points_per_cell) != points_per_cell:
        raise PopulationMetricError("points_per_cell must be a positive integer")
    points = int(points_per_cell)
    if points <= 0:
        raise PopulationMetricError("points_per_cell must be positive")
    edges = _as_positive_radii(edges_m)
    if edges.size < 3 or np.any(np.diff(edges) <= 0.0):
        raise PopulationMetricError("cell edges must be strictly increasing with at least two cells")
    number = _as_nonnegative_weights(cell_number_m3, edges.size - 1)
    local_nodes, local_weights = np.polynomial.legendre.leggauss(points)
    active = np.flatnonzero(number > 0.0)
    radii = np.empty(active.size * points, dtype=np.float64)
    weights = np.empty(active.size * points, dtype=np.float64)
    offset = 0
    for index in active:
        lower = float(edges[index])
        upper = float(edges[index + 1])
        if points == 1:
            # One atom cannot reproduce all four cell moments.  Preserve the
            # inventory-bearing cubic moment as well as cell number exactly.
            cubic_mean = (upper**4 - lower**4) / (4.0 * (upper - lower))
            radii[offset] = cubic_mean ** (1.0 / 3.0)
            weights[offset] = float(number[index])
        else:
            half_width = 0.5 * (upper - lower)
            midpoint = 0.5 * (upper + lower)
            radii[offset : offset + points] = midpoint + half_width * local_nodes
            weights[offset : offset + points] = 0.5 * float(number[index]) * local_weights
        offset += points
    if np.any(weights <= 0.0) or np.any(radii <= 0.0):
        raise PopulationMetricError("positive cell quadrature generated a non-positive node or weight")
    return radii, weights


def beta_fraction_from_m3(m3_dimensionless: float) -> float:
    """Return spherical beta volume fraction from M3."""

    m3 = float(m3_dimensionless)
    if not math.isfinite(m3) or m3 < 0.0:
        raise PopulationMetricError("M3 must be finite and non-negative")
    return 4.0 * math.pi * m3 / 3.0


def beta_inventory_from_m3(
    m3_dimensionless: float, *, x_b: float, molar_volume_m3_mol: float
) -> float:
    """Return beta B inventory in mol m^-3 from M3."""

    if not 0.0 < float(x_b) <= 1.0 or float(molar_volume_m3_mol) <= 0.0:
        raise PopulationMetricError("beta composition and molar volume must be physical")
    return beta_fraction_from_m3(m3_dimensionless) * float(x_b) / float(molar_volume_m3_mol)


def close_matrix_from_precipitates(
    *,
    total_b_mol_m3: float,
    matrix_molar_volume_m3_mol: float,
    precipitate_volume_fraction: float,
    precipitate_inventory_mol_m3: float,
) -> MatrixClosure:
    """Close matrix composition from fixed total inventory and precipitates."""

    total = float(total_b_mol_m3)
    matrix_volume = float(matrix_molar_volume_m3_mol)
    precipitate_fraction = float(precipitate_volume_fraction)
    precipitate_inventory = float(precipitate_inventory_mol_m3)
    if total < 0.0 or matrix_volume <= 0.0:
        raise PopulationMetricError("total inventory and matrix molar volume must be physical")
    if not math.isfinite(precipitate_fraction) or not 0.0 <= precipitate_fraction < 1.0:
        raise PopulationMetricError("precipitate volume fraction must lie in [0, 1)")
    if not math.isfinite(precipitate_inventory) or precipitate_inventory < 0.0:
        raise PopulationMetricError("precipitate inventory must be finite and non-negative")
    matrix_fraction = 1.0 - precipitate_fraction
    matrix_xb = matrix_volume * (total - precipitate_inventory) / matrix_fraction
    if not math.isfinite(matrix_xb) or not 0.0 <= matrix_xb <= 1.0:
        raise PopulationMetricError("inventory closure requires an unphysical matrix composition")
    matrix_inventory = matrix_fraction * matrix_xb / matrix_volume
    residual = matrix_inventory + precipitate_inventory - total
    relative = abs(residual) / max(abs(total), 1.0e-300)
    return MatrixClosure(
        matrix_xb=matrix_xb,
        matrix_fraction=matrix_fraction,
        matrix_inventory_mol_m3=matrix_inventory,
        precipitate_inventory_mol_m3=precipitate_inventory,
        total_inventory_mol_m3=total,
        residual_mol_m3=residual,
        relative_residual=relative,
    )
