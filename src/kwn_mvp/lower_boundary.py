"""Shared physical lower-radius boundary contract for KWN populations.

The resolved KWN radius domain is closed at its physical lower edge ``Rmin``.
This module centralises the quantities that must be identical for a
finite-volume population and a characteristic cohort: the edge location,
growth velocity, particle inventory, outward-flux convention, and event
residual.  It deliberately does *not* update the matrix inventory: the
algebraic ledger remains the sole matrix authority.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any

import numpy as np

from .growth import growth_rate_m_s
from .populations import PopulationParameters


class LowerBoundaryError(ValueError):
    """Raised when a physical lower-boundary quantity is not valid."""


@dataclass(frozen=True)
class ParticleInventory:
    """Spherical-equivalent inventory carried by one resolved particle."""

    radius_m: float
    volume_m3: float
    beta_moles_mol: float
    b_moles_mol: float


@dataclass(frozen=True)
class BoundaryInventoryFlux:
    """Outward lower-boundary fluxes derived from a positive number flux."""

    number_flux_out_m3_s: float
    beta_volume_flux_out_s: float
    beta_mol_flux_out_mol_m3_s: float
    b_mol_flux_out_mol_m3_s: float


def boundary_radius(grid_or_edges: Any) -> float:
    """Return the exact binary64 physical ``Rmin`` from a grid or edge value.

    No unit conversion, rounding, or text reparse occurs here.  Near the
    validation-contract lower edge, even a seemingly harmless rounded-nm
    representation can substantially change the Gibbs--Thomson velocity.
    """

    if hasattr(grid_or_edges, "edges_m"):
        edges = np.asarray(grid_or_edges.edges_m, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise LowerBoundaryError("radius grid must expose at least two finite edges")
        value = float(edges[0])
    elif np.isscalar(grid_or_edges):
        value = float(grid_or_edges)
    else:
        edges = np.asarray(grid_or_edges, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 2:
            raise LowerBoundaryError("radius edges must contain at least two values")
        value = float(edges[0])
    if not math.isfinite(value) or value <= 0.0:
        raise LowerBoundaryError("physical lower radius must be finite and positive")
    return value


def sphere_volume_m3(radius_m: float) -> float:
    """Return the spherical-equivalent volume of one particle at ``radius_m``."""

    radius = boundary_radius(radius_m)
    return 4.0 * math.pi * radius**3 / 3.0


def boundary_growth_velocity(
    *,
    radius_m: float,
    matrix_xb: float,
    parameters: PopulationParameters,
    equilibrium_adapter: Any,
) -> float:
    """Evaluate the shared growth law exactly at physical ``Rmin``.

    ``equilibrium_adapter`` is the already-frozen validation-contract
    thermodynamic path used by both solvers.  The return value follows the
    radius-space sign convention: negative means outward dissolution through
    the lower boundary.
    """

    radius = boundary_radius(radius_m)
    composition = float(matrix_xb)
    if not math.isfinite(composition) or not 0.0 <= composition <= 1.0:
        raise LowerBoundaryError("matrix composition must be finite and lie in [0, 1]")
    radii = np.asarray([radius], dtype=np.float64)
    equilibrium = equilibrium_adapter.equilibrium_xb(radii, parameters)
    velocity = growth_rate_m_s(
        radii_m=radii,
        matrix_xb=composition,
        equilibrium_xb=equilibrium,
        parameters=parameters,
    )
    value = float(np.asarray(velocity, dtype=np.float64)[0])
    if not math.isfinite(value):
        raise LowerBoundaryError("lower-boundary growth velocity must be finite")
    return value


def particle_inventory_at_radius(
    radius_m: float,
    *,
    x_b: float,
    molar_volume_m3_mol: float,
) -> ParticleInventory:
    """Return particle volume and beta/B molar inventory at one radius."""

    radius = boundary_radius(radius_m)
    composition = float(x_b)
    molar_volume = float(molar_volume_m3_mol)
    if not math.isfinite(composition) or not 0.0 < composition <= 1.0:
        raise LowerBoundaryError("precipitate B composition must lie in (0, 1]")
    if not math.isfinite(molar_volume) or molar_volume <= 0.0:
        raise LowerBoundaryError("precipitate molar volume must be finite and positive")
    volume = sphere_volume_m3(radius)
    beta_moles = volume / molar_volume
    return ParticleInventory(
        radius_m=radius,
        volume_m3=volume,
        beta_moles_mol=beta_moles,
        b_moles_mol=composition * beta_moles,
    )


def boundary_event_residual(radius_m: float, rmin_m: float) -> float:
    """Return the signed characteristic residual ``R - Rmin`` in metres."""

    radius = boundary_radius(radius_m)
    return radius - boundary_radius(rmin_m)


def boundary_number_flux_diagnostic(
    boundary_velocity_m_s: float, density_upwind_per_m4: float
) -> float:
    """Return outward number flux with the unique convention ``J_N,out >= 0``.

    A negative radius velocity is outward at ``Rmin`` and uses the resolved
    first-cell donor reconstruction.  Positive or zero velocity has no
    external sub-grid source, so its lower-boundary inflow is explicitly zero.
    """

    velocity = float(boundary_velocity_m_s)
    density = float(density_upwind_per_m4)
    if not math.isfinite(velocity):
        raise LowerBoundaryError("boundary velocity must be finite")
    if not math.isfinite(density) or density < 0.0:
        raise LowerBoundaryError("upwind boundary density must be finite and non-negative")
    return max(-velocity * density, 0.0)


def boundary_inventory_diagnostic(
    number_flux_out_m3_s: float, inventory: ParticleInventory
) -> BoundaryInventoryFlux:
    """Convert a positive outward number flux to physical inventory fluxes."""

    number_flux = float(number_flux_out_m3_s)
    if not math.isfinite(number_flux) or number_flux < 0.0:
        raise LowerBoundaryError("outward number flux must be finite and non-negative")
    beta_volume = number_flux * inventory.volume_m3
    beta_moles = number_flux * inventory.beta_moles_mol
    return BoundaryInventoryFlux(
        number_flux_out_m3_s=number_flux,
        beta_volume_flux_out_s=beta_volume,
        beta_mol_flux_out_mol_m3_s=beta_moles,
        b_mol_flux_out_mol_m3_s=number_flux * inventory.b_moles_mol,
    )
