"""Radius-distributed GP-like and beta population state containers."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict

import numpy as np
from numpy.typing import NDArray

from .radius_grid import RadiusGrid


@dataclass(frozen=True)
class PopulationParameters:
    """Physical and effective parameters for one KWN precipitate population."""

    name: str
    x_b: float
    molar_volume_m3_mol: float
    diffusivity_m2_s: float
    gamma_j_m2: float
    xeq_infinity: float
    shape_factor: float = 1.0
    elastic_penalty_j_m3: float = 0.0
    nucleation: Dict[str, Any] = field(default_factory=lambda: {"mode": "off"})

    def __post_init__(self) -> None:
        if self.name not in {"g", "beta"}:
            raise ValueError(f"Only g and beta populations are supported; got {self.name!r}")
        if not 0.0 < self.x_b <= 1.0:
            raise ValueError(f"{self.name}.x_b must be in (0, 1]")
        if self.molar_volume_m3_mol <= 0.0:
            raise ValueError(f"{self.name}.molar_volume_m3_mol must be positive")
        if self.diffusivity_m2_s < 0.0:
            raise ValueError(f"{self.name}.diffusivity_m2_s must be non-negative")
        if self.gamma_j_m2 < 0.0:
            raise ValueError(f"{self.name}.gamma_j_m2 must be non-negative")
        if not 0.0 < self.xeq_infinity < 1.0:
            raise ValueError(f"{self.name}.xeq_infinity must be in (0, 1)")
        if self.shape_factor <= 0.0:
            raise ValueError(f"{self.name}.shape_factor must be positive")


@dataclass
class Population:
    """Number-density-per-radius state for one population.

    ``number_density_per_m4`` is n(R), with units m^-4 so that integrating
    against a radius interval returns number density in m^-3.
    """

    parameters: PopulationParameters
    grid: RadiusGrid
    number_density_per_m4: NDArray[np.float64]

    def __post_init__(self) -> None:
        density = np.asarray(self.number_density_per_m4, dtype=np.float64)
        if density.shape != (self.grid.bins,):
            raise ValueError(f"{self.parameters.name} density shape must match grid bins")
        if not np.all(np.isfinite(density)) or np.any(density < 0.0):
            raise ValueError(f"{self.parameters.name} density must be finite and non-negative")
        self.number_density_per_m4 = density

    @classmethod
    def empty(cls, parameters: PopulationParameters, grid: RadiusGrid) -> "Population":
        """Create a population with no particles."""

        return cls(parameters, grid, np.zeros(grid.bins, dtype=np.float64))

    def copy(self) -> "Population":
        """Return an independent copy suitable for deterministic restart tests."""

        return Population(self.parameters, self.grid, self.number_density_per_m4.copy())

    def add_number_at_radius(self, radius_m: float, number_density_m3: float) -> None:
        """Add a finite number density to the bin containing ``radius_m``."""

        if number_density_m3 < 0.0:
            raise ValueError("Cannot add a negative number density")
        index = self.grid.bin_index(radius_m)
        self.number_density_per_m4[index] += number_density_m3 / self.grid.widths_m[index]

    def number_density_m3(self) -> float:
        """Return total number density in m^-3."""

        return float(np.sum(self.number_density_per_m4 * self.grid.widths_m))

    def volume_fraction(self) -> float:
        """Return particle volume per unit material volume (dimensionless)."""

        radii = self.grid.centres_m
        sphere_volume = (4.0 * np.pi / 3.0) * radii**3
        return float(np.sum(self.number_density_per_m4 * self.grid.widths_m * sphere_volume))

    def b_inventory_mol_m3(self) -> float:
        """Return pseudo-binary B inventory in mol m^-3 for this population."""

        return self.volume_fraction() * self.parameters.x_b / self.parameters.molar_volume_m3_mol

    def mean_radius_m(self) -> float:
        """Return number-weighted mean radius in metres, or zero if empty."""

        weights = self.number_density_per_m4 * self.grid.widths_m
        total = float(np.sum(weights))
        if total == 0.0:
            return 0.0
        return float(np.sum(weights * self.grid.centres_m) / total)

    def mean_radius_cubed_m3(self) -> float:
        """Return number-weighted mean R^3 in m^3, or zero if empty."""

        weights = self.number_density_per_m4 * self.grid.widths_m
        total = float(np.sum(weights))
        if total == 0.0:
            return 0.0
        return float(np.sum(weights * self.grid.centres_m**3) / total)

    def specific_surface_area_m_inv(self) -> float:
        """Return spherical-equivalent interfacial area density in m^-1."""

        radii = self.grid.centres_m
        return float(np.sum(self.number_density_per_m4 * self.grid.widths_m * 4.0 * np.pi * radii**2))
