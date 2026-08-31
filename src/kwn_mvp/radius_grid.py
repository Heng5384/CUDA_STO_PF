"""Conservative radius-grid definitions for finite-volume KWN transport."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


class RadiusGridError(ValueError):
    """Raised when radius-grid construction or lookup is invalid."""


@dataclass(frozen=True)
class RadiusGrid:
    """Strictly increasing radius-bin edges in metres."""

    edges_m: NDArray[np.float64]

    def __post_init__(self) -> None:
        edges = np.asarray(self.edges_m, dtype=np.float64)
        if edges.ndim != 1 or edges.size < 3:
            raise RadiusGridError("Radius grid needs at least two bins")
        if not np.all(np.isfinite(edges)) or np.any(edges <= 0.0):
            raise RadiusGridError("Radius edges must be finite and strictly positive")
        if not np.all(np.diff(edges) > 0.0):
            raise RadiusGridError("Radius edges must be strictly increasing")
        object.__setattr__(self, "edges_m", edges)

    @classmethod
    def logarithmic(cls, minimum_m: float, maximum_m: float, bins: int) -> "RadiusGrid":
        """Create a logarithmically spaced radius grid in SI units."""

        if minimum_m <= 0.0 or maximum_m <= minimum_m or bins < 2:
            raise RadiusGridError("Require 0 < minimum < maximum and at least two bins")
        return cls(np.geomspace(float(minimum_m), float(maximum_m), int(bins) + 1))

    @property
    def centres_m(self) -> NDArray[np.float64]:
        """Return geometric bin centres, appropriate for a log grid."""

        return np.sqrt(self.edges_m[:-1] * self.edges_m[1:])

    @property
    def widths_m(self) -> NDArray[np.float64]:
        """Return linear bin widths used by the finite-volume representation."""

        return np.diff(self.edges_m)

    @property
    def bins(self) -> int:
        """Return the number of finite-volume cells."""

        return int(self.edges_m.size - 1)

    def bin_index(self, radius_m: float) -> int:
        """Return the bin containing a radius, rejecting unsupported radii."""

        radius = float(radius_m)
        if not self.edges_m[0] <= radius < self.edges_m[-1]:
            raise RadiusGridError(
                f"Radius {radius:.6e} m is outside [{self.edges_m[0]:.6e}, "
                f"{self.edges_m[-1]:.6e}) m"
            )
        return int(np.searchsorted(self.edges_m, radius, side="right") - 1)
