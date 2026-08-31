"""Typed experimental observation records and pseudo-binary mapping helpers."""

from __future__ import annotations

from dataclasses import dataclass

from .composition_mapping import ag_at_fraction_to_xb


@dataclass(frozen=True)
class MatrixAgObservation:
    """Observed matrix Ag atomic fraction with source-role provenance."""

    state: str
    time_h: float
    ag_at_fraction: float
    uncertainty: float
    source: str
    role: str

    @property
    def x_b_observation(self) -> float:
        """Return model xB through the explicit ideal pseudo-binary operator."""

        return ag_at_fraction_to_xb(self.ag_at_fraction)
