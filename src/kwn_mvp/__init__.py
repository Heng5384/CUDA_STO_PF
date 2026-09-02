"""Minimal, mass-conserving effective KWN model for PbTe--Ag2Te.

The package is intentionally independent of the CUDA phase-field executable.
It provides a one-way snapshot layer only; it never advances a phase-field
state or implements GP-to-beta conversion.
"""

from .composition_mapping import ag_at_fraction_to_xb, xb_to_ag_at_fraction
from .characteristic_reference import CharacteristicReferenceSolver
from .solver import KWNSolver, SolverConfig

__all__ = [
    "CharacteristicReferenceSolver",
    "KWNSolver",
    "SolverConfig",
    "ag_at_fraction_to_xb",
    "xb_to_ag_at_fraction",
]
