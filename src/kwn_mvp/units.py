"""SI-unit constants and explicit conversion helpers used by the KWN MVP."""

from __future__ import annotations


def gas_constant_j_mol_k() -> float:
    """Return the molar gas constant in J mol^-1 K^-1."""

    return 8.31446261815324


def boltzmann_constant_j_k() -> float:
    """Return the Boltzmann constant in J K^-1."""

    return 1.380649e-23


def nm_to_m(value_nm: float) -> float:
    """Convert a length from nm to m."""

    return float(value_nm) * 1.0e-9


def m_to_nm(value_m: float) -> float:
    """Convert a length from m to nm."""

    return float(value_m) * 1.0e9


def hours_to_seconds(value_h: float) -> float:
    """Convert time from hours to seconds."""

    return float(value_h) * 3600.0


def seconds_to_hours(value_s: float) -> float:
    """Convert time from seconds to hours."""

    return float(value_s) / 3600.0
