"""Observation operator between measured Ag atomic fraction and pseudo-binary xB."""

from __future__ import annotations



class CompositionMappingError(ValueError):
    """Raised when a composition cannot be represented by the pseudo-binary map."""


def ag_at_fraction_to_xb(y_ag: float) -> float:
    """Map total Ag atomic fraction to pseudo-binary Ag2Te fraction xB.

    For ``(PbTe)_(1-x) (Ag2Te)_x``, the observation operator is
    ``x = 2*y_Ag/(2-y_Ag)``.  It is a model observation operator, not an
    assertion that APT matrix chemistry lies exactly on the ideal tie line.
    """

    y_value = float(y_ag)
    if not 0.0 <= y_value < 1.0:
        raise CompositionMappingError(f"Ag atomic fraction must lie in [0, 1); got {y_value}")
    return 2.0 * y_value / (2.0 - y_value)


def xb_to_ag_at_fraction(x_b: float) -> float:
    """Map pseudo-binary Ag2Te fraction xB to total Ag atomic fraction."""

    x_value = float(x_b)
    if not 0.0 <= x_value <= 1.0:
        raise CompositionMappingError(f"Pseudo-binary xB must lie in [0, 1]; got {x_value}")
    return 2.0 * x_value / (2.0 + x_value)
