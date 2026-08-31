"""Diffusion-controlled spherical growth and dissolution law."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from .populations import PopulationParameters


class GrowthLawError(ValueError):
    """Raised when a growth-law parameter or composition is physically invalid."""


def growth_rate_m_s(
    *,
    radii_m: NDArray[np.float64],
    matrix_xb: float,
    equilibrium_xb: NDArray[np.float64],
    parameters: PopulationParameters,
) -> NDArray[np.float64]:
    """Return diffusion-controlled spherical growth rate in m s^-1.

    ``Rdot=(D/R)*(x_alpha-xeq)/(x_p-xeq)*F_shape``.  A negative rate is a
    dissolution flux and is returned to the matrix through the independent
    inventory ledger after the conservative size-space update.
    """

    radii = np.asarray(radii_m, dtype=np.float64)
    xeq = np.asarray(equilibrium_xb, dtype=np.float64)
    if radii.shape != xeq.shape:
        raise GrowthLawError("radii and equilibrium arrays must have equal shapes")
    denominator = parameters.x_b - xeq
    if np.any(denominator <= 0.0):
        raise GrowthLawError(
            f"{parameters.name} composition must exceed curvature equilibrium for growth law"
        )
    return (
        parameters.diffusivity_m2_s
        / radii
        * (float(matrix_xb) - xeq)
        / denominator
        * parameters.shape_factor
    )
