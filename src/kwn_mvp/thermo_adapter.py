"""Thermodynamic adapters with explicit PF-contract blocking semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .populations import PopulationParameters
from .units import gas_constant_j_mol_k


class ThermodynamicContractBlocked(RuntimeError):
    """Raised when a requested PF-linked calculation lacks an authority contract."""


class ThermodynamicDomainError(ValueError):
    """Raised when a curvature-corrected equilibrium leaves physical bounds."""


@dataclass(frozen=True)
class PFContractReference:
    """Provenance-only reference to the audited PF thermodynamic contract."""

    status: str
    source: str
    source_commit: str
    planar_xb: Optional[float] = None

    def require_resolved(self) -> None:
        """Reject PF-linked execution until the project's contract is authoritative."""

        if self.status != "RESOLVED":
            raise ThermodynamicContractBlocked(
                "PF thermodynamic contract is not resolved: "
                f"status={self.status}; source={self.source}. "
                "The KWN model must not silently choose legacy or exact-candidate parameters."
            )


@dataclass(frozen=True)
class DiluteEquilibriumAdapter:
    """Explicit Gibbs--Thomson dilute approximation for a KWN population.

    The adapter is deliberately named ``APPROXIMATE_BETA_THERMO`` for beta.
    It is suitable for numerical qualification and effective-GP feasibility,
    but it cannot be labelled PF-consistent while the PF contract is blocked.
    """

    temperature_k: float
    label: str = "APPROXIMATE_BETA_THERMO"
    planar_reference_xb: Optional[float] = None

    def __post_init__(self) -> None:
        if self.temperature_k <= 0.0:
            raise ValueError("temperature_k must be positive")

    def equilibrium_xb(
        self, radii_m: NDArray[np.float64], parameters: PopulationParameters
    ) -> NDArray[np.float64]:
        """Return curvature-corrected matrix equilibrium composition.

        ``xeq(R)=xeq(infinity)*exp[(2 gamma + R*E_el) Vm/(R R_gas T)]``.
        The optional elastic penalty is a mean-field energy density; no spatial
        elastic field is inferred or reconstructed here.
        """

        radii = np.asarray(radii_m, dtype=np.float64)
        if np.any(radii <= 0.0) or not np.all(np.isfinite(radii)):
            raise ThermodynamicDomainError("Radii must be finite and positive")
        capillary_energy_j_m3 = 2.0 * parameters.gamma_j_m2 / radii
        exponent = (
            (capillary_energy_j_m3 + parameters.elastic_penalty_j_m3)
            * parameters.molar_volume_m3_mol
            / (gas_constant_j_mol_k() * self.temperature_k)
        )
        values = parameters.xeq_infinity * np.exp(exponent)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0) or np.any(values >= 1.0):
            raise ThermodynamicDomainError(
                f"{parameters.name} Gibbs--Thomson equilibrium left (0, 1); "
                "increase the minimum radius or use physically coherent parameters"
            )
        return values

    def planar_relative_error(self, parameters: PopulationParameters) -> Optional[float]:
        """Compare the configured planar solvus with a separately supplied reference."""

        if self.planar_reference_xb is None:
            return None
        if self.planar_reference_xb <= 0.0:
            raise ThermodynamicDomainError("planar_reference_xb must be positive")
        return abs(parameters.xeq_infinity - self.planar_reference_xb) / self.planar_reference_xb


def build_equilibrium_adapter(
    *,
    mode: str,
    temperature_k: float,
    pf_contract: Optional[PFContractReference] = None,
    planar_reference_xb: Optional[float] = None,
) -> DiluteEquilibriumAdapter:
    """Build the selected equilibrium adapter without falling back silently.

    ``pf_contract`` requests a fully shared PF thermodynamic path.  The MVP
    rejects it when audit status is unresolved instead of copying a different
    regular-solution expression.
    """

    if mode == "pf_contract":
        if pf_contract is None:
            raise ThermodynamicContractBlocked("pf_contract mode requires PF contract provenance")
        pf_contract.require_resolved()
        # A resolved adapter must be implemented from the frozen PF contract,
        # not inferred from this fallback.  Keeping this explicit prevents a
        # numerically plausible but different beta thermodynamic model.
        raise ThermodynamicContractBlocked(
            "PF contract is marked RESOLVED but no frozen executable adapter was supplied"
        )
    if mode != "approximate_dilute":
        raise ValueError(f"Unsupported beta thermodynamic mode: {mode!r}")
    return DiluteEquilibriumAdapter(
        temperature_k=temperature_k,
        planar_reference_xb=planar_reference_xb,
    )
