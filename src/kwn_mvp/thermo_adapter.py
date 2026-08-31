"""Thermodynamic adapters with explicit PF-contract blocking semantics."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from .contract import PFKWNValidationContract, load_validation_contract
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
    contract_path: Optional[str] = None
    contract_hash: Optional[str] = None

    def require_resolved(self) -> None:
        """Reject PF-linked execution until the project's contract is authoritative."""

        if self.status not in {"RESOLVED", "VALIDATION_CONTRACT_RESOLVED"}:
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


@dataclass(frozen=True)
class ValidationContractEquilibriumAdapter:
    """Exact beta equilibrium adapter evaluated from the hash-bound JSON contract.

    This is deliberately limited to the validation-control beta population.
    The current contract contains no GP thermodynamic model, and the adapter
    therefore refuses to turn it into one by approximation.
    """

    contract: PFKWNValidationContract
    temperature_k: float
    planar_reference_xb: Optional[float] = None
    label: str = "PF_KWN_VALIDATION_CONTRACT_V1"
    _planar_solvus_xb: float = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.temperature_k <= 0.0 or not math.isfinite(self.temperature_k):
            raise ValueError("temperature_k must be finite and positive")
        if not math.isclose(self.temperature_k, self.contract.temperature_k, rel_tol=0.0, abs_tol=1.0e-12):
            raise ThermodynamicContractBlocked(
                "validation contract temperature mismatch: "
                f"requested={self.temperature_k:.17g} K contract={self.contract.temperature_k:.17g} K"
            )
        object.__setattr__(
            self,
            "_planar_solvus_xb",
            self.contract.planar_solvus_xb(self.temperature_k),
        )
        if self.planar_reference_xb is not None:
            expected = self._planar_solvus_xb
            if not math.isclose(self.planar_reference_xb, expected, rel_tol=1.0e-12, abs_tol=1.0e-15):
                raise ThermodynamicContractBlocked(
                    "validation contract planar solvus mismatch: "
                    f"configured={self.planar_reference_xb:.17g} contract={expected:.17g}"
                )

    @property
    def contract_hash(self) -> str:
        """Return the canonical hash that must be recorded with KWN output."""

        return self.contract.sha256

    @staticmethod
    def _same_quantity(observed: float, expected: float) -> bool:
        return math.isclose(observed, expected, rel_tol=1.0e-12, abs_tol=1.0e-30)

    def _require_contract_beta_parameters(self, parameters: PopulationParameters) -> None:
        """Reject KWN-side retuning before it can masquerade as PF agreement."""

        if parameters.name != "beta":
            raise ThermodynamicContractBlocked(
                "PF_KWN_VALIDATION_CONTRACT_V1 defines the beta control only; "
                "GP thermodynamics remain outside this adapter"
            )
        expected = {
            "x_b": self.contract.beta_xb,
            "molar_volume_m3_mol": self.contract.vm_beta_m3_mol,
            "diffusivity_m2_s": self.contract.matrix_diffusivity_m2_s(self.temperature_k),
            "gamma_j_m2": self.contract.gamma_j_m2,
            "xeq_infinity": self._planar_solvus_xb,
            "elastic_penalty_j_m3": self.contract.kwn_elastic_penalty_j_m3,
        }
        for field, contract_value in expected.items():
            configured_value = float(getattr(parameters, field))
            if not self._same_quantity(configured_value, contract_value):
                raise ThermodynamicContractBlocked(
                    "PF_KWN_VALIDATION_CONTRACT_V1 parameter mismatch: "
                    f"{field} configured={configured_value:.17g} contract={contract_value:.17g}"
                )

    def equilibrium_xb(
        self, radii_m: NDArray[np.float64], parameters: PopulationParameters
    ) -> NDArray[np.float64]:
        """Return contract-exact curvature equilibrium for the beta population."""

        self._require_contract_beta_parameters(parameters)
        radii = np.asarray(radii_m, dtype=np.float64)
        if np.any(radii <= 0.0) or not np.all(np.isfinite(radii)):
            raise ThermodynamicDomainError("Radii must be finite and positive")
        exponent = (
            (2.0 * self.contract.gamma_j_m2 / radii + self.contract.kwn_elastic_penalty_j_m3)
            * self.contract.vm_beta_m3_mol
            / (self.contract.gas_constant_j_mol_k * self.temperature_k)
        )
        values = self._planar_solvus_xb * np.exp(exponent)
        if not np.all(np.isfinite(values)) or np.any(values <= 0.0) or np.any(values >= 1.0):
            raise ThermodynamicDomainError("contract Gibbs--Thomson equilibrium left (0, 1)")
        return values

    def planar_relative_error(self, parameters: PopulationParameters) -> Optional[float]:
        """Return exact-planar mismatch after enforcing all contract-owned beta fields."""

        self._require_contract_beta_parameters(parameters)
        reference = self._planar_solvus_xb
        return abs(parameters.xeq_infinity - reference) / reference


def build_equilibrium_adapter(
    *,
    mode: str,
    temperature_k: float,
    pf_contract: Optional[PFContractReference] = None,
    planar_reference_xb: Optional[float] = None,
    contract_path: str | Path | None = None,
    expected_contract_hash: Optional[str] = None,
) -> DiluteEquilibriumAdapter | ValidationContractEquilibriumAdapter:
    """Build the selected equilibrium adapter without falling back silently.

    ``pf_contract`` requests a fully shared PF thermodynamic path.  Legacy
    calls remain blocked unless a caller explicitly supplies the frozen
    validation-contract path (directly or through a resolved reference).
    This prevents an implicit switch from historical/mixed runtime metadata to
    the new validation-only control.
    """

    if mode == "pf_contract":
        resolved_path = None if contract_path is None else Path(contract_path)
        resolved_hash = expected_contract_hash
        if pf_contract is not None:
            pf_contract.require_resolved()
            if pf_contract.planar_xb is not None:
                if planar_reference_xb is None:
                    planar_reference_xb = pf_contract.planar_xb
                elif not math.isclose(
                    planar_reference_xb,
                    pf_contract.planar_xb,
                    rel_tol=1.0e-12,
                    abs_tol=1.0e-15,
                ):
                    raise ThermodynamicContractBlocked(
                        "PF contract reference and explicit planar solvus disagree"
                    )
            if pf_contract.contract_path is not None:
                reference_path = Path(pf_contract.contract_path)
                if resolved_path is None:
                    resolved_path = reference_path
                elif resolved_path.resolve() != reference_path.resolve():
                    raise ThermodynamicContractBlocked(
                        "PF contract reference and explicit validation-contract path disagree"
                    )
            if pf_contract.contract_hash is not None:
                if resolved_hash is None:
                    resolved_hash = pf_contract.contract_hash
                elif (
                    not isinstance(resolved_hash, str)
                    or not isinstance(pf_contract.contract_hash, str)
                    or resolved_hash.lower() != pf_contract.contract_hash.lower()
                ):
                    raise ThermodynamicContractBlocked(
                        "PF contract reference and explicit expected hash disagree"
                    )
        if resolved_path is None:
            if pf_contract is None:
                raise ThermodynamicContractBlocked(
                    "pf_contract mode requires an explicit validation-contract path or resolved PF provenance"
                )
            raise ThermodynamicContractBlocked(
                "PF contract is marked resolved but no frozen executable validation-contract path was supplied"
            )
        try:
            validation_contract = load_validation_contract(
                resolved_path,
                expected_hash=resolved_hash,
            )
        except ValueError as exc:
            raise ThermodynamicContractBlocked(f"cannot load validation PF contract: {exc}") from exc
        return ValidationContractEquilibriumAdapter(
            contract=validation_contract,
            temperature_k=temperature_k,
            planar_reference_xb=planar_reference_xb,
        )
    if mode != "approximate_dilute":
        raise ValueError(f"Unsupported beta thermodynamic mode: {mode!r}")
    return DiluteEquilibriumAdapter(
        temperature_k=temperature_k,
        planar_reference_xb=planar_reference_xb,
    )
