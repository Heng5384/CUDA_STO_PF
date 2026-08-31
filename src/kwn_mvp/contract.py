"""Hash-bound Python view of the PF--KWN validation thermodynamic contract.

The canonical JSON file is the numerical authority for the *validation-only*
96-cube control.  This module deliberately evaluates every quantity from that
file instead of maintaining a second hand-copied set of thermodynamic or
kinetic constants in Python.  It must not be used to relabel historical PF
production, whose as-run authority remains separate.
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence


VALIDATION_CONTRACT_SCHEMA = "PF_KWN_VALIDATION_CONTRACT_V1"
DEFAULT_VALIDATION_CONTRACT_PATH = (
    Path(__file__).resolve().parents[2] / "contracts" / "pf_kwn_validation_contract_v1.json"
)
_PROVENANCE_KEYS = (
    "value",
    "unit",
    "source_file",
    "source_line_or_symbol",
    "authority_role",
    "used_by_pf",
    "used_by_kwn",
    "assumption_status",
)


class ValidationContractError(ValueError):
    """Raised when a validation contract is malformed or semantically invalid."""


class ValidationContractHashMismatch(ValidationContractError):
    """Raised when a caller's expected hash differs from the canonical JSON hash."""


def _reject_nonfinite_json_constant(value: str) -> None:
    """Reject JSON NaN/Infinity spellings instead of accepting non-portable values."""

    raise ValidationContractError(f"contract contains non-finite JSON literal {value!r}")


def canonical_contract_json(data: Mapping[str, Any]) -> str:
    """Return the exact canonical JSON representation used for the contract hash."""

    try:
        return json.dumps(
            data,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
            allow_nan=False,
        )
    except (TypeError, ValueError) as exc:
        raise ValidationContractError(f"contract cannot be canonically encoded: {exc}") from exc


def validation_contract_hash(data: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of canonical validation-contract bytes."""

    return hashlib.sha256(canonical_contract_json(data).encode("utf-8")).hexdigest()


def _mapping(value: Any, context: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ValidationContractError(f"{context} must be an object")
    return value


def _finite_number(value: Any, context: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValidationContractError(f"{context} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValidationContractError(f"{context} must be finite")
    return result


def _positive_number(value: Any, context: str, *, allow_zero: bool = False) -> float:
    result = _finite_number(value, context)
    if result < 0.0 or (not allow_zero and result == 0.0):
        comparator = "non-negative" if allow_zero else "positive"
        raise ValidationContractError(f"{context} must be {comparator}")
    return result


def _finite_vector(value: Any, context: str, *, length: int | None = None) -> tuple[float, ...]:
    if not isinstance(value, Sequence) or isinstance(value, (str, bytes)):
        raise ValidationContractError(f"{context} must be a numeric array")
    result = tuple(_finite_number(item, f"{context}[{index}]") for index, item in enumerate(value))
    if length is not None and len(result) != length:
        raise ValidationContractError(f"{context} must have length {length}")
    return result


def _walk_finite_json_numbers(value: Any, context: str = "contract") -> None:
    """Reject accidental non-finite values nested inside provenance payloads."""

    if isinstance(value, float) and not math.isfinite(value):
        raise ValidationContractError(f"{context} contains a non-finite number")
    if isinstance(value, Mapping):
        for key, child in value.items():
            _walk_finite_json_numbers(child, f"{context}.{key}")
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk_finite_json_numbers(child, f"{context}[{index}]")


@dataclass(frozen=True)
class PFKWNValidationContract:
    """The parsed, canonicalized validation-only PF--KWN numerical contract."""

    path: Path
    data: Mapping[str, Any]
    sha256: str

    def _provenance_record(self, dotted_path: str) -> Mapping[str, Any]:
        current: Any = self.data
        for component in dotted_path.split("."):
            current = _mapping(current, dotted_path)
            if component not in current:
                raise ValidationContractError(f"missing contract field {dotted_path}")
            current = current[component]
        record = _mapping(current, dotted_path)
        missing = [key for key in _PROVENANCE_KEYS if key not in record]
        if missing:
            raise ValidationContractError(
                f"{dotted_path} lacks required provenance field(s): {', '.join(missing)}"
            )
        return record

    def value(self, dotted_path: str) -> Any:
        """Return a provenance-wrapped contract payload by dotted field path."""

        return self._provenance_record(dotted_path)["value"]

    def number(self, dotted_path: str, *, positive: bool = False, allow_zero: bool = True) -> float:
        """Return one finite, provenance-wrapped numerical contract value."""

        value = self.value(dotted_path)
        if positive:
            return _positive_number(value, dotted_path, allow_zero=allow_zero)
        return _finite_number(value, dotted_path)

    @property
    def temperature_k(self) -> float:
        return self.number("temperature.temperature_K", positive=True)

    @property
    def gas_constant_j_mol_k(self) -> float:
        return self.number("thermodynamics.gas_constant_j_mol_k", positive=True)

    @property
    def delta_h_j_mol(self) -> float:
        return self.number("thermodynamics.delta_H_J_mol")

    @property
    def delta_s_j_mol_k(self) -> float:
        return self.number("thermodynamics.delta_S_J_mol_K")

    @property
    def gamma_j_m2(self) -> float:
        return self.number("interface.gamma_J_m2", positive=True, allow_zero=True)

    @property
    def vm_alpha_m3_mol(self) -> float:
        return self.number("volumes.Vm_alpha_m3_mol", positive=True)

    @property
    def vm_beta_m3_mol(self) -> float:
        return self.number("volumes.Vm_beta_m3_mol", positive=True)

    @property
    def beta_xb(self) -> float:
        return self.number("composition.v_B", positive=True)

    @property
    def kwn_elastic_penalty_j_m3(self) -> float:
        return self.number("elasticity.kwn_elastic_penalty_J_m3", positive=True, allow_zero=True)

    def _standard_state_records(self) -> Mapping[str, Any]:
        return _mapping(
            self.value("thermodynamics.standard_state_coefficients"),
            "thermodynamics.standard_state_coefficients.value",
        )

    def standard_state_j_mol(self, name: str, temperature_k: float) -> float:
        """Evaluate one SGTE standard-state polynomial declared in the JSON."""

        temperature = _positive_number(temperature_k, "temperature_K")
        record = _mapping(self._standard_state_records().get(name), f"standard state {name}")
        transition = _positive_number(record.get("transition_K"), f"standard state {name}.transition_K")
        branch_name = "low" if temperature < transition else "high"
        coefficients = _finite_vector(record.get(branch_name), f"standard state {name}.{branch_name}", length=7)
        a0, a1, a_log, a2, a3, a_power, power = coefficients
        tail = 0.0 if a_power == 0.0 else a_power * math.pow(temperature, power)
        return (
            a0
            + a1 * temperature
            + a_log * temperature * math.log(temperature)
            + a2 * temperature * temperature
            + a3 * temperature * temperature * temperature
            + tail
        )

    def _compound_standard_state_j_mol(self, name: str, temperature_k: float) -> float:
        record = _mapping(self._standard_state_records().get(name), f"compound standard state {name}")
        if "base" in record:
            base = _finite_vector(record["base"], f"compound standard state {name}.base", length=2)
            terms = record.get("terms")
            if not isinstance(terms, Sequence) or isinstance(terms, (str, bytes)):
                raise ValidationContractError(f"compound standard state {name}.terms must be an array")
            return base[0] + base[1] * temperature_k + sum(
                self.standard_state_j_mol(str(term), temperature_k) for term in terms
            )
        base = _finite_vector(
            record.get("base_per_atom"), f"compound standard state {name}.base_per_atom", length=2
        )
        terms = record.get("terms")
        weights = record.get("atom_weights")
        if (
            not isinstance(terms, Sequence)
            or isinstance(terms, (str, bytes))
            or not isinstance(weights, Sequence)
            or isinstance(weights, (str, bytes))
        ):
            raise ValidationContractError(f"compound standard state {name} requires terms and atom_weights arrays")
        if len(terms) != len(weights) or not terms:
            raise ValidationContractError(f"compound standard state {name} has inconsistent terms/weights")
        multiplier = _finite_number(record.get("molecular_multiplier"), f"compound standard state {name}.molecular_multiplier")
        atom_energy = base[0] + base[1] * temperature_k + sum(
            _finite_number(weight, f"compound standard state {name}.atom_weights[{index}]")
            * self.standard_state_j_mol(str(term), temperature_k)
            for index, (term, weight) in enumerate(zip(terms, weights))
        )
        return multiplier * atom_energy

    def g_pbte_j_mol(self, temperature_k: float) -> float:
        """Return the PbTe pseudo-component standard Gibbs energy."""

        return self._compound_standard_state_j_mol("G_PbTe", _positive_number(temperature_k, "temperature_K"))

    def g_ag2te_j_mol(self, temperature_k: float) -> float:
        """Return the Ag2Te pseudo-component standard Gibbs energy."""

        return self._compound_standard_state_j_mol("G_Ag2Te", _positive_number(temperature_k, "temperature_K"))

    @staticmethod
    def _clamp_fraction(x_b: float) -> float:
        value = _finite_number(x_b, "xB")
        return min(max(value, 1.0e-12), 1.0 - 1.0e-12)

    def regular_solution_parameter_j_mol(self, temperature_k: float) -> float:
        """Return the declared regular-solution interaction parameter L(T)."""

        temperature = _positive_number(temperature_k, "temperature_K")
        return self.delta_h_j_mol - self.delta_s_j_mol_k * temperature

    def _convex_settings(self) -> tuple[bool, float, float]:
        record = _mapping(
            self.value("thermodynamics.convex_extrapolation"),
            "thermodynamics.convex_extrapolation.value",
        )
        enabled = record.get("enabled")
        if not isinstance(enabled, bool):
            raise ValidationContractError("convex_extrapolation.enabled must be boolean")
        limit = _positive_number(record.get("xB_limit"), "convex_extrapolation.xB_limit")
        if limit >= 1.0:
            raise ValidationContractError("convex_extrapolation.xB_limit must lie below one")
        penalty = _positive_number(
            record.get("penalty_J_mol"), "convex_extrapolation.penalty_J_mol", allow_zero=True
        )
        return enabled, limit, penalty

    def _physical_mu_a_j_mol(self, temperature_k: float, x_b: float) -> float:
        x = self._clamp_fraction(x_b)
        return (
            self.g_pbte_j_mol(temperature_k)
            + self.gas_constant_j_mol_k * temperature_k * math.log(1.0 - x)
            + self.regular_solution_parameter_j_mol(temperature_k) * x * x
        )

    def _physical_mu_b_j_mol(self, temperature_k: float, x_b: float) -> float:
        x = self._clamp_fraction(x_b)
        return (
            self.g_ag2te_j_mol(temperature_k)
            + self.gas_constant_j_mol_k * temperature_k * math.log(x)
            + self.regular_solution_parameter_j_mol(temperature_k) * (1.0 - x) * (1.0 - x)
        )

    def _physical_dmu_a_dx_j_mol(self, temperature_k: float, x_b: float) -> float:
        x = self._clamp_fraction(x_b)
        return (
            -self.gas_constant_j_mol_k * temperature_k / (1.0 - x)
            + 2.0 * self.regular_solution_parameter_j_mol(temperature_k) * x
        )

    def _physical_dmu_b_dx_j_mol(self, temperature_k: float, x_b: float) -> float:
        x = self._clamp_fraction(x_b)
        return (
            self.gas_constant_j_mol_k * temperature_k / x
            - 2.0 * self.regular_solution_parameter_j_mol(temperature_k) * (1.0 - x)
        )

    def _convex_mu(
        self,
        *,
        temperature_k: float,
        x_b: float,
        physical_mu: float,
        physical_slope: float,
    ) -> float:
        enabled, limit, penalty = self._convex_settings()
        if not enabled or x_b <= limit:
            return physical_mu
        delta = x_b - limit
        return physical_mu + physical_slope * delta + penalty * delta * delta

    def chemical_potential_a_j_mol(self, temperature_k: float, x_b: float) -> float:
        """Return the PbTe chemical potential including declared convex handling."""

        temperature = _positive_number(temperature_k, "temperature_K")
        raw_x = _finite_number(x_b, "xB")
        enabled, limit, _ = self._convex_settings()
        if enabled and raw_x > limit:
            return self._convex_mu(
                temperature_k=temperature,
                x_b=raw_x,
                physical_mu=self._physical_mu_a_j_mol(temperature, limit),
                physical_slope=self._physical_dmu_a_dx_j_mol(temperature, limit),
            )
        return self._physical_mu_a_j_mol(temperature, raw_x)

    def chemical_potential_b_j_mol(self, temperature_k: float, x_b: float) -> float:
        """Return the Ag2Te chemical potential including declared convex handling."""

        temperature = _positive_number(temperature_k, "temperature_K")
        raw_x = _finite_number(x_b, "xB")
        enabled, limit, _ = self._convex_settings()
        if enabled and raw_x > limit:
            return self._convex_mu(
                temperature_k=temperature,
                x_b=raw_x,
                physical_mu=self._physical_mu_b_j_mol(temperature, limit),
                physical_slope=self._physical_dmu_b_dx_j_mol(temperature, limit),
            )
        return self._physical_mu_b_j_mol(temperature, raw_x)

    def dchemical_potential_a_dx_j_mol(self, temperature_k: float, x_b: float) -> float:
        """Return d(mu_PbTe)/dx for the JSON-declared thermodynamic branch."""

        temperature = _positive_number(temperature_k, "temperature_K")
        raw_x = _finite_number(x_b, "xB")
        enabled, limit, penalty = self._convex_settings()
        if enabled and raw_x > limit:
            return self._physical_dmu_a_dx_j_mol(temperature, limit) + 2.0 * penalty * (raw_x - limit)
        return self._physical_dmu_a_dx_j_mol(temperature, raw_x)

    def dchemical_potential_b_dx_j_mol(self, temperature_k: float, x_b: float) -> float:
        """Return d(mu_Ag2Te)/dx for the JSON-declared thermodynamic branch."""

        temperature = _positive_number(temperature_k, "temperature_K")
        raw_x = _finite_number(x_b, "xB")
        enabled, limit, penalty = self._convex_settings()
        if enabled and raw_x > limit:
            return self._physical_dmu_b_dx_j_mol(temperature, limit) + 2.0 * penalty * (raw_x - limit)
        return self._physical_dmu_b_dx_j_mol(temperature, raw_x)

    def g_alpha_j_mol(self, temperature_k: float, x_b: float) -> float:
        """Return the declared alpha regular-solution Gibbs energy G_alpha(x, T)."""

        temperature = _positive_number(temperature_k, "temperature_K")
        x = self._clamp_fraction(x_b)
        return (
            (1.0 - x) * self.g_pbte_j_mol(temperature)
            + x * self.g_ag2te_j_mol(temperature)
            + self.gas_constant_j_mol_k
            * temperature
            * ((1.0 - x) * math.log(1.0 - x) + x * math.log(x))
            + self.regular_solution_parameter_j_mol(temperature) * x * (1.0 - x)
        )

    def dg_alpha_dx_j_mol(self, temperature_k: float, x_b: float) -> float:
        """Return dG_alpha/dx, i.e. mu_Ag2Te - mu_PbTe."""

        return self.chemical_potential_b_j_mol(temperature_k, x_b) - self.chemical_potential_a_j_mol(
            temperature_k, x_b
        )

    def d2g_alpha_dx2_j_mol(self, temperature_k: float, x_b: float) -> float:
        """Return d²G_alpha/dx² using the same branch as the chemical potentials."""

        return self.dchemical_potential_b_dx_j_mol(
            temperature_k, x_b
        ) - self.dchemical_potential_a_dx_j_mol(temperature_k, x_b)

    def mu_alpha_j_mol(self, temperature_k: float, x_b: float) -> float:
        """Alias for the alpha composition chemical potential dG_alpha/dx."""

        return self.dg_alpha_dx_j_mol(temperature_k, x_b)

    def dmu_alpha_dx_j_mol(self, temperature_k: float, x_b: float) -> float:
        """Alias for d²G_alpha/dx² used by the PF thermodynamic factor."""

        return self.d2g_alpha_dx2_j_mol(temperature_k, x_b)

    def beta_driving_force_j_mol(self, temperature_k: float, x_b: float) -> float:
        """Return the planar beta root convention explicitly declared in the contract."""

        temperature = _positive_number(temperature_k, "temperature_K")
        x = self._clamp_fraction(x_b)
        return self.gas_constant_j_mol_k * temperature * math.log(x) + self.regular_solution_parameter_j_mol(
            temperature
        ) * (1.0 - x) * (1.0 - x)

    def planar_solvus_xb(self, temperature_k: float, *, iterations: int = 200) -> float:
        """Solve the declared beta-driving-force root by deterministic bisection."""

        temperature = _positive_number(temperature_k, "temperature_K")
        if iterations <= 0:
            raise ValidationContractError("solvus bisection iteration count must be positive")
        lower = 1.0e-12
        upper = 0.5
        lower_value = self.beta_driving_force_j_mol(temperature, lower)
        upper_value = self.beta_driving_force_j_mol(temperature, upper)
        if not lower_value < 0.0 < upper_value:
            raise ValidationContractError(
                "validation solvus root is not bracketed on the declared (1e-12, 0.5) interval"
            )
        for _ in range(iterations):
            middle = 0.5 * (lower + upper)
            if self.beta_driving_force_j_mol(temperature, middle) > 0.0:
                upper = middle
            else:
                lower = middle
        return 0.5 * (lower + upper)

    def matrix_diffusivity_m2_s(self, temperature_k: float) -> float:
        """Return D_alpha(T), converting the declared cm²/s prefactor to m²/s."""

        temperature = _positive_number(temperature_k, "temperature_K")
        d0_cm2_s = self.number("kinetics.D0_cm2_s", positive=True)
        activation = self.number("kinetics.activation_energy_J_mol", positive=True)
        return d0_cm2_s * math.exp(-activation / (self.gas_constant_j_mol_k * temperature)) * 1.0e-4

    def curvature_equilibrium_xb(
        self,
        temperature_k: float,
        radius_m: float,
        *,
        elastic_penalty_j_m3: float | None = None,
    ) -> float:
        """Return same-contract Gibbs--Thomson equilibrium for a beta sphere."""

        temperature = _positive_number(temperature_k, "temperature_K")
        radius = _positive_number(radius_m, "radius_m")
        penalty = (
            self.kwn_elastic_penalty_j_m3
            if elastic_penalty_j_m3 is None
            else _finite_number(elastic_penalty_j_m3, "elastic_penalty_j_m3")
        )
        exponent = (
            (2.0 * self.gamma_j_m2 / radius + penalty)
            * self.vm_beta_m3_mol
            / (self.gas_constant_j_mol_k * temperature)
        )
        try:
            result = self.planar_solvus_xb(temperature) * math.exp(exponent)
        except OverflowError as exc:
            raise ValidationContractError("curvature equilibrium overflowed") from exc
        if not 0.0 < result < 1.0 or not math.isfinite(result):
            raise ValidationContractError("curvature equilibrium left the physical composition interval")
        return result

    def h_of_phi(self, phi: float) -> float:
        """Evaluate the PF quintic storage interpolation without clipping ``phi``.

        ``phase_functions.h`` evaluates the polynomial on the raw phase field;
        callers that need bounded observational support must apply their own
        explicitly declared threshold or clamp.  Clamping here would make the
        Python ledger disagree with the PF storage equation for overshoots.
        """

        value = _finite_number(phi, "phi")
        return 6.0 * value**5 - 15.0 * value**4 + 10.0 * value**3


def _validate_contract_shape(data: Mapping[str, Any]) -> None:
    """Validate the closed schema subset required by the Python/PF parity path."""

    _walk_finite_json_numbers(data)
    if data.get("schema_version") != VALIDATION_CONTRACT_SCHEMA:
        raise ValidationContractError(
            f"unexpected validation contract schema {data.get('schema_version')!r}; "
            f"expected {VALIDATION_CONTRACT_SCHEMA!r}"
        )
    if data.get("name") != VALIDATION_CONTRACT_SCHEMA:
        raise ValidationContractError("validation contract name must match its schema version")
    if data.get("historical_as_run_claim") is not False:
        raise ValidationContractError(
            "the validation contract must explicitly declare historical_as_run_claim=false"
        )
    canonicalization = _mapping(data.get("canonicalization"), "canonicalization")
    if canonicalization.get("algorithm") != "SHA-256 over UTF-8 JSON with sort_keys=true, separators=(',', ':'), allow_nan=false":
        raise ValidationContractError("validation contract canonicalization algorithm is not the frozen SHA-256 rule")
    if canonicalization.get("float_precision") != "IEEE-754 binary64":
        raise ValidationContractError("validation contract must declare IEEE-754 binary64")

    # Instantiate only after the structural gate, then force each numerical
    # quantity and coefficient family needed by the shared equations to parse.
    provisional = PFKWNValidationContract(path=Path("<unbound>"), data=data, sha256="")
    for path in (
        "temperature.temperature_K",
        "composition.v_A",
        "composition.v_B",
        "thermodynamics.gas_constant_j_mol_k",
        "thermodynamics.standard_state_coefficients",
        "thermodynamics.delta_H_J_mol",
        "thermodynamics.delta_S_J_mol_K",
        "thermodynamics.regular_solution",
        "thermodynamics.solvus_root_equation",
        "thermodynamics.convex_extrapolation",
        "kinetics.D0_cm2_s",
        "kinetics.activation_energy_J_mol",
        "interface.gamma_J_m2",
        "interface.interpolation_h_phi",
        "volumes.Vm_alpha_m3_mol",
        "volumes.Vm_beta_m3_mol",
        "volumes.dVm_alpha_dxB_m3_mol",
        "elasticity.kwn_elastic_penalty_J_m3",
    ):
        provisional._provenance_record(path)
    # Exercise every generic evaluator once.  This catches malformed standard
    # state arrays or compound mappings before a numerical KWN run begins.
    temperature = provisional.temperature_k
    provisional.g_pbte_j_mol(temperature)
    provisional.g_ag2te_j_mol(temperature)
    provisional.g_alpha_j_mol(temperature, 0.01)
    provisional.d2g_alpha_dx2_j_mol(temperature, 0.01)
    provisional.planar_solvus_xb(temperature)
    provisional.matrix_diffusivity_m2_s(temperature)
    provisional.curvature_equilibrium_xb(temperature, 10.0e-9)
    provisional.h_of_phi(0.5)


def load_validation_contract(
    path: str | Path = DEFAULT_VALIDATION_CONTRACT_PATH,
    *,
    expected_hash: str | None = None,
) -> PFKWNValidationContract:
    """Load, validate, and hash the canonical validation-only JSON contract.

    ``expected_hash`` is intentionally an equality gate rather than a warning:
    callers that already recorded a contract hash cannot continue with a
    different contract payload.
    """

    contract_path = Path(path)
    try:
        data = json.loads(
            contract_path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json_constant
        )
    except FileNotFoundError as exc:
        raise ValidationContractError(f"validation contract does not exist: {contract_path}") from exc
    except json.JSONDecodeError as exc:
        raise ValidationContractError(
            f"invalid validation-contract JSON {contract_path}: line {exc.lineno}, column {exc.colno}: {exc.msg}"
        ) from exc
    data_mapping = _mapping(data, "validation contract root")
    _validate_contract_shape(data_mapping)
    digest = validation_contract_hash(data_mapping)
    if expected_hash is not None:
        if not isinstance(expected_hash, str) or len(expected_hash) != 64:
            raise ValidationContractHashMismatch("expected validation contract hash must be a 64-character SHA-256")
        if digest != expected_hash.lower():
            raise ValidationContractHashMismatch(
                f"validation contract hash mismatch: expected={expected_hash.lower()} actual={digest}"
            )
    return PFKWNValidationContract(path=contract_path.resolve(), data=data_mapping, sha256=digest)


# Short alias for callers that do not need to spell out the PF/KWN domain.
ValidationContract = PFKWNValidationContract
