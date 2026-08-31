"""Explicit nucleation-source modes for the effective KWN populations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict

from .populations import PopulationParameters
from .units import boltzmann_constant_j_k, gas_constant_j_mol_k


@dataclass(frozen=True)
class NucleationResult:
    """Source-rate result and transparent intermediate terms."""

    rate_m3_s: float
    radius_m: float
    critical_radius_m: float | None
    barrier_j: float | None
    driving_energy_j_m3: float | None


def _require_number(config: Dict[str, Any], name: str, *, positive: bool = False) -> float:
    """Read a numeric source parameter with a clear error message."""

    value = config.get(name)
    if not isinstance(value, (float, int)):
        raise ValueError(f"Nucleation parameter '{name}' must be numeric")
    result = float(value)
    if positive and result <= 0.0:
        raise ValueError(f"Nucleation parameter '{name}' must be positive")
    return result


def nucleation_rate(
    *,
    population: PopulationParameters,
    matrix_xb: float,
    temperature_k: float,
    time_s: float,
) -> NucleationResult:
    """Evaluate the configured ``off``, ``prescribed_source``, or effective-CNT source.

    No source transfers mass from g to beta.  Both populations can only draw
    material from the shared matrix ledger.
    """

    config = population.nucleation
    mode = str(config.get("mode", "off"))
    if mode == "off":
        return NucleationResult(0.0, 0.0, None, None, None)
    if mode == "prescribed_source":
        peak_rate = _require_number(config, "peak_rate_m3_s", positive=True)
        centre = _require_number(config, "centre_time_s")
        width = _require_number(config, "width_s", positive=True)
        radius = _require_number(config, "radius_m", positive=True)
        exponent = -0.5 * ((time_s - centre) / width) ** 2
        return NucleationResult(peak_rate * math.exp(exponent), radius, None, None, None)
    if mode != "effective_cnt":
        raise ValueError(f"Unsupported nucleation mode {mode!r} for {population.name}")
    if population.name != "g":
        raise ValueError("effective_cnt is only enabled for g in the MVP; beta nucleation is off")
    if matrix_xb <= population.xeq_infinity:
        return NucleationResult(0.0, 0.0, None, None, 0.0)
    site_density = _require_number(config, "site_density_m3", positive=True)
    zeldovich = _require_number(config, "zeldovich_factor", positive=True)
    attachment = _require_number(config, "attachment_prefactor_s_inv", positive=True)
    radius_floor = _require_number(config, "radius_floor_m", positive=True)
    log_supersaturation = math.log(matrix_xb / population.xeq_infinity)
    driving = (
        gas_constant_j_mol_k() * temperature_k * log_supersaturation / population.molar_volume_m3_mol
        - population.elastic_penalty_j_m3
    )
    if driving <= 0.0:
        return NucleationResult(0.0, 0.0, None, None, driving)
    critical_radius = 2.0 * population.gamma_j_m2 / driving
    barrier = 16.0 * math.pi * population.gamma_j_m2**3 / (3.0 * driving**2)
    exponent = -barrier / (boltzmann_constant_j_k() * temperature_k)
    rate = site_density * zeldovich * attachment * math.exp(max(exponent, -745.0))
    return NucleationResult(rate, max(critical_radius, radius_floor), critical_radius, barrier, driving)
