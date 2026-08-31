#!/usr/bin/env python3
"""Generate the sole C/CUDA numerical view of the validation contract.

The JSON itself is the authority for the 96-cube validation control.  This
script deliberately emits a small, host/CUDA compatible header instead of
maintaining a second manually edited set of thermodynamic constants.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_CONTRACT = ROOT / "contracts" / "pf_kwn_validation_contract_v1.json"
DEFAULT_HEADER = ROOT / "generated" / "pf_kwn_validation_contract_v1.h"


class ContractGenerationError(ValueError):
    """Raised when the validation contract cannot be safely generated."""


def _reject_nonfinite(value: str) -> None:
    raise ContractGenerationError(f"contract contains non-finite JSON literal {value!r}")


def canonical_json(data: Mapping[str, Any]) -> str:
    """Return the exact JSON byte representation bound into the header."""

    return json.dumps(
        data,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )


def contract_hash(data: Mapping[str, Any]) -> str:
    """Return the SHA-256 digest of the canonical validation contract."""

    return hashlib.sha256(canonical_json(data).encode("utf-8")).hexdigest()


def load_contract(path: Path) -> dict[str, Any]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite)
    except FileNotFoundError as exc:
        raise ContractGenerationError(f"missing contract: {path}") from exc
    except json.JSONDecodeError as exc:
        raise ContractGenerationError(f"invalid JSON in {path}: {exc}") from exc
    if not isinstance(data, dict):
        raise ContractGenerationError("contract root must be a JSON object")
    if data.get("schema_version") != "PF_KWN_VALIDATION_CONTRACT_V1":
        raise ContractGenerationError("unexpected validation contract schema")
    if data.get("historical_as_run_claim") is not False:
        raise ContractGenerationError("validation contract must explicitly reject historical-as-run claim")
    return data


def value(data: Mapping[str, Any], dotted_path: str) -> Any:
    """Read a provenance-wrapped scalar from the canonical contract."""

    current: Any = data
    for component in dotted_path.split("."):
        if not isinstance(current, Mapping) or component not in current:
            raise ContractGenerationError(f"missing contract field {dotted_path}")
        current = current[component]
    if not isinstance(current, Mapping) or "value" not in current:
        raise ContractGenerationError(f"{dotted_path} must be a provenance-wrapped value")
    return current["value"]


def finite_float(data: Mapping[str, Any], dotted_path: str) -> float:
    result = value(data, dotted_path)
    if not isinstance(result, (int, float)) or isinstance(result, bool) or not math.isfinite(float(result)):
        raise ContractGenerationError(f"{dotted_path} must be a finite number")
    return float(result)


def cpp_number(number: float) -> str:
    """Render a round-trippable binary64 literal for generated C++."""

    return format(number, ".17g")


def render_header(contract: Mapping[str, Any]) -> str:
    """Render deterministic C++14/CUDA helpers from the contract."""

    digest = contract_hash(contract)
    rgas = finite_float(contract, "thermodynamics.gas_constant_j_mol_k")
    delta_h = finite_float(contract, "thermodynamics.delta_H_J_mol")
    delta_s = finite_float(contract, "thermodynamics.delta_S_J_mol_K")
    d0_cm2 = finite_float(contract, "kinetics.D0_cm2_s")
    activation = finite_float(contract, "kinetics.activation_energy_J_mol")
    gamma = finite_float(contract, "interface.gamma_J_m2")
    vm_alpha = finite_float(contract, "volumes.Vm_alpha_m3_mol")
    vm_beta = finite_float(contract, "volumes.Vm_beta_m3_mol")
    v_b = finite_float(contract, "composition.v_B")
    convex = value(contract, "thermodynamics.convex_extrapolation")
    if not isinstance(convex, Mapping):
        raise ContractGenerationError("convex extrapolation settings must be an object")
    convex_enabled = convex.get("enabled")
    if not isinstance(convex_enabled, bool):
        raise ContractGenerationError("convex extrapolation enabled flag must be boolean")
    convex_limit = convex.get("xB_limit")
    convex_penalty = convex.get("penalty_J_mol")
    if (not isinstance(convex_limit, (int, float)) or isinstance(convex_limit, bool) or
            not math.isfinite(float(convex_limit)) or not 0.0 < float(convex_limit) < 1.0):
        raise ContractGenerationError("convex extrapolation xB_limit must lie in (0, 1)")
    if (not isinstance(convex_penalty, (int, float)) or isinstance(convex_penalty, bool) or
            not math.isfinite(float(convex_penalty)) or float(convex_penalty) < 0.0):
        raise ContractGenerationError("convex extrapolation penalty must be non-negative")
    coeffs = value(contract, "thermodynamics.standard_state_coefficients")
    if not isinstance(coeffs, Mapping):
        raise ContractGenerationError("standard-state coefficients must be an object")

    def phase(name: str) -> tuple[float, list[float], list[float]]:
        record = coeffs.get(name)
        if not isinstance(record, Mapping):
            raise ContractGenerationError(f"missing standard-state record {name}")
        transition = float(record["transition_K"])
        low = [float(x) for x in record["low"]]
        high = [float(x) for x in record["high"]]
        if len(low) != 7 or len(high) != 7:
            raise ContractGenerationError(f"{name} coefficient vectors must have length 7")
        return transition, low, high

    pb = phase("GHSER_Pb")
    ag = phase("GHSER_Ag")
    te = phase("GHSER_Te")
    pbte = coeffs.get("G_PbTe")
    ag2te = coeffs.get("G_Ag2Te")
    if not isinstance(pbte, Mapping) or not isinstance(ag2te, Mapping):
        raise ContractGenerationError("compound standard-state records are missing")
    pbte_base = [float(x) for x in pbte["base"]]
    ag2te_base = [float(x) for x in ag2te["base_per_atom"]]
    ag2te_weights = [float(x) for x in ag2te["atom_weights"]]
    multiplier = float(ag2te["molecular_multiplier"])

    def array(values: list[float]) -> str:
        return ", ".join(cpp_number(item) for item in values)

    return f"""// GENERATED FILE. DO NOT EDIT.
// Source: contracts/pf_kwn_validation_contract_v1.json
// Generator: tools/generate_pf_contract_header.py
#ifndef PF_KWN_VALIDATION_CONTRACT_V1_H
#define PF_KWN_VALIDATION_CONTRACT_V1_H

#include <math.h>

#if defined(__CUDACC__)
#define PF_KWN_HD __host__ __device__
#else
#define PF_KWN_HD
#endif

#define PF_KWN_VALIDATION_CONTRACT_HASH \"{digest}\"
#define PF_KWN_VALIDATION_CONTRACT_SCHEMA \"PF_KWN_VALIDATION_CONTRACT_V1\"

static constexpr double PF_KWN_R_GAS = {cpp_number(rgas)};
static constexpr double PF_KWN_DELTA_H_J_PER_MOL = {cpp_number(delta_h)};
static constexpr double PF_KWN_DELTA_S_J_PER_MOL_K = {cpp_number(delta_s)};
static constexpr double PF_KWN_D0_CM2_PER_S = {cpp_number(d0_cm2)};
static constexpr double PF_KWN_ACTIVATION_ENERGY_J_PER_MOL = {cpp_number(activation)};
static constexpr double PF_KWN_GAMMA_J_PER_M2 = {cpp_number(gamma)};
static constexpr double PF_KWN_VM_ALPHA_M3_PER_MOL = {cpp_number(vm_alpha)};
static constexpr double PF_KWN_VM_BETA_M3_PER_MOL = {cpp_number(vm_beta)};
static constexpr double PF_KWN_V_B = {cpp_number(v_b)};
static constexpr int PF_KWN_CONVEX_EXTRAPOLATION_ENABLED = {1 if convex_enabled else 0};
static constexpr double PF_KWN_CONVEX_EXTRAPOLATION_XB_LIMIT = {cpp_number(float(convex_limit))};
static constexpr double PF_KWN_CONVEX_EXTRAPOLATION_PENALTY_J_PER_MOL = {cpp_number(float(convex_penalty))};

PF_KWN_HD static inline double pf_kwn_clamp_fraction(double x) {{
    return x < 1.0e-12 ? 1.0e-12 : (x > 1.0 - 1.0e-12 ? 1.0 - 1.0e-12 : x);
}}

PF_KWN_HD static inline double pf_kwn_standard_state_piecewise(
    double temperature_K, double transition_K,
    double a0_low, double a1_low, double alog_low, double a2_low, double a3_low, double ainv_low, double pinv_low,
    double a0_high, double a1_high, double alog_high, double a2_high, double a3_high, double ainv_high, double pinv_high) {{
    const bool low = temperature_K < transition_K;
    const double a0 = low ? a0_low : a0_high;
    const double a1 = low ? a1_low : a1_high;
    const double alog = low ? alog_low : alog_high;
    const double a2 = low ? a2_low : a2_high;
    const double a3 = low ? a3_low : a3_high;
    const double ainv = low ? ainv_low : ainv_high;
    const double pinv = low ? pinv_low : pinv_high;
    return a0 + a1 * temperature_K + alog * temperature_K * log(temperature_K) +
           a2 * temperature_K * temperature_K + a3 * temperature_K * temperature_K * temperature_K +
           (ainv == 0.0 ? 0.0 : ainv * pow(temperature_K, pinv));
}}

PF_KWN_HD static inline double pf_kwn_GHSER_Pb(double T) {{
    return pf_kwn_standard_state_piecewise(T, {cpp_number(pb[0])}, {array(pb[1])}, {array(pb[2])});
}}
PF_KWN_HD static inline double pf_kwn_GHSER_Ag(double T) {{
    return pf_kwn_standard_state_piecewise(T, {cpp_number(ag[0])}, {array(ag[1])}, {array(ag[2])});
}}
PF_KWN_HD static inline double pf_kwn_GHSER_Te(double T) {{
    return pf_kwn_standard_state_piecewise(T, {cpp_number(te[0])}, {array(te[1])}, {array(te[2])});
}}
PF_KWN_HD static inline double pf_kwn_G_PbTe(double T) {{
    return {cpp_number(pbte_base[0])} + {cpp_number(pbte_base[1])} * T + pf_kwn_GHSER_Pb(T) + pf_kwn_GHSER_Te(T);
}}
PF_KWN_HD static inline double pf_kwn_G_Ag2Te(double T) {{
    const double atom = {cpp_number(ag2te_base[0])} + {cpp_number(ag2te_base[1])} * T +
        {cpp_number(ag2te_weights[0])} * pf_kwn_GHSER_Ag(T) +
        {cpp_number(ag2te_weights[1])} * pf_kwn_GHSER_Te(T);
    return {cpp_number(multiplier)} * atom;
}}
PF_KWN_HD static inline double pf_kwn_L(double T) {{
    return PF_KWN_DELTA_H_J_PER_MOL - PF_KWN_DELTA_S_J_PER_MOL_K * T;
}}
PF_KWN_HD static inline double pf_kwn_mu_A(double T, double xB) {{
    const double x = pf_kwn_clamp_fraction(xB);
    return pf_kwn_G_PbTe(T) + PF_KWN_R_GAS * T * log(1.0 - x) + pf_kwn_L(T) * x * x;
}}
PF_KWN_HD static inline double pf_kwn_mu_B(double T, double xB) {{
    const double x = pf_kwn_clamp_fraction(xB);
    return pf_kwn_G_Ag2Te(T) + PF_KWN_R_GAS * T * log(x) + pf_kwn_L(T) * (1.0 - x) * (1.0 - x);
}}
PF_KWN_HD static inline double pf_kwn_G_alpha(double T, double xB) {{
    const double x = pf_kwn_clamp_fraction(xB);
    return (1.0 - x) * pf_kwn_G_PbTe(T) + x * pf_kwn_G_Ag2Te(T) +
        PF_KWN_R_GAS * T * ((1.0 - x) * log(1.0 - x) + x * log(x)) +
        pf_kwn_L(T) * x * (1.0 - x);
}}
PF_KWN_HD static inline double pf_kwn_dGdx(double T, double xB) {{
    return pf_kwn_mu_B(T, xB) - pf_kwn_mu_A(T, xB);
}}
PF_KWN_HD static inline double pf_kwn_d2Gdx2(double T, double xB) {{
    const double x = pf_kwn_clamp_fraction(xB);
    return PF_KWN_R_GAS * T / (x * (1.0 - x)) - 2.0 * pf_kwn_L(T);
}}
PF_KWN_HD static inline double pf_kwn_beta_driving_force(double T, double xB) {{
    const double x = pf_kwn_clamp_fraction(xB);
    return PF_KWN_R_GAS * T * log(x) + pf_kwn_L(T) * (1.0 - x) * (1.0 - x);
}}
PF_KWN_HD static inline double pf_kwn_D_alpha_m2_s(double T) {{
    return PF_KWN_D0_CM2_PER_S * exp(-PF_KWN_ACTIVATION_ENERGY_J_PER_MOL / (PF_KWN_R_GAS * T)) * 1.0e-4;
}}
PF_KWN_HD static inline double pf_kwn_planar_solvus(double T) {{
    double lo = 1.0e-12;
    double hi = 0.5;
    for (int iter = 0; iter < 200; ++iter) {{
        const double mid = 0.5 * (lo + hi);
        const double f = pf_kwn_beta_driving_force(T, mid);
        if (f > 0.0) hi = mid; else lo = mid;
    }}
    return 0.5 * (lo + hi);
}}
PF_KWN_HD static inline double pf_kwn_curvature_equilibrium(double T, double radius_m, double elastic_penalty_J_m3) {{
    const double xeq = pf_kwn_planar_solvus(T);
    return xeq * exp(((2.0 * PF_KWN_GAMMA_J_PER_M2 / radius_m) + elastic_penalty_J_m3) *
                      PF_KWN_VM_BETA_M3_PER_MOL / (PF_KWN_R_GAS * T));
}}
PF_KWN_HD static inline double pf_kwn_h_of_phi(double phi) {{
    const double p = phi;
    return 6.0 * p * p * p * p * p - 15.0 * p * p * p * p + 10.0 * p * p * p;
}}

#endif  // PF_KWN_VALIDATION_CONTRACT_V1_H
"""


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--contract", type=Path, default=DEFAULT_CONTRACT)
    parser.add_argument("--header", type=Path, default=DEFAULT_HEADER)
    parser.add_argument("--check", action="store_true", help="fail if the header is stale")
    args = parser.parse_args()
    contract = load_contract(args.contract)
    rendered = render_header(contract)
    if args.check:
        if not args.header.exists() or args.header.read_text(encoding="utf-8") != rendered:
            raise SystemExit("generated PF validation-contract header is missing or stale")
    else:
        args.header.parent.mkdir(parents=True, exist_ok=True)
        args.header.write_text(rendered, encoding="utf-8")
    print(json.dumps({"contract": str(args.contract), "hash": contract_hash(contract), "header": str(args.header)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
