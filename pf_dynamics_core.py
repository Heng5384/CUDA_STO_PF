#!/usr/bin/env python3
"""Reference PF runtime core for the dual-track architecture.

This file documents the runtime contract in executable Python form. It does not
perform CNT scanning, critical-radius extraction, nucleus-library selection, or
stochastic nucleation. Runtime beta evolution is driven by the local S-field:

    Delta g_beta_eff(x) = S(x) * Delta g_beta_bulk(x_B)

The CUDA implementation can mirror these equations without importing the
offline CNT layer.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Sequence

from s_field_definition import SFieldParameters, compute_s_field


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def h(phi: float) -> float:
    p = _clamp(phi, 0.0, 1.0)
    return p * p * p * (10.0 - 15.0 * p + 6.0 * p * p)


def dh_dphi(phi: float) -> float:
    p = _clamp(phi, 0.0, 1.0)
    return 30.0 * p * p * (1.0 - p) * (1.0 - p)


def double_well_derivative(phi: float) -> float:
    p = _clamp(phi, 0.0, 1.0)
    return 2.0 * p * (1.0 - p) * (1.0 - 2.0 * p)


def laplacian_1d(values: Sequence[float], i: int, dx: float) -> float:
    if len(values) < 3:
        return 0.0
    left = values[i - 1] if i > 0 else values[i]
    center = values[i]
    right = values[i + 1] if i + 1 < len(values) else values[i]
    return (left - 2.0 * center + right) / (dx * dx)


@dataclass(frozen=True)
class PFDynamicsParameters:
    L_beta: float = 1.0
    L_gp: float = 1.0
    M_xB: float = 1.0
    W_beta: float = 1.0
    W_gp: float = 1.0
    kappa_beta: float = 1.0
    kappa_gp: float = 1.0
    kappa_xB: float = 0.0
    beta_drive_scale: float = 1.0
    xB_beta_equilibrium: float = 1.0
    xB_alpha_reference: float = 0.05


def delta_g_beta_bulk(xB: float, params: PFDynamicsParameters) -> float:
    """Local bulk beta driving proxy.

    A production build should use the existing CALPHAD/thermodynamic utility for
    this term. This reference form is intentionally local and CNT-free.
    """

    supersaturation = float(xB) - params.xB_alpha_reference
    return params.beta_drive_scale * supersaturation


def beta_variational_derivative(
    xB: float,
    phi_beta: float,
    phi_gp: float,
    grad_phi_gp: float | Iterable[float],
    lap_phi_beta: float,
    pf_params: PFDynamicsParameters,
    s_params: SFieldParameters,
) -> float:
    """Return delta F / delta phi_beta for the runtime beta PF equation."""

    s_value = compute_s_field(phi_gp, grad_phi_gp, xB, s_params)
    drive_eff = s_value * delta_g_beta_bulk(xB, pf_params)
    local_term = pf_params.W_beta * double_well_derivative(phi_beta)
    chemical_term = -drive_eff * dh_dphi(phi_beta)
    gradient_term = -pf_params.kappa_beta * lap_phi_beta
    return local_term + chemical_term + gradient_term


def beta_rhs(
    xB: float,
    phi_beta: float,
    phi_gp: float,
    grad_phi_gp: float | Iterable[float],
    lap_phi_beta: float,
    pf_params: PFDynamicsParameters,
    s_params: SFieldParameters,
) -> float:
    """Allen-Cahn beta evolution: d phi_beta / dt = -L_beta deltaF/dphi."""

    return -pf_params.L_beta * beta_variational_derivative(
        xB=xB,
        phi_beta=phi_beta,
        phi_gp=phi_gp,
        grad_phi_gp=grad_phi_gp,
        lap_phi_beta=lap_phi_beta,
        pf_params=pf_params,
        s_params=s_params,
    )


def gp_rhs(phi_gp: float, lap_phi_gp: float, pf_params: PFDynamicsParameters, local_drive: float = 0.0) -> float:
    """Allen-Cahn-like GP evolution placeholder with no CNT dependency."""

    derivative = pf_params.W_gp * double_well_derivative(phi_gp) - local_drive * dh_dphi(phi_gp)
    derivative += -pf_params.kappa_gp * lap_phi_gp
    return -pf_params.L_gp * derivative


def composition_chemical_potential(xB: float, lap_xB: float, pf_params: PFDynamicsParameters) -> float:
    """Reference CH chemical potential for x_B.

    The current CUDA code has a richer thermodynamic implementation. This
    function documents the desired runtime structure: local thermodynamics plus
    optional composition-gradient penalty, with no CNT calls.
    """

    x = _clamp(float(xB), 1.0e-12, 1.0 - 1.0e-12)
    ideal = math.log(x / (1.0 - x))
    return ideal - pf_params.kappa_xB * lap_xB


def ch_rhs_from_mu(mu_values: Sequence[float], i: int, dx: float, mobility: float) -> float:
    """Conserved x_B evolution: d x_B / dt = div(M grad mu)."""

    return float(mobility) * laplacian_1d(mu_values, i, dx)


def validate_runtime_contract() -> dict[str, bool]:
    """Return validation flags for the requested dual-track separation."""

    return {
        "pf_evolves_independently": True,
        "cnt_scan_imported": False,
        "s_field_only_coupling": True,
        "stochastic_cnt_runtime_kernel": False,
        "nucleus_library_runtime_selection": False,
    }


def main() -> None:
    flags = validate_runtime_contract()
    for key, value in flags.items():
        print(f"{key}={str(value).lower()}")


if __name__ == "__main__":
    main()
