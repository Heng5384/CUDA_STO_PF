#!/usr/bin/env python3
"""Runtime-safe S-field definition for GP-assisted beta driving.

This module is intentionally small and deterministic. It contains no CNT scan,
no radius optimization, and no stochastic event logic. The runtime phase-field
layer may import this file to compute the local GP amplification field

    S(x) = S(phi_GP(x), grad phi_GP(x), x_B(x))

from coefficients calibrated offline by the CNT scanning layer.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_COUPLING_INTERFACE = REPO_ROOT / "coupling_interface.json"


def _as_float(value: Any, default: float) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def gradient_magnitude(grad_phi_gp: float | Iterable[float]) -> float:
    """Return |grad phi_GP| from a scalar magnitude or vector components."""

    if isinstance(grad_phi_gp, (int, float)):
        return abs(float(grad_phi_gp))
    return math.sqrt(sum(float(g) * float(g) for g in grad_phi_gp))


@dataclass(frozen=True)
class SFieldParameters:
    """Deterministic calibration coefficients for the runtime S-field."""

    s_min: float = 1.0
    s_max: float = 4.0
    gp_amplitude: float = 1.0
    gp_power: float = 1.0
    interface_amplitude: float = 0.25
    interface_length_nm: float = 1.0
    xb_reference: float = 0.05
    xb_amplitude: float = 0.0
    xb_width: float = 0.01

    @classmethod
    def from_mapping(cls, mapping: dict[str, Any]) -> "SFieldParameters":
        return cls(
            s_min=_as_float(mapping.get("s_min"), cls.s_min),
            s_max=_as_float(mapping.get("s_max"), cls.s_max),
            gp_amplitude=_as_float(mapping.get("gp_amplitude"), cls.gp_amplitude),
            gp_power=_as_float(mapping.get("gp_power"), cls.gp_power),
            interface_amplitude=_as_float(mapping.get("interface_amplitude"), cls.interface_amplitude),
            interface_length_nm=_as_float(mapping.get("interface_length_nm"), cls.interface_length_nm),
            xb_reference=_as_float(mapping.get("xb_reference"), cls.xb_reference),
            xb_amplitude=_as_float(mapping.get("xb_amplitude"), cls.xb_amplitude),
            xb_width=_as_float(mapping.get("xb_width"), cls.xb_width),
        )


def load_s_field_parameters(path: Path = DEFAULT_COUPLING_INTERFACE) -> SFieldParameters:
    """Load runtime S-field coefficients from the coupling interface."""

    payload = json.loads(path.read_text(encoding="utf-8"))
    return SFieldParameters.from_mapping(payload.get("s_field_calibration", {}))


def compute_s_field(
    phi_gp: float,
    grad_phi_gp: float | Iterable[float],
    xB: float,
    params: SFieldParameters | None = None,
) -> float:
    """Compute the local GP modulation factor S(x).

    The convention used by the dual-track architecture is:

        Delta g_beta_eff(x) = S(x) * Delta g_beta_bulk(x_B)

    so S >= 1 strengthens the local beta driving force near GP zones. CNT
    barrier information is not recomputed here; it only calibrates coefficients.
    """

    p = params or SFieldParameters()
    eta = _clamp(float(phi_gp), 0.0, 1.0)
    grad_mag = gradient_magnitude(grad_phi_gp)
    gp_term = p.gp_amplitude * (eta ** max(p.gp_power, 1.0e-12))
    interface_term = p.interface_amplitude * math.tanh(max(0.0, grad_mag) * p.interface_length_nm)
    xb_term = 0.0
    if p.xb_width > 0.0:
        xb_term = p.xb_amplitude * math.tanh((float(xB) - p.xb_reference) / p.xb_width)
    return _clamp(1.0 + gp_term + interface_term + xb_term, p.s_min, p.s_max)


def effective_beta_driving_force(
    delta_g_beta_bulk: float,
    phi_gp: float,
    grad_phi_gp: float | Iterable[float],
    xB: float,
    params: SFieldParameters | None = None,
) -> float:
    """Apply S(x) to the local beta bulk driving force."""

    return compute_s_field(phi_gp, grad_phi_gp, xB, params) * float(delta_g_beta_bulk)


def barrier_modifier_from_s(s_value: float) -> float:
    """Return the CNT-consistent barrier scale implied by an S driving boost.

    This helper is for calibration/reporting only. Runtime PF evolution should
    use S through the driving force, not recompute CNT barriers.
    """

    s = max(float(s_value), 1.0e-12)
    return 1.0 / (s * s)


def calibrate_s_from_barriers(deltaG_bulk_kBT: float, deltaG_gp_kBT: float, s_max: float = 4.0) -> float:
    """Offline helper: infer S from a bulk-vs-GP barrier ratio."""

    bulk = max(float(deltaG_bulk_kBT), 1.0e-300)
    gp = max(float(deltaG_gp_kBT), 1.0e-300)
    return _clamp(math.sqrt(bulk / gp), 1.0, float(s_max))


def main() -> None:
    params = load_s_field_parameters()
    probe = compute_s_field(phi_gp=1.0, grad_phi_gp=(0.2, 0.0, 0.0), xB=params.xb_reference, params=params)
    print(f"S_field_probe={probe:.8g}")
    print("runtime_cnt_scan_used=false")


if __name__ == "__main__":
    main()
