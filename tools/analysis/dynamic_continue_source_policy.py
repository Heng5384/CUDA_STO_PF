#!/usr/bin/env python3
"""Source-radius policy for dynamic-continue starts.

Production dynamic-continue should not start from the fitted peak itself.  It
should start from the first available completed radius point to the right of
the peak with a post-critical safety margin.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Any


NO_SAFE_POSTCRITICAL_RADIUS_AVAILABLE = "NO_SAFE_POSTCRITICAL_RADIUS_AVAILABLE"


@dataclass(frozen=True)
class RadiusChoice:
    status: str
    radius_nm: float | None
    row: Any | None
    policy_used: str
    margin_nm: float | None
    reason: str


def _radius(row: Any) -> float:
    if isinstance(row, dict):
        value = row.get("actual_r_eff_nm") or row.get("source_equiv_radius_nm") or row.get("radius_nm")
    else:
        value = row
    return float(value)


def choose_dynamic_continue_source(
    radius_points: Iterable[Any],
    peak_radius_nm: float,
    *,
    min_margin_nm: float = 0.10,
    fallback_margin_nm: float = 0.05,
    allow_exact_peak_debug: bool = False,
) -> RadiusChoice:
    """Choose first radius to the right of the peak with production margin.

    `radius_points` may be floats or dict rows containing `radius_nm`.
    """
    points = sorted(radius_points, key=_radius)
    if allow_exact_peak_debug:
        for row in points:
            r = _radius(row)
            if abs(r - peak_radius_nm) <= 1.0e-9:
                return RadiusChoice(
                    status="DEBUG_EXACT_PEAK",
                    radius_nm=r,
                    row=row,
                    policy_used="debug_exact_peak",
                    margin_nm=0.0,
                    reason="exact peak allowed only in debug mode",
                )
    for margin, label in ((min_margin_nm, "min_margin"), (fallback_margin_nm, "fallback_margin")):
        threshold = peak_radius_nm + max(margin, 0.0)
        for row in points:
            r = _radius(row)
            if r >= threshold - 1.0e-9:
                return RadiusChoice(
                    status="OK",
                    radius_nm=r,
                    row=row,
                    policy_used=f"first_right_of_peak_with_{label}",
                    margin_nm=margin,
                    reason=f"selected first radius >= peak + {margin:g} nm",
                )
    return RadiusChoice(
        status=NO_SAFE_POSTCRITICAL_RADIUS_AVAILABLE,
        radius_nm=None,
        row=None,
        policy_used="first_right_of_peak_with_margin",
        margin_nm=None,
        reason="no completed radius point to the right of peak satisfies fallback margin",
    )
