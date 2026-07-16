#!/usr/bin/env python3
"""Backend-neutral arithmetic for future real Next8 trajectories.

The routines do not define a physical backend mapping.  They only implement
auditable arithmetic once a selected backend has supplied traceable quantities
in the frozen project convention.  The CLI refuses an empty/header-only input.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Sequence

import numpy as np


def linear_slope(time: Sequence[float], value: Sequence[float]) -> tuple[float, float]:
    t = np.asarray(time, dtype=float)
    y = np.asarray(value, dtype=float)
    if t.ndim != 1 or y.ndim != 1 or len(t) != len(y) or len(t) < 3:
        raise ValueError("at least three paired samples are required")
    if not np.all(np.isfinite(t)) or not np.all(np.isfinite(y)) or np.ptp(t) <= 0.0:
        raise ValueError("finite, non-degenerate trajectory required")
    design = np.column_stack([t, np.ones_like(t)])
    slope, intercept = np.linalg.lstsq(design, y, rcond=None)[0]
    residual = y - (slope * t + intercept)
    return float(slope), float(np.sqrt(np.mean(residual * residual)))


def equimolar_position(z: Sequence[float], ctot: Sequence[float], c_alpha: float, c_beta: float) -> float:
    """One-interface beta-left equimolar position on a uniform/ordered z grid."""
    z_arr = np.asarray(z, dtype=float)
    c_arr = np.asarray(ctot, dtype=float)
    if len(z_arr) != len(c_arr) or len(z_arr) < 2 or not np.all(np.diff(z_arr) > 0):
        raise ValueError("ordered paired profile required")
    jump = float(c_beta - c_alpha)
    if not math.isfinite(jump) or abs(jump) < 1.0e-30:
        raise ValueError("nonzero phase concentration jump required")
    excess = np.trapezoid(c_arr - c_alpha, z_arr)
    return float(z_arr[0] + excess / jump)


def inventory_flux(delta_inventory_mol: float, area_m2: float, duration_s: float, interface_count: int = 1) -> float:
    if area_m2 <= 0.0 or duration_s <= 0.0 or interface_count <= 0:
        raise ValueError("positive area, duration, and interface count required")
    return float(delta_inventory_mol / (area_m2 * duration_s * interface_count))


def crossing_flux(net_crossing_mol: float, area_m2: float, duration_s: float, interface_count: int = 1) -> float:
    return inventory_flux(net_crossing_mol, area_m2, duration_s, interface_count)


def stefan_flux(vn_m_s: float, vB_mol_m3: float, c_alpha_mol_m3: float) -> float:
    return float(vn_m_s * (vB_mol_m3 - c_alpha_mol_m3))


def block_mean_covariance(vn_samples: Sequence[float], jb_samples: Sequence[float], block_size: int) -> tuple[np.ndarray, int]:
    v = np.asarray(vn_samples, dtype=float)
    j = np.asarray(jb_samples, dtype=float)
    if len(v) != len(j) or block_size <= 0:
        raise ValueError("paired samples and positive block size required")
    nblock = len(v) // block_size
    if nblock < 2:
        raise ValueError("at least two complete blocks required")
    vb = v[: nblock * block_size].reshape(nblock, block_size).mean(axis=1)
    jb = j[: nblock * block_size].reshape(nblock, block_size).mean(axis=1)
    covariance_of_mean = np.cov(np.vstack([vb, jb]), ddof=1) / nblock
    return covariance_of_mean, nblock


def linearity_deviation(full_response: float, half_response: float) -> float:
    if half_response == 0.0:
        raise ValueError("half-driving response is zero")
    return abs(full_response / half_response - 2.0) / 2.0


def validate_A_I(x_phase: float, vn: float, sigma_vn: float, A_expected: float) -> dict[str, float | bool]:
    if vn == 0.0 or sigma_vn < 0.0 or A_expected <= 0.0:
        raise ValueError("valid phase response and positive A_expected required")
    measured = x_phase / vn
    sigma = abs(x_phase / (vn * vn)) * sigma_vn
    rel = abs(measured - A_expected) / A_expected
    return {
        "A_measured": measured,
        "A_sigma": sigma,
        "relative_error": rel,
        "expected_inside_95CI": abs(measured - A_expected) <= 1.96 * sigma,
    }


def predict_diagonal(x_phase: float, x_diff: float, A: float, C: float) -> tuple[float, float]:
    if A <= 0.0 or C <= 0.0:
        raise ValueError("positive diagonal resistance required")
    return x_phase / A, x_diff / C


def predict_reciprocal(x_phase: float, x_diff: float, A: float, B: float, C: float) -> tuple[float, float]:
    determinant = A * C - B * B
    if A <= 0.0 or C <= 0.0 or determinant <= 0.0:
        raise ValueError("strictly SPD resistance matrix required")
    return ((C * x_phase - B * x_diff) / determinant,
            (-B * x_phase + A * x_diff) / determinant)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("trajectory_csv", type=Path)
    args = parser.parse_args()
    if not args.trajectory_csv.is_file():
        parser.error("real trajectory CSV does not exist")
    with args.trajectory_csv.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        parser.error("real trajectory CSV has no DATA rows")
    raise SystemExit("backend-specific column and provenance adapter is required before analysis")


if __name__ == "__main__":
    raise SystemExit(main())
