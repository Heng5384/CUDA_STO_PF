#!/usr/bin/env python3
"""Select fixed-inventory R1/R2 exact-library radius histograms.

This selector changes only the integer choice of already-qualified 15-entry
target profiles.  It does not materialize fields, rescale profiles, or alter
any physical input.  The exact field/inventory audit happens in the separate
materializer.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp


SCHEMA = "PF_246CUBE_RESOLVED_SEED_THERMAL_CASE_INTEGER_SELECTION_V1"
LIBRARY_SHA256 = "de4142e0268e379f70fd1c860aab9e004421d07df4f8d872f4eaeb3dbef7af5b"
TARGET_H_VOLUME_NM3 = 356237.61016735336
TARGET_MEAN_C_BTOT = 0.03
TARGET_MATRIX_XAG = 0.0062
H_VOLUME_TOLERANCE_NM3 = 0.1
BOX_VOLUME_NM3 = float(246**3)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def profile_table(library_root: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    manifest_path = library_root / "library_manifest.json"
    if not manifest_path.is_file() or sha256(manifest_path) != LIBRARY_SHA256:
        raise SystemExit("[fatal] 15-entry library manifest identity mismatch")
    library = load_json(manifest_path)
    radii = np.asarray(library.get("radius_ladder_nm", []), dtype=float)
    expected = np.asarray([8.0 + 0.25 * index for index in range(15)], dtype=float)
    if radii.shape != expected.shape or not np.array_equal(radii, expected):
        raise SystemExit("[fatal] 15-entry radius ladder mismatch")
    h_values: list[float] = []
    for entry in library.get("profiles", []):
        profile_path = (library_root / entry["profile_manifest_path"]).resolve()
        if not profile_path.is_file() or sha256(profile_path) != entry["profile_manifest_sha256"]:
            raise SystemExit(f"[fatal] profile manifest mismatch: {profile_path}")
        profile = load_json(profile_path)
        h = float(profile["geometry"]["h_volume_nm3"])
        if not math.isfinite(h) or h <= 0.0:
            raise SystemExit(f"[fatal] invalid h-volume: {profile_path}")
        h_values.append(h)
    return radii, np.asarray(h_values, dtype=float), library


def solve_for_count(
    h: np.ndarray,
    radii: np.ndarray,
    particle_count: int,
    allowed: set[int],
    extra_constraints: list[tuple[np.ndarray, float, float]],
) -> tuple[np.ndarray, float] | None:
    """Minimize absolute h-volume residual at a fixed integer particle count."""
    count = h.size
    lower = np.zeros(count + 1, dtype=float)
    upper = np.full(count + 1, np.inf, dtype=float)
    for index in set(range(count)) - allowed:
        upper[index] = 0.0
    # The final continuous variable is the absolute volume residual.  The
    # minuscule radius tie breaker keeps R1 closest to the smallest entries.
    objective = np.r_[1.0e-8 * (radii - 8.0) ** 2, 1.0]
    integrality = np.r_[np.ones(count, dtype=int), 0]
    rows = [np.r_[np.ones(count), 0.0], np.r_[h, -1.0], np.r_[-h, -1.0]]
    lo = [float(particle_count), -np.inf, -np.inf]
    hi = [float(particle_count), TARGET_H_VOLUME_NM3, -TARGET_H_VOLUME_NM3]
    for coefficients, low, high in extra_constraints:
        rows.append(np.r_[coefficients, 0.0])
        lo.append(low)
        hi.append(high)
    result = milp(
        c=objective,
        integrality=integrality,
        bounds=Bounds(lower, upper),
        constraints=LinearConstraint(np.asarray(rows), np.asarray(lo), np.asarray(hi)),
    )
    if not result.success:
        return None
    histogram = np.rint(result.x[:count]).astype(int)
    if np.any(histogram < 0) or int(np.sum(histogram)) != particle_count:
        raise RuntimeError("MILP returned invalid integer histogram")
    return histogram, float(np.dot(h, histogram) - TARGET_H_VOLUME_NM3)


def first_exact_count(
    name: str,
    radii: np.ndarray,
    h: np.ndarray,
    maximum_count: int,
    minimum_count: int,
    allowed: set[int],
    extras: list[tuple[np.ndarray, float, float]],
) -> tuple[np.ndarray, float]:
    for count in range(maximum_count, minimum_count - 1, -1):
        result = solve_for_count(h, radii, count, allowed, extras)
        if result is not None and abs(result[1]) <= H_VOLUME_TOLERANCE_NM3:
            return result
    raise RuntimeError(f"{name}: no integer histogram reaches the h-volume tolerance")


def summarize_case(name: str, radii: np.ndarray, h: np.ndarray, histogram: np.ndarray, h_error: float, seed: int) -> dict[str, Any]:
    count = int(np.sum(histogram))
    selected_h = float(np.dot(h, histogram))
    radial_second = float(np.dot(radii**2, histogram))
    return {
        "case": name,
        "particle_count": count,
        "histogram": {f"{radius:.2f}": int(value) for radius, value in zip(radii, histogram) if int(value)},
        "selected_h_volume_nm3": selected_h,
        "target_h_volume_nm3": TARGET_H_VOLUME_NM3,
        "h_volume_error_nm3": h_error,
        "h_volume_relative_error": abs(h_error) / TARGET_H_VOLUME_NM3,
        "mean_registered_radius_nm": float(np.dot(radii, histogram) / count),
        "registered_radius_cv": float(np.sqrt(np.dot((radii - np.dot(radii, histogram) / count) ** 2, histogram) / count) / (np.dot(radii, histogram) / count)),
        "spherical_selector_Sv_nm_inv": 4.0 * math.pi * radial_second / BOX_VOLUME_NM3,
        "placement_seed_unsigned64": seed,
        "placement_policy": "deterministic_periodic_hard_core_rejection_integer_grid_v1",
        "profile_scaling_used": False,
        "profile_interpolation_used": False,
        "analytic_profile_used": False,
        "physical_parameter_retuning_used": False,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"[fatal] refusing to overwrite selection: {args.out}")
    radii, h, library = profile_table(args.library_root.resolve())

    # R1: the highest exact-library count whose h-volume can meet the
    # pre-registered tolerance.  The tie-breaker makes it overwhelmingly 8 nm.
    r1_histogram, r1_error = first_exact_count(
        "R1", radii, h, maximum_count=170, minimum_count=1,
        allowed=set(range(15)), extras=[],
    )
    small = np.zeros(15)
    small[:3] = 1.0
    large = np.zeros(15)
    large[12:] = 1.0
    # R2: only 8--8.5 nm and 11--11.5 nm entries; at least 100 small and
    # 15 large particles ensure a real two-mode resolved population.
    r2_histogram, r2_error = first_exact_count(
        "R2", radii, h, maximum_count=160, minimum_count=1,
        allowed={0, 1, 2, 12, 13, 14},
        extras=[(small, 100.0, np.inf), (large, 15.0, np.inf)],
    )
    payload = {
        "schema": SCHEMA,
        "status": "PASS_RESOLVED_SEED_INTEGER_PSD_SELECTION_V1",
        "profile_library_manifest_sha256": LIBRARY_SHA256,
        "profile_library_canonical_content_sha256": library["canonical_content_sha256"],
        "registered_radius_ladder_nm": radii.tolist(),
        "target_contract": {
            "domain": [246, 246, 246],
            "dx_nm": 1.0,
            "lambda_sm_nm": 4.0,
            "temperature_C": 380.0,
            "target_mean_C_Btot": TARGET_MEAN_C_BTOT,
            "target_matrix_xAg": TARGET_MATRIX_XAG,
            "target_effective_h_volume_nm3": TARGET_H_VOLUME_NM3,
            "h_volume_tolerance_nm3": H_VOLUME_TOLERANCE_NM3,
        },
        "optimizer": {
            "backend": "SCIPY_HIGHS_MILP_INTEGER_COUNTS_V1",
            "objective": "max_particle_count_then_minimize_absolute_exact_library_h_volume_residual_then_minimize_radius_distance_from_8nm",
            "R1_policy": "maximum_resolved_interface_count_with_exact_library_profiles",
            "R2_policy": "bimodal_8_to_8p5_and_11_to_11p5_nm_with_at_least_100_small_and_15_large_particles",
        },
        "cases": {
            "R1": summarize_case("R1", radii, h, r1_histogram, r1_error, 11807945950896733510),
            "R2": summarize_case("R2", radii, h, r2_histogram, r2_error, 14886717532826618077),
        },
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"status": payload["status"], "selection_sha256": sha256(args.out), "R1_count": payload["cases"]["R1"]["particle_count"], "R2_count": payload["cases"]["R2"]["particle_count"]}, sort_keys=True))


if __name__ == "__main__":
    main()
