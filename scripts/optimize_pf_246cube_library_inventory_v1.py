#!/usr/bin/env python3
"""Freeze a minimum-distortion integer PSD on a registered radius ladder.

The target is the historical fixture's integrated h-volume, not its analytic
tanh radius parameter.  Every historical particle is assigned to exactly one
registered library radius.  The assignment minimizes squared error in the
h-equivalent radius while matching the nearest representable total cubic
moment.  No profile field is interpolated or scaled.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from decimal import Decimal
from pathlib import Path

import numpy as np


SCHEMA = "PF_246CUBE_LIBRARY_INVENTORY_SELECTION_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def decimal_places(values: list[float]) -> int:
    return max(max(0, -Decimal(str(value)).as_tuple().exponent) for value in values)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--historical-manifest", type=Path, required=True)
    parser.add_argument("--library-manifest", type=Path, required=True)
    parser.add_argument("--library-selection-provenance", type=Path, required=True)
    parser.add_argument("--target-mean-C-B-tot", type=float, default=0.03)
    parser.add_argument("--experimental-matrix-xAg-center", type=float, default=0.0062)
    parser.add_argument(
        "--experimental-matrix-xAg-interval",
        type=float,
        nargs=2,
        default=(0.0058, 0.0066),
    )
    parser.add_argument(
        "--max-assignment-deviation-nm", type=float, default=0.26
    )
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"[fatal] refusing to overwrite output: {args.out}")

    historical = json.loads(args.historical_manifest.read_text(encoding="utf-8"))
    library = json.loads(args.library_manifest.read_text(encoding="utf-8"))
    provenance = json.loads(
        args.library_selection_provenance.read_text(encoding="utf-8")
    )
    particles = sorted(
        historical["particles"], key=lambda row: int(row["particle_id"])
    )
    if len(particles) != 96:
        raise SystemExit("[fatal] historical fixture does not contain 96 particles")
    library_sha = sha256(args.library_manifest)
    if provenance.get("library_manifest_sha256") != library_sha:
        raise SystemExit("[fatal] library selection provenance does not pin library")
    radii = sorted(float(value) for value in library["radius_ladder_nm"])
    if len(radii) < 2 or len(set(radii)) != len(radii):
        raise SystemExit("[fatal] registered radius ladder is invalid")
    places = decimal_places(radii)
    scale = 10**places
    scaled_radii = np.asarray([round(value * scale) for value in radii], dtype=np.int64)
    if any(abs(integer / scale - value) > 1.0e-12 for integer, value in zip(scaled_radii, radii)):
        raise SystemExit("[fatal] radius ladder is not representable on decimal grid")
    cubic_units = scaled_radii**3

    h_volumes = np.asarray(
        [float(row["h_volume_nm3"]) for row in particles], dtype=np.float64
    )
    effective_radii = (3.0 * h_volumes / (4.0 * math.pi)) ** (1.0 / 3.0)
    effective_radii_sorted = np.sort(effective_radii)
    target_h_volume = float(np.sum(h_volumes, dtype=np.float64))
    target_cubic_units_real = (
        target_h_volume / (4.0 * math.pi / 3.0) * scale**3
    )
    target_cubic_units = int(round(target_cubic_units_real))

    particle_count = len(particles)
    radius_count = len(radii)

    # Exact dynamic programming over the sorted particles.  The state key is
    # (integer cubic moment, last radius index).  This proves the nearest
    # representable total within one registered 0.25 nm ladder step while
    # enforcing the monotone minimum-distortion assignment contract.  The
    # lexicographic objective is absolute volume error first, squared radius
    # distortion second.
    states: dict[tuple[int, int], tuple[float, bytes]] = {
        (0, -1): (0.0, b"")
    }
    for effective in effective_radii_sorted:
        allowed = [
            index
            for index, radius in enumerate(radii)
            if abs(radius - effective) <= args.max_assignment_deviation_nm
        ]
        if not allowed:
            raise SystemExit(
                "[fatal] no registered radius satisfies the particle "
                "assignment-deviation contract"
            )
        next_states: dict[tuple[int, int], tuple[float, bytes]] = {}
        for (cubic_sum, previous_index), (cost, path) in states.items():
            for radius_index in allowed:
                if radius_index < previous_index:
                    continue
                key = (cubic_sum + int(cubic_units[radius_index]), radius_index)
                candidate = (
                    cost + (radii[radius_index] - effective) ** 2,
                    path + bytes((radius_index,)),
                )
                current = next_states.get(key)
                if current is None or candidate[0] < current[0]:
                    next_states[key] = candidate
        if not next_states:
            raise SystemExit("[fatal] monotone integer PSD state space is empty")
        states = next_states
    (selected_cubic_units, _), (selected_cost, selected_path) = min(
        states.items(),
        key=lambda item: (
            abs(item[0][0] - target_cubic_units),
            item[1][0],
            item[1][1],
        ),
    )
    selected_indices = np.frombuffer(selected_path, dtype=np.uint8).astype(
        np.int64
    )
    if selected_indices.size != particle_count:
        raise SystemExit("[fatal] optimized particle assignment is incomplete")
    selected = np.asarray(radii)[selected_indices]
    minimum_cubic_gap = abs(selected_cubic_units - target_cubic_units)
    histogram_counter = Counter(float(value) for value in selected)
    histogram = {
        f"{radius:.{places}f}": int(histogram_counter.get(radius, 0))
        for radius in radii
    }
    selected_h_volume = float(
        np.sum(4.0 * math.pi * selected**3 / 3.0, dtype=np.float64)
    )
    if abs(selected_cubic_units - target_cubic_units) != minimum_cubic_gap:
        raise SystemExit("[fatal] selected PSD does not retain minimum cubic gap")
    sorted_assignments = sorted(
        zip(effective_radii_sorted.tolist(), selected.tolist(), strict=True)
    )
    monotone = all(
        right[1] >= left[1]
        for left, right in zip(sorted_assignments, sorted_assignments[1:])
    )
    if not monotone:
        raise SystemExit("[fatal] optimized assignment is not monotone")
    payload = {
        "schema": SCHEMA,
        "selection_policy": (
            "NEAREST_REPRESENTABLE_TOTAL_CUBIC_MOMENT_THEN_MINIMUM_SQUARED_"
            "H_EQUIVALENT_RADIUS_ERROR_WITHIN_ONE_QUARTER_NM_STEP_V1"
        ),
        "source_identities": {
            "historical_fixture_file_sha256": sha256(args.historical_manifest),
            "historical_canonical_manifest_sha256": historical[
                "canonical_manifest_sha256"
            ],
            "profile_library_manifest_sha256": library_sha,
            "profile_library_selection_provenance_sha256": sha256(
                args.library_selection_provenance
            ),
        },
        "particle_count": particle_count,
        "registered_radii_nm": radii,
        "selected_histogram": histogram,
        "particle_assignment_rule": (
            "sort_historical_h_equivalent_radius_then_particle_id_and_fill_"
            "selected_histogram_in_ascending_radius_order"
        ),
        "target_effective_h_volume_nm3": target_h_volume,
        "selected_effective_h_volume_nm3": selected_h_volume,
        "effective_h_volume_error_nm3": selected_h_volume - target_h_volume,
        "target_cubic_units_real": target_cubic_units_real,
        "target_cubic_units_rounded": target_cubic_units,
        "selected_cubic_units": selected_cubic_units,
        "minimum_absolute_cubic_unit_gap": abs(
            selected_cubic_units - target_cubic_units
        ),
        "decimal_radius_scale": scale,
        "effective_radius_rmse_nm": float(
            np.sqrt(np.mean((selected - effective_radii_sorted) ** 2))
        ),
        "effective_radius_max_abs_error_nm": float(
            np.max(np.abs(selected - effective_radii_sorted))
        ),
        "selected_mean_radius_nm": float(np.mean(selected)),
        "selected_radius_cv": float(np.std(selected) / np.mean(selected)),
        "selected_cubic_mean_radius_nm": float(np.mean(selected**3) ** (1.0 / 3.0)),
        "target_mean_C_B_tot": float(args.target_mean_C_B_tot),
        "experimental_matrix_xAg_center": float(
            args.experimental_matrix_xAg_center
        ),
        "experimental_matrix_xAg_interval": [
            float(args.experimental_matrix_xAg_interval[0]),
            float(args.experimental_matrix_xAg_interval[1]),
        ],
        "matrix_baseline_mode": "DERIVE_FROM_EXACT_ASSEMBLED_CANONICAL_LEDGER",
        "unresolved_inventory_used": False,
        "profile_scaling_used": False,
        "profile_interpolation_used": False,
        "physical_parameter_retuning_used": False,
        "maximum_particle_assignment_deviation_nm": float(
            args.max_assignment_deviation_nm
        ),
        "optimizer_backend": "EXACT_MONOTONE_INTEGER_DYNAMIC_PROGRAM_V1",
        "final_dynamic_program_state_count": len(states),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "status": "PASS_PF_246CUBE_LIBRARY_INVENTORY_OPTIMIZATION_V1",
                "selection_sha256": sha256(args.out),
                "selected_histogram": histogram,
                "effective_h_volume_error_nm3": (
                    selected_h_volume - target_h_volume
                ),
                "effective_radius_rmse_nm": payload[
                    "effective_radius_rmse_nm"
                ],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
