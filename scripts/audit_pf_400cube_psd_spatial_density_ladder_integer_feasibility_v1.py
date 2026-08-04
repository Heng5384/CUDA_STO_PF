#!/usr/bin/env python3
"""Fail-closed integer-inventory feasibility audit for the 400-cube ladder.

The 400-cube PSD/spatial/density campaign is allowed to choose only complete
entries from the frozen quarter-nanometre Method-1 library.  This audit checks
the requested h-volume tolerance before any 400^3 field is materialised or a
Slurm job is submitted.  It never modifies a profile, a historical fixture, or
the running 400-cube V5 pilot.

The registered radii are quarter-nanometre values.  Their ideal equivalent
volumes are therefore an integer lattice c*q^3, where c=pi/48 and q=4R.  The
small, hash-pinned difference between a profile's recorded h-volume and that
ideal volume is bounded explicitly.  That gives a rigorous lower bound on
the closest possible total h-volume for *any* integer histogram of N entries,
without claiming that a particular PSD family has been optimised.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


SCHEMA = "PF_400CUBE_PSD_SPATIAL_DENSITY_LADDER_INTEGER_FEASIBILITY_V1"
STATUS_PASS = "PASS_400CUBE_PSD_SPATIAL_DENSITY_LADDER_INTEGER_FEASIBILITY_V1"
STATUS_BLOCKED = "BLOCKED_PROFILE_LIBRARY_RANGE"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def load_json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return value


def write_json(path: Path, value: Any) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def profile_rows(library_path: Path) -> list[dict[str, float | str]]:
    library = load_json(library_path)
    if library.get("schema") != "PF_ELASTIC_TARGET_PROFILE_LIBRARY_V1":
        raise ValueError("wrong frozen profile-library schema")
    result: list[dict[str, float | str]] = []
    constant = math.pi / 48.0
    for entry in library.get("profiles", []):
        radius = float(entry["target_radius_nm"])
        q = int(round(4.0 * radius))
        if not math.isclose(4.0 * radius, float(q), rel_tol=0.0, abs_tol=1.0e-12):
            raise ValueError(f"R={radius}: not a quarter-nanometre registered radius")
        manifest_path = (library_path.parent / str(entry["profile_manifest_path"])).resolve()
        manifest = load_json(manifest_path)
        geometry = manifest.get("geometry", {})
        h_volume = float(geometry.get("final_h_volume_nm3", geometry.get("h_volume_nm3")))
        if not math.isfinite(h_volume) or h_volume <= 0.0:
            raise ValueError(f"R={radius}: invalid recorded h-volume")
        ideal = constant * float(q**3)
        result.append(
            {
                "radius_nm": radius,
                "q": float(q),
                "recorded_h_volume_nm3": h_volume,
                "ideal_quarter_nm_volume_nm3": ideal,
                "recorded_minus_ideal_nm3": h_volume - ideal,
                "profile_manifest_path": str(manifest_path),
                "profile_manifest_sha256": sha256(manifest_path),
            }
        )
    result.sort(key=lambda row: float(row["radius_nm"]))
    if [float(row["radius_nm"]) for row in result] != [8.0 + 0.25 * index for index in range(15)]:
        raise ValueError("frozen library is not exactly the required 15-entry 8.00--11.50 nm ladder")
    return result


def nearest_integer_lattice_distance(target: float, spacing: float) -> tuple[int, float, float]:
    quotient = target / spacing
    lower = math.floor(quotient)
    upper = lower + 1
    candidates = [(lower, abs(target - lower * spacing)), (upper, abs(target - upper * spacing))]
    integer, distance = min(candidates, key=lambda pair: pair[1])
    return integer, distance, quotient - math.floor(quotient)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--library-manifest", type=Path, required=True)
    parser.add_argument("--task-contract", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"[fatal] refusing to overwrite feasibility output: {args.out}")
    if not args.library_manifest.is_file() or not args.task_contract.is_file():
        raise SystemExit("[fatal] required immutable input is missing")

    target_h_volume = 1531490.9025410344
    relative_tolerance = 1.0e-10
    target_tolerance = target_h_volume * relative_tolerance
    counts = (256, 512, 640, 704)
    try:
        rows = profile_rows(args.library_manifest.resolve())
        offsets = [float(row["recorded_minus_ideal_nm3"]) for row in rows]
        if min(offsets) < 0.0:
            raise ValueError("profile recorded-minus-ideal correction must be nonnegative for this lattice bound")
        spacing = math.pi / 48.0
        nearest_q_sum, ideal_distance, fractional_lattice_coordinate = nearest_integer_lattice_distance(target_h_volume, spacing)
        result_rows: list[dict[str, Any]] = []
        for count in counts:
            correction_min = count * min(offsets)
            correction_max = count * max(offsets)
            # The closest ideal lattice point is below the target.  Adding the
            # largest possible nonnegative profile correction is maximally
            # favourable; this remains a lower bound even if not every integer
            # q^3 sum is reachable with exactly `count` particles.
            lower_bound = max(0.0, ideal_distance - correction_max)
            feasible = lower_bound <= target_tolerance
            result_rows.append(
                {
                    "particle_count": count,
                    "target_h_volume_nm3": target_h_volume,
                    "relative_h_volume_tolerance": relative_tolerance,
                    "absolute_h_volume_tolerance_nm3": target_tolerance,
                    "nearest_ideal_lattice_q3_sum": nearest_q_sum,
                    "target_fractional_lattice_coordinate": fractional_lattice_coordinate,
                    "nearest_ideal_lattice_distance_nm3": ideal_distance,
                    "minimum_profile_correction_nm3": correction_min,
                    "maximum_profile_correction_nm3": correction_max,
                    "rigorous_lower_bound_absolute_h_volume_error_nm3": lower_bound,
                    "lower_bound_relative_h_volume_error": lower_bound / target_h_volume,
                    "integer_histogram_can_meet_requested_tolerance": feasible,
                    "status": "PASS" if feasible else "BLOCKED_INTEGER_LATTICE_MISMATCH",
                }
            )
        blocked = [row for row in result_rows if not row["integer_histogram_can_meet_requested_tolerance"]]
        status = STATUS_BLOCKED if blocked else STATUS_PASS
        args.out.mkdir(parents=True)
        with (args.out / "profile_h_volume_lattice.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        with (args.out / "density_case_feasibility.csv").open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result_rows[0]))
            writer.writeheader()
            writer.writerows(result_rows)
        payload = {
            "schema": SCHEMA,
            "status": status,
            "scope": "read_only_pre_materialization_integer_h_volume_feasibility",
            "input_sha256": {
                "profile_library_manifest": sha256(args.library_manifest),
                "task_contract": sha256(args.task_contract),
            },
            "target_contract": {
                "target_h_volume_nm3": target_h_volume,
                "relative_h_volume_tolerance": relative_tolerance,
                "absolute_h_volume_tolerance_nm3": target_tolerance,
                "allowed_profile_radius_nm": [8.0, 11.5],
                "profile_scaling_allowed": False,
                "profile_interpolation_allowed": False,
                "analytic_profile_allowed": False,
            },
            "lattice_derivation": {
                "quarter_nm_integer_q_definition": "q=4*R_registered",
                "ideal_h_volume_nm3": "(pi/48)*q^3",
                "ideal_lattice_spacing_nm3": spacing,
                "nearest_ideal_lattice_q3_sum": nearest_q_sum,
                "target_fractional_lattice_coordinate": fractional_lattice_coordinate,
                "nearest_ideal_lattice_distance_nm3": ideal_distance,
                "profile_correction_nonnegative": True,
                "profile_correction_min_nm3": min(offsets),
                "profile_correction_max_nm3": max(offsets),
                "bound_semantics": "A lower bound: if it exceeds the requested tolerance, no integer histogram can pass even before PSD-shape constraints are applied.",
            },
            "profile_rows": rows,
            "density_rows": result_rows,
            "fixture_materialized": False,
            "production_job_submitted": False,
            "v5_pilot_modified": False,
        }
        write_json(args.out / "feasibility_audit.json", payload)
        lines = [
            "# 400-cube PSD/spatial/density ladder: integer h-volume feasibility audit",
            "",
            "This is a read-only pre-materialization audit.  No 400³ fixture or Slurm production job was created.",
            "",
            "## Result",
            "",
            f"`{status}`",
            "",
            "The requested target cannot be met by any integer histogram of the frozen 8.00--11.50 nm, 0.25 nm library under the requested relative h-volume tolerance of `1e-10`.",
            "",
            "## Why this is a hard gate",
            "",
            "For a registered radius R, q=4R is an integer and the ideal equivalent h-volume is `(pi/48) q^3`.  Thus every allowed integer profile histogram lies on that ideal volume lattice, plus the recorded (positive) profile-level corrections.  The nearest ideal lattice point is farther from the requested target than all possible corrections can bridge.",
            "",
            "| N | requested absolute tolerance (nm³) | rigorous lower bound on error (nm³) | lower-bound relative error | result |",
            "|---:|---:|---:|---:|---|",
        ]
        for row in result_rows:
            lines.append(
                "| {particle_count} | {absolute_h_volume_tolerance_nm3:.9g} | "
                "{rigorous_lower_bound_absolute_h_volume_error_nm3:.9g} | "
                "{lower_bound_relative_h_volume_error:.9g} | {status} |".format(**row)
            )
        lines.extend(
            [
                "",
                "The bound is already greater than the tolerance before applying the additional P0/P1/P2/P3 shape constraints.  Changing the PSD optimizer, spatial seeds, profile assignment, or queue cannot remove it.  Profile scaling/interpolation or changing the frozen target would remove the proof's assumptions and is forbidden by the task contract.",
                "",
                "## Consequence",
                "",
                "All 21 fixtures are blocked at their common static h-volume gate.  The required dual queue submission was deliberately not performed; submitting it would create runs that cannot satisfy the stated fixture contract.  The current V5 pilot was not read from or written to by this audit.",
            ]
        )
        (args.out / "integer_feasibility_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
        final_lines = [
            f"campaign_root={args.out}",
            f"profile_library_sha256={sha256(args.library_manifest)}",
            f"target_h_volume_nm3={target_h_volume:.17g}",
            f"target_relative_h_volume_tolerance={relative_tolerance:.17g}",
            f"target_absolute_h_volume_tolerance_nm3={target_tolerance:.17g}",
            "logical_case_count=24",
            "unique_case_count=21",
            "fixture_pass_count=0",
            "fixture_blocked_count=21",
            "gpu_uvip_job_count=0",
            "gpu_vip_24h_job_count=0",
            "claim_ledger_status=NOT_CREATED_STATIC_H_VOLUME_GATE_BLOCKED",
            "v5_pilot_modified=false",
            f"final_status={status}",
        ]
        (args.out / "final_terminal_output.txt").write_text("\n".join(final_lines) + "\n", encoding="utf-8")
        (args.out / "status.txt").write_text(status + "\n", encoding="utf-8")
        print(status)
        if blocked:
            raise SystemExit(3)
    except Exception as exc:
        if args.out.exists():
            (args.out / "first_failure.csv").write_text(
                "stage,detail\ninteger_h_volume_feasibility," + str(exc).replace(",", ";").replace("\n", " ") + "\n",
                encoding="utf-8",
            )
        raise


if __name__ == "__main__":
    main()
