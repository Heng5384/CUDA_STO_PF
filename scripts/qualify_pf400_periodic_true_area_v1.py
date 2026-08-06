#!/usr/bin/env python3
"""Synthetic qualification for periodic PF400 true-area extraction."""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np


PASS = "PASS_PF400_PERIODIC_TRUE_AREA_V1"


def load_module(path: Path) -> Any:
    spec = importlib.util.spec_from_file_location("pf400_periodic_geometry_v1", path)
    if spec is None or spec.loader is None:
        raise ValueError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def minimal_delta(coordinate: np.ndarray, center: float, period: float) -> np.ndarray:
    return (coordinate - center + period / 2.0) % period - period / 2.0


def sphere_field(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    radius: float,
    width: float,
    dx: float = 1.0,
) -> np.ndarray:
    axes = np.meshgrid(
        *(np.arange(size, dtype=np.float64) * dx for size in shape),
        indexing="ij",
        sparse=True,
    )
    periods = tuple(size * dx for size in shape)
    distance = np.sqrt(
        sum(minimal_delta(axes[i], center[i], periods[i]) ** 2 for i in range(3))
    )
    return 0.5 * (1.0 - np.tanh((distance - radius) / width))


def ellipsoid_field(
    shape: tuple[int, int, int],
    center: tuple[float, float, float],
    axes_nm: tuple[float, float, float],
    width: float,
    dx: float = 1.0,
) -> np.ndarray:
    grid = np.meshgrid(
        *(np.arange(size, dtype=np.float64) * dx for size in shape),
        indexing="ij",
        sparse=True,
    )
    periods = tuple(size * dx for size in shape)
    normalized = np.sqrt(
        sum((minimal_delta(grid[i], center[i], periods[i]) / axes_nm[i]) ** 2 for i in range(3))
    )
    physical_scale = float(np.mean(axes_nm))
    return 0.5 * (1.0 - np.tanh((normalized - 1.0) * physical_scale / width))


def bounded_union(*fields: np.ndarray) -> np.ndarray:
    result = np.ones_like(fields[0])
    for field in fields:
        result *= 1.0 - field
    return 1.0 - result


def run_case(
    geometry: Any,
    name: str,
    phi: np.ndarray,
    dx: float,
    expected_count: int,
) -> tuple[list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    rows, aggregate = geometry.extract_geometry(
        phi, case_id=name, time_h=0.0, dx_nm=dx
    )
    result = {
        "test": name,
        "expected_count": expected_count,
        "observed_count": aggregate["particle_count"],
        "count_pass": aggregate["particle_count"] == expected_count,
        "watertight_pass": bool(aggregate["all_meshes_watertight"]),
        "wrapped_object_count": aggregate["wrapped_object_count"],
        "necked_object_count": aggregate["necked_object_count"],
    }
    return rows, aggregate, result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--geometry-module", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    geometry = load_module(args.geometry_module.resolve())
    checks: list[dict[str, Any]] = []
    sensitivity: list[dict[str, Any]] = []
    shape = (48, 48, 48)
    radius = 8.0
    analytic_area = 4.0 * math.pi * radius**2

    definitions = [
        ("sphere_inside", (24.0, 24.0, 24.0), 0),
        ("sphere_cross_one_seam", (1.0, 24.0, 24.0), 1),
        ("sphere_cross_two_seams", (1.0, 1.0, 24.0), 1),
        ("sphere_cross_three_seams", (1.0, 1.0, 1.0), 1),
    ]
    reference_area = None
    for name, center, minimum_wrap in definitions:
        rows, aggregate, check = run_case(
            geometry, name, sphere_field(shape, center, radius, 1.0), 1.0, 1
        )
        area = float(rows[0]["true_area_nm2"])
        check.update(
            {
                "analytic_area_nm2": analytic_area,
                "measured_area_nm2": area,
                "area_relative_error": abs(area - analytic_area) / analytic_area,
                "wrap_pass": int(aggregate["wrapped_object_count"]) >= minimum_wrap,
            }
        )
        reference_area = area if reference_area is None else reference_area
        check["seam_area_relative_to_inside"] = abs(area - reference_area) / reference_area
        checks.append(check)
        for level in geometry.AREA_LEVELS:
            tag = f"phi{int(round(100 * level)):02d}"
            sensitivity.append(
                {
                    "test": name,
                    "level_set_phi": level,
                    "area_nm2": rows[0][f"area_nm2_{tag}"],
                    "Sv_nm-1": aggregate[f"Sv_true_{tag}_nm-1"],
                    "canonical": level == geometry.PRIMARY_LEVEL,
                }
            )

    separated = bounded_union(
        sphere_field(shape, (14.0, 24.0, 24.0), 6.0, 1.0),
        sphere_field(shape, (34.0, 24.0, 24.0), 6.0, 1.0),
    )
    _, _, separated_check = run_case(geometry, "two_separated_spheres", separated, 1.0, 2)
    separated_check["topology_pass"] = separated_check["observed_count"] == 2
    checks.append(separated_check)

    necked = bounded_union(
        sphere_field(shape, (17.0, 24.0, 24.0), 6.0, 1.0),
        sphere_field(shape, (31.0, 24.0, 24.0), 6.0, 1.0),
    )
    neck_rows, _, neck_check = run_case(geometry, "necked_pair", necked, 1.0, 1)
    neck_check["topology_pass"] = bool(neck_rows[0]["merged_or_necked"])
    checks.append(neck_check)

    ellipsoid = ellipsoid_field(shape, (24.0, 24.0, 24.0), (11.0, 7.0, 5.0), 1.0)
    ellipsoid_rows, _, ellipsoid_check = run_case(geometry, "ellipsoid", ellipsoid, 1.0, 1)
    ellipsoid_check["topology_pass"] = float(ellipsoid_rows[0]["aspect_ratio_1_3"]) > 1.7
    ellipsoid_check["measured_aspect_ratio"] = ellipsoid_rows[0]["aspect_ratio_1_3"]
    checks.append(ellipsoid_check)

    convergence: list[dict[str, Any]] = []
    for n in (32, 48, 64, 96):
        physical_length = 32.0
        dx = physical_length / n
        local_shape = (n, n, n)
        rows, aggregate, check = run_case(
            geometry,
            f"sphere_refinement_n{n}",
            sphere_field(local_shape, (16.0, 16.0, 16.0), 6.0, max(dx, 0.35), dx),
            dx,
            1,
        )
        exact = 4.0 * math.pi * 6.0**2
        error = abs(float(rows[0]["true_area_nm2"]) - exact) / exact
        convergence.append({"grid_n": n, "dx_nm": dx, "area_relative_error": error})
        check.update({"area_relative_error": error, "analytic_area_nm2": exact, "measured_area_nm2": rows[0]["true_area_nm2"]})
        checks.append(check)
    refinement_pass = convergence[-1]["area_relative_error"] <= 0.01 and convergence[-1]["area_relative_error"] < convergence[0]["area_relative_error"]

    origin = sphere_field(shape, (24.0, 24.0, 24.0), radius, 1.0)
    shifted = np.roll(origin, (7, -9, 11), axis=(0, 1, 2))
    origin_rows, _, _ = run_case(geometry, "origin_reference", origin, 1.0, 1)
    shifted_rows, _, shifted_check = run_case(geometry, "origin_integer_translation", shifted, 1.0, 1)
    origin_error = abs(float(origin_rows[0]["true_area_nm2"]) - float(shifted_rows[0]["true_area_nm2"])) / float(origin_rows[0]["true_area_nm2"])
    # VTK receives float32 scalar data; periodic integer translations therefore
    # close to single-precision geometry tolerance rather than byte identity.
    geometry_repeat_tolerance = 1.0e-7
    shifted_check.update({"origin_area_relative_difference": origin_error, "origin_invariance_pass": origin_error <= geometry_repeat_tolerance})
    checks.append(shifted_check)

    for check in checks:
        check["pass"] = bool(
            check.get("count_pass", True)
            and check.get("watertight_pass", True)
            and check.get("wrap_pass", True)
            and check.get("topology_pass", True)
            and check.get("origin_invariance_pass", True)
        )
    seam_pass = all(
        float(check.get("seam_area_relative_to_inside", 0.0)) <= geometry_repeat_tolerance
        for check in checks
        if str(check["test"]).startswith("sphere_cross")
    )
    all_pass = all(bool(check["pass"]) for check in checks) and refinement_pass and seam_pass
    write_csv(args.out / "synthetic_geometry_checks.csv", checks)
    write_csv(args.out / "true_area_threshold_sensitivity.csv", sensitivity)
    write_csv(args.out / "mesh_refinement.csv", convergence)
    status = PASS if all_pass else "FAIL_PF400_PERIODIC_TRUE_AREA_V1"
    report = f"""# PF400 periodic true-area validation

Final status: `{status}`

The qualified object contract is periodic six-neighbour connectivity on
`h(phi)>1e-4`.  The interface contract is object-local periodic unwrapping and
watertight marching cubes at `phi=0.50`; `phi=0.45` and `0.55` are sensitivity
levels only.  Connected objects are not promoted to independent physical
particles, and a low-support object containing multiple `phi>=0.5` cores is
flagged as necked/merged.

The suite covers an interior sphere; spheres crossing one, two and three
periodic seams; two separated spheres; a necked pair; an ellipsoid; an
analytic sphere; four mesh resolutions; three level sets; and integer periodic
translation.  Seam-crossing areas agree with the interior representation to
`{max(float(check.get('seam_area_relative_to_inside', 0.0)) for check in checks):.3e}`.
The finest-grid analytic-sphere area error is
`{convergence[-1]['area_relative_error']:.3%}` and the coarse-grid error is
`{convergence[0]['area_relative_error']:.3%}`.  Refinement acceptance is
`{refinement_pass}`; integer-origin invariance is `{origin_error:.3e}`.

Mesh-volume versus diffuse `h`-volume is reported per object in production;
it is a representation comparison rather than an equality constraint because
the former encloses `phi=0.5` while the latter integrates the interpolation
function over the diffuse interface.
"""
    (args.out / "true_area_validation_report.md").write_text(report, encoding="utf-8")
    provenance = {
        "schema": "PF400_PERIODIC_TRUE_AREA_QUALIFICATION_V1",
        "status": status,
        "geometry_module": str(args.geometry_module.resolve()),
        "geometry_module_sha256": sha256(args.geometry_module.resolve()),
        "qualification_script_sha256": sha256(Path(__file__).resolve()),
        "refinement_pass": refinement_pass,
        "periodic_seam_pass": seam_pass,
        "test_count": len(checks),
    }
    (args.out / "qualification.json").write_text(json.dumps(provenance, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (args.out / "status.txt").write_text(status + "\n", encoding="utf-8")
    print(status)
    return 0 if all_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
