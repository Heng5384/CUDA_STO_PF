#!/usr/bin/env python3
"""Offline CNT scan and nucleus-library generation pipeline.

This is Track 1 of the dual-track architecture. It is analysis/calibration code
only. Runtime PF evolution must not import this module.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = REPO_ROOT / "Results" / "dual_track_calibration"


@dataclass(frozen=True)
class CNTCase:
    T_C: float
    xB: float
    strain: float
    strain_mode: str = "eyy"
    gamma_J_m2: float = 0.05
    delta_g_bulk_J_m3: float = 1.0e8
    elastic_penalty_J_m3: float = 0.0

    @property
    def T_K(self) -> float:
        return self.T_C + 273.15


@dataclass(frozen=True)
class ShapeDescriptor:
    shape_type: str
    aspect_ratio: float
    axis_ratios: tuple[float, float, float]
    surface_factor: float
    source: str = "offline_shape_descriptor"


SHAPE_LIBRARY = [
    ShapeDescriptor("spherical", 1.0, (1.0, 1.0, 1.0), 1.0),
    ShapeDescriptor("faceted", 1.25, (1.25, 1.0, 0.8), 1.08),
    ShapeDescriptor("anisotropic", 1.5, (1.5, 1.0, 0.67), 1.18),
]


def delta_g_radius_J(r_nm: float, case: CNTCase, shape: ShapeDescriptor) -> float:
    """CNT free energy DeltaG(r) for offline scanning."""

    r_m = float(r_nm) * 1.0e-9
    effective_drive = max(case.delta_g_bulk_J_m3 - case.elastic_penalty_J_m3, 1.0e-30)
    surface = 4.0 * math.pi * r_m * r_m * case.gamma_J_m2 * shape.surface_factor
    volume = (4.0 / 3.0) * math.pi * r_m * r_m * r_m * effective_drive
    return surface - volume


def r_scan(
    case: CNTCase,
    r_min_nm: float,
    r_max_nm: float,
    dr_nm: float,
    shapes: list[ShapeDescriptor] | None = None,
) -> list[dict[str, Any]]:
    """Scan DeltaG(r, T, xB, strain) for each shape descriptor."""

    if dr_nm <= 0.0:
        raise ValueError("dr_nm must be positive")
    rows: list[dict[str, Any]] = []
    shape_set = shapes or SHAPE_LIBRARY
    n_steps = int(math.floor((r_max_nm - r_min_nm) / dr_nm)) + 1
    for shape in shape_set:
        for i in range(max(n_steps, 0)):
            r_nm = r_min_nm + i * dr_nm
            rows.append(
                {
                    "T_C": case.T_C,
                    "xB": case.xB,
                    "strain": case.strain,
                    "strain_mode": case.strain_mode,
                    "shape_type": shape.shape_type,
                    "aspect_ratio": shape.aspect_ratio,
                    "r_nm": r_nm,
                    "DeltaG_J": delta_g_radius_J(r_nm, case, shape),
                }
            )
    return rows


def extract_critical_nuclei(scan_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Extract r* and DeltaG* as the maximum along each scanned shape curve."""

    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = {}
    for row in scan_rows:
        key = (row["T_C"], row["xB"], row["strain"], row["strain_mode"], row["shape_type"])
        grouped.setdefault(key, []).append(row)

    entries: list[dict[str, Any]] = []
    for key, rows in grouped.items():
        best = max(rows, key=lambda item: item["DeltaG_J"])
        T_C, xB, strain, strain_mode, shape_type = key
        kBT = 1.380649e-23 * (float(T_C) + 273.15)
        shape = next((s for s in SHAPE_LIBRARY if s.shape_type == shape_type), SHAPE_LIBRARY[0])
        entries.append(
            {
                "id": f"offline_cnt_T{float(T_C):.3f}_xB{float(xB):.5f}_{strain_mode}_{float(strain):.5f}_{shape_type}",
                "T_C": float(T_C),
                "xB": float(xB),
                "strain_mode": strain_mode,
                "strain_value": float(strain),
                "rc_nm": float(best["r_nm"]),
                "energy_barrier_J": float(best["DeltaG_J"]),
                "energy_barrier_kBT": float(best["DeltaG_J"]) / kBT,
                "shape_type": shape_type,
                "aspect_ratio": shape.aspect_ratio,
                "axis_ratios": list(shape.axis_ratios),
                "surface_factor": shape.surface_factor,
                "source_dyn_dir": "",
                "profile_dir": "",
                "confidence_score": 0.5,
                "origin": "CNT_offline_scan",
                "runtime_usable": False,
                "runtime_note": "Offline calibration only; PF runtime consumes fitted S-field/barrier surfaces, not this entry directly.",
            }
        )
    return sorted(entries, key=lambda item: (item["T_C"], item["xB"], item["strain_value"], item["energy_barrier_kBT"]))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        if fieldnames is None:
            fieldnames = []
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=names)
        writer.writeheader()
        writer.writerows(rows)


def write_shape_library(output_dir: Path) -> Path:
    shape_dir = output_dir / "shape_library"
    shape_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "library_role": "offline critical nucleus geometry descriptors",
        "runtime_policy": "calibration_only_not_runtime_selection",
        "shapes": [asdict(shape) for shape in SHAPE_LIBRARY],
    }
    path = shape_dir / "shape_descriptors.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_catalog(output_dir: Path, entries: list[dict[str, Any]]) -> Path:
    payload = {
        "schema_version": 2,
        "system_type": "dual-track hybrid PF + CNT calibrated nucleation system",
        "track": "TRACK_1_OFFLINE_CNT_SCANNING",
        "runtime_policy": "nucleus library is calibration-only; runtime PF does not select insertion templates from this catalog",
        "entry_count": len(entries),
        "entries": entries,
    }
    path = output_dir / "nucleus_catalog.json"
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return path


def write_barrier_surface_fit(output_dir: Path, entries: list[dict[str, Any]]) -> Path:
    """Write a deterministic calibration table for DeltaG*(T,xB,strain)."""

    rows = [
        {
            "T_C": e["T_C"],
            "xB": e["xB"],
            "strain": e["strain_value"],
            "strain_mode": e["strain_mode"],
            "shape_type": e["shape_type"],
            "rc_nm": e["rc_nm"],
            "DeltaG_star_kBT": e["energy_barrier_kBT"],
            "fit_family": "nearest_or_interpolated_surface",
            "runtime_role": "calibration_lookup_only",
        }
        for e in entries
    ]
    path = output_dir / "barrier_surface_fit.csv"
    write_csv(path, rows, fieldnames=[
        "T_C",
        "xB",
        "strain",
        "strain_mode",
        "shape_type",
        "rc_nm",
        "DeltaG_star_kBT",
        "fit_family",
        "runtime_role",
    ])
    return path


def run_offline_cnt_pipeline(
    case: CNTCase,
    output_dir: Path,
    r_min_nm: float,
    r_max_nm: float,
    dr_nm: float,
) -> dict[str, str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    scan_rows = r_scan(case, r_min_nm, r_max_nm, dr_nm)
    scan_path = output_dir / "cnt_r_scan.csv"
    write_csv(scan_path, scan_rows)
    entries = extract_critical_nuclei(scan_rows)
    catalog_path = write_catalog(output_dir, entries)
    barrier_path = write_barrier_surface_fit(output_dir, entries)
    shape_path = write_shape_library(output_dir)
    return {
        "cnt_r_scan_csv": str(scan_path),
        "nucleus_catalog_json": str(catalog_path),
        "barrier_surface_fit_csv": str(barrier_path),
        "shape_library_json": str(shape_path),
        "runtime_cnt_enabled": "false",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--T-C", type=float, required=True)
    parser.add_argument("--xB", type=float, required=True)
    parser.add_argument("--strain", type=float, default=0.0)
    parser.add_argument("--strain-mode", default="eyy")
    parser.add_argument("--gamma-J-m2", type=float, default=0.05)
    parser.add_argument("--delta-g-bulk-J-m3", type=float, default=1.0e8)
    parser.add_argument("--elastic-penalty-J-m3", type=float, default=0.0)
    parser.add_argument("--r-min-nm", type=float, default=0.25)
    parser.add_argument("--r-max-nm", type=float, default=8.0)
    parser.add_argument("--dr-nm", type=float, default=0.05)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    args = parser.parse_args()

    case = CNTCase(
        T_C=args.T_C,
        xB=args.xB,
        strain=args.strain,
        strain_mode=args.strain_mode,
        gamma_J_m2=args.gamma_J_m2,
        delta_g_bulk_J_m3=args.delta_g_bulk_J_m3,
        elastic_penalty_J_m3=args.elastic_penalty_J_m3,
    )
    outputs = run_offline_cnt_pipeline(case, args.output_dir, args.r_min_nm, args.r_max_nm, args.dr_nm)
    for key, value in outputs.items():
        print(f"{key}={value}")
    print("track_1_offline_only=true")


if __name__ == "__main__":
    main()
