#!/usr/bin/env python3
"""Tile an accepted P1 elastic seed state for cubic cost profiling only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def read_ascii_structured_points(path: Path) -> tuple[np.ndarray, tuple[int, int, int]]:
    text = path.read_text()
    marker = "LOOKUP_TABLE default\n"
    if marker not in text:
        raise ValueError(f"missing VTK lookup-table marker: {path}")
    dimensions = None
    for line in text.splitlines():
        if line.startswith("DIMENSIONS "):
            dimensions = tuple(int(value) for value in line.split()[1:4])
            break
    if dimensions is None:
        raise ValueError(f"missing VTK dimensions: {path}")
    values = np.fromstring(text.split(marker, 1)[1], sep=" ", dtype=np.float64)
    if values.size != int(np.prod(dimensions)):
        raise ValueError(
            f"VTK value count {values.size} does not match {dimensions}: {path}"
        )
    return values.reshape(dimensions), dimensions


def tile_field(field: np.ndarray, target_size: int) -> np.ndarray:
    if field.ndim != 3 or len(set(field.shape)) != 1:
        raise ValueError(f"source must be cubic, got {field.shape}")
    source_size = field.shape[0]
    if target_size % source_size:
        raise ValueError(
            f"target size {target_size} is not divisible by source size {source_size}"
        )
    repeats = target_size // source_size
    return np.tile(field, (repeats, repeats, repeats))


def h_poly(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def alpha_stable(phi: np.ndarray) -> np.ndarray:
    return np.where(phi > 0.5, h_poly(1.0 - phi), 1.0 - h_poly(phi))


def reconstruct_ctot(phi: np.ndarray, x_b_alpha: np.ndarray) -> np.ndarray:
    alpha = alpha_stable(phi)
    return (1.0 - alpha) + alpha * x_b_alpha


def build_metadata(
    size: int, source_label: str, source_dimensions: tuple[int, int, int]
) -> dict[str, object]:
    return {
        "schema": "ctot_checkpoint_v1",
        "Nx": size,
        "Ny": size,
        "Nz": size,
        "dx_nm": 0.05,
        "interface_width_nm": 0.6,
        "dtype": "float64",
        "order": "C",
        "authoritative_state": "Ctot",
        "transport_operator_name": "mimetic_shared_face_v1",
        "transport_operator_version": "1",
        "gradient_operator_name": "periodic_positive_face_difference_v1",
        "divergence_operator_name": "periodic_face_incidence_v1",
        "adjoint_identity_mode": "D_equals_negative_G_star_exact",
        "face_mobility_mode": "symmetric_harmonic_zero_endpoint_v1",
        "phase_solver_name": "semismooth_pdas_v1",
        "phase_solver_version": "1",
        "finite_interface_correction_enabled": 0,
        "step": 0,
        "time_code": 0.0,
        "migrated_from_legacy_restart": False,
        "reference_type": "tiled_accepted_p1_elastic_cost_profile_only",
        "source_label": source_label,
        "source_dimensions": list(source_dimensions),
        "physical_benchmark_evidence": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, choices=(32, 64), required=True)
    parser.add_argument("--source-label", required=True)
    args = parser.parse_args()

    inputs = {
        "phi": "phi_init.vtk",
        "xB": "xB_alpha_active_init.vtk",
        "Ctot": "Ctot_init.vtk",
    }
    fields: dict[str, np.ndarray] = {}
    source_dimensions = None
    for name, filename in inputs.items():
        field, dimensions = read_ascii_structured_points(args.source_dir / filename)
        if source_dimensions is None:
            source_dimensions = dimensions
        elif dimensions != source_dimensions:
            raise ValueError(f"inconsistent source dimensions: {dimensions}")
        fields[name] = tile_field(field, args.size)
    source_ctot = fields["Ctot"]
    fields["Ctot"] = reconstruct_ctot(fields["phi"], fields["xB"])

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    fields["phi"].tofile(out / "phi_init.raw")
    fields["xB"].tofile(out / "xB_init.raw")
    fields["Ctot"].tofile(out / "Ctot_init.raw")
    metadata = build_metadata(
        args.size, args.source_label, source_dimensions or (0, 0, 0)
    )
    (out / "init_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    profile_metrics = {
        "grid": [args.size, args.size, args.size],
        "profile_role": "representative_3d_cost_profile_only",
        "source_label": args.source_label,
        "source_dimensions": list(source_dimensions or ()),
        "tiling_repeats_per_axis": args.size // int(source_dimensions[0]),
        "phi_min": float(np.min(fields["phi"])),
        "phi_max": float(np.max(fields["phi"])),
        "xB_min": float(np.min(fields["xB"])),
        "xB_max": float(np.max(fields["xB"])),
        "Ctot_min": float(np.min(fields["Ctot"])),
        "Ctot_max": float(np.max(fields["Ctot"])),
        "source_rounded_Ctot_reconstruction_Linf": float(
            np.max(np.abs(source_ctot - fields["Ctot"]))
        ),
        "storage_reconstruction_Linf": float(
            np.max(
                np.abs(
                    fields["Ctot"]
                    - reconstruct_ctot(fields["phi"], fields["xB"])
                )
            )
        ),
        "physical_benchmark_evidence": False,
    }
    (out / "profile_metrics.json").write_text(
        json.dumps(profile_metrics, indent=2) + "\n"
    )
    print(f"tiled_p1_elastic_profile_out={out}")
    print(f"grid={args.size}x{args.size}x{args.size}")
    print(f"source_label={args.source_label}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
