#!/usr/bin/env python3
"""Audit finite-box convergence of a portable elastic profile correction."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from compare_pf_elastic_target_profile_box_size_v1 import (
    crop_periodic,
    find_energy,
    load_field,
    load_manifest,
    sha256,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profiles", type=Path, nargs="+", required=True)
    parser.add_argument(
        "--energy-audits", type=Path, nargs="+", required=True
    )
    parser.add_argument("--far-baseline-xB", type=float, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-nm", type=float, default=64.0)
    parser.add_argument("--radius-relative-max", type=float, default=2.0e-3)
    parser.add_argument("--axis-relative-max", type=float, default=2.0e-2)
    parser.add_argument(
        "--axis-ratio-relative-max", type=float, default=2.0e-2
    )
    parser.add_argument(
        "--phi-l1-normalized-max", type=float, default=2.0e-2
    )
    parser.add_argument(
        "--reference-subtracted-xB-mean-absolute-max",
        type=float,
        default=2.0e-5,
    )
    parser.add_argument(
        "--absolute-far-xB-diagnostic-max", type=float, default=2.0e-5
    )
    parser.add_argument(
        "--elastic-energy-relative-max", type=float, default=5.0e-2
    )
    parser.add_argument("--mass-relative-max", type=float, default=1.0e-12)
    parser.add_argument(
        "--offset-inventory-relative-spread-max",
        type=float,
        default=2.0e-2,
    )
    args = parser.parse_args()

    if len(args.profiles) < 3:
        raise SystemExit("at least three box sizes are required")
    if len(args.profiles) != len(args.energy_audits):
        raise SystemExit("profiles and energy audits must have equal length")
    if not math.isfinite(args.far_baseline_xB):
        raise SystemExit("far baseline must be finite")
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    manifests = [load_manifest(path) for path in args.profiles]
    grids = [
        tuple(int(manifest["grid"][key]) for key in ("Nx", "Ny", "Nz"))
        for manifest in manifests
    ]
    if any(len(set(grid)) != 1 for grid in grids):
        raise SystemExit("only cubic boxes are supported")
    box_sizes = [grid[0] for grid in grids]
    if box_sizes != sorted(box_sizes) or len(set(box_sizes)) != len(box_sizes):
        raise SystemExit("box sizes must be unique and increasing")

    first = manifests[0]
    for manifest in manifests[1:]:
        for key in (
            "temperature_C",
            "lambda_sm_nm",
            "v_B",
            "orientation_label",
            "source_tree_sha256",
            "binary_sha256",
        ):
            if manifest[key] != first[key]:
                raise SystemExit(f"profile invariant differs: {key}")
        if (
            float(manifest["geometry"]["target_equivalent_radius_nm"])
            != float(first["geometry"]["target_equivalent_radius_nm"])
        ):
            raise SystemExit("target radii differ")
        if float(manifest["grid"]["dx_nm"]) != float(first["grid"]["dx_nm"]):
            raise SystemExit("grid spacings differ")

    radius = float(first["geometry"]["target_equivalent_radius_nm"])
    dx_nm = float(first["grid"]["dx_nm"])
    window_n = int(round(args.window_nm / dx_nm))
    if window_n <= 0 or window_n > min(box_sizes):
        raise SystemExit("comparison window does not fit every box")

    fields: list[dict[str, np.ndarray]] = []
    energies: list[float] = []
    energy_rows: list[dict[str, Any]] = []
    offset_inventories: list[float] = []
    for manifest, profile_path, energy_audit, box_n in zip(
        manifests, args.profiles, args.energy_audits, box_sizes
    ):
        cropped: dict[str, np.ndarray] = {}
        for name in ("phi", "xB_alpha", "h_phi"):
            cropped[name] = crop_periodic(
                load_field(manifest, profile_path, name),
                manifest["geometry"]["periodic_h_centroid_nm"],
                dx_nm,
                window_n,
            )
        fields.append(cropped)
        energy_density, energy_row = find_energy(energy_audit, radius)
        energies.append(energy_density * box_n**3 * dx_nm**3)
        energy_rows.append(energy_row)
        far_xb = float(manifest["composition"]["far_field_xB_mean"])
        offset_inventories.append(
            (far_xb - args.far_baseline_xB) * box_n**3
        )

    pairs: list[dict[str, Any]] = []
    for index in range(len(manifests) - 1):
        ref = manifests[index]
        cand = manifests[index + 1]
        ref_fields = fields[index]
        cand_fields = fields[index + 1]
        ref_h_sum = float(np.sum(ref_fields["h_phi"], dtype=np.float64))
        cand_h_sum = float(np.sum(cand_fields["h_phi"], dtype=np.float64))
        phi_l1 = float(
            np.sum(
                np.abs(ref_fields["phi"] - cand_fields["phi"]),
                dtype=np.float64,
            )
            / max(0.5 * (ref_h_sum + cand_h_sum), 1.0)
        )
        ref_far = float(ref["composition"]["far_field_xB_mean"])
        cand_far = float(cand["composition"]["far_field_xB_mean"])
        reference_subtracted_xb = float(
            np.mean(
                np.abs(
                    (ref_fields["xB_alpha"] - ref_far)
                    - (cand_fields["xB_alpha"] - cand_far)
                ),
                dtype=np.float64,
            )
        )
        absolute_xb = float(
            np.mean(
                np.abs(
                    ref_fields["xB_alpha"] - cand_fields["xB_alpha"]
                ),
                dtype=np.float64,
            )
        )
        ref_radius = float(ref["geometry"]["actual_equivalent_radius_nm"])
        cand_radius = float(cand["geometry"]["actual_equivalent_radius_nm"])
        radius_relative = abs(ref_radius - cand_radius) / max(
            abs(ref_radius), 1.0e-30
        )
        ref_axes = np.asarray(
            ref["geometry"]["ellipsoid_semi_axes_nm"], dtype=float
        )
        cand_axes = np.asarray(
            cand["geometry"]["ellipsoid_semi_axes_nm"], dtype=float
        )
        axis_relative = float(
            np.max(
                np.abs(ref_axes - cand_axes)
                / np.maximum(np.abs(ref_axes), 1.0e-30)
            )
        )
        ref_ratio = float(ref["geometry"]["axis_ratio_major_minor"])
        cand_ratio = float(cand["geometry"]["axis_ratio_major_minor"])
        axis_ratio_relative = abs(ref_ratio - cand_ratio) / max(
            abs(ref_ratio), 1.0e-30
        )
        energy_relative = abs(energies[index] - energies[index + 1]) / max(
            abs(energies[index]), 1.0e-30
        )
        far_absolute = abs(ref_far - cand_far)
        ref_mass = float(ref["constraint_contract"]["mass_error_relative"])
        cand_mass = float(cand["constraint_contract"]["mass_error_relative"])
        checks = {
            "radius": radius_relative <= args.radius_relative_max,
            "axes": axis_relative <= args.axis_relative_max,
            "axis_ratio": (
                axis_ratio_relative <= args.axis_ratio_relative_max
            ),
            "local_phi": phi_l1 <= args.phi_l1_normalized_max,
            "portable_local_xB": (
                reference_subtracted_xb
                <= args.reference_subtracted_xB_mean_absolute_max
            ),
            "integrated_elastic_energy": (
                energy_relative <= args.elastic_energy_relative_max
            ),
            "mass": max(ref_mass, cand_mass) <= args.mass_relative_max,
        }
        pairs.append(
            {
                "reference_grid": box_sizes[index],
                "candidate_grid": box_sizes[index + 1],
                "metrics": {
                    "radius_relative": radius_relative,
                    "axis_relative_max": axis_relative,
                    "axis_ratio_relative": axis_ratio_relative,
                    "local_phi_l1_normalized": phi_l1,
                    "portable_local_xB_mean_absolute": (
                        reference_subtracted_xb
                    ),
                    "absolute_local_xB_mean_absolute": absolute_xb,
                    "absolute_far_xB": far_absolute,
                    "integrated_elastic_energy_relative": energy_relative,
                    "reference_integrated_F_el_hat_code_nm3": energies[index],
                    "candidate_integrated_F_el_hat_code_nm3": energies[
                        index + 1
                    ],
                },
                "checks": checks,
                "portable_status": "PASS" if all(checks.values()) else "FAIL",
                "absolute_composition_diagnostic": (
                    "PASS"
                    if far_absolute <= args.absolute_far_xB_diagnostic_max
                    and absolute_xb
                    <= args.reference_subtracted_xB_mean_absolute_max
                    else "FAIL"
                ),
            }
        )

    offset_mean = float(np.mean(offset_inventories, dtype=np.float64))
    offset_relative_spread = (
        (max(offset_inventories) - min(offset_inventories))
        / max(abs(offset_mean), 1.0e-30)
    )
    dilution_pass = (
        offset_relative_spread
        <= args.offset_inventory_relative_spread_max
    )
    portable_pass = dilution_pass and all(
        pair["portable_status"] == "PASS" for pair in pairs
    )
    absolute_pass = all(
        pair["absolute_composition_diagnostic"] == "PASS"
        for pair in pairs
    )
    audit = {
        "schema": "PF_ELASTIC_TARGET_PROFILE_BOX_SERIES_AUDIT_V1",
        "status": (
            "PASS_ELASTIC_TARGET_PROFILE_PORTABLE_CORRECTION_BOX_SIZE_V1"
            if portable_pass
            else "FAIL_ELASTIC_TARGET_PROFILE_PORTABLE_CORRECTION_BOX_SIZE_V1"
        ),
        "absolute_raw_composition_status": (
            "PASS_ABSOLUTE_RAW_COMPOSITION_BOX_SIZE_V1"
            if absolute_pass
            else "FAIL_ABSOLUTE_RAW_COMPOSITION_BOX_SIZE_V1"
        ),
        "scientific_interpretation": (
            "Portable correction is xB_alpha minus the independently "
            "registered far-field baseline. Absolute raw xB remains a "
            "separate diagnostic because fixed single-particle inventory "
            "produces an O(box_volume^-1) uniform offset."
        ),
        "profiles": [
            {
                "path": str(path),
                "sha256": sha256(path),
                "grid_n": box_n,
                "far_xB": float(
                    manifest["composition"]["far_field_xB_mean"]
                ),
                "far_offset_inventory_code": offset,
                "integrated_F_el_hat_code_nm3": energy,
                "energy_provenance": energy_row,
            }
            for path, box_n, manifest, offset, energy, energy_row in zip(
                args.profiles,
                box_sizes,
                manifests,
                offset_inventories,
                energies,
                energy_rows,
            )
        ],
        "target_radius_nm": radius,
        "window_nm": args.window_nm,
        "far_baseline_xB": args.far_baseline_xB,
        "far_offset_inventory_relative_spread": offset_relative_spread,
        "far_offset_inventory_scaling_status": (
            "PASS" if dilution_pass else "FAIL"
        ),
        "pairs": pairs,
        "thresholds": {
            "radius_relative_max": args.radius_relative_max,
            "axis_relative_max": args.axis_relative_max,
            "axis_ratio_relative_max": args.axis_ratio_relative_max,
            "phi_l1_normalized_max": args.phi_l1_normalized_max,
            "reference_subtracted_xB_mean_absolute_max": (
                args.reference_subtracted_xB_mean_absolute_max
            ),
            "absolute_far_xB_diagnostic_max": (
                args.absolute_far_xB_diagnostic_max
            ),
            "elastic_energy_relative_max": (
                args.elastic_energy_relative_max
            ),
            "mass_relative_max": args.mass_relative_max,
            "offset_inventory_relative_spread_max": (
                args.offset_inventory_relative_spread_max
            ),
        },
        "source_identity": {
            "source_tree_sha256": first["source_tree_sha256"],
            "binary_sha256": first["binary_sha256"],
            "commit_labels": [
                manifest["source_commit"] for manifest in manifests
            ],
        },
        "analyzer_sha256": sha256(Path(__file__)),
    }
    audit_path = args.out / "box_series_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "final_terminal_output.txt").write_text(
        f"portable_box_size_status={audit['status']}\n"
        f"absolute_raw_composition_status="
        f"{audit['absolute_raw_composition_status']}\n"
        f"box_sizes={','.join(map(str, box_sizes))}\n"
        f"target_radius_nm={radius:.17g}\n"
        f"offset_inventory_relative_spread="
        f"{offset_relative_spread:.17e}\n"
        f"audit_sha256={hashlib.sha256(audit_path.read_bytes()).hexdigest()}\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not portable_pass:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
