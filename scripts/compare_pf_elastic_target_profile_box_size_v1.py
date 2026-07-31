#!/usr/bin/env python3
"""Compare one elastic target profile across two periodic box sizes."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_manifest(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema") != "PF_ELASTIC_TARGET_PROFILE_V1":
        raise SystemExit(f"{path}: wrong profile schema")
    return payload


def load_field(
    manifest: dict[str, Any], manifest_path: Path, name: str
) -> np.ndarray:
    field = manifest["fields"][name]
    path = manifest_path.parent / field["path"]
    if sha256(path) != field["sha256"]:
        raise SystemExit(f"{path}: field hash mismatch")
    shape = tuple(
        int(manifest["grid"][key]) for key in ("Nx", "Ny", "Nz")
    )
    values = np.fromfile(path, dtype="<f8")
    if values.size != math.prod(shape):
        raise SystemExit(f"{path}: field size mismatch")
    return values.reshape(shape, order="C")


def crop_periodic(
    field: np.ndarray, centroid_nm: list[float], dx_nm: float, size: int
) -> np.ndarray:
    indices = []
    for axis, count in enumerate(field.shape):
        center = float(centroid_nm[axis]) / dx_nm
        # The symmetric even-grid fixtures are centered on an integer voxel.
        # Round to that voxel before cropping.  A raw floor is not stable to
        # harmless O(1e-6)-voxel centroid noise and can introduce an
        # artificial whole-voxel translation between box sizes.
        start = int(math.floor(center + 0.5)) - size // 2
        indices.append(np.arange(start, start + size) % count)
    return field[np.ix_(*indices)]


def find_energy(
    audit_path: Path, radius_nm: float
) -> tuple[float, dict[str, Any]]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if (
        audit.get("status")
        != "PASS_ELASTIC_TARGET_PROFILE_ENERGY_PROVENANCE_V1"
    ):
        raise SystemExit(f"{audit_path}: energy audit is not PASS")
    matches = [
        row
        for row in audit["profiles"]
        if abs(float(row["target_radius_nm"]) - radius_nm) <= 1.0e-9
    ]
    if len(matches) != 1:
        raise SystemExit(f"{audit_path}: expected one matching radius")
    return float(matches[0]["F_el_hat"]), matches[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-profile", type=Path, required=True)
    parser.add_argument("--candidate-profile", type=Path, required=True)
    parser.add_argument("--reference-energy-audit", type=Path, required=True)
    parser.add_argument("--candidate-energy-audit", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--window-nm", type=float, default=64.0)
    parser.add_argument("--radius-relative-max", type=float, default=2.0e-3)
    parser.add_argument("--axis-relative-max", type=float, default=2.0e-2)
    parser.add_argument("--axis-ratio-relative-max", type=float, default=2.0e-2)
    parser.add_argument("--phi-l1-normalized-max", type=float, default=2.0e-2)
    parser.add_argument("--xB-mean-absolute-max", type=float, default=2.0e-5)
    parser.add_argument("--far-xB-absolute-max", type=float, default=2.0e-5)
    parser.add_argument("--elastic-energy-relative-max", type=float, default=5.0e-2)
    parser.add_argument("--mass-relative-max", type=float, default=1.0e-12)
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)
    ref = load_manifest(args.reference_profile)
    cand = load_manifest(args.candidate_profile)
    for key in (
        "temperature_C",
        "lambda_sm_nm",
        "v_B",
        "orientation_label",
        "source_tree_sha256",
        "binary_sha256",
    ):
        if ref[key] != cand[key]:
            raise SystemExit(f"profile invariant differs: {key}")
    radius = float(ref["geometry"]["target_equivalent_radius_nm"])
    if abs(radius - float(cand["geometry"]["target_equivalent_radius_nm"])) > 1.0e-9:
        raise SystemExit("target radii differ")
    dx_ref = float(ref["grid"]["dx_nm"])
    dx_cand = float(cand["grid"]["dx_nm"])
    if dx_ref != dx_cand:
        raise SystemExit("grid spacing differs")
    window_n = int(round(args.window_nm / dx_ref))
    if (
        window_n <= 0
        or window_n > min(
            *(int(ref["grid"][key]) for key in ("Nx", "Ny", "Nz")),
            *(int(cand["grid"][key]) for key in ("Nx", "Ny", "Nz")),
        )
    ):
        raise SystemExit("comparison window does not fit both boxes")

    ref_phi = crop_periodic(
        load_field(ref, args.reference_profile, "phi"),
        ref["geometry"]["periodic_h_centroid_nm"],
        dx_ref,
        window_n,
    )
    cand_phi = crop_periodic(
        load_field(cand, args.candidate_profile, "phi"),
        cand["geometry"]["periodic_h_centroid_nm"],
        dx_cand,
        window_n,
    )
    ref_xb = crop_periodic(
        load_field(ref, args.reference_profile, "xB_alpha"),
        ref["geometry"]["periodic_h_centroid_nm"],
        dx_ref,
        window_n,
    )
    cand_xb = crop_periodic(
        load_field(cand, args.candidate_profile, "xB_alpha"),
        cand["geometry"]["periodic_h_centroid_nm"],
        dx_cand,
        window_n,
    )
    ref_h = crop_periodic(
        load_field(ref, args.reference_profile, "h_phi"),
        ref["geometry"]["periodic_h_centroid_nm"],
        dx_ref,
        window_n,
    )
    cand_h = crop_periodic(
        load_field(cand, args.candidate_profile, "h_phi"),
        cand["geometry"]["periodic_h_centroid_nm"],
        dx_cand,
        window_n,
    )

    phi_l1 = float(
        np.sum(np.abs(ref_phi - cand_phi), dtype=np.float64)
        / max(
            0.5
            * float(
                np.sum(ref_h, dtype=np.float64)
                + np.sum(cand_h, dtype=np.float64)
            ),
            1.0,
        )
    )
    xB_mean_abs = float(
        np.mean(np.abs(ref_xb - cand_xb), dtype=np.float64)
    )
    ref_far_xB = float(ref["composition"]["far_field_xB_mean"])
    cand_far_xB = float(cand["composition"]["far_field_xB_mean"])
    xB_reference_subtracted_mean_abs = float(
        np.mean(
            np.abs(
                (ref_xb - ref_far_xB)
                - (cand_xb - cand_far_xB)
            ),
            dtype=np.float64,
        )
    )
    ref_radius = float(ref["geometry"]["actual_equivalent_radius_nm"])
    cand_radius = float(cand["geometry"]["actual_equivalent_radius_nm"])
    radius_relative = abs(ref_radius - cand_radius) / max(abs(ref_radius), 1e-30)
    ref_axes = np.asarray(ref["geometry"]["ellipsoid_semi_axes_nm"], dtype=float)
    cand_axes = np.asarray(cand["geometry"]["ellipsoid_semi_axes_nm"], dtype=float)
    axis_relative = float(
        np.max(np.abs(ref_axes - cand_axes) / np.maximum(np.abs(ref_axes), 1e-30))
    )
    ref_ratio = float(ref["geometry"]["axis_ratio_major_minor"])
    cand_ratio = float(cand["geometry"]["axis_ratio_major_minor"])
    axis_ratio_relative = abs(ref_ratio - cand_ratio) / max(abs(ref_ratio), 1e-30)
    far_xB_absolute = abs(ref_far_xB - cand_far_xB)
    ref_energy_density, ref_energy_row = find_energy(
        args.reference_energy_audit, radius
    )
    cand_energy_density, cand_energy_row = find_energy(
        args.candidate_energy_audit, radius
    )
    ref_voxels = math.prod(
        int(ref["grid"][key]) for key in ("Nx", "Ny", "Nz")
    )
    cand_voxels = math.prod(
        int(cand["grid"][key]) for key in ("Nx", "Ny", "Nz")
    )
    ref_energy = ref_energy_density * ref_voxels * dx_ref**3
    cand_energy = cand_energy_density * cand_voxels * dx_cand**3
    energy_relative = abs(ref_energy - cand_energy) / max(abs(ref_energy), 1e-30)
    ref_mass = float(ref["constraint_contract"]["mass_error_relative"])
    cand_mass = float(cand["constraint_contract"]["mass_error_relative"])
    checks = {
        "radius": radius_relative <= args.radius_relative_max,
        "axes": axis_relative <= args.axis_relative_max,
        "axis_ratio": axis_ratio_relative <= args.axis_ratio_relative_max,
        "local_phi": phi_l1 <= args.phi_l1_normalized_max,
        "local_xB": xB_mean_abs <= args.xB_mean_absolute_max,
        "far_xB": far_xB_absolute <= args.far_xB_absolute_max,
        "elastic_energy": energy_relative <= args.elastic_energy_relative_max,
        "mass": max(ref_mass, cand_mass) <= args.mass_relative_max,
    }
    passed = all(checks.values())
    audit = {
        "schema": "PF_ELASTIC_TARGET_PROFILE_BOX_SIZE_AUDIT_V1",
        "status": (
            "PASS_ELASTIC_TARGET_PROFILE_BOX_SIZE_REFINEMENT_V1"
            if passed
            else "FAIL_ELASTIC_TARGET_PROFILE_BOX_SIZE_REFINEMENT_V1"
        ),
        "reference_profile": str(args.reference_profile),
        "reference_profile_sha256": sha256(args.reference_profile),
        "candidate_profile": str(args.candidate_profile),
        "candidate_profile_sha256": sha256(args.candidate_profile),
        "source_identity": {
            "reference_commit": ref["source_commit"],
            "candidate_commit": cand["source_commit"],
            "source_tree_sha256": ref["source_tree_sha256"],
            "binary_sha256": ref["binary_sha256"],
            "commit_labels_equal": (
                ref["source_commit"] == cand["source_commit"]
            ),
            "identity_basis": "source_tree_sha256_and_binary_sha256",
        },
        "reference_grid": ref["grid"],
        "candidate_grid": cand["grid"],
        "target_radius_nm": radius,
        "window_nm": args.window_nm,
        "metrics": {
            "radius_relative": radius_relative,
            "axis_relative_max": axis_relative,
            "axis_ratio_relative": axis_ratio_relative,
            "local_phi_l1_normalized": phi_l1,
            "local_xB_mean_absolute": xB_mean_abs,
            "local_xB_reference_subtracted_mean_absolute": (
                xB_reference_subtracted_mean_abs
            ),
            "far_xB_absolute": far_xB_absolute,
            "elastic_energy_relative": energy_relative,
            "reference_F_el_hat_box_mean": ref_energy_density,
            "candidate_F_el_hat_box_mean": cand_energy_density,
            "reference_integrated_F_el_hat_code_nm3": ref_energy,
            "candidate_integrated_F_el_hat_code_nm3": cand_energy,
            "reference_mass_error_relative": ref_mass,
            "candidate_mass_error_relative": cand_mass,
        },
        "thresholds": {
            "radius_relative_max": args.radius_relative_max,
            "axis_relative_max": args.axis_relative_max,
            "axis_ratio_relative_max": args.axis_ratio_relative_max,
            "phi_l1_normalized_max": args.phi_l1_normalized_max,
            "xB_mean_absolute_max": args.xB_mean_absolute_max,
            "far_xB_absolute_max": args.far_xB_absolute_max,
            "elastic_energy_relative_max": args.elastic_energy_relative_max,
            "mass_relative_max": args.mass_relative_max,
        },
        "checks": checks,
        "energy_provenance": {
            "reference": ref_energy_row,
            "candidate": cand_energy_row,
        },
    }
    audit_path = args.out / "box_size_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "final_terminal_output.txt").write_text(
        f"box_size_status={audit['status']}\n"
        f"reference_grid={ref['grid']['Nx']}^3\n"
        f"candidate_grid={cand['grid']['Nx']}^3\n"
        f"target_radius_nm={radius}\n"
        f"axis_ratio_relative={axis_ratio_relative:.17e}\n"
        f"elastic_energy_relative={energy_relative:.17e}\n"
        f"audit_sha256={sha256(audit_path)}\n",
        encoding="utf-8",
    )
    print(json.dumps(audit, indent=2, sort_keys=True))
    if not passed:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
