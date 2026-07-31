#!/usr/bin/env python3
"""Cross-device/timestep comparison for elastic target-profile libraries."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


LIBRARY_SCHEMA = "PF_ELASTIC_TARGET_PROFILE_LIBRARY_V1"
PROFILE_SCHEMA = "PF_ELASTIC_TARGET_PROFILE_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_library(path: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = path / "library_manifest.json" if path.is_dir() else path
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema") != LIBRARY_SCHEMA:
        raise ValueError(f"{manifest_path}: wrong library schema")
    return manifest, manifest_path


def load_profile(
    library_manifest_path: Path, entry: dict[str, Any]
) -> tuple[dict[str, Any], Path]:
    profile_manifest_path = (
        library_manifest_path.parent / entry["profile_manifest_path"]
    ).resolve()
    if sha256(profile_manifest_path) != entry["profile_manifest_sha256"]:
        raise ValueError(f"{profile_manifest_path}: profile manifest hash mismatch")
    profile = json.loads(profile_manifest_path.read_text(encoding="utf-8"))
    if profile.get("schema") != PROFILE_SCHEMA:
        raise ValueError(f"{profile_manifest_path}: wrong profile schema")
    return profile, profile_manifest_path


def load_field(
    profile: dict[str, Any], profile_manifest_path: Path, name: str
) -> np.ndarray:
    field = profile["fields"][name]
    path = profile_manifest_path.parent / field["path"]
    if sha256(path) != field["sha256"]:
        raise ValueError(f"{path}: field hash mismatch")
    shape = (
        int(profile["grid"]["Nx"]),
        int(profile["grid"]["Ny"]),
        int(profile["grid"]["Nz"]),
    )
    array = np.fromfile(path, dtype="<f8")
    if array.size != math.prod(shape):
        raise ValueError(f"{path}: wrong element count")
    return array.reshape(shape, order="C")


def periodic_delta(a: float, b: float, period: float) -> float:
    delta = abs(a - b)
    return min(delta, period - delta)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--phi-l1-normalized-max", type=float, default=2.0e-2)
    parser.add_argument("--xB-mean-absolute-max", type=float, default=2.0e-5)
    parser.add_argument("--canonical-l1-relative-max", type=float, default=1.0e-4)
    parser.add_argument("--axis-relative-max", type=float, default=2.0e-2)
    parser.add_argument("--centroid-periodic-max-nm", type=float, default=1.0e-1)
    parser.add_argument("--far-xB-absolute-max", type=float, default=2.0e-5)
    parser.add_argument("--require-min-pseudo-time", type=float, default=0.0)
    parser.add_argument(
        "--pseudo-time-alignment-relative-max",
        type=float,
        default=2.5e-2,
        help=(
            "Maximum converged pseudo-time mismatch as a fraction of the "
            "required common relaxation depth. The effective gate is the "
            "larger of this value and one consecutive-step window."
        ),
    )
    args = parser.parse_args()

    if (
        not math.isfinite(args.require_min_pseudo_time)
        or args.require_min_pseudo_time < 0.0
    ):
        raise SystemExit("--require-min-pseudo-time must be finite and >= 0")
    if (
        not math.isfinite(args.pseudo_time_alignment_relative_max)
        or args.pseudo_time_alignment_relative_max < 0.0
    ):
        raise SystemExit(
            "--pseudo-time-alignment-relative-max must be finite and >= 0"
        )
    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty output: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    reference, reference_path = load_library(args.reference)
    candidate, candidate_path = load_library(args.candidate)
    if reference["radius_ladder_nm"] != candidate["radius_ladder_nm"]:
        raise SystemExit("radius ladders differ")
    if reference["source_commit"] != candidate["source_commit"]:
        raise SystemExit("source commits differ")
    if reference["source_tree_sha256"] != candidate["source_tree_sha256"]:
        raise SystemExit("runtime source-tree hashes differ")
    invariant_keys = (
        "temperature_C",
        "grid",
        "lambda_sm_nm",
        "v_B",
        "orientation_scope",
    )
    for key in invariant_keys:
        if reference[key] != candidate[key]:
            raise SystemExit(f"library invariant differs: {key}")

    reference_entries = {
        float(entry["target_radius_nm"]): entry
        for entry in reference["profiles"]
    }
    candidate_entries = {
        float(entry["target_radius_nm"]): entry
        for entry in candidate["profiles"]
    }
    rows: list[dict[str, Any]] = []
    for radius in reference["radius_ladder_nm"]:
        ref_profile, ref_manifest_path = load_profile(
            reference_path, reference_entries[float(radius)]
        )
        cand_profile, cand_manifest_path = load_profile(
            candidate_path, candidate_entries[float(radius)]
        )
        ref_phi = load_field(ref_profile, ref_manifest_path, "phi")
        cand_phi = load_field(cand_profile, cand_manifest_path, "phi")
        ref_xb = load_field(ref_profile, ref_manifest_path, "xB_alpha")
        cand_xb = load_field(cand_profile, cand_manifest_path, "xB_alpha")
        ref_c = load_field(ref_profile, ref_manifest_path, "C_B_tot")
        cand_c = load_field(cand_profile, cand_manifest_path, "C_B_tot")
        ref_h = load_field(ref_profile, ref_manifest_path, "h_phi")
        cand_h = load_field(cand_profile, cand_manifest_path, "h_phi")

        phi_l1_normalized = float(
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
        xB_mean_absolute = float(
            np.mean(np.abs(ref_xb - cand_xb), dtype=np.float64)
        )
        canonical_l1_relative = float(
            np.sum(np.abs(ref_c - cand_c), dtype=np.float64)
            / max(float(np.sum(np.abs(ref_c), dtype=np.float64)), 1.0)
        )
        ref_axes = np.asarray(
            ref_profile["geometry"]["ellipsoid_semi_axes_nm"], dtype=float
        )
        cand_axes = np.asarray(
            cand_profile["geometry"]["ellipsoid_semi_axes_nm"], dtype=float
        )
        axis_relative_max = float(
            np.max(np.abs(ref_axes - cand_axes) / np.maximum(np.abs(ref_axes), 1e-30))
        )
        ref_centroid = ref_profile["geometry"]["periodic_h_centroid_nm"]
        cand_centroid = cand_profile["geometry"]["periodic_h_centroid_nm"]
        grid = ref_profile["grid"]
        periods = [
            float(grid["Nx"]) * float(grid["dx_nm"]),
            float(grid["Ny"]) * float(grid["dx_nm"]),
            float(grid["Nz"]) * float(grid["dx_nm"]),
        ]
        centroid_periodic_max_nm = max(
            periodic_delta(float(a), float(b), period)
            for a, b, period in zip(ref_centroid, cand_centroid, periods)
        )
        far_xB_absolute = abs(
            float(ref_profile["composition"]["far_field_xB_mean"])
            - float(cand_profile["composition"]["far_field_xB_mean"])
        )
        ref_marker = ref_profile["constraint_contract"]["final_marker"]
        cand_marker = cand_profile["constraint_contract"]["final_marker"]
        ref_iter = int(ref_marker["convergence_converged_iter"])
        cand_iter = int(cand_marker["convergence_converged_iter"])
        ref_pseudo_marker = ref_marker.get(
            "convergence_converged_pseudo_time"
        )
        cand_pseudo_marker = cand_marker.get(
            "convergence_converged_pseudo_time"
        )
        ref_dt = (
            float(ref_pseudo_marker) / ref_iter
            if ref_pseudo_marker is not None and ref_iter > 0
            else math.nan
        )
        cand_dt = (
            float(cand_pseudo_marker) / cand_iter
            if cand_pseudo_marker is not None and cand_iter > 0
            else math.nan
        )
        ref_min_pseudo_time = float(
            ref_marker.get("convergence_min_pseudo_time", 0.0)
        )
        cand_min_pseudo_time = float(
            cand_marker.get("convergence_min_pseudo_time", 0.0)
        )
        ref_converged_pseudo_time = float(
            ref_marker.get(
                "convergence_converged_pseudo_time",
                0.0,
            )
        )
        cand_converged_pseudo_time = float(
            cand_marker.get(
                "convergence_converged_pseudo_time",
                0.0,
            )
        )
        pseudo_time_difference = abs(
            ref_converged_pseudo_time - cand_converged_pseudo_time
        )
        convergence_steps = max(
            int(ref_marker["convergence_consecutive_required"]),
            int(cand_marker["convergence_consecutive_required"]),
        )
        consecutive_window = (
            max(ref_dt, cand_dt) * convergence_steps + 1.0e-12
            if math.isfinite(ref_dt) and math.isfinite(cand_dt)
            else math.nan
        )
        relative_window = (
            args.pseudo_time_alignment_relative_max
            * args.require_min_pseudo_time
            if args.require_min_pseudo_time > 0.0
            else 0.0
        )
        pseudo_time_alignment_max = (
            max(consecutive_window, relative_window)
            if math.isfinite(consecutive_window)
            else math.nan
        )
        checks = {
            "phi_l1": phi_l1_normalized <= args.phi_l1_normalized_max,
            "xB_l1": xB_mean_absolute <= args.xB_mean_absolute_max,
            "canonical_l1": (
                canonical_l1_relative <= args.canonical_l1_relative_max
            ),
            "shape_axes": axis_relative_max <= args.axis_relative_max,
            "centroid": (
                centroid_periodic_max_nm <= args.centroid_periodic_max_nm
            ),
            "far_xB": far_xB_absolute <= args.far_xB_absolute_max,
            "minimum_pseudo_time": (
                ref_min_pseudo_time >= args.require_min_pseudo_time
                and cand_min_pseudo_time >= args.require_min_pseudo_time
                and ref_converged_pseudo_time >= args.require_min_pseudo_time
                and cand_converged_pseudo_time >= args.require_min_pseudo_time
            ),
            "pseudo_time_alignment": (
                (
                    math.isfinite(pseudo_time_alignment_max)
                    and pseudo_time_difference <= pseudo_time_alignment_max
                )
                if args.require_min_pseudo_time > 0.0
                else (
                    not math.isfinite(pseudo_time_alignment_max)
                    or pseudo_time_difference <= pseudo_time_alignment_max
                )
            ),
        }
        rows.append(
            {
                "target_radius_nm": radius,
                "phi_l1_normalized": phi_l1_normalized,
                "xB_mean_absolute": xB_mean_absolute,
                "canonical_l1_relative": canonical_l1_relative,
                "axis_relative_max": axis_relative_max,
                "centroid_periodic_max_nm": centroid_periodic_max_nm,
                "far_xB_absolute": far_xB_absolute,
                "reference_min_pseudo_time": ref_min_pseudo_time,
                "candidate_min_pseudo_time": cand_min_pseudo_time,
                "reference_converged_pseudo_time": ref_converged_pseudo_time,
                "candidate_converged_pseudo_time": cand_converged_pseudo_time,
                "pseudo_time_difference": pseudo_time_difference,
                "pseudo_time_alignment_max": pseudo_time_alignment_max,
                "reference_axis_ratio": ref_profile["geometry"][
                    "axis_ratio_major_minor"
                ],
                "candidate_axis_ratio": cand_profile["geometry"][
                    "axis_ratio_major_minor"
                ],
                "status": "PASS" if all(checks.values()) else "FAIL",
                "failed_checks": ",".join(
                    name for name, passed in checks.items() if not passed
                ),
            }
        )

    passed = all(row["status"] == "PASS" for row in rows)
    csv_path = args.out / "cross_device_profile_comparison.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    audit = {
        "schema": "PF_ELASTIC_TARGET_PROFILE_CROSS_DEVICE_AUDIT_V1",
        "status": (
            "PASS_ELASTIC_TARGET_PROFILE_DT_AND_DEVICE_REFINEMENT_V1"
            if passed
            else "FAIL_ELASTIC_TARGET_PROFILE_DT_OR_DEVICE_REFINEMENT_V1"
        ),
        "reference_library_manifest": str(reference_path),
        "reference_library_sha256": sha256(reference_path),
        "candidate_library_manifest": str(candidate_path),
        "candidate_library_sha256": sha256(candidate_path),
        "common_source_commit": reference["source_commit"],
        "common_source_tree_sha256": reference["source_tree_sha256"],
        "different_binary_hashes_expected": (
            reference["binary_sha256"] != candidate["binary_sha256"]
        ),
        "analyzer_sha256": sha256(Path(__file__)),
        "thresholds": {
            "phi_l1_normalized_max": args.phi_l1_normalized_max,
            "xB_mean_absolute_max": args.xB_mean_absolute_max,
            "canonical_l1_relative_max": args.canonical_l1_relative_max,
            "axis_relative_max": args.axis_relative_max,
            "centroid_periodic_max_nm": args.centroid_periodic_max_nm,
            "far_xB_absolute_max": args.far_xB_absolute_max,
            "require_min_pseudo_time": args.require_min_pseudo_time,
            "pseudo_time_alignment_relative_max": (
                args.pseudo_time_alignment_relative_max
            ),
        },
        "profiles": rows,
    }
    audit_path = args.out / "cross_device_audit.json"
    audit_path.write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "final_terminal_output.txt").write_text(
        f"cross_device_status={audit['status']}\n"
        f"profile_count={len(rows)}\n"
        f"common_source_tree_sha256={reference['source_tree_sha256']}\n"
        f"audit_sha256={sha256(audit_path)}\n",
        encoding="utf-8",
    )
    if not passed:
        raise SystemExit(2)
    print(json.dumps(audit, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
