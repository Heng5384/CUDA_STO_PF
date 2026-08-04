#!/usr/bin/env python3
"""Assemble individually qualified elastic profiles into a frozen library."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
from typing import Any


SCHEMA = "PF_ELASTIC_TARGET_PROFILE_LIBRARY_V1"
PROFILE_SCHEMA = "PF_ELASTIC_TARGET_PROFILE_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_json_sha(payload: Any) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profiles",
        type=Path,
        nargs="+",
        required=True,
        help="qualified profile directories or profile_manifest.json paths",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--expected-radii-nm", type=float, nargs="+", required=True)
    parser.add_argument("--radius-absolute-tolerance-nm", type=float, default=1.0e-6)
    parser.add_argument(
        "--allow-mixed-binary-same-source",
        action="store_true",
        help=(
            "permit individually qualified profiles built for different GPU "
            "architectures only when source commit/tree and all physical "
            "invariants are identical"
        ),
    )
    args = parser.parse_args()

    if args.out.exists() and any(args.out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty library root: {args.out}")
    args.out.mkdir(parents=True, exist_ok=True)

    entries: list[dict[str, Any]] = []
    for item in args.profiles:
        manifest_path = item / "profile_manifest.json" if item.is_dir() else item
        if not manifest_path.is_file():
            raise SystemExit(f"missing profile manifest: {manifest_path}")
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema") != PROFILE_SCHEMA:
            raise SystemExit(f"wrong profile schema: {manifest_path}")
        terminal_path = manifest_path.parent / "final_terminal_output.txt"
        if not terminal_path.is_file() or (
            "profile_status=PASS_ELASTIC_CONSTRAINED_TARGET_PROFILE_V1"
            not in terminal_path.read_text(encoding="utf-8")
        ):
            raise SystemExit(f"profile has no exact PASS marker: {manifest_path}")
        for field in manifest["fields"].values():
            path = manifest_path.parent / field["path"]
            if not path.is_file() or sha256(path) != field["sha256"]:
                raise SystemExit(f"profile field hash mismatch: {path}")
        geometry = manifest["geometry"]
        entries.append(
            {
                "target_radius_nm": float(geometry["target_equivalent_radius_nm"]),
                "actual_radius_nm": float(geometry["actual_equivalent_radius_nm"]),
                "axis_ratio_major_minor": float(geometry["axis_ratio_major_minor"]),
                "far_field_xB_mean": float(manifest["composition"]["far_field_xB_mean"]),
                "mass_error_relative": float(
                    manifest["constraint_contract"]["mass_error_relative"]
                ),
                "profile_binary_sha256": str(manifest["binary_sha256"]),
                "profile_manifest_path": os.path.relpath(
                    manifest_path.resolve(), args.out.resolve()
                ),
                "profile_manifest_sha256": sha256(manifest_path),
                "profile_directory_sha256": canonical_json_sha(
                    {
                        "manifest": sha256(manifest_path),
                        "fields": manifest["fields"],
                    }
                ),
                "manifest": manifest,
            }
        )

    entries.sort(key=lambda row: row["target_radius_nm"])
    actual_ladder = [row["target_radius_nm"] for row in entries]
    expected_ladder = sorted(args.expected_radii_nm)
    if len(actual_ladder) != len(expected_ladder):
        raise SystemExit("profile count does not match the registered radius ladder")
    for actual, expected in zip(actual_ladder, expected_ladder):
        if abs(actual - expected) > args.radius_absolute_tolerance_nm:
            raise SystemExit(
                f"radius ladder mismatch: actual={actual}, expected={expected}"
            )
    if len(set(actual_ladder)) != len(actual_ladder):
        raise SystemExit("duplicate target radii")

    invariant_keys = (
        "temperature_C",
        "grid",
        "lambda_sm_nm",
        "v_B",
        "orientation_label",
        "source_tree_sha256",
    )
    reference = entries[0]["manifest"]
    for entry in entries[1:]:
        manifest = entry["manifest"]
        for key in invariant_keys:
            if manifest[key] != reference[key]:
                raise SystemExit(f"library invariant differs for key {key}")
    binary_hashes = sorted(
        {str(entry["manifest"]["binary_sha256"]) for entry in entries}
    )
    source_commits = sorted(
        {str(entry["manifest"].get("source_commit", "UNKNOWN")) for entry in entries}
    )
    if len(source_commits) != 1 and not args.allow_mixed_binary_same_source:
        raise SystemExit(
            "library source labels differ; use the explicit same-source mode "
            "only when the source-tree SHA-256 is identical"
        )
    if len(binary_hashes) != 1 and not args.allow_mixed_binary_same_source:
        raise SystemExit(
            "library invariant differs for key binary_sha256; use the explicit "
            "same-source mixed-binary mode only for a device-qualified extension"
        )

    compact_entries = []
    for entry in entries:
        compact_entries.append(
            {key: value for key, value in entry.items() if key != "manifest"}
        )
    library: dict[str, Any] = {
        "schema": SCHEMA,
        "scientific_role": "radius_resolved_single_variant_elastic_profile_library",
        "interpolation_status": "NOT_YET_QUALIFIED",
        "orientation_scope": reference["orientation_scope"],
        "temperature_C": reference["temperature_C"],
        "grid": reference["grid"],
        "lambda_sm_nm": reference["lambda_sm_nm"],
        "v_B": reference["v_B"],
        "source_commit": (
            source_commits[0]
            if len(source_commits) == 1
            else "MIXED_SOURCE_LABELS_IDENTICAL_TREE"
        ),
        "source_commit_set": source_commits,
        "mixed_source_labels": len(source_commits) != 1,
        "source_tree_sha256": reference["source_tree_sha256"],
        "binary_sha256": (
            binary_hashes[0]
            if len(binary_hashes) == 1
            else "MIXED_PROFILE_BINARIES"
        ),
        "binary_sha256_set": binary_hashes,
        "mixed_binary_profiles": len(binary_hashes) != 1,
        "mixed_binary_contract": (
            "IDENTICAL_SOURCE_COMMIT_TREE_AND_PHYSICS_PER_PROFILE_HASH_PINNED"
            if len(binary_hashes) != 1
            else "SINGLE_BINARY"
        ),
        "radius_ladder_nm": actual_ladder,
        "profile_count": len(entries),
        "profiles": compact_entries,
    }
    library["canonical_content_sha256"] = canonical_json_sha(library)
    manifest_path = args.out / "library_manifest.json"
    manifest_path.write_text(
        json.dumps(library, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    with (args.out / "profile_index.csv").open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        fieldnames = [
            "target_radius_nm",
            "actual_radius_nm",
            "axis_ratio_major_minor",
            "far_field_xB_mean",
            "mass_error_relative",
            "profile_manifest_sha256",
            "profile_directory_sha256",
            "profile_binary_sha256",
            "profile_manifest_path",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(compact_entries)

    terminal = {
        "library_status": "PASS_ELASTIC_TARGET_PROFILE_LIBRARY_ASSEMBLY_V1",
        "profile_count": len(entries),
        "radius_ladder_nm": ",".join(f"{value:g}" for value in actual_ladder),
        "library_manifest_sha256": sha256(manifest_path),
        "canonical_content_sha256": library["canonical_content_sha256"],
        "interpolation_status": library["interpolation_status"],
        "mixed_binary_profiles": str(library["mixed_binary_profiles"]).lower(),
    }
    (args.out / "final_terminal_output.txt").write_text(
        "\n".join(f"{key}={value}" for key, value in terminal.items()) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(terminal, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
