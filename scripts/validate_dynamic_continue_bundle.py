#!/usr/bin/env python3
"""Validate the versioned dynamic-continue bundle and its negative cases."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import tempfile
from pathlib import Path


class BundleError(RuntimeError):
    pass


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def validate_bundle(root: Path, expected_version: str = "dynamic_continue_v1") -> dict[str, object]:
    manifest_path = root / "bundle_manifest.csv"
    if not manifest_path.is_file():
        raise BundleError(f"missing bundle manifest: {manifest_path}")
    with manifest_path.open(newline="", encoding="utf-8") as handle:
        manifest = list(csv.DictReader(handle))
    if not manifest:
        raise BundleError("empty bundle manifest")
    for row in manifest:
        if row["bundle_version"] != expected_version:
            raise BundleError(f"wrong bundle version: {row['bundle_version']}")
        rel = row["relative_path"]
        if not rel or rel.startswith("/") or ".." in Path(rel).parts:
            raise BundleError(f"unsafe bundle path: {rel}")
        path = (root / rel).resolve()
        if root.resolve() not in path.parents:
            raise BundleError(f"bundle path escapes root: {rel}")
        if not path.is_file():
            raise BundleError(f"missing bundle file: {rel}")
        if path.stat().st_size != int(row["size_bytes"]):
            raise BundleError(f"size mismatch: {rel}")
        if digest(path) != row["sha256"]:
            raise BundleError(f"checksum mismatch: {rel}")
        text = path.read_text(encoding="utf-8", errors="ignore")
        if any(token in text for token in ("/Users/", "/home/", "/data/home/", "Results/")):
            raise BundleError(f"non-portable provenance remains in: {rel}")

    csv_path = root / "nucleus_library.dynamic_continue.v1.csv"
    json_path = root / "nucleus_library.dynamic_continue.v1.json"
    with csv_path.open(newline="", encoding="utf-8") as handle:
        csv_rows = list(csv.DictReader(handle))
    payload = json.loads(json_path.read_text(encoding="utf-8"))
    json_rows = payload["entries"]
    if [row["library_entry_id"] for row in csv_rows] != [row["library_entry_id"] for row in json_rows]:
        raise BundleError("CSV/JSON entry order mismatch")
    for csv_row, json_row in zip(csv_rows, json_rows):
        for key, value in csv_row.items():
            if str(json_row.get(key, "")) != value:
                raise BundleError(f"CSV/JSON mismatch entry={csv_row['library_entry_id']} field={key}")
        entry_root = root / csv_row["profile_relative_dir"]
        for required in (
            csv_row["profile_relative_path"],
            csv_row["metadata_relative_path"],
            csv_row["source_dynamic_relative_dir"],
        ):
            if not (root / required).exists():
                raise BundleError(f"entry {csv_row['library_entry_id']} missing {required}")
        if not entry_root.is_dir():
            raise BundleError(f"entry directory missing: {entry_root}")
    return {
        "bundle_version": expected_version,
        "manifest_rows": len(manifest),
        "entry_count": len(csv_rows),
        "bundle_size_bytes": sum(path.stat().st_size for path in root.rglob("*") if path.is_file()),
        "entry_ids": [row["library_entry_id"] for row in csv_rows],
    }


def run_negative_tests(root: Path) -> list[tuple[str, str]]:
    results: list[tuple[str, str]] = []
    with tempfile.TemporaryDirectory(prefix="cuda_bundle_negative_") as tmp:
        tmp_root = Path(tmp) / "bundle"
        shutil.copytree(root, tmp_root)

        missing = tmp_root / "entries/nlib_00006/source_dyn_dir/summary.txt"
        missing.unlink()
        try:
            validate_bundle(tmp_root)
        except BundleError:
            results.append(("missing_source_dynamic", "PASS"))
        else:
            results.append(("missing_source_dynamic", "FAIL"))

        shutil.rmtree(tmp_root)
        shutil.copytree(root, tmp_root)
        manifest = tmp_root / "bundle_manifest.csv"
        text = manifest.read_text(encoding="utf-8").replace("dynamic_continue_v1", "wrong_version", 1)
        manifest.write_text(text, encoding="utf-8")
        try:
            validate_bundle(tmp_root)
        except BundleError:
            results.append(("wrong_bundle_version", "PASS"))
        else:
            results.append(("wrong_bundle_version", "FAIL"))

        shutil.rmtree(tmp_root)
        shutil.copytree(root, tmp_root)
        corrupt = tmp_root / "entries/nlib_dc_T400_xB003/faceted_family_profiles.csv"
        corrupt.write_bytes(corrupt.read_bytes() + b"corrupt")
        try:
            validate_bundle(tmp_root)
        except BundleError:
            results.append(("checksum_mismatch", "PASS"))
        else:
            results.append(("checksum_mismatch", "FAIL"))

        shutil.rmtree(tmp_root)
        shutil.copytree(root, tmp_root)
        manifest = tmp_root / "bundle_manifest.csv"
        lines = manifest.read_text(encoding="utf-8").splitlines()
        lines[1] = lines[1].replace("nucleus_library.dynamic_continue.v1.csv", "../escaped.csv")
        manifest.write_text("\n".join(lines) + "\n", encoding="utf-8")
        try:
            validate_bundle(tmp_root)
        except BundleError:
            results.append(("path_traversal", "PASS"))
        else:
            results.append(("path_traversal", "FAIL"))
    return results


def main() -> None:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--bundle", type=Path, required=True)
    parser.add_argument("--negative-tests", action="store_true")
    args = parser.parse_args()
    summary = validate_bundle(args.bundle.resolve())
    print(json.dumps(summary, sort_keys=True))
    if args.negative_tests:
        results = run_negative_tests(args.bundle.resolve())
        for name, status in results:
            print(f"{name}={status}")
        if any(status != "PASS" for _, status in results):
            raise SystemExit(1)


if __name__ == "__main__":
    main()
