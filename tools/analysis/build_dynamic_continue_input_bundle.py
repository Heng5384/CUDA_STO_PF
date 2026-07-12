#!/usr/bin/env python3
"""Build the small, versioned runtime input bundle from audited staging data.

This tool deliberately takes the staging library and profile cache as explicit
inputs. It does not search Results, and it strips machine-local provenance paths
from the emitted bundle while retaining the scientific row values unchanged.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import shutil
import subprocess
from pathlib import Path


ENTRY_IDS = ("nlib_00006", "nlib_dc_T400_xB003", "nlib_dc_T450_xB003")
BUNDLE_VERSION = "dynamic_continue_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def git_head(repo_root: Path) -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"], text=True
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return "UNKNOWN_SOURCE_COMMIT"


def sanitize_provenance(value: str) -> str:
    if not value:
        return value
    if value.startswith("/"):
        return "EXTERNAL_SOURCE_NOT_PACKAGED"
    return re.sub(r"/(?:Users|home|data/home)/[^,; ]+", "EXTERNAL_SOURCE_NOT_PACKAGED", value)


def load_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    by_id = {row["library_entry_id"]: row for row in rows}
    missing = [entry_id for entry_id in ENTRY_IDS if entry_id not in by_id]
    if missing:
        raise SystemExit(f"missing audited staging entries: {', '.join(missing)}")
    return [dict(by_id[entry_id]) for entry_id in ENTRY_IDS]


def emit_library(bundle: Path, rows: list[dict[str, str]], source_commit: str, source_sha: str) -> None:
    extras = {
        "bundle_version": BUNDLE_VERSION,
        "source_commit": source_commit,
        "source_library_sha256": source_sha,
        "generator_version": "build_dynamic_continue_input_bundle.py:v1",
        "entry_acceptance_status": "BRIDGE_VALIDATED",
    }
    for row in rows:
        entry_id = row["library_entry_id"]
        for key in ("seed_source_path", "bridge_source_dir"):
            row[key] = sanitize_provenance(row.get(key, ""))
        row["notes"] = sanitize_provenance(row.get("notes", ""))
        row.update(extras)
        row["profile_bundle_id"] = f"{BUNDLE_VERSION}:{entry_id}"
        row["profile_relative_dir"] = f"entries/{entry_id}"
        row["metadata_relative_path"] = f"entries/{entry_id}/seed_profile_metadata.json"
        row["source_dynamic_relative_dir"] = f"entries/{entry_id}/source_dyn_dir"
        row["profile_relative_path"] = f"entries/{entry_id}/faceted_family_profiles.csv"

    fields = list(rows[0])
    csv_path = bundle / "nucleus_library.dynamic_continue.v1.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    json_path = bundle / "nucleus_library.dynamic_continue.v1.json"
    payload = {
        "schema_version": 1,
        "bundle_version": BUNDLE_VERSION,
        "source_commit": source_commit,
        "source_library_sha256": source_sha,
        "generator_version": extras["generator_version"],
        "scientific_status": "VALIDATION_REFERENCE_ONLY",
        "entries": rows,
    }
    json_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def emit_profiles(bundle: Path, cache_root: Path) -> list[Path]:
    emitted: list[Path] = []
    for entry_id in ENTRY_IDS:
        source = cache_root / entry_id
        if not source.is_dir():
            raise SystemExit(f"missing runtime profile source: {source}")
        target = bundle / "entries" / entry_id
        target.mkdir(parents=True, exist_ok=True)
        for relative in (
            Path("faceted_family_profiles.csv"),
            Path("seed_profile_metadata.json"),
            Path("README_profile_dir.md"),
            Path("source_dyn_dir") / "summary.txt",
        ):
            src = source / relative
            if not src.is_file():
                raise SystemExit(f"missing required runtime profile file: {src}")
            dst = target / relative
            dst.parent.mkdir(parents=True, exist_ok=True)
            if relative.name == "seed_profile_metadata.json":
                payload = json.loads(src.read_text(encoding="utf-8"))
                payload["profile_dir"] = f"entries/{entry_id}"
                payload["source_dyn_dir"] = f"entries/{entry_id}/source_dyn_dir"
                payload["summary_txt"] = f"entries/{entry_id}/source_dyn_dir/summary.txt"
                dst.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            elif relative.name == "summary.txt":
                text = src.read_text(encoding="utf-8")
                text = re.sub(r"summary_file\s+: .*", f"summary_file              : entries/{entry_id}/source_dyn_dir/summary.txt", text)
                dst.write_text(text, encoding="utf-8")
            else:
                dst.write_bytes(src.read_bytes().replace(b"\r\n", b"\n"))
            emitted.append(dst)
    return emitted


def emit_manifest(bundle: Path, source_commit: str) -> None:
    files = [
        bundle / "nucleus_library.dynamic_continue.v1.csv",
        bundle / "nucleus_library.dynamic_continue.v1.json",
        *sorted((bundle / "entries").rglob("*")),
    ]
    rows = []
    for path in files:
        if not path.is_file():
            continue
        rel = path.relative_to(bundle).as_posix()
        entry_id = rel.split("/", 2)[1] if rel.startswith("entries/") else "library"
        role = "library" if entry_id == "library" else (
            "profile" if rel.endswith("faceted_family_profiles.csv") else
            "source_dynamic" if rel.endswith("summary.txt") else
            "metadata" if rel.endswith("seed_profile_metadata.json") else "documentation"
        )
        rows.append({
            "bundle_version": BUNDLE_VERSION,
            "entry_id": entry_id,
            "role": role,
            "relative_path": rel,
            "size_bytes": path.stat().st_size,
            "sha256": sha256(path),
            "source_commit": source_commit,
            "scientific_status": "VALIDATION_REFERENCE_ONLY",
        })
    manifest = bundle / "bundle_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def emit_readme(bundle: Path, source_commit: str, source_sha: str) -> None:
    text = f"""# Dynamic-Continue Runtime Input Bundle v1

This is a small, versioned CUDA runtime input bundle containing the three audited
reference entries required by the T380/T400/T450 overlays. It is not a production
GP-growth acceptance bundle. Scientific status: `VALIDATION_REFERENCE_ONLY`.

- bundle version: `{BUNDLE_VERSION}`
- source code commit: `{source_commit}`
- staging library source SHA-256: `{source_sha}`
- profile representation: synthetic runtime-compatible faceted-family profiles
- integrity manifest: `bundle_manifest.csv`

The bundle is addressed through `bundle:<relative-path>` parameter values. The
runtime must verify the manifest before selecting an entry. Historical output-tree
paths and machine-local provenance paths are intentionally absent.
"""
    (bundle / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--staging-library", type=Path, required=True)
    parser.add_argument("--profile-cache-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    repo_root = args.repo_root.resolve()
    bundle = (args.output or repo_root / "data/runtime_profiles/dynamic_continue_v1").resolve()
    bundle.mkdir(parents=True, exist_ok=True)
    rows = load_rows(args.staging_library)
    source_commit = git_head(repo_root)
    source_sha = sha256(args.staging_library)
    emit_library(bundle, rows, source_commit, source_sha)
    emit_profiles(bundle, args.profile_cache_root.resolve())
    emit_readme(bundle, source_commit, source_sha)
    emit_manifest(bundle, source_commit)
    print(f"bundle={bundle}")
    print(f"bundle_version={BUNDLE_VERSION}")
    print(f"entries={','.join(ENTRY_IDS)}")
    print(f"source_library_sha256={source_sha}")


if __name__ == "__main__":
    main()
