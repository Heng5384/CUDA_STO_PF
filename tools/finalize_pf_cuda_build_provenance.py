#!/usr/bin/env python3
"""Bind a compiled CUDA validation binary to its pre-build provenance record."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping


SCHEMA_VERSION = "PF_CUDA_CONTROLLED_BINARY_MANIFEST_V1"


class FinalizationError(ValueError):
    """Raised when a build artifact cannot be bound to its declared identity."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_hash(rows: Iterable[Mapping[str, str]]) -> str:
    payload = json.dumps(
        list(rows), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise FinalizationError(f"cannot read build provenance JSON: {path}") from error
    if document.get("schema_version") != "PF_CUDA_BUILD_PROVENANCE_V1":
        raise FinalizationError("unexpected build provenance schema")
    return document


def _embedded_json(header: Path) -> dict[str, Any]:
    text = header.read_text(encoding="utf-8")
    match = re.search(r"^#define PF_CUDA_BUILD_PROVENANCE_JSON (.+)$", text, re.MULTILINE)
    if match is None:
        raise FinalizationError("compiled provenance header has no JSON macro")
    try:
        encoded = json.loads(match.group(1))
        document = json.loads(encoded)
    except json.JSONDecodeError as error:
        raise FinalizationError("compiled provenance header contains invalid JSON") from error
    if not isinstance(document, dict):
        raise FinalizationError("compiled provenance header JSON is not an object")
    return document


def _directory_rows(build_dir: Path, exclude: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(build_dir.rglob("*")):
        if not path.is_file() or path.resolve() == exclude.resolve():
            continue
        rows.append(
            {
                "path": str(path.relative_to(build_dir)),
                "sha256": _sha256_file(path),
            }
        )
    return rows


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--build-provenance", type=Path, required=True)
    parser.add_argument("--build-header", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--allow-uncontrolled",
        action="store_true",
        help="test/debug only: emit a manifest for a non-controlled build",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    binary = args.binary.resolve()
    build_provenance = args.build_provenance.resolve()
    build_header = args.build_header.resolve()
    output = args.output.resolve()
    if not binary.is_file() or binary.stat().st_size == 0:
        raise FinalizationError(f"compiled CUDA binary is missing or empty: {binary}")
    if output == binary or output == build_provenance or output == build_header:
        raise FinalizationError("manifest output must not overwrite a build input")

    provenance = _load_json(build_provenance)
    embedded = _embedded_json(build_header)
    if embedded != provenance:
        raise FinalizationError("build header does not embed the companion provenance JSON exactly")
    controlled = bool(provenance.get("controlled_binary_eligible"))
    if not controlled and not args.allow_uncontrolled:
        raise FinalizationError("refusing to finalize an uncontrolled or dirty-source CUDA binary")

    build_dir = binary.parent
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = _directory_rows(build_dir, output)
    document = {
        "schema_version": SCHEMA_VERSION,
        "status": (
            "PASS_CONTROLLED_CUDA_BINARY_PROVENANCE_V1"
            if controlled
            else "UNCONTROLLED_CUDA_BINARY_PROVENANCE_V1"
        ),
        "controlled_binary_eligible": controlled,
        "build_provenance": provenance,
        "artifacts": {
            "binary": {
                "path": str(binary.name),
                "sha256": _sha256_file(binary),
                "bytes": binary.stat().st_size,
            },
            "build_provenance_json": {
                "path": str(build_provenance.relative_to(build_dir)),
                "sha256": _sha256_file(build_provenance),
            },
            "compiled_provenance_header": {
                "path": str(build_header.relative_to(build_dir)),
                "sha256": _sha256_file(build_header),
            },
            "build_directory": {
                "sha256": _canonical_hash(rows),
                "file_count": len(rows),
                "files": rows,
            },
        },
    }
    output.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "binary_sha256": document["artifacts"]["binary"]["sha256"],
                "build_directory_sha256": document["artifacts"]["build_directory"]["sha256"],
                "manifest": str(output),
                "status": document["status"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except FinalizationError as error:
        raise SystemExit(f"[fatal] {error}") from error
