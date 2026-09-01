#!/usr/bin/env python3
"""Generate the compile-time identity for a CUDA validation binary.

The header emitted by this tool is deliberately a build artifact.  It records
the exact source/contract/toolchain identity that was present *before* NVCC
was invoked, and is compiled into ``main_cuda --provenance``.  The companion
JSON is consumed by ``finalize_pf_cuda_build_provenance.py`` after the binary
exists, when its SHA-256 can be recorded without pretending that a binary can
contain its own final hash.

This is a provenance gate, not a parameter source.  It does not select a
physical case or alter any PF/KWN numerical input.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence


SCHEMA_VERSION = "PF_CUDA_BUILD_PROVENANCE_V1"
CONTRACT_SCHEMA = "PF_KWN_VALIDATION_CONTRACT_V1"


class BuildProvenanceError(ValueError):
    """Raised when a binary identity cannot be described faithfully."""


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _canonical_contract_hash(document: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        document,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        allow_nan=False,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _run(command: Sequence[str], *, cwd: Path) -> str:
    try:
        completed = subprocess.run(
            command,
            cwd=cwd,
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError) as error:
        raise BuildProvenanceError(
            f"command failed while collecting build provenance: {' '.join(command)}"
        ) from error
    return completed.stdout.strip()


def _tool_version(command: str) -> str:
    try:
        completed = subprocess.run(
            [command, "--version"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return f"UNAVAILABLE:{command}"
    return " ".join(completed.stdout.split())


def _extract_define(header: Path, name: str) -> str:
    pattern = re.compile(rf'^\s*#define\s+{re.escape(name)}\s+"([^"]+)"\s*$', re.MULTILINE)
    match = pattern.search(header.read_text(encoding="utf-8"))
    if match is None:
        raise BuildProvenanceError(f"missing {name} in generated contract header: {header}")
    return match.group(1)


def _git_identity(source_root: Path) -> tuple[str, bool, list[str]]:
    commit = _run(["git", "rev-parse", "HEAD"], cwd=source_root)
    status = _run(
        ["git", "status", "--porcelain=v1", "--untracked-files=all"],
        cwd=source_root,
    )
    changed = [line for line in status.splitlines() if line]
    return commit, not changed, changed


def _load_json(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise BuildProvenanceError(f"cannot read {label}: {path}") from error
    if not isinstance(document, dict):
        raise BuildProvenanceError(f"{label} must be a JSON object: {path}")
    return document


def _write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def _header_text(document: Mapping[str, Any]) -> str:
    embedded_json = json.dumps(
        document, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    )
    encoded = json.dumps(embedded_json, ensure_ascii=True)
    source_clean = 1 if document["source"]["clean"] else 0
    controlled = 1 if document["controlled_binary_eligible"] else 0
    return "\n".join(
        (
            "// GENERATED FILE. DO NOT EDIT.",
            "// Generator: tools/generate_pf_cuda_build_provenance.py",
            "#ifndef PF_CUDA_BUILD_PROVENANCE_V1_H",
            "#define PF_CUDA_BUILD_PROVENANCE_V1_H",
            "",
            f'#define PF_CUDA_BUILD_PROVENANCE_SCHEMA "{SCHEMA_VERSION}"',
            f'#define PF_CUDA_BUILD_SOURCE_COMMIT "{document["source"]["commit"]}"',
            f"#define PF_CUDA_BUILD_SOURCE_CLEAN {source_clean}",
            f"#define PF_CUDA_BUILD_CONTROLLED_BINARY {controlled}",
            f'#define PF_CUDA_BUILD_VALIDATION_CONTRACT_HASH "{document["contract"]["canonical_hash"]}"',
            f'#define PF_CUDA_BUILD_VALIDATION_CONTRACT_HEADER_SHA256 "{document["contract"]["header_sha256"]}"',
            f'#define PF_CUDA_BUILD_TIMESTAMP_UTC "{document["build_utc"]}"',
            f"#define PF_CUDA_BUILD_PROVENANCE_JSON {encoded}",
            "",
            "#endif  // PF_CUDA_BUILD_PROVENANCE_V1_H",
            "",
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--contract", type=Path, required=True)
    parser.add_argument("--contract-header", type=Path, required=True)
    parser.add_argument("--fixture-spec", type=Path, required=True)
    parser.add_argument("--output-header", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--nvcc", required=True)
    parser.add_argument("--host-cxx", required=True)
    parser.add_argument("--cuda-root", required=True)
    parser.add_argument("--cuda-arch", required=True)
    parser.add_argument("--nvccflags", required=True)
    parser.add_argument("--includes", required=True)
    parser.add_argument("--ldflags", required=True)
    parser.add_argument("--ldlibs", required=True)
    parser.add_argument("--precision", default="IEEE754_BINARY64")
    parser.add_argument("--checkpoint-schema", default="PF_ZERO_MODE_CHECKPOINT_V6")
    parser.add_argument("--checkpoint-disk-magic", default="PFZMCHK6")
    parser.add_argument("--auxiliary-sidecar-schema", default="PF_AUXILIARY_HANDOFF_V2_SIDECAR_V1")
    parser.add_argument("--auxiliary-population-schema", default="AUXILIARY_POPULATION_STORAGE_ONLY_V1")
    parser.add_argument(
        "--require-clean",
        action="store_true",
        help="refuse to create a controlled-binary identity from modified source",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    source_root = args.source_root.resolve()
    contract_path = args.contract.resolve()
    contract_header = args.contract_header.resolve()
    fixture_spec = args.fixture_spec.resolve()
    output_header = args.output_header.resolve()
    output_json = args.output_json.resolve()
    if output_header == output_json:
        raise BuildProvenanceError("build provenance header and JSON must be distinct files")

    commit, source_clean, dirty_paths = _git_identity(source_root)
    if args.require_clean and not source_clean:
        dirty_summary = "; ".join(dirty_paths[:8])
        raise BuildProvenanceError(
            "controlled CUDA binary requires a clean source tree; "
            f"found: {dirty_summary}"
        )

    contract = _load_json(contract_path, "validation contract")
    if contract.get("schema_version") != CONTRACT_SCHEMA:
        raise BuildProvenanceError("unexpected validation-contract schema")
    contract_header_generator = source_root / "tools" / "generate_pf_contract_header.py"
    if not contract_header_generator.is_file():
        raise BuildProvenanceError(
            f"missing validation-contract header generator: {contract_header_generator}"
        )
    # The contract hash macro alone is not sufficient: a manually modified
    # ignored header could retain that macro while changing a thermodynamic
    # expression.  Require the sole header generator's exact output before
    # allowing its bytes to be compiled and hash-recorded.
    _run(
        [
            sys.executable,
            str(contract_header_generator),
            "--contract",
            str(contract_path),
            "--header",
            str(contract_header),
            "--check",
        ],
        cwd=source_root,
    )
    canonical_hash = _canonical_contract_hash(contract)
    header_contract_hash = _extract_define(
        contract_header, "PF_KWN_VALIDATION_CONTRACT_HASH"
    )
    if header_contract_hash != canonical_hash:
        raise BuildProvenanceError(
            "generated validation-contract header hash does not match the canonical JSON contract"
        )

    fixture = _load_json(fixture_spec, "fixture specification")
    fixture_schema = fixture.get("schema")
    fixture_id = fixture.get("fixture_id")
    target = fixture.get("target")
    if not isinstance(fixture_schema, str) or not fixture_schema:
        raise BuildProvenanceError("fixture specification has no schema")
    if not isinstance(fixture_id, str) or not fixture_id:
        raise BuildProvenanceError("fixture specification has no fixture_id")
    if not isinstance(target, Mapping) or target.get("grid") != [96, 96, 96]:
        raise BuildProvenanceError("controlled CUDA validation binary requires the 96^3 fixture contract")

    nvcc_version = _tool_version(args.nvcc)
    build_time = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace(
        "+00:00", "Z"
    )
    document: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "build_utc": build_time,
        "build_flavor": "CONTROLLED_CLEAN" if args.require_clean else "UNCONTROLLED_OR_DEBUG",
        "controlled_binary_eligible": bool(args.require_clean and source_clean),
        "source": {
            "commit": commit,
            "clean": source_clean,
            "dirty_path_count": len(dirty_paths),
        },
        "contract": {
            "schema_version": CONTRACT_SCHEMA,
            "canonical_hash": canonical_hash,
            "json_sha256": _sha256_file(contract_path),
            "header_sha256": _sha256_file(contract_header),
            "header_contract_hash": header_contract_hash,
            "path": str(contract_path.relative_to(source_root)),
            "header_path": str(contract_header.relative_to(source_root)),
        },
        "toolchain": {
            "nvcc_path": args.nvcc,
            "nvcc_version": nvcc_version,
            "host_cxx": args.host_cxx,
            "host_cxx_version": _tool_version(args.host_cxx),
            "cuda_root": args.cuda_root,
            "cuda_arch": args.cuda_arch,
            "nvccflags": args.nvccflags,
            "includes": args.includes,
            "ldflags": args.ldflags,
            "ldlibs": args.ldlibs,
        },
        "precision": {
            "pf_state": args.precision,
            "checkpoint_field_payload": "IEEE754_BINARY64",
            "elastic_warm_state": "COMPLEX_FLOAT32",
        },
        "fixture": {
            "schema": fixture_schema,
            "fixture_id": fixture_id,
            "sha256": _sha256_file(fixture_spec),
            "target_grid": target["grid"],
            "spec_path": str(fixture_spec.relative_to(source_root)),
        },
        "checkpoint": {
            "schema": args.checkpoint_schema,
            "disk_magic": args.checkpoint_disk_magic,
            "auxiliary_sidecar_schema": args.auxiliary_sidecar_schema,
            "auxiliary_population_schema": args.auxiliary_population_schema,
        },
    }
    _write(output_header, _header_text(document))
    _write(output_json, json.dumps(document, indent=2, sort_keys=True) + "\n")
    print(
        json.dumps(
            {
                "controlled_binary_eligible": document["controlled_binary_eligible"],
                "header": str(output_header),
                "provenance": str(output_json),
                "source_clean": source_clean,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except BuildProvenanceError as error:
        raise SystemExit(f"[fatal] {error}") from error
