#!/usr/bin/env python3
"""Copy checked CUDA A--E assets and rewrite only their absolute asset paths.

The asset preparer intentionally records absolute local paths for provenance.
Those cannot be consumed after a repository/asset rsync to gpu_uvip.  This
small staging adapter preserves the original index hash, copies the asset tree
into a caller-owned run root, and rewrites references below the old asset root
to their copied paths.  It never changes raw-field bytes, sidecars, ledgers,
or the source asset tree.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path
from typing import Any


INDEX_NAME = "cuda_ae_asset_index.json"
SCHEMA = "PF_CUDA_AE_VALIDATION_ASSET_STAGING_V1"


class StageError(RuntimeError):
    pass


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _read_index(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise StageError(f"cannot read asset index: {path}") from error
    if not isinstance(value, dict) or value.get("schema_version") != "PF_CUDA_AE_VALIDATION_ASSETS_V1":
        raise StageError("unexpected CUDA A--E asset-index schema")
    return value


def _asset_root_recorded_by_index(index: dict[str, Any]) -> Path:
    try:
        case_a = index["cases"]["A"]
        case_directory = Path(case_a["case_directory"])
    except (KeyError, TypeError) as error:
        raise StageError("asset index lacks Case A directory needed for relocation") from error
    if not case_directory.is_absolute() or case_directory.name != "case_A_baseline_legacy_zero_aux":
        raise StageError("asset index Case A path cannot define its original asset root")
    return case_directory.parent


def _rewrite(value: Any, old_root: Path, new_root: Path) -> Any:
    if isinstance(value, dict):
        return {key: _rewrite(item, old_root, new_root) for key, item in value.items()}
    if isinstance(value, list):
        return [_rewrite(item, old_root, new_root) for item in value]
    if isinstance(value, str):
        candidate = Path(value)
        if candidate.is_absolute():
            try:
                return str(new_root / candidate.relative_to(old_root))
            except ValueError:
                pass
    return value


def stage(asset_root: Path, out: Path) -> dict[str, Any]:
    source_root = asset_root.resolve()
    source_index = source_root / INDEX_NAME
    if not source_index.is_file():
        raise StageError(f"asset index is missing: {source_index}")
    if out.exists():
        raise StageError(f"refusing to overwrite staged asset root: {out}")
    index = _read_index(source_index)
    recorded_root = _asset_root_recorded_by_index(index)
    original_index_sha256 = _sha256(source_index)
    shutil.copytree(source_root, out)
    copied_index = out / INDEX_NAME
    copied_before_rewrite_sha256 = _sha256(copied_index)
    if copied_before_rewrite_sha256 != original_index_sha256:
        raise StageError("copy changed the original asset-index bytes")
    rewritten = _rewrite(index, recorded_root, out.resolve())
    copied_index.write_text(json.dumps(rewritten, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    manifest = {
        "schema_version": SCHEMA,
        "status": "PASS_CUDA_AE_ASSET_RELOCATION_V1",
        "source_asset_root": str(source_root),
        "recorded_asset_root": str(recorded_root),
        "staged_asset_root": str(out.resolve()),
        "source_index_sha256": original_index_sha256,
        "copied_index_before_rewrite_sha256": copied_before_rewrite_sha256,
        "rewritten_index_sha256": _sha256(copied_index),
        "rewritten_paths_scope": "ONLY_ABSOLUTE_PATHS_DESCENDING_FROM_RECORDED_ASSET_ROOT",
    }
    (out / "asset_relocation_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    try:
        manifest = stage(args.asset_root, args.out)
    except StageError as error:
        print(f"CUDA A--E asset staging failed: {error}", file=sys.stderr)
        return 2
    print(json.dumps(manifest, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
