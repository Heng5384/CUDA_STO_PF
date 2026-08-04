#!/usr/bin/env python3
"""Freeze declared 400-cube particle anchors for merge-aware lineage audit.

The component tracker independently verifies these declared periodic centres
against the h-threshold connected components at step zero.  Therefore this is
an identity registration file, not a substitute for a CCL measurement.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path


STATUS = "PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_INITIAL_ANCHORS_V1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fixture-manifest", required=True, type=Path)
    parser.add_argument("--initial-particles", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    fixture = json.loads(args.fixture_manifest.read_text(encoding="utf-8"))
    if fixture.get("schema") != "PF_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_FIXTURE_V1":
        raise SystemExit("wrong fixture schema")
    grid = fixture.get("grid", {})
    if [grid.get(key) for key in ("Nx", "Ny", "Nz")] != [400, 400, 400]:
        raise SystemExit("this anchor builder is exclusively for the 400^3 contract")
    rows = list(csv.DictReader(args.initial_particles.open(encoding="utf-8", newline="")))
    expected = int(fixture["component_contract"]["expected_count"])
    if len(rows) != expected or expected != 64:
        raise SystemExit("unexpected initial particle population")
    anchors = []
    seen = set()
    for row in rows:
        particle_id = str(row["particle_id"])
        centre = json.loads(row["center_grid"])
        if particle_id in seen or len(centre) != 3:
            raise SystemExit("duplicate id or invalid centre")
        if any(int(value) < 0 or int(value) >= 400 for value in centre):
            raise SystemExit("centre is outside the periodic 400 nm domain")
        seen.add(particle_id)
        anchors.append({"particle_id": particle_id, "centroid_nm": json.dumps([float(value) for value in centre])})
    anchors.sort(key=lambda row: row["particle_id"])
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["particle_id", "centroid_nm"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(anchors)
    manifest = {
        "schema": "PF_400CUBE_DECLARED_COMPONENT_ANCHORS_V1",
        "status": STATUS,
        "fixture_manifest_sha256": sha256(args.fixture_manifest),
        "initial_particles_sha256": sha256(args.initial_particles),
        "anchor_count": len(anchors),
        "domain_nm": 400.0,
        "identity_semantics": "declared placement anchors verified against independent step-zero periodic CCL",
    }
    manifest_path = args.out.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(STATUS)


if __name__ == "__main__":
    main()
