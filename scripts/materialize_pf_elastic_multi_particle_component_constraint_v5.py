#!/usr/bin/env python3
"""Materialize the immutable periodic ownership map for V5 profile relaxation.

The six resolved V2 components are separated at the 6 h handoff.  V5 assigns
every voxel to the nearest component center on the periodic grid, then uses
that fixed ownership partition to constrain each component's integral
``sum(h(phi))`` independently.  The map is a fixture artifact, never inferred
from an evolving thresholded field, so it cannot silently relabel a particle
during the preparatory relaxation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

import materialize_pf_elastic_multi_particle_6h_fixture_v1 as base


SEED_SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_COMMON_MATRIX_SEED_V3"
SCHEMA = "PF_ELASTIC_MULTI_PARTICLE_COMPONENT_CONSTRAINT_V5"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def field(seed: Dict[str, Any], root: Path, name: str, shape: tuple[int, int, int]) -> np.ndarray:
    row = seed["fields"][name]
    path = root / str(row["path"])
    if not path.is_file() or sha256(path) != row["sha256"]:
        raise ValueError(f"seed field hash mismatch: {name}")
    data = np.fromfile(path, dtype="<f8")
    if data.size != math.prod(shape) or not np.all(np.isfinite(data)):
        raise ValueError(f"invalid seed field: {name}")
    return data.reshape(shape, order="C")


def periodic_delta(coords: np.ndarray, center: float, length: int) -> np.ndarray:
    return (coords - center + 0.5 * length) % length - 0.5 * length


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"[fatal] refusing to overwrite: {args.out}")
    seed_path = args.seed / "seed_manifest.json"
    if not seed_path.is_file():
        raise SystemExit("[fatal] V3 seed manifest missing")
    seed = json.loads(seed_path.read_text(encoding="utf-8"))
    if seed.get("schema") != SEED_SCHEMA:
        raise SystemExit("[fatal] V5 ownership map requires a V3 common-matrix seed")
    grid = seed["grid"]
    shape = tuple(int(grid[key]) for key in ("Nx", "Ny", "Nz"))
    dx_nm = float(grid["dx_nm"])
    phi = field(seed, args.seed, "phi", shape)
    h = base.h_of_phi(phi)
    threshold = float(seed["component_contract"]["h_threshold"])
    threshold_labels, rows = base.components(h, threshold, dx_nm)
    expected = int(seed["component_contract"]["expected_count"])
    if len(rows) != expected or expected < 2:
        raise SystemExit("[fatal] V5 ownership map requires the frozen separated multi-particle seed")

    # components() returns deterministic label order; preserve it in the
    # manifest rather than sorting by a mutable size statistic.
    centers_grid: List[List[float]] = []
    for row in rows:
        centroid_nm = [float(value) for value in row["centroid_nm"]]
        if len(centroid_nm) != 3 or not all(math.isfinite(value) for value in centroid_nm):
            raise SystemExit("[fatal] invalid periodic component centroid")
        centers_grid.append([value / dx_nm for value in centroid_nm])

    coords = np.indices(shape, dtype=np.float64)
    best_dist2 = np.full(shape, np.inf, dtype=np.float64)
    owner = np.zeros(shape, dtype=np.int32)
    for component, center in enumerate(centers_grid):
        dist2 = np.zeros(shape, dtype=np.float64)
        for axis, length in enumerate(shape):
            delta = periodic_delta(coords[axis], center[axis], length)
            dist2 += delta * delta
        update = dist2 < best_dist2
        owner[update] = component
        best_dist2[update] = dist2[update]
    del coords, best_dist2

    # Every thresholded seed component must remain entirely inside its own
    # ownership cell.  Otherwise a Voronoi boundary crosses an interface and
    # this fixture fails closed rather than applying an ambiguous constraint.
    component_rows: List[Dict[str, Any]] = []
    for component, row in enumerate(rows):
        source_label = int(row["component_label"])
        support = threshold_labels == source_label
        if not np.any(support) or not np.all(owner[support] == component):
            raise SystemExit(
                f"[fatal] V5 periodic ownership boundary intersects seed component {component}")
        target_h_sum = float(np.sum(h[owner == component], dtype=np.float64))
        if not math.isfinite(target_h_sum) or target_h_sum <= 0.0:
            raise SystemExit(f"[fatal] V5 component {component} has invalid h-volume target")
        component_rows.append({
            "constraint_component": component,
            "source_component_label": source_label,
            "centroid_nm": row["centroid_nm"],
            "target_h_sum": target_h_sum,
            "target_h_volume_nm3": target_h_sum * dx_nm ** 3,
            "ownership_voxel_count": int(np.count_nonzero(owner == component)),
            "threshold_voxel_count": int(np.count_nonzero(support)),
        })
    target_total = float(sum(row["target_h_sum"] for row in component_rows))
    h_total = float(np.sum(h, dtype=np.float64))
    if abs(target_total - h_total) > 1.0e-10 * max(abs(h_total), 1.0):
        raise SystemExit("[fatal] V5 ownership partition does not close h-volume")

    args.out.mkdir(parents=True, exist_ok=False)
    raw_path = args.out / "component_ownership_labels.raw.i32"
    np.asarray(owner, dtype="<i4").ravel(order="C").tofile(raw_path)
    manifest: Dict[str, Any] = {
        "schema": SCHEMA,
        "validation_only": True,
        "seed_manifest_sha256": sha256(seed_path),
        "grid": grid,
        "ownership_contract": {
            "kind": "PERIODIC_NEAREST_CENTROID_VORONOI_FIXED_V5",
            "component_count": expected,
            "inter_component_volume_exchange": False,
            "threshold_boundary_intersection": False,
            "label_raw": {
                "path": raw_path.name,
                "dtype": "int32-le",
                "order": "C",
                "sha256": sha256(raw_path),
            },
        },
        "components": component_rows,
        "target_h_sum_total": target_total,
        "seed_h_sum_total": h_total,
    }
    manifest_path = args.out / "component_constraint_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS_PF_ELASTIC_MULTI_PARTICLE_COMPONENT_CONSTRAINT_V5",
        "component_count": expected,
        "constraint_manifest_sha256": sha256(manifest_path),
        "label_raw_sha256": sha256(raw_path),
        "target_h_sum_total": target_total,
    }, sort_keys=True))


if __name__ == "__main__":
    main()
