#!/usr/bin/env python3
"""Extract deterministic particle metrics from multi-particle V2 VTK outputs.

The audit identity is the connected component of ``phi >= 0.5`` (periodic
centroid unwrapping is performed around the registered seed centers).  The
reported volume is the integral of h(phi) over that component; it is not a
voxel-count radius.  This is an audit helper for the isolated profile
extension and never changes the PF runtime.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def read_vtk(path: Path) -> tuple[np.ndarray, tuple[int, int, int]]:
    with path.open("r", encoding="ascii") as f:
        header = [next(f) for _ in range(10)]
    dims = tuple(int(v) for v in header[4].split()[1:4])
    data = np.loadtxt(path, skiprows=10, dtype=np.float64)
    if data.size != int(np.prod(dims)):
        raise ValueError(f"{path}: scalar count mismatch")
    # write_vtk_cuda emits the scalar stream with the first coordinate
    # varying fastest; Fortran-order reshape restores runtime (i,j,k).
    return data.reshape(dims, order="F"), dims


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def unwrap_component(coords: np.ndarray, center: np.ndarray, n: np.ndarray) -> np.ndarray:
    d = coords.astype(np.float64) - center[None, :]
    d -= n[None, :] * np.rint(d / n[None, :])
    return center[None, :] + d


def metrics(phi: np.ndarray, seed_centers: np.ndarray) -> list[dict[str, float]]:
    # Core connectivity is intentionally conservative; all components in the
    # registered fixture are well separated from one another and boundaries.
    labels, count = ndimage.label(phi >= 0.5, structure=ndimage.generate_binary_structure(3, 1))
    hp = h(np.clip(phi, 0.0, 1.0))
    n = np.asarray(phi.shape, dtype=np.float64)
    rows = []
    for lab in range(1, count + 1):
        vox = np.argwhere(labels == lab)
        if len(vox) < 8:
            continue
        # Match to the closest registered center after periodic unwrapping.
        raw = vox.astype(np.float64)
        best = None
        for pid, c in enumerate(seed_centers):
            uw = unwrap_component(raw, c, n)
            centroid = uw.mean(axis=0)
            dist = float(np.linalg.norm(centroid - c))
            candidate = (dist, pid, centroid)
            if best is None or candidate[0] < best[0]:
                best = candidate
        assert best is not None
        dist, pid, centroid = best
        comp_h = float(np.sum(hp[labels == lab]))
        r_eq = float((3.0 * comp_h / (4.0 * np.pi)) ** (1.0 / 3.0))
        rows.append({
            "component_label": int(lab), "particle_index": int(pid),
            "voxel_count_phi_ge_0p5": int(len(vox)),
            "h_volume_voxel_units": comp_h, "equivalent_radius_nm": r_eq,
            "centroid_x_nm": float(centroid[0]), "centroid_y_nm": float(centroid[1]),
            "centroid_z_nm": float(centroid[2]), "centroid_distance_to_registered_nm": dist,
        })
    return sorted(rows, key=lambda x: x["particle_index"])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", action="append", nargs=2, metavar=("NAME", "PHI_VTK"), required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    centers = np.asarray([[20.0, 20.0, 20.0], [68.0, 20.0, 20.0], [20.0, 68.0, 20.0]])
    result = {"grid": None, "component_threshold": "phi>=0.5", "volume_definition": "sum h(phi) over component", "cases": {}}
    for name, raw_path in args.case:
        phi, dims = read_vtk(Path(raw_path))
        if result["grid"] is None:
            result["grid"] = list(dims)
        elif result["grid"] != list(dims):
            raise ValueError("grid mismatch")
        result["cases"][name] = metrics(phi, centers)
    args.out.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
