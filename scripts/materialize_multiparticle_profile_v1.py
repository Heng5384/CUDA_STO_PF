#!/usr/bin/env python3
"""Materialize a deterministic multi-particle profile-A audit fixture.

This is an equilibrium-audit-only field generator.  It uses the same spherical
tanh convention as ``init_phi_kernel`` (maximum over seeds), keeps the matrix
composition at the frozen T380 chemical root, and writes the raw-field format
accepted by the isolated fresh-profile extension.  It does not modify any
production parameter or claim an experimental particle distribution.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def h_of_phi(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def periodic_delta(x: np.ndarray, c: float, length: float) -> np.ndarray:
    d = x - c
    return d - length * np.rint(d / length)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--n", type=int, default=96)
    ap.add_argument("--dx-nm", type=float, default=1.0)
    ap.add_argument("--interface-width-nm", type=float, default=4.0)
    ap.add_argument("--xB-root", type=float, default=0.004664951821454195)
    args = ap.parse_args()

    if args.n < 64 or args.dx_nm <= 0 or args.interface_width_nm <= 0:
        raise SystemExit("invalid grid or interface width")

    # Registered validation-only three-particle ladder.  The minimum periodic
    # center distance is 48 nm, greater than R_i+R_j+4*lambda=34 nm.
    seeds = [
        {"id": "particle_small", "center_nm": [20.0, 20.0, 20.0], "radius_nm": 8.0},
        {"id": "particle_medium", "center_nm": [68.0, 20.0, 20.0], "radius_nm": 10.0},
        {"id": "particle_large", "center_nm": [20.0, 68.0, 20.0], "radius_nm": 12.0},
    ]
    L = args.n * args.dx_nm
    for s in seeds:
        if not all(0.0 <= c < L for c in s["center_nm"]):
            raise SystemExit(f"seed center outside periodic domain: {s}")
    for i, a in enumerate(seeds):
        for b in seeds[i + 1 :]:
            d = [periodic_delta(np.asarray([a["center_nm"][q]]), b["center_nm"][q], L)[0]
                 for q in range(3)]
            dist = math.sqrt(sum(v * v for v in d))
            required = a["radius_nm"] + b["radius_nm"] + 4.0 * args.interface_width_nm
            if not dist > required:
                raise SystemExit(f"periodic seed separation failed: {a['id']} {b['id']}")

    coords = np.arange(args.n, dtype=np.float64) * args.dx_nm
    xx, yy, zz = np.meshgrid(coords, coords, coords, indexing="ij")
    width = args.interface_width_nm / 2.0
    phi = np.zeros((args.n, args.n, args.n), dtype=np.float64)
    for s in seeds:
        dx = periodic_delta(xx, s["center_nm"][0], L)
        dy = periodic_delta(yy, s["center_nm"][1], L)
        dz = periodic_delta(zz, s["center_nm"][2], L)
        r = np.sqrt(dx * dx + dy * dy + dz * dz)
        # Exact init_phi_kernel convention for a spherical seed.
        candidate = 0.5 * (1.0 + np.tanh((s["radius_nm"] - r) / width))
        phi = np.maximum(phi, candidate)
    phi = np.clip(phi, 0.0, 1.0)
    xB = np.full_like(phi, args.xB_root)
    out = args.out
    out.mkdir(parents=True, exist_ok=True)
    phi_path = out / "phi_init.raw"
    xb_path = out / "xB_init.raw"
    phi.astype("<f8").tofile(phi_path)
    xB.astype("<f8").tofile(xb_path)
    meta = {
        "Nx": args.n, "Ny": args.n, "Nz": args.n,
        "dx_nm": args.dx_nm, "interface_width_nm": args.interface_width_nm,
        "dtype": "float64", "order": "C",
        "mean_phi": float(np.mean(phi)),
        "mean_h_phi": float(np.mean(h_of_phi(phi))),
        "mean_xB": float(np.mean(xB)),
        "seed_count": len(seeds), "seeds": seeds,
        "separation_rule": "periodic_center_distance_gt_Rsum_plus_4lambda",
        "classification": "VALIDATION_ONLY_MULTI_PARTICLE_PROFILE_A",
        "physical_claim": "Not fitted experimental distribution; deterministic V2 audit fixture only.",
        "phi_sha256": sha256(phi_path), "xB_sha256": sha256(xb_path),
    }
    (out / "init_meta.json").write_text(json.dumps(meta, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
