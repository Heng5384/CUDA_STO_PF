#!/usr/bin/env python3
"""Materialize the deterministic PF-only 6 h handoff fixture.

The generated fields are deliberately small and validation-only: they represent
an effective resolved beta population, not the full APT Ag-rich object count.
The profile is the same tanh spherical profile used by ``init_phi_kernel`` and
the radius scale is solved from the registered diffuse h-volume fraction.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


def h_of_phi(phi: np.ndarray) -> np.ndarray:
    p2 = phi * phi
    return p2 * phi * (6.0 * p2 - 15.0 * phi + 10.0)


def periodic_delta(values: np.ndarray, center: float, length: float) -> np.ndarray:
    d = np.abs(values - center)
    return np.minimum(d, length - d)


def component_stats(phi: np.ndarray, dx_nm: float) -> list[dict]:
    """6-neighbor periodic CCL for the initial structural mask phi > 0.5."""
    mask = phi > 0.5
    nx, ny, nz = mask.shape
    active = np.flatnonzero(mask.ravel(order="C"))
    active_set = set(int(v) for v in active)
    parent = {int(v): int(v) for v in active}

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def flat(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k

    for raw in active:
        v = int(raw)
        i, rem = divmod(v, ny * nz)
        j, k = divmod(rem, nz)
        # Only positive directions are needed for deterministic unions.
        for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            u = flat((i + di) % nx, (j + dj) % ny, (k + dk) % nz)
            if u in active_set:
                union(v, u)

    groups: dict[int, list[int]] = {}
    for raw in active:
        v = int(raw)
        groups.setdefault(find(v), []).append(v)

    cell_volume = dx_nm ** 3
    box_volume = (nx * dx_nm) * (ny * dx_nm) * (nz * dx_nm)
    rows = []
    for comp_id, indices in enumerate(sorted(groups.values(), key=lambda q: min(q)), start=1):
        arr = np.asarray(indices, dtype=np.int64)
        ii = arr // (ny * nz)
        rem = arr % (ny * nz)
        jj = rem // nz
        kk = rem % nz
        hvals = h_of_phi(phi.ravel(order="C")[arr])
        hvol = float(np.sum(hvals) * cell_volume)
        # Circular mean followed by periodic unwrapping gives a stable centroid.
        coords = (ii, jj, kk)
        cent = []
        for vals, n in zip(coords, (nx, ny, nz)):
            angles = 2.0 * math.pi * (vals.astype(float) + 0.5) / n
            ang = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
            c = ((ang % (2.0 * math.pi)) * n / (2.0 * math.pi))
            cent.append(c)
        eq_r = (3.0 * hvol / (4.0 * math.pi)) ** (1.0 / 3.0)
        rows.append({
            "particle_id": comp_id,
            "component_cell_count": len(indices),
            "h_volume_nm3": hvol,
            "equivalent_radius_nm": eq_r,
            "centroid_x_nm": cent[0] * dx_nm,
            "centroid_y_nm": cent[1] * dx_nm,
            "centroid_z_nm": cent[2] * dx_nm,
            "beta_volume_fraction": hvol / box_volume,
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--n", type=int, default=128)
    ap.add_argument("--dx-nm", type=float, default=1.0)
    ap.add_argument("--lambda-nm", type=float, default=4.0)
    ap.add_argument("--temperature-c", type=float, default=380.0)
    ap.add_argument("--matrix-xb", type=float, default=0.006219279767278563)
    ap.add_argument("--mean-ctot", type=float, default=0.03)
    ap.add_argument("--seed-count", type=int, default=32)
    args = ap.parse_args()
    if args.n != 128 or args.seed_count != 32:
        raise SystemExit("this registered fixture is intentionally fixed at 128^3 and 32 seeds")
    if not (0.0 < args.matrix_xb < 1.0 and 0.0 < args.mean_ctot < 1.0):
        raise SystemExit("invalid composition")

    out = args.out
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"refusing to overwrite non-empty fixture root: {out}")
    out.mkdir(parents=True, exist_ok=True)
    n = args.n
    box = n * args.dx_nm
    target_h = (args.mean_ctot - args.matrix_xb) / (1.0 - args.matrix_xb)
    if not (0.0 < target_h < 1.0):
        raise SystemExit("registered mean composition implies invalid beta fraction")

    # Four-by-four-by-two periodic lattice.  The ladder is intentionally
    # validation-only and is scaled in the cubic moment by the diffuse profile.
    centers = []
    for ix in range(4):
        for iy in range(4):
            for iz in range(2):
                centers.append(((ix + 0.5) * box / 4.0,
                                (iy + 0.5) * box / 4.0,
                                (iz + 0.5) * box / 2.0))
    centers = np.asarray(centers, dtype=float)
    # Keep the three-radius contrast while leaving a positive periodic
    # separation margin on the 128^3 registered lattice.
    ladder = np.asarray([0.9, 1.0, 1.1, 1.0] * 8, dtype=float)
    base_radii = 7.0 * ladder
    w = args.lambda_nm / 2.0  # ic_phi_iface_w=2 at dx=1 nm, as in init_phi_kernel
    grid = np.arange(n, dtype=float) * args.dx_nm
    X = grid[:, None, None]
    Y = grid[None, :, None]
    Z = grid[None, None, :]

    def build(scale: float) -> np.ndarray:
        phi = np.zeros((n, n, n), dtype=np.float64)
        for (cx, cy, cz), radius in zip(centers, base_radii * scale):
            rx = periodic_delta(X, cx, box)
            ry = periodic_delta(Y, cy, box)
            rz = periodic_delta(Z, cz, box)
            r = np.sqrt(rx * rx + ry * ry + rz * rz)
            phi = np.maximum(phi, 0.5 * (1.0 + np.tanh((radius - r) * radius / w)))
        return np.clip(phi, 0.0, 1.0)

    lo, hi = 0.5, 1.5
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if float(np.mean(h_of_phi(build(mid)))) < target_h:
            lo = mid
        else:
            hi = mid
    scale = 0.5 * (lo + hi)
    phi = build(scale)
    h = h_of_phi(phi)
    mean_h = float(np.mean(h))
    xB = np.full_like(phi, args.matrix_xb)
    xBtot = (1.0 - h) * xB + h
    mass_error = float(np.mean(xBtot) - args.mean_ctot)
    if abs(mass_error) > 5.0e-14:
        raise SystemExit(f"fixture mass solve failed: {mass_error:.3e}")

    phi_path = out / "phi.raw.f64"
    xb_path = out / "xB.raw.f64"
    meta_path = out / "raw_init_meta.json"
    phi.astype("<f8", copy=False).ravel(order="C").tofile(phi_path)
    xB.astype("<f8", copy=False).ravel(order="C").tofile(xb_path)
    meta = {
        "Nx": n, "Ny": n, "Nz": n, "dx_nm": args.dx_nm,
        "interface_width_nm": args.lambda_nm, "dt_recommended": 0.02,
        "mean_xBtot": args.mean_ctot, "xB_max_safe": 0.95,
        "dtype": "float64", "order": "C",
    }
    meta_path.write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")

    particles = component_stats(phi, args.dx_nm)
    with (out / "initial_particles.csv").open("w", newline="", encoding="utf-8") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(particles[0]))
        writer.writeheader()
        writer.writerows(particles)

    def sha(path: Path) -> str:
        digest = hashlib.sha256()
        with path.open("rb") as fp:
            for block in iter(lambda: fp.read(1024 * 1024), b""):
                digest.update(block)
        return digest.hexdigest()

    manifest = {
        "schema": "PF_6H_EFFECTIVE_RESOLVED_BETA_FIXTURE_V1",
        "validation_only": True,
        "temperature_C": args.temperature_c,
        "grid": [n, n, n], "dx_nm": args.dx_nm, "lambda_sm_nm": args.lambda_nm,
        "interface_width_grids": args.lambda_nm / args.dx_nm,
        "mean_C_B_tot": args.mean_ctot,
        "matrix_xB_alpha": args.matrix_xb,
        "matrix_xAg_alpha": 2.0 * args.matrix_xb / (2.0 + args.matrix_xb),
        "target_beta_fraction": target_h,
        "actual_beta_fraction": mean_h,
        "mass_error": mass_error,
        "seed_count": len(centers), "scale_factor": scale,
        "seed_centers_nm": centers.tolist(),
        "seed_base_radii_nm": base_radii.tolist(),
        "resolved_particle_count": len(particles),
        "separation_contract": "periodic center distance > Ri+Rj+4*lambda_sm",
        "source_paths": {"phi": phi_path.name, "xB": xb_path.name, "meta": meta_path.name},
        "field_sha256": {"phi": sha(phi_path), "xB": sha(xb_path), "meta": sha(meta_path)},
    }
    (out / "fixture_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    lines = [f"{name}_sha256={value}" for name, value in manifest["field_sha256"].items()]
    (out / "fixture_hashes.md").write_text("# Fixture hashes\n\n" + "\n".join(f"`{x}`" for x in lines) + "\n", encoding="utf-8")
    print(json.dumps({
        "status": "PASS_PF_6H_EFFECTIVE_FIXTURE_MATERIALIZED",
        "grid": [n, n, n], "seed_count": len(centers),
        "resolved_particle_count": len(particles), "beta_fraction": mean_h,
        "mass_error": mass_error, "scale_factor": scale,
        "field_sha256": manifest["field_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
