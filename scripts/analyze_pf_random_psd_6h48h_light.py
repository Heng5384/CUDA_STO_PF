#!/usr/bin/env python3
"""Compact, read-only 6 h -> 48 h audit for the random-PSD PF fixture.

It deliberately avoids the 3x3x3 tiled distance operator: the full periodic
distance field is unnecessary for the population/composition trend gates and
would consume several GiB for a 246^3 snapshot.  Component identities are
still counted with periodic face unions, while the canonical voxel-overlap
tracker remains the authoritative identity audit for 6--12 h.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy import ndimage


def read_vtk(path: Path, n: int) -> np.ndarray:
    payload = path.read_bytes()
    marker = b"LOOKUP_TABLE default"
    pos = payload.find(marker)
    if pos < 0:
        raise RuntimeError(f"VTK marker missing: {path}")
    pos = payload.find(b"\n", pos)
    arr = np.fromstring(payload[pos + 1 :], sep=" ", dtype=np.float64, count=n**3)
    if arr.size != n**3:
        raise RuntimeError(f"expected {n**3} values, got {arr.size}: {path}")
    del payload
    return arr.reshape((n, n, n))


def h_of_phi(phi: np.ndarray) -> np.ndarray:
    p = np.clip(phi, 0.0, 1.0)
    p2 = p * p
    return p2 * p * (6.0 * p2 - 15.0 * p + 10.0)


def periodic_component_count(mask: np.ndarray) -> int:
    structure = ndimage.generate_binary_structure(3, 1)
    labels, raw = ndimage.label(mask, structure=structure)
    if raw == 0:
        return 0
    parent = np.arange(raw + 1, dtype=np.int32)

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = int(parent[x])
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    for axis in range(3):
        a = np.take(labels, 0, axis=axis)
        b = np.take(labels, -1, axis=axis)
        ma = np.take(mask, 0, axis=axis)
        mb = np.take(mask, -1, axis=axis)
        for x, y in zip(a[ma & mb].ravel(), b[ma & mb].ravel()):
            union(int(x), int(y))
    roots = {find(i) for i in range(1, raw + 1)}
    return len(roots)


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def snapshot_rows(preflight: Path, continuation: Path) -> list[tuple[float, Path, Path]]:
    rows: list[tuple[float, Path, Path]] = []
    pre = preflight / "continuous_dt0p02" / "results" / "ch_T380_cuda_246x246x246_dt0.02_steps21800_xB0.006" / "continuous_dt0p02"
    rows.append((6.0, pre / "phi_init.vtk", pre / "xB_0.vtk"))
    for step, age in zip((3633, 7266, 10899, 14532, 18165, 21798), (7, 8, 9, 10, 11, 12)):
        rows.append((float(age), pre / f"phi_{step}.vtk", pre / f"xB_{step}.vtk"))
    con = continuation / "Results" / "ch_T380_cuda_246x246x246_dt0.02_steps152600_xB0.030"
    con_steps = (25431, 29064, 32697, 36330, 39963, 43596, 47229, 50862,
                 54495, 58128, 61761, 65394, 69027, 72660, 76293, 79926,
                 83559, 87192, 90825, 94458, 98091, 101724, 105357, 108990,
                 112623, 116256, 119889, 123522, 127155, 130788, 134421,
                 138054, 141687, 145320, 148953, 152586)
    for i, step in enumerate(con_steps, start=13):
        rows.append((float(i), con / f"phi_{step}.vtk", con / f"xB_{step}.vtk"))
    # Do not mix a separately materialized restart-final snapshot into the
    # continuous trend series.  Restart equality is audited from checkpoint
    # hashes; the population trend must use the continuous 48 h snapshot.
    return rows


def analyze_one(age: float, phi_path: Path, xb_path: Path, n: int) -> dict[str, float | int | str]:
    phi = read_vtk(phi_path, n)
    xb = read_vtk(xb_path, n)
    h = h_of_phi(phi)
    alpha = np.maximum(1.0 - h, 0.0)
    mass = (alpha * xb + h).mean()
    beta_fraction = float(h.mean())
    matrix_mask = h < 0.005
    matrix_xb = float(np.mean(xb[matrix_mask])) if np.any(matrix_mask) else float("nan")
    matrix_xag = float(2.0 * matrix_xb / (2.0 + matrix_xb))
    alpha_weighted_xb = float(np.sum(alpha * xb) / np.sum(alpha))
    alpha_weighted_xag = float(2.0 * alpha_weighted_xb / (2.0 + alpha_weighted_xb))
    comp_count = periodic_component_count(h > 1.0e-4)
    # A bounded finite-difference interface-area proxy, normalized by volume.
    gx = 0.5 * (np.roll(phi, -1, 0) - np.roll(phi, 1, 0))
    gy = 0.5 * (np.roll(phi, -1, 1) - np.roll(phi, 1, 1))
    gz = 0.5 * (np.roll(phi, -1, 2) - np.roll(phi, 1, 2))
    area_density = float(np.mean(np.sqrt(gx * gx + gy * gy + gz * gz)))
    del gx, gy, gz
    labels, raw = ndimage.label(h > 1.0e-4, structure=ndimage.generate_binary_structure(3, 1))
    hvol = np.bincount(labels.ravel(), weights=h.ravel(), minlength=raw + 1)[1:]
    hvol = hvol[hvol > 1.0]
    radii = np.cbrt(3.0 * hvol / (4.0 * np.pi)) if hvol.size else np.empty(0)
    out: dict[str, float | int | str] = {
        "age_h": age,
        "phi_sha256": sha256(phi_path),
        "xB_sha256": sha256(xb_path),
        "mean_C_B_tot": float(mass),
        "beta_volume_fraction": beta_fraction,
        "matrix_xB_h_lt_0p005": matrix_xb,
        "matrix_xAg_h_lt_0p005": matrix_xag,
        "matrix_xB_alpha_weighted": alpha_weighted_xb,
        "matrix_xAg_alpha_weighted": alpha_weighted_xag,
        "resolved_component_count_periodic": comp_count,
        "resolved_component_count_raw": int(raw),
        "mean_radius_nm": float(np.mean(radii)) if radii.size else float("nan"),
        "R3_moment_mean_nm3": float(np.mean(radii**3)) if radii.size else float("nan"),
        "interface_area_proxy_nm_inv": area_density,
    }
    del phi, xb, h, alpha, labels, hvol, radii
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--preflight-root", type=Path, required=True)
    ap.add_argument("--continuation-root", type=Path, required=True)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n", type=int, default=246)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for age, phi, xb in snapshot_rows(args.preflight_root, args.continuation_root):
        if not phi.is_file() or not xb.is_file():
            raise SystemExit(f"missing snapshot at {age:g} h: {phi} {xb}")
        print(f"[snapshot] age_h={age:g} phi={phi.name}", flush=True)
        rows.append(analyze_one(age, phi, xb, args.n))
    fields = list(rows[0])
    with (args.out_dir / "population_matrix_time_series.csv").open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields); w.writeheader(); w.writerows(rows)
    summary = {
        "status": "PASS_RANDOM_PSD_6H48H_LIGHT_OBSERVATION_AUDIT",
        "rows": len(rows),
        "age_h": [r["age_h"] for r in rows],
        "initial_particle_count": rows[0]["resolved_component_count_periodic"],
        "particle_count_48h": rows[-1]["resolved_component_count_periodic"],
        "mean_radius_6h_nm": rows[0]["mean_radius_nm"],
        "mean_radius_48h_nm": rows[-1]["mean_radius_nm"],
        "matrix_xAg_min": min(float(r["matrix_xAg_h_lt_0p005"]) for r in rows),
        "matrix_xAg_max": max(float(r["matrix_xAg_h_lt_0p005"]) for r in rows),
        "max_abs_mass_drift": max(abs(float(r["mean_C_B_tot"]) - 0.03) for r in rows),
        "structural_change_12h_to_48h": {
            "particle_count_pct": 100.0 * (rows[-1]["resolved_component_count_periodic"] / rows[6]["resolved_component_count_periodic"] - 1.0),
            "mean_radius_pct": 100.0 * (rows[-1]["mean_radius_nm"] / rows[6]["mean_radius_nm"] - 1.0),
            "R3_moment_pct": 100.0 * (rows[-1]["R3_moment_mean_nm3"] / rows[6]["R3_moment_mean_nm3"] - 1.0),
            "interface_area_pct": 100.0 * (rows[-1]["interface_area_proxy_nm_inv"] / rows[6]["interface_area_proxy_nm_inv"] - 1.0),
        },
    }
    (args.out_dir / "population_matrix_time_series_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (args.out_dir / "status.txt").write_text(summary["status"] + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
