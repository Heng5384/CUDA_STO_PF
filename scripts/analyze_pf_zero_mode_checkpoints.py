#!/usr/bin/env python3
"""Read PF zero-mode checkpoints and produce periodic particle trend tables."""

from __future__ import annotations

import argparse
import csv
import glob
import math
import re
import struct
from pathlib import Path

import numpy as np


HEADER_V2 = struct.Struct("<8sIIQQ4i6d3Q64s64s64s64s64s64sQ")
HEADER_V3 = struct.Struct(
    "<8sIIQQ4i6d3Q64s64s64s64s64s64s96s96s96sQ"
)
HEADER_V4 = struct.Struct(
    "<8sIIQQQ4i6d6Qd64s64s64s64s64s64s64s96s96s96sQ"
)
PREFIX = struct.Struct("<8sII")


def read_checkpoint(path: Path):
    with path.open("rb") as fp:
        prefix_raw = fp.read(PREFIX.size)
        if len(prefix_raw) != PREFIX.size:
            raise ValueError(f"short checkpoint header: {path}")
        magic, version, header_bytes = PREFIX.unpack(prefix_raw)
        if magic == b"PFZMCHK2" and version == 2:
            header = HEADER_V2
        elif magic == b"PFZMCHK3" and version == 3:
            header = HEADER_V3
        elif magic == b"PFZMCHK4" and version == 4:
            header = HEADER_V4
        else:
            raise ValueError(f"checkpoint schema mismatch: {path}")
        if header_bytes != header.size:
            raise ValueError(f"checkpoint header-size mismatch: {path}")
        fp.seek(0)
        raw = fp.read(header.size)
        if len(raw) != header.size:
            raise ValueError(f"short checkpoint header: {path}")
        fields = header.unpack(raw)
        if version == 4:
            magic, version, header_bytes, count, k_count, step = fields[:6]
            nx, ny, nz, elastic_state_present = fields[6:10]
            dt, temp, target, lam, residual, deriv = fields[10:16]
            if (
                elastic_state_present != 1
                or k_count != nx * ny * (nz // 2 + 1)
            ):
                raise ValueError(f"checkpoint V4 elastic state invalid: {path}")
        else:
            magic, version, header_bytes, count, step = fields[:5]
            nx, ny, nz, _reserved = fields[5:9]
            dt, temp, target, lam, residual, deriv = fields[9:15]
        if count != nx * ny * nz:
            raise ValueError(f"checkpoint count/grid mismatch: {path}")
        arrays = [np.fromfile(fp, dtype="<f8", count=count) for _ in range(4)]
        if any(a.size != count or not np.isfinite(a).all() for a in arrays):
            raise ValueError(f"checkpoint field invalid: {path}")
    return int(step), (nx, ny, nz), float(dt), float(temp), float(target), *arrays


def ccl_periodic(mask: np.ndarray):
    nx, ny, nz = mask.shape
    active = np.flatnonzero(mask.ravel(order="C"))
    active_set = set(int(v) for v in active)
    parent = {int(v): int(v) for v in active}

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = parent[v]
        return v

    def union(a: int, b: int):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[rb] = ra

    def flat(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k

    for raw in active:
        v = int(raw)
        i, rem = divmod(v, ny * nz)
        j, k = divmod(rem, nz)
        for di, dj, dk in ((1, 0, 0), (0, 1, 0), (0, 0, 1)):
            u = flat((i + di) % nx, (j + dj) % ny, (k + dk) % nz)
            if u in active_set:
                union(v, u)
    groups = {}
    for raw in active:
        v = int(raw)
        groups.setdefault(find(v), []).append(v)
    return [np.asarray(q, dtype=np.int64) for q in sorted(groups.values(), key=lambda q: min(q))]


def centroid_periodic(indices: np.ndarray, shape: tuple[int, int, int]):
    nx, ny, nz = shape
    coords = []
    for n, stride in ((nx, ny * nz), (ny, nz), (nz, 1)):
        vals = (indices // stride) % n
        angles = 2.0 * math.pi * (vals.astype(float) + 0.5) / n
        angle = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
        coords.append(((angle % (2.0 * math.pi)) * n / (2.0 * math.pi)))
    return np.asarray(coords)


def analyze_one(phi, xb, shape, dx_nm, prev_labels=None, prev_ids=None):
    nx, ny, nz = shape
    phi_grid = phi.reshape(shape, order="C")
    h = h_of_phi(phi)
    mask = phi_grid > 0.5
    groups = ccl_periodic(mask)
    labels = np.full(phi.size, -1, dtype=np.int32)
    if prev_ids is None:
        prev_ids = []
    rows = []
    proposed = []
    for indices in groups:
        labels[indices] = 0  # populated after identity assignment
        overlap = {}
        if prev_labels is not None:
            vals = prev_labels[indices]
            vals = vals[vals >= 0]
            if vals.size:
                uniq, counts = np.unique(vals, return_counts=True)
                overlap = {int(a): int(b) for a, b in zip(uniq, counts)}
        proposed.append((indices, overlap))

    used = set()
    next_id = (max(prev_ids) + 1) if prev_ids else 1
    for indices, overlap in proposed:
        candidates = sorted(overlap.items(), key=lambda q: (-q[1], q[0]))
        ident = None
        for cand, _count in candidates:
            if cand not in used:
                ident = cand
                break
        if ident is None:
            ident = next_id
            next_id += 1
        used.add(ident)
        labels[indices] = ident
        cent = centroid_periodic(indices, shape)
        volume = float(np.sum(h[indices]) * dx_nm ** 3)
        rows.append({
            "particle_id": int(ident),
            "component_cell_count": int(indices.size),
            "h_volume_nm3": volume,
            "equivalent_radius_nm": (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0),
            "centroid_x_nm": float(cent[0] * dx_nm),
            "centroid_y_nm": float(cent[1] * dx_nm),
            "centroid_z_nm": float(cent[2] * dx_nm),
            "overlap_parent_count": len(overlap),
        })
    rows.sort(key=lambda q: q["particle_id"])
    unexpected = False
    if prev_labels is not None:
        parent_hits = {}
        for _indices, overlap in proposed:
            for pid in overlap:
                parent_hits[pid] = parent_hits.get(pid, 0) + 1
        unexpected = any(v > 1 for v in parent_hits.values())
    return rows, labels, unexpected


def h_of_phi(phi):
    p2 = phi * phi
    return p2 * phi * (6.0 * p2 - 15.0 * phi + 10.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoints", nargs="*", default=[])
    ap.add_argument("--glob", dest="glob_pattern")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--dx-nm", type=float, default=1.0)
    ap.add_argument("--matrix-xb", type=float, default=0.006219279767278563)
    ap.add_argument("--mean-ctot", type=float, default=0.03)
    ap.add_argument(
        "--physical-dt-s",
        type=float,
        default=None,
        help="physical seconds per code step; if omitted, read ../time_contract.txt",
    )
    ap.add_argument("--start-age-h", type=float, default=6.0)
    args = ap.parse_args()
    paths = [Path(p) for p in args.checkpoints]
    if args.glob_pattern:
        paths += [Path(p) for p in glob.glob(args.glob_pattern)]
    def step_key(path: Path):
        match = re.search(r"step[_-]?(\d+)", path.stem)
        return (0, int(match.group(1))) if match else (1, path.name)
    paths = sorted(set(paths), key=step_key)
    if not paths:
        raise SystemExit("no checkpoints supplied")
    physical_dt_s = args.physical_dt_s
    if physical_dt_s is None:
        # Pilot outputs carry the authoritative converter result next to the
        # analysis directory.  Never infer physical time from checkpoint dt:
        # that field is the dimensionless/code timestep.
        contract = args.out.parent / "time_contract.txt"
        if contract.exists():
            for raw in contract.read_text(encoding="utf-8").splitlines():
                key, sep, value = raw.partition("=")
                if sep and key.strip() == "dt_physical_s":
                    physical_dt_s = float(value.strip())
                    break
    if physical_dt_s is None:
        # Preserve standalone analyzer compatibility for old checkpoints.
        # Callers running a physical-time pilot must provide a contract or
        # --physical-dt-s explicitly.
        physical_dt_s = None
    args.out.mkdir(parents=True, exist_ok=True)
    trajectories = []
    populations = []
    mass_rows = []
    prev_labels = None
    prev_ids = None
    first_step = None
    for path in paths:
        step, shape, dt, temp, target, phi_flat, _Y, xb_flat, _dY = read_checkpoint(path)
        if first_step is None:
            first_step = step
        phi = phi_flat.reshape(shape, order="C")
        xb = xb_flat.reshape(shape, order="C")
        elapsed_physical_time_s = step * (physical_dt_s if physical_dt_s is not None else dt)
        experimental_age_h = args.start_age_h + elapsed_physical_time_s / 3600.0
        h = h_of_phi(phi)
        rows, labels, unexpected = analyze_one(phi_flat, xb_flat, shape, args.dx_nm, prev_labels, prev_ids)
        for row in rows:
            row.update({"step": step, "elapsed_physical_time_s": elapsed_physical_time_s,
                        "experimental_age_h": experimental_age_h,
                        "temperature_K": temp,
                        "checkpoint": path.name, "unexpected_merge": int(unexpected)})
            trajectories.append(row)
        beta_fraction = float(np.mean(h))
        ctot = (1.0 - h) * xb + h
        interfacial_faces = 0
        mask = phi > 0.5
        for axis in range(3):
            interfacial_faces += int(np.count_nonzero(mask != np.roll(mask, -1, axis=axis)))
        # Each disagreement is one outward face in this positive-direction tally.
        box_volume = np.prod(shape) * args.dx_nm ** 3
        matrix_mask = h < 0.05
        matrix_xb = float(np.mean(xb[matrix_mask])) if np.any(matrix_mask) else float("nan")
        mass_rows.append({"step": step, "elapsed_physical_time_s": elapsed_physical_time_s,
                          "experimental_age_h": experimental_age_h,
                          "mean_C_B_tot": float(np.mean(ctot)),
                          "target_C_B_tot": target / np.prod(shape),
                          "mass_error_mean": float(np.mean(ctot) - args.mean_ctot),
                          "beta_volume_fraction": beta_fraction,
                          "matrix_xB_alpha": matrix_xb,
                          "matrix_xAg_alpha": 2.0 * matrix_xb / (2.0 + matrix_xb),
                          "interfacial_area_density_nm_inv": interfacial_faces * args.dx_nm ** 2 / box_volume,
                          "resolved_particle_count": len(rows),
                          "unexpected_merge": int(unexpected)})
        populations.append({"step": step, "elapsed_physical_time_s": elapsed_physical_time_s,
                            "experimental_age_h": experimental_age_h,
                            "resolved_particle_count": len(rows),
                            "beta_volume_fraction": beta_fraction,
                            "interfacial_area_density_nm_inv": interfacial_faces * args.dx_nm ** 2 / box_volume})
        prev_labels, prev_ids = labels, [r["particle_id"] for r in rows]

    def write(name, rows):
        if not rows:
            return
        with (args.out / name).open("w", newline="", encoding="utf-8") as fp:
            writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
            writer.writeheader(); writer.writerows(rows)

    write("particle_trajectories.csv", trajectories)
    write("population_time_series.csv", populations)
    write("mass_and_zero_mode_time_series.csv", mass_rows)
    if any(r["unexpected_merge"] for r in mass_rows):
        status = "BLOCKED_UNEXPECTED_PARTICLE_MERGE_SPLIT"
    else:
        status = "PASS_PF_PERIODIC_PARTICLE_OBSERVATION_V1"
    (args.out / "status.txt").write_text(status + "\n", encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
