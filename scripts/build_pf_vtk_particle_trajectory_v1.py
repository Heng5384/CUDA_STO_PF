#!/usr/bin/env python3
"""Build a periodic particle and matrix time series directly from ASCII VTK.

This is an observation-only adapter for completed PF runs.  It intentionally
does not alter solver state and keeps the registered 6--48 h snapshot ladder
small enough to process on the cluster without copying multi-gigabyte fields to
the workstation.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np
from scipy import ndimage


def read_vtk_ascii(path: Path, n: int) -> np.ndarray:
    payload = path.read_bytes()
    marker = b"LOOKUP_TABLE default"
    pos = payload.find(marker)
    if pos < 0:
        raise RuntimeError(f"VTK marker missing: {path}")
    pos = payload.find(b"\n", pos)
    values = np.fromstring(payload[pos + 1 :], sep=" ", dtype=np.float64, count=n**3)
    if values.size != n**3:
        raise RuntimeError(f"{path}: expected {n**3}, got {values.size}")
    return values.reshape((n, n, n))


def h_of_phi(phi: np.ndarray) -> np.ndarray:
    p = np.clip(phi, 0.0, 1.0)
    return p * p * p * (6.0 * p * p - 15.0 * p + 10.0)


def periodic_components(mask: np.ndarray) -> np.ndarray:
    """Six-neighbour CCL with the three periodic face identifications."""
    structure = np.zeros((3, 3, 3), dtype=np.int8)
    structure[1, 1, 1] = 1
    structure[0, 1, 1] = structure[2, 1, 1] = 1
    structure[1, 0, 1] = structure[1, 2, 1] = 1
    structure[1, 1, 0] = structure[1, 1, 2] = 1
    raw, count = ndimage.label(mask, structure=structure)
    if count == 0:
        return raw.astype(np.int32)
    parent = np.arange(count + 1, dtype=np.int32)

    def find(v: int) -> int:
        while parent[v] != v:
            parent[v] = parent[parent[v]]
            v = int(parent[v])
        return v

    def union(a: np.ndarray, b: np.ndarray) -> None:
        pairs = np.stack((a.ravel(), b.ravel()), axis=1)
        pairs = pairs[(pairs[:, 0] != 0) & (pairs[:, 1] != 0)]
        if pairs.size == 0:
            return
        for left, right in np.unique(pairs, axis=0):
            ra, rb = find(int(left)), find(int(right))
            if ra != rb:
                parent[rb] = ra

    union(raw[0, :, :], raw[-1, :, :])
    union(raw[:, 0, :], raw[:, -1, :])
    union(raw[:, :, 0], raw[:, :, -1])
    roots = np.arange(count + 1, dtype=np.int32)
    for idx in range(1, count + 1):
        roots[idx] = find(idx)
    unique, inverse = np.unique(roots, return_inverse=True)
    # inverse[0] is the inactive label; preserve it as zero.
    dense = inverse[raw]
    dense[dense == inverse[0]] = 0
    if inverse[0] != 0:
        dense[dense > inverse[0]] -= 1
    return dense.astype(np.int32)


def centroid_periodic(indices, shape):
    coords = np.unravel_index(indices, shape)
    out = []
    for values, size in zip(coords, shape):
        angles = 2.0 * math.pi * (values.astype(np.float64) + 0.5) / size
        angle = math.atan2(float(np.mean(np.sin(angles))), float(np.mean(np.cos(angles))))
        out.append((angle % (2.0 * math.pi)) * size / (2.0 * math.pi))
    return tuple(out)


def identify(phi, dx_nm, previous_ids, next_id, particle_h_threshold):
    h = h_of_phi(phi)
    # The registered resolved-particle contract is the small positive h tail;
    # this keeps the h-volume closure with mean(h(phi)) while avoiding a full
    # grid transfer to the workstation.
    labels = periodic_components(h > particle_h_threshold)
    rows = []
    used = set()
    assigned = np.zeros(labels.size, dtype=np.int32)
    for component in range(1, int(labels.max()) + 1):
        indices = np.flatnonzero(labels.ravel() == component)
        if indices.size == 0:
            continue
        overlap = {}
        if previous_ids is not None:
            old = previous_ids[indices]
            old = old[old > 0]
            if old.size:
                ids, counts = np.unique(old, return_counts=True)
                overlap = {int(pid): int(count) for pid, count in zip(ids, counts)}
        candidates = sorted(overlap, key=lambda pid: (-overlap[pid], pid))
        pid = next((candidate for candidate in candidates if candidate not in used), None)
        if pid is None:
            pid = next_id
            next_id += 1
        used.add(pid)
        assigned[indices] = pid
        volume = float(np.sum(h.ravel()[indices]) * dx_nm**3)
        cent = centroid_periodic(indices, phi.shape)
        rows.append({
            "particle_id": pid,
            "component_cell_count": int(indices.size),
            "h_volume_nm3": volume,
            "equivalent_radius_nm": (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0),
            "centroid_x_nm": cent[0] * dx_nm,
            "centroid_y_nm": cent[1] * dx_nm,
            "centroid_z_nm": cent[2] * dx_nm,
            "overlap_parent_count": len(overlap),
        })
    parent_hits: dict[int, int] = {}
    for pid in used:
        if previous_ids is not None:
            # A parent appearing in multiple assigned components indicates a split.
            pass
    if previous_ids is None:
        unexpected = False
    else:
        parent_components: dict[int, set[int]] = {}
        flat_labels = labels.ravel()
        for component in range(1, int(labels.max()) + 1):
            indices = np.flatnonzero(flat_labels == component)
            old = previous_ids[indices]
            for parent_id in np.unique(old[old > 0]):
                parent_components.setdefault(int(parent_id), set()).add(component)
        unexpected = any(len(components) > 1 for components in parent_components.values())
        # A many-parent overlap is also fail-closed as an unexpected merge.
        unexpected = unexpected or any(int(row["overlap_parent_count"]) > 1 for row in rows)
    return rows, assigned, next_id, unexpected


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--stage1-root", type=Path, required=True)
    parser.add_argument("--stage2-root", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=246)
    parser.add_argument("--dx-nm", type=float, default=1.0)
    parser.add_argument("--matrix-xb-total", type=float, default=0.03)
    parser.add_argument("--particle-h-threshold", type=float, default=1.0e-4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    snapshots: list[tuple[float, int, Path, Path]] = []
    for age, step in [(6.0, 0), (7.0, 3633), (8.0, 7266), (9.0, 10899), (10.0, 14532), (11.0, 18165), (12.0, 21798)]:
        phi_name = "phi_init.vtk" if step == 0 else f"phi_{step}.vtk"
        xb_name = "xB_0.vtk" if step == 0 else f"xB_{step}.vtk"
        snapshots.append((age, step, args.stage1_root / phi_name, args.stage1_root / xb_name))
    for age, step in zip(range(13, 48), range(25431, 148954, 3633)):
        snapshots.append((float(age), step, args.stage2_root / f"phi_{step}.vtk", args.stage2_root / f"xB_{step}.vtk"))
    snapshots.append((48.0, 152600, args.stage2_root / "phi_43.vtk", args.stage2_root / "xB_43.vtk"))
    if len(snapshots) != 43:
        raise RuntimeError(f"registered snapshot count is {len(snapshots)}, expected 43")

    particle_rows: list[dict[str, object]] = []
    matrix_rows: list[dict[str, object]] = []
    previous_ids = None
    next_id = 1
    unexpected_steps: list[int] = []
    for ordinal, (age, step, phi_path, xb_path) in enumerate(snapshots, start=1):
        print(f"[vtk {ordinal}/{len(snapshots)}] age_h={age:g} step={step}", flush=True)
        if not phi_path.is_file() or not xb_path.is_file():
            raise RuntimeError(f"missing pair: {phi_path} {xb_path}")
        phi = read_vtk_ascii(phi_path, args.grid)
        xb = read_vtk_ascii(xb_path, args.grid)
        h = h_of_phi(phi)
        rows, current_ids, next_id, unexpected = identify(phi, args.dx_nm, previous_ids, next_id, args.particle_h_threshold)
        if unexpected:
            unexpected_steps.append(step)
        for row in rows:
            row.update({"step": step, "age_h": age, "registered_age_h": age,
                        "elapsed_physical_time_s": (age - 6.0) * 3600.0,
                        "temperature_K": 653.15, "source_phi": str(phi_path),
                        "source_xB": str(xb_path), "unexpected_merge": int(unexpected)})
            particle_rows.append(row)
        matrix_mask = h < 0.005
        matrix_xb = float(np.mean(xb[matrix_mask])) if np.any(matrix_mask) else float("nan")
        ctot = (1.0 - h) * xb + h
        matrix_rows.append({
            "step": step, "age_h": age, "registered_age_h": age,
            "elapsed_physical_time_s": (age - 6.0) * 3600.0,
            "source_phi": str(phi_path), "source_xB": str(xb_path),
            "beta_volume_fraction": float(np.mean(h)),
            "beta_volume_fraction_from_particles": float(sum(float(r["h_volume_nm3"]) for r in rows) / (args.grid * args.dx_nm) ** 3),
            "matrix_xB_h_lt_0p005": matrix_xb,
            "matrix_xAg_h_lt_0p005": 2.0 * matrix_xb / (2.0 + matrix_xb),
            "mean_C_B_tot": float(np.mean(ctot)),
            "target_C_B_tot": args.matrix_xb_total,
            "resolved_particle_count": len(rows),
            "unexpected_merge": int(unexpected),
        })
        previous_ids = current_ids
    write_csv(args.out_dir / "particle_trajectories.csv", particle_rows)
    write_csv(args.out_dir / "matrix_composition_time_series.csv", matrix_rows)
    status = "PASS_PF_VTK_PERIODIC_PARTICLE_TRACKING_V1" if not unexpected_steps else "BLOCKED_UNEXPECTED_PARTICLE_MERGE_SPLIT"
    (args.out_dir / "status.txt").write_text(status + "\n", encoding="utf-8")
    (args.out_dir / "snapshot_manifest.csv").write_text(
        "age_h,step,phi,xB\n" + "".join(f"{age},{step},{phi},{xb}\n" for age, step, phi, xb in snapshots),
        encoding="utf-8",
    )
    print(status)


if __name__ == "__main__":
    main()
