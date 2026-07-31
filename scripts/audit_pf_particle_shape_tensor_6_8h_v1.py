#!/usr/bin/env python3
"""Extract periodic h(phi)-weighted particle shape tensors from PF VTK fields."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import build_pf_vtk_particle_trajectory_v1 as tracker


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def canonical_axis(vector):
    vector = np.asarray(vector, dtype=np.float64)
    pivot = int(np.argmax(np.abs(vector)))
    if vector[pivot] < 0.0:
        vector = -vector
    return vector


def weighted_periodic_centroid(coords, weights, shape):
    center = []
    for axis, size in enumerate(shape):
        angle = 2.0 * math.pi * (coords[axis].astype(np.float64) + 0.5) / size
        x = float(np.sum(weights * np.cos(angle)))
        y = float(np.sum(weights * np.sin(angle)))
        center.append((math.atan2(y, x) % (2.0 * math.pi)) * size / (2.0 * math.pi))
    return np.asarray(center, dtype=np.float64)


def particle_tensor(h, assigned, particle_id, dx_nm):
    flat_ids = assigned.ravel()
    indices = np.flatnonzero(flat_ids == particle_id)
    coords = np.asarray(np.unravel_index(indices, h.shape), dtype=np.float64)
    weights = h.ravel()[indices].astype(np.float64)
    weight_sum = float(np.sum(weights))
    center_cell = weighted_periodic_centroid(coords, weights, h.shape)
    displacement = coords + 0.5 - center_cell[:, None]
    for axis, size in enumerate(h.shape):
        displacement[axis] = (displacement[axis] + 0.5 * size) % size - 0.5 * size
    displacement *= dx_nm
    weighted_second = (displacement * weights[None, :]) @ displacement.T
    covariance = weighted_second / weight_sum
    volume_nm3 = weight_sum * dx_nm**3
    inertia = (np.trace(weighted_second) * np.eye(3) - weighted_second) * dx_nm**3
    inertia_per_volume = inertia / volume_nm3
    values, vectors = np.linalg.eigh(covariance)
    order = np.argsort(values)[::-1]
    values = np.maximum(values[order], 0.0)
    vectors = vectors[:, order]
    axes = [canonical_axis(vectors[:, index]) for index in range(3)]
    semi = np.sqrt(5.0 * values)
    major, intermediate, minor = (float(value) for value in semi)
    major_axis = axes[0]
    direction_angles = np.degrees(np.arccos(np.clip(np.abs(major_axis), 0.0, 1.0)))
    closest = int(np.argmin(direction_angles))
    closest_name = ("[100]", "[010]", "[001]")[closest]
    denom = major * major - minor * minor
    triaxiality = (
        (major * major - intermediate * intermediate) / denom
        if denom > 1.0e-24 else float("nan")
    )
    eigen_gap = (
        (values[0] - values[1]) / values[0]
        if values[0] > 1.0e-24 else 0.0
    )
    return {
        "h_volume_nm3_tensor": volume_nm3,
        "weighted_centroid_x_nm": center_cell[0] * dx_nm,
        "weighted_centroid_y_nm": center_cell[1] * dx_nm,
        "weighted_centroid_z_nm": center_cell[2] * dx_nm,
        "cov_xx_nm2": covariance[0, 0],
        "cov_xy_nm2": covariance[0, 1],
        "cov_xz_nm2": covariance[0, 2],
        "cov_yy_nm2": covariance[1, 1],
        "cov_yz_nm2": covariance[1, 2],
        "cov_zz_nm2": covariance[2, 2],
        "inertia_xx_nm5": inertia[0, 0],
        "inertia_xy_nm5": inertia[0, 1],
        "inertia_xz_nm5": inertia[0, 2],
        "inertia_yy_nm5": inertia[1, 1],
        "inertia_yz_nm5": inertia[1, 2],
        "inertia_zz_nm5": inertia[2, 2],
        "inertia_per_volume_xx_nm2": inertia_per_volume[0, 0],
        "inertia_per_volume_xy_nm2": inertia_per_volume[0, 1],
        "inertia_per_volume_xz_nm2": inertia_per_volume[0, 2],
        "inertia_per_volume_yy_nm2": inertia_per_volume[1, 1],
        "inertia_per_volume_yz_nm2": inertia_per_volume[1, 2],
        "inertia_per_volume_zz_nm2": inertia_per_volume[2, 2],
        "shape_eigenvalue_major_nm2": values[0],
        "shape_eigenvalue_intermediate_nm2": values[1],
        "shape_eigenvalue_minor_nm2": values[2],
        "semi_axis_major_nm": major,
        "semi_axis_intermediate_nm": intermediate,
        "semi_axis_minor_nm": minor,
        "aspect_ratio_major_minor": major / minor if minor > 1.0e-12 else float("nan"),
        "aspect_ratio_major_intermediate": major / intermediate if intermediate > 1.0e-12 else float("nan"),
        "aspect_ratio_intermediate_minor": intermediate / minor if minor > 1.0e-12 else float("nan"),
        "triaxiality": triaxiality,
        "major_axis_x": major_axis[0],
        "major_axis_y": major_axis[1],
        "major_axis_z": major_axis[2],
        "intermediate_axis_x": axes[1][0],
        "intermediate_axis_y": axes[1][1],
        "intermediate_axis_z": axes[1][2],
        "minor_axis_x": axes[2][0],
        "minor_axis_y": axes[2][1],
        "minor_axis_z": axes[2][2],
        "major_axis_angle_to_x_deg": direction_angles[0],
        "major_axis_angle_to_y_deg": direction_angles[1],
        "major_axis_angle_to_z_deg": direction_angles[2],
        "closest_box_crystal_axis": closest_name,
        "misorientation_to_closest_box_axis_deg": direction_angles[closest],
        "major_axis_eigen_gap": eigen_gap,
        "major_axis_orientation_well_conditioned": int(eigen_gap >= 0.05),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshot-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=246)
    parser.add_argument("--dx-nm", type=float, default=1.0)
    parser.add_argument("--particle-h-threshold", type=float, default=1.0e-4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.snapshot_csv.open(newline="", encoding="utf-8") as stream:
        snapshot_rows = list(csv.DictReader(stream))
    previous = None
    next_id = 1
    output = []
    unexpected_ages = []
    per_frame = []
    for ordinal, snapshot in enumerate(snapshot_rows, start=1):
        age = float(snapshot["age_h"])
        phi_path = Path(snapshot["source_phi"])
        print("[shape %d/%d] age_h=%.9g" % (ordinal, len(snapshot_rows), age), flush=True)
        phi = tracker.read_vtk_ascii(phi_path, args.grid)
        h = tracker.h_of_phi(phi)
        particle_rows, assigned, next_id, unexpected = tracker.identify(
            phi, args.dx_nm, previous, next_id, args.particle_h_threshold
        )
        if unexpected:
            unexpected_ages.append(age)
        ill_conditioned = 0
        for particle in particle_rows:
            tensor = particle_tensor(h, assigned, int(particle["particle_id"]), args.dx_nm)
            if not tensor["major_axis_orientation_well_conditioned"]:
                ill_conditioned += 1
            output.append({
                "age_h": age,
                "step": int(round(float(snapshot["step_estimate"]))),
                "particle_id": int(particle["particle_id"]),
                "component_cell_count": int(particle["component_cell_count"]),
                "equivalent_radius_nm": float(particle["equivalent_radius_nm"]),
                **tensor,
                "source_phi": str(phi_path),
            })
        per_frame.append({
            "age_h": age,
            "particle_count": len(particle_rows),
            "orientation_well_conditioned_count": len(particle_rows) - ill_conditioned,
            "orientation_ill_conditioned_count": ill_conditioned,
            "unexpected_merge_split": int(unexpected),
        })
        previous = assigned
    write_csv(args.out_dir / "particle_shape_tensor_trajectories.csv", output)
    write_csv(args.out_dir / "shape_tensor_frame_summary.csv", per_frame)
    finite_aspect = [
        float(row["aspect_ratio_major_minor"]) for row in output
        if math.isfinite(float(row["aspect_ratio_major_minor"]))
    ]
    summary = {
        "schema": "PF_PARTICLE_SHAPE_TENSOR_6_8H_V1",
        "snapshot_count": len(snapshot_rows),
        "particle_row_count": len(output),
        "particle_h_threshold": args.particle_h_threshold,
        "unexpected_ages_h": unexpected_ages,
        "major_axis_orientation_condition": "(lambda_major-lambda_intermediate)/lambda_major >= 0.05",
        "aspect_ratio_major_minor_min": min(finite_aspect),
        "aspect_ratio_major_minor_max": max(finite_aspect),
        "aspect_ratio_major_minor_mean": float(np.mean(finite_aspect)),
        "orientation_well_conditioned_fraction": float(np.mean([
            int(row["major_axis_orientation_well_conditioned"]) for row in output
        ])),
    }
    status = (
        "PASS_PARTICLE_SHAPE_TENSOR_6_8H_V1"
        if not unexpected_ages else "BLOCKED_SHAPE_TENSOR_MERGE_SPLIT"
    )
    summary["status"] = status
    (args.out_dir / "shape_tensor_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "status.txt").write_text(status + "\n", encoding="utf-8")
    report = """# 6–8 h particle shape-tensor audit

`%s`

Each particle uses periodic unwrapping and the diffuse `h(phi)` field as its
volume weight.  The shape covariance has units nm2.  The physical geometric
inertia tensor is `integral h(r) [r^2 I - r r] dV` and has units nm5.  Effective
ellipsoid semi-axes use `a_i=sqrt(5 lambda_i)`.

The major axis is an unoriented line: its sign is canonicalized only for stable
serialization.  Angles use absolute direction cosines relative to the box
axes, interpreted as [100], [010], [001].  Orientations with a major/intermediate
eigenvalue gap below 5%% are marked ill-conditioned and should not be used as
crystallographic evidence.
""" % status
    (args.out_dir / "shape_tensor_report.md").write_text(report, encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
