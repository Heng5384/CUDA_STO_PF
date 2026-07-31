#!/usr/bin/env python3
"""Audit whether a PF particle merge has a persistent multi-threshold neck."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

import build_pf_vtk_particle_trajectory_v1 as tracker


def local_maximum_marker(h, anchor_nm, dx_nm, radius_nm):
    shape = np.asarray(h.shape, dtype=np.int64)
    center = np.asarray(anchor_nm, dtype=np.float64) / dx_nm - 0.5
    radius_cells = int(math.ceil(radius_nm / dx_nm))
    best = None
    for i in range(-radius_cells, radius_cells + 1):
        for j in range(-radius_cells, radius_cells + 1):
            for k in range(-radius_cells, radius_cells + 1):
                delta = np.asarray((i, j, k), dtype=np.float64) * dx_nm
                if float(np.dot(delta, delta)) > radius_nm * radius_nm:
                    continue
                index = tuple(((np.rint(center).astype(np.int64) + (i, j, k)) % shape).tolist())
                candidate = (float(h[index]), index)
                if best is None or candidate[0] > best[0]:
                    best = candidate
    return best


def periodic_shortest_delta(left, right, box_nm):
    delta = np.asarray(right, dtype=np.float64) - np.asarray(left, dtype=np.float64)
    return (delta + 0.5 * box_nm) % box_nm - 0.5 * box_nm


def straight_line_profile(h, left_nm, right_nm, dx_nm, sample_count):
    box_nm = h.shape[0] * dx_nm
    delta = periodic_shortest_delta(left_nm, right_nm, box_nm)
    fraction = np.linspace(0.0, 1.0, sample_count)
    points_nm = (np.asarray(left_nm)[:, None] + delta[:, None] * fraction[None, :]) % box_nm
    coords = points_nm / dx_nm - 0.5
    values = ndimage.map_coordinates(h, coords, order=1, mode="wrap")
    index = int(np.argmin(values))
    return {
        "straight_line_min_h": float(values[index]),
        "straight_line_min_fraction": float(fraction[index]),
        "straight_line_mean_h": float(np.mean(values)),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", nargs="+", required=True, help="age_h=phi.vtk")
    parser.add_argument("--particle-rows", type=Path, required=True)
    parser.add_argument("--parent-ids", nargs=2, type=int, default=(46, 68))
    parser.add_argument("--anchor-age-h", type=float, default=15.0)
    parser.add_argument("--thresholds", nargs="+", type=float, default=[
        1.0e-4, 5.0e-4, 1.0e-3, 5.0e-3, 1.0e-2,
        5.0e-2, 1.0e-1, 2.5e-1, 5.0e-1, 7.5e-1,
    ])
    parser.add_argument("--anchor-search-nm", type=float, default=12.0)
    parser.add_argument("--grid", type=int, default=246)
    parser.add_argument("--dx-nm", type=float, default=1.0)
    parser.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.particle_rows.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    anchors = {}
    for row in rows:
        if (
            abs(float(row["age_h"]) - args.anchor_age_h) < 1.0e-9
            and int(row["canonical_particle_id"]) in args.parent_ids
        ):
            anchors[int(row["canonical_particle_id"])] = (
                float(row["centroid_x_nm"]),
                float(row["centroid_y_nm"]),
                float(row["centroid_z_nm"]),
            )
    if set(anchors) != set(args.parent_ids):
        raise SystemExit("missing parent anchors at registered anchor age")
    specs = []
    for item in args.snapshots:
        age, sep, path = item.partition("=")
        if not sep:
            raise SystemExit("snapshot must be age_h=phi.vtk")
        specs.append((float(age), Path(path)))
    specs.sort()
    threshold_rows = []
    age_rows = []
    for ordinal, (age, path) in enumerate(specs, start=1):
        print("[neck %d/%d] age_h=%g" % (ordinal, len(specs), age), flush=True)
        phi = tracker.read_vtk_ascii(path, args.grid)
        h = tracker.h_of_phi(phi)
        marker_info = [
            local_maximum_marker(h, anchors[parent_id], args.dx_nm, args.anchor_search_nm)
            for parent_id in args.parent_ids
        ]
        marker_nm = [
            tuple((np.asarray(info[1], dtype=np.float64) + 0.5) * args.dx_nm)
            for info in marker_info
        ]
        line = straight_line_profile(h, marker_nm[0], marker_nm[1], args.dx_nm, 1025)
        connected_thresholds = []
        for threshold in sorted(args.thresholds):
            labels = tracker.periodic_components(h > threshold)
            left_label = int(labels[marker_info[0][1]])
            right_label = int(labels[marker_info[1][1]])
            connected = left_label > 0 and left_label == right_label
            if connected:
                connected_thresholds.append(threshold)
            threshold_rows.append({
                "age_h": age,
                "threshold_h": threshold,
                "connected": int(connected),
                "left_marker_h": marker_info[0][0],
                "right_marker_h": marker_info[1][0],
                "left_component_label": left_label,
                "right_component_label": right_label,
                "connected_component_cell_count": int(np.count_nonzero(labels == left_label)) if connected else 0,
                **line,
                "source_phi": str(path),
            })
        age_rows.append({
            "age_h": age,
            "left_parent_id": args.parent_ids[0],
            "right_parent_id": args.parent_ids[1],
            "left_marker_x_nm": marker_nm[0][0],
            "left_marker_y_nm": marker_nm[0][1],
            "left_marker_z_nm": marker_nm[0][2],
            "right_marker_x_nm": marker_nm[1][0],
            "right_marker_y_nm": marker_nm[1][1],
            "right_marker_z_nm": marker_nm[1][2],
            "marker_periodic_distance_nm": float(np.linalg.norm(periodic_shortest_delta(
                marker_nm[0], marker_nm[1], args.grid * args.dx_nm
            ))),
            "straight_line_min_h": line["straight_line_min_h"],
            "straight_line_min_fraction": line["straight_line_min_fraction"],
            "max_registered_connected_threshold_h": max(connected_thresholds) if connected_thresholds else 0.0,
            "connected_threshold_count": len(connected_thresholds),
            "source_phi": str(path),
        })
    for name, output in (
        ("neck_threshold_matrix.csv", threshold_rows),
        ("neck_age_summary.csv", age_rows),
    ):
        with (args.out_dir / name).open("w", newline="", encoding="utf-8") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator="\n")
            writer.writeheader()
            writer.writerows(output)
    by_age_threshold = {
        (float(row["age_h"]), float(row["threshold_h"])): bool(int(row["connected"]))
        for row in threshold_rows
    }
    ages = {float(row["age_h"]) for row in age_rows}
    persistent = [
        threshold for threshold in sorted(args.thresholds)
        if 16.0 in ages and 17.0 in ages
        and by_age_threshold.get((16.0, threshold), False)
        and by_age_threshold.get((17.0, threshold), False)
    ]
    appeared_after_15 = [
        threshold for threshold in persistent
        if not by_age_threshold.get((15.0, threshold), False)
    ]
    if persistent and max(persistent) >= 5.0e-3:
        status = "PASS_MULTI_THRESHOLD_NECK_PERSISTS_16H_17H"
    elif persistent:
        status = "TREND_ONLY_DIFFUSE_TAIL_NECK"
    else:
        status = "BLOCKED_TRANSIENT_OR_ABSENT_NECK"
    summary = {
        "schema": "PF_MERGE_NECK_PERSISTENCE_V1",
        "status": status,
        "parent_ids": list(args.parent_ids),
        "anchor_age_h": args.anchor_age_h,
        "thresholds_h": sorted(args.thresholds),
        "persistent_16_17h_thresholds_h": persistent,
        "newly_connected_after_15h_thresholds_h": appeared_after_15,
        "max_persistent_threshold_h": max(persistent) if persistent else 0.0,
        "age_summary": age_rows,
    }
    (args.out_dir / "neck_persistence_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "status.txt").write_text(status + "\n", encoding="utf-8")
    report = """# 15–16 h merge-neck persistence audit

`%s`

Connectivity is evaluated on periodic six-neighbour components of `h(phi)`
at every registered threshold.  The two markers are local h maxima searched
around the 15 h parent centroids.  A connection at only `h=1e-4` is treated as
a diffuse-tail contact; persistence at `h>=0.005` at both 16 h and 17 h is the
pre-registered material-neck gate.
""" % status
    (args.out_dir / "neck_persistence_report.md").write_text(report, encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
