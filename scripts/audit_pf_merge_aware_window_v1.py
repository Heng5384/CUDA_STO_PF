#!/usr/bin/env python3
"""Audit a short PF particle merge/split window with explicit parent edges."""

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

import build_pf_vtk_particle_trajectory_v1 as tracker


def read_global_rows(path, age):
    rows = []
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if abs(float(row["age_h"]) - age) < 1.0e-9:
                rows.append(row)
    return rows


def periodic_distance(a, b, size):
    d = np.abs(np.asarray(a) - np.asarray(b))
    d = np.minimum(d, size - d)
    return float(np.sqrt(np.sum(d * d)))


def assign_global_ids(local_rows, global_rows, grid):
    pairs = []
    for li, local in enumerate(local_rows):
        lc = [float(local["centroid_x_nm"]), float(local["centroid_y_nm"]), float(local["centroid_z_nm"])]
        lr = float(local["equivalent_radius_nm"])
        for gi, glob in enumerate(global_rows):
            gc = [float(glob["centroid_x_nm"]), float(glob["centroid_y_nm"]), float(glob["centroid_z_nm"])]
            gr = float(glob["equivalent_radius_nm"])
            pairs.append((periodic_distance(lc, gc, grid) + 0.25 * abs(lr - gr), li, gi))
    mapping = {}
    used = set()
    for _score, li, gi in sorted(pairs):
        if li not in mapping and gi not in used:
            mapping[li] = int(global_rows[gi]["particle_id"])
            used.add(gi)
    return mapping


def write_csv(path, rows):
    if not rows:
        return
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader(); writer.writerows(rows)


def periodic_weighted_centroid(rows, size):
    """Return a volume-weighted centroid without breaking periodic wrapping."""
    result = []
    weights = np.asarray([float(row["h_volume_nm3"]) for row in rows], dtype=np.float64)
    for key in ("centroid_x_nm", "centroid_y_nm", "centroid_z_nm"):
        theta = 2.0 * math.pi * np.asarray([float(row[key]) for row in rows]) / size
        x = float(np.sum(weights * np.cos(theta)))
        y = float(np.sum(weights * np.sin(theta)))
        result.append((math.atan2(y, x) % (2.0 * math.pi)) * size / (2.0 * math.pi))
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--snapshots", nargs="+", required=True, help="age_h=phi.vtk entries, in increasing age")
    parser.add_argument("--trajectory-csv", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--grid", type=int, default=246)
    parser.add_argument("--dx-nm", type=float, default=1.0)
    parser.add_argument("--dt-code", type=float, default=0.02)
    parser.add_argument("--code-time-to-physical-s", type=float, default=49.54630476715921)
    parser.add_argument("--particle-h-threshold", type=float, default=1.0e-4)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    specs = []
    for item in args.snapshots:
        age, sep, raw = item.partition("=")
        if not sep:
            raise SystemExit("snapshot must be age_h=phi.vtk")
        specs.append((float(age), Path(raw)))
    specs.sort()
    previous_ids = None
    next_id = 1
    local_by_age = {}
    label_by_age = {}
    rows_by_age = {}
    unexpected = []
    for age, phi_path in specs:
        phi = tracker.read_vtk_ascii(phi_path, args.grid)
        rows, assigned, next_id, bad = tracker.identify(phi, args.dx_nm, previous_ids, next_id, args.particle_h_threshold)
        for row in rows:
            row.update({
                "age_h": age,
                "step": int(round(
                    (age - 6.0) * 3600.0
                    / (args.dt_code * args.code_time_to_physical_s)
                )),
                "source_phi": str(phi_path),
            })
        rows_by_age[age] = rows
        label_by_age[age] = assigned
        if bad:
            unexpected.append(age)
        previous_ids = assigned
    # Map the local identifiers at the first tracked age to canonical IDs from
    # the full 6--48 h trajectory table.  This makes the edge report usable
    # together with the existing 96-particle CSV.
    first_age = specs[0][0]
    canonical = read_global_rows(args.trajectory_csv, first_age)
    local_to_global = assign_global_ids(rows_by_age[first_age], canonical, args.grid * args.dx_nm)
    for age in sorted(rows_by_age):
        for row in rows_by_age[age]:
            row["canonical_particle_id"] = local_to_global.get(int(row["particle_id"]), int(row["particle_id"]))

    edge_rows = []
    for (parent_age, _), (child_age, _) in zip(specs[:-1], specs[1:]):
        parent_labels = label_by_age[parent_age]
        child_labels = label_by_age[child_age]
        parent_rows = {int(row["particle_id"]): row for row in rows_by_age[parent_age]}
        child_rows = {int(row["particle_id"]): row for row in rows_by_age[child_age]}
        flat_child = child_labels.ravel()
        for child_id, child in child_rows.items():
            indices = np.flatnonzero(flat_child == child_id)
            old = parent_labels[indices]
            old = old[old > 0]
            if old.size == 0:
                edge_rows.append({"parent_age_h": parent_age, "child_age_h": child_age, "parent_local_id": "", "parent_canonical_id": "", "child_local_id": child_id, "child_canonical_id": child["canonical_particle_id"], "overlap_voxels": 0, "parent_overlap_fraction": 0.0, "child_overlap_fraction": 0.0, "relation": "new_or_unmatched"})
                continue
            ids, counts = np.unique(old, return_counts=True)
            for parent_id, count in zip(ids, counts):
                parent_id = int(parent_id)
                parent = parent_rows[parent_id]
                pf = float(count) / max(int(parent["component_cell_count"]), 1)
                cf = float(count) / max(int(child["component_cell_count"]), 1)
                relation = "merge" if len(ids) > 1 else "continuation"
                edge_rows.append({"parent_age_h": parent_age, "child_age_h": child_age, "parent_local_id": parent_id, "parent_canonical_id": parent["canonical_particle_id"], "child_local_id": child_id, "child_canonical_id": child["canonical_particle_id"], "overlap_voxels": int(count), "parent_overlap_fraction": pf, "child_overlap_fraction": cf, "relation": relation})
    parent_to_children = {}
    for edge in edge_rows:
        if edge["parent_canonical_id"] != "":
            key = (edge["parent_age_h"], edge["parent_canonical_id"])
            parent_to_children.setdefault(key, set()).add(edge["child_canonical_id"])
    for edge in edge_rows:
        if edge["parent_canonical_id"] != "" and len(parent_to_children[(edge["parent_age_h"], edge["parent_canonical_id"])]) > 1:
            edge["relation"] = "split"
    write_csv(args.out_dir / "merge_edges.csv", edge_rows)
    trajectory_rows = []
    for age in sorted(rows_by_age):
        trajectory_rows.extend(rows_by_age[age])
    write_csv(args.out_dir / "window_particle_rows.csv", trajectory_rows)
    merge_edges = [row for row in edge_rows if row["relation"] == "merge"]
    split_edges = [row for row in edge_rows if row["relation"] == "split"]
    # Promote each detected many-to-one overlap into a persistent lineage
    # group.  Before the event its volume is the sum of the parents; at and
    # after the event it follows the merged child.  This prevents the smaller
    # parent from being silently classified as dissolution.
    merge_defs = {}
    for edge in merge_edges:
        key = (edge["parent_age_h"], edge["child_age_h"], edge["child_canonical_id"])
        merge_defs.setdefault(key, set()).add(int(edge["parent_canonical_id"]))
    group_rows = []
    group_summaries = []
    for (parent_age, child_age, child_id), parent_ids_set in sorted(merge_defs.items()):
        parent_ids = sorted(parent_ids_set)
        group_id = "MG_%gh_%s_to_%s" % (
            parent_age, "_".join(str(value) for value in parent_ids), child_id
        )
        immediate_before = None
        immediate_after = None
        for age in sorted(rows_by_age):
            if age <= parent_age:
                selected = [
                    row for row in rows_by_age[age]
                    if int(row["canonical_particle_id"]) in parent_ids
                ]
                state = "pre_merge_members"
            elif age >= child_age:
                selected = [
                    row for row in rows_by_age[age]
                    if int(row["canonical_particle_id"]) == int(child_id)
                ]
                state = "merged_child" if age == child_age else "post_merge_continuation"
            else:
                continue
            if not selected:
                continue
            volume = sum(float(row["h_volume_nm3"]) for row in selected)
            centroid = periodic_weighted_centroid(selected, args.grid * args.dx_nm)
            item = {
                "merge_group_id": group_id,
                "age_h": age,
                "step": int(round(
                    (age - 6.0) * 3600.0
                    / (args.dt_code * args.code_time_to_physical_s)
                )),
                "lineage_parent_ids": "+".join(str(value) for value in parent_ids),
                "tracked_child_id": child_id,
                "state": state,
                "component_count": len(selected),
                "h_volume_nm3": volume,
                "equivalent_group_radius_nm": (3.0 * volume / (4.0 * math.pi)) ** (1.0 / 3.0),
                "centroid_x_nm": centroid[0],
                "centroid_y_nm": centroid[1],
                "centroid_z_nm": centroid[2],
            }
            group_rows.append(item)
            if age == parent_age:
                immediate_before = volume
            if age == child_age:
                immediate_after = volume
        delta_fraction = None
        if immediate_before and immediate_after is not None:
            delta_fraction = (immediate_after - immediate_before) / immediate_before
        group_summaries.append({
            "merge_group_id": group_id,
            "parent_age_h": parent_age,
            "child_age_h": child_age,
            "parent_ids": parent_ids,
            "child_id": child_id,
            "h_volume_before_nm3": immediate_before,
            "h_volume_after_nm3": immediate_after,
            "h_volume_change_fraction": delta_fraction,
        })
    write_csv(args.out_dir / "merge_group_trajectories.csv", group_rows)
    explained_unexpected = {float(key[1]) for key in merge_defs}
    unexplained_unexpected = [
        age for age in unexpected if float(age) not in explained_unexpected
    ]
    fail_closed = bool(split_edges or unexplained_unexpected)
    summary = {"schema": "PF_MERGE_AWARE_WINDOW_AUDIT_V1", "window_age_h": [specs[0][0], specs[-1][0]], "snapshot_count": len(specs), "particle_h_threshold": args.particle_h_threshold, "unexpected_tracker_ages_h": unexpected, "unexplained_tracker_ages_h": unexplained_unexpected, "merge_edge_count": len(merge_edges), "split_edge_count": len(split_edges), "merge_group_count": len(group_summaries), "merge_edges": merge_edges, "split_edges": split_edges, "merge_groups": group_summaries, "fail_closed": fail_closed}
    (args.out_dir / "merge_aware_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    if fail_closed:
        status = "BLOCKED_MERGE_AWARE_UNEXPLAINED_EVENT"
    elif merge_edges and group_summaries:
        status = "PASS_MERGE_AWARE_GROUP_TRACKING"
    else:
        status = "PASS_MERGE_AWARE_WINDOW_NO_MERGE_SPLIT"
    (args.out_dir / "status.txt").write_text(status + "\n", encoding="utf-8")
    report = "# Merge-aware 15--16 h particle audit\n\n"
    report += "`%s`\n\n" % status
    report += "The audit uses periodic h(phi)>1e-4 components and writes explicit voxel-overlap parent/child edges. A merge is promoted to a persistent lineage group rather than silently treating a parent as dissolution; a split remains fail-closed.\n\n"
    report += "- snapshots: %s\n- merge edges: %d\n- merge groups: %d\n- split edges: %d\n- unexplained tracker ages: %s\n- fail-closed: %s\n" % (", ".join("%.0f h" % s[0] for s in specs), len(merge_edges), len(group_summaries), len(split_edges), unexplained_unexpected, fail_closed)
    if merge_edges:
        report += "\n## Merge edges\n\n|parent age|parent id|child age|child id|overlap voxels|parent fraction|child fraction|\n|---:|---:|---:|---:|---:|---:|---:|\n"
        for edge in merge_edges:
            report += "|%s|%s|%s|%s|%s|%.6g|%.6g|\n" % (edge["parent_age_h"], edge["parent_canonical_id"], edge["child_age_h"], edge["child_canonical_id"], edge["overlap_voxels"], float(edge["parent_overlap_fraction"]), float(edge["child_overlap_fraction"]))
    if group_summaries:
        report += "\n## Persistent merge groups\n\n|group|parents|child|h-volume before (nm3)|h-volume after (nm3)|change|\n|---|---|---:|---:|---:|---:|\n"
        for group in group_summaries:
            report += "|%s|%s|%s|%.9g|%.9g|%.6g|\n" % (
                group["merge_group_id"],
                "+".join(str(value) for value in group["parent_ids"]),
                group["child_id"],
                float(group["h_volume_before_nm3"]),
                float(group["h_volume_after_nm3"]),
                float(group["h_volume_change_fraction"]),
            )
    (args.out_dir / "merge_aware_report.md").write_text(report, encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
