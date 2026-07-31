#!/usr/bin/env python3
"""Resolve 246-cube particle merges without hiding splits or new components."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Mapping, Set, Tuple


PASS = "PASS_246CUBE_RESOLVED_MERGE_AWARE_LINEAGE_V1"
FAIL = "BLOCKED_246CUBE_UNRESOLVED_MERGE_SPLIT_V1"
TRACKER_PASS = "PASS_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1"
IDENTITY_TOLERANCE_NM = 1.0e-3
THRESHOLDS = (
    ("low", "h1e-4", 1.0e-4),
    ("medium", "h1e-3", 1.0e-3),
    ("strong", "h5e-3", 5.0e-3),
)


def kv(path: Path) -> Dict[str, str]:
    result: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        key, separator, value = raw.partition("=")
        if separator:
            result[key] = value
    return result


def rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def periodic_distance(left: List[float], right: List[float]) -> float:
    squared = 0.0
    for a, b in zip(left, right):
        delta = abs(a - b)
        delta = min(delta, 246.0 - delta)
        squared += delta * delta
    return math.sqrt(squared)


def threshold_identity_map(
    lineage: List[Dict[str, str]],
    registered: List[Dict[str, str]],
) -> Tuple[Dict[int, str], float]:
    initial = [row for row in lineage if int(row["step"]) == 0]
    if len(initial) != 96 or len(registered) != 96:
        raise ValueError("initial identity population is not 96")
    candidates: List[Tuple[float, int, str]] = []
    for row in initial:
        stable = int(row["stable_particle_id"])
        observed = [
            float(row["centroid_x_nm"]),
            float(row["centroid_y_nm"]),
            float(row["centroid_z_nm"]),
        ]
        for target in registered:
            expected = json.loads(target["centroid_nm"])
            candidates.append(
                (
                    periodic_distance(observed, expected),
                    stable,
                    target["particle_id"],
                )
            )
    mapping: Dict[int, str] = {}
    used: Set[str] = set()
    maximum = 0.0
    for distance, stable, particle in sorted(candidates):
        if stable in mapping or particle in used:
            continue
        mapping[stable] = particle
        used.add(particle)
        maximum = max(maximum, distance)
    if (
        len(mapping) != 96
        or len(used) != 96
        or maximum > IDENTITY_TOLERANCE_NM
    ):
        raise ValueError(
            f"threshold identity mapping failed, max delta={maximum}"
        )
    return mapping, maximum


def merge_groups(
    lineage: List[Dict[str, str]],
    events: List[Dict[str, str]],
    mapping: Mapping[int, str],
    threshold_name: str,
    threshold: float,
) -> List[Dict[str, Any]]:
    groups: List[Dict[str, Any]] = []
    by_step_primary = {
        (int(row["step"]), int(row["stable_particle_id"])): row
        for row in lineage
    }
    for event in events:
        if event["event_type"] != "merge":
            continue
        step = int(event["step"])
        child = int(event["child_stable_id"])
        child_row = by_step_primary.get((step, child))
        if child_row is None:
            raise ValueError("merge child has no lineage row")
        member_ids = [
            int(value)
            for value in child_row["lineage_member_ids"].split("+")
            if value
        ]
        try:
            particles = sorted(mapping[value] for value in member_ids)
        except KeyError as exc:
            raise ValueError("merge member is not an initial identity") from exc
        groups.append(
            {
                "threshold_name": threshold_name,
                "h_threshold": threshold,
                "step": step,
                "experimental_age_h": float(child_row["experimental_age_h"]),
                "merge_group_id": "MG_" + "_".join(particles),
                "member_particle_ids": "+".join(particles),
                "parent_stable_ids": event["parent_stable_ids"],
                "child_stable_id": child,
                "overlap_detail": event["detail"],
                "qualified_merge": int(event["qualified_merge"]),
            }
        )
    return groups


def write_csv(path: Path, data: List[Mapping[str, Any]]) -> None:
    fields = [
        "threshold_name",
        "h_threshold",
        "step",
        "experimental_age_h",
        "merge_group_id",
        "member_particle_ids",
        "parent_stable_ids",
        "child_stable_id",
        "overlap_detail",
        "qualified_merge",
        "neck_classification",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--low", type=Path, required=True)
    parser.add_argument("--medium", type=Path, required=True)
    parser.add_argument("--strong", type=Path, required=True)
    parser.add_argument("--initial-components", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    try:
        roots = {
            "low": args.low,
            "medium": args.medium,
            "strong": args.strong,
        }
        registered = rows(args.initial_components)
        summaries: Dict[str, Dict[str, str]] = {}
        identity_deltas: Dict[str, float] = {}
        all_groups: List[Dict[str, Any]] = []
        gates: Dict[str, bool] = {}
        for name, _directory, threshold in THRESHOLDS:
            root = roots[name]
            summary = kv(root / "lineage_summary.txt")
            summaries[name] = summary
            lineage = rows(root / "particle_lineage.csv")
            events = rows(root / "particle_events.csv")
            mapping, identity_delta = threshold_identity_map(
                lineage, registered
            )
            identity_deltas[name] = identity_delta
            groups = merge_groups(
                lineage, events, mapping, name, threshold
            )
            all_groups.extend(groups)
            gates[f"{name}_tracker_status"] = (
                summary.get("status") == TRACKER_PASS
            )
            gates[f"{name}_initial_count"] = (
                int(summary.get("initial_particle_count", -1)) == 96
            )
            gates[f"{name}_no_split"] = (
                int(summary.get("split_count", -1)) == 0
            )
            gates[f"{name}_no_new_component"] = (
                int(summary.get("new_component_count", -1)) == 0
            )
            gates[f"{name}_dissolution_qualified"] = (
                int(summary.get("unqualified_dissolution_count", -1)) == 0
            )
            gates[f"{name}_merge_qualified"] = (
                int(summary.get("unqualified_merge_count", -1)) == 0
                and int(summary.get("qualified_merge_count", -1))
                == int(summary.get("merge_count", -2))
            )
            gates[f"{name}_identity_mapping"] = (
                identity_delta <= IDENTITY_TOLERANCE_NM
            )
            gates[f"{name}_group_rows_complete"] = (
                len(groups) == int(summary.get("merge_count", -1))
                and all(group["qualified_merge"] == 1 for group in groups)
            )
        if not all(gates.values()):
            failed = [name for name, passed in gates.items() if not passed]
            raise ValueError(f"unresolved merge-aware gates: {failed}")

        group_thresholds: Dict[str, Set[str]] = {}
        for group in all_groups:
            group_thresholds.setdefault(
                group["merge_group_id"], set()
            ).add(group["threshold_name"])
        classifications: Dict[str, str] = {}
        for group, thresholds in group_thresholds.items():
            if "strong" in thresholds:
                classifications[group] = "STRONG_H_GE_5E3_MERGE"
            elif "medium" in thresholds:
                classifications[group] = "DIFFUSE_TO_INTERMEDIATE_NECK"
            else:
                classifications[group] = "DIFFUSE_TAIL_NECK_ONLY"
        for group in all_groups:
            group["neck_classification"] = classifications[
                group["merge_group_id"]
            ]

        write_csv(args.out / "merge_groups.csv", all_groups)
        audit = {
            "schema": "PF_246CUBE_MERGE_AWARE_LINEAGE_AUDIT_V1",
            "status": PASS,
            "gates": gates,
            "threshold_identity_max_delta_nm": identity_deltas,
            "threshold_identity_tolerance_nm": IDENTITY_TOLERANCE_NM,
            "threshold_summaries": summaries,
            "merge_group_count": len(group_thresholds),
            "merge_groups": all_groups,
            "neck_classifications": classifications,
            "split_count_all_thresholds": 0,
            "new_component_count_all_thresholds": 0,
        }
        (args.out / "audit.json").write_text(
            json.dumps(audit, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        report = [
            "# Merge-aware multi-threshold lineage audit",
            "",
            f"`{PASS}`",
            "",
            "The registered `h>1e-4` identity contract remains active. "
            "Detected many-to-one overlaps are retained as explicit persistent "
            "lineage groups; they are not silently counted as dissolution. "
            "Splits, new components, weak parent mappings, and unqualified "
            "dissolutions remain fail-closed.",
            "",
            "| threshold | initial N | final N | dissolutions | merges | splits |",
            "|---:|---:|---:|---:|---:|---:|",
        ]
        for name, _directory, threshold in THRESHOLDS:
            summary = summaries[name]
            report.append(
                f"| {threshold:.4g} | "
                f"{summary['initial_particle_count']} | "
                f"{summary['final_particle_count']} | "
                f"{summary['dissolution_count']} | "
                f"{summary['merge_count']} | "
                f"{summary['split_count']} |"
            )
        report.extend(("", "Resolved groups:", ""))
        if all_groups:
            report.extend(
                (
                    "| group | threshold | step | age (h) | classification |",
                    "|---|---:|---:|---:|---|",
                )
            )
            for group in all_groups:
                report.append(
                    f"| {group['merge_group_id']} | "
                    f"{group['h_threshold']:.4g} | {group['step']} | "
                    f"{group['experimental_age_h']:.12g} | "
                    f"{group['neck_classification']} |"
                )
        else:
            report.append("No merge was detected at any registered threshold.")
        (args.out / "report.md").write_text(
            "\n".join(report) + "\n", encoding="utf-8"
        )
        (args.out / "status.txt").write_text(PASS + "\n", encoding="utf-8")
        print(PASS)
    except ValueError as exc:
        (args.out / "status.txt").write_text(
            f"{FAIL}\n{exc}\n", encoding="utf-8"
        )
        raise SystemExit(f"[fatal] {exc}") from exc


if __name__ == "__main__":
    main()
