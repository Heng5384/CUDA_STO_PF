#!/usr/bin/env python3
"""Contract tests for the hour-cadence merge/dissolution audit."""

from __future__ import annotations

import csv
import hashlib
import json
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Sequence


PASS = "PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1"
FAIL = "BLOCKED_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1"
ROOT = Path(__file__).resolve().parents[1]
AUDITOR = ROOT / "scripts" / "audit_pf_246cube_hourly_merge_dissolution_v1.py"
STEPS = (0, 10, 20)
PARTICLES = ("P000", "P001", "P002", "P003")


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def component(
    step: int,
    stable: int,
    members: str,
    volume: float,
    centroid: Sequence[float],
) -> Dict[str, object]:
    return {
        "step": step,
        "elapsed_physical_time_s": step,
        "experimental_age_h": 6.0 + step / 3600.0,
        "stable_particle_id": stable,
        "lineage_member_ids": members,
        "lineage_group_size": len(members.split("+")),
        "component_label": stable,
        "threshold_voxel_count": max(1, int(volume * 10)),
        "h_volume_nm3": volume,
        "equivalent_radius_nm": max(0.1, volume ** (1.0 / 3.0)),
        "centroid_x_nm": centroid[0],
        "centroid_y_nm": centroid[1],
        "centroid_z_nm": centroid[2],
    }


def event(
    step: int,
    kind: str,
    stable: int,
    parent_count: int,
    child_count: int,
    qualified_dissolution: int = 0,
    qualified_merge: int = 0,
    parent_ids: str = "",
    child_id: int = -1,
    detail: str = "",
) -> Dict[str, object]:
    return {
        "step": step,
        "event_type": kind,
        "stable_particle_id": stable,
        "parent_count": parent_count,
        "child_count": child_count,
        "qualified_dissolution": qualified_dissolution,
        "qualified_merge": qualified_merge,
        "parent_stable_ids": parent_ids,
        "child_stable_id": child_id,
        "detail": detail,
    }


LINEAGE_FIELDS = (
    "step",
    "elapsed_physical_time_s",
    "experimental_age_h",
    "stable_particle_id",
    "lineage_member_ids",
    "lineage_group_size",
    "component_label",
    "threshold_voxel_count",
    "h_volume_nm3",
    "equivalent_radius_nm",
    "centroid_x_nm",
    "centroid_y_nm",
    "centroid_z_nm",
)
EVENT_FIELDS = (
    "step",
    "event_type",
    "stable_particle_id",
    "parent_count",
    "child_count",
    "qualified_dissolution",
    "qualified_merge",
    "parent_stable_ids",
    "child_stable_id",
    "detail",
)
OBS_FIELDS = (
    "step",
    "elapsed_physical_time_s",
    "experimental_age_h",
    "particle_count",
    "beta_volume_fraction",
    "mean_radius_nm",
    "Sv_nm_inv",
    "M6_nm3",
    "far_field_matrix_xB",
    "far_field_matrix_xAg",
    "far4_matrix_xB",
    "far4_matrix_xAg",
    "far3_fraction",
    "far4_fraction",
    "far3_far4_xAg_delta",
    "broad_matrix_xB",
    "broad_matrix_xAg",
    "canonical_inventory_code",
    "mass_relative_error",
    "phi_normalized_l1_from_initial",
    "xB_mean_absolute_from_initial",
    "phi_min",
    "phi_max",
    "xB_min",
    "xB_max",
    "finite",
    "bounds",
)


def base_lineage(name: str) -> List[Dict[str, object]]:
    centroids = ((1, 1, 1), (3, 3, 3), (5, 5, 5), (7, 7, 7))
    volume = {"low": 6.0, "medium": 5.99, "strong": 5.8}[name]
    rows = [
        component(0, index, str(index), volume, centroids[index])
        for index in range(4)
    ]
    if name == "low":
        rows.extend(
            (
                component(10, 1, "1+2", 12.0, (4, 4, 4)),
                component(10, 3, "3", 6.0, centroids[3]),
                component(20, 1, "1+2", 12.1, (4, 4, 4)),
                component(20, 3, "3", 6.0, centroids[3]),
            )
        )
    elif name == "medium":
        rows.extend(
            (
                component(10, 1, "1", 5.99, centroids[1]),
                component(10, 2, "2", 5.99, centroids[2]),
                component(10, 3, "3", 5.99, centroids[3]),
                component(20, 1, "1+2", 12.0, (4, 4, 4)),
                component(20, 3, "3", 5.99, centroids[3]),
            )
        )
    else:
        for step in (10, 20):
            rows.extend(
                component(step, index, str(index), 5.8, centroids[index])
                for index in (1, 2, 3)
            )
    return rows


def base_events(name: str) -> List[Dict[str, object]]:
    rows = [
        event(
            10,
            "dissolution",
            0,
            1,
            0,
            qualified_dissolution=0,
            parent_ids="0",
            detail="insufficient continuous decay evidence",
        )
    ]
    merge_detail = (
        "parent_id:overlap_voxels:parent_fraction:child_fraction="
        "1:60:1:0.5+2:60:1:0.5"
    )
    if name == "low":
        rows.append(event(10, "merge", 1, 2, 1, 0, 1, "1+2", 1, merge_detail))
    elif name == "medium":
        rows.append(event(20, "merge", 1, 2, 1, 0, 1, "1+2", 1, merge_detail))
    return rows


def observations(name: str, lineage: Sequence[Mapping[str, object]]) -> List[Dict[str, object]]:
    global_h = {0: 24.05, 10: 18.05, 20: 18.10}
    result: List[Dict[str, object]] = []
    for step in STEPS:
        count = sum(int(row["step"]) == step for row in lineage)
        result.append(
            {
                "step": step,
                "elapsed_physical_time_s": step,
                "experimental_age_h": 6.0 + step / 3600.0,
                "particle_count": count,
                "beta_volume_fraction": global_h[step] / 1000.0,
                "mean_radius_nm": 1.0,
                "Sv_nm_inv": 0.1,
                "M6_nm3": 1.0,
                "far_field_matrix_xB": 0.0062,
                "far_field_matrix_xAg": 0.00618,
                "far4_matrix_xB": 0.0062,
                "far4_matrix_xAg": 0.00618,
                "far3_fraction": 0.8,
                "far4_fraction": 0.7,
                "far3_far4_xAg_delta": 0.0,
                "broad_matrix_xB": 0.0062,
                "broad_matrix_xAg": 0.00618,
                "canonical_inventory_code": 100.0,
                "mass_relative_error": 0.0,
                "phi_normalized_l1_from_initial": 0.0,
                "xB_mean_absolute_from_initial": 0.0,
                "phi_min": 0.0,
                "phi_max": 1.0,
                "xB_min": 0.005,
                "xB_max": 0.01,
                "finite": 1,
                "bounds": 1,
            }
        )
    return result


def materialize_case(root: Path, fault: str | None = None) -> Dict[str, Path]:
    root.mkdir(parents=True)
    roots: Dict[str, Path] = {}
    initial_rows = []
    centroids = ((1, 1, 1), (3, 3, 3), (5, 5, 5), (7, 7, 7))
    for index, particle in enumerate(PARTICLES):
        initial_rows.append(
            {
                "component_label": index,
                "threshold_voxel_count": 10,
                "h_volume_nm3": 6.0,
                "equivalent_radius_nm": 1.0,
                "centroid_nm": json.dumps(centroids[index]),
                "semi_axes_nm": "[]",
                "axis_ratio": 1.0,
                "principal_axes_rows": "[]",
                "particle_id": particle,
            }
        )
    initial = root / "initial_components.csv"
    write_csv(initial, tuple(initial_rows[0]), initial_rows)
    steps = root / "expected_steps.txt"
    steps.write_text("10\n20\n", encoding="utf-8")

    for name in ("low", "medium", "strong"):
        target = root / name
        target.mkdir()
        roots[name] = target
        lineage = base_lineage(name)
        events = base_events(name)
        if fault == "split" and name == "low":
            events.append(event(20, "split", 3, 1, 2, parent_ids="3"))
        if fault == "new_component" and name == "low":
            events.append(event(20, "new_component_without_overlap", -1, 0, 1))
        if fault == "unqualified_merge" and name == "low":
            events[1]["qualified_merge"] = 0
        if fault == "missing_dissolution_event" and name == "low":
            events = [row for row in events if row["event_type"] != "dissolution"]
        if fault == "reappearance" and name == "low":
            lineage.append(component(20, 0, "0", 0.05, (1, 1, 1)))
        if fault == "duplicate_identity" and name == "low":
            lineage.append(component(10, 99, "3", 0.05, (7, 7, 7)))
        if fault == "threshold_order" and name == "strong":
            lineage.append(component(10, 0, "0", 0.1, (1, 1, 1)))
            events = [row for row in events if row["event_type"] != "dissolution"]
            events.append(
                event(20, "dissolution", 0, 1, 0, 0, 0, "0", -1, "late")
            )
        obs = observations(name, lineage)
        if fault == "mass_drift" and name == "low":
            obs[-1]["mass_relative_error"] = 1.0e-5
        if fault == "missing_step" and name == "strong":
            lineage = [row for row in lineage if int(row["step"]) != 20]
            obs = [row for row in obs if int(row["step"]) != 20]
        write_csv(target / "particle_lineage.csv", LINEAGE_FIELDS, lineage)
        write_csv(target / "particle_events.csv", EVENT_FIELDS, events)
        write_csv(target / "ensemble_observables.csv", OBS_FIELDS, obs)
        initial_count = sum(int(row["step"]) == 0 for row in lineage)
        final_count = sum(int(row["step"]) == 20 for row in lineage)
        dissolution_count = sum(row["event_type"] == "dissolution" for row in events)
        merge_count = sum(row["event_type"] == "merge" for row in events)
        split_count = sum(row["event_type"] == "split" for row in events)
        new_count = sum(
            row["event_type"] == "new_component_without_overlap" for row in events
        )
        summary = (
            "status=BLOCKED_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1\n"
            f"initial_particle_count={initial_count}\n"
            f"final_particle_count={final_count}\n"
            "snapshot_count=3\n"
            f"dissolution_count={dissolution_count}\n"
            f"unqualified_dissolution_count={dissolution_count}\n"
            f"merge_count={merge_count}\n"
            f"qualified_merge_count={sum(int(row['qualified_merge']) for row in events)}\n"
            f"unqualified_merge_count={sum(row['event_type'] == 'merge' and not int(row['qualified_merge']) for row in events)}\n"
            f"split_count={split_count}\n"
            f"new_component_count={new_count}\n"
            "max_mass_relative_error=0\n"
        )
        (target / "lineage_summary.txt").write_text(summary, encoding="utf-8")
        (target / "status.txt").write_text(
            "BLOCKED_PERIODIC_OVERLAP_PARTICLE_LINEAGE_V1\n", encoding="utf-8"
        )
    return {**roots, "initial": initial, "steps": steps}


def run_audit(inputs: Mapping[str, Path], out: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        (
            sys.executable,
            str(AUDITOR),
            "--low",
            str(inputs["low"]),
            "--medium",
            str(inputs["medium"]),
            "--strong",
            str(inputs["strong"]),
            "--initial-components",
            str(inputs["initial"]),
            "--expected-steps",
            str(inputs["steps"]),
            "--out",
            str(out),
            "--expected-initial-count",
            "4",
            "--domain-nm",
            "10",
            "--physical-dt-s",
            "1",
        ),
        text=True,
        capture_output=True,
        check=False,
    )


def tree_hashes(root: Path) -> Dict[str, str]:
    return {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


class HourlyAuditContractTest(unittest.TestCase):
    def test_positive_interval_and_determinism(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            inputs = materialize_case(root / "inputs")
            first = run_audit(inputs, root / "out_a")
            second = run_audit(inputs, root / "out_b")
            self.assertEqual(first.returncode, 0, first.stderr)
            self.assertEqual(second.returncode, 0, second.stderr)
            self.assertEqual(
                (root / "out_a/status.txt").read_text().strip(), PASS
            )
            self.assertEqual(tree_hashes(root / "out_a"), tree_hashes(root / "out_b"))
            audit = json.loads((root / "out_a/audit.json").read_text())
            self.assertEqual(audit["metrics"]["legacy_medium_unqualified_dissolution_count"], 1)
            self.assertEqual(audit["metrics"]["dissolved_initial_identity_count"], 1)
            self.assertFalse(audit["contract"]["exact_event_time_claimed"])

    def test_fail_closed_negative_matrix(self) -> None:
        faults = (
            "split",
            "new_component",
            "unqualified_merge",
            "missing_dissolution_event",
            "reappearance",
            "duplicate_identity",
            "threshold_order",
            "mass_drift",
            "missing_step",
        )
        for fault in faults:
            with self.subTest(fault=fault), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                inputs = materialize_case(root / "inputs", fault=fault)
                result = run_audit(inputs, root / "out")
                self.assertNotEqual(result.returncode, 0)
                self.assertEqual(
                    (root / "out/status.txt").read_text().splitlines()[0], FAIL
                )


if __name__ == "__main__":
    unittest.main()
