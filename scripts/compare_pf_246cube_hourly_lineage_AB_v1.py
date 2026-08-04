#!/usr/bin/env python3
"""Deterministic A/B comparison for qualified hourly lineage audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping, Sequence


LINEAGE_PASS = "PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1"
PASS = "PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AB_COMPARISON_V1"
REGISTERED = ((0, 6), (21798, 12), (43596, 18), (65393, 24), (108989, 36), (152585, 48))
FIELDS = (
    "particle_count",
    "beta_volume_fraction",
    "mean_radius_nm",
    "Sv_nm_inv",
    "M6_nm3",
    "far_field_matrix_xAg",
    "mass_relative_error",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def load_case(audit_path: Path, observations_path: Path) -> Dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != LINEAGE_PASS or not all(audit.get("gates", {}).values()):
        raise ValueError(f"lineage audit is not an all-gate PASS: {audit_path}")
    rows = {int(row["step"]): row for row in read_csv(observations_path)}
    if any(step not in rows for step, _age in REGISTERED):
        raise ValueError(f"registered observation is missing: {observations_path}")
    metrics = audit["metrics"]
    initial = int(metrics["initial_particle_count"])
    strong_dissolutions = int(
        audit["threshold_summaries"]["strong"]["dissolution_count"]
    )
    strong_identity_lower = initial - strong_dissolutions
    medium_identity_upper = int(metrics["final_resolved_core_identity_count"])
    if strong_identity_lower > medium_identity_upper:
        raise ValueError("strong/medium final identity bounds are inverted")
    return {
        "audit": audit,
        "observations": rows,
        "strong_identity_lower": strong_identity_lower,
        "medium_identity_upper": medium_identity_upper,
    }


def run(args: argparse.Namespace) -> Dict[str, Any]:
    if args.out.exists():
        raise ValueError(f"refusing to overwrite output: {args.out}")
    cases = {
        "A": load_case(args.a_audit, args.a_observations),
        "B": load_case(args.b_audit, args.b_observations),
    }
    args.out.mkdir(parents=True)
    observation_rows = []
    difference_rows = []
    for step, nominal_age in REGISTERED:
        for case_name in ("A", "B"):
            source = cases[case_name]["observations"][step]
            observation_rows.append(
                {
                    "case": case_name,
                    "step": step,
                    "nominal_age_h": nominal_age,
                    "actual_age_h": source["experimental_age_h"],
                    **{field: source[field] for field in FIELDS},
                }
            )
        for field in FIELDS:
            left = float(cases["A"]["observations"][step][field])
            right = float(cases["B"]["observations"][step][field])
            difference_rows.append(
                {
                    "step": step,
                    "nominal_age_h": nominal_age,
                    "observable": field,
                    "A": left,
                    "B": right,
                    "B_minus_A": right - left,
                    "B_minus_A_relative_to_A": "" if left == 0.0 else (right - left) / left,
                }
            )
    write_csv(
        args.out / "registered_observables_AB.csv",
        ("case", "step", "nominal_age_h", "actual_age_h", *FIELDS),
        observation_rows,
    )
    write_csv(
        args.out / "registered_observable_differences_B_minus_A.csv",
        ("step", "nominal_age_h", "observable", "A", "B", "B_minus_A", "B_minus_A_relative_to_A"),
        difference_rows,
    )
    final = {
        case_name: {
            field: float(case["observations"][152585][field])
            for field in FIELDS
        }
        for case_name, case in cases.items()
    }
    lineage = {}
    for case_name, case in cases.items():
        metrics = case["audit"]["metrics"]
        lineage[case_name] = {
            "qualified_dissolution_count": int(metrics["dissolved_initial_identity_count"]),
            "final_strong_identity_lower_bound": case["strong_identity_lower"],
            "final_medium_identity_upper_bound": case["medium_identity_upper"],
            "final_low_support_component_count": int(metrics["final_low_support_component_count"]),
            "threshold_contact_group_count": int(metrics["threshold_contact_group_count"]),
            "persistent_strong_core_merge_group_count": int(metrics["persistent_strong_core_merge_group_count"]),
            "unresolved_split_count": 0,
            "new_component_count": 0,
            "max_mass_relative_error": float(metrics["max_mass_relative_error"]),
        }
    summary = {
        "schema": "PF_246CUBE_HOURLY_MERGE_DISSOLUTION_AB_COMPARISON_V1",
        "status": PASS,
        "comparison_semantics": "DESCRIPTIVE_HASH_PINNED_REPLICATE_COMPARISON_NOT_ENSEMBLE_CONVERGENCE",
        "lineage": lineage,
        "final_48h": final,
        "final_48h_B_minus_A": {
            field: final["B"][field] - final["A"][field] for field in FIELDS
        },
        "final_48h_B_minus_A_relative_to_A": {
            field: (final["B"][field] - final["A"][field]) / final["A"][field]
            if final["A"][field] != 0.0
            else None
            for field in FIELDS
        },
        "input_sha256": {
            "A_audit": sha256(args.a_audit),
            "A_observations": sha256(args.a_observations),
            "B_audit": sha256(args.b_audit),
            "B_observations": sha256(args.b_observations),
        },
    }
    (args.out / "comparison_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# A/B hourly merge-aware and coarsening comparison",
        "",
        f"`{PASS}`",
        "",
        "Both cases passed the identical three-threshold lineage audit. This is a two-replicate descriptive comparison, not an ensemble-convergence claim.",
        "",
        "## Lineage",
        "",
        "| case | qualified dissolutions | final identity interval | final support components | threshold contacts | persistent strong-core merges |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for case_name in ("A", "B"):
        item = lineage[case_name]
        lower = item["final_strong_identity_lower_bound"]
        upper = item["final_medium_identity_upper_bound"]
        interval = str(lower) if lower == upper else f"{lower}--{upper}"
        lines.append(
            f"| {case_name} | {item['qualified_dissolution_count']} | {interval} | "
            f"{item['final_low_support_component_count']} | {item['threshold_contact_group_count']} | "
            f"{item['persistent_strong_core_merge_group_count']} |"
        )
    lines.extend(
        [
            "",
            "## Registered observables",
            "",
            "| age (h) | case | N | mean R (nm) | beta volume fraction | Sv (nm^-1) | M6 (nm^3) | far xAg (at.%) |",
            "|---:|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for step, nominal_age in REGISTERED:
        for case_name in ("A", "B"):
            row = cases[case_name]["observations"][step]
            lines.append(
                f"| {nominal_age} | {case_name} | {int(row['particle_count'])} | "
                f"{float(row['mean_radius_nm']):.6g} | {float(row['beta_volume_fraction']):.9g} | "
                f"{float(row['Sv_nm_inv']):.9g} | {float(row['M6_nm3']):.9g} | "
                f"{100.0 * float(row['far_field_matrix_xAg']):.6g} |"
            )
    lines.extend(
        [
            "",
            "The A and B fixtures begin with practically identical inventory and size statistics. Differences therefore quantify sensitivity to the hash-pinned spatial realization, subject to the two-sample limitation.",
        ]
    )
    (args.out / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "status.txt").write_text(PASS + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a-audit", type=Path, required=True)
    parser.add_argument("--a-observations", type=Path, required=True)
    parser.add_argument("--b-audit", type=Path, required=True)
    parser.add_argument("--b-observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(result["status"])


if __name__ == "__main__":
    main()
