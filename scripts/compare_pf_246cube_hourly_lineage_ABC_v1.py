#!/usr/bin/env python3
"""Deterministic three-replicate summary for qualified hourly PF lineage audits."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


LINEAGE_PASS = "PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1"
PASS = "PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_ABC_ENSEMBLE_V1"
REGISTERED = ((0, 6), (21798, 12), (43596, 18), (65393, 24), (108989, 36), (152585, 48))
FIELDS = (
    "particle_count",
    "number_density_m3",
    "beta_volume_fraction",
    "mean_radius_nm",
    "Sv_nm_inv",
    "M6_nm3",
    "far_field_matrix_xAg",
    "mass_relative_error",
)
EXPERIMENTAL_FAR_XAG_BAND = (0.0058, 0.0066)
BOX_VOLUME_M3 = (246.0e-9) ** 3


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


def field_value(row: Mapping[str, str], field: str) -> float:
    if field == "number_density_m3":
        return float(row["particle_count"]) / BOX_VOLUME_M3
    return float(row[field])


def load_case(audit_path: Path, observations_path: Path) -> dict[str, Any]:
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != LINEAGE_PASS or not all(audit.get("gates", {}).values()):
        raise ValueError(f"lineage audit is not an all-gate PASS: {audit_path}")
    rows = {int(row["step"]): row for row in read_csv(observations_path)}
    if any(step not in rows for step, _age in REGISTERED):
        raise ValueError(f"registered observation is missing: {observations_path}")
    metrics = audit["metrics"]
    initial = int(metrics["initial_particle_count"])
    strong_dissolutions = int(audit["threshold_summaries"]["strong"]["dissolution_count"])
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


def run(args: argparse.Namespace) -> dict[str, Any]:
    if args.out.exists():
        raise ValueError(f"refusing to overwrite output: {args.out}")
    paths = {
        "A": (args.a_audit, args.a_observations),
        "B": (args.b_audit, args.b_observations),
        "C": (args.c_audit, args.c_observations),
    }
    cases = {name: load_case(*case_paths) for name, case_paths in paths.items()}
    args.out.mkdir(parents=True)

    observation_rows: list[dict[str, Any]] = []
    statistics_rows: list[dict[str, Any]] = []
    for step, nominal_age in REGISTERED:
        for case_name in ("A", "B", "C"):
            source = cases[case_name]["observations"][step]
            observation_rows.append(
                {
                    "case": case_name,
                    "step": step,
                    "nominal_age_h": nominal_age,
                    "actual_age_h": source["experimental_age_h"],
                    **{field: field_value(source, field) for field in FIELDS},
                }
            )
        for field in FIELDS:
            values = [field_value(cases[name]["observations"][step], field) for name in ("A", "B", "C")]
            mean = statistics.mean(values)
            sample_std = statistics.stdev(values)
            statistics_rows.append(
                {
                    "step": step,
                    "nominal_age_h": nominal_age,
                    "observable": field,
                    "sample_count": 3,
                    "mean": mean,
                    "sample_std": sample_std,
                    "minimum": min(values),
                    "maximum": max(values),
                    "range": max(values) - min(values),
                    "coefficient_of_variation": "" if mean == 0.0 else sample_std / abs(mean),
                }
            )
    write_csv(
        args.out / "registered_observables_ABC.csv",
        ("case", "step", "nominal_age_h", "actual_age_h", *FIELDS),
        observation_rows,
    )
    write_csv(
        args.out / "registered_observable_ensemble_statistics_ABC.csv",
        (
            "step",
            "nominal_age_h",
            "observable",
            "sample_count",
            "mean",
            "sample_std",
            "minimum",
            "maximum",
            "range",
            "coefficient_of_variation",
        ),
        statistics_rows,
    )

    lineage: dict[str, Any] = {}
    final: dict[str, Any] = {}
    changes: dict[str, Any] = {}
    for name, case in cases.items():
        metrics = case["audit"]["metrics"]
        lineage[name] = {
            "qualified_dissolution_count": int(metrics["dissolved_initial_identity_count"]),
            "final_strong_identity_lower_bound": case["strong_identity_lower"],
            "final_medium_identity_upper_bound": case["medium_identity_upper"],
            "final_low_support_component_count": int(metrics["final_low_support_component_count"]),
            "threshold_contact_group_count": int(metrics["threshold_contact_group_count"]),
            "persistent_strong_core_merge_group_count": int(metrics["persistent_strong_core_merge_group_count"]),
            "max_mass_relative_error": float(metrics["max_mass_relative_error"]),
        }
        initial_row = case["observations"][0]
        final_row = case["observations"][152585]
        final[name] = {field: field_value(final_row, field) for field in FIELDS}
        changes[name] = {
            field: field_value(final_row, field) - field_value(initial_row, field) for field in FIELDS
        }

    gates = {
        "all_lineage_audits_pass": True,
        "particle_count_decreases_in_every_replicate": all(
            changes[name]["particle_count"] < 0.0 for name in cases
        ),
        "mean_radius_increases_in_every_replicate": all(
            changes[name]["mean_radius_nm"] > 0.0 for name in cases
        ),
        "interface_area_density_decreases_in_every_replicate": all(
            changes[name]["Sv_nm_inv"] < 0.0 for name in cases
        ),
        "M6_increases_in_every_replicate": all(changes[name]["M6_nm3"] > 0.0 for name in cases),
        "final_far_xAg_in_experimental_band_in_every_replicate": all(
            EXPERIMENTAL_FAR_XAG_BAND[0]
            <= final[name]["far_field_matrix_xAg"]
            <= EXPERIMENTAL_FAR_XAG_BAND[1]
            for name in cases
        ),
        "mass_closure_below_1e-10_in_every_replicate": all(
            lineage[name]["max_mass_relative_error"] <= 1.0e-10 for name in cases
        ),
        "all_merge_split_and_identity_events_resolved_fail_closed": all(
            all(cases[name]["audit"]["gates"].values()) for name in cases
        ),
    }
    if not all(gates.values()):
        raise ValueError(f"ABC comparison gate failed: {gates}")

    summary = {
        "schema": "PF_246CUBE_HOURLY_MERGE_DISSOLUTION_ABC_ENSEMBLE_V1",
        "status": PASS,
        "comparison_semantics": "THREE_HASH_PINNED_REPLICATE_PRELIMINARY_ENSEMBLE_NOT_ENSEMBLE_CONVERGENCE",
        "experimental_far_xAg_band_fraction": list(EXPERIMENTAL_FAR_XAG_BAND),
        "gates": gates,
        "lineage": lineage,
        "final_48h": final,
        "change_6h_to_48h": changes,
        "input_sha256": {
            f"{name}_{kind}": sha256(path)
            for name, case_paths in paths.items()
            for kind, path in zip(("audit", "observations"), case_paths)
        },
    }
    (args.out / "comparison_audit.json").write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    lines = [
        "# A/B/C hourly merge-aware coarsening ensemble",
        "",
        f"`{PASS}`",
        "",
        "All three hash-pinned spatial realizations pass the same three-threshold lineage contract. The statistics below are a preliminary three-replicate ensemble, not an ensemble- or box-size-convergence claim.",
        "",
        "## Lineage",
        "",
        "| case | qualified dissolutions | final identity interval | final support components | threshold contacts | persistent strong-core merges |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for name in ("A", "B", "C"):
        item = lineage[name]
        lower = item["final_strong_identity_lower_bound"]
        upper = item["final_medium_identity_upper_bound"]
        interval = str(lower) if lower == upper else f"{lower}--{upper}"
        lines.append(
            f"| {name} | {item['qualified_dissolution_count']} | {interval} | "
            f"{item['final_low_support_component_count']} | {item['threshold_contact_group_count']} | "
            f"{item['persistent_strong_core_merge_group_count']} |"
        )
    lines.extend(
        [
            "",
            "## Registered observables",
            "",
            "| age (h) | case | N | Nv (m^-3) | mean R (nm) | beta volume fraction | Sv (nm^-1) | M6 (nm^3) | far xAg (at.%) |",
            "|---:|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for step, nominal_age in REGISTERED:
        for name in ("A", "B", "C"):
            row = cases[name]["observations"][step]
            lines.append(
                f"| {nominal_age} | {name} | {int(row['particle_count'])} | "
                f"{field_value(row, 'number_density_m3'):.6g} | "
                f"{float(row['mean_radius_nm']):.6g} | {float(row['beta_volume_fraction']):.9g} | "
                f"{float(row['Sv_nm_inv']):.9g} | {float(row['M6_nm3']):.9g} | "
                f"{100.0 * float(row['far_field_matrix_xAg']):.6g} |"
            )
    lines.extend(
        [
            "",
            "The three fixtures share the same physics and conserved inventory and differ only in their hash-pinned spatial realization. The resulting spread measures spatial-realization sensitivity for this three-member fixture family; it does not justify parameter retuning or selective exclusion of a path.",
        ]
    )
    (args.out / "comparison_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    (args.out / "status.txt").write_text(PASS + "\n", encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    for name in ("a", "b", "c"):
        parser.add_argument(f"--{name}-audit", type=Path, required=True)
        parser.add_argument(f"--{name}-observations", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    result = run(args)
    print(result["status"])


if __name__ == "__main__":
    main()
