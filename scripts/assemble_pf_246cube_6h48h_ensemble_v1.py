#!/usr/bin/env python3
"""Assemble A/B/C 6--48 h conditional-path ensemble statistics."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Dict, List


PASS = "PASS_246CUBE_THREE_REPLICATE_6H48H_ENSEMBLE_V1"
CASE_PASS = "PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1"
STEPS = [0, 21798, 43596, 65393, 108989, 152585]
METRICS = (
    "particle_count",
    "number_density_m3",
    "mean_radius_nm",
    "beta_volume_fraction",
    "Sv_nm_inv",
    "M6_nm3",
    "far_field_matrix_xAg",
)


def rows(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case-a", type=Path, required=True)
    parser.add_argument("--case-b", type=Path, required=True)
    parser.add_argument("--case-c", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    cases = {
        "A": args.case_a,
        "B": args.case_b,
        "C": args.case_c,
    }
    audits = {}
    observations = {}
    psd = {}
    for label, root in cases.items():
        status = (root / "status.txt").read_text(encoding="utf-8").strip()
        if status != CASE_PASS:
            raise SystemExit(f"case {label} is not exact PASS: {status}")
        audits[label] = json.loads(
            (root / "audit.json").read_text(encoding="utf-8")
        )
        observations[label] = {
            int(row["step"]): row
            for row in rows(root / "registered_observables.csv")
        }
        psd[label] = rows(root / "registered_particle_psd.csv")
        if sorted(observations[label]) != STEPS:
            raise SystemExit(f"case {label} registered steps mismatch")

    ensemble_rows = []
    for step in STEPS:
        for metric in METRICS:
            values = [
                float(observations[label][step][metric])
                for label in "ABC"
            ]
            ensemble_rows.append(
                {
                    "step": step,
                    "experimental_age_h": observations["A"][step][
                        "experimental_age_h"
                    ],
                    "metric": metric,
                    "mean": statistics.fmean(values),
                    "population_std": statistics.pstdev(values),
                    "minimum": min(values),
                    "maximum": max(values),
                }
            )
    with (args.out / "ensemble_statistics.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(ensemble_rows[0])
        )
        writer.writeheader()
        writer.writerows(ensemble_rows)

    psd_rows = []
    for step in STEPS:
        per_replicate = {}
        pooled = []
        for label in "ABC":
            values = [
                float(row["equivalent_radius_nm"])
                for row in psd[label]
                if int(row["step"]) == step
            ]
            if not values:
                raise SystemExit(f"empty PSD for {label} step {step}")
            pooled.extend(values)
            per_replicate[label] = {
                "count": len(values),
                "mean": statistics.fmean(values),
                "std": statistics.pstdev(values),
                "min": min(values),
                "max": max(values),
            }
        psd_rows.append(
            {
                "step": step,
                "pooled_particle_count": len(pooled),
                "pooled_radius_mean_nm": statistics.fmean(pooled),
                "pooled_radius_std_nm": statistics.pstdev(pooled),
                "pooled_radius_min_nm": min(pooled),
                "pooled_radius_max_nm": max(pooled),
                "replicate_mean_radius_std_nm": statistics.pstdev(
                    [per_replicate[x]["mean"] for x in "ABC"]
                ),
                "replicate_particle_count_min": min(
                    per_replicate[x]["count"] for x in "ABC"
                ),
                "replicate_particle_count_max": max(
                    per_replicate[x]["count"] for x in "ABC"
                ),
            }
        )
    with (args.out / "ensemble_psd_summary.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(psd_rows[0]))
        writer.writeheader()
        writer.writerows(psd_rows)

    classifications = {
        label: audits[label]["scientific_status"] for label in "ABC"
    }
    allowed = sum(
        value == "EXPERIMENT_ALLOWED_CONDITIONAL_PATH"
        for value in classifications.values()
    )
    excluded = sum(
        value == "SCIENTIFICALLY_EXCLUDED_TOTAL_INVENTORY_PSD_FAMILY"
        for value in classifications.values()
    )
    if allowed == 3:
        family_status = "EXPERIMENT_ALLOWED_CONDITIONAL_PATH_ENSEMBLE"
    elif excluded == 3:
        family_status = (
            "SCIENTIFICALLY_EXCLUDED_TOTAL_INVENTORY_PSD_FAMILY"
        )
    else:
        family_status = "MIXED_CONDITIONAL_PATH_ENSEMBLE"
    result = {
        "schema": "PF_246CUBE_THREE_REPLICATE_6H48H_ENSEMBLE_AUDIT_V1",
        "numerical_status": PASS,
        "scientific_family_status": family_status,
        "case_classifications": classifications,
        "experiment_far_field_xAg_band": [0.0058, 0.0066],
        "physical_parameter_retuning": False,
    }
    (args.out / "ensemble_decision.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    (args.out / "final_terminal_output.txt").write_text(
        "\n".join(
            (
                f"numerical_status={PASS}",
                f"scientific_family_status={family_status}",
                f"replicate_A_status={classifications['A']}",
                f"replicate_B_status={classifications['B']}",
                f"replicate_C_status={classifications['C']}",
                "physical_parameter_retuning=false",
            )
        )
        + "\n",
        encoding="utf-8",
    )
    print(PASS)


if __name__ == "__main__":
    main()
