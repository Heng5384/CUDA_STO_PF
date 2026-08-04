#!/usr/bin/env python3
"""Qualify accepted-field elasticity replay against an online synchronized solve."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np


STRAIN = ("xx", "yy", "zz", "xy", "xz", "yz")
STRESS = STRAIN
DISPLACEMENT = ("x", "y", "z")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def read_summary(root: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for line in (root / "replay_summary.txt").read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            values[key] = value
    return values


def normalized_metrics(reference: np.ndarray, candidate: np.ndarray) -> dict[str, float]:
    delta = candidate.astype(np.float64) - reference.astype(np.float64)
    ref = reference.astype(np.float64)
    floor = np.finfo(np.float64).tiny
    ref_l1 = max(float(np.sum(np.abs(ref))), floor)
    ref_l2 = max(float(np.linalg.norm(ref.ravel())), floor)
    ref_centered = ref - float(np.mean(ref))
    candidate_centered = candidate.astype(np.float64) - float(np.mean(candidate))
    corr_denom = float(np.linalg.norm(ref_centered.ravel())) * float(
        np.linalg.norm(candidate_centered.ravel())
    )
    correlation = (
        float(np.dot(ref_centered.ravel(), candidate_centered.ravel()) / corr_denom)
        if corr_denom > floor
        else (1.0 if np.array_equal(reference, candidate) else float("nan"))
    )
    return {
        "normalized_l1": float(np.sum(np.abs(delta))) / ref_l1,
        "normalized_l2": float(np.linalg.norm(delta.ravel())) / ref_l2,
        "max_abs": float(np.max(np.abs(delta))),
        "correlation": correlation,
    }


def load_field(root: Path, name: str, dtype: str, count: int) -> np.ndarray:
    path = root / name
    field = np.fromfile(path, dtype=dtype)
    if field.size != count:
        raise ValueError(f"{path}: expected {count} values, found {field.size}")
    return field


def compare_pair(reference_root: Path, candidate_root: Path, label: str) -> tuple[list[dict], dict]:
    reference_summary = read_summary(reference_root)
    candidate_summary = read_summary(candidate_root)
    if reference_summary.get("schema") != "MECHANICS_ONLY_ACCEPTED_FIELD_REPLAY_V1":
        raise ValueError("reference schema mismatch")
    if candidate_summary.get("schema") != "MECHANICS_ONLY_ACCEPTED_FIELD_REPLAY_V1":
        raise ValueError(f"{label}: candidate schema mismatch")
    grid = tuple(int(value) for value in reference_summary["grid"].split(","))
    if candidate_summary.get("grid") != reference_summary["grid"]:
        raise ValueError(f"{label}: grid mismatch")
    count = math.prod(grid)

    rows: list[dict] = []
    source_exact = True
    for source_name in ("accepted_phi.raw.f64", "accepted_xB.raw.f64"):
        reference_path = reference_root / source_name
        candidate_path = candidate_root / source_name
        exact = sha256(reference_path) == sha256(candidate_path)
        source_exact = source_exact and exact
        rows.append(
            {
                "comparison": label,
                "family": "accepted_source",
                "component": source_name,
                "normalized_l1": 0.0 if exact else float("nan"),
                "normalized_l2": 0.0 if exact else float("nan"),
                "max_abs": 0.0 if exact else float("nan"),
                "correlation": 1.0 if exact else float("nan"),
                "byte_identical": exact,
            }
        )

    family_specs = (
        ("strain", STRAIN, "strain_{}.raw.f32"),
        ("stress", STRESS, "stress_{}.raw.f32"),
        ("displacement", DISPLACEMENT, "displacement_{}.raw.f32"),
    )
    maxima = {"strain": 0.0, "stress": 0.0, "displacement": 0.0}
    for family, components, template in family_specs:
        for component in components:
            name = template.format(component)
            reference = load_field(reference_root, name, "<f4", count)
            candidate = load_field(candidate_root, name, "<f4", count)
            metrics = normalized_metrics(reference, candidate)
            maxima[family] = max(maxima[family], metrics["normalized_l2"])
            rows.append(
                {
                    "comparison": label,
                    "family": family,
                    "component": component,
                    **metrics,
                    "byte_identical": bool(np.array_equal(reference, candidate)),
                }
            )

    reference_energy = float(reference_summary["total_elastic_energy_J"])
    candidate_energy = float(candidate_summary["total_elastic_energy_J"])
    energy_relative_error = abs(candidate_energy - reference_energy) / max(
        abs(reference_energy), np.finfo(np.float64).tiny
    )
    immutable_flags = ("time_advanced", "phi_advanced", "xB_advanced", "checkpoint_written")
    no_advance = all(candidate_summary.get(key) == "false" for key in immutable_flags)
    same_accepted_step = (
        candidate_summary.get("accepted_field_step")
        == reference_summary.get("accepted_field_step")
    )
    pair_pass = bool(
        source_exact
        and no_advance
        and same_accepted_step
        and energy_relative_error <= 1.0e-6
        and maxima["strain"] <= 1.0e-5
        and maxima["stress"] <= 1.0e-5
    )
    summary = {
        "comparison": label,
        "source_fields_byte_identical": source_exact,
        "same_accepted_field_step": same_accepted_step,
        "no_source_or_time_advance": no_advance,
        "energy_relative_error": energy_relative_error,
        "max_strain_normalized_l2": maxima["strain"],
        "max_stress_normalized_l2": maxima["stress"],
        "max_displacement_normalized_l2": maxima["displacement"],
        "status": "PASS" if pair_pass else "FAIL",
    }
    return rows, summary


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--online", type=Path, required=True)
    parser.add_argument("--warm-replay", type=Path, required=True)
    parser.add_argument("--zero-replay", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)

    all_rows: list[dict] = []
    summaries: list[dict] = []
    for candidate, label in (
        (args.warm_replay, "online_vs_checkpoint_warm_replay"),
        (args.zero_replay, "online_vs_zero_initialized_replay"),
    ):
        rows, summary = compare_pair(args.online, candidate, label)
        all_rows.extend(rows)
        summaries.append(summary)

    with (args.out / "replay_online_vs_offline_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)

    warm_summary = next(
        item
        for item in summaries
        if item["comparison"] == "online_vs_checkpoint_warm_replay"
    )
    zero_summary = next(
        item
        for item in summaries
        if item["comparison"] == "online_vs_zero_initialized_replay"
    )
    # Historical V4 checkpoints always carry the registered n-1 warm state.
    # Only that exact production path is required and qualified for authority
    # replay.  The zero-initialized calculation is retained as a deliberately
    # fail-closed sensitivity check; a failure there cannot be relabeled or
    # used as an accepted historical replay path.
    overall_pass = warm_summary["status"] == "PASS"
    bundle_files = sorted(
        path.name
        for path in args.online.iterdir()
        if path.is_file() and path.name != "replay_summary.txt"
    )
    with (args.out / "replay_field_hashes.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=("role", "file", "sha256"))
        writer.writeheader()
        for role, root in (
            ("online_synchronized_reference", args.online),
            ("checkpoint_warm_replay", args.warm_replay),
            ("zero_initialized_replay", args.zero_replay),
        ):
            for name in bundle_files:
                writer.writerow({"role": role, "file": name, "sha256": sha256(root / name)})
    audit = {
        "schema": "MECHANICS_ONLY_ACCEPTED_FIELD_REPLAY_QUALIFICATION_V1",
        "thresholds": {
            "energy_relative_error_max": 1.0e-6,
            "strain_normalized_l2_max": 1.0e-5,
            "stress_normalized_l2_max": 1.0e-5,
        },
        "comparisons": summaries,
        "accepted_replay_initialization": "checkpoint_warm",
        "zero_initialized_replay_status": (
            "QUALIFIED" if zero_summary["status"] == "PASS" else "REJECTED_NOT_QUALIFIED"
        ),
        "zero_initialized_replay_permitted_for_authority": False,
        "status": (
            "PASS_MECHANICS_ONLY_ACCEPTED_FIELD_REPLAY_V1"
            if overall_pass
            else "BLOCKED_MECHANICS_ONLY_REPLAY"
        ),
    }
    (args.out / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    lines = [
        "# Accepted-field mechanics replay qualification",
        "",
        f"Status: `{audit['status']}`",
        "",
        "The online reference and both offline replays solve the same accepted",
        "phi/xB field before any PF, source-field, time, or checkpoint advance.",
        "The authority path is checkpoint-warm replay, matching historical V4",
        "production checkpoints. Zero initialization is an additional sensitivity",
        "test only and is never permitted for authority replay unless it independently",
        "passes every field gate.",
        "",
        "| Comparison | Energy rel. error | Max strain L2 | Max stress L2 | Source exact | Status |",
        "|---|---:|---:|---:|---|---|",
    ]
    for item in summaries:
        lines.append(
            "| {comparison} | {energy_relative_error:.3e} | "
            "{max_strain_normalized_l2:.3e} | {max_stress_normalized_l2:.3e} | "
            "{source_fields_byte_identical} | {status} |".format(**item)
        )
    if zero_summary["status"] != "PASS":
        lines.extend(
            [
                "",
                "The zero-initialized sensitivity did not pass the frozen field",
                "tolerances and is explicitly rejected. No tolerance was relaxed;",
                "all historical replays are therefore locked to checkpoint_warm.",
            ]
        )
    lines.extend(
        [
            "",
            "Acceptance gates: energy relative error <= 1e-6; normalized L2",
            "error <= 1e-5 for every strain and stress component; source fields",
            "must be byte-identical and all no-advance flags must be true.",
            "",
        ]
    )
    (args.out / "replay_determinism_audit.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    with (args.out / "replay_energy_closure.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=(
                "comparison",
                "energy_relative_error",
                "threshold",
                "status",
            ),
        )
        writer.writeheader()
        for item in summaries:
            writer.writerow(
                {
                    "comparison": item["comparison"],
                    "energy_relative_error": item["energy_relative_error"],
                    "threshold": 1.0e-6,
                    "status": (
                        "PASS"
                        if item["energy_relative_error"] <= 1.0e-6
                        else "FAIL"
                    ),
                }
            )
    (args.out / "status.txt").write_text(audit["status"] + "\n", encoding="utf-8")
    print(json.dumps(audit, indent=2, sort_keys=True))
    return 0 if overall_pass else 2


if __name__ == "__main__":
    raise SystemExit(main())
