#!/usr/bin/env python3
"""Compare PF elastic diagnostics and final fields across continuous/restart runs."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


DIAG_COLUMNS = (
    "time",
    "mean_elastic_energy",
    "max_elastic_energy",
    "stress_hydro_min",
    "stress_hydro_max",
    "mean_xBtot_end_step",
    "total_delta_mass_step",
)


def find_one(root, name):
    paths = list(root.rglob(name))
    if len(paths) != 1:
        raise RuntimeError("%s: expected one %s, found %d" % (root, name, len(paths)))
    return paths[0]


def read_diag(root):
    path = find_one(root, "dynamics_mass_diagnostics.csv")
    with path.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    return path, {int(row["step"]): row for row in rows}


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--continuous-root", type=Path, required=True)
    parser.add_argument("--first-half-root", type=Path, required=True)
    parser.add_argument("--restart-root", type=Path, required=True)
    parser.add_argument("--continuous-checkpoint", type=Path, required=True)
    parser.add_argument("--restart-checkpoint", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--final-step", type=int, default=512)
    args = parser.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    cont_path, continuous = read_diag(args.continuous_root)
    half_path, first_half = read_diag(args.first_half_root)
    restart_path, restarted_leg = read_diag(args.restart_root)
    combined = dict(first_half)
    overlap = set(combined) & set(restarted_leg)
    if overlap:
        raise RuntimeError("first-half/restart diagnostics overlap: %s" % sorted(overlap))
    combined.update(restarted_leg)
    if set(continuous) != set(combined):
        raise RuntimeError(
            "diagnostic step mismatch: continuous=%s combined=%s"
            % (sorted(continuous), sorted(combined))
        )
    comparison = []
    exact = True
    for step in sorted(continuous):
        row = {"step": step, "source_leg": "first_half" if step in first_half else "restart"}
        for column in DIAG_COLUMNS:
            left = continuous[step][column]
            right = combined[step][column]
            equal = left == right
            exact = exact and equal
            row["continuous_" + column] = left
            row["restarted_" + column] = right
            row[column + "_exact"] = int(equal)
            row[column + "_abs_diff"] = abs(float(left) - float(right))
        comparison.append(row)
    with (args.out_dir / "elastic_diag_comparison.csv").open(
        "w", newline="", encoding="utf-8"
    ) as stream:
        writer = csv.DictWriter(stream, fieldnames=list(comparison[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(comparison)
    field_results = {}
    fields_equal = True
    for stem in ("phi", "xB", "xBtot"):
        name = "%s_%d.vtk" % (stem, args.final_step)
        continuous_path = find_one(args.continuous_root, name)
        restart_field_path = find_one(args.restart_root, name)
        left_hash = sha256(continuous_path)
        right_hash = sha256(restart_field_path)
        equal = left_hash == right_hash and continuous_path.read_bytes() == restart_field_path.read_bytes()
        fields_equal = fields_equal and equal
        field_results[stem] = {
            "continuous_path": str(continuous_path),
            "restart_path": str(restart_field_path),
            "continuous_sha256": left_hash,
            "restart_sha256": right_hash,
            "bytewise_equal": equal,
        }
    checkpoint_left = sha256(args.continuous_checkpoint)
    checkpoint_right = sha256(args.restart_checkpoint)
    checkpoint_equal = (
        checkpoint_left == checkpoint_right
        and args.continuous_checkpoint.read_bytes() == args.restart_checkpoint.read_bytes()
    )
    status = (
        "PASS_PF_ELASTIC_DIAGNOSTIC_AND_FIELD_RESTART_V1"
        if exact and fields_equal and checkpoint_equal
        else "BLOCKED_PF_ELASTIC_DIAGNOSTIC_OR_FIELD_RESTART"
    )
    summary = {
        "schema": "PF_ELASTIC_DIAGNOSTIC_RESTART_AUDIT_V1",
        "status": status,
        "continuous_diagnostics": str(cont_path),
        "first_half_diagnostics": str(half_path),
        "restart_diagnostics": str(restart_path),
        "diagnostic_steps": sorted(continuous),
        "diagnostic_columns": list(DIAG_COLUMNS),
        "diagnostic_trajectory_string_exact": exact,
        "final_fields": field_results,
        "final_fields_bytewise_equal": fields_equal,
        "continuous_checkpoint_sha256": checkpoint_left,
        "restart_checkpoint_sha256": checkpoint_right,
        "checkpoint_bytewise_equal": checkpoint_equal,
    }
    (args.out_dir / "restart_validation.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )
    (args.out_dir / "status.txt").write_text(status + "\n", encoding="utf-8")
    report = """# PF elastic diagnostic restart audit

`%s`

The continuous diagnostic trajectory is compared to the concatenation of the
first-half and restarted legs at every registered absolute step.  Elastic
mean/max, hydrostatic-stress extrema, mass diagnostics, final phi/xB/xBtot VTK
fields and the complete final checkpoint are required to match exactly.
""" % status
    (args.out_dir / "restart_validation.md").write_text(report, encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
