#!/usr/bin/env python3
"""Run the frozen KWN positivity diagnosis over the required numerical matrix."""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUTPUT_ROOT = ROOT / "outputs" / "kwn_pf_cuda_runtime_closure_v1"
DIAGNOSTIC = ROOT / "scripts" / "diagnose_kwn_positivity_failure.py"
BIN_COUNTS = (100, 200, 400)
DT_FACTORS = (1.0, 0.5, 0.25, 0.125)


def _label(value: float) -> str:
    return format(value, ".3g").replace(".", "p")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--end-time-h", type=float, default=0.5)
    parser.add_argument("--max-accepted-steps", type=int, default=5000)
    args = parser.parse_args()
    root = args.output_root.resolve()
    try:
        root.relative_to(OUTPUT_ROOT.resolve())
    except ValueError as error:
        raise SystemExit("refusing output outside this task's output root") from error
    matrix_root = root / "kwn_positivity_diagnosis_matrix"
    matrix_root.mkdir(parents=True, exist_ok=True)
    summary_path = matrix_root / "kwn_positivity_diagnosis_matrix.csv"
    if summary_path.exists():
        raise SystemExit(f"refusing to overwrite {summary_path}")
    rows: list[dict[str, object]] = []
    for bins in BIN_COUNTS:
        for factor in DT_FACTORS:
            case_root = matrix_root / f"bins_{bins}" / f"max_dt_factor_{_label(factor)}"
            completed = subprocess.run(
                [
                    sys.executable,
                    str(DIAGNOSTIC),
                    "--output-root",
                    str(case_root),
                    "--bins",
                    str(bins),
                    "--max-dt-factor",
                    format(factor, ".17g"),
                    "--end-time-h",
                    format(args.end_time_h, ".17g"),
                    "--max-accepted-steps",
                    str(args.max_accepted_steps),
                ],
                cwd=ROOT,
                text=True,
                capture_output=True,
                check=False,
            )
            if completed.returncode != 0:
                raise RuntimeError(
                    f"diagnostic matrix case bins={bins} factor={factor} failed:\n"
                    f"stdout:\n{completed.stdout}\nstderr:\n{completed.stderr}"
                )
            metadata_path = case_root / "kwn_positivity_failure_diagnosis.json"
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            failure = metadata["failure"]
            beta = None if failure is None else failure["populations"]["beta"]
            rows.append(
                {
                    "radius_bins": bins,
                    "max_dt_factor": factor,
                    "status": metadata["status"],
                    "failure_time_h": "" if failure is None else failure["time_h"],
                    "failure_step": "" if failure is None else failure["step"],
                    "failure_beta_bin": "" if beta is None else beta["first_candidate_negative_bin"],
                    "minimum_candidate_density_per_m4": (
                        "" if beta is None else beta["minimum_candidate_density_per_m4"]
                    ),
                    "candidate_dt_s": metadata["classification_evidence"]["candidate_dt_s"],
                    "donor_dt_max_s": (
                        "" if beta is None else beta["first_candidate_negative_dt_max_s"]
                    ),
                    "inventory_relative_residual": metadata["completion"]["inventory_relative_residual"],
                    "completion_time_h": metadata["completion"]["time_h"],
                    "completion_steps": metadata["completion"]["accepted_steps"],
                    "reached_time_ceiling": metadata["completion"]["reached_time_ceiling"],
                    "reached_step_ceiling": metadata["completion"]["reached_step_ceiling"],
                    "trace": str(case_root / "kwn_positivity_failure_trace.csv"),
                    "snapshot": str(case_root / "kwn_positivity_failure_snapshot.npz"),
                }
            )
    with summary_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    print(json.dumps({"status": "PASS_KWN_POSITIVITY_DIAGNOSIS_MATRIX", "csv": str(summary_path)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
