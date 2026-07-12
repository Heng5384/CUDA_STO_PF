#!/usr/bin/env python3
"""Run the prepared RSMD validation matrix on one workstation GPU."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PARAM_ROOT = ROOT / "params" / "rsmd_equal_time_domain_low_overshoot"
REPORT_ROOT = ROOT / "reports" / "rsmd_equal_time_domain_low_overshoot"


def load_manifest() -> list[dict[str, str]]:
    with (PARAM_ROOT / "run_manifest.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def result_dir(row: dict[str, str]) -> Path:
    dt = float(row["dt"])
    steps = int(row["nsteps"])
    T = int(row["T_C"])
    parent = ROOT / "Results" / f"chel_T{T}_cuda_128x128x128_dt{dt:g}_steps{steps}_xB0.008"
    return parent / row["case"]


def run_case(row: dict[str, str]) -> dict[str, str]:
    out = result_dir(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    logs = REPORT_ROOT / "logs"
    logs.mkdir(parents=True, exist_ok=True)
    log_path = logs / f"{row['case']}.log"
    param = PARAM_ROOT / Path(row["param_file"]).name
    diag_every = max(1, round(0.05 / float(row["dt"])))
    cmd = [
        str(ROOT / "main_cuda"), "--Nx", "128", "--Ny", "128", "--Nz", "128",
        "--pf-param-file", str(param), "--nsteps", row["nsteps"],
        "--out-every", row["nsteps"], "--csv-out-every", str(diag_every),
        "--elastic", "1",
        "--init-case-tag", row["case"],
    ]
    started = time.time()
    with log_path.open("w") as log:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                              env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
    return {
        "case": row["case"], "phase": row["phase"], "scheme": row["scheme"],
        "returncode": str(proc.returncode), "wall_time_s": f"{time.time()-started:.6f}",
        "output_dir": str(out), "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", action="append", default=[])
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    rows = [r for r in load_manifest()
            if (not args.phase or r["phase"] in args.phase)
            and (not args.case or r["case"] in args.case)]
    REPORT_ROOT.mkdir(parents=True, exist_ok=True)
    status_path = REPORT_ROOT / "workstation_run_status.csv"
    existing: list[dict[str, str]] = []
    if status_path.exists():
        with status_path.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
    for row in rows:
        completed = next((old for old in existing
                          if old.get("case") == row["case"] and old.get("returncode") == "0"), None)
        if completed and not args.rerun:
            print(f"SKIP completed {row['case']}", flush=True)
            continue
        print(f"RUN {row['case']}", flush=True)
        result = run_case(row)
        existing = [old for old in existing if old.get("case") != result["case"]]
        existing.append(result)
        with status_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result))
            writer.writeheader()
            writer.writerows(existing)
        print(f"DONE {result}", flush=True)
        if result["returncode"] != "0":
            raise SystemExit(f"case failed: {result['case']}")


if __name__ == "__main__":
    main()
