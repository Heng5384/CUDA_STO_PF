#!/usr/bin/env python3
"""Run selected storage-exact PF-only acceptance cases on one workstation GPU."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PARAM = ROOT / "params/pf_only_baseline_closure/remediation"
REPORT = ROOT / "reports/pf_only_baseline_closure"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--temperature", type=int, action="append")
    parser.add_argument("--dt", type=float, action="append")
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    cases = list(csv.DictReader((PARAM / "acceptance_manifest.csv").open()))
    if args.temperature:
        cases = [r for r in cases if int(r["T_C"]) in args.temperature]
    if args.dt:
        cases = [r for r in cases if any(abs(float(r["dt"])-v) < 1e-15 for v in args.dt)]
    status_path = REPORT / "storage_exact_workstation_status.csv"
    existing = list(csv.DictReader(status_path.open())) if status_path.exists() else []
    (REPORT / "logs").mkdir(parents=True, exist_ok=True)
    for row in cases:
        old = next((r for r in existing if r["case"] == row["case"] and r["returncode"] == "0"), None)
        if old and not args.rerun:
            print(f"SKIP {row['case']}", flush=True)
            continue
        log = REPORT / "logs" / f"{row['case']}.log"
        cmd = [str(ROOT / "main_cuda"), "--Nx", "128", "--Ny", "128", "--Nz", "128",
               "--pf-param-file", row["param_file"], "--nsteps", row["nsteps"],
               "--out-every", row["nsteps"], "--csv-out-every", "10",
               "--elastic", "1", "--init-case-tag", row["case"]]
        print(f"RUN {row['case']}", flush=True)
        started = time.time()
        with log.open("w") as handle:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        result = {"case": row["case"], "T_C": row["T_C"], "dt": row["dt"],
                  "nsteps": row["nsteps"], "returncode": str(proc.returncode),
                  "wall_time_s": f"{time.time()-started:.6f}", "log": str(log)}
        existing = [r for r in existing if r["case"] != row["case"]] + [result]
        with status_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result))
            writer.writeheader(); writer.writerows(existing)
        print(f"DONE {result}", flush=True)
        if proc.returncode:
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
