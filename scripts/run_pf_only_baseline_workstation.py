#!/usr/bin/env python3
"""Execute prepared PF-only operator controls on one workstation GPU."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PARAM_ROOT = ROOT / "params/pf_only_baseline_closure"
REPORT = ROOT / "reports/pf_only_baseline_closure"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", default=[])
    parser.add_argument("--rerun", action="store_true")
    args = parser.parse_args()
    rows = list(csv.DictReader((PARAM_ROOT / "operator_manifest.csv").open()))
    rows = [row for row in rows if row["param_file"] != "SHARED_WITH_P1" and
            row["param_file"] != "SHARED_WITH_P6" and
            (not args.case or row["operator_case"] in args.case or row["case"] in args.case)]
    REPORT.mkdir(parents=True, exist_ok=True)
    (REPORT / "logs").mkdir(exist_ok=True)
    status_path = REPORT / "workstation_run_status.csv"
    existing = list(csv.DictReader(status_path.open())) if status_path.exists() else []
    for row in rows:
        old = next((item for item in existing if item["case"] == row["case"] and
                    item["returncode"] == "0"), None)
        if old and not args.rerun:
            print(f"SKIP {row['case']}", flush=True)
            continue
        log = REPORT / "logs" / f"{row['case']}.log"
        output = (ROOT / f"Results/chel_T400_cuda_128x128x128_dt0.002_steps500_xB0.008/{row['case']}")
        param_file = PARAM_ROOT / Path(row["param_file"]).name
        cmd = [str(ROOT / "main_cuda"), "--Nx", "128", "--Ny", "128", "--Nz", "128",
               "--pf-param-file", str(param_file), "--nsteps", row["nsteps"],
               "--out-every", row["nsteps"], "--csv-out-every", "10",
               "--elastic", "1", "--init-case-tag", row["case"]]
        print(f"RUN {row['case']}", flush=True)
        started = time.time()
        with log.open("w") as handle:
            proc = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT,
                                  env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        result = {"case": row["case"], "operator_case": row["operator_case"],
                  "returncode": str(proc.returncode), "wall_time_s": f"{time.time()-started:.6f}",
                  "output_dir": str(output), "log": str(log)}
        existing = [item for item in existing if item["case"] != row["case"]] + [result]
        with status_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result))
            writer.writeheader(); writer.writerows(existing)
        print(f"DONE {result}", flush=True)
        if proc.returncode:
            raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
