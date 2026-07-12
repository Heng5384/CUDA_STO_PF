#!/usr/bin/env python3
"""Run selected workstation-only RSMD validation cases with an xB guard."""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path
import shlex
import subprocess
import time


ROOT = Path(__file__).resolve().parents[1]
PARAM_ROOT = ROOT / "params" / "rsmd_post_history_sync_validation"


def load_manifest() -> list[dict[str, str]]:
    with (PARAM_ROOT / "run_manifest.csv").open(newline="") as handle:
        return list(csv.DictReader(handle))


def result_dir(row: dict[str, str]) -> Path:
    dt = float(row["dt"])
    steps = int(row["nsteps"])
    T = int(row["T_C"])
    parent = ROOT / "Results" / f"chel_T{T}_cuda_128x128x128_dt{dt:g}_steps{steps}_xB0.008"
    return parent / row["case"]


def max_projection_xb(path: Path) -> float:
    if not path.exists():
        return float("-inf")
    try:
        with path.open(newline="") as handle:
            values = [float(r["max_xB_before_projection"]) for r in csv.DictReader(handle)
                      if r.get("max_xB_before_projection")]
        return max(values, default=float("-inf"))
    except (OSError, ValueError, KeyError):
        return float("-inf")


def run(row: dict[str, str]) -> dict[str, str]:
    out = result_dir(row)
    out.parent.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "reports" / "rsmd_post_history_sync_validation" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / f"{row['case']}.log"
    dt = float(row["dt"])
    diag_every = max(1, round(0.05 / dt))
    param = PARAM_ROOT / Path(row["param_file"]).name
    cmd = [
        str(ROOT / "main_cuda"), "--Nx", "128", "--Ny", "128", "--Nz", "128",
        "--pf-param-file", str(param), "--nsteps", row["nsteps"],
        "--out-every", row["nsteps"], "--csv-out-every", str(diag_every),
        "--elastic", "1", "--init-case-tag", row["case"],
    ] + shlex.split(row.get("extra_args", ""))
    started = time.time()
    guard = "not_triggered"
    with log_path.open("w") as log:
        proc = subprocess.Popen(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT,
                                env={**os.environ, "CUDA_VISIBLE_DEVICES": "0"})
        while proc.poll() is None:
            time.sleep(2.0)
            observed = max_projection_xb(out / "y_update_mass_projection.csv")
            if observed >= 0.99:
                guard = f"terminated_max_xB_{observed:.12g}"
                proc.terminate()
                try:
                    proc.wait(timeout=15)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    proc.wait()
                break
    return {
        "case": row["case"], "phase": row["phase"], "history_mode": row["history_mode"],
        "returncode": str(proc.returncode), "guard_status": guard,
        "wall_time_s": f"{time.time() - started:.6f}", "output_dir": str(out),
        "log": str(log_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", choices=["history_scheme", "dt_convergence"])
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()
    selected = [r for r in load_manifest()
                if (not args.phase or r["phase"] == args.phase)
                and (not args.case or r["case"] in args.case)]
    status_path = ROOT / "reports" / "rsmd_post_history_sync_validation" / "workstation_run_status.csv"
    existing: list[dict[str, str]] = []
    if status_path.exists():
        with status_path.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
    for row in selected:
        print(f"RUN {row['case']}", flush=True)
        result = run(row)
        existing = [old for old in existing if old.get("case") != result["case"]]
        existing.append(result)
        status_path.parent.mkdir(parents=True, exist_ok=True)
        with status_path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(result))
            writer.writeheader()
            writer.writerows(existing)
        print(f"DONE {result}", flush=True)


if __name__ == "__main__":
    main()
