#!/usr/bin/env python3
"""Extract the runtime elastic energy scalar already written by PF diagnostics."""

import argparse
import csv
import json
import math
from pathlib import Path


def finite(value):
    try:
        value = float(value)
        return math.isfinite(value)
    except (TypeError, ValueError):
        return False


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--diagnostics", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--age-start-h", type=float, default=6.0)
    parser.add_argument("--code-time-to-physical-s", type=float, default=49.54630476715921)
    parser.add_argument("--elastic-energy-scale-j-m3", type=float, default=None)
    args = parser.parse_args(); args.out_dir.mkdir(parents=True, exist_ok=True)
    with args.diagnostics.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    required = ["step", "time", "mean_elastic_energy", "max_elastic_energy", "stress_hydro_min", "stress_hydro_max"]
    missing = [key for key in required if key not in (rows[0] if rows else {})]
    if missing:
        raise SystemExit("missing runtime elastic columns: " + ",".join(missing))
    output = []
    for row in rows:
        if not finite(row["mean_elastic_energy"]):
            continue
        time_code = float(row["time"]); mean_hat = float(row["mean_elastic_energy"]); max_hat = float(row["max_elastic_energy"])
        item = {"step": int(float(row["step"])), "time_code": time_code, "age_h": args.age_start_h + time_code * args.code_time_to_physical_s / 3600.0, "mean_elastic_energy_hat": mean_hat, "max_elastic_energy_hat": max_hat, "stress_hydro_min": float(row["stress_hydro_min"]), "stress_hydro_max": float(row["stress_hydro_max"])}
        if args.elastic_energy_scale_j_m3 is not None:
            item["mean_elastic_energy_J_m^-3"] = mean_hat * args.elastic_energy_scale_j_m3
            item["max_elastic_energy_J_m^-3"] = max_hat * args.elastic_energy_scale_j_m3
        output.append(item)
    with (args.out_dir / "runtime_elastic_energy_time_series.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(output[0]), lineterminator="\n"); writer.writeheader(); writer.writerows(output)
    window = [row for row in output if 15.0 <= row["age_h"] <= 16.0]
    summary = {"schema": "PF_RUNTIME_ELASTIC_SCALAR_AUDIT_V1", "source_diagnostics": str(args.diagnostics), "row_count": len(output), "15_16h_row_count": len(window), "energy_unit": "gel_hat (dimensionless) unless an explicit physical scale is supplied", "mean_elastic_energy_hat_min": min(row["mean_elastic_energy_hat"] for row in output), "mean_elastic_energy_hat_max": max(row["mean_elastic_energy_hat"] for row in output), "window_15_16h": window}
    (args.out_dir / "runtime_elastic_energy_summary.json").write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    status = "PASS_RUNTIME_ELASTIC_SCALAR_EXTRACTED" if output else "BLOCKED_RUNTIME_ELASTIC_SCALAR_EMPTY"
    (args.out_dir / "status.txt").write_text(status + "\n", encoding="utf-8")
    (args.out_dir / "runtime_elastic_energy_report.md").write_text("# Runtime elastic-energy scalar audit\n\n`%s`\n\nThe scalar is taken from the solver's `mean_elastic_energy` and `max_elastic_energy` diagnostic columns. These are `gel_hat` values when `elastic_gel_is_dimless=1`; no zero is substituted. A physical J/m3 conversion is emitted only when the run's audited `12*gamma/lambda_sm` scale is supplied.\n\nRows in the 15--16 h physical window: %d.\n" % (status, len(window)), encoding="utf-8")
    print(status)


if __name__ == "__main__":
    main()
