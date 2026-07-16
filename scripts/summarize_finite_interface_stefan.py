#!/usr/bin/env python3
"""Summarize final eligible windows without fitting physical parameters."""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


def read_final_rows(paths: list[Path]) -> list[dict[str, str]]:
    final: dict[tuple[str, str, str, str], dict[str, str]] = {}
    for path in paths:
        with path.open(newline="") as handle:
            for row in csv.DictReader(handle):
                key = (
                    row["case"], row["dx_nm"], row["L_phi_factor"],
                    str(row.get("finite_interface_correction", "0")),
                )
                previous = final.get(key)
                if previous is None or int(row["new_step"]) > int(previous["new_step"]):
                    item = dict(row)
                    item["source_csv"] = str(path)
                    final[key] = item
    return sorted(final.values(), key=lambda row: (
        float(row["dx_nm"]), float(row["L_phi_factor"]), row["case"]
    ))


def summarize(rows: list[dict[str, str]]) -> list[dict[str, object]]:
    output: list[dict[str, object]] = []
    for row in rows:
        ratio = float(row["phi_velocity_ratio_to_sharp"])
        completed = int(row["accepted_steps"]) == int(row["requested_steps"])
        eligible = completed and row["equal_requested_dt"].lower() == "true"
        output.append({
            "case": row["case"],
            "source_csv": row["source_csv"],
            "dx_nm": row["dx_nm"],
            "interface_resolution": row["interface_resolution"],
            "L_phi_factor": row["L_phi_factor"],
            "finite_interface_correction": row["finite_interface_correction"],
            "old_step": row["old_step"],
            "new_step": row["new_step"],
            "accepted_steps": row["accepted_steps"],
            "requested_steps": row["requested_steps"],
            "retry_count": row["retry_count"],
            "eligible_equal_dt_complete": eligible,
            "phi_velocity_ratio_to_sharp": ratio,
            "sharp_velocity_error_rel": abs(ratio - 1.0),
            "local_stefan_ratio_phi_over_flux":
                row["local_stefan_ratio_phi_over_flux"],
            "mu_surface_new_excess": row["mu_surface_new_excess"],
            "runtime_mass_error_max": row["runtime_mass_error_max"],
            "energy_balance_rel_max": row["energy_balance_rel_max"],
            "doubling_change_from_previous": math.nan,
            "provenance":
                "finite_interface_asymptotic_audit_not_solver_compensation",
        })
    groups: dict[tuple[float, int], list[dict[str, object]]] = {}
    for row in output:
        groups.setdefault((float(row["dx_nm"]),
                           int(row["finite_interface_correction"])), []).append(row)
    for group in groups.values():
        eligible = sorted(
            (row for row in group if row["eligible_equal_dt_complete"]),
            key=lambda row: float(row["L_phi_factor"]),
        )
        for previous, current in zip(eligible, eligible[1:]):
            if math.isclose(float(current["L_phi_factor"]),
                            2.0 * float(previous["L_phi_factor"])):
                current["doubling_change_from_previous"] = abs(
                    float(current["phi_velocity_ratio_to_sharp"]) -
                    float(previous["phi_velocity_ratio_to_sharp"])
                )
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows = summarize(read_final_rows(args.input))
    if not rows:
        raise SystemExit("no rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"summary_rows={len(rows)}")
    print(f"eligible_rows={sum(bool(row['eligible_equal_dt_complete']) for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
