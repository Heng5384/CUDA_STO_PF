#!/usr/bin/env python3
"""Extract the auditable tail of the historical cumulative-step-1902 failure."""

from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


FAILURE_RE = re.compile(
    r"CTOT_PHASE_INNER physical_step=(?P<step>\d+).*?iterations=(?P<iterations>\d+) "
    r"initial_KKT=(?P<initial>\S+) final_KKT=(?P<final>\S+).*?converged=(?P<converged>[01])"
)
FAILURE_CELL_RE = re.compile(
    r"CTOT_PHASE_INNER_MAX_KKT index=(?P<index>\d+) i=(?P<i>\d+) j=(?P<j>\d+) "
    r"k=(?P<k>\d+) phi_n=(?P<phi_n>\S+) phi=(?P<phi>\S+) C=(?P<C>\S+) "
    r"x=(?P<x>\S+).*?raw=(?P<raw>\S+) KKT=(?P<KKT>\S+)"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as stream:
        return list(csv.DictReader(stream))


def finite_or_na(value: str | None) -> str:
    if value is None:
        return "NA_NOT_LOGGED"
    try:
        number = float(value)
    except ValueError:
        return value
    return value if math.isfinite(number) else "NA_FAILED_STAGE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--first-step", type=int, default=852)
    parser.add_argument("--last-step", type=int, default=902)
    args = parser.parse_args()

    outer_rows = read_csv(args.source / "ctot_outer_iterations.csv")
    retry_rows = read_csv(args.source / "ctot_retry_attempts.csv")
    energy_rows = read_csv(args.source / "ctot_energy_work.csv")

    outer_by_step: dict[int, dict[str, str]] = {}
    for row in outer_rows:
        step = int(row["physical_step"])
        if args.first_step <= step <= args.last_step:
            previous = outer_by_step.get(step)
            if previous is None or int(row["outer_iter"]) >= int(previous["outer_iter"]):
                outer_by_step[step] = row

    retry_by_step = {
        int(row["physical_step_id"]): row
        for row in retry_rows
        if args.first_step <= int(row["physical_step_id"]) <= args.last_step
    }
    energy_by_step = {
        int(row["physical_step_id"]): row
        for row in energy_rows
        if args.first_step <= int(row["physical_step_id"]) <= args.last_step
    }

    failure_phase: dict[str, str] = {}
    failure_cell: dict[str, str] = {}
    for line in args.run_log.read_text(errors="replace").splitlines():
        match = FAILURE_RE.search(line)
        if match and int(match.group("step")) == args.last_step:
            failure_phase = match.groupdict()
        match = FAILURE_CELL_RE.search(line)
        if match:
            failure_cell = match.groupdict()

    fieldnames = [
        "physical_step",
        "accepted",
        "accepted_time_before",
        "accepted_time_after",
        "dt_requested",
        "dt_try",
        "outer_iter_last",
        "transport_solve_residual",
        "final_coupled_transport_residual",
        "phase_solve_residual",
        "final_coupled_phase_residual",
        "phase_KKT_residual",
        "phase_inner_iterations",
        "phase_initial_KKT",
        "mechanical_residual",
        "mass_residual",
        "local_phase_storage_residual",
        "D_transport_base",
        "W_finite_interface",
        "energy_balance_residual",
        "phi_half_radius_nm",
        "h_volume_radius_nm",
        "instantaneous_R_over_lambda",
        "minimum_q_alpha",
        "minimum_matrix_capacity",
        "lower_KKT_active_cell_count",
        "upper_KKT_active_cell_count",
        "base_divergence",
        "finite_interface_divergence",
        "chemical_potential_jump",
        "failure_stage",
        "failure_reason_logged",
        "exact_failure_operator",
        "max_kkt_cell_index",
        "max_kkt_phi_n",
        "max_kkt_phi",
        "max_kkt_Ctot",
        "max_kkt_xB_alpha_context",
        "max_kkt_raw_phase_residual",
        "evidence_limit",
    ]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        for step in range(args.first_step, args.last_step + 1):
            retry = retry_by_step.get(step, {})
            outer = outer_by_step.get(step, {})
            energy = energy_by_step.get(step, {})
            accepted = retry.get("accepted", "0")
            failed = accepted != "1"
            writer.writerow(
                {
                    "physical_step": step,
                    "accepted": accepted,
                    "accepted_time_before": retry.get("accepted_time_before", "NA_NOT_LOGGED"),
                    "accepted_time_after": retry.get("accepted_time_after", "NA_NOT_LOGGED"),
                    "dt_requested": retry.get("dt_requested", "NA_NOT_LOGGED"),
                    "dt_try": retry.get("dt_try", "NA_NOT_LOGGED"),
                    "outer_iter_last": outer.get("outer_iter", "NA_PHASE_FAILED_BEFORE_OUTER_ROW") if failed else outer.get("outer_iter", "NA_NOT_LOGGED"),
                    "transport_solve_residual": finite_or_na(outer.get("transport_solve_residual")),
                    "final_coupled_transport_residual": finite_or_na(outer.get("final_coupled_transport_residual")),
                    "phase_solve_residual": finite_or_na(outer.get("phase_solve_residual")),
                    "final_coupled_phase_residual": finite_or_na(outer.get("final_coupled_phase_residual")),
                    "phase_KKT_residual": failure_phase.get("final", "NA_NOT_LOGGED") if failed else finite_or_na(outer.get("phase_KKT_residual")),
                    "phase_inner_iterations": failure_phase.get("iterations", "NA_NOT_LOGGED") if failed else "3",
                    "phase_initial_KKT": failure_phase.get("initial", "NA_NOT_LOGGED") if failed else "NA_NOT_LOGGED",
                    "mechanical_residual": finite_or_na(outer.get("mechanical_residual")),
                    "mass_residual": finite_or_na(outer.get("mass_residual")),
                    "local_phase_storage_residual": finite_or_na(outer.get("local_phase_storage_residual")),
                    "D_transport_base": finite_or_na(energy.get("D_transport")),
                    "W_finite_interface": finite_or_na(energy.get("W_finite_interface")),
                    "energy_balance_residual": finite_or_na(energy.get("energy_balance_residual")),
                    "phi_half_radius_nm": "NA_NOT_LOGGED",
                    "h_volume_radius_nm": "NA_NOT_LOGGED",
                    "instantaneous_R_over_lambda": "NA_NOT_LOGGED",
                    "minimum_q_alpha": "NA_NOT_LOGGED",
                    "minimum_matrix_capacity": "NA_NOT_LOGGED",
                    "lower_KKT_active_cell_count": "0" if failed else "NA_NOT_LOGGED",
                    "upper_KKT_active_cell_count": "0" if failed else "NA_NOT_LOGGED",
                    "base_divergence": "NA_NOT_LOGGED",
                    "finite_interface_divergence": "NA_NOT_LOGGED",
                    "chemical_potential_jump": "NA_NOT_LOGGED",
                    "failure_stage": retry.get("failure_stage", "NA_NOT_LOGGED"),
                    "failure_reason_logged": retry.get("failure_reason", "NA_NOT_LOGGED"),
                    "exact_failure_operator": "phase_inner_KKT_nonconvergence" if failed else "none",
                    "max_kkt_cell_index": failure_cell.get("index", "") if failed else "",
                    "max_kkt_phi_n": failure_cell.get("phi_n", "") if failed else "",
                    "max_kkt_phi": failure_cell.get("phi", "") if failed else "",
                    "max_kkt_Ctot": failure_cell.get("C", "") if failed else "",
                    "max_kkt_xB_alpha_context": failure_cell.get("x", "") if failed else "",
                    "max_kkt_raw_phase_residual": failure_cell.get("raw", "") if failed else "",
                    "evidence_limit": "radius_q_divergence_jump_not_emitted_by_historical_run",
                }
            )


if __name__ == "__main__":
    main()
