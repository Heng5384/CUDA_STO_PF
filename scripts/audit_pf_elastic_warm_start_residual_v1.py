#!/usr/bin/env python3
"""Qualify elastic warm-start/residual control against the fixed-iteration path."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict

import numpy as np


PASS = "PASS_ELASTIC_WARM_START_RESIDUAL_V1"
FAIL = "BLOCKED_ELASTIC_WARM_START_RESIDUAL_V1"
NESTED_PASS = "PASS_246CUBE_SHORT_RESTART_AND_OBSERVABLES_V1"


def last_csv_row(path: Path) -> Dict[str, str]:
    with path.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"empty CSV: {path}")
    return rows[-1]


def relative(value: float, reference: float) -> float:
    return abs(value - reference) / max(abs(reference), 1.0e-30)


def load_checkpoint_reader(source_root: Path):
    sys.path.insert(0, str(source_root / "scripts"))
    import analyze_pf_zero_mode_checkpoints as checkpoint_reader

    return checkpoint_reader


def checkpoint_fields(reader, path: Path):
    step, shape, dt, temperature, target, phi, y, xb, dydt = (
        reader.read_checkpoint(path)
    )
    return {
        "step": step,
        "shape": shape,
        "dt": dt,
        "temperature": temperature,
        "target": target,
        "phi": phi,
        "Y": y,
        "xB": xb,
        "dYdt": dydt,
    }


def trace_rows(root: Path):
    candidates = list(
        (root / "continuous" / "results").rglob("elastic_solver_trace.csv")
    )
    if len(candidates) != 1:
        raise ValueError(
            f"expected exactly one continuous elastic trace, found {candidates}"
        )
    with candidates[0].open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError("accelerated elastic trace is empty")
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--baseline-root", type=Path, required=True)
    parser.add_argument("--accelerated-root", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)

    try:
        baseline_status = (
            args.baseline_root / "status.txt"
        ).read_text(encoding="utf-8").strip()
        accelerated_status = (
            args.accelerated_root / "status.txt"
        ).read_text(encoding="utf-8").strip()
        if baseline_status != NESTED_PASS or accelerated_status != NESTED_PASS:
            raise ValueError(
                "baseline or accelerated short-restart qualification failed"
            )

        reader = load_checkpoint_reader(args.source_root)
        baseline_checkpoint_path = (
            args.baseline_root / "continuous" / "final.chk"
        )
        accelerated_checkpoint_path = (
            args.accelerated_root / "continuous" / "final.chk"
        )
        baseline = checkpoint_fields(reader, baseline_checkpoint_path)
        accelerated = checkpoint_fields(reader, accelerated_checkpoint_path)
        if (
            baseline["step"] != accelerated["step"]
            or baseline["shape"] != accelerated["shape"]
            or baseline["dt"] != accelerated["dt"]
            or baseline["temperature"] != accelerated["temperature"]
            or baseline["target"] != accelerated["target"]
        ):
            raise ValueError("baseline/accelerated endpoint contract mismatch")

        baseline_observables = last_csv_row(
            args.baseline_root / "lineage" / "ensemble_observables.csv"
        )
        accelerated_observables = last_csv_row(
            args.accelerated_root / "lineage" / "ensemble_observables.csv"
        )
        baseline_mass = last_csv_row(
            args.baseline_root / "continuous" /
            "dynamics_mass_diagnostics.csv"
        )
        accelerated_mass = last_csv_row(
            args.accelerated_root / "continuous" /
            "dynamics_mass_diagnostics.csv"
        )
        baseline_audit = json.loads(
            (args.baseline_root / "audit" / "audit.json").read_text(
                encoding="utf-8"
            )
        )
        accelerated_audit = json.loads(
            (args.accelerated_root / "audit" / "audit.json").read_text(
                encoding="utf-8"
            )
        )
        baseline_wall = float(
            baseline_audit["metrics"]["continuous_wall_seconds"]
        )
        accelerated_wall = float(
            accelerated_audit["metrics"]["continuous_wall_seconds"]
        )
        speedup = baseline_wall / accelerated_wall

        phi_scale = max(
            float(np.sum(np.abs(baseline["phi"]), dtype=np.float64)), 1.0
        )
        field_differences = {
            "phi_normalized_L1": float(
                np.sum(
                    np.abs(accelerated["phi"] - baseline["phi"]),
                    dtype=np.float64,
                )
                / phi_scale
            ),
            "Y_mean_absolute": float(
                np.mean(
                    np.abs(accelerated["Y"] - baseline["Y"]),
                    dtype=np.float64,
                )
            ),
            "xB_mean_absolute": float(
                np.mean(
                    np.abs(accelerated["xB"] - baseline["xB"]),
                    dtype=np.float64,
                )
            ),
            "dYdt_mean_absolute": float(
                np.mean(
                    np.abs(accelerated["dYdt"] - baseline["dYdt"]),
                    dtype=np.float64,
                )
            ),
        }

        observable_differences = {
            "beta_volume_fraction_relative": relative(
                float(accelerated_observables["beta_volume_fraction"]),
                float(baseline_observables["beta_volume_fraction"]),
            ),
            "mean_radius_relative": relative(
                float(accelerated_observables["mean_radius_nm"]),
                float(baseline_observables["mean_radius_nm"]),
            ),
            "Sv_relative": relative(
                float(accelerated_observables["Sv_nm_inv"]),
                float(baseline_observables["Sv_nm_inv"]),
            ),
            "M6_relative": relative(
                float(accelerated_observables["M6_nm3"]),
                float(baseline_observables["M6_nm3"]),
            ),
            "matrix_xAg_absolute": abs(
                float(accelerated_observables["far_field_matrix_xAg"])
                - float(baseline_observables["far_field_matrix_xAg"])
            ),
            "mean_elastic_energy_relative": relative(
                float(accelerated_mass["mean_elastic_energy"]),
                float(baseline_mass["mean_elastic_energy"]),
            ),
        }

        rows = trace_rows(args.accelerated_root)
        trace_iterations = [int(row["iterations"]) for row in rows]
        trace_residuals = [float(row["relative_residual"]) for row in rows]
        trace_tolerances = [float(row["tolerance"]) for row in rows]
        trace_converged = [int(row["converged"]) for row in rows]
        trace_warm = [int(row["warm_start_used"]) for row in rows]
        accelerated_stdout = (
            args.accelerated_root / "continuous" / "stdout.log"
        ).read_text(encoding="utf-8", errors="replace")
        v4_magic = accelerated_checkpoint_path.read_bytes()[:8] == b"PFZMCHK4"

        gates = {
            "nested_baseline_pass": baseline_status == NESTED_PASS,
            "nested_accelerated_pass": accelerated_status == NESTED_PASS,
            "accelerated_checkpoint_v4": v4_magic,
            "accelerated_restart_bytewise": bool(
                accelerated_audit["metrics"]["restart_bytewise_equal"]
            ),
            "elastic_provenance": (
                "elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1"
                in accelerated_stdout
                and "checkpoint_state=V4" in accelerated_stdout
            ),
            "trace_complete": len(rows) == int(accelerated["step"]),
            "trace_converged": all(value == 1 for value in trace_converged),
            "trace_time_level": (
                trace_warm[0] == 0
                and all(value == 1 for value in trace_warm[1:])
            ),
            "trace_residual": all(
                math.isfinite(value)
                and value <= tolerance
                for value, tolerance in zip(
                    trace_residuals, trace_tolerances
                )
            ),
            "iteration_reduction": (
                sum(trace_iterations) / len(trace_iterations) < 19.0
            ),
            "performance_improved": speedup > 1.0,
            "particle_identity": (
                int(baseline_observables["particle_count"])
                == int(accelerated_observables["particle_count"])
                == 96
            ),
            "phi_field": field_differences["phi_normalized_L1"] <= 2.0e-2,
            "beta_volume_fraction": (
                observable_differences[
                    "beta_volume_fraction_relative"
                ]
                <= 1.0e-2
            ),
            "mean_radius": (
                observable_differences["mean_radius_relative"] <= 1.0e-2
            ),
            "Sv": observable_differences["Sv_relative"] <= 2.0e-2,
            "M6": observable_differences["M6_relative"] <= 5.0e-2,
            "matrix_xAg": (
                observable_differences["matrix_xAg_absolute"] <= 4.0e-4
            ),
            "mean_elastic_energy": (
                observable_differences[
                    "mean_elastic_energy_relative"
                ]
                <= 5.0e-2
            ),
            "mass_tier": (
                float(accelerated_observables["mass_relative_error"])
                <= 1.0e-10
            ),
        }
        status = PASS if all(gates.values()) else FAIL
        report = {
            "schema": "PF_ELASTIC_WARM_START_RESIDUAL_AUDIT_V1",
            "status": status,
            "gates": gates,
            "configuration": {
                "warm_start": True,
                "residual_control": True,
                "minimum_iterations": min(trace_iterations),
                "hard_cap": max(int(row["hard_cap"]) for row in rows),
                "residual_tolerance": trace_tolerances[0],
            },
            "performance": {
                "baseline_wall_seconds": baseline_wall,
                "accelerated_wall_seconds": accelerated_wall,
                "speedup": speedup,
                "baseline_seconds_per_step": (
                    baseline_wall / int(baseline["step"])
                ),
                "accelerated_seconds_per_step": (
                    accelerated_wall / int(accelerated["step"])
                ),
                "mean_elastic_iterations": (
                    sum(trace_iterations) / len(trace_iterations)
                ),
                "maximum_elastic_iterations": max(trace_iterations),
            },
            "field_differences": field_differences,
            "observable_differences": observable_differences,
        }
        (args.out / "audit.json").write_text(
            json.dumps(report, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        (args.out / "status.txt").write_text(
            status + "\n", encoding="utf-8"
        )
        (args.out / "final_terminal_output.txt").write_text(
            "\n".join(
                (
                    f"status={status}",
                    "solver=ELASTIC_WARM_START_RESIDUAL_V1",
                    f"mean_elastic_iterations={report['performance']['mean_elastic_iterations']:.9f}",
                    f"baseline_seconds_per_step={report['performance']['baseline_seconds_per_step']:.9f}",
                    f"accelerated_seconds_per_step={report['performance']['accelerated_seconds_per_step']:.9f}",
                    f"speedup={speedup:.9f}",
                    "restart_status=PASS_BYTEWISE"
                    if gates["accelerated_restart_bytewise"]
                    else "restart_status=FAIL",
                    "particle_identity_status=PASS"
                    if gates["particle_identity"]
                    else "particle_identity_status=FAIL",
                    "GP_enabled=false",
                    "physical_parameter_retuning=false",
                )
            )
            + "\n",
            encoding="utf-8",
        )
        if status != PASS:
            failed = [key for key, value in gates.items() if not value]
            raise ValueError(f"failed gates: {failed}")
    except (KeyError, OSError, ValueError) as exc:
        if not (args.out / "status.txt").exists():
            (args.out / "status.txt").write_text(
                f"{FAIL}\n{exc}\n", encoding="utf-8"
            )
        raise SystemExit(f"[fatal] {exc}") from exc
    print(PASS)


if __name__ == "__main__":
    main()
