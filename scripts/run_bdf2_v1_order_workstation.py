#!/usr/bin/env python3
"""Consistent-history equal-time order test for fixed-step IMEX-BDF2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
from pathlib import Path

import numpy as np

from run_bdf2_v1_startup_restart_workstation import run_case


ACCEPT_RE = re.compile(
    r"CTOT_MIMETIC_BE_ACCEPT step=(\d+) nonlinear_iters=(\d+).*?"
    r"mass_error=([^ ]+).*?transport_solves=(\d+) phase_solves=(\d+) "
    r"mechanics_solves=(\d+)"
)


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def l2(values: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(values, dtype=np.float64))))


def right_interface_position(phi: np.ndarray, dx: float = 1.0) -> float:
    center = int(np.argmax(phi))
    n = phi.size
    for offset in range(1, n // 2 + 1):
        left = (center + offset - 1) % n
        right = (center + offset) % n
        a = float(phi[left])
        b = float(phi[right])
        if a >= 0.5 and b < 0.5:
            fraction = 0.0 if a == b else (a - 0.5) / (a - b)
            return (center + offset - 1 + fraction) * dx
    return math.nan


def parse_log(path: Path, expected_steps: int) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    accepted = [
        {
            "step": int(m.group(1)),
            "iterations": int(m.group(2)),
            "mass_error": abs(float(m.group(3))),
            "transport_solves": int(m.group(4)),
            "phase_solves": int(m.group(5)),
            "mechanics_solves": int(m.group(6)),
        }
        for m in ACCEPT_RE.finditer(text)
    ]
    bdf2_energy = re.findall(r"CTOT_IMEX_BDF2_ENERGY_WORK .*?pass=(\d+)", text)
    be_energy = re.findall(r"CTOT_LIE_BE_SUBSTEP_ENERGY .*?pass=(\d+)", text)
    contexts = re.findall(r"CTOT_IMEX_BDF2_STEP_CONTEXT .*?integrator=([^ ]+)", text)
    iterations = np.asarray([row["iterations"] for row in accepted], dtype=float)
    return {
        "accepted_steps": len(accepted),
        "expected_steps": expected_steps,
        "reject_count": text.count("_REJECT step="),
        "bdf2_steps": sum(value == "BDF2" for value in contexts),
        "be_fallback_steps": sum(value != "BDF2" for value in contexts),
        "energy_all_pass": all(value == "1" for value in bdf2_energy + be_energy),
        "max_mass_error": max((row["mass_error"] for row in accepted), default=math.nan),
        "p99_iterations": float(np.quantile(iterations, 0.99)) if iterations.size else math.nan,
        "max_transport_solves": max((row["transport_solves"] for row in accepted), default=0),
        "max_phase_solves": max((row["phase_solves"] for row in accepted), default=0),
        "max_mechanics_solves": max((row["mechanics_solves"] for row in accepted), default=0),
        "hard_gates_pass": (
            len(accepted) == expected_steps
            and "_REJECT step=" not in text
            and all(value == "1" for value in bdf2_energy + be_energy)
            and "clip_count=0" in text
            and "projection_mass=0" in text
        ),
    }


def adjacent_error(
    a: dict[str, np.ndarray | float],
    b: dict[str, np.ndarray | float],
) -> dict[str, float]:
    delta_C_a = a["Ctot"] - a["Ctot_measurement_start"]
    delta_C_b = b["Ctot"] - b["Ctot_measurement_start"]
    delta_phi_a = a["phi"] - a["phi_measurement_start"]
    delta_phi_b = b["phi"] - b["phi_measurement_start"]
    delta_h_a = float(a["hvolume"]) - float(a["hvolume_measurement_start"])
    delta_h_b = float(b["hvolume"]) - float(b["hvolume_measurement_start"])
    return {
        "Ctot_L2": l2(a["Ctot"] - b["Ctot"]),
        "Ctot_Linf": float(np.max(np.abs(a["Ctot"] - b["Ctot"]))),
        "Ctot_increment_L2": l2(delta_C_a - delta_C_b),
        "phi_L2": l2(a["phi"] - b["phi"]),
        "phi_Linf": float(np.max(np.abs(a["phi"] - b["phi"]))),
        "phi_increment_L2": l2(delta_phi_a - delta_phi_b),
        "hvolume_abs": abs(float(a["hvolume"]) - float(b["hvolume"])),
        "hvolume_increment_abs": abs(delta_h_a - delta_h_b),
        "interface_abs": abs(float(a["interface"]) - float(b["interface"])),
        "interface_increment_abs": abs(
            (float(a["interface"]) - float(a["interface_measurement_start"]))
            - (float(b["interface"]) - float(b["interface_measurement_start"]))
        ),
        "matrix_L1": float(np.mean(np.abs(a["xB"] - b["xB"]))),
        "matrix_Linf": float(np.max(np.abs(a["xB"] - b["xB"]))),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--base-dt", type=float, default=0.003125)
    parser.add_argument(
        "--base-original-factor",
        type=int,
        default=1,
        help="Label the base dt as original_dt divided by this factor.",
    )
    parser.add_argument("--warmup-coarse-steps", type=int, default=8)
    parser.add_argument("--measurement-coarse-steps", type=int, default=4)
    parser.add_argument(
        "--output-root", type=Path,
        default=Path("runs/bdf2_v1/order_validation"),
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--active-manifold",
        action="store_true",
        help="Use the default-off active-manifold IMEX-BDF2 context.",
    )
    args = parser.parse_args()
    repo = args.repo.resolve()
    root = args.output_root
    if not root.is_absolute():
        root = repo / root
    if root.exists():
        if not args.overwrite:
            raise RuntimeError(f"refusing to overwrite {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True)

    if args.base_dt <= 0.0:
        raise ValueError("--base-dt must be positive")
    if args.base_original_factor <= 0:
        raise ValueError("--base-original-factor must be positive")
    base_dt = args.base_dt
    warmup_time = base_dt * args.warmup_coarse_steps
    measurement_time = base_dt * args.measurement_coarse_steps
    total_time = warmup_time + measurement_time
    initial = {
        "Ctot": np.fromfile(repo / "runs/frozen_input/ctot_checkpoint_step000054_Ctot.raw", dtype=np.float64),
        "phi": np.fromfile(repo / "runs/frozen_input/ctot_checkpoint_step000054_phi.raw", dtype=np.float64),
        "xB": np.fromfile(repo / "runs/frozen_input/ctot_checkpoint_step000054_xB_alpha.raw", dtype=np.float64),
    }
    solutions: dict[int, dict[str, object]] = {}
    param_overrides = None
    if args.active_manifold:
        param_overrides = {
            "ctot_numerics_contract":
                "ctot_jichen_imex_bdf2_active_manifold_v1",
            "BDF2_EVENT_PREFLIGHT_V1": "1",
            "BDF2_EVENT_BE_SUBCYCLING_V1": "1",
        }
    for factor in (1, 2, 4, 8):
        original_factor = args.base_original_factor * factor
        dt = base_dt / factor
        warmup_steps = args.warmup_coarse_steps * factor
        steps = (args.warmup_coarse_steps + args.measurement_coarse_steps) * factor
        warmup_result = run_case(
            repo, root / "warmup", f"dt_div{original_factor}", dt,
            warmup_steps, param_overrides=param_overrides,
        )
        result = run_case(
            repo, root / "full", f"dt_div{original_factor}", dt, steps,
            param_overrides=param_overrides,
        )
        warmup_state = {
            "Ctot": np.fromfile(warmup_result["Ctot"], dtype=np.float64),
            "phi": np.fromfile(warmup_result["phi"], dtype=np.float64),
            "xB": np.fromfile(warmup_result["xB"], dtype=np.float64),
        }
        state = {
            "Ctot": np.fromfile(result["Ctot"], dtype=np.float64),
            "phi": np.fromfile(result["phi"], dtype=np.float64),
            "xB": np.fromfile(result["xB"], dtype=np.float64),
        }
        state["hvolume"] = float(np.sum(h(state["phi"])))
        state["interface"] = right_interface_position(state["phi"])
        state["Ctot_measurement_start"] = warmup_state["Ctot"]
        state["phi_measurement_start"] = warmup_state["phi"]
        state["xB_measurement_start"] = warmup_state["xB"]
        state["hvolume_measurement_start"] = float(
            np.sum(h(warmup_state["phi"]))
        )
        state["interface_measurement_start"] = right_interface_position(
            warmup_state["phi"]
        )
        log_metrics = parse_log(Path(result["log"]), steps)
        solutions[factor] = {
            **state,
            **log_metrics,
            "dt": dt,
            "steps": steps,
            "wall_seconds": float(result["wall_seconds"]),
            "history_generation_wall_seconds": float(warmup_result["wall_seconds"]),
            "warmup_steps": warmup_steps,
            "physical_time_code": total_time,
        }

    pair_errors: dict[str, dict[str, float]] = {}
    for coarse, fine in ((1, 2), (2, 4), (4, 8)):
        pair_errors[f"{coarse}_to_{fine}"] = adjacent_error(
            solutions[coarse], solutions[fine]
        )
    ratios: dict[str, dict[str, float]] = {}
    for coarse_pair, fine_pair, label in (
        ("1_to_2", "2_to_4", "dt_over_dt2"),
        ("2_to_4", "4_to_8", "dt2_over_dt4"),
    ):
        ratios[label] = {}
        for key in pair_errors[coarse_pair]:
            numerator = pair_errors[coarse_pair][key]
            denominator = pair_errors[fine_pair][key]
            ratios[label][key] = numerator / denominator if denominator > 0.0 else math.nan

    reference = solutions[8]
    reference_delta_C = (
        reference["Ctot"] - reference["Ctot_measurement_start"]
    )
    reference_delta_phi = (
        reference["phi"] - reference["phi_measurement_start"]
    )
    reference_delta_h = (
        float(reference["hvolume"])
        - float(reference["hvolume_measurement_start"])
    )
    reference_delta_interface = (
        float(reference["interface"])
        - float(reference["interface_measurement_start"])
    )
    rows = []
    for factor in (1, 2, 4, 8):
        original_factor = args.base_original_factor * factor
        state = solutions[factor]
        c_error = state["Ctot"] - reference["Ctot"]
        phi_error = state["phi"] - reference["phi"]
        c_increment_error = (
            state["Ctot"] - state["Ctot_measurement_start"]
            - reference_delta_C
        )
        phi_increment_error = (
            state["phi"] - state["phi_measurement_start"]
            - reference_delta_phi
        )
        h_increment = (
            float(state["hvolume"])
            - float(state["hvolume_measurement_start"])
        )
        interface_increment = (
            float(state["interface"])
            - float(state["interface_measurement_start"])
        )
        rows.append(
            {
                "factor": original_factor,
                "local_refinement_factor": factor,
                "dt": state["dt"],
                "steps": state["steps"],
                "Ctot_field_L2_error": l2(c_error),
                "Ctot_field_Linf_error": float(np.max(np.abs(c_error))),
                "Ctot_increment_rel_L2_error": l2(c_increment_error)
                / max(l2(reference_delta_C), 1.0e-30),
                "phi_field_L2_error": l2(phi_error),
                "phi_field_Linf_error": float(np.max(np.abs(phi_error))),
                "phi_increment_rel_L2_error": l2(phi_increment_error)
                / max(l2(reference_delta_phi), 1.0e-30),
                "hvolume": state["hvolume"],
                "hvolume_field_rel_error": abs(
                    state["hvolume"] - reference["hvolume"]
                ) / max(abs(reference["hvolume"]), 1.0e-30),
                "hvolume_increment_rel_error": abs(
                    h_increment - reference_delta_h
                ) / max(abs(reference_delta_h), 1.0e-30),
                "interface_position": state["interface"],
                "interface_error_dx": abs(state["interface"] - reference["interface"]),
                "interface_increment": interface_increment,
                "interface_increment_error_dx": abs(
                    interface_increment - reference_delta_interface
                ),
                "matrix_L1_error": float(np.mean(np.abs(state["xB"] - reference["xB"]))),
                "matrix_Linf_error": float(np.max(np.abs(state["xB"] - reference["xB"]))),
                "accepted_steps": state["accepted_steps"],
                "bdf2_steps": state["bdf2_steps"],
                "be_fallback_steps": state["be_fallback_steps"],
                "max_mass_error": state["max_mass_error"],
                "p99_iterations": state["p99_iterations"],
                "max_transport_solves": state["max_transport_solves"],
                "max_phase_solves": state["max_phase_solves"],
                "wall_seconds": state["wall_seconds"],
                "history_generation_wall_seconds": state[
                    "history_generation_wall_seconds"
                ],
                "hard_gates_pass": state["hard_gates_pass"],
            }
        )
    with (root / "fixed_dt_order_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    order_keys = (
        "Ctot_L2",
        "Ctot_increment_L2",
        "phi_L2",
        "phi_increment_L2",
        "hvolume_abs",
        "hvolume_increment_abs",
    )
    second_order = all(
        3.0 <= ratios[level][key] <= 5.0
        for level in ratios for key in order_keys
    )
    coarse_row = rows[0]
    short_accuracy = (
        coarse_row["Ctot_increment_rel_L2_error"] <= 0.02
        and coarse_row["phi_increment_rel_L2_error"] <= 0.02
        and coarse_row["hvolume_increment_rel_error"] <= 0.02
        and coarse_row["interface_error_dx"] <= 0.5
        and all(
            math.copysign(1.0, row["interface_increment"])
            == math.copysign(1.0, reference_delta_interface)
            for row in rows
            if row["interface_increment"] != 0.0
            and reference_delta_interface != 0.0
        )
    )
    summary = {
        "common_initial_checkpoint": "runs/frozen_input/ctot_checkpoint_step000054",
        "original_dt_code": 0.003125,
        "base_dt_code": base_dt,
        "base_original_factor": args.base_original_factor,
        "active_manifold_context": args.active_manifold,
        "original_dt_factors": [
            args.base_original_factor * factor for factor in (1, 2, 4, 8)
        ],
        "consistent_history_method": (
            "independent_one_BE_startup_then_BDF2_warmup_per_dt_from_common_state"
        ),
        "startup_steps_excluded_from_order_interval": True,
        "warmup_coarse_steps": args.warmup_coarse_steps,
        "warmup_time_code": warmup_time,
        "measurement_coarse_steps": args.measurement_coarse_steps,
        "measurement_time_code": measurement_time,
        "total_time_code": total_time,
        "total_time_physical_s": total_time * 41.12958542455477,
        "pair_errors": pair_errors,
        "ratios": ratios,
        "second_order_status": second_order,
        "short_window_accuracy_status": short_accuracy,
        "all_hard_gates_pass": all(row["hard_gates_pass"] for row in rows),
        "status": (
            "PASS_BDF2_ORDER_AND_SHORT_WINDOW"
            if second_order and short_accuracy and all(row["hard_gates_pass"] for row in rows)
            else "FAIL_BDF2_ORDER_OR_SHORT_WINDOW"
        ),
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True))
    return 0 if summary["status"].startswith("PASS") else 1


if __name__ == "__main__":
    raise SystemExit(main())
