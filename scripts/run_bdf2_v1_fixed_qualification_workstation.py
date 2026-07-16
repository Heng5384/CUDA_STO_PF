#!/usr/bin/env python3
"""Run fixed-step IMEX-BDF2 qualification candidates without retries."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
import shutil
import sys
from pathlib import Path

import numpy as np

from run_bdf2_v1_startup_restart_workstation import run_case


TIME_UNIT_S = 41.12958542455477
DT_CANDIDATES = (
    (2, 0.0015625),
    (4, 0.00078125),
    (8, 0.000390625),
    (16, 0.0001953125),
)

ACCEPT_RE = re.compile(
    r"CTOT_MIMETIC_BE_ACCEPT step=(\d+) nonlinear_iters=(\d+) "
    r"res_inf=([^ ]+) res_l2_rel=([^ ]+) mass_error=([^ ]+).*?"
    r"outer_iters=(\d+).*?transport_solves=(\d+) phase_solves=(\d+) "
    r"mechanics_solves=(\d+).*?projection_mass=(\d+) clip_count=(\d+)"
)
PHASE_RE = re.compile(
    r"CTOT_PHASE_INNER physical_step=(\d+).*?iterations=(\d+) .*?"
    r"final_KKT=([^ ]+).*?converged=(\d+)"
)
CONTEXT_RE = re.compile(
    r"CTOT_IMEX_BDF2_STEP_CONTEXT step=(\d+).*?integrator=([^ ]+) "
    r"history_valid=(\d+).*?fallback_reason=([^\s]+)"
)


def finite_quantile(values: list[float], q: float) -> float:
    return float(np.quantile(np.asarray(values, dtype=np.float64), q)) \
        if values else math.nan


def iteration_stability(
    values: list[float],
) -> tuple[bool, float, float, float, list[float]]:
    """Detect unresolved tail growth without conflating physical regime shifts."""
    if len(values) < 300:
        return False, math.nan, math.nan, math.nan, []
    windows = [
        float(np.median(values[start:start + 100]))
        for start in range(0, len(values), 100)
        if len(values[start:start + 100]) == 100
    ]
    tail = windows[-4:]
    slope = float(np.polyfit(np.arange(len(tail)), tail, 1)[0])
    tail_differences = np.diff(np.asarray(tail[-3:], dtype=np.float64))
    # A bounded transition to a harder physical state is allowed.  Unresolved
    # numerical growth requires consecutive rises in the latest three complete
    # windows, not merely a positive regression slope caused by an earlier
    # transition peak.  The independent p99 gate below still requires at least
    # 75% headroom in the configured 400-iteration nonlinear budget.
    persistent = (
        len(tail_differences) == 2
        and bool(np.all(tail_differences > 0.5))
        and tail[-1] > tail[-3] + 3.0
    )
    return persistent, windows[0], windows[-1], slope, windows


def parse_log(path: Path, expected_steps: int, wall_seconds: float,
              factor: int, dt: float) -> dict[str, object]:
    text = path.read_text(encoding="utf-8")
    accepted = [
        {
            "step": int(m.group(1)),
            "transport_iterations": int(m.group(2)),
            "residual": abs(float(m.group(3))),
            "residual_l2": abs(float(m.group(4))),
            "mass_error": abs(float(m.group(5))),
            "outer_iterations": int(m.group(6)),
            "transport_solves": int(m.group(7)),
            "phase_solves": int(m.group(8)),
            "mechanics_solves": int(m.group(9)),
            "projection_mass": int(m.group(10)),
            "clip_count": int(m.group(11)),
        }
        for m in ACCEPT_RE.finditer(text)
    ]
    phases = [
        {
            "step": int(m.group(1)),
            "iterations": int(m.group(2)),
            "kkt": abs(float(m.group(3))),
            "converged": int(m.group(4)),
        }
        for m in PHASE_RE.finditer(text)
    ]
    contexts = [
        {
            "step": int(m.group(1)),
            "integrator": m.group(2),
            "history_valid": int(m.group(3)),
            "fallback_reason": m.group(4),
        }
        for m in CONTEXT_RE.finditer(text)
    ]
    transport_iterations = [row["transport_iterations"] for row in accepted]
    phase_iterations = [row["iterations"] for row in phases]
    (transport_growth, transport_first, transport_last,
     transport_tail_slope, transport_windows) = iteration_stability(
        transport_iterations[1:]
    )
    (phase_growth, phase_first, phase_last,
     phase_tail_slope, phase_windows) = iteration_stability(
        phase_iterations[1:]
    )
    transport_p99 = finite_quantile(transport_iterations, 0.99)
    phase_p99 = finite_quantile(phase_iterations, 0.99)
    iteration_budget_margin_pass = (
        transport_p99 <= 100.0 and phase_p99 <= 100.0
    )
    bdf2_energy = re.findall(r"CTOT_IMEX_BDF2_ENERGY_WORK .*?pass=(\d+)", text)
    be_energy = re.findall(r"CTOT_LIE_BE_SUBSTEP_ENERGY .*?pass=(\d+)", text)
    be_contexts = [row for row in contexts if row["integrator"] != "BDF2"]
    repeated_fallback = any(
        row["step"] != 1 or row["fallback_reason"] != "startup_history_unavailable"
        for row in be_contexts
    )
    all_bdf2_history_valid = all(
        row["history_valid"] == 1
        for row in contexts if row["integrator"] == "BDF2"
    )
    accepted_time_code = expected_steps * dt
    accepted_time_s = accepted_time_code * TIME_UNIT_S
    hard_gates = (
        len(accepted) == expected_steps
        and len(phases) == expected_steps
        and len(contexts) == expected_steps
        and text.count("_REJECT step=") == 0
        and max((row["mass_error"] for row in accepted), default=math.inf) <= 1.0e-10
        and max((row["residual"] for row in accepted), default=math.inf) <= 1.01e-12
        and max((row["kkt"] for row in phases), default=math.inf) <= 1.01e-10
        and all(row["converged"] == 1 for row in phases)
        and len(bdf2_energy) == expected_steps - 1
        and all(value == "1" for value in bdf2_energy + be_energy)
        and all(row["projection_mass"] == 0 for row in accepted)
        and all(row["clip_count"] == 0 for row in accepted)
        and all(row["transport_solves"] == 1 for row in accepted)
        and all(row["phase_solves"] == 1 for row in accepted)
        and len(be_contexts) == 1
        and not repeated_fallback
        and all_bdf2_history_valid
        and not transport_growth
        and not phase_growth
        and iteration_budget_margin_pass
    )
    return {
        "original_dt_factor": factor,
        "dt_code": dt,
        "dt_physical_s": dt * TIME_UNIT_S,
        "expected_steps": expected_steps,
        "accepted_steps": len(accepted),
        "reject_count": text.count("_REJECT step="),
        "wall_seconds": wall_seconds,
        "accepted_time_code": accepted_time_code,
        "accepted_time_physical_s": accepted_time_s,
        "accepted_physical_time_per_GPU_hour_s": (
            accepted_time_s * 3600.0 / wall_seconds if wall_seconds > 0.0 else math.nan
        ),
        "max_transport_residual": max(
            (row["residual"] for row in accepted), default=math.nan
        ),
        "max_phase_KKT": max((row["kkt"] for row in phases), default=math.nan),
        "max_mass_error": max(
            (row["mass_error"] for row in accepted), default=math.nan
        ),
        "p99_transport_iterations": transport_p99,
        "p99_phase_iterations": phase_p99,
        "transport_first100_median": transport_first,
        "transport_last100_median": transport_last,
        "transport_persistent_growth": transport_growth,
        "transport_tail_window_slope": transport_tail_slope,
        "transport_window_medians": json.dumps(transport_windows),
        "phase_first100_median": phase_first,
        "phase_last100_median": phase_last,
        "phase_persistent_growth": phase_growth,
        "phase_tail_window_slope": phase_tail_slope,
        "phase_window_medians": json.dumps(phase_windows),
        "iteration_budget_margin_pass": iteration_budget_margin_pass,
        "transport_solves_per_step": (
            sum(row["transport_solves"] for row in accepted) / len(accepted)
            if accepted else math.nan
        ),
        "phase_solves_per_step": (
            sum(row["phase_solves"] for row in accepted) / len(accepted)
            if accepted else math.nan
        ),
        "mechanics_solves_per_step": (
            sum(row["mechanics_solves"] for row in accepted) / len(accepted)
            if accepted else math.nan
        ),
        "be_fallback_steps": len(be_contexts),
        "be_fallback_fraction": len(be_contexts) / expected_steps,
        "repeated_be_fallback": repeated_fallback,
        "all_bdf2_history_valid": all_bdf2_history_valid,
        "energy_work_pass_count": sum(value == "1" for value in bdf2_energy),
        "energy_work_expected_count": expected_steps - 1,
        "zero_clipping": all(row["clip_count"] == 0 for row in accepted),
        "zero_physical_projection": all(
            row["projection_mass"] == 0 for row in accepted
        ),
        "hard_gates_pass": hard_gates,
        "status": "PASS_FIXED_DT_QUALIFICATION" if hard_gates else "FAIL_FIXED_DT_QUALIFICATION",
        "log": str(path),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--analyze-existing", action="store_true")
    args = parser.parse_args()
    if not 1000 <= args.steps <= 2000:
        raise ValueError("qualification requires 1000-2000 fixed accepted steps")
    repo = args.repo.resolve()
    root = repo / "runs/bdf2_v1/fixed_dt_qualification"
    prior_wall: dict[int, float] = {}
    if args.analyze_existing and (root / "summary.json").exists():
        prior = json.loads((root / "summary.json").read_text(encoding="utf-8"))
        for row in prior.get("candidates", []):
            wall = row.get("wall_seconds")
            if wall is not None and isinstance(wall, (int, float)) and math.isfinite(wall):
                prior_wall[int(row["original_dt_factor"])] = float(wall)
    if root.exists() and not args.analyze_existing:
        if not args.overwrite:
            raise RuntimeError(f"refusing to overwrite {root}")
        shutil.rmtree(root)
    root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for factor, dt in DT_CANDIDATES:
        tag = f"dt_div{factor}"
        try:
            if args.analyze_existing:
                log = root / tag / "run" / "run.log"
                if not log.exists():
                    raise FileNotFoundError(log)
                result = {
                    "log": log,
                    "wall_seconds": prior_wall.get(factor, math.nan),
                }
            else:
                result = run_case(repo, root, tag, dt, args.steps)
            row = parse_log(
                Path(result["log"]), args.steps,
                float(result["wall_seconds"]), factor, dt,
            )
        except Exception as exc:
            case = root / tag / "run"
            log = case / "run.log"
            row = {
                "original_dt_factor": factor,
                "dt_code": dt,
                "dt_physical_s": dt * TIME_UNIT_S,
                "expected_steps": args.steps,
                "accepted_steps": 0,
                "reject_count": math.nan,
                "wall_seconds": math.nan,
                "hard_gates_pass": False,
                "status": "FAIL_RUNTIME",
                "log": str(log),
                "failure": str(exc),
            }
        rows.append(row)
        with (root / "fixed_dt_metrics.partial.json").open("w", encoding="utf-8") as stream:
            json.dump(rows, stream, indent=2, allow_nan=True)
            stream.write("\n")
    fieldnames = list(dict.fromkeys(key for row in rows for key in row))
    with (root / "fixed_dt_metrics.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    passed = [row for row in rows if row.get("hard_gates_pass")]
    selected = max(
        passed,
        key=lambda row: float(row["accepted_physical_time_per_GPU_hour_s"]),
        default=None,
    )
    summary = {
        "qualification_steps": args.steps,
        "retry_enabled": False,
        "candidates": rows,
        "selected_original_dt_factor": (
            selected["original_dt_factor"] if selected else None
        ),
        "selected_dt_code": selected["dt_code"] if selected else None,
        "selected_dt_physical_s": selected["dt_physical_s"] if selected else None,
        "status": "PASS_FIXED_DT_SELECTION" if selected else "FAIL_NO_FIXED_DT_QUALIFIED",
    }
    (root / "summary.json").write_text(
        json.dumps(summary, indent=2, allow_nan=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2, allow_nan=True))
    return 0 if selected else 1


if __name__ == "__main__":
    raise SystemExit(main())
