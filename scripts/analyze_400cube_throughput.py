#!/usr/bin/env python3
"""Analyze actual 400-cube throughput without extrapolating from small grids."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path


SUMMARY_RE = re.compile(
    r"macro_steps=(\d+).*internal_trial_rejects=(\d+).*retry_fraction=([^ ]+).*"
    r"fallback_macros=(\d+).*fallback_fraction=([^ ]+).*"
    r"reject_trial_wall_fraction=([^ ]+)")
ACCEPT_RE = re.compile(r"CTOT_MIMETIC_BE_ACCEPT step=(\d+) nonlinear_iters=(\d+).*")
VARIABLE_DT_RE = re.compile(
    r"CTOT_VARIABLE_BDF2_CONTROLLER_COMMIT step=(\d+).*accepted_dt=([^ ]+)")
ACTIVE_RE = re.compile(r"inactive_support=(\d+)")


def only(root: Path, name: str) -> Path:
    matches = list(root.rglob(name))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} under {root}, got {matches}")
    return matches[0]


def stage_fractions(path: Path) -> tuple[float, float, float]:
    attempt = transport = phase = 0.0
    with path.open(newline="", encoding="utf-8") as stream:
        for row in csv.DictReader(stream):
            if int(row["step"]) < 4:
                continue
            value = float(row["gpu_ms"])
            stage = row["stage"]
            if stage == "attempt.total":
                attempt += value
            elif stage == "transport.residual_evaluation":
                transport += value
            elif stage in ("phase.local_driving", "phase.pdas_solve",
                           "phase.final_reconstruction"):
                phase += value
    if attempt <= 0.0:
        return math.nan, math.nan, math.nan
    # The residual and PDAS scopes are the non-overlapping top-level scopes.
    # Remaining time contains hard-gate reductions, history, and orchestration;
    # no dedicated non-authoritative audit scope exists.
    return transport / attempt, phase / attempt, max(0.0, 1.0-(transport+phase)/attempt)


def analyze_case(case: Path, geometry: str, candidate: str,
                 t_real: float) -> dict:
    log = (case / "run.log").read_text(errors="replace")
    perf = json.loads(only(case, "performance_summary.json").read_text())
    summary_matches = SUMMARY_RE.findall(log)
    if not summary_matches:
        raise RuntimeError(f"missing retry summary in {case}")
    macros, rejects, retry_fraction, fallback, fallback_fraction, retry_wall = \
        summary_matches[-1]
    accepts = [(int(a), int(b)) for a, b in ACCEPT_RE.findall(log)]
    timed_iterations = [iters for step, iters in accepts if step >= 4]
    dt = float(perf["dt"])
    variable = [(int(step), float(value))
                for step, value in VARIABLE_DT_RE.findall(log)]
    if variable:
        timed_dt = [value for step, value in variable if step >= 4]
        dt = sum(timed_dt) / len(timed_dt)
    wall = float(perf["warmup_excluded_avg_walltime_per_step_s"])
    throughput = dt * t_real * 3600.0 / wall
    mem_values = []
    with (case / "gpu_memory_samples.csv").open(encoding="utf-8") as stream:
        for line in stream:
            fields = [x.strip() for x in line.split(",")]
            if len(fields) >= 2:
                try:
                    mem_values.append(float(fields[1]))
                except ValueError:
                    pass
    transport, phase, other = stage_fractions(
        only(case, "ctot_performance_stage_timing.csv"))
    inactive = [int(x) for x in ACTIVE_RE.findall(log)]
    retry_value = float(retry_fraction)
    fallback_value = float(fallback_fraction)
    retry_wall_value = float(retry_wall)
    peak_memory = max(mem_values, default=math.nan)
    gate_failures = []
    if int(macros) != 8:
        gate_failures.append("macro_count")
    if retry_value > 0.01:
        gate_failures.append("retry_fraction")
    if fallback_value > 0.01:
        gate_failures.append("fallback_fraction")
    if retry_wall_value > 0.05:
        gate_failures.append("retry_wall_fraction")
    if not math.isfinite(peak_memory) or peak_memory > 0.85 * 16303.0:
        gate_failures.append("memory_peak")
    return {
        "geometry": geometry,
        "candidate": candidate,
        "grid": "400x400x400",
        "memory_mask": 187,
        "timed_steps": 5,
        "mean_accepted_dt_code": dt,
        "mean_accepted_dt_physical_s": dt*t_real,
        "wall_s_per_accepted_step": wall,
        "physical_s_per_GPU_hour": throughput,
        "speedup_vs_frozen_0p378860553": throughput/0.37886055291310844,
        "transport_fraction": transport,
        "phase_fraction": phase,
        "other_hard_gate_or_orchestration_fraction": other,
        "non_authoritative_audit_fraction": "not_separately_instrumented",
        "peak_memory_MiB": peak_memory,
        "retry_fraction": retry_value,
        "fallback_fraction": fallback_value,
        "retry_wall_fraction": retry_wall_value,
        "mean_timed_nonlinear_iterations": (
            sum(timed_iterations)/len(timed_iterations) if timed_iterations else math.nan),
        "max_timed_nonlinear_iterations": max(timed_iterations, default=0),
        "mean_inactive_support_fraction": (
            sum(inactive)/len(inactive)/(400**3) if inactive else math.nan),
        "macro_steps": int(macros),
        "internal_trial_rejects": int(rejects),
        "fallback_macros": int(fallback),
        "status": ("PASS_PRODUCTION_RUNTIME_GATES" if not gate_failures
                   else "FAIL_PRODUCTION_RUNTIME_GATES"),
        "gate_failures": ";".join(gate_failures),
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for geometry in ("P", "M"):
        for candidate in ("strict_dt16", "fixed_G9_dt4", "variable_G9"):
            rows.append(analyze_case(
                args.run_root / f"{geometry}_{candidate}", geometry, candidate,
                41.1295854245547687))
    args.report_root.mkdir(parents=True, exist_ok=True)
    with (args.report_root / "400cube_throughput.csv").open(
            "w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)

    by_candidate = {}
    for candidate in ("strict_dt16", "fixed_G9_dt4", "variable_G9"):
        values = [r["physical_s_per_GPU_hour"] for r in rows
                  if r["candidate"] == candidate]
        by_candidate[candidate] = min(values)
    candidate_qualified = {
        candidate: all(
            row["status"] == "PASS_PRODUCTION_RUNTIME_GATES"
            for row in rows if row["candidate"] == candidate
        )
        for candidate in ("fixed_G9_dt4", "variable_G9")
    }
    eligible = [name for name, passed in candidate_qualified.items() if passed]
    selected = max(eligible, key=by_candidate.get) if eligible else None
    measured_fastest = max(("fixed_G9_dt4", "variable_G9"), key=by_candidate.get)
    projection_candidate = selected or measured_fastest
    selected_rate = by_candidate[projection_candidate]
    speedup = selected_rate / 0.37886055291310844
    cost1 = 3600.0/selected_rate
    cost10 = 36000.0/selected_rate
    cost50 = 180000.0/selected_rate
    short = selected is not None and cost1 <= 500.0
    intermediate = selected is not None and cost10 <= 2000.0
    long = selected is not None and cost50 <= 5000.0
    table = "\n".join(
        f"| {r['geometry']} | {r['candidate']} | {r['wall_s_per_accepted_step']:.3f} | "
        f"{r['mean_accepted_dt_code']:.6e} | {r['physical_s_per_GPU_hour']:.6f} | "
        f"{r['speedup_vs_frozen_0p378860553']:.2f}x | {r['retry_fraction']:.3%} | "
        f"{r['fallback_fraction']:.3%} | {r['status']} |" for r in rows)
    decision = f"""# Actual 400-cube throughput decision

All values below are measured on the same RTX 5080 with memory mask 187. Each
row uses three warmup accepted steps followed by five timed accepted steps.
Initialization, plan creation, and field I/O are excluded from steady-state time.

| Geometry | Candidate | wall s/step | mean dt code | physical s/GPUh | vs frozen | retry | fallback | status |
|---|---|---:|---:|---:|---:|---:|---:|---|
{table}

The conservative cross-geometry rate is the minimum of P and M. The fastest
qualified candidate is `{selected if selected else 'NONE'}`. The fastest measured
candidate is `{measured_fastest}` at `{by_candidate[measured_fastest]:.6f}`
physical s/GPUh. The cost projection uses `{projection_candidate}` at
`{selected_rate:.6f}` physical s/GPUh and is
`{'a qualified production projection' if selected else 'an optimistic unqualified upper bound'}`.
The corresponding measured speedup is `{speedup:.2f}x` relative to the frozen
0.378861 baseline.

Status: `{'PASS_MINIMUM_USEFUL_10X' if selected and speedup >= 10 else ('QUALIFIED_FULL_PF_THROUGHPUT_BELOW_10X_USEFUL_GATE' if selected else 'NO_400CUBE_PRODUCTION_CANDIDATE_PASSED_RUNTIME_GATES')}`.
"""
    (args.report_root / "400cube_throughput_decision.md").write_text(decision)
    projection = f"""# Full-PF cost projection

Using the conservative P/M rate `{selected_rate:.9f}` physical s/GPUh from
`{projection_candidate}` (`{'qualified' if selected else 'unqualified optimistic upper bound'}`):

| Target physical time | GPU hours |
|---:|---:|
| 1 h | {cost1:.1f} |
| 10 h | {cost10:.1f} |
| 50 h | {cost50:.1f} |

These costs exclude future GP release, elasticity, and particle-analysis cost.
"""
    (args.report_root / "full_pf_cost_projection.md").write_text(projection)
    feasibility = f"""# Full-PF feasibility decision

- `FULL_PF_SHORT_TRANSIENT_FEASIBLE`: `{str(short).lower()}`
- `FULL_PF_INTERMEDIATE_FEASIBLE`: `{str(intermediate).lower()}`
- `FULL_PF_LONG_COARSENING_FEASIBLE`: `{str(long).lower()}`

Engineering classification: `{'FULL_PF_LONG_TIME_NOT_PRACTICAL; HYBRID_HANDOFF_REQUIRED' if not long else 'FULL_PF_LONG_COARSENING_FEASIBLE'}`.
"""
    (args.report_root / "full_pf_feasibility_decision.md").write_text(feasibility)
    print(json.dumps({"selected": selected, "measured_fastest": measured_fastest,
                      "projection_candidate": projection_candidate,
                      "candidate_qualified": candidate_qualified,
                      "rate": selected_rate, "speedup": speedup,
                      "cost50": cost50, "long": long}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
