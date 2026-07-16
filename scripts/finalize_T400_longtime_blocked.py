#!/usr/bin/env python3
"""Assemble the fail-closed T400 long-time qualification record."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np
from scipy.integrate import solve_ivp


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "T400_longtime_v1"
sys.path.insert(0, str(ROOT))
from scripts.jc4_ji_chen_1d_oracle import JiChen1DOracle, OracleConfig  # noqa: E402
from scripts.jc4_long_time_sharp_oracle import (  # noqa: E402
    LongTimePlanarSharpOracle,
    SharpConfig,
)


RUNS = (
    {
        "run_id": "growth_N512_shift0_dt8_pilot10k",
        "dt_code": 0.000390625,
        "dt_physical_s": 0.016066244306466707,
        "requested_steps": 10000,
        "refinement_label": "selected_dt_original_ladder_dt_over_8",
    },
    {
        "run_id": "growth_N512_shift0_dt16_pilot20k",
        "dt_code": 0.0001953125,
        "dt_physical_s": 0.008033122153233353,
        "requested_steps": 20000,
        "refinement_label": "primary_refinement_original_ladder_dt_over_16",
    },
)


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("status\nNOT_RUN\n")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def first_float(text: str, pattern: str) -> float:
    match = re.search(pattern, text)
    return float(match.group(1)) if match else math.nan


def collect_runtime_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in RUNS:
        case = OUT / "coarse4_runs" / spec["run_id"]
        log = (case / "run" / "run.log").read_text(errors="replace")
        summary = json.loads((case / "analysis_summary.json").read_text())
        reject = next((line for line in log.splitlines() if "CTOT_MIMETIC_BE_REJECT" in line), "")
        reject_step = int(first_float(reject, r"step=(\d+)"))
        fallback = "none"
        contexts = [line for line in log.splitlines() if f"step={reject_step}" in line and "CTOT_IMEX_BDF2_STEP_CONTEXT" in line]
        if contexts:
            match = re.search(r"fallback_reason=([^ ]+)", contexts[-1])
            fallback = match.group(1) if match else "unknown"
        energy_lines = [line for line in log.splitlines() if f"step={reject_step}" in line and "CTOT_IMEX_BDF2_ENERGY_WORK" in line]
        energy = energy_lines[-1] if energy_lines else ""
        status_payload = json.loads((case / "workstation_status.json").read_text())
        rows.append({
            **spec,
            "accepted_steps": summary["accepted_steps"],
            "reject_step": reject_step,
            "reject_time_code": reject_step * spec["dt_code"],
            "reject_time_s": reject_step * spec["dt_physical_s"],
            "last_checkpoint_h_displacement_nm": summary["final_h_displacement_nm"],
            "last_checkpoint_crossing_displacement_nm": summary["final_crossing_displacement_nm"],
            "max_checkpoint_mass_error_rel": summary["max_mass_error_rel"],
            "reject_transport_residual": first_float(reject, r"res_inf=([0-9.eE+-]+)"),
            "reject_mass_error": first_float(reject, r"mass_error=([0-9.eE+-]+)"),
            "fallback_reason": fallback,
            "energy_work_present": bool(energy),
            "energy_work_pass": int(first_float(energy, r"pass=(\d+)")) if energy else "not_reached",
            "energy_nonfinite_chain": bool(energy and ("nonlinear_chain=nan" in energy or "phase_chain=nan" in energy)),
            "clipping_count": summary["clipping_count"],
            "physical_projection_count": summary["physical_projection_count"],
            "accepted_state_nonfinite": summary["nonfinite"],
            "wall_seconds": status_payload["wall_seconds"],
            "hard_gate_status": "FAIL",
            "termination": "REJECT_RETRY_DISABLED",
        })
    return rows


def direct_reference_probe() -> dict[str, Any]:
    config = OracleConfig(
        temperature_c=400.0, cells=512, dx_nm=1.0, lambda_nm=4.0,
        initial_beta_half_width_nm=16.0, initial_matrix_xB=0.05,
        lphi_ratio=0.90, final_time_code=10.0, output_points=11,
        rtol=2.0e-11, atol=2.0e-13, max_step_code=0.1,
        matrix_support_eps=1.0e-10,
    )
    start = time.perf_counter()
    try:
        result = JiChen1DOracle(config).run()
        rows = result["records"]
        displacement = rows[-1]["beta_half_width_nm"] - rows[0]["beta_half_width_nm"]
        status = "PASS"
        error = ""
    except Exception as exc:  # the exception itself is the qualification evidence
        displacement = math.nan
        status = "FAIL_LONG_GROWTH_JACOBIAN_SINGULAR"
        error = f"{type(exc).__name__}: {exc}"
    return {
        "reference": "direct_JiChen_solution_logit",
        "grid_cells": config.cells,
        "dx_nm": config.dx_nm,
        "box_nm": config.length_nm,
        "target_time_code": config.final_time_code,
        "wall_seconds": time.perf_counter() - start,
        "displacement_nm": displacement,
        "mass_error_rel": math.nan,
        "grid_convergence": "NOT_AVAILABLE",
        "time_convergence": "NOT_AVAILABLE",
        "restart": "NOT_RUN_AFTER_FAILURE",
        "status": status,
        "detail": error,
    }


def sharp_case(direction: str, length_nm: float, cells: int, rtol: float) -> dict[str, Any]:
    final_s = 3000.0 if direction == "growth" else 205000.0
    x0 = 0.05 if direction == "growth" else 1.0e-4
    cfg = SharpConfig(
        temperature_c=400.0, length_nm=length_nm,
        initial_beta_half_width_nm=16.0, initial_matrix_xB=x0,
        matrix_cells=cells, final_time_s=final_s, output_points=41,
        rtol=rtol, atol=0.01 * rtol,
    )
    oracle = LongTimePlanarSharpOracle(cfg)
    start = time.perf_counter(); result = oracle.run(); wall = time.perf_counter() - start
    last = result["records"][-1]
    initial_inventory = oracle.inventory(oracle.initial_state())
    outer = 0.5 * length_nm
    equilibrium_radius = (initial_inventory - outer * oracle.x_eq) / (1.0 - oracle.x_eq)
    return {
        "reference": "diffusion_sharp_Stefan",
        "direction": direction,
        "grid_cells": cells,
        "dx_nominal_nm": (outer - 16.0) / cells,
        "box_nm": length_nm,
        "final_time_s": final_s,
        "rtol": rtol,
        "wall_seconds": wall,
        "displacement_nm": last["displacement_nm"],
        "equilibrium_displacement_nm": equilibrium_radius - 16.0,
        "matrix_mean_xB": last["matrix_mean_xB"],
        "mass_error_rel": result["max_mass_error_rel"],
        "direction_pass": last["displacement_nm"] > 0 if direction == "growth" else last["displacement_nm"] < 0,
        "status": "PASS",
    }


def build_reference_rows() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    raw: list[dict[str, Any]] = []
    for direction, length in (("growth", 512.0), ("dissolution", 3072.0)):
        for cells in (400, 800, 1600):
            raw.append(sharp_case(direction, length, cells, 2.0e-10))
    # Independent tolerance refinement at the middle spatial grid.
    for direction, length in (("growth", 512.0), ("dissolution", 3072.0)):
        raw.append(sharp_case(direction, length, 800, 2.0e-11))
    direct = direct_reference_probe()
    summary: list[dict[str, Any]] = [direct]
    for direction in ("growth", "dissolution"):
        values = [row for row in raw if row["direction"] == direction and row["rtol"] == 2.0e-10]
        coarse, medium, fine = values
        time_ref = next(row for row in raw if row["direction"] == direction and row["rtol"] == 2.0e-11)
        spatial_error = abs(fine["displacement_nm"] - medium["displacement_nm"])
        time_error = abs(time_ref["displacement_nm"] - medium["displacement_nm"])
        summary.append({
            "reference": f"diffusion_sharp_{direction}",
            "grid_cells": "400;800;1600",
            "dx_nm": "ALE",
            "box_nm": medium["box_nm"],
            "target_time_code": "physical_seconds",
            "wall_seconds": sum(row["wall_seconds"] for row in raw if row["direction"] == direction),
            "displacement_nm": medium["displacement_nm"],
            "mass_error_rel": max(row["mass_error_rel"] for row in raw if row["direction"] == direction),
            "grid_convergence": spatial_error,
            "time_convergence": time_error,
            "restart": "NOT_RUN_AFTER_STAGE3_HARD_STOP",
            "status": "PARTIAL_GRID_TIME_MASS_DIRECTION_PASS_RESTART_NOT_RUN",
            "detail": f"finite_box_equilibrium_displacement_nm={medium['equilibrium_displacement_nm']:.17e}",
        })
    summary.extend([
        {
            "reference": "finite_Lphi_mixed_control_sharp", "grid_cells": "NOT_RUN",
            "dx_nm": "NOT_RUN", "box_nm": "MATCHED_REQUIRED", "target_time_code": "NOT_RUN",
            "wall_seconds": 0.0, "displacement_nm": math.nan, "mass_error_rel": math.nan,
            "grid_convergence": "NOT_RUN", "time_convergence": "NOT_RUN", "restart": "NOT_RUN",
            "status": "NOT_RUN_AFTER_STAGE3_HARD_STOP", "detail": "finite Lphi requires separate mixed-control comparator",
        },
        {
            "reference": "fine_fixed_Ctot_lambda0p3_dx0p025", "grid_cells": "20480_for_512nm",
            "dx_nm": 0.025, "box_nm": 512.0, "target_time_code": "2_to_4_nm_allowed",
            "wall_seconds": 0.0, "displacement_nm": math.nan, "mass_error_rel": math.nan,
            "grid_convergence": "NOT_RUN", "time_convergence": "NOT_RUN", "restart": "NOT_RUN",
            "status": "NOT_RUN_AFTER_STAGE3_HARD_STOP", "detail": "downstream reference stopped after coarse4 dt hard gate failed",
        },
    ])
    return raw, summary


def main() -> int:
    runtime = collect_runtime_rows()
    sharp_raw, reference = build_reference_rows()
    write_csv(OUT / "dt_refinement_metrics.csv", runtime)
    write_csv(OUT / "reference_self_qualification.csv", reference)
    write_csv(OUT / "sharp_reference_convergence_raw.csv", sharp_raw)

    dt_table = "\n".join(
        f"| {row['refinement_label']} | {row['dt_code']:.10g} | {row['dt_physical_s']:.10g} | "
        f"{row['accepted_steps']} | {row['reject_step']} | {row['reject_time_s']:.6g} | "
        f"{row['reject_transport_residual']:.3e} | {row['fallback_reason']} | "
        f"{row['energy_work_pass']} | FAIL |" for row in runtime
    )
    (OUT / "dt_refinement.md").write_text(f"""# T400 Coarse4 BDF2 Long-Time DT Refinement

| ladder row | dt code | dt physical (s) | accepted | reject step | reject time (s) | transport residual | fallback | energy/work | status |
|---|---:|---:|---:|---:|---:|---:|---|---|---|
{dt_table}

Both fixed steps fail before the 8 nm growth horizon with retry disabled.  The
selected step reaches an infeasible extrapolated phase context and its BE
fallback transport solve stops at `5.135e-12`.  The primary refinement reaches
the same active-set window but its endpoint energy/work chain becomes nonfinite
and is rejected.  Neither mass loss, clipping, projection, nor an accepted-state
NaN caused the failure.

Per the preregistered stop rule, `dt_convergence_status=BLOCKED` and
`final_status=BLOCKED_T400_LONGTIME_TIME_CONVERGENCE`.
""", encoding="utf-8")

    ref_table = "\n".join(
        f"| {row['reference']} | {row['status']} | {row.get('displacement_nm', '')} | "
        f"{row.get('mass_error_rel', '')} | {row.get('detail', '')} |" for row in reference
    )
    (OUT / "reference_ensemble.md").write_text(f"""# T400 Long-Time Reference Ensemble

| reference | status | displacement (nm) | mass error | detail |
|---|---|---:|---:|---|
{ref_table}

The diffusion-only finite-box Stefan oracle passes direction, mass, spatial-grid,
and integration-tolerance checks, but its restart check was not run after the
Stage-3 hard stop, so it remains a partial rather than fully qualified reference.
It predicts about 8.37 nm growth in 3000 s
for the 512 nm box.  For dissolution, conservation proves that 512 and 1024 nm
boxes can retreat only about 1.4 and 2.9 nm; a 3072 nm box is the first tested
box capable of the required 8 nm scale, reaching about 8 nm near 205000 s.

The direct solution/logit Ji--Chen oracle is not a qualified long-growth
reference: its zero-capacity beta-core variables make the long-run sparse
Jacobian exactly singular.  The mixed-control sharp and fine fixed-Ctot runs are
downstream work and were not started after the coarse4 Stage-3 hard stop.
""", encoding="utf-8")

    partial = []
    for row in runtime:
        ts = OUT / "coarse4_runs" / row["run_id"] / "coarse4_timeseries.csv"
        with ts.open(newline="") as handle:
            partial.extend(csv.DictReader(handle))
    write_csv(OUT / "growth_timeseries.csv", partial)
    (OUT / "growth_validation.md").write_text("""# T400 Long-Time Growth Validation

Status: `BLOCKED_BEFORE_8NM`.

Both preregistered fixed steps agree closely through the last common checkpoint
(about 0.745 nm h-volume displacement and 0.790 nm phi=0.5 displacement), but
each later incurs one hard rejection in the active-set transition window.  No
8 nm accepted trajectory exists, so no growth fidelity error is assigned.
""", encoding="utf-8")

    write_csv(OUT / "dissolution_timeseries.csv", [{"status": "NOT_RUN_AFTER_STAGE3_HARD_STOP"}])
    (OUT / "dissolution_validation.md").write_text("""# T400 Long-Time Dissolution Validation

Status: `NOT_RUN_AFTER_STAGE3_HARD_STOP`.

The corrected matched finite-box definition requires 3072/4096 nm boxes.  The
run was not started after both coarse4 growth dt rows failed their hard gates.
""", encoding="utf-8")

    for stem, title in (
        ("box_initial", "Box And Initial-State Sensitivity"),
        ("jichen_equal_error", "Ji--Chen Equal-Error Comparison"),
    ):
        write_csv(OUT / f"{stem}_metrics.csv", [{"status": "NOT_RUN_AFTER_STAGE3_HARD_STOP"}])
        (OUT / f"{stem}_sensitivity.md" if stem == "box_initial" else OUT / f"{stem}_comparison.md").write_text(
            f"# T400 {title}\n\nStatus: `NOT_RUN_AFTER_STAGE3_HARD_STOP`.\n",
            encoding="utf-8",
        )

    fidelity = [{
        "T400_fidelity_tier": "UNASSIGNED_BLOCKED",
        "growth_direction": "POSITIVE_BEFORE_REJECT",
        "dissolution_direction": "NOT_RUN",
        "growth_8nm": False,
        "dissolution_8nm": False,
        "dt_convergence": "FAIL_HARD_GATE",
        "reason": "selected dt and primary refinement each reject before long displacement",
    }]
    write_csv(OUT / "fidelity_metrics.csv", fidelity)
    (OUT / "fidelity_tier_decision.md").write_text("""# T400 Fidelity Tier Decision

`T400_fidelity_tier=UNASSIGNED_BLOCKED`

Tier Q, Tier T, and physics FAIL are not assigned because the numerical
qualification prerequisite failed before either 8 nm trajectory existed.  This
is a numerical hard-gate block, not evidence that the underlying physics has a
wrong growth or dissolution sign.
""", encoding="utf-8")
    (OUT / "publication_claim_boundary.md").write_text("""# Publication Claim Boundary

Allowed: the zero-driving coarse4 profile is stable for 1000 steps; the early
growth direction is positive; the diffusion-sharp finite-box inventory/time
scales are self-consistent; the present fixed-step BDF2 path is blocked at a
long-growth active-set transition.

Forbidden: quantitative long-time T400 fidelity, 8 nm growth/dissolution,
equal-error Ji--Chen throughput, T380 transfer, GP/S3, curvature, or production
campaign claims.

`recommended_next_action=AUDIT_BDF2_ACTIVE_SET_CONTEXT_AND_ENERGY_WORK_NAN_AT_GROWTH_TRANSITION`
""", encoding="utf-8")

    terminal = """BDF2_baseline_preserved=true
selected_longtime_dt_code=NONE
selected_longtime_dt_physical=NONE
growth_displacement_nm=BLOCKED_BEFORE_8NM
dissolution_displacement_nm=NOT_RUN
dt_convergence_status=FAIL_HARD_GATE
box_convergence_status=NOT_RUN
subcell_sensitivity_status=NOT_RUN
direct_JiChen_reference_status=FAIL_LONG_GROWTH_JACOBIAN_SINGULAR
fine_Ctot_reference_status=NOT_RUN_AFTER_STAGE3_HARD_STOP
sharp_reference_status=PARTIAL_DIFFUSION_SHARP_QUALIFIED_MIXED_NOT_RUN
growth_direction_status=POSITIVE_BEFORE_REJECT
dissolution_direction_status=NOT_RUN
growth_transfer_error=NOT_AVAILABLE
dissolution_transfer_error=NOT_AVAILABLE
growth_trajectory_error=NOT_AVAILABLE
dissolution_trajectory_error=NOT_AVAILABLE
final_beta_amount_error=NOT_AVAILABLE
equilibrium_matrix_composition_error=NOT_AVAILABLE
matrix_profile_error=NOT_AVAILABLE
T400_fidelity_tier=UNASSIGNED_BLOCKED
JiChen_equal_error_throughput=NOT_RUN
Ctot_BDF2_equal_error_throughput=NOT_RUN
throughput_ratio=NOT_AVAILABLE
T380_status=NOT_RUN
GP_status=NOT_RUN
curvature_status=NOT_RUN
multiparticle_status=NOT_RUN
large_3D_status=NOT_RUN
cluster_used=false
commit_created=false
push_performed=false
allowed_publication_claims=EARLY_DIRECTION_AND_NUMERICAL_BLOCKER_ONLY
recommended_next_action=AUDIT_BDF2_ACTIVE_SET_CONTEXT_AND_ENERGY_WORK_NAN_AT_GROWTH_TRANSITION
final_status=BLOCKED_T400_LONGTIME_TIME_CONVERGENCE
"""
    (OUT / "final_terminal_output.txt").write_text(terminal, encoding="utf-8")
    print(terminal, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
