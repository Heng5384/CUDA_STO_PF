#!/usr/bin/env python3
"""Finalize fail-closed JC4 qualification reports from workstation evidence."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/pf_ctot_production_candidate"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, rows: Iterable[dict[str, object]]) -> None:
    rows = list(rows)
    if not rows:
        raise ValueError(f"empty output {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def find_line(path: Path, needle: str) -> int:
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if needle in line:
            return index
    raise ValueError(f"{needle!r} absent from {path}")


def one_by_direction(rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["direction"]: row for row in rows}


def status_row(
    source: str,
    row: dict[str, str] | None,
    direction: str,
    status: str,
    note: str,
) -> dict[str, object]:
    keys = (
        "case_id", "grid", "dx_nm", "lambda_nm", "dt_code", "nsteps",
        "requested_elapsed_s", "actual_elapsed_s", "actual_to_requested_time_ratio",
        "accepted_steps", "retry_count", "reject_count", "PF_h_displacement_nm",
        "sharp_displacement_nm", "trajectory_error_abs_nm", "trajectory_error_rel",
        "final_phase_fraction_error_rel", "mass_error_rel_recomputed",
        "runtime_mass_error_max", "phase_KKT_max", "numerical_hard_gates_pass",
        "long_time_research_gate_pass",
    )
    result: dict[str, object] = {
        "evidence_source": source,
        "direction": direction,
        "qualification_status": status,
    }
    for key in keys:
        result[key] = row.get(key, "") if row else ""
    result["notes"] = note
    return result


def build_reports(
    refined_analysis: Path,
    research_analysis: Path,
    support_analysis: Path,
    width_manifest: Path,
    report_root: Path,
) -> None:
    refined = one_by_direction(read_rows(refined_analysis / "jc4_long_time_planar_metrics.csv"))
    research = one_by_direction(read_rows(research_analysis / "jc4_long_time_planar_metrics.csv"))
    support = one_by_direction(read_rows(support_analysis / "jc4_long_time_planar_metrics.csv"))
    long_rows = [
        status_row(
            str(refined_analysis), refined["growth"], "growth",
            "PASS_SHORT_NUMERICAL_ONLY_DISPLACEMENT_LT_5DX",
            "zero retry/reject and hard gates pass; not a long-displacement gate",
        ),
        status_row(
            str(refined_analysis), refined["dissolution"], "dissolution",
            "PASS_SHORT_NUMERICAL_ONLY_DISPLACEMENT_LT_5DX",
            "zero retry/reject and hard gates pass; not a long-displacement gate",
        ),
        status_row(
            str(research_analysis), research["growth"], "growth",
            "FAIL_EQUAL_TIME_DT_COLLAPSE_AT_Q_LOWER_BOUND",
            "16000 accepted step labels but only 2.075% of requested physical time; 22 retries",
        ),
        status_row(
            str(support_analysis), support["growth"], "growth",
            "FAIL_SUPPORT_CUTOFF_SENSITIVITY_DID_NOT_CLOSE",
            "matrix_support_eps=1e-8 only delayed the first retry and did not prevent dt collapse",
        ),
        status_row(
            "projected_from_safe_fixed_dt_and_sharp_reference", None, "dissolution",
            "NOT_RUN_BLOCKED_LONG_TIME_THROUGHPUT",
            "640000 fixed-dt steps projected; upstream growth crossing already exposed the same bound-active solver defect",
        ),
    ]
    write_rows(report_root / "jc4_long_time_planar_metrics.csv", long_rows)

    profile_source = research_analysis / "jc4_long_time_profiles.csv"
    profiles = read_rows(profile_source)
    for row in profiles:
        row["profile_qualification"] = "FAILED_EQUAL_TIME_ATTEMPT_FINAL_ACCEPTED_STATE"
        row["actual_time_warning"] = "requested_time_not_reached_due_dt_collapse"
    write_rows(report_root / "jc4_long_time_profiles.csv", profiles)

    main_line_preconditioner = find_line(
        ROOT / "main_cuda.cu", "__global__ void ctot_local_storage_preconditioner_kernel"
    )
    main_line_solver = find_line(
        ROOT / "main_cuda.cu", "auto solve_ctot_transport_operator ="
    )
    kernel_line_reconstruct = find_line(
        ROOT / "cuda_kernels.cu", "__global__ void reconstruct_x_q_Y_from_ctot_kernel"
    )
    kernel_line_candidate = find_line(
        ROOT / "cuda_kernels.cu", "__global__ void ctot_candidate_state_from_Y_kernel"
    )
    growth = research["growth"]
    support_growth = support["growth"]
    report_root.joinpath("jc4_long_time_transport_bound_blocker.md").write_text(f"""# JC4 long-time transport bound blocker

## Observed failure

The T400 growth research-gate attempt accepted 16,000 step labels but advanced
only `{float(growth['actual_elapsed_s']):.12g} s` of the requested
`{float(growth['requested_elapsed_s']):.12g} s`. It incurred
`{growth['retry_count']}` retries and reduced `dt` from `3.125e-3` to
`7.450580596923828e-10`. The first collapse began at accepted code time
`1.028125`, after about one cell of interface motion.

Mass was not lost: the final recomputed relative error was
`{float(growth['mass_error_rel_recomputed']):.3e}` and the largest runtime
mass residual was `{float(growth['runtime_mass_error_max']):.3e}`.

## Mathematical source

The state is authoritative in

`C = h(phi) + alpha*x`, `q = C-h`, `alpha=1-h`, and `Y=logit(x)`.

The current local transport preconditioner therefore divides by

`dC/dY = alpha*x*(1-x)`.

At the observed bound crossing, an accepted cell had approximately
`alpha=1.637e-9`, `q=1.110e-16`, `x=6.780e-8`, hence
`dC/dY approximately 1.110e-16`. The physical lower bound is `q=0`, but the
transport nonlinear solve has no lower-bound active-set representation. The
logit coordinate approaches negative infinity and the local correction becomes
ill-conditioned; line search rejects even while Ctot mass remains closed.

## Code trace

- `main_cuda.cu:{main_line_preconditioner}` computes `dC_dY=alpha*x*(1-x)` and `residual/dC_dY`.
- `main_cuda.cu:{main_line_solver}` runs the Y-space nonlinear iteration and line search.
- `cuda_kernels.cu:{kernel_line_reconstruct}` reconstructs `Y=log(x/(1-x))` whenever alpha exceeds the support cutoff.
- `cuda_kernels.cu:{kernel_line_candidate}` maps every active trial back through sigmoid without a q-lower-bound PDAS state.

## Rejected workaround

Raising `ctot_matrix_support_eps` from `1e-10` to `1e-8` moved the first
retry only from code time `1.028125` to `1.046875`; the run still froze near
code time `1.063014`, with `{support_growth['retry_count']}` retries. Therefore
support-cutoff tuning is rejected.

## Classification

This is a numerical bound-active nonlinear-solver defect, not a finite-interface
physics error, mobility-calibration error, mass-closure error, or evidence that
the Ji-Chen physical mechanism is wrong. A q/C lower-bound active-set transport
solve is required before equal-time long-displacement qualification.

`root_cause_status=PASS_LONG_TIME_BLOCKER_ATTRIBUTED_TO_MISSING_Q_LOWER_BOUND_ACTIVE_SET`
""", encoding="utf-8")

    report_root.joinpath("jc4_long_time_planar_validation.md").write_text(f"""# JC4 long-time planar validation

## Verdict

`fixed_ctot_long_time_planar_status=BLOCKED_LONG_TIME_THROUGHPUT`

The short fixed-dt T400 growth and dissolution runs pass mass, bounds, KKT,
energy/work, and no-clipping/no-projection gates, but their displacements are
below `5 dx`. The attempted long growth run is not an equal-time result:
its actual/requested time ratio is `{float(growth['actual_to_requested_time_ratio']):.6%}`.
It therefore cannot be used for a long-time trajectory or phase-fraction claim.

The failure is traced in `jc4_long_time_transport_bound_blocker.md`. The
standalone Ji-Chen and sharp oracles remain valid, but the current runtime
transport nonlinear solver cannot cross the q=0 bound robustly. Dissolution is
not launched because its projected 640,000-step fixed-dt run would not repair
the already demonstrated bound defect.

No long-time fixed-Ctot PASS is claimed.
""", encoding="utf-8")

    width = json.loads(width_manifest.read_text(encoding="utf-8"))
    width_rows = []
    for case in width["cases"]:
        width_rows.append({
            "case_id": case["case_id"],
            "direction": case["direction"],
            "grid": "x".join(map(str, case["grid"])),
            "dx_nm": case["dx_nm"],
            "lambda_nm": case["lambda_nm"],
            "planned_dt_code": case["dt_code"],
            "planned_steps": case["nsteps"],
            "runtime_status": "NOT_RUN_UPSTREAM_LONG_PLANAR_GATE_BLOCKED",
            "acceptance_status": "NOT_EVALUATED",
        })
    write_rows(report_root / "jc4_width_grid_metrics.csv", width_rows)
    report_root.joinpath("jc4_width_grid_sensitivity.md").write_text("""# JC4 width/grid sensitivity

The equal-time matrix is prepared, including fixed-lambda grid refinement and
lambda 3/4/5 nm comparisons. Runtime execution is intentionally gated because
the production lambda=4, dx=1 path has not crossed the long-planar q=0 solver
boundary. Running comparators now would not qualify the selected baseline.

`width_grid_sensitivity_status=NOT_RUN_UPSTREAM_LONG_PLANAR_GATE_BLOCKED`
""", encoding="utf-8")

    curvature_rows = []
    for radius in (5, 8, 12, 20):
        for fate in ("stationary", "growth", "dissolution"):
            curvature_rows.append({
                "radius_nm": radius,
                "case_type": fate,
                "production_resolution_cells": radius,
                "runtime_status": "NOT_RUN_UPSTREAM_LONG_PLANAR_GATE_BLOCKED",
                "acceptance_status": "NOT_EVALUATED",
            })
    write_rows(report_root / "jc4_curvature_metrics.csv", curvature_rows)
    report_root.joinpath("jc4_curvature_validation.md").write_text("""# JC4 curvature validation

The required R=5/8/12/20 nm stationary/growth/dissolution matrix is not opened.
The planar long-displacement gate is upstream and currently blocked by the
transport lower-bound solver.

`curvature_status=NOT_RUN_UPSTREAM_LONG_PLANAR_GATE_BLOCKED`
""", encoding="utf-8")
    report_root.joinpath("jc4_handoff_radius.md").write_text("""# JC4 handoff radius

The geometric policy remains hard minimum 5 nm and preferred minimum 8 nm.
Neither value is runtime-qualified by this goal because curvature validation is
gated. No sub-5-nm quantitative claim is allowed.

`minimum_resolved_beta_radius_nm=UNQUALIFIED`
""", encoding="utf-8")

    particle_rows = [
        {"case_id": "two_particle_size_competition", "runtime_status": "NOT_RUN_UPSTREAM_GATES_BLOCKED"},
        {"case_id": "three_particle_competition", "runtime_status": "NOT_RUN_UPSTREAM_GATES_BLOCKED"},
        {"case_id": "small_multiparticle_distribution", "runtime_status": "NOT_RUN_UPSTREAM_GATES_BLOCKED"},
    ]
    write_rows(report_root / "jc4_particle_timeseries.csv", particle_rows)
    report_root.joinpath("jc4_multiparticle_validation.md").write_text("""# JC4 multiparticle validation

Not run. Long planar and curvature gates are not closed, so a one-particle or
short-time result is not relabeled as coarsening evidence.

`multiparticle_status=NOT_RUN_UPSTREAM_GATES_BLOCKED`
""", encoding="utf-8")

    report_root.joinpath("jc4_formulation_decision.md").write_text("""# JC4 formulation decision

The fixed-Ctot route is not rejected on physical-equivalence evidence; it is
blocked numerically before reaching the required displacement. The independent
Ji-Chen oracle passes, but a direct `ji_chen_diagonal_logit_v1` runtime is not
implemented or selected because doing so now would bypass the authoritative
Ctot production contract without first closing the identified q-lower-bound
transaction.

`direct_Ji_Chen_runtime_required=UNRESOLVED_AFTER_BOUND_ACTIVE_CTOT_SOLVER`

`selected_research_model=NONE_UNQUALIFIED`

`formulation_decision_status=BLOCKED_FIXED_CTOT_LONG_TIME_EQUIVALENCE`
""", encoding="utf-8")

    report_root.joinpath("jc4_accuracy_and_claims.md").write_text("""# JC4 accuracy and claims

Allowed now: model definition, thermodynamic mapping, independent-oracle
limiting behavior, short-time numerical mass/KKT/energy stability, and the
identified long-time solver limitation.

Not allowed: long-time beta growth/dissolution accuracy, 5-nm displacement
equivalence, curvature accuracy, multiparticle/coarsening ranking, GP-mediated
growth, 3D throughput, or production readiness.

The 4-nm interface remains a numerical mesoscale interface and is not an
atomistic-width claim.
""", encoding="utf-8")

    gated_reports = {
        "jc4_gp_source_implementation.md": "GP source reintegration",
        "jc4_gp_pf_ledger.md": "GP+PF ledger validation",
        "jc4_gp_end_to_end.md": "GP end-to-end validation",
        "jc4_3d_grid_decision.md": "3D grid decision",
        "jc4_adaptive_BE.md": "adaptive BE",
        "jc4_BDF2.md": "BDF2",
    }
    for filename, title in gated_reports.items():
        report_root.joinpath(filename).write_text(
            f"# {title}\n\nNot run. The long-planar fixed-dt gate is upstream "
            "and remains blocked. GP/S3/RSMD stay disabled.\n\n"
            f"`status=NOT_RUN_UPSTREAM_LONG_PLANAR_GATE_BLOCKED`\n",
            encoding="utf-8",
        )
    write_rows(report_root / "jc4_3d_performance.csv", [{
        "grid": "64^3/128^3/192^3/256^3",
        "runtime_status": "NOT_RUN_UPSTREAM_LONG_PLANAR_GATE_BLOCKED",
        "formal_production": False,
    }])
    write_rows(report_root / "jc4_physical_time_throughput.csv", [{
        "path": "fixed_ctot_ji_chen_coarse4_gp_v1",
        "safe_startup_dt_code": 0.003125,
        "first_long_growth_retry_code_time": 1.028125,
        "final_effective_dt_code": 7.450580596923828e-10,
        "actual_to_requested_time_ratio": growth["actual_to_requested_time_ratio"],
        "status": "BLOCKED_LONG_TIME_THROUGHPUT",
    }])

    final_lines = [
        "fixed_ctot_baseline_preserved=true",
        "mechanics_contract_preserved=true",
        "research_model_candidate=fixed_ctot_ji_chen_coarse4_gp_v1",
        "dx_nm=1.0", "lambda_nm=4.0", "lambda_over_dx=4",
        "finite_interface_mode=off", "a_M=0",
        "Ji_Chen_1D_oracle_status=PASS_LIMITING_SUITE",
        "fixed_ctot_long_time_planar_status=BLOCKED_LONG_TIME_THROUGHPUT",
        "long_time_growth_error=NOT_COMPARABLE_EQUAL_TIME_NOT_REACHED",
        "long_time_dissolution_error=NOT_RUN",
        "final_phase_fraction_error=NOT_QUALIFIED",
        "equilibrium_composition_status=NOT_REACHED",
        "width_grid_sensitivity_status=NOT_RUN_UPSTREAM_GATE",
        "curvature_status=NOT_RUN_UPSTREAM_GATE",
        "minimum_resolved_beta_radius_nm=UNQUALIFIED",
        "multiparticle_status=NOT_RUN_UPSTREAM_GATE",
        "competition_ranking_status=NOT_EVALUATED",
        "direct_Ji_Chen_runtime_required=UNRESOLVED_AFTER_BOUND_ACTIVE_CTOT_SOLVER",
        "selected_research_model=NONE_UNQUALIFIED",
        "GP_source_reintegrated=false", "GP_PF_ledger_status=NOT_RUN",
        "GP_growth_enabled=false", "GP_coarsening_enabled=false",
        "GP_migration_enabled=false", "grid_128_status=NOT_RUN",
        "grid_192_status=NOT_RUN", "grid_256_status=NOT_RUN",
        "accepted_physical_time_per_wall_hour=NOT_QUALIFIED_DT_COLLAPSE",
        "adaptive_BE_status=NOT_RUN_GATED", "BDF2_status=NOT_RUN_GATED",
        "formal_production_executed=false", "cluster_used=false",
        "commit_created=false", "push_performed=false",
        "recommended_next_action=IMPLEMENT_Q_OR_C_LOWER_BOUND_ACTIVE_SET_TRANSPORT_SOLVER_WITHOUT_CHANGING_PHYSICS",
        "final_status=BLOCKED_LONG_TIME_THROUGHPUT",
    ]
    report_root.joinpath("jc4_final_terminal_output.txt").write_text(
        "\n".join(final_lines) + "\n", encoding="utf-8"
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--refined-analysis", type=Path, required=True)
    parser.add_argument("--research-analysis", type=Path, required=True)
    parser.add_argument("--support-analysis", type=Path, required=True)
    parser.add_argument("--width-manifest", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    build_reports(
        args.refined_analysis.resolve(), args.research_analysis.resolve(),
        args.support_analysis.resolve(), args.width_manifest.resolve(),
        args.report_root.resolve(),
    )
    print("fixed_ctot_long_time_planar_status=BLOCKED_LONG_TIME_THROUGHPUT")
    print("final_status=BLOCKED_LONG_TIME_THROUGHPUT")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
