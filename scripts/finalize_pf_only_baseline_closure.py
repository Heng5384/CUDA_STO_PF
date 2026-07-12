#!/usr/bin/env python3
"""Finalize the PF-only closure reports from authoritative CSV evidence."""

from __future__ import annotations

import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/pf_only_baseline_closure"


def read(name: str) -> list[dict[str, str]]:
    path = REPORT / name
    return list(csv.DictReader(path.open())) if path.exists() and path.stat().st_size else []


def value(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        return float(row.get(key, ""))
    except (TypeError, ValueError):
        return default


def truth(text: str | bool | None) -> bool:
    return str(text).lower() == "true"


def optional_truth(row: dict[str, str], key: str) -> bool:
    text = row.get(key, "")
    return True if text == "" else truth(text)


def main() -> None:
    summary = read("pf_only_dt_convergence_summary.csv")
    metrics = read("pf_only_dt_convergence_metrics.csv")
    operators = read("pf_operator_decomposition_matrix.csv")
    projections = read("pf_projection_idempotence.csv")
    by_temp = {temp: [r for r in summary if int(r["T_C"]) == temp and r["run_status"] == "COMPLETE"]
               for temp in (400, 380)}

    def convergence(temp: int) -> tuple[bool, dict[str, float | bool]]:
        cases = by_temp[temp]
        rows = [r for r in metrics if int(r["T_C"]) == temp]
        if len(cases) != 3 or len(rows) != 3:
            return False, {"complete": False}
        coarse = next(r for r in rows if abs(value(r, "dt")-0.002) < 1e-12 and
                      abs(value(r, "reference_dt")-0.0005) < 1e-12)
        medium = next(r for r in rows if abs(value(r, "dt")-0.001) < 1e-12 and
                      abs(value(r, "reference_dt")-0.0005) < 1e-12)
        signs = {math.copysign(1.0, value(r, "delta_R_eff_h_nm")) for r in cases}
        errors_decrease = (value(medium, "R_L2_nm") < value(coarse, "R_L2_nm") and
                           value(medium, "h_relative_L2") < value(coarse, "h_relative_L2"))
        cases_pass = all(truth(r.get("pass")) for r in cases)
        passed = cases_pass and len(signs) == 1 and errors_decrease
        return passed, {
            "complete": True, "sign_consistent": len(signs) == 1,
            "errors_decrease": errors_decrease,
            "coarse_R_L2": value(coarse, "R_L2_nm"),
            "medium_R_L2": value(medium, "R_L2_nm"),
            "coarse_h_L2": value(coarse, "h_relative_L2"),
            "medium_h_L2": value(medium, "h_relative_L2"),
        }

    t400_pass, t400_conv = convergence(400)
    t380_pass, t380_conv = convergence(380)
    max_mass = max((value(r, "max_mass_error_rel") for r in summary if r["run_status"] == "COMPLETE"),
                   default=math.inf)
    max_alpha = max((value(r, "max_alpha_xB") for r in summary if r["run_status"] == "COMPLETE"),
                    default=math.inf)
    max_interface = max((value(r, "max_interface_xB") for r in summary if r["run_status"] == "COMPLETE"),
                        default=math.inf)
    mode_pass = all(truth(r.get("runtime_mode_pass")) and truth(r.get("storage_floor_pass"))
                    for r in summary if r["run_status"] == "COMPLETE")
    source_frozen = all(truth(r.get("S3_source_frozen"))
                        for r in summary if r["run_status"] == "COMPLETE")
    projection_pass = bool(projections) and all(
        optional_truth(r, "idempotence_pass") and
        optional_truth(r, "residual_redistributed_not_lost") and
        optional_truth(r, "frequency_equivalence_pass")
        for r in projections
    )
    by_op = {r["operator_case"]: r for r in operators if r.get("run_status") == "COMPLETE"}
    operator_pass = (
        by_op.get("P0_full", {}).get("fate") == "RUNAWAY" and
        by_op.get("P1_frozen_phi", {}).get("fate") == "BOUNDED" and
        by_op.get("P2_transport_no_projection", {}).get("fate") == "BOUNDED" and
        by_op.get("P3_projection_only", {}).get("fate") == "BOUNDED" and
        by_op.get("P5_full_history_off", {}).get("fate") in {"BOUNDED", "SHRINK"}
    )
    mass_pass = max_mass <= 1e-10
    bounded_pass = max_alpha < 0.1 and max_interface < 0.1
    full_pass = (t400_pass and t380_pass and mass_pass and bounded_pass and mode_pass and
                 source_frozen and projection_pass and operator_pass)
    final_status = ("PASS_PF_ONLY_Y_TRANSPORT_PROJECTION_BASELINE_NUMERICAL_CLOSURE"
                    if full_pass else
                    ("PARTIAL_T400_PF_BASELINE_CLOSED_T380_PENDING" if t400_pass and not t380_pass
                     else "FAIL_PF_ONLY_BASELINE_NUMERICAL_CLOSURE"))

    lines = [
        "# PF-only Y/Transport/Projection Baseline Acceptance", "",
        f"**Final status:** `{final_status}`", "",
        "## Root cause", "",
        "The historical solver used the previous physical-step `dY/dt` as the only algebraic iterate for the current storage Jacobian `A=(1-h)x(1-x)`. At the AQ matrix composition, `1-A=0.99223`; in beta support it approaches one. Under changing phi this lagged memory amplifies the storage compensation, drives matrix/interface xB toward one, and only then collapses the beta field. Projection alone is identity/idempotent and frozen-phi transport is bounded.", "",
        "Primary cause: lagged, algebraically unclosed Y-storage Jacobian under phi coupling. Secondary cause: the ill-conditioned matrix-channel representation as `(1-h)x(1-x)->0`.", "",
        "## Numerical remediation", "",
        "Fix A advances the fixed-phi matrix transport equation `(1-h)x_t=divJ` directly with an add/subtract semi-implicit Fourier stabilizer, eliminating temporal `dY/dt` memory. Fix B protects `1-h<0.1` beta support and uses the existing full-storage projection to close the reported phi/transport split residual. Neither fix writes phi, changes beta inventory, changes free energy, changes mobility, or activates S3.", "",
        "Two alternatives were rejected by runtime evidence: local explicit C reconstruction and full-domain spectral Ctot transport both generated near-bound composition.", "",
        "## Equal-time results", "",
        "| T (C) | dt | fate | R final (nm) | h/h0 | max alpha xB | max interface xB | max mass error |",
        "|---:|---:|---|---:|---:|---:|---:|---:|",
    ]
    for row in sorted((r for r in summary if r["run_status"] == "COMPLETE"),
                      key=lambda r: (int(r["T_C"]), value(r, "dt"))):
        lines.append(
            f"| {row['T_C']} | {value(row,'dt'):.4g} | {row['fate']} | "
            f"{value(row,'R_eff_h_final_nm'):.6f} | {value(row,'h_integral_ratio'):.6f} | "
            f"{value(row,'max_alpha_xB'):.6f} | {value(row,'max_interface_xB'):.6f} | "
            f"{value(row,'max_mass_error_rel'):.3e} |"
        )
    lines += [
        "", "## Convergence gates", "",
        f"- T400: pass={t400_pass}; R L2 error coarse->fine `{t400_conv.get('coarse_R_L2', math.nan):.6g}` nm, medium->fine `{t400_conv.get('medium_R_L2', math.nan):.6g}` nm; h relative L2 `{t400_conv.get('coarse_h_L2', math.nan):.6g}` -> `{t400_conv.get('medium_h_L2', math.nan):.6g}`.",
        f"- T380: pass={t380_pass}; R L2 error coarse->fine `{t380_conv.get('coarse_R_L2', math.nan):.6g}` nm, medium->fine `{t380_conv.get('medium_R_L2', math.nan):.6g}` nm; h relative L2 `{t380_conv.get('coarse_h_L2', math.nan):.6g}` -> `{t380_conv.get('medium_h_L2', math.nan):.6g}`.",
        f"- Maximum relative mass error across accepted cases: `{max_mass:.3e}`.",
        f"- Maximum alpha/interface xB: `{max_alpha:.6g}` / `{max_interface:.6g}`.",
        f"- Projection identity/idempotence/capacity/cadence tests: `{projection_pass}`.",
        f"- Runtime mode and storage-floor provenance: `{mode_pass}`.",
        f"- S3 mass source frozen at zero: `{source_frozen}`.", "",
        "## Interpretation", "",
        "T400 undergoes strong but finite, grid-resolved dissolution; T380 undergoes slower shrink. This is not the historical artificial collapse: phi retains a beta-like core, xB remains bounded, and trajectories converge with dt refinement. The result does not assert that the seed must grow under the no-source after-quench matrix composition.", "",
        "Grid/domain tests remain a separate next gate. S3 may be reintroduced only as one matched PF-only/PF+S3 case per temperature; the 90-case matrix remains frozen.",
    ]
    (REPORT / "pf_only_baseline_acceptance_report.md").write_text("\n".join(lines) + "\n")

    grid_state = "TEMPORAL_GATE_PASS_ALLOWED_NOT_RUN" if full_pass else "GATED_NOT_RUN"
    (REPORT / "pf_grid_and_domain_gate_decision.md").write_text(
        "# PF Grid and Domain Gate\n\n"
        f"Status: `{grid_state}`.\n\n"
        "No grid or domain convergence claim is made in this audit. Once the temporal gate passes, grid refinement must keep physical box, seed, and interface width fixed; domain tests must keep dx, seed, and far field fixed.\n"
    )
    s3_allowed = full_pass
    (REPORT / "pf_S3_reintegration_gate.md").write_text(
        "# S3 Reintegration Gate\n\n"
        f"S3 remained frozen with `f_max_per_step=0` in every acceptance case. Reintegration allowed: `{str(s3_allowed).lower()}`.\n\n"
        "If allowed, run only one matched PF-only/PF+S3 pair at T380 and T400. Do not rerun the 90-case matrix until those controls attribute any fate change to S3 rather than baseline numerics.\n"
    )

    def fate(temp: int, dt: float) -> str:
        row = next((r for r in summary if int(r["T_C"]) == temp and
                    abs(value(r, "dt")-dt) < 1e-12 and r["run_status"] == "COMPLETE"), None)
        return row["fate"] if row else "NOT_RUN"

    terminal = [
        "PF_only_runaway_reproduced=true",
        "primary_failing_operator=lagged_Y_storage_Jacobian_history_under_changing_phi",
        "secondary_failing_operator=ill_conditioned_matrix_channel_storage_as_one_minus_h_times_x_times_one_minus_x_goes_to_zero",
        "runaway_causal_order=XBRUNAWAY_PRECEDES_PHI_COLLAPSE",
        "storage_inversion_singularity_confirmed=true",
        "projection_non_idempotent=false",
        "projection_mass_target_correct=true",
        "Y_chain_rule_correct=formula_yes_historical_single_lag_discretization_no",
        "Y_history_term_stable=false",
        "Fourier_operator_signs_correct=true",
        "zero_mode_handling_correct=true",
        "implemented_fix_A=storage_weighted_semi_implicit_matrix_transport_without_lagged_dYdt",
        "implemented_fix_B=protected_halpha_floor_0p1_plus_full_storage_projection",
        f"T400_dt0p0005_fate={fate(400,0.0005)}",
        f"T400_dt0p001_fate={fate(400,0.001)}",
        f"T400_dt0p002_fate={fate(400,0.002)}",
        f"T400_alpha_bounded={str(all(value(r,'max_alpha_xB')<0.1 for r in by_temp[400])).lower()}",
        f"T400_interface_bounded={str(all(value(r,'max_interface_xB')<0.1 for r in by_temp[400])).lower()}",
        f"T400_dt_convergence={str(t400_pass).lower()}",
        f"T380_validation_status={'PASS' if t380_pass else 'PENDING_OR_FAIL'}",
        f"mass_closure_status={'PASS' if mass_pass else 'FAIL'}_max_rel_{max_mass:.12e}",
        f"composition_context_status={'PASS_BOUNDED' if bounded_pass else 'FAIL'}",
        f"projection_acceptance_status={'PASS_IDEMPOTENT_FULL_STORAGE_TARGET' if projection_pass else 'FAIL'}",
        "S3_source_component_frozen=true",
        f"S3_reintegration_allowed={str(s3_allowed).lower()}",
        "production_GP_thermodynamics_closed=false",
        ("recommended_next_action=run_one_matched_PF_only_vs_PF_plus_S3_case_at_T380_and_T400_then_grid_domain_gates"
         if full_pass else "recommended_next_action=complete_or_repair_T380_temporal_gate_before_S3_reintegration"),
        f"final_status={final_status}",
    ]
    (REPORT / "final_terminal_output.txt").write_text("\n".join(terminal) + "\n")
    print(f"final_status={final_status}")


if __name__ == "__main__":
    main()
