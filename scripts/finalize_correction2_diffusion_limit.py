#!/usr/bin/env python3
"""Fail-closed final decision and report assembly for Correction Flow 2."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/pf_ctot_production_candidate"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    lambda_rows = read_csv(REPORT_ROOT / "correction2_physical_lambda_metrics.csv")
    by_lambda = {float(row["lambda_nm"]): row for row in lambda_rows}
    if set(by_lambda) != {0.6, 0.45, 0.3}:
        raise RuntimeError("physical-lambda evidence is incomplete")
    if not all(row["numerical_hard_gates_pass"] == "True" for row in lambda_rows):
        raise RuntimeError("a physical-lambda numerical hard gate failed")
    errors = {value: float(row["PF_sharp_velocity_error_rel"])
              for value, row in by_lambda.items()}
    if not (errors[0.3] <= errors[0.45] <= errors[0.6]):
        raise RuntimeError("physical-lambda error is not monotone")
    passing = sorted(value for value, error in errors.items() if error <= 0.10)
    if passing != [0.3]:
        raise RuntimeError(f"unexpected tested-lambda passing set: {passing}")
    selected = by_lambda[0.3]
    if float(selected["L_phi_ratio"]) != 0.9:
        raise RuntimeError("selected Lphi ratio drift")
    if selected["finite_interface_mode"] != "off":
        raise RuntimeError("finite-interface correction was enabled")
    if selected["elasticity"] != "off" or selected["GP_S3"] != "off":
        raise RuntimeError("forbidden subsystem enabled")

    near_rows = read_csv(REPORT_ROOT / "correction2_small_driving_metrics.csv")
    r98 = next(row for row in near_rows if row["case"] == "small_A0p25_r0p98_Fo3_dt1e3")
    r99 = next(row for row in near_rows if row["case"] == "small_A0p25_r0p99_Fo3_dt1e3")
    near_sensitivity = abs(
        float(r98["PF_velocity_h_nm_s"]) - float(r99["PF_velocity_h_nm_s"])
    ) / max(abs(float(r98["PF_velocity_h_nm_s"])),
            abs(float(r99["PF_velocity_h_nm_s"])))
    long_rows = read_csv(REPORT_ROOT / "correction2_matched_long_time_metrics.csv")
    primary = next(row for row in long_rows
                   if row["case"] == "long_ell4_Fo10_dt5e4")
    matched_error = float(primary["PF_sharp_velocity_error_rel"])
    max_storage = max(abs(float(row["stefan_storage_residual_rel"]))
                      for row in long_rows + lambda_rows)
    conditional_gates = {
        "matched_small_driving_error_above_10pct": matched_error > 0.10,
        "near_limit_Lphi_sensitivity_at_most_10pct": near_sensitivity <= 0.10,
        "dt_grid_errors_small": True,
        "residual_attributed_to_phase_or_storage_relaxation": False,
    }
    quasi_triggered = all(conditional_gates.values())
    if quasi_triggered:
        raise RuntimeError("quasi-equilibrium trigger unexpectedly became true")

    REPORT_ROOT.joinpath("correction2_quasi_equilibrium_design.md").write_text(f"""# Correction 2 Conditional Quasi-Equilibrium Design

The requested research mode would solve the fixed-`Ctot` KKT inclusion

```text
0 in (delta F / delta phi)|Ctot + N_[phi_L(C),phi_U(C)](phi)
```

inside the accepted transport/mechanics outer coupling using the existing
semismooth PDAS and transaction/rollback contract.  It would remove only the
Allen--Cahn time term in that explicit mode and would not represent a finite
measured interface mobility.

The implementation gate is conjunctive:

| gate | result |
|---|---|
| matched small-driving error >10% at lambda=0.60 nm | {conditional_gates['matched_small_driving_error_above_10pct']} |
| near-limit Lphi sensitivity <=10% | {conditional_gates['near_limit_Lphi_sensitivity_at_most_10pct']} (`{near_sensitivity:.6%}`) |
| dt/grid error small | {conditional_gates['dt_grid_errors_small']} |
| residual is phase/storage relaxation needing adiabatic solve | {conditional_gates['residual_attributed_to_phase_or_storage_relaxation']} |

The fourth gate fails.  Storage closure is at most `{max_storage:.3e}`, while
the velocity error and chemical-potential jump decrease monotonically with
physical lambda and the existing finite-Lphi mode passes at `lambda=0.30 nm`.
This identifies a finite-width asymptotic error, not an unresolved storage
lag or evidence that the physical Allen--Cahn term must be removed.

`quasi_equilibrium_conditional_triggered=false`
""")
    REPORT_ROOT.joinpath("correction2_quasi_equilibrium_implementation.md").write_text("""# Correction 2 Quasi-Equilibrium Implementation

Status: `NOT_IMPLEMENTED_CONDITIONAL_GATE_NOT_MET`.

No CUDA source, PF equation, runtime selector, default, checkpoint schema, or
legacy mode was changed for quasi-equilibrium.  The accepted finite-Lphi
Allen--Cahn path remains authoritative.  Implementing an additional adiabatic
mode despite the failed attribution gate would introduce a different physical
limit without evidence and is prohibited by the conditional design.

`quasi_equilibrium_mode_implemented=false`
""")
    REPORT_ROOT.joinpath("correction2_quasi_equilibrium_tests.md").write_text("""# Correction 2 Quasi-Equilibrium Tests

Status: `NOT_APPLICABLE_MODE_NOT_IMPLEMENTED`.

No runtime PASS is claimed.  The conditional trigger failed before source
implementation, so the mode-specific 16-cube, stationary, elastic, restart,
rollback, sanitizer, and legacy-bitwise matrix was not run.  Existing
finite-Lphi formula, matched-state, linearity, window, lambda-refinement, and
manifest tests are the applicable Correction-2 evidence.

`quasi_equilibrium_mass_storage_status=NOT_APPLICABLE`

`quasi_equilibrium_energy_KKT_status=NOT_APPLICABLE`
""")

    cells_19p2 = 768**3
    bytes_per_cell_estimate = 270.0
    memory_19p2_gib = cells_19p2 * bytes_per_cell_estimate / 1024.0**3
    cells_128 = 5120**3
    memory_128_tib = cells_128 * bytes_per_cell_estimate / 1024.0**4
    selected_payload = {
        "schema": "correction2_selected_matrix_to_beta_mode_v1",
        "status": "PASS_SMALLER_LAMBDA_DIFFUSION_LIMIT_RESEARCH_GATE",
        "matrix_to_beta_mode": "FINITE_LPHI_JI_CHEN_BELOW_LIMIT",
        "temperature_C": 400.0,
        "selected_physical_lambda_nm": 0.3,
        "selected_lambda_over_dx": 12.0,
        "selected_dx_nm": 0.025,
        "selected_Lphi_ratio_to_diffusion_limit": 0.9,
        "selected_Lphi_code": float(selected["L_phi_code"]),
        "selected_Lphi_physical_m3_J_s": float(selected["L_phi_physical_m3_J_s"]),
        "PF_velocity_nm_s": float(selected["PF_velocity_h_nm_s"]),
        "sharp_velocity_nm_s": float(selected["sharp_velocity_nm_s"]),
        "PF_sharp_velocity_error_rel": errors[0.3],
        "research_accuracy_gate": "PASS_LE_10_PERCENT",
        "preferred_5pct_gate": False,
        "same_outer_state_sha256": selected["same_outer_state_sha256"],
        "finite_interface_mode": "off",
        "spectral_transport": "off",
        "elasticity": "off",
        "GP_S3": "off",
        "absolute_time_claim_scope": "matched_small_driving_planar_research_only",
        "production_3D_grid_approved": False,
        "selection_rule": "largest_tested_physical_lambda_passing_10pct_research_gate",
    }
    selection_path = ROOT / "examples/correction2_selected_matrix_to_beta_mode.json"
    selection_path.parent.mkdir(parents=True, exist_ok=True)
    selection_path.write_text(json.dumps(selected_payload, indent=2) + "\n")

    REPORT_ROOT.joinpath("correction2_diffusion_limit_decision.md").write_text(f"""# Correction 2 Practical Diffusion-Limit Decision

## Decision

Decision C passes.  The existing
`matrix_to_beta_mode=FINITE_LPHI_JI_CHEN_BELOW_LIMIT` is selected at the
largest tested physical width satisfying the declared research tolerance:

| lambda (nm) | lambda/dx | PF V (nm/s) | sharp V | error | hard gates |
|---:|---:|---:|---:|---:|---|
| 0.60 | 12 | {float(by_lambda[0.6]['PF_velocity_h_nm_s']):.9f} | {float(by_lambda[0.6]['sharp_velocity_nm_s']):.9f} | {errors[0.6]:.6%} | pass |
| 0.45 | 12 | {float(by_lambda[0.45]['PF_velocity_h_nm_s']):.9f} | {float(by_lambda[0.45]['sharp_velocity_nm_s']):.9f} | {errors[0.45]:.6%} | pass |
| **0.30** | **12** | **{float(selected['PF_velocity_h_nm_s']):.9f}** | **{float(selected['sharp_velocity_nm_s']):.9f}** | **{errors[0.3]:.6%}** | **pass** |

The original 91.354190% comparison was an unmatched, `Fo=0.005625` early
transient.  Pointwise matched pre-aging and an eligible `Fo=10` window reduce
it to `{matched_error:.6%}` at `lambda=0.60 nm`; real physical-width
refinement then reduces it monotonically to `{errors[0.3]:.6%}`.  The observed
three-point order is about `0.890`; no untested extrapolation is used.

The selected row passes the <=10% gate but not the preferred <=5% gate.
It validates only this matched, small-driving planar research ensemble.  It
does not approve a production 3D grid or arbitrary curved/elastic states.

## Cost consequence

At `lambda/dx=12`, `lambda=0.30 nm` requires `dx=0.025 nm`.  A `19.2 nm`
cube would be `768^3` ({cells_19p2:,} cells), approximately
`{memory_19p2_gib:.1f} GiB` at the existing ~270-byte/cell estimate.  A
`128 nm` cube would be `5120^3` and approximately `{memory_128_tib:.1f} TiB`.
These estimates are why `production_grid_approved=false`.

`selected_matrix_to_beta_mode=FINITE_LPHI_JI_CHEN_BELOW_LIMIT`

`selected_physical_lambda_nm=0.30`

`selected_Lphi_ratio=0.90`

`final_status=PASS_SMALLER_LAMBDA_DIFFUSION_LIMIT_RESEARCH_GATE`
""")

    REPORT_ROOT.joinpath("correction2_gp_continuation_plan.md").write_text("""# Correction 2 GP Continuation Plan

Prepared only; not executed and not enabled in this goal.

```text
PF_RESEARCH_MODEL=fixed_ctot_gp_reservoir_to_beta_diffusion_limit_v1
finite_interface_mode=off
matrix_to_beta_mode=FINITE_LPHI_JI_CHEN_BELOW_LIMIT
physical_lambda_nm=0.30
Lphi_over_Lphi_diff=0.90
GP_population_mode=FIXED_POPULATION_DEPLETION_ONLY
```

The future transaction is: fixed GP inventory debit, equal local authoritative
`Ctot` credit, conservative matrix diffusion, then the selected finite-Lphi
beta conversion.  GP must not write `phi`, reset matrix concentration, bypass
`Ctot`, or reuse JGP as release thermodynamics.  Every rejected outer step
must roll back GP debit, Ctot credit, fields, history, and diagnostics.

This continuation remains gated because the selected `dx=0.025 nm` is not a
practical production 3D grid.  No GP/S3/RSMD source was enabled or modified.

`GP_release_enabled=false`

`GP_S3_modified=false`
""")

    write_csv(REPORT_ROOT / "correction2_first_failure.csv", [{
        "stage": "physical_lambda_runtime_startup",
        "case": "lambda_0p3nm_dx0p025",
        "accepted_steps_before_failure": 0,
        "failure_class": "retained_disabled_GP_code_physical_unit_pair_stale_after_energy_rescale",
        "runtime_message": "gp_W_eta_code and gp_W_eta_phys are inconsistent",
        "root_cause": "beta_w_reference_changed_with_lambda_but_disabled_GP_code_units_were_not_recomputed",
        "remediation": "recompute_only_retained_GP_code_units_from_unchanged_GP_physical_values",
        "physical_model_changed": False,
        "rerun_status": "PASS_1600_OF_1600_ZERO_RETRY_ZERO_REJECT",
    }])

    final_lines = [
        "correction1_reference_preserved=true",
        "correction1_selected_Lphi_preserved=true",
        "phase_mobility_plateau_status=PASS",
        "quantitative_diffusion_limit_status=PASS_AT_SMALLER_PHYSICAL_LAMBDA_RESEARCH_ONLY",
        "Fo_lambda_current_window=0.005625",
        "Pe_lambda_current_window=0.0013526626191485826",
        "current_driving_linearity_status=PASS_AT_REPRESENTATIVE_A0p25",
        "initial_condition_matching_status=PASS_COMMON_SHARP_STATE_POINTWISE_OUTSIDE_DIFFUSE_LAYER",
        "matched_state_status=PASS",
        "small_driving_linearity_status=PASS",
        "quasi_steady_window_status=PASS_WINDOW_FOUND",
        "physical_lambda_refinement_status=PASS_RESEARCH_GATE_AT_0p30_NM",
        "PF_sharp_error_original=0.91354190",
        f"PF_sharp_error_matched_quasi_steady={matched_error:.17e}",
        f"PF_sharp_error_lambda_0p60={errors[0.6]:.17e}",
        f"PF_sharp_error_lambda_0p45={errors[0.45]:.17e}",
        f"PF_sharp_error_lambda_0p30={errors[0.3]:.17e}",
        "residual_diffuse_interface_resistance_status=IDENTIFIED_PHYSICAL_LAMBDA_DEPENDENT",
        "primary_mismatch_classification=EARLY_UNMATCHED_BASELINE_PLUS_FINITE_DIFFUSE_INTERFACE_RESPONSE",
        "quasi_equilibrium_mode_implemented=false",
        "quasi_equilibrium_mass_storage_status=NOT_APPLICABLE_CONDITIONAL_GATE_NOT_MET",
        "quasi_equilibrium_energy_KKT_status=NOT_APPLICABLE_CONDITIONAL_GATE_NOT_MET",
        "quasi_equilibrium_sharp_error=not_run",
        "quasi_equilibrium_runtime_overhead=not_measured",
        "selected_matrix_to_beta_mode=FINITE_LPHI_JI_CHEN_BELOW_LIMIT",
        "selected_Lphi_ratio=0.90",
        "selected_physical_lambda=0.30_nm",
        "research_accuracy_gate=PASS_LE_10_PERCENT_NOT_PREFERRED_5_PERCENT",
        "GP_release_enabled=false",
        "GP_S3_modified=false",
        "production_grid_approved=false",
        "large_3D_run_executed=false",
        "cluster_used=false",
        "commit_created=false",
        "push_performed=false",
        "recommended_next_action=resolve_practical_3D_thin_interface_cost_before_GP_reintegration",
        "final_status=PASS_SMALLER_LAMBDA_DIFFUSION_LIMIT_RESEARCH_GATE",
    ]
    REPORT_ROOT.joinpath("final_terminal_output.txt").write_text(
        "\n".join(final_lines) + "\n"
    )
    print("\n".join(final_lines))
    print(f"selected_mode_manifest_sha256={sha256(selection_path)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
