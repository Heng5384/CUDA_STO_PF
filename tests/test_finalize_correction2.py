#!/usr/bin/env python3
"""Acceptance-contract checks for the Correction-2 final decision."""

from __future__ import annotations

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_selected_mode_contract() -> None:
    path = ROOT / "examples/correction2_selected_matrix_to_beta_mode.json"
    payload = json.loads(path.read_text())
    assert payload["status"] == "PASS_SMALLER_LAMBDA_DIFFUSION_LIMIT_RESEARCH_GATE"
    assert payload["matrix_to_beta_mode"] == "FINITE_LPHI_JI_CHEN_BELOW_LIMIT"
    assert payload["selected_physical_lambda_nm"] == 0.3
    assert payload["selected_lambda_over_dx"] == 12.0
    assert payload["selected_dx_nm"] == 0.025
    assert payload["selected_Lphi_ratio_to_diffusion_limit"] == 0.9
    assert payload["PF_sharp_velocity_error_rel"] <= 0.10
    assert payload["finite_interface_mode"] == "off"
    assert payload["spectral_transport"] == "off"
    assert payload["elasticity"] == "off"
    assert payload["GP_S3"] == "off"
    assert payload["production_3D_grid_approved"] is False


def test_all_required_reports_and_evidence() -> None:
    report = ROOT / "reports/pf_ctot_production_candidate"
    required = [
        "correction2_status_reclassification.md",
        "correction2_asymptotic_window_audit.md",
        "correction2_dimensionless_windows.csv",
        "correction2_initial_condition_audit.md",
        "correction2_matched_state_construction.md",
        "correction2_small_driving_linearity.md",
        "correction2_small_driving_metrics.csv",
        "correction2_matched_long_time_planar.md",
        "correction2_matched_long_time_metrics.csv",
        "correction2_windowed_velocity.csv",
        "correction2_physical_lambda_refinement.md",
        "correction2_physical_lambda_metrics.csv",
        "correction2_resistance_decomposition.md",
        "correction2_resistance_metrics.csv",
        "correction2_quasi_equilibrium_design.md",
        "correction2_quasi_equilibrium_implementation.md",
        "correction2_quasi_equilibrium_tests.md",
        "correction2_diffusion_limit_decision.md",
        "correction2_gp_continuation_plan.md",
        "correction2_first_failure.csv",
        "final_terminal_output.txt",
    ]
    assert all((report / name).is_file() and (report / name).stat().st_size > 0
               for name in required)
    rows = list(csv.DictReader(
        (report / "correction2_physical_lambda_metrics.csv").open()
    ))
    assert len(rows) == 3
    assert all(row["numerical_hard_gates_pass"] == "True" for row in rows)
    assert all(row["artificial_shape_relaxation_below_2e3"] == "True"
               for row in rows)
    assert max(float(row["final_profile_vs_translated_equilibrium_Linf"])
               for row in rows) <= 2.0e-3
    terminal = (report / "final_terminal_output.txt").read_text()
    assert "cluster_used=false" in terminal
    assert "GP_release_enabled=false" in terminal
    assert "production_grid_approved=false" in terminal
    assert "final_status=PASS_SMALLER_LAMBDA_DIFFUSION_LIMIT_RESEARCH_GATE" in terminal


if __name__ == "__main__":
    test_selected_mode_contract()
    test_all_required_reports_and_evidence()
    print("test_finalize_correction2=PASS")
