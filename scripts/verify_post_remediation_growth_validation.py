#!/usr/bin/env python3
"""Verify that the post-remediation evidence bundle supports its stated status.

This verifier does not reclassify physics or change any runtime configuration.
It checks coverage and the bookkeeping/semantic gates that the generated report
claims to have evaluated.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


REQUIRED = (
    "post_remediation_code_path_audit.md",
    "post_remediation_runtime_semantics.csv",
    "post_remediation_validation_manifest.csv",
    "post_remediation_seed_growth_time_series.csv",
    "post_remediation_interface_supply_time_series.csv",
    "post_remediation_interface_radial_profiles.csv",
    "post_remediation_interface_coverage.csv",
    "post_remediation_phi_rhs_interface_decomposition.csv",
    "post_remediation_mass_transfer_efficiency.csv",
    "post_remediation_growth_classification_protocol.md",
    "post_remediation_growth_classification_table.csv",
    "post_remediation_source_on_off_response.csv",
    "post_remediation_T380_T400_growth_comparison.csv",
    "post_remediation_growth_summary_table.csv",
    "post_remediation_paper_level_growth_results.md",
    "post_remediation_figure_plan.csv",
    "post_remediation_acceptance_report.md",
    "final_terminal_output.txt",
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def number(value: str | None) -> float:
    try:
        return float(value or "nan")
    except ValueError:
        return math.nan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report-root", type=Path, required=True)
    args = parser.parse_args()
    root = args.report_root

    missing = [name for name in REQUIRED if not (root / name).exists()]
    summaries = rows(root / "post_remediation_growth_summary_table.csv") if not missing else []
    enabled = [row for row in summaries if number(row.get("f_max_per_step")) > 0.0]
    statuses_ok = bool(summaries) and all(row.get("status") == "EXIT 0" for row in summaries)
    direct_semantics = {row["key"]: row["value"] for row in rows(root / "post_remediation_runtime_semantics.csv")} if not missing else {}
    semantics_ok = all(direct_semantics.get(key) == "false" for key in (
        "direct_phi_write", "direct_GP_to_beta_transfer", "matrix_reset", "JGP_release_thermo_reuse"
    ))
    source_ok = bool(enabled) and all(
        row.get("source_normalization_status") == "PASS" and row.get("locality_status") == "PASS"
        for row in enabled
    )
    mass_ok = bool(summaries) and all(number(row.get("max_mass_error_rel")) <= 1.0e-10 for row in summaries)
    far_ok = bool(summaries) and all(number(row.get("max_far_field_xB")) < 0.010 for row in summaries)
    genuine = [row for row in summaries if row.get("growth_class") == "GENUINE_GROWTH"]
    temperatures = sorted({row.get("temperature_C") for row in summaries})
    final_status = next((line.split("=", 1)[1] for line in
                         (root / "final_terminal_output.txt").read_text().splitlines()
                         if line.startswith("final_status=")), "NOT_FOUND") if not missing else "NOT_FOUND"

    passed = not missing and statuses_ok and semantics_ok and source_ok and mass_ok and far_ok and bool(genuine)
    audit_status = ("PASS_EVIDENCE_BUNDLE_COMPLETE_WITH_STATED_LIMITATIONS" if passed else
                    "FAIL_EVIDENCE_BUNDLE_INCOMPLETE_OR_GATE_FAILED")
    text = "\n".join([
        "# Post-Remediation Evidence Bundle Verification",
        "",
        f"audit_status={audit_status}",
        f"reported_final_status={final_status}",
        f"required_files_missing={missing}",
        f"manifest_cases={len(summaries)}",
        f"temperatures={temperatures}",
        f"all_runtime_status_exit0={statuses_ok}",
        f"direct_write_semantics_pass={semantics_ok}",
        f"source_normalization_and_locality_pass={source_ok}",
        f"mass_closure_pass={mass_ok}",
        f"far_field_pass={far_ok}",
        f"genuine_growth_cases={[row.get('run_id') for row in genuine]}",
        "",
        "This verifier confirms report coverage and stated numerical/semantic gates. "
        "It does not convert a scenario relay into calibrated GP thermodynamics.",
        "",
    ])
    (root / "post_remediation_evidence_bundle_verification.md").write_text(text)
    print(audit_status)
    if not passed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
