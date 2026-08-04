#!/usr/bin/env python3
"""Assemble the read-only global density/strain audit decision."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def read_markers(path: Path) -> dict[str, str]:
    answer: dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if "=" in line:
            key, value = line.split("=", 1)
            answer[key] = value
    return answer


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.out.resolve() if args.out.is_absolute() else (root / args.out).resolve()
    # Keep the figure names named in the audit contract as stable aliases.  The
    # scan writer uses a "heatmap_" prefix for its implementation filenames;
    # the aliases are byte-for-byte copies, not regenerated figures.
    for source_stem, contract_stem in (
        ("heatmap_delta_kappa", "delta_kappa_heatmap"),
        ("heatmap_relative_change", "relative_change_heatmap"),
        ("heatmap_J_abs", "objective_heatmap"),
    ):
        for extension in (".png", ".pdf"):
            shutil.copyfile(out / f"{source_stem}{extension}", out / f"{contract_stem}{extension}")
    density = read_markers(out / "final_terminal_density_stage.txt")
    field_manifest = rows(out / "reconstructed_elastic_field_manifest.csv")
    closure = rows(out / "elastic_energy_closure.csv")
    math_summary = json.loads((out / "global_2d_non_pf_mathematical_diagnostic_summary.json").read_text())
    require_density = density["density_only_global_status"] == "NO_GO_RESOLVED_DENSITY_CONTRAST_GLOBAL_RANGE"
    require_strain = all("NOT_RECOVERABLE" in row["reconstruction_status"] for row in field_manifest)
    if not (require_density and require_strain):
        raise RuntimeError("unexpected prerequisite state for integrated decision")
    final_status = "INCONCLUSIVE_STRAIN_FIELDS_NOT_RECOVERABLE"
    report = "# Integrated resolved-particle thermal-trend decision\n\n"
    report += "## Density-contrast result\n\n"
    report += "`NO_GO_RESOLVED_DENSITY_CONTRAST_GLOBAL_RANGE`\n\n"
    report += f"The global resolved one-dimensional minimum is κ={density['resolved_radius_scan_min_kappa']} W m^-1 K^-1 at R={density['resolved_radius_scan_min_R_nm']} nm, above the 6 h diagnostic upper gate 0.8925 W m^-1 K^-1. The largest resolved coarsening-compatible positive change is {density['maximum_positive_delta_kappa']} W m^-1 K^-1, only {float(density['delta_kappa_fraction_of_experiment'])*100:.3f}% of the required 0.18 W m^-1 K^-1. No resolved pair passes both endpoint gates. The additional R<8 nm mathematical diagnostic likewise has no endpoint pair within 5%; it is not a PF-seed recommendation.\n\n"
    report += "This rules out the proposition that the observed discrepancy is cured merely by selecting a different number, mean radius, width, or tested bimodal distribution of the same resolved beta inventory under the frozen Yu density-contrast mechanism.\n\n"
    report += "## Strain-field result\n\n"
    report += "`INCONCLUSIVE_STRAIN_FIELDS_NOT_RECOVERABLE`\n\n"
    report += "The immutable checkpoint V4 files do retain packed displacement warm states, but each belongs to source field n−1. The matching n−1 phi/xB fields are absent at all registered 12–48 h endpoints, while the 6 h fixture has no assembled-cell runtime displacement. Consequently an exact energy closure and field/shell/spectrum statistics cannot be calculated without mixing time levels. That approximation was rejected. The recorded runtime elastic-energy and hydrostatic-stress extrema are scalar provenance only and do not establish a strain-scattering relaxation law.\n\n"
    report += "## Production decision\n\n"
    report += "`R1_R2_dynamic_PF_status=DEFERRED`\n\n"
    report += "Do not launch R1/R2 from this audit: the density-only scan contains no resolved feasible region, and the exact strain-field evidence required to justify a separate strain route is unavailable. A future, separately authorized production/checkpoint design would need source-field-consistent elastic snapshots (or matching prior fields plus runtime integrated energy) before a literature-constrained coherent-strain phonon-scattering model can be evaluated.\n\n"
    report += f"## Final status\n\n`{final_status}`\n"
    (out / "integrated_resolved_only_decision.md").write_text(report, encoding="utf-8")
    # This is deliberately a provenance supplement rather than a reconstruction
    # of strain.  The checkpoint audit established that exact fields are not
    # recoverable at a common time level.
    (out / "production_elastic_contract.md").write_text(
        "# Frozen production elastic contract\n\n"
        "- Authority: `reports/pf_246cube_method1_production_authority_v1`.\n"
        "- Runtime binary SHA-256: `516489b3e4dbafd6ba5876beb2858df8309fbfcbd1065d455b73f1782f5fe8f5`.\n"
        "- Main CUDA source SHA-256: `76b09b9334e1dace77b39d21ca489061ae9e5aa104f45afca3f04b47e362f9d7`.\n"
        "- CUDA kernels SHA-256: `0d36841845992e22ff4db3b4903200f49dc2b3ad1d42ab415b27477d72217f5a`.\n"
        "- Checkpoint source SHA-256: `e9c65556ab68ba231493924b4c90b36cc0c3a69430930f2b540774f6c00c6003`.\n"
        "- Parameter-file SHA-256: `ecbdd0ac070bdf5e5d214322b5248a08f5ca5dd4e0670427513f5ef977ea977a`.\n"
        "- Fixed periodic cell, zero external strain/stress, identity orientation, and eigenstrain `[0.046, -0.022, -0.017, 0, 0, 0]`.\n"
        "- Solver inputs named `S_ij`: S11=214.2857142857, S12=S13=11.9047619048, "
        "S22=S33=140.8730158730, S23=85.3174603175, S44=101.1904761905, "
        "S55=S66=27.7777777778 (frozen parameter-file units).\n"
        "- Checkpoint V4 stores displacement warm state from source field `n-1`, while stored phi/Y/xB are accepted field `n`; matching n-1 fields are absent.\n",
        encoding="utf-8",
    )
    terminal = {
        "baseline_reproduction_status": density["baseline_reproduction_status"],
        "global_radius_scan_min_R_nm": density["global_radius_scan_min_R_nm"],
        "global_radius_scan_min_kappa": density["global_radius_scan_min_kappa"],
        "resolved_radius_scan_min_R_nm": density["resolved_radius_scan_min_R_nm"],
        "resolved_radius_scan_min_kappa": density["resolved_radius_scan_min_kappa"],
        "library_range_min_R_nm": density["library_range_min_R_nm"],
        "library_range_min_kappa": density["library_range_min_kappa"],
        "original_15nm_boundary_artifact": density["original_15nm_boundary_artifact"],
        "best_absolute_R6_nm": density["best_absolute_R6_nm"],
        "best_absolute_R48_nm": density["best_absolute_R48_nm"],
        "best_absolute_kappa6": density["best_absolute_kappa6"],
        "best_absolute_kappa48": density["best_absolute_kappa48"],
        "best_absolute_J": density["best_absolute_J"],
        "maximum_positive_delta_kappa": density["maximum_positive_delta_kappa"],
        "maximum_relative_increase_percent": density["maximum_relative_increase_percent"],
        "delta_kappa_fraction_of_experiment": density["delta_kappa_fraction_of_experiment"],
        "any_resolved_endpoint_pair_within_5percent": density["any_resolved_endpoint_pair_within_5percent"],
        "density_only_global_status": density["density_only_global_status"],
        "non_pf_mathematical_endpoint_pair_within_5percent": math_summary["any_endpoint_pair_within_5percent"],
        "strain_reconstruction_status": "INCONCLUSIVE_STRAIN_FIELDS_NOT_RECOVERABLE",
        "elastic_energy_closure_max_relative": "NOT_COMPUTABLE_WITH_EXACT_TIME_LEVEL",
        "hydrostatic_variance_ratio_48h_6h": "NOT_COMPUTABLE",
        "deviatoric_variance_ratio_48h_6h": "NOT_COMPUTABLE",
        "stress_variance_ratio_48h_6h": "NOT_COMPUTABLE",
        "elastic_energy_ratio_48h_6h": "NOT_COMPUTABLE",
        "low_q_power_ratio_48h_6h": "NOT_COMPUTABLE",
        "mid_q_power_ratio_48h_6h": "NOT_COMPUTABLE",
        "high_q_power_ratio_48h_6h": "NOT_COMPUTABLE",
        "strain_relaxation_leverage_status": "INCONCLUSIVE_STRAIN_FIELDS_NOT_RECOVERABLE",
        "R1_R2_dynamic_PF_status": "DEFERRED",
        "recommended_next_action": "Do not run R1/R2 for density-only matching. Preserve the authority; a future separately authorized production contract must retain source-time-level elastic fields before evaluating a literature-constrained strain-scattering route.",
        "final_status": final_status,
    }
    (out / "final_terminal_output.txt").write_text("\n".join(f"{key}={value}" for key, value in terminal.items()) + "\n", encoding="utf-8")
    manifest = {
        "schema": "GLOBAL_RESOLVED_PSD_NO_GO_AND_STRAIN_FIELD_AUDIT_V1",
        # A manifest cannot include its own digest: a previous invocation may
        # leave the old manifest in place before this one is written.
        "outputs": {
            path.name: sha256(path)
            for path in sorted(out.iterdir())
            if path.is_file() and path.name != "analysis_manifest.json"
        },
        "density_only_status": density["density_only_global_status"],
        "strain_status": "INCONCLUSIVE_STRAIN_FIELDS_NOT_RECOVERABLE",
        "checkpoint_energy_closure_rows": len(closure),
        "final_status": final_status,
    }
    (out / "analysis_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(final_status)


if __name__ == "__main__":
    main()
