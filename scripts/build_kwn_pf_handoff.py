#!/usr/bin/env python3
"""Build and audit one authority-gated, one-way KWN-to-PF handoff package.

The script deliberately writes a package and a PF initialization *plan* only.
It never materializes PF raw fields or starts CUDA: current PF state cannot
represent non-zero GP or sub-grid beta inventory without changing its physics.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coupling.handoff_audit import audit_handoff, write_handoff_audit  # noqa: E402
from coupling.kwn_pf_schema import (  # noqa: E402
    assert_roundtrip_equivalent,
    read_handoff_package,
    write_handoff_package,
)
from coupling.kwn_to_pf import adapt_kwn_to_pf, build_handoff_from_kwn_solver  # noqa: E402
from kwn_mvp.config import load_config  # noqa: E402
from kwn_mvp.solver import KWNSolver  # noqa: E402
from kwn_mvp.units import hours_to_seconds  # noqa: E402


def _sha256(path: Path) -> str:
    """Return the SHA-256 digest of one checked-in local input file."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_commit(root: Path) -> str:
    """Return the checked-out commit used as the KWN source provenance."""

    return subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=root,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write homogeneous audit rows with stable field ordering."""

    row_list = list(rows)
    if not row_list:
        raise ValueError("handoff audit rows must not be empty")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(row_list[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(row_list)


def _write_report(path: Path, audit: Mapping[str, Any], package_dir: Path) -> None:
    """Write the human-readable outcome from the machine-readable ledger audit."""

    source = audit["source_ledger"]
    destinations = audit["destination_by_bucket"]
    lines = [
        "# KWN-to-PF handoff mass audit",
        "",
        f"Status: `{audit['status']}`",
        "",
        "## Package",
        "",
        f"- schema: `kwn_pf_handoff_v1`",
        f"- package: `{package_dir.relative_to(ROOT)}`",
        "- inventory basis: `mol_B_per_m3`",
        "- PF raw initialization emitted: `false`",
        f"- PF raw initialization allowed: `{str(audit['pf_raw_initialization_allowed']).lower()}`",
        "",
        "## Four-bucket ledger",
        "",
        "| bucket | mol B m⁻³ | destination |",
        "|---|---:|---|",
    ]
    for key in ("C_B_matrix", "C_B_GP", "C_B_beta_subgrid", "C_B_beta_resolved"):
        lines.append(f"| `{key}` | {source[key]:.16e} | {destinations[key]} |")
    lines.extend(
        [
            f"| `C_B_total` | {source['C_B_total']:.16e} | source total |",
            "",
            f"Package-level relative accounting residual: `{audit['relative_accounting_residual']:.3e}`.",
            "",
            "The ledger closes only at package level when status is "
            "`PARTIAL_PF_STATE_NOT_CLOSED`: unmapped GP/sub-grid beta material remains "
            "explicitly retained in the package and is not transferred into `xB_alpha` or "
            "resolved `phi`.  This package therefore does not authorize a PF/CUDA run.",
            "",
            "## Provenance",
            "",
            "`metadata.json` records the source commit, KWN configuration hash, backend, "
            "analysis-script hash, fixture hash, and the explicit absence of a compiled KWN binary.",
        ]
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    """Run prescribed KWN to the configured handoff time and produce an audited package."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--kwn-config", default=str(ROOT / "configs/kwn/gp_prescribed_source_mvp.yaml"))
    parser.add_argument(
        "--handoff-config",
        default=str(ROOT / "configs/kwn/kwn_pf_handoff_existing_geometry.yaml"),
    )
    parser.add_argument(
        "--sampled-geometry-config",
        default=str(ROOT / "configs/kwn/kwn_pf_handoff_sampled_geometry.yaml"),
    )
    parser.add_argument(
        "--package-dir",
        default=str(ROOT / "outputs/kwn_pf_mvp_v1/kwn_pf_handoff_6h"),
    )
    parser.add_argument(
        "--audit-json",
        default=str(ROOT / "outputs/kwn_pf_mvp_v1/handoff_audit.json"),
    )
    parser.add_argument(
        "--ledger-csv",
        default=str(ROOT / "outputs/kwn_pf_mvp_v1/handoff_ledger.csv"),
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "reports/kwn_pf_mvp_v1/05_handoff_mass_audit.md"),
    )
    args = parser.parse_args()

    kwn_document = load_config(args.kwn_config)
    handoff_document = load_config(args.handoff_config)
    sampled_document = load_config(args.sampled_geometry_config)
    handoff = handoff_document.data["handoff"]
    sampled = sampled_document.data["handoff"]
    fixture = ROOT / str(handoff["pf_fixture"])
    if not fixture.is_file():
        raise FileNotFoundError(f"Declared PF fixture is missing: {fixture}")

    solver = KWNSolver.from_document(kwn_document)
    handoff_time_h = float(handoff["handoff_time_h"])
    solver.run_to_time(hours_to_seconds(handoff_time_h))
    package = build_handoff_from_kwn_solver(
        solver,
        source_git_commit=_git_commit(ROOT),
        pf_box_lengths_m=handoff["pf_box_dimensions_m"],
        observation_dataset_role="Sheskin_2018_primary_prescribed_source_numerical_exercise",
        beta_handoff_radius_m=float(sampled["handoff_radius_m"]),
        assumptions=[
            "PRESCRIBED_SOURCE_IS_NOT_A_GP_NUCLEATION_PREDICTION.",
            "PF thermodynamic contract status is P0_CONTRACT_CONFLICT; this package is not PF-run eligible.",
            "R_h is a numerical PF profile-support classification, not a physical GP-to-beta transformation radius.",
            "Existing-geometry handoff preserves authoritative resolved-beta diffuse profiles and delta_C_relaxation rather than overwriting them with a scalar matrix value.",
        ],
        resolved_shape_orientation_metadata={
            "mode": "existing_geometry",
            "fixture": str(handoff["pf_fixture"]),
            "profile_support_radii_nm": handoff["profile_support_radii_nm"],
        },
        source_binary_hash="NOT_APPLICABLE_INTERNAL_PYTHON_KWN_BACKEND",
        source_fixture_hash=_sha256(fixture),
        source_analysis_hash=_sha256(Path(__file__)),
    )
    package.metadata["source"]["additional_provenance"] = {
        "kwn_config_path": str(Path(args.kwn_config).relative_to(ROOT)),
        "kwn_config_sha256": kwn_document.sha256,
        "handoff_config_path": str(Path(args.handoff_config).relative_to(ROOT)),
        "handoff_config_sha256": handoff_document.sha256,
        "sampled_geometry_config_path": str(Path(args.sampled_geometry_config).relative_to(ROOT)),
        "sampled_geometry_config_sha256": sampled_document.sha256,
        "pf_contract_status": "P0_CONTRACT_CONFLICT",
        "pf_fixture_path": str(handoff["pf_fixture"]),
        "pf_fixture_sha256": _sha256(fixture),
    }
    package_dir = Path(args.package_dir)
    write_handoff_package(package, package_dir)
    restored = read_handoff_package(package_dir)
    assert_roundtrip_equivalent(package, restored)
    plan = adapt_kwn_to_pf(restored, mode="existing_geometry")
    audit = audit_handoff(restored, plan)
    audit_dict = audit.as_dict()
    write_handoff_audit(audit, Path(args.audit_json))
    _write_csv(
        Path(args.ledger_csv),
        [
            {
                "bucket": key,
                "inventory_mol_B_m3": value,
                "destination": audit_dict["destination_by_bucket"].get(key, "source total"),
                "pf_materialized_mol_B_m3": audit_dict["pf_materialized_inventory"].get(key, 0.0),
                "package_only_mol_B_m3": audit_dict["package_only_inventory"].get(key, 0.0),
                "status": audit_dict["status"],
            }
            for key, value in audit_dict["source_ledger"].items()
            if key != "residual"
        ],
    )
    _write_report(Path(args.report), audit_dict, package_dir)
    print(json.dumps({"status": audit.status, "package": str(package_dir)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
