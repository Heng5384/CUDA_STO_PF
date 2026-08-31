#!/usr/bin/env python3
"""Build one validation-only, four-bucket fixture-conditioned v2 handoff.

The script only writes below ``outputs/kwn_pf_state_closure_v1``.  It never
modifies the legacy v1 handoff, source fixture, raw profile fields or PF/CUDA
inputs.  When the host-only frozen profile library is unavailable it can make a
clearly labelled deterministic *synthetic storage control* instead; that
fallback is intentionally not historical or production evidence.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from coupling.fixture_conditioned_handoff_v2 import (  # noqa: E402
    FixtureConditionedHandoffError,
    build_fixture_conditioned_handoff_v2,
    load_host_96cube_fixture,
    load_validation_contract,
    make_synthetic_fixture_control,
    read_fixture_conditioned_handoff_v2,
    write_fixture_conditioned_handoff_v2,
)


OUTPUT_ROOT = ROOT / "outputs" / "kwn_pf_state_closure_v1"
DEFAULT_HOST_PROFILE_ROOT = (
    ROOT.parent
    / "CUDA_STO_PF"
    / "data"
    / "qualification"
    / "pf_elastic_target_profile_quarter_nm_v2"
    / "profiles"
)


def _inside_output_root(path: Path) -> bool:
    try:
        path.resolve().relative_to(OUTPUT_ROOT.resolve())
        return True
    except ValueError:
        return False


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--contract",
        type=Path,
        default=ROOT / "contracts" / "pf_kwn_validation_contract_v1.json",
    )
    parser.add_argument(
        "--fixture-spec",
        type=Path,
        default=ROOT
        / "data"
        / "qualification"
        / "pf_mass_conserving_library_handoff_v1"
        / "six_particle_96cube_spec.json",
    )
    parser.add_argument(
        "--profile-root",
        type=Path,
        default=DEFAULT_HOST_PROFILE_ROOT,
        help="read-only host path containing R8p0/R9p5/R10p5 frozen profiles",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=OUTPUT_ROOT / "kwn_pf_handoff_fixture_conditioned_v2",
    )
    parser.add_argument(
        "--gp-fraction-of-source-matrix",
        type=float,
        default=0.01,
        help="prescribed storage-control fraction; it is not a GP prediction",
    )
    parser.add_argument(
        "--beta-subgrid-fraction-of-source-matrix",
        type=float,
        default=0.0,
    )
    parser.add_argument("--gp-xB", type=float, default=0.03)
    parser.add_argument("--beta-subgrid-xB", type=float, default=1.0)
    parser.add_argument(
        "--no-synthetic-fallback",
        action="store_true",
        help="fail if host raw profile fields are not available instead of emitting a synthetic control",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not _inside_output_root(args.out):
        raise SystemExit(
            f"refusing output outside validation result root: {args.out}"
        )
    contract = load_validation_contract(args.contract)
    fixture_error = None
    try:
        fixture = load_host_96cube_fixture(
            args.fixture_spec, args.profile_root, contract
        )
    except FixtureConditionedHandoffError as error:
        if args.no_synthetic_fallback:
            raise SystemExit(f"host fixture unavailable: {error}") from error
        fixture_error = str(error)
        fixture = make_synthetic_fixture_control(contract, args.fixture_spec)
    metadata, arrays, _ = build_fixture_conditioned_handoff_v2(
        fixture,
        contract,
        gp_fraction_of_source_matrix=args.gp_fraction_of_source_matrix,
        beta_subgrid_fraction_of_source_matrix=args.beta_subgrid_fraction_of_source_matrix,
        gp_x_b=args.gp_xB,
        beta_subgrid_x_b=args.beta_subgrid_xB,
    )
    metadata["build_context"] = {
        "host_fixture_attempt_error": fixture_error,
        "synthetic_fallback_used": fixture_error is not None,
        "pf_execution": "NOT_RUN_STORAGE_PACKAGE_ONLY",
    }
    paths = write_fixture_conditioned_handoff_v2(
        args.out, metadata, arrays, fixture, contract
    )
    restored, _arrays, report = read_fixture_conditioned_handoff_v2(
        args.out, fixture, contract
    )
    print(
        json.dumps(
            {
                "status": report["status"],
                "contract_hash": contract.contract_hash,
                "fixture_source_kind": fixture.source_kind,
                "fixture_hash": fixture.fixture_hash,
                "package_hash": restored["package_hash"],
                "relative_residual": report["relative_residual"],
                "nonzero_GP_mol": report["ledger"]["Q_B_GP_mol"],
                "fixed_resolved_beta_mol": report["ledger"]["Q_B_beta_resolved_fixed_mol"],
                "double_count_check": report["double_count_check"]["status"],
                "paths": {name: str(path) for name, path in paths.items()},
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

