#!/usr/bin/env python3
"""Emit the smallest-fixture PF smoke-test gate without bypassing authority.

This wrapper intentionally has no CUDA launch path.  While the thermodynamic
contract is blocked or the four-bucket state is partial, invoking a PF binary
would produce a result that cannot be interpreted as the requested coupling.
The gate retains the exact fixture and required future observables instead.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

ROOT = Path(__file__).resolve().parents[1]


def _sha256(path: Path) -> str:
    """Return SHA-256 for a declared local provenance input."""

    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_csv(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    """Write a small stable CSV used by the final acceptance report."""

    rows_list = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_list[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows_list)


def _gate_status(contract_text: str, handoff_status: str) -> str:
    """Return the first authority gate that prevents a PF executable run."""

    if "BLOCKED_CONTRACT_CONFLICT" in contract_text:
        return "NOT_RUN_P0_CONTRACT_CONFLICT"
    if handoff_status == "PARTIAL_PF_STATE_NOT_CLOSED":
        return "NOT_RUN_PARTIAL_PF_STATE_NOT_CLOSED"
    return "NOT_RUN_NO_QUALIFIED_PF_EXECUTION_PATH"


def main() -> int:
    """Write a gated smoke-test record for the smallest eligible PF fixture."""

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--fixture",
        default=str(
            ROOT
            / "data/qualification/pf_mass_conserving_library_handoff_v1/six_particle_96cube_spec.json"
        ),
    )
    parser.add_argument(
        "--contract",
        default=str(ROOT / "thermodynamic_contract_freeze_v1/07_frozen_thermodynamic_contract.yaml"),
    )
    parser.add_argument(
        "--handoff-audit",
        default=str(ROOT / "outputs/kwn_pf_mvp_v1/handoff_audit.json"),
    )
    parser.add_argument(
        "--output",
        default=str(ROOT / "outputs/kwn_pf_mvp_v1/pf_smoke_trajectories.csv"),
    )
    parser.add_argument(
        "--gate-output",
        default=str(ROOT / "outputs/kwn_pf_mvp_v1/pf_smoke_gate.json"),
    )
    parser.add_argument(
        "--report",
        default=str(ROOT / "reports/kwn_pf_mvp_v1/06_pf_smoke_test.md"),
    )
    args = parser.parse_args()
    fixture_path = Path(args.fixture)
    contract_path = Path(args.contract)
    audit_path = Path(args.handoff_audit)
    if not fixture_path.is_file() or not contract_path.is_file() or not audit_path.is_file():
        raise FileNotFoundError("fixture, contract, and audited KWN handoff must exist")
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    contract_text = contract_path.read_text(encoding="utf-8")
    status = _gate_status(contract_text, str(audit["status"]))
    target = fixture["target"]
    _write_csv(
        Path(args.output),
        [
            {
                "status": status,
                "time_h": time_h,
                "N_beta_m3": "",
                "mean_radius_nm": "",
                "matrix_xB_alpha": "",
                "mean_C_B_tot": "",
                "new_beta_nucleation": "OFF_REQUIRED",
                "GP": "OFF_REQUIRED",
                "source": "NOT_RUN",
            }
            for time_h in (0.0, 6.0, 48.0)
        ],
    )
    gate: Dict[str, Any] = {
        "status": status,
        "pf_executable_invoked": False,
        "fixture_id": fixture["fixture_id"],
        "fixture_path": str(fixture_path.relative_to(ROOT)),
        "fixture_sha256": _sha256(fixture_path),
        "fixture_grid": target["grid"],
        "fixture_temperature_C": target["temperature_C"],
        "fixture_lambda_sm_nm": target["lambda_sm_nm"],
        "handoff_audit_status": audit["status"],
        "thermodynamic_contract_path": str(contract_path.relative_to(ROOT)),
        "required_future_checks": [
            "0/6/48 h N_beta, mean radius, matrix xB_alpha, and total B inventory continuity",
            "new beta nucleation OFF",
            "GP, GP birth, GP release, external source, and GP-assisted beta modes OFF",
            "particle identity fail-closed on unresolved merge/split events",
        ],
    }
    gate_output = Path(args.gate_output)
    gate_output.parent.mkdir(parents=True, exist_ok=True)
    gate_output.write_text(json.dumps(gate, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    report = Path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        "# PF smoke-test gate\n\n"
        f"Status: `{status}`\n\n"
        "No PF/CUDA executable was invoked.  The selected target is the smallest existing conditional fixture: "
        f"`{fixture['fixture_id']}` ({target['grid'][0]}³, {target['temperature_C']} °C, "
        f"lambda={target['lambda_sm_nm']} nm).  It is validation-only.\n\n"
        "The run is blocked first by `BLOCKED_CONTRACT_CONFLICT`, and independently by the handoff state's "
        f"`{audit['status']}` status.  Therefore the blank trajectory cells are intentionally not treated as zeroes "
        "or failed physical observations.\n\n"
        "A future smoke test must preserve the existing diffuse profiles and `delta_C_relaxation`, start from the "
        "qualified fixture only, keep all prohibited birth/GP mechanisms off, and report continuity at 0, 6, and 48 h.\n",
        encoding="utf-8",
    )
    print(json.dumps({"status": status, "pf_executable_invoked": False}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
