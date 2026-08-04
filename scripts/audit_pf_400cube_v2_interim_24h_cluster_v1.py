#!/usr/bin/env python3
"""Fail-closed, read-only integrity audit through the exact 24 h checkpoint.

This is an interim extraction gate for applying the immutable transport-V2
equations to a new 400^3 PF input.  It is deliberately not the completion
audit and cannot emit the 48 h production PASS token.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path


PASS = "PASS_400CUBE_V2_INTERIM_24H_CLUSTER_EXTRACTION_GATE_V1"
FINAL_STEP = 65393
CADENCE = 3633
EXPECTED_CHECKPOINTS = sorted(set(range(CADENCE, FINAL_STEP, CADENCE)) | {FINAL_STEP})


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--production-root", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)

    root = args.production_root.resolve()
    campaign_path = root / "provenance/campaign_manifest.json"
    fixture_path = root / "provenance/fixture_manifest.json"
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    failures: list[str] = []

    if campaign.get("schema") != "PF_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_CAMPAIGN_V1":
        failures.append("campaign schema mismatch")
    if campaign.get("grid") != [400, 400, 400] or campaign.get("dx_nm") != 1.0:
        failures.append("grid contract mismatch")
    if campaign.get("elasticity_enabled") is not True:
        failures.append("elasticity is not enabled")
    for key in (
        "GP_enabled",
        "GP_birth_enabled",
        "GP_release_enabled",
        "external_source_enabled",
        "new_beta_nucleation_enabled",
    ):
        if campaign.get(key) is not False:
            failures.append(f"prohibited path enabled: {key}")
    if sha256(fixture_path) != campaign.get("fixture_manifest_sha256"):
        failures.append("fixture hash does not match campaign")
    if fixture.get("component_contract", {}).get("actual_count") != 64:
        failures.append("initial population is not exactly 64")

    checkpoint_hashes: dict[str, str] = {}
    segment_metrics: dict[str, dict[str, float | int | str]] = {}
    for index, step in enumerate(EXPECTED_CHECKPOINTS):
        segment = root / "segments" / f"step_{step}"
        checkpoint = root / "checkpoints" / f"step_{step}.chk"
        required = [
            segment / "status.txt",
            segment / "stdout.log",
            segment / "stderr.log",
            segment / "checkpoint.sha256",
            segment / "dynamics_mass_diagnostics.csv",
            checkpoint,
        ]
        missing = [str(path) for path in required if not path.is_file()]
        if missing:
            failures.append(f"step {step}: missing {missing}")
            continue
        if (segment / "status.txt").read_text(encoding="utf-8").strip() != (
            "PASS_400CUBE_FIXED_XB03_ELASTIC_PSD_SIGN_SEGMENT_V1"
        ):
            failures.append(f"step {step}: segment is not exact PASS")
        if (segment / "stderr.log").stat().st_size:
            failures.append(f"step {step}: nonempty stderr")
        stdout = (segment / "stdout.log").read_text(encoding="utf-8", errors="replace")
        for token in (
            "PF_ZERO_MODE_FINAL_AUDIT status=PASS",
            "elastic_solver_status=PASS_ELASTIC_WARM_START_RESIDUAL_V1",
            "elastic_nonconverged_steps=0",
        ):
            if token not in stdout:
                failures.append(f"step {step}: missing {token}")
        if index and "checkpoint_restart_provenance=RESTORED_AND_VALIDATED" not in stdout:
            failures.append(f"step {step}: restart provenance is not restored and validated")
        actual = sha256(checkpoint)
        checkpoint_hashes[str(step)] = actual
        registered = (segment / "checkpoint.sha256").read_text(encoding="utf-8").split()[0]
        if actual != registered:
            failures.append(f"step {step}: checkpoint SHA-256 mismatch")
        mass_hits = re.findall(r"final_mass_code=([0-9.eE+-]+)", stdout)
        residual_hits = re.findall(r"elastic_last_residual=([0-9.eE+-]+)", stdout)
        iteration_hits = re.findall(r"elastic_mean_iterations=([0-9.eE+-]+)", stdout)
        segment_metrics[str(step)] = {
            "checkpoint_sha256": actual,
            "checkpoint_size_bytes": checkpoint.stat().st_size,
            "final_mass_code": float(mass_hits[-1]) if mass_hits else "NOT_PARSED",
            "elastic_last_residual": float(residual_hits[-1]) if residual_hits else "NOT_PARSED",
            "elastic_mean_iterations": float(iteration_hits[-1]) if iteration_hits else "NOT_PARSED",
        }

    audit = {
        "schema": "PF_400CUBE_V2_INTERIM_24H_CLUSTER_EXTRACTION_AUDIT_V1",
        "status": PASS if not failures else "BLOCKED_400CUBE_V2_INTERIM_24H_CLUSTER_EXTRACTION_GATE_V1",
        "scope": "read-only integrity through exact 24 h step; not a 48 h completion audit",
        "production_root": str(root),
        "production_root_status": (root / "status.txt").read_text(encoding="utf-8").strip(),
        "exact_24h_step": FINAL_STEP,
        "checkpoint_count_through_24h": len(checkpoint_hashes),
        "checkpoint_hashes": checkpoint_hashes,
        "segment_metrics": segment_metrics,
        "input_sha256": {
            "campaign_manifest": sha256(campaign_path),
            "fixture_manifest": sha256(fixture_path),
        },
        "failures": failures,
    }
    (args.out / "audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (args.out / "status.txt").write_text(audit["status"] + "\n", encoding="utf-8")
    if failures:
        raise SystemExit("[fatal] " + "; ".join(failures))
    print(PASS)


if __name__ == "__main__":
    main()
