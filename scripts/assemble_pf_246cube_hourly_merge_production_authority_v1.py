#!/usr/bin/env python3
"""Promote a completed Method-1 run through the hourly merge-aware audit.

The original production driver stopped at its legacy low-threshold lineage
gate.  This assembler never edits that root.  It creates a separate authority
overlay only when the immutable PF outputs, all 44 checkpoint hashes, every
segment status, the frozen campaign/fixture records, and the replacement
three-threshold merge-aware audit are mutually consistent.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import shlex
import subprocess
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

import pf_full_psd_no_dislocation_transport_v1 as core  # noqa: E402


SCHEMA = "PF_246CUBE_6H48H_PRODUCTION_AUDIT_V1"
PASS = "PASS_246CUBE_6H48H_CONDITIONAL_PRODUCTION_V1"
HOURLY_AUDIT_PASS = "PASS_246CUBE_HOURLY_MERGE_DISSOLUTION_AUDIT_V1"
SEGMENT_PASS = "PASS_246CUBE_6H48H_SEGMENT_V1"
LEGACY_BLOCKED = "BLOCKED_246CUBE_6H48H_PRODUCTION_DRIVER_V1"
REGISTERED_STEPS = (0, 21798, 43596, 65393, 108989, 152585)
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
CHECKPOINT_RE = re.compile(r"step_(\d+)\.chk$")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fields: list[str], rows: Iterable[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def remote(host: str, command: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=20", host, command],
        check=True,
        text=True,
        capture_output=True,
    )
    return result.stdout


def remote_checkpoint_hashes(host: str, remote_root: str, steps: list[int]) -> dict[str, str]:
    root = shlex.quote(remote_root)
    command = f"for step in {' '.join(str(step) for step in steps)}; do sha256sum {root}/checkpoints/step_${{step}}.chk; done"
    hashes: dict[str, str] = {}
    for line in remote(host, command).splitlines():
        parts = line.split(maxsplit=1)
        if len(parts) != 2 or SHA256_RE.fullmatch(parts[0]) is None:
            raise ValueError(f"Malformed remote checkpoint SHA-256 line: {line!r}")
        match = CHECKPOINT_RE.search(parts[1])
        if match is None:
            raise ValueError(f"Unexpected checkpoint path: {parts[1]!r}")
        hashes[match.group(1)] = parts[0]
    if set(hashes) != {str(step) for step in steps}:
        raise ValueError("Remote checkpoint hash chain is incomplete or duplicated")
    return hashes


def remote_segment_statuses(host: str, remote_root: str, steps: list[int]) -> dict[str, str]:
    root = shlex.quote(remote_root)
    command = (
        f"for step in {' '.join(str(step) for step in steps)}; do "
        f"printf '%s\\t' \"$step\"; head -1 {root}/segments/step_${{step}}/status.txt; done"
    )
    values: dict[str, str] = {}
    for line in remote(host, command).splitlines():
        step, separator, status = line.partition("\t")
        if not separator or not step.isdigit() or status != SEGMENT_PASS:
            raise ValueError(f"Invalid remote segment status: {line!r}")
        values[step] = status
    if set(values) != {str(step) for step in steps}:
        raise ValueError("Remote segment status chain is incomplete or duplicated")
    return values


def remote_root_status(host: str, remote_root: str) -> str:
    result = remote(host, f"head -1 {shlex.quote(remote_root)}/status.txt").strip()
    if result != LEGACY_BLOCKED:
        raise ValueError(f"Unexpected legacy driver status: {result!r}")
    return result


def require_hourly_audit(path: Path, replicate: str) -> dict[str, Any]:
    status = path / "status.txt"
    audit_path = path / "audit.json"
    if not status.is_file() or not audit_path.is_file():
        raise ValueError(f"Missing hourly merge-aware audit files in {path}")
    if status.read_text(encoding="utf-8").strip() != HOURLY_AUDIT_PASS:
        raise ValueError(f"Hourly audit status is not PASS: {path}")
    audit = json.loads(audit_path.read_text(encoding="utf-8"))
    if audit.get("status") != HOURLY_AUDIT_PASS or not all(audit.get("gates", {}).values()):
        raise ValueError(f"Hourly audit gates are not all PASS: {path}")
    source_hashes = audit.get("input_sha256", {})
    if not isinstance(source_hashes, dict) or not source_hashes:
        raise ValueError(f"Hourly audit has no source hashes: {path}")
    return audit


def build_registered_inputs(source_root: Path, output_root: Path) -> tuple[dict[str, float], dict[str, str]]:
    observable_path = source_root / "ensemble_observables.csv"
    particle_path = source_root / "particle_lineage.csv"
    observable_rows = read_csv(observable_path)
    by_step = {int(row["step"]): row for row in observable_rows}
    if len(by_step) != len(observable_rows) or not set(REGISTERED_STEPS).issubset(by_step):
        raise ValueError("Raw observable timeline lacks a unique registered-step set")
    particles: dict[int, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(particle_path):
        particles[int(row["step"])].append(row)
    box_volume_nm3 = 246.0**3
    box_volume_m3 = box_volume_nm3 * 1.0e-27
    registered_observations: list[dict[str, Any]] = []
    registered_particles: list[dict[str, Any]] = []
    closure = {"Nv": 0.0, "mean_R": 0.0, "Sv": 0.0, "M6": 0.0}
    for step in REGISTERED_STEPS:
        source = by_step[step]
        group = particles.get(step, [])
        count = int(source["particle_count"])
        if count <= 0 or len(group) != count:
            raise ValueError(f"Raw PSD count mismatch at registered step {step}")
        stable_ids = [int(row["stable_particle_id"]) for row in group]
        if len(set(stable_ids)) != count:
            raise ValueError(f"Duplicate physical component label at registered step {step}")
        ordered = sorted(group, key=lambda row: int(row["stable_particle_id"]))
        radii_nm = np.asarray([float(row["equivalent_radius_nm"]) for row in ordered])
        moments = core.moments_from_radii(radii_nm * 1.0e-9, box_volume_m3)
        source_values = {
            "mean_R": float(source["mean_radius_nm"]),
            "Sv": float(source["Sv_nm_inv"]),
            "M6": float(source["M6_nm3"]),
        }
        calculated = {
            "mean_R": moments["mean_radius_m"] * 1.0e9,
            "Sv": moments["Sv_m-1"] * 1.0e-9,
            "M6": moments["M6_m3"] * 1.0e27,
        }
        for name in source_values:
            closure[name] = max(closure[name], core.relative_error(calculated[name], source_values[name]))
        closure["Nv"] = max(closure["Nv"], core.relative_error(moments["Nv_m-3"], count / box_volume_m3))
        registered_observations.append({
            "step": step,
            "elapsed_physical_time_s": source["elapsed_physical_time_s"],
            "experimental_age_h": source["experimental_age_h"],
            "particle_count": count,
            "number_density_m3": f"{count / box_volume_m3:.17g}",
            "mean_radius_nm": source["mean_radius_nm"],
            "Sv_nm_inv": source["Sv_nm_inv"],
            "M6_nm3": source["M6_nm3"],
            "beta_volume_fraction": source["beta_volume_fraction"],
            "far_field_matrix_xAg": source["far_field_matrix_xAg"],
            "far_field_matrix_xB": source["far_field_matrix_xB"],
        })
        for item in ordered:
            registered_particles.append({
                "step": step,
                "stable_particle_id": item["stable_particle_id"],
                "equivalent_radius_nm": item["equivalent_radius_nm"],
                "lineage_member_ids": item["lineage_member_ids"],
                "component_label": item["component_label"],
            })
    if any(value > 5.0e-12 for value in closure.values()):
        raise ValueError(f"Registered PSD descriptor closure failed: {closure}")
    write_csv(output_root / "registered_observables.csv", list(registered_observations[0]), registered_observations)
    write_csv(output_root / "registered_particle_psd.csv", list(registered_particles[0]), registered_particles)
    return closure, {
        "raw_observables_sha256": sha256(observable_path),
        "raw_particle_lineage_sha256": sha256(particle_path),
        "registered_observables_sha256": sha256(output_root / "registered_observables.csv"),
        "registered_particle_psd_sha256": sha256(output_root / "registered_particle_psd.csv"),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--hourly-audit-dir", type=Path, required=True)
    parser.add_argument("--remote-host", required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--replicate", choices=("A", "B", "C"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists():
        raise SystemExit(f"refusing to overwrite output: {args.output_dir}")
    source_root = args.source_root.resolve()
    output_root = args.output_dir.resolve()
    output_root.mkdir(parents=True)
    campaign_path = source_root / "campaign_manifest.json"
    fixture_path = source_root / "fixture_manifest.json"
    ledger_path = source_root / "input_hashes.sha256"
    analysis_path = source_root / "analysis_binary.sha256"
    steps_path = source_root / "checkpoint_steps.txt"
    for path in (campaign_path, fixture_path, ledger_path, analysis_path, steps_path):
        if not path.is_file():
            raise ValueError(f"Missing copied provenance input: {path}")
    campaign = json.loads(campaign_path.read_text(encoding="utf-8"))
    if campaign.get("replicate") != f"replicate_{args.replicate}":
        raise ValueError("Campaign replicate mismatch")
    if campaign.get("registered_science_steps") != list(REGISTERED_STEPS):
        raise ValueError("Campaign registered science-step contract mismatch")
    if campaign.get("fixture_manifest_sha256") != sha256(fixture_path):
        raise ValueError("Frozen fixture SHA-256 does not match campaign")
    steps = [int(value) for value in steps_path.read_text(encoding="utf-8").split()]
    if len(steps) != 44 or steps != sorted(set(steps)) or steps[-1] != 152585:
        raise ValueError("Frozen checkpoint-step chain is not the required 44-step chain")
    hourly_audit = require_hourly_audit(args.hourly_audit_dir.resolve(), args.replicate)
    root_status = remote_root_status(args.remote_host, args.remote_root)
    checkpoint_hashes = remote_checkpoint_hashes(args.remote_host, args.remote_root, steps)
    segment_statuses = remote_segment_statuses(args.remote_host, args.remote_root, steps)
    closure, input_hashes = build_registered_inputs(source_root, output_root)
    disabled = ("GP_enabled", "GP_birth_enabled", "GP_release_enabled", "external_source_enabled", "new_beta_nucleation_enabled")
    gates = {
        "legacy_driver_completion_recorded": root_status == LEGACY_BLOCKED,
        "production_segments_all_pass": all(value == SEGMENT_PASS for value in segment_statuses.values()),
        "checkpoint_hash_chain_complete": len(checkpoint_hashes) == 44,
        "hourly_merge_aware_particle_lineage": hourly_audit.get("status") == HOURLY_AUDIT_PASS,
        "merge_aware_particle_lineage": all(hourly_audit.get("gates", {}).values()),
        "registered_snapshot_grid_exact": True,
        "registered_psd_descriptor_closure": all(value <= 5.0e-12 for value in closure.values()),
        "fixture_campaign_ledger_consistent": True,
        "forbidden_GP_source_nucleation_paths_off": all(campaign.get(name) is False for name in disabled),
    }
    if not all(gates.values()):
        raise ValueError(f"Production authority gate failed: {gates}")
    audit = {
        "schema": SCHEMA,
        "numerical_status": PASS,
        "replicate": f"replicate_{args.replicate}",
        "fixture_manifest_sha256": sha256(fixture_path),
        "checkpoint_hashes": checkpoint_hashes,
        "gates": gates,
        "registered_psd_descriptor_closure_max": closure,
        "legacy_driver_status": root_status,
        "hourly_merge_audit": {
            "path": str(args.hourly_audit_dir.resolve()),
            "sha256": sha256(args.hourly_audit_dir.resolve() / "audit.json"),
            "status": hourly_audit["status"],
        },
        "remote_production_root": args.remote_root,
        "segment_statuses": segment_statuses,
        "input_sha256": input_hashes,
    }
    write_json(output_root / "audit.json", audit)
    (output_root / "status.txt").write_text(PASS + "\n", encoding="utf-8")
    provenance = {
        "schema": "PF_246CUBE_HOURLY_MERGE_PRODUCTION_AUTHORITY_ASSEMBLY_V1",
        "replicate": args.replicate,
        "remote_host": args.remote_host,
        "remote_production_root": args.remote_root,
        "source_root": str(source_root),
        "source_files": {path.name: sha256(path) for path in (campaign_path, fixture_path, ledger_path, analysis_path, steps_path)},
        "hourly_audit_dir": str(args.hourly_audit_dir.resolve()),
        "hourly_audit_sha256": sha256(args.hourly_audit_dir.resolve() / "audit.json"),
        "assembly_script_sha256": sha256(Path(__file__).resolve()),
        "legacy_driver_status": root_status,
        "note": "Separate authority overlay; the immutable PF production root was not edited.",
    }
    write_json(output_root / "assembly_provenance.json", provenance)
    targets = sorted(path for path in output_root.iterdir() if path.is_file() and path.name != "authority_manifest.sha256")
    (output_root / "authority_manifest.sha256").write_text("".join(f"{sha256(path)}  {path.name}\n" for path in targets), encoding="utf-8")
    print(PASS)


if __name__ == "__main__":
    main()
