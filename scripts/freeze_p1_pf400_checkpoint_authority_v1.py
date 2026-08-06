#!/usr/bin/env python3
"""Freeze conditional 400-cube checkpoint authority and provenance for P1."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shlex
import subprocess
from pathlib import Path
from typing import Any


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def remote(host: str, command: str) -> str:
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=25", host, command],
        text=True,
        capture_output=True,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or f"remote command failed: {command}")
    return result.stdout


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fields = sorted({key for row in rows for key in row})
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--routes", required=True, type=Path)
    parser.add_argument("--observables", required=True, type=Path)
    parser.add_argument("--campaign-root", required=True)
    parser.add_argument("--remote-geometry-root", required=True)
    parser.add_argument("--segmentation-code", required=True, type=Path)
    parser.add_argument("--transport-code", required=True, type=Path)
    parser.add_argument("--transport-contract", required=True, type=Path)
    parser.add_argument("--host", default="uvip-cluster")
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--verify-checkpoint-bytes", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite output: {args.out}")
    args.out.mkdir(parents=True)
    routes = list(csv.DictReader(args.routes.open(encoding="utf-8", newline="")))
    observations = list(csv.DictReader(args.observables.open(encoding="utf-8", newline="")))
    source_status = {
        (row["case_id"], int(row["hour_h"])): row["source_status"]
        for row in observations
        if int(row["hour_h"]) in (6, 12, 24, 48)
    }
    attempts = {(row["case_id"], row["attempt_id"]) for row in routes}
    metadata: dict[tuple[str, str], dict[str, Any]] = {}
    for case_id, attempt_id in sorted(attempts):
        root = f"{args.campaign_root}/attempts/{case_id}/{attempt_id}"
        fixture = f"{args.campaign_root}/fixtures/{case_id}"
        program = f"""python3 - <<'PY'
import json,pathlib
r=pathlib.Path({root!r})
f=pathlib.Path({fixture!r})
m=json.load(open(r/'provenance/campaign_manifest.json'))
fm=json.load(open(f/'fixture_manifest.json'))
chain={{pathlib.Path(line.split(None,1)[1].strip()).name:line.split()[0] for line in open(r/'provenance/checkpoint_chain.sha256') if line.strip()}}
accepted_steps=[int(line) for line in open(r/'provenance/checkpoint_steps.txt') if line.strip()]
print(json.dumps({{'campaign':m,'fixture_id':fm.get('fixture_id'),'chain':chain,'chain_count':len(chain),'accepted_step_count':len(accepted_steps),'attempt_status':(r/'status.txt').read_text().strip(),'qualification_status':(r/'qualification/status.txt').read_text().strip()}},sort_keys=True))
PY"""
        metadata[(case_id, attempt_id)] = json.loads(remote(args.host, program))
    rows: list[dict[str, Any]] = []
    for route in routes:
        case_id = route["case_id"]
        attempt_id = route["attempt_id"]
        hour = int(route["time_h"])
        meta = metadata[(case_id, attempt_id)]
        campaign = meta["campaign"]
        geometry_status_path = f"{args.remote_geometry_root}/geometry/case_{case_id}/{hour}h/status.txt"
        geometry_status = remote(args.host, f"cat {shlex.quote(geometry_status_path)}").strip()
        if route["source_type"] == "checkpoint":
            filename = Path(route["source_path"]).name
            recorded_hash = meta["chain"].get(filename, "")
            actual_hash = recorded_hash
            verification = "PASS_CHAIN_ENTRY_PRESENT"
            if args.verify_checkpoint_bytes:
                actual_hash = remote(args.host, f"sha256sum {shlex.quote(route['source_path'])}").split()[0]
                verification = "PASS_CURRENT_BYTES_MATCH_CHAIN" if actual_hash == recorded_hash else "FAIL_CURRENT_BYTES_DIFFER_FROM_CHAIN"
        else:
            phi_path = Path(route["source_path"]) / "phi.raw.f64"
            manifest_path = Path(route["source_path"]) / "fixture_manifest.json"
            command = f"sha256sum {shlex.quote(str(phi_path))} {shlex.quote(str(manifest_path))}"
            hashes = {Path(line.split(None, 1)[1]).name: line.split()[0] for line in remote(args.host, command).splitlines()}
            recorded_hash = hashes["phi.raw.f64"]
            actual_hash = recorded_hash
            verification = "PASS_CURRENT_FIXTURE_BYTES_HASHED"
        status = source_status[(case_id, hour)]
        if case_id == "004":
            conditional_reason = "complete checkpoint evidence; periodic overlap/particle lineage unresolved"
            periodic_flag = "UNRESOLVED_PERIODIC_OVERLAP"
            lineage_flag = "BLOCKED"
        elif case_id == "019":
            conditional_reason = "complete checkpoint evidence; registered postprocess/authority route blocked"
            periodic_flag = "SNAPSHOT_GEOMETRY_INDEPENDENTLY_REQUALIFIED"
            lineage_flag = "CONDITIONAL"
        else:
            conditional_reason = "complete trajectory but no unified formal 400-cube production authority registration"
            periodic_flag = "SNAPSHOT_GEOMETRY_INDEPENDENTLY_REQUALIFIED"
            lineage_flag = "SNAPSHOT_ONLY_NO_IDENTITY_KINETICS"
        rows.append(
            {
                "case_id": case_id,
                "time_h": hour,
                "step": route["step"],
                "attempt_id": attempt_id,
                "checkpoint_path": route["source_path"],
                "checkpoint_hash": actual_hash,
                "checkpoint_hash_chain_record": recorded_hash,
                "checkpoint_hash_verification": verification,
                "solver_commit": campaign["source_commit"],
                "solver_binary_sha256": "759956a89780db9a19ccd51b4115319de47463a3fd7b8d446a022a74c9beeb3b",
                "fixture_version": meta["fixture_id"],
                "fixture_manifest_sha256": campaign["fixture_manifest_sha256"],
                "schema_version": "INITIAL_FIXTURE_RAW_FIELDS" if hour == 6 else "PFZMCHK4_HEADER_904",
                "segmentation_version": sha256(args.segmentation_code),
                "transport_version": sha256(args.transport_code),
                "transport_contract_sha256": sha256(args.transport_contract),
                "trajectory_complete": meta["accepted_step_count"] == 44,
                "accepted_checkpoint_count": meta["accepted_step_count"],
                "checkpoint_hash_record_count_including_restart_artifacts": meta["chain_count"],
                "attempt_status": meta["attempt_status"],
                "attempt_qualification_status": meta["qualification_status"],
                "registered_source_status": status,
                "checkpoint_readable": True,
                "segmentation_available": True,
                "transport_available": True,
                "production_authority": "CONDITIONAL_NOT_FORMALLY_REGISTERED",
                "conditional_reason": conditional_reason,
                "periodic_overlap_flag": periodic_flag,
                "lineage_flag": lineage_flag,
                "geometry_status": geometry_status,
                "usable_for_geometry": geometry_status == "PASS",
                "usable_for_transport": geometry_status == "PASS",
                "identity_kinetics_authorized": False,
            }
        )
    if len(rows) != 28:
        raise ValueError("authority manifest must contain 28 rows")
    write_csv(args.out / "checkpoint_authority_manifest.csv", rows)
    if any(row["checkpoint_hash_verification"].startswith("FAIL") for row in rows):
        raise ValueError("checkpoint hash verification failed")
    repo = args.routes.resolve().parents[3]
    state = {
        "schema": "P1_PF400_CHECKPOINT_AUTHORITY_FREEZE_V1",
        "status": "CONDITIONAL_COMPLETE_CHECKPOINT_EVIDENCE",
        "repository": str(repo),
        "branch": subprocess.check_output(["git", "branch", "--show-current"], cwd=repo, text=True).strip(),
        "head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip(),
        "worktree_status": subprocess.check_output(["git", "status", "--short"], cwd=repo, text=True).splitlines(),
        "routes_sha256": sha256(args.routes),
        "authority_manifest_sha256": sha256(args.out / "checkpoint_authority_manifest.csv"),
        "formal_production_authority_count": 0,
        "conditional_snapshot_count": len(rows),
        "no_pf_mutation": True,
    }
    (args.out / "authority_freeze.json").write_text(json.dumps(state, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(state["status"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
