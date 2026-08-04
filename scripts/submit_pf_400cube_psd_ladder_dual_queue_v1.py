#!/usr/bin/env python3
"""Prepare and incrementally submit the dual-queue PSD ladder chains.

The cluster enforces one active job per GPU QOS and a four-job submit ceiling
on gpu_vip_24h, so the 42 logical submissions are issued incrementally while
the manifests and CSVs always describe the complete 21-case chains.
"""

from __future__ import annotations

import argparse
import csv
import datetime as dt
import hashlib
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


SCHEMA = "PF_400CUBE_PSD_SPATIAL_DENSITY_LADDER_SUBMISSION_V1"
STATIC_AUDIT_PASS = "PASS_400CUBE_PSD_CAMPAIGN_INTEGER_GATE_STATIC_AUDIT_V2"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError(f"refusing empty CSV: {path}")
    fields = list(rows[0])
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def run(command: list[str]) -> str:
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(
            f"command failed ({result.returncode}): {' '.join(command)}\n{result.stderr}"
        )
    return result.stdout.strip()


def active_jobs() -> dict[str, list[str]]:
    user = os.environ.get("USER", os.environ.get("LOGNAME", "luozhiheng"))
    raw = run(["squeue", "-h", "-u", user, "-o", "%i %P %T %j"])
    result: dict[str, list[str]] = {}
    for line in raw.splitlines():
        fields = line.split()
        if len(fields) < 4:
            continue
        job_id, partition, state, _name = fields[:4]
        if state in ("RUNNING", "PENDING"):
            result.setdefault(partition, []).append(job_id)
    return result


def job_state(job_id: str) -> str:
    try:
        raw = run(["sacct", "-j", job_id, "-X", "-n", "-P", "-o", "State"])
    except RuntimeError:
        return "UNKNOWN"
    for line in raw.splitlines():
        state = line.strip()
        if state:
            return state
    return "UNKNOWN"


def load_plan(campaign: Path) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    manifest = json.loads((campaign / "manifests/plan/campaign_case_manifest.json").read_text(encoding="utf-8"))
    cases = manifest["cases"]
    by_id = {row["case_id"]: row for row in cases}
    if len(cases) != 21 or len(by_id) != 21:
        raise ValueError("campaign manifest must contain exactly 21 unique cases")
    return cases, by_id


def fixture_passes(campaign: Path, case_id: str) -> bool:
    status = campaign / "fixtures" / case_id / "static_audit" / "status.txt"
    return status.is_file() and status.read_text(encoding="utf-8").strip() == STATIC_AUDIT_PASS


def param_path(campaign: Path, case: dict[str, Any]) -> Path:
    return campaign / "params" / f"{case['case_id']}_{case['case_label']}.params"


def build_manifest(campaign: Path, source: Path, dry_run: bool) -> dict[str, Any]:
    cases, by_id = load_plan(campaign)
    forward = [row["case_id"] for row in sorted(cases, key=lambda row: int(row["case_id"]))]
    reverse = list(reversed(forward))
    forward_rows = [
        {"position": i + 1, "case_id": case_id, "case_label": by_id[case_id]["case_label"]}
        for i, case_id in enumerate(forward)
    ]
    reverse_rows = [
        {"position": i + 1, "case_id": case_id, "case_label": by_id[case_id]["case_label"]}
        for i, case_id in enumerate(reverse)
    ]
    write_csv(campaign / "case_order_forward.csv", forward_rows)
    write_csv(campaign / "case_order_reverse.csv", reverse_rows)

    binary_sha = sha256(source / "main_cuda")
    param_hashes: dict[str, str] = {}
    fixture_hashes: dict[str, str] = {}
    for case in cases:
        path = param_path(campaign, case)
        if not path.is_file():
            raise ValueError(f"missing parameter file for {case['case_id']}: {path}")
        param_hashes[case["case_id"]] = sha256(path)
        if not fixture_passes(campaign, case["case_id"]):
            raise ValueError(f"fixture static audit missing or failed for {case['case_id']}")
        fixture_hashes[case["case_id"]] = sha256(
            campaign / "fixtures" / case["case_id"] / "fixture_manifest.json"
        )

    queues = [
        {"queue": "gpu_uvip", "order": forward, "role": "forward"},
        {"queue": "gpu_vip_24h", "order": reverse, "role": "reverse"},
    ]
    entries: list[dict[str, Any]] = []
    for queue in queues:
        for position, case_id in enumerate(queue["order"], start=1):
            entries.append({
                "case_id": case_id,
                "case_label": by_id[case_id]["case_label"],
                "queue": queue["queue"],
                "role": queue["role"],
                "chain_position": position,
                "slurm_job_id": "",
                "dependency_job_id": "",
                "attempt_root": f"attempts/{case_id}/{queue['queue']}_PENDING",
                "fixture_sha256": fixture_hashes[case_id],
                "binary_sha256": binary_sha,
                "parameter_sha256": param_hashes[case_id],
                "submitted_time": "",
                "initial_state": "PENDING_SUBMISSION" if dry_run else "UNSUBMITTED",
            })
    return {
        "schema": SCHEMA,
        "campaign_root": str(campaign),
        "source_root": str(source),
        "binary_sha256": binary_sha,
        "logical_case_count": 24,
        "unique_case_count": 21,
        "dual_queue_logical_submission_count": 42,
        "forward_first": forward[0],
        "forward_last": forward[-1],
        "reverse_first": reverse[0],
        "reverse_last": reverse[-1],
        "fixture_pass_count": sum(1 for case in cases if fixture_passes(campaign, case["case_id"])),
        "fixture_blocked_count": sum(1 for case in cases if not fixture_passes(campaign, case["case_id"])),
        "entries": entries,
        "available_storage_TB": shutil.disk_usage(campaign).free / (1024 ** 4),
    }


def submit_entry(
    entry: dict[str, Any],
    campaign: Path,
    source: Path,
    sbatch: Path,
    case_by_id: dict[str, dict[str, Any]],
    queue: str,
    actual_queue: str,
    previous_job: str,
) -> None:
    case = case_by_id[entry["case_id"]]
    param = param_path(campaign, case)
    env = (
        f"ALL,SOURCE_ROOT={source},CAMPAIGN_ROOT={campaign},CASE_ID={case['case_id']},"
        f"QUEUE={actual_queue},PARAM_FILE={param}"
    )
    command = ["sbatch", "--parsable", "--partition", actual_queue, "--qos", actual_queue, "--export", env, str(sbatch)]
    if previous_job:
        command.extend(["--dependency", f"afterany:{previous_job}"])
    job_id = run(command)
    now = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    entry.update({
        "slurm_job_id": job_id,
        "dependency_job_id": previous_job,
        "actual_queue": actual_queue,
        "attempt_root": f"attempts/{case['case_id']}/{actual_queue}_{job_id}",
        "submitted_time": now,
        "initial_state": "SUBMITTED",
    })


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--campaign-root", type=Path, required=True)
    parser.add_argument("--source-root", type=Path, required=True)
    parser.add_argument("--sbatch", type=Path, required=True)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--submit-uvip", action="store_true")
    parser.add_argument("--submit-vip", action="store_true")
    parser.add_argument("--max-active", type=int, default=1)
    parser.add_argument("--max-submitted", type=int, default=0)
    parser.add_argument("--force-queue", choices=("gpu_uvip", "gpu_vip_24h"), default="")
    args = parser.parse_args()

    campaign = args.campaign_root.resolve()
    source = args.source_root.resolve()
    sbatch = args.sbatch.resolve()
    cases, case_by_id = load_plan(campaign)
    missing = [row["case_id"] for row in cases if not fixture_passes(campaign, row["case_id"])]
    if missing:
        (campaign / "submission_manifest.json").write_text(json.dumps({
            "schema": SCHEMA,
            "status": "BLOCKED_FIXTURE_QUALIFICATION",
            "missing_fixture_audits": missing,
        }, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        print("BLOCKED_FIXTURE_QUALIFICATION", ",".join(missing))
        raise SystemExit(2)

    manifest = build_manifest(campaign, source, args.dry_run)
    entries = manifest["entries"]
    previous_entries: dict[tuple[str, str], dict[str, Any]] = {}
    previous_path = campaign / "submission_manifest.json"
    if previous_path.is_file():
        try:
            previous = json.loads(previous_path.read_text(encoding="utf-8"))
            for row in previous.get("entries", []):
                if row.get("slurm_job_id"):
                    previous_entries[(row["case_id"], row["queue"])] = row
        except (json.JSONDecodeError, TypeError):
            previous_entries = {}
    for entry in entries:
        saved = previous_entries.get((entry["case_id"], entry["queue"]))
        if saved:
            entry.update(saved)
        if args.force_queue:
            entry["actual_queue"] = args.force_queue
    if not args.dry_run:
        active = active_jobs()
        active_ids = {job_id for ids in active.values() for job_id in ids}
        for entry in entries:
            job_id = entry.get("slurm_job_id", "")
            if not job_id or job_id in active_ids:
                continue
            case_id = entry["case_id"]
            if (campaign / "authority" / case_id / "PASS").exists():
                continue
            claim = campaign / "claims" / f"{case_id}.claim"
            actual_queue = entry.get("actual_queue", entry["queue"])
            attempt_status = campaign / "attempts" / case_id / f"{actual_queue}_{job_id}" / "status.txt"
            retryable = False
            if claim.exists():
                if attempt_status.is_file():
                    text = attempt_status.read_text(encoding="utf-8", errors="replace")
                    if "BLOCKED" in text:
                        continue
                    retryable = True
                else:
                    retryable = True
            else:
                retryable = True
            if retryable:
                if claim.exists():
                    try:
                        claim.rmdir()
                    except OSError:
                        pass
                entry.update({
                    "slurm_job_id": "",
                    "dependency_job_id": "",
                    "submitted_time": "",
                    "initial_state": "RETRYABLE_RESET",
                })
        for position, entry in enumerate(entries):
            queue = entry["queue"]
            if queue == "gpu_uvip" and not args.submit_uvip:
                continue
            if queue == "gpu_vip_24h" and not args.submit_vip:
                continue
            if entry["slurm_job_id"]:
                continue
            if (campaign / "claims" / f"{entry['case_id']}.claim").exists():
                continue
            actual_queue = entry.get("actual_queue", queue)
            current_submitted = len(active.get(actual_queue, []))
            submit_limit = args.max_submitted if args.max_submitted else args.max_active
            if current_submitted >= submit_limit:
                continue
            previous = ""
            for earlier in entries[:position]:
                if earlier["queue"] == queue and earlier["slurm_job_id"]:
                    previous = earlier["slurm_job_id"]
            submit_entry(entry, campaign, source, sbatch, case_by_id, queue, actual_queue, previous)
            active.setdefault(actual_queue, []).append(entry["slurm_job_id"])

    write_csv(campaign / "slurm_submission_log.csv", [{**row} for row in entries])
    write_csv(campaign / "claim_state_ledger.csv", [
        {
            "case_id": row["case_id"],
            "queue": row["queue"],
            "job_id": row["slurm_job_id"] or "",
            "claim_state": "AUTHORITY" if (campaign / "authority" / row["case_id"] / "PASS").exists() else "NOT_CLAIMED_OR_PENDING",
        }
        for row in entries
    ])
    write_csv(campaign / "authority_selection_ledger.csv", [
        {
            "case_id": row["case_id"],
            "queue": row["queue"],
            "job_id": row["slurm_job_id"] or "",
            "authority_owner": (
                (campaign / "authority" / row["case_id"] / "source.txt").read_text().strip()
                if (campaign / "authority" / row["case_id"] / "source.txt").exists() else ""
            ),
        }
        for row in entries
    ])
    write_json(campaign / "submission_manifest.json", manifest)
    (campaign / "submission_manifest.sha256").write_text(
        f"{sha256(campaign / 'submission_manifest.json')}  submission_manifest.json\n",
        encoding="utf-8",
    )

    submitted = [row for row in entries if row["slurm_job_id"]]
    validation = {
        "forward_first_case": next(row["case_id"] for row in entries if row["queue"] == "gpu_uvip"),
        "reverse_first_case": next(row["case_id"] for row in entries if row["queue"] == "gpu_vip_24h"),
        "actual_queue": args.force_queue or "native_dual_queue",
        "submitted_count": len(submitted),
        "uvip_submitted_count": len([row for row in entries if row["queue"] == "gpu_uvip" and row["slurm_job_id"]]),
        "vip_submitted_count": len([row for row in entries if row["queue"] == "gpu_vip_24h" and row["slurm_job_id"]]),
        "uvip_job_ids": [row["slurm_job_id"] for row in entries if row["queue"] == "gpu_uvip"],
        "vip_job_ids": [row["slurm_job_id"] for row in entries if row["queue"] == "gpu_vip_24h"],
    }
    (campaign / "submission_validation.json").write_text(
        json.dumps(validation, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(validation, sort_keys=True))
    if not args.dry_run and len(submitted) == 42:
        print("PASS_DUAL_QUEUE_FORWARD_REVERSE_SUBMISSION")


if __name__ == "__main__":
    main()
