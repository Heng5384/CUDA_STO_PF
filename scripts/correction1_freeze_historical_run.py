#!/usr/bin/env python3
"""Hash and summarize a completed Research2 historical run tree."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def active_processes() -> list[dict[str, object]]:
    found = []
    proc = Path("/proc")
    if not proc.is_dir():
        return found
    own = os.getpid()
    for entry in proc.iterdir():
        if not entry.name.isdigit() or int(entry.name) == own:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
            cwd = os.readlink(entry / "cwd")
        except (FileNotFoundError, PermissionError, ProcessLookupError):
            continue
        if "main_cuda" in command or "run_research2_fast_interface_plateau" in command:
            found.append({"pid": int(entry.name), "command": command, "cwd": cwd})
    return sorted(found, key=lambda row: int(row["pid"]))


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--assets", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    statuses = sorted(root.glob("T*/**/status.json"))
    if not statuses:
        raise FileNotFoundError(f"no status files under {root}")
    cases = []
    assets = []
    binary_hashes = set()
    for status_path in statuses:
        status = json.loads(status_path.read_text())
        case_dir = status_path.parent
        manifest_path = case_dir / "input/benchmark_manifest.json"
        manifest = json.loads(manifest_path.read_text()) if manifest_path.is_file() else {}
        checkpoints = []
        for path in case_dir.rglob("ctot_checkpoint_step*_meta.json"):
            match = re.search(r"step(\d+)_", path.name)
            if match:
                checkpoints.append((int(match.group(1)), path))
        latest_checkpoint = max(checkpoints, default=(0, None), key=lambda item: item[0])
        status_mtime = status_path.stat().st_mtime
        wall_time = float(status.get("wall_time_s") or 0.0)
        command = status.get("command", [])
        binary = Path(command[0]) if command else None
        actual_binary_hash = sha256(binary) if binary and binary.is_file() else "MISSING"
        binary_hashes.add(actual_binary_hash)
        cases.append({
            "case": status.get("case", case_dir.name),
            "case_path": str(case_dir.relative_to(root)),
            "temperature_C": status.get("temperature_C"),
            "L_phi_factor": status.get("L_phi_factor"),
            "L_phi_code": status.get("L_phi_code"),
            "L_phi_reference_code": status.get("L_phi_reference_code"),
            "returncode": status.get("returncode"),
            "accepted_steps": status.get("accepted_steps"),
            "nsteps": status.get("nsteps"),
            "dt_code": status.get("dt_code"),
            "final_code_time": (
                float(status.get("nsteps") or 0) * float(status.get("dt_code") or 0.0)
            ),
            "retry_count": status.get("retry_count"),
            "reject_count": status.get("reject_count"),
            "observation_time_s": status.get("observation_time_s"),
            "wall_time_s": status.get("wall_time_s"),
            "status_mtime_unix_s": status_mtime,
            "inferred_start_unix_s": status_mtime - wall_time,
            "latest_checkpoint_step": latest_checkpoint[0],
            "latest_checkpoint": (
                str(latest_checkpoint[1].relative_to(root))
                if latest_checkpoint[1] is not None else "MISSING"
            ),
            "lambda_over_dx": manifest.get("interface_resolution"),
            "finite_interface_mode": status.get("finite_interface_mode"),
            "elasticity": status.get("elasticity"),
            "GP_S3": status.get("GP_S3"),
            "binary_sha256_recorded": status.get("binary_sha256"),
            "binary_sha256_actual": actual_binary_hash,
            "params_sha256_recorded": status.get("params_sha256"),
            "command": json.dumps(command, separators=(",", ":")),
        })
        for path in sorted(p for p in case_dir.rglob("*") if p.is_file()):
            stat = path.stat()
            assets.append({
                "case": case_dir.name,
                "path": str(path.relative_to(root)),
                "size_bytes": stat.st_size,
                "mtime_unix_s": f"{stat.st_mtime:.9f}",
                "sha256": sha256(path),
                "asset_role": (
                    "status" if path.name == "status.json" else
                    "input" if "input" in path.relative_to(case_dir).parts else
                    "runtime_output"
                ),
            })
    args.assets.parent.mkdir(parents=True, exist_ok=True)
    with args.assets.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(assets[0]))
        writer.writeheader()
        writer.writerows(assets)
    summary = {
        "root": str(root),
        "active_processes": active_processes(),
        "case_count": len(cases),
        "completed_count": sum(int(row["returncode"] or 0) == 0 for row in cases),
        "all_binary_hashes": sorted(binary_hashes),
        "cases": cases,
        "classification": "HISTORICAL_BASELINE_UNDER_PREVIOUS_LPHI_SELECTION",
    }
    args.summary.parent.mkdir(parents=True, exist_ok=True)
    args.summary.write_text(json.dumps(summary, indent=2) + "\n")
    print(f"historical_case_count={len(cases)}")
    print(f"active_process_count={len(summary['active_processes'])}")
    print(f"asset_count={len(assets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
