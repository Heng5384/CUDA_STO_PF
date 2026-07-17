#!/usr/bin/env python3
"""Freeze V1 evidence after the unchanged formal queue has terminated."""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "transport_residual_gate_v1"
LOCAL_LONG = ROOT / "runs" / "transport_residual_gate_v1_long"
REMOTE_LONG = "/home/zhiheng/PF/CUDA_STO_PF_transport_gate_v1/runs/transport_residual_gate_v1_long"
REMOTE_BINARY = "/home/zhiheng/PF/CUDA_STO_PF_transport_gate_v1/main_cuda"
FROZEN_COMMIT = "f440c0dcd4c02cd45d9079c35d3838ebfa9b37e2"
REMOTE_REPO = "/home/zhiheng/PF/CUDA_STO_PF_transport_gate_v1"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def formal_queue_is_terminal() -> None:
    summary = REPORT / "long_window_candidate_summary.csv"
    if not summary.is_file():
        raise RuntimeError("long candidate summary missing")
    with summary.open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    if len(rows) != 13 or any(
        row["long_window_status"] in ("", "PENDING") for row in rows
    ):
        raise RuntimeError("V1 formal queue is not terminal")


def local_rows() -> list[dict[str, object]]:
    excluded = {
        REPORT / "preregistered_v1_hash_manifest.csv",
        REPORT / "preregistered_v1_preservation.md",
    }
    result: list[dict[str, object]] = []
    roots = (("local_report", REPORT), ("local_long_cache", LOCAL_LONG))
    for scope, root in roots:
        if not root.exists():
            continue
        for path in sorted(item for item in root.rglob("*") if item.is_file()):
            if path in excluded:
                continue
            result.append(
                {
                    "scope": scope,
                    "path": str(path.relative_to(ROOT)),
                    "size_bytes": path.stat().st_size,
                    "sha256": sha256(path),
                }
            )
    return result


def remote_rows(host: str) -> list[dict[str, object]]:
    command = f"""python3 - <<'PY'
from pathlib import Path
import hashlib, json
root=Path({REMOTE_LONG!r})
for path in sorted(p for p in root.rglob('*') if p.is_file()):
    h=hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024*1024), b''):
            h.update(chunk)
    print(json.dumps({{'path': str(path.relative_to(root)), 'size_bytes': path.stat().st_size, 'sha256': h.hexdigest()}}, sort_keys=True))
PY"""
    result = subprocess.run(
        ["ssh", host, command], text=True, capture_output=True, check=True
    )
    rows = []
    for line in result.stdout.splitlines():
        item = json.loads(line)
        rows.append({"scope": "remote_formal_long", **item})
    binary = subprocess.run(
        ["ssh", host, f"sha256sum {REMOTE_BINARY}"],
        text=True,
        capture_output=True,
        check=True,
    ).stdout.split()
    rows.append(
        {
            "scope": "remote_binary",
            "path": REMOTE_BINARY,
            "size_bytes": "",
            "sha256": binary[0],
        }
    )
    return rows


def remote_source_names(host: str) -> list[str]:
    command = (
        f"cd {REMOTE_REPO} && "
        "find . -maxdepth 1 -type f "
        "\\( -name '*.cu' -o -name '*.h' -o -name 'Makefile' \\) "
        "-printf '%f\\n' | LC_ALL=C sort"
    )
    result = subprocess.run(
        ["ssh", host, command], text=True, capture_output=True, check=True
    )
    names = [line for line in result.stdout.splitlines() if line]
    if "main_cuda.cu" not in names or "Makefile" not in names:
        raise RuntimeError(f"remote V1 source enumeration is incomplete: {names}")
    return names


def snapshot_remote_source(host: str, names: list[str]) -> None:
    snapshot = REPORT / "frozen_v1_source_snapshot"
    snapshot.mkdir(parents=True, exist_ok=True)
    for name in names:
        subprocess.run(
            [
                "rsync",
                "-az",
                f"{host}:{REMOTE_REPO}/{name}",
                str(snapshot / name),
            ],
            check=True,
        )


def source_rows() -> list[dict[str, object]]:
    listing = subprocess.run(
        ["git", "ls-tree", "-r", "--name-only", FROZEN_COMMIT],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    ).stdout.splitlines()
    files = sorted(
        name
        for name in listing
        if "/" not in name
        and (name == "Makefile" or name.endswith(".cu") or name.endswith(".h"))
    )
    if "main_cuda.cu" not in files or "Makefile" not in files:
        raise RuntimeError(f"frozen commit source enumeration is incomplete: {files}")
    result = []
    for name in files:
        blob = subprocess.run(
            ["git", "rev-parse", f"{FROZEN_COMMIT}:{name}"],
            cwd=ROOT,
            text=True,
            capture_output=True,
            check=True,
        ).stdout.strip()
        result.append(
            {
                "scope": "frozen_source_blob",
                "path": name,
                "size_bytes": "",
                "sha256": "git_blob:" + blob,
            }
        )
    return result


def main() -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="workstation-tail")
    args = parser.parse_args()
    formal_queue_is_terminal()
    remote_names = remote_source_names(args.host)
    snapshot_remote_source(args.host, remote_names)
    records = source_rows() + local_rows() + remote_rows(args.host)
    output = REPORT / "preregistered_v1_hash_manifest.csv"
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=("scope", "path", "size_bytes", "sha256")
        )
        writer.writeheader()
        writer.writerows(records)
    print(
        json.dumps(
            {
                "frozen_commit": FROZEN_COMMIT,
                "records": len(records),
                "manifest": str(output),
                "manifest_sha256": sha256(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
