#!/usr/bin/env python3
"""Run the frozen-input Lie-BE v2 step655 ladder on a workstation GPU."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import socket
import subprocess
import time


WORKSTATION_HOSTNAMES = {"fuxin"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_param(path: Path, key: str, default: str | None = None) -> str:
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" not in line:
            continue
        found, value = line.split("=", 1)
        if found.strip() == key:
            return value.strip()
    if default is None:
        raise KeyError(key)
    return default


def build_command(
    binary: Path,
    case: dict[str, object],
    params: Path,
    meta: Path,
    frozen_root: Path,
) -> list[str]:
    metadata = json.loads(meta.read_text(encoding="utf-8"))
    nx, ny, nz = (int(metadata[name]) for name in ("Nx", "Ny", "Nz"))
    elastic = int(read_param(params, "elastic_enabled"))
    steps = int(case["steps"])
    dt = float(case["run_dt"])
    case_id = str(case["case_id"])
    prefix = frozen_root / "ctot_checkpoint_step000054"
    return [
        str(binary), str(nx), str(ny), str(nz), f"{dt:.17e}",
        str(steps), str(steps), "1", str(elastic),
        "--mode", "dynamics", "--pf-param-file", str(params),
        "--init-mode", "raw_fields",
        "--init-phi-raw", str(prefix) + "_phi.raw",
        "--init-xB-raw", str(prefix) + "_xB_alpha.raw",
        "--init-Ctot-raw", str(prefix) + "_Ctot.raw",
        "--init-meta", str(meta), "--init-case-tag", case_id,
    ]


def select_cases(
    cases: list[dict[str, object]], selected: set[str]
) -> list[dict[str, object]]:
    matched = [
        case for case in cases
        if not selected or str(case["case_id"]) in selected
    ]
    matched_ids = {str(case["case_id"]) for case in matched}
    if selected and selected != matched_ids:
        raise KeyError(f"unknown cases: {sorted(selected - matched_ids)}")
    return matched


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--preparation-root", type=Path, required=True)
    parser.add_argument("--frozen-root", type=Path, required=True)
    parser.add_argument("--allow-host", action="append", default=[])
    parser.add_argument("--case", action="append", default=[])
    args = parser.parse_args()

    allowed = WORKSTATION_HOSTNAMES | set(args.allow_host)
    hostname = socket.gethostname().split(".", 1)[0]
    if hostname not in allowed:
        raise RuntimeError(f"workstation-only runner refused host={hostname}")
    if not args.binary.is_file() or not os.access(args.binary, os.X_OK):
        raise FileNotFoundError(args.binary)
    manifest_path = args.preparation_root / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    selected = set(args.case)
    cases = select_cases(manifest["cases"], selected)

    records: list[dict[str, object]] = []
    for case in cases:
        case_id = str(case["case_id"])
        case_dir = args.preparation_root / "cases" / case_id
        params = case_dir / "runtime.params"
        meta = case_dir / "init_meta.json"
        run_dir = case_dir / "run"
        if run_dir.exists():
            raise FileExistsError(f"refusing to overwrite {run_dir}")
        run_dir.mkdir()
        command = build_command(args.binary, case, params, meta, args.frozen_root)
        (run_dir / "command.json").write_text(
            json.dumps(command, indent=2) + "\n", encoding="utf-8"
        )
        start = time.monotonic()
        with (run_dir / "run.log").open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=run_dir, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        wall = time.monotonic() - start
        (run_dir / "exit_code.txt").write_text(
            f"{completed.returncode}\n", encoding="utf-8"
        )
        record = {
            "case_id": case_id,
            "exit_code": completed.returncode,
            "wall_s": wall,
            "binary_sha256": sha256(args.binary),
            "params_sha256": sha256(params),
            "meta_sha256": sha256(meta),
        }
        records.append(record)
        print(
            f"case={case_id} exit_code={completed.returncode} "
            f"wall_s={wall:.6f}", flush=True
        )
        if completed.returncode != 0:
            break
    status = {
        "host": hostname,
        "binary": str(args.binary),
        "binary_sha256": sha256(args.binary),
        "manifest_sha256": sha256(manifest_path),
        "records": records,
        "all_passed": len(records) == len(cases) and
            all(record["exit_code"] == 0 for record in records),
    }
    status_path = args.preparation_root / "workstation_run_status.json"
    status_path.write_text(json.dumps(status, indent=2) + "\n", encoding="utf-8")
    print(f"status={status_path}")
    return 0 if status["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
