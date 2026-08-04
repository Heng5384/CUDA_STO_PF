#!/usr/bin/env python3
"""Compare two isolated global density analyses byte-for-byte."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def files(root: Path) -> dict[str, str]:
    return {
        path.relative_to(root).as_posix(): digest(path)
        for path in sorted(root.rglob("*"))
        if path.is_file()
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--first", type=Path, required=True)
    parser.add_argument("--second", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    first, second, out = args.first.resolve(), args.second.resolve(), args.out.resolve()
    first_files, second_files = files(first), files(second)
    names = sorted(set(first_files) | set(second_files))
    comparisons = [
        {
            "path": name,
            "first_sha256": first_files.get(name, "MISSING"),
            "second_sha256": second_files.get(name, "MISSING"),
            "bytewise_equal": first_files.get(name) == second_files.get(name),
        }
        for name in names
    ]
    passed = all(row["bytewise_equal"] for row in comparisons)
    payload = {
        "schema": "GLOBAL_RESOLVED_PSD_NO_GO_DETERMINISM_V1",
        "source_date_epoch": 0,
        "first_root": str(first),
        "second_root": str(second),
        "per_file": comparisons,
        "status": "PASS_BYTEWISE_DETERMINISTIC" if passed else "FAIL_NONDETERMINISTIC_OUTPUT",
    }
    (out / "determinism_audit.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    table = "\n".join(
        f"| `{row['path']}` | `{row['bytewise_equal']}` |"
        for row in comparisons
    )
    (out / "determinism_audit.md").write_text(
        "# Global density-analysis determinism audit\n\n"
        "Two isolated analyses were run with `SOURCE_DATE_EPOCH=0`; no PF run or historical output was modified.\n\n"
        f"Status: `{payload['status']}`.\n\n"
        "| file | bytewise equal |\n|---|---:|\n" + table + "\n",
        encoding="utf-8",
    )
    print(payload["status"])


if __name__ == "__main__":
    main()
