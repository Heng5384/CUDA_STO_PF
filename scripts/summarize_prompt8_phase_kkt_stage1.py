#!/usr/bin/env python3
"""Combine Prompt-8 phase-KKT diagnostics without altering raw evidence."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path


def parse_input(value: str) -> tuple[str, Path]:
    if "=" not in value:
        raise argparse.ArgumentTypeError("input must be CASE=CSV")
    case, path = value.split("=", 1)
    if not case or not path:
        raise argparse.ArgumentTypeError("input must be CASE=CSV")
    return case, Path(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", action="append", type=parse_input,
                        required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    rows: list[dict[str, str]] = []
    fieldnames: list[str] | None = None
    for case, path in args.input:
        if not path.is_file():
            raise FileNotFoundError(path)
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if reader.fieldnames is None:
                raise ValueError(f"missing CSV header: {path}")
            if fieldnames is None:
                fieldnames = list(reader.fieldnames)
            elif reader.fieldnames != fieldnames:
                raise ValueError(
                    f"incompatible headers: {path}: {reader.fieldnames}")
            for row in reader:
                rows.append({"case": case, "source_csv": str(path), **row})

    if not rows or fieldnames is None:
        raise ValueError("no diagnostic rows")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    output_fields = ["case", "source_csv", *fieldnames]
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=output_fields)
        writer.writeheader()
        writer.writerows(rows)
    print(f"phase_kkt_rows={len(rows)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
