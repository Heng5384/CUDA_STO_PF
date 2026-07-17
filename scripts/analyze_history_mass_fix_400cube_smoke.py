#!/usr/bin/env python3
"""Summarize the large-grid BDF2 history-mass preflight smoke."""

from __future__ import annotations

import argparse
import json
import pathlib
import re


def field(line: str, name: str) -> str | None:
    match = re.search(rf"(?:^|\s){re.escape(name)}=([^\s]+)", line)
    return match.group(1) if match else None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=pathlib.Path)
    parser.add_argument("--exit-code", required=True, type=int)
    args = parser.parse_args()

    log_text = (args.run_dir / "run.log").read_text(errors="replace")
    contexts = []
    for line in log_text.splitlines():
        if "CTOT_IMEX_BDF2_STEP_CONTEXT" not in line:
            continue
        contexts.append({
            key: field(line, key)
            for key in (
                "step", "attempt_id", "integrator", "history_valid", "dt",
                "dt_transport", "history_mass_delta", "history_mass_scale",
                "history_mass_tolerance", "fallback_pending", "fallback_reason",
            )
        })

    full_bdf2 = [
        row for row in contexts
        if row["integrator"] == "BDF2" and row["fallback_pending"] == "0"
    ]
    accepted_macros = len(re.findall(r"^CTOT_MIMETIC_BE_ACCEPT\b", log_text, re.M))
    hard_gate_rows = len(re.findall(
        r"^CTOT_FULL_AUDIT_RESULT\b.*\bhard_gates_executed=1\b",
        log_text,
        re.M,
    ))
    event_subcycle_retries = len(re.findall(
        r"^CTOT_BDF2_EVENT_SUBCYCLE_RETRY\b", log_text, re.M
    ))
    history_mismatches = len(re.findall(r"history_mass_mismatch", log_text))

    memory_values = []
    memory_file = args.run_dir / "gpu_memory_samples.csv"
    if memory_file.exists():
        for line in memory_file.read_text(errors="replace").splitlines():
            parts = [part.strip() for part in line.split(",")]
            if len(parts) >= 2:
                try:
                    memory_values.append(float(parts[1]))
                except ValueError:
                    pass

    passed = (
        args.exit_code == 0
        and history_mismatches == 0
        and len(full_bdf2) >= 1
        and accepted_macros == 4
        and hard_gate_rows == 4
    )
    summary = {
        "exit_code": args.exit_code,
        "accepted_macros": accepted_macros,
        "hard_gate_rows": hard_gate_rows,
        "full_bdf2_contexts": len(full_bdf2),
        "event_subcycle_retries": event_subcycle_retries,
        "history_mass_mismatches": history_mismatches,
        "peak_memory_MiB": max(memory_values) if memory_values else None,
        "step_contexts": contexts,
        "pass": passed,
    }
    (args.run_dir / "smoke_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
