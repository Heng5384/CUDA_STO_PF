#!/usr/bin/env python3
"""Create deterministic Prompt-8 Phase-KKT runtime test overlays."""

from __future__ import annotations

import argparse
from pathlib import Path


OVERLAYS = {
    "retry": {
        "dt": "1.0e-4",
        "ctot_debug_force_first_attempt_reject": "1",
        "ctot_step_max_retries": "1",
        "ctot_phase_semismooth_pdas_enabled": "1",
        "ctot_phase_restart_solver_migration_allowed": "0",
    },
    "direct_half_dt": {
        "dt": "5.0e-5",
        "ctot_debug_force_first_attempt_reject": "0",
        "ctot_step_max_retries": "0",
        "ctot_phase_semismooth_pdas_enabled": "1",
        "ctot_phase_restart_solver_migration_allowed": "0",
    },
    "restart": {
        "dt": "5.0e-5",
        "ctot_debug_force_first_attempt_reject": "0",
        "ctot_step_max_retries": "0",
        "ctot_phase_semismooth_pdas_enabled": "1",
        "ctot_phase_restart_solver_migration_allowed": "0",
    },
    "old_solver": {
        "dt": "5.0e-5",
        "ctot_debug_force_first_attempt_reject": "0",
        "ctot_step_max_retries": "0",
        "ctot_phase_semismooth_pdas_enabled": "0",
        "ctot_phase_restart_solver_migration_allowed": "0",
    },
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()

    base = args.base.read_text(encoding="utf-8").rstrip()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for name, values in OVERLAYS.items():
        lines = [base, "", f"# Prompt-8 generated overlay: {name}"]
        lines.extend(f"{key}={value}" for key, value in values.items())
        output = args.output_dir / f"prompt8_phase_kkt_{name}.params"
        output.write_text("\n".join(lines) + "\n", encoding="utf-8")
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
