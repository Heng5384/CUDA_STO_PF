"""Small command-line entry point for a config-owned KWN run."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .config import load_config
from .diagnostics import solver_observables, write_csv
from .solver import KWNSolver
from .units import hours_to_seconds


def main() -> int:
    """Run a KWN configuration to a requested physical time."""

    parser = argparse.ArgumentParser(description="Run the internal KWN finite-volume MVP")
    parser.add_argument("--config", required=True, help="JSON-subset YAML KWN configuration")
    parser.add_argument("--end-time-h", type=float, required=True, help="Physical end time in hours")
    parser.add_argument("--output", required=True, help="Output directory")
    args = parser.parse_args()
    document = load_config(args.config)
    solver = KWNSolver.from_document(document)
    solver.run_to_time(hours_to_seconds(args.end_time_h))
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "final_observables.csv", [solver_observables(solver)])
    (output / "run_metadata.json").write_text(
        json.dumps(
            {
                "config_sha256": document.sha256,
                "solver_version": solver.solver_version,
                "end_time_h": args.end_time_h,
            },
            indent=2,
            sort_keys=True,
        )
        + "\n",
        encoding="utf-8",
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
