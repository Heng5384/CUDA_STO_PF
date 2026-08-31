#!/usr/bin/env python3
"""Audit and, when authorised, run the beta-only KWN-to-PF consistency comparison."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.config import load_config  # noqa: E402


def _write(path: Path, content: str) -> None:
    """Write a small UTF-8 report, creating its parent directory."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    """Emit the authority-gated beta-only comparison status."""

    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(ROOT / "configs/kwn/beta_only_pf_consistency.yaml"))
    parser.add_argument("--report", default=str(ROOT / "reports/kwn_pf_mvp_v1/03_beta_only_pf_consistency.md"))
    parser.add_argument("--output", default=str(ROOT / "outputs/kwn_pf_mvp_v1/beta_only_comparison.csv"))
    args = parser.parse_args()
    document = load_config(args.config)
    thermo = document.data["thermodynamics"]
    routes = document.data["pf_snapshot_routes"]
    report_path = Path(args.report)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    status = "P0_CONTRACT_CONFLICT"
    blocker = (
        "The PF thermodynamic freeze remains BLOCKED_CONTRACT_CONFLICT between legacy runtime "
        "and exact-candidate local source. The requested same-PF thermodynamic/mobility beta-only "
        "run cannot select either value silently."
    )
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "status",
                "case",
                "snapshot_route",
                "comparison_executed",
                "reason",
                "config_sha256",
            ],
        )
        writer.writeheader()
        for label, route in routes.items():
            writer.writerow(
                {
                    "status": status,
                    "case": label,
                    "snapshot_route": route,
                    "comparison_executed": "false",
                    "reason": blocker,
                    "config_sha256": document.sha256,
                }
            )
    _write(
        report_path,
        "# Beta-only KWN versus PF consistency\n\n"
        f"Status: `{status}`\n\n"
        "## Result\n\n"
        f"{blocker}\n\n"
        "The repository retains 246-cube A/B registered PSD at 12 h, but it does not retain the requested "
        "full individual-radius Broad and Narrow 400-cube PSD locally. Scalar moments are not substituted for "
        "a PSD. The listed 246 A/B routes are retained as future proxy routes only.\n\n"
        "## Required before execution\n\n"
        "1. Resolve and hash-bind the PF thermodynamic contract.\n"
        "2. Export full 12 h resolved-beta PSD for one BROAD/REF and one NARROW case from authoritative checkpoints.\n"
        "3. Re-run with nucleation and GP off, PF-consistent diffusivity, and no fitted `D_scale`.\n\n"
        "The blocked state is not reported as a failed KWN numerical test; it is an authority/provenance gate.\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
