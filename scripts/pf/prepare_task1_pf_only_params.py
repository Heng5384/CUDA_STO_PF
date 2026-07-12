#!/usr/bin/env python3
"""Create deterministic PF-only Task-1 parameter snapshots from a complete file."""

from __future__ import annotations

import argparse
from pathlib import Path


OVERRIDES = {
    "enable_gp_assisted_beta_nucleation": "0",
    "enable_gp_runtime_library_nucleation": "0",
    "enable_runtime_nucleus_library": "0",
    "enable_dynamic_continue_bridge": "0",
    "gp_initial_population_enabled": "0",
    "diagnostic_rsmd_enabled": "0",
    "y_update_mass_projection_enabled": "0",
    "y_update_mass_projection_report_enabled": "0",
    "elastic_enabled": "0",
    "dt": "0.0001",
    "dt_code": "0.0001",
    "pf_composition_mode": "legacy",
    "pf_y_update_mode": "lagged_rhs",
    "ic_vf_init_phi": "0.015",
    "ic_vf_target_phi": "0.015",
    "ic_phi_seed_radius": "2.0",
    "ic_23d_xB_out": "0.03",
    "ic_xB_eq_matrix": "0.03",
    "seed": "20260712",
}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()

    lines = args.source.read_text(encoding="utf-8").splitlines()
    seen: set[str] = set()
    output: list[str] = ["# Generated for Task 1 PF-only replay; physics values come from source."]
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in line:
            output.append(line)
            continue
        key = line.split("=", 1)[0].strip()
        if key in OVERRIDES:
            output.append(f"{key}={OVERRIDES[key]}")
            seen.add(key)
        else:
            output.append(line)
    for key, value in OVERRIDES.items():
        if key not in seen:
            output.append(f"{key}={value}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(output) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
