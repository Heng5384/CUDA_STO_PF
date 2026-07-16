#!/usr/bin/env python3
"""Run mesh/time convergence for the independent Next2 sharp oracle."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.next2_finite_box_sharp_oracle import run_case  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving-root", type=Path, required=True)
    parser.add_argument("--stationary-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--benchmark-dt", type=float, default=6.25e-6)
    parser.add_argument("--final-time", type=float, default=0.003125)
    args = parser.parse_args()
    output = args.output_root.resolve()
    if output.exists():
        raise FileExistsError(f"refusing to overwrite {output}")
    output.mkdir(parents=True)
    rows: list[dict[str, object]] = []
    references: dict[str, object] = {}
    configurations = (
        (256, 0.25 * args.benchmark_dt, "mesh_256"),
        (512, 0.25 * args.benchmark_dt, "mesh_512"),
        (1024, 0.25 * args.benchmark_dt, "mesh_1024"),
        (1024, 0.125 * args.benchmark_dt, "time_refined_1024"),
    )
    for ratio in (5, 8, 10, 15, 20):
        state = json.loads(
            (args.moving_root / "shared" / f"r{ratio}"
             / "moving_state_manifest.json").read_text()
        )
        params = args.stationary_root / f"r{ratio}" / "benchmark.params"
        results: dict[str, dict[str, object]] = {}
        for nodes, sharp_dt, label in configurations:
            result = run_case(
                params,
                float(state["radius_h_nm"]),
                float(state["box_nm"]),
                float(state["far_xB_requested"]),
                float(state["diffusion_age_code"]),
                args.final_time,
                sharp_dt,
                nodes,
                float(state["pf_inventory_nm2"]),
            )
            results[label] = result
        reference = results["time_refined_1024"]
        reference_velocity = float(reference["velocity_full_nm_per_code_time"])
        reference_radius = float(reference["radius_final_nm"])
        for nodes, sharp_dt, label in configurations:
            result = results[label]
            velocity = float(result["velocity_full_nm_per_code_time"])
            radius = float(result["radius_final_nm"])
            rows.append({
                "R_over_lambda": ratio,
                "configuration": label,
                "radial_nodes": nodes,
                "sharp_dt_code": sharp_dt,
                "final_time_code": args.final_time,
                "radius_initial_nm": result["radius_initial_nm"],
                "radius_final_nm": radius,
                "velocity_full_nm_per_code_time": velocity,
                "velocity_initial_nm_per_code_time":
                    result["velocity_initial_nm_per_code_time"],
                "velocity_final_nm_per_code_time":
                    result["velocity_final_nm_per_code_time"],
                "velocity_abs_error_to_reference": abs(
                    velocity - reference_velocity
                ),
                "velocity_rel_error_to_reference": abs(
                    velocity - reference_velocity
                ) / max(abs(reference_velocity), 1.0e-300),
                "radius_abs_error_to_reference_nm": abs(radius - reference_radius),
                "max_inventory_error_rel": result["max_inventory_error_rel"],
                "inventory_match_correction_xB":
                    result["inventory_match_correction_xB"],
                "surface_xB_initial": result["surface_xB_initial"],
                "surface_xB_final": result["surface_xB_final"],
                "provenance": (
                    "independent_equal_area_finite_outer_no_flux_cylindrical_"
                    "sharp_oracle_no_PF_velocity_fit"
                ),
            })
        references[str(ratio)] = {
            key: value for key, value in reference.items()
            if key not in {
                "radial_nodes_initial", "profile_initial", "radial_nodes_final",
                "profile_final",
            }
        }
    with (output / "next2_sharp_convergence.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "sharp_reference.json").write_text(
        json.dumps(references, indent=2) + "\n"
    )
    print(f"sharp_convergence_rows={len(rows)}")
    print(f"sharp_reference_cases={len(references)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
