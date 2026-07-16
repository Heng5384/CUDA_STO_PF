#!/usr/bin/env python3
"""Deterministic, resumable process-parallel Next4 oracle matrix."""

from __future__ import annotations

import os

for variable in (
    "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
):
    os.environ[variable] = "1"

import argparse
import concurrent.futures
import csv
import hashlib
import json
import math
import multiprocessing
from pathlib import Path
import platform
import sys
import time

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.next4_numba_root_kernels import warmup_numba_kernels  # noqa: E402
from scripts.next4_optimized_sharp_oracle import run_case_optimized  # noqa: E402


RATIOS = (5, 8, 10, 15, 20)
CONFIGURATIONS = (
    (256, 1.5625e-6, "mesh_256"),
    (512, 1.5625e-6, "mesh_512"),
    (1024, 1.5625e-6, "mesh_1024"),
    (1024, 7.8125e-7, "time_refined_1024"),
)
SOLVER_VERSION = "next4_optimized_oracle_v2_full_bracket_fallback"


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def serializable(result: dict[str, object]) -> dict[str, object]:
    excluded = {
        "radial_nodes_initial", "profile_initial", "radial_nodes_final",
        "profile_final", "prepared_capillary_state",
    }
    return {key: value for key, value in result.items() if key not in excluded}


def save_capillary_cache(path: Path, state: dict[str, object]) -> None:
    with path.open("wb") as handle:
        np.savez(
            handle,
            radius_nm=state["radius_nm"],
            outer_radius_nm=state["outer_radius_nm"],
            nodes=state["nodes"],
            radial_dimension=state["radial_dimension"],
            surface_initial_capillary=state["surface_initial_capillary"],
            radial_nodes=state["radial_nodes"],
            profile=state["profile"],
            inventory_target=state["inventory_target"],
            inventory_match_correction_xB=state["inventory_match_correction_xB"],
        )


def load_capillary_cache(path: Path) -> dict[str, object]:
    with np.load(path, allow_pickle=False) as data:
        return {
            "radius_nm": float(data["radius_nm"]),
            "outer_radius_nm": float(data["outer_radius_nm"]),
            "nodes": int(data["nodes"]),
            "radial_dimension": int(data["radial_dimension"]),
            "surface_initial_capillary": float(data["surface_initial_capillary"]),
            "radial_nodes": data["radial_nodes"].copy(),
            "profile": data["profile"].copy(),
            "inventory_target": float(data["inventory_target"]),
            "inventory_match_correction_xB": float(
                data["inventory_match_correction_xB"]
            ),
        }


def ratio_cases(ratio: int) -> list[dict[str, object]]:
    cases: list[dict[str, object]] = []
    for mode, finite, multiplier, configurations in (
        ("zero_kinetic", False, 1.0, CONFIGURATIONS),
        ("finite_Lphi", True, 1.0, CONFIGURATIONS),
        ("large_Lphi_1e6", True, 1.0e6, CONFIGURATIONS[-1:]),
    ):
        for nodes, dt, label in configurations:
            cases.append({
                "ratio": ratio,
                "boundary_mode": mode,
                "finite_lphi": finite,
                "l_phi_multiplier": multiplier,
                "nodes": nodes,
                "sharp_dt": dt,
                "configuration": label,
                "case_id": f"r{ratio}_{mode}_{label}",
            })
    return cases


def run_ratio_group(arguments: dict[str, object]) -> dict[str, object]:
    ratio = int(arguments["ratio"])
    moving_root = Path(str(arguments["moving_root"]))
    stationary_root = Path(str(arguments["stationary_root"]))
    output_root = Path(str(arguments["output_root"]))
    final_time = float(arguments["final_time"])
    ratio_root = output_root / f"r{ratio}"
    ratio_root.mkdir(parents=True, exist_ok=True)
    warmup_numba_kernels()
    state = json.loads(
        (moving_root / "shared" / f"r{ratio}"
         / "moving_state_manifest.json").read_text()
    )
    params = stationary_root / f"r{ratio}" / "benchmark.params"
    capillary_cache: dict[int, dict[str, object]] = {}
    rows: list[dict[str, object]] = []
    group_start = time.perf_counter()
    for case in ratio_cases(ratio):
        case_id = str(case["case_id"])
        result_path = ratio_root / f"{case_id}.json"
        manifest_payload = {
            **case,
            "solver_version": SOLVER_VERSION,
            "final_time": final_time,
            "moving_state": state,
            "params_sha256": hashlib.sha256(params.read_bytes()).hexdigest(),
        }
        case_hash = sha256_bytes(
            json.dumps(manifest_payload, sort_keys=True).encode()
        )
        if result_path.exists():
            previous = json.loads(result_path.read_text())
            if previous.get("case_input_hash") == case_hash:
                rows.append(previous)
                continue
            raise RuntimeError(f"completed case hash mismatch: {case_id}")
        nodes = int(case["nodes"])
        cache_path = ratio_root / f"capillary_state_nodes{nodes}.npz"
        if nodes not in capillary_cache and cache_path.exists():
            capillary_cache[nodes] = load_capillary_cache(cache_path)
        checkpoint = ratio_root / f"{case_id}.checkpoint.npz"
        resume = checkpoint if checkpoint.exists() else None
        start = time.perf_counter()
        result = run_case_optimized(
            params,
            float(state["radius_h_nm"]),
            float(state["box_nm"]),
            float(state["far_xB_requested"]),
            float(state["diffusion_age_code"]),
            final_time,
            float(case["sharp_dt"]),
            nodes,
            float(state["pf_inventory_nm2"]),
            finite_lphi_kinetics=bool(case["finite_lphi"]),
            radial_dimension=2,
            code_length_unit_nm=1.0,
            l_phi_multiplier=float(case["l_phi_multiplier"]),
            prepared_capillary_state=capillary_cache.get(nodes),
            root_backend="numba",
            checkpoint_path=checkpoint,
            checkpoint_interval_steps=500,
            resume_checkpoint=resume,
            checkpoint_input_hash=case_hash,
        )
        wall = time.perf_counter() - start
        if nodes not in capillary_cache:
            capillary_cache[nodes] = result["prepared_capillary_state"]
            save_capillary_cache(cache_path, capillary_cache[nodes])
        row = {
            **serializable(result),
            **case,
            "case_input_hash": case_hash,
            "case_wall_time_s": wall,
            "worker_pid": os.getpid(),
            "solver_version": SOLVER_VERSION,
            "status": "PASS",
        }
        temporary = result_path.with_suffix(".json.tmp")
        temporary.write_text(json.dumps(row, indent=2, allow_nan=True) + "\n")
        os.replace(temporary, result_path)
        rows.append(row)
    return {
        "ratio": ratio,
        "worker_pid": os.getpid(),
        "group_wall_time_s": time.perf_counter() - group_start,
        "rows": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--moving-root", type=Path, required=True)
    parser.add_argument("--stationary-root", type=Path, required=True)
    parser.add_argument("--gold-root", type=Path, required=True)
    parser.add_argument("--frozen-pf-metrics", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, required=True)
    parser.add_argument("--final-time", type=float, default=0.003125)
    parser.add_argument("--workers", type=int, default=5)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    args.report_root.mkdir(parents=True, exist_ok=True)
    worker_count = max(1, min(args.workers, len(RATIOS), os.cpu_count() or 1))
    manifest = {
        "solver_version": SOLVER_VERSION,
        "workers": worker_count,
        "thread_environment": {
            key: os.environ[key] for key in (
                "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS",
            )
        },
        "python": sys.version,
        "platform": platform.platform(),
        "cases": [case for ratio in RATIOS for case in ratio_cases(ratio)],
    }
    (args.output_root / "matrix_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    start = time.perf_counter()
    context = multiprocessing.get_context("spawn")
    groups: list[dict[str, object]] = []
    with concurrent.futures.ProcessPoolExecutor(
        max_workers=worker_count, mp_context=context
    ) as executor:
        futures = [
            executor.submit(run_ratio_group, {
                "ratio": ratio,
                "moving_root": str(args.moving_root),
                "stationary_root": str(args.stationary_root),
                "output_root": str(args.output_root),
                "final_time": args.final_time,
            }) for ratio in RATIOS
        ]
        for future in concurrent.futures.as_completed(futures):
            groups.append(future.result())
    matrix_wall = time.perf_counter() - start
    all_rows = [row for group in groups for row in group["rows"]]
    all_rows.sort(key=lambda row: (
        int(row["ratio"]),
        ("zero_kinetic", "finite_Lphi", "large_Lphi_1e6").index(
            str(row["boundary_mode"])
        ),
        ("mesh_256", "mesh_512", "mesh_1024", "time_refined_1024").index(
            str(row["configuration"])
        ),
    ))
    gold_rows = list(csv.DictReader(
        (args.gold_root / "outputs" / "next4_sharp_oracle_convergence.csv").open()
    ))
    gold = {
        (int(row["R_over_lambda"]), row["boundary_mode"], row["configuration"]): row
        for row in gold_rows
    }
    table: list[dict[str, object]] = []
    references: dict[int, dict[str, dict[str, object]]] = {}
    for row in all_rows:
        key = (int(row["ratio"]), str(row["boundary_mode"]), str(row["configuration"]))
        gold_row = gold[key]
        velocity = float(row["velocity_full_nm_per_code_time"])
        gold_velocity = float(gold_row["velocity_full_nm_per_code_time"])
        table.append({
            "R_over_lambda": row["ratio"],
            "boundary_mode": row["boundary_mode"],
            "configuration": row["configuration"],
            "radial_nodes": row["nodes"],
            "sharp_dt_code": row["sharp_dt_code"],
            "final_time_code": row["final_time_code"],
            "velocity_full_nm_per_code_time": velocity,
            "gold_velocity_full_nm_per_code_time": gold_velocity,
            "velocity_abs_difference_to_gold": abs(velocity-gold_velocity),
            "velocity_rel_difference_to_gold": abs(velocity-gold_velocity)
            / max(abs(gold_velocity), 1e-300),
            "radius_final_nm": row["radius_final_nm"],
            "gold_radius_final_nm": gold_row["radius_final_nm"],
            "radius_abs_difference_to_gold": abs(
                float(row["radius_final_nm"])-float(gold_row["radius_final_nm"])
            ),
            "max_inventory_error_rel": row["max_inventory_error_rel"],
            "gold_max_inventory_error_rel": gold_row["max_inventory_error_rel"],
            "max_kinetic_boundary_residual": row["max_kinetic_boundary_residual"],
            "gold_max_kinetic_boundary_residual": gold_row["max_kinetic_boundary_residual"],
            "case_wall_time_s": row["case_wall_time_s"],
            "worker_pid": row["worker_pid"],
            "root_backend": row["root_backend"],
            "coupled_fallback_count": row["solver_metrics"]["coupled_fallback_count"],
            "capillary_function_evaluations": row["solver_metrics"]["capillary"]["function_evaluations"],
            "kinetic_function_evaluations": row["solver_metrics"]["kinetic"]["function_evaluations"],
            "coupled_newton_iterations": row["solver_metrics"]["coupled_newton_iterations"],
            "ale_coupled_iterations": row["solver_metrics"]["ale_coupled_iterations"],
            "status": row["status"],
        })
        if row["configuration"] == "time_refined_1024":
            references.setdefault(int(row["ratio"]), {})[
                str(row["boundary_mode"])
            ] = serializable(row)
    write_csv(args.report_root / "next4_full_matrix_results.csv", table)
    (args.report_root / "next4_sharp_reference_optimized.json").write_text(
        json.dumps(references, indent=2, allow_nan=True) + "\n"
    )

    pf_rows = [
        row for row in csv.DictReader(args.frozen_pf_metrics.open())
        if row["window"] == "full"
    ]
    reanalysis: list[dict[str, object]] = []
    for row in pf_rows:
        ratio = int(row["R_over_lambda"])
        pf_velocity = float(row["PF_h_velocity_nm_per_code_time"])
        sharp = float(references[ratio]["finite_Lphi"]["velocity_full_nm_per_code_time"])
        error = abs(pf_velocity-sharp)/max(abs(sharp), 1e-300)
        reanalysis.append({
            "case": row["case"],
            "R_over_lambda": ratio,
            "correction": row["correction"],
            "PF_dt_code": row["dt_code"],
            "PF_h_velocity_nm_per_code_time": pf_velocity,
            "optimized_finite_Lphi_sharp_velocity_nm_per_code_time": sharp,
            "velocity_error_rel": error,
            "direction_match": int(pf_velocity*sharp >= 0.0),
            "error_below_2pct": int(error <= 0.02),
            "error_below_5pct": int(error <= 0.05),
            "provenance": "frozen_Next2_PF_reanalysis_no_PF_rerun",
        })
    write_csv(
        args.report_root / "next4_reanalyzed_velocity_matrix_optimized.csv",
        reanalysis,
    )
    summary = {
        "matrix_wall_time_s": matrix_wall,
        "workers": worker_count,
        "cases": len(table),
        "maximum_velocity_abs_difference_to_gold": max(
            float(row["velocity_abs_difference_to_gold"]) for row in table
        ),
        "maximum_velocity_rel_difference_to_gold": max(
            float(row["velocity_rel_difference_to_gold"]) for row in table
        ),
        "maximum_radius_abs_difference_to_gold": max(
            float(row["radius_abs_difference_to_gold"]) for row in table
        ),
        "maximum_inventory_residual": max(
            float(row["max_inventory_error_rel"]) for row in table
        ),
        "maximum_root_residual": max(
            float(row["max_kinetic_boundary_residual"]) for row in table
        ),
        "fallback_count": sum(int(row["coupled_fallback_count"]) for row in table),
        "group_wall_times": {
            str(group["ratio"]): group["group_wall_time_s"] for group in groups
        },
    }
    (args.output_root / "matrix_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
