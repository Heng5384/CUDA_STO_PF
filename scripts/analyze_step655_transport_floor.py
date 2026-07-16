#!/usr/bin/env python3
"""Independent host-FP64 analysis of the step655 D5 frozen transport state."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_rows(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def as_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("case_dir", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--dt", type=float, default=0.003125)
    parser.add_argument("--dx", type=float, default=1.0)
    parser.add_argument("--dy", type=float, default=1.0)
    parser.add_argument("--dz", type=float, default=1.0)
    args = parser.parse_args()

    summary = read_rows(args.case_dir / "ctot_d5_runtime_cold_summary.csv")[0]
    frozen = read_rows(args.case_dir / "ctot_d5_frozen_state.csv")
    polish = read_rows(args.case_dir / "ctot_d5_transport_polish.csv")
    aref = read_rows(args.case_dir / "ctot_d5_aref_invariance.csv")
    if not frozen:
        raise RuntimeError("frozen-state CSV is empty")

    nx = max(int(row["i"]) for row in frozen) + 1
    ny = max(int(row["j"]) for row in frozen) + 1
    nz = max(int(row["k"]) for row in frozen) + 1
    n = len(frozen)
    if n != nx * ny * nz:
        raise RuntimeError(f"incomplete grid: rows={n}, shape={nx}x{ny}x{nz}")

    def idx(i: int, j: int, k: int) -> int:
        return (i * ny + j) * nz + k

    mu = [as_float(row, "mu") for row in frozen]
    mfx = [as_float(row, "face_mobility_x") for row in frozen]
    mfy = [as_float(row, "face_mobility_y") for row in frozen]
    mfz = [as_float(row, "face_mobility_z") for row in frozen]
    fx_host = [0.0] * n
    fy_host = [0.0] * n
    fz_host = [0.0] * n
    for row in frozen:
        i, j, k = int(row["i"]), int(row["j"]), int(row["k"])
        p = idx(i, j, k)
        fx_host[p] = mfx[p] * (mu[idx((i + 1) % nx, j, k)] - mu[p]) / args.dx
        fy_host[p] = mfy[p] * (mu[idx(i, (j + 1) % ny, k)] - mu[p]) / args.dy
        fz_host[p] = mfz[p] * (mu[idx(i, j, (k + 1) % nz)] - mu[p]) / args.dz

    div_host = [0.0] * n
    residual_host = [0.0] * n
    decomposition: list[dict[str, object]] = []
    for row in frozen:
        i, j, k = int(row["i"]), int(row["j"]), int(row["k"])
        p = idx(i, j, k)
        im = idx((i - 1) % nx, j, k)
        jm = idx(i, (j - 1) % ny, k)
        km = idx(i, j, (k - 1) % nz)
        div = ((fx_host[p] - fx_host[im]) / args.dx +
               (fy_host[p] - fy_host[jm]) / args.dy +
               (fz_host[p] - fz_host[km]) / args.dz)
        div_host[p] = div
        c_old = as_float(row, "Cn")
        c_new = as_float(row, "C_trial")
        delta_c = c_new - c_old
        dt_divj = args.dt * div
        residual = delta_c - dt_divj
        residual_host[p] = residual
        cancellation = ((abs(delta_c) + abs(dt_divj)) / abs(residual)
                        if residual != 0.0 else math.inf)
        backward_scale = abs(c_new) + abs(c_old) + abs(dt_divj)
        backward = abs(residual) / max(backward_scale, math.ulp(1.0))
        decomposition.append({
            "idx": p,
            "i": i,
            "j": j,
            "k": k,
            "DeltaC": f"{delta_c:.17e}",
            "dt_divJ_host": f"{dt_divj:.17e}",
            "R_host": f"{residual:.17e}",
            "R_gpu_cold": row["cold_residual"],
            "cancellation_ratio": f"{cancellation:.17e}",
            "normalized_backward_error": f"{backward:.17e}",
        })

    write_rows(
        args.output_dir / "residual_term_decomposition.csv",
        list(decomposition[0]), decomposition,
    )

    runtime_r = [as_float(row, "cold_residual") for row in frozen]
    runtime_div = [as_float(row, "divJ") for row in frozen]
    runtime_fx = [as_float(row, "face_x") for row in frozen]
    runtime_fy = [as_float(row, "face_y") for row in frozen]
    runtime_fz = [as_float(row, "face_z") for row in frozen]
    host_linf = max(abs(value) for value in residual_host)
    host_worst = max(range(n), key=lambda p: abs(residual_host[p]))
    host_l2 = math.sqrt(math.fsum(value * value for value in residual_host))
    host_l2_rms = host_l2 / math.sqrt(n)
    runtime_gate = float(summary["runtime_gate"])
    host_wrms_gate = math.sqrt(math.fsum(
        (value / runtime_gate) ** 2 for value in residual_host
    ) / n)
    host_sum_fsum = math.fsum(residual_host)
    host_sumsq_fsum = math.fsum(value * value for value in residual_host)
    sequential_sum = 0.0
    sequential_sumsq = 0.0
    for value in residual_host:
        sequential_sum += value
        sequential_sumsq += value * value

    comparisons = [{
        "runtime_residual_Linf": summary["runtime_residual"],
        "cold_gpu_residual_Linf": summary["cold_residual"],
        "deterministic_gpu_residual_Linf": summary["gpu_deterministic_linf"],
        "host_fp64_residual_Linf": f"{host_linf:.17e}",
        "runtime_gate": summary["runtime_gate"],
        "host_worst_idx": host_worst,
        "gpu_worst_idx": summary["gpu_deterministic_worst_idx"],
        "host_L2": f"{host_l2:.17e}",
        "host_L2_RMS": f"{host_l2_rms:.17e}",
        "host_WRMS_over_runtime_gate": f"{host_wrms_gate:.17e}",
        "face_x_Linf_host_gpu": f"{max(abs(a-b) for a,b in zip(fx_host,runtime_fx)):.17e}",
        "face_y_Linf_host_gpu": f"{max(abs(a-b) for a,b in zip(fy_host,runtime_fy)):.17e}",
        "face_z_Linf_host_gpu": f"{max(abs(a-b) for a,b in zip(fz_host,runtime_fz)):.17e}",
        "divJ_Linf_host_gpu": f"{max(abs(a-b) for a,b in zip(div_host,runtime_div)):.17e}",
        "residual_Linf_host_gpu": f"{max(abs(a-b) for a,b in zip(residual_host,runtime_r)):.17e}",
    }]
    write_rows(
        args.output_dir / "runtime_vs_cold_residual.csv",
        list(comparisons[0]), comparisons,
    )

    reductions = [
        {"reduction": "gpu_runtime_current", "Linf": summary["cold_residual"],
         "sum": "not_recorded", "sumsq": "not_recorded", "worst_idx": "not_recorded"},
        {"reduction": "gpu_deterministic_fixed_order", "Linf": summary["gpu_deterministic_linf"],
         "sum": summary["gpu_deterministic_sum"], "sumsq": summary["gpu_deterministic_sumsq"],
         "worst_idx": summary["gpu_deterministic_worst_idx"]},
        {"reduction": "host_fp64_sequential", "Linf": f"{host_linf:.17e}",
         "sum": f"{sequential_sum:.17e}", "sumsq": f"{sequential_sumsq:.17e}",
         "worst_idx": host_worst},
        {"reduction": "host_fp64_fsum", "Linf": f"{host_linf:.17e}",
         "sum": f"{host_sum_fsum:.17e}", "sumsq": f"{host_sumsq_fsum:.17e}",
         "worst_idx": host_worst},
    ]
    write_rows(args.output_dir / "reduction_comparison.csv", list(reductions[0]), reductions)
    write_rows(args.output_dir / "transport_polish_matrix.csv", list(polish[0]), polish)
    write_rows(args.output_dir / "aref_invariance.csv", list(aref[0]), aref)

    runtime_residual = float(summary["runtime_residual"])
    cold_residual = float(summary["cold_residual"])
    polish_pass = any(int(row["original_gate_pass"]) == 1 for row in polish)
    if cold_residual <= runtime_gate < runtime_residual:
        classification = "STALE_OR_INCONSISTENT_RUNTIME_RESIDUAL"
        recommendation = "FIX_RUNTIME_RESIDUAL_REFRESH_AND_REPLAY_STEP655"
    elif polish_pass:
        classification = "FIXED_PHI_TRANSPORT_POLISH_CLOSES_RESIDUAL"
        recommendation = "RETAIN_BLOCK_M3_AND_ENTER_STAGE_E_QUALIFICATION"
    else:
        classification = "FIXED_PHI_TRANSPORT_NONLINEAR_FLOOR"
        recommendation = "IMPLEMENT_BOUND_AWARE_FIXED_PHI_TRANSPORT_JFNK"

    aref_hash_fields = [
        "C_hash", "phi_hash", "q_hash", "x_hash", "mu_hash", "face_x_hash",
        "face_y_hash", "face_z_hash", "divJ_hash", "residual_hash",
        "active_mask_hash", "phase_raw_hash", "phase_KKT_hash",
    ]
    aref_hash_invariant = all(
        len({row[field] for row in aref}) == 1 for field in aref_hash_fields
    )
    result = {
        "grid": [nx, ny, nz],
        "dt": args.dt,
        "runtime_residual": runtime_residual,
        "cold_residual": cold_residual,
        "runtime_cold_difference": cold_residual - runtime_residual,
        "runtime_gate": runtime_gate,
        "worst_cell_index": host_worst,
        "DeltaC_scale": max(abs(as_float(row, "C_trial") - as_float(row, "Cn")) for row in frozen),
        "dt_divJ_scale": max(abs(args.dt * value) for value in div_host),
        "cancellation_ratio_worst": float(decomposition[host_worst]["cancellation_ratio"]),
        "normalized_transport_backward_error_worst": float(
            decomposition[host_worst]["normalized_backward_error"]),
        "host_double_residual": host_linf,
        "reduction_order_sensitivity_Linf": max(
            abs(host_linf - float(summary["gpu_deterministic_linf"])),
            abs(host_linf - cold_residual),
        ),
        "a_ref_physics_array_invariance": aref_hash_invariant,
        "residual_floor_classification": classification,
        "recommended_next_action": recommendation,
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "analysis_summary.json").write_text(
        json.dumps(result, indent=2, sort_keys=True) + "\n"
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
