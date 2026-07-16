#!/usr/bin/env python3
"""Summarize the final P1 cubic cost profile without changing runtime data."""

from __future__ import annotations

import argparse
import csv
import importlib.util
import re
import sys
from collections import defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MEMORY_SCRIPT = ROOT / "scripts" / "audit_pf_ctot_performance_memory.py"
SPEC = importlib.util.spec_from_file_location("memory_ledger", MEMORY_SCRIPT)
memory_ledger = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules[SPEC.name] = memory_ledger
SPEC.loader.exec_module(memory_ledger)

CASES = (
    ("cubic32_elastic_off", 32, 0),
    ("cubic32_elastic_on", 32, 1),
    ("cubic64_elastic_off", 64, 0),
    ("cubic64_elastic_on", 64, 1),
)


def rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def nsys_rows(path: Path) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    start = next(index for index, line in enumerate(lines) if line.startswith("Time (%)"))
    return list(csv.DictReader(lines[start:]))


def one_result(case_dir: Path, name: str) -> Path:
    matches = list((case_dir / "results").glob(f"**/{name}"))
    if len(matches) != 1:
        raise RuntimeError(f"expected one {name} below {case_dir}, got {matches}")
    return matches[0]


def stage_sums(path: Path) -> dict[str, float]:
    result: dict[str, float] = defaultdict(float)
    for row in rows(path):
        result[row["stage"]] += float(row["gpu_ms"])
    return dict(result)


def api_metrics(path: Path) -> dict[str, int]:
    result = {"runtime_launches": 0, "driver_launches": 0, "syncs": 0}
    for row in nsys_rows(path):
        name = row["Name"]
        calls = int(row["Num Calls"])
        if name == "cudaLaunchKernel":
            result["runtime_launches"] = calls
        elif name == "cuLaunchKernel":
            result["driver_launches"] = calls
        elif name == "cudaDeviceSynchronize":
            result["syncs"] = calls
    return result


def kernel_metrics(path: Path) -> dict[str, float | int]:
    result: dict[str, float | int] = {
        "total_ns": 0.0,
        "fft_ns": 0.0,
        "fft_instances": 0,
        "r2c_instances": 0,
        "c2r_instances": 0,
        "trial_compare_ns": 0.0,
        "trial_compare_instances": 0,
        "elastic_specific_ns_lower_bound": 0.0,
    }
    elastic_tokens = (
        "dealias_23", "displacement", "strain", "stress", "eigenstrain",
        "hij_green", "normalized_diff_complex_float",
    )
    for row in nsys_rows(path):
        name = row["Name"]
        elapsed = float(row["Total Time (ns)"])
        instances = int(row["Instances"])
        result["total_ns"] += elapsed
        if "fft" in name.lower():
            result["fft_ns"] += elapsed
            result["fft_instances"] += instances
        if "vector_fft_r2c" in name:
            result["r2c_instances"] += instances
        if "vector_fft_c2r" in name:
            result["c2r_instances"] += instances
        if "ctot_phase_pdas_compare_trial_active_kernel" in name:
            result["trial_compare_ns"] += elapsed
            result["trial_compare_instances"] += instances
        if any(token in name for token in elastic_tokens):
            result["elastic_specific_ns_lower_bound"] += elapsed
    return result


def source_memory(size: int, elastic: bool) -> tuple[dict[str, int], int]:
    n = size**3
    k = size * size * (size // 2 + 1)
    categories: dict[str, int] = defaultdict(int)
    total = 0
    for item in memory_ledger.ALLOCATIONS:
        if elastic and not item.elastic_on:
            continue
        if not elastic and not item.elastic_off:
            continue
        count = memory_ledger.count(item.count_expr, n, k)
        amount = count * item.bytes_per_element
        categories[item.category] += amount
        total += amount
    return dict(categories), total


def write(path: Path, fieldnames: list[str], data: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(data)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile-root", type=Path, required=True)
    parser.add_argument("--report-dir", type=Path, required=True)
    args = parser.parse_args()
    report = args.report_dir.resolve()
    timing_source = {row["case"]: row for row in rows(report / "next_cubic_unprofiled_timing_source.csv")}
    d2h_source = {row["case"]: row for row in rows(report / "next_cubic_d2h_source.csv")}
    stage_output: list[dict[str, object]] = []
    iteration_output: list[dict[str, object]] = []
    measured_peak: dict[tuple[int, int], float] = {}

    for case, size, elastic in CASES:
        case_dir = args.profile_root / case
        log = (case_dir / "run.log").read_text()
        accepts = re.findall(r"CTOT_MIMETIC_BE_ACCEPT[^\n]+", log)
        rejects = len(re.findall(r"CTOT_MIMETIC_BE_REJECT", log))
        retries = len(re.findall(r"CTOT_MIMETIC_BE_RETRY", log))
        accept_nonlinear = [int(re.search(r"nonlinear_iters=(\d+)", line).group(1)) for line in accepts]
        accept_outer = [int(re.search(r"outer_iters=(\d+)", line).group(1)) for line in accepts]
        summary = rows(one_result(case_dir, "performance_summary.csv"))[0]
        stages = stage_sums(one_result(case_dir, "ctot_performance_stage_timing.csv"))
        phase_rows = rows(one_result(case_dir, "ctot_phase_kkt_iterations.csv"))
        api = api_metrics(case_dir / "nsys_cuda_api_sum.csv")
        kernels = kernel_metrics(case_dir / "nsys_cuda_gpu_kern_sum.csv")
        timing = timing_source[case]
        d2h = d2h_source[case]
        attempt_ms = stages.get("attempt.total", 0.0)
        outer_ms = stages.get("outer.iteration", 0.0)
        phase_ms = stages.get("phase.pdas_solve", 0.0)
        transport_ms = stages.get("transport.residual_evaluation", 0.0)
        pcg_ms = stages.get("phase.pcg_solve", 0.0)
        jv_ms = stages.get("phase.pcg_jv", 0.0)
        outer_remainder_ms = max(0.0, outer_ms - phase_ms - transport_ms)
        audit_commit_ms = max(0.0, attempt_ms - outer_ms)
        stepping_wall = float(timing["stepping_wall_s"])
        accepted = len(accepts)
        stage_output.append({
            "case": case, "grid": f"{size}^3", "elastic_enabled": elastic,
            "dt": float(summary["dt"]), "attempted_steps": accepted + rejects,
            "accepted_steps": accepted, "rejected_steps": rejects,
            "retry_count": retries,
            "external_process_wall_s": float(timing["external_process_wall_s"]),
            "stepping_wall_s": stepping_wall,
            "wall_per_accepted_step_s": float(timing["avg_wall_per_accepted_step_s"]),
            "median_wall_per_accepted_step_s": float(timing["median_wall_per_accepted_step_s"]),
            "accepted_code_time_per_wall_hour": accepted * float(summary["dt"]) / stepping_wall * 3600.0,
            "attempt_gpu_ms": attempt_ms, "outer_gpu_ms": outer_ms,
            "phase_pdas_gpu_ms": phase_ms, "transport_gpu_ms": transport_ms,
            "pcg_gpu_ms": pcg_ms, "phase_jv_gpu_ms": jv_ms,
            "fft_gpu_ms": float(kernels["fft_ns"]) / 1.0e6,
            "phase_trial_compare_gpu_ms": float(kernels["trial_compare_ns"]) / 1.0e6,
            "elastic_specific_gpu_ms_lower_bound": float(kernels["elastic_specific_ns_lower_bound"]) / 1.0e6,
            "outer_unattributed_coupling_gpu_ms": outer_remainder_ms,
            "audit_commit_gpu_ms": audit_commit_ms,
            "startup_io_teardown_wall_s": max(0.0, float(timing["external_process_wall_s"]) - stepping_wall),
            "phase_fraction_of_attempt_gpu": phase_ms / attempt_ms,
            "transport_fraction_of_attempt_gpu": transport_ms / attempt_ms,
            "fft_fraction_of_profiled_kernel": float(kernels["fft_ns"]) / float(kernels["total_ns"]),
            "trial_compare_fraction_of_profiled_kernel": float(kernels["trial_compare_ns"]) / float(kernels["total_ns"]),
        })
        green = [int(value) for value in re.findall(r"Green_iterations=(\d+)", log)]
        memory_rows = rows(case_dir / "gpu_memory_samples.csv")
        peak = max(float(row["used_memory_MiB"]) for row in memory_rows)
        measured_peak[(size, elastic)] = peak
        iteration_output.append({
            "case": case, "grid": f"{size}^3", "elastic_enabled": elastic,
            "accepted_steps": accepted,
            "transport_nonlinear_iterations_total": sum(accept_nonlinear),
            "phase_iteration_records": len(phase_rows),
            "phase_linear_pcg_iterations_total": sum(int(row["linear_iterations"]) for row in phase_rows),
            "phase_linear_pcg_iterations_max": max(int(row["linear_iterations"]) for row in phase_rows),
            "outer_iterations_total": sum(accept_outer),
            "outer_iterations_per_step": sum(accept_outer) / accepted,
            "Green_iterations_total": sum(green),
            "Green_iterations_max_per_solve": max(green) if green else 0,
            "runtime_kernel_launch_count": api["runtime_launches"],
            "driver_kernel_launch_count": api["driver_launches"],
            "global_device_synchronize_count": api["syncs"],
            "D2H_total_count": int(d2h["d2h_total"]),
            "D2H_scalar_status_count_le_32_bytes": int(d2h["d2h_scalar_status_le_32_bytes"]),
            "D2H_reduction_or_small_field_count": int(d2h["d2h_reduction_or_small_field_gt32_lt_full"]),
            "D2H_full_field_count": int(d2h["d2h_full_field_ge_262144"]),
            "FFT_implementation_kernel_instances": int(kernels["fft_instances"]),
            "R2C_implementation_kernel_instances": int(kernels["r2c_instances"]),
            "C2R_implementation_kernel_instances": int(kernels["c2r_instances"]),
            "phase_trial_compare_instances": int(kernels["trial_compare_instances"]),
        })

    stage_fields = list(stage_output[0])
    iteration_fields = list(iteration_output[0])
    write(report / "next_cubic_stage_timing.csv", stage_fields, stage_output)
    write(report / "next_cubic_iteration_counts.csv", iteration_fields, iteration_output)

    overhead_gib: dict[int, float] = {}
    for elastic in (0, 1):
        _, source_bytes = source_memory(64, bool(elastic))
        overhead_gib[elastic] = max(
            0.0, measured_peak[(64, elastic)] / 1024.0 - source_bytes / 2**30
        )
    memory_output: list[dict[str, object]] = []
    for size in (32, 64, 128, 256, 320):
        for elastic in (0, 1):
            categories, source_bytes = source_memory(size, bool(elastic))
            source_gib = source_bytes / 2**30
            subtotal = source_gib + overhead_gib[elastic]
            safety = 0.20
            memory_output.append({
                "grid": f"{size}^3", "elastic_enabled": elastic,
                "measured_process_peak_MiB": measured_peak.get((size, elastic), ""),
                "persistent_accepted_and_checkpoint_GiB": categories.get("persistent_accepted_state", 0) / 2**30,
                "persistent_trial_GiB": categories.get("persistent_trial_state", 0) / 2**30,
                "transport_scratch_GiB": categories.get("transport_scratch", 0) / 2**30,
                "phase_scratch_GiB": categories.get("phase_PDAS_scratch", 0) / 2**30,
                "fft_state_scratch_GiB": categories.get("FFT_state_scratch", 0) / 2**30,
                "mechanical_GiB": categories.get("mechanical_state_scratch", 0) / 2**30,
                "diagnostic_GiB": categories.get("diagnostic_status", 0) / 2**30,
                "source_device_allocations_GiB": source_gib,
                "cufft_external_work_area_GiB": 0.0,
                "calibrated_context_allocator_overhead_GiB": overhead_gib[elastic],
                "subtotal_before_safety_GiB": subtotal,
                "safety_margin_fraction": safety,
                "estimated_peak_with_safety_GiB": subtotal * (1.0 + safety),
            })
    write(report / "next_cubic_memory_scaling.csv", list(memory_output[0]), memory_output)
    print("next_cubic_profile_analysis_complete")
    print(f"stage_rows={len(stage_output)}")
    print(f"iteration_rows={len(iteration_output)}")
    print(f"memory_rows={len(memory_output)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
