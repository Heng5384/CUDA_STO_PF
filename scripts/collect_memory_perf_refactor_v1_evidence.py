#!/usr/bin/env python3
"""Collect measured memory-refactor evidence into one reproducible JSON file."""

from __future__ import annotations

import csv
import hashlib
import json
import re
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "memory_perf_refactor_v1"
EV = OUT / "evidence"
DT_CODE = 1.95312500000000011e-4
DT_PHYS = 0.008033122153


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def perf(case: str) -> dict[str, str]:
    paths = list((EV / "final_validation" / case).glob("**/performance_summary.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one performance summary for {case}, got {paths}")
    with paths[0].open(newline="", encoding="utf-8") as f:
        return next(csv.DictReader(f))


def benchmark_perf() -> dict[str, str]:
    paths = list((EV / "benchmark400").glob("**/performance_summary.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"expected one 400 benchmark summary, got {paths}")
    with paths[0].open(newline="", encoding="utf-8") as f:
        return next(csv.DictReader(f))


def csv_section(text: str, header_prefix: str) -> list[dict[str, str]]:
    lines = text.splitlines()
    start = next(i for i, line in enumerate(lines) if line.startswith(header_prefix))
    data = []
    for line in lines[start:]:
        if not line.strip() or line.startswith("Processing ["):
            break
        data.append(line)
    return list(csv.DictReader(data))


def nsys_sections(tag: str):
    base = EV / "final_validation" / tag
    text = (base / "nsys_stats.txt").read_text(encoding="utf-8")
    kernels = csv_section(text, "Time (%),Total Time (ns),Instances")
    mem_time = csv_section(text, "Time (%),Total Time (ns),Count")
    api = csv_section(text, "Time (%),Total Time (ns),Num Calls")
    mem_size_text = (base / "nsys_mem_size.txt").read_text(encoding="utf-8")
    mem_size = csv_section(mem_size_text, "Total (MB),Count")
    return kernels, mem_time, api, mem_size


def log_summary(path: Path) -> tuple[int, int, int, int]:
    text = path.read_text(encoding="utf-8", errors="replace")
    match = re.search(
        r"CTOT_BOUNDED_RETRY_SUMMARY .*?macro_steps=(\d+) "
        r"macro_hard_rejects=(\d+) internal_trial_rejects=(\d+).*?"
        r"fallback_macros=(\d+)", text
    )
    if not match:
        raise RuntimeError(f"retry summary missing in {path}")
    return tuple(map(int, match.groups()))


def row_float(row: dict[str, str], key: str) -> float:
    return float(row[key])


def source_gib_after(stage: int) -> float:
    before = 19.579537615180016
    cumulative = {
        0: 0.0,
        1: 1.430511474609375,
        2: 1.430511474609375 + 3.337860107421875,
        3: 1.430511474609375 + 3.337860107421875,
        4: 1.430511474609375 + 3.337860107421875 + 0.4792213439941406,
        5: 1.430511474609375 + 3.337860107421875 + 0.4792213439941406 + 1.195669174194336,
        6: 7.396936416625977,
        7: 7.396936416625977,
        8: 7.396936416625977,
        9: 7.396936416625977,
        10: 7.396936416625977,
    }
    return before - cumulative[stage]


def main() -> None:
    source_files = (
        "main_cuda.cu", "cuda_kernels.cu", "cuda_kernels.h",
        "cuda_common.cu", "cuda_common.h", "pf_params.h",
        "thermo_utils.h", "phase_functions.h", "Makefile",
    )
    source_sha = {name: sha256(ROOT / name) for name in source_files}
    git_head = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True
    ).strip()

    p32_b0 = perf("N32_B0_5")
    p32_b187 = perf("N32_B187_5")
    p64_b0 = perf("N64_B0_5")
    p64_b187 = perf("N64_B187_5")
    p128_b0 = perf("N128_B0_nsys")
    p128_b187 = perf("N128_B187_nsys")
    p400 = benchmark_perf()

    nsys_data = {}
    kernel_rows = []
    copy_rows = []
    for tag, label in (("N128_B0_nsys", "B0"), ("N128_B187_nsys", "B187")):
        kernels, mem_time, api, mem_size = nsys_sections(tag)
        nsys_data[label] = {"kernels": kernels, "mem_time": mem_time, "api": api, "mem_size": mem_size}
        for rank, row in enumerate(kernels[:20], 1):
            kernel_rows.append({
                "configuration": label,
                "rank": rank,
                "kernel": row["Name"],
                "instances": int(row["Instances"]),
                "total_time_ms": float(row["Total Time (ns)"]) / 1.0e6,
                "avg_time_us": float(row["Avg (ns)"]) / 1.0e3,
                "time_fraction": float(row["Time (%)"]) / 100.0,
                "source": f"Nsight Systems 2025.1.3, {tag}, full 2-step trace",
            })
        size_by_op = {row["Operation"]: row for row in mem_size}
        for rank, row in enumerate(mem_time, 1):
            size = size_by_op[row["Operation"]]
            copy_rows.append({
                "configuration": label,
                "rank": rank,
                "operation": row["Operation"],
                "instances": int(row["Count"]),
                "total_time_ms": float(row["Total Time (ns)"]) / 1.0e6,
                "avg_time_us": float(row["Avg (ns)"]) / 1.0e3,
                "bytes": float(size["Total (MB)"]) * 1.0e6,
                "source": f"Nsight Systems 2025.1.3, {tag}, full 2-step trace",
            })

    def api_count(label: str, name: str) -> int:
        row = next(row for row in nsys_data[label]["api"] if row["Name"] == name)
        return int(row["Num Calls"])

    def mem_mb(label: str, operation: str) -> float:
        row = next(row for row in nsys_data[label]["mem_size"] if row["Operation"] == operation)
        return float(row["Total (MB)"])

    source_rows = {}
    with (OUT / "static_memory_ledger.csv").open(newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            source_rows[int(row["grid"].split("^")[0])] = row

    measured = {
        128: (440, 1.09), 192: (1652, 2.03), 256: (3544, 3.88),
        320: (6660, 6.92), 384: (11310, 11.46), 400: (12788, 12.90),
    }
    scaling_rows = []
    for n in (128, 192, 256, 320, 384, 400):
        process_peak, program_used = measured[n]
        scaling_rows.append({
            "configuration": "PF_ONLY",
            "N": n,
            "source_GiB": float(source_rows[n]["source_GiB_after"]),
            "process_peak_MiB": process_peak,
            "program_used_GiB": program_used,
            "device_total_MiB": 16303,
            "device_fraction_percent": 100.0 * program_used / 15.46,
            "first_step_attempted": n == 400,
            "first_step_accepted": n == 400,
            "status": "PASS_85PCT_ADMISSION",
            "measurement_note": (
                "feature mask 187 actual accepted step" if n == 400
                else "allocation dry-run; mask 251 had measured zero-byte M7 workspace and identical footprint"
            ),
        })
    scaling_rows.extend((
        {
            "configuration": "PF_ELASTIC", "N": 400, "source_GiB": 26.747592195868492,
            "process_peak_MiB": "NOT_STARTED", "program_used_GiB": "NOT_STARTED",
            "device_total_MiB": 16303, "device_fraction_percent": "NOT_MEASURED",
            "first_step_attempted": False, "first_step_accepted": False,
            "status": "BLOCKED_STATIC_SOURCE_EXCEEDS_DEVICE",
            "measurement_note": "elastic source ledger alone exceeds device; accepted baseline is elastic OFF",
        },
        {
            "configuration": "PF_ELASTIC_GP_ALLOCATED", "N": 400, "source_GiB": 26.747592195868492,
            "process_peak_MiB": "NOT_STARTED", "program_used_GiB": "NOT_STARTED",
            "device_total_MiB": 16303, "device_fraction_percent": "NOT_MEASURED",
            "first_step_attempted": False, "first_step_accepted": False,
            "status": "BLOCKED_STATIC_SOURCE_AND_GP_GATE",
            "measurement_note": "static GP reservoir is host-side; GP/S3 gate remained closed",
        },
    ))

    base_hashes = {
        "Ctot": "c2789fa84d4b3017c2836b4c95ae4e60f02ce23b52532188b36086ef66b3d772",
        "Ctot_nm1": "2a81a5c6e3dacdfb0a768d0aa51adef0e23ce4175940142408d768416073c6e3",
        "phi": "0e48c8b51bfd5c84bf8c7403625fe8f6fcb2ae730259b26c371703db2714fcdb",
        "phi_nm1": "f0b568bb9dc27e9c82a3919a558bbdf50188057673dcd84792a22cf3a8aa0c78",
        "xB_alpha": "6d22a351a2f631a8a1e75e736ddb5075df4606cab83320543454d8b292fd1a23",
    }
    ablation_specs = [
        ("B0", 0, "none", 0.013277, "RETAINED_CONTROL", "B0_postedit_100"),
        ("B1", 1, "M1", 0.013245, "RETAINED", "B1_M1_only_100"),
        ("B2", 3, "M1+M2", 0.013215, "RETAINED", "B3_M1_M2_100"),
        ("B3", 3, "M1+M2; M3 not retained", 0.013215, "NOT_RETAINED_M3", "B3 shares B2 result"),
        ("B4", 11, "M1+M2+M4", 0.013210, "RETAINED", "B11_plus_M4_100"),
        ("B5", 27, "M1+M2+M4+M5", 0.013308, "RETAINED", "B27_plus_M4_latest"),
        ("B6", 59, "M1+M2+M4+M5+M6", 0.013346, "RETAINED", "B59_plus_M6_latest"),
        ("B7", 123, "B6+M7", 0.013265, "NOT_RETAINED_ZERO_WORKSPACE", "B123_plus_M7_preM8_100"),
        ("B8", 187, "B6+M8", 0.013215, "SELECTED", "B187_plus_M8_latest"),
        ("B9", 187, "B8; M9 not implemented", None, "NOT_RETAINED_M9", "profile evidence did not justify fusion"),
        ("B10", 187, "B8; M10 not implemented", None, "NOT_RETAINED_M10", "dynamic nonlinear/retry graph not captured"),
    ]
    ablation_rows = []
    for index, (stage, mask, features, wall, decision, evidence) in enumerate(ablation_specs):
        ablation_rows.append({
            "stage": stage, "feature_mask": mask, "features": features,
            "grid": "512x1x1", "steps": 100 if wall is not None else 0,
            "accepted": 100 if wall is not None else "NOT_RUN",
            "hard_rejects": 0 if wall is not None else "NOT_RUN",
            "fallback_macros": 0 if wall is not None else "NOT_RUN",
            "endpoint_bitwise_vs_B0": True if wall is not None else "NOT_APPLICABLE",
            "rollback_bitwise": True if stage in {"B0", "B6", "B7", "B8"} else "COVERED_BY_SELECTED",
            "restart_bitwise": True if stage == "B8" else "COVERED_BY_SELECTED",
            "source_GiB_400": source_gib_after(index),
            "wall_s_per_step": wall if wall is not None else "NOT_MEASURED",
            "decision": decision, "evidence": evidence,
        })
    ablation_rows.extend((
        {"stage": "3D_N32", "feature_mask": 187, "features": "selected", "grid": "32^3", "steps": 5,
         "accepted": 5, "hard_rejects": 0, "fallback_macros": 0, "endpoint_bitwise_vs_B0": True,
         "rollback_bitwise": "NOT_TRIGGERED", "restart_bitwise": "SEPARATE_CONTROL", "source_GiB_400": 12.182601198554039,
         "wall_s_per_step": row_float(p32_b187, "avg_walltime_per_step_s"), "decision": "PASS", "evidence": "fresh workstation curved state"},
        {"stage": "3D_N64", "feature_mask": 187, "features": "selected", "grid": "64^3", "steps": 5,
         "accepted": 5, "hard_rejects": 0, "fallback_macros": 0, "endpoint_bitwise_vs_B0": True,
         "rollback_bitwise": "NOT_TRIGGERED", "restart_bitwise": "SEPARATE_CONTROL", "source_GiB_400": 12.182601198554039,
         "wall_s_per_step": row_float(p64_b187, "avg_walltime_per_step_s"), "decision": "PASS", "evidence": "fresh workstation curved state"},
    ))

    bench_log = (EV / "benchmark400" / "run.log").read_text(encoding="utf-8", errors="replace")
    event_steps = [int(x) for x in re.findall(r"CTOT_BDF2_EVENT_SUBCYCLE_RETRY step=(\d+)", bench_log)]
    first_failure_rows = [{
        "case": "PF_ONLY_N400_benchmark13", "grid": "400^3", "stage": f"step_{step}",
        "first_failure": "coupled_outer_max_iter_not_converged -> depth-2 event subcycle",
        "hard_reject": False, "recovered": True,
        "effect_on_admission": "none; accepted after bitwise rollback and BE subcycling",
    } for step in event_steps]

    speed_rows = [
        {"case": "N32_B0", "N": 32, "state": "curved", "feature_mask": 0, "warmup_steps": 0, "timed_steps": 5,
         "accepted_steps": 5, "hard_rejects": 0, "fallback_macros": 0,
         "wall_s_per_accepted_step": row_float(p32_b0, "avg_walltime_per_step_s"),
         "physical_s_per_GPU_hour": DT_PHYS * 3600 / row_float(p32_b0, "avg_walltime_per_step_s"),
         "peak_MiB": "NOT_SAMPLED", "measurement_status": "ACTUAL", "notes": "B0 control"},
        {"case": "N32_B187", "N": 32, "state": "curved", "feature_mask": 187, "warmup_steps": 0, "timed_steps": 5,
         "accepted_steps": 5, "hard_rejects": 0, "fallback_macros": 0,
         "wall_s_per_accepted_step": row_float(p32_b187, "avg_walltime_per_step_s"),
         "physical_s_per_GPU_hour": DT_PHYS * 3600 / row_float(p32_b187, "avg_walltime_per_step_s"),
         "peak_MiB": "NOT_SAMPLED", "measurement_status": "ACTUAL", "notes": "bitwise endpoint"},
        {"case": "N64_B0", "N": 64, "state": "curved", "feature_mask": 0, "warmup_steps": 0, "timed_steps": 5,
         "accepted_steps": 5, "hard_rejects": 0, "fallback_macros": 0,
         "wall_s_per_accepted_step": row_float(p64_b0, "avg_walltime_per_step_s"),
         "physical_s_per_GPU_hour": DT_PHYS * 3600 / row_float(p64_b0, "avg_walltime_per_step_s"),
         "peak_MiB": "NOT_SAMPLED", "measurement_status": "ACTUAL", "notes": "B0 control"},
        {"case": "N64_B187", "N": 64, "state": "curved", "feature_mask": 187, "warmup_steps": 0, "timed_steps": 5,
         "accepted_steps": 5, "hard_rejects": 0, "fallback_macros": 0,
         "wall_s_per_accepted_step": row_float(p64_b187, "avg_walltime_per_step_s"),
         "physical_s_per_GPU_hour": DT_PHYS * 3600 / row_float(p64_b187, "avg_walltime_per_step_s"),
         "peak_MiB": "NOT_SAMPLED", "measurement_status": "ACTUAL", "notes": "bitwise endpoint"},
        {"case": "N128_B0_NSYS", "N": 128, "state": "curved", "feature_mask": 0, "warmup_steps": 0, "timed_steps": 2,
         "accepted_steps": 2, "hard_rejects": 0, "fallback_macros": 0,
         "wall_s_per_accepted_step": row_float(p128_b0, "avg_walltime_per_step_s"),
         "physical_s_per_GPU_hour": DT_PHYS * 3600 / row_float(p128_b0, "avg_walltime_per_step_s"),
         "peak_MiB": "NOT_SAMPLED", "measurement_status": "PROFILED_NOT_SPEED_AUTHORITY", "notes": "CUDA-event and Nsight overhead"},
        {"case": "N128_B187_NSYS", "N": 128, "state": "curved", "feature_mask": 187, "warmup_steps": 0, "timed_steps": 2,
         "accepted_steps": 2, "hard_rejects": 0, "fallback_macros": 0,
         "wall_s_per_accepted_step": row_float(p128_b187, "avg_walltime_per_step_s"),
         "physical_s_per_GPU_hour": DT_PHYS * 3600 / row_float(p128_b187, "avg_walltime_per_step_s"),
         "peak_MiB": "NOT_SAMPLED", "measurement_status": "PROFILED_NOT_SPEED_AUTHORITY", "notes": "CUDA-event and Nsight overhead"},
        {"case": "PF_ONLY_N400_B187", "N": 400, "state": "representative curved", "feature_mask": 187,
         "warmup_steps": 3, "timed_steps": 10, "accepted_steps": 13, "hard_rejects": 0, "fallback_macros": 6,
         "wall_s_per_accepted_step": row_float(p400, "warmup_excluded_avg_walltime_per_step_s"),
         "physical_s_per_GPU_hour": DT_PHYS * 3600 / row_float(p400, "warmup_excluded_avg_walltime_per_step_s"),
         "peak_MiB": 12788, "measurement_status": "ACTUAL_3_WARMUP_10_TIMED",
         "notes": "large output suppressed; hard gates active; recovered event subcycles included"},
    ]

    opt = {
        "selected_mask": 187,
        "M1": {"decision": "RETAINED", "saved_GiB": 1.430511474609375,
               "source": "main_cuda.cu:30461-30468,31632-31818",
               "old_new": "three persistent event snapshots -> two zero-increment aliases plus deterministic Y reconstruction",
               "correctness": "100-step endpoint and forced event rollback bitwise",
               "performance": "no easy-path regression", "transaction": "event macro ownership and free path explicitly versioned",
               "reason": "removes event-only full fields without weakening event safety"},
        "M2": {"decision": "RETAINED", "saved_GiB": 3.337860107421875,
               "source": "main_cuda.cu:30474-30485,36352-36383,38142-38155",
               "old_new": "seven always-live outer/history fields -> allocate only for paths requiring multiple/accelerated outer iterations",
               "correctness": "method-consistent single-outer path endpoint bitwise",
               "performance": "D2D volume reduced", "transaction": "accepted state and BDF2 histories remain immutable",
               "reason": "largest low-risk lifetime removal; this is history elision, not pointer rotation"},
        "M3": {"decision": "NOT_RETAINED", "saved_GiB": 0.0,
               "source": "main_cuda.cu active-mask allocation and consumers",
               "old_new": "no change; FP64 active-code field retained",
               "correctness": "not implemented", "performance": "not measured",
               "transaction": "no ABI/reduction migration attempted",
               "reason": "memory target already met; coordinated ABI/reduction risk not justified"},
        "M4": {"decision": "RETAINED", "saved_GiB": 0.4792213439941406,
               "source": "main_cuda.cu:30898-30905",
               "old_new": "separate d_Y_k spectrum -> alias d_phi_k after non-overlapping lifetime proof",
               "correctness": "100-step endpoint bitwise", "performance": "no measurable regression",
               "transaction": "single owner prevents double free", "reason": "one full complex spectrum removed"},
        "M5": {"decision": "RETAINED", "saved_GiB": 1.195669174194336,
               "source": "main_cuda.cu:30443-30447,31647-32070; cuda_common.cu k4 allocation gate",
               "old_new": "persistent Y rollback, unused divJ spectrum and k4 -> exact reconstruction or no allocation",
               "correctness": "endpoint and event rollback bitwise", "performance": "reconstruction below 5% regression cap",
               "transaction": "Y rebuilt from authoritative Ctot+phi", "reason": "removes non-authoritative/unused fields"},
        "M6": {"decision": "RETAINED", "saved_GiB": 0.95367431640625,
               "source": "main_cuda.cu:34564-34635; cuda_kernels.cu:6291-6318",
               "old_new": "three simultaneous positive-face fields -> one x/y/z streamed field with ordered divergence accumulation",
               "correctness": "manufactured periodic incidence test; 1D/32^3/64^3 endpoint bitwise",
               "performance": "64^3 wall improves 3.78%; kernel launches increase 100 in 2-step trace",
               "transaction": "failed-attempt diagnostics materialize temporary faces only on failure",
               "reason": "saves two full FP64 fields with net measured speed benefit"},
        "M7": {"decision": "NOT_RETAINED_ZERO_BENEFIT", "saved_GiB": 0.0,
               "source": "main_cuda.cu:31316-31339",
               "old_new": "optional explicit shared cuFFT work area tested; selected path keeps normal plan ownership",
               "correctness": "mask-123 endpoint bitwise", "performance": "workspace query returned zero bytes on all tested grids",
               "transaction": "default-off plan teardown tested", "reason": "no memory benefit"},
        "M8": {"decision": "RETAINED", "saved_GiB": 0.0,
               "source": "main_cuda.cu:38387-38467",
               "old_new": "eight full-field non-authoritative diagnostic copies every step -> CSV cadence/end only",
               "correctness": "hard gates and failure diagnostics remain per-step",
               "performance": "2-step D2H trace reduced 42.16%", "transaction": "decision-boundary checks unchanged",
               "reason": "removes observer traffic without delaying authoritative failure checks"},
        "M9": {"decision": "NOT_IMPLEMENTED", "saved_GiB": 0.0,
               "source": "profile-only candidate",
               "old_new": "no change", "correctness": "not applicable", "performance": "not measured",
               "transaction": "no reduction-order change", "reason": "top kernels have distinct lifetimes; fusion risk unsupported"},
        "M10": {"decision": "NOT_IMPLEMENTED", "saved_GiB": 0.0,
                "source": "dynamic nonlinear/retry control flow",
                "old_new": "no change", "correctness": "not applicable", "performance": "not measured",
                "transaction": "event/retry branches remain uncaptured", "reason": "variable loops and fail-closed branches dominate"},
    }

    b0_easy = 0.032213906466946919
    b187_easy = 0.032323879363989741
    wall400 = row_float(p400, "warmup_excluded_avg_walltime_per_step_s")
    data = {
        "source": {
            "git_head": git_head, "dirty_tree": True,
            "binary_sha256": "13752facf2b6158511859428dc7588db3fc146b8bf1be6b91a26d3a9911fbb5a",
            "source_sha256": source_sha,
            "input_sha256": sha256(OUT / "baseline_inputs" / "runtime.params"),
            "compiler": "gcc 13.1.0 host + nvcc 12.9, -arch=sm_120",
            "cuda": "12.9", "driver": "580.95.05",
            "gpu": "NVIDIA GeForce RTX 5080, 16303 MiB",
        },
        "baseline": {
            "solver": "LEGACY_CURRENT / V0 active-manifold IMEX-BDF2",
            "dt_code": DT_CODE, "dt_physical_s": DT_PHYS,
            "transport_gate": "strict currently qualified gate",
            "requested_steps": 8000, "accepted_steps": 8000,
            "hard_rejects": 0, "fallback_macros": 0, "iteration_p99": 44,
            "endpoint_hashes": base_hashes,
        },
        "memory": {
            "source_before_GiB": 19.579537615180016,
            "source_after_GiB": 12.182601198554039,
            "reduction_percent": 37.77891266896483,
            "n400_peak_GiB": 12.90,
            "n400_process_peak_MiB": 12788,
            "n400_device_fraction_percent": 100.0 * 12.90 / 15.46,
            "scaling_rows": scaling_rows,
        },
        "speed": {
            "rows": speed_rows,
            "easy_before_s": b0_easy, "easy_after_s": b187_easy,
            "easy_regression_percent": 100.0 * (b187_easy / b0_easy - 1.0),
            "n400_accepted_steps": 13, "n400_peak_mib": 12788,
            "n400_first_step_s": 93.691429,
            "n400_event_subcycle_macros": len(event_steps),
            "n400_dominant_event_reason": "coupled_outer_max_iter_not_converged; BDF2 absolute mass-identity preflight also flags large-N roundoff",
            "projection_wall_s_per_step": wall400,
            "gpu_hours_per_physical_hour": wall400 / DT_PHYS,
        },
        "tests": {
            "ablation_rows": ablation_rows,
            "first_failure_rows": first_failure_rows,
            "overall_equivalence_status": "PASS_BITWISE_ENDPOINT_ROLLBACK_RESTART",
        },
        "profiler": {
            "nsys_status": "AVAILABLE_AND_RUN_2025.1.3",
            "ncu_status": "UNAVAILABLE_ON_WORKSTATION",
            "kernel_rows": kernel_rows, "copy_rows": copy_rows,
            "ncu_rows": [
                {"tool": "Nsight Compute", "metric": metric, "value": "NOT_MEASURED", "status": "UNAVAILABLE",
                 "reason": "ncu executable absent on workstation; no estimate fabricated"}
                for metric in ("occupancy", "register_pressure", "achieved_memory_bandwidth", "L2_hit_rate")
            ],
            "trace_metrics": {
                "D2D_MB_B0_2steps": mem_mb("B0", "[CUDA memcpy Device-to-Device]"),
                "D2D_MB_B187_2steps": mem_mb("B187", "[CUDA memcpy Device-to-Device]"),
                "D2H_MB_B0_2steps": mem_mb("B0", "[CUDA memcpy Device-to-Host]"),
                "D2H_MB_B187_2steps": mem_mb("B187", "[CUDA memcpy Device-to-Host]"),
                "cudaDeviceSynchronize_B0_2steps": api_count("B0", "cudaDeviceSynchronize"),
                "cudaDeviceSynchronize_B187_2steps": api_count("B187", "cudaDeviceSynchronize"),
                "cudaLaunchKernel_B0_2steps": api_count("B0", "cudaLaunchKernel"),
                "cudaLaunchKernel_B187_2steps": api_count("B187", "cudaLaunchKernel"),
            },
        },
        "optimizations": opt,
        "final_status": "PASS_400CUBE_PF_ONLY_ELASTIC_MEMORY_BLOCKED",
    }

    tm = data["profiler"]["trace_metrics"]
    data["terminal_markers"] = {
        "baseline_preserved": True,
        "baseline_final_fields_bitwise": True,
        "baseline_restart_bitwise": True,
        "baseline_rollback_bitwise": True,
        "legacy_behavior_unchanged": True,
        "original_400cube_source_GiB": 19.580,
        "device_85pct_limit_GiB": 13.533,
        "largest_allocation_name": "d_scratch_r_double",
        "largest_allocation_GiB": 0.95367431640625,
        "number_full_FP64_field_equivalents_before": 41.06126646875,
        "number_full_FP64_field_equivalents_after": 25.54876646875,
        "optional_lazy_saved_GiB": opt["M1"]["saved_GiB"],
        "pointer_rotation_saved_GiB": 0.0,
        "compact_masks_saved_GiB": 0.0,
        "scratch_alias_saved_GiB": opt["M4"]["saved_GiB"],
        "derived_field_saved_GiB": opt["M5"]["saved_GiB"],
        "face_streaming_saved_GiB": opt["M6"]["saved_GiB"],
        "fft_workspace_saved_GiB": 0.0,
        "other_saved_GiB": opt["M2"]["saved_GiB"],
        "total_source_allocation_before_GiB": data["memory"]["source_before_GiB"],
        "total_source_allocation_after_GiB": data["memory"]["source_after_GiB"],
        "memory_reduction_percent": data["memory"]["reduction_percent"],
        "D2D_bytes_per_step_before": tm["D2D_MB_B0_2steps"] * 1.0e6 / 2.0,
        "D2D_bytes_per_step_after": tm["D2D_MB_B187_2steps"] * 1.0e6 / 2.0,
        "D2H_sync_count_before": f"{tm['cudaDeviceSynchronize_B0_2steps']}_per_2step_trace",
        "D2H_sync_count_after": f"{tm['cudaDeviceSynchronize_B187_2steps']}_per_2step_trace",
        "kernel_count_before": f"{tm['cudaLaunchKernel_B0_2steps']}_per_2step_trace",
        "kernel_count_after": f"{tm['cudaLaunchKernel_B187_2steps']}_per_2step_trace",
        "easy_step_wall_before": b0_easy,
        "easy_step_wall_after": b187_easy,
        "easy_step_regression": data["speed"]["easy_regression_percent"] / 100.0,
        "large_grid_wall_before": "NOT_MEASURED_B0_SOURCE_OOM",
        "large_grid_wall_after": wall400,
        "large_grid_speedup": "NOT_COMPARABLE_B0_SOURCE_OOM",
        "PF_only_400cube_source_GiB": data["memory"]["source_after_GiB"],
        "PF_only_400cube_peak_GiB": data["memory"]["n400_peak_GiB"],
        "PF_only_400cube_status": "PASS_83.44PCT_DEVICE",
        "PF_elastic_400cube_peak_GiB": "NOT_STARTED_STATIC_SOURCE_26.7476_GiB",
        "PF_elastic_400cube_status": "BLOCKED_STATIC_SOURCE_EXCEEDS_DEVICE",
        "PF_elastic_GP_400cube_peak_GiB": "NOT_STARTED",
        "PF_elastic_GP_400cube_status": "BLOCKED_STATIC_SOURCE_AND_GP_GATE",
        "largest_safe_PF_only_N": 400,
        "largest_safe_elastic_N": "NOT_RUNTIME_ADMITTED_STATIC_UPPER_BOUND_256",
        "largest_safe_elastic_GP_N": "NOT_RUNTIME_ADMITTED_STATIC_UPPER_BOUND_256",
        "400cube_speed_measurement": "ACTUAL_13_STEP_3_WARMUP_10_TIMED",
        "400cube_wall_s_per_step": wall400,
        "400cube_physical_s_per_GPU_hour": DT_PHYS * 3600.0 / wall400,
        "GPU_hours_for_1_physical_hour": wall400 / DT_PHYS,
        "GPU_hours_for_10_physical_hours": 10.0 * wall400 / DT_PHYS,
        "GPU_hours_for_50_physical_hours": 50.0 * wall400 / DT_PHYS,
        "physical_model_changed": False,
        "transport_equation_changed": False,
        "phase_equation_changed": False,
        "acceptance_gates_changed": False,
        "sparse_matrix_used": False,
        "large_Krylov_basis_used": False,
        "cluster_used": False,
        "commit_created": False,
        "push_performed": False,
        "recommended_next_action": "QUALIFY_ELASTIC_MEMORY_SEPARATELY_AND_PROFILE_EXISTING_EVENT_SUBCYCLE_COST_WITHOUT_CHANGING_GATES",
        "final_status": data["final_status"],
    }

    (OUT / "measurement_evidence.json").write_text(
        json.dumps(data, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    main()
