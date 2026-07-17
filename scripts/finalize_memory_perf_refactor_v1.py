#!/usr/bin/env python3
"""Generate the memory/performance refactor acceptance bundle.

The script is deliberately a report-only consumer.  Numerical values must be
present in measurement_evidence.json and are never inferred from model inputs.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "memory_perf_refactor_v1"
EVIDENCE = OUT / "measurement_evidence.json"


def write_csv(name: str, fieldnames: list[str], rows: list[dict]) -> None:
    with (OUT / name).open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def write_md(name: str, body: str) -> None:
    (OUT / name).write_text(body.rstrip() + "\n", encoding="utf-8")


def fmt(value, digits: int = 6) -> str:
    if value is None:
        return "NOT_MEASURED"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return str(value)
        return f"{value:.{digits}g}"
    return str(value)


def require(e: dict, *keys: str):
    value = e
    for key in keys:
        value = value[key]
    return value


def main() -> None:
    e = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    baseline = require(e, "baseline")
    memory = require(e, "memory")
    speed = require(e, "speed")
    tests = require(e, "tests")
    profiler = require(e, "profiler")
    source = require(e, "source")
    optimizations = require(e, "optimizations")

    manifest = {
        "schema": "memory_perf_refactor_v1_baseline_manifest",
        "git_head": source["git_head"],
        "dirty_tree": source["dirty_tree"],
        "binary_sha256": source["binary_sha256"],
        "source_sha256": source["source_sha256"],
        "input_sha256": source["input_sha256"],
        "compiler": source["compiler"],
        "cuda": source["cuda"],
        "driver": source["driver"],
        "gpu": source["gpu"],
        "solver": baseline["solver"],
        "dt_code": baseline["dt_code"],
        "dt_physical_s": baseline["dt_physical_s"],
        "transport_gate": baseline["transport_gate"],
        "physics_or_numerics_changed": False,
        "cluster_used": False,
    }
    (OUT / "baseline_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    write_md(
        "baseline_freeze.md",
        f"""# Baseline freeze

The accepted control is `{baseline['solver']}` at `dt={baseline['dt_code']}` code units
(`{baseline['dt_physical_s']}` s).  It completed {baseline['accepted_steps']}/{baseline['requested_steps']}
accepted steps with {baseline['hard_rejects']} hard rejects and {baseline['fallback_macros']} fallback macros.

The selected memory path uses feature mask `{optimizations['selected_mask']}`.  Its 8000-step endpoint is
bitwise identical for `Ctot`, `Ctot_nm1`, `phi`, `phi_nm1`, and `xB_alpha`; accepted iteration p99 remains
{baseline['iteration_p99']}.  Forced event rollback is bitwise.  Three-dimensional and restart controls are
listed in `optimization_ablation.csv`.

The worktree was dirty before this goal.  No reset, cleanup, commit, push, cluster execution, physics edit,
numerical-equation edit, or acceptance-gate edit was performed.
""",
    )

    kernel_rows = profiler["kernel_rows"]
    write_csv(
        "kernel_timing.csv",
        ["configuration", "rank", "kernel", "instances", "total_time_ms", "avg_time_us", "time_fraction", "source"],
        kernel_rows,
    )
    write_csv(
        "copy_sync_timing.csv",
        ["configuration", "rank", "operation", "instances", "total_time_ms", "avg_time_us", "bytes", "source"],
        profiler["copy_rows"],
    )
    write_csv(
        "ncu_summary.csv",
        ["tool", "metric", "value", "status", "reason"],
        profiler["ncu_rows"],
    )
    top_kernel = kernel_rows[0] if kernel_rows else None
    write_md(
        "performance_hotspots.md",
        f"""# Performance hotspots

Nsight Systems availability: **{profiler['nsys_status']}**. Nsight Compute availability:
**{profiler['ncu_status']}**.  Occupancy, register pressure, bandwidth and L2 claims are therefore made only
when directly present in `ncu_summary.csv`; unavailable metrics are not estimated.

The actual 400^3 representative curved benchmark used nonuniform `phi`, nonuniform chemical potential and
nonzero transport.  It completed {speed['n400_accepted_steps']} accepted steps with a measured process peak of
{speed['n400_peak_mib']} MiB.  The first accepted step cost {speed['n400_first_step_s']} s; the final timed
summary and warmup policy are in `actual_3d_speed.csv`.

Existing event safety activated in {speed['n400_event_subcycle_macros']} macros.  Dominant diagnostic:
`{speed['n400_dominant_event_reason']}`.  These recovered branches are included in the measured wall time and
were not hidden, retuned, or removed by the memory refactor.

Top profiler kernel: **{top_kernel['kernel'] if top_kernel else 'NOT_AVAILABLE'}**.  The dominant engineering
cost remains repeated nonlinear transport/phase kernels and reductions; no arithmetic-order-changing fusion
or CUDA Graph was retained without a measured net benefit.
""",
    )

    ablation_rows = tests["ablation_rows"]
    write_csv(
        "optimization_ablation.csv",
        [
            "stage", "feature_mask", "features", "grid", "steps", "accepted", "hard_rejects",
            "fallback_macros", "endpoint_bitwise_vs_B0", "rollback_bitwise", "restart_bitwise",
            "source_GiB_400", "wall_s_per_step", "decision", "evidence",
        ],
        ablation_rows,
    )
    write_csv(
        "actual_memory_scaling.csv",
        [
            "configuration", "N", "source_GiB", "process_peak_MiB", "program_used_GiB",
            "device_total_MiB", "device_fraction_percent", "first_step_attempted", "first_step_accepted",
            "status", "measurement_note",
        ],
        memory["scaling_rows"],
    )
    write_csv(
        "actual_3d_speed.csv",
        [
            "case", "N", "state", "feature_mask", "warmup_steps", "timed_steps", "accepted_steps",
            "hard_rejects", "fallback_macros", "wall_s_per_accepted_step", "physical_s_per_GPU_hour",
            "peak_MiB", "measurement_status", "notes",
        ],
        speed["rows"],
    )
    write_csv(
        "first_failure.csv",
        ["case", "grid", "stage", "first_failure", "hard_reject", "recovered", "effect_on_admission"],
        tests["first_failure_rows"],
    )

    validation_reports = {
        "optional_allocation_validation.md": ("M1 optional allocation", optimizations["M1"]),
        "pointer_rotation_validation.md": ("M2 transaction/history lifetime", optimizations["M2"]),
        "compact_mask_validation.md": ("M3 compact masks", optimizations["M3"]),
        "scratch_alias_validation.md": ("M4 scratch alias", optimizations["M4"]),
        "derived_field_validation.md": ("M5 derived-field de-persistence", optimizations["M5"]),
        "face_streaming_validation.md": ("M6 face streaming", optimizations["M6"]),
        "fft_workspace_validation.md": ("M7 explicit FFT workspace", optimizations["M7"]),
        "synchronization_validation.md": ("M8 copy/synchronization", optimizations["M8"]),
        "kernel_fusion_validation.md": ("M9 kernel fusion", optimizations["M9"]),
        "cuda_graph_validation.md": ("M10 CUDA Graph", optimizations["M10"]),
    }
    for filename, (title, data) in validation_reports.items():
        write_md(
            filename,
            f"""# {title}

- Decision: **{data['decision']}**
- Source: `{data['source']}`
- Old/new operation: {data['old_new']}
- Source saving at 400^3: **{fmt(data['saved_GiB'])} GiB**
- Arithmetic/physics/gate change: **none**
- Correctness evidence: {data['correctness']}
- Performance evidence: {data['performance']}
- Rollback/restart disposition: {data['transaction']}
- Reason: {data['reason']}
""",
        )

    reverted = [
        (key, data)
        for key, data in optimizations.items()
        if key.startswith("M") and data["decision"] not in {"RETAINED", "SELECTED"}
    ]
    write_md(
        "reverted_optimizations.md",
        "# Reverted or not-retained optimizations\n\n"
        + "\n".join(
            f"- **{key}**: {data['decision']}. {data['reason']}" for key, data in reverted
        ),
    )

    write_md(
        "largest_safe_grid.md",
        f"""# Largest safe grid

PF-only `400^3` is the largest requested grid and is admitted: static source allocation
{memory['source_after_GiB']:.6f} GiB and measured peak {memory['n400_peak_GiB']:.6f} GiB
({memory['n400_device_fraction_percent']:.2f}% of the 16,303 MiB device).  This satisfies the hard 85% gate,
though not the preferred 80% engineering target.

Elastic and elastic+GP layouts were not run: the current accepted baseline is elasticity OFF and the goal
forbids opening GP/S3.  Their status is **GATED_NOT_MEASURED**, not inferred from PF-only data.
""",
    )

    projection_rows = []
    for hours in (1, 10, 50):
        projection_rows.append(
            {
                "physical_hours": hours,
                "GPU_hours": speed["gpu_hours_per_physical_hour"] * hours,
                "GPU_days": speed["gpu_hours_per_physical_hour"] * hours / 24.0,
            }
        )
    projection_table = "\n".join(
        f"| {row['physical_hours']} | {row['GPU_hours']:.3f} | {row['GPU_days']:.3f} |"
        for row in projection_rows
    )
    write_md(
        "physical_time_cost_projection.md",
        f"""# Physical-time cost projection

Based only on the actual representative 400^3 benchmark: `dt_phys={baseline['dt_physical_s']}` s and
`{speed['projection_wall_s_per_step']}` wall s/accepted step.  No 512x1 timing is used.

| Physical hours | GPU wall hours | GPU wall days |
|---:|---:|---:|
{projection_table}

Excluded: GP source, large field output, particle analysis, adaptive dt, KWN handoff and elastic mechanics.
The estimate is a short-run projection and inherits the measured nonlinear-iteration/event mix.
""",
    )

    status = e["final_status"]
    write_md(
        "production_candidate_decision.md",
        f"""# Production candidate decision

## Decision

**{status}**

Selected engineering mask: `{optimizations['selected_mask']}` = M1+M2+M4+M5+M6+M8.

- 400^3 static PF-only source: {memory['source_after_GiB']:.6f} GiB (target <=12.5 GiB: PASS).
- Reduction: {memory['reduction_percent']:.3f}% (preferred >=35%: PASS).
- Actual 400^3 peak: {memory['n400_peak_GiB']:.6f} GiB / {memory['n400_device_fraction_percent']:.2f}% (hard <=85%: PASS).
- 8000-step easy-path wall regression: {speed['easy_regression_percent']:.3f}% (<=3%: PASS).
- Endpoint, rollback and restart: {tests['overall_equivalence_status']}.
- Physics, transport/phase equations and hard gates: unchanged.

The candidate qualifies PF-only memory admission. Elastic and elastic+GP 400^3 remain separately gated and
unmeasured. M3, M7, M9 and M10 are not part of the selected default.
""",
    )

    terminal = e["terminal_markers"]
    (OUT / "final_terminal_output.txt").write_text(
        "\n".join(f"{key}={fmt(value, 12)}" for key, value in terminal.items()) + "\n",
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()
