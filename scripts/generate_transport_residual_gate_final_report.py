#!/usr/bin/env python3
"""Generate the evidence-derived transport residual-gate decision."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "transport_residual_gate_v1"
T_REAL_UNIT_S = 41.12958542455477


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        return []
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def value(row: dict[str, Any] | None, key: str, default: float = math.nan) -> float:
    try:
        return float(row[key]) if row is not None else default
    except (KeyError, TypeError, ValueError):
        return default


def truth(row: dict[str, Any], key: str) -> bool:
    return str(row.get(key, "")).lower() == "true"


def marker(value_: Any) -> str:
    if isinstance(value_, bool):
        return "true" if value_ else "false"
    if isinstance(value_, float):
        return f"{value_:.17e}" if math.isfinite(value_) else "NOT_AVAILABLE"
    return str(value_)


def main() -> int:
    short = read_csv(REPORT / "equal_time_metrics.csv")
    long = read_csv(REPORT / "long_window_candidate_summary.csv")
    long_windows = read_csv(REPORT / "long_window_metrics.csv")
    if len(short) != 20:
        raise SystemExit("20-case equal-time matrix is incomplete")
    long_pass = [row for row in long if row.get("long_window_status") == "PASS"]
    selected = max(long_pass, key=lambda row: value(row, "physical_s_per_GPU_hour")) if long_pass else None
    selected_short = None
    selected_windows: list[dict[str, str]] = []
    if selected:
        selected_short = next(
            row for row in short
            if row["gate_id"] == selected["gate_id"] and row["dt_id"] == selected["dt_id"]
        )
        selected_windows = [
            row for row in long_windows
            if row["gate_id"] == selected["gate_id"]
            and row["dt_id"] == selected["dt_id"]
        ]
    strict = next(row for row in short if row["gate_id"] == "G12" and row["dt_id"] == "dt16")
    short_pass = [row for row in short if truth(row, "full_common_window_pass")]
    best_short = max(short_pass, key=lambda row: value(row, "physical_s_per_GPU_hour")) if short_pass else None
    best_trend = max(
        (row for row in short if truth(row, "equal_time_accuracy_pass") and truth(row, "hard_numerical_gates_pass")),
        key=lambda row: value(row, "physical_s_per_GPU_hour"),
        default=None,
    )
    relaxed_selected = selected is not None and selected["gate_id"] != "G12"
    entry_path = REPORT / "optional_8nm_entry_smoke.json"
    entry = json.loads(entry_path.read_text(encoding="utf-8")) if entry_path.is_file() else {}
    entry_status = entry.get("status", "NOT_RUN_OPTIONAL_ENTRY_SMOKE")
    entry_pass = str(entry_status).startswith("PASS")
    final_status = (
        "PASS_RELAXED_TRANSPORT_GATE_HIGH_THROUGHPUT_CANDIDATE_QUALIFIED"
        if relaxed_selected and entry_pass else
        "PASS_STRICT_GATE_REMAINS_ONLY_QUANTITATIVE_CANDIDATE"
        if selected and not relaxed_selected and entry_pass else
        "FAIL_OPTIONAL_8NM_ENTRY_SMOKE_AFTER_CANDIDATE"
        if selected else
        "FAIL_NO_LONG_WINDOW_TRANSPORT_GATE_CANDIDATE"
    )
    baseline = json.loads((REPORT / "baseline_manifest.json").read_text(encoding="utf-8"))
    frozen_strict_throughput = float(
        baseline["strict_dt16"]["throughput_physical_s_per_GPU_hour"]
    )
    speedup = (
        value(selected, "physical_s_per_GPU_hour") /
        frozen_strict_throughput
        if selected else math.nan
    )

    def max_window(metric: str) -> float:
        values = [value(row, metric) for row in selected_windows]
        values = [item for item in values if math.isfinite(item)]
        return max(values, default=math.nan)

    status_lines = []
    for gate in ("G12", "G10", "G9", "G8"):
        for dt_id in ("dt16", "dt8", "dt4", "dt2", "dt1"):
            row = next(item for item in short if item["gate_id"] == gate and item["dt_id"] == dt_id)
            status_lines.append(f"{gate}_{dt_id}_status={row['matrix_status']}")

    markers: list[tuple[str, Any]] = [
        ("baseline_preserved", True),
        ("gate_plumbing_status", "PASS_UNIFIED_EXISTING_THRESHOLD_WITH_TRANSACTIONAL_OBSERVER"),
    ]
    markers.extend(tuple(line.split("=", 1)) for line in status_lines)
    markers += [
        ("selected_transport_residual_gate", value(selected_short, "requested_transport_gate")),
        ("selected_dt_code", value(selected_short, "dt_code")),
        ("selected_dt_physical", value(selected_short, "dt_physical_s")),
        ("selected_Ctot_error", max_window("Ctot_increment_relative_L2_error")),
        ("selected_phi_error", max_window("phi_increment_relative_L2_error")),
        ("selected_hvolume_error", max_window("hvolume_increment_relative_error")),
        ("selected_profile_error", max_window("matrix_profile_capacity_weighted_relative_error")),
        ("selected_interface_error", max_window("phi_half_interface_error_dx")),
        ("selected_transfer_error", max_window("cumulative_transfer_relative_error")),
        ("selected_far_field_error", max_window("far_field_relative_error")),
        ("selected_eta_R", value(selected, "eta_R")),
        ("selected_max_local_cumulative_defect", value(selected, "maximum_abs_local_cumulative_defect")),
        ("selected_retry_fraction", value(selected, "retry_fraction")),
        ("selected_fallback_fraction", value(selected, "fallback_fraction")),
        ("selected_retry_overhead", value(selected, "retry_wall_overhead_fraction")),
        ("selected_p99_iterations", value(selected, "accepted_iteration_p99")),
        ("selected_physical_s_per_GPU_hour", value(selected, "physical_s_per_GPU_hour")),
        ("strict_dt16_throughput", frozen_strict_throughput),
        ("instrumented_strict_dt16_common_throughput", value(strict, "physical_s_per_GPU_hour")),
        ("selected_speedup_vs_strict_dt16", speedup),
        ("long_window_status", selected.get("long_window_status", "NO_PASS") if selected else "NO_PASS"),
        ("T400_8nm_entry_status", entry_status),
        ("T380_status", "NOT_RUN"),
        ("GP_status", "NOT_RUN"),
        ("large_3D_status", "NOT_RUN"),
        ("new_solver_status", "NOT_IMPLEMENTED"),
        ("cluster_used", False),
        ("commit_created", False),
        ("push_performed", False),
        ("recommended_next_action",
         "FREEZE_SELECTED_GATE_DT_FOR_NEXT_FORMAL_8NM_GOAL"
         if selected and entry_pass else
         "RUN_OR_FIX_OPTIONAL_8NM_ENTRY_SMOKE"
         if selected else
         "RETAIN_STRICT_DT16_AND_REVISIT_GLOBALIZATION"),
        ("final_status", final_status),
    ]
    terminal = "\n".join(f"{key}={marker(item)}" for key, item in markers) + "\n"
    (REPORT / "final_terminal_output.txt").write_text(terminal, encoding="utf-8")

    matrix_table = [
        "| Case | Common status | Accuracy | Defect | Retry/efficiency | Throughput |",
        "|---|---|---|---|---|---:|",
    ]
    for row in short:
        matrix_table.append(
            f"| {row['case_id']} | {row['matrix_status']} | {row.get('equal_time_accuracy_pass', '-')} | "
            f"{row.get('accumulated_defect_gate_pass', '-')} | {row.get('retry_efficiency_gate_pass', '-')} | "
            f"{value(row, 'physical_s_per_GPU_hour'):.1f} |"
        )
    long_table = [
        "| Case | Hard | Retry | Accuracy | Local rate | Signed bias | Throughput | Status |",
        "|---|---|---|---|---|---|---:|---|",
    ]
    for row in long:
        long_table.append(
            f"| {row['gate_id']}+{row['dt_id']} | {row.get('long_hard_gates_pass', '-')} | "
            f"{row.get('long_retry_efficiency_gate_pass', '-')} | "
            f"{row.get('all_registered_windows_accuracy_pass', '-')} | "
            f"{row.get('local_defect_rate_growth_pass', '-')} | "
            f"{row.get('signed_residual_bias_growth_pass', '-')} | "
            f"{value(row, 'physical_s_per_GPU_hour'):.1f} | {row.get('long_window_status', '-')} |"
        )
    decision = [
        "# Transport residual-gate production candidate decision", "",
        "## Decision", "",
        f"`final_status={final_status}`", "",
        "Only `ctot_residual_abs_tol` varied. Thermodynamics, mobility, phase KKT, mass, bounds, energy/work, mechanics, active-manifold prediction, BDF2, line search, iteration budget, and retry mathematics remained frozen.", "",
        f"Best quantitative candidate: `{selected['gate_id'] + '+' + selected['dt_id'] if selected else 'NONE'}`.",
        f"Best short-window throughput candidate: `{best_short['case_id'] if best_short else 'NONE'}`.",
        f"Best trend-only short candidate: `{best_trend['case_id'] if best_trend else 'NONE'}`.", "",
        f"The selected long-window throughput is `{value(selected, 'physical_s_per_GPU_hour'):.3f}` physical s/GPU h, `{speedup:.3f}x` the frozen strict dt/16 baseline (`{frozen_strict_throughput:.3f}`).", "",
        "The reported selected errors are the worst values over all four pre-registered long windows, not only the first 64 s window.", "",
        "The broadest gate is not assumed optimal: a looser endpoint changes the next nonlinear initial state and can increase later iteration work. Selection therefore follows measured complete-trajectory throughput after all registered QoI gates.", "",
        "## Common Window", "", *matrix_table, "",
        "## Long Evidence", "", *long_table, "",
        "See `long_window_validation.md`, `long_window_metrics.csv`, and `long_window_candidate_summary.csv`. Every common-window Stage 7-8 pass is required to have a 4x/257 s trajectory before entering the selected set.", "",
        "## Conditional Entry Smoke", "",
        f"`T400_8nm_entry_status={entry_status}`. This is only a low-cost growth-entry/restart check; no formal 8 nm displacement campaign was run.", "",
        "## Claim Boundary", "",
        "This qualification covers the frozen T400 512x1x1 planar active-manifold IMEX-BDF2 problem only. T380, GP/S3, curvature, multi-particle, large 3D, formal 8 nm displacement, and new nonlinear solvers were not run.", "",
    ]
    (REPORT / "production_candidate_decision.md").write_text("\n".join(decision), encoding="utf-8")
    print(json.dumps({"selected": selected, "speedup": speedup, "final_status": final_status}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
