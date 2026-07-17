#!/usr/bin/env python3
"""Assemble the low-memory transport qualification evidence without inventing runs."""

from __future__ import annotations

import array
import csv
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports" / "low_memory_transport_v1"
RUNS = REPORT / "workstation_runs"
RAW = REPORT / "workstation_remote_results"

ACCEPT_RE = re.compile(
    r"CTOT_MIMETIC_BE_ACCEPT step=(\d+) nonlinear_iters=(\d+) "
    r"res_inf=([0-9.eE+-]+).*?mass_error=([0-9.eE+-]+).*?lambda=([0-9.eE+-]+)"
)
LOW_RE = re.compile(
    r"CTOT_LOW_MEMORY_TRANSPORT_SUMMARY step=(\d+).*?converged=(\d+).*?"
    r"nonlinear_iters=(\d+) line_search_trials=(\d+).*?"
    r"plateau_triggers=(\d+).*?fraction_to_boundary_calls=(\d+).*?"
    r"minimum_fraction_to_boundary=([0-9.eE+-]+).*?final_residual=([0-9.eE+-]+)"
)
RETRY_RE = re.compile(
    r"CTOT_BOUNDED_RETRY_SUMMARY .*?macro_steps=(\d+) .*?"
    r"macro_hard_rejects=(\d+) .*?internal_trial_rejects=(\d+) .*?"
    r"fallback_macros=(\d+).*?accepted_iteration_p99=([0-9.eE+-]+)"
)
REJECT_RE = re.compile(
    r"\[reject\] (CTOT_[A-Z0-9_]+) step=(\d+).*?"
    r"first_failure=([^ ]+).*?res_inf=([0-9.eE+-]+)"
)


def percentile(values: list[float], probability: float) -> float | None:
    if not values:
        return None
    ordered = sorted(values)
    return ordered[max(0, math.ceil(probability * len(ordered)) - 1)]


def fmt(value: object) -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        return f"{value:.17e}"
    return str(value)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fields})


def parse_case(case_dir: Path) -> dict[str, object]:
    status_path = case_dir / "workstation_status.json"
    status = json.loads(status_path.read_text()) if status_path.exists() else {}
    log_path = case_dir / "run.log"
    text = log_path.read_text(encoding="utf-8", errors="replace") if log_path.exists() else ""
    accepts = [
        (int(m.group(1)), int(m.group(2)), float(m.group(3)),
         float(m.group(4)), float(m.group(5)))
        for m in ACCEPT_RE.finditer(text)
    ]
    low = [
        (int(m.group(1)), int(m.group(2)), int(m.group(3)), int(m.group(4)),
         int(m.group(5)), int(m.group(6)), float(m.group(7)), float(m.group(8)))
        for m in LOW_RE.finditer(text)
    ]
    retries = list(RETRY_RE.finditer(text))
    retry = retries[-1].groups() if retries else (0, 0, 0, 0, 0.0)
    rejects = list(REJECT_RE.finditer(text))
    case_id = status.get("case_id", case_dir.name)
    version, dt_id, requested = re.match(r"(V\d+)_(dt\d+)_(\d+)steps", case_id).groups()
    return {
        "case_id": case_id,
        "version": version,
        "dt_id": dt_id,
        "feature_mask": status.get("features"),
        "requested_steps": int(requested),
        "completed_steps": len(accepts),
        "returncode": status.get("returncode"),
        "wall_seconds": status.get("remote_wall_seconds"),
        "macro_hard_rejects": int(retry[1]),
        "internal_trial_rejects": int(retry[2]),
        "fallback_macros": int(retry[3]),
        "homotopy_triggers": text.count("CTOT_INTERNAL_ALGEBRAIC_HOMOTOPY_TRIGGER"),
        "homotopy_level_failures": len(re.findall(
            r"CTOT_INTERNAL_ALGEBRAIC_HOMOTOPY_LEVEL .*?status=FAIL", text
        )),
        "plateau_triggers": sum(item[4] for item in low),
        "fraction_to_boundary_calls": sum(item[5] for item in low),
        "nonlinear_p50": percentile([float(item[1]) for item in accepts], 0.50),
        "nonlinear_p95": percentile([float(item[1]) for item in accepts], 0.95),
        "nonlinear_p99": percentile([float(item[1]) for item in accepts], 0.99),
        "nonlinear_max": max((item[1] for item in accepts), default=None),
        "line_search_p99": percentile([float(item[3]) for item in low], 0.99),
        "min_lambda": min((item[4] for item in accepts), default=None),
        "max_accepted_residual": max((item[2] for item in accepts), default=None),
        "max_abs_mass_error": max((abs(item[3]) for item in accepts), default=None),
        "first_reject_type": rejects[0].group(1) if rejects else None,
        "first_reject_step": int(rejects[0].group(2)) if rejects else None,
        "first_failure": rejects[0].group(3) if rejects else None,
        "first_failure_residual": float(rejects[0].group(4)) if rejects else None,
        "rollback_bitwise_pass": "CTOT_BOUNDED_RETRY_ROLLBACK" not in text or
            not bool(re.search(r"CTOT_BOUNDED_RETRY_ROLLBACK .*?bitwise=0", text)),
        "selector_seen": bool(status.get("runtime_selector_seen")),
        "production_gate_qualified": False,
        "status": (
            "COMPLETE" if len(accepts) == int(requested) and status.get("returncode") == 0
            else ("COMPUTE_COMPLETE_RUNTIME_FAILED"
                  if len(accepts) == int(requested) else "FAILED")
        ),
        "teardown_or_runtime_signal": "SIGSEGV_RC139" if status.get("returncode") == 139 else "none",
    }


def raw_field(case: str, field: str) -> tuple[Path, list[float]] | None:
    case_root = RAW / case
    if not case_root.exists():
        return None
    paths = sorted(case_root.rglob(f"*_{field}.raw"))
    if not paths:
        return None
    values = array.array("d")
    values.frombytes(paths[-1].read_bytes())
    return paths[-1], list(values)


def field_delta(case_a: str, case_b: str, field: str) -> tuple[float, float] | None:
    a = raw_field(case_a, field)
    b = raw_field(case_b, field)
    if not a or not b or len(a[1]) != len(b[1]):
        return None
    delta = [x - y for x, y in zip(a[1], b[1])]
    return max(map(abs, delta)), math.sqrt(sum(x * x for x in delta) / len(delta))


def source_hash(path: str) -> str:
    data = (ROOT / path).read_bytes()
    return hashlib.sha256(data).hexdigest()


def main() -> int:
    REPORT.mkdir(parents=True, exist_ok=True)
    cases = [parse_case(path) for path in sorted(RUNS.iterdir())
             if path.is_dir() and (path / "workstation_status.json").exists()]
    fields = [
        "case_id", "version", "feature_mask", "dt_id", "requested_steps",
        "completed_steps", "returncode", "status", "wall_seconds",
        "macro_hard_rejects", "internal_trial_rejects", "fallback_macros",
        "homotopy_triggers", "homotopy_level_failures", "plateau_triggers",
        "fraction_to_boundary_calls", "nonlinear_p50", "nonlinear_p95",
        "nonlinear_p99", "nonlinear_max", "line_search_p99", "min_lambda",
        "max_accepted_residual", "max_abs_mass_error", "first_reject_step",
        "first_failure", "first_failure_residual", "rollback_bitwise_pass",
        "teardown_or_runtime_signal", "production_gate_qualified",
    ]
    write_csv(REPORT / "ablation_matrix.csv", cases, fields)

    failures = [row for row in cases if row["status"] == "FAILED" or
                row["internal_trial_rejects"] or row["fallback_macros"] or
                row["teardown_or_runtime_signal"] != "none"]
    write_csv(REPORT / "first_failure.csv", failures, [
        "case_id", "status", "completed_steps", "requested_steps", "returncode",
        "first_reject_type", "first_reject_step", "first_failure",
        "first_failure_residual", "internal_trial_rejects", "fallback_macros",
        "homotopy_level_failures", "rollback_bitwise_pass",
        "teardown_or_runtime_signal",
    ])

    comparisons: list[dict[str, object]] = []
    for left, right in [
        ("V0_dt16_20steps", "V1_dt16_20steps"),
        ("V0_dt16_20steps", "V2_dt16_20steps"),
        ("V0_dt16_20steps", "V3_dt16_20steps"),
        ("V0_dt16_20steps", "V4_dt16_20steps"),
        ("V0_dt16_8000steps", "V5_dt16_8000steps"),
        ("V0_dt4_2000steps", "V4_dt4_2000steps"),
        ("V0_dt4_2000steps", "V5_dt4_2000steps"),
    ]:
        row: dict[str, object] = {"reference": left, "candidate": right}
        for field in ("Ctot", "phi"):
            delta = field_delta(left, right, field)
            row[f"{field}_Linf"] = delta[0] if delta else None
            row[f"{field}_L2"] = delta[1] if delta else None
        row["bitwise"] = row["Ctot_Linf"] == 0.0 and row["phi_Linf"] == 0.0
        comparisons.append(row)
    write_csv(REPORT / "equal_time_field_comparison.csv", comparisons, [
        "reference", "candidate", "Ctot_Linf", "Ctot_L2", "phi_Linf",
        "phi_L2", "bitwise",
    ])

    by_id = {row["case_id"]: row for row in cases}
    baseline = by_id.get("V0_dt16_8000steps", {})
    candidate = by_id.get("V5_dt16_8000steps", {})
    baseline_wall = baseline.get("wall_seconds")
    candidate_wall = candidate.get("wall_seconds")
    easy_regression = ((candidate_wall / baseline_wall) - 1.0) if baseline_wall and candidate_wall else None
    perf_rows = []
    for row in cases:
        dt_code = {"dt16": 1.953125e-4, "dt8": 3.90625e-4,
                   "dt4": 7.8125e-4, "dt2": 1.5625e-3}.get(row["dt_id"])
        physical_per_code = 0.00803312 / 1.953125e-4
        accepted_phys = row["completed_steps"] * dt_code * physical_per_code
        wall = row.get("wall_seconds") or 0.0
        perf_rows.append({
            **row,
            "accepted_physical_s": accepted_phys,
            "physical_s_per_GPU_hour": accepted_phys / wall * 3600.0 if wall else None,
        })
    write_csv(REPORT / "production_performance.csv", perf_rows, fields + [
        "accepted_physical_s", "physical_s_per_GPU_hour",
    ])

    (REPORT / "equal_time_accuracy.md").write_text(f"""# Equal-time accuracy

The formal V2 residual-gate qualification ended in `FAIL_V2_MATERIAL_TRANSPORT_DEFECT`.
Therefore no relaxed gate is available for quantitative production acceptance and
G10 results remain solver-development diagnostics only.

The 20-step dt/16 V1--V4 checkpoints are bitwise identical to V0 for both Ctot
and phi. Long-window V5 is not equivalent because it invokes fallback: at dt/16
the endpoint differences relative to V0 are Ctot Linf
`{fmt(next((r['Ctot_Linf'] for r in comparisons if r['candidate']=='V5_dt16_8000steps'), None))}`
and phi Linf
`{fmt(next((r['phi_Linf'] for r in comparisons if r['candidate']=='V5_dt16_8000steps'), None))}`.
These differences are reported, not accepted as qualified error.
""", encoding="utf-8")

    (REPORT / "long_window_validation.md").write_text(f"""# Long-window validation

| Run | Completed | Internal rejects | Fallbacks | Result |
|---|---:|---:|---:|---|
| V0 dt/16 | {baseline.get('completed_steps','NA')}/8000 | {baseline.get('internal_trial_rejects','NA')} | {baseline.get('fallback_macros','NA')} | baseline zero-fallback pass |
| V5 dt/16 | {candidate.get('completed_steps','NA')}/8000 | {candidate.get('internal_trial_rejects','NA')} | {candidate.get('fallback_macros','NA')} | production fail |
| V5 dt/8 | {by_id.get('V5_dt8_4000steps',{}).get('completed_steps','NA')}/4000 | {by_id.get('V5_dt8_4000steps',{}).get('internal_trial_rejects','NA')} | {by_id.get('V5_dt8_4000steps',{}).get('fallback_macros','NA')} | exhausted at step 3465 |
| V5 dt/2 | {by_id.get('V5_dt2_1000steps',{}).get('completed_steps','NA')}/1000 | {by_id.get('V5_dt2_1000steps',{}).get('internal_trial_rejects','NA')} | {by_id.get('V5_dt2_1000steps',{}).get('fallback_macros','NA')} | exhausted at step 831 |

V5 recovers many individual difficult solves, but it does not satisfy the
pre-registered zero internal retry and zero fallback contract. The old retry
safety net remains bitwise restorative where exercised.
""", encoding="utf-8")

    (REPORT / "frozen_reject_replay.md").write_text("""# Frozen reject replay

The catalog contains 31 historical reject records, but the repository evidence
does not contain the exact predecessor `Ctot`, `phi`, BDF2 history, active set,
and mechanics arrays for all 31 records. Exact state replay is therefore not
possible and no convergence count is fabricated.

The available replacement evidence is trajectory recurrence from the common
frozen initial state. It reproduces the same globalization failure class:
`transport_line_search_stagnation`, repeated internal fallback at coarse dt,
and bitwise rollback when a trial is rejected. This evidence is sufficient to
reject V1--V5 as production candidates, but it is not labelled an exact 31-state
replay. Status: `INCOMPLETE_EXACT_STATE_PAYLOADS_NOT_RECORDED`.
""", encoding="utf-8")

    (REPORT / "production_performance.md").write_text(f"""# Production performance

The only complete zero-retry long-window production baseline is V0 dt/16:
`{fmt(baseline_wall)}` wall seconds for 8000 steps. V5 dt/16 took
`{fmt(candidate_wall)}` seconds, an apparent wall change of
`{fmt(easy_regression)}`, but required {candidate.get('fallback_macros','NA')}
fallback macros. It therefore fails the production throughput gate regardless
of raw wall time. V5 dt/4's short-window speedup is likewise diagnostic only.

No solver version simultaneously achieved a qualified residual gate, zero
fallback, long-window completion at a larger dt, and the required accuracy
holdouts. `selected_solver_name=NONE`.
""", encoding="utf-8")

    memory_rows = [
        {"optimization": "M1_POINTER_ROTATION", "implemented": False,
         "saved_MiB_400cube": 0.0, "status": "NOT_ENTERED_NO_SOLVER_CANDIDATE",
         "reason": "transaction-authoritative pointer rotation requires separate rollback qualification"},
        {"optimization": "M2_UINT8_ACTIVE_MASK", "implemented": False,
         "saved_MiB_400cube": 64_000_000 * 7 / 2**20,
         "status": "ANALYZED_NOT_RETAINED",
         "reason": "insufficient alone and active-code ABI touches all transport/PDAS kernels"},
        {"optimization": "M3_SCRATCH_LIFETIME_REUSE", "implemented": False,
         "saved_MiB_400cube": 0.0, "status": "NOT_ENTERED_NO_SOLVER_CANDIDATE",
         "reason": "requires stream last-use and rollback proof"},
        {"optimization": "M4_FACE_BUFFER_STREAMING", "implemented": False,
         "saved_MiB_400cube": 2 * 64_000_000 * 8 / 2**20,
         "status": "ANALYZED_NOT_RETAINED",
         "reason": "existing outer shared-face-change audit keeps prior directional faces live"},
    ]
    write_csv(REPORT / "memory_optimization_ablation.csv", memory_rows, [
        "optimization", "implemented", "saved_MiB_400cube", "status", "reason",
    ])

    manifest = {
        "status": "PREPARED_NOT_EXECUTED",
        "blocking_gates": [
            "FAIL_V2_MATERIAL_TRANSPORT_DEFECT",
            "NO_ZERO_FALLBACK_LOW_MEMORY_SOLVER_CANDIDATE",
            "400CUBE_SOURCE_ALLOCATION_EXCEEDS_85_PERCENT_VRAM",
        ],
        "case_P": {"name": "PLANAR_TILED_3D", "state": "not_constructed"},
        "case_M": {"name": "MULTIPARTICLE_3D", "state": "not_constructed"},
        "cluster_used": False,
        "fabricated_measurements": False,
    }
    (REPORT / "3d_benchmark_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_csv(REPORT / "3d_speed_scaling.csv", [{
        "status": "NOT_RUN_BLOCKED_BY_PRECONDITIONS",
        "reason": ";".join(manifest["blocking_gates"]),
    }], ["status", "reason"])

    (REPORT / "400cube_speed_benchmark.md").write_text("""# 400-cube speed benchmark

No 400-cube speed run was attempted. The source-exact PF-only allocation ledger
is 19.580 GiB at 400^3, already above the RTX 5080 85% admission ceiling of
13.533 GiB before CUDA context and cuFFT workspace. In addition, no low-memory
solver passed the residual/fallback prerequisites. Consequently the result is
`400CUBE_SPEED_NOT_MEASURED_MEMORY_AND_SOLVER_BLOCKED`, not an extrapolated or
actual throughput claim.
""", encoding="utf-8")

    (REPORT / "physical_time_cost_projection.md").write_text("""# Physical-time cost projection

A 400^3 physical-time cost projection is intentionally not reported. The goal
requires measured safe-size 3D points before extrapolation; those benchmarks
were gated off by the failed solver qualification and source-only memory
admission. A 512x1x1 timing cannot be scaled into a credible 400^3 estimate.
""", encoding="utf-8")

    decision = f"""# Production candidate decision

## Decision

No new production candidate is selected.

1. V2 has no formally qualified relaxed transport gate
   (`FAIL_V2_MATERIAL_TRANSPORT_DEFECT`). G10 is diagnostic only.
2. V5 improves a 2000-step dt/4 diagnostic but still requires fallback; in the
   8000-step dt/16 holdout it records {candidate.get('internal_trial_rejects','NA')}
   internal rejects and {candidate.get('fallback_macros','NA')} fallback macros.
3. V5 dt/8 stops at step {by_id.get('V5_dt8_4000steps',{}).get('completed_steps','NA') + 1 if by_id.get('V5_dt8_4000steps') else 'NA'};
   V5 dt/2 stops at step {by_id.get('V5_dt2_1000steps',{}).get('completed_steps','NA') + 1 if by_id.get('V5_dt2_1000steps') else 'NA'}.
4. The 400^3 source allocation is over the workstation admission limit before
   FFT/context overhead.

`LEGACY_CURRENT` remains the default and the accepted dt/16 baseline. The new
mode remains default-off research code. The correct next action is to resolve
the formal residual-gate defect and the active-bound globalization root cause
before memory refactoring or 3D campaign work.
"""
    (REPORT / "production_candidate_decision.md").write_text(decision, encoding="utf-8")

    # Update feature reports with runtime evidence rather than only unit tests.
    (REPORT / "homotopy_validation.md").write_text(f"""# Internal homotopy validation

The fixed theta sequence 0.25, 0.5, 1.0 uses the same macro anchor/history and
does not advance time or commit history at intermediate levels. Unit contract
tests pass. Runtime qualification fails: V5 dt/4 triggered homotopy 9 times and
two levels failed, leaving two fallback macros; V5 dt/16 accumulated
{candidate.get('homotopy_triggers','NA')} triggers and
{candidate.get('homotopy_level_failures','NA')} failed levels. V5 dt/8 and dt/2
eventually exhausted their safety path. Status: `IMPLEMENTED_DEFAULT_OFF_NOT_PRODUCTION_QUALIFIED`.
""", encoding="utf-8")

    (REPORT / "plateau_detector_validation.md").write_text("""# Plateau detector validation

The nine-value window implements the frozen `R_k/R_(k-8) >= 0.98` rule and four
consecutive sub-one-percent improvements. Host tests pass. Runtime tests show
that the detector exits V1 at step 1325 instead of burning the full nonlinear
budget, while never accepting a residual above the unchanged gate. With later
strategies enabled it escalates rather than accepting. Status:
`IMPLEMENTED_DEFAULT_OFF_FAIL_CLOSED_VALIDATED`.
""", encoding="utf-8")

    (REPORT / "fraction_to_boundary_validation.md").write_text("""# Exact fraction-to-boundary validation

The GPU kernel evaluates the final mass-tangent direction after mean removal
and uses the fixed 0.995 safety factor. Host manufactured lower/upper/mixed
states and the workstation GPU parity binary pass. Runtime V2--V5 records finite
positive feasible lambdas and no clipping or physical mass projection. The
feature does not by itself eliminate the coarse-dt plateau. Status:
`IMPLEMENTED_GPU_HOST_PARITY_PASS_NOT_PRODUCTION_QUALIFIED`.
""", encoding="utf-8")

    (REPORT / "hybrid_merit_validation.md").write_text("""# Hybrid L2/Linf merit validation

The far/near/very-near gate regions preserve the frozen merit contract, and
unit tests pass. Runtime V3 delays the dt/4 first terminal failure from step
1325 to step 1380 but does not complete the trajectory. V4/V5 can complete some
diagnostic trajectories only with fallback. The final cold Linf gate is never
relaxed. Status: `IMPLEMENTED_DEFAULT_OFF_NOT_SUFFICIENT`.
""", encoding="utf-8")

    (REPORT / "adaptive_preconditioner_validation.md").write_text("""# Adaptive scalar preconditioner validation

The rule uses only an existing mobility reduction and clips `a_ref_eff` and
`D_ref_eff` to the preregistered tenfold interval. Unit tests cover finite and
fallback paths. Runtime V4 completes the 2000-step dt/4 diagnostic 20.5% faster
than V0 but requires 15 internal rejects and 10 fallback macros. V5 reduces
that short-window count to 2/2, yet fails the long-window zero-fallback gate.
Status: `RUNTIME_BENEFIT_OBSERVED_NOT_PRODUCTION_QUALIFIED`.
""", encoding="utf-8")

    head = subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          text=True, capture_output=True, check=True).stdout.strip()
    final = f"""baseline_commit=f440c0dcd4c02cd45d9079c35d3838ebfa9b37e2
candidate_worktree_head={head}
baseline_solver=adaptive_logit_feasible_ctot_v1
qualified_transport_gate=NONE
qualified_transport_gate_status=FAIL_V2_MATERIAL_TRANSPORT_DEFECT
selected_solver_name=NONE
selected_solver_version=NONE
selected_feature_mask=NONE
plateau_detector_status=IMPLEMENTED_DEFAULT_OFF_NOT_QUALIFIED
fraction_to_boundary_status=IMPLEMENTED_GPU_HOST_PARITY_NOT_QUALIFIED
hybrid_merit_status=IMPLEMENTED_DEFAULT_OFF_NOT_QUALIFIED
adaptive_spectral_status=IMPLEMENTED_DEFAULT_OFF_RUNTIME_BENEFIT_NOT_QUALIFIED
internal_homotopy_status=IMPLEMENTED_DEFAULT_OFF_LONG_WINDOW_FAIL
frozen_reject_cases_total=31
frozen_reject_direct_convergence_count=NOT_REPLAYABLE_EXACT_STATES_NOT_RECORDED
frozen_reject_retry_count=NOT_REPLAYABLE_EXACT_STATES_NOT_RECORDED
frozen_reject_fallback_count=NOT_REPLAYABLE_EXACT_STATES_NOT_RECORDED
dt16_status=V0_PASS_ZERO_FALLBACK_V5_FAIL_ZERO_FALLBACK_GATE
dt8_status=FAILED_STEP_3465
dt4_status=DIAGNOSTIC_COMPLETE_WITH_FALLBACK_NOT_QUALIFIED
dt2_status=FAILED_STEP_831
selected_dt_code=1.953125e-4_LEGACY_BASELINE_ONLY
selected_dt_physical_s=0.00803312_LEGACY_BASELINE_ONLY
selected_retry_count=0_LEGACY_BASELINE
selected_fallback_count=0_LEGACY_BASELINE
selected_macro_hard_reject_count=0_LEGACY_BASELINE
selected_V2_defect_status=FAIL_V2_MATERIAL_TRANSPORT_DEFECT
selected_holdout_status=FAIL
baseline_easy_step_wall_s={fmt((baseline_wall / 8000.0) if baseline_wall else None)}
selected_easy_step_wall_s=NA_NO_SELECTED_CANDIDATE
easy_path_regression=NA_NO_SELECTED_CANDIDATE
hard_path_speedup=NA_NO_QUALIFIED_CANDIDATE
baseline_peak_VRAM_GiB=SOURCE_LEDGER_DEPENDS_ON_N
selected_peak_VRAM_GiB=NO_SELECTED_CANDIDATE
extra_peak_VRAM_MiB=0_LOW_MEMORY_GLOBALIZATION
persistent_full_grid_fields_added=0
pointer_rotation_saved_MiB=0
uint8_mask_saved_MiB=0_NOT_RETAINED
scratch_reuse_saved_MiB=0
face_streaming_saved_MiB=0_NOT_RETAINED
device_total_VRAM_GiB=15.921
PF_only_400cube_required_GiB=19.580_SOURCE_EXACT_BEFORE_CUFFT
PF_elastic_400cube_required_GiB=34.145_SOURCE_EXACT_BEFORE_CUFFT
PF_elastic_GP_400cube_required_GiB=34.145_SOURCE_EXACT_BEFORE_CUFFT_S3_GATED
400cube_memory_status=BLOCKED_SOURCE_ALLOCATION_EXCEEDS_85_PERCENT
largest_safe_cubic_N=UNMEASURED_SOURCE_UPPER_BOUND_LT_400
400cube_speed_measurement_type=NOT_MEASURED
sparse_matrix_assembled=false
large_Krylov_basis_used=false
physical_model_changed=false
mass_contract_changed=false
cluster_used=false
commit_created=false
push_performed=false
recommended_next_action=RESOLVE_V2_DEFECT_AND_ACTIVE_BOUND_GLOBALIZATION_BEFORE_MEMORY_3D_WORK
final_status=FAIL_NO_QUALIFIED_LOW_MEMORY_PRODUCTION_CANDIDATE
"""
    (REPORT / "final_terminal_output.txt").write_text(final, encoding="utf-8")

    manifest_path = REPORT / "baseline_manifest.json"
    if manifest_path.exists():
        manifest_data = json.loads(manifest_path.read_text())
        manifest_data["finalization"] = {
            "source_hashes": {
                name: source_hash(name) for name in
                ("main_cuda.cu", "cuda_kernels.cu", "cuda_kernels.h",
                 "pf_params.h", "low_memory_transport_v1_utils.h")
            },
            "formal_gate": "FAIL_V2_MATERIAL_TRANSPORT_DEFECT",
            "selected_candidate": None,
        }
        manifest_path.write_text(json.dumps(manifest_data, indent=2, sort_keys=True) + "\n")
    print(final, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
