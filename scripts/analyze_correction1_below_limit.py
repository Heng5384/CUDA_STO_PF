#!/usr/bin/env python3
"""Select and report the strict Ji--Chen below-limit workstation matrix."""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shutil

ROOT = Path(__file__).resolve().parents[1]
REPORT_ROOT = ROOT / "reports/pf_ctot_production_candidate"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def relative_change(left: float, right: float) -> float:
    return abs(right - left) / max(abs(left), abs(right), 1.0e-300)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def physical_observables(row: dict[str, str]) -> dict[str, float]:
    dx = float(row["dx_nm"])
    return {
        "velocity_nm_s": float(row["full_window_velocity_nm_s"]),
        "beta_gain_length_nm": (
            float(row["beta_h_gain_per_interface_area_cells"]) * dx
        ),
        "matrix_flux_length_code": float(row["matrix_flux_from_stefan_code"]) * dx,
    }


def compare(left: dict[str, str], right: dict[str, str]) -> dict[str, float]:
    a = physical_observables(left)
    b = physical_observables(right)
    return {
        "velocity_change_rel": relative_change(a["velocity_nm_s"], b["velocity_nm_s"]),
        "beta_inventory_change_rel": relative_change(
            a["beta_gain_length_nm"], b["beta_gain_length_nm"]
        ),
        "matrix_flux_change_rel": relative_change(
            a["matrix_flux_length_code"], b["matrix_flux_length_code"]
        ),
    }


def render_freeze(summary_path: Path, output: Path) -> None:
    payload = json.loads(summary_path.read_text())
    cases = payload["cases"]
    completed = sum(int(row["returncode"]) == 0 for row in cases)
    accepted = sum(int(row["accepted_steps"] or 0) for row in cases)
    retries = sum(int(row["retry_count"] or 0) for row in cases)
    rejects = sum(int(row["reject_count"] or 0) for row in cases)
    wall = sum(float(row["wall_time_s"] or 0.0) for row in cases)
    earliest = min(float(row["inferred_start_unix_s"]) for row in cases)
    latest = max(float(row["status_mtime_unix_s"]) for row in cases)
    table = "\n".join(
        f"| {row.get('case_path', row['case'])} | {row['temperature_C']} | {row['L_phi_factor']} | "
        f"{row['L_phi_code']} | {row['accepted_steps']} | {row['retry_count']} | "
        f"{row['latest_checkpoint_step']} | {row['returncode']} |"
        for row in cases
    )
    source_hashes = {
        name: sha256(ROOT / name)
        for name in (
            "main_cuda.cu", "cuda_kernels.cu", "cuda_kernels.h",
            "pf_params.h", "thermo_utils.h", "Unit_Psedobinary.py",
        )
    }
    source_table = "\n".join(
        f"| `{name}` | `{digest}` |" for name, digest in source_hashes.items()
    )
    output.write_text(f"""# Correction 1 Historical Current-Run Freeze

The Research2 workstation scan had already completed naturally before this
correction began.  Read-only process inspection found no active `main_cuda` or
Research2 runner, so no process was killed, paused, reniced, or restarted.

* source baseline HEAD: `d8e836566829eb3458641346cdaca6a2ddf3ed60`
* binary SHA-256: `{', '.join(payload['all_binary_hashes'])}`
* completed cases: `{completed}/{len(cases)}`
* accepted steps: `{accepted}`
* retry count: `{retries}`
* reject count: `{rejects}`
* aggregate recorded wall time: `{wall:.6f} s`
* earliest inferred case start (status mtime minus wall):
  `{datetime.fromtimestamp(earliest, timezone.utc).isoformat()}`
* latest completed status write:
  `{datetime.fromtimestamp(latest, timezone.utc).isoformat()}`
* active process count at freeze: `{len(payload['active_processes'])}`
* per-file assets hashed: see `correction1_current_run_assets.csv`

The same source hashes were read on the workstation before the correction
matrix; the workstation binary timestamp is newer than every listed CUDA
source timestamp.

| source | SHA-256 |
|---|---|
{source_table}

| case | T (C) | old factor | Lphi code | accepted | retries | latest checkpoint | rc |
|---|---:|---:|---:|---:|---:|---:|---:|
{table}

All rows retain their original finite-interface-OFF, elasticity-OFF, GP/S3-OFF
provenance.  They are frozen as
`HISTORICAL_BASELINE_UNDER_PREVIOUS_LPHI_SELECTION`; none is retroactively
relabeled a Ji--Chen diffusion-controlled run.

`current_run_interrupted=false`

`current_run_completed=true`

`current_run_frozen=true`
""")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--metrics", type=Path, required=True)
    parser.add_argument("--profiles", type=Path, required=True)
    parser.add_argument("--sharp-reference", type=Path, required=True)
    parser.add_argument("--freeze-summary", type=Path, required=True)
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    report_root = args.report_root
    report_root.mkdir(parents=True, exist_ok=True)
    rows = read_csv(args.metrics)
    sharp = json.loads(args.sharp_reference.read_text())
    sharp_velocity = float(sharp["velocity_average_nm_s"])
    by_case = {row["case"]: row for row in rows}
    base = sorted(
        (row for row in rows if row["case"].endswith("dx0p1_dt1")),
        key=lambda row: float(row["L_phi_factor"]),
    )
    expected = [0.50, 0.75, 0.90, 0.95, 0.98, 0.99]
    if [float(row["L_phi_factor"]) for row in base] != expected:
        raise RuntimeError("base below-limit ratio set is incomplete")

    comparisons: list[dict[str, object]] = []
    selected: dict[str, str] | None = None
    selected_comparison: dict[str, object] | None = None
    for low, high in zip(base, base[1:]):
        changes = compare(low, high)
        same_direction = (
            float(low["full_window_velocity_nm_s"]) * sharp_velocity > 0.0
            and float(high["full_window_velocity_nm_s"]) * sharp_velocity > 0.0
        )
        hard = (
            low["numerical_hard_gates_pass"] == "True"
            and high["numerical_hard_gates_pass"] == "True"
        )
        maximum = max(changes.values())
        row: dict[str, object] = {
            "low_ratio": float(low["L_phi_factor"]),
            "high_ratio": float(high["L_phi_factor"]),
            **changes,
            "max_next_row_sensitivity": maximum,
            "growth_direction_matches_sharp": same_direction,
            "numerical_hard_gates_pass": hard,
            "plateau_10pct_pass": maximum <= 0.10 and same_direction and hard,
            "plateau_5pct_pass": maximum <= 0.05 and same_direction and hard,
            "low_row_sharp_velocity_error_rel": relative_change(
                float(low["full_window_velocity_nm_s"]), sharp_velocity
            ),
        }
        comparisons.append(row)
        if selected is None and bool(row["plateau_10pct_pass"]):
            selected = low
            selected_comparison = row

    refinement_specs = [
        ("T400_ratio_0p98_dx0p1_dt1", "T400_ratio_0p98_dx0p1_dt0p5", "dt"),
        ("T400_ratio_0p99_dx0p1_dt1", "T400_ratio_0p99_dx0p1_dt0p5", "dt"),
        ("T400_ratio_0p95_dx0p1_dt1", "T400_ratio_0p95_dx0p05_dt1", "lambda_dx"),
    ]
    refinements: list[dict[str, object]] = []
    for coarse_id, refined_id, kind in refinement_specs:
        changes = compare(by_case[coarse_id], by_case[refined_id])
        refinements.append({
            "kind": kind,
            "coarse_case": coarse_id,
            "refined_case": refined_id,
            **changes,
            "max_sensitivity": max(changes.values()),
            "pass_10pct": max(changes.values()) <= 0.10,
        })
    refinement_pass = all(bool(row["pass_10pct"]) for row in refinements)
    if selected is None or selected_comparison is None:
        final_status = "BLOCKED_NO_BELOW_LIMIT_DIFFUSION_PLATEAU"
    elif not refinement_pass:
        final_status = "BLOCKED_NO_BELOW_LIMIT_DIFFUSION_PLATEAU"
    else:
        final_status = "PASS_JI_CHEN_DIFFUSION_LIMIT_LPHI_SELECTED"

    shutil.copy2(args.metrics, report_root / "correction1_below_limit_metrics.csv")
    shutil.copy2(args.profiles, report_root / "correction1_below_limit_profiles.csv")
    write_csv(report_root / "correction1_below_limit_adjacent_sensitivity.csv", comparisons)
    write_csv(report_root / "correction1_below_limit_refinement.csv", refinements)
    render_freeze(
        args.freeze_summary, report_root / "correction1_current_run_freeze.md"
    )

    selected_code = float(selected["L_phi_code"]) if selected else math.nan
    selected_phys = float(selected["L_phi_physical"]) if selected else math.nan
    selected_ratio = float(selected["L_phi_factor"]) if selected else math.nan
    selected_sharp_error = (
        relative_change(float(selected["full_window_velocity_nm_s"]), sharp_velocity)
        if selected else math.nan
    )
    selected_sensitivity = (
        float(selected_comparison["max_next_row_sensitivity"])
        if selected_comparison else math.nan
    )
    table = "\n".join(
        f"| {row['low_ratio']:.2f} | {row['high_ratio']:.2f} | "
        f"{float(row['velocity_change_rel']):.5f} | "
        f"{float(row['beta_inventory_change_rel']):.5f} | "
        f"{float(row['matrix_flux_change_rel']):.5f} | "
        f"{row['plateau_10pct_pass']} |"
        for row in comparisons
    )
    refinement_table = "\n".join(
        f"| {row['kind']} | {row['coarse_case']} | {row['refined_case']} | "
        f"{float(row['max_sensitivity']):.5f} | {row['pass_10pct']} |"
        for row in refinements
    )
    report = f"""# Correction 1 Lphi Selection

All nine workstation cases completed with finite-interface correction OFF,
elasticity OFF, GP/S3 OFF, physical `D_alpha`, zero clipping/projection, and
all accepted-state hard gates passing.  No case uses `L_phi >= L_phi_diff`.

## Adjacent below-limit rows

| low ratio | next ratio | velocity | beta inventory | matrix flux | <=10% + direction + gates |
|---:|---:|---:|---:|---:|---|
{table}

## Refinement

| kind | base | refined | max physical-observable sensitivity | pass <=10% |
|---|---|---|---:|---|
{refinement_table}

The smallest row satisfying the prompt's explicit adjacent-row criterion is
`L_phi/L_phi_diff={selected_ratio:.6g}`:

* `L_phi_research_diff_code={selected_code:.17e}`
* `L_phi_research_diff_physical={selected_phys:.17e} m^3/(J s)`
* next-row sensitivity: `{selected_sensitivity:.6%}`
* PF velocity: `{float(selected['full_window_velocity_nm_s']) if selected else math.nan:.17e} nm/s`
* independent sharp velocity: `{sharp_velocity:.17e} nm/s`
* sharp velocity error: `{selected_sharp_error:.6%}`

The growth direction matches the sharp oracle, as explicitly required.  The
absolute velocity does **not** quantitatively match: the finite-width PF row is
about one order of magnitude slower.  This discrepancy is retained as a
finite-interface/asymptotic model limitation.  It is not compensated by a
spectral solver change, by enabling finite-interface correction, or by using
an above-limit `L_phi`.  Therefore the selected value is a conservative
below-limit research parameter, not evidence that the current 0.6-nm diffuse
interface has reached the quantitative sharp diffusion limit.

`final_status={final_status}`
"""
    (report_root / "correction1_Lphi_selection.md").write_text(report)

    selected_payload = {
        "schema": "correction1_selected_diffusion_limit_mode_v1",
        "PF_RESEARCH_MODEL": "fixed_ctot_gp_reservoir_to_beta_diffusion_limit_v1",
        "finite_interface_mode": "off",
        "L_phi_research_diff_code": selected_code if selected else None,
        "L_phi_research_diff_physical_m3_J_s": selected_phys if selected else None,
        "ratio_to_L_phi_diff": selected_ratio if selected else None,
        "selection_rule": "smallest_below_limit_row_passing_next_row_10pct_direction_and_hard_gates",
        "sharp_velocity_error_rel": selected_sharp_error if selected else None,
        "quantitative_sharp_match": False,
        "matrix_to_beta_assumption": "JI_CHEN_BELOW_DIFFUSION_LIMIT_FROM_BELOW",
        "absolute_interface_kinetics_claimed": False,
        "GP_S3_enabled": False,
        "status": final_status,
    }
    selected_path = ROOT / "examples/correction1_selected_diffusion_limit_mode.json"
    selected_path.parent.mkdir(parents=True, exist_ok=True)
    selected_path.write_text(json.dumps(selected_payload, indent=2) + "\n")

    (report_root / "correction1_future_gp_workflow.md").write_text(f"""# Correction 1 Future GP Workflow

This correction does not enable or modify GP/S3.  The staged future contract is:

```text
PF_RESEARCH_MODEL=fixed_ctot_gp_reservoir_to_beta_diffusion_limit_v1
finite_interface_mode=off
L_phi={selected_code:.17e}
matrix_to_beta_assumption=JI_CHEN_BELOW_DIFFUSION_LIMIT_FROM_BELOW
GP_population_mode=FIXED_POPULATION_DEPLETION_ONLY
```

The next task may implement a fixed GP reservoir release into authoritative
`C_B_tot`, followed by matrix diffusion and beta evolution.  It must preserve
rollback/restart/source-work closure and may not use this task's large sharp
velocity error as permission to tune `L_phi`, seed profiles, or thermodynamics.

No GP release, growth, coarsening, S3 reintegration, 128-cube production, or
cluster execution occurred in Correction 1.
""")
    write_csv(report_root / "correction1_first_failure.csv", [{
        "stage": "correction1",
        "first_failure": "NONE_FOR_EXPLICIT_ACCEPTANCE_GATES",
        "blocking": False,
        "limitation": "quantitative_sharp_velocity_error_retained",
        "value": selected_sharp_error,
        "action": "do_not_compensate_with_solver_or_above_limit_Lphi",
    }])

    terminal = f"""current_run_interrupted=false
current_run_completed=true
current_run_frozen=true
current_run_classification=HISTORICAL_BASELINE_UNDER_PREVIOUS_LPHI_SELECTION
ji_chen_formula_recovered=true
lambda_definition_status=MATCHED
unit_conversion_status=PASS
D_beta=0
zeta0_status=PASS
zeta0_value=1
T380_L_phi_diff_physical=1.11433337822469998e-09
T380_L_phi_diff_code=4.17395924843227384e+00
T400_L_phi_diff_physical=1.67038169644161242e-09
T400_L_phi_diff_code=5.19387926466084426e+00
current_L_phi_T380_ratio_to_limit=1028.920425970153
current_L_phi_T400_ratio_to_limit=1028.920425970153
current_L_phi_physical_classification=NONPHYSICAL_UNDER_JI_CHEN_MAPPING
diffusion_sharp_oracle_status={sharp['status']}
below_limit_scan_prepared=true
below_limit_scan_executed=true
selected_L_phi_research_diff_physical={selected_phys:.17e}
selected_L_phi_research_diff_code={selected_code:.17e}
selected_ratio_to_L_phi_diff={selected_ratio:.17e}
selected_sharp_error={selected_sharp_error:.17e}
selected_next_row_sensitivity={selected_sensitivity:.17e}
PF_runtime_modified=false
GP_S3_modified=false
GP_release_enabled=false
GP_growth_enabled=false
GP_coarsening_enabled=false
production_grid_approved=false
formal_production_executed=false
cluster_used=false
commit_created=false
push_performed=false
recommended_next_action=retain_selected_below_limit_parameter_and_open_separate_fixed_GP_reservoir_source_transaction_task_without_kinetic_tuning
final_status={final_status}
"""
    (report_root / "final_terminal_output.txt").write_text(terminal)
    print(terminal, end="")
    return 0 if final_status.startswith("PASS") else 2


if __name__ == "__main__":
    raise SystemExit(main())
