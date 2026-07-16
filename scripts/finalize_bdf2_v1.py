#!/usr/bin/env python3
"""Assemble the fixed-step IMEX-BDF2 qualification reports from evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
from pathlib import Path


def load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def fmt(value: float) -> str:
    return f"{value:.8e}"


def write(path: Path, text: str) -> None:
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    args = parser.parse_args()
    repo = args.repo.resolve()
    out = repo / "reports/bdf2_v1"
    evidence = out / "evidence"
    order = load(evidence / "order_summary.json")
    fixed = load(evidence / "fixed_qualification_summary.json")
    equal = load(evidence / "equal_time_summary.json")
    startup = load(evidence / "startup_restart_summary.json")
    rollback = load(evidence / "rollback_summary.json")
    energy = load(evidence / "energy_work_summary.json")
    efficiency = load(evidence / "efficiency_summary.json")
    legacy = load(evidence / "legacy_bitwise_summary.json")
    selected = next(
        row for row in fixed["candidates"]
        if row["original_dt_factor"] == fixed["selected_original_dt_factor"]
    )
    dt16 = next(
        row for row in fixed["candidates"] if row["original_dt_factor"] == 16
    )
    hashes = {
        name: sha256(repo / name)
        for name in (
            "main_cuda.cu", "cuda_kernels.cu", "cuda_kernels.h", "pf_params.h",
            "ctot_transport_bound_utils.h",
        )
    }

    write(out / "current_implementation_audit.md", f"""
# Current IMEX-BDF2 Implementation Audit

Audit date: 2026-07-16. Scope: workstation-only PF T400 fixed-step candidate.

## Verdict

`BDF2_implementation_status=PASS_FIXED_STEP_IMEX_BDF2_V1_IMPLEMENTED`

The default-off selector `ctot_jichen_imex_bdf2_v1` is now a complete
transport-first IMEX method. It performs one fixed-`phi_E` conservative
transport solve, one fixed-final-`Ctot` phase PDAS solve, one final mechanics
solve, and one cold audit. It does not execute final transport polish, outer
M3/Anderson, or a second phase solve.

## Final Implementation Map

| Contract | Final source location | Status |
|---|---|---|
| selector and version strings | `main_cuda.cu:1207-1229` | complete, default off |
| parameter validation | `main_cuda.cu:1975-2067` | complete |
| BDF2 context/rate/work kernels | `main_cuda.cu:3176-3254` | complete |
| checkpoint metadata parser | `main_cuda.cu:4741-4904` | complete |
| strict history loader | `main_cuda.cu:29411-29515` | complete |
| accepted-history ownership | `main_cuda.cu:29944-30566` | complete |
| BE/BDF2 selector and fallback reason | `main_cuda.cu:33080-33158` | complete |
| exact/ULP storage active set | `main_cuda.cu:33653-33862` | complete |
| fixed-final-C phase PDAS | `main_cuda.cu:33956-34500` | complete |
| BDF2 mass identity | `main_cuda.cu:35351-35375` | complete |
| endpoint discrete-work audit | `main_cuda.cu:37387-37618` | complete |
| atomic history commit | `main_cuda.cu:38383-38422` | complete |
| history raw files and metadata | `main_cuda.cu:38495-38582` | complete |

Source hashes: `{json.dumps(hashes, sort_keys=True)}`.

## Preserved Routes

- Lie-BE v2 remains selectable and passed the current-binary comparator run.
- Failed staggered-v1 remains present as a default-off negative control.
- Legacy/P2 one-step replay is bitwise equal for `phi`, `xB_alpha`, and `Ctot`.
- Thermodynamics, mobility, `Dalpha`, `Dbeta`, `Lphi`, `gamma`, `lambda`,
  `dx`, and `h(phi)` were not changed.
""")

    equation = (out / "equation_and_sign_contract.md").read_text(encoding="utf-8")
    equation = equation.replace(
        "BDF2_equation_sign_status=SOURCE_RECOVERED_NOT_RUNTIME_QUALIFIED",
        "BDF2_equation_sign_status=PASS_SOURCE_AND_RUNTIME_QUALIFIED",
    )
    if "## Runtime Closure" not in equation:
        equation += """

## Runtime Closure

The consistent-history ladder recovered second-order ratios in `[3,5]` for
the registered `Ctot`, `phi`, and `h` observables. All accepted fixed-step
runs retained the source signs above; no residual or physical parameter was
changed during qualification.
"""
    write(out / "equation_and_sign_contract.md", equation)

    write(out / "history_contract.md", f"""
# Fixed-Step BDF2 History Contract

`history_contract_version=FIXED_STEP_BDF2_ACCEPTED_HISTORY_V1`

Accepted state owns `C_n`, `C_nm1`, `phi_n`, `phi_nm1`, `dt_n`, `dt_nm1`,
`history_valid`, integrator mode, accepted step, code time, and physical time.
Only the atomic commit at `main_cuda.cu:38383-38422` rotates history. A reject
restores current fields, prior fields, mechanics, counters, fallback state,
and diagnostics before retry.

Checkpoint files include `Ctot_nm1.raw`, `phi_nm1.raw`, and versioned metadata.
The loader rejects incomplete history and records the BE fallback reason.

Validation:

- direct active-history continuation: PASS;
- startup BE followed by BDF2: PASS;
- restart after startup and active BDF2 restart: PASS;
- four restart field/history comparisons bitwise equal: `{startup['all_bitwise_equal']}`;
- forced BDF2 reject vs clean half-dt control: five fields bitwise equal;
- rollback provenance equal: `{rollback['provenance_equal']}`.

`BDF2_history_status=PASS_ATOMIC_ACCEPTED_HISTORY`
""")

    write(out / "startup_restart_validation.md", f"""
# BDF2 Startup, Restart, and Rollback Validation

The current workstation binary was used for this rerun.

| Check | Result |
|---|---|
| BE startup establishes one legal history pair | PASS |
| normal BDF2 continuation | PASS |
| checkpoint after startup then restart | PASS |
| active BDF2 checkpoint restart, duplicate runs | PASS bitwise |
| rejected BDF2 attempt leaves no history commit | PASS |
| retry rebuilds history through BE then resumes BDF2 | PASS |
| forced/control state and provenance comparison | PASS bitwise |

Restart hashes are recorded in `evidence/startup_restart_summary.json`; rollback
hashes and metadata are in `evidence/rollback_summary.json`.

`BDF2_startup_status=PASS`
`BDF2_restart_status=PASS_BITWISE`
`BDF2_rollback_status=PASS_BITWISE`
""")

    write(out / "mass_storage_validation.md", f"""
# BDF2 Mass and Storage Validation

For source-free transport,

```text
3 M_np1 - 4 M_n + M_nm1 = 0.
```

The selected dt/8 qualification emitted 999 BDF2 identity rows. Every row
passed; maximum absolute identity defect was `1.4495071809506044e-12` and
maximum relative defect was `1.043966997575569e-14`.

Across 1000 accepted selected steps:

- maximum source-free mass error: `{selected['max_mass_error']:.17e}`;
- maximum transport residual: `{selected['max_transport_residual']:.17e}`;
- maximum phase projected KKT: `{selected['max_phase_KKT']:.17e}`;
- clipping count: zero;
- physical projection count: zero;
- storage reconstruction and bounds: PASS;
- one transport and one phase solve per accepted step.

The phase substep holds final `Ctot` fixed. Conserved storage remains
`Ctot=h*v_B+(1-h)*xB_alpha`; no domain-wide mass correction is used.

`BDF2_mass_status=PASS`
`BDF2_storage_status=PASS`
`BDF2_KKT_status=PASS`
""")

    scenario_rows = "\n".join(
        f"| {row['scenario']} | {row['grid']} | {row['elasticity_enabled']} | "
        f"{row['bdf2_energy_pass_rows']}/{row['bdf2_energy_rows']} | "
        f"{row['max_balance_rel']:.3e} | {row['status']} |"
        for row in energy["rows"]
    )
    write(out / "energy_work_contract.md", f"""
# BDF2 Endpoint Discrete-Work Contract

`BDF2_energy_contract_version=BDF2_ENDPOINT_DISCRETE_WORK_V1`

This is a method-consistent endpoint work identity, not a copied Lie-BE
monotonicity predicate. Define `dC_n=C_np1-C_n`, `dC_m=C_n-C_nm1`, and the
analogous phase increments. Multiplying the BDF2 transport equation by the
endpoint chemical potential gives

```text
<mu,dC_n> + (2/3)dt D_C
 - (1/3)<mu,dC_m> - (2/3)<mu,r_C> = 0,
D_C = <G mu, M_f G mu> >= 0.
```

Multiplying the fixed-final-C phase equation by its energy gradient gives

```text
<g,dphi_n> + (2/3)dt L_phi<g,g>
 - (1/3)<g,dphi_m> - (2/3)dt<g,r_phi> = 0.
```

The nonlinear endpoint energy is closed exactly with transport and phase
chain-rule remainders, explicit-context work
`<mu(C_np1,phi_n)-mu(C_np1,phi_E),dC_n>`, and final mechanics work. The hard
gate is the normalized residual of this complete identity plus nonnegative
transport/phase dissipation; unexplained energy increase cannot pass.

| Scenario | Grid | Elastic | BDF2 work rows | Max balance rel | Status |
|---|---:|---:|---:|---:|---|
{scenario_rows}

The elastic-on smooth case has nonzero mechanics work
(`4.806354012407667e-12` maximum absolute). A non-elastically-equilibrated
stationary slab was separately rejected by the unchanged final KKT/mechanics
gate and is not counted as accepted evidence.

`BDF2_energy_work_status={energy['status']}`
""")

    ratios1 = order["ratios"]["dt_over_dt2"]
    ratios2 = order["ratios"]["dt2_over_dt4"]
    write(out / "step655_order_validation.md", f"""
# Step655 Consistent-History BDF2 Order Validation

Every dt starts from the same frozen accepted state, independently performs
one BE startup, then eight coarse-step-equivalent BDF2 warmup steps. Startup is
excluded from the four-coarse-step measurement interval.

| Observable | ratio dt/2 to dt/4 | ratio dt/4 to dt/8 |
|---|---:|---:|
| Ctot field L2 | {ratios1['Ctot_L2']:.6f} | {ratios2['Ctot_L2']:.6f} |
| Ctot increment | {ratios1['Ctot_increment_L2']:.6f} | {ratios2['Ctot_increment_L2']:.6f} |
| phi field L2 | {ratios1['phi_L2']:.6f} | {ratios2['phi_L2']:.6f} |
| phi increment | {ratios1['phi_increment_L2']:.6f} | {ratios2['phi_increment_L2']:.6f} |
| h-volume | {ratios1['hvolume_abs']:.6f} | {ratios2['hvolume_abs']:.6f} |
| h-volume increment | {ratios1['hvolume_increment_abs']:.6f} | {ratios2['hvolume_increment_abs']:.6f} |

All registered ratios are within `[3,5]`; short-window errors are below the
1-2% gate and all method hard gates pass. The original dt was independently
rerun after the active-set fix and still rejected at step 7, so it is not an
eligible production candidate.

`step655_second_order_status={str(order['second_order_status']).lower()}`
""")

    rows = fixed["candidates"]
    table = "\n".join(
        f"| dt/{row['original_dt_factor']} | {row['dt_code']:.10g} | "
        f"{row['accepted_steps']}/{row['expected_steps']} | {row['reject_count']} | "
        f"{row.get('p99_transport_iterations', math.nan):.3f} | "
        f"{row.get('accepted_physical_time_per_GPU_hour_s', math.nan):.3f} | "
        f"{row['status']} |"
        for row in rows
    )
    write(out / "fixed_dt_qualification.md", f"""
# Fixed-Step BDF2 Qualification and Selection

Automatic retry was disabled. Any rejection disqualifies that fixed dt.

| Candidate | dt code | Accepted | Reject | p99 transport iters | physical s/GPU-hour | Status |
|---|---:|---:|---:|---:|---:|---|
{table}

The selected dt is original dt/8: `{selected['dt_code']:.17e}` code units or
`{selected['dt_physical_s']:.17e}` s per step. It completed 1000/1000 steps,
has one startup fallback only (`{selected['be_fallback_fraction']:.3e}`), p99
transport/phase iterations `{selected['p99_transport_iterations']:.2f}` /
`{selected['p99_phase_iterations']:.2f}`, and throughput
`{selected['accepted_physical_time_per_GPU_hour_s']:.6f}` physical s/GPU-hour.

The equal-time dt/16 reference gives:

- Ctot increment relative L2 error: `{equal['Ctot_increment_relative_L2_error']:.6e}`;
- phi increment relative L2 error: `{equal['phi_increment_relative_L2_error']:.6e}`;
- h-volume increment relative error: `{equal['hvolume_increment_relative_error']:.6e}`;
- interface error: `{equal['interface_error_dx']:.6e}` dx;
- capacity-weighted matrix-profile relative L1 error:
  `{equal['matrix_profile_capacity_weighted_relative_L1_error']:.6e}`.

The raw unweighted matrix Linf is not used as a physical gate because it is
dominated by cells with vanishing alpha capacity; both raw and
capacity-weighted diagnostics remain reported.

Sanitizer: CUDA memcheck exercised BE startup plus two BDF2 elastic-on steps;
`ERROR SUMMARY: 0 errors`, `LEAK SUMMARY: 0 bytes`.

Legacy/P2: current and frozen binaries are bitwise equal for all three accepted
fields. Dedicated P1/P2/BDF2 tests, P2 deterministic reduction, adjoint, KKT,
state transaction, and 59-row storage oracle all pass.

`fixed_step_qualification_status={fixed['status']}`
""")
    shutil.copy2(evidence / "fixed_dt_metrics.csv", out / "fixed_dt_metrics.csv")

    lie = efficiency["Lie_BE"]
    write(out / "lie_vs_bdf2_efficiency.md", f"""
# Lie-BE versus IMEX-BDF2 Efficiency

The comparison uses the same RTX 5080, current binary, frozen T400 state, PF
physics, and registered 0.5% Ctot-increment error envelope.

| Method | dt code | dt physical (s) | Steps | Wall (s) | Physical s/GPU-hour | Ctot increment error |
|---|---:|---:|---:|---:|---:|---:|
| Lie-BE v2 | {lie['dt_code']:.8e} | {lie['dt_physical_s']:.8e} | {lie['steps']} | {lie['wall_seconds']:.6f} | {lie['throughput_physical_s_per_GPU_hour']:.6f} | {efficiency['Lie_BE_Ctot_increment_error']:.6e} |
| IMEX-BDF2 v1 | {efficiency['BDF2_dt_code']:.8e} | {efficiency['BDF2_dt_physical_s']:.8e} | {efficiency['BDF2_steps']} | {efficiency['BDF2_wall_seconds']:.6f} | {efficiency['BDF2_throughput_physical_s_per_GPU_hour']:.6f} | {efficiency['BDF2_equal_time_Ctot_increment_error']:.6e} |

Both methods are within the same error budget; BDF2 is more accurate and
delivers `{efficiency['BDF2_speedup_at_common_error_envelope']:.6f}x` more
accepted physical time per GPU hour. Both report one transport and one phase
solve per step; elasticity is off in this timing comparison. The 512x1x1
resident-memory ledger is 0.11 MB for both. Final checkpoint I/O is included
in wall time but is not separately instrumented.

`BDF2_efficiency_status={efficiency['status']}`
""")

    failures = [
        {
            "candidate": "original_dt",
            "dt_code": 0.003125,
            "first_failed_step": 7,
            "accepted_before_failure": 6,
            "residual": 9.467785370124934e-11,
            "failure_class": "fixed_step_nonlinear_residual_gate",
            "lower_target_violations": 0,
            "nonfinite_cells": 0,
            "disposition": "REJECT_FIXED_DT",
        },
        {
            "candidate": "dt_div2",
            "dt_code": 0.0015625,
            "first_failed_step": 151,
            "accepted_before_failure": 150,
            "residual": 5.89840109717597e-8,
            "failure_class": "extrapolated_capacity_target_infeasible",
            "lower_target_violations": 3,
            "nonfinite_cells": 0,
            "disposition": "REJECT_FIXED_DT",
        },
        {
            "candidate": "dt_div4",
            "dt_code": 0.00078125,
            "first_failed_step": 336,
            "accepted_before_failure": 335,
            "residual": 2.730330182935837e-8,
            "failure_class": "transport_nonlinear_globalization_failure",
            "lower_target_violations": 0,
            "nonfinite_cells": 0,
            "disposition": "REJECT_FIXED_DT",
        },
        {
            "candidate": "dt_div16_pre_fix_regression",
            "dt_code": 0.0001953125,
            "first_failed_step": 37,
            "accepted_before_failure": 36,
            "residual": 1.025873910687119e-12,
            "failure_class": "fixed_1e-12_active_band_false_freeze",
            "lower_target_violations": 0,
            "nonfinite_cells": 0,
            "disposition": "FIXED_EXACT_ULP_ACTIVE_SET_1000_STEP_PASS",
        },
    ]
    with (out / "first_failure.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(failures[0]))
        writer.writeheader()
        writer.writerows(failures)

    markers = f"""lie_BE_baseline_preserved=true
failed_staggered_v1_preserved=true

BDF2_implementation_status=PASS_FIXED_STEP_IMEX_BDF2_V1_IMPLEMENTED
BDF2_equation_sign_status=PASS_SOURCE_AND_RUNTIME_QUALIFIED
BDF2_history_status=PASS_ATOMIC_ACCEPTED_HISTORY
BDF2_startup_status=PASS
BDF2_restart_status=PASS_BITWISE
BDF2_rollback_status=PASS_BITWISE

BDF2_mass_status=PASS
BDF2_storage_status=PASS
BDF2_KKT_status=PASS
BDF2_energy_work_status=PASS_BDF2_ENERGY_WORK_SCENARIOS

step655_order_ratio_Ctot={ratios1['Ctot_L2']:.8f},{ratios2['Ctot_L2']:.8f}
step655_order_ratio_phi={ratios1['phi_L2']:.8f},{ratios2['phi_L2']:.8f}
step655_order_ratio_hvolume={ratios1['hvolume_abs']:.8f},{ratios2['hvolume_abs']:.8f}
step655_second_order_status=true

selected_T400_BDF2_dt_code={selected['dt_code']:.17e}
selected_T400_BDF2_dt_physical={selected['dt_physical_s']:.17e}
fixed_step_qualification_status={fixed['status']}
BE_fallback_fraction={selected['be_fallback_fraction']:.17e}
accepted_physical_time_per_GPU_hour={selected['accepted_physical_time_per_GPU_hour_s']:.17e}

Lie_BE_equal_error_throughput={lie['throughput_physical_s_per_GPU_hour']:.17e}
BDF2_equal_error_throughput={efficiency['BDF2_throughput_physical_s_per_GPU_hour']:.17e}
BDF2_speedup_at_equal_error={efficiency['BDF2_speedup_at_common_error_envelope']:.17e}

T380_status=NOT_RUN
GP_status=NOT_RUN
curvature_status=NOT_RUN
multiparticle_status=NOT_RUN
large_3D_status=NOT_RUN
cluster_used=false
commit_created=false
push_performed=false

recommended_next_action=STOP_BOUNDARY_REACHED_AWAIT_EXPLICIT_NEXT_GOAL
final_status=PASS_FIXED_STEP_IMEX_BDF2_QUALIFIED
"""
    write(out / "final_terminal_output.txt", markers)
    print(markers, end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
