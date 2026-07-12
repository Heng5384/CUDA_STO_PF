#!/usr/bin/env python3
"""Merge completed RSMD and timestep-refinement evidence without changing physics."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def f(value: object, default: float = math.nan) -> float:
    try:
        return float(str(value))
    except (TypeError, ValueError):
        return default


def get(rows_: list[dict[str, str]], temperature: int, group: str) -> dict[str, str] | None:
    return next((row for row in rows_ if int(f(row.get("temperature_C"), -1)) == temperature and
                 row.get("group") == group), None)


def yes(value: object) -> bool:
    return str(value).strip().lower() == "true"


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    baseline_root = root / "reports" / "post_remediation_growth_validation_transport_fixed"
    refinement_root = root / "reports" / "post_remediation_dt_refinement"
    out = baseline_root / "post_remediation_interface_supply_final_evidence.md"
    terminal = baseline_root / "final_terminal_output_after_dt_refinement.txt"

    baseline = rows(baseline_root / "post_remediation_growth_summary_table.csv")
    refined = rows(refinement_root / "dt_refinement_summary.csv")
    semantics = {row.get("key"): row.get("value")
                 for row in rows(baseline_root / "post_remediation_runtime_semantics.csv")}
    moving_rows = rows(baseline_root / "post_remediation_interface_supply_time_series.csv")
    mass_rows = rows(baseline_root / "post_remediation_mass_transfer_efficiency.csv")

    prerequisites = {
        "baseline_matrix_complete": len(baseline) == 12 and all(row.get("status") == "EXIT 0" for row in baseline),
        "dt_refinement_complete": len(refined) == 4 and all(row.get("status") == "EXIT 0" for row in refined),
        "no_direct_phi_write": semantics.get("direct_phi_write") == "false",
        "no_direct_GP_to_beta_transfer": semantics.get("direct_GP_to_beta_transfer") == "false",
        "no_matrix_reset": semantics.get("matrix_reset") == "false",
        "no_JGP_thermo_reuse": semantics.get("JGP_release_thermo_reuse") == "false",
        "moving_interface_rows_present": bool(moving_rows),
        "mass_transfer_rows_present": bool(mass_rows),
    }

    refined_by_temp: dict[int, dict[str, dict[str, str]]] = {380: {}, 400: {}}
    for row in refined:
        temp = int(f(row.get("temperature_C"), -1))
        if temp in refined_by_temp:
            refined_by_temp[temp][row.get("group", "")] = row

    evidence: list[str] = []
    both_growth = True
    stable_all = True
    source_off_completed = True
    for temp in (380, 400):
        on = refined_by_temp[temp].get("source_on")
        off = refined_by_temp[temp].get("source_off_tail")
        if not on or not off:
            stable_all = False
            source_off_completed = False
            both_growth = False
            evidence.append(f"- T{temp}: missing refined source-on or source-off-tail row.")
            continue
        stable_on = yes(on.get("timestep_refinement_stable"))
        stable_off = yes(off.get("timestep_refinement_stable"))
        d_r = f(on.get("relative_delta_R_eff_h"))
        d_h = f(on.get("relative_delta_h_integral"))
        source_off_slope = f(off.get("source_off_tail_R_slope_nm_per_step"))
        growth = stable_on and d_r > 0.0 and d_h > 0.0
        both_growth &= growth
        stable_all &= stable_on and stable_off
        source_off_completed &= int(f(off.get("source_off_tail_steps"), 0)) > 0
        tail = ("self-sustained candidate" if source_off_slope > 0.0 else
                "source-maintained/transient under this scenario" if math.isfinite(source_off_slope) else
                "tail not resolved")
        evidence.append(
            f"- T{temp}: source-on stable={stable_on}; dR/R={d_r:.6g}; dh/h={d_h:.6g}; "
            f"source-off tail slope={source_off_slope:.6g} nm/step ({tail})."
        )

    if not all(prerequisites.values()):
        status = "INCOMPLETE_INTERFACE_SUPPLY_EVIDENCE_BUNDLE"
    elif not stable_all:
        status = "FAIL_INTERFACE_SUPPLY_TIMESTEP_STABILITY_NOT_CONFIRMED"
    elif not both_growth:
        status = "PARTIAL_INTERFACE_SUPPLY_STABLE_BUT_GROWTH_NOT_ROBUST_AT_BOTH_TEMPERATURES"
    elif not source_off_completed:
        status = "PARTIAL_INTERFACE_SUPPLY_GROWTH_WITHOUT_SOURCE_OFF_TAIL"
    else:
        status = "PASS_INTERFACE_SUPPLY_LOCALIZED_HALO_GROWTH_MECHANISM_AUDIT"

    mass_gate = all(f(row.get("mass_error_rel_max")) <= 1.0e-10 for row in mass_rows if row.get("run_id"))
    text = [
        "# Interface-Supply / Localized-Halo GP-to-Beta Evidence",
        "",
        f"final_status={status}",
        "",
        "## Verified semantic boundary",
        "",
        "The scenario relay moves ledger inventory only from eligible static GP reservoirs "
        "to matrix-side alpha halo cells. It does not directly write beta phi, beta volume, "
        "or reset the external matrix. The selected dynamic-continued seed is pre-existing "
        "before the relay is assessed.",
        "",
        "## Completion gates",
        "",
    ]
    text.extend(f"- {key}: {value}" for key, value in prerequisites.items())
    text.extend([
        f"- mass_closure_max_1e-10_for_baseline_rows: {mass_gate}",
        "",
        "## Rate-preserving timestep-refinement evidence",
        "",
        "The refinement changes dt from 0.005 to 0.001 and uses f_max_per_step=8e-5, "
        "rather than 4e-4, to preserve the original physical-time source ceiling. It is "
        "therefore a numerical stability test, not a stronger GP supply scenario.",
        "",
    ])
    text.extend(evidence)
    text.extend([
        "",
        "## Publication boundary",
        "",
        "This evidence can support a bounded, required-supply-informed scenario statement: "
        "eligible GP-reservoir inventory can be relayed conservatively into a localized "
        "matrix halo, and the PF beta response is measured against source-on and source-off "
        "controls. It does not calibrate a GP solvus, GP release kinetics, or a moving-interface "
        "transport law. The source shell remains referenced to the selected seed radius; moving "
        "interface bands are observational diagnostics rather than a moving source geometry.",
        "",
    ])
    out.write_text("\n".join(text))
    terminal.write_text(
        f"baseline_matrix_complete={prerequisites['baseline_matrix_complete']}\n"
        f"dt_refinement_complete={prerequisites['dt_refinement_complete']}\n"
        f"T380_T400_refined_growth={both_growth}\n"
        f"source_off_tail_completed={source_off_completed}\n"
        f"mass_closure_gate={mass_gate}\n"
        "production_GP_thermodynamics_closed=false\n"
        f"final_status={status}\n"
    )
    print(f"final_status={status}")
    if status.startswith("FAIL") or status.startswith("INCOMPLETE"):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
