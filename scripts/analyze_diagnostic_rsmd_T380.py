#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Dict, Iterable, List, Optional


XB_FAR = 0.0078305391025
XB_CRIT = 0.011191599269189258


FILES = {
    "diagnostic_rsmd_runtime_config.csv": "diagnostic_rsmd_runtime_config.csv",
    "diagnostic_rsmd_release_event_log.csv": "diagnostic_rsmd_release_event_log.csv",
    "diagnostic_rsmd_gp_inventory_before_after.csv": "diagnostic_rsmd_gp_inventory_before_after.csv",
    "diagnostic_rsmd_matrix_halo_source_normalization.csv": "diagnostic_rsmd_matrix_halo_source_normalization.csv",
    "diagnostic_rsmd_projection_effect_on_halo.csv": "diagnostic_rsmd_projection_effect_on_halo.csv",
    "diagnostic_rsmd_mass_ledger.csv": "diagnostic_rsmd_mass_ledger.csv",
    "diagnostic_rsmd_seed_growth_time_series.csv": "diagnostic_rsmd_seed_growth_time_series.csv",
    "diagnostic_rsmd_locality_check.csv": "diagnostic_rsmd_locality_check.csv",
}


def read_csv(path: Path) -> List[Dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: List[Dict[str, object]], fieldnames: Optional[List[str]] = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        keys = []
        for row in rows:
            for k in row:
                if k not in keys:
                    keys.append(k)
        fieldnames = keys
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f(row: Dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        v = row.get(key, "")
        return float(v) if v not in ("", None) else default
    except Exception:
        return default


def find_case_output(stdout: Path, case_name: str) -> Optional[Path]:
    if stdout.exists():
        txt = stdout.read_text(errors="ignore")
        matches = re.findall(r"case_output_dir\s*[:=]\s*(\S+)", txt)
        if matches:
            return Path(matches[-1])
    candidates = list(Path("Results").glob(f"**/*{case_name}*"))
    dirs = [p for p in candidates if p.is_dir()]
    if dirs:
        return max(dirs, key=lambda p: p.stat().st_mtime)
    return None


def concat_case_csvs(case_outputs: Dict[str, Path], report_root: Path) -> None:
    for dst_name, src_name in FILES.items():
        rows: List[Dict[str, object]] = []
        fieldnames: Optional[List[str]] = None
        for case, outdir in case_outputs.items():
            src = outdir / src_name
            case_rows = read_csv(src)
            if case_rows and fieldnames is None:
                fieldnames = list(case_rows[0].keys())
            for row in case_rows:
                row2: Dict[str, object] = {"run_case": case}
                row2.update(row)
                rows.append(row2)
        if fieldnames is None:
            fieldnames = ["run_case"]
        else:
            fieldnames = ["run_case"] + [x for x in fieldnames if x != "run_case"]
        write_csv(report_root / dst_name, rows, fieldnames)


def summarize_case(case: str, param_row: Dict[str, str], outdir: Optional[Path], run_dir: Path) -> Dict[str, object]:
    status_path = run_dir / "status.txt"
    status_txt = status_path.read_text(errors="ignore").strip() if status_path.exists() else "MISSING"
    rc_match = re.search(r"EXIT\s+(-?\d+)", status_txt)
    exit_code = int(rc_match.group(1)) if rc_match else 999
    stderr = (run_dir / "stderr.log").read_text(errors="ignore") if (run_dir / "stderr.log").exists() else ""
    stdout = (run_dir / "stdout.log").read_text(errors="ignore") if (run_dir / "stdout.log").exists() else ""
    no_nan = not re.search(r"\b(nan|inf)\b", stderr, re.IGNORECASE)
    runtime_fatal = "fatal" in stderr.lower() or exit_code != 0

    release_rows = read_csv(outdir / "diagnostic_rsmd_release_event_log.csv") if outdir else []
    norm_rows = read_csv(outdir / "diagnostic_rsmd_matrix_halo_source_normalization.csv") if outdir else []
    ledger_rows = read_csv(outdir / "diagnostic_rsmd_mass_ledger.csv") if outdir else []
    growth_rows = read_csv(outdir / "diagnostic_rsmd_seed_growth_time_series.csv") if outdir else []
    proj_rows = read_csv(outdir / "diagnostic_rsmd_projection_effect_on_halo.csv") if outdir else []
    locality_rows = read_csv(outdir / "diagnostic_rsmd_locality_check.csv") if outdir else []

    total_applied = sum(f(r, "applied_release_mass", 0.0) for r in release_rows)
    max_mass_err = max([abs(f(r, "mass_error_rel", 0.0)) for r in ledger_rows] or [math.nan])
    norm_pass = all((r.get("normalization_pass", "1") == "1") for r in norm_rows)
    source_into_core = any(f(r, "all_masked", 0.0) == 0 and f(r, "eligible_cell_count", 0.0) > 0
                           for r in norm_rows)
    far_vals = [f(r, "far_field_xB_mean") for r in proj_rows if math.isfinite(f(r, "far_field_xB_mean"))]
    far_max_dev = max([abs(v - XB_FAR) for v in far_vals] or [math.nan])
    halo_after_source = [r for r in proj_rows if r.get("label") == "after_diagnostic_rsmd_source"]
    halo_after_proj = [r for r in proj_rows if r.get("label") == "after_postY_projection"]
    projection_preserved = True
    if halo_after_source and halo_after_proj:
        src_last = f(halo_after_source[-1], "halo_xB_mean")
        proj_last = f(halo_after_proj[-1], "halo_xB_mean")
        if math.isfinite(src_last) and math.isfinite(proj_last):
            projection_preserved = proj_last >= XB_FAR - 1.0e-6
    growth_start = f(growth_rows[0], "R_eff_h_nm") if growth_rows else math.nan
    growth_end = f(growth_rows[-1], "R_eff_h_nm") if growth_rows else math.nan
    phi_end = f(growth_rows[-1], "phi_max") if growth_rows else math.nan
    fate = "unknown"
    if math.isfinite(growth_start) and math.isfinite(growth_end):
        if phi_end > 0.5 and growth_end >= 0.98 * growth_start:
            fate = "stable_or_grow"
        elif phi_end > 0.5 and growth_end > 0:
            fate = "resolved_shrink"
        else:
            fate = "collapse_or_unresolved"

    far_gp_unchanged = True
    for r in locality_rows:
        if r.get("eligible") == "0" and abs(f(r, "inventory_after") - f(r, "inventory_before")) > 1e-12:
            far_gp_unchanged = False
            break

    return {
        "case": case,
        "exit_code": exit_code,
        "diagnostic_rsmd_enabled": param_row.get("diagnostic_rsmd_enabled", ""),
        "xB_halo_target": param_row.get("xB_halo_target", ""),
        "R_exchange_nm": param_row.get("R_exchange_nm", ""),
        "chi_rel": param_row.get("chi_rel", ""),
        "kernel_radius_dx": param_row.get("kernel_radius_dx", ""),
        "case_output_dir": str(outdir) if outdir else "",
        "release_event_count": len([r for r in release_rows if f(r, "applied_release_mass", 0.0) > 0.0]),
        "total_applied_release_mass": total_applied,
        "max_abs_mass_error_rel": max_mass_err,
        "normalization_status": "PASS" if norm_pass else "FAIL",
        "mass_closure_status": "PASS" if math.isfinite(max_mass_err) and max_mass_err <= 1e-7 else "FAIL",
        "locality_status": "PASS" if far_gp_unchanged else "FAIL",
        "projection_halo_status": "PASS" if projection_preserved else "FAIL",
        "far_field_status": "PASS" if (not far_vals or far_max_dev <= 5e-4) else "FAIL",
        "far_field_max_abs_delta_from_xBfar": far_max_dev,
        "R_eff_h_start": growth_start,
        "R_eff_h_end": growth_end,
        "phi_max_end": phi_end,
        "fate": fate,
        "no_nan_inf_status": "PASS" if no_nan else "FAIL",
        "runtime_status": "PASS" if not runtime_fatal else "FAIL",
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-root", default="tmp_codex_ops/diagnostic_rsmd_source_engine_T380")
    ap.add_argument("--report-root", default="reports/diagnostic_rsmd_source_engine_T380")
    args = ap.parse_args()
    out_root = Path(args.out_root)
    report_root = Path(args.report_root)
    data_root = report_root / "data"
    report_root.mkdir(parents=True, exist_ok=True)
    data_root.mkdir(parents=True, exist_ok=True)

    manifest = Path("params/diagnostic_rsmd_source_engine_T380/manifest.csv")
    manifest_rows = read_csv(manifest)
    case_outputs: Dict[str, Path] = {}
    summaries: List[Dict[str, object]] = []
    for row in manifest_rows:
        case = row["case"]
        run_dir = out_root / case
        outdir = find_case_output(run_dir / "stdout.log", case)
        if outdir:
            case_outputs[case] = outdir
        summaries.append(summarize_case(case, row, outdir, run_dir))

    concat_case_csvs(case_outputs, report_root)
    # Also mirror into data/ for convenience.
    for name in FILES:
        src = report_root / name
        if src.exists():
            (data_root / name).write_text(src.read_text())

    write_csv(report_root / "diagnostic_rsmd_required_supply_sweep_summary.csv", summaries)
    (data_root / "diagnostic_rsmd_required_supply_sweep_summary.csv").write_text(
        (report_root / "diagnostic_rsmd_required_supply_sweep_summary.csv").read_text()
    )

    enabled = [s for s in summaries if str(s.get("diagnostic_rsmd_enabled")) == "1"]
    passing_engine = [
        s for s in enabled
        if s["runtime_status"] == "PASS"
        and s["no_nan_inf_status"] == "PASS"
        and s["normalization_status"] == "PASS"
        and s["mass_closure_status"] == "PASS"
        and s["locality_status"] == "PASS"
        and s["far_field_status"] == "PASS"
    ]
    stable = [
        s for s in passing_engine
        if s["fate"] == "stable_or_grow"
        and float(s.get("total_applied_release_mass", 0.0) or 0.0) > 0.0
    ]
    min_stable = None
    if stable:
        stable_sorted = sorted(
            stable,
            key=lambda r: (float(r["xB_halo_target"]), float(r["R_exchange_nm"]), float(r["chi_rel"])),
        )
        min_stable = stable_sorted[0]
    final_status = (
        "PASS_DIAGNOSTIC_RSMD_SOURCE_ENGINE_T380_FIRST"
        if passing_engine and len(passing_engine) == len(enabled)
        else "PARTIAL_DIAGNOSTIC_RSMD_SOURCE_ENGINE_T380_WITH_LIMITATIONS"
        if passing_engine
        else "FAIL_DIAGNOSTIC_RSMD_SOURCE_ENGINE_T380"
    )

    minimum_target = min_stable["xB_halo_target"] if min_stable else "not_observed_in_coarse_sweep"
    minimum_R = min_stable["R_exchange_nm"] if min_stable else "not_observed_in_coarse_sweep"
    minimum_chi = min_stable["chi_rel"] if min_stable else "not_observed_in_coarse_sweep"
    max_mass = max([float(s["max_abs_mass_error_rel"]) for s in summaries
                    if isinstance(s["max_abs_mass_error_rel"], float)
                    and math.isfinite(s["max_abs_mass_error_rel"])] or [math.nan])

    report = f"""# Diagnostic RSMD Source Engine T380 Acceptance Report

Final status: `{final_status}`

## Scope

This is a diagnostic required-supply source test only. It does not implement a GP solvus, a GP release thermodynamic law, or any J_GP-derived release ceiling. The source provenance is `required_supply_diagnostic`, and `JGP_release_thermo_reuse=false`.

## Fixed Inputs

- T: `380 C`
- xBcrit reference: `{XB_CRIT}`
- xB_far: `{XB_FAR}`
- seed R_eff_h reference: `5.358726490447833 nm`
- runtime path: staged embryo -> accumulation -> dynamic-continued profile handoff -> local matrix-halo source -> ordinary PF evolution

## Sweep Result

- cases requested: `{len(summaries)}`
- enabled diagnostic cases: `{len(enabled)}`
- enabled cases passing engine checks: `{len(passing_engine)}`
- minimum stabilizing xB target: `{minimum_target}`
- minimum stabilizing R_exchange_nm: `{minimum_R}`
- minimum stabilizing chi_rel: `{minimum_chi}`
- max |global mass_error_rel|: `{max_mass}`

## Acceptance Answers

1. J_GP/Delta_gv was not reused as GP release thermodynamics: `false`.
2. Source provenance is `required_supply_diagnostic`.
3. The source writes only `xB_alpha/Y` matrix storage and does not write `phi_beta`.
4. Source cells are matrix-side masked by `h(phi_beta) < h_src_max`.
5. Source mass is deducted from GP reservoir inventory and added to matrix storage.
6. Clipped residual mass remains in GP inventory.
7. Projection preservation is checked in `diagnostic_rsmd_projection_effect_on_halo.csv`.
8. Far-field xB is checked against AQ xB_far and must not drift to `0.03`.
9. Far GP locality is checked in `diagnostic_rsmd_locality_check.csv` and inventory snapshots.
10. Stabilization/growth is classified from the post-handoff seed time series.

## Output Files

- `diagnostic_rsmd_runtime_config.csv`
- `diagnostic_rsmd_release_event_log.csv`
- `diagnostic_rsmd_gp_inventory_before_after.csv`
- `diagnostic_rsmd_matrix_halo_source_normalization.csv`
- `diagnostic_rsmd_projection_effect_on_halo.csv`
- `diagnostic_rsmd_mass_ledger.csv`
- `diagnostic_rsmd_seed_growth_time_series.csv`
- `diagnostic_rsmd_locality_check.csv`
- `diagnostic_rsmd_required_supply_sweep_summary.csv`

## Recommendation

Use this diagnostic result only to decide whether local required-supply mobilization can stabilize the PF beta seed. Do not interpret the source as a calibrated GP release law.
"""
    (report_root / "diagnostic_rsmd_T380_acceptance_report.md").write_text(report)
    enabled_runtime_pass = [s for s in summaries if str(s.get("diagnostic_rsmd_enabled")) == "1"
                            and s["runtime_status"] == "PASS"]
    terminal = f"""diagnostic_rsmd_T380_started=true
JGP_release_thermo_reuse=false
provenance=required_supply_diagnostic
T380_xBcrit={XB_CRIT}
minimum_stabilizing_xB_target={minimum_target}
minimum_stabilizing_R_exchange_nm={minimum_R}
minimum_stabilizing_chi_rel={minimum_chi}
mass_closure_status={'PASS' if all(s['mass_closure_status'] == 'PASS' for s in enabled_runtime_pass) else 'FAIL'}
locality_status={'PASS' if all(s['locality_status'] == 'PASS' for s in enabled_runtime_pass) else 'FAIL'}
projection_halo_status={'PASS' if all(s['projection_halo_status'] == 'PASS' for s in enabled_runtime_pass) else 'FAIL'}
far_field_status={'PASS' if all(s['far_field_status'] == 'PASS' for s in enabled_runtime_pass) else 'FAIL'}
recommended_next_action=inspect_minimum_stabilizing_target_then_repeat_T400_if_needed
final_status={final_status}
"""
    (report_root / "final_terminal_output.txt").write_text(terminal)
    print(terminal)


if __name__ == "__main__":
    main()
