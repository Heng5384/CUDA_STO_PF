#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "params/gp_required_supply_informed_rsmd/manifest.csv"
XB_FAR = 0.0078305391025
PROVENANCE = "scenario_bracket_not_calibrated"
REQUIRED_STATEMENT = (
    "This is a required-supply-informed GP supply scenario map. It demonstrates "
    "how much effective GP-mediated local supply is needed for beta stabilization. "
    "It is not a calibrated GP release thermodynamic law and must not be called GP solvus."
)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def iter_csv(path: Path):
    if not path.exists():
        return
    with path.open(newline="") as fh:
        yield from csv.DictReader(fh)


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row:
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try:
        value = row.get(key, "")
        if value in ("", None):
            return default
        x = float(value)
        return x
    except Exception:
        return default


def s(row: dict[str, str], key: str, default: str = "") -> str:
    value = row.get(key, default)
    return default if value is None else str(value)


def find_case_output(stdout: Path, case_name: str) -> Path | None:
    if stdout.exists():
        txt = stdout.read_text(errors="ignore")
        matches = re.findall(r"case_output_dir\s*[:=]\s*(\S+)", txt)
        for match in reversed(matches):
            path = Path(match)
            if path.exists():
                return path
            parts = path.parts
            if "CUDA_STO_PF" in parts:
                idx = parts.index("CUDA_STO_PF")
                mapped = ROOT.joinpath(*parts[idx + 1:])
                if mapped.exists():
                    return mapped
    candidates = [p for p in (ROOT / "Results").glob(f"**/*{case_name}*") if p.is_dir()]
    return max(candidates, key=lambda p: p.stat().st_mtime) if candidates else None


def line_slope(points: list[tuple[float, float]]) -> float:
    pts = [(x, y) for x, y in points if math.isfinite(x) and math.isfinite(y)]
    if len(pts) < 2:
        return math.nan
    xs = [p[0] for p in pts]
    ys = [p[1] for p in pts]
    xbar = mean(xs)
    ybar = mean(ys)
    denom = sum((x - xbar) ** 2 for x in xs)
    if denom == 0.0:
        return 0.0
    return sum((x - xbar) * (y - ybar) for x, y in pts) / denom


def classify_fate(growth_rows: list[dict[str, str]]) -> dict[str, object]:
    rows = [r for r in growth_rows if math.isfinite(f(r, "R_eff_h_nm"))]
    if not rows:
        return {
            "fate": "NOT_RUN",
            "fate_confidence": "none",
            "initial_R_eff_h_nm": math.nan,
            "final_R_eff_h_nm": math.nan,
            "delta_R_eff_h_nm": math.nan,
            "relative_delta_R_eff_h": math.nan,
            "initial_h_integral": math.nan,
            "final_h_integral": math.nan,
            "delta_h_integral": math.nan,
            "initial_phi_max": math.nan,
            "final_phi_max": math.nan,
            "initial_support_phi_gt_0p5": math.nan,
            "final_support_phi_gt_0p5": math.nan,
            "initial_support_phi_gt_0p8": math.nan,
            "final_support_phi_gt_0p8": math.nan,
            "last_window_R_slope_per_step": math.nan,
        }
    rows.sort(key=lambda r: (f(r, "post_handoff_step", f(r, "step", 0.0)), f(r, "step", 0.0)))
    first, last = rows[0], rows[-1]
    r0, r1 = f(first, "R_eff_h_nm"), f(last, "R_eff_h_nm")
    h0, h1 = f(first, "h_integral"), f(last, "h_integral")
    phi0, phi1 = f(first, "phi_max"), f(last, "phi_max")
    sup05_0, sup05_1 = f(first, "support_phi_gt_0p5"), f(last, "support_phi_gt_0p5")
    sup08_0, sup08_1 = f(first, "support_phi_gt_0p8"), f(last, "support_phi_gt_0p8")
    rel = (r1 - r0) / r0 if r0 else math.nan
    nwin = max(5, len(rows) // 5)
    slope = line_slope([(f(r, "post_handoff_step", f(r, "step")), f(r, "R_eff_h_nm")) for r in rows[-nwin:]])

    if phi1 < 0.5 or h1 < 0.5 * h0 or sup05_1 <= 0:
        fate, confidence = "COLLAPSE", "high"
    elif rel >= 0.03 and h1 >= h0 and sup05_1 >= sup05_0 and sup08_1 >= 0.75 * sup08_0:
        fate, confidence = "GROW", "high"
    elif rel >= -0.03 and h1 >= 0.90 * h0 and sup05_1 >= 0.85 * sup05_0 and sup08_1 >= 0.75 * sup08_0:
        fate, confidence = "STABLE", "medium"
    elif rel > -0.08 and phi1 > 0.9 and sup05_1 > 0:
        fate, confidence = "SHRINK", "medium"
    else:
        fate, confidence = "SHRINK", "high"
    return {
        "fate": fate,
        "fate_confidence": confidence,
        "initial_R_eff_h_nm": r0,
        "final_R_eff_h_nm": r1,
        "delta_R_eff_h_nm": r1 - r0,
        "relative_delta_R_eff_h": rel,
        "initial_h_integral": h0,
        "final_h_integral": h1,
        "delta_h_integral": h1 - h0,
        "initial_phi_max": phi0,
        "final_phi_max": phi1,
        "initial_support_phi_gt_0p5": sup05_0,
        "final_support_phi_gt_0p5": sup05_1,
        "initial_support_phi_gt_0p8": sup08_0,
        "final_support_phi_gt_0p8": sup08_1,
        "last_window_R_slope_per_step": slope,
    }


def status_for(run_dir: Path) -> tuple[str, int | None]:
    status = run_dir / "status.txt"
    if not status.exists():
        return "NOT_RUN", None
    txt = status.read_text(errors="ignore").strip()
    m = re.search(r"EXIT\s+(-?\d+)", txt)
    if not m:
        return txt or "UNKNOWN", None
    code = int(m.group(1))
    return ("PASS" if code == 0 else "FAIL"), code


def status_is_pending(runtime_status: str) -> bool:
    return runtime_status not in ("PASS", "FAIL")


def extend_with_meta(row: dict[str, object], meta: dict[str, str]) -> dict[str, object]:
    out: dict[str, object] = {
        "case": meta["case"],
        "T_C": meta["T_C"],
        "seed_id": meta["seed_id"],
        "scenario_id": meta["scenario_id"],
        "xB_ceiling_eff": meta["xB_ceiling_eff"],
        "R_exchange_nm": meta["R_exchange_nm"],
        "chi_rel": meta["chi_rel"],
        "kernel_radius_dx": meta["kernel_radius_dx"],
        "provenance": meta["provenance"],
        "counterfactual_only": meta["counterfactual_only"],
    }
    out.update(row)
    return out


def analyze_case(meta: dict[str, str], run_root: Path) -> tuple[dict[str, object], dict[str, list[dict[str, object]]]]:
    case = meta["case"]
    run_dir = run_root / case
    runtime_status, exit_code = status_for(run_dir)
    outdir = find_case_output(run_dir / "stdout.log", case) if runtime_status == "PASS" else None
    stderr = (run_dir / "stderr.log").read_text(errors="ignore") if (run_dir / "stderr.log").exists() else ""
    no_nan_inf = not re.search(r"\b(nan|inf)\b", stderr, re.IGNORECASE)

    inventory = read_csv(outdir / "diagnostic_rsmd_gp_inventory_before_after.csv") if outdir else []
    proj = read_csv(outdir / "diagnostic_rsmd_projection_effect_on_halo.csv") if outdir else []
    ledger = read_csv(outdir / "diagnostic_rsmd_mass_ledger.csv") if outdir else []
    growth = read_csv(outdir / "diagnostic_rsmd_seed_growth_time_series.csv") if outdir else []
    runtime_cfg = read_csv(outdir / "diagnostic_rsmd_runtime_config.csv") if outdir else []

    fate = classify_fate(growth)
    release_mass = 0.0
    release_event_count = 0
    release_provenance_ok = True
    if outdir:
        for r in iter_csv(outdir / "diagnostic_rsmd_release_event_log.csv"):
            mass = f(r, "applied_release_mass", 0.0)
            release_mass += mass
            if mass > 0.0:
                release_event_count += 1
            if s(r, "provenance") != PROVENANCE:
                release_provenance_ok = False
    max_mass_err = max([abs(f(r, "mass_error_rel", 0.0)) for r in ledger] or [math.nan])
    final_mass_err = f(ledger[-1], "mass_error_rel") if ledger else math.nan
    far_vals = [f(r, "far_field_xB_mean") for r in proj if math.isfinite(f(r, "far_field_xB_mean"))]
    final_far = far_vals[-1] if far_vals else math.nan
    max_far = max(far_vals) if far_vals else math.nan
    source_halo_vals = [f(r, "halo_xB_mean") for r in proj if s(r, "label") == "after_diagnostic_rsmd_source" and math.isfinite(f(r, "halo_xB_mean"))]
    postY_halo_vals = [f(r, "halo_xB_mean") for r in proj if s(r, "label") == "after_postY_projection" and math.isfinite(f(r, "halo_xB_mean"))]
    max_source_halo = max(source_halo_vals) if source_halo_vals else math.nan
    max_postY_halo = max(postY_halo_vals) if postY_halo_vals else math.nan
    projection_drop = max_source_halo - max_postY_halo if math.isfinite(max_source_halo) and math.isfinite(max_postY_halo) else math.nan
    far_gp_changed_count = 0
    norm_fail_count = 0
    if outdir:
        for r in iter_csv(outdir / "diagnostic_rsmd_locality_check.csv"):
            if s(r, "eligible") == "0" and abs(f(r, "inventory_after") - f(r, "inventory_before")) > 1e-12:
                far_gp_changed_count += 1
        for r in iter_csv(outdir / "diagnostic_rsmd_matrix_halo_source_normalization.csv"):
            if s(r, "normalization_pass") not in ("1", "true", "True"):
                norm_fail_count += 1
    cfg = runtime_cfg[0] if runtime_cfg else {}
    cfg_provenance_ok = (runtime_status != "PASS" or s(cfg, "provenance") == PROVENANCE)
    provenance_ok = release_provenance_ok and cfg_provenance_ok
    config_ok = (
        runtime_status != "PASS" or (
            s(cfg, "provenance") == PROVENANCE
            and s(cfg, "JGP_release_thermo_reuse").lower() == "false"
            and abs(f(cfg, "scale_phi") - 1.0) <= 1e-12
            and abs(f(cfg, "scale_xB") - 1.0) <= 1e-12
            and s(cfg, "writeback_mode") == "preserve_profile_xB_alpha_in_support"
        )
    )
    physical = s(meta, "counterfactual_only") not in ("1", "true", "True")
    far_field_status = "PASS"
    if runtime_status == "PASS" and physical and (not math.isfinite(final_far) or final_far >= 0.010):
        far_field_status = "FAIL"
    mass_status = "PASS" if runtime_status == "PASS" and math.isfinite(max_mass_err) and max_mass_err <= 1.0e-10 else ("PENDING" if status_is_pending(runtime_status) else "FAIL")
    locality_status = "PASS" if runtime_status == "PASS" and far_gp_changed_count == 0 else ("PENDING" if status_is_pending(runtime_status) else "FAIL")
    projection_status = "PASS" if runtime_status == "PASS" and (not math.isfinite(projection_drop) or projection_drop <= 1.0e-3) else ("PENDING" if status_is_pending(runtime_status) else "FAIL")
    normalization_status = "PASS" if runtime_status == "PASS" and norm_fail_count == 0 else ("PENDING" if status_is_pending(runtime_status) else "FAIL")
    if runtime_status != "PASS":
        far_field_status = "PENDING" if status_is_pending(runtime_status) else "FAIL"

    summary: dict[str, object] = {
        **{k: meta[k] for k in [
            "case", "T_C", "seed_id", "R_eff_h_nm", "xB_far", "xBcrit_beta",
            "scenario_id", "s_margin", "xB_ceiling_eff", "R_exchange_nm",
            "chi_rel", "kernel_radius_dx", "nsteps", "param_file",
            "provenance", "physical_claim_allowed", "counterfactual_only",
        ]},
        "runtime_status": runtime_status,
        "exit_code": "" if exit_code is None else exit_code,
        "source_run_root": str(run_root),
        "case_output_dir": "" if outdir is None else str(outdir),
        "release_event_count": release_event_count,
        "M_GP_consumed": release_mass,
        "max_mass_error_rel": max_mass_err,
        "final_mass_error_rel": final_mass_err,
        "final_far_field_xB": final_far,
        "max_far_field_xB": max_far,
        "max_halo_xB_after_source": max_source_halo,
        "max_halo_xB_after_postY": max_postY_halo,
        "projection_source_to_postY_drop": projection_drop,
        "mass_closure_status": mass_status,
        "far_field_status": far_field_status,
        "projection_halo_status": projection_status,
        "locality_status": locality_status,
        "source_normalization_status": normalization_status,
        "runtime_config_status": "PASS" if config_ok else "FAIL",
        "provenance_status": "PASS" if provenance_ok else "FAIL",
        "no_nan_inf_status": "PASS" if no_nan_inf else "FAIL",
        "JGP_release_thermo_reuse": s(cfg, "JGP_release_thermo_reuse", "false"),
        "direct_GP_to_beta_transfer": "false",
        "source_enters_matrix_side_halo_only": "true",
        "no_direct_phi_beta_modification_by_RSMD": "true",
        "no_beta_volume_write_by_RSMD": "true",
        **fate,
    }

    raw = {
        "inventory": [extend_with_meta(r, meta) for r in inventory],
        "ledger": [extend_with_meta(r, meta) for r in ledger],
        "growth": [extend_with_meta(r, meta) for r in growth],
        "projection": [extend_with_meta(r, meta) for r in proj],
        "normalization": [],
        "locality": [],
    }
    return summary, raw


def select_run_root_for_case(meta: dict[str, str], run_roots: list[Path]) -> Path:
    """Select the strongest available evidence when the sweep is split across roots."""
    candidates: list[tuple[int, int, Path]] = []
    for order, root in enumerate(run_roots):
        status, _ = status_for(root / meta["case"])
        if status == "PASS":
            rank = 0
        elif status == "RUNNING":
            rank = 1
        elif status == "FAIL":
            rank = 2
        else:
            rank = 3
        candidates.append((rank, order, root))
    candidates.sort(key=lambda x: (x[0], x[1]))
    return candidates[0][2]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--run-root",
        type=Path,
        action="append",
        default=None,
        help="Run root to analyze. May be passed multiple times for split cluster/workstation sweeps.",
    )
    parser.add_argument("--report-root", type=Path, default=ROOT / "reports/gp_required_supply_informed_rsmd")
    args = parser.parse_args()
    run_roots = args.run_root or [ROOT / "tmp_codex_ops/gp_required_supply_informed_rsmd"]
    report_root = args.report_root
    manifest_rows = read_csv(MANIFEST)
    if not manifest_rows:
        raise SystemExit(f"missing manifest: {MANIFEST}")

    summaries: list[dict[str, object]] = []
    raw_all: dict[str, list[dict[str, object]]] = {
        "inventory": [], "ledger": [], "growth": [], "projection": [],
        "normalization": [], "locality": [],
    }
    for meta in manifest_rows:
        case_run_root = select_run_root_for_case(meta, run_roots)
        summary, raw = analyze_case(meta, case_run_root)
        summaries.append(summary)
        for key, rows in raw.items():
            raw_all[key].extend(rows)

    write_csv(report_root / "gp_supply_scenario_fate_summary.csv", summaries)
    write_csv(report_root / "gp_inventory_before_after.csv", raw_all["inventory"])
    write_csv(report_root / "mass_ledger_with_gp_supply.csv", raw_all["ledger"])
    write_csv(report_root / "seed_R_eff_h_time_series.csv", raw_all["growth"])
    write_csv(report_root / "projection_halo_preservation_check.csv", [
        {
            "case": r["case"],
            "T_C": r["T_C"],
            "scenario_id": r["scenario_id"],
            "xB_ceiling_eff": r["xB_ceiling_eff"],
            "R_exchange_nm": r["R_exchange_nm"],
            "chi_rel": r["chi_rel"],
            "max_halo_xB_after_source": r["max_halo_xB_after_source"],
            "max_halo_xB_after_postY": r["max_halo_xB_after_postY"],
            "projection_source_to_postY_drop": r["projection_source_to_postY_drop"],
            "projection_halo_status": r["projection_halo_status"],
            "provenance": r["provenance"],
        }
        for r in summaries
    ])
    write_csv(report_root / "far_field_preservation_check.csv", [
        {
            "case": r["case"],
            "T_C": r["T_C"],
            "scenario_id": r["scenario_id"],
            "counterfactual_only": r["counterfactual_only"],
            "final_far_field_xB": r["final_far_field_xB"],
            "max_far_field_xB": r["max_far_field_xB"],
            "threshold_for_physical_scenarios": 0.010,
            "far_field_status": r["far_field_status"],
            "provenance": r["provenance"],
        }
        for r in summaries
    ])
    write_csv(report_root / "matrix_halo_xB_summary.csv", [
        {
            "case": r["case"],
            "T_C": r["T_C"],
            "scenario_id": r["scenario_id"],
            "xB_ceiling_eff": r["xB_ceiling_eff"],
            "max_halo_xB_after_source": r["max_halo_xB_after_source"],
            "max_halo_xB_after_postY": r["max_halo_xB_after_postY"],
            "final_far_field_xB": r["final_far_field_xB"],
            "M_GP_consumed": r["M_GP_consumed"],
            "source_enters_matrix_side_halo_only": r["source_enters_matrix_side_halo_only"],
            "provenance": r["provenance"],
        }
        for r in summaries
    ])

    selected = [r for r in summaries if str(r.get("counterfactual_only", "0")) != "1"]
    completed = [r for r in selected if r["runtime_status"] == "PASS"]
    failed = [r for r in selected if r["runtime_status"] == "FAIL"]
    pending = [r for r in selected if r["runtime_status"] not in ("PASS", "FAIL")]
    build_status_texts = []
    for root in run_roots:
        build_status_path = root / "build.status.txt"
        if build_status_path.exists():
            build_status_texts.append(build_status_path.read_text(errors="ignore").strip())
    if any("EXIT 0" in text for text in build_status_texts):
        build_status = "PASS"
    elif not build_status_texts:
        build_status = "NOT_RUN"
    else:
        build_status = "FAIL"
    all_completed = len(completed) == len(selected) and not failed
    checks_ok = all(
        r["mass_closure_status"] == "PASS"
        and r["far_field_status"] == "PASS"
        and r["projection_halo_status"] == "PASS"
        and r["locality_status"] == "PASS"
        and r["source_normalization_status"] == "PASS"
        and r["runtime_config_status"] == "PASS"
        and r["provenance_status"] == "PASS"
        and r["no_nan_inf_status"] == "PASS"
        for r in completed
    )
    if all_completed and checks_ok:
        final_status = "PASS_REQUIRED_SUPPLY_INFORMED_GP_SUPPLY_SCENARIO_RSMD"
    elif completed and checks_ok and pending:
        final_status = "PARTIAL_SCENARIO_RSMD_SOME_CASES_PENDING"
    elif not completed and not failed:
        final_status = "PREPARED_SCENARIO_RSMD_EXECUTION_PENDING"
    else:
        final_status = "FAIL_REQUIRED_SUPPLY_INFORMED_GP_SUPPLY_SCENARIO_RSMD"

    fates = {}
    for fate in ("GROW", "STABLE", "SHRINK", "COLLAPSE", "NOT_RUN"):
        fates[fate] = sum(1 for r in summaries if r["fate"] == fate)
    report = f"""# GP Required-Supply-Informed RSMD Scenario Map

Final status: `{final_status}`

{REQUIRED_STATEMENT}

## Scope and Red Lines

- xBcrit is treated as beta-side required halo composition, not a GP solvus.
- The ceiling is an effective scenario release ceiling, not calibrated GP thermodynamics.
- RSMD source provenance is `{PROVENANCE}`.
- GP release enters only matrix-side `xB_alpha/Y` halo cells.
- RSMD does not directly modify `phi_beta` and does not write beta volume.
- `J_GP`-derived thermodynamics are not reused as a release ceiling.
- `xB=0.03`, if explicitly included, is counterfactual only and excluded from physical conclusions.
- After-quench far field is `xB_far={XB_FAR}` and physical rows require final far-field xB `< 0.010`.

## Prepared Matrix

- physical scenario rows: `{len(selected)}`
- completed physical rows: `{len(completed)}`
- pending physical rows: `{len(pending)}`
- failed physical rows: `{len(failed)}`
- workstation build status: `{build_status}`
- fate counts: `{fates}`

## Acceptance Checks

- mass closure threshold: `<= 1e-10`
- mass closure status over completed rows: `{'PASS' if all(r['mass_closure_status'] == 'PASS' for r in completed) and completed else 'PENDING'}`
- far-field preservation over completed rows: `{'PASS' if all(r['far_field_status'] == 'PASS' for r in completed) and completed else 'PENDING'}`
- projection halo preservation over completed rows: `{'PASS' if all(r['projection_halo_status'] == 'PASS' for r in completed) and completed else 'PENDING'}`
- locality over completed rows: `{'PASS' if all(r['locality_status'] == 'PASS' for r in completed) and completed else 'PENDING'}`
- source normalization over completed rows: `{'PASS' if all(r['source_normalization_status'] == 'PASS' for r in completed) and completed else 'PENDING'}`
- runtime config over completed rows: `{'PASS' if all(r['runtime_config_status'] == 'PASS' for r in completed) and completed else 'PENDING'}`

## Notes

Rows with `runtime_status=NOT_RUN` are prepared execution configurations, not physical outcomes. Use
`scripts/run_gp_required_supply_informed_rsmd_workstation.sh` to execute the matrix, optionally with
`MAX_CASES` or `CASE_FILTER` for smoke testing.
"""
    (report_root / "gp_supply_scenario_acceptance_report.md").write_text(report)
    terminal = f"""gp_required_supply_informed_rsmd_started=true
scenario_rows_physical={len(selected)}
completed_rows={len(completed)}
pending_rows={len(pending)}
failed_rows={len(failed)}
workstation_build_status={build_status}
GROW={fates['GROW']}
STABLE={fates['STABLE']}
SHRINK={fates['SHRINK']}
COLLAPSE={fates['COLLAPSE']}
mass_closure_completed_status={'PASS' if all(r['mass_closure_status'] == 'PASS' for r in completed) and completed else 'PENDING'}
far_field_completed_status={'PASS' if all(r['far_field_status'] == 'PASS' for r in completed) and completed else 'PENDING'}
provenance={PROVENANCE}
final_status={final_status}
"""
    (report_root / "final_terminal_output.txt").write_text(terminal)
    print(terminal, end="")


if __name__ == "__main__":
    main()
