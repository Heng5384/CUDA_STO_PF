#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
LIBRARY_ENTRY = "nlib_dc_T400_xB003"
SEED_R_EFF_H_NM = 4.735880528792827
MASS_TOL_REL = 1.0e-7


def f(v: Any, default: float = math.nan) -> float:
    try:
        if v is None:
            return default
        if isinstance(v, str) and not v.strip():
            return default
        x = float(v)
        return x if math.isfinite(x) else default
    except Exception:
        return default


def read_text(path: Path) -> str:
    return path.read_text(errors="ignore") if path.exists() else ""


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
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
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def parse_params(path: Path) -> dict[str, str]:
    vals: dict[str, str] = {}
    for line in read_text(path).splitlines():
        s = line.strip()
        if not s or s.startswith("#") or "=" not in s:
            continue
        key, value = s.split("=", 1)
        vals[key.strip()] = value.strip()
    return vals


def case_output_dir(control_dir: Path) -> Path:
    text = read_text(control_dir / "stdout.log")
    m = re.search(r"case_output_dir\s*:\s*(\S+)", text)
    if not m:
        return control_dir
    p = Path(m.group(1))
    return p if p.is_absolute() else ROOT / p


def r_eff_from_h(h_sum: float, dx_nm: float = 1.0) -> float:
    if not (h_sum > 0.0):
        return math.nan
    return (3.0 * h_sum * dx_nm**3 / (4.0 * math.pi)) ** (1.0 / 3.0)


def slope_last(rows: list[dict[str, Any]], window_code_time: float = 6.0) -> float:
    pts = [(f(r.get("post_handoff_time_code")), f(r.get("R_eff_h_nm"))) for r in rows]
    pts = [(x, y) for x, y in pts if math.isfinite(x) and math.isfinite(y)]
    if len(pts) < 2:
        return math.nan
    max_x = max(x for x, _ in pts)
    pts = [(x, y) for x, y in pts if x >= max_x - window_code_time]
    if len(pts) < 2:
        return math.nan
    n = len(pts)
    sx = sum(x for x, _ in pts)
    sy = sum(y for _, y in pts)
    sxx = sum(x * x for x, _ in pts)
    sxy = sum(x * y for x, y in pts)
    den = n * sxx - sx * sx
    return (n * sxy - sx * sy) / den if abs(den) > 0 else math.nan


def detect_nan_inf(*paths: Path) -> bool:
    pat = re.compile(r"(nan|inf|overflow|nonfinite)", re.IGNORECASE)
    for path in paths:
        txt = read_text(path)
        if "NaN_Inf=1" in txt:
            return True
        if pat.search(txt):
            placeholders = re.findall(r":\s*nan\b", txt, flags=re.IGNORECASE)
            if len(placeholders) == 0 or "nonfinite" in txt.lower():
                return True
    return False


def classify(initial_r: float, final_r: float, initial_h: float, final_h: float,
             phi_max: float, slope: float, mass_ok: bool) -> tuple[str, str, str]:
    if not mass_ok:
        return "NUMERICALLY_UNRELIABLE", "high", "mass closure exceeds refinement tolerance"
    threshold = max(0.10, 0.02 * initial_r)
    slope_threshold = threshold / 20.0
    delta_r = final_r - initial_r
    dh = (final_h - initial_h) / initial_h if initial_h > 0 else math.nan
    if phi_max < 0.1 or final_h < max(1e-12, 1e-4 * initial_h):
        return "COLLAPSE", "high", "phi support disappears or h integral is nearly zero"
    if delta_r > 0.5 * initial_r or dh > 0.5:
        return "OVERSHOOT", "high", "R_eff_h or h_integral increases by more than 50%"
    if delta_r > threshold and (not math.isfinite(slope) or slope > -slope_threshold):
        return "GROW", "medium_high", "R_eff_h increase exceeds threshold with nonnegative final trend"
    if abs(delta_r) <= threshold and (not math.isfinite(slope) or abs(slope) <= slope_threshold):
        return "STABLE", "medium", "R_eff_h change and final slope are within thresholds"
    if delta_r < -threshold and (not math.isfinite(slope) or slope < slope_threshold):
        return "SHRINK", "medium_high", "R_eff_h decrease exceeds threshold with nonpositive final trend"
    return "STABLE", "low", "mixed metrics near threshold"


def parse_case(control_dir: Path) -> dict[str, Any]:
    out_dir = case_output_dir(control_dir)
    params = parse_params(out_dir / "pf_input.params")
    if not params:
        params = parse_params(control_dir / "pf_input.params")
    probes = read_csv(out_dir / "handoff_profile_probes.csv")
    source = read_csv(out_dir / "resolved_seed_source_diagnostics.csv")
    tx = read_csv(out_dir / "resolved_seed_handoff_transactions.csv")
    status = read_text(control_dir / "status.txt").strip()
    dt = f(params.get("dt"), 0.02)
    dx = f(params.get("dx"), 1.0)
    xB = f(params.get("ic_23d_xB_out"))
    step_rows = [r for r in probes if r.get("probe_label") == "PROBE_STEP_END"]
    handoff_rows = [
        r for r in step_rows
        if f(r.get("beta_phi_sum")) > 0.0 and int(f(r.get("resolved_seed_count"), 0)) > 0
    ]
    first = handoff_rows[0] if handoff_rows else None
    last = handoff_rows[-1] if handoff_rows else None
    handoff_step = int(f(first.get("step"))) if first else math.nan
    ts_rows: list[dict[str, Any]] = []
    for r in handoff_rows:
        step = int(f(r.get("step")))
        h_sum = f(r.get("beta_phi_sum"))
        ts_rows.append({
            "run_id": control_dir.name,
            "xB_uniform": xB,
            "dt_code": dt,
            "step": step,
            "time_code": step * dt,
            "post_handoff_step": step - handoff_step,
            "post_handoff_time_code": (step - handoff_step) * dt,
            "h_integral": h_sum,
            "R_eff_h_nm": r_eff_from_h(h_sum, dx),
            "beta_phi_max": f(r.get("beta_phi_max")),
            "xB_alpha_mean": f(r.get("xB_alpha_mean")),
            "xB_alpha_min": f(r.get("xB_alpha_min")),
            "xB_alpha_max": f(r.get("xB_alpha_max")),
            "Y_min": f(r.get("Y_min")),
            "Y_max": f(r.get("Y_max")),
            "M_matrix": f(r.get("M_matrix")),
            "M_beta": f(r.get("M_beta")),
            "M_staged": f(r.get("M_staged")),
            "mass_error_rel": f(r.get("mass_error_rel")),
        })
    closure_probe_rows = [r for r in probes if r.get("probe_label") == "PROBE_STEP_END"]
    mass_vals = [abs(f(r.get("mass_error_rel"))) for r in closure_probe_rows if math.isfinite(f(r.get("mass_error_rel")))]
    max_mass = max(mass_vals) if mass_vals else math.nan
    xB_mins = [f(r.get("xB_alpha_min")) for r in probes if math.isfinite(f(r.get("xB_alpha_min")))]
    xB_maxs = [f(r.get("xB_alpha_max")) for r in probes if math.isfinite(f(r.get("xB_alpha_max")))]
    y_mins = [f(r.get("Y_min")) for r in probes if math.isfinite(f(r.get("Y_min")))]
    y_maxs = [f(r.get("Y_max")) for r in probes if math.isfinite(f(r.get("Y_max")))]
    initial_h = f(first.get("beta_phi_sum")) if first else math.nan
    final_h = f(last.get("beta_phi_sum")) if last else math.nan
    initial_r = r_eff_from_h(initial_h, dx)
    final_r = r_eff_from_h(final_h, dx)
    slope = slope_last(ts_rows)
    mass_ok = math.isfinite(max_mass) and max_mass <= MASS_TOL_REL
    fate, confidence, reason = classify(initial_r, final_r, initial_h, final_h, f(last.get("beta_phi_max")) if last else math.nan, slope, mass_ok) if first else (
        "NO_HANDOFF", "high", "resolved handoff did not occur"
    )
    source_last = source[-1] if source else {}
    config_ok = (
        abs(dx - 1.0) < 1e-12 and
        abs(f(params.get("lambda_sm_m")) - 6.0e-10) < 1e-15 and
        abs(f(params.get("scheduled_nuc_scale_interface_width"), 1.0) - 1.0) < 1e-12 and
        abs(f(params.get("scheduled_nuc_scale_xB_profile_width"), 1.0) - 1.0) < 1e-12 and
        params.get("resolved_handoff_xB_write_mode") == "preserve_profile_xB_alpha_in_support" and
        params.get("gp_initial_population_enabled") in {"0", "0.0"} and
        params.get("gp_literature_model_enabled") in {"0", "0.0"} and
        source_last.get("library_entry_id", LIBRARY_ENTRY) == LIBRARY_ENTRY and
        source_last.get("analytic_fallback_used", "0") in {"0", "false", "False", ""}
    )
    return {
        "run_id": control_dir.name,
        "control_dir": str(control_dir),
        "run_dir": str(out_dir),
        "T_C": 400,
        "xB_uniform": xB,
        "dt_code": dt,
        "nsteps": f(params.get("nsteps"), ""),
        "run_completed": status.endswith("0") and bool(first),
        "no_nan_inf": (not detect_nan_inf(control_dir / "stdout.log", control_dir / "stderr.log")) and status.endswith("0"),
        "config_ok": config_ok,
        "library_entry": source_last.get("library_entry_id", ""),
        "handoff_step": handoff_step,
        "post_handoff_steps": int(f(last.get("step")) - handoff_step) if first and last else "",
        "post_handoff_time_code": ((f(last.get("step")) - handoff_step) * dt) if first and last else "",
        "initial_R_eff_h_nm": initial_r,
        "final_R_eff_h_nm": final_r,
        "delta_R_eff_h_nm": final_r - initial_r if math.isfinite(final_r) and math.isfinite(initial_r) else "",
        "initial_h_integral": initial_h,
        "final_h_integral": final_h,
        "delta_h_integral_frac": (final_h - initial_h) / initial_h if initial_h > 0 else "",
        "final_phi_max": f(last.get("beta_phi_max")) if last else "",
        "slope_last_window_nm_per_code_time": slope,
        "mass_closure_max_rel": max_mass,
        "mass_closure_status": "PASS" if mass_ok else "FAIL",
        "first_mass_failure_step": next((r.get("step") for r in closure_probe_rows if abs(f(r.get("mass_error_rel"))) > MASS_TOL_REL), ""),
        "xB_min_global": min(xB_mins) if xB_mins else "",
        "xB_max_global": max(xB_maxs) if xB_maxs else "",
        "Y_min_global": min(y_mins) if y_mins else "",
        "Y_max_global": max(y_maxs) if y_maxs else "",
        "xB_clipping_or_jump_detected": bool((xB_mins and min(xB_mins) <= 1.0e-8) or (xB_maxs and max(xB_maxs) >= 0.999)),
        "final_fate": fate,
        "fate_confidence": confidence,
        "classification_reason": reason,
        "source_last": source_last,
        "tx_last": tx[-1] if tx else {},
        "series": ts_rows,
    }


def discover_cases(run_root: Path) -> list[Path]:
    if not run_root.exists():
        return []
    return sorted(p for p in run_root.iterdir() if p.is_dir() and p.name.startswith("T400_"))


def summarize_by_xb(cases: list[dict[str, Any]]) -> tuple[list[dict[str, Any]], str, str]:
    rows: list[dict[str, Any]] = []
    status = "T400_HIGH_XB_NUMERICALLY_UNRESOLVED"
    bracket = "unresolved"
    for xb in sorted({round(f(c["xB_uniform"]), 12) for c in cases if math.isfinite(f(c["xB_uniform"]))}):
        subset = sorted([c for c in cases if abs(f(c["xB_uniform"]) - xb) < 1e-12], key=lambda c: f(c["dt_code"]))
        reliable = [c for c in subset if c["run_completed"] and c["no_nan_inf"] and c["config_ok"] and c["mass_closure_status"] == "PASS"]
        best = reliable[0] if reliable else subset[0] if subset else {}
        rows.append({
            "T_C": 400,
            "xB_uniform": xb,
            "reliable_dt_count": len(reliable),
            "tested_dt_codes": ";".join(str(c["dt_code"]) for c in subset),
            "selected_dt_code": best.get("dt_code", ""),
            "selected_mass_closure_max_rel": best.get("mass_closure_max_rel", ""),
            "selected_fate": best.get("final_fate", ""),
            "selected_final_R_eff_h_nm": best.get("final_R_eff_h_nm", ""),
            "selected_delta_R_eff_h_nm": best.get("delta_R_eff_h_nm", ""),
            "selected_xB_clipping_or_jump_detected": best.get("xB_clipping_or_jump_detected", ""),
            "status": "RELIABLE" if reliable else "NUMERICALLY_UNRESOLVED",
        })
    by_xb = {round(f(r["xB_uniform"]), 12): r for r in rows}
    r20 = by_xb.get(round(0.020, 12), {})
    r24 = by_xb.get(round(0.024, 12), {})
    if r20.get("status") == "RELIABLE" and r24.get("status") == "RELIABLE":
        f20 = str(r20.get("selected_fate"))
        f24 = str(r24.get("selected_fate"))
        if f20 in {"STABLE", "SHRINK", "COLLAPSE"} and f24 in {"GROW", "OVERSHOOT", "STABLE"}:
            status = "T400_XBCRIT_BRACKET_RETAINED"
            bracket = "[0.020,0.024]"
    return rows, status, bracket


def build_report(cases: list[dict[str, Any]], selected_rows: list[dict[str, Any]], status: str, bracket: str) -> str:
    lines = [
        "# T400 High-xB Mass Closure Refinement",
        "",
        f"final_refinement_status = `{status}`",
        f"T400_xBcrit_bracket_after_refinement = `{bracket}`",
        "",
        "## Scope",
        "",
        "This refinement reruns the T400 transition/high-xB range with the stable unscaled dynamic-continued seed path. AQ GP inventory, J_GP, RSMD source, GP release, scheduled scaling, and 4-grid runtime mode are disabled.",
        "",
        "## Runtime Matrix",
        "",
        "| xB | dt codes tested | reliable dt count | selected dt | selected fate | selected mass rel | status |",
        "|---:|---|---:|---:|---|---:|---|",
    ]
    for r in selected_rows:
        lines.append(
            f"| {r['xB_uniform']} | {r['tested_dt_codes']} | {r['reliable_dt_count']} | {r['selected_dt_code']} | "
            f"{r['selected_fate']} | {r['selected_mass_closure_max_rel']} | {r['status']} |"
        )
    lines.extend([
        "",
        "## Drift Attribution",
        "",
        "Rows with `mass_closure_status=FAIL` are not used as physical growth/shrink evidence. They are reported as numerical storage/projection/Y-rebuild failures when global ledger residual exceeds `1e-7`, especially if `xB_alpha` clips or jumps to `[0,1]`.",
        "",
        "## Decision",
        "",
    ])
    if status == "T400_XBCRIT_BRACKET_RETAINED":
        lines.append("The existing `T400 xBcrit bracket = [0.020, 0.024]` is retained for supply-map preparation, while all numerically failed high-xB rows remain flagged as caveats.")
    else:
        lines.append("The T400 high-xB transition is marked `T400_HIGH_XB_NUMERICALLY_UNRESOLVED`; do not force a physical interpretation from failed rows.")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-root", default="tmp_codex_ops/t400_xbcrit_high_range_refinement")
    parser.add_argument("--report-root", default="reports/t400_xbcrit_high_range_refinement")
    args = parser.parse_args()
    run_root = Path(args.run_root)
    report_root = Path(args.report_root)
    cases = [parse_case(p) for p in discover_cases(run_root)]
    report_root.mkdir(parents=True, exist_ok=True)
    run_rows = []
    ts_rows = []
    proj_rows = []
    for c in cases:
        run_rows.append({
            "run_id": c["run_id"],
            "T_C": c["T_C"],
            "xB_uniform": c["xB_uniform"],
            "dt_code": c["dt_code"],
            "run_completed": c["run_completed"],
            "no_nan_inf": c["no_nan_inf"],
            "config_ok": c["config_ok"],
            "handoff_step": c["handoff_step"],
            "post_handoff_steps": c["post_handoff_steps"],
            "post_handoff_time_code": c["post_handoff_time_code"],
            "initial_R_eff_h_nm": c["initial_R_eff_h_nm"],
            "final_R_eff_h_nm": c["final_R_eff_h_nm"],
            "delta_R_eff_h_nm": c["delta_R_eff_h_nm"],
            "initial_h_integral": c["initial_h_integral"],
            "final_h_integral": c["final_h_integral"],
            "delta_h_integral_frac": c["delta_h_integral_frac"],
            "final_phi_max": c["final_phi_max"],
            "mass_closure_max_rel": c["mass_closure_max_rel"],
            "mass_closure_status": c["mass_closure_status"],
            "xB_min_global": c["xB_min_global"],
            "xB_max_global": c["xB_max_global"],
            "xB_clipping_or_jump_detected": c["xB_clipping_or_jump_detected"],
            "final_fate": c["final_fate"],
            "fate_confidence": c["fate_confidence"],
            "comments": c["classification_reason"],
        })
        ts_rows.extend(c["series"])
        proj_rows.append({
            "run_id": c["run_id"],
            "xB_uniform": c["xB_uniform"],
            "dt_code": c["dt_code"],
            "mass_closure_max_rel": c["mass_closure_max_rel"],
            "first_mass_failure_step": c["first_mass_failure_step"],
            "xB_min_global": c["xB_min_global"],
            "xB_max_global": c["xB_max_global"],
            "Y_min_global": c["Y_min_global"],
            "Y_max_global": c["Y_max_global"],
            "xB_clipping_or_jump_detected": c["xB_clipping_or_jump_detected"],
            "projection_correction_csv_available": False,
            "storage_reconstruction_failure_suspected": c["xB_clipping_or_jump_detected"] or f(c["mass_closure_max_rel"], 0.0) > MASS_TOL_REL,
            "notes": "diagnostic from handoff_profile_probes; no dedicated per-projection correction CSV found",
        })
    selected_rows, status, bracket = summarize_by_xb(cases)
    write_csv(report_root / "t400_high_xb_mass_closure_refinement.csv", run_rows)
    write_csv(report_root / "t400_high_xb_time_series.csv", ts_rows)
    write_csv(report_root / "t400_high_xb_projection_audit.csv", proj_rows)
    write_csv(report_root / "t400_high_xb_final_classification.csv", selected_rows)
    (report_root / "t400_high_xb_refinement_report.md").write_text(build_report(cases, selected_rows, status, bracket))
    terminal = [
        "t400_xbcrit_high_range_refinement_analyzed",
        f"runs_parsed={len(cases)}",
        f"reliable_runs={sum(1 for c in cases if c['mass_closure_status'] == 'PASS' and c['run_completed'] and c['config_ok'])}",
        f"max_mass_closure_rel={max([f(c['mass_closure_max_rel'], 0.0) for c in cases], default=0.0)}",
        f"T400_refinement_status={status}",
        f"T400_xBcrit_bracket_after_refinement={bracket}",
    ]
    (report_root / "final_terminal_output.txt").write_text("\n".join(terminal) + "\n")
    print("\n".join(terminal))


if __name__ == "__main__":
    main()
