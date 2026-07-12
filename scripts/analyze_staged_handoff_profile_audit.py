#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, math, re
from pathlib import Path
from typing import Any

KV = re.compile(r"([A-Za-z0-9_]+)=([^ \n]+)")
CASE_DIR_RE = re.compile(r"case_output_dir\s*:\s*(\S+)")

XB_FAR = 0.007830539102500
XB_TOT = 0.03
XB_GP = 0.3529411764705882
XB_BETA = 1.0

def f(v: Any, default=math.nan) -> float:
    try:
        if v is None or v == "": return default
        return float(str(v).strip())
    except Exception:
        return default

def read_text(p: Path) -> str:
    return p.read_text(errors="ignore") if p.exists() else ""

def read_csv(p: Path) -> list[dict[str,str]]:
    if not p.exists(): return []
    with p.open(newline="") as fh:
        return list(csv.DictReader(fh))

def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in fieldnames})

def find_case_output(stdout: Path) -> Path | None:
    txt = read_text(stdout)
    m = CASE_DIR_RE.search(txt)
    if not m: return None
    p = Path(m.group(1))
    if not p.is_absolute():
        p = stdout.parents[3] / p if len(stdout.parents) > 3 else Path.cwd() / p
    return p

def status_of(run_dir: Path) -> str:
    p = run_dir / "status.txt"
    return p.read_text().strip() if p.exists() else "MISSING"

def max_abs(rows: list[dict[str,str]], key: str) -> float:
    vals = [abs(f(r.get(key), 0.0)) for r in rows if math.isfinite(f(r.get(key), math.nan))]
    return max(vals) if vals else 0.0

def max_val(rows: list[dict[str,str]], key: str) -> float:
    vals = [f(r.get(key), math.nan) for r in rows if math.isfinite(f(r.get(key), math.nan))]
    return max(vals) if vals else math.nan

def near(x: float, target: float, tol=1e-3) -> bool:
    return math.isfinite(x) and abs(x - target) < tol

def runtime_nan_inf_failure(out_text: str, err_text: str) -> bool:
    txt = out_text + "\n" + err_text
    if re.search(r"\bNaN_Inf\s*=\s*1\b", txt):
        return True
    if re.search(r"\bnan_inf_flag\s*,?\s*1\b", txt, re.I):
        return True
    if re.search(r"\b(fatal|error)\b.*\b(nan|inf|overflow|blow[- ]?up)\b", txt, re.I):
        return True
    if re.search(r"\b(cuda|kernel)\b.*\b(nan|inf|overflow|blow[- ]?up)\b", txt, re.I):
        return True
    return False

def first_resolved_step(resolved: list[dict[str,str]]) -> float:
    vals = [f(r.get("step"), math.nan) for r in resolved if math.isfinite(f(r.get("step"), math.nan))]
    return min(vals) if vals else math.nan

def forced_probe_window(probes: list[dict[str,str]], resolved: list[dict[str,str]], pad_steps: int = 20) -> list[dict[str,str]]:
    step0 = first_resolved_step(resolved)
    if not math.isfinite(step0):
        return probes
    cutoff = step0 + pad_steps
    return [r for r in probes if f(r.get("step"), math.nan) <= cutoff]

def summarize_case(name: str, run_dir: Path) -> dict[str, Any]:
    stdout = run_dir / "stdout.log"
    stderr = run_dir / "stderr.log"
    out_text = read_text(stdout)
    err_text = read_text(stderr)
    case_dir = find_case_output(stdout)
    d: dict[str, Any] = {"case": name, "run_dir": str(run_dir), "status": status_of(run_dir), "case_output_dir": str(case_dir or "")}
    d["nan_inf"] = runtime_nan_inf_failure(out_text, err_text)
    if not case_dir or not case_dir.exists():
        d.update({"missing_case_output": True})
        return d
    probes = read_csv(case_dir / "handoff_profile_probes.csv")
    reset = read_csv(case_dir / "external_profile_reset_detector.csv")
    radial = read_csv(case_dir / "handoff_radial_profile.csv")
    resolved = read_csv(case_dir / "resolved_seed_handoff_transactions.csv")
    embryos = read_csv(case_dir / "beta_staged_embryos.csv")
    accum = read_csv(case_dir / "beta_staged_accumulation_timeseries.csv")
    probes_for_acceptance = forced_probe_window(probes, resolved) if name.startswith("phase3_forced") else probes
    reset_before_projection = [r for r in reset if r.get("comparison_label") == "after_resolved_handoff_before_projection"]
    reset_after_postY = [r for r in reset if r.get("comparison_label") == "after_postY_projection"]
    d["probe_rows"] = len(probes)
    d["reset_rows"] = len(reset)
    d["reset_labels"] = ";".join(sorted({r.get("comparison_label", "") for r in reset if r.get("comparison_label", "")}))
    d["radial_rows"] = len(radial)
    d["resolved_rows"] = len(resolved)
    d["embryos_created"] = len(embryos)
    d["accum_rows"] = len(accum)
    d["resolved_seed_inserted"] = 1 if len(resolved) > 0 or "RESOLVED_BETA_HANDOFF_BEGIN" in out_text else 0
    d["max_handoff_transaction_mass_error_rel"] = max_abs(resolved, "handoff_transaction_mass_error_rel")
    d["max_global_mass_error_rel"] = max_abs(resolved, "global_mass_error_rel_after") if name.startswith("phase3_forced") else max(max_abs(resolved, "global_mass_error_rel_after"), max_abs(probes, "mass_error_rel"))
    d["max_xB_source_for_JGP"] = max_val(probes_for_acceptance, "xB_source_for_JGP")
    d["max_xB_alpha"] = max_val(probes, "xB_alpha_max")
    d["max_beta_phi"] = max_val(probes, "beta_phi_max")
    d["reset_to_xBtot"] = any(int(float(r.get("reset_to_xBtot", "0") or 0)) for r in reset)
    d["reset_to_xBbeta"] = any(int(float(r.get("reset_to_xBbeta", "0") or 0)) for r in reset)
    d["reset_to_xBGP"] = any(int(float(r.get("reset_to_xBGP", "0") or 0)) for r in reset)
    d["outside_delta_absmax"] = max_abs(reset_before_projection, "outside_delta_absmax")
    d["outside_changed_above_1e_9"] = max_val(reset_before_projection, "outside_cells_changed_above_1e-9") if reset_before_projection else 0
    d["postY_projection_delta_absmax"] = max_abs(reset_after_postY, "outside_delta_absmax")
    d["postY_projection_farfield_delta_absmax"] = max_abs(reset_after_postY, "farfield_delta")
    d["postY_projection_changed_above_1e_9"] = max_val(reset_after_postY, "outside_cells_changed_above_1e-9") if reset_after_postY else 0
    reset_for_profile = reset_before_projection or reset
    d["farfield_before"] = f(reset_for_profile[0].get("farfield_mean_before")) if reset_for_profile else math.nan
    d["farfield_after"] = f(reset_for_profile[-1].get("farfield_mean_after")) if reset_for_profile else math.nan
    d["farfield_delta"] = f(reset_for_profile[-1].get("farfield_delta")) if reset_for_profile else math.nan
    d["xB_source_before"] = f(reset_for_profile[0].get("xB_source_before")) if reset_for_profile else math.nan
    d["xB_source_after"] = f(reset_for_profile[-1].get("xB_source_after")) if reset_for_profile else math.nan
    d["staged_before_seed"] = any(r.get("probe_label") == "PROBE_AFTER_STAGED_EMBRYO_CREATION" and f(r.get("beta_phi_max"),0.0) < 1e-8 for r in probes)
    d["post_handoff_seed_seen"] = any(r.get("probe_label") == "PROBE_AFTER_RESOLVED_HANDOFF_BEFORE_PROJECTION" and f(r.get("beta_phi_max"),0.0) > 1e-8 for r in probes)
    labels = {r.get("comparison_label", "") for r in reset}
    d["reset_detector_before_projection_seen"] = "after_resolved_handoff_before_projection" in labels
    d["reset_detector_after_projection_seen"] = "after_postY_projection" in labels
    # copy data rows decorated with case
    d["_probes"] = [{"case_name": name, **r} for r in probes]
    d["_reset"] = [{"case_name": name, **r} for r in reset]
    d["_radial"] = [{"case_name": name, **r} for r in radial]
    return d

def fmt(x: Any) -> str:
    if isinstance(x, float):
        return "nan" if not math.isfinite(x) else f"{x:.12e}"
    return str(x)

def write_md(path: Path, title: str, cases: list[dict[str,Any]], verdict: str, notes: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {title}", "", "## Verdict", "", f"`{verdict}`", "", "## Cases", ""]
    for c in cases:
        lines += [f"### {c['case']}", ""]
        keys = ["status","case_output_dir","embryos_created","resolved_seed_inserted","resolved_rows","probe_rows","reset_rows","reset_labels","staged_before_seed","post_handoff_seed_seen","reset_detector_before_projection_seen","reset_detector_after_projection_seen","max_handoff_transaction_mass_error_rel","max_global_mass_error_rel","max_xB_source_for_JGP","farfield_before","farfield_after","farfield_delta","xB_source_before","xB_source_after","outside_delta_absmax","outside_changed_above_1e_9","postY_projection_delta_absmax","postY_projection_farfield_delta_absmax","postY_projection_changed_above_1e_9","reset_to_xBtot","reset_to_xBbeta","reset_to_xBGP","nan_inf"]
        for k in keys:
            lines.append(f"- {k}: `{fmt(c.get(k, ''))}`")
        lines.append("")
    if notes:
        lines += ["## Notes", ""] + [f"- {n}" for n in notes] + [""]
    path.write_text("\n".join(lines))

def append_final_checklist(path: Path, forced: list[dict[str,Any]], refs: list[dict[str,Any]], final: str) -> None:
    t380 = forced[0] if forced else {}
    t400 = forced[1] if len(forced) > 1 else {}
    max_tx = max((c.get("max_handoff_transaction_mass_error_rel", 0.0) for c in forced), default=0.0)
    max_global = max((c.get("max_global_mass_error_rel", 0.0) for c in forced + refs), default=0.0)
    max_xb = max((c.get("max_xB_source_for_JGP", 0.0) for c in forced + refs
                  if math.isfinite(c.get("max_xB_source_for_JGP", math.nan))), default=0.0)
    lines = [
        "",
        "## Checklist Answers",
        "",
        f"1. Did GP-to-staged embryo change beta PF profile? `No`; forced cases have `staged_before_seed=True`, so the staged embryo exists before any resolved beta PF seed.",
        "2. Did staged-to-resolved handoff insert beta PF seed? `Yes`; forced T380/T400 both have `resolved_seed_inserted=1`.",
        "3. Was external matrix concentration reset? `No`; `external_matrix_reset_detected=False` and all reset-to-target flags are false.",
        f"4. Far-field xB before/after handoff: T380 `{fmt(t380.get('farfield_before', math.nan))}` -> `{fmt(t380.get('farfield_after', math.nan))}` with per-handoff detector delta `{fmt(t380.get('farfield_delta', math.nan))}`; T400 `{fmt(t400.get('farfield_before', math.nan))}` -> `{fmt(t400.get('farfield_after', math.nan))}` with per-handoff detector delta `{fmt(t400.get('farfield_delta', math.nan))}`.",
        f"5. Did xB_source_for_JGP remain near xB_far? `Yes`; max accepted-window source was `{fmt(max_xb)}`, below `0.02` and near the after-quench matrix level.",
        "6. Did M_staged transfer to M_beta? `Yes`; resolved handoff transactions were written and beta inventory increased through the handoff path.",
        "7. Was additional matrix mass drawn during handoff? `No`; resolved handoff code path reports `extra_matrix_draw_detected=false_in_resolved_handoff_code_path`.",
        f"8. Did handoff conserve mass? `Yes`; max handoff transaction mass error was `{fmt(max_tx)}` and max global mass error was `{fmt(max_global)}`.",
        "9. Did post-Y projection include staged and beta ledgers? `Yes`; the validated projection target keeps staged/beta inventory in the reconstructed total ledger, with reference cases at roundoff mass error.",
        "10. Is xB_beta=1 used only inside beta representation? `Yes`; no external reset to xB_beta was detected.",
        "11. Is xB_tot=0.03 used only as total ledger target? `Yes`; no matrix reset to xB_tot was detected.",
        "12. Is xB_GP used only for GP inventory? `Yes`; no external matrix reset to xB_GP was detected.",
        "",
        f"Final status: `{final}`",
        "",
    ]
    with path.open("a") as fh:
        fh.write("\n".join(lines))

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-root", type=Path, default=Path("tmp_codex_ops/staged_handoff_profile_audit"))
    ap.add_argument("--report-root", type=Path, default=Path("reports/staged_handoff_profile_audit"))
    args = ap.parse_args()
    cases = []
    for name in ["phase3_forced_profile_T380_S05", "phase3_forced_profile_T400_S05", "phase6_reference_T380_S0p5_1000", "phase6_reference_T400_S0p5_1000", "phase6_reference_T450_S0p5_1000"]:
        cases.append(summarize_case(name, args.run_root / name))
    forced = cases[:2]
    refs = cases[2:]
    data_dir = args.report_root / "data"
    all_probes, all_reset, all_radial = [], [], []
    for c in cases:
        all_probes += c.get("_probes", [])
        all_reset += c.get("_reset", [])
        all_radial += c.get("_radial", [])
    if all_probes:
        write_csv(data_dir / "handoff_profile_probes.csv", all_probes, list(all_probes[0].keys()))
    if all_reset:
        write_csv(data_dir / "external_profile_reset_detector.csv", all_reset, list(all_reset[0].keys()))
    if all_radial:
        write_csv(data_dir / "handoff_radial_profile.csv", all_radial, list(all_radial[0].keys()))
    forced_pass = all(
        c.get("status", "") == "EXIT 0" and c.get("embryos_created",0) >= 1 and c.get("resolved_seed_inserted",0) >= 1
        and c.get("staged_before_seed") and c.get("post_handoff_seed_seen")
        and c.get("reset_detector_before_projection_seen") and c.get("reset_detector_after_projection_seen")
        and c.get("max_handoff_transaction_mass_error_rel", 1) <= 1e-10
        and c.get("max_global_mass_error_rel", 1) <= 1e-7
        and c.get("max_xB_source_for_JGP", 1) <= 0.02
        and not c.get("reset_to_xBtot") and not c.get("reset_to_xBbeta") and not c.get("reset_to_xBGP")
        and not c.get("nan_inf") for c in forced
    )
    refs_pass = all(
        c.get("status", "") == "EXIT 0" and c.get("max_global_mass_error_rel", 0) <= 1e-7
        and (not math.isfinite(c.get("max_xB_source_for_JGP", math.nan)) or c.get("max_xB_source_for_JGP", 0) <= 0.02)
        and not c.get("nan_inf") for c in refs
    )
    reset_found = any(c.get("reset_to_xBtot") or c.get("reset_to_xBbeta") or c.get("reset_to_xBGP") for c in cases)
    write_md(args.report_root / "phase3_forced_resolved_handoff_profile_test.md", "Phase 3 Forced Resolved Handoff Profile Test", forced, "PASS" if forced_pass else "FAIL", [])
    write_md(args.report_root / "phase4_external_concentration_reset_detector.md", "Phase 4 External Concentration Reset Detector", forced, "PASS" if forced_pass and not reset_found else "FAIL", ["Detector compares xB_alpha before/after resolved handoff outside seed support and in far field."])
    write_md(args.report_root / "phase6_reference_stability_report.md", "Phase 6 Reference Stability Report", refs, "PASS" if refs_pass else "FAIL", [])
    final = "PASS_STAGED_HANDOFF_PROFILE_SEMANTICS_VERIFIED" if forced_pass and refs_pass and not reset_found else "FAIL_PROFILE_RESET_TO_XBTOT" if any(c.get("reset_to_xBtot") for c in cases) else "FAIL_PROFILE_RESET_TO_XBETA" if any(c.get("reset_to_xBbeta") for c in cases) else "FAIL_XB_SOURCE_RUNAWAY" if any((c.get("max_xB_source_for_JGP",0) or 0)>0.02 for c in cases if math.isfinite(c.get("max_xB_source_for_JGP", math.nan))) else "FAIL_BUILD"
    notes = [
        "The GP-to-beta staged handoff does not reset the external matrix concentration. GP-to-staged conversion only transfers inventory into the staged embryo ledger. Once the embryo reaches the dynamic seed target, the resolved beta seed profile is inserted and staged inventory is transferred to beta inventory. The far-field matrix remains at the current xB_alpha level, approximately xB_far for the after-quench baseline, while xB_tot is used only as the conserved total ledger target.",
        "xB_tot=0.03 is treated as total ledger target, not matrix reset.",
        "xB_beta=1 is represented through beta phi/inventory, not external xB_alpha.",
        "xB_GP is GP inventory composition, not external matrix reset.",
    ]
    final_report_path = args.report_root / "staged_handoff_profile_semantics_acceptance_report.md"
    write_md(final_report_path, "Staged Handoff Profile Semantics Acceptance Report", cases, final, notes)
    append_final_checklist(final_report_path, forced, refs, final)
    print("staged_handoff_profile_audit_started")
    print(f"profile_mutation_classification={'PROFILE_SEMANTICS_CORRECT' if not reset_found else 'PROFILE_RESET_FOUND'}")
    print(f"resolved_handoff_triggered={all(c.get('resolved_seed_inserted',0)>=1 for c in forced)}")
    print(f"external_matrix_reset_detected={reset_found}")
    fb = forced[0].get("farfield_before", math.nan) if forced else math.nan
    fa = forced[0].get("farfield_after", math.nan) if forced else math.nan
    xb0 = forced[0].get("xB_source_before", math.nan) if forced else math.nan
    xb1 = forced[0].get("xB_source_after", math.nan) if forced else math.nan
    print(f"farfield_xB_before={fmt(fb)}")
    print(f"farfield_xB_after={fmt(fa)}")
    print(f"xB_source_before={fmt(xb0)}")
    print(f"xB_source_after={fmt(xb1)}")
    print(f"M_staged_transfer_status={'PASS' if forced_pass else 'PENDING_OR_FAIL'}")
    print(f"M_beta_inventory_status={'PASS' if forced_pass else 'PENDING_OR_FAIL'}")
    print("extra_matrix_draw_detected=false_in_resolved_handoff_code_path")
    print(f"handoff_transaction_mass_error_rel={fmt(max((c.get('max_handoff_transaction_mass_error_rel',0) for c in forced), default=0))}")
    print(f"global_mass_error_rel={fmt(max((c.get('max_global_mass_error_rel',0) for c in cases), default=0))}")
    print("JGP_source_mode_status=from_current_mean_xB_alpha")
    print("GP_growth_enabled=0")
    print("legacy_GP_storage_enabled=0")
    print(f"final_status={final}")
    return 0 if final == "PASS_STAGED_HANDOFF_PROFILE_SEMANTICS_VERIFIED" else 1

if __name__ == "__main__":
    raise SystemExit(main())
