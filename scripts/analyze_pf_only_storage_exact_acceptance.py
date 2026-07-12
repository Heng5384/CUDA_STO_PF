#!/usr/bin/env python3
"""Analyze equal-time PF-only storage-exact acceptance cases."""

from __future__ import annotations

import csv
import math
from pathlib import Path
from collections import defaultdict


ROOT = Path(__file__).resolve().parents[1]
PARAM = ROOT / "params/pf_only_baseline_closure/remediation/acceptance_manifest.csv"
REPORT = ROOT / "reports/pf_only_baseline_closure"


def read(path: Path) -> list[dict[str, str]]:
    if not path.exists(): return []
    with path.open(newline="") as handle: return list(csv.DictReader(handle))


def num(row: dict[str, str], key: str, default: float = math.nan) -> float:
    try: return float(row.get(key, ""))
    except (TypeError, ValueError): return default


def locate(row: dict[str, str]) -> Path | None:
    dt = str(float(row["dt"]))
    base = ROOT / f"Results/chel_T{row['T_C']}_cuda_128x128x128_dt{dt}_steps{row['nsteps']}_xB0.008"
    candidate = base / row["case"]
    if candidate.exists(): return candidate
    found = list((ROOT / "Results").glob(f"chel_T{row['T_C']}_cuda_128x128x128_dt*_steps{row['nsteps']}_xB0.008/{row['case']}"))
    return found[0] if found else None


def dedupe_growth(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    by_step: dict[int, dict[str, str]] = {}
    for item in sorted(rows, key=lambda r: (int(r["step"]), int(r["post_handoff_step"]))):
        by_step[int(item["step"])] = item
    return [by_step[key] for key in sorted(by_step)]


def interpolate(rows: list[dict[str, str]], dt: float, key: str, t_code: float) -> float:
    points = sorted((num(r, "post_handoff_step") * dt, num(r, key)) for r in rows)
    points = [(t, value) for t, value in points if math.isfinite(t) and math.isfinite(value)]
    if not points or t_code < points[0][0] - 1e-12 or t_code > points[-1][0] + 1e-12:
        return math.nan
    for t, value in points:
        if abs(t - t_code) <= 1e-12:
            return value
    for (ta, va), (tb, vb) in zip(points, points[1:]):
        if ta <= t_code <= tb and tb > ta:
            w = (t_code - ta) / (tb - ta)
            return va + w * (vb - va)
    return points[-1][1]


def rms(values: list[float]) -> float:
    finite = [v for v in values if math.isfinite(v)]
    return math.sqrt(sum(v*v for v in finite) / len(finite)) if finite else math.nan


def main() -> None:
    manifest = read(PARAM)
    summaries: list[dict[str, object]] = []
    timeline: list[dict[str, object]] = []
    curves: dict[tuple[int, float], list[dict[str, str]]] = {}
    for meta in manifest:
        out = locate(meta)
        growth = read(out / "diagnostic_rsmd_seed_growth_time_series.csv") if out else []
        regional = read(out / "diagnostic_rsmd_regional_xB_context.csv") if out else []
        projection = read(out / "y_update_mass_projection.csv") if out else []
        if not growth:
            summaries.append({**meta, "run_status": "NOT_RUN", "pass": False})
            continue
        # The runtime may emit pre-projection and step-end rows at one step.
        growth = dedupe_growth(growth)
        curves[(int(meta["T_C"]), float(meta["dt"]))] = growth
        initial, final = growth[0], growth[-1]
        h0, hf = num(initial, "h_integral"), num(final, "h_integral")
        r0, rf = num(initial, "R_eff_h_nm"), num(final, "R_eff_h_nm")
        phi_max_final = num(final, "phi_max")
        support_phi_gt_0p5_final = num(final, "support_phi_gt_0p5")
        alpha = [r for r in regional if r.get("region") == "h_lt_0p1"]
        interface = [r for r in regional if r.get("region") in {"h_0p1_0p5", "h_0p5_0p9"}]
        max_alpha = max((num(r, "xB_alpha_max") for r in alpha), default=math.nan)
        max_interface = max((num(r, "xB_alpha_max") for r in interface), default=math.nan)
        max_ctot = max((num(r, "C_tot_max") for r in regional), default=math.nan)
        max_mass = max((abs(num(r, "mass_error_rel")) for r in growth), default=math.nan)
        max_proj_after = max((abs(num(r, "delta_after_projection")) for r in projection), default=math.nan)
        config_rows = read(out / "diagnostic_rsmd_runtime_config.csv")
        config = config_rows[-1] if config_rows else {}
        runtime_mode = config.get("pf_y_update_mode", "MISSING")
        storage_floor = num(config, "pf_matrix_storage_floor")
        source_fmax = num(config, "f_max_per_step")
        source_frozen = source_fmax == 0.0
        nan_inf = any(not math.isfinite(num(r, "R_eff_h_nm")) or
                      not math.isfinite(num(r, "h_integral")) for r in growth)
        bounded = max_alpha < 0.10 and max_interface < 0.10 and max_ctot <= 1.0 + 1e-8
        # Distinguish a finite, grid-resolved beta-like core from the historical
        # artificial collapse (R=0 and h/h0~1e-70). Strong physical shrink is
        # allowed by the requested acceptance as long as the core remains resolved.
        retained = (hf > 1.0e-4*h0 and rf >= 1.0 and
                    phi_max_final >= 0.8 and support_phi_gt_0p5_final >= 1.0)
        mass_pass = max_mass <= 1e-10
        mode_pass = runtime_mode == "x_transport_projection_split"
        floor_pass = math.isfinite(storage_floor) and abs(storage_floor - 0.1) <= 1e-12
        case_pass = bounded and retained and mass_pass and not nan_inf and mode_pass and floor_pass and source_frozen
        if not retained: fate = "COLLAPSE"
        elif hf < 0.8*h0: fate = "SHRINK"
        elif hf > 1.2*h0: fate = "GROW"
        else: fate = "STABLE_RELAXATION"
        summaries.append({
            **meta, "run_status": "COMPLETE", "output_dir": str(out),
            "R_eff_h_initial_nm": r0, "R_eff_h_final_nm": rf,
            "delta_R_eff_h_nm": rf-r0, "h_integral_initial": h0,
            "h_integral_final": hf, "h_integral_ratio": hf/h0 if h0 else math.nan,
            "phi_max_final": phi_max_final,
            "support_phi_gt_0p5_final": support_phi_gt_0p5_final,
            "max_alpha_xB": max_alpha, "max_interface_xB": max_interface,
            "max_C_tot": max_ctot, "max_mass_error_rel": max_mass,
            "max_projection_residual_after": max_proj_after,
            "runtime_pf_y_update_mode": runtime_mode,
            "runtime_storage_floor": storage_floor,
            "S3_f_max_per_step": source_fmax,
            "S3_source_frozen": source_frozen,
            "runtime_mode_pass": mode_pass,
            "storage_floor_pass": floor_pass,
            "bounded": bounded, "retained": retained, "nan_inf": nan_inf,
            "fate": fate, "pass": case_pass,
        })
        for row in growth:
            timeline.append({"case": meta["case"], "T_C": meta["T_C"], "dt": meta["dt"],
                             "post_handoff_code_time": num(row, "post_handoff_step") * float(meta["dt"]),
                             **row})

    convergence_rows: list[dict[str, object]] = []
    convergence_by_temp: dict[int, dict[str, object]] = {}
    for temp in (400, 380):
        dts = sorted(dt for (case_temp, dt) in curves if case_temp == temp)
        if len(dts) != 3:
            convergence_by_temp[temp] = {"complete": False, "pass": False}
            continue
        fine, medium, coarse = dts[0], dts[1], dts[2]
        t_end = min(num(curves[(temp, dt)][-1], "post_handoff_step") * dt for dt in dts)
        n = int(math.floor(t_end / 0.02 + 1e-9))
        common_times = [0.02 * i for i in range(n + 1)]
        pair_metrics: dict[tuple[float, float], dict[str, float]] = {}
        for dt, ref_dt in ((coarse, fine), (medium, fine), (coarse, medium)):
            dr: list[float] = []
            dh: list[float] = []
            for t in common_times:
                r_dt = interpolate(curves[(temp, dt)], dt, "R_eff_h_nm", t)
                r_ref = interpolate(curves[(temp, ref_dt)], ref_dt, "R_eff_h_nm", t)
                h_dt = interpolate(curves[(temp, dt)], dt, "h_integral", t)
                h_ref = interpolate(curves[(temp, ref_dt)], ref_dt, "h_integral", t)
                dr.append(r_dt-r_ref)
                dh.append((h_dt-h_ref) / max(abs(h_ref), 1e-30))
            pair_metrics[(dt, ref_dt)] = {
                "R_L2_nm": rms(dr),
                "R_Linf_nm": max((abs(v) for v in dr if math.isfinite(v)), default=math.nan),
                "h_relative_L2": rms(dh),
                "h_relative_Linf": max((abs(v) for v in dh if math.isfinite(v)), default=math.nan),
            }
        coarse_fine = pair_metrics[(coarse, fine)]
        medium_fine = pair_metrics[(medium, fine)]
        errors_decrease = (
            medium_fine["R_L2_nm"] < coarse_fine["R_L2_nm"] and
            medium_fine["h_relative_L2"] < coarse_fine["h_relative_L2"]
        )
        finals = {dt: curves[(temp, dt)][-1] for dt in dts}
        signs = []
        for dt in dts:
            rows = curves[(temp, dt)]
            signs.append(math.copysign(1.0, num(rows[-1], "R_eff_h_nm") - num(rows[0], "R_eff_h_nm")))
        sign_consistent = len(set(signs)) == 1
        case_rows = [r for r in summaries if int(r["T_C"]) == temp and r.get("run_status") == "COMPLETE"]
        cases_pass = len(case_rows) == 3 and all(r.get("pass") is True for r in case_rows)
        conv_pass = cases_pass and sign_consistent and errors_decrease
        convergence_by_temp[temp] = {
            "complete": True, "pass": conv_pass, "sign_consistent": sign_consistent,
            "errors_decrease": errors_decrease,
            "coarse_fine_R_L2_nm": coarse_fine["R_L2_nm"],
            "medium_fine_R_L2_nm": medium_fine["R_L2_nm"],
            "coarse_fine_h_relative_L2": coarse_fine["h_relative_L2"],
            "medium_fine_h_relative_L2": medium_fine["h_relative_L2"],
        }
        for (dt, ref_dt), metrics in pair_metrics.items():
            convergence_rows.append({"T_C": temp, "dt": dt, "reference_dt": ref_dt,
                                     "common_time_end_code": t_end,
                                     "common_time_points": len(common_times), **metrics})

    complete_fields = list(summaries[0])
    for row in summaries:
        for key in row:
            if key not in complete_fields: complete_fields.append(key)
    with (REPORT / "pf_only_dt_convergence_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=complete_fields)
        writer.writeheader(); writer.writerows(summaries)
    if timeline:
        fields = list(timeline[0])
        with (REPORT / "pf_only_dt_convergence_time_series.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
            writer.writeheader(); writer.writerows(timeline)
    if convergence_rows:
        with (REPORT / "pf_only_dt_convergence_metrics.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(convergence_rows[0]))
            writer.writeheader(); writer.writerows(convergence_rows)

    t400 = bool(convergence_by_temp.get(400, {}).get("pass"))
    t380 = bool(convergence_by_temp.get(380, {}).get("pass"))
    complete = [r for r in summaries if r.get("run_status") == "COMPLETE"]
    report = ["# PF-only Baseline Acceptance", "",
              "The accepted candidate evolves matrix-channel composition with a semi-implicit Fourier transport step, protects ill-conditioned beta support, and uses the existing full-storage projection to close the phi/transport split residual.", "",
              f"T400 three-dt pass: `{t400}`. T380 three-dt pass: `{t380}`.", "",
              "| T (C) | dt | fate | max alpha xB | max interface xB | mass error | R final (nm) | pass |",
              "|---:|---:|---|---:|---:|---:|---:|---|" ]
    for r in complete:
        report.append(f"| {r['T_C']} | {float(r['dt']):.4g} | {r['fate']} | {float(r['max_alpha_xB']):.6g} | {float(r['max_interface_xB']):.6g} | {float(r['max_mass_error_rel']):.3e} | {float(r['R_eff_h_final_nm']):.6g} | {r['pass']} |")
    report += ["", "## Equal-time convergence"]
    for temp in (400, 380):
        c = convergence_by_temp.get(temp, {})
        report.append(
            f"- T{temp}: complete={c.get('complete', False)}, sign_consistent={c.get('sign_consistent', False)}, "
            f"trajectory_errors_decrease={c.get('errors_decrease', False)}, pass={c.get('pass', False)}."
        )
    (REPORT / "pf_only_baseline_acceptance_report.md").write_text("\n".join(report) + "\n")
    print(f"T400_pass={str(t400).lower()}")
    print(f"T380_pass={str(t380).lower()}")


if __name__ == "__main__": main()
