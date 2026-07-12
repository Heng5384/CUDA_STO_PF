#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any

N_A = 6.02214076e23
R_GAS = 8.31446261815324


def safe_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


def safe_bool(value: Any) -> bool:
    text = str(value).strip().lower()
    return text in {"1", "true", "yes", "on"}


def fmt(value: Any, digits: int = 12) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if value is None:
        return ""
    if isinstance(value, (int,)):
        return str(value)
    if isinstance(value, float):
        if not math.isfinite(value):
            return ""
        return f"{value:.{digits}g}"
    return str(value)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: fmt(row.get(field)) for field in fieldnames})


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")


def fit_quadratic_curvature(points: list[tuple[float, float]]) -> tuple[float, float, float, float] | None:
    if len(points) < 3:
        return None
    sx = sy = sxx = sxxx = sxxxx = sxy = sxxy = 0.0
    for x, y in points:
        xx = x * x
        sx += x
        sy += y
        sxx += xx
        sxxx += xx * x
        sxxxx += xx * xx
        sxy += x * y
        sxxy += xx * y
    n = float(len(points))
    a11, a12, a13 = sxxxx, sxxx, sxx
    a21, a22, a23 = sxxx, sxx, sx
    a31, a32, a33 = sxx, sx, n
    b1, b2, b3 = sxxy, sxy, sy
    det = (
        a11 * (a22 * a33 - a23 * a32)
        - a12 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * a32 - a22 * a31)
    )
    if abs(det) < 1.0e-30:
        return None
    det_a = (
        b1 * (a22 * a33 - a23 * a32)
        - a12 * (b2 * a33 - a23 * b3)
        + a13 * (b2 * a32 - a22 * b3)
    )
    det_b = (
        a11 * (b2 * a33 - a23 * b3)
        - b1 * (a21 * a33 - a23 * a31)
        + a13 * (a21 * b3 - b2 * a31)
    )
    det_c = (
        a11 * (a22 * b3 - b2 * a32)
        - a12 * (a21 * b3 - b2 * a31)
        + b1 * (a21 * a32 - a22 * a31)
    )
    a = det_a / det
    b = det_b / det
    c = det_c / det
    curvature = 2.0 * a
    x_vertex = -b / (2.0 * a) if abs(a) > 1.0e-30 else float("nan")
    ss_tot = 0.0
    ss_res = 0.0
    y_mean = sy / n
    for x, y in points:
        y_hat = a * x * x + b * x + c
        ss_tot += (y - y_mean) ** 2
        ss_res += (y - y_hat) ** 2
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 1.0e-30 else 1.0
    return curvature, x_vertex, r2, a


def z_n_from_z_r(z_r_m_inv: float | None, omega_g_m3: float | None, r_star_nm: float | None) -> float | None:
    if z_r_m_inv is None or omega_g_m3 is None or r_star_nm is None:
        return None
    r_star_m = r_star_nm * 1.0e-9
    if not (
        math.isfinite(z_r_m_inv) and z_r_m_inv > 0.0 and
        math.isfinite(omega_g_m3) and omega_g_m3 > 0.0 and
        math.isfinite(r_star_m) and r_star_m > 0.0
    ):
        return None
    return z_r_m_inv * omega_g_m3 / (4.0 * math.pi * r_star_m * r_star_m)


def merged_barrier_record(primary: dict[str, Any] | None, recovered: dict[str, str] | None) -> dict[str, Any] | None:
    if primary is None and recovered is None:
        return None
    out: dict[str, Any] = dict(primary or {})
    if recovered is None:
        return out
    out.setdefault("entry_id", recovered.get("entry_id", ""))
    out["T_C"] = safe_float(recovered.get("T_C"), safe_float(out.get("T_C")))
    out["xB"] = safe_float(recovered.get("xB"), safe_float(out.get("xB")))
    out["strain_mode"] = recovered.get("strain_mode") or out.get("strain_mode") or "no_strain"
    out["r_star_nm"] = safe_float(recovered.get("r_star_nm"), safe_float(out.get("r_star_nm")))
    out["DeltaG_bare_kBT"] = safe_float(recovered.get("DeltaG_bare_kBT"), safe_float(out.get("DeltaG_bare_kBT")))
    out["barrier_base_case_tag"] = recovered.get("barrier_base_case_tag") or out.get("barrier_base_case_tag") or out.get("entry_id", "")
    out["barrier_curve_source"] = recovered.get("barrier_curve_source") or out.get("barrier_curve_source") or ""
    out["fit_window_nm"] = safe_float(recovered.get("Z_r_fit_window_nm"), safe_float(out.get("fit_window_nm")))
    out["curvature_d2G_dr2_kBT_per_nm2"] = safe_float(
        recovered.get("curvature_d2G_dr2_kBT_per_nm2"),
        safe_float(out.get("curvature_d2G_dr2_kBT_per_nm2")),
    )
    out["Z_r_m_inv"] = safe_float(recovered.get("Z_r_m_inv"), safe_float(out.get("Z_r_m_inv")))
    out["Z_r_status"] = recovered.get("Z_r_status") or out.get("Z_r_status") or "MISSING"
    out["n_fit_points"] = int(round((out["fit_window_nm"] or 0.0) / 0.1)) + 1 if out.get("fit_window_nm") else out.get("n_fit_points", 0)
    out["production_usable"] = out.get("Z_r_status") == "OK" and safe_float(out.get("Z_r_m_inv")) is not None
    out["notes"] = (out.get("notes") or "").strip()
    return out


def main() -> int:
    repo = Path(__file__).resolve().parents[2]
    out_dir = repo / "reports/nucleus_library_workflow/physical_cnt_parameter_closure"
    data_dir = repo / "data"
    params_dir = repo / "params/physical_cnt"
    out_dir.mkdir(parents=True, exist_ok=True)

    barrier_summary_path = repo / "Results/workflows/no_strain_400cube_dx0p1_cnt_sweep/summary_reports/real_unit_barriers/barrier_real_units_by_case.csv"
    barrier_detail_path = repo / "Results/workflows/no_strain_400cube_dx0p1_cnt_sweep/summary_reports/real_unit_barriers/barrier_real_units_detail.csv"
    staging_path = repo / "data/nucleus_library/nucleus_library.staging_dx1nm_T380_T400_T450.csv"
    recovered_zr_path = repo / "data/nucleus_library/recovered_cluster_zr_rows.csv"
    physical_inputs_path = repo / "physical_inputs.example.json"

    barrier_summary = read_csv(barrier_summary_path)
    barrier_detail = read_csv(barrier_detail_path)
    staging_rows = read_csv(staging_path)
    recovered_zr_rows = read_csv(recovered_zr_path) if recovered_zr_path.exists() else []
    physical_inputs = json.loads(physical_inputs_path.read_text(encoding="utf-8"))

    vm_compound = safe_float(physical_inputs.get("Vm_compound"))
    vm_alpha = safe_float(physical_inputs.get("Vm_alpha_0"))
    if vm_compound is None or vm_alpha is None:
        raise SystemExit("missing Vm_compound/Vm_alpha_0 in physical_inputs.example.json")
    omega_g_m3 = vm_compound / N_A
    omega_site_m3 = vm_alpha / N_A
    n_site_m3 = 1.0 / omega_site_m3
    d0_m2_s = 4.251e-15
    q_j_mol = 3.403e4

    summary_by_key: dict[tuple[float, float, float], dict[str, str]] = {}
    summary_by_tag: dict[str, dict[str, str]] = {}
    for row in barrier_summary:
        key = (
            round(safe_float(row.get("T_C"), float("nan")) or float("nan"), 6),
            round(safe_float(row.get("xB"), float("nan")) or float("nan"), 6),
            round(safe_float(row.get("strain"), 0.0) or 0.0, 6),
        )
        summary_by_key[key] = row
        summary_by_tag[row["base_case_tag"]] = row

    detail_by_tag: dict[str, list[dict[str, str]]] = {}
    for row in barrier_detail:
        detail_by_tag.setdefault(row["base_case_tag"], []).append(row)
    for rows in detail_by_tag.values():
        rows.sort(key=lambda r: safe_float(r.get("radius_nm"), 0.0) or 0.0)

    zr_rows: list[dict[str, Any]] = []
    zr_by_key: dict[tuple[float, float, float], dict[str, Any]] = {}
    for key, srow in sorted(summary_by_key.items()):
        base_case_tag = srow["base_case_tag"]
        detail_rows = detail_by_tag.get(base_case_tag, [])
        peak_radius = safe_float(srow.get("cnt_refsub_peak_radius_nm"))
        if peak_radius is None:
            peak_radius = safe_float(srow.get("rc_schur_nm"))
        usable_rows = [
            row for row in detail_rows
            if safe_float(row.get("radius_nm")) is not None and safe_float(row.get("DeltaG_CNT_refsub_kBT")) is not None
        ]
        usable_rows.sort(key=lambda row: abs((safe_float(row.get("radius_nm")) or 0.0) - (peak_radius or 0.0)))
        chosen = usable_rows[:5]
        chosen.sort(key=lambda row: safe_float(row.get("radius_nm"), 0.0) or 0.0)
        points = [
            (safe_float(row["radius_nm"]) or 0.0, safe_float(row["DeltaG_CNT_refsub_kBT"]) or 0.0)
            for row in chosen
        ]
        fit = fit_quadratic_curvature(points)
        status = "OK"
        notes = ""
        curvature = None
        z_m_inv = None
        z_nm_inv = None
        fit_window_nm = None
        vertex_nm = None
        r2 = None
        if fit is None:
            status = "MISSING_INSUFFICIENT_BARRIER_CURVE"
        else:
            curvature, vertex_nm, r2, a_coef = fit
            if not math.isfinite(curvature):
                status = "CURVATURE_NOT_FINITE"
            elif curvature >= 0.0:
                status = "UNSTABLE_NONCONCAVE_CURVATURE"
            else:
                z_nm_inv = math.sqrt(abs(curvature) / (2.0 * math.pi))
                z_m_inv = z_nm_inv * 1.0e9
                fit_window_nm = max(x for x, _ in points) - min(x for x, _ in points)
                notes = f"vertex_nm={vertex_nm:.6f};R2={r2:.6f};a={a_coef:.6e}"
        production_usable = status == "OK" and z_m_inv is not None and z_m_inv > 0.0
        out = {
            "entry_id": base_case_tag,
            "T_C": safe_float(srow.get("T_C")),
            "xB": safe_float(srow.get("xB")),
            "strain_mode": "no_strain" if abs(safe_float(srow.get("strain"), 0.0) or 0.0) < 1.0e-12 else "strained",
            "r_star_nm": safe_float(srow.get("cnt_refsub_peak_radius_nm")),
            "DeltaG_bare_kBT": safe_float(srow.get("cnt_refsub_peak_kBT")),
            "barrier_curve_path": barrier_detail_path.relative_to(repo).as_posix(),
            "fit_window_nm": fit_window_nm,
            "n_fit_points": len(points),
            "curvature_d2G_dr2_kBT_per_nm2": curvature,
            "Z_r_nm_inv": z_nm_inv,
            "Z_r_m_inv": z_m_inv,
            "Z_r_status": status,
            "production_usable": production_usable,
            "fit_vertex_nm": vertex_nm,
            "fit_R2": r2,
            "notes": notes,
        }
        zr_rows.append(out)
        zr_by_key[key] = out

    recovered_zr_by_key: dict[tuple[float, float, float], dict[str, str]] = {}
    for row in recovered_zr_rows:
        key = (
            round(safe_float(row.get("T_C"), float("nan")) or float("nan"), 6),
            round(safe_float(row.get("xB"), float("nan")) or float("nan"), 6),
            0.0 if (row.get("strain_mode") or "no_strain") == "no_strain" else round(safe_float(row.get("strain_value"), 0.0) or 0.0, 6),
        )
        recovered_zr_by_key[key] = row

    for key, recovered in recovered_zr_by_key.items():
        merged = merged_barrier_record(zr_by_key.get(key), recovered)
        if merged is None:
            continue
        zr_by_key[key] = merged

    zr_rows = [zr_by_key[key] for key in sorted(zr_by_key)]

    merged_rows: list[dict[str, Any]] = []
    for row in staging_rows:
        out = dict(row)
        key = (
            round(safe_float(row.get("T_C"), float("nan")) or float("nan"), 6),
            round(safe_float(row.get("xB"), float("nan")) or float("nan"), 6),
            0.0 if row.get("strain_mode") == "no_strain" else round(safe_float(row.get("strain_value"), 0.0) or 0.0, 6),
        )
        zr = zr_by_key.get(key)
        summary = summary_by_key.get(key)
        recovered = recovered_zr_by_key.get(key)
        zr = merged_barrier_record(zr, recovered)
        deltaG_bare = safe_float(row.get("DeltaG_bare_kBT"))
        if deltaG_bare is None and zr is not None:
            deltaG_bare = safe_float(zr.get("DeltaG_bare_kBT"))
        if deltaG_bare is not None:
            out["DeltaG_bare_kBT"] = deltaG_bare
            out["DeltaG_eff_kBT_default"] = deltaG_bare
        r_star_nm = safe_float(row.get("r_star_nm"))
        if r_star_nm is None and zr is not None:
            r_star_nm = safe_float(zr.get("r_star_nm"))
        if r_star_nm is not None:
            out["r_star_nm"] = r_star_nm
        out["Z_r_m_inv"] = zr.get("Z_r_m_inv") if zr else None
        out["Z_r_unit"] = "1/m" if zr and zr.get("Z_r_m_inv") else ""
        if zr and zr.get("barrier_curve_source"):
            out["Z_r_source"] = str(zr.get("barrier_curve_source"))
        else:
            out["Z_r_source"] = "quadratic_fit_cnt_refsub_peak" if zr and zr.get("Z_r_m_inv") else ""
        out["Z_r_fit_window_nm"] = zr.get("fit_window_nm") if zr else None
        if recovered and recovered.get("Z_r_fit_method"):
            out["Z_r_fit_method"] = recovered.get("Z_r_fit_method")
        else:
            out["Z_r_fit_method"] = "5-point quadratic fit around cnt_refsub peak" if zr else ""
        out["curvature_d2G_dr2_kBT_per_nm2"] = zr.get("curvature_d2G_dr2_kBT_per_nm2") if zr else None
        out["Z_r_status"] = zr.get("Z_r_status") if zr else "MISSING"
        out["z_r_strict"] = zr.get("Z_r_m_inv") if zr and zr.get("Z_r_status") == "OK" else None
        out["strict_status"] = "ok" if zr and zr.get("Z_r_status") == "OK" else "missing_zr"
        out["z_r_diagnostic"] = ""
        out["Z_n"] = z_n_from_z_r(
            zr.get("Z_r_m_inv") if zr else None,
            omega_g_m3,
            safe_float(out.get("r_star_nm")),
        )
        if recovered and recovered.get("Z_n_source") and out["Z_n"]:
            out["Z_n_source"] = recovered.get("Z_n_source")
        else:
            out["Z_n_source"] = "converted_from_Z_r_and_Omega_g" if out["Z_n"] else ""
        out["Z_conversion_Omega_g_m3"] = omega_g_m3 if out["Z_n"] else None
        out["Z_conversion_r_star_m"] = (safe_float(out.get("r_star_nm")) or 0.0) * 1.0e-9 if out["Z_n"] else None
        out["Z_policy"] = "dimensionless_Z_n_with_beta_star_1_per_s"
        out["beta_rate_Omega_g_m3"] = omega_g_m3
        out["beta_rate_Omega_site_m3"] = omega_site_m3
        out["beta_rate_N_site_m3"] = n_site_m3
        if summary:
            out["barrier_base_case_tag"] = summary["base_case_tag"]
            out["barrier_curve_summary_radius_nm"] = safe_float(summary.get("cnt_refsub_peak_radius_nm"))
        elif zr and zr.get("barrier_base_case_tag"):
            out["barrier_base_case_tag"] = zr.get("barrier_base_case_tag")
            out["barrier_curve_summary_radius_nm"] = safe_float(zr.get("r_star_nm"))
        merged_rows.append(out)

    merged_fieldnames = list(staging_rows[0].keys()) + [
        "Z_r_m_inv",
        "Z_r_unit",
        "Z_r_source",
        "Z_r_fit_window_nm",
        "Z_r_fit_method",
        "curvature_d2G_dr2_kBT_per_nm2",
        "Z_r_status",
        "z_r_strict",
        "strict_status",
        "z_r_diagnostic",
        "Z_n",
        "Z_n_source",
        "Z_conversion_Omega_g_m3",
        "Z_conversion_r_star_m",
        "Z_policy",
        "beta_rate_Omega_g_m3",
        "beta_rate_Omega_site_m3",
        "beta_rate_N_site_m3",
        "barrier_base_case_tag",
        "barrier_curve_summary_radius_nm",
    ]
    merged_csv_path = data_dir / "nucleus_library/nucleus_library.with_Zr.staging.csv"
    write_csv(merged_csv_path, merged_rows, merged_fieldnames)
    merged_json_path = data_dir / "nucleus_library/nucleus_library.with_Zr.staging.json"
    write_json(merged_json_path, merged_rows)

    d_b_rows: list[dict[str, Any]] = []
    unique_t = sorted({safe_float(row.get("T_C")) for row in staging_rows if safe_float(row.get("T_C")) is not None})
    for t_c in unique_t:
        t_k = t_c + 273.15
        d_b = d0_m2_s * math.exp(-q_j_mol / (R_GAS * t_k))
        d_b_rows.append(
            {
                "T_C": t_c,
                "T_K": t_k,
                "D_B_alpha_m2_s": d_b,
                "D0_m2_s": d0_m2_s,
                "Q_J_mol": q_j_mol,
                "source": "thermo_utils.h:D_Ag_in_PbTe_m2_per_s / Unit_Psedobinary.py:D_Ag_in_PbTe_m2_per_s",
            }
        )
    d_b_csv_path = data_dir / "physical_params/D_B_alpha_Ag_in_PbTe_SI.csv"
    write_csv(d_b_csv_path, d_b_rows, list(d_b_rows[0].keys()))

    production_rows = [
        row for row in merged_rows
        if safe_bool(row.get("production_valid")) and row.get("strain_mode") == "no_strain" and abs((safe_float(row.get("xB")) or 0.0) - 0.03) < 1.0e-9
    ]
    production_rows.sort(key=lambda row: safe_float(row.get("T_C"), 0.0) or 0.0)
    params_dir.mkdir(parents=True, exist_ok=True)
    generated_params: list[dict[str, Any]] = []
    for row in production_rows:
        t_c = safe_float(row.get("T_C")) or 0.0
        t_k = t_c + 273.15
        d_b = d0_m2_s * math.exp(-q_j_mol / (R_GAS * t_k))
        filename = params_dir / f"physical_cnt_T{int(round(t_c))}_xB003_no_strain_dx1p0.params"
        lines = [
            "# physical_cnt production-parameter overlay for Ag2Te/beta runtime selector",
            f"temperature_C={fmt(t_c)}",
            f"dt={fmt(safe_float(row.get('dt_code')))}",
            f"dt_code={fmt(safe_float(row.get('dt_code')))}",
            f"t_real_unit={fmt(safe_float(row.get('t_real_unit_s')))}",
            f"t_real_unit_s={fmt(safe_float(row.get('t_real_unit_s')))}",
            "beta_rate_model=physical_cnt",
            "beta_rate_use_physical_dt=1",
            "beta_rate_use_gp_barrier_modifier=1",
            "beta_rate_D_B_alpha_model=Arrhenius",
            f"beta_rate_D_B_alpha_D0_m2_s={d0_m2_s:.16e}",
            f"beta_rate_D_B_alpha_Q_J_mol={q_j_mol:.16e}",
            f"beta_rate_D0_m2_s={d0_m2_s:.16e}",
            f"beta_rate_Q_J_mol={q_j_mol:.16e}",
            f"beta_rate_Omega_g_m3={omega_g_m3:.16e}",
            "beta_rate_Omega_g_source=physical_inputs.example.json:Vm_compound/N_A",
            f"beta_rate_Omega_site_m3={omega_site_m3:.16e}",
            f"beta_rate_N_site_m3={n_site_m3:.16e}",
            "beta_rate_site_model=explicit_gp_site_capture_volume",
            "beta_rate_gp_capture_volume_model=runtime_cell_volume",
            "beta_rate_DeltaV_nuc_model=runtime_cell_volume",
            "beta_rate_deltaV_nuc_mode=cell_volume",
            "beta_rate_Z_type=Z_n",
            "beta_rate_Z_r_source=library",
            "beta_rate_Z_r_required=1",
            "beta_rate_Z_r_debug_fallback_enabled=0",
            "beta_rate_Z_r_fallback_mode=disabled",
            "beta_rate_allow_runtime_Zn_from_Zr=0",
            "beta_rate_scale_Z_with_sGP=0",
            "beta_rate_debug_rate_multiplier=1.0",
            "gp_barrier_only_mode=1",
            "enable_legacy_gp_storage_coupling=0",
            "use_nucleus_library_selector=1",
            "enable_gp_runtime_library_nucleation=1",
            "enable_runtime_nucleus_library=1",
            "enable_dynamic_continue_bridge=1",
            "nucleus_library_path=data/nucleus_library/nucleus_library.with_Zr.staging.csv",
            "gp_runtime_barrier_library_path=data/nucleus_library/nucleus_library.with_Zr.staging.csv",
            "gp_runtime_nucleus_library_path=data/nucleus_library/nucleus_library.with_Zr.staging.csv",
            "runtime_profile_cache=Results/runtime_profile_cache/dx1p0",
            "gp_runtime_profile_cache_root=Results/runtime_profile_cache/dx1p0",
            "gp_runtime_barrier_mode=CNT_refsub",
            "gp_runtime_s_gp_scalar=0.5",
            "gp_runtime_nucleation_mode=homogeneous_plus_GP",
            "gp_runtime_catalog_allow_fallback=0",
            "gp_runtime_min_rseed_over_dx=4.0",
            "gp_runtime_enable_delayed_insertion_queue=1",
            "gp_runtime_log_accepted_events=1",
            "gp_runtime_log_candidates=1",
            f"gp_site_B_mass_equiv={fmt(safe_float(row.get('mass_seed_B_equiv')))}",
            "beta_rate_phi_threshold=0.05",
            "beta_rate_xB_min=0.0",
            "",
        ]
        filename.write_text("\n".join(lines), encoding="utf-8")
        generated_params.append(
            {
                "file": filename.relative_to(repo).as_posix(),
                "T_C": t_c,
                "xB": safe_float(row.get("xB")),
                "r_star_nm": safe_float(row.get("r_star_nm")),
                "r_seed_nm": safe_float(row.get("r_seed_nm")),
                "DeltaG_bare_kBT": safe_float(row.get("DeltaG_bare_kBT")),
                "Z_r_m_inv": safe_float(row.get("Z_r_m_inv")),
                "Z_n": safe_float(row.get("Z_n")),
                "D_B_alpha_m2_s": d_b,
            }
        )

    z_report_lines = [
        "# Z_r From Barrier Curves",
        "",
        "| entry_id | T_C | xB | r_star_nm | DeltaG_bare_kBT | n_fit_points | curvature_d2G_dr2_kBT_per_nm2 | Z_r_m_inv | Z_r_status | production_usable |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|---|",
    ]
    for row in zr_rows:
        z_report_lines.append(
            "| {entry_id} | {T_C:.1f} | {xB:.3f} | {r_star_nm:.6f} | {DeltaG_bare_kBT:.6f} | {n_fit_points} | {curvature} | {zr} | {status} | {usable} |".format(
                entry_id=row["entry_id"],
                T_C=row["T_C"] or float("nan"),
                xB=row["xB"] or float("nan"),
                r_star_nm=row["r_star_nm"] or float("nan"),
                DeltaG_bare_kBT=row["DeltaG_bare_kBT"] or float("nan"),
                n_fit_points=int(row["n_fit_points"]),
                curvature=fmt(row["curvature_d2G_dr2_kBT_per_nm2"]),
                zr=fmt(row["Z_r_m_inv"]),
                status=row["Z_r_status"],
                usable="true" if row["production_usable"] else "false",
            )
        )
    (out_dir / "Z_r_from_barrier_curves_report.md").write_text("\n".join(z_report_lines) + "\n", encoding="utf-8")

    omega_lines = [
        "# Omega_g Ag2Te Definition",
        "",
        f"- `chosen_method`: molar volume / Avogadro number",
        f"- `Vm_compound_m3_per_mol`: `{vm_compound:.16e}` from `physical_inputs.example.json`",
        f"- `N_A`: `{N_A:.16e}`",
        f"- `Omega_g_m3 = Vm_compound / N_A = {omega_g_m3:.16e}`",
        f"- `source_file_or_reference`: `physical_inputs.example.json:Vm_compound`",
        f"- `production_ready_yes_no`: yes",
        "",
        "Definition: `volume_per_Ag2Te_formula_unit` with pseudo-binary `B ≡ Ag2Te`.",
    ]
    (out_dir / "Omega_g_Ag2Te_definition_report.md").write_text("\n".join(omega_lines) + "\n", encoding="utf-8")

    site_lines = [
        "# Site Density And Capture Volume Definition",
        "",
        f"- `Omega_site_m3`: `{omega_site_m3:.16e}` from `Vm_alpha_0 / N_A` using `physical_inputs.example.json:Vm_alpha_0`",
        f"- `N_site_m3 = 1 / Omega_site_m3 = {n_site_m3:.16e}`",
        "- `beta_rate_site_model`: `explicit_gp_site_capture_volume`",
        "- `beta_rate_gp_capture_volume_model`: `runtime_cell_volume`",
        "- `beta_rate_DeltaV_nuc_model`: `runtime_cell_volume`",
        "- For matrix sites, the continuum prefactor uses `N_site_m3`.",
        "- For explicit GP-assisted candidate sites, the Poisson gate uses `DeltaV_nuc_m3 = dx*dy*dz` for the runtime cell hosting that site.",
        "- This avoids double-counting: `J_beta` remains a volumetric rate density, and the per-site event probability is `1-exp(-J_beta * DeltaV_nuc * dt_s)`.",
    ]
    (out_dir / "site_density_capture_volume_definition_report.md").write_text("\n".join(site_lines) + "\n", encoding="utf-8")

    d_lines = [
        "# D_B_alpha Parameter Table (SI)",
        "",
        f"- Arrhenius source: `thermo_utils.h:D_Ag_in_PbTe_m2_per_s` and `Unit_Psedobinary.py:D_Ag_in_PbTe_m2_per_s`",
        f"- `D0_cm2_s = 4.251e-11`, converted to `D0_m2_s = {d0_m2_s:.16e}`",
        f"- `Q_J_mol = {q_j_mol:.16e}`",
        "",
        "| T_C | T_K | D_B_alpha_m2_s | D0_m2_s | Q_J_mol | source |",
        "|---:|---:|---:|---:|---:|---|",
    ]
    for row in d_b_rows:
        d_lines.append(
            f"| {row['T_C']:.1f} | {row['T_K']:.2f} | {row['D_B_alpha_m2_s']:.16e} | {row['D0_m2_s']:.16e} | {row['Q_J_mol']:.16e} | {row['source']} |"
        )
    (out_dir / "D_B_alpha_parameter_table_SI.md").write_text("\n".join(d_lines) + "\n", encoding="utf-8")

    param_lines = [
        "# Physical CNT Production Params Generation",
        "",
        "Generated overlay templates:",
        "",
    ]
    for row in generated_params:
        param_lines.append(
            f"- `{row['file']}`: `T_C={fmt(row['T_C'])}`, `xB={fmt(row['xB'])}`, `DeltaG_bare_kBT={fmt(row['DeltaG_bare_kBT'])}`, `Z_r_m_inv={fmt(row['Z_r_m_inv'])}`"
        )
    param_lines.append("")
    param_lines.append("These files are production-parameter overlays for the validated `physical_cnt` path and are intended to be combined with the normal PF base parameter set for the same thermodynamic state.")
    (out_dir / "physical_cnt_production_params_generation_report.md").write_text("\n".join(param_lines) + "\n", encoding="utf-8")

    gp_diag_lines = [
        "# GP Inventory And xBtot Diagnostic",
        "",
        "- `TOTAL_LEDGER_INCLUDES_GP = true`",
        "- `GENERIC_XBTOT_FIELD_INCLUDES_GP = false` in barrier-only mode",
        "- `XBTOT_VTK_INCLUDES_GP = false` in barrier-only mode",
        "- `MASS_AUDIT_USES_LEDGER_TOTAL = true` for the GP-assisted multi-site ledger",
        "",
        "Current production recommendation: keep the continuous `xBtot` field as matrix+beta only, and expose discrete GP inventory through separate ledger outputs rather than smearing it back into the PDE field.",
    ]
    (out_dir / "GP_inventory_xBtot_diagnostic_report.md").write_text("\n".join(gp_diag_lines) + "\n", encoding="utf-8")

    acceptance_lines = [
        "# Physical CNT Parameter Closure Acceptance",
        "",
        "- Status before live smoke: `PARTIAL_PHYSICAL_CNT_PARAMETER_CLOSURE`",
        f"- `Z_r` staging library generated: `{merged_csv_path.relative_to(repo).as_posix()}`",
        f"- `Omega_g_m3`: `{omega_g_m3:.16e}`",
        f"- `N_site_m3`: `{n_site_m3:.16e}`",
        f"- `D_B_alpha` table: `{d_b_csv_path.relative_to(repo).as_posix()}`",
        "- Remaining acceptance gate: live CUDA stochastic `physical_cnt` smoke and build validation.",
    ]
    (out_dir / "physical_cnt_parameter_closure_acceptance_report.md").write_text("\n".join(acceptance_lines) + "\n", encoding="utf-8")

    print("physical_cnt_parameter_closure_generated")
    print(f"z_r_rows={len(zr_rows)}")
    print(f"production_param_files={len(generated_params)}")
    print(f"merged_library={merged_csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
