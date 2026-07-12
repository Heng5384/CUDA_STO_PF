#!/usr/bin/env python3
"""Build a unified nucleus library from CNT/minimize/dynamic-continue outputs."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_BARRIER = ROOT / "Results/workflows/no_strain_400cube_dx0p1_cnt_sweep/summary_reports/real_unit_barriers/barrier_real_units_by_case.csv"
OUT_DIR = ROOT / "data/nucleus_library"
REPORT_DIR = ROOT / "reports/nucleus_library_workflow"

FIELDS = [
    "library_entry_id",
    "case_id",
    "source_priority",
    "T_C",
    "T_K",
    "xB",
    "strain_mode",
    "strain_value",
    "barrier_mode",
    "DeltaG_bare_kBT",
    "DeltaG_bare_J",
    "DeltaG_eff_kBT_default",
    "r_star_nm",
    "r_star_source_column",
    "Z_n",
    "Z_n_source",
    "Z_r",
    "Z_r_source",
    "Z_r_status",
    "Z_policy",
    "r_seed_nm",
    "source_dx_nm",
    "source_internal_unit_to_nm",
    "r_seed_source_internal",
    "semiaxes_source_internal",
    "r_seed_over_dx_for_dx_0p1",
    "r_seed_over_dx_for_dx_0p25",
    "r_seed_over_dx_for_dx_0p5",
    "r_seed_over_dx_for_dx_1p0",
    "tau_bridge_s",
    "tau_bridge_steps",
    "dt_code",
    "t_real_unit_s",
    "tau_bridge_code_time",
    "dynamic_continue_dt_s",
    "seed_shape_type",
    "semiaxes_nm",
    "center_of_mass_nm",
    "bbox_length_xyz_nm",
    "orientation_matrix",
    "shape_tensor",
    "shape_tensor_unit",
    "mass_seed_B_equiv",
    "seed_profile_type",
    "seed_source_type",
    "seed_source_path",
    "bridge_source_dir",
    "seed_insertable_dx_0p1",
    "seed_insertable_dx_0p25",
    "seed_insertable_dx_0p5",
    "seed_insertable_dx_1p0",
    "bridge_validation_status",
    "production_valid",
    "debug_only",
    "missing_reason",
    "notes",
]

RSTAR_PRIORITY = [
    "cnt_refsub_peak_radius_nm",
    "cnt_peak_radius_nm",
    "barrier_peak_radius_nm",
    "cnt_absolute_peak_radius_nm",
    "r_star_nm",
    "r_star_cnt_nm",
    "r_eff_star_nm",
    "rc_schur_nm",
]


def safe_float(v: Any, default: float | None = None) -> float | None:
    if v is None:
        return default
    text = str(v).strip()
    if not text or text.lower() in {"nan", "none", "null", "na"}:
        return default
    try:
        out = float(text)
    except ValueError:
        return default
    return out if math.isfinite(out) else default


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


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str] = FIELDS) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def bool_text(value: bool) -> str:
    return "true" if value else "false"


def rstar_from_barrier_row(row: dict[str, str]) -> tuple[float | None, str]:
    for name in RSTAR_PRIORITY:
        value = safe_float(row.get(name))
        if value is not None:
            return value, name
    return None, "missing"


def strain_mode_from_row(row: dict[str, Any]) -> str:
    strain = safe_float(row.get("strain"), safe_float(row.get("strain_value"), 0.0)) or 0.0
    text = " ".join(str(row.get(k, "")) for k in ("case_id", "base_case_tag", "workflow_name", "seed_source_path")).lower()
    if abs(strain) > 1.0e-12:
        return "external_strain"
    if "no_strain" in text or "s000" in text:
        return "no_strain"
    if "exx" in text and "s000" not in text:
        return "external_strain"
    return "no_strain"


def find_dynamic_continue_summaries(root: Path) -> dict[str, Path]:
    out: dict[str, Path] = {}
    for pattern in ("Results/**/continue_dyn_*/summary*.txt", "Results/**/continue_dyn_*/summary.txt"):
        for path in root.glob(pattern):
            tag = path.parent.parent.name
            out.setdefault(tag, path)
    return out


def parse_summary(path: Path) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for line in path.read_text(errors="ignore").splitlines():
        if "=" in line:
            key, val = line.split("=", 1)
        elif ":" in line:
            key, val = line.split(":", 1)
        else:
            continue
        key = key.strip().lower().replace(" ", "_")
        values[key] = val.strip()
    return values


def parse_float_triplet(text: Any) -> tuple[float, float, float] | None:
    if text is None:
        return None
    raw = str(text).strip()
    if not raw:
        return None
    cleaned = raw.replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    parts = [p.strip() for p in cleaned.split(",")]
    if len(parts) != 3:
        parts = cleaned.split()
    if len(parts) != 3:
        return None
    vals = [safe_float(p) for p in parts]
    if any(v is None for v in vals):
        return None
    return float(vals[0]), float(vals[1]), float(vals[2])


def format_triplet(values: tuple[float, float, float] | None) -> str:
    if values is None:
        return ""
    return " ".join(f"{v:.12g}" for v in values)


def mean_triplet(values: tuple[float, float, float] | None) -> float | None:
    if values is None:
        return None
    return sum(values) / 3.0


def orientation_matrix_from_summary(values: dict[str, Any]) -> str:
    axes = [
        parse_float_triplet(values.get("long_axis")),
        parse_float_triplet(values.get("mid_axis")),
        parse_float_triplet(values.get("short_axis")),
    ]
    if any(axis is None for axis in axes):
        return ""
    return "; ".join(" ".join(f"{v:.12g}" for v in axis) for axis in axes if axis is not None)


def semiaxes_nm_from_summary(values: dict[str, Any]) -> tuple[float, float, float] | None:
    direct = parse_float_triplet(values.get("semiaxes_nm"))
    if direct is not None:
        return direct
    l1 = safe_float(values.get("l1_long_nm"))
    l2 = safe_float(values.get("l2_mid_nm"))
    l3 = safe_float(values.get("l3_short_nm"))
    if all(v is not None and v > 0.0 for v in (l1, l2, l3)):
        return (0.5 * float(l1), 0.5 * float(l2), 0.5 * float(l3))
    bbox = parse_float_triplet(values.get("bbox_length_xyz_nm"))
    if bbox is not None and all(v > 0.0 for v in bbox):
        return tuple(0.5 * v for v in bbox)
    return None


def semiaxes_internal_from_summary(values: dict[str, Any]) -> tuple[float, float, float] | None:
    direct = parse_float_triplet(values.get("semiaxes_source_internal"))
    if direct is not None:
        return direct
    l1 = safe_float(values.get("l1_long"))
    l2 = safe_float(values.get("l2_mid"))
    l3 = safe_float(values.get("l3_short"))
    if all(v is not None and v > 0.0 for v in (l1, l2, l3)):
        return (0.5 * float(l1), 0.5 * float(l2), 0.5 * float(l3))
    bbox = parse_float_triplet(values.get("bbox_length_xyz"))
    if bbox is not None and all(v > 0.0 for v in bbox):
        return tuple(0.5 * v for v in bbox)
    return None


def source_dx_nm_from_summary(values: dict[str, Any], unit_to_nm: float | None) -> float | None:
    direct = safe_float(values.get("source_dx_nm"))
    if direct is not None and direct > 0.0:
        return direct
    spacing_nm = parse_float_triplet(values.get("spacing_nm"))
    if spacing_nm is not None and spacing_nm[0] > 0.0:
        return float(spacing_nm[0])
    spacing_internal = parse_float_triplet(values.get("spacing_sim_units"))
    if spacing_internal is not None and unit_to_nm is not None and unit_to_nm > 0.0:
        return float(spacing_internal[0]) * unit_to_nm
    return None


def shape_tensor_from_semiaxes_nm(semiaxes_nm: tuple[float, float, float] | None) -> str:
    if semiaxes_nm is None:
        return ""
    vals = [v * v for v in semiaxes_nm]
    return "; ".join(
        " ".join(f"{val:.12g}" for val in row)
        for row in (
            (vals[0], 0.0, 0.0),
            (0.0, vals[1], 0.0),
            (0.0, 0.0, vals[2]),
        )
    )


def build_library(barrier_path: Path = DEFAULT_BARRIER) -> list[dict[str, Any]]:
    barriers = read_csv(barrier_path)
    continue_summaries = find_dynamic_continue_summaries(ROOT)
    rows: list[dict[str, Any]] = []
    for idx, row in enumerate(barriers, start=1):
        r_star, rstar_col = rstar_from_barrier_row(row)
        case_id = row.get("base_case_tag") or row.get("workflow_name") or f"barrier_case_{idx}"
        T_C = safe_float(row.get("T_C"))
        T_K = safe_float(row.get("T_K"), (T_C + 273.15 if T_C is not None else None))
        xB = safe_float(row.get("xB"))
        strain_value = safe_float(row.get("strain"), 0.0) or 0.0
        strain_mode = strain_mode_from_row({**row, "case_id": case_id})
        barrier_kBT = safe_float(row.get("cnt_refsub_peak_kBT"), safe_float(row.get("DeltaG_bare_kBT")))
        barrier_J = safe_float(row.get("cnt_refsub_peak_J"), safe_float(row.get("DeltaG_bare_J")))
        omega_g_m3 = safe_float(row.get("beta_rate_Omega_g_m3"), safe_float(row.get("Omega_g_m3")))
        z_n = safe_float(row.get("Z_n"))
        z_r = safe_float(row.get("Z_r_strict"), safe_float(row.get("Z_r_1_m"), safe_float(row.get("Z_r"))))
        if z_n is None:
            z_n = z_n_from_z_r(z_r, omega_g_m3, r_star)
        z_n_source = ""
        z_r_source = ""
        z_r_status = ""
        if z_n is not None:
            if safe_float(row.get("Z_n")) is not None:
                z_n_source = "Z_n"
            elif z_r is not None:
                z_n_source = "derived_from_Z_r"
        if z_r is not None:
            if safe_float(row.get("Z_r_strict")) is not None:
                z_r_source = "Z_r_strict"
                z_r_status = "strict"
            elif safe_float(row.get("Z_r_1_m")) is not None or safe_float(row.get("Z_r")) is not None:
                z_r_source = "Z_r_1_m"
                z_r_status = "strict_or_precomputed"
        elif safe_float(row.get("Z_r_diagnostic")) is not None:
            z_r = safe_float(row.get("Z_r_diagnostic"))
            z_r_source = "Z_r_diagnostic"
            z_r_status = "diagnostic_only"
        else:
            z_r_status = str(row.get("strict_status") or "missing")
        dyn_summary = None
        dyn_key = ""
        for key, path in continue_summaries.items():
            if case_id in str(path) or case_id in key:
                dyn_summary = path
                dyn_key = key
                break
        dyn_values = parse_summary(dyn_summary) if dyn_summary else {}
        source_internal_unit_to_nm = safe_float(dyn_values.get("internal_unit_to_nm"))
        source_dx_nm = source_dx_nm_from_summary(dyn_values, source_internal_unit_to_nm)
        semiaxes_nm = semiaxes_nm_from_summary(dyn_values)
        semiaxes_internal = semiaxes_internal_from_summary(dyn_values)
        r_seed = safe_float(dyn_values.get("r_seed_nm"), safe_float(dyn_values.get("r_eff_nm")))
        if r_seed is None:
            r_seed = mean_triplet(semiaxes_nm)
        r_seed_source_internal = safe_float(dyn_values.get("r_seed_source_internal"))
        if r_seed_source_internal is None and r_seed is not None and source_internal_unit_to_nm and source_internal_unit_to_nm > 0.0:
            r_seed_source_internal = r_seed / source_internal_unit_to_nm
        tau_steps = safe_float(dyn_values.get("tau_bridge_steps"), safe_float(dyn_values.get("steps")))
        dt_s = safe_float(dyn_values.get("dynamic_continue_dt_s"), safe_float(dyn_values.get("dt_s")))
        tau_s = safe_float(dyn_values.get("tau_bridge_s"))
        if tau_s is None and tau_steps is not None and dt_s is not None:
            tau_s = tau_steps * dt_s
        seed_source_type = "dynamic_continue_summary" if dyn_summary else "missing_bridge"
        seed_source_path = str(dyn_summary.relative_to(ROOT)) if dyn_summary else ""
        missing_reason = ""
        debug_only = False
        if r_star is None or barrier_kBT is None:
            missing_reason = "BARRIER_OR_RSTAR_MISSING"
        elif r_seed is None or tau_s is None:
            missing_reason = "DYNAMIC_CONTINUE_BRIDGE_MISSING"
            debug_only = True
        def ratio(dx: float) -> str:
            return "" if r_seed is None else f"{r_seed / dx:.12g}"
        def insertable(dx: float) -> str:
            return bool_text(r_seed is not None and r_seed / dx >= 4.0 and tau_s is not None)
        production_valid = bool(
            barrier_kBT is not None
            and r_star is not None
            and r_seed is not None
            and tau_s is not None
            and r_seed / 0.1 >= 4.0
            and seed_source_type != "missing_bridge"
        )
        rows.append({
            "library_entry_id": f"nlib_{idx:05d}",
            "case_id": case_id,
            "source_priority": 0 if production_valid else 50,
            "T_C": "" if T_C is None else T_C,
            "T_K": "" if T_K is None else T_K,
            "xB": "" if xB is None else xB,
            "strain_mode": strain_mode,
            "strain_value": strain_value,
            "barrier_mode": "CNT_refsub",
            "DeltaG_bare_kBT": "" if barrier_kBT is None else barrier_kBT,
            "DeltaG_bare_J": "" if barrier_J is None else barrier_J,
            "DeltaG_eff_kBT_default": "" if barrier_kBT is None else barrier_kBT,
            "r_star_nm": "" if r_star is None else r_star,
            "r_star_source_column": rstar_col,
            "Z_n": "" if z_n is None else z_n,
            "Z_n_source": z_n_source,
            "Z_r": "" if z_r is None else z_r,
            "Z_r_source": z_r_source,
            "Z_r_status": z_r_status,
            "Z_policy": "dimensionless_Z_n_with_beta_star_1_per_s",
            "r_seed_nm": "" if r_seed is None else r_seed,
            "source_dx_nm": "" if source_dx_nm is None else source_dx_nm,
            "source_internal_unit_to_nm": "" if source_internal_unit_to_nm is None else source_internal_unit_to_nm,
            "r_seed_source_internal": "" if r_seed_source_internal is None else r_seed_source_internal,
            "semiaxes_source_internal": format_triplet(semiaxes_internal),
            "r_seed_over_dx_for_dx_0p1": ratio(0.1),
            "r_seed_over_dx_for_dx_0p25": ratio(0.25),
            "r_seed_over_dx_for_dx_0p5": ratio(0.5),
            "r_seed_over_dx_for_dx_1p0": ratio(1.0),
            "tau_bridge_s": "" if tau_s is None else tau_s,
            "tau_bridge_steps": "" if tau_steps is None else tau_steps,
            "dt_code": dyn_values.get("dynamic_continue_dt_code", dyn_values.get("dt_code", "")),
            "t_real_unit_s": dyn_values.get("t_real_unit_s", ""),
            "tau_bridge_code_time": (
                ""
                if tau_steps is None or safe_float(dyn_values.get("dynamic_continue_dt_code", dyn_values.get("dt_code"))) is None
                else tau_steps * safe_float(dyn_values.get("dynamic_continue_dt_code", dyn_values.get("dt_code")))
            ),
            "dynamic_continue_dt_s": "" if dt_s is None else dt_s,
            "seed_shape_type": dyn_values.get("seed_shape_type", dyn_values.get("shape_type", "")),
            "semiaxes_nm": format_triplet(semiaxes_nm),
            "center_of_mass_nm": dyn_values.get("center_of_mass_nm", ""),
            "bbox_length_xyz_nm": dyn_values.get("bbox_length_xyz_nm", ""),
            "orientation_matrix": dyn_values.get("orientation_matrix", orientation_matrix_from_summary(dyn_values)),
            "shape_tensor": dyn_values.get("shape_tensor", shape_tensor_from_semiaxes_nm(semiaxes_nm)),
            "shape_tensor_unit": dyn_values.get("shape_tensor_unit", "nm2" if semiaxes_nm is not None else ""),
            "mass_seed_B_equiv": dyn_values.get("mass_seed_B_equiv", ""),
            "seed_profile_type": "summary_descriptor" if dyn_summary else "",
            "seed_source_type": seed_source_type,
            "seed_source_path": seed_source_path,
            "bridge_source_dir": str(dyn_summary.parent.relative_to(ROOT)) if dyn_summary else "",
            "seed_insertable_dx_0p1": insertable(0.1),
            "seed_insertable_dx_0p25": insertable(0.25),
            "seed_insertable_dx_0p5": insertable(0.5),
            "seed_insertable_dx_1p0": insertable(1.0),
            "bridge_validation_status": "PASS" if production_valid else ("MISSING" if not dyn_summary else "PARTIAL"),
            "production_valid": bool_text(production_valid),
            "debug_only": bool_text(debug_only),
            "missing_reason": missing_reason,
            "notes": f"dynamic_key={dyn_key}" if dyn_key else "bridge row retained for future fill",
        })
    return rows


def write_reports(rows: list[dict[str, Any]], barrier_path: Path) -> None:
    REPORT_DIR.mkdir(parents=True, exist_ok=True)
    prod = [r for r in rows if r["production_valid"] == "true"]
    debug = [r for r in rows if r["debug_only"] == "true"]
    missing = [r for r in rows if r["missing_reason"] == "DYNAMIC_CONTINUE_BRIDGE_MISSING"]
    t380 = [r for r in rows if str(r["T_C"]) in {"380.0", "380"} and abs(float(r["xB"]) - 0.04) < 1e-12]
    (REPORT_DIR / "nucleus_library_build_report.md").write_text(
        "# Nucleus Library Build Report\n\n"
        f"- barrier_cases_found: {len(rows)}\n"
        f"- library_entries_created: {len(rows)}\n"
        f"- production_valid_entries: {len(prod)}\n"
        f"- debug_only_entries: {len(debug)}\n"
        f"- missing_dynamic_continue_entries: {len(missing)}\n"
        "- VTK_only_entries: 0\n"
        f"- T380_xB004_entry_status: {t380[0]['missing_reason'] if t380 else 'MISSING'}\n"
        f"- barrier_source: `{barrier_path}`\n",
        encoding="utf-8",
    )


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--barrier", type=Path, default=DEFAULT_BARRIER)
    ap.add_argument("--csv", type=Path, default=OUT_DIR / "nucleus_library.csv")
    ap.add_argument("--json", type=Path, default=OUT_DIR / "nucleus_library.json")
    args = ap.parse_args()
    rows = build_library(args.barrier)
    write_csv(args.csv, rows)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps({"schema_version": 1, "entries": rows}, indent=2), encoding="utf-8")
    write_reports(rows, args.barrier)
    print(f"nucleus_library_csv={args.csv}")
    print(f"nucleus_library_json={args.json}")
    print(f"library_entries_count={len(rows)}")
    print(f"production_valid_entries_count={sum(r['production_valid']=='true' for r in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
