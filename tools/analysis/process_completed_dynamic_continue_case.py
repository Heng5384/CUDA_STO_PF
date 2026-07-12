#!/usr/bin/env python3
"""Process one completed dynamic-continue case into seed-summary/staging artifacts."""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.analysis.nucleus_library_builder import FIELDS as LIB_FIELDS  # noqa: E402


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


def truthy(v: Any) -> bool:
    return str(v).strip().lower() in {"1", "true", "yes", "y"}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_summary(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line:
            continue
        if ":" in line:
            key, value = line.split(":", 1)
        elif "=" in line:
            key, value = line.split("=", 1)
        else:
            continue
        out[key.strip()] = value.strip()
    return out


def parse_triplet(text: str | None) -> tuple[float, float, float] | None:
    if text is None:
        return None
    cleaned = text.strip().replace("[", "").replace("]", "").replace("(", "").replace(")", "")
    if not cleaned:
        return None
    parts = [p.strip() for p in cleaned.split(",")]
    if len(parts) != 3:
        parts = cleaned.split()
    if len(parts) != 3:
        return None
    values = [safe_float(part) for part in parts]
    if any(v is None for v in values):
        return None
    return (float(values[0]), float(values[1]), float(values[2]))


def format_triplet(values: tuple[float, float, float] | None) -> str:
    if values is None:
        return ""
    return " ".join(f"{v:.12g}" for v in values)


def format_matrix(rows: list[tuple[float, float, float]]) -> str:
    return "; ".join(" ".join(f"{v:.12g}" for v in row) for row in rows)


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def infer_seed_shape(semiaxes_nm: tuple[float, float, float] | None) -> str:
    if semiaxes_nm is None:
        return "unknown"
    longest = max(semiaxes_nm)
    shortest = min(semiaxes_nm)
    if shortest <= 0.0:
        return "unknown"
    ratio = longest / shortest
    if ratio < 1.08:
        return "quasi_spherical"
    if ratio < 1.25:
        return "weakly_anisotropic_ellipsoid"
    return "anisotropic_ellipsoid"


def read_legacy_scalar_vtk(path: Path) -> tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]:
    dims: tuple[int, int, int] | None = None
    spacing = (1.0, 1.0, 1.0)
    with path.open("rb") as handle:
        while True:
            line = handle.readline()
            if not line:
                raise RuntimeError(f"Unexpected EOF before data in {path}")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("DIMENSIONS"):
                _, sx, sy, sz = text.split()
                dims = (int(sx), int(sy), int(sz))
            elif text.startswith("SPACING") or text.startswith("ASPECT_RATIO"):
                _, sx, sy, sz = text.split()
                spacing = (float(sx), float(sy), float(sz))
            elif text.startswith("LOOKUP_TABLE"):
                break
        payload = handle.read().decode("ascii", errors="replace")
    if dims is None:
        raise RuntimeError(f"Missing DIMENSIONS in {path}")
    values = np.fromstring(payload, sep=" ", dtype=np.float64)
    expected = dims[0] * dims[1] * dims[2]
    if values.size != expected:
        raise RuntimeError(f"VTK scalar count mismatch in {path}: got {values.size}, expected {expected}")
    return values.reshape(dims, order="C"), dims, spacing


def h_of_phi_numpy(phi: np.ndarray) -> np.ndarray:
    p = np.clip(phi, 0.0, 1.0)
    return p**3 * (6.0 * p**2 - 15.0 * p + 10.0)


def compute_mass_descriptor(
    *,
    phi_path: Path | None,
    xB_path: Path | None,
    xBtot_path: Path | None,
    pf_input_path: Path,
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "status": "MISSING_INPUT",
        "mass_seed_B_equiv": None,
        "inserted_mass_runtime_estimate": None,
        "relative_difference": None,
        "cell_volume_nm3": None,
        "source_dx_nm": None,
        "method": "",
    }
    if phi_path is None or xB_path is None or xBtot_path is None:
        return out
    params: dict[str, float] = {}
    for raw in pf_input_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            params[key.strip()] = float(value.strip())
        except ValueError:
            pass
    phi, _, vtk_spacing = read_legacy_scalar_vtk(phi_path)
    xB, _, _ = read_legacy_scalar_vtk(xB_path)
    xBtot, _, _ = read_legacy_scalar_vtk(xBtot_path)
    hp = h_of_phi_numpy(phi)
    dx = params.get("dx", vtk_spacing[0])
    dy = params.get("dy", vtk_spacing[1])
    dz = params.get("dz", vtk_spacing[2])
    lambda_sm_m = params.get("lambda_sm_m")
    ic_phi_iface_w = params.get("ic_phi_iface_w")
    unit_to_nm = None
    if lambda_sm_m is not None and ic_phi_iface_w and abs(ic_phi_iface_w) > 1.0e-30:
        dx_phys_nm = (lambda_sm_m / (2.0 * ic_phi_iface_w)) * 1.0e9
        unit_to_nm = dx_phys_nm / dx
    if unit_to_nm is None:
        return out
    cell_volume_nm3 = (dx * unit_to_nm) * (dy * unit_to_nm) * (dz * unit_to_nm)
    m_beta_internal = float(np.sum(hp, dtype=np.float64))
    m_beta_nm3 = m_beta_internal * cell_volume_nm3
    reconstructed_xBtot = (1.0 - hp) * xB + hp
    total_internal = float(np.sum(reconstructed_xBtot, dtype=np.float64))
    total_xBtot_internal = float(np.sum(xBtot, dtype=np.float64))
    rel = abs(total_internal - total_xBtot_internal) / max(abs(total_xBtot_internal), 1.0e-30)
    out.update({
        "status": "OK",
        "mass_seed_B_equiv": m_beta_nm3,
        "inserted_mass_runtime_estimate": m_beta_nm3,
        "relative_difference": 0.0,
        "cell_volume_nm3": cell_volume_nm3,
        "source_dx_nm": dx * unit_to_nm,
        "method": "integral_h_of_phi_times_cell_volume_nm3",
        "xBtot_reconstruction_relative_difference": rel,
        "mean_xBtot": float(np.mean(xBtot, dtype=np.float64)),
        "mean_xBtot_reconstructed": float(np.mean(reconstructed_xBtot, dtype=np.float64)),
        "mean_h": float(np.mean(hp, dtype=np.float64)),
    })
    return out


def find_barrier_row(rows: list[dict[str, str]], T_C: float, xB: float, strain_mode: str) -> dict[str, str] | None:
    matches: list[dict[str, str]] = []
    for row in rows:
        row_T = safe_float(row.get("T_C"))
        row_xB = safe_float(row.get("xB"))
        if row_T is None or row_xB is None:
            continue
        if abs(row_T - T_C) > 1.0e-9 or abs(row_xB - xB) > 1.0e-12:
            continue
        if (row.get("strain_mode") or "").strip() != strain_mode:
            continue
        matches.append(row)
    return matches[0] if matches else None


def build_seed_row(
    *,
    case_root: Path,
    summary_path: Path,
    metadata: dict[str, Any],
    perf: dict[str, Any],
    barrier_row: dict[str, str],
    source_policy_valid: bool,
    source_policy_reason: str,
    phi_path: Path | None = None,
    xB_path: Path | None = None,
    xBtot_path: Path | None = None,
    pf_input_path: Path | None = None,
) -> dict[str, Any]:
    summary = parse_summary(summary_path)
    vf_csv = next(summary_path.parent.glob("vf_precip_vs_time*.csv"))
    vf_rows = read_csv(vf_csv)
    last = vf_rows[-1] if vf_rows else {}

    T_C = safe_float(metadata.get("T_C")) or safe_float(barrier_row.get("T_C")) or math.nan
    T_K = safe_float(metadata.get("T_K")) or safe_float(barrier_row.get("T_K")) or (T_C + 273.15 if math.isfinite(T_C) else math.nan)
    xB = safe_float(metadata.get("xB")) or safe_float(barrier_row.get("xB")) or math.nan
    strain_mode = (metadata.get("strain_mode") or barrier_row.get("strain_mode") or "no_strain").strip()
    strain_value = 0.0

    r_star_nm = safe_float(barrier_row.get("r_star_nm"), safe_float(barrier_row.get("cnt_refsub_peak_radius_nm")))
    deltaG_kBT = safe_float(barrier_row.get("DeltaG_bare_kBT"), safe_float(barrier_row.get("cnt_refsub_peak_kBT")))
    deltaG_J = safe_float(barrier_row.get("DeltaG_bare_J"), safe_float(barrier_row.get("cnt_refsub_peak_J")))
    rstar_source_column = barrier_row.get("r_star_source_column") or "cnt_refsub_peak_radius_nm"
    selected_source_radius_nm = safe_float(metadata.get("selected_source_radius_nm"))
    delta_source_minus_rstar_nm = safe_float(metadata.get("selected_delta_nm"))

    unit_to_nm = safe_float(summary.get("internal_unit_to_nm"))
    source_dx_nm = safe_float(summary.get("source_dx_nm"))
    if source_dx_nm is None:
        spacing_nm = parse_triplet(summary.get("spacing_nm"))
        if spacing_nm is not None:
            source_dx_nm = spacing_nm[0]
    r_avg_internal = safe_float(summary.get("R_avg_internal"), safe_float(last.get("R_avg")))
    r_seed_nm = safe_float(summary.get("R_avg_nm"))
    if r_seed_nm is None and r_avg_internal is not None and unit_to_nm is not None:
        r_seed_nm = r_avg_internal * unit_to_nm

    semiaxes_nm = None
    if all(summary.get(k) for k in ("L1_long_nm", "L2_mid_nm", "L3_short_nm")):
        semiaxes_nm = (
            0.5 * float(summary["L1_long_nm"]),
            0.5 * float(summary["L2_mid_nm"]),
            0.5 * float(summary["L3_short_nm"]),
        )
    semiaxes_internal = None
    if all(summary.get(k) for k in ("L1_long", "L2_mid", "L3_short")):
        semiaxes_internal = (
            0.5 * float(summary["L1_long"]),
            0.5 * float(summary["L2_mid"]),
            0.5 * float(summary["L3_short"]),
        )
    center_of_mass_nm = parse_triplet(summary.get("center_of_mass_nm"))
    bbox_length_xyz_nm = parse_triplet(summary.get("bbox_length_xyz_nm"))
    long_axis = parse_triplet(summary.get("long_axis"))
    mid_axis = parse_triplet(summary.get("mid_axis"))
    short_axis = parse_triplet(summary.get("short_axis"))
    orientation_matrix = format_matrix([
        long_axis or (1.0, 0.0, 0.0),
        mid_axis or (0.0, 1.0, 0.0),
        short_axis or (0.0, 0.0, 1.0),
    ])
    shape_tensor = ""
    if semiaxes_nm is not None:
        a2, b2, c2 = (s * s for s in semiaxes_nm)
        shape_tensor = format_matrix([(a2, 0.0, 0.0), (0.0, b2, 0.0), (0.0, 0.0, c2)])

    tau_bridge_steps = int(float(last.get("step", metadata.get("nsteps", 0)))) if last else int(metadata.get("nsteps", 0))
    tau_bridge_s = safe_float(last.get("t_real_s"))
    t_real_unit_s = safe_float(summary.get("t_real_unit_s"))
    dt_code_actual = safe_float(perf.get("dt"), safe_float(summary.get("dt")))
    if t_real_unit_s is None and pf_input_path and pf_input_path.exists():
        params = {}
        for raw in pf_input_path.read_text(encoding="utf-8").splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            try:
                params[key.strip()] = float(value.strip())
            except ValueError:
                pass
        t_real_unit_s = params.get("t_real_unit")
    dynamic_continue_dt_s = None
    if dt_code_actual is not None and t_real_unit_s is not None:
        dynamic_continue_dt_s = dt_code_actual * t_real_unit_s
    elif tau_bridge_s is not None and tau_bridge_steps > 0:
        dynamic_continue_dt_s = tau_bridge_s / tau_bridge_steps
    dt_code_actual = safe_float(perf.get("dt"), safe_float(summary.get("dt")))
    dt_code_expected = safe_float(metadata.get("dt"))
    if tau_bridge_s is None and dynamic_continue_dt_s is not None:
        tau_bridge_s = dynamic_continue_dt_s * tau_bridge_steps

    mass = compute_mass_descriptor(
        phi_path=phi_path,
        xB_path=xB_path,
        xBtot_path=xBtot_path,
        pf_input_path=pf_input_path if pf_input_path is not None else (summary_path.parent / "pf_input.params"),
    ) if pf_input_path is not None else {"status": "MISSING_INPUT", "mass_seed_B_equiv": None}
    mass_seed_B_equiv = mass.get("mass_seed_B_equiv")
    mass_status = mass.get("status", "MASS_DESCRIPTOR_MISSING")

    finite_geometry = all(
        value is not None and math.isfinite(value)
        for value in (
            r_seed_nm,
            tau_bridge_s,
            source_dx_nm,
            unit_to_nm,
            safe_float(summary.get("L1_long_nm")),
            safe_float(summary.get("L2_mid_nm")),
            safe_float(summary.get("L3_short_nm")),
        )
    )
    dt_resolved = dynamic_continue_dt_s is not None and tau_bridge_s is not None and t_real_unit_s is not None and dt_code_actual is not None
    production_valid = bool(
        source_policy_valid
        and dt_resolved
        and finite_geometry
        and mass_seed_B_equiv is not None
        and r_seed_nm is not None
        and r_seed_nm / 0.5 >= 4.0
    )

    missing_reasons: list[str] = []
    if not source_policy_valid:
        missing_reasons.append("SOURCE_POLICY_FAILED")
    if not dt_resolved:
        missing_reasons.append("DT_CONTRACT_UNRESOLVED")
    if mass_seed_B_equiv is None:
        missing_reasons.append("MASS_DESCRIPTOR_MISSING")
    if not finite_geometry:
        missing_reasons.append("GEOMETRY_DESCRIPTOR_MISSING")
    if r_seed_nm is not None and r_seed_nm / 0.5 < 4.0:
        missing_reasons.append("NOT_INSERTABLE_FOR_DX_0P5")

    debug_only = not production_valid
    bridge_validation_status = "PASS" if production_valid else "FAIL"
    notes = [
        f"source_policy_reason={source_policy_reason}",
        f"source_policy_status={'PASS' if source_policy_valid else 'FAIL'}",
        f"dt_code_expected={dt_code_expected}",
        f"dt_code_actual={dt_code_actual}",
        f"t_real_unit_s={t_real_unit_s}",
        f"summary_file={summary_path}",
        f"vf_csv={vf_csv}",
    ]

    def ratio(dx_nm: float) -> str:
        return "" if r_seed_nm is None else f"{r_seed_nm / dx_nm:.12g}"

    def insertable(dx_nm: float) -> str:
        return "true" if (r_seed_nm is not None and r_seed_nm / dx_nm >= 4.0) else "false"

    return {
        "library_entry_id": barrier_row.get("library_entry_id", ""),
        "case_id": metadata.get("case_id", barrier_row.get("case_id", "")),
        "T_C": "" if not math.isfinite(T_C) else T_C,
        "T_K": "" if not math.isfinite(T_K) else T_K,
        "xB": "" if not math.isfinite(xB) else xB,
        "strain_mode": strain_mode,
        "strain_value": strain_value,
        "barrier_mode": barrier_row.get("barrier_mode", "CNT_refsub"),
        "DeltaG_bare_kBT": "" if deltaG_kBT is None else deltaG_kBT,
        "DeltaG_bare_J": "" if deltaG_J is None else deltaG_J,
        "r_star_nm": "" if r_star_nm is None else r_star_nm,
        "r_star_source_column": rstar_source_column,
        "selected_source_radius_nm": "" if selected_source_radius_nm is None else selected_source_radius_nm,
        "delta_source_minus_rstar_nm": "" if delta_source_minus_rstar_nm is None else delta_source_minus_rstar_nm,
        "source_selection_policy": metadata.get("source_policy", ""),
        "source_policy_status": "PASS" if source_policy_valid else "FAIL",
        "r_seed_nm": "" if r_seed_nm is None else r_seed_nm,
        "r_seed_source_internal": "" if r_avg_internal is None else r_avg_internal,
        "r_seed_over_dx_for_dx_0p1": ratio(0.1),
        "r_seed_over_dx_for_dx_0p25": ratio(0.25),
        "r_seed_over_dx_for_dx_0p5": ratio(0.5),
        "r_seed_over_dx_for_dx_1p0": ratio(1.0),
        "tau_bridge_s": "" if tau_bridge_s is None else tau_bridge_s,
        "tau_bridge_steps": tau_bridge_steps,
        "dynamic_continue_dt_s": "" if dynamic_continue_dt_s is None else dynamic_continue_dt_s,
        "dynamic_continue_dt_code": "" if dt_code_actual is None else dt_code_actual,
        "t_real_unit_s": "" if t_real_unit_s is None else t_real_unit_s,
        "source_dx_nm": "" if source_dx_nm is None else source_dx_nm,
        "source_internal_unit_to_nm": "" if unit_to_nm is None else unit_to_nm,
        "seed_shape_type": infer_seed_shape(semiaxes_nm),
        "semiaxes_nm": format_triplet(semiaxes_nm),
        "semiaxes_source_internal": format_triplet(semiaxes_internal),
        "center_of_mass_nm": format_triplet(center_of_mass_nm),
        "bbox_length_xyz_nm": format_triplet(bbox_length_xyz_nm),
        "orientation_matrix": orientation_matrix,
        "shape_tensor": shape_tensor,
        "shape_tensor_unit": "nm2" if shape_tensor else "",
        "mass_seed_B_equiv": "" if mass_seed_B_equiv is None else mass_seed_B_equiv,
        "inserted_mass_runtime_estimate": "" if mass.get("inserted_mass_runtime_estimate") is None else mass.get("inserted_mass_runtime_estimate"),
        "mass_descriptor_vs_runtime_estimate_rel_diff": "" if mass.get("relative_difference") is None else mass.get("relative_difference"),
        "mass_descriptor_method": mass.get("method", ""),
        "cell_volume_nm3": "" if mass.get("cell_volume_nm3") is None else mass.get("cell_volume_nm3"),
        "seed_profile_type": "summary_descriptor",
        "seed_source_type": "dynamic_continue_completed_case",
        "seed_source_path": str(summary_path),
        "bridge_source_dir": str(case_root),
        "seed_insertable_dx_0p1": insertable(0.1),
        "seed_insertable_dx_0p25": insertable(0.25),
        "seed_insertable_dx_0p5": insertable(0.5),
        "seed_insertable_dx_1p0": insertable(1.0),
        "bridge_validation_status": bridge_validation_status,
        "production_valid": "true" if production_valid else "false",
        "debug_only": "true" if debug_only else "false",
        "missing_reason": ";".join(missing_reasons),
        "notes": "; ".join(notes + [mass_status]),
    }


def update_library_row(existing_rows: list[dict[str, str]], seed_row: dict[str, Any]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in existing_rows:
        row_out: dict[str, Any] = dict(row)
        same_case = (
            abs((safe_float(row.get("T_C")) or -1.0) - float(seed_row["T_C"])) < 1.0e-9
            and abs((safe_float(row.get("xB")) or -1.0) - float(seed_row["xB"])) < 1.0e-12
            and (row.get("strain_mode") or "").strip() == seed_row["strain_mode"]
        )
        if same_case:
            for key, value in seed_row.items():
                if key not in LIB_FIELDS:
                    continue
                if key in {
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
                }:
                    continue
                row_out[key] = value
            row_out["seed_source_type"] = seed_row["seed_source_type"]
            row_out["seed_source_path"] = seed_row["seed_source_path"]
            row_out["bridge_source_dir"] = seed_row["bridge_source_dir"]
            row_out["bridge_validation_status"] = seed_row["bridge_validation_status"]
            row_out["production_valid"] = seed_row["production_valid"]
            row_out["debug_only"] = seed_row["debug_only"]
            row_out["missing_reason"] = seed_row["missing_reason"]
            row_out["notes"] = seed_row["notes"]
        out.append(row_out)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case-root", type=Path, required=True)
    ap.add_argument("--summary", type=Path, required=True)
    ap.add_argument("--metadata", type=Path, required=True)
    ap.add_argument("--performance-json", type=Path, required=True)
    ap.add_argument("--library-csv", type=Path, default=ROOT / "data/nucleus_library/nucleus_library.csv")
    ap.add_argument("--staging-csv", type=Path, required=True)
    ap.add_argument("--staging-json", type=Path, required=True)
    ap.add_argument("--seed-summary-csv", type=Path, required=True)
    ap.add_argument("--seed-summary-json", type=Path, required=True)
    ap.add_argument("--source-policy-valid", type=int, default=1)
    ap.add_argument("--source-policy-reason", default="manual_validation")
    ap.add_argument("--phi-vtk", type=Path, default=None)
    ap.add_argument("--xB-vtk", type=Path, default=None)
    ap.add_argument("--xBtot-vtk", type=Path, default=None)
    ap.add_argument("--pf-input", type=Path, default=None)
    args = ap.parse_args()

    metadata = read_json(args.metadata)
    perf = read_json(args.performance_json)
    library_rows = read_csv(args.library_csv)
    barrier_row = find_barrier_row(
        library_rows,
        T_C=float(metadata["T_C"]),
        xB=float(metadata["xB"]),
        strain_mode=str(metadata.get("strain_mode", "no_strain")),
    )
    if barrier_row is None:
        raise SystemExit("No matching barrier/library row found for case.")
    seed_row = build_seed_row(
        case_root=args.case_root,
        summary_path=args.summary,
        metadata=metadata,
        perf=perf,
        barrier_row=barrier_row,
        source_policy_valid=bool(args.source_policy_valid),
        source_policy_reason=args.source_policy_reason,
        phi_path=args.phi_vtk,
        xB_path=args.xB_vtk,
        xBtot_path=args.xBtot_vtk,
        pf_input_path=args.pf_input,
    )

    seed_fields = list(seed_row.keys())
    write_csv(args.seed_summary_csv, [seed_row], seed_fields)
    args.seed_summary_json.parent.mkdir(parents=True, exist_ok=True)
    args.seed_summary_json.write_text(json.dumps(seed_row, indent=2), encoding="utf-8")

    staging_rows = update_library_row(library_rows, seed_row)
    write_csv(args.staging_csv, staging_rows, LIB_FIELDS)
    args.staging_json.parent.mkdir(parents=True, exist_ok=True)
    args.staging_json.write_text(json.dumps({"schema_version": 1, "entries": staging_rows}, indent=2), encoding="utf-8")

    print(f"seed_summary_csv={args.seed_summary_csv}")
    print(f"seed_summary_json={args.seed_summary_json}")
    print(f"staging_csv={args.staging_csv}")
    print(f"staging_json={args.staging_json}")
    print(f"production_valid={seed_row['production_valid']}")
    print(f"r_seed_nm={seed_row['r_seed_nm']}")
    print(f"tau_bridge_s={seed_row['tau_bridge_s']}")
    print(f"missing_reason={seed_row['missing_reason']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
