#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.analyze_cnt_peak_table import fit_local_peak, parse_summary_file
from tools.analysis.workflow_utils import cnt_summary_filename


def _resolve_repo_path(repo_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return repo_root / p


def _pick_existing_result(case_dir: Path, preferred: Path, pattern: str) -> Path | None:
    if preferred.exists():
        return preferred
    matches = sorted(case_dir.glob(pattern))
    return matches[0] if matches else None


def _pick_cnt_summary(case_dir: Path, preferred: Path) -> Path | None:
    canonical = case_dir / cnt_summary_filename()
    if canonical.exists():
        return canonical
    return _pick_existing_result(case_dir, preferred, "summary*.txt")


def _union_fieldnames(records: list[dict[str, object]]) -> list[str]:
    fieldnames: list[str] = []
    seen: set[str] = set()
    for record in records:
        for key in record.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    return fieldnames


def _safe_float(value: object) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if number == number else None


def _load_last_energy_row(path: Path) -> dict[str, str] | None:
    try:
        with path.open(newline="", encoding="utf-8") as handle:
            rows = list(csv.DictReader(handle))
    except (OSError, csv.Error):
        return None
    return rows[-1] if rows else None


def _load_reference_record(repo_root: Path, row: dict[str, str]) -> tuple[dict[str, object] | None, list[str]]:
    warnings: list[str] = []
    ref_rel = row.get("reference_energy_rel", "")
    if ref_rel:
        ref_path = _resolve_repo_path(repo_root, ref_rel)
        if ref_path.exists():
            try:
                with ref_path.open(newline="", encoding="utf-8") as handle:
                    ref_rows = list(csv.DictReader(handle))
            except (OSError, csv.Error) as exc:
                warnings.append(f"failed to read reference csv {ref_path}: {exc}")
            else:
                if ref_rows:
                    record = dict(ref_rows[0])
                    record["_reference_path"] = str(ref_path)
                    return record, warnings
                warnings.append(f"reference csv is empty: {ref_path}")

    case_dir = _resolve_repo_path(repo_root, row["case_dir_rel"])
    energy_csv = _pick_existing_result(case_dir, _resolve_repo_path(repo_root, row["energy_csv_rel"]), "energy_minimize_*.csv")
    if energy_csv is None:
        warnings.append("missing reference_energy.csv and energy_minimize_*.csv for reference row")
        return None, warnings
    last = _load_last_energy_row(energy_csv)
    if last is None:
        warnings.append(f"missing usable last-row energy data in {energy_csv}")
        return None, warnings
    record = {
        "base_case_tag": row["base_case_tag"],
        "case_dir": str(case_dir),
        "reference_type": row.get("reference_type", "matrix_only_same_strain"),
        "T_input": row.get("T_C", ""),
        "T_K": (_safe_float(row.get("T_C")) + 273.15) if _safe_float(row.get("T_C")) is not None else "",
        "xB0": row.get("xB_out", ""),
        "xB_out": row.get("xB_out", ""),
        "mode": row.get("mode", ""),
        "exx": row.get("E0_xx", ""),
        "eyy": row.get("E0_yy", ""),
        "ezz": row.get("E0_zz", ""),
        "exy": row.get("E0_xy", ""),
        "exz": row.get("E0_xz", ""),
        "eyz": row.get("E0_yz", ""),
        "Nx": row.get("nx", ""),
        "Ny": row.get("ny", ""),
        "Nz": row.get("nz", ""),
        "dx_nm": row.get("pf_dx_nm", ""),
        "dy_nm": row.get("pf_dx_nm", ""),
        "dz_nm": row.get("pf_dx_nm", ""),
        "V_box_m3": "",
        "gamma_Jm2": "",
        "lambda_sm_m": (str(_safe_float(row.get("lambda_sm_nm")) * 1.0e-9) if _safe_float(row.get("lambda_sm_nm")) is not None else ""),
        "w_phys_J_m3": "",
        "F_surf_ref_hat": last.get("F_surf_hat", ""),
        "F_chem_ref_hat": last.get("F_chem_excess_hat", ""),
        "F_el_ref_hat": last.get("F_el_hat", ""),
        "F_total_ref_hat": last.get("F_total_excess_hat", ""),
        "F_total_ref_J": "",
        "mean_phi": "0.0",
        "mean_h": last.get("mean_h", ""),
        "mean_xB": row.get("xB_out", ""),
        "voxel_count": "0",
        "status": "fallback_from_energy_csv",
        "warnings": "reference_energy.csv missing; fell back to F_total_excess_hat from last energy row",
        "_reference_path": str(energy_csv),
    }
    warnings.append("reference_energy.csv missing; using last-row F_total_excess_hat fallback")
    return record, warnings


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize CNT scan results from a single guide CSV.")
    parser.add_argument("--guide-csv", type=Path, required=True, help="guide_cnt_scan.csv generated by setup_cnt_workflow.py")
    parser.add_argument("--repo-root", "--results-root", dest="repo_root", type=Path, default=Path("."), help="repo root containing Results/")
    parser.add_argument("--output", type=Path, default=None, help="output summary csv")
    args = parser.parse_args()

    guide_csv = args.guide_csv.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    workflow_dir = guide_csv.parent.parent
    output = args.output.expanduser().resolve() if args.output else (workflow_dir / "cnt_scan" / "current_results_master_table_fitted.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with guide_csv.open(newline="", encoding="utf-8") as f:
        guide_rows = list(csv.DictReader(f))

    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    meta_by_base: dict[str, dict[str, str]] = {}
    for row in guide_rows:
        base = row["base_case_tag"]
        grouped[base].append(row)
        meta_by_base[base] = row

    records: list[dict[str, object]] = []
    reference_records: list[dict[str, object]] = []
    reference_output = output.parent / "reference_energy.csv"
    for base_case_tag, rows in sorted(grouped.items(), key=lambda kv: (kv[1][0]["mode"], float(kv[1][0]["strain"]))):
        reference_rows = [row for row in rows if row.get("row_type") == "reference"]
        scan_rows = [row for row in rows if row.get("row_type") != "reference"]
        rows_sorted = sorted(scan_rows, key=lambda r: float(r["radius_nm"]))
        points_absolute: list[tuple[float, float]] = []
        points_excess: list[tuple[float, float]] = []
        point_info: dict[float, dict[str, object]] = {}
        reference_record = None
        reference_warnings: list[str] = []
        if reference_rows:
            reference_record, reference_warnings = _load_reference_record(repo_root, reference_rows[0])
            if reference_record is not None:
                reference_records.append(reference_record)
        else:
            reference_warnings.append("missing explicit reference row in guide csv")
        f_ref_hat = _safe_float(reference_record.get("F_total_ref_hat")) if reference_record is not None else None

        for row in rows_sorted:
            radius_nm = float(row["radius_nm"])
            case_dir = _resolve_repo_path(repo_root, row["case_dir_rel"])
            energy_csv = _pick_existing_result(case_dir, _resolve_repo_path(repo_root, row["energy_csv_rel"]), "energy_minimize_*.csv")
            summary_path = _pick_cnt_summary(case_dir, _resolve_repo_path(repo_root, row["summary_rel"]))
            if energy_csv is None:
                continue
            last_row = _load_last_energy_row(energy_csv)
            if not last_row:
                continue
            fcnt = _safe_float(last_row.get("F_total_CNT_hat"))
            if fcnt is None:
                continue
            points_absolute.append((radius_nm, fcnt))
            delta_hat = fcnt - f_ref_hat if f_ref_hat is not None else None
            if delta_hat is not None:
                points_excess.append((radius_nm, delta_hat))
            point_info[radius_nm] = {
                "summary_path": summary_path,
                "case_dir_path": case_dir,
                "energy_csv_path": energy_csv,
                "F_total_CNT_hat": fcnt,
                "DeltaF_CNT_hat": delta_hat,
            }

        peak_found = False
        peak_index = None
        rc_cnt = None
        fcnt_peak_used = None
        rc_fit = None
        fcnt_fit_used = None
        fit_method = ""
        delta_discrete = None
        delta_fit = None
        peak_index_absolute = None
        rc_cnt_absolute = None
        fcnt_peak_absolute = None
        rc_fit_absolute = None
        fcnt_fit_absolute = None
        fit_method_absolute = ""
        peak_index_excess = None
        rc_cnt_excess = None
        fcnt_peak_excess = None
        rc_fit_excess = None
        fcnt_fit_excess = None
        fit_method_excess = ""

        working_points = points_excess if points_excess else points_absolute
        barrier_reference_type = "matrix_only_same_strain" if points_excess else "absolute_peak_no_reference"
        barrier_reference_warning = " | ".join(reference_warnings) if reference_warnings else ""

        if points_absolute:
            vals_absolute = [p[1] for p in points_absolute]
            peak_index_absolute = max(range(len(vals_absolute)), key=lambda i: vals_absolute[i])
            rc_cnt_absolute = points_absolute[peak_index_absolute][0]
            fcnt_peak_absolute = points_absolute[peak_index_absolute][1]
            if 0 < peak_index_absolute < len(points_absolute) - 1:
                rc_fit_absolute, fcnt_fit_absolute, fit_method_absolute = fit_local_peak(points_absolute, peak_index_absolute)

        if points_excess:
            vals_excess = [p[1] for p in points_excess]
            peak_index_excess = max(range(len(vals_excess)), key=lambda i: vals_excess[i])
            rc_cnt_excess = points_excess[peak_index_excess][0]
            fcnt_peak_excess = points_excess[peak_index_excess][1]
            if 0 < peak_index_excess < len(points_excess) - 1:
                rc_fit_excess, fcnt_fit_excess, fit_method_excess = fit_local_peak(points_excess, peak_index_excess)

        if working_points:
            vals = [p[1] for p in working_points]
            peak_index = max(range(len(vals)), key=lambda i: vals[i])
            rc_cnt = working_points[peak_index][0]
            fcnt_peak_used = working_points[peak_index][1]
            if 0 < peak_index < len(working_points) - 1:
                peak_found = True
                rc_fit, fcnt_fit_used, fit_method = fit_local_peak(working_points, peak_index)
                delta_discrete = rc_cnt - float(meta_by_base[base_case_tag]["rc_schur_nm"])
                if rc_fit is not None:
                    delta_fit = rc_fit - float(meta_by_base[base_case_tag]["rc_schur_nm"])

        summary_data: dict[str, object] = {}
        if rc_cnt is not None:
            summary_file = point_info.get(rc_cnt, {}).get("summary_path")
            if isinstance(summary_file, Path) and summary_file.exists():
                summary_data = parse_summary_file(summary_file)
            else:
                summary_data = {"summary_path": str(summary_file) if summary_file else ""}

        expected_points = len(rows_sorted)
        available_points = len(points_absolute)
        if available_points == 0:
            status = "not_started"
        elif available_points < expected_points:
            status = "partial_peak_found" if peak_found else "partial_no_peak_yet"
        else:
            status = "complete_peak_found" if peak_found else "complete_no_peak_in_window"
        if f_ref_hat is None:
            status = f"{status}_missing_reference"

        meta = meta_by_base[base_case_tag]
        record = {
            "workflow_name": meta["workflow_name"],
            "guide_csv": str(guide_csv),
            "base_case_tag": base_case_tag,
            "mode": meta["mode"],
            "strain": float(meta["strain"]),
            "T_C": float(meta["T_C"]),
            "xB_out": float(meta["xB_out"]),
            "rc_schur_nm": float(meta["rc_schur_nm"]),
            "window_nm": float(meta["window_nm"]),
            "step_nm": float(meta["step_nm"]),
            "expected_points": expected_points,
            "available_points": available_points,
            "peak_found": int(peak_found),
            "rc_cnt_nm": rc_cnt,
            "delta_cnt_minus_schur_nm": delta_discrete,
            "F_CNT_peak_hat": fcnt_peak_used,
            "rc_cnt_fit_nm": rc_fit,
            "delta_fit_minus_schur_nm": delta_fit,
            "F_CNT_fit_hat": fcnt_fit_used,
            "fit_method": fit_method,
            "peak_index": peak_index,
            "F_ref_hat_same_strain": f_ref_hat,
            "F_ref_source": reference_record.get("_reference_path", "") if reference_record is not None else "",
            "F_ref_confidence": meta.get("reference_confidence", "high") if f_ref_hat is not None else "missing",
            "F_CNT_peak_hat_absolute": fcnt_peak_absolute,
            "F_CNT_peak_hat_excess": fcnt_peak_excess,
            "F_CNT_peak_hat_used": fcnt_peak_used,
            "F_CNT_fit_hat_absolute": fcnt_fit_absolute,
            "F_CNT_fit_hat_excess": fcnt_fit_excess,
            "F_CNT_fit_hat_used": fcnt_fit_used,
            "rc_cnt_nm_absolute": rc_cnt_absolute,
            "rc_cnt_nm_excess": rc_cnt_excess,
            "rc_cnt_fit_nm_absolute": rc_fit_absolute,
            "rc_cnt_fit_nm_excess": rc_fit_excess,
            "fit_method_absolute": fit_method_absolute,
            "fit_method_excess": fit_method_excess,
            "barrier_reference_type": barrier_reference_type,
            "barrier_reference_warning": barrier_reference_warning,
            "status": status,
        }
        record.update(summary_data)
        records.append(record)

    with output.open("w", newline="", encoding="utf-8") as f:
        if records:
            writer = csv.DictWriter(f, fieldnames=_union_fieldnames(records))
            writer.writeheader()
            writer.writerows(records)
        else:
            f.write("")

    with reference_output.open("w", newline="", encoding="utf-8") as f:
        if reference_records:
            writer = csv.DictWriter(f, fieldnames=_union_fieldnames(reference_records))
            writer.writeheader()
            writer.writerows(reference_records)
        else:
            f.write("")

    print(output)
    print(reference_output)
    print(f"records={len(records)}")
    print(f"complete_peak_found={sum(1 for r in records if r['status'] == 'complete_peak_found')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
