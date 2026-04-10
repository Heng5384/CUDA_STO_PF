#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from pathlib import Path

import numpy as np


def canon(tag: str) -> str:
    tag = tag.replace("_s010_", "_s0p01_")
    tag = tag.replace("_sm010_", "_sm0p01_")
    if tag.endswith("_s010"):
        tag = tag[:-5] + "_s0p01"
    if tag.endswith("_sm010"):
        tag = tag[:-6] + "_sm0p01"
    return tag


def fit_local_peak(points: list[tuple[float, float]], peak_index: int) -> tuple[float | None, float | None, str]:
    if len(points) < 4 or not (0 < peak_index < len(points) - 1):
        return None, None, "insufficient"

    left = max(0, peak_index - 2)
    right = min(len(points), peak_index + 3)
    local = points[left:right]

    # Prefer quartic on 5 points, otherwise cubic on 4 points.
    degree = min(4 if len(local) >= 5 else 3, len(local) - 1)
    if degree < 3:
        return None, None, "insufficient"

    xs = np.array([p[0] for p in local], dtype=float)
    ys = np.array([p[1] for p in local], dtype=float)
    x0 = points[peak_index][0]
    xt = xs - x0

    try:
        coeff = np.polyfit(xt, ys, degree)
    except np.linalg.LinAlgError:
        return None, None, f"poly{degree}_failed"

    poly = np.poly1d(coeff)
    dpoly = poly.deriv()
    roots = dpoly.r

    lo = float(xt.min())
    hi = float(xt.max())
    candidates = []
    for root in roots:
        if abs(root.imag) > 1e-8:
            continue
        xr = float(root.real)
        if lo <= xr <= hi:
            candidates.append(xr)

    if not candidates:
        return None, None, f"poly{degree}_no_root"

    best_x = max(candidates, key=lambda xr: float(poly(xr)))
    return x0 + best_x, float(poly(best_x)), f"poly{degree}"


def parse_vector(text: str) -> tuple[float, ...]:
    nums = [float(x.strip()) for x in text.strip().strip("[]").split(",")]
    return tuple(nums)


def parse_summary_file(path: Path) -> dict[str, object]:
    data: dict[str, object] = {
        "summary_path": str(path),
        "voxel_count": None,
        "bbox_x": None,
        "bbox_y": None,
        "bbox_z": None,
        "long_axis_x": None,
        "long_axis_y": None,
        "long_axis_z": None,
        "mid_axis_x": None,
        "mid_axis_y": None,
        "mid_axis_z": None,
        "short_axis_x": None,
        "short_axis_y": None,
        "short_axis_z": None,
        "L1_long": None,
        "L2_mid": None,
        "L3_short": None,
        "L1_over_L3": None,
        "L2_over_L3": None,
        "L1_over_L2": None,
        "long_vs_x_deg": None,
        "long_vs_y_deg": None,
        "long_vs_z_deg": None,
        "mid_vs_x_deg": None,
        "mid_vs_y_deg": None,
        "mid_vs_z_deg": None,
        "short_vs_x_deg": None,
        "short_vs_y_deg": None,
        "short_vs_z_deg": None,
        "short_face_normal_x": None,
        "short_face_normal_y": None,
        "short_face_normal_z": None,
        "short_face_angle_x_deg": None,
        "short_face_angle_y_deg": None,
        "short_face_angle_z_deg": None,
        "short_face_vs_long_deg": None,
        "short_face_vs_mid_deg": None,
        "short_face_vs_short_deg": None,
    }
    current_face = None
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.rstrip()
        stripped = line.strip()
        if not stripped:
            continue
        if stripped in {"+long face", "-long face", "+mid face", "-mid face", "+short face", "-short face"}:
            current_face = stripped
            continue
        if ":" not in stripped:
            continue
        key, value = [x.strip() for x in stripped.split(":", 1)]
        try:
            if key == "voxel_count":
                data["voxel_count"] = int(value)
            elif key == "bbox_length_xyz":
                vx, vy, vz = parse_vector(value)
                data["bbox_x"], data["bbox_y"], data["bbox_z"] = vx, vy, vz
            elif key == "long_axis":
                vx, vy, vz = parse_vector(value)
                data["long_axis_x"], data["long_axis_y"], data["long_axis_z"] = vx, vy, vz
            elif key == "mid_axis":
                vx, vy, vz = parse_vector(value)
                data["mid_axis_x"], data["mid_axis_y"], data["mid_axis_z"] = vx, vy, vz
            elif key == "short_axis":
                vx, vy, vz = parse_vector(value)
                data["short_axis_x"], data["short_axis_y"], data["short_axis_z"] = vx, vy, vz
            elif key == "L1_long":
                data["L1_long"] = float(value)
            elif key == "L2_mid":
                data["L2_mid"] = float(value)
            elif key == "L3_short":
                data["L3_short"] = float(value)
            elif key == "L1/L3":
                data["L1_over_L3"] = float(value)
            elif key == "L2/L3":
                data["L2_over_L3"] = float(value)
            elif key == "L1/L2":
                data["L1_over_L2"] = float(value)
            elif key == "long_vs_x":
                data["long_vs_x_deg"] = float(value)
            elif key == "long_vs_y":
                data["long_vs_y_deg"] = float(value)
            elif key == "long_vs_z":
                data["long_vs_z_deg"] = float(value)
            elif key == "mid_vs_x":
                data["mid_vs_x_deg"] = float(value)
            elif key == "mid_vs_y":
                data["mid_vs_y_deg"] = float(value)
            elif key == "mid_vs_z":
                data["mid_vs_z_deg"] = float(value)
            elif key == "short_vs_x":
                data["short_vs_x_deg"] = float(value)
            elif key == "short_vs_y":
                data["short_vs_y_deg"] = float(value)
            elif key == "short_vs_z":
                data["short_vs_z_deg"] = float(value)
            elif current_face == "+short face" and key == "mean normal":
                vx, vy, vz = parse_vector(value)
                data["short_face_normal_x"], data["short_face_normal_y"], data["short_face_normal_z"] = vx, vy, vz
            elif current_face == "+short face" and key == "angle with x":
                data["short_face_angle_x_deg"] = float(value)
            elif current_face == "+short face" and key == "angle with y":
                data["short_face_angle_y_deg"] = float(value)
            elif current_face == "+short face" and key == "angle with z":
                data["short_face_angle_z_deg"] = float(value)
            elif current_face == "+short face" and key == "angle with long axis":
                data["short_face_vs_long_deg"] = float(value)
            elif current_face == "+short face" and key == "angle with mid axis":
                data["short_face_vs_mid_deg"] = float(value)
            elif current_face == "+short face" and key == "angle with short axis":
                data["short_face_vs_short_deg"] = float(value)
        except Exception:
            continue
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description="Build CNT peak summary table with local cubic/quartic fitting.")
    parser.add_argument("--case-dir", default="Results_scan/constraint_cnt_strictref_test_T400_x0_0p03")
    parser.add_argument("--results-root", default="Results")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    case_dir = Path(args.case_dir)
    results_root = Path(args.results_root)
    output = Path(args.output) if args.output else case_dir / "current_results_master_table_fitted.csv"

    intended: dict[str, dict[str, object]] = {}
    for csv_path in sorted(case_dir.glob("case_*_L5.csv")):
        with csv_path.open("r", encoding="utf-8", newline="") as f:
            row = next(csv.DictReader(f))
        base = canon(row["base_case_tag"])
        window = float(row["window_nm"])
        step = float(row["step_nm"])
        intended[base] = {
            "original_base_case_tag": row["base_case_tag"],
            "mode": row["mode"],
            "strain": float(row["strain"]),
            "rc_schur_nm": float(row["rc_schur_nm"]),
            "window_nm": window,
            "step_nm": step,
            "expected_points": int(round(2 * window / step)) + 1,
        }

    pattern = re.compile(r"^(cntcon_T400_xB0p030_strictref_.*)_r(\d+)p(\d+)$")
    series: dict[str, list[tuple[float, float]]] = {k: [] for k in intended}
    point_info: dict[str, dict[float, dict[str, object]]] = {k: {} for k in intended}
    for energy_path in results_root.glob(
        "chel_T400_cuda_400x400x400_dt0.1_steps30000_r*.030/cntcon_T400_xB0p030_strictref*/energy_minimize_*.csv"
    ):
        name = energy_path.parent.name
        match = pattern.match(name)
        if not match:
            continue
        base = canon(match.group(1))
        if base not in intended:
            continue
        radius_nm = float(f"{match.group(2)}.{match.group(3)}")
        with energy_path.open("r", encoding="utf-8", newline="") as f:
            rows = list(csv.DictReader(f))
        if not rows or "F_total_CNT_hat" not in rows[-1]:
            continue
        summary_candidates = list(energy_path.parent.glob("summary_*.txt"))
        summary_path = str(summary_candidates[0]) if summary_candidates else ""
        series[base].append((radius_nm, float(rows[-1]["F_total_CNT_hat"])))
        point_info[base][radius_nm] = {
            "summary_path": summary_path,
            "case_dir_path": str(energy_path.parent),
            "energy_csv_path": str(energy_path),
        }

    records = []
    for base, meta in sorted(intended.items(), key=lambda kv: (kv[1]["mode"], kv[1]["strain"])):
        points = sorted(series.get(base, []), key=lambda x: x[0])
        available = len(points)
        rc_cnt = None
        fcnt_peak = None
        peak_found = False
        fit_method = ""
        rc_fit = None
        fcnt_fit = None
        delta_discrete = None
        delta_fit = None
        peak_index = None

        if points:
            vals = [p[1] for p in points]
            peak_index = max(range(len(vals)), key=lambda i: vals[i])
            rc_cnt = points[peak_index][0]
            fcnt_peak = points[peak_index][1]
            if 0 < peak_index < len(points) - 1:
                peak_found = True
                delta_discrete = rc_cnt - float(meta["rc_schur_nm"])
                rc_fit, fcnt_fit, fit_method = fit_local_peak(points, peak_index)
                if rc_fit is not None:
                    delta_fit = rc_fit - float(meta["rc_schur_nm"])

        summary_data: dict[str, object] = {}
        if rc_cnt is not None:
            info = point_info.get(base, {}).get(rc_cnt, {})
            summary_file = info.get("summary_path", "")
            if summary_file and Path(summary_file).exists():
                summary_data = parse_summary_file(Path(summary_file))
            else:
                summary_data = {"summary_path": summary_file}

        if available == 0:
            status = "not_started"
        elif available < int(meta["expected_points"]):
            status = "partial_peak_found" if peak_found else "partial_no_peak_yet"
        else:
            status = "complete_peak_found" if peak_found else "complete_no_peak_in_window"

        record = {
            "base_case_tag": base,
            "original_base_case_tag": meta["original_base_case_tag"],
            "mode": meta["mode"],
            "strain": meta["strain"],
            "rc_schur_nm": meta["rc_schur_nm"],
            "expected_points": meta["expected_points"],
            "available_points": available,
            "peak_found": int(peak_found),
            "rc_cnt_nm": rc_cnt,
            "delta_cnt_minus_schur_nm": delta_discrete,
            "F_CNT_peak_hat": fcnt_peak,
            "rc_cnt_fit_nm": rc_fit,
            "delta_fit_minus_schur_nm": delta_fit,
            "F_CNT_fit_hat": fcnt_fit,
            "fit_method": fit_method,
            "peak_index": peak_index,
            "status": status,
        }
        record.update(summary_data)
        records.append(record)

    with output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(output)
    print("status_counts", Counter(r["status"] for r in records))


if __name__ == "__main__":
    main()
