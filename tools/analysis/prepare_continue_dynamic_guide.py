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

from tools.analysis.analyze_cnt_peak_table import parse_summary_file
from tools.analysis.workflow_utils import cnt_summary_filename, continue_summary_rel, voxel_count_to_radius_nm


def _resolve_repo_path(repo_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return repo_root / p


def _continue_pf_rel(output_root_rel: str) -> str:
    return str(Path(output_root_rel) / "continue_dyn_1" / "pf_input.params")


def _source_equiv_radius_nm(summary_path: Path, dx_nm: float) -> float | None:
    if not summary_path.exists():
        return None
    summary_data = parse_summary_file(summary_path)
    voxel_count = summary_data.get("voxel_count")
    if not isinstance(voxel_count, int):
        return None
    return voxel_count_to_radius_nm(voxel_count, dx_nm)


def _cnt_summary_path(row: dict[str, str], repo_root: Path) -> Path:
    case_dir = _resolve_repo_path(repo_root, row["case_dir_rel"])
    canonical = case_dir / cnt_summary_filename()
    if canonical.exists():
        return canonical
    return _resolve_repo_path(repo_root, row["summary_rel"])


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a single guide CSV for continue dynamic from CNT summary + guide.")
    parser.add_argument("--summary-csv", type=Path, required=True, help="current_results_master_table_fitted.csv")
    parser.add_argument("--guide-csv", type=Path, required=True, help="guide_cnt_scan.csv")
    parser.add_argument("--repo-root", "--results-root", dest="repo_root", type=Path, default=Path("."), help="repo root containing Results/")
    parser.add_argument("--radius-offset-nm", type=float, default=0.1, help="legacy fallback: nominal discrete radius margin above rc_cnt_nm when no fitted/actual-radius match is available")
    parser.add_argument("--fit-radius-margin-nm", type=float, default=0.055, help="require source nominal discrete radius >= rc_cnt_fit_nm + margin")
    parser.add_argument("--dx-nm", type=float, default=0.1, help="grid spacing used to convert voxel_count to equivalent radius")
    parser.add_argument("--dt", type=float, default=0.1, help="continue dynamic dt")
    parser.add_argument("--steps", type=int, default=5000, help="continue dynamic steps")
    parser.add_argument("--out-every", type=int, default=2500, help="continue dynamic VTK output interval")
    parser.add_argument("--csv-out-every", type=int, default=10, help="continue dynamic CSV output interval")
    parser.add_argument("--allow-no-peak", action="store_true", help="also allow complete_no_peak_in_window rows for smoke validation")
    parser.add_argument("--output", type=Path, default=None, help="output guide csv")
    args = parser.parse_args()

    summary_csv = args.summary_csv.expanduser().resolve()
    guide_csv = args.guide_csv.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    workflow_dir = summary_csv.parent.parent
    output = args.output.expanduser().resolve() if args.output else (workflow_dir / "input" / "guide_continue_dynamic.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with summary_csv.open(newline="", encoding="utf-8") as f:
        summary_rows = list(csv.DictReader(f))
    with guide_csv.open(newline="", encoding="utf-8") as f:
        guide_rows = list(csv.DictReader(f))

    guide_by_base: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in guide_rows:
        if row.get("row_type") == "reference":
            continue
        guide_by_base[row["base_case_tag"]].append(row)
    for rows in guide_by_base.values():
        for candidate in rows:
            summary_path = _cnt_summary_path(candidate, repo_root)
            actual_radius = _source_equiv_radius_nm(summary_path, args.dx_nm)
            candidate["_source_equiv_radius_nm"] = actual_radius
        rows.sort(key=lambda r: float(r["radius_nm"]))

    out_rows: list[dict[str, object]] = []
    for row in summary_rows:
        allowed_status = {"complete_peak_found"}
        if args.allow_no_peak:
            allowed_status.add("complete_no_peak_in_window")
        if row.get("status") not in allowed_status:
            continue
        base = row["base_case_tag"]
        rc_cnt = float(row["rc_cnt_nm"])
        rc_cnt_fit = float(row["rc_cnt_fit_nm"]) if row.get("rc_cnt_fit_nm") else None
        target_fit_radius = None
        if rc_cnt_fit is not None:
            target_fit_radius = rc_cnt_fit + max(args.fit_radius_margin_nm, 0.0)
        target_nominal_radius = rc_cnt + max(args.radius_offset_nm, 0.0)
        candidates = guide_by_base.get(base, [])
        source_row = None
        start_radius_source = ""
        for candidate in candidates:
            if target_fit_radius is None:
                continue
            if float(candidate["radius_nm"]) >= target_fit_radius - 1e-9:
                source_row = candidate
                start_radius_source = f"first_discrete_radius_ge_rc_cnt_fit_plus_{args.fit_radius_margin_nm:g}nm"
                break
        if source_row is None:
            for candidate in candidates:
                if float(candidate["radius_nm"]) >= target_nominal_radius - 1e-9:
                    source_row = candidate
                    start_radius_source = f"fallback_first_discrete_radius_ge_rc_cnt_plus_{args.radius_offset_nm:g}nm"
                    break
        if source_row is None and candidates:
            source_row = candidates[-1]
            start_radius_source = "fallback_largest_discrete_radius_in_scan"
        if source_row is None:
            continue

        phi_vtk = _resolve_repo_path(repo_root, source_row["phi_final_rel"])
        xb_vtk = _resolve_repo_path(repo_root, source_row["xb_final_rel"])
        summary_path = _cnt_summary_path(source_row, repo_root)
        source_equiv_radius = source_row.get("_source_equiv_radius_nm")
        start_radius_nm = f"{float(source_row['radius_nm']):.6f}"

        out_rows.append(
            {
                "workflow_name": source_row["workflow_name"],
                "base_case_tag": base,
                "mode": source_row["mode"],
                "strain": source_row["strain"],
                "T_C": source_row["T_C"],
                "xB_out": source_row["xB_out"],
                "raw_results_root_rel": source_row["raw_results_root_rel"],
                "nx": source_row["nx"],
                "ny": source_row["ny"],
                "nz": source_row["nz"],
                "dt": args.dt,
                "nsteps": args.steps,
                "out_every": args.out_every,
                "csv_out_every": args.csv_out_every,
                "elastic": source_row["elastic"],
                "rc_schur_nm": row["rc_schur_nm"],
                "rc_cnt_nm": row["rc_cnt_nm"],
                "rc_cnt_fit_nm": row["rc_cnt_fit_nm"],
                "start_radius_nm": start_radius_nm,
                "start_radius_source": start_radius_source,
                "source_case_tag": source_row["case_tag"],
                "source_radius_nm": source_row["radius_nm"],
                "source_equiv_radius_nm": f"{float(source_equiv_radius):.6f}" if source_equiv_radius is not None else "",
                "selection_target_fit_radius_nm": f"{target_fit_radius:.6f}" if target_fit_radius is not None else "",
                "source_summary_rel": str(Path(source_row["case_dir_rel"]) / cnt_summary_filename()),
                "source_phi_final_rel": source_row["phi_final_rel"],
                "source_xb_final_rel": source_row["xb_final_rel"],
                "continue_summary_rel": continue_summary_rel(source_row["output_root_rel"], "dynamic"),
                "continue_pf_input_rel": _continue_pf_rel(source_row["output_root_rel"]),
                "continue_phi_vtk": str(phi_vtk),
                "continue_xb_vtk": str(xb_vtk),
                "continue_source_summary_path": str(summary_path),
                "output_root_rel": source_row["output_root_rel"],
            }
        )

    with output.open("w", newline="", encoding="utf-8") as f:
        if out_rows:
            writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
            writer.writeheader()
            writer.writerows(out_rows)
        else:
            f.write("")

    print(output)
    print(f"records={len(out_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
