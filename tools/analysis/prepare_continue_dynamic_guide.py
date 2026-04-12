#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path


def _resolve_repo_path(repo_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return repo_root / p


def _continue_summary_rel(output_root_rel: str) -> str:
    return str(Path(output_root_rel) / "continue_dyn_1" / "summary.txt")


def _continue_pf_rel(output_root_rel: str) -> str:
    return str(Path(output_root_rel) / "continue_dyn_1" / "pf_input.params")


def main() -> int:
    parser = argparse.ArgumentParser(description="Prepare a single guide CSV for continue dynamic from CNT summary + guide.")
    parser.add_argument("--summary-csv", type=Path, required=True, help="current_results_master_table_fitted.csv")
    parser.add_argument("--guide-csv", type=Path, required=True, help="guide_cnt_scan.csv")
    parser.add_argument("--repo-root", "--results-root", dest="repo_root", type=Path, default=Path("."), help="repo root containing Results/")
    parser.add_argument("--radius-offset-nm", type=float, default=0.1, help="start continue from discrete peak + offset")
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
        guide_by_base[row["base_case_tag"]].append(row)
    for rows in guide_by_base.values():
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
        target_radius = rc_cnt + max(args.radius_offset_nm, 0.0)
        candidates = guide_by_base.get(base, [])
        source_row = None
        for candidate in candidates:
            if float(candidate["radius_nm"]) >= target_radius - 1e-9:
                source_row = candidate
                break
        if source_row is None and candidates:
            source_row = candidates[-1]
        if source_row is None:
            continue

        phi_vtk = _resolve_repo_path(repo_root, source_row["phi_final_rel"])
        xb_vtk = _resolve_repo_path(repo_root, source_row["xb_final_rel"])
        summary_path = _resolve_repo_path(repo_root, source_row["summary_rel"])

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
                "start_radius_nm": f"{float(source_row['radius_nm']):.6f}",
                "start_radius_source": f"first_discrete_radius_ge_rc_cnt_plus_{args.radius_offset_nm:g}nm",
                "source_case_tag": source_row["case_tag"],
                "source_radius_nm": source_row["radius_nm"],
                "source_summary_rel": source_row["summary_rel"],
                "source_phi_final_rel": source_row["phi_final_rel"],
                "source_xb_final_rel": source_row["xb_final_rel"],
                "continue_summary_rel": _continue_summary_rel(source_row["output_root_rel"]),
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
