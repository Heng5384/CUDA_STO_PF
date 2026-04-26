#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.analyze_cnt_peak_table import parse_summary_file
from tools.analysis.workflow_utils import continue_summary_rel
from tools.analysis.workflow_utils import voxel_count_to_radius_nm


def _resolve_repo_path(repo_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return repo_root / p


def _continue_summary_path(row: dict[str, str], repo_root: Path, continue_kind: str) -> Path:
    if continue_kind == "dynamic":
        rel = row.get("continue_summary_rel")
        if rel:
            return _resolve_repo_path(repo_root, rel)
        return _resolve_repo_path(repo_root, continue_summary_rel(row["output_root_rel"], "dynamic"))

    rel = row.get("continue_minimize_summary_rel")
    if rel:
        preferred = _resolve_repo_path(repo_root, rel)
        if preferred.exists():
            return preferred
    canonical = _resolve_repo_path(repo_root, continue_summary_rel(row["output_root_rel"], "minimize"))
    if canonical.exists():
        return canonical
    return _resolve_repo_path(
        repo_root,
        str(Path(row["output_root_rel"]) / "continue_min_1" / "summary_continue_min_1.txt"),
    )


def main(default_continue_kind: str = "dynamic") -> int:
    parser = argparse.ArgumentParser(description="Summarize continue results from a single guide CSV.")
    parser.add_argument("--guide-csv", type=Path, required=True, help="guide_continue_dynamic.csv")
    parser.add_argument("--repo-root", "--results-root", dest="repo_root", type=Path, default=Path("."), help="repo root containing Results/")
    parser.add_argument("--continue-kind", choices=("dynamic", "minimize"), default=default_continue_kind, help="which continue result directory/summary naming to use")
    parser.add_argument("--dx-nm", type=float, default=0.1, help="grid spacing for equivalent radius conversion")
    parser.add_argument("--output", type=Path, default=None, help="output growth summary csv")
    args = parser.parse_args()

    guide_csv = args.guide_csv.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    workflow_dir = guide_csv.parent.parent
    default_dir = "continue_dynamic" if args.continue_kind == "dynamic" else "continue_minimize"
    output_name = "growth_summary.csv" if args.continue_kind == "dynamic" else "minimize_summary.csv"
    output = args.output.expanduser().resolve() if args.output else (workflow_dir / default_dir / output_name)
    output.parent.mkdir(parents=True, exist_ok=True)

    with guide_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records: list[dict[str, object]] = []
    for row in rows:
        summary_path = _continue_summary_path(row, repo_root, args.continue_kind)
        summary_data: dict[str, object] = {}
        if summary_path.exists():
            summary_data = parse_summary_file(summary_path)
        voxel_count = summary_data.get("voxel_count")
        grown_radius = None
        if isinstance(voxel_count, int):
            grown_radius = voxel_count_to_radius_nm(voxel_count, args.dx_nm)
        record = {
            "workflow_name": row["workflow_name"],
            "base_case_tag": row["base_case_tag"],
            "mode": row["mode"],
            "continue_kind": args.continue_kind,
            "strain": float(row["strain"]),
            "T_C": float(row["T_C"]),
            "xB_out": float(row["xB_out"]),
            "rc_schur_nm": float(row["rc_schur_nm"]),
            "rc_cnt_nm": float(row["rc_cnt_nm"]),
            "rc_cnt_fit_nm": float(row["rc_cnt_fit_nm"]) if row["rc_cnt_fit_nm"] else None,
            "source_radius_nm": float(row["source_radius_nm"]),
            "start_radius_nm": float(row["start_radius_nm"]),
            "source_equiv_radius_nm": float(row["source_equiv_radius_nm"]) if row.get("source_equiv_radius_nm") else None,
            "summary_path": str(summary_path),
            "summary_exists": int(summary_path.exists()),
            "voxel_count": voxel_count if isinstance(voxel_count, int) else None,
            "grown_equiv_radius_nm": grown_radius,
            "delta_growth_nm": (grown_radius - float(row["start_radius_nm"])) if grown_radius is not None else None,
        }
        for key, value in summary_data.items():
            if key == "summary_path":
                continue
            record[key] = value
        records.append(record)

    with output.open("w", newline="", encoding="utf-8") as f:
        if records:
            writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
            writer.writeheader()
            writer.writerows(records)
        else:
            f.write("")

    print(output)
    print(f"records={len(records)}")
    print(f"with_summary={sum(1 for r in records if r['summary_exists'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
