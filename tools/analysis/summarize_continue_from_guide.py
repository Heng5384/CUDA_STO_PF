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
from tools.analysis.workflow_utils import voxel_count_to_radius_nm


def _resolve_repo_path(repo_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return repo_root / p


def main() -> int:
    parser = argparse.ArgumentParser(description="Summarize continue dynamic results from a single guide CSV.")
    parser.add_argument("--guide-csv", type=Path, required=True, help="guide_continue_dynamic.csv")
    parser.add_argument("--repo-root", "--results-root", dest="repo_root", type=Path, default=Path("."), help="repo root containing Results/")
    parser.add_argument("--dx-nm", type=float, default=0.1, help="grid spacing for equivalent radius conversion")
    parser.add_argument("--output", type=Path, default=None, help="output growth summary csv")
    args = parser.parse_args()

    guide_csv = args.guide_csv.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    workflow_dir = guide_csv.parent.parent
    output = args.output.expanduser().resolve() if args.output else (workflow_dir / "continue_dynamic" / "growth_summary.csv")
    output.parent.mkdir(parents=True, exist_ok=True)

    with guide_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    records: list[dict[str, object]] = []
    for row in rows:
        summary_path = _resolve_repo_path(repo_root, row["continue_summary_rel"])
        summary_data: dict[str, object] = {}
        if summary_path.exists():
            summary_data = parse_summary_file(summary_path)
        voxel_count = summary_data.get("voxel_count")
        grown_radius = None
        if isinstance(voxel_count, int):
            grown_radius = voxel_count_to_radius_nm(voxel_count, args.dx_nm)
        records.append(
            {
                "workflow_name": row["workflow_name"],
                "base_case_tag": row["base_case_tag"],
                "mode": row["mode"],
                "strain": float(row["strain"]),
                "T_C": float(row["T_C"]),
                "xB_out": float(row["xB_out"]),
                "rc_schur_nm": float(row["rc_schur_nm"]),
                "rc_cnt_nm": float(row["rc_cnt_nm"]),
                "rc_cnt_fit_nm": float(row["rc_cnt_fit_nm"]) if row["rc_cnt_fit_nm"] else None,
                "source_radius_nm": float(row["source_radius_nm"]),
                "start_radius_nm": float(row["start_radius_nm"]),
                "summary_path": str(summary_path),
                "summary_exists": int(summary_path.exists()),
                "voxel_count": voxel_count if isinstance(voxel_count, int) else None,
                "grown_equiv_radius_nm": grown_radius,
                "delta_growth_nm": (grown_radius - float(row["start_radius_nm"])) if grown_radius is not None else None,
                "L1_long": summary_data.get("L1_long"),
                "L2_mid": summary_data.get("L2_mid"),
                "L3_short": summary_data.get("L3_short"),
                "L1_over_L3": summary_data.get("L1_over_L3"),
            }
        )

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
