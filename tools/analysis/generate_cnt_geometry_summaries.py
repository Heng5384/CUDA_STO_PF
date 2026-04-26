#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.generate_continue_dynamic_geometry_summaries import (
    _compute_geometry,
    _parse_pf_params,
    _read_legacy_scalar_vtk,
    _resolve_repo_path,
    _write_summary,
)
from tools.analysis.workflow_utils import cnt_summary_filename


def _pick_phi_vtk(case_dir: Path, preferred: Path) -> Path | None:
    if preferred.exists():
        return preferred
    finals = sorted(case_dir.glob("phi_final_*.vtk"))
    if finals:
        return finals[0]
    numeric = []
    for p in case_dir.glob("phi_*.vtk"):
        stem = p.stem
        if stem == "phi_init":
            continue
        suffix = stem.removeprefix("phi_")
        if suffix.isdigit():
            numeric.append((int(suffix), p))
    if numeric:
        numeric.sort()
        return numeric[-1][1]
    init = case_dir / "phi_init.vtk"
    return init if init.exists() else None


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate geometry summary.txt files for CNT minimize results by reading VTK files from guide_cnt_scan.csv.")
    parser.add_argument("--guide-csv", type=Path, required=True, help="guide_cnt_scan.csv")
    parser.add_argument("--repo-root", "--results-root", dest="repo_root", type=Path, default=Path("."), help="repo root containing Results/")
    parser.add_argument("--threshold", type=float, default=0.5, help="phi threshold used for nucleus masking")
    parser.add_argument("--overwrite", action="store_true", help="rebuild summary.txt even if it already exists")
    args = parser.parse_args()

    guide_csv = args.guide_csv.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    rows = list(csv.DictReader(guide_csv.open(newline="", encoding="utf-8")))

    built = 0
    skipped = 0
    failed = 0
    for idx, row in enumerate(rows, start=1):
        case_dir = _resolve_repo_path(repo_root, row["case_dir_rel"])
        summary_path = case_dir / cnt_summary_filename()
        legacy_summary_path = case_dir / cnt_summary_filename(row.get("case_tag", ""), legacy=True)
        if summary_path.exists() and not args.overwrite:
            skipped += 1
            continue

        try:
            preferred_phi = _resolve_repo_path(repo_root, row["phi_final_rel"])
            phi_vtk = _pick_phi_vtk(case_dir, preferred_phi)
            if phi_vtk is None or not phi_vtk.exists():
                raise FileNotFoundError(f"no CNT minimize phi vtk in {case_dir}")

            pf_params = _parse_pf_params(_resolve_repo_path(repo_root, row["pf_input_rel"]))
            phi, _dims, vtk_spacing = _read_legacy_scalar_vtk(phi_vtk)
            dx = float(pf_params.get("dx", vtk_spacing[0]))
            dy = float(pf_params.get("dy", vtk_spacing[1]))
            dz = float(pf_params.get("dz", vtk_spacing[2]))

            geom = _compute_geometry(phi, dx, dy, dz, args.threshold)
            if geom is None:
                raise RuntimeError("no valid nucleus geometry found above threshold")

            _write_summary(summary_path, phi_vtk, geom, "minimize")
            if legacy_summary_path != summary_path:
                legacy_summary_path.write_text(summary_path.read_text(encoding="utf-8"), encoding="utf-8")
            built += 1
            print(f"[built] row={idx} base={row['base_case_tag']} case={row['case_tag']} summary={summary_path}")
        except Exception as exc:
            failed += 1
            print(f"[failed] row={idx} base={row.get('base_case_tag', '')} case={row.get('case_tag', '')} error={exc}")

    print(f"built={built}")
    print(f"skipped={skipped}")
    print(f"failed={failed}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
