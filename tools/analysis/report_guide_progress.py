#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path
import sys
from typing import Iterable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.workflow_utils import cnt_summary_filename, continue_summary_filename


def _resolve_repo_path(repo_root: Path, rel_or_abs: str) -> Path:
    p = Path(rel_or_abs)
    if p.is_absolute():
        return p
    return repo_root / p


def _pick_existing(preferred: Path, candidates: Iterable[Path]) -> Path | None:
    if preferred.exists():
        return preferred
    for cand in candidates:
        if cand.exists():
            return cand
    return None


def _cnt_row_status(row: dict[str, str], repo_root: Path) -> tuple[str, dict[str, str]]:
    case_dir = _resolve_repo_path(repo_root, row["case_dir_rel"])
    energy = _pick_existing(
        _resolve_repo_path(repo_root, row["energy_csv_rel"]),
        sorted(case_dir.glob("energy_minimize_*.csv")),
    )
    phi_final = _pick_existing(
        _resolve_repo_path(repo_root, row["phi_final_rel"]),
        sorted(case_dir.glob("phi_final_*.vtk")),
    )
    summary = _pick_existing(
        case_dir / cnt_summary_filename(),
        [_resolve_repo_path(repo_root, row["summary_rel"]), *sorted(case_dir.glob("summary*.txt"))],
    )
    if energy and phi_final:
        return "complete", {
            "energy_csv": str(energy),
            "phi_final": str(phi_final),
            "summary": str(summary) if summary else "",
        }
    if case_dir.exists():
        return "partial", {
            "energy_csv": str(energy) if energy else "",
            "phi_final": str(phi_final) if phi_final else "",
            "summary": str(summary) if summary else "",
        }
    return "pending", {"energy_csv": "", "phi_final": "", "summary": ""}


def _continue_row_status(row: dict[str, str], repo_root: Path, continue_kind: str) -> tuple[str, dict[str, str]]:
    suffix = "continue_dyn_1" if continue_kind == "dynamic" else "continue_min_1"
    run_root = _resolve_repo_path(repo_root, row["output_root_rel"])
    cont_dir = run_root / suffix
    summary = _pick_existing(
        cont_dir / continue_summary_filename(continue_kind),
        [cont_dir / "summary_continue_min_1.txt"] if continue_kind == "minimize" else [],
    ) or (cont_dir / continue_summary_filename(continue_kind))
    final_phi = cont_dir / (
        f"phi_{row['nsteps']}.vtk" if continue_kind == "dynamic" else "phi_final_continue_min_1.vtk"
    )
    final_xb = cont_dir / (
        f"xB_{row['nsteps']}.vtk" if continue_kind == "dynamic" else "xB_final_continue_min_1.vtk"
    )
    energy = cont_dir / ("energy_minimize_continue_min_1.csv" if continue_kind == "minimize" else "")

    if continue_kind == "dynamic":
        if summary.exists() or (final_phi.exists() and final_xb.exists()):
            return "complete", {
                "continue_dir": str(cont_dir),
                "summary": str(summary) if summary.exists() else "",
                "phi_final": str(final_phi) if final_phi.exists() else "",
                "xb_final": str(final_xb) if final_xb.exists() else "",
            }
        if cont_dir.exists():
            return "partial", {
                "continue_dir": str(cont_dir),
                "summary": str(summary) if summary.exists() else "",
                "phi_final": str(final_phi) if final_phi.exists() else "",
                "xb_final": str(final_xb) if final_xb.exists() else "",
            }
        return "pending", {"continue_dir": str(cont_dir), "summary": "", "phi_final": "", "xb_final": ""}

    if summary.exists() or (energy.exists() and final_phi.exists()):
        return "complete", {
            "continue_dir": str(cont_dir),
            "summary": str(summary) if summary.exists() else "",
            "energy_csv": str(energy) if energy.exists() else "",
            "phi_final": str(final_phi) if final_phi.exists() else "",
            "xb_final": str(final_xb) if final_xb.exists() else "",
        }
    if cont_dir.exists():
        return "partial", {
            "continue_dir": str(cont_dir),
            "summary": str(summary) if summary.exists() else "",
            "energy_csv": str(energy) if energy.exists() else "",
            "phi_final": str(final_phi) if final_phi.exists() else "",
            "xb_final": str(final_xb) if final_xb.exists() else "",
        }
    return "pending", {"continue_dir": str(cont_dir), "summary": "", "energy_csv": "", "phi_final": "", "xb_final": ""}


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Report completed/pending rows from a workflow guide CSV and optionally write a pending-only guide."
    )
    parser.add_argument("--guide-csv", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path("."))
    parser.add_argument(
        "--guide-type",
        choices=("cnt", "continue-dynamic", "continue-minimize"),
        required=True,
        help="guide type to inspect",
    )
    parser.add_argument("--output", type=Path, default=None, help="status report csv")
    parser.add_argument("--pending-guide-output", type=Path, default=None, help="write only pending+partial rows as a new guide csv")
    args = parser.parse_args()

    guide_csv = args.guide_csv.expanduser().resolve()
    repo_root = args.repo_root.expanduser().resolve()
    with guide_csv.open(newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))

    report_rows: list[dict[str, object]] = []
    pending_rows: list[dict[str, str]] = []
    for idx, row in enumerate(rows):
        if args.guide_type == "cnt":
            status, info = _cnt_row_status(row, repo_root)
        elif args.guide_type == "continue-dynamic":
            status, info = _continue_row_status(row, repo_root, "dynamic")
        else:
            status, info = _continue_row_status(row, repo_root, "minimize")

        report_row = dict(row)
        report_row["row_index"] = idx
        report_row["progress_status"] = status
        report_row.update(info)
        report_rows.append(report_row)
        if status != "complete":
            pending_rows.append(row)

    output = args.output.expanduser().resolve() if args.output else guide_csv.with_name(guide_csv.stem + "_progress.csv")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as f:
        if report_rows:
            writer = csv.DictWriter(f, fieldnames=list(report_rows[0].keys()))
            writer.writeheader()
            writer.writerows(report_rows)
        else:
            f.write("")

    if args.pending_guide_output:
        pending_out = args.pending_guide_output.expanduser().resolve()
        pending_out.parent.mkdir(parents=True, exist_ok=True)
        with pending_out.open("w", newline="", encoding="utf-8") as f:
            if rows:
                writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                writer.writeheader()
                writer.writerows(pending_rows)
            else:
                f.write("")

    complete = sum(1 for r in report_rows if r["progress_status"] == "complete")
    partial = sum(1 for r in report_rows if r["progress_status"] == "partial")
    pending = sum(1 for r in report_rows if r["progress_status"] == "pending")
    print(output)
    print(f"complete={complete}")
    print(f"partial={partial}")
    print(f"pending={pending}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
