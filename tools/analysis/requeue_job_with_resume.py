#!/usr/bin/env python3
"""Dry-run helper for safe requeue using original run directories and checkpoints.

This tool is intentionally conservative:

* dry-run is the default
* it never creates replacement run directories
* it never deletes files
* if resume semantics are ambiguous, it returns MANUAL_REVIEW

Current automated support focuses on CNT/minimize scan workflow roots generated
by ``setup_cnt_workflow.py`` and launched with ``jobs/submit_cnt_guide_serial.sbatch``.
Other continue-style jobs are inspected, but currently reported as manual review
unless their restart contract is unambiguous.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]


def run(cmd: list[str]) -> tuple[int, str]:
    proc = subprocess.run(
        cmd,
        cwd=ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    return proc.returncode, proc.stdout


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def file_nonempty(path: Path) -> bool:
    return path.is_file() and path.stat().st_size > 0


def first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if file_nonempty(path):
            return path
    return None


def shell_quote(text: str) -> str:
    return "'" + text.replace("'", "'\"'\"'") + "'"


@dataclass
class RowState:
    row_idx: int
    case_tag: str
    case_dir_rel: str
    status: str
    energy_csv: str = ""
    phi_final: str = ""
    checkpoint_files: list[str] = field(default_factory=list)
    notes: str = ""


@dataclass
class Decision:
    run_dir: str
    job_type: str
    decision: str
    safe_to_submit: bool
    safe_for_resume: bool
    resume_command: str
    dependency: str
    completion_marker: str
    latest_checkpoint: str
    risk: str
    notes: list[str]
    row_states: list[RowState] = field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["row_states"] = [asdict(row) for row in self.row_states]
        return payload


def detect_job_type(run_dir: Path) -> str:
    if (run_dir / "input" / "guide_cnt_scan.csv").exists():
        return "cnt_scan_workflow"
    if (run_dir / "input" / "guide_continue_dynamic.csv").exists():
        return "continue_dynamic_workflow"
    if (run_dir / "continue_dyn_1").exists() or run_dir.name.startswith("continue_dyn_"):
        return "continue_dynamic_case"
    if (run_dir / "continue_min_1").exists() or run_dir.name.startswith("continue_min_"):
        return "continue_minimize_case"
    return "unknown"


def inspect_cnt_scan_workflow(run_dir: Path, dependency: str) -> Decision:
    guide_csv = run_dir / "input" / "guide_cnt_scan.csv"
    rows = read_csv(guide_csv)
    rel_run_dir = run_dir.relative_to(ROOT)

    complete_rows = 0
    partial_rows = 0
    missing_rows = 0
    latest_checkpoint = ""
    notes: list[str] = []
    row_states: list[RowState] = []
    guide_path_mismatch = False

    for row_idx, row in enumerate(rows):
        case_dir = ROOT / row["case_dir_rel"]
        preferred_energy = ROOT / row["energy_csv_rel"]
        preferred_phi_final = ROOT / row["phi_final_rel"]
        energy_csv = first_existing([preferred_energy, *sorted(case_dir.glob("energy_minimize_*.csv"))])
        phi_final = first_existing([preferred_phi_final, *sorted(case_dir.glob("phi_final_*.vtk"))])
        checkpoint_files = [
            str(path.relative_to(ROOT))
            for path in sorted(case_dir.glob("*"))
            if path.name.startswith(("phi_", "xB_", "vf_precip_vs_time", "energy_minimize", "summary", "pf_input"))
        ]
        if row.get("workflow_dir_rel") and row["workflow_dir_rel"] != str(rel_run_dir):
            guide_path_mismatch = True
        if energy_csv and phi_final:
            status = "complete"
            complete_rows += 1
            latest_checkpoint = str(phi_final.relative_to(ROOT))
        elif checkpoint_files:
            status = "partial"
            partial_rows += 1
            latest_checkpoint = checkpoint_files[-1]
        else:
            status = "missing"
            missing_rows += 1
        row_states.append(
            RowState(
                row_idx=row_idx,
                case_tag=row.get("case_tag", f"row_{row_idx}"),
                case_dir_rel=row.get("case_dir_rel", ""),
                status=status,
                energy_csv="" if energy_csv is None else str(energy_csv.relative_to(ROOT)),
                phi_final="" if phi_final is None else str(phi_final.relative_to(ROOT)),
                checkpoint_files=checkpoint_files,
                notes=(
                    "guide workflow_dir_rel mismatches this run_dir"
                    if row.get("workflow_dir_rel") and row["workflow_dir_rel"] != str(rel_run_dir)
                    else ""
                ),
            )
        )

    workflow_name = rows[0].get("workflow_name", run_dir.name) if rows else run_dir.name
    submit_cmd = (
        "sbatch "
        + (f"--dependency={dependency} " if dependency else "")
        + f"--export=ALL,GUIDE_CSV={shell_quote(str(guide_csv))},MANIFEST_NAME={workflow_name} "
        + "jobs/submit_cnt_guide_serial.sbatch"
    )

    if complete_rows == len(rows):
        decision = "COMPLETE_ALREADY_EXISTS"
        safe_to_submit = False
        safe_for_resume = True
        risk = "low"
        completion_marker = f"all {len(rows)} guide rows have energy_minimize_*.csv + phi_final_*.vtk"
        notes.append("No resubmission needed; rerun would only hit skip-complete logic.")
    elif guide_path_mismatch:
        decision = "MANUAL_REVIEW"
        safe_to_submit = False
        safe_for_resume = False
        risk = "high"
        completion_marker = "guide_csv exists but at least one workflow_dir_rel points outside this workflow root"
        notes.append("Guide path mismatch detected; same workflow root cannot be trusted for automated requeue.")
    elif partial_rows > 0:
        decision = "RESUBMIT_SAME_WORKFLOW_DIR"
        safe_to_submit = True
        safe_for_resume = False
        risk = "medium"
        completion_marker = "per-row complete if energy_minimize_*.csv + phi_final_*.vtk exist"
        notes.append(
            "Completed rows will be skipped by submit_cnt_guide_serial.sbatch, but partial rows do not have lossless mid-minimize checkpoints."
        )
        notes.append(
            "Partial scan points will restart from their original initialization inside the same case directory, not from a replacement folder."
        )
    else:
        decision = "RESUBMIT_SAME_WORKFLOW_DIR"
        safe_to_submit = True
        safe_for_resume = True
        risk = "low"
        completion_marker = "per-row complete if energy_minimize_*.csv + phi_final_*.vtk exist"
        notes.append("No partial checkpoints detected; missing rows can be resumed safely by reusing the same workflow root.")

    if any("_requeued_after_dc_" in row.case_dir_rel for row in row_states):
        notes.append("This workflow already carries a replacement-folder suffix; prefer the original workflow root on future reprioritization.")

    notes.append(f"row_counts: complete={complete_rows}, partial={partial_rows}, missing={missing_rows}")

    return Decision(
        run_dir=str(rel_run_dir),
        job_type="cnt_scan_workflow",
        decision=decision,
        safe_to_submit=safe_to_submit,
        safe_for_resume=safe_for_resume,
        resume_command=submit_cmd,
        dependency=dependency,
        completion_marker=completion_marker,
        latest_checkpoint=latest_checkpoint,
        risk=risk,
        notes=notes,
        row_states=row_states,
    )


def inspect_continue_case(run_dir: Path, dependency: str, mode: str) -> Decision:
    latest_phi = ""
    latest_xb = ""
    phi_hits = sorted(run_dir.glob("phi_*.vtk"))
    xb_hits = sorted(run_dir.glob("xB_*.vtk"))
    if phi_hits:
        latest_phi = str(phi_hits[-1].relative_to(ROOT))
    if xb_hits:
        latest_xb = str(xb_hits[-1].relative_to(ROOT))
    notes = [
        f"Detected {mode} case directory.",
        "Underlying main_cuda continuation exists, but the current serial continue wrapper does not auto-rebind to the latest in-progress checkpoint.",
        "Use the latest phi/xB pair in this same directory when preparing a manual resume command, or extend the wrapper before automated submit.",
    ]
    return Decision(
        run_dir=str(run_dir.relative_to(ROOT)),
        job_type=f"{mode}_case",
        decision="MANUAL_REVIEW",
        safe_to_submit=False,
        safe_for_resume=False,
        resume_command="",
        dependency=dependency,
        completion_marker="summary.txt or final phi/xB pair in continue directory",
        latest_checkpoint=latest_phi or latest_xb,
        risk="medium",
        notes=notes,
    )


def inspect_unknown(run_dir: Path, dependency: str) -> Decision:
    return Decision(
        run_dir=str(run_dir.relative_to(ROOT)) if run_dir.is_relative_to(ROOT) else str(run_dir),
        job_type="unknown",
        decision="MANUAL_REVIEW",
        safe_to_submit=False,
        safe_for_resume=False,
        resume_command="",
        dependency=dependency,
        completion_marker="",
        latest_checkpoint="",
        risk="high",
        notes=["No supported workflow marker found under this run_dir."],
    )


def build_report_md(decision: Decision) -> str:
    lines = [
        f"# Requeue Resume Audit",
        "",
        f"- run_dir: `{decision.run_dir}`",
        f"- job_type: `{decision.job_type}`",
        f"- decision: `{decision.decision}`",
        f"- safe_to_submit: `{str(decision.safe_to_submit).lower()}`",
        f"- safe_for_resume: `{str(decision.safe_for_resume).lower()}`",
        f"- risk: `{decision.risk}`",
        f"- completion_marker: `{decision.completion_marker}`",
        f"- latest_checkpoint: `{decision.latest_checkpoint}`",
        "",
        "## Suggested Command",
        "",
        "```bash",
        decision.resume_command or "# manual review required",
        "```",
        "",
        "## Notes",
        "",
    ]
    for note in decision.notes:
        lines.append(f"- {note}")
    if decision.row_states:
        lines.extend(
            [
                "",
                "## Row States",
                "",
                "| row_idx | case_tag | status | case_dir_rel |",
                "|---|---|---|---|",
            ]
        )
        for row in decision.row_states:
            lines.append(f"| {row.row_idx} | {row.case_tag} | {row.status} | `{row.case_dir_rel}` |")
    return "\n".join(lines) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-dir", type=Path, help="Original workflow or case directory under the project root.")
    ap.add_argument("--old-job-id", help="Reserved for future scheduler lookup support.")
    ap.add_argument("--job-type", default="auto", choices=["auto", "cnt_scan_workflow", "continue_dynamic_case", "continue_minimize_case"])
    ap.add_argument("--mode", default="dry-run", choices=["dry-run", "submit"])
    ap.add_argument("--dependency", default="", help="Optional dependency clause, e.g. afterany:65047")
    ap.add_argument("--report-json", type=Path, default=None)
    ap.add_argument("--report-md", type=Path, default=None)
    args = ap.parse_args()

    if args.run_dir is None and not args.old_job_id:
        raise SystemExit("Either --run-dir or --old-job-id is required.")
    if args.old_job_id:
        raise SystemExit("old_job_id lookup is not implemented yet; pass --run-dir for now.")

    run_dir = args.run_dir
    if not run_dir.is_absolute():
        run_dir = (ROOT / run_dir).resolve()
    if not run_dir.exists():
        raise SystemExit(f"run_dir does not exist: {run_dir}")

    job_type = args.job_type if args.job_type != "auto" else detect_job_type(run_dir)
    if job_type == "cnt_scan_workflow":
        decision = inspect_cnt_scan_workflow(run_dir, args.dependency)
    elif job_type == "continue_dynamic_case":
        decision = inspect_continue_case(run_dir, args.dependency, "continue_dynamic")
    elif job_type == "continue_minimize_case":
        decision = inspect_continue_case(run_dir, args.dependency, "continue_minimize")
    else:
        decision = inspect_unknown(run_dir, args.dependency)

    if args.report_json is not None:
        args.report_json.parent.mkdir(parents=True, exist_ok=True)
        args.report_json.write_text(json.dumps(decision.to_json_dict(), indent=2), encoding="utf-8")
    if args.report_md is not None:
        args.report_md.parent.mkdir(parents=True, exist_ok=True)
        args.report_md.write_text(build_report_md(decision), encoding="utf-8")

    print(json.dumps(decision.to_json_dict(), ensure_ascii=False, indent=2))

    if args.mode == "submit":
        if not decision.safe_to_submit or not decision.resume_command:
            raise SystemExit("Decision is not safe_to_submit; refusing submit.")
        rc, output = run(["bash", "-lc", decision.resume_command])
        sys.stdout.write(output)
        if rc != 0:
            raise SystemExit(rc)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
