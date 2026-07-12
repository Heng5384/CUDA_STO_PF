#!/usr/bin/env python3
"""Workstation batch CNT scanning launcher for no-strain nucleus catalog builds.

This script runs the offline CNT / energy-minimization layer only. It does not
modify PF equations and does not introduce S(x).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import subprocess
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[2]
K_B = 1.380649e-23


T_GRID_K = [350.0, 380.0, 400.0, 450.0]
XB_GRID = [0.03, 0.04, 0.05]
STRAIN_GRID = [0.0]


def parse_float_list(text: str) -> list[float]:
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def safe_tag_float(value: float, digits: int = 5) -> str:
    text = f"{value:.{digits}f}".rstrip("0").rstrip(".")
    if text == "-0":
        text = "0"
    return text.replace("-", "m").replace(".", "p")


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def tail_text(path: Path, max_chars: int = 4000) -> str:
    if not path.exists():
        return ""
    data = path.read_text(encoding="utf-8", errors="ignore")
    return data[-max_chars:]


@dataclass(frozen=True)
class ScanCase:
    case_id: str
    T_K: float
    T_C: float
    xB: float
    strain: float
    strain_mode: str = "exx"

    @property
    def case_dir(self) -> Path:
        return REPO_ROOT / "CNT_SCAN_WORKSTATION_RUN" / "results" / f"T_{safe_tag_float(self.T_K, 2)}K" / f"xB_{safe_tag_float(self.xB, 4)}" / f"strain_{safe_tag_float(self.strain, 5)}"


def generate_cases(T_grid_K: list[float], xB_grid: list[float]) -> list[ScanCase]:
    cases: list[ScanCase] = []
    for T_K in T_grid_K:
        for xB in xB_grid:
            for strain in STRAIN_GRID:
                case_id = f"T{safe_tag_float(T_K, 2)}K_xB{safe_tag_float(xB, 4)}_strain{safe_tag_float(strain, 5)}"
                cases.append(ScanCase(case_id=case_id, T_K=T_K, T_C=T_K - 273.15, xB=xB, strain=strain))
    return cases


def setup_workflow(case: ScanCase, args: argparse.Namespace) -> tuple[int, str]:
    workflow_name = case.case_id
    cmd = [
        "python3",
        "tools/analysis/setup_cnt_workflow.py",
        "--temp-c",
        f"{case.T_C:.12g}",
        "--xb-out",
        f"{case.xB:.12g}",
        "--workflow-root",
        "CNT_SCAN_WORKSTATION_RUN/workflows",
        "--workflow-name",
        workflow_name,
        "--grid",
        args.grid,
        "--dt",
        str(args.dt),
        "--steps",
        str(args.steps),
        "--out-every",
        str(args.out_every),
        "--csv-out-every",
        str(args.csv_out_every),
        "--elastic",
        str(args.elastic),
        "--window-nm",
        str(args.window_nm),
        "--step-nm",
        str(args.step_nm),
        "--strains",
        "0",
        "--modes",
        "exx",
    ]
    proc = subprocess.run(cmd, cwd=REPO_ROOT, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    return proc.returncode, proc.stdout


def run_scan(case: ScanCase, args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    out_dir = case.case_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "job.log"
    summary_path = out_dir / "summary.json"
    if summary_path.exists() and not args.force:
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            payload = {}
        if payload.get("status") in {"complete", "skipped_no_supersaturation"}:
            return {
                "case_id": case.case_id,
                "T_K": case.T_K,
                "xB": case.xB,
                "strain": case.strain,
                "status": payload.get("status"),
                "workflow_dir": payload.get("workflow_dir", ""),
                "result_dir": str(out_dir.relative_to(REPO_ROOT)),
                "log": str(log_path.relative_to(REPO_ROOT)),
                "elapsed_s": 0.0,
                "notes": "resume_skip_existing_summary",
            }

    if case.xB <= 0.0:
        payload = {
            "case_id": case.case_id,
            "T_K": case.T_K,
            "T_C": case.T_C,
            "xB": case.xB,
            "strain": case.strain,
            "status": "skipped_no_supersaturation",
            "reason": "xB=0 gives undefined log(xB) in CNT chemical driving force and no Ag2Te supersaturation.",
            "r_star": None,
            "deltaG_star": None,
            "nucleus_shape": "none",
        }
        summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        for name in ("energy_profile.csv", "r_scan_curve.csv"):
            write_csv(out_dir / name, [], ["r_nm", "DeltaG", "status"])
        (out_dir / "nucleus_geometry.dat").write_text("status skipped_no_supersaturation\n", encoding="utf-8")
        (out_dir / "minimized_configuration.xyz").write_text("0\nskipped_no_supersaturation\n", encoding="utf-8")
        log_path.write_text(payload["reason"] + "\n", encoding="utf-8")
        return {
            "case_id": case.case_id,
            "T_K": case.T_K,
            "xB": case.xB,
            "strain": case.strain,
            "status": "skipped_no_supersaturation",
            "workflow_dir": "",
            "result_dir": str(out_dir.relative_to(REPO_ROOT)),
            "log": str(log_path.relative_to(REPO_ROOT)),
            "elapsed_s": time.time() - started,
            "notes": payload["reason"],
        }

    rc, setup_log = setup_workflow(case, args)
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("========== setup_cnt_workflow ==========\n")
        handle.write(setup_log)
        handle.write("\n")
    workflow_dir = REPO_ROOT / "CNT_SCAN_WORKSTATION_RUN" / "workflows" / case.case_id
    guide_csv = workflow_dir / "input" / "guide_cnt_scan.csv"
    if rc != 0 or not guide_csv.exists():
        payload = {
            "case_id": case.case_id,
            "T_K": case.T_K,
            "T_C": case.T_C,
            "xB": case.xB,
            "strain": case.strain,
            "status": "setup_failed",
            "workflow_dir": str(workflow_dir.relative_to(REPO_ROOT)),
            "log_tail": tail_text(log_path),
        }
        summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        return {
            "case_id": case.case_id,
            "T_K": case.T_K,
            "xB": case.xB,
            "strain": case.strain,
            "status": "setup_failed",
            "workflow_dir": str(workflow_dir.relative_to(REPO_ROOT)),
            "result_dir": str(out_dir.relative_to(REPO_ROOT)),
            "log": str(log_path.relative_to(REPO_ROOT)),
            "elapsed_s": time.time() - started,
            "notes": "setup_cnt_workflow failed",
        }

    run_env = os.environ.copy()
    run_env.update(
        {
            "GUIDE_CSV": str(guide_csv),
            "MANIFEST_NAME": f"workstation_cnt_{case.case_id}",
            "JOB_TAG": case.case_id,
            "CUDA_VISIBLE_DEVICES": args.cuda_visible_devices,
        }
    )
    cmd = ["bash", "jobs/submit_cnt_guide_serial.sbatch"]
    with log_path.open("a", encoding="utf-8") as handle:
        handle.write("========== submit_cnt_guide_serial ==========\n")
        handle.write(" ".join(cmd) + "\n")
        handle.flush()
        proc = subprocess.run(cmd, cwd=REPO_ROOT, env=run_env, text=True, stdout=handle, stderr=subprocess.STDOUT)

    status = "complete" if proc.returncode == 0 else "failed"
    payload = summarize_case(case, workflow_dir, out_dir, status)
    payload["returncode"] = proc.returncode
    payload["elapsed_s"] = time.time() - started
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return {
        "case_id": case.case_id,
        "T_K": case.T_K,
        "xB": case.xB,
        "strain": case.strain,
        "status": status,
        "workflow_dir": str(workflow_dir.relative_to(REPO_ROOT)),
        "result_dir": str(out_dir.relative_to(REPO_ROOT)),
        "log": str(log_path.relative_to(REPO_ROOT)),
        "elapsed_s": time.time() - started,
        "notes": "" if status == "complete" else "see job.log",
    }


def summarize_case(case: ScanCase, workflow_dir: Path, out_dir: Path, status: str) -> dict[str, Any]:
    table = workflow_dir / "cnt_scan" / "current_results_master_table_fitted.csv"
    if not table.exists():
        # Build geometry and CNT summary if the serial runner did not do it.
        guide = workflow_dir / "input" / "guide_cnt_scan.csv"
        if guide.exists():
            subprocess.run(["python3", "tools/analysis/generate_cnt_geometry_summaries.py", "--guide-csv", str(guide)], cwd=REPO_ROOT)
            subprocess.run(["python3", "tools/analysis/summarize_cnt_scan_from_guide.py", "--guide-csv", str(guide)], cwd=REPO_ROOT)
    rows = read_csv(table)
    scan_rows = [r for r in rows if r.get("row_type") != "reference"]
    best = None
    for row in scan_rows:
        barrier = as_float(row.get("F_CNT_peak_hat"), as_float(row.get("F_CNT_fit_hat")))
        if barrier is None:
            continue
        if best is None or barrier < best[0]:
            best = (barrier, row)

    r_scan_rows = []
    for row in scan_rows:
        r_scan_rows.append(
            {
                "r_nm": row.get("radius_nm") or row.get("r_eff_nm") or "",
                "DeltaG": row.get("F_CNT_peak_hat") or row.get("F_CNT_fit_hat") or "",
                "status": row.get("status", ""),
                "case_dir": row.get("case_dir", ""),
            }
        )
    write_csv(out_dir / "r_scan_curve.csv", r_scan_rows, ["r_nm", "DeltaG", "status", "case_dir"])
    write_csv(out_dir / "energy_profile.csv", r_scan_rows, ["r_nm", "DeltaG", "status", "case_dir"])

    r_star = None
    deltaG_star = None
    shape = "unknown"
    file_path = ""
    if best is not None:
        row = best[1]
        deltaG_star = best[0]
        r_star = as_float(row.get("rc_cnt_fit_nm"), as_float(row.get("rc_cnt_nm"), as_float(row.get("radius_nm"))))
        aspect = as_float(row.get("L1_over_L3"))
        if aspect is None:
            shape = "unknown"
        elif aspect < 1.10:
            shape = "spherical"
        elif aspect < 1.8:
            shape = "anisotropic"
        else:
            shape = "faceted"
        file_path = row.get("case_dir", "")
        (out_dir / "nucleus_geometry.dat").write_text(
            "\n".join(
                [
                    f"shape {shape}",
                    f"r_star_nm {r_star}",
                    f"aspect_ratio {aspect}",
                    f"source_case_dir {file_path}",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
    else:
        (out_dir / "nucleus_geometry.dat").write_text("shape unknown\n", encoding="utf-8")
    (out_dir / "minimized_configuration.xyz").write_text(
        "0\nNo atomistic XYZ/POSCAR generated by PF CNT scan; see source VTK/raw case directory.\n",
        encoding="utf-8",
    )
    return {
        "case_id": case.case_id,
        "T_K": case.T_K,
        "T_C": case.T_C,
        "xB": case.xB,
        "strain": case.strain,
        "status": status,
        "r_star": r_star,
        "deltaG_star": deltaG_star,
        "nucleus_shape": shape,
        "energy_landscape": str((out_dir / "r_scan_curve.csv").relative_to(REPO_ROOT)),
        "file_path": file_path,
        "workflow_dir": str(workflow_dir.relative_to(REPO_ROOT)),
        "result_dir": str(out_dir.relative_to(REPO_ROOT)),
    }


def as_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def build_catalog(status_rows: list[dict[str, Any]], output_root: Path) -> dict[str, Any]:
    entries = []
    for row in status_rows:
        summary_path = REPO_ROOT / row["result_dir"] / "summary.json"
        if not summary_path.exists():
            continue
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        if payload.get("status") not in {"complete", "skipped_no_supersaturation"}:
            continue
        entries.append(
            {
                "T": payload.get("T_K"),
                "xB": payload.get("xB"),
                "strain": payload.get("strain"),
                "r_star": payload.get("r_star"),
                "deltaG_star": payload.get("deltaG_star"),
                "nucleus_shape": payload.get("nucleus_shape"),
                "energy_landscape": payload.get("energy_landscape"),
                "file_path": payload.get("file_path") or payload.get("result_dir"),
                "status": payload.get("status"),
            }
        )
    catalog = {
        "schema_version": 1,
        "generated_by": "tools/analysis/workstation_cnt_batch.py",
        "run_root": str(output_root.relative_to(REPO_ROOT)),
        "parameter_grid": {
            "T_K": sorted({row["T_K"] for row in status_rows}),
            "xB": sorted({row["xB"] for row in status_rows}),
            "strain": STRAIN_GRID,
            "note": "No-strain run only per user correction.",
        },
        "entry_count": len(entries),
        "entries": entries,
    }
    (output_root / "nucleus_catalog.json").write_text(json.dumps(catalog, indent=2) + "\n", encoding="utf-8")
    return catalog


def write_reports(output_root: Path, status_rows: list[dict[str, Any]], catalog: dict[str, Any]) -> None:
    fields = ["case_id", "T_K", "xB", "strain", "status", "workflow_dir", "result_dir", "log", "elapsed_s", "notes"]
    write_csv(output_root / "job_status_table.csv", status_rows, fields)

    counts: dict[str, int] = {}
    for row in status_rows:
        counts[row["status"]] = counts.get(row["status"], 0) + 1
    report = [
        "# Workstation CNT Scan Completion Report",
        "",
        "## Parameter Grid",
        "",
        f"- Temperature: {', '.join(str(x) for x in sorted({row['T_K'] for row in status_rows}))} K",
        f"- Composition xB: {', '.join(str(x) for x in sorted({row['xB'] for row in status_rows}))}",
        "- Strain: 0.0 only",
        f"- Total cases: {len(status_rows)}",
        "",
        "## Status Counts",
        "",
    ]
    for key, value in sorted(counts.items()):
        report.append(f"- {key}: {value}")
    report.extend(
        [
            "",
            "## Consistency Controls",
            "",
            "- Same solver path: `jobs/submit_cnt_guide_serial.sbatch` and existing `main_cuda` minimize mode.",
            "- Same scan settings are written into each workflow metadata.",
            "- Resume is enabled by per-case `summary.json` checks.",
            "- Independent per-case logs are stored under `CNT_SCAN_WORKSTATION_RUN/results/.../job.log`.",
            "- `xB=0.00` is excluded per user correction.",
        ]
    )
    (output_root / "scan_completion_report.md").write_text("\n".join(report) + "\n", encoding="utf-8")

    conv = [
        "# CNT Scan Convergence Summary",
        "",
        f"- Catalog entries: {catalog.get('entry_count', 0)}",
        f"- Complete cases: {counts.get('complete', 0)}",
        f"- Skipped no-supersaturation cases: {counts.get('skipped_no_supersaturation', 0)}",
        f"- Failed cases: {counts.get('failed', 0) + counts.get('setup_failed', 0)}",
        "",
        "Per-case convergence should be read from each `energy_profile.csv` and `summary.json` once long-running minimizations finish.",
    ]
    (output_root / "convergence_summary.md").write_text("\n".join(conv) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid", default="256,256,256")
    parser.add_argument("--T-list-K", default="350,380,400,450")
    parser.add_argument("--xB-list", default="0.03,0.04,0.05")
    parser.add_argument("--dt", type=float, default=0.1)
    parser.add_argument("--steps", type=int, default=30000)
    parser.add_argument("--out-every", type=int, default=250)
    parser.add_argument("--csv-out-every", type=int, default=10)
    parser.add_argument("--elastic", type=int, default=1)
    parser.add_argument("--window-nm", type=float, default=0.05)
    parser.add_argument("--step-nm", type=float, default=0.025)
    parser.add_argument("--max-parallel", type=int, default=1)
    parser.add_argument("--cuda-visible-devices", default="0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    output_root = REPO_ROOT / "CNT_SCAN_WORKSTATION_RUN"
    output_root.mkdir(parents=True, exist_ok=True)
    cases = generate_cases(parse_float_list(args.T_list_K), parse_float_list(args.xB_list))
    status_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(args.max_parallel, 1)) as pool:
        futures = {pool.submit(run_scan, case, args): case for case in cases}
        for fut in as_completed(futures):
            row = fut.result()
            status_rows.append(row)
            status_rows.sort(key=lambda r: (r["T_K"], r["xB"], r["strain"]))
            catalog = build_catalog(status_rows, output_root)
            write_reports(output_root, status_rows, catalog)
            print(f"[{row['status']}] {row['case_id']} elapsed={row['elapsed_s']:.1f}s", flush=True)
    catalog = build_catalog(status_rows, output_root)
    write_reports(output_root, status_rows, catalog)
    print(f"nucleus_catalog={output_root / 'nucleus_catalog.json'}")
    print(f"job_status_table={output_root / 'job_status_table.csv'}")
    print(f"scan_completion_report={output_root / 'scan_completion_report.md'}")
    print(f"convergence_summary={output_root / 'convergence_summary.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
