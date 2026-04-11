#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import subprocess
import sys
from collections import defaultdict
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.prepare_dynamics_continue_from_summary import (  # noqa: E402
    DEFAULT_CSV,
    derive_continue_source_files,
    get_radius_nm,
    load_rows,
    parse_case_metadata,
    resolve_existing_path,
)


QUEUE_ORDER = [
    ("gpu_vip_24h", "gpu_vip_24h"),
    ("gpu_uvip", "gpu_uvip"),
]


def eligible_rows(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    out = []
    for row in rows:
        status = (row.get("status") or "").strip()
        if status == "complete_peak_found":
            out.append(row)
    return out


def queue_for_index(idx: int, start_with: str) -> tuple[str, str]:
    normalized = start_with.lower()
    if normalized not in {"24h", "uvip"}:
        raise ValueError("--start-queue must be either 24h or uvip")
    base = QUEUE_ORDER if normalized == "24h" else list(reversed(QUEUE_ORDER))
    return base[idx % 2]


def manifest_row(
    row: dict[str, str],
    source_index: int,
    queue_name: str,
    csv_path: Path,
    results_root: Path | None,
    extra_steps: int,
    mode: str,
    dt_override: float | None,
    strict_paths: bool,
    radius_mode: str,
    radius_offset_nm: float,
) -> dict[str, str]:
    summary_path = resolve_existing_path(row["summary_path"], csv_path, results_root=results_root, strict=strict_paths)
    phi_vtk, xb_vtk, source_summary_path, source_radius_nm = derive_continue_source_files(
        row=row,
        csv_path=csv_path,
        summary_path=summary_path,
        strict=strict_paths,
    )
    meta = parse_case_metadata(summary_path)
    radius_nm, radius_source, rc_cnt_nm, rc_fit_nm = get_radius_nm(
        row,
        radius_mode=radius_mode,
        offset_nm=radius_offset_nm,
    )
    if source_radius_nm is not None:
        radius_nm = source_radius_nm
        radius_source = "next_discrete_case_radius_nm"
    dt_value = meta["dt"] if dt_override is None else dt_override

    return {
        "SOURCE_ROW_INDEX": str(source_index + 1),
        "BASE_CASE_TAG": row.get("base_case_tag", ""),
        "ORIGINAL_BASE_CASE_TAG": row.get("original_base_case_tag", ""),
        "QUEUE_NAME": queue_name,
        "MODE": mode,
        "NX": str(meta["NX"]),
        "NY": str(meta["NY"]),
        "NZ": str(meta["NZ"]),
        "DT": f"{dt_value}",
        "MIN_DT": f"{dt_value}",
        "NSTEPS": str(extra_steps),
        "OUT_EVERY": str(max(1, extra_steps // 2)),
        "CSV_OUT_EVERY": "10",
        "ELASTIC": str(meta["elastic"]),
        "TEMP_C": f"{meta['temperature_C']}",
        "RADIUS": f"{radius_nm:.6f}",
        "XB_OUT": f"{meta['xB_out']:.6f}",
        "CONTINUE_PHI_VTK": str(phi_vtk),
        "CONTINUE_XB_VTK": str(xb_vtk),
        "MINIMIZE_FULL_MODEL": "1",
        "MIN_RMS_DPHI_THRESHOLD": "1e-5",
        "MIN_RMS_DY_THRESHOLD": "1e-5",
        "MIN_ENERGY_DIFF_REL_THRESHOLD": "1e-7",
        "MIN_RMS_RES_FOR_ENERGY_PLATEAU": "1e-5",
        "MIN_RMS_RES_THRESHOLD": "1e-4",
        "MIN_VOL_ERR_REL_THRESHOLD": "3e-2",
        "MIN_CONVERGENCE_STEPS": "100",
        "MIN_DT_SAFETY_LIMIT": "1e-6",
        "ETA_LAMBDA_VOL": "0.9",
        "POST_PROJ_ITERS": "1",
        "SUMMARY_PATH": str(summary_path),
        "CONTINUE_SOURCE_SUMMARY_PATH": str(source_summary_path),
        "CONTINUE_SOURCE_RADIUS_NM": "" if source_radius_nm is None else f"{source_radius_nm:.6f}",
        "PHI_FINAL_VTK": str(phi_vtk),
        "XB_FINAL_VTK": str(xb_vtk),
        "STATUS": row.get("status", ""),
        "STRAIN": row.get("strain", ""),
        "RC_SCHUR_NM": row.get("rc_schur_nm", ""),
        "RC_CNT_NM": row.get("rc_cnt_nm", ""),
        "RC_CNT_FIT_NM": row.get("rc_cnt_fit_nm", ""),
        "START_RADIUS_NM": f"{radius_nm:.6f}",
        "START_RADIUS_SOURCE": radius_source,
        "START_RADIUS_OFFSET_NM": f"{radius_offset_nm:.6f}",
        "DISCRETE_RC_NM": "" if rc_cnt_nm is None else f"{rc_cnt_nm:.6f}",
        "FIT_RC_NM": "" if rc_fit_nm is None else f"{rc_fit_nm:.6f}",
        "F_CNT_PEAK_HAT": row.get("F_CNT_peak_hat", ""),
    }


def write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def submit_serial_job(
    manifest_path: Path,
    queue: str,
    manifest_name: str,
    mode: str,
    submit: bool,
) -> None:
    cmd = [
        "sbatch",
        f"--partition={queue}",
        f"--qos={queue}",
        f"--job-name={manifest_name}_{queue}",
        f"--export=ALL,MANIFEST_CSV={manifest_path},MANIFEST_NAME={manifest_name},MODE={mode}",
        str(REPO_ROOT / "jobs" / "submit_continue_manifest_serial.sbatch"),
    ]
    print("[submit]", " ".join(map(str, cmd)))
    if submit:
        subprocess.run(cmd, check=True, cwd=REPO_ROOT)


def main() -> int:
    parser = argparse.ArgumentParser(description="Split all completed strictref cases into 24h and uvip continue jobs.")
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="master table CSV path")
    parser.add_argument("--results-root", type=Path, default=None, help="optional repo root to resolve summary/VTK paths")
    parser.add_argument("--extra-steps", type=int, default=5000, help="continue step count")
    parser.add_argument("--mode", choices=("dynamics-continue", "minimize-continue"), default="dynamics-continue", help="continue mode")
    parser.add_argument("--dt", type=float, default=None, help="override dt for all generated continue cases")
    parser.add_argument("--radius-mode", choices=("supercritical", "fit", "discrete", "schur"), default="supercritical", help="starting radius selection policy")
    parser.add_argument("--radius-offset-nm", type=float, default=0.1, help="offset added in supercritical mode")
    parser.add_argument("--start-queue", choices=("24h", "uvip"), default="24h", help="first queue in alternating assignment")
    parser.add_argument("--strict-paths", action="store_true", help="fail if summary/VTK paths cannot be resolved")
    parser.add_argument("--manifest-dir", type=Path, default=None, help="where to write queue manifests")
    parser.add_argument("--dry-run", action="store_true", help="do not invoke sbatch, only print commands")
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    results_root = args.results_root.expanduser().resolve() if args.results_root else None
    rows = eligible_rows(load_rows(csv_path))
    if not rows:
        print("[warn] no eligible rows found.")
        return 0

    manifest_base = args.manifest_dir.expanduser().resolve() if args.manifest_dir else (csv_path.parent / "dynamics_continue_manifests")
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    selected_rows = []
    for i, row in enumerate(rows):
        queue_alias, queue_name = queue_for_index(i, args.start_queue)
        manifest = manifest_row(
            row,
            i,
            queue_alias,
            csv_path,
            results_root,
            args.extra_steps,
            args.mode,
            args.dt,
            args.strict_paths,
            args.radius_mode,
            args.radius_offset_nm,
        )
        grouped[queue_name].append(manifest)
        selected_rows.append((i + 1, queue_name, row.get("base_case_tag", ""), row.get("strain", ""), row.get("status", "")))

    manifest_paths = {}
    for queue_name, manifest_rows in grouped.items():
        mode_slug = args.mode.replace("-", "_")
        manifest_path = manifest_base / f"{queue_name}_{mode_slug}_manifest.csv"
        write_manifest(manifest_path, manifest_rows)
        manifest_paths[queue_name] = manifest_path

    print("[summary] selected rows:")
    for idx, queue_name, base_tag, strain, status in selected_rows:
        print(f"  row {idx:02d} -> {queue_name:12s} {base_tag} strain={strain} status={status}")
    for queue_name, manifest_path in manifest_paths.items():
        print(f"[manifest] {queue_name}: {manifest_path} ({len(grouped[queue_name])} rows)")
    print(f"[policy] mode={args.mode} dt={args.dt if args.dt is not None else 'source'} radius_mode={args.radius_mode} radius_offset_nm={args.radius_offset_nm}")

    if not args.dry_run:
        for queue_name, manifest_path in manifest_paths.items():
            submit_serial_job(
                manifest_path=manifest_path,
                queue=queue_name,
                manifest_name=f"{args.mode.replace('-', '_')}_{queue_name}",
                mode=args.mode,
                submit=True,
            )
    else:
        for queue_name, manifest_path in manifest_paths.items():
            submit_serial_job(
                manifest_path=manifest_path,
                queue=queue_name,
                manifest_name=f"{args.mode.replace('-', '_')}_{queue_name}",
                mode=args.mode,
                submit=False,
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
