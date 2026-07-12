#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import re
import shlex
import sys
from pathlib import Path

from tools.analysis.dynamic_continue_source_policy import (
    NO_SAFE_POSTCRITICAL_RADIUS_AVAILABLE,
    choose_dynamic_continue_source,
)
from tools.analysis.workflow_utils import cnt_summary_filename


DEFAULT_CSV = Path("Results_scan/constraint_cnt_strictref_test_T400_x0_0p03/current_results_master_table_fitted.csv")


def shell_quote(value: str) -> str:
    return shlex.quote(str(value))


def resolve_existing_path(
    path_text: str,
    csv_path: Path,
    results_root: Path | None = None,
    strict: bool = False,
) -> Path:
    """
    Resolve a path that may still use old Results/ or Results_r40_scan/ prefixes.
    """
    candidates: list[Path] = []
    raw = Path(path_text)
    text_variants = [path_text]
    remapped_absolute_candidates: list[Path] = []
    if path_text.startswith("Results/"):
        text_variants.append(path_text.replace("Results/", "Results_scan/", 1))
    if path_text.startswith("Results_r40_scan/"):
        text_variants.append(path_text.replace("Results_r40_scan/", "Results_scan/", 1))
    if raw.is_absolute() and results_root is not None:
        marker = "CUDA_STO_PF/"
        if marker in path_text:
            suffix = path_text.split(marker, 1)[1]
            text_variants.append(suffix)
            remapped_absolute_candidates.append(results_root / suffix)
            if suffix.startswith("Results/"):
                text_variants.append(suffix.replace("Results/", "Results_scan/", 1))
                remapped_absolute_candidates.append(results_root / suffix.replace("Results/", "Results_scan/", 1))
            if suffix.startswith("Results_r40_scan/"):
                text_variants.append(suffix.replace("Results_r40_scan/", "Results_scan/", 1))
                remapped_absolute_candidates.append(results_root / suffix.replace("Results_r40_scan/", "Results_scan/", 1))

    # When a target results_root is provided (for example, cluster-side generation from a
    # local CSV), prefer the remapped absolute path under that root over any still-existing
    # local absolute path. This keeps manifests runnable on the target machine.
    seen_remap: set[str] = set()
    for cand in remapped_absolute_candidates:
        key = str(cand)
        if key in seen_remap:
            continue
        seen_remap.add(key)
        candidates.append(cand)

    for t in text_variants:
        p = Path(t)
        if p.is_absolute():
            candidates.append(p)
        else:
            if results_root is not None:
                candidates.append(results_root / p)
            candidates.append(Path.cwd() / p)
            candidates.append(csv_path.parent.parent / p)

    seen: set[str] = set()
    for cand in candidates:
        key = str(cand.resolve()) if cand.exists() else str(cand)
        if key in seen:
            continue
        seen.add(key)
        if cand.exists():
            return cand.resolve()

    if strict:
        raise FileNotFoundError(f"Cannot resolve existing path from {path_text!r}")
    # Return the first plausible candidate so the caller can still emit a useful manifest.
    if candidates:
        return candidates[0].resolve() if candidates[0].exists() else candidates[0]
    return raw


def load_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def find_case_csv_for_base_tag(base_case_tag: str, csv_path: Path) -> Path:
    for case_csv in sorted(csv_path.parent.glob("case_*.csv")):
        with case_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first = next(reader, None)
        if first and first.get("base_case_tag") == base_case_tag:
            return case_csv.resolve()
    raise FileNotFoundError(f"No case_*.csv found for base_case_tag={base_case_tag}")


def select_row(rows: list[dict[str, str]], *, row_index: int | None, case_tag: str | None) -> dict[str, str]:
    if row_index is not None:
        if row_index < 1 or row_index > len(rows):
            raise IndexError(f"row index {row_index} out of range 1..{len(rows)}")
        return rows[row_index - 1]
    if case_tag:
        for row in rows:
            if row.get("base_case_tag") == case_tag or row.get("original_base_case_tag") == case_tag:
                return row
        raise KeyError(f"case-tag not found: {case_tag}")
    raise ValueError("Need --row or --case-tag")


def parse_case_metadata(summary_path: Path) -> dict[str, object]:
    """
    Parse metadata from:
      Results_scan/chel_T400_cuda_400x400x400_dt0.1_steps30000_r2.836nm_xB0.030/...
    """
    case_dir_path = summary_path.parent
    out_root = case_dir_path.parent.name
    case_dir = case_dir_path.name

    m = re.match(
        r"^(?P<prefix>chel|ch)_T(?P<T>\d+)_cuda_(?P<NX>\d+)x(?P<NY>\d+)x(?P<NZ>\d+)_dt(?P<dt>[0-9.]+)_steps(?P<steps>\d+)_r(?P<radius>[0-9.]+)nm_xB(?P<xb>[0-9.]+)$",
        out_root,
    )
    if not m:
        raise ValueError(f"Cannot parse run-root metadata from: {out_root}")

    prefix = m.group("prefix")
    return {
        "prefix": prefix,
        "elastic": 1 if prefix == "chel" else 0,
        "temperature_C": float(m.group("T")),
        "NX": int(m.group("NX")),
        "NY": int(m.group("NY")),
        "NZ": int(m.group("NZ")),
        "dt": float(m.group("dt")),
        "steps": int(m.group("steps")),
        "radius_label_nm": float(m.group("radius")),
        "xB_out": float(m.group("xb")),
        "output_root": out_root,
        "case_dir": case_dir,
    }


def derive_continue_files(summary_path: Path, strict: bool = False) -> tuple[Path, Path]:
    case_dir = summary_path.parent
    summary_stem = summary_path.stem
    case_tag = case_dir.name if summary_stem == "summary" else summary_stem.removeprefix("summary_")

    phi_final = case_dir / f"phi_final_{case_tag}.vtk"
    xb_final = case_dir / f"xB_final_{case_tag}.vtk"
    phi_init = case_dir / f"phi_init_{case_tag}.vtk"
    xb_init = case_dir / f"xB_init_{case_tag}.vtk"

    if phi_final.exists():
        phi_path = phi_final
    elif phi_init.exists():
        phi_path = phi_init
    else:
        if strict:
            raise FileNotFoundError(f"Neither final nor init phi vtk found in {case_dir}")
        phi_path = phi_final

    if xb_final.exists():
        xb_path = xb_final
    elif xb_init.exists():
        xb_path = xb_init
    else:
        if strict:
            raise FileNotFoundError(f"Neither final nor init xB vtk found in {case_dir}")
        xb_path = xb_final  # preferred path for diagnostics

    if strict:
        if not phi_path.exists():
            raise FileNotFoundError(phi_path)
        if not xb_path.exists():
            raise FileNotFoundError(xb_path)
    return phi_path.resolve(), xb_path.resolve()


def derive_next_discrete_summary_path(
    *,
    row: dict[str, str],
    csv_path: Path,
    peak_summary_path: Path,
) -> tuple[Path, float]:
    rc_cnt = _parse_positive_float(row.get("rc_cnt_nm"))
    if rc_cnt is None:
        raise ValueError(f"No rc_cnt_nm available for {row.get('base_case_tag')}")

    case_csv = find_case_csv_for_base_tag(row.get("base_case_tag", ""), csv_path)
    with case_csv.open(newline="", encoding="utf-8") as f:
        meta = next(csv.DictReader(f))

    rc_schur = float(meta["rc_schur_nm"])
    window_nm = float(meta["window_nm"])
    step_nm = float(meta["step_nm"])
    npts = int(round(2.0 * window_nm / step_nm)) + 1
    scan_radii = [round(rc_schur - window_nm + i * step_nm, 6) for i in range(npts)]
    choice = choose_dynamic_continue_source(
        scan_radii,
        rc_cnt,
        min_margin_nm=0.10,
        fallback_margin_nm=0.05,
        allow_exact_peak_debug=False,
    )
    if choice.status == NO_SAFE_POSTCRITICAL_RADIUS_AVAILABLE or choice.radius_nm is None:
        raise ValueError(
            f"No safe postcritical radius >= rc_cnt_nm+0.05 for rc_cnt_nm={rc_cnt:.6f} "
            f"case={row.get('base_case_tag')}"
        )
    next_radius = choice.radius_nm

    root_dir = peak_summary_path.parent.parent
    case_dir = peak_summary_path.parent
    summary_name = peak_summary_path.name

    root_name = re.sub(r"_r[0-9.]+nm_xB", f"_r{next_radius:.3f}nm_xB", root_dir.name)
    case_name = re.sub(r"_r[0-9p]+$", f"_r{next_radius:.6f}".replace(".", "p"), case_dir.name)
    summary_new = cnt_summary_filename()
    new_path = root_dir.parent / root_name / case_name / summary_new
    return new_path.resolve(), next_radius


def derive_continue_source_files(
    *,
    row: dict[str, str],
    csv_path: Path,
    summary_path: Path,
    strict: bool = False,
) -> tuple[Path, Path, Path, float | None]:
    next_summary_path, next_radius = derive_next_discrete_summary_path(
        row=row,
        csv_path=csv_path,
        peak_summary_path=summary_path,
    )
    try:
        phi_path, xb_path = derive_continue_files(next_summary_path, strict=True)
        return phi_path, xb_path, next_summary_path, next_radius
    except FileNotFoundError:
        if strict:
            raise
    phi_path, xb_path = derive_continue_files(summary_path, strict=strict)
    return phi_path, xb_path, summary_path, None


def _parse_positive_float(value: str | None) -> float | None:
    if value in (None, "", "nan", "NaN"):
        return None
    try:
        f = float(value)
    except ValueError:
        return None
    if f > 0.0 and f == f:
        return f
    return None


def get_radius_nm(
    row: dict[str, str],
    *,
    radius_mode: str = "supercritical",
    offset_nm: float = 0.1,
) -> tuple[float, str, float | None, float | None]:
    """
    Return the starting radius for dynamics-continue.

    radius_mode:
      - supercritical: use discrete peak + offset_nm (default, preferred for growth)
      - fit: use local fitted peak directly
      - discrete: use discrete peak directly
      - schur: use Schur prediction directly
    """
    rc_fit = _parse_positive_float(row.get("rc_cnt_fit_nm"))
    rc_cnt = _parse_positive_float(row.get("rc_cnt_nm"))
    rc_schur = _parse_positive_float(row.get("rc_schur_nm"))

    mode = radius_mode.lower().strip()
    if mode == "fit":
        if rc_fit is not None:
            return rc_fit, "rc_cnt_fit_nm", rc_cnt, rc_fit
        mode = "supercritical"
    elif mode == "discrete":
        if rc_cnt is not None:
            return rc_cnt, "rc_cnt_nm", rc_cnt, rc_fit
        mode = "supercritical"
    elif mode == "schur":
        if rc_schur is not None:
            return rc_schur, "rc_schur_nm", rc_cnt, rc_fit
        mode = "supercritical"

    # Default: start slightly above the discrete peak to ensure supercritical growth.
    if rc_cnt is not None:
        start = rc_cnt + max(float(offset_nm), 0.0)
        return start, f"rc_cnt_nm+{offset_nm:g}nm", rc_cnt, rc_fit
    if rc_fit is not None:
        start = rc_fit + max(float(offset_nm), 0.0)
        return start, f"rc_cnt_fit_nm+{offset_nm:g}nm", rc_cnt, rc_fit
    if rc_schur is not None:
        start = rc_schur + max(float(offset_nm), 0.0)
        return start, f"rc_schur_nm+{offset_nm:g}nm", rc_cnt, rc_fit
    raise ValueError("No usable radius found in row")


def format_env_lines(env: dict[str, object]) -> str:
    lines = []
    for k, v in env.items():
        if v is None:
            continue
        lines.append(f"export {k}={shell_quote(v)}")
    return "\n".join(lines)


def build_command_block(repo_root: Path, env: dict[str, object], template: str) -> str:
    env_block = format_env_lines(env).replace("\n", " \\\n  ")
    if template == "local":
        cmd = "bash jobs/template_run_local_continue.sh"
    else:
        cmd = "sbatch jobs/template_submit_slurm_continue.sbatch"
    return f"cd {shell_quote(repo_root)} && \\\n  {env_block} && \\\n  {cmd}"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Read current_results_master_table_fitted.csv and generate a dynamics-continue launch command or sbatch manifest."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV, help="master table CSV path")
    parser.add_argument("--row", type=int, default=None, help="1-based row index")
    parser.add_argument("--case-tag", type=str, default=None, help="base_case_tag or original_base_case_tag")
    parser.add_argument("--emit", choices=("both", "command", "manifest"), default="both")
    parser.add_argument("--template", choices=("local", "sbatch"), default="sbatch")
    parser.add_argument("--extra-steps", type=int, default=5000, help="continuation length in steps")
    parser.add_argument("--radius-mode", choices=("supercritical", "fit", "discrete", "schur"), default="supercritical", help="starting radius selection policy")
    parser.add_argument("--radius-offset-nm", type=float, default=0.1, help="offset used when radius-mode=supercritical")
    parser.add_argument("--results-root", type=Path, default=None, help="optional absolute root for Results_scan / cluster sync")
    parser.add_argument("--strict-paths", action="store_true", help="fail if summary/VTK paths cannot be resolved")
    parser.add_argument("--out", type=Path, default=None, help="optional output file for the generated snippet")
    args = parser.parse_args()

    csv_path = args.csv.expanduser().resolve()
    rows = load_rows(csv_path)
    row = select_row(rows, row_index=args.row, case_tag=args.case_tag)
    results_root = args.results_root.expanduser().resolve() if args.results_root else None
    summary_path = resolve_existing_path(row["summary_path"], csv_path, results_root=results_root, strict=args.strict_paths)
    phi_vtk, xb_vtk, source_summary_path, source_radius_nm = derive_continue_source_files(
        row=row,
        csv_path=csv_path,
        summary_path=summary_path,
        strict=args.strict_paths,
    )
    meta = parse_case_metadata(summary_path)

    radius_nm, radius_source, rc_cnt_nm, rc_fit_nm = get_radius_nm(
        row,
        radius_mode=args.radius_mode,
        offset_nm=args.radius_offset_nm,
    )
    if source_radius_nm is not None:
        radius_nm = source_radius_nm
        radius_source = "next_discrete_case_radius_nm"
    mode = "dynamics-continue"
    env_local = {
        "MODE": mode,
        "NX": meta["NX"],
        "NY": meta["NY"],
        "NZ": meta["NZ"],
        "DT": meta["dt"],
        "NSTEPS": args.extra_steps,
        "OUT_EVERY": max(1, args.extra_steps // 2),
        "CSV_OUT_EVERY": 10,
        "ELASTIC": meta["elastic"],
        "TEMP_C": meta["temperature_C"],
        "RADIUS": f"{radius_nm:.6f}",
        "XB_OUT": f"{meta['xB_out']:.6f}",
        "CONTINUE_PHI_VTK": str(phi_vtk),
        "CONTINUE_XB_VTK": str(xb_vtk),
        "MINIMIZE_FULL_MODEL": 1,
    }
    env_sbatch = dict(env_local)
    env_sbatch["MIN_DT"] = env_sbatch.pop("DT")

    # human-readable report
    report_lines = [
        "Selected row:",
        f"  base_case_tag = {row.get('base_case_tag')}",
        f"  mode          = {row.get('mode')}",
        f"  strain        = {row.get('strain')}",
        f"  rc_cnt_fit_nm = {row.get('rc_cnt_fit_nm')}",
        f"  rc_cnt_nm     = {row.get('rc_cnt_nm')}",
        f"  start_radius   = {radius_nm:.6f} ({radius_source})",
        f"  status        = {row.get('status')}",
        f"  summary_path   = {summary_path}",
        f"  continue_source_summary_path = {source_summary_path}",
        f"  continue_source_radius_nm    = {source_radius_nm}",
        f"  phi_final_vtk  = {phi_vtk}",
        f"  xB_final_vtk   = {xb_vtk}",
        f"  results_root   = {results_root if results_root else '<auto>'}",
        "",
        f"Generated {args.template} snippet for mode={mode}:",
        build_command_block(Path(__file__).resolve().parents[2], env_local if args.template == "local" else env_sbatch, args.template),
    ]
    report = "\n".join(report_lines)
    print(report)

    if args.out:
        args.out.write_text(report + "\n", encoding="utf-8")
        print(f"\n[wrote] {args.out}")

    if args.emit == "command":
        print("\n[command]")
        print(build_command_block(Path(__file__).resolve().parents[2], env_local, "local"))
    elif args.emit == "manifest":
        print("\n[manifest]")
        print(build_command_block(Path(__file__).resolve().parents[2], env_sbatch, "sbatch"))
    else:
        print("\n[command]")
        print(build_command_block(Path(__file__).resolve().parents[2], env_local, "local"))
        print("\n[manifest]")
        print(build_command_block(Path(__file__).resolve().parents[2], env_sbatch, "sbatch"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
