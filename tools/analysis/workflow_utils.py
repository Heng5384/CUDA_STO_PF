#!/usr/bin/env python3
from __future__ import annotations

import math
from pathlib import Path


def format_xb_tag(xb_out: float) -> str:
    return f"xB{xb_out:.3f}".replace(".", "p")


def format_temp_tag(temp_c: float) -> str:
    return f"T{int(round(temp_c))}"


def workflow_name(temp_c: float, xb_out: float) -> str:
    return f"{format_temp_tag(temp_c)}_{format_xb_tag(xb_out)}"


def build_workflow_dir(workflow_root: Path, temp_c: float, xb_out: float) -> Path:
    return workflow_root / workflow_name(temp_c, xb_out)


def ensure_workflow_subdirs(workflow_dir: Path) -> dict[str, Path]:
    subdirs = {
        "root": workflow_dir,
        "input": workflow_dir / "input",
        "raw": workflow_dir / "raw",
        "cnt_scan": workflow_dir / "cnt_scan",
        "continue_dynamic": workflow_dir / "continue_dynamic",
        "summaries": workflow_dir / "summaries",
    }
    for path in subdirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return subdirs


def build_cnt_prefix(temp_c: float, xb_out: float) -> str:
    return f"cntcon_{format_temp_tag(temp_c)}_{format_xb_tag(xb_out)}_strictref"


def radius_root_tag(radius_nm: float) -> str:
    return f"{radius_nm:.3f}"


def radius_case_tag(radius_nm: float) -> str:
    return f"{radius_nm:.6f}".replace(".", "p")


def case_run_root_rel(
    *,
    results_root_rel: str | Path = "Results",
    elastic: int,
    temp_c: float,
    nx: int,
    ny: int,
    nz: int,
    dt: float,
    nsteps: int,
    radius_nm: float,
    xb_out: float,
) -> Path:
    prefix = "chel" if int(elastic) else "ch"
    name = (
        f"{prefix}_{format_temp_tag(temp_c)}_cuda_{nx}x{ny}x{nz}_dt{dt:g}_steps{nsteps}"
        f"_r{radius_root_tag(radius_nm)}nm_xB{xb_out:.3f}"
    )
    return Path(results_root_rel) / name


def case_dir_name(base_case_tag: str, radius_nm: float) -> str:
    return f"{base_case_tag}_r{radius_case_tag(radius_nm)}"


def case_output_rel(
    *,
    results_root_rel: str | Path = "Results",
    base_case_tag: str,
    elastic: int,
    temp_c: float,
    nx: int,
    ny: int,
    nz: int,
    dt: float,
    nsteps: int,
    radius_nm: float,
    xb_out: float,
) -> dict[str, str]:
    root_rel = case_run_root_rel(
        results_root_rel=results_root_rel,
        elastic=elastic,
        temp_c=temp_c,
        nx=nx,
        ny=ny,
        nz=nz,
        dt=dt,
        nsteps=nsteps,
        radius_nm=radius_nm,
        xb_out=xb_out,
    )
    case_name = case_dir_name(base_case_tag, radius_nm)
    case_dir_rel = root_rel / case_name
    return {
        "output_root_rel": str(root_rel),
        "case_dir_rel": str(case_dir_rel),
        "summary_rel": str(case_dir_rel / f"summary_{case_name}.txt"),
        "energy_csv_rel": str(case_dir_rel / f"energy_minimize_{case_name}.csv"),
        "phi_final_rel": str(case_dir_rel / f"phi_final_{case_name}.vtk"),
        "xb_final_rel": str(case_dir_rel / f"xB_final_{case_name}.vtk"),
        "pf_input_rel": str(case_dir_rel / "pf_input.params"),
    }


def voxel_count_to_radius_nm(voxel_count: int, dx_nm: float) -> float:
    volume_nm3 = float(voxel_count) * (dx_nm ** 3)
    return ((3.0 * volume_nm3) / (4.0 * math.pi)) ** (1.0 / 3.0)
