#!/usr/bin/env python3
"""
Audit same-strain-reference-subtracted CNT barrier components.

This script analyzes constrained nucleus / CNT radius-scan workflows after
explicit matrix-only same-strain references have been generated. For each
base_case_tag, it builds a reference-subtracted free-energy profile

  DeltaF_i_hat(r) = F_i_hat(r) - F_i_ref_hat

for i in {surf, chem, el, total}, then converts each component to physical
barrier units with

  DeltaG_i_J(r) = DeltaF_i_hat(r) * w_phys * V_box

using the same physical energy-density scale and box volume already used in
the CNT summary / nucleation-rate workflow.

The goal is not to repair Zeldovich extraction. The goal is to determine why
the same-strain-reference-subtracted excess barrier remains high by separating
surface, chemical, and elastic contributions at the saddle and along the full
radius scan.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.analyze_cnt_peak_table import parse_summary_file

K_B_J_PER_K = 1.380649e-23


@dataclass
class ReferenceRow:
    base_case_tag: str
    case_dir: Path
    T_input: float | None
    T_K: float | None
    xB0: float | None
    xB_out: float | None
    mode: str
    exx: float | None
    eyy: float | None
    ezz: float | None
    exy: float | None
    exz: float | None
    eyz: float | None
    Nx: int | None
    Ny: int | None
    Nz: int | None
    dx_nm: float | None
    dy_nm: float | None
    dz_nm: float | None
    V_box_m3: float | None
    w_phys_J_m3: float | None
    F_surf_ref_hat: float | None
    F_chem_ref_hat: float | None
    F_el_ref_hat: float | None
    F_total_ref_hat: float | None
    mean_h: float | None
    mean_phi: float | None
    mean_xB: float | None
    voxel_count: int | None
    status: str
    warnings: list[str]
    source: str


@dataclass
class ProfileRow:
    base_case_tag: str
    case_dir: Path
    radius_nominal_nm: float | None
    r_eff_nm: float | None
    voxel_count: int | None
    mean_h: float | None
    F_surf_hat: float | None
    F_chem_hat: float | None
    F_el_hat: float | None
    F_total_hat: float | None
    DeltaF_surf_hat: float | None
    DeltaF_chem_hat: float | None
    DeltaF_el_hat: float | None
    DeltaF_total_hat: float | None
    DeltaG_surf_kBT: float | None
    DeltaG_chem_kBT: float | None
    DeltaG_el_kBT: float | None
    DeltaG_total_kBT: float | None
    is_saddle_candidate: int
    warnings: list[str]

    def to_row(self) -> dict[str, Any]:
        return {
            "base_case_tag": self.base_case_tag,
            "case_dir": str(self.case_dir),
            "radius_nominal": self.radius_nominal_nm,
            "r_eff_nm": self.r_eff_nm,
            "voxel_count": self.voxel_count,
            "mean_h": self.mean_h,
            "DeltaG_surf_kBT": self.DeltaG_surf_kBT,
            "DeltaG_chem_kBT": self.DeltaG_chem_kBT,
            "DeltaG_el_kBT": self.DeltaG_el_kBT,
            "DeltaG_total_kBT": self.DeltaG_total_kBT,
            "is_saddle_candidate": self.is_saddle_candidate,
            "warnings": " | ".join(sorted(dict.fromkeys(self.warnings))),
        }


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        number = float(text)
    except ValueError:
        return None
    return number if math.isfinite(number) else None


def safe_int(value: Any) -> int | None:
    number = safe_float(value)
    if number is None:
        return None
    return int(round(number))


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def find_first_existing(case_dir: Path, preferred: Path, pattern: str) -> Path | None:
    if preferred.exists():
        return preferred
    matches = sorted(case_dir.glob(pattern))
    return matches[0] if matches else None


def nominal_radius_from_case_name(case_name: str) -> float | None:
    marker = "_r"
    idx = case_name.rfind(marker)
    if idx < 0:
        return None
    token = case_name[idx + len(marker):]
    try:
        return float(token.replace("p", ".").replace("m", "-"))
    except ValueError:
        return None


def equivalent_radius_nm(voxel_count: int | None, dx_nm: float | None, dy_nm: float | None, dz_nm: float | None) -> float | None:
    if voxel_count is None or dx_nm is None or dy_nm is None or dz_nm is None:
        return None
    volume_nm3 = float(voxel_count) * dx_nm * dy_nm * dz_nm
    if volume_nm3 <= 0.0:
        return 0.0
    return ((3.0 * volume_nm3) / (4.0 * math.pi)) ** (1.0 / 3.0)


def component_kbt(delta_hat: float | None, ref: ReferenceRow) -> float | None:
    if delta_hat is None or ref.w_phys_J_m3 is None or ref.V_box_m3 is None or ref.T_K is None or ref.T_K <= 0.0:
        return None
    delta_g_j = delta_hat * ref.w_phys_J_m3 * ref.V_box_m3
    return delta_g_j / (K_B_J_PER_K * ref.T_K)


def load_reference_rows(reference_csv: Path) -> dict[str, ReferenceRow]:
    references: dict[str, ReferenceRow] = {}
    for row in read_csv_rows(reference_csv):
        base = row["base_case_tag"]
        references[base] = ReferenceRow(
            base_case_tag=base,
            case_dir=Path(row["case_dir"]),
            T_input=safe_float(row.get("T_input")),
            T_K=safe_float(row.get("T_K")),
            xB0=safe_float(row.get("xB0")),
            xB_out=safe_float(row.get("xB_out")),
            mode=row.get("mode", ""),
            exx=safe_float(row.get("exx")),
            eyy=safe_float(row.get("eyy")),
            ezz=safe_float(row.get("ezz")),
            exy=safe_float(row.get("exy")),
            exz=safe_float(row.get("exz")),
            eyz=safe_float(row.get("eyz")),
            Nx=safe_int(row.get("Nx")),
            Ny=safe_int(row.get("Ny")),
            Nz=safe_int(row.get("Nz")),
            dx_nm=safe_float(row.get("dx_nm")),
            dy_nm=safe_float(row.get("dy_nm")),
            dz_nm=safe_float(row.get("dz_nm")),
            V_box_m3=safe_float(row.get("V_box_m3")),
            w_phys_J_m3=safe_float(row.get("w_phys_J_m3")),
            F_surf_ref_hat=safe_float(row.get("F_surf_ref_hat")),
            F_chem_ref_hat=safe_float(row.get("F_chem_ref_hat")),
            F_el_ref_hat=safe_float(row.get("F_el_ref_hat")),
            F_total_ref_hat=safe_float(row.get("F_total_ref_hat")),
            mean_h=safe_float(row.get("mean_h")),
            mean_phi=safe_float(row.get("mean_phi")),
            mean_xB=safe_float(row.get("mean_xB")),
            voxel_count=safe_int(row.get("voxel_count")),
            status=row.get("status", ""),
            warnings=[w.strip() for w in str(row.get("warnings", "")).split("|") if w.strip()],
            source=row.get("_reference_path") or row.get("case_dir", ""),
        )
    return references


def supplement_reference(reference: ReferenceRow, guide_row: dict[str, str], workflow_root: Path) -> ReferenceRow:
    dx_nm = reference.dx_nm if reference.dx_nm is not None else safe_float(guide_row.get("pf_dx_nm"))
    dy_nm = reference.dy_nm if reference.dy_nm is not None else dx_nm
    dz_nm = reference.dz_nm if reference.dz_nm is not None else dx_nm
    nx = reference.Nx if reference.Nx is not None else safe_int(guide_row.get("nx"))
    ny = reference.Ny if reference.Ny is not None else safe_int(guide_row.get("ny"))
    nz = reference.Nz if reference.Nz is not None else safe_int(guide_row.get("nz"))
    t_input = reference.T_input if reference.T_input is not None else safe_float(guide_row.get("T_C"))
    t_k = reference.T_K if reference.T_K is not None else (t_input + 273.15 if t_input is not None else None)
    x_b_out = reference.xB_out if reference.xB_out is not None else safe_float(guide_row.get("xB_out"))
    x_b0 = reference.xB0 if reference.xB0 is not None else x_b_out
    v_box_m3 = reference.V_box_m3
    if v_box_m3 is None and None not in (dx_nm, dy_nm, dz_nm, nx, ny, nz):
        v_box_m3 = int(nx) * int(ny) * int(nz) * (dx_nm * 1.0e-9) * (dy_nm * 1.0e-9) * (dz_nm * 1.0e-9)
    w_phys = reference.w_phys_J_m3
    if w_phys is None:
        phys_json = workflow_root / "input" / "physical_inputs.json"
        if phys_json.exists():
            payload = json.loads(phys_json.read_text(encoding="utf-8"))
            gamma = safe_float(payload.get("gamma"))
            lambda_sm = safe_float(payload.get("lambda_sm"))
            if gamma is not None and lambda_sm is not None and lambda_sm > 0.0:
                w_phys = 12.0 * gamma / lambda_sm
    warnings = list(reference.warnings)
    if reference.V_box_m3 is None and v_box_m3 is not None:
        warnings.append("supplemented V_box_m3 from guide row grid and dx")
    if reference.w_phys_J_m3 is None and w_phys is not None:
        warnings.append("supplemented w_phys_J_m3 from physical_inputs.json")
    return ReferenceRow(
        base_case_tag=reference.base_case_tag,
        case_dir=reference.case_dir,
        T_input=t_input,
        T_K=t_k,
        xB0=x_b0,
        xB_out=x_b_out,
        mode=reference.mode or guide_row.get("mode", ""),
        exx=reference.exx if reference.exx is not None else safe_float(guide_row.get("E0_xx")),
        eyy=reference.eyy if reference.eyy is not None else safe_float(guide_row.get("E0_yy")),
        ezz=reference.ezz if reference.ezz is not None else safe_float(guide_row.get("E0_zz")),
        exy=reference.exy if reference.exy is not None else safe_float(guide_row.get("E0_xy")),
        exz=reference.exz if reference.exz is not None else safe_float(guide_row.get("E0_xz")),
        eyz=reference.eyz if reference.eyz is not None else safe_float(guide_row.get("E0_yz")),
        Nx=nx,
        Ny=ny,
        Nz=nz,
        dx_nm=dx_nm,
        dy_nm=dy_nm,
        dz_nm=dz_nm,
        V_box_m3=v_box_m3,
        w_phys_J_m3=w_phys,
        F_surf_ref_hat=reference.F_surf_ref_hat,
        F_chem_ref_hat=reference.F_chem_ref_hat,
        F_el_ref_hat=reference.F_el_ref_hat,
        F_total_ref_hat=reference.F_total_ref_hat,
        mean_h=reference.mean_h,
        mean_phi=reference.mean_phi,
        mean_xB=reference.mean_xB,
        voxel_count=reference.voxel_count,
        status=reference.status,
        warnings=warnings,
        source=reference.source,
    )


def make_reference_from_energy(guide_row: dict[str, str], workflow_root: Path) -> ReferenceRow | None:
    case_dir = workflow_root.parent.parent / guide_row["case_dir_rel"]
    preferred = workflow_root.parent.parent / guide_row["energy_csv_rel"]
    energy_csv = find_first_existing(case_dir, preferred, "energy_minimize_*.csv")
    if energy_csv is None:
        return None
    rows = read_csv_rows(energy_csv)
    if not rows:
        return None
    last = rows[-1]
    t_c = safe_float(guide_row.get("T_C"))
    t_k = t_c + 273.15 if t_c is not None else None
    dx_nm = safe_float(guide_row.get("pf_dx_nm"))
    nx = safe_int(guide_row.get("nx"))
    ny = safe_int(guide_row.get("ny"))
    nz = safe_int(guide_row.get("nz"))
    v_box_m3 = None
    if None not in (dx_nm, nx, ny, nz):
        v_box_m3 = int(nx) * int(ny) * int(nz) * (dx_nm * 1.0e-9) ** 3
    phys_json = workflow_root / "input" / "physical_inputs.json"
    w_phys = None
    if phys_json.exists():
        payload = json.loads(phys_json.read_text(encoding="utf-8"))
        gamma = safe_float(payload.get("gamma"))
        lambda_sm = safe_float(payload.get("lambda_sm"))
        if gamma is not None and lambda_sm is not None and lambda_sm > 0.0:
            w_phys = 12.0 * gamma / lambda_sm
    return ReferenceRow(
        base_case_tag=guide_row["base_case_tag"],
        case_dir=case_dir,
        T_input=t_c,
        T_K=t_k,
        xB0=safe_float(guide_row.get("xB_out")),
        xB_out=safe_float(guide_row.get("xB_out")),
        mode=guide_row.get("mode", ""),
        exx=safe_float(guide_row.get("E0_xx")),
        eyy=safe_float(guide_row.get("E0_yy")),
        ezz=safe_float(guide_row.get("E0_zz")),
        exy=safe_float(guide_row.get("E0_xy")),
        exz=safe_float(guide_row.get("E0_xz")),
        eyz=safe_float(guide_row.get("E0_yz")),
        Nx=nx,
        Ny=ny,
        Nz=nz,
        dx_nm=dx_nm,
        dy_nm=dx_nm,
        dz_nm=dx_nm,
        V_box_m3=v_box_m3,
        w_phys_J_m3=w_phys,
        F_surf_ref_hat=safe_float(last.get("F_surf_hat")),
        F_chem_ref_hat=safe_float(last.get("F_chem_excess_hat")),
        F_el_ref_hat=safe_float(last.get("F_el_hat")),
        F_total_ref_hat=safe_float(last.get("F_total_excess_hat")),
        mean_h=safe_float(last.get("mean_h")),
        mean_phi=0.0,
        mean_xB=safe_float(guide_row.get("xB_out")),
        voxel_count=0,
        status="fallback_from_energy_csv",
        warnings=["workflow-level reference csv missing; used last-row energy_minimize fallback"],
        source=str(energy_csv),
    )


def pick_saddle(profile_rows: list[ProfileRow]) -> ProfileRow | None:
    usable = [row for row in profile_rows if row.DeltaF_total_hat is not None]
    if not usable:
        return None
    return max(usable, key=lambda row: float(row.DeltaF_total_hat))


def representative_case_keys(groups: dict[str, list[ProfileRow]]) -> list[str]:
    keys = sorted(groups)
    if len(keys) <= 6:
        return keys
    chosen = [keys[0], keys[len(keys) // 4], keys[len(keys) // 2], keys[(3 * len(keys)) // 4], keys[-1]]
    unique: list[str] = []
    for key in chosen:
        if key not in unique:
            unique.append(key)
    return unique


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                seen.add(key)
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def make_plots(
    audit_rows: list[dict[str, Any]],
    profile_rows: list[ProfileRow],
    out_dir: Path,
) -> list[str]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    plot_dir = out_dir / "plots"
    plot_dir.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []

    valid_audit = audit_rows
    if valid_audit:
        labels = [row["base_case_tag"] for row in valid_audit]
        xs = np.arange(len(labels))
        surf = np.array([safe_float(row.get("DeltaG_surf_kBT")) or 0.0 for row in valid_audit], dtype=float)
        chem = np.array([safe_float(row.get("DeltaG_chem_kBT")) or 0.0 for row in valid_audit], dtype=float)
        el = np.array([safe_float(row.get("DeltaG_el_kBT")) or 0.0 for row in valid_audit], dtype=float)
        fig, ax = plt.subplots(figsize=(12.0, 5.5), constrained_layout=True)
        ax.bar(xs, surf, label="surface")
        ax.bar(xs, chem, bottom=surf, label="chemical")
        ax.bar(xs, el, bottom=surf + chem, label="elastic")
        ax.set_title("Excess CNT barrier components after same-strain reference subtraction")
        ax.set_xlabel("Case")
        ax.set_ylabel("DeltaG component [kB T]")
        ax.set_xticks(xs)
        ax.set_xticklabels(labels, rotation=75, ha="right", fontsize=7)
        ax.grid(True, axis="y", alpha=0.25)
        ax.legend(fontsize=8)
        target = plot_dir / "barrier_components_stacked.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        paths.append(str(target))

    by_case: dict[str, list[ProfileRow]] = defaultdict(list)
    for row in profile_rows:
        by_case[row.base_case_tag].append(row)
    rep_keys = representative_case_keys(by_case)

    if rep_keys:
        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        for key in rep_keys:
            items = sorted(
                [row for row in by_case[key] if row.r_eff_nm is not None and row.DeltaG_total_kBT is not None],
                key=lambda row: float(row.r_eff_nm),
            )
            if not items:
                continue
            ax.plot(
                [float(row.r_eff_nm) for row in items],
                [float(row.DeltaG_total_kBT) for row in items],
                marker="o",
                linewidth=1.6,
                markersize=4.0,
                label=key,
            )
        ax.set_title("Reference-subtracted free-energy profile versus effective radius")
        ax.set_xlabel("Effective radius [nm]")
        ax.set_ylabel("DeltaG_total [kB T]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        target = plot_dir / "DeltaG_total_vs_r_eff_nm.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        paths.append(str(target))

        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        for key in rep_keys:
            items = sorted(
                [row for row in by_case[key] if row.r_eff_nm is not None and row.DeltaG_chem_kBT is not None],
                key=lambda row: float(row.r_eff_nm),
            )
            if not items:
                continue
            ax.plot(
                [float(row.r_eff_nm) for row in items],
                [float(row.DeltaG_chem_kBT) for row in items],
                marker="o",
                linewidth=1.6,
                markersize=4.0,
                label=key,
            )
        ax.set_title("Chemical driving force contribution along constrained nucleus path")
        ax.set_xlabel("Effective radius [nm]")
        ax.set_ylabel("DeltaG_chem [kB T]")
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7)
        target = plot_dir / "DeltaG_chem_vs_r_eff_nm.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        paths.append(str(target))

    elastic_rows = [
        row for row in audit_rows
        if safe_float(row.get("DeltaG_el_kBT")) is not None
    ]
    if elastic_rows:
        grouped: dict[str, list[float]] = defaultdict(list)
        for row in elastic_rows:
            grouped[str(row.get("mode", "unknown"))].append(float(row["DeltaG_el_kBT"]))
        fig, ax = plt.subplots(figsize=(8.0, 5.0), constrained_layout=True)
        labels = sorted(grouped)
        ax.boxplot([grouped[label] for label in labels], tick_labels=labels)
        ax.set_title("Elastic excess contribution after removing uniform strained background")
        ax.set_xlabel("Strain mode")
        ax.set_ylabel("DeltaG_el [kB T]")
        ax.grid(True, axis="y", alpha=0.25)
        target = plot_dir / "DeltaG_el_vs_mode.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        paths.append(str(target))

    supersat_rows = [
        row for row in audit_rows
        if safe_float(row.get("xB_out_if_available")) is not None and safe_float(row.get("DeltaG_total_kBT")) is not None
    ]
    x_values = {safe_float(row.get("xB_out_if_available")) for row in supersat_rows}
    if supersat_rows and len({x for x in x_values if x is not None}) > 1:
        fig, ax = plt.subplots(figsize=(7.0, 5.0), constrained_layout=True)
        ax.scatter(
            [float(row["xB_out_if_available"]) for row in supersat_rows if safe_float(row.get("xB_out_if_available")) is not None],
            [float(row["DeltaG_total_kBT"]) for row in supersat_rows if safe_float(row.get("xB_out_if_available")) is not None],
            s=40,
        )
        ax.set_title("Excess nucleation barrier versus local supersaturation")
        ax.set_xlabel("xB_out")
        ax.set_ylabel("DeltaG_total [kB T]")
        ax.grid(True, alpha=0.25)
        target = plot_dir / "DeltaG_total_vs_xB_out.png"
        fig.savefig(target, dpi=180)
        plt.close(fig)
        paths.append(str(target))

    return paths


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Audit same-strain-reference-subtracted CNT barrier components.")
    parser.add_argument("--workflow-root", required=True, help="workflow root like Results/workflows/T400_xB0p050")
    parser.add_argument("--repo-root", default=".", help="repo root containing the workflow")
    parser.add_argument(
        "--out",
        default=None,
        help="output directory; default is <workflow-root>/analysis/energy_component_barrier",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    repo_root = Path(args.repo_root).expanduser().resolve()
    workflow_root = (repo_root / args.workflow_root).resolve() if not Path(args.workflow_root).is_absolute() else Path(args.workflow_root).resolve()
    out_dir = (
        Path(args.out).expanduser().resolve()
        if args.out
        else (workflow_root / "analysis" / "energy_component_barrier")
    )

    guide_csv = workflow_root / "input" / "guide_cnt_scan.csv"
    summary_csv = workflow_root / "cnt_scan" / "current_results_master_table_fitted.csv"
    reference_csv = workflow_root / "cnt_scan" / "reference_energy.csv"

    guide_rows = read_csv_rows(guide_csv)
    summary_rows = {row["base_case_tag"]: row for row in read_csv_rows(summary_csv)}
    references = load_reference_rows(reference_csv) if reference_csv.exists() else {}

    grouped_scan_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    grouped_reference_rows: dict[str, dict[str, str]] = {}
    for row in guide_rows:
        if row.get("row_type") == "reference":
            grouped_reference_rows[row["base_case_tag"]] = row
        else:
            grouped_scan_rows[row["base_case_tag"]].append(row)

    profile_rows: list[ProfileRow] = []
    audit_rows: list[dict[str, Any]] = []
    warnings_global: list[str] = []
    reference_mismatch_detected = False

    for base_case_tag, scan_rows in sorted(grouped_scan_rows.items()):
        reference = references.get(base_case_tag)
        if reference is None and base_case_tag in grouped_reference_rows:
            reference = make_reference_from_energy(grouped_reference_rows[base_case_tag], workflow_root)
        if reference is None:
            warnings_global.append(f"{base_case_tag}: missing same-strain reference")
            continue
        if base_case_tag in grouped_reference_rows:
            reference = supplement_reference(reference, grouped_reference_rows[base_case_tag], workflow_root)

        per_case_rows: list[ProfileRow] = []
        xB_reference_used = "xB_out" if reference.xB_out is not None else "xB0"

        for guide_row in sorted(scan_rows, key=lambda row: safe_float(row.get("radius_nm")) or math.inf):
            case_dir = repo_root / guide_row["case_dir_rel"]
            preferred_energy = repo_root / guide_row["energy_csv_rel"]
            energy_csv = find_first_existing(case_dir, preferred_energy, "energy_minimize_*.csv")
            if energy_csv is None:
                continue
            rows = read_csv_rows(energy_csv)
            if not rows:
                continue
            last = rows[-1]
            summary_path = case_dir / "summary.txt"
            summary_data = parse_summary_file(summary_path) if summary_path.exists() else {}
            dx_nm = safe_float(summary_data.get("spacing_x")) or reference.dx_nm
            dy_nm = safe_float(summary_data.get("spacing_y")) or reference.dy_nm
            dz_nm = safe_float(summary_data.get("spacing_z")) or reference.dz_nm
            voxel_count = safe_int(summary_data.get("voxel_count"))
            r_eff_nm = equivalent_radius_nm(voxel_count, dx_nm, dy_nm, dz_nm)
            if r_eff_nm is None:
                r_eff_nm = equivalent_radius_nm(safe_int(last.get("voxel_count")), reference.dx_nm, reference.dy_nm, reference.dz_nm)

            f_surf = safe_float(last.get("F_surf_hat"))
            f_chem = safe_float(last.get("F_chem_CNT_hat"))
            f_el = safe_float(last.get("F_el_hat"))
            f_total = safe_float(last.get("F_total_CNT_hat"))

            delta_surf = (f_surf - reference.F_surf_ref_hat) if f_surf is not None and reference.F_surf_ref_hat is not None else None
            delta_chem = (f_chem - reference.F_chem_ref_hat) if f_chem is not None and reference.F_chem_ref_hat is not None else None
            delta_el = (f_el - reference.F_el_ref_hat) if f_el is not None and reference.F_el_ref_hat is not None else None
            delta_total = (f_total - reference.F_total_ref_hat) if f_total is not None and reference.F_total_ref_hat is not None else None

            row_warnings: list[str] = []
            total_from_parts = None
            if None not in (delta_surf, delta_chem, delta_el):
                total_from_parts = delta_surf + delta_chem + delta_el
            if delta_total is not None and total_from_parts is not None and abs(delta_total - total_from_parts) > 1.0e-9:
                row_warnings.append("DeltaF_total_hat does not equal sum of surf+chem+el within tolerance")

            scan_xb_out = safe_float(guide_row.get("xB_out"))
            if reference.xB_out is not None and scan_xb_out is not None and abs(reference.xB_out - scan_xb_out) > 1.0e-12:
                row_warnings.append("reference far-field composition does not match scan xB_out")
                reference_mismatch_detected = True

            per_case_rows.append(
                ProfileRow(
                    base_case_tag=base_case_tag,
                    case_dir=case_dir,
                    radius_nominal_nm=safe_float(guide_row.get("radius_nm")) or nominal_radius_from_case_name(case_dir.name),
                    r_eff_nm=r_eff_nm,
                    voxel_count=voxel_count,
                    mean_h=safe_float(last.get("mean_h")),
                    F_surf_hat=f_surf,
                    F_chem_hat=f_chem,
                    F_el_hat=f_el,
                    F_total_hat=f_total,
                    DeltaF_surf_hat=delta_surf,
                    DeltaF_chem_hat=delta_chem,
                    DeltaF_el_hat=delta_el,
                    DeltaF_total_hat=delta_total,
                    DeltaG_surf_kBT=component_kbt(delta_surf, reference),
                    DeltaG_chem_kBT=component_kbt(delta_chem, reference),
                    DeltaG_el_kBT=component_kbt(delta_el, reference),
                    DeltaG_total_kBT=component_kbt(delta_total, reference),
                    is_saddle_candidate=0,
                    warnings=row_warnings,
                )
            )

        saddle = pick_saddle(per_case_rows)
        if saddle is None:
            warnings_global.append(f"{base_case_tag}: no usable profile rows")
            continue
        saddle.is_saddle_candidate = 1
        profile_rows.extend(per_case_rows)

        saddle_warnings = list(saddle.warnings) + list(reference.warnings)
        if saddle.DeltaG_chem_kBT is None or saddle.DeltaG_chem_kBT >= -1.0e-9:
            saddle_warnings.append("chemical_driving_force_not_negative_or_too_weak")
        if saddle.DeltaG_surf_kBT is not None and saddle.DeltaG_surf_kBT < -1.0e-9:
            saddle_warnings.append("surface_contribution_not_positive")

        fraction_surf = None
        fraction_chem = None
        fraction_el = None
        if saddle.DeltaG_total_kBT not in (None, 0.0):
            if saddle.DeltaG_surf_kBT is not None:
                fraction_surf = saddle.DeltaG_surf_kBT / saddle.DeltaG_total_kBT
            if saddle.DeltaG_chem_kBT is not None:
                fraction_chem = saddle.DeltaG_chem_kBT / saddle.DeltaG_total_kBT
            if saddle.DeltaG_el_kBT is not None:
                fraction_el = saddle.DeltaG_el_kBT / saddle.DeltaG_total_kBT

        master = summary_rows.get(base_case_tag, {})
        master_kbt = None
        if saddle.DeltaF_total_hat is not None and reference.T_K is not None and reference.T_K > 0.0 and reference.V_box_m3 is not None and reference.w_phys_J_m3 is not None:
            delta_g_total_j = saddle.DeltaF_total_hat * reference.w_phys_J_m3 * reference.V_box_m3
        else:
            delta_g_total_j = None
        if reference.T_K is not None and reference.T_K > 0.0:
            master_excess_hat = safe_float(master.get("F_CNT_peak_hat_excess")) or safe_float(master.get("F_CNT_peak_hat_used"))
            if master_excess_hat is not None and reference.V_box_m3 is not None and reference.w_phys_J_m3 is not None:
                master_kbt = master_excess_hat * reference.w_phys_J_m3 * reference.V_box_m3 / (K_B_J_PER_K * reference.T_K)

        audit_rows.append(
            {
                "base_case_tag": base_case_tag,
                "case_dir_saddle": str(saddle.case_dir),
                "T_input": reference.T_input,
                "T_K": reference.T_K,
                "xB0": reference.xB0,
                "xB_out_if_available": reference.xB_out,
                "xB_reference_used": xB_reference_used,
                "mode": reference.mode,
                "exx": reference.exx,
                "eyy": reference.eyy,
                "ezz": reference.ezz,
                "exy": reference.exy,
                "exz": reference.exz,
                "eyz": reference.eyz,
                "Nx": reference.Nx,
                "Ny": reference.Ny,
                "Nz": reference.Nz,
                "dx_nm": reference.dx_nm,
                "V_box_m3": reference.V_box_m3,
                "w_phys_J_m3": reference.w_phys_J_m3,
                "r_eff_star_nm": saddle.r_eff_nm,
                "voxel_count_saddle": saddle.voxel_count,
                "mean_h_saddle": saddle.mean_h,
                "F_surf_saddle_hat": saddle.F_surf_hat,
                "F_chem_saddle_hat": saddle.F_chem_hat,
                "F_el_saddle_hat": saddle.F_el_hat,
                "F_total_saddle_hat": saddle.F_total_hat,
                "F_surf_ref_hat": reference.F_surf_ref_hat,
                "F_chem_ref_hat": reference.F_chem_ref_hat,
                "F_el_ref_hat": reference.F_el_ref_hat,
                "F_total_ref_hat": reference.F_total_ref_hat,
                "DeltaF_surf_hat": saddle.DeltaF_surf_hat,
                "DeltaF_chem_hat": saddle.DeltaF_chem_hat,
                "DeltaF_el_hat": saddle.DeltaF_el_hat,
                "DeltaF_total_hat": saddle.DeltaF_total_hat,
                "DeltaG_surf_J": (saddle.DeltaF_surf_hat * reference.w_phys_J_m3 * reference.V_box_m3) if None not in (saddle.DeltaF_surf_hat, reference.w_phys_J_m3, reference.V_box_m3) else None,
                "DeltaG_chem_J": (saddle.DeltaF_chem_hat * reference.w_phys_J_m3 * reference.V_box_m3) if None not in (saddle.DeltaF_chem_hat, reference.w_phys_J_m3, reference.V_box_m3) else None,
                "DeltaG_el_J": (saddle.DeltaF_el_hat * reference.w_phys_J_m3 * reference.V_box_m3) if None not in (saddle.DeltaF_el_hat, reference.w_phys_J_m3, reference.V_box_m3) else None,
                "DeltaG_total_J": delta_g_total_j,
                "DeltaG_surf_kBT": saddle.DeltaG_surf_kBT,
                "DeltaG_chem_kBT": saddle.DeltaG_chem_kBT,
                "DeltaG_el_kBT": saddle.DeltaG_el_kBT,
                "DeltaG_total_kBT": saddle.DeltaG_total_kBT,
                "fraction_surf": fraction_surf,
                "fraction_chem": fraction_chem,
                "fraction_el": fraction_el,
                "barrier_kBT_from_master_table": master_kbt,
                "difference_vs_master_kBT": (saddle.DeltaG_total_kBT - master_kbt) if saddle.DeltaG_total_kBT is not None and master_kbt is not None else None,
                "warnings": " | ".join(sorted(dict.fromkeys(saddle_warnings))),
            }
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    profile_csv = out_dir / "energy_profile_excess_by_case.csv"
    audit_csv = out_dir / "energy_component_barrier_audit.csv"
    metadata_json = out_dir / "energy_component_barrier_metadata.json"

    write_csv(profile_csv, [row.to_row() for row in profile_rows])
    write_csv(audit_csv, audit_rows)
    plot_paths = make_plots(audit_rows, profile_rows, out_dir)

    total_vals = [safe_float(row.get("DeltaG_total_kBT")) for row in audit_rows if safe_float(row.get("DeltaG_total_kBT")) is not None]
    surf_vals = [safe_float(row.get("DeltaG_surf_kBT")) for row in audit_rows if safe_float(row.get("DeltaG_surf_kBT")) is not None]
    chem_vals = [safe_float(row.get("DeltaG_chem_kBT")) for row in audit_rows if safe_float(row.get("DeltaG_chem_kBT")) is not None]
    el_vals = [safe_float(row.get("DeltaG_el_kBT")) for row in audit_rows if safe_float(row.get("DeltaG_el_kBT")) is not None]

    negative_chem = sum(1 for value in chem_vals if value is not None and value < -1.0e-9)
    nonnegative_chem = sum(1 for value in chem_vals if value is not None and value >= -1.0e-9)
    median_surf = float(sorted(surf_vals)[len(surf_vals) // 2]) if surf_vals else float("nan")
    median_chem = float(sorted(chem_vals)[len(chem_vals) // 2]) if chem_vals else float("nan")
    median_el = float(sorted(el_vals)[len(el_vals) // 2]) if el_vals else float("nan")
    positive_components = {
        "surface": median_surf if math.isfinite(median_surf) else float("-inf"),
        "elastic": median_el if math.isfinite(median_el) else float("-inf"),
    }
    dominant_positive = max(positive_components, key=positive_components.get) if positive_components else "unknown"
    if nonnegative_chem > 0:
        dominant_reason = "weak chemical driving force"
    elif math.isfinite(median_surf) and math.isfinite(median_el):
        dominant_reason = "surface" if median_surf >= median_el else "elastic"
    else:
        dominant_reason = dominant_positive

    metadata = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "workflow_root": str(workflow_root),
        "guide_csv": str(guide_csv),
        "summary_csv": str(summary_csv),
        "reference_csv": str(reference_csv),
        "plot_paths": plot_paths,
        "warnings": warnings_global,
        "reference_mismatch_detected": reference_mismatch_detected,
    }
    metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"cases_scanned={len(audit_rows)}")
    print(f"median_DeltaG_total_kBT={float(sorted(total_vals)[len(total_vals)//2]) if total_vals else 'nan'}")
    print(f"median_DeltaG_surf_kBT={median_surf if math.isfinite(median_surf) else 'nan'}")
    print(f"median_DeltaG_chem_kBT={median_chem if math.isfinite(median_chem) else 'nan'}")
    print(f"median_DeltaG_el_kBT={median_el if math.isfinite(median_el) else 'nan'}")
    print(f"negative_chemical_cases={negative_chem}")
    print(f"nonnegative_chemical_cases={nonnegative_chem}")
    print(f"dominant_positive_component={dominant_positive}")
    print(f"dominant_barrier_reason={dominant_reason}")
    print(f"xB_reference_mismatch_detected={int(reference_mismatch_detected)}")
    print(f"output_dir={out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
