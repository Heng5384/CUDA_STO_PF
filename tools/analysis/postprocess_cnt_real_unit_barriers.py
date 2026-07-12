#!/usr/bin/env python3
"""Convert CNT/minimize hat energies in a workflow to physical barrier units.

This is a read-only postprocess for guide-based CNT scans.  It preserves the
three energy definitions emitted by the workflow:

* ``F_total_excess_hat``: PF/full-model excess free energy already referenced
  to the matrix/bulk baseline inside ``main_cuda``.
* ``F_total_CNT_hat``: CNT diagnostic using the far-field chemical-potential
  driving force.  This is not reference-subtracted by itself.
* ``DeltaF_excess_refsub_hat``: ``F_total_excess_hat - F_total_ref_hat``.  This
  is the strict no-nucleus-reference-subtracted PF excess barrier.
* ``DeltaF_CNT_refsub_hat``: ``F_total_CNT_hat - F_total_ref_hat`` using the
  same-strain matrix-only reference row from the guide.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

K_B_J_PER_K = 1.380649e-23
EV_J = 1.602176634e-19


def safe_float(value: Any) -> float | None:
    if value is None:
        return None
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    try:
        out = float(text)
    except ValueError:
        return None
    return out if math.isfinite(out) else None


def safe_int(value: Any) -> int | None:
    f = safe_float(value)
    return int(f) if f is not None else None


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def last_csv_row(path: Path) -> dict[str, str] | None:
    rows = read_csv(path)
    return rows[-1] if rows else None


def resolve(repo_root: Path, maybe_rel: str | None) -> Path | None:
    if not maybe_rel:
        return None
    path = Path(maybe_rel)
    if path.is_absolute():
        return path
    return repo_root / path


def first_existing(case_dir: Path, preferred: Path | None, pattern: str) -> Path | None:
    if preferred and preferred.exists():
        return preferred
    matches = sorted(case_dir.glob(pattern))
    return matches[0] if matches else None


def load_physical_inputs(repo_root: Path, row: dict[str, str]) -> dict[str, Any]:
    path = resolve(repo_root, row.get("physical_input_json_rel"))
    if path and path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    return {}


def energy_scale(repo_root: Path, row: dict[str, str]) -> tuple[float | None, float | None, str]:
    payload = load_physical_inputs(repo_root, row)
    gamma = safe_float(payload.get("gamma"))
    lambda_sm = safe_float(payload.get("lambda_sm"))
    if lambda_sm is None:
        lambda_nm = safe_float(row.get("lambda_sm_nm"))
        lambda_sm = lambda_nm * 1.0e-9 if lambda_nm is not None else None
    if gamma is None or lambda_sm is None or lambda_sm <= 0.0:
        return None, None, "missing gamma/lambda_sm"
    return 12.0 * gamma / lambda_sm, gamma, "physical_inputs.json"


def box_volume_m3(row: dict[str, str]) -> tuple[float | None, str]:
    nx = safe_int(row.get("nx"))
    ny = safe_int(row.get("ny"))
    nz = safe_int(row.get("nz"))
    dx_nm = safe_float(row.get("pf_dx_nm"))
    if None in (nx, ny, nz, dx_nm):
        return None, "missing grid or pf_dx_nm"
    dx_m = dx_nm * 1.0e-9
    return float(nx * ny * nz) * dx_m**3, "guide_cnt_scan.csv"


def convert_hat(delta_hat: float | None, factor_j_per_hat: float | None, temperature_k: float | None) -> dict[str, Any]:
    if delta_hat is None or factor_j_per_hat is None:
        return {"J": "", "eV": "", "kBT": ""}
    value_j = delta_hat * factor_j_per_hat
    out: dict[str, Any] = {
        "J": value_j,
        "eV": value_j / EV_J,
        "kBT": "",
    }
    if temperature_k is not None and temperature_k > 0.0:
        out["kBT"] = value_j / (K_B_J_PER_K * temperature_k)
    return out


def radius_sort_key(row: dict[str, Any]) -> tuple[float, str]:
    radius = safe_float(row.get("radius_nm"))
    return (radius if radius is not None else float("inf"), str(row.get("case_tag", "")))


def parse_temperature_label(workflow_name: str) -> tuple[float | None, str]:
    match = re.search(r"_T([0-9]+(?:p[0-9]+)?)([CK])_", workflow_name)
    if not match:
        return None, ""
    value = float(match.group(1).replace("p", "."))
    unit = match.group(2)
    return value, unit


def build_reference_by_base(repo_root: Path, rows: list[dict[str, str]]) -> dict[str, dict[str, Any]]:
    refs: dict[str, dict[str, Any]] = {}
    for row in rows:
        if row.get("row_type") != "reference":
            continue
        case_dir = resolve(repo_root, row.get("case_dir_rel"))
        if case_dir is None:
            continue
        ref_path = resolve(repo_root, row.get("reference_energy_rel"))
        record: dict[str, Any] | None = None
        if ref_path and ref_path.exists():
            ref_rows = read_csv(ref_path)
            record = dict(ref_rows[0]) if ref_rows else None
            source = str(ref_path)
        else:
            energy_csv = first_existing(case_dir, resolve(repo_root, row.get("energy_csv_rel")), "energy_minimize_*.csv")
            last = last_csv_row(energy_csv) if energy_csv else None
            record = dict(last) if last else None
            source = str(energy_csv) if energy_csv else ""
        if record is None:
            continue
        refs[row["base_case_tag"]] = {
            "F_total_ref_hat": safe_float(record.get("F_total_ref_hat")) or safe_float(record.get("F_total_excess_hat")),
            "F_surf_ref_hat": safe_float(record.get("F_surf_ref_hat")) or safe_float(record.get("F_surf_hat")),
            "F_el_ref_hat": safe_float(record.get("F_el_ref_hat")) or safe_float(record.get("F_el_hat")),
            "F_chem_ref_hat": safe_float(record.get("F_chem_ref_hat")) or safe_float(record.get("F_chem_excess_hat")),
            "reference_source": source,
        }
    return refs


DETAIL_FIELDS = [
    "workflow_name",
    "base_case_tag",
    "case_tag",
    "T_C",
    "T_K",
    "xB",
    "strain",
    "radius_nm",
    "rc_schur_nm",
    "Nx",
    "Ny",
    "Nz",
    "dx_nm",
    "gamma_J_m2",
    "lambda_sm_m",
    "w_phys_J_m3",
    "V_box_m3",
    "J_per_hat",
    "F_total_excess_hat",
    "DeltaG_excess_J",
    "DeltaG_excess_eV",
    "DeltaG_excess_kBT",
    "DeltaF_excess_refsub_hat",
    "DeltaG_excess_refsub_J",
    "DeltaG_excess_refsub_eV",
    "DeltaG_excess_refsub_kBT",
    "F_total_CNT_hat",
    "DeltaG_CNT_absolute_J",
    "DeltaG_CNT_absolute_eV",
    "DeltaG_CNT_absolute_kBT",
    "F_total_ref_hat",
    "DeltaF_CNT_refsub_hat",
    "DeltaG_CNT_refsub_J",
    "DeltaG_CNT_refsub_eV",
    "DeltaG_CNT_refsub_kBT",
    "reference_source",
    "energy_csv",
    "scale_source",
    "volume_source",
    "definition_note",
]

SUMMARY_FIELDS = [
    "workflow_name",
    "base_case_tag",
    "T_C",
    "T_K",
    "xB",
    "strain",
    "n_points",
    "rc_schur_nm",
    "excess_peak_radius_nm",
    "excess_peak_kBT",
    "excess_peak_J",
    "excess_refsub_peak_radius_nm",
    "excess_refsub_peak_kBT",
    "excess_refsub_peak_J",
    "cnt_absolute_peak_radius_nm",
    "cnt_absolute_peak_kBT",
    "cnt_absolute_peak_J",
    "cnt_refsub_peak_radius_nm",
    "cnt_refsub_peak_kBT",
    "cnt_refsub_peak_J",
    "J_per_hat",
    "w_phys_J_m3",
    "V_box_m3",
    "reference_source",
]


def summarize_detail_rows(detail_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summary_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = defaultdict(list)
    for row in detail_rows:
        grouped[(str(row["workflow_name"]), str(row["base_case_tag"]))].append(row)
    for (_workflow, _base), group in sorted(grouped.items()):
        group = sorted(group, key=radius_sort_key)

        def peak(key: str) -> dict[str, Any] | None:
            valid = [r for r in group if safe_float(r.get(key)) is not None]
            return max(valid, key=lambda r: float(r[key])) if valid else None

        p_excess = peak("DeltaG_excess_kBT")
        p_excess_refsub = peak("DeltaG_excess_refsub_kBT")
        p_cnt_abs = peak("DeltaG_CNT_absolute_kBT")
        p_cnt_ref = peak("DeltaG_CNT_refsub_kBT")
        first = group[0]
        summary_rows.append(
            {
                "workflow_name": first["workflow_name"],
                "base_case_tag": first["base_case_tag"],
                "T_C": first["T_C"],
                "T_K": first["T_K"],
                "xB": first["xB"],
                "strain": first["strain"],
                "n_points": len(group),
                "rc_schur_nm": first["rc_schur_nm"],
                "excess_peak_radius_nm": p_excess.get("radius_nm", "") if p_excess else "",
                "excess_peak_kBT": p_excess.get("DeltaG_excess_kBT", "") if p_excess else "",
                "excess_peak_J": p_excess.get("DeltaG_excess_J", "") if p_excess else "",
                "excess_refsub_peak_radius_nm": p_excess_refsub.get("radius_nm", "") if p_excess_refsub else "",
                "excess_refsub_peak_kBT": p_excess_refsub.get("DeltaG_excess_refsub_kBT", "") if p_excess_refsub else "",
                "excess_refsub_peak_J": p_excess_refsub.get("DeltaG_excess_refsub_J", "") if p_excess_refsub else "",
                "cnt_absolute_peak_radius_nm": p_cnt_abs.get("radius_nm", "") if p_cnt_abs else "",
                "cnt_absolute_peak_kBT": p_cnt_abs.get("DeltaG_CNT_absolute_kBT", "") if p_cnt_abs else "",
                "cnt_absolute_peak_J": p_cnt_abs.get("DeltaG_CNT_absolute_J", "") if p_cnt_abs else "",
                "cnt_refsub_peak_radius_nm": p_cnt_ref.get("radius_nm", "") if p_cnt_ref else "",
                "cnt_refsub_peak_kBT": p_cnt_ref.get("DeltaG_CNT_refsub_kBT", "") if p_cnt_ref else "",
                "cnt_refsub_peak_J": p_cnt_ref.get("DeltaG_CNT_refsub_J", "") if p_cnt_ref else "",
                "J_per_hat": first["J_per_hat"],
                "w_phys_J_m3": first["w_phys_J_m3"],
                "V_box_m3": first["V_box_m3"],
                "reference_source": first["reference_source"],
            }
        )
    return summary_rows


def write_report(output_dir: Path, n_detail: int, n_summary: int) -> None:
    report = output_dir / "barrier_real_units_report.md"
    lines = [
        "# CNT real-unit barrier postprocess",
        "",
        "Definitions:",
        "- `F_total_excess_hat`: PF/full-model excess energy already referenced to the matrix/bulk baseline inside `main_cuda`.",
        "- `F_total_CNT_hat`: CNT diagnostic using far-field chemical-potential driving force; not reference-subtracted by itself.",
        "- `DeltaF_excess_refsub_hat`: `F_total_excess_hat - F_total_ref_hat`; strict no-nucleus-reference-subtracted PF excess barrier.",
        "- `DeltaF_CNT_refsub_hat`: `F_total_CNT_hat - F_total_ref_hat` using the matrix-only same-strain reference.",
        "",
        "Conversion:",
        "- `DeltaG_J = DeltaF_hat * w_phys * V_box`",
        "- `w_phys = 12 * gamma / lambda_sm`",
        "- `V_box = Nx * Ny * Nz * dx^3`",
        "- `DeltaG_kBT = DeltaG_J / (k_B * T_K)`",
        "",
        f"Processed scan points: {n_detail}",
        f"Processed cases: {n_summary}",
        "",
        "Outputs:",
        "- `barrier_real_units_detail.csv`",
        "- `barrier_real_units_by_case.csv`",
    ]
    report.write_text("\n".join(lines) + "\n", encoding="utf-8")


def postprocess_workflow(workflow_root: Path, repo_root: Path, output_dir: Path, *, write_outputs: bool = True) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    guide = workflow_root / "input" / "guide_cnt_scan.csv"
    if not guide.exists():
        raise FileNotFoundError(f"missing guide: {guide}")
    rows = read_csv(guide)
    refs = build_reference_by_base(repo_root, rows)
    detail_rows: list[dict[str, Any]] = []

    for row in rows:
        if row.get("row_type") != "scan_point":
            continue
        case_dir = resolve(repo_root, row.get("case_dir_rel"))
        if case_dir is None:
            continue
        energy_csv = first_existing(case_dir, resolve(repo_root, row.get("energy_csv_rel")), "energy_minimize_*.csv")
        last = last_csv_row(energy_csv) if energy_csv else None
        if not last:
            continue

        w_phys, gamma, scale_source = energy_scale(repo_root, row)
        v_box, volume_source = box_volume_m3(row)
        factor = w_phys * v_box if w_phys is not None and v_box is not None else None
        t_c = safe_float(row.get("T_C"))
        t_k = t_c + 273.15 if t_c is not None else None
        ref = refs.get(row["base_case_tag"], {})
        f_total_cnt = safe_float(last.get("F_total_CNT_hat"))
        f_total_excess = safe_float(last.get("F_total_excess_hat"))
        f_total_ref = safe_float(ref.get("F_total_ref_hat"))
        delta_excess_refsub = f_total_excess - f_total_ref if f_total_excess is not None and f_total_ref is not None else None
        delta_cnt_refsub = f_total_cnt - f_total_ref if f_total_cnt is not None and f_total_ref is not None else None
        cnt_abs = convert_hat(f_total_cnt, factor, t_k)
        cnt_refsub = convert_hat(delta_cnt_refsub, factor, t_k)
        excess = convert_hat(f_total_excess, factor, t_k)
        excess_refsub = convert_hat(delta_excess_refsub, factor, t_k)

        detail_rows.append(
            {
                "workflow_name": row.get("workflow_name", ""),
                "base_case_tag": row.get("base_case_tag", ""),
                "case_tag": row.get("case_tag", ""),
                "T_C": t_c,
                "T_K": t_k,
                "xB": safe_float(row.get("xB_out")),
                "strain": safe_float(row.get("strain")) or 0.0,
                "radius_nm": safe_float(row.get("radius_nm")),
                "rc_schur_nm": safe_float(row.get("rc_schur_nm")),
                "Nx": safe_int(row.get("nx")),
                "Ny": safe_int(row.get("ny")),
                "Nz": safe_int(row.get("nz")),
                "dx_nm": safe_float(row.get("pf_dx_nm")),
                "gamma_J_m2": gamma,
                "lambda_sm_m": safe_float(load_physical_inputs(repo_root, row).get("lambda_sm")),
                "w_phys_J_m3": w_phys,
                "V_box_m3": v_box,
                "J_per_hat": factor,
                "F_total_excess_hat": f_total_excess,
                "DeltaG_excess_J": excess["J"],
                "DeltaG_excess_eV": excess["eV"],
                "DeltaG_excess_kBT": excess["kBT"],
                "DeltaF_excess_refsub_hat": delta_excess_refsub,
                "DeltaG_excess_refsub_J": excess_refsub["J"],
                "DeltaG_excess_refsub_eV": excess_refsub["eV"],
                "DeltaG_excess_refsub_kBT": excess_refsub["kBT"],
                "F_total_CNT_hat": f_total_cnt,
                "DeltaG_CNT_absolute_J": cnt_abs["J"],
                "DeltaG_CNT_absolute_eV": cnt_abs["eV"],
                "DeltaG_CNT_absolute_kBT": cnt_abs["kBT"],
                "F_total_ref_hat": f_total_ref,
                "DeltaF_CNT_refsub_hat": delta_cnt_refsub,
                "DeltaG_CNT_refsub_J": cnt_refsub["J"],
                "DeltaG_CNT_refsub_eV": cnt_refsub["eV"],
                "DeltaG_CNT_refsub_kBT": cnt_refsub["kBT"],
                "reference_source": ref.get("reference_source", ""),
                "energy_csv": str(energy_csv) if energy_csv else "",
                "scale_source": scale_source,
                "volume_source": volume_source,
                "definition_note": "excess=PF full-model excess baseline; excess_refsub=excess minus matrix-only reference; CNT_absolute=far-field chemical-potential CNT diagnostic; CNT_refsub=CNT_absolute minus matrix-only reference",
            }
        )

    detail_rows.sort(key=lambda row: (str(row.get("workflow_name", "")), radius_sort_key(row)))
    summary_rows = summarize_detail_rows(detail_rows)
    if write_outputs:
        write_csv(output_dir / "barrier_real_units_detail.csv", detail_rows, DETAIL_FIELDS)
        write_csv(output_dir / "barrier_real_units_by_case.csv", summary_rows, SUMMARY_FIELDS)
        write_report(output_dir, len(detail_rows), len(summary_rows))
    return detail_rows, summary_rows


def discover_workflows(root: Path) -> list[Path]:
    if (root / "input" / "guide_cnt_scan.csv").exists():
        return [root]
    return sorted(path.parent.parent for path in root.glob("*/input/guide_cnt_scan.csv"))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workflow-root", type=Path, required=True)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--output-dir", type=Path)
    args = parser.parse_args()

    workflow_root = args.workflow_root.resolve()
    repo_root = args.repo_root.resolve()
    output_dir = args.output_dir or (workflow_root / "summary_reports" / "real_unit_barriers")
    workflows = discover_workflows(workflow_root)
    if not workflows:
        raise FileNotFoundError(f"no guide_cnt_scan.csv found under {workflow_root}")
    if len(workflows) == 1 and workflows[0] == workflow_root:
        detail, summary = postprocess_workflow(workflow_root, repo_root, output_dir)
    else:
        detail = []
        for child in workflows:
            child_detail, _child_summary = postprocess_workflow(child, repo_root, output_dir, write_outputs=False)
            detail.extend(child_detail)
        detail.sort(key=lambda row: (str(row.get("workflow_name", "")), radius_sort_key(row)))
        summary = summarize_detail_rows(detail)
        write_csv(output_dir / "barrier_real_units_detail.csv", detail, DETAIL_FIELDS)
        write_csv(output_dir / "barrier_real_units_by_case.csv", summary, SUMMARY_FIELDS)
        write_report(output_dir, len(detail), len(summary))
    print(f"real_unit_barrier_postprocess_complete")
    print(f"workflow_roots={len(workflows)}")
    print(f"detail_rows={len(detail)}")
    print(f"case_rows={len(summary)}")
    print(f"output_dir={output_dir}")


if __name__ == "__main__":
    main()
