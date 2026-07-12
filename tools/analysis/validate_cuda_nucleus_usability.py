#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
from scipy import ndimage


def _safe_float(value: Any, default: float | None = None) -> float | None:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _union_fields(rows: list[dict[str, Any]]) -> list[str]:
    seen: set[str] = set()
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                fields.append(key)
    return fields


def _parse_pf_params(path: Path) -> dict[str, float]:
    out: dict[str, float] = {}
    if not path.exists():
        return out
    for raw in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        try:
            out[key.strip()] = float(value.strip())
        except ValueError:
            continue
    return out


def _read_legacy_scalar_vtk(path: Path) -> tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]:
    dims: tuple[int, int, int] | None = None
    spacing = (1.0, 1.0, 1.0)
    with path.open("rb") as handle:
        while True:
            raw = handle.readline()
            if not raw:
                raise ValueError(f"unexpected EOF before LOOKUP_TABLE in {path}")
            line = raw.decode("ascii", errors="replace").strip()
            if line.startswith("DIMENSIONS"):
                _, sx, sy, sz = line.split()
                dims = (int(sx), int(sy), int(sz))
            elif line.startswith("SPACING") or line.startswith("ASPECT_RATIO"):
                parts = line.split()
                spacing = (float(parts[1]), float(parts[2]), float(parts[3]))
            elif line.startswith("LOOKUP_TABLE"):
                break
        if dims is None:
            raise ValueError(f"missing DIMENSIONS in {path}")
        values = np.fromstring(handle.read().decode("ascii", errors="replace"), sep=" ", dtype=np.float32)
    expected = dims[0] * dims[1] * dims[2]
    if values.size != expected:
        raise ValueError(f"VTK scalar count mismatch in {path}: got {values.size}, expected {expected}")
    return values.reshape(dims, order="C"), dims, spacing


def _last_energy_row(path: Path | None) -> dict[str, str]:
    if path is None or not path.exists():
        return {}
    rows = _read_csv_rows(path)
    return rows[-1] if rows else {}


def _radius_from_path(path: Path) -> float | None:
    text = str(path)
    m = re.search(r"_r([0-9]+(?:\.[0-9]+)?)nm_", text)
    if m:
        return float(m.group(1))
    m = re.search(r"_r([0-9]+)p([0-9]+)", text)
    if m:
        return float(f"{m.group(1)}.{m.group(2)}")
    m = re.search(r"_r([0-9]+(?:\.[0-9]+))_", text)
    if m:
        return float(m.group(1))
    return None


def _condition_from_path(path: Path) -> dict[str, Any]:
    text = str(path)
    out: dict[str, Any] = {"T_K": "", "T_C": "", "xB": "", "strain": ""}
    m = re.search(r"T_?([0-9]+(?:p[0-9]+)?)K", text)
    if m:
        out["T_K"] = float(m.group(1).replace("p", "."))
        out["T_C"] = out["T_K"] - 273.15
    else:
        m = re.search(r"_T([0-9]+)_", text)
        if m:
            out["T_C"] = float(m.group(1))
            out["T_K"] = out["T_C"] + 273.15
    m = re.search(r"xB_?([0-9]+p[0-9]+|[0-9]+\.[0-9]+)", text)
    if m:
        out["xB"] = float(m.group(1).replace("p", "."))
    m = re.search(r"strain_?([m0-9p\.-]+)", text)
    if m:
        raw = m.group(1).replace("m", "-").replace("p", ".")
        out["strain"] = _safe_float(raw, "")
    return out


def _find_partner(case_dir: Path, prefix: str) -> Path | None:
    files = sorted(case_dir.glob(prefix))
    return files[0] if files else None


def _case_records(root: Path) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for phi in sorted(root.rglob("phi_final_*.vtk")):
        if "_ref_matrix_only" in phi.name or "r0.000nm" in str(phi):
            continue
        case_dir = phi.parent
        energy = _find_partner(case_dir, "energy_minimize_*.csv")
        xB = _find_partner(case_dir, "xB_final_*.vtk")
        summary = _find_partner(case_dir, "summary*.txt")
        pf_input = case_dir / "pf_input.params"
        if not pf_input.exists():
            pf_candidates = sorted(case_dir.glob("*.params"))
            pf_input = pf_candidates[0] if pf_candidates else pf_input
        record = {
            "case_dir": str(case_dir),
            "phi_final": str(phi),
            "xB_final": str(xB) if xB else "",
            "energy_csv": str(energy) if energy else "",
            "summary": str(summary) if summary else "",
            "pf_input": str(pf_input) if pf_input.exists() else "",
            "radius_nm": _radius_from_path(phi),
        }
        record.update(_condition_from_path(phi))
        out.append(record)
    return out


def _analyze_phi(phi: np.ndarray, dx_nm: float, threshold: float) -> dict[str, Any]:
    phi_min = float(np.nanmin(phi))
    phi_max = float(np.nanmax(phi))
    phi_mean = float(np.nanmean(phi))
    mask = np.isfinite(phi) & (phi > threshold)
    voxel_count = int(mask.sum())
    if voxel_count == 0:
        return {
            "phi_min": phi_min,
            "phi_max": phi_max,
            "phi_mean": phi_mean,
            "voxel_count": 0,
            "component_count": 0,
            "largest_component_fraction": 0.0,
            "compactness_score": 0.0,
            "interface_voxel_count": int(((phi > 0.1) & (phi < 0.9)).sum()),
            "interface_smoothness": "missing",
            "fragmented": True,
            "effective_radius_nm": 0.0,
            "aspect_ratio": "",
            "bbox_cells_x": 0,
            "bbox_cells_y": 0,
            "bbox_cells_z": 0,
            "shape_type_observed": "none",
        }

    structure = ndimage.generate_binary_structure(3, 2)
    labeled, ncomp = ndimage.label(mask, structure=structure)
    counts = np.bincount(labeled.ravel())
    counts[0] = 0
    largest_label = int(counts.argmax())
    largest_count = int(counts[largest_label])
    largest = labeled == largest_label
    largest_fraction = largest_count / max(voxel_count, 1)
    coords = np.argwhere(largest)
    mins = coords.min(axis=0)
    maxs = coords.max(axis=0)
    bbox_cells = (maxs - mins + 1).astype(int)
    volume_nm3 = largest_count * dx_nm**3
    r_eff = (3.0 * volume_nm3 / (4.0 * math.pi)) ** (1.0 / 3.0)
    bbox_volume = int(bbox_cells[0] * bbox_cells[1] * bbox_cells[2])
    compactness = largest_count / max(bbox_volume, 1)

    rel = (coords - coords.mean(axis=0)) * dx_nm
    aspect = ""
    shape = "unknown"
    if coords.shape[0] >= 4:
        cov = np.cov(rel, rowvar=False, bias=True)
        eig = np.linalg.eigvalsh(cov)
        eig = np.sort(np.maximum(eig, 0.0))[::-1]
        if eig[-1] > 1.0e-20:
            aspect_value = math.sqrt(float(eig[0] / eig[-1]))
            aspect = f"{aspect_value:.6g}"
            if aspect_value < 1.10:
                shape = "spherical"
            elif aspect_value < 1.80:
                shape = "anisotropic"
            else:
                shape = "faceted_or_elongated"

    interface = np.isfinite(phi) & (phi > 0.1) & (phi < 0.9)
    interface_count = int(interface.sum())
    interface_ratio = interface_count / max(voxel_count, 1)
    if interface_count == 0:
        smooth = "subgrid_or_binary"
    elif interface_ratio < 0.15:
        smooth = "thin_or_noisy"
    else:
        smooth = "resolved"

    return {
        "phi_min": phi_min,
        "phi_max": phi_max,
        "phi_mean": phi_mean,
        "voxel_count": voxel_count,
        "component_count": int(ncomp),
        "largest_component_fraction": largest_fraction,
        "compactness_score": compactness,
        "interface_voxel_count": interface_count,
        "interface_to_core_ratio": interface_ratio,
        "interface_smoothness": smooth,
        "fragmented": bool(ncomp > 1 and largest_fraction < 0.98),
        "effective_radius_nm": r_eff,
        "aspect_ratio": aspect,
        "bbox_cells_x": int(bbox_cells[0]),
        "bbox_cells_y": int(bbox_cells[1]),
        "bbox_cells_z": int(bbox_cells[2]),
        "shape_type_observed": shape,
    }


def _classify(row: dict[str, Any]) -> tuple[str, bool, str, str]:
    rdx = _safe_float(row.get("radius_over_dx"), 0.0) or 0.0
    eff_rdx = _safe_float(row.get("effective_radius_over_dx"), 0.0) or 0.0
    largest = _safe_float(row.get("largest_component_fraction"), 0.0) or 0.0
    comp = int(_safe_float(row.get("component_count"), 0) or 0)
    iface = str(row.get("interface_smoothness", ""))
    reasons: list[str] = []
    fixes: list[str] = []

    if comp == 0:
        reasons.append("no phi>threshold nucleus")
        fixes.append("re-minimization")
        return "C", False, "re-minimization", "; ".join(reasons)
    if rdx < 2.0 or eff_rdx < 2.0:
        reasons.append("sub-grid or barely resolved radius")
        fixes.append("grid refinement")
    elif rdx < 4.0 or eff_rdx < 4.0:
        reasons.append("under-resolved for direct insertion")
        fixes.append("interpolation / resampling")
    if comp > 1 and largest < 0.98:
        reasons.append("fragmented morphology")
        fixes.append("geometry smoothing")
    if iface in {"subgrid_or_binary", "thin_or_noisy"}:
        reasons.append("interface not smoothly resolved")
        fixes.append("geometry smoothing")

    if not reasons and rdx >= 4.0 and eff_rdx >= 4.0 and largest >= 0.98:
        return "A", True, "none", "directly usable"
    if any("sub-grid" in item or "fragmented" in item for item in reasons):
        return "C", False, " + ".join(dict.fromkeys(fixes)) or "rejection", "; ".join(reasons)
    return "B", False, " + ".join(dict.fromkeys(fixes)) or "interpolation / resampling", "; ".join(reasons)


@dataclass
class TemplateRecord:
    template_id: str
    source_case_dir: str
    phi_template: str
    xB_template: str
    radius_nm: float
    dx_nm: float
    grid: list[int]
    shape_type: str


def _write_template(case: dict[str, Any], phi: np.ndarray, out_dir: Path, root: Path) -> TemplateRecord:
    template_id = re.sub(r"[^A-Za-z0-9_.-]+", "_", Path(case["case_dir"]).name)
    template_dir = out_dir / template_id
    template_dir.mkdir(parents=True, exist_ok=True)
    phi_path = template_dir / f"phi_init_{template_id}.npy"
    np.save(phi_path, phi.astype(np.float32, copy=False))
    xB_template = ""
    xB_path = Path(str(case.get("xB_final", "")))
    if xB_path.exists():
        xB_template = str(xB_path.relative_to(root) if xB_path.is_relative_to(root) else xB_path)
    return TemplateRecord(
        template_id=template_id,
        source_case_dir=str(Path(case["case_dir"]).relative_to(root) if Path(case["case_dir"]).is_relative_to(root) else case["case_dir"]),
        phi_template=str(phi_path.relative_to(root) if phi_path.is_relative_to(root) else phi_path),
        xB_template=xB_template,
        radius_nm=float(case.get("radius_nm") or 0.0),
        dx_nm=float(case.get("dx_nm") or 1.0),
        grid=[int(x) for x in case.get("grid", [])],
        shape_type=str(case.get("shape_type_observed", "unknown")),
    )


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate whether CNT/minimized nuclei can be used as CUDA insertion templates.")
    parser.add_argument("--root", type=Path, default=Path("CNT_SCAN_WORKSTATION_RUN"), help="CNT scan root")
    parser.add_argument("--output-dir", type=Path, default=Path("CNT_SCAN_WORKSTATION_RUN/cuda_nucleus_compatibility"))
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--direct-min-rdx", type=float, default=4.0)
    args = parser.parse_args()

    root = args.root.resolve()
    output_dir = args.output_dir.resolve()
    template_dir = root / "cuda_nucleus_templates"
    template_dir.mkdir(parents=True, exist_ok=True)
    cases = _case_records(root)

    geometry_rows: list[dict[str, Any]] = []
    physics_rows: list[dict[str, Any]] = []
    compat_rows: list[dict[str, Any]] = []
    templates: list[TemplateRecord] = []

    for case in cases:
        phi_path = Path(case["phi_final"])
        energy_path = Path(case["energy_csv"]) if case.get("energy_csv") else None
        pf_params = _parse_pf_params(Path(case["pf_input"])) if case.get("pf_input") else {}
        try:
            phi, dims, spacing = _read_legacy_scalar_vtk(phi_path)
            dx_nm = float(pf_params.get("dx", pf_params.get("dx_nm", spacing[0])))
            if dx_nm <= 0:
                dx_nm = float(spacing[0]) if spacing[0] > 0 else 1.0
            geom = _analyze_phi(phi, dx_nm, args.threshold)
            status = "analyzed"
            error = ""
        except Exception as exc:
            phi = np.zeros((0, 0, 0), dtype=np.float32)
            dims = (0, 0, 0)
            dx_nm = float(pf_params.get("dx", 1.0))
            geom = _analyze_phi(phi, dx_nm, args.threshold) if phi.size else {
                "phi_min": "", "phi_max": "", "phi_mean": "", "voxel_count": 0,
                "component_count": 0, "largest_component_fraction": 0.0,
                "compactness_score": 0.0, "interface_voxel_count": 0,
                "interface_smoothness": "read_failed", "fragmented": True,
                "effective_radius_nm": 0.0, "aspect_ratio": "",
                "bbox_cells_x": 0, "bbox_cells_y": 0, "bbox_cells_z": 0,
                "shape_type_observed": "unknown",
            }
            status = "read_failed"
            error = str(exc)

        radius_nm = _safe_float(case.get("radius_nm"), None)
        rdx = (radius_nm / dx_nm) if radius_nm is not None and dx_nm > 0 else ""
        eff_rdx = geom["effective_radius_nm"] / dx_nm if dx_nm > 0 else ""
        energy_last = _last_energy_row(energy_path)
        f_total = _safe_float(energy_last.get("F_total_CNT_hat"), _safe_float(energy_last.get("F_total_excess_hat")))
        rms_dphi = _safe_float(energy_last.get("rms_dphi"), _safe_float(energy_last.get("rms_res_phi")))
        rms_dy = _safe_float(energy_last.get("rms_dY"), _safe_float(energy_last.get("rms_res_Y")))

        row = {
            **case,
            "grid": list(dims),
            "dx_nm": dx_nm,
            "radius_over_dx": rdx,
            "effective_radius_over_dx": eff_rdx,
            "analysis_status": status,
            "analysis_error": error,
            **geom,
        }
        compatibility_class, usable, required_fix, reason = _classify(row)
        row["cuda_compatibility_class"] = compatibility_class
        row["usable_for_insertion"] = str(bool(usable)).lower()
        row["required_fix"] = required_fix
        row["classification_reason"] = reason
        geometry_rows.append(row)

        physics_rows.append({
            "case_dir": case["case_dir"],
            "T_K": case.get("T_K", ""),
            "xB": case.get("xB", ""),
            "strain": case.get("strain", ""),
            "radius_nm": radius_nm if radius_nm is not None else "",
            "F_total_CNT_hat_or_excess": f_total if f_total is not None else "",
            "rms_dphi": rms_dphi if rms_dphi is not None else "",
            "rms_dY": rms_dy if rms_dy is not None else "",
            "energetically_localized": str(geom["voxel_count"] > 0 and geom["largest_component_fraction"] >= 0.98).lower(),
            "shape_stable_proxy": "true" if compatibility_class in {"A", "B"} else "false",
            "rstar_minimum_check": "requires full scan summary" if "steps1" in str(case["case_dir"]) else "available_from_scan_curve",
            "physics_status": "partial" if compatibility_class == "B" else ("pass" if compatibility_class == "A" else "fail"),
            "notes": reason,
        })

        compat = {
            "case_dir": case["case_dir"],
            "radius_nm": radius_nm if radius_nm is not None else "",
            "dx_nm": dx_nm,
            "radius_over_dx": rdx,
            "effective_radius_nm": geom["effective_radius_nm"],
            "effective_radius_over_dx": eff_rdx,
            "shape_type_observed": geom["shape_type_observed"],
            "cuda_compatibility_class": compatibility_class,
            "usable_for_insertion": str(bool(usable)).lower(),
            "required_fix": required_fix,
            "classification_reason": reason,
            "phi_final": case["phi_final"],
            "xB_final": case.get("xB_final", ""),
        }
        compat_rows.append(compat)

        if usable and phi.size:
            template_case = {**case, **row}
            templates.append(_write_template(template_case, phi, template_dir, root))

    _write_csv(output_dir / "geometry_validity_report.csv", geometry_rows, _union_fields(geometry_rows))
    _write_csv(output_dir / "physics_consistency_report.csv", physics_rows, _union_fields(physics_rows))
    _write_csv(output_dir / "cuda_insertion_compatibility_table.csv", compat_rows, _union_fields(compat_rows))

    metadata = {
        "template_count": len(templates),
        "templates": [asdict(t) for t in templates],
        "classification_policy": {
            "A": "direct CUDA insertion: r/dx>=4, compact, connected, smooth interface",
            "B": "requires interpolation/resampling or smoothing before insertion",
            "C": "not usable without re-minimization/refinement/rejection",
        },
    }
    (template_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    total = len(compat_rows)
    ready = sum(1 for r in compat_rows if r["cuda_compatibility_class"] == "A")
    partial = sum(1 for r in compat_rows if r["cuda_compatibility_class"] == "B")
    rejected = sum(1 for r in compat_rows if r["cuda_compatibility_class"] == "C")
    usable_ratio = ready / total if total else 0.0
    dominant_failure = "none"
    if rejected or partial:
        failures: dict[str, int] = {}
        for row in compat_rows:
            if row["cuda_compatibility_class"] == "A":
                continue
            key = str(row["required_fix"])
            failures[key] = failures.get(key, 0) + 1
        dominant_failure = max(failures.items(), key=lambda kv: kv[1])[0] if failures else "none"
    status = "READY" if total and ready == total else ("PARTIAL" if ready or partial else "NOT READY")
    report = f"""# CUDA Nucleus Usability Report

## Summary

- scanned nuclei with `phi_final_*.vtk`: {total}
- class A directly usable nuclei: {ready}
- class B interpolation/resampling required: {partial}
- class C rejected nuclei: {rejected}
- usable nuclei ratio: {usable_ratio:.6f}
- dominant failure mode: {dominant_failure}
- system status: {status}

## Decision

CNT/minimized nuclei are considered direct CUDA insertion templates only when
their phase-field geometry is sufficiently resolved, connected, compact, and
has a smooth interface. Under-resolved nuclei are not promoted to direct
templates, because they can introduce sub-grid insertion artifacts.

## CNT To CUDA Mapping Validity

Current mapping is `{status}` for direct insertion. Class A entries are exported
under `cuda_nucleus_templates/`. Class B entries require interpolation or
resampling before scheduled insertion. Class C entries should be rejected or
re-minimized on a finer grid.

## Recommendation

{"Proceed to PF insertion for class A templates." if ready else "Do not directly insert the current minimized nuclei; refine scan/grid or generate resampled templates first."}
"""
    (output_dir / "cuda_nucleus_usability_report.md").write_text(report, encoding="utf-8")

    print(f"usable_nuclei_ratio={usable_ratio:.6f}")
    print(f"insertion_ready_count={ready}")
    print(f"rejected_count={rejected}")
    print(f"system_status = {status}")
    print(f"class_B_resample_required={partial}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
