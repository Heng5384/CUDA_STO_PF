#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import subprocess
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.analyze_cnt_peak_table import parse_summary_file
from tools.analysis.generate_continue_dynamic_geometry_summaries import _read_legacy_scalar_vtk


FAMILY_LABELS = [f"{sx:+d}{sy:+d}{sz:+d}" for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)]
FAMILY_VECS = np.array([
    np.array([sx, sy, sz], dtype=float) / np.sqrt(3.0)
    for sx in (-1, 1)
    for sy in (-1, 1)
    for sz in (-1, 1)
])


def _h_phi(phi: np.ndarray) -> np.ndarray:
    return 6.0 * phi**5 - 15.0 * phi**4 + 10.0 * phi**3


def _smoothstep01(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return 3.0 * s**2 - 2.0 * s**3


def _resolve_path(path: Path) -> Path:
    path = path.expanduser()
    if path.is_absolute():
        return path
    return (Path.cwd() / path).resolve()


def _interp_const(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x, y, left=float(y[0]), right=float(y[-1]))


def _interp_delta(
    x: np.ndarray,
    y: np.ndarray,
    x_new: np.ndarray,
    extrapolate: str,
) -> np.ndarray:
    if extrapolate == "clamp":
        return _interp_const(x, y, x_new)
    return np.interp(x_new, x, y, left=0.0, right=0.0)


def _write_vtk_scalar(path: Path, data: np.ndarray, name: str, dx_nm: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = data.shape
    with path.open("w", encoding="ascii") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{name}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write("ORIGIN 0 0 0\n")
        f.write(f"SPACING {dx_nm:.12g} {dx_nm:.12g} {dx_nm:.12g}\n")
        f.write(f"POINT_DATA {nx * ny * nz}\n")
        f.write(f"SCALARS {name} double 1\n")
        f.write("LOOKUP_TABLE default\n")
        flat = data.ravel(order="C")
        for start in range(0, flat.size, 6):
            f.write(" ".join(f"{v:.10e}" for v in flat[start:start + 6]))
            f.write("\n")


def _read_legacy_scalar_vtk_with_meta(
    path: Path,
) -> tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float], tuple[float, float, float]]:
    dims: tuple[int, int, int] | None = None
    spacing = (1.0, 1.0, 1.0)
    origin = (0.0, 0.0, 0.0)

    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF before data in {path}")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("DIMENSIONS"):
                _, sx, sy, sz = text.split()
                dims = (int(sx), int(sy), int(sz))
            elif text.startswith("SPACING"):
                _, sx, sy, sz = text.split()
                spacing = (float(sx), float(sy), float(sz))
            elif text.startswith("ASPECT_RATIO"):
                _, sx, sy, sz = text.split()
                spacing = (float(sx), float(sy), float(sz))
            elif text.startswith("ORIGIN"):
                parts = text.split()
                if len(parts) >= 4:
                    origin = (float(parts[1]), float(parts[2]), float(parts[3]))
            elif text.startswith("LOOKUP_TABLE"):
                break

        if dims is None:
            raise ValueError(f"Missing DIMENSIONS in {path}")

        payload = f.read().decode("ascii", errors="replace")

    values = np.fromstring(payload, sep=" ", dtype=np.float64)
    expected = dims[0] * dims[1] * dims[2]
    if values.size != expected:
        raise ValueError(f"VTK scalar count mismatch in {path}: got {values.size}, expected {expected}")
    return values.reshape(dims, order="C"), dims, spacing, origin


def _load_profiles(profile_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    profile_dir = _resolve_path(profile_dir)
    files = sorted(p for p in profile_dir.iterdir()) if profile_dir.exists() else []
    profile_csv = profile_dir / "faceted_family_profiles.csv"
    if not profile_csv.exists():
        csv_candidates = [p for p in files if p.suffix.lower() == ".csv" and "profile" in p.name.lower()]
        if csv_candidates:
            profile_csv = csv_candidates[0]
    if not profile_csv.exists():
        listing = "\n".join(str(p) for p in files) or "(profile dir missing or empty)"
        raise FileNotFoundError(
            f"Could not identify faceted profile CSV in {profile_dir}.\nFiles:\n{listing}"
        )

    rows = list(csv.DictReader(profile_csv.open(newline="", encoding="utf-8")))
    required = {"family", "u_nm", "phi_mean", "xB_mean"}
    if not rows or not required.issubset(rows[0].keys()):
        fields = list(rows[0].keys()) if rows else []
        raise ValueError(f"Profile file {profile_csv} does not contain required fields {sorted(required)}; fields={fields}")

    profile_by_family: dict[str, dict[str, np.ndarray]] = {}
    for family in sorted({r["family"] for r in rows}):
        family_rows = [r for r in rows if r["family"] == family]
        regions = [r.get("region", "all") for r in family_rows]
        region = "face" if "face" in regions else regions[0]
        pts = [r for r in family_rows if r.get("region", "all") == region]
        pts.sort(key=lambda r: float(r["u_nm"]))
        profile_by_family[family] = {
            "u_nm": np.array([float(r["u_nm"]) for r in pts], dtype=float),
            "phi_mean": np.array([float(r["phi_mean"]) for r in pts], dtype=float),
            "xB_mean": np.array([float(r["xB_mean"]) for r in pts], dtype=float),
            "region": np.array([region], dtype=object),
        }

    meta_path = profile_dir / "faceted_family_metadata.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    all_u = np.concatenate([v["u_nm"] for v in profile_by_family.values()])
    loader_info = {
        "profile_dir": str(profile_dir),
        "profile_file": str(profile_csv),
        "metadata_file": str(meta_path) if meta_path.exists() else None,
        "fields": list(rows[0].keys()),
        "family_count": len(profile_by_family),
        "families": sorted(profile_by_family),
        "distance_min_nm": float(np.min(all_u)),
        "distance_max_nm": float(np.max(all_u)),
        "profile_metadata": meta,
    }
    return profile_by_family, loader_info


def _source_paths(source_dyn_dir: Path | None, phi_name: str, xb_name: str) -> tuple[Path | None, Path | None, Path | None]:
    if source_dyn_dir is None:
        return None, None, None
    d = _resolve_path(source_dyn_dir)
    return d / phi_name, d / xb_name, d / "summary.txt"


def _source_center_from_phi(
    phi: np.ndarray,
    dx_nm: float,
    origin: tuple[float, float, float],
) -> tuple[np.ndarray, str]:
    h = _h_phi(np.clip(phi, 0.0, 1.0))
    total = float(np.sum(h))
    if total <= 1.0e-14:
        mask = phi > 0.5
        if not np.any(mask):
            nx, ny, nz = phi.shape
            return np.array([
                origin[0] + 0.5 * nx * dx_nm,
                origin[1] + 0.5 * ny * dx_nm,
                origin[2] + 0.5 * nz * dx_nm,
            ], dtype=float), "box_center_fallback"
        wx = mask.sum(axis=(1, 2)).astype(float)
        wy = mask.sum(axis=(0, 2)).astype(float)
        wz = mask.sum(axis=(0, 1)).astype(float)
        method = "phi_gt_0p5_mask_centroid"
        total = float(np.sum(wx))
    else:
        wx = h.sum(axis=(1, 2))
        wy = h.sum(axis=(0, 2))
        wz = h.sum(axis=(0, 1))
        method = "h_phi_weighted_centroid"
    x = origin[0] + np.arange(phi.shape[0], dtype=float) * dx_nm
    y = origin[1] + np.arange(phi.shape[1], dtype=float) * dx_nm
    z = origin[2] + np.arange(phi.shape[2], dtype=float) * dx_nm
    return np.array([
        float(np.dot(x, wx) / total),
        float(np.dot(y, wy) / total),
        float(np.dot(z, wz) / total),
    ], dtype=float), method


def _source_mass_target_stats(source_phi: np.ndarray | None, source_xb: np.ndarray | None, nominal: float) -> dict[str, float | None]:
    if source_phi is None or source_xb is None:
        return {
            "source_vtk_mean_xBtot_actual": None,
            "source_vtk_mean_xB_actual": None,
            "source_vtk_mean_phi_actual": None,
            "source_vtk_mean_hphi_actual": None,
            "source_vtk_mass_difference_from_nominal": None,
        }
    h = _h_phi(np.clip(source_phi, 0.0, 1.0))
    xBtot = (1.0 - h) * source_xb + h
    mean_xBtot = float(np.mean(xBtot))
    return {
        "source_vtk_mean_xBtot_actual": mean_xBtot,
        "source_vtk_mean_xB_actual": float(np.mean(source_xb)),
        "source_vtk_mean_phi_actual": float(np.mean(source_phi)),
        "source_vtk_mean_hphi_actual": float(np.mean(h)),
        "source_vtk_mass_difference_from_nominal": mean_xBtot - nominal,
    }


def _axis_from_summary(summary_path: Path | None) -> np.ndarray | None:
    if summary_path is None or not summary_path.exists():
        return None
    info = parse_summary_file(summary_path)
    try:
        axes = np.array([
            [float(info["long_axis_x"]), float(info["long_axis_y"]), float(info["long_axis_z"])],
            [float(info["mid_axis_x"]), float(info["mid_axis_y"]), float(info["mid_axis_z"])],
            [float(info["short_axis_x"]), float(info["short_axis_y"]), float(info["short_axis_z"])],
        ], dtype=float)
    except KeyError:
        return None
    norms = np.linalg.norm(axes, axis=1)
    if np.any(norms < 1.0e-12):
        return None
    return axes / norms[:, None]


def _spacing_from_summary(summary_path: Path | None, fallback: tuple[float, float, float]) -> tuple[float, float, float]:
    if summary_path is None or not summary_path.exists():
        return fallback
    info = parse_summary_file(summary_path)
    try:
        return (
            float(info["spacing_x"]),
            float(info["spacing_y"]),
            float(info["spacing_z"]),
        )
    except KeyError:
        return fallback


def _trilinear_sample(field: np.ndarray, spacing: tuple[float, float, float], points_xyz: np.ndarray) -> np.ndarray:
    dx, dy, dz = spacing
    nx, ny, nz = field.shape
    gx = np.clip(points_xyz[:, 0] / dx, 0.0, nx - 1.000001)
    gy = np.clip(points_xyz[:, 1] / dy, 0.0, ny - 1.000001)
    gz = np.clip(points_xyz[:, 2] / dz, 0.0, nz - 1.000001)
    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    z0 = np.floor(gz).astype(int)
    x1 = np.clip(x0 + 1, 0, nx - 1)
    y1 = np.clip(y0 + 1, 0, ny - 1)
    z1 = np.clip(z0 + 1, 0, nz - 1)
    tx = gx - x0
    ty = gy - y0
    tz = gz - z0
    c000 = field[x0, y0, z0]
    c100 = field[x1, y0, z0]
    c010 = field[x0, y1, z0]
    c110 = field[x1, y1, z0]
    c001 = field[x0, y0, z1]
    c101 = field[x1, y0, z1]
    c011 = field[x0, y1, z1]
    c111 = field[x1, y1, z1]
    c00 = c000 * (1.0 - tx) + c100 * tx
    c10 = c010 * (1.0 - tx) + c110 * tx
    c01 = c001 * (1.0 - tx) + c101 * tx
    c11 = c011 * (1.0 - tx) + c111 * tx
    c0 = c00 * (1.0 - ty) + c10 * ty
    c1 = c01 * (1.0 - ty) + c11 * ty
    return c0 * (1.0 - tz) + c1 * tz


def _find_crossing(s: np.ndarray, phi: np.ndarray, threshold: float, positive_side: bool) -> float | None:
    center_idx = int(np.argmin(np.abs(s)))
    indices = range(center_idx, len(s) - 1) if positive_side else range(center_idx, 0, -1)
    for i in indices:
        j = i + 1 if positive_side else i - 1
        pi = phi[i]
        pj = phi[j]
        if pi == threshold:
            return float(s[i])
        if (pi - threshold) * (pj - threshold) <= 0.0 and pi != pj:
            t = (threshold - pi) / (pj - pi)
            return float(s[i] + t * (s[j] - s[i]))
    return None


def _semiaxes_from_source_phi(
    phi: np.ndarray | None,
    spacing: tuple[float, float, float] | None,
    axes: np.ndarray | None,
    fallback_radius_nm: float,
) -> tuple[np.ndarray, str]:
    if phi is None or spacing is None or axes is None:
        return np.array([fallback_radius_nm] * 3, dtype=float), "spherical_fallback"
    nx, ny, nz = phi.shape
    center = np.array([0.5 * nx * spacing[0], 0.5 * ny * spacing[1], 0.5 * nz * spacing[2]], dtype=float)
    half_range = 0.5 * min(nx * spacing[0], ny * spacing[1], nz * spacing[2])
    s_grid = np.arange(-half_range, half_range + 0.05, 0.05)
    semiaxes = []
    for axis in axes:
        pts = center[None, :] + s_grid[:, None] * axis[None, :]
        vals = _trilinear_sample(phi, spacing, pts)
        s_neg = _find_crossing(s_grid, vals, 0.5, positive_side=False)
        s_pos = _find_crossing(s_grid, vals, 0.5, positive_side=True)
        if s_neg is None or s_pos is None:
            return np.array([fallback_radius_nm] * 3, dtype=float), "spherical_fallback"
        semiaxes.append(0.5 * (abs(s_neg) + abs(s_pos)))
    return np.maximum(np.array(semiaxes, dtype=float), 1.0e-6), "ellipsoid_from_source_phi"


def _source_stats(phi: np.ndarray | None, xb: np.ndarray | None) -> dict[str, float]:
    if phi is None or xb is None:
        return {}
    h = _h_phi(np.clip(phi, 0.0, 1.0))
    stats: dict[str, float] = {
        "actual_phi_min": float(np.min(phi)),
        "actual_phi_max": float(np.max(phi)),
        "actual_phi_mean": float(np.mean(phi)),
        "actual_xB_min": float(np.min(xb)),
        "actual_xB_max": float(np.max(xb)),
        "actual_xB_mean": float(np.mean(xb)),
        "actual_xBtot_mean": float(np.mean((1.0 - h) * xb + h)),
    }
    masks = {
        "xB_mean_phi_lt_0p05": phi < 0.05,
        "xB_mean_phi_lt_0p10": phi < 0.10,
        "xB_mean_phi_gt_0p90": phi > 0.90,
        "xB_mean_phi_gt_0p95": phi > 0.95,
        "xB_mean_interface_0p1_0p9": (phi > 0.10) & (phi < 0.90),
    }
    for key, mask in masks.items():
        stats[key] = float(np.mean(xb[mask])) if np.any(mask) else float("nan")
    return stats


def _family_index_from_local(local: np.ndarray) -> np.ndarray:
    signs = np.where(local >= 0.0, 1, -1)
    return ((signs[..., 0] > 0).astype(np.int8) * 4
            + (signs[..., 1] > 0).astype(np.int8) * 2
            + (signs[..., 2] > 0).astype(np.int8))


def _profile_plateau_reference(profiles: dict[str, dict[str, np.ndarray]]) -> float:
    vals = []
    for prof in profiles.values():
        idx = int(np.argmax(prof["u_nm"]))
        vals.append(float(prof["xB_mean"][idx]))
    if not vals:
        raise ValueError("No profile values available to estimate xB_matrix_reference")
    return float(np.mean(vals))


def _build_geometry(
    nx: int,
    ny: int,
    nz: int,
    dx_nm: float,
    center_nm: np.ndarray,
    axes: np.ndarray,
    semiaxes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.arange(nx, dtype=np.float32) * dx_nm - center_nm[0]
    y = np.arange(ny, dtype=np.float32) * dx_nm - center_nm[1]
    z = np.arange(nz, dtype=np.float32) * dx_nm - center_nm[2]
    xx, yy, zz = np.meshgrid(x, y, z, indexing="ij")
    rel = np.stack([xx, yy, zz], axis=-1)
    local = np.einsum("...j,ij->...i", rel, axes).astype(np.float32)
    scaled = local / semiaxes.astype(np.float32)
    rho = np.sqrt(np.sum(scaled * scaled, axis=-1)).astype(np.float32)
    r = np.sqrt(np.sum(local * local, axis=-1)).astype(np.float32)
    boundary_radius = np.where(rho > 1.0e-8, r / np.maximum(rho, 1.0e-8), float(np.mean(semiaxes)))
    d_nm = ((rho - 1.0) * boundary_radius).astype(np.float32)
    family_idx = _family_index_from_local(local)
    return rho, d_nm, family_idx, local


def _rebuild_from_profiles(
    profiles: dict[str, dict[str, np.ndarray]],
    d_nm: np.ndarray,
    family_idx: np.ndarray,
    xB_matrix_reference: float,
    alpha_interface: float,
    xB_rebuild_mode: str,
    xB_matrix_near: float,
    profile_extrapolate: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    phi = np.empty(d_nm.shape, dtype=np.float32)
    delta = np.empty(d_nm.shape, dtype=np.float32)
    for idx, label in enumerate(FAMILY_LABELS):
        prof = profiles.get(label)
        if prof is None:
            prof = next(iter(profiles.values()))
        mask = family_idx == idx
        if not np.any(mask):
            continue
        phi[mask] = _interp_const(prof["u_nm"], prof["phi_mean"], d_nm[mask]).astype(np.float32)
        delta[mask] = _interp_delta(
            prof["u_nm"],
            prof["xB_mean"] - xB_matrix_reference,
            d_nm[mask],
            profile_extrapolate,
        ).astype(np.float32)
    phi = np.clip(phi, 0.0, 1.0).astype(np.float32)
    if xB_rebuild_mode == "constant-matrix":
        xB_near = np.full(d_nm.shape, xB_matrix_near, dtype=np.float32)
    else:
        xB_near = (xB_matrix_near + alpha_interface * delta).astype(np.float32)
    return phi, delta, xB_near


def _center_axis_rows(
    phi: np.ndarray,
    d_nm: np.ndarray,
    xB_local: np.ndarray,
    xB_before_clip: np.ndarray,
    xB: np.ndarray,
    xBtot: np.ndarray,
    W_dep: np.ndarray,
    rho: np.ndarray,
    family_idx: np.ndarray,
    dx_nm: float,
    center_nm: np.ndarray,
    mass_mode: str,
) -> list[dict[str, object]]:
    nx, ny, nz = phi.shape
    cx = int(round(center_nm[0] / dx_nm))
    cy = int(round(center_nm[1] / dx_nm))
    cz = int(round(center_nm[2] / dx_nm))
    cx = int(np.clip(cx, 0, nx - 1))
    cy = int(np.clip(cy, 0, ny - 1))
    cz = int(np.clip(cz, 0, nz - 1))
    rows: list[dict[str, object]] = []
    axes = {
        "x": [(i, cy, cz, i * dx_nm - center_nm[0]) for i in range(nx)],
        "y": [(cx, j, cz, j * dx_nm - center_nm[1]) for j in range(ny)],
        "z": [(cx, cy, k, k * dx_nm - center_nm[2]) for k in range(nz)],
    }
    for axis, pts in axes.items():
        for n, (i, j, k, coord) in enumerate(pts):
            rows.append({
                "axis": axis,
                "index": n,
                "coordinate_nm": float(coord),
                "phi": float(phi[i, j, k]),
                "h_phi": float(_h_phi(np.array(phi[i, j, k]))),
                "d_nm": float(d_nm[i, j, k]),
                "xB_local": float(xB_local[i, j, k]),
                "xB_before_clip": float(xB_before_clip[i, j, k]),
                "xB_after_clip": float(xB[i, j, k]),
                "xB": float(xB[i, j, k]),
                "xBtot": float(xBtot[i, j, k]),
                "W_dep": float(W_dep[i, j, k]),
                "W_blend": float(W_dep[i, j, k]),
                "rho": float(rho[i, j, k]),
                "face_family": FAMILY_LABELS[int(family_idx[i, j, k])],
                "mass_mode": mass_mode,
            })
    return rows


def _write_center_axis_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = [
        "axis",
        "index",
        "coordinate_nm",
        "phi",
        "h_phi",
        "d_nm",
        "xB_local",
        "xB_before_clip",
        "xB_after_clip",
        "xB",
        "xBtot",
        "W_dep",
        "W_blend",
        "rho",
        "face_family",
        "mass_mode",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def _plot_center_axes(path: Path, rows: list[dict[str, object]], xB_far: float) -> None:
    fig, axs = plt.subplots(3, 6, figsize=(22, 10), constrained_layout=True)
    fields = ["phi", "xB_local", "xB_before_clip", "xB_after_clip", "xBtot", "W_dep"]
    for i, axis in enumerate(["x", "y", "z"]):
        axis_rows = [r for r in rows if r["axis"] == axis]
        coord = np.array([float(r["coordinate_nm"]) for r in axis_rows])
        for j, field in enumerate(fields):
            vals = np.array([float(r[field]) for r in axis_rows])
            axs[i, j].plot(coord, vals, lw=1.6)
            if field.startswith("xB"):
                axs[i, j].axhline(xB_far, color="#666666", lw=1.0, ls=":", label="xB_far")
                axs[i, j].legend(frameon=False, fontsize=7)
            axs[i, j].set_title(f"{axis}-axis {field}")
            axs[i, j].set_xlabel(f"{axis} coordinate from center (nm)")
            axs[i, j].grid(True, alpha=0.25)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def _write_txt(path: Path, data: dict[str, Any]) -> None:
    lines: list[str] = []
    for section, values in data.items():
        lines.append(f"[{section}]")
        if isinstance(values, dict):
            for key, value in values.items():
                lines.append(f"{key}: {value}")
        else:
            lines.append(str(values))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def _clip_report(xB: np.ndarray, xB_min: float, xB_max: float) -> tuple[np.ndarray, float, float]:
    below = xB < xB_min
    above = xB > xB_max
    lower_clip_fraction = float(np.mean(below))
    upper_clip_fraction = float(np.mean(above))
    return np.clip(xB, xB_min, xB_max).astype(np.float32), lower_clip_fraction, upper_clip_fraction


def _mass_summary(
    xBtot: np.ndarray,
    xBtot_target_used: float,
    dx_nm: float,
    xB0_target_nominal: float | None = None,
) -> dict[str, float]:
    excess = xBtot - xBtot_target_used
    out = {
        "mean_xBtot": float(np.mean(xBtot)),
        "local_mass_excess_mean": float(np.mean(excess)),
        "local_mass_excess_integral_grid": float(np.sum(excess)),
        "local_mass_excess_integral_physical_composition_nm3": float(np.sum(excess) * dx_nm**3),
    }
    if xB0_target_nominal is not None:
        out["local_mass_excess_vs_nominal"] = float(np.mean(xBtot) - xB0_target_nominal)
    return out


def _sample_center_axis(field: np.ndarray, dx_nm: float, center_nm: np.ndarray, axis: str, coords: np.ndarray) -> np.ndarray:
    axis_unit = {
        "x": np.array([1.0, 0.0, 0.0], dtype=float),
        "y": np.array([0.0, 1.0, 0.0], dtype=float),
        "z": np.array([0.0, 0.0, 1.0], dtype=float),
    }[axis]
    pts = center_nm[None, :] + coords[:, None] * axis_unit[None, :]
    return _trilinear_sample(field, (dx_nm, dx_nm, dx_nm), pts)


def _actual_comparison(
    out_dir: Path,
    source_phi: np.ndarray | None,
    source_xb: np.ndarray | None,
    phi: np.ndarray,
    xB: np.ndarray,
    xBtot: np.ndarray,
    W_dep: np.ndarray,
    rows: list[dict[str, object]],
    dx_nm: float,
    center_nm: np.ndarray,
    xB_far: float,
    mass_mode: str,
    xB0_target_nominal: float,
    write_plots: bool,
) -> dict[str, Any]:
    if source_phi is None or source_xb is None:
        return {"available": False}
    if source_phi.shape != phi.shape or source_xb.shape != xB.shape:
        return {
            "available": False,
            "reason": f"shape mismatch actual phi/xB {source_phi.shape}/{source_xb.shape} vs recon {phi.shape}",
        }

    actual_h = _h_phi(np.clip(source_phi, 0.0, 1.0))
    actual_xBtot = ((1.0 - actual_h) * source_xb + actual_h).astype(np.float32)
    interface_mask = (source_phi > 0.1) & (source_phi < 0.9)
    matrix_mask = source_phi < 0.05
    far_mask = (W_dep < 1.0e-3) & (phi < 0.05)
    near_matrix_mask = (W_dep > 0.999) & (phi < 0.1)

    axis_metrics: list[dict[str, float | str]] = []
    plot_rows: list[dict[str, object]] = []
    for axis in ("x", "y", "z"):
        axis_rows = [r for r in rows if r["axis"] == axis]
        coords = np.array([float(r["coordinate_nm"]) for r in axis_rows], dtype=float)
        phi_actual_line = _sample_center_axis(source_phi, dx_nm, center_nm, axis, coords)
        xb_actual_line = _sample_center_axis(source_xb, dx_nm, center_nm, axis, coords)
        xbtot_actual_line = (1.0 - _h_phi(np.clip(phi_actual_line, 0.0, 1.0))) * xb_actual_line + _h_phi(np.clip(phi_actual_line, 0.0, 1.0))
        phi_recon_line = np.array([float(r["phi"]) for r in axis_rows], dtype=float)
        xb_recon_line = np.array([float(r["xB_after_clip"]) for r in axis_rows], dtype=float)
        xbtot_recon_line = np.array([float(r["xBtot"]) for r in axis_rows], dtype=float)
        W_line = np.array([float(r["W_dep"]) for r in axis_rows], dtype=float)
        axis_metrics.append({
            "axis": axis,
            "phi_rmse": float(np.sqrt(np.mean((phi_actual_line - phi_recon_line) ** 2))),
            "xB_rmse": float(np.sqrt(np.mean((xb_actual_line - xb_recon_line) ** 2))),
            "xBtot_rmse": float(np.sqrt(np.mean((xbtot_actual_line - xbtot_recon_line) ** 2))),
        })
        for values in zip(coords, phi_actual_line, phi_recon_line, xb_actual_line, xb_recon_line, xbtot_actual_line, xbtot_recon_line, W_line):
            plot_rows.append({
                "axis": axis,
                "coordinate_nm": float(values[0]),
                "phi_actual": float(values[1]),
                "phi_recon": float(values[2]),
                "xB_actual": float(values[3]),
                "xB_recon": float(values[4]),
                "xBtot_actual": float(values[5]),
                "xBtot_recon": float(values[6]),
                "W_dep": float(values[7]),
            })

    if plot_rows:
        csv_path = out_dir / "actual_vs_reconstructed_center_axes.csv"
        with csv_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(plot_rows[0].keys()))
            writer.writeheader()
            writer.writerows(plot_rows)

    if write_plots and plot_rows:
        fig, axs = plt.subplots(3, 4, figsize=(19, 10.5), constrained_layout=True)
        for i, axis in enumerate(("x", "y", "z")):
            axis_rows = [r for r in plot_rows if r["axis"] == axis]
            coord = np.array([float(r["coordinate_nm"]) for r in axis_rows])
            phi_actual_line = np.array([float(r["phi_actual"]) for r in axis_rows])
            phi_recon_line = np.array([float(r["phi_recon"]) for r in axis_rows])
            xb_actual_line = np.array([float(r["xB_actual"]) for r in axis_rows])
            xb_recon_line = np.array([float(r["xB_recon"]) for r in axis_rows])
            xbtot_actual_line = np.array([float(r["xBtot_actual"]) for r in axis_rows])
            xbtot_recon_line = np.array([float(r["xBtot_recon"]) for r in axis_rows])
            W_line = np.array([float(r["W_dep"]) for r in axis_rows])
            axs[i, 0].plot(coord, phi_actual_line, label="actual", color="#263746")
            axs[i, 0].plot(coord, phi_recon_line, "--", label="recon", color="#d95f02")
            axs[i, 1].plot(coord, xb_actual_line, label="actual xB", color="#263746")
            axs[i, 1].plot(coord, xb_recon_line, "--", label="recon xB", color="#d95f02")
            axs[i, 1].axhline(xB_far, color="#777777", ls=":", label="xB_far")
            axs[i, 2].plot(coord, xbtot_actual_line, label="actual xBtot", color="#263746")
            axs[i, 2].plot(coord, xbtot_recon_line, "--", label="recon xBtot", color="#1b9e77")
            axs[i, 3].plot(coord, W_line, label="W_dep", color="#2c7fb8")
            axs[i, 0].set_title(f"{axis}-axis phi")
            axs[i, 1].set_title(f"{axis}-axis xB")
            axs[i, 2].set_title(f"{axis}-axis xBtot")
            axs[i, 3].set_title(f"{axis}-axis W_dep")
            axs[i, 1].set_ylim(min(float(np.min(xb_actual_line)), float(np.min(xb_recon_line)), 0.0) - 0.001, max(float(np.max(xb_actual_line)), float(np.max(xb_recon_line)), xB_far) + 0.001)
            for ax in axs[i, :]:
                ax.set_xlabel(f"{axis} coordinate from center (nm)")
                ax.grid(True, alpha=0.25)
                ax.legend(frameon=False, fontsize=8)
        fig.suptitle(f"Actual source VTK vs reconstructed: {mass_mode}", fontsize=14)
        fig.savefig(out_dir / "actual_vs_reconstructed_center_axes.png", dpi=180)
        plt.close(fig)

    comparison = {
        "available": True,
        "note": "xB actual mismatch is not necessarily a failure under embedding-report mode because embedding-report intentionally restores far field to xB_far.",
        "whole_box_phi_rmse": float(np.sqrt(np.mean((source_phi - phi) ** 2))),
        "whole_box_xB_rmse": float(np.sqrt(np.mean((source_xb - xB) ** 2))),
        "whole_box_xBtot_rmse": float(np.sqrt(np.mean((actual_xBtot - xBtot) ** 2))),
        "interface_phi_rmse": float(np.sqrt(np.mean((source_phi[interface_mask] - phi[interface_mask]) ** 2))) if np.any(interface_mask) else float("nan"),
        "axis_metrics": axis_metrics,
        "actual_near_matrix_mean_xB": float(np.mean(source_xb[near_matrix_mask])) if np.any(near_matrix_mask) else float("nan"),
        "recon_near_matrix_mean_xB": float(np.mean(xB[near_matrix_mask])) if np.any(near_matrix_mask) else float("nan"),
        "actual_far_matrix_mean_xB": float(np.mean(source_xb[far_mask])) if np.any(far_mask) else float("nan"),
        "recon_far_matrix_mean_xB": float(np.mean(xB[far_mask])) if np.any(far_mask) else float("nan"),
        "actual_matrix_mean_xB_phi_lt_0p05": float(np.mean(source_xb[matrix_mask])) if np.any(matrix_mask) else float("nan"),
        "actual_mean_xBtot": float(np.mean(actual_xBtot)),
        "recon_mean_xBtot": float(np.mean(xBtot)),
        "recon_minus_actual_mean_xBtot": float(np.mean(xBtot) - np.mean(actual_xBtot)),
        "recon_minus_nominal_xB0_target": float(np.mean(xBtot) - xB0_target_nominal),
        "actual_xBtot_min": float(np.min(actual_xBtot)),
        "actual_xBtot_max": float(np.max(actual_xBtot)),
        "recon_xBtot_min": float(np.min(xBtot)),
        "recon_xBtot_max": float(np.max(xBtot)),
    }
    (out_dir / "actual_vs_reconstructed_metrics.json").write_text(json.dumps(comparison, indent=2), encoding="utf-8")
    return comparison


def _strip_cli_option(argv: list[str], option: str, has_value: bool) -> list[str]:
    out: list[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg == option:
            i += 2 if has_value else 1
            continue
        if has_value and arg.startswith(option + "="):
            i += 1
            continue
        out.append(arg)
        i += 1
    return out


def _run_all_mass_modes(argv: list[str], out_dir: Path) -> int:
    base_argv = list(argv)
    for option, has_value in (
        ("--run-all-mass-modes", False),
        ("--mass-mode", True),
        ("--out-dir", True),
    ):
        base_argv = _strip_cli_option(base_argv, option, has_value)

    mode_dirs = {
        "embedding-report": "embedding_report",
        "closed-box": "closed_box",
        "far-field-compensate": "far_field_compensate",
    }
    out_dir.mkdir(parents=True, exist_ok=True)
    summary_rows: list[dict[str, Any]] = []
    global_info: dict[str, Any] = {}
    for mode, dirname in mode_dirs.items():
        child_out = out_dir / dirname
        cmd = [
            sys.executable,
            str(Path(__file__).resolve()),
            *base_argv,
            "--out-dir",
            str(child_out),
            "--mass-mode",
            mode,
        ]
        result = subprocess.run(cmd, cwd=Path.cwd(), check=False)
        if result.returncode != 0:
            return result.returncode
        diag_path = child_out / "diagnostics.json"
        diag = json.loads(diag_path.read_text(encoding="utf-8"))
        if not global_info:
            mass_target = diag.get("MassTarget", {})
            global_info = {
                "mass_target_mode": mass_target.get("mass_target_mode"),
                "xB0_target_nominal": mass_target.get("xB0_target_nominal"),
                "xBtot_target_used": mass_target.get("xBtot_target_used"),
                "source_vtk_mean_xBtot_actual": mass_target.get("source_vtk_mean_xBtot_actual"),
                "source_vtk_mass_difference_from_nominal": mass_target.get("source_vtk_mass_difference_from_nominal"),
            }
        history = diag.get("Mass", {}).get("mass_correction_history", [])
        last_c = None
        if history:
            last = history[-1]
            last_c = last.get("correction_C", last.get("correction_C_far"))
        summary_rows.append({
            "mass_mode": mode,
            "output_dir": str(child_out),
            "mean_xBtot": diag.get("Mass", {}).get("final_mean_xBtot"),
            "mass_error_vs_target": diag.get("Mass", {}).get("final_mass_error"),
            "mass_error_vs_nominal": diag.get("Mass", {}).get("mass_error_vs_nominal"),
            "xB_min": diag.get("xB", {}).get("xB_min_after_final_clipping"),
            "xB_max": diag.get("xB", {}).get("xB_max_after_final_clipping"),
            "lower_clipped_fraction": diag.get("xB", {}).get("lower_clipped_fraction"),
            "upper_clipped_fraction": diag.get("xB", {}).get("upper_clipped_fraction"),
            "far_field_mean_xB": diag.get("Blending", {}).get("far_field_mean_xB"),
            "near_matrix_mean_xB": diag.get("Blending", {}).get("near_matrix_mean_xB"),
            "final_correction": last_c,
            "warnings_count": len(diag.get("Warnings", [])),
        })
    (out_dir / "same_size_summary.json").write_text(json.dumps({
        **global_info,
        "results": summary_rows,
    }, indent=2), encoding="utf-8")
    print(f"same_size_summary={out_dir / 'same_size_summary.json'}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Test stable phi/xB initialization rebuilt from dynamic faceted profiles.")
    parser.add_argument("--profile-dir", type=Path, required=True)
    parser.add_argument("--source-dyn-dir", type=Path, default=None)
    parser.add_argument("--source-phi-vtk", type=Path, default=None)
    parser.add_argument("--source-xB-vtk", type=Path, default=None)
    parser.add_argument("--phi-vtk-name", default="phi_1500.vtk")
    parser.add_argument("--xB-vtk-name", default="xB_1500.vtk")
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--same-as-source-vtk", action="store_true")
    parser.add_argument("--use-source-center", action="store_true")
    parser.add_argument("--run-all-mass-modes", action="store_true")
    parser.add_argument("--compare-actual", action="store_true", default=None)
    parser.add_argument("--no-compare-actual", dest="compare_actual", action="store_false")
    parser.add_argument("--Nx", type=int, default=192)
    parser.add_argument("--Ny", type=int, default=192)
    parser.add_argument("--Nz", type=int, default=192)
    parser.add_argument("--dx-nm", type=float, default=0.1)
    parser.add_argument("--center-mode", choices=("box_center", "manual"), default="box_center")
    parser.add_argument("--center-x-nm", type=float, default=None)
    parser.add_argument("--center-y-nm", type=float, default=None)
    parser.add_argument("--center-z-nm", type=float, default=None)
    parser.add_argument("--xB-rebuild-mode", choices=("constant-matrix", "delta-interface"), default="delta-interface")
    parser.add_argument("--alpha-interface", type=float, default=0.25)
    parser.add_argument("--xB-matrix-near", type=float, default=None)
    parser.add_argument("--xB-matrix-reference", type=float, default=None)
    parser.add_argument("--xB-far", type=float, default=None)
    parser.add_argument("--profile-extrapolate", choices=("clamp", "far"), default="far")
    parser.add_argument("--mass-mode", choices=("closed-box", "embedding-report", "far-field-compensate"), default="embedding-report")
    parser.add_argument("--blend-mode", choices=("signed-distance", "rho"), default="signed-distance")
    parser.add_argument("--blend-d1-nm", type=float, default=0.5)
    parser.add_argument("--blend-d2-nm", type=float, default=10.0)
    parser.add_argument("--blend-rho1", type=float, default=1.2)
    parser.add_argument("--blend-rho2", type=float, default=2.5)
    parser.add_argument("--W-comp-threshold", type=float, default=1.0e-3)
    parser.add_argument("--phi-matrix-threshold", type=float, default=0.05)
    parser.add_argument("--fallback-radius-nm", type=float, default=7.0)
    parser.add_argument("--xB0-target", type=float, default=0.030)
    parser.add_argument("--mass-target-mode", choices=("nominal", "source-vtk", "manual"), default=None)
    parser.add_argument("--xBtot-target-manual", type=float, default=None)
    parser.add_argument("--xB-min", type=float, default=1.0e-8)
    parser.add_argument("--xB-max", type=float, default=0.035)
    parser.add_argument("--mass-correction-iters", type=int, default=5)
    parser.add_argument("--mass-tol", type=float, default=1.0e-9)
    parser.add_argument("--write-vtk", action="store_true")
    parser.add_argument("--write-plots", action="store_true", default=True)
    parser.add_argument("--no-write-plots", dest="write_plots", action="store_false")
    args = parser.parse_args()

    out_dir = _resolve_path(args.out_dir)
    if args.run_all_mass_modes:
        return _run_all_mass_modes(sys.argv[1:], out_dir)

    out_dir.mkdir(parents=True, exist_ok=True)
    warnings: list[str] = []

    profiles, loader_info = _load_profiles(args.profile_dir)
    source_phi_path, source_xb_path, source_summary_path = _source_paths(args.source_dyn_dir, args.phi_vtk_name, args.xB_vtk_name)
    if args.source_phi_vtk is not None:
        source_phi_path = _resolve_path(args.source_phi_vtk)
    if args.source_xB_vtk is not None:
        source_xb_path = _resolve_path(args.source_xB_vtk)

    source_phi = None
    source_xb = None
    source_spacing = None
    source_dims = None
    source_origin = (0.0, 0.0, 0.0)
    source_vtk_spacing = None
    if source_phi_path and source_phi_path.exists() and source_xb_path and source_xb_path.exists():
        source_phi, source_dims, source_spacing_vtk, source_origin = _read_legacy_scalar_vtk_with_meta(source_phi_path)
        source_vtk_spacing = source_spacing_vtk
        source_spacing = _spacing_from_summary(source_summary_path, source_spacing_vtk)
        source_xb, _, source_xb_spacing_vtk, _source_xb_origin = _read_legacy_scalar_vtk_with_meta(source_xb_path)
    elif args.source_dyn_dir is not None:
        warnings.append("source dynamic dir provided but phi/xB VTK could not be read")

    compare_actual = (source_phi is not None and source_xb is not None) if args.compare_actual is None else bool(args.compare_actual)
    source_mass_stats = _source_mass_target_stats(source_phi, source_xb, args.xB0_target)
    mass_target_mode = args.mass_target_mode
    if mass_target_mode is None:
        mass_target_mode = "source-vtk" if args.same_as_source_vtk and compare_actual and source_phi is not None and source_xb is not None else "nominal"
    if mass_target_mode == "nominal":
        xBtot_target_used = float(args.xB0_target)
    elif mass_target_mode == "source-vtk":
        source_target = source_mass_stats.get("source_vtk_mean_xBtot_actual")
        if source_target is None:
            raise ValueError("source-vtk target requires both actual phi and actual xB fields.")
        xBtot_target_used = float(source_target)
    else:
        if args.xBtot_target_manual is None:
            raise ValueError("mass-target-mode=manual requires --xBtot-target-manual")
        xBtot_target_used = float(args.xBtot_target_manual)

    axes = _axis_from_summary(source_summary_path)
    if axes is None:
        axes = np.eye(3, dtype=float)
        warnings.append("principal axes unavailable; using identity axes")
    semiaxes, rho_mode = _semiaxes_from_source_phi(source_phi, source_spacing, axes, args.fallback_radius_nm)
    if rho_mode == "spherical_fallback":
        warnings.append("spherical rho fallback used")

    if args.same_as_source_vtk:
        if source_dims is None:
            raise ValueError("--same-as-source-vtk requires a readable source phi VTK")
        warnings.append("Nx/Ny/Nz overridden by source VTK because --same-as-source-vtk is enabled")
        args.Nx, args.Ny, args.Nz = source_dims
        if source_vtk_spacing is not None and max(source_vtk_spacing) - min(source_vtk_spacing) > 1.0e-12:
            warnings.append("source VTK spacing is anisotropic; using --dx-nm scalar for reconstruction")
        elif source_vtk_spacing is not None and abs(source_vtk_spacing[0] - 1.0) > 1.0e-12:
            args.dx_nm = float(source_vtk_spacing[0])
        else:
            warnings.append("using dx_nm from command line because VTK spacing may not be physical nm")

    box_size = np.array([args.Nx * args.dx_nm, args.Ny * args.dx_nm, args.Nz * args.dx_nm], dtype=float)
    center_method = "box_center"
    if (args.use_source_center or args.same_as_source_vtk) and source_phi is not None:
        center_nm, center_method = _source_center_from_phi(source_phi, args.dx_nm, source_origin)
    elif args.center_mode == "box_center":
        center_nm = 0.5 * box_size
    else:
        if args.center_x_nm is None or args.center_y_nm is None or args.center_z_nm is None:
            raise ValueError("manual center mode requires --center-x-nm --center-y-nm --center-z-nm")
        center_nm = np.array([args.center_x_nm, args.center_y_nm, args.center_z_nm], dtype=float)
        center_method = "manual"

    source_stats = _source_stats(source_phi, source_xb)
    xB_matrix_near = args.xB_matrix_near
    if xB_matrix_near is None:
        xB_matrix_near = source_stats.get("xB_mean_phi_lt_0p05", args.xB0_target)
    xB_far = args.xB_far if args.xB_far is not None else args.xB0_target
    xB_matrix_reference = args.xB_matrix_reference
    if xB_matrix_reference is None:
        try:
            xB_matrix_reference = _profile_plateau_reference(profiles)
        except Exception:
            xB_matrix_reference = xB_matrix_near
            warnings.append("profile plateau reference failed; using xB_matrix_near")

    rho, d_nm, family_idx, _local = _build_geometry(args.Nx, args.Ny, args.Nz, args.dx_nm, center_nm, axes, semiaxes)
    phi, delta_xB, xB_near = _rebuild_from_profiles(
        profiles,
        d_nm,
        family_idx,
        float(xB_matrix_reference),
        args.alpha_interface,
        args.xB_rebuild_mode,
        float(xB_matrix_near),
        args.profile_extrapolate,
    )
    if not np.all(np.isfinite(phi)):
        raise FloatingPointError("phi contains NaN or Inf")
    phi = np.clip(phi, 0.0, 1.0).astype(np.float32)
    h = _h_phi(phi).astype(np.float32)
    if args.blend_mode == "signed-distance":
        if args.blend_d2_nm <= args.blend_d1_nm:
            raise ValueError("--blend-d2-nm must be larger than --blend-d1-nm")
        W_dep = (1.0 - _smoothstep01((d_nm - args.blend_d1_nm) / (args.blend_d2_nm - args.blend_d1_nm))).astype(np.float32)
    else:
        if args.blend_rho2 <= args.blend_rho1:
            raise ValueError("--blend-rho2 must be larger than --blend-rho1")
        W_dep = (1.0 - _smoothstep01((rho - args.blend_rho1) / (args.blend_rho2 - args.blend_rho1))).astype(np.float32)

    xB_before_clip = (xB_far + W_dep * (xB_near - xB_far)).astype(np.float32)
    if not np.all(np.isfinite(xB_before_clip)):
        raise FloatingPointError("xB contains NaN or Inf before clipping")

    mass_history: list[dict[str, float]] = []
    compensation_mask_count = 0
    compensation_mask_fraction = 0.0
    xB, lower_clip_fraction, upper_clip_fraction = _clip_report(xB_before_clip, args.xB_min, args.xB_max)

    if args.mass_mode == "closed-box":
        warnings.append(
            "closed-box mode treats the current test box as a finite closed system; "
            "fixed xB_far may be incompatible with a large pre-existing nucleus"
        )
        xB = xB_before_clip.copy()
        for it in range(args.mass_correction_iters):
            xBtot_iter = (1.0 - h) * xB + h
            mean_xBtot_before = float(np.mean(xBtot_iter))
            denom = float(np.mean(1.0 - h))
            if denom <= 1.0e-14:
                C = 0.0
                warnings.append("mean(1-h(phi)) too small for mass correction")
            else:
                C = (xBtot_target_used - mean_xBtot_before) / denom
            xB = xB + C
            xB, lower_clip_fraction, upper_clip_fraction = _clip_report(xB, args.xB_min, args.xB_max)
            xBtot_after = (1.0 - h) * xB + h
            mean_xBtot_after = float(np.mean(xBtot_after))
            mass_history.append({
                "iter": float(it),
                "mean_xBtot_before": mean_xBtot_before,
                "correction_C": float(C),
                "mean_xBtot_after": mean_xBtot_after,
                "mass_error_after": float(mean_xBtot_after - xBtot_target_used),
                "lower_clip_fraction": lower_clip_fraction,
                "upper_clip_fraction": upper_clip_fraction,
            })
            if abs(mean_xBtot_after - xBtot_target_used) <= args.mass_tol:
                break
    elif args.mass_mode == "far-field-compensate":
        far_mask = (W_dep < args.W_comp_threshold) & (phi < args.phi_matrix_threshold)
        compensation_mask_count = int(np.count_nonzero(far_mask))
        compensation_mask_fraction = float(np.mean(far_mask))
        if compensation_mask_count == 0:
            warnings.append("far-field-compensate found no far-field matrix points; no compensation applied")
        else:
            for it in range(args.mass_correction_iters):
                xBtot_iter = (1.0 - h) * xB + h
                mean_xBtot_before = float(np.mean(xBtot_iter))
                mass_error = xBtot_target_used - mean_xBtot_before
                denom = float(np.sum((1.0 - h)[far_mask]) / h.size)
                if denom <= 1.0e-14:
                    C_far = 0.0
                    warnings.append("far-field compensation denominator too small")
                else:
                    C_far = mass_error / denom
                if abs(C_far) > 1.0e-3:
                    warnings.append(
                        "far-field compensation |C_far| exceeds 1e-3; "
                        "this test box may be too small or the inserted nucleus too large"
                    )
                if abs(C_far) > 5.0e-3:
                    warnings.append(
                        "far-field compensation |C_far| exceeds 5e-3; "
                        "this test box is likely too small for far-field-only compensation"
                    )
                xB[far_mask] = xB[far_mask] + C_far
                xB, lower_clip_fraction, upper_clip_fraction = _clip_report(xB, args.xB_min, args.xB_max)
                xBtot_after = (1.0 - h) * xB + h
                mean_xBtot_after = float(np.mean(xBtot_after))
                mass_history.append({
                    "iter": float(it),
                    "mean_xBtot_before": mean_xBtot_before,
                    "correction_C_far": float(C_far),
                    "mean_xBtot_after": mean_xBtot_after,
                    "mass_error_after": float(mean_xBtot_after - xBtot_target_used),
                    "lower_clip_fraction": lower_clip_fraction,
                    "upper_clip_fraction": upper_clip_fraction,
                    "far_mask_fraction": compensation_mask_fraction,
                })
                if abs(mean_xBtot_after - xBtot_target_used) <= args.mass_tol:
                    break

    xBtot = ((1.0 - h) * xB + h).astype(np.float32)
    final_mass_error = float(np.mean(xBtot) - xBtot_target_used)
    converged = abs(final_mass_error) <= args.mass_tol if args.mass_mode != "embedding-report" else False
    if args.mass_mode != "embedding-report" and abs(final_mass_error) > args.mass_tol:
        warnings.append("final mass error exceeds mass_tol")
    if upper_clip_fraction > 1.0e-4 or lower_clip_fraction > 1.0e-4:
        warnings.append("clipped fraction exceeds 1e-4")
    if args.xB_max < args.xB0_target:
        warnings.append("xB_max is below xB0_target")
    if args.mass_mode != "embedding-report" and float(np.mean(h)) > xBtot_target_used:
        warnings.append("mean h(phi) exceeds xBtot_target_used; target total composition is impossible without negative xB")
    if not np.all(np.isfinite(xB)) or not np.all(np.isfinite(xBtot)):
        raise FloatingPointError("final xB/xBtot contains NaN or Inf")

    far_field_mask = W_dep < 1.0e-3
    near_field_mask = W_dep > 0.999
    near_matrix_mask = near_field_mask & (phi < 0.1)
    far_field_mean_xB = float(np.mean(xB[far_field_mask])) if np.any(far_field_mask) else float("nan")
    near_matrix_mean_xB = float(np.mean(xB[near_matrix_mask])) if np.any(near_matrix_mask) else float("nan")
    if args.mass_mode == "embedding-report" and np.isfinite(far_field_mean_xB) and abs(far_field_mean_xB - xB_far) > 1.0e-5:
        warnings.append("far-field mean xB deviates from xB_far by more than 1e-5 in embedding-report mode")

    rows = _center_axis_rows(
        phi,
        d_nm,
        xB_near,
        xB_before_clip,
        xB,
        xBtot,
        W_dep,
        rho,
        family_idx,
        args.dx_nm,
        center_nm,
        args.mass_mode,
    )
    _write_center_axis_csv(out_dir / "center_axis_profiles.csv", rows)
    if args.write_plots:
        _plot_center_axes(out_dir / "center_axis_profiles.png", rows, float(xB_far))
    if args.write_vtk:
        _write_vtk_scalar(out_dir / "phi_reconstructed.vtk", phi, "phi", args.dx_nm)
        _write_vtk_scalar(out_dir / "xB_reconstructed.vtk", xB, "xB", args.dx_nm)
        _write_vtk_scalar(out_dir / "xBtot_reconstructed.vtk", xBtot, "xB_tot", args.dx_nm)
        _write_vtk_scalar(out_dir / "W_blend.vtk", W_dep, "W_blend", args.dx_nm)
        _write_vtk_scalar(out_dir / "W_dep.vtk", W_dep, "W_dep", args.dx_nm)
        _write_vtk_scalar(out_dir / "rho.vtk", rho, "rho", args.dx_nm)

    actual_compare = _actual_comparison(
        out_dir=out_dir,
        source_phi=source_phi if compare_actual else None,
        source_xb=source_xb if compare_actual else None,
        phi=phi,
        xB=xB,
        xBtot=xBtot,
        W_dep=W_dep,
        rows=rows,
        dx_nm=args.dx_nm,
        center_nm=center_nm,
        xB_far=float(xB_far),
        mass_mode=args.mass_mode,
        xB0_target_nominal=args.xB0_target,
        write_plots=args.write_plots,
    )

    diagnostics: dict[str, Any] = {
        "Geometry": {
            "Nx": args.Nx,
            "Ny": args.Ny,
            "Nz": args.Nz,
            "dx_nm": args.dx_nm,
            "box_size_nm": [float(v) for v in box_size],
            "center_nm": [float(v) for v in center_nm],
            "center_method": center_method,
            "rho_mode": rho_mode,
            "fallback_radius_nm": args.fallback_radius_nm,
            "semi_axes_long_mid_short_nm": [float(v) for v in semiaxes],
            "phi_min": float(np.min(phi)),
            "phi_max": float(np.max(phi)),
            "phi_mean": float(np.mean(phi)),
            "mean_h_phi": float(np.mean(h)),
            "same_as_source_vtk": bool(args.same_as_source_vtk),
            "source_vtk_dimensions": list(source_dims) if source_dims is not None else None,
            "source_vtk_spacing": list(source_vtk_spacing) if source_vtk_spacing is not None else None,
            "source_vtk_origin": list(source_origin),
            "effective_dx_nm": args.dx_nm,
        },
        "ProfileLoader": loader_info,
        "MassTarget": {
            "mass_target_mode": mass_target_mode,
            "xB0_target_nominal": args.xB0_target,
            "xBtot_target_used": xBtot_target_used,
            "xBtot_target_manual": args.xBtot_target_manual,
            **source_mass_stats,
            "target_note": (
                "source-vtk mass target is used to reproduce the total composition of the actual dynamic VTK state."
                if mass_target_mode == "source-vtk"
                else "nominal mass target is used to construct a closed system with prescribed alloy composition."
                if mass_target_mode == "nominal"
                else "manual mass target is used as specified by --xBtot-target-manual."
            ),
        },
        "Blending": {
            "blend_mode": args.blend_mode,
            "signed_distance_convention": "d_nm = 0 at phi~0.5 interface; d_nm > 0 in matrix/outside; d_nm < 0 in precipitate/inside",
            "blend_d1_nm": args.blend_d1_nm,
            "blend_d2_nm": args.blend_d2_nm,
            "blend_rho1": args.blend_rho1,
            "blend_rho2": args.blend_rho2,
            "W_dep_min": float(np.min(W_dep)),
            "W_dep_max": float(np.max(W_dep)),
            "W_dep_mean": float(np.mean(W_dep)),
            "near_field_fraction_W_dep_gt_0p999": float(np.mean(near_field_mask)),
            "far_field_fraction_W_dep_lt_1e_minus_3": float(np.mean(far_field_mask)),
            "far_field_mean_xB": far_field_mean_xB,
            "near_matrix_mean_xB": near_matrix_mean_xB,
        },
        "xB": {
            "xB_rebuild_mode": args.xB_rebuild_mode,
            "alpha_interface": args.alpha_interface,
            "xB_matrix_near": float(xB_matrix_near),
            "xB_matrix_reference": float(xB_matrix_reference),
            "xB_far": float(xB_far),
            "profile_extrapolate": args.profile_extrapolate,
            "xB_local_min": float(np.min(xB_near)),
            "xB_local_max": float(np.max(xB_near)),
            "xB_local_mean": float(np.mean(xB_near)),
            "xB_min_before_clipping": float(np.min(xB_before_clip)),
            "xB_max_before_clipping": float(np.max(xB_before_clip)),
            "xB_mean_before_clipping": float(np.mean(xB_before_clip)),
            "xB_min_after_final_clipping": float(np.min(xB)),
            "xB_max_after_final_clipping": float(np.max(xB)),
            "xB_mean_after_final_clipping": float(np.mean(xB)),
            "lower_clipped_fraction": lower_clip_fraction,
            "upper_clipped_fraction": upper_clip_fraction,
            "xB_min_bound": args.xB_min,
            "xB_max_bound": args.xB_max,
        },
        "Mass": {
            "mass_mode": args.mass_mode,
            "mass_target_mode": mass_target_mode,
            "xB0_target_nominal": args.xB0_target,
            "xBtot_target_used": xBtot_target_used,
            "xB0_target": args.xB0_target,
            "xB_far": float(xB_far),
            "initial_mean_xBtot": float(np.mean((1.0 - h) * xB_before_clip + h)),
            "final_mean_xBtot": float(np.mean(xBtot)),
            "final_mass_error": final_mass_error,
            "mass_error_vs_target": final_mass_error,
            "mass_error_vs_nominal": float(np.mean(xBtot) - args.xB0_target),
            **_mass_summary(xBtot, xBtot_target_used, args.dx_nm, args.xB0_target),
            "mass_correction_history": mass_history,
            "embedding_report_note": "no mass correction applied by design" if args.mass_mode == "embedding-report" else "",
            "far_field_compensation_mask_count": compensation_mask_count,
            "far_field_compensation_mask_fraction": compensation_mask_fraction,
            "W_comp_threshold": args.W_comp_threshold,
            "phi_matrix_threshold": args.phi_matrix_threshold,
            "converged": converged,
        },
        "SourceDynamicStats": source_stats,
        "ActualComparison": actual_compare,
        "Warnings": warnings,
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    _write_txt(out_dir / "diagnostics.txt", diagnostics)

    print(f"output_dir={out_dir}")
    print(f"final_mean_xBtot={float(np.mean(xBtot)):.12e}")
    print(f"final_mass_error={final_mass_error:.12e}")
    print(f"xB_min={float(np.min(xB)):.12e}")
    print(f"xB_max={float(np.max(xB)):.12e}")
    print(f"lower_clipped_fraction={lower_clip_fraction:.12e}")
    print(f"upper_clipped_fraction={upper_clip_fraction:.12e}")
    print(f"warnings_count={len(warnings)}")
    for warning in warnings:
        print(f"WARNING: {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
