#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass, field
from pathlib import Path
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


def h_phi(phi: np.ndarray) -> np.ndarray:
    p = np.clip(phi, 0.0, 1.0)
    return 6.0 * p**5 - 15.0 * p**4 + 10.0 * p**3


def smoothstep01(s: np.ndarray) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return 3.0 * s**2 - 2.0 * s**3


def resolve_path(path_value: str | None, *, config_dir: Path) -> Path | None:
    if not path_value:
        return None
    path = Path(path_value).expanduser()
    if path.is_absolute():
        return path
    from_config = (config_dir / path).resolve()
    if from_config.exists():
        return from_config
    return (Path.cwd() / path).resolve()


def interp_const(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x, y, left=float(y[0]), right=float(y[-1]))


def interp_delta(x: np.ndarray, y: np.ndarray, x_new: np.ndarray, mode: str) -> np.ndarray:
    if mode == "clamp":
        return interp_const(x, y, x_new)
    return np.interp(x_new, x, y, left=0.0, right=0.0)


def read_vtk_with_meta(path: Path) -> tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]:
    data, dims, spacing = _read_legacy_scalar_vtk(path)
    return np.asarray(data, dtype=np.float32), tuple(int(v) for v in dims), tuple(float(v) for v in spacing)


def _write_vtk_header(f, name: str, shape: tuple[int, int, int], dx_nm: float) -> None:
    nx, ny, nz = shape
    f.write("# vtk DataFile Version 3.0\n")
    f.write(f"{name}\n")
    f.write("ASCII\n")
    f.write("DATASET STRUCTURED_POINTS\n")
    f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
    f.write("ORIGIN 0 0 0\n")
    f.write(f"SPACING {dx_nm:.12g} {dx_nm:.12g} {dx_nm:.12g}\n")
    f.write(f"POINT_DATA {nx * ny * nz}\n")
    f.write(f"SCALARS {name} float 1\n")
    f.write("LOOKUP_TABLE default\n")


def _write_flat_values(f, data: np.ndarray) -> None:
    flat = np.asarray(data).ravel(order="C")
    block = 6
    for start in range(0, flat.size, block):
        f.write(" ".join(f"{float(v):.8e}" for v in flat[start : start + block]))
        f.write("\n")


def write_legacy_vtk_scalar(path: Path, data: np.ndarray, name: str, dx_nm: float) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        _write_vtk_header(f, name, data.shape, dx_nm)
        _write_flat_values(f, data)


def write_legacy_vtk_scalar_chunked(
    path: Path,
    shape: tuple[int, int, int],
    name: str,
    dx_nm: float,
    chunk_z: int,
    chunk_fn,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _nx, _ny, nz = shape
    with path.open("w", encoding="ascii") as f:
        _write_vtk_header(f, name, shape, dx_nm)
        for z0 in range(0, nz, chunk_z):
            z1 = min(nz, z0 + chunk_z)
            _write_flat_values(f, chunk_fn(z0, z1))


def latest_step_vtk(source_dir: Path, prefix: str, default_name: str) -> Path | None:
    preferred = source_dir / default_name
    if preferred.exists():
        return preferred
    matches: list[tuple[int, Path]] = []
    for path in source_dir.glob(f"{prefix}_*.vtk"):
        stem = path.stem
        suffix = stem[len(prefix) + 1 :]
        if suffix.isdigit():
            matches.append((int(suffix), path))
    if matches:
        return sorted(matches)[-1][1]
    final = source_dir / f"{prefix}_final.vtk"
    return final if final.exists() else None


def load_profiles(profile_dir: Path) -> tuple[dict[str, dict[str, np.ndarray]], dict[str, Any]]:
    if not profile_dir.exists():
        raise FileNotFoundError(f"profile_dir does not exist: {profile_dir}")
    profile_csv = profile_dir / "faceted_family_profiles.csv"
    if not profile_csv.exists():
        candidates = sorted(p for p in profile_dir.glob("*.csv") if "profile" in p.name.lower())
        if candidates:
            profile_csv = candidates[0]
    if not profile_csv.exists():
        listing = "\n".join(str(p) for p in sorted(profile_dir.iterdir()))
        raise FileNotFoundError(f"Cannot identify faceted profile CSV in {profile_dir}\nFiles:\n{listing}")

    rows = list(csv.DictReader(profile_csv.open(newline="", encoding="utf-8")))
    fields = set(rows[0].keys()) if rows else set()
    distance_field = "u_nm" if "u_nm" in fields else "d_nm" if "d_nm" in fields else None
    required = {"family", "phi_mean", "xB_mean"}
    if not rows or not required.issubset(fields) or distance_field is None:
        fields = list(rows[0].keys()) if rows else []
        raise ValueError(
            f"{profile_csv} missing required fields {sorted(required)} plus u_nm or d_nm; fields={fields}"
        )

    profiles: dict[str, dict[str, np.ndarray]] = {}
    for family in sorted({row["family"] for row in rows}):
        family_rows = [row for row in rows if row["family"] == family]
        regions = {row.get("region", "all") for row in family_rows}
        region = "face" if "face" in regions else sorted(regions)[0]
        pts = [row for row in family_rows if row.get("region", "all") == region]
        pts.sort(key=lambda row: float(row[distance_field]))
        profiles[family] = {
            "u_nm": np.array([float(row[distance_field]) for row in pts], dtype=np.float64),
            "phi_mean": np.array([float(row["phi_mean"]) for row in pts], dtype=np.float64),
            "xB_mean": np.array([float(row["xB_mean"]) for row in pts], dtype=np.float64),
        }

    meta_path = profile_dir / "faceted_family_metadata.json"
    meta: dict[str, Any] = {}
    if meta_path.exists():
        try:
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            meta = {}
    all_u = np.concatenate([p["u_nm"] for p in profiles.values()])
    info = {
        "profile_dir": str(profile_dir),
        "profile_file": str(profile_csv),
        "metadata_file": str(meta_path) if meta_path.exists() else None,
        "fields": list(rows[0].keys()),
        "distance_field_used": distance_field,
        "family_count": len(profiles),
        "families": sorted(profiles),
        "distance_min_nm": float(np.min(all_u)),
        "distance_max_nm": float(np.max(all_u)),
        "profile_metadata": meta,
    }
    return profiles, info


def profile_plateau_reference(profiles: dict[str, dict[str, np.ndarray]]) -> float:
    vals = []
    for prof in profiles.values():
        idx = int(np.argmax(prof["u_nm"]))
        vals.append(float(prof["xB_mean"][idx]))
    if not vals:
        raise ValueError("No profile values available to estimate xB_matrix_reference")
    return float(np.mean(vals))


def axes_from_summary(summary_path: Path | None) -> np.ndarray:
    if summary_path and summary_path.exists():
        info = parse_summary_file(summary_path)
        try:
            axes = np.array(
                [
                    [float(info["long_axis_x"]), float(info["long_axis_y"]), float(info["long_axis_z"])],
                    [float(info["mid_axis_x"]), float(info["mid_axis_y"]), float(info["mid_axis_z"])],
                    [float(info["short_axis_x"]), float(info["short_axis_y"]), float(info["short_axis_z"])],
                ],
                dtype=np.float64,
            )
            norms = np.linalg.norm(axes, axis=1)
            if np.all(norms > 1.0e-12):
                return axes / norms[:, None]
        except KeyError:
            pass
    return np.eye(3, dtype=np.float64)


def trilinear_sample(field: np.ndarray, spacing: tuple[float, float, float], points_xyz: np.ndarray) -> np.ndarray:
    dx, dy, dz = spacing
    nx, ny, nz = field.shape
    gx = np.clip(points_xyz[:, 0] / dx, 0.0, nx - 1.000001)
    gy = np.clip(points_xyz[:, 1] / dy, 0.0, ny - 1.000001)
    gz = np.clip(points_xyz[:, 2] / dz, 0.0, nz - 1.000001)
    x0 = np.floor(gx).astype(np.int64)
    y0 = np.floor(gy).astype(np.int64)
    z0 = np.floor(gz).astype(np.int64)
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


def find_crossing(s: np.ndarray, phi: np.ndarray, threshold: float, positive_side: bool) -> float | None:
    center_idx = int(np.argmin(np.abs(s)))
    indices = range(center_idx, len(s) - 1) if positive_side else range(center_idx, 0, -1)
    for i in indices:
        j = i + 1 if positive_side else i - 1
        pi = phi[i]
        pj = phi[j]
        if (pi - threshold) * (pj - threshold) <= 0.0 and pi != pj:
            t = (threshold - pi) / (pj - pi)
            return float(s[i] + t * (s[j] - s[i]))
    return None


def semiaxes_from_source_phi(
    phi: np.ndarray | None,
    spacing: tuple[float, float, float] | None,
    axes: np.ndarray,
    fallback_radius_nm: float,
) -> tuple[np.ndarray, str]:
    if phi is None or spacing is None:
        return np.array([fallback_radius_nm] * 3, dtype=np.float64), "spherical_fallback"
    nx, ny, nz = phi.shape
    center = np.array([0.5 * nx * spacing[0], 0.5 * ny * spacing[1], 0.5 * nz * spacing[2]], dtype=np.float64)
    half_range = 0.5 * min(nx * spacing[0], ny * spacing[1], nz * spacing[2])
    s_grid = np.arange(-half_range, half_range + 0.05, 0.05)
    semiaxes: list[float] = []
    for axis in axes:
        points = center[None, :] + s_grid[:, None] * axis[None, :]
        vals = trilinear_sample(phi, spacing, points)
        s_neg = find_crossing(s_grid, vals, 0.5, positive_side=False)
        s_pos = find_crossing(s_grid, vals, 0.5, positive_side=True)
        if s_neg is None or s_pos is None:
            return np.array([fallback_radius_nm] * 3, dtype=np.float64), "spherical_fallback"
        semiaxes.append(0.5 * (abs(s_neg) + abs(s_pos)))
    return np.maximum(np.array(semiaxes, dtype=np.float64), 1.0e-6), "ellipsoid_from_source_phi"


def source_stats(phi: np.ndarray | None, xb: np.ndarray | None) -> dict[str, float]:
    if phi is None or xb is None:
        return {}
    hp = h_phi(phi)
    xbtot = (1.0 - hp) * xb + hp
    out = {
        "source_phi_min": float(np.min(phi)),
        "source_phi_max": float(np.max(phi)),
        "source_phi_mean": float(np.mean(phi)),
        "source_xB_min": float(np.min(xb)),
        "source_xB_max": float(np.max(xb)),
        "source_xB_mean": float(np.mean(xb)),
        "source_xBtot_mean": float(np.mean(xbtot)),
        "source_mean_hphi": float(np.mean(hp)),
    }
    masks = {
        "source_xB_mean_phi_lt_0p05": phi < 0.05,
        "source_xB_mean_phi_0p1_0p9": (phi > 0.1) & (phi < 0.9),
        "source_xB_mean_phi_gt_0p9": phi > 0.9,
    }
    for key, mask in masks.items():
        out[key] = float(np.mean(xb[mask])) if np.any(mask) else float("nan")
    return out


def family_index_from_local(local0: np.ndarray, local1: np.ndarray, local2: np.ndarray) -> np.ndarray:
    return (
        (local0 >= 0.0).astype(np.uint8) * 4
        + (local1 >= 0.0).astype(np.uint8) * 2
        + (local2 >= 0.0).astype(np.uint8)
    )


@dataclass
class Nucleus:
    index: int
    name: str
    center_nm: np.ndarray
    profile_dir: Path
    source_dyn_dir: Path | None
    source_box_n: tuple[int, int, int]
    source_dx_nm: float
    blend_inner_margin_nm: float
    blend_outer_margin_nm: float
    alpha_interface: float
    xB_rebuild_mode: str
    profile_extrapolate: str
    fallback_radius_nm: float
    profiles: dict[str, dict[str, np.ndarray]]
    profile_info: dict[str, Any]
    axes: np.ndarray
    semiaxes_nm: np.ndarray
    semiaxes_source: str
    xB_matrix_reference: float
    xB_matrix_near: float
    source_stats: dict[str, float] = field(default_factory=dict)
    accum: dict[str, float] = field(default_factory=dict)


def load_nucleus(raw: dict[str, Any], index: int, *, config_dir: Path, xB_far: float) -> Nucleus | None:
    if raw.get("enabled", True) is False:
        return None
    name = str(raw.get("name") or f"nucleus_{index}")
    profile_dir = resolve_path(raw.get("profile_dir"), config_dir=config_dir)
    source_dyn_dir = resolve_path(raw.get("source_dyn_dir"), config_dir=config_dir)
    if profile_dir is None:
        raise ValueError(f"nucleus {name}: profile_dir is required")
    if not profile_dir.exists():
        raise FileNotFoundError(
            f"nucleus {name}: profile_dir not found: {profile_dir}. "
            "Run extract_faceted_rebuild_profiles.py first."
        )
    profiles, profile_info = load_profiles(profile_dir)

    source_phi = None
    source_xb = None
    source_spacing = None
    source_box_n: tuple[int, int, int] | None = None
    summary_path = source_dyn_dir / "summary.txt" if source_dyn_dir else None
    if source_dyn_dir:
        phi_path = latest_step_vtk(source_dyn_dir, "phi", "phi_1500.vtk")
        xb_path = latest_step_vtk(source_dyn_dir, "xB", "xB_1500.vtk")
        if phi_path and phi_path.exists():
            source_phi, source_box_n, source_spacing = read_vtk_with_meta(phi_path)
        if xb_path and xb_path.exists():
            source_xb, _, _ = read_vtk_with_meta(xb_path)

    if raw.get("source_box_N"):
        source_box_n = tuple(int(v) for v in raw["source_box_N"])
    if source_box_n is None:
        raise ValueError(f"nucleus {name}: source_box_N missing and source VTK dimensions unavailable")

    source_dx_nm = float(raw.get("source_dx_nm") or (source_spacing[0] if source_spacing else 0.1))
    center_nm = np.array(raw["center_nm"], dtype=np.float64)
    axes = axes_from_summary(summary_path)
    fallback_radius_nm = float(raw.get("fallback_radius_nm", 7.0))
    semiaxes, semiaxes_source = semiaxes_from_source_phi(
        source_phi,
        (source_dx_nm, source_dx_nm, source_dx_nm) if source_phi is not None else None,
        axes,
        fallback_radius_nm,
    )
    stats = source_stats(source_phi, source_xb)
    xB_matrix_reference = float(raw.get("xB_matrix_reference", profile_plateau_reference(profiles)))
    if raw.get("xB_matrix_near") is not None:
        xB_matrix_near = float(raw["xB_matrix_near"])
    elif "source_xB_mean_phi_lt_0p05" in stats and math.isfinite(stats["source_xB_mean_phi_lt_0p05"]):
        xB_matrix_near = float(stats["source_xB_mean_phi_lt_0p05"])
    else:
        xB_matrix_near = xB_far

    accum = {
        "W_box_nonzero_points": 0.0,
        "phi_gt_0p5_points": 0.0,
        "phi_gt_0p05_points": 0.0,
        "max_phi_contribution": 0.0,
        "near_matrix_xB_sum": 0.0,
        "near_matrix_xB_count": 0.0,
        "interface_xB_sum": 0.0,
        "interface_xB_count": 0.0,
    }
    return Nucleus(
        index=index,
        name=name,
        center_nm=center_nm,
        profile_dir=profile_dir,
        source_dyn_dir=source_dyn_dir,
        source_box_n=source_box_n,
        source_dx_nm=source_dx_nm,
        blend_inner_margin_nm=float(raw.get("blend_inner_margin_nm", 3.0)),
        blend_outer_margin_nm=float(raw.get("blend_outer_margin_nm", 0.0)),
        alpha_interface=float(raw.get("alpha_interface", 0.25)),
        xB_rebuild_mode=str(raw.get("xB_rebuild_mode", "delta-interface")),
        profile_extrapolate=str(raw.get("profile_extrapolate", "far")),
        fallback_radius_nm=fallback_radius_nm,
        profiles=profiles,
        profile_info=profile_info,
        axes=axes,
        semiaxes_nm=semiaxes,
        semiaxes_source=semiaxes_source,
        xB_matrix_reference=xB_matrix_reference,
        xB_matrix_near=xB_matrix_near,
        source_stats=stats,
        accum=accum,
    )


def memmap_array(path: Path, shape: tuple[int, int, int], dtype: np.dtype | type, fill: float | int) -> np.memmap:
    arr = np.memmap(path, dtype=dtype, mode="w+", shape=shape)
    arr[:] = fill
    arr.flush()
    return arr


def chunk_geometry(
    *,
    nucleus: Nucleus,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    qx = x[:, None, None] - nucleus.center_nm[0]
    qy = y[None, :, None] - nucleus.center_nm[1]
    qz = z[None, None, :] - nucleus.center_nm[2]

    axes = nucleus.axes
    l0 = axes[0, 0] * qx + axes[0, 1] * qy + axes[0, 2] * qz
    l1 = axes[1, 0] * qx + axes[1, 1] * qy + axes[1, 2] * qz
    l2 = axes[2, 0] * qx + axes[2, 1] * qy + axes[2, 2] * qz
    local0 = np.asarray(l0, dtype=np.float32)
    local1 = np.asarray(l1, dtype=np.float32)
    local2 = np.asarray(l2, dtype=np.float32)

    scaled0 = local0 / np.float32(nucleus.semiaxes_nm[0])
    scaled1 = local1 / np.float32(nucleus.semiaxes_nm[1])
    scaled2 = local2 / np.float32(nucleus.semiaxes_nm[2])
    rho = np.sqrt(scaled0 * scaled0 + scaled1 * scaled1 + scaled2 * scaled2).astype(np.float32, copy=False)
    r = np.sqrt(local0 * local0 + local1 * local1 + local2 * local2).astype(np.float32, copy=False)
    mean_radius = np.float32(np.mean(nucleus.semiaxes_nm))
    boundary_radius = np.where(rho > 1.0e-8, r / np.maximum(rho, 1.0e-8), mean_radius)
    d_nm = (rho - np.float32(1.0)) * boundary_radius
    family_idx = family_index_from_local(local0, local1, local2)
    return rho, d_nm.astype(np.float32), family_idx, (qx, qy, qz)


def rebuild_profile_chunk(nucleus: Nucleus, d_nm: np.ndarray, family_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    phi = np.empty(d_nm.shape, dtype=np.float32)
    delta = np.empty(d_nm.shape, dtype=np.float32)
    fallback = next(iter(nucleus.profiles.values()))
    for idx, label in enumerate(FAMILY_LABELS):
        prof = nucleus.profiles.get(label, fallback)
        mask = family_idx == idx
        if not np.any(mask):
            continue
        phi[mask] = interp_const(prof["u_nm"], prof["phi_mean"], d_nm[mask]).astype(np.float32)
        delta[mask] = interp_delta(
            prof["u_nm"],
            prof["xB_mean"] - nucleus.xB_matrix_reference,
            d_nm[mask],
            nucleus.profile_extrapolate,
        ).astype(np.float32)
    phi = np.clip(phi, 0.0, 1.0).astype(np.float32)
    if nucleus.xB_rebuild_mode == "constant-matrix":
        xB_local = np.full(d_nm.shape, nucleus.xB_matrix_near, dtype=np.float32)
    else:
        xB_local = (nucleus.xB_matrix_near + nucleus.alpha_interface * delta).astype(np.float32)
    return phi, xB_local


def source_box_window(nucleus: Nucleus, qx: np.ndarray, qy: np.ndarray, qz: np.ndarray) -> np.ndarray:
    sx, sy, sz = nucleus.source_box_n
    hx = np.float32(0.5 * sx * nucleus.source_dx_nm)
    hy = np.float32(0.5 * sy * nucleus.source_dx_nm)
    hz = np.float32(0.5 * sz * nucleus.source_dx_nm)
    margin = max(float(nucleus.blend_inner_margin_nm), 1.0e-12)
    m = np.minimum(np.minimum(hx - np.abs(qx), hy - np.abs(qy)), hz - np.abs(qz))
    w = np.zeros(np.broadcast_shapes(qx.shape, qy.shape, qz.shape), dtype=np.float32)
    inside = m > 0.0
    if np.any(inside):
        s = np.clip(m[inside] / margin, 0.0, 1.0)
        w[inside] = smoothstep01(s).astype(np.float32)
    return w


def compute_mean_xbtot(
    phi_total: np.memmap,
    xB: np.memmap,
    *,
    chunk_z: int,
) -> tuple[float, float, float, float]:
    nx, ny, nz = phi_total.shape
    total = 0.0
    xb_sum = 0.0
    phi_sum = 0.0
    h_sum = 0.0
    n = nx * ny * nz
    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        phi = np.asarray(phi_total[:, :, z0:z1], dtype=np.float32)
        xb = np.asarray(xB[:, :, z0:z1], dtype=np.float32)
        hp = h_phi(phi)
        total += float(np.sum((1.0 - hp) * xb + hp, dtype=np.float64))
        xb_sum += float(np.sum(xb, dtype=np.float64))
        phi_sum += float(np.sum(phi, dtype=np.float64))
        h_sum += float(np.sum(hp, dtype=np.float64))
    return total / n, xb_sum / n, phi_sum / n, h_sum / n


def mass_compensate_far_field(
    phi_total: np.memmap,
    xB: np.memmap,
    W_box_max: np.memmap,
    *,
    xBtot_target: float,
    xB_min: float,
    xB_max: float,
    W_comp_threshold: float,
    phi_matrix_threshold: float,
    iters: int,
    tol: float,
    chunk_z: int,
) -> tuple[list[dict[str, float]], float, float, list[str]]:
    warnings: list[str] = []
    nx, ny, nz = phi_total.shape
    n = nx * ny * nz
    correction_total = 0.0
    history: list[dict[str, float]] = []
    for iteration in range(iters):
        mean_xbtot, _mean_xb, _mean_phi, _mean_h = compute_mean_xbtot(phi_total, xB, chunk_z=chunk_z)
        mass_error = xBtot_target - mean_xbtot
        denom_sum = 0.0
        far_count = 0
        for z0 in range(0, nz, chunk_z):
            z1 = min(nz, z0 + chunk_z)
            phi = np.asarray(phi_total[:, :, z0:z1], dtype=np.float32)
            wmax = np.asarray(W_box_max[:, :, z0:z1], dtype=np.float32)
            far_mask = (phi < phi_matrix_threshold) & (wmax < W_comp_threshold)
            if np.any(far_mask):
                hp = h_phi(phi)
                denom_sum += float(np.sum((1.0 - hp)[far_mask], dtype=np.float64))
                far_count += int(np.count_nonzero(far_mask))
        denom = denom_sum / n
        if denom <= 1.0e-20:
            warnings.append("far_mask is empty or denom too small; cannot compensate mass in far-field matrix.")
            break
        c_far = mass_error / denom
        correction_total += c_far
        lower_clip = 0
        upper_clip = 0
        for z0 in range(0, nz, chunk_z):
            z1 = min(nz, z0 + chunk_z)
            phi = np.asarray(phi_total[:, :, z0:z1], dtype=np.float32)
            wmax = np.asarray(W_box_max[:, :, z0:z1], dtype=np.float32)
            far_mask = (phi < phi_matrix_threshold) & (wmax < W_comp_threshold)
            xb = np.array(xB[:, :, z0:z1], dtype=np.float32, copy=True)
            xb[far_mask] += np.float32(c_far)
            lower_clip += int(np.count_nonzero(xb < xB_min))
            upper_clip += int(np.count_nonzero(xb > xB_max))
            np.clip(xb, xB_min, xB_max, out=xb)
            xB[:, :, z0:z1] = xb
        final_mean, _mean_xb2, _mean_phi2, _mean_h2 = compute_mean_xbtot(phi_total, xB, chunk_z=chunk_z)
        final_error = final_mean - xBtot_target
        rec = {
            "iter": float(iteration),
            "mean_xBtot_before": float(mean_xbtot),
            "mass_error_before": float(mass_error),
            "C_far": float(c_far),
            "C_far_total": float(correction_total),
            "far_mask_fraction": float(far_count / n),
            "lower_clipped_fraction": float(lower_clip / n),
            "upper_clipped_fraction": float(upper_clip / n),
            "mean_xBtot_after": float(final_mean),
            "mass_error_after": float(final_error),
        }
        history.append(rec)
        if abs(c_far) > 0.01:
            warnings.append(f"strong warning: |C_far|={abs(c_far):.4g} > 0.01; compensation is physically questionable.")
        elif abs(c_far) > 0.003:
            warnings.append(f"warning: |C_far|={abs(c_far):.4g} > 0.003; box may be too small or too many nuclei inserted.")
        if rec["lower_clipped_fraction"] > 1.0e-4 or rec["upper_clipped_fraction"] > 1.0e-4:
            warnings.append(
                f"clipping fraction exceeded 1e-4 at iter {iteration}: "
                f"lower={rec['lower_clipped_fraction']:.3e}, upper={rec['upper_clipped_fraction']:.3e}"
            )
        if abs(final_error) <= tol:
            break
    return history, correction_total, history[-1]["far_mask_fraction"] if history else 0.0, warnings


def write_summary_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fields = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_diagnostics_txt(path: Path, diagnostics: dict[str, Any]) -> None:
    lines: list[str] = []
    for section, values in diagnostics.items():
        lines.append(f"[{section}]")
        if isinstance(values, dict):
            for key, value in values.items():
                lines.append(f"{key}: {value}")
        else:
            lines.append(str(values))
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def collect_global_stats(
    phi_total: np.memmap,
    xB: np.memmap,
    W_box_max: np.memmap,
    active_count: np.memmap,
    W_active_count: np.memmap,
    *,
    xBtot_target: float,
    xB_min: float,
    xB_max: float,
    W_comp_threshold: float,
    phi_matrix_threshold: float,
    chunk_z: int,
) -> dict[str, float]:
    nx, ny, nz = phi_total.shape
    n = nx * ny * nz
    mean_xbtot, mean_xb, mean_phi, mean_h = compute_mean_xbtot(phi_total, xB, chunk_z=chunk_z)
    phi_min = math.inf
    phi_max = -math.inf
    xb_min = math.inf
    xb_max = -math.inf
    far_count = 0
    active_max = 0
    phi_overlap_count = 0
    w_overlap_count = 0
    lower_count = 0
    upper_count = 0
    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        phi = np.asarray(phi_total[:, :, z0:z1])
        xb = np.asarray(xB[:, :, z0:z1])
        wmax = np.asarray(W_box_max[:, :, z0:z1])
        active = np.asarray(active_count[:, :, z0:z1])
        w_active = np.asarray(W_active_count[:, :, z0:z1])
        phi_min = min(phi_min, float(np.min(phi)))
        phi_max = max(phi_max, float(np.max(phi)))
        xb_min = min(xb_min, float(np.min(xb)))
        xb_max = max(xb_max, float(np.max(xb)))
        far_count += int(np.count_nonzero((phi < phi_matrix_threshold) & (wmax < W_comp_threshold)))
        active_max = max(active_max, int(np.max(active)))
        phi_overlap_count += int(np.count_nonzero(active > 1))
        w_overlap_count += int(np.count_nonzero(w_active > 1))
        lower_count += int(np.count_nonzero(xb <= xB_min))
        upper_count += int(np.count_nonzero(xb >= xB_max))
    return {
        "mean_phi": float(mean_phi),
        "mean_h_phi": float(mean_h),
        "phi_min": phi_min,
        "phi_max": phi_max,
        "mean_xB": float(mean_xb),
        "xB_min": xb_min,
        "xB_max": xb_max,
        "mean_xBtot": float(mean_xbtot),
        "mass_error_vs_target": float(mean_xbtot - xBtot_target),
        "far_mask_fraction": float(far_count / n),
        "active_nucleus_count_max": float(active_max),
        "overlap_fraction_phi_i_gt_0p05": float(phi_overlap_count / n),
        "overlap_fraction_W_box_gt_0p01": float(w_overlap_count / n),
        "lower_clipped_fraction": float(lower_count / n),
        "upper_clipped_fraction": float(upper_count / n),
    }


def write_center_profiles(
    out_csv: Path,
    out_png: Path,
    phi_total: np.memmap,
    xB: np.memmap,
    *,
    nuclei: list[Nucleus],
    dx_nm: float,
    chunk_z: int,
) -> None:
    nx, ny, nz = phi_total.shape
    rows: list[dict[str, Any]] = []

    def add_axis(scope: str, center: np.ndarray, axis: str) -> None:
        cx = int(np.clip(round(center[0] / dx_nm), 0, nx - 1))
        cy = int(np.clip(round(center[1] / dx_nm), 0, ny - 1))
        cz = int(np.clip(round(center[2] / dx_nm), 0, nz - 1))
        if axis == "x":
            coords = (np.arange(nx) * dx_nm) - center[0]
            p = np.asarray(phi_total[:, cy, cz])
            xb = np.asarray(xB[:, cy, cz])
        elif axis == "y":
            coords = (np.arange(ny) * dx_nm) - center[1]
            p = np.asarray(phi_total[cx, :, cz])
            xb = np.asarray(xB[cx, :, cz])
        else:
            coords = (np.arange(nz) * dx_nm) - center[2]
            p = np.asarray(phi_total[cx, cy, :])
            xb = np.asarray(xB[cx, cy, :])
        hp = h_phi(p)
        xt = (1.0 - hp) * xb + hp
        for i in range(coords.size):
            rows.append(
                {
                    "scope": scope,
                    "axis": axis,
                    "index": i,
                    "coordinate_nm": float(coords[i]),
                    "phi": float(p[i]),
                    "xB": float(xb[i]),
                    "xBtot": float(xt[i]),
                }
            )

    box_center = np.array([0.5 * nx * dx_nm, 0.5 * ny * dx_nm, 0.5 * nz * dx_nm], dtype=np.float64)
    for axis in ("x", "y", "z"):
        add_axis("global_box_center", box_center, axis)
    for nucleus in nuclei:
        for axis in ("x", "y", "z"):
            add_axis(nucleus.name, nucleus.center_nm, axis)

    write_summary_csv(out_csv, rows)

    fig, axs = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for scope in ["global_box_center"] + [n.name for n in nuclei]:
        if scope != "global_box_center" and len(nuclei) > 8:
            continue
        sub = [row for row in rows if row["scope"] == scope]
        for ai, axis in enumerate(("x", "y", "z")):
            axis_rows = [row for row in sub if row["axis"] == axis]
            coord = np.array([row["coordinate_nm"] for row in axis_rows], dtype=float)
            phi = np.array([row["phi"] for row in axis_rows], dtype=float)
            xb = np.array([row["xB"] for row in axis_rows], dtype=float)
            axs[ai, 0].plot(coord, phi, lw=1.2, label=scope)
            axs[ai, 1].plot(coord, xb, lw=1.2, label=scope)
            axs[ai, 0].set_title(f"{axis}-axis phi")
            axs[ai, 1].set_title(f"{axis}-axis xB")
            axs[ai, 0].grid(alpha=0.25)
            axs[ai, 1].grid(alpha=0.25)
    axs[0, 0].legend(fontsize=7, frameon=False)
    axs[0, 1].legend(fontsize=7, frameon=False)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Embed multiple reconstructed dynamic nuclei into a larger test box.")
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--chunk-z", type=int, default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default=None)
    parser.add_argument("--write-vtk", action="store_true", default=None)
    parser.add_argument("--write-plots", action="store_true", default=None)
    parser.add_argument("--xB-combine-mode", choices=("min", "weighted-average"), default=None)
    args = parser.parse_args()

    config_path = args.config.expanduser().resolve()
    config_dir = config_path.parent
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    global_cfg = cfg.get("global", {})
    out_dir = args.out_dir.expanduser().resolve() if args.out_dir else resolve_path(global_cfg.get("out_dir"), config_dir=config_dir)
    if out_dir is None:
        raise ValueError("--out-dir or global.out_dir is required")
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir = out_dir / "_work_arrays"
    work_dir.mkdir(parents=True, exist_ok=True)

    nx = int(global_cfg["Nx"])
    ny = int(global_cfg["Ny"])
    nz = int(global_cfg["Nz"])
    dx_nm = float(global_cfg.get("dx_nm", 0.1))
    xB_far = float(global_cfg.get("xB_far", 0.030))
    xB0_nominal = float(global_cfg.get("xB0_target_nominal", xB_far))
    xBtot_target_mode = str(global_cfg.get("xBtot_target_mode", "nominal"))
    if xBtot_target_mode == "manual":
        xBtot_target = float(global_cfg["xBtot_target_manual"])
    elif xBtot_target_mode == "nominal":
        xBtot_target = xB0_nominal
    else:
        raise ValueError(f"Unsupported xBtot_target_mode for this script version: {xBtot_target_mode}")
    xB_min = float(global_cfg.get("xB_min", 1.0e-8))
    xB_max = float(global_cfg.get("xB_max", 0.035))
    mass_mode = str(global_cfg.get("mass_mode", "far-field-compensate"))
    W_comp_threshold = float(global_cfg.get("W_comp_threshold", 1.0e-3))
    phi_matrix_threshold = float(global_cfg.get("phi_matrix_threshold", 0.05))
    mass_iters = int(global_cfg.get("mass_correction_iters", 10))
    mass_tol = float(global_cfg.get("mass_tol", 1.0e-9))
    write_vtk = bool(global_cfg.get("write_vtk", False)) if args.write_vtk is None else bool(args.write_vtk)
    write_plots = bool(global_cfg.get("write_plots", True)) if args.write_plots is None else bool(args.write_plots)
    dtype = np.dtype(args.dtype or global_cfg.get("dtype", "float32"))
    xB_combine_mode = args.xB_combine_mode or str(global_cfg.get("xB_combine_mode", "min"))
    if xB_combine_mode not in {"min", "weighted-average"}:
        raise ValueError(f"Unsupported xB_combine_mode: {xB_combine_mode}")
    chunk_z = args.chunk_z or int(global_cfg.get("chunk_z") or (32 if nx * ny * nz >= 256**3 else nz))
    chunk_z = max(1, min(chunk_z, nz))

    nuclei = [
        nuc
        for idx, raw in enumerate(cfg.get("nuclei", []))
        if (nuc := load_nucleus(raw, idx, config_dir=config_dir, xB_far=xB_far)) is not None
    ]
    if not nuclei:
        raise ValueError("No enabled nuclei in config.")

    shape = (nx, ny, nz)
    phi_total = memmap_array(work_dir / "phi_total.dat", shape, dtype, 0.0)
    W_box_max = memmap_array(work_dir / "W_box_max.dat", shape, dtype, 0.0)
    active_count = memmap_array(work_dir / "active_nucleus_count.dat", shape, np.uint16, 0)
    W_active_count = memmap_array(work_dir / "W_active_count.dat", shape, np.uint16, 0)
    if xB_combine_mode == "min":
        xB = memmap_array(work_dir / "xB.dat", shape, dtype, xB_far)
        weight_sum = None
        weighted_sum = None
    else:
        xB = memmap_array(work_dir / "xB.dat", shape, dtype, xB_far)
        weight_sum = memmap_array(work_dir / "weight_sum.dat", shape, dtype, 1.0)
        weighted_sum = memmap_array(work_dir / "weighted_sum.dat", shape, dtype, xB_far)

    x_coords = np.arange(nx, dtype=np.float32) * np.float32(dx_nm)
    y_coords = np.arange(ny, dtype=np.float32) * np.float32(dx_nm)

    print(f"[info] embedding {len(nuclei)} nuclei into {nx}x{ny}x{nz}, chunk_z={chunk_z}, combine={xB_combine_mode}")
    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        z_coords = np.arange(z0, z1, dtype=np.float32) * np.float32(dx_nm)
        phi_chunk = np.array(phi_total[:, :, z0:z1], dtype=np.float32, copy=True)
        wmax_chunk = np.array(W_box_max[:, :, z0:z1], dtype=np.float32, copy=True)
        active_chunk = np.array(active_count[:, :, z0:z1], dtype=np.uint16, copy=True)
        w_active_chunk = np.array(W_active_count[:, :, z0:z1], dtype=np.uint16, copy=True)
        if xB_combine_mode == "min":
            xb_chunk = np.array(xB[:, :, z0:z1], dtype=np.float32, copy=True)
        else:
            wsum_chunk = np.array(weight_sum[:, :, z0:z1], dtype=np.float32, copy=True)  # type: ignore[index]
            xsum_chunk = np.array(weighted_sum[:, :, z0:z1], dtype=np.float32, copy=True)  # type: ignore[index]

        for nucleus in nuclei:
            _rho, d_nm, family_idx, q = chunk_geometry(nucleus=nucleus, x=x_coords, y=y_coords, z=z_coords)
            phi_i, xB_local = rebuild_profile_chunk(nucleus, d_nm, family_idx)
            W_box = source_box_window(nucleus, *q)
            xB_candidate = (xB_far + W_box * (xB_local - xB_far)).astype(np.float32)
            if xB_combine_mode == "min":
                xb_chunk = np.minimum(xb_chunk, xB_candidate)
            else:
                wsum_chunk += W_box
                xsum_chunk += W_box * xB_candidate
            phi_chunk = np.maximum(phi_chunk, phi_i)
            wmax_chunk = np.maximum(wmax_chunk, W_box)
            active_chunk += (phi_i > 0.05).astype(np.uint16)
            w_active_chunk += (W_box > 0.01).astype(np.uint16)

            nucleus.accum["W_box_nonzero_points"] += float(np.count_nonzero(W_box > 1.0e-6))
            nucleus.accum["phi_gt_0p5_points"] += float(np.count_nonzero(phi_i > 0.5))
            nucleus.accum["phi_gt_0p05_points"] += float(np.count_nonzero(phi_i > 0.05))
            nucleus.accum["max_phi_contribution"] = max(nucleus.accum["max_phi_contribution"], float(np.max(phi_i)))
            near_mask = (W_box > 0.999) & (phi_i < 0.05)
            interface_mask = (W_box > 0.999) & (phi_i > 0.1) & (phi_i < 0.9)
            if np.any(near_mask):
                nucleus.accum["near_matrix_xB_sum"] += float(np.sum(xB_candidate[near_mask], dtype=np.float64))
                nucleus.accum["near_matrix_xB_count"] += float(np.count_nonzero(near_mask))
            if np.any(interface_mask):
                nucleus.accum["interface_xB_sum"] += float(np.sum(xB_candidate[interface_mask], dtype=np.float64))
                nucleus.accum["interface_xB_count"] += float(np.count_nonzero(interface_mask))

        if xB_combine_mode == "weighted-average":
            xb_chunk = xsum_chunk / np.maximum(wsum_chunk, 1.0e-12)
            weight_sum[:, :, z0:z1] = wsum_chunk  # type: ignore[index]
            weighted_sum[:, :, z0:z1] = xsum_chunk  # type: ignore[index]
        np.clip(xb_chunk, xB_min, xB_max, out=xb_chunk)
        phi_total[:, :, z0:z1] = phi_chunk.astype(dtype, copy=False)
        xB[:, :, z0:z1] = xb_chunk.astype(dtype, copy=False)
        W_box_max[:, :, z0:z1] = wmax_chunk.astype(dtype, copy=False)
        active_count[:, :, z0:z1] = active_chunk
        W_active_count[:, :, z0:z1] = w_active_chunk

    phi_total.flush()
    xB.flush()
    W_box_max.flush()
    active_count.flush()
    W_active_count.flush()

    stats_before = collect_global_stats(
        phi_total,
        xB,
        W_box_max,
        active_count,
        W_active_count,
        xBtot_target=xBtot_target,
        xB_min=xB_min,
        xB_max=xB_max,
        W_comp_threshold=W_comp_threshold,
        phi_matrix_threshold=phi_matrix_threshold,
        chunk_z=chunk_z,
    )
    warnings: list[str] = []
    if stats_before["overlap_fraction_phi_i_gt_0p05"] > 0.0:
        warnings.append(
            f"nucleus overlap detected: phi_i>0.05 overlap_fraction={stats_before['overlap_fraction_phi_i_gt_0p05']:.3e}"
        )

    correction_history: list[dict[str, float]] = []
    correction_total = 0.0
    if mass_mode == "far-field-compensate":
        correction_history, correction_total, _far_fraction, mass_warnings = mass_compensate_far_field(
            phi_total,
            xB,
            W_box_max,
            xBtot_target=xBtot_target,
            xB_min=xB_min,
            xB_max=xB_max,
            W_comp_threshold=W_comp_threshold,
            phi_matrix_threshold=phi_matrix_threshold,
            iters=mass_iters,
            tol=mass_tol,
            chunk_z=chunk_z,
        )
        warnings.extend(mass_warnings)
    elif mass_mode != "embedding-report":
        raise ValueError(f"Unsupported mass_mode: {mass_mode}. Use embedding-report or far-field-compensate.")

    stats_after = collect_global_stats(
        phi_total,
        xB,
        W_box_max,
        active_count,
        W_active_count,
        xBtot_target=xBtot_target,
        xB_min=xB_min,
        xB_max=xB_max,
        W_comp_threshold=W_comp_threshold,
        phi_matrix_threshold=phi_matrix_threshold,
        chunk_z=chunk_z,
    )
    if abs(stats_after["mass_error_vs_target"]) > mass_tol and mass_mode == "far-field-compensate":
        warnings.append(
            f"final mass error {stats_after['mass_error_vs_target']:.6e} exceeds mass_tol={mass_tol:.3e}"
        )

    nuclei_rows: list[dict[str, Any]] = []
    total_points = float(nx * ny * nz)
    for nucleus in nuclei:
        near_count = nucleus.accum["near_matrix_xB_count"]
        interface_count = nucleus.accum["interface_xB_count"]
        row: dict[str, Any] = {
            "name": nucleus.name,
            "center_nm": " ".join(f"{v:.6g}" for v in nucleus.center_nm),
            "source_dyn_dir": str(nucleus.source_dyn_dir) if nucleus.source_dyn_dir else "",
            "profile_dir": str(nucleus.profile_dir),
            "source_box_size_nm": " ".join(f"{nucleus.source_box_n[i] * nucleus.source_dx_nm:.6g}" for i in range(3)),
            "source_dx_nm": nucleus.source_dx_nm,
            "semiaxes_nm": " ".join(f"{v:.6g}" for v in nucleus.semiaxes_nm),
            "semiaxes_source": nucleus.semiaxes_source,
            "xB_matrix_near": nucleus.xB_matrix_near,
            "xB_matrix_reference": nucleus.xB_matrix_reference,
            "alpha_interface": nucleus.alpha_interface,
            "source_actual_mean_xBtot": nucleus.source_stats.get("source_xBtot_mean", ""),
            "source_final_equiv_radius_est_nm": (3.0 * nucleus.accum["phi_gt_0p5_points"] * dx_nm**3 / (4.0 * math.pi)) ** (1.0 / 3.0),
            "local_matrix_xB_estimate": nucleus.accum["near_matrix_xB_sum"] / near_count if near_count else float("nan"),
            "local_interface_xB_estimate": nucleus.accum["interface_xB_sum"] / interface_count if interface_count else float("nan"),
            "W_box_nonzero_fraction": nucleus.accum["W_box_nonzero_points"] / total_points,
            "near_profile_mean_after_embedding": nucleus.accum["near_matrix_xB_sum"] / near_count if near_count else float("nan"),
            "near_profile_mean_after_compensation": nucleus.accum["near_matrix_xB_sum"] / near_count if near_count else float("nan"),
            "max_phi_contribution": nucleus.accum["max_phi_contribution"],
            "estimated_precipitate_volume_contribution_nm3": nucleus.accum["phi_gt_0p5_points"] * dx_nm**3,
            "profile_file": nucleus.profile_info.get("profile_file", ""),
        }
        nuclei_rows.append(row)

    write_summary_csv(out_dir / "nuclei_summary.csv", nuclei_rows)

    diagnostics = {
        "Global": {
            "Nx": nx,
            "Ny": ny,
            "Nz": nz,
            "dx_nm": dx_nm,
            "box_size_nm": [nx * dx_nm, ny * dx_nm, nz * dx_nm],
            "number_of_nuclei": len(nuclei),
            "xB_far_requested": xB_far,
            "xB0_target_nominal": xB0_nominal,
            "xBtot_target_mode": xBtot_target_mode,
            "xBtot_target_used": xBtot_target,
            "mass_mode": mass_mode,
            "xB_combine_mode": xB_combine_mode,
            "dtype": str(dtype),
            "chunk_z": chunk_z,
        },
        "Before compensation": stats_before,
        "After compensation": {
            **stats_after,
            "final_mass_error": stats_after["mass_error_vs_target"],
            "C_far_per_iteration": [r["C_far"] for r in correction_history],
            "final_C_far_total": correction_total,
            "xB_far_after_mass_compensation": xB_far + correction_total,
            "mass_converged": abs(stats_after["mass_error_vs_target"]) <= mass_tol,
        },
        "Overlap": {
            "active_nucleus_count_max": stats_after["active_nucleus_count_max"],
            "overlap_fraction_phi_i_gt_0p05": stats_after["overlap_fraction_phi_i_gt_0p05"],
            "overlap_fraction_W_box_gt_0p01": stats_after["overlap_fraction_W_box_gt_0p01"],
        },
        "Mass correction history": correction_history,
        "Warnings": warnings,
        "Per nucleus": nuclei_rows,
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    write_diagnostics_txt(out_dir / "diagnostics.txt", diagnostics)

    if write_plots:
        write_center_profiles(
            out_dir / "center_axis_profiles.csv",
            out_dir / "center_axis_profiles.png",
            phi_total,
            xB,
            nuclei=nuclei,
            dx_nm=dx_nm,
            chunk_z=chunk_z,
        )

    if write_vtk:
        write_legacy_vtk_scalar(out_dir / "phi_embedded.vtk", phi_total, "phi_embedded", dx_nm)
        write_legacy_vtk_scalar(out_dir / "xB_embedded.vtk", xB, "xB_embedded", dx_nm)
        write_legacy_vtk_scalar_chunked(
            out_dir / "xBtot_embedded.vtk",
            shape,
            "xBtot_embedded",
            dx_nm,
            chunk_z,
            lambda z0, z1: (
                lambda phi, xb: ((1.0 - h_phi(phi)) * xb + h_phi(phi)).astype(np.float32)
            )(np.asarray(phi_total[:, :, z0:z1], dtype=np.float32), np.asarray(xB[:, :, z0:z1], dtype=np.float32)),
        )
        write_legacy_vtk_scalar_chunked(
            out_dir / "far_mask.vtk",
            shape,
            "far_mask",
            dx_nm,
            chunk_z,
            lambda z0, z1: ((np.asarray(phi_total[:, :, z0:z1]) < phi_matrix_threshold)
                            & (np.asarray(W_box_max[:, :, z0:z1]) < W_comp_threshold)).astype(np.float32),
        )
        write_legacy_vtk_scalar(out_dir / "active_nucleus_count.vtk", active_count, "active_nucleus_count", dx_nm)
        write_legacy_vtk_scalar(out_dir / "W_box_max.vtk", W_box_max, "W_box_max", dx_nm)

    print(f"out_dir={out_dir}")
    print(f"nuclei={len(nuclei)}")
    print(f"mean_xBtot_before={stats_before['mean_xBtot']:.10e}")
    print(f"mean_xBtot_after={stats_after['mean_xBtot']:.10e}")
    print(f"mass_error_after={stats_after['mass_error_vs_target']:.10e}")
    print(f"C_far_total={correction_total:.10e}")
    print(f"far_mask_fraction={stats_after['far_mask_fraction']:.6e}")
    print(f"warnings={len(warnings)}")
    for warning in warnings:
        print(f"[warn] {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
