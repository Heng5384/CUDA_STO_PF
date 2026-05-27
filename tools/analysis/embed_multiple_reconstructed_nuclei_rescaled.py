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

from tools.analysis.embed_multiple_reconstructed_nuclei import (
    FAMILY_LABELS,
    axes_from_summary,
    compute_mean_xbtot,
    family_index_from_local,
    h_phi,
    interp_const,
    interp_delta,
    latest_step_vtk,
    load_profiles,
    memmap_array,
    profile_plateau_reference,
    read_vtk_with_meta,
    resolve_path,
    semiaxes_from_source_phi,
    smoothstep01,
    source_stats,
    write_diagnostics_txt,
    write_summary_csv,
)


@dataclass
class RescaledNucleus:
    index: int
    name: str
    center_nm: np.ndarray
    profile_dir: Path
    source_dyn_dir: Path | None
    source_grid_n: tuple[int, int, int]
    source_dx_nm: float
    source_interface_width_nm: float
    target_interface_width_nm: float
    source_box_size_nm: np.ndarray
    source_box_size_target_cells: np.ndarray
    source_box_blend_margin_nm: float
    scale_source_box_with_geometry: bool
    scale_factor_geometry: float
    scale_factor_interface_width: float
    scale_factor_xB_profile_width: float
    alpha_interface: float
    xB_rebuild_mode: str
    profile_extrapolate: str
    profiles: dict[str, dict[str, np.ndarray]]
    profile_info: dict[str, Any]
    axes: np.ndarray
    semiaxes_source_nm: np.ndarray
    semiaxes_target_nm: np.ndarray
    semiaxes_source: str
    xB_matrix_reference: float
    xB_matrix_near: float
    xB_edge_mode: str
    xB_edge_value_config: float | None
    xB_edge_sample_radius_nm: float | None
    xB_edge_shell_inner_margin_nm: float | None
    xB_edge_shell_outer_margin_nm: float | None
    xB_edge_i: float = float("nan")
    xB_edge_sample_count: int = 0
    xB_edge_sample_mean: float = float("nan")
    xB_edge_sample_median: float = float("nan")
    xB_edge_sample_stat: str = "mean"
    source_stats: dict[str, float] = field(default_factory=dict)
    accum: dict[str, float] = field(default_factory=dict)


def _to_int3(value: Any, *, key: str) -> tuple[int, int, int]:
    if value is None or len(value) != 3:
        raise ValueError(f"{key} must be a 3-element list")
    return tuple(int(v) for v in value)


def _target_shape(global_cfg: dict[str, Any]) -> tuple[int, int, int]:
    if "target_N" in global_cfg:
        return _to_int3(global_cfg["target_N"], key="global.target_N")
    return (
        int(global_cfg["Nx"]),
        int(global_cfg["Ny"]),
        int(global_cfg["Nz"]),
    )


def _write_vtk_header_with_origin(
    f,
    name: str,
    shape: tuple[int, int, int],
    dx_nm: float,
    origin_nm: np.ndarray,
) -> None:
    nx, ny, nz = shape
    f.write("# vtk DataFile Version 3.0\n")
    f.write(f"{name}\n")
    f.write("ASCII\n")
    f.write("DATASET STRUCTURED_POINTS\n")
    f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
    f.write(f"ORIGIN {origin_nm[0]:.12g} {origin_nm[1]:.12g} {origin_nm[2]:.12g}\n")
    f.write(f"SPACING {dx_nm:.12g} {dx_nm:.12g} {dx_nm:.12g}\n")
    f.write(f"POINT_DATA {nx * ny * nz}\n")
    f.write(f"SCALARS {name} float 1\n")
    f.write("LOOKUP_TABLE default\n")


def _write_flat_values(f, data: np.ndarray) -> None:
    flat = np.asarray(data).ravel(order="C")
    for start in range(0, flat.size, 6):
        f.write(" ".join(f"{float(v):.8e}" for v in flat[start : start + 6]))
        f.write("\n")


def write_vtk_scalar(
    path: Path,
    data: np.ndarray,
    name: str,
    dx_nm: float,
    origin_nm: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="ascii") as f:
        _write_vtk_header_with_origin(f, name, data.shape, dx_nm, origin_nm)
        _write_flat_values(f, data)


def write_vtk_scalar_chunked(
    path: Path,
    shape: tuple[int, int, int],
    name: str,
    dx_nm: float,
    origin_nm: np.ndarray,
    chunk_z: int,
    chunk_fn,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    _nx, _ny, nz = shape
    with path.open("w", encoding="ascii") as f:
        _write_vtk_header_with_origin(f, name, shape, dx_nm, origin_nm)
        for z0 in range(0, nz, chunk_z):
            z1 = min(nz, z0 + chunk_z)
            _write_flat_values(f, chunk_fn(z0, z1))


def _source_grid_from_vtk(source_dyn_dir: Path | None) -> tuple[tuple[int, int, int] | None, float | None]:
    if not source_dyn_dir:
        return None, None
    phi_path = latest_step_vtk(source_dyn_dir, "phi", "phi_1500.vtk")
    if not phi_path or not phi_path.exists():
        return None, None
    _phi, dims, spacing = read_vtk_with_meta(phi_path)
    return dims, float(spacing[0])


def _load_source_fields(
    source_dyn_dir: Path | None,
    source_dx_nm: float,
) -> tuple[np.ndarray | None, np.ndarray | None, tuple[int, int, int] | None]:
    if not source_dyn_dir:
        return None, None, None
    phi = None
    xb = None
    dims = None
    phi_path = latest_step_vtk(source_dyn_dir, "phi", "phi_1500.vtk")
    xb_path = latest_step_vtk(source_dyn_dir, "xB", "xB_1500.vtk")
    if phi_path and phi_path.exists():
        phi, dims, _spacing = read_vtk_with_meta(phi_path)
    if xb_path and xb_path.exists():
        xb, _dims, _spacing = read_vtk_with_meta(xb_path)
    return phi, xb, dims


def load_nucleus(
    raw: dict[str, Any],
    index: int,
    *,
    config_dir: Path,
    global_cfg: dict[str, Any],
    target_dx_nm: float,
    target_interface_width_nm: float | None,
    xB_far: float,
) -> RescaledNucleus | None:
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
    inferred_grid, inferred_dx = _source_grid_from_vtk(source_dyn_dir)
    if raw.get("source_grid_N") is not None:
        source_grid_n = _to_int3(raw["source_grid_N"], key=f"nucleus {name}.source_grid_N")
    elif inferred_grid is not None:
        source_grid_n = inferred_grid
    else:
        raise ValueError(f"nucleus {name}: source_grid_N missing and source VTK dimensions unavailable")

    source_dx_nm = float(raw.get("source_dx_nm") or global_cfg.get("source_default_dx_nm") or inferred_dx or 0.1)
    source_interface_width_nm = float(
        raw.get("source_interface_width_nm")
        or global_cfg.get("source_default_interface_width_nm")
        or 0.6
    )
    nucleus_target_interface_width_nm = float(
        raw.get("target_interface_width_nm")
        or target_interface_width_nm
        or source_interface_width_nm
    )
    if source_interface_width_nm <= 0.0 or nucleus_target_interface_width_nm <= 0.0:
        raise ValueError(
            f"nucleus {name}: source_interface_width_nm and target_interface_width_nm must be > 0; "
            f"got {source_interface_width_nm}, {nucleus_target_interface_width_nm}"
        )
    source_phi, source_xb, _source_dims = _load_source_fields(source_dyn_dir, source_dx_nm)
    source_box_size_nm = np.array(source_grid_n, dtype=np.float64) * source_dx_nm

    scale_factor_geometry = float(raw.get("scale_factor_geometry", 1.0))
    if raw.get("scale_factor_interface_width") is not None:
        scale_factor_interface_width = float(raw["scale_factor_interface_width"])
    else:
        scale_factor_interface_width = nucleus_target_interface_width_nm / source_interface_width_nm
    if raw.get("scale_factor_xB_profile_width") is not None:
        scale_factor_xB_profile_width = float(raw["scale_factor_xB_profile_width"])
    else:
        scale_factor_xB_profile_width = scale_factor_interface_width
    scale_source_box_with_geometry = bool(raw.get("scale_source_box_with_geometry", False))
    source_box_effective_nm = source_box_size_nm * (scale_factor_geometry if scale_source_box_with_geometry else 1.0)

    center_nm = np.array(raw["center_nm"], dtype=np.float64)
    summary_path = source_dyn_dir / "summary.txt" if source_dyn_dir else None
    axes = axes_from_summary(summary_path)
    fallback_radius_nm = float(raw.get("fallback_radius_nm", 7.0))
    semiaxes_source_nm, semiaxes_source = semiaxes_from_source_phi(
        source_phi,
        (source_dx_nm, source_dx_nm, source_dx_nm) if source_phi is not None else None,
        axes,
        fallback_radius_nm,
    )
    semiaxes_target_nm = semiaxes_source_nm * scale_factor_geometry

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
        "boundary_xB_before_sum": 0.0,
        "boundary_xB_before_count": 0.0,
        "boundary_xB_before_min": math.inf,
        "boundary_xB_before_max": -math.inf,
        "transition_xB_min": math.inf,
        "transition_xB_max": -math.inf,
        "boundary_xB_after_sum": 0.0,
        "boundary_xB_after_count": 0.0,
        "boundary_xB_after_min": math.inf,
        "boundary_xB_after_max": -math.inf,
    }
    return RescaledNucleus(
        index=index,
        name=name,
        center_nm=center_nm,
        profile_dir=profile_dir,
        source_dyn_dir=source_dyn_dir,
        source_grid_n=source_grid_n,
        source_dx_nm=source_dx_nm,
        source_interface_width_nm=source_interface_width_nm,
        target_interface_width_nm=nucleus_target_interface_width_nm,
        source_box_size_nm=source_box_effective_nm,
        source_box_size_target_cells=source_box_effective_nm / target_dx_nm,
        source_box_blend_margin_nm=float(raw.get("source_box_blend_margin_nm", raw.get("blend_inner_margin_nm", 3.0))),
        scale_source_box_with_geometry=scale_source_box_with_geometry,
        scale_factor_geometry=scale_factor_geometry,
        scale_factor_interface_width=scale_factor_interface_width,
        scale_factor_xB_profile_width=scale_factor_xB_profile_width,
        alpha_interface=float(raw.get("alpha_interface", 0.25)),
        xB_rebuild_mode=str(raw.get("xB_rebuild_mode", "delta-interface")),
        profile_extrapolate=str(raw.get("profile_extrapolate", "far")),
        profiles=profiles,
        profile_info=profile_info,
        axes=axes,
        semiaxes_source_nm=semiaxes_source_nm,
        semiaxes_target_nm=semiaxes_target_nm,
        semiaxes_source=semiaxes_source,
        xB_matrix_reference=xB_matrix_reference,
        xB_matrix_near=xB_matrix_near,
        xB_edge_mode=str(raw.get("xB_edge_mode", "")),
        xB_edge_value_config=float(raw["xB_edge_value"]) if raw.get("xB_edge_value") is not None else None,
        xB_edge_sample_radius_nm=float(raw["xB_edge_sample_radius_nm"]) if raw.get("xB_edge_sample_radius_nm") is not None else None,
        xB_edge_shell_inner_margin_nm=float(raw["xB_edge_shell_inner_margin_nm"]) if raw.get("xB_edge_shell_inner_margin_nm") is not None else None,
        xB_edge_shell_outer_margin_nm=float(raw["xB_edge_shell_outer_margin_nm"]) if raw.get("xB_edge_shell_outer_margin_nm") is not None else None,
        source_stats=stats,
        accum=accum,
    )


def chunk_geometry_physical(
    *,
    nucleus: RescaledNucleus,
    x_nm: np.ndarray,
    y_nm: np.ndarray,
    z_nm: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    qx = x_nm[:, None, None] - nucleus.center_nm[0]
    qy = y_nm[None, :, None] - nucleus.center_nm[1]
    qz = z_nm[None, None, :] - nucleus.center_nm[2]

    axes = nucleus.axes
    l0 = axes[0, 0] * qx + axes[0, 1] * qy + axes[0, 2] * qz
    l1 = axes[1, 0] * qx + axes[1, 1] * qy + axes[1, 2] * qz
    l2 = axes[2, 0] * qx + axes[2, 1] * qy + axes[2, 2] * qz
    local0 = np.asarray(l0, dtype=np.float32)
    local1 = np.asarray(l1, dtype=np.float32)
    local2 = np.asarray(l2, dtype=np.float32)

    scaled0 = local0 / np.float32(nucleus.semiaxes_target_nm[0])
    scaled1 = local1 / np.float32(nucleus.semiaxes_target_nm[1])
    scaled2 = local2 / np.float32(nucleus.semiaxes_target_nm[2])
    rho = np.sqrt(scaled0 * scaled0 + scaled1 * scaled1 + scaled2 * scaled2).astype(np.float32, copy=False)
    r = np.sqrt(local0 * local0 + local1 * local1 + local2 * local2).astype(np.float32, copy=False)
    mean_radius = np.float32(np.mean(nucleus.semiaxes_target_nm))
    boundary_radius = np.where(rho > 1.0e-8, r / np.maximum(rho, 1.0e-8), mean_radius)
    d_target_nm = (rho - np.float32(1.0)) * boundary_radius
    family_idx = family_index_from_local(local0, local1, local2)
    return d_target_nm.astype(np.float32), family_idx, rho, (qx, qy, qz)


def rebuild_rescaled_profile_chunk(
    nucleus: RescaledNucleus,
    d_target_nm: np.ndarray,
    family_idx: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    # geometry scale controls nucleus size; interface/profile scales control diffuse broadening.
    # Do not use target/source grid index ratios here: profiles are interpolated in physical nm.
    d_phi_profile_nm = d_target_nm / np.float32(nucleus.scale_factor_interface_width)
    d_xb_profile_nm = d_target_nm / np.float32(nucleus.scale_factor_xB_profile_width)
    phi = np.empty(d_target_nm.shape, dtype=np.float32)
    delta = np.empty(d_target_nm.shape, dtype=np.float32)
    fallback = next(iter(nucleus.profiles.values()))
    for idx, label in enumerate(FAMILY_LABELS):
        prof = nucleus.profiles.get(label, fallback)
        mask = family_idx == idx
        if not np.any(mask):
            continue
        phi[mask] = interp_const(prof["u_nm"], prof["phi_mean"], d_phi_profile_nm[mask]).astype(np.float32)
        delta[mask] = interp_delta(
            prof["u_nm"],
            prof["xB_mean"] - nucleus.xB_matrix_reference,
            d_xb_profile_nm[mask],
            nucleus.profile_extrapolate,
        ).astype(np.float32)
    phi = np.clip(phi, 0.0, 1.0).astype(np.float32)
    if nucleus.xB_rebuild_mode == "constant-matrix":
        xB_local = np.full(d_target_nm.shape, nucleus.xB_matrix_near, dtype=np.float32)
    else:
        xB_local = (nucleus.xB_matrix_near + nucleus.alpha_interface * delta).astype(np.float32)
    return phi, xB_local


def source_box_window_physical(
    nucleus: RescaledNucleus,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
) -> np.ndarray:
    hx, hy, hz = (0.5 * nucleus.source_box_size_nm).astype(np.float32)
    margin = max(float(nucleus.source_box_blend_margin_nm), 1.0e-12)
    m = np.minimum(np.minimum(hx - np.abs(qx), hy - np.abs(qy)), hz - np.abs(qz))
    w = np.zeros(np.broadcast_shapes(qx.shape, qy.shape, qz.shape), dtype=np.float32)
    inside = m > 0.0
    if np.any(inside):
        s = np.clip(m[inside] / margin, 0.0, 1.0)
        w[inside] = smoothstep01(s).astype(np.float32)
    return w


def source_box_shell_weight_physical(
    nucleus: RescaledNucleus,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
    *,
    inner_nm: float,
    outer_nm: float,
) -> np.ndarray:
    if outer_nm <= inner_nm:
        raise ValueError(f"local compensation shell outer_nm ({outer_nm}) must be > inner_nm ({inner_nm})")
    m = chunk_source_box_margin(nucleus, qx, qy, qz)
    outside_dist = -m
    return ((outside_dist >= inner_nm) & (outside_dist <= outer_nm)).astype(np.float32)


def chunk_source_box_margin(
    nucleus: RescaledNucleus,
    qx: np.ndarray,
    qy: np.ndarray,
    qz: np.ndarray,
) -> np.ndarray:
    hx, hy, hz = (0.5 * nucleus.source_box_size_nm).astype(np.float32)
    return np.minimum(np.minimum(hx - np.abs(qx), hy - np.abs(qy)), hz - np.abs(qz)).astype(np.float32)


def array_stats_chunked(arr: np.memmap, *, chunk_z: int) -> dict[str, float]:
    nx, ny, nz = arr.shape
    n = nx * ny * nz
    total = 0.0
    vmin = math.inf
    vmax = -math.inf
    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        chunk = np.asarray(arr[:, :, z0:z1], dtype=np.float32)
        total += float(np.sum(chunk, dtype=np.float64))
        vmin = min(vmin, float(np.min(chunk)))
        vmax = max(vmax, float(np.max(chunk)))
    return {"min": vmin, "max": vmax, "mean": total / n}


def initialize_background(
    xB: np.memmap,
    *,
    global_cfg: dict[str, Any],
    config_dir: Path,
    target_shape: tuple[int, int, int],
    target_dx_nm: float,
    constant_default: float,
    chunk_z: int,
) -> tuple[dict[str, Any], list[str]]:
    warnings: list[str] = []
    mode = str(global_cfg.get("background_mode", "constant"))
    xB_background_constant = float(global_cfg.get("xB_background_constant", constant_default))
    vtk_path_value = global_cfg.get("xB_background_vtk")
    vtk_path = resolve_path(vtk_path_value, config_dir=config_dir) if vtk_path_value else None

    if mode == "constant":
        xB[:] = np.float32(xB_background_constant)
        xB.flush()
    elif mode == "vtk":
        if vtk_path is None or not vtk_path.exists():
            raise FileNotFoundError(f"background_mode=vtk requires an existing xB_background_vtk; got {vtk_path_value}")
        data, dims, spacing = read_vtk_with_meta(vtk_path)
        if tuple(dims) != tuple(target_shape):
            raise ValueError(f"xB_background_vtk dimensions {dims} do not match target_N {target_shape}")
        if any(abs(float(s) - target_dx_nm) > 1.0e-6 for s in spacing):
            raise ValueError(f"xB_background_vtk spacing {spacing} does not match target_dx_nm={target_dx_nm}")
        xB[:] = data.astype(xB.dtype, copy=False)
        xB.flush()
    elif mode == "function":
        raise NotImplementedError("background_mode=function is reserved but not implemented yet. Use constant or vtk.")
    else:
        raise ValueError(f"Unsupported background_mode: {mode}")

    stats = array_stats_chunked(xB, chunk_z=chunk_z)
    return (
        {
            "background_mode": mode,
            "xB_background_constant": xB_background_constant,
            "xB_background_vtk": str(vtk_path) if vtk_path else None,
            "xB_background_min": stats["min"],
            "xB_background_max": stats["max"],
            "xB_background_mean": stats["mean"],
        },
        warnings,
    )


def _sample_stat(values: list[np.ndarray], stat: str) -> tuple[int, float, float, float]:
    if not values:
        return 0, float("nan"), float("nan"), float("nan")
    data = np.concatenate(values)
    if data.size == 0:
        return 0, float("nan"), float("nan"), float("nan")
    mean = float(np.mean(data, dtype=np.float64))
    median = float(np.median(data))
    selected = median if stat == "median" else mean
    return int(data.size), mean, median, selected


def sample_background_for_edge(
    background: np.memmap,
    nucleus: RescaledNucleus,
    *,
    mode: str,
    stat: str,
    radius_nm: float,
    shell_inner_nm: float,
    shell_outer_nm: float,
    target_dx_nm: float,
    origin_nm: np.ndarray,
    chunk_z: int,
) -> tuple[int, float, float, float]:
    nx, ny, nz = background.shape
    x_coords = origin_nm[0] + (np.arange(nx, dtype=np.float32) + np.float32(0.5)) * np.float32(target_dx_nm)
    y_coords = origin_nm[1] + (np.arange(ny, dtype=np.float32) + np.float32(0.5)) * np.float32(target_dx_nm)
    values: list[np.ndarray] = []
    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        z_coords = origin_nm[2] + (np.arange(z0, z1, dtype=np.float32) + np.float32(0.5)) * np.float32(target_dx_nm)
        qx = x_coords[:, None, None] - nucleus.center_nm[0]
        qy = y_coords[None, :, None] - nucleus.center_nm[1]
        qz = z_coords[None, None, :] - nucleus.center_nm[2]
        if mode == "sample-background":
            r2 = qx * qx + qy * qy + qz * qz
            mask = r2 <= np.float32(radius_nm * radius_nm)
        elif mode == "sample-background-shell":
            m = chunk_source_box_margin(nucleus, qx, qy, qz)
            outside_dist = -m
            mask = (outside_dist >= shell_inner_nm) & (outside_dist <= shell_outer_nm)
        else:
            raise ValueError(f"Unsupported sampling edge mode: {mode}")
        if np.any(mask):
            values.append(np.asarray(background[:, :, z0:z1], dtype=np.float32)[mask])
    return _sample_stat(values, stat)


def resolve_nucleus_edges(
    nuclei: list[RescaledNucleus],
    background: np.memmap,
    *,
    global_cfg: dict[str, Any],
    background_info: dict[str, Any],
    target_dx_nm: float,
    origin_nm: np.ndarray,
    xB_min: float,
    xB_max: float,
    chunk_z: int,
) -> list[str]:
    warnings: list[str] = []
    default_mode = str(global_cfg.get("xB_edge_mode_default", "global-constant"))
    edge_stat = str(global_cfg.get("edge_sample_stat", "mean"))
    if edge_stat not in {"mean", "median"}:
        raise ValueError(f"edge_sample_stat must be mean or median, got {edge_stat}")
    constant = float(global_cfg.get("xB_background_constant", background_info.get("xB_background_mean", 0.030)))
    default_radius = float(global_cfg.get("xB_edge_sample_radius_nm", target_dx_nm))
    default_shell_inner = float(global_cfg.get("xB_edge_shell_inner_margin_nm", 0.0))
    default_shell_outer = float(global_cfg.get("xB_edge_shell_outer_margin_nm", 3.0))

    for nucleus in nuclei:
        mode = nucleus.xB_edge_mode or default_mode
        nucleus.xB_edge_mode = mode
        nucleus.xB_edge_sample_stat = edge_stat
        sample_count = 0
        sample_mean = float("nan")
        sample_median = float("nan")
        if mode == "global-constant":
            edge = constant
        elif mode == "per-nucleus":
            if nucleus.xB_edge_value_config is None:
                raise ValueError(f"nucleus {nucleus.name}: xB_edge_mode=per-nucleus requires xB_edge_value")
            edge = float(nucleus.xB_edge_value_config)
        elif mode in {"sample-background", "sample-background-shell"}:
            if background_info["background_mode"] == "constant":
                edge = constant
                sample_mean = constant
                sample_median = constant
                sample_count = 1
            else:
                radius = nucleus.xB_edge_sample_radius_nm if nucleus.xB_edge_sample_radius_nm is not None else default_radius
                shell_inner = (
                    nucleus.xB_edge_shell_inner_margin_nm
                    if nucleus.xB_edge_shell_inner_margin_nm is not None
                    else default_shell_inner
                )
                shell_outer = (
                    nucleus.xB_edge_shell_outer_margin_nm
                    if nucleus.xB_edge_shell_outer_margin_nm is not None
                    else default_shell_outer
                )
                sample_count, sample_mean, sample_median, edge = sample_background_for_edge(
                    background,
                    nucleus,
                    mode=mode,
                    stat=edge_stat,
                    radius_nm=float(radius),
                    shell_inner_nm=float(shell_inner),
                    shell_outer_nm=float(shell_outer),
                    target_dx_nm=target_dx_nm,
                    origin_nm=origin_nm,
                    chunk_z=chunk_z,
                )
                if sample_count < 8:
                    warnings.append(
                        f"nucleus {nucleus.name}: {mode} sampled only {sample_count} background points; edge value may be noisy."
                    )
        else:
            raise ValueError(f"Unsupported xB_edge_mode for nucleus {nucleus.name}: {mode}")

        nucleus.xB_edge_i = float(edge)
        nucleus.xB_edge_sample_count = int(sample_count)
        nucleus.xB_edge_sample_mean = float(sample_mean)
        nucleus.xB_edge_sample_median = float(sample_median)
        if not (xB_min <= nucleus.xB_edge_i <= xB_max):
            warnings.append(
                f"nucleus {nucleus.name}: xB_edge_i={nucleus.xB_edge_i:.6g} is outside clipping bounds [{xB_min}, {xB_max}]"
            )
    return warnings


def recompute_boundary_xB_stats(
    nuclei: list[RescaledNucleus],
    xB: np.memmap,
    *,
    phase: str,
    target_dx_nm: float,
    origin_nm: np.ndarray,
    W_comp_threshold: float,
    chunk_z: int,
) -> None:
    if phase not in {"before", "after"}:
        raise ValueError(f"phase must be before or after, got {phase}")
    nx, ny, nz = xB.shape
    x_coords = origin_nm[0] + (np.arange(nx, dtype=np.float32) + np.float32(0.5)) * np.float32(target_dx_nm)
    y_coords = origin_nm[1] + (np.arange(ny, dtype=np.float32) + np.float32(0.5)) * np.float32(target_dx_nm)
    for nucleus in nuclei:
        nucleus.accum[f"boundary_xB_{phase}_sum"] = 0.0
        nucleus.accum[f"boundary_xB_{phase}_count"] = 0.0
        nucleus.accum[f"boundary_xB_{phase}_min"] = math.inf
        nucleus.accum[f"boundary_xB_{phase}_max"] = -math.inf
        for z0 in range(0, nz, chunk_z):
            z1 = min(nz, z0 + chunk_z)
            z_coords = origin_nm[2] + (np.arange(z0, z1, dtype=np.float32) + np.float32(0.5)) * np.float32(target_dx_nm)
            qx = x_coords[:, None, None] - nucleus.center_nm[0]
            qy = y_coords[None, :, None] - nucleus.center_nm[1]
            qz = z_coords[None, None, :] - nucleus.center_nm[2]
            W_box = source_box_window_physical(nucleus, qx, qy, qz)
            mask = (W_box > W_comp_threshold) & (W_box < 0.999)
            if np.any(mask):
                vals = np.asarray(xB[:, :, z0:z1], dtype=np.float32)[mask]
                nucleus.accum[f"boundary_xB_{phase}_sum"] += float(np.sum(vals, dtype=np.float64))
                nucleus.accum[f"boundary_xB_{phase}_count"] += float(vals.size)
                nucleus.accum[f"boundary_xB_{phase}_min"] = min(
                    nucleus.accum[f"boundary_xB_{phase}_min"], float(np.min(vals))
                )
                nucleus.accum[f"boundary_xB_{phase}_max"] = max(
                    nucleus.accum[f"boundary_xB_{phase}_max"], float(np.max(vals))
                )


def collect_stats(
    phi_total: np.memmap,
    xB: np.memmap,
    W_box_max: np.memmap,
    active_phi_count: np.memmap,
    active_source_box_count: np.memmap,
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
    far_xb_sum = 0.0
    active_phi_max = 0
    active_source_box_max = 0
    phi_overlap_count = 0
    source_box_overlap_count = 0
    lower_count = 0
    upper_count = 0
    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        phi = np.asarray(phi_total[:, :, z0:z1])
        xb = np.asarray(xB[:, :, z0:z1])
        wmax = np.asarray(W_box_max[:, :, z0:z1])
        active_phi = np.asarray(active_phi_count[:, :, z0:z1])
        active_source = np.asarray(active_source_box_count[:, :, z0:z1])
        phi_min = min(phi_min, float(np.min(phi)))
        phi_max = max(phi_max, float(np.max(phi)))
        xb_min = min(xb_min, float(np.min(xb)))
        xb_max = max(xb_max, float(np.max(xb)))
        far_mask = (phi < phi_matrix_threshold) & (wmax < W_comp_threshold)
        far_count += int(np.count_nonzero(far_mask))
        if np.any(far_mask):
            far_xb_sum += float(np.sum(xb[far_mask], dtype=np.float64))
        active_phi_max = max(active_phi_max, int(np.max(active_phi)))
        active_source_box_max = max(active_source_box_max, int(np.max(active_source)))
        phi_overlap_count += int(np.count_nonzero(active_phi > 1))
        source_box_overlap_count += int(np.count_nonzero(active_source > 1))
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
        "far_field_mean_xB": float(far_xb_sum / far_count) if far_count else float("nan"),
        "active_phi_count_max": float(active_phi_max),
        "active_source_box_count_max": float(active_source_box_max),
        "overlap_fraction_phi_i_gt_0p05": float(phi_overlap_count / n),
        "overlap_fraction_source_box_gt_threshold": float(source_box_overlap_count / n),
        "lower_clipped_fraction": float(lower_count / n),
        "upper_clipped_fraction": float(upper_count / n),
    }


def _region_delta_stats(values: dict[str, list[float]]) -> dict[str, float]:
    count = int(values["count"][0]) if values["count"] else 0
    out: dict[str, float] = {"count": float(count)}
    if count == 0:
        for key in ("sum_delta_xBtot", "max_abs_delta_xB", "mean_abs_delta_xB"):
            out[key] = float("nan")
        return out
    out["sum_delta_xBtot"] = values["sum_delta_xBtot"][0]
    out["max_abs_delta_xB"] = values["max_abs_delta_xB"][0]
    out["mean_abs_delta_xB"] = values["sum_abs_delta_xB"][0] / count
    return out


def compute_delta_diagnostics(
    phi_total: np.memmap,
    xB_before: np.memmap,
    xB_after: np.memmap,
    W_box_max: np.memmap,
    local_comp_weight: np.memmap,
    *,
    W_comp_threshold: float,
    phi_matrix_threshold: float,
    chunk_z: int,
) -> dict[str, Any]:
    nx, ny, nz = phi_total.shape
    n = nx * ny * nz
    delta_sum = 0.0
    delta2_sum = 0.0
    delta_min = math.inf
    delta_max = -math.inf
    delta_tot_sum = 0.0
    delta_tot2_sum = 0.0
    delta_tot_min = math.inf
    delta_tot_max = -math.inf
    regions = {
        "inside_source_boxes": {"count": [0], "sum_delta_xBtot": [0.0], "sum_abs_delta_xB": [0.0], "max_abs_delta_xB": [0.0]},
        "local_comp_shell": {"count": [0], "sum_delta_xBtot": [0.0], "sum_abs_delta_xB": [0.0], "max_abs_delta_xB": [0.0]},
        "far_outside_shell": {"count": [0], "sum_delta_xBtot": [0.0], "sum_abs_delta_xB": [0.0], "max_abs_delta_xB": [0.0]},
    }

    def add_region(name: str, mask: np.ndarray, dx: np.ndarray, dxt: np.ndarray) -> None:
        if not np.any(mask):
            return
        vals = dx[mask]
        vals_tot = dxt[mask]
        rec = regions[name]
        rec["count"][0] += int(vals.size)
        rec["sum_delta_xBtot"][0] += float(np.sum(vals_tot, dtype=np.float64))
        rec["sum_abs_delta_xB"][0] += float(np.sum(np.abs(vals), dtype=np.float64))
        rec["max_abs_delta_xB"][0] = max(rec["max_abs_delta_xB"][0], float(np.max(np.abs(vals))))

    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        phi = np.asarray(phi_total[:, :, z0:z1], dtype=np.float32)
        xb0 = np.asarray(xB_before[:, :, z0:z1], dtype=np.float32)
        xb1 = np.asarray(xB_after[:, :, z0:z1], dtype=np.float32)
        wmax = np.asarray(W_box_max[:, :, z0:z1], dtype=np.float32)
        lweight = np.asarray(local_comp_weight[:, :, z0:z1], dtype=np.float32)
        hp = h_phi(phi)
        dx = xb1 - xb0
        dxt = (1.0 - hp) * dx
        delta_sum += float(np.sum(dx, dtype=np.float64))
        delta2_sum += float(np.sum(dx * dx, dtype=np.float64))
        delta_min = min(delta_min, float(np.min(dx)))
        delta_max = max(delta_max, float(np.max(dx)))
        delta_tot_sum += float(np.sum(dxt, dtype=np.float64))
        delta_tot2_sum += float(np.sum(dxt * dxt, dtype=np.float64))
        delta_tot_min = min(delta_tot_min, float(np.min(dxt)))
        delta_tot_max = max(delta_tot_max, float(np.max(dxt)))
        inside = wmax > W_comp_threshold
        shell = lweight > 1.0e-6
        far = (wmax < W_comp_threshold) & (lweight < 1.0e-6) & (phi < phi_matrix_threshold)
        add_region("inside_source_boxes", inside, dx, dxt)
        add_region("local_comp_shell", shell, dx, dxt)
        add_region("far_outside_shell", far, dx, dxt)

    delta_mean = delta_sum / n
    delta_tot_mean = delta_tot_sum / n
    shell_mass = regions["local_comp_shell"]["sum_delta_xBtot"][0]
    fraction_shell = shell_mass / delta_tot_sum if abs(delta_tot_sum) > 1.0e-30 else float("nan")
    return {
        "delta_xB_min": delta_min,
        "delta_xB_max": delta_max,
        "delta_xB_mean": delta_mean,
        "delta_xB_std": math.sqrt(max(delta2_sum / n - delta_mean * delta_mean, 0.0)),
        "delta_xBtot_min": delta_tot_min,
        "delta_xBtot_max": delta_tot_max,
        "delta_xBtot_mean": delta_tot_mean,
        "delta_xBtot_std": math.sqrt(max(delta_tot2_sum / n - delta_tot_mean * delta_tot_mean, 0.0)),
        "max_abs_delta_xB": max(abs(delta_min), abs(delta_max)),
        "total_delta_mass_global": delta_tot_sum,
        "delta_mass_inside_source_boxes": regions["inside_source_boxes"]["sum_delta_xBtot"][0],
        "delta_mass_local_comp_shell": shell_mass,
        "delta_mass_far_outside_shell": regions["far_outside_shell"]["sum_delta_xBtot"][0],
        "fraction_of_delta_mass_in_local_comp_shell": fraction_shell,
        "inside_source_boxes": _region_delta_stats(regions["inside_source_boxes"]),
        "local_comp_shell": _region_delta_stats(regions["local_comp_shell"]),
        "far_outside_shell": _region_delta_stats(regions["far_outside_shell"]),
        "max_abs_delta_xB_inside_source_boxes": regions["inside_source_boxes"]["max_abs_delta_xB"][0],
        "mean_abs_delta_xB_inside_source_boxes": (
            regions["inside_source_boxes"]["sum_abs_delta_xB"][0] / regions["inside_source_boxes"]["count"][0]
            if regions["inside_source_boxes"]["count"][0]
            else float("nan")
        ),
        "max_abs_delta_xB_in_local_comp_shell": regions["local_comp_shell"]["max_abs_delta_xB"][0],
        "max_abs_delta_xB_far_outside_shell": regions["far_outside_shell"]["max_abs_delta_xB"][0],
        "mean_abs_delta_xB_far_outside_shell": (
            regions["far_outside_shell"]["sum_abs_delta_xB"][0] / regions["far_outside_shell"]["count"][0]
            if regions["far_outside_shell"]["count"][0]
            else float("nan")
        ),
    }


def write_before_after_fields(
    out_dir: Path,
    *,
    write_vtk: bool,
    phi_total: np.memmap,
    xB_before: np.memmap,
    xB_after: np.memmap,
    W_box_max: np.memmap,
    local_comp_weight: np.memmap,
    active_source_box_count: np.memmap,
    active_phi_count: np.memmap,
    target_dx_nm: float,
    origin_nm: np.ndarray,
    chunk_z: int,
) -> None:
    shape = phi_total.shape
    if write_vtk:
        write_vtk_scalar(out_dir / "phi_embedded.vtk", phi_total, "phi_embedded", target_dx_nm, origin_nm)
        write_vtk_scalar(out_dir / "xB_before_compensation.vtk", xB_before, "xB_before_compensation", target_dx_nm, origin_nm)
        write_vtk_scalar(out_dir / "xB_after_compensation.vtk", xB_after, "xB_after_compensation", target_dx_nm, origin_nm)
        write_vtk_scalar(out_dir / "xB_embedded.vtk", xB_after, "xB_embedded", target_dx_nm, origin_nm)
        write_vtk_scalar_chunked(
            out_dir / "xB_delta_compensation.vtk",
            shape,
            "xB_delta_compensation",
            target_dx_nm,
            origin_nm,
            chunk_z,
            lambda z0, z1: (
                np.asarray(xB_after[:, :, z0:z1], dtype=np.float32)
                - np.asarray(xB_before[:, :, z0:z1], dtype=np.float32)
            ).astype(np.float32),
        )
        write_vtk_scalar_chunked(
            out_dir / "xBtot_before_compensation.vtk",
            shape,
            "xBtot_before_compensation",
            target_dx_nm,
            origin_nm,
            chunk_z,
            lambda z0, z1: (
                lambda phi, xb: ((1.0 - h_phi(phi)) * xb + h_phi(phi)).astype(np.float32)
            )(np.asarray(phi_total[:, :, z0:z1], dtype=np.float32), np.asarray(xB_before[:, :, z0:z1], dtype=np.float32)),
        )
        write_vtk_scalar_chunked(
            out_dir / "xBtot_after_compensation.vtk",
            shape,
            "xBtot_after_compensation",
            target_dx_nm,
            origin_nm,
            chunk_z,
            lambda z0, z1: (
                lambda phi, xb: ((1.0 - h_phi(phi)) * xb + h_phi(phi)).astype(np.float32)
            )(np.asarray(phi_total[:, :, z0:z1], dtype=np.float32), np.asarray(xB_after[:, :, z0:z1], dtype=np.float32)),
        )
        write_vtk_scalar_chunked(
            out_dir / "xBtot_embedded.vtk",
            shape,
            "xBtot_embedded",
            target_dx_nm,
            origin_nm,
            chunk_z,
            lambda z0, z1: (
                lambda phi, xb: ((1.0 - h_phi(phi)) * xb + h_phi(phi)).astype(np.float32)
            )(np.asarray(phi_total[:, :, z0:z1], dtype=np.float32), np.asarray(xB_after[:, :, z0:z1], dtype=np.float32)),
        )
        write_vtk_scalar_chunked(
            out_dir / "xBtot_delta_compensation.vtk",
            shape,
            "xBtot_delta_compensation",
            target_dx_nm,
            origin_nm,
            chunk_z,
            lambda z0, z1: (
                lambda phi, xb0, xb1: ((1.0 - h_phi(phi)) * (xb1 - xb0)).astype(np.float32)
            )(
                np.asarray(phi_total[:, :, z0:z1], dtype=np.float32),
                np.asarray(xB_before[:, :, z0:z1], dtype=np.float32),
                np.asarray(xB_after[:, :, z0:z1], dtype=np.float32),
            ),
        )
        write_vtk_scalar(out_dir / "local_comp_weight.vtk", local_comp_weight, "local_comp_weight", target_dx_nm, origin_nm)
        write_vtk_scalar(out_dir / "W_box_max.vtk", W_box_max, "W_box_max", target_dx_nm, origin_nm)
        write_vtk_scalar(out_dir / "active_source_box_count.vtk", active_source_box_count, "active_source_box_count", target_dx_nm, origin_nm)
        write_vtk_scalar(out_dir / "active_phi_count.vtk", active_phi_count, "active_phi_count", target_dx_nm, origin_nm)
    else:
        phi = np.asarray(phi_total, dtype=np.float32)
        xb0 = np.asarray(xB_before, dtype=np.float32)
        xb1 = np.asarray(xB_after, dtype=np.float32)
        hp = h_phi(phi)
        np.savez_compressed(
            out_dir / "compensation_before_after_fields.npz",
            phi_embedded=phi,
            xB_before_compensation=xb0,
            xB_after_compensation=xb1,
            xB_delta_compensation=(xb1 - xb0).astype(np.float32),
            xBtot_before_compensation=((1.0 - hp) * xb0 + hp).astype(np.float32),
            xBtot_after_compensation=((1.0 - hp) * xb1 + hp).astype(np.float32),
            xBtot_delta_compensation=((1.0 - hp) * (xb1 - xb0)).astype(np.float32),
            local_comp_weight=np.asarray(local_comp_weight, dtype=np.float32),
            W_box_max=np.asarray(W_box_max, dtype=np.float32),
            active_source_box_count=np.asarray(active_source_box_count, dtype=np.float32),
        )


def _slice_arrays(
    phi: np.memmap,
    xb0: np.memmap,
    xb1: np.memmap,
    wbox: np.memmap,
    local_weight: np.memmap,
    active_source: np.memmap,
    plane: str,
) -> tuple[dict[str, np.ndarray], int]:
    nx, ny, nz = phi.shape
    if plane == "mid_x":
        idx = nx // 2
        sl = (idx, slice(None), slice(None))
    elif plane == "mid_y":
        idx = ny // 2
        sl = (slice(None), idx, slice(None))
    else:
        idx = nz // 2
        sl = (slice(None), slice(None), idx)
    p = np.asarray(phi[sl], dtype=np.float32)
    b = np.asarray(xb0[sl], dtype=np.float32)
    a = np.asarray(xb1[sl], dtype=np.float32)
    hp = h_phi(p)
    xbt0 = (1.0 - hp) * b + hp
    xbt1 = (1.0 - hp) * a + hp
    return (
        {
            "phi": p,
            "xB_before": b,
            "xB_after": a,
            "delta_xB": a - b,
            "xBtot_before": xbt0,
            "xBtot_after": xbt1,
            "delta_xBtot": xbt1 - xbt0,
            "W_box_max": np.asarray(wbox[sl], dtype=np.float32),
            "local_comp_weight": np.asarray(local_weight[sl], dtype=np.float32),
            "active_source_box_count": np.asarray(active_source[sl], dtype=np.float32),
        },
        idx,
    )


def _slice_extent(shape2: tuple[int, int], plane: str, target_dx_nm: float, origin_nm: np.ndarray) -> tuple[float, float, float, float]:
    if plane == "mid_x":
        return (
            origin_nm[1],
            origin_nm[1] + shape2[0] * target_dx_nm,
            origin_nm[2],
            origin_nm[2] + shape2[1] * target_dx_nm,
        )
    if plane == "mid_y":
        return (
            origin_nm[0],
            origin_nm[0] + shape2[0] * target_dx_nm,
            origin_nm[2],
            origin_nm[2] + shape2[1] * target_dx_nm,
        )
    return (
        origin_nm[0],
        origin_nm[0] + shape2[0] * target_dx_nm,
        origin_nm[1],
        origin_nm[1] + shape2[1] * target_dx_nm,
    )


def _plot_source_boxes(ax, nuclei: list[RescaledNucleus], plane: str, slice_pos_nm: float, local_comp_outer_nm: float) -> None:
    import matplotlib.patches as patches

    for nuc in nuclei:
        hx, hy, hz = 0.5 * nuc.source_box_size_nm
        cx, cy, cz = nuc.center_nm
        if plane == "mid_z":
            if abs(slice_pos_nm - cz) > hz:
                continue
            xy = (cx - hx, cy - hy)
            wh = (2 * hx, 2 * hy)
            shell_xy = (cx - hx - local_comp_outer_nm, cy - hy - local_comp_outer_nm)
            shell_wh = (2 * (hx + local_comp_outer_nm), 2 * (hy + local_comp_outer_nm))
            marker = (cx, cy)
        elif plane == "mid_y":
            if abs(slice_pos_nm - cy) > hy:
                continue
            xy = (cx - hx, cz - hz)
            wh = (2 * hx, 2 * hz)
            shell_xy = (cx - hx - local_comp_outer_nm, cz - hz - local_comp_outer_nm)
            shell_wh = (2 * (hx + local_comp_outer_nm), 2 * (hz + local_comp_outer_nm))
            marker = (cx, cz)
        else:
            if abs(slice_pos_nm - cx) > hx:
                continue
            xy = (cy - hy, cz - hz)
            wh = (2 * hy, 2 * hz)
            shell_xy = (cy - hy - local_comp_outer_nm, cz - hz - local_comp_outer_nm)
            shell_wh = (2 * (hy + local_comp_outer_nm), 2 * (hz + local_comp_outer_nm))
            marker = (cy, cz)
        ax.add_patch(patches.Rectangle(xy, *wh, fill=False, linestyle="--", linewidth=0.8))
        ax.add_patch(patches.Rectangle(shell_xy, *shell_wh, fill=False, linestyle=":", linewidth=0.8))
        ax.plot(marker[0], marker[1], marker="x", markersize=4)
        ax.text(marker[0], marker[1], nuc.name, fontsize=6)


def plot_global_slice_comparison(
    out_path: Path,
    *,
    plane: str,
    phi_total: np.memmap,
    xB_before: np.memmap,
    xB_after: np.memmap,
    W_box_max: np.memmap,
    local_comp_weight: np.memmap,
    active_source_box_count: np.memmap,
    nuclei: list[RescaledNucleus],
    target_dx_nm: float,
    origin_nm: np.ndarray,
    local_comp_outer_nm: float,
    dpi: int,
) -> None:
    arrays, idx = _slice_arrays(phi_total, xB_before, xB_after, W_box_max, local_comp_weight, active_source_box_count, plane)
    shape2 = arrays["xB_before"].shape
    extent = _slice_extent(shape2, plane, target_dx_nm, origin_nm)
    axis_index = {"mid_x": 0, "mid_y": 1, "mid_z": 2}[plane]
    slice_pos_nm = origin_nm[axis_index] + (idx + 0.5) * target_dx_nm
    fig, axs = plt.subplots(4, 3, figsize=(14, 14), constrained_layout=True)
    rows = [
        ("xB", "xB_before", "xB_after", "delta_xB"),
        ("xBtot", "xBtot_before", "xBtot_after", "delta_xBtot"),
        ("local_comp_weight", "local_comp_weight", "local_comp_weight", "local_comp_weight"),
        ("W_box_max", "W_box_max", "active_source_box_count", "W_box_max"),
    ]
    for ri, (label, before_key, after_key, delta_key) in enumerate(rows):
        if label in {"xB", "xBtot"}:
            vmin = float(min(np.min(arrays[before_key]), np.min(arrays[after_key])))
            vmax = float(max(np.max(arrays[before_key]), np.max(arrays[after_key])))
            dmax = float(max(np.max(np.abs(arrays[delta_key])), 1.0e-12))
            configs = [(before_key, vmin, vmax), (after_key, vmin, vmax), (delta_key, -dmax, dmax)]
        elif label == "local_comp_weight":
            configs = [(before_key, 0.0, 1.0), (after_key, 0.0, 1.0), (delta_key, 0.0, 1.0)]
        else:
            configs = [(before_key, 0.0, 1.0), (after_key, 0.0, max(float(np.max(arrays[after_key])), 1.0)), (delta_key, 0.0, 1.0)]
        for ci, (key, vmin, vmax) in enumerate(configs):
            ax = axs[ri, ci]
            im = ax.imshow(arrays[key].T, origin="lower", extent=extent, aspect="equal", vmin=vmin, vmax=vmax)
            title = key.replace("_", " ")
            ax.set_title(f"{plane}: {title}\nmin={np.min(arrays[key]):.4g}, max={np.max(arrays[key]):.4g}, mean={np.mean(arrays[key]):.4g}")
            _plot_source_boxes(ax, nuclei, plane, slice_pos_nm, local_comp_outer_nm)
            fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)


def write_center_axis_before_after(
    out_csv: Path,
    out_png: Path,
    *,
    phi_total: np.memmap,
    xB_before: np.memmap,
    xB_after: np.memmap,
    W_box_max: np.memmap,
    local_comp_weight: np.memmap,
    active_source_box_count: np.memmap,
    target_dx_nm: float,
    origin_nm: np.ndarray,
    xB_background_constant: float,
    dpi: int,
) -> None:
    nx, ny, nz = phi_total.shape
    center = np.array([nx, ny, nz]) // 2
    rows: list[dict[str, Any]] = []
    for axis in ("x", "y", "z"):
        if axis == "x":
            coords = origin_nm[0] + (np.arange(nx) + 0.5) * target_dx_nm
            sl = (slice(None), center[1], center[2])
        elif axis == "y":
            coords = origin_nm[1] + (np.arange(ny) + 0.5) * target_dx_nm
            sl = (center[0], slice(None), center[2])
        else:
            coords = origin_nm[2] + (np.arange(nz) + 0.5) * target_dx_nm
            sl = (center[0], center[1], slice(None))
        p = np.asarray(phi_total[sl], dtype=np.float32)
        b = np.asarray(xB_before[sl], dtype=np.float32)
        a = np.asarray(xB_after[sl], dtype=np.float32)
        hp = h_phi(p)
        xtb = (1.0 - hp) * b + hp
        xta = (1.0 - hp) * a + hp
        w = np.asarray(W_box_max[sl], dtype=np.float32)
        lw = np.asarray(local_comp_weight[sl], dtype=np.float32)
        act = np.asarray(active_source_box_count[sl], dtype=np.float32)
        for i in range(coords.size):
            rows.append(
                {
                    "axis": axis,
                    "index": i,
                    "coordinate_nm": float(coords[i]),
                    "phi": float(p[i]),
                    "xB_before": float(b[i]),
                    "xB_after": float(a[i]),
                    "delta_xB": float(a[i] - b[i]),
                    "xBtot_before": float(xtb[i]),
                    "xBtot_after": float(xta[i]),
                    "delta_xBtot": float(xta[i] - xtb[i]),
                    "W_box_max": float(w[i]),
                    "local_comp_weight": float(lw[i]),
                    "active_source_box_count": float(act[i]),
                    "xB_background_constant": xB_background_constant,
                }
            )
    write_summary_csv(out_csv, rows)
    fig, axs = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
    for ai, axis in enumerate(("x", "y", "z")):
        sub = [r for r in rows if r["axis"] == axis]
        x = np.array([r["coordinate_nm"] for r in sub])
        p = np.array([r["phi"] for r in sub])
        b = np.array([r["xB_before"] for r in sub])
        a = np.array([r["xB_after"] for r in sub])
        dx = np.array([r["delta_xB"] for r in sub])
        xtb = np.array([r["xBtot_before"] for r in sub])
        xta = np.array([r["xBtot_after"] for r in sub])
        w = np.array([r["W_box_max"] for r in sub])
        lw = np.array([r["local_comp_weight"] for r in sub])
        axs[ai, 0].plot(x, p, label="phi")
        axs[ai, 0].plot(x, w, label="W_box_max")
        axs[ai, 1].plot(x, b, label="xB before")
        axs[ai, 1].plot(x, a, label="xB after")
        axs[ai, 1].axhline(xB_background_constant, color="0.4", ls=":", lw=0.8)
        axs[ai, 2].plot(x, dx, label="delta xB")
        axs[ai, 2].plot(x, lw, label="local weight")
        axs[ai, 3].plot(x, xtb, label="xBtot before")
        axs[ai, 3].plot(x, xta, label="xBtot after")
        for ax in axs[ai]:
            ax.grid(alpha=0.25)
            ax.set_title(f"{axis}-axis")
    for ax in axs[0]:
        ax.legend(fontsize=7, frameon=False)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)


def write_per_nucleus_profiles(
    out_dir: Path,
    *,
    nucleus: RescaledNucleus,
    phi_total: np.memmap,
    xB_before: np.memmap,
    xB_after: np.memmap,
    W_box_max: np.memmap,
    local_comp_weight: np.memmap,
    target_dx_nm: float,
    origin_nm: np.ndarray,
    local_comp_inner_nm: float,
    local_comp_outer_nm: float,
    dpi: int,
) -> None:
    nx, ny, nz = phi_total.shape
    rows: list[dict[str, Any]] = []
    half_box = 0.5 * nucleus.source_box_size_nm
    profile_range = float(np.max(half_box) + local_comp_outer_nm + 10.0)
    for axis_idx, axis in enumerate(("x", "y", "z")):
        s_vals = np.arange(-profile_range, profile_range + 0.5 * target_dx_nm, target_dx_nm)
        points = np.repeat(nucleus.center_nm[None, :], s_vals.size, axis=0)
        points[:, axis_idx] += s_vals
        ix = np.clip(np.floor((points[:, 0] - origin_nm[0]) / target_dx_nm).astype(int), 0, nx - 1)
        iy = np.clip(np.floor((points[:, 1] - origin_nm[1]) / target_dx_nm).astype(int), 0, ny - 1)
        iz = np.clip(np.floor((points[:, 2] - origin_nm[2]) / target_dx_nm).astype(int), 0, nz - 1)
        p = np.asarray(phi_total[ix, iy, iz], dtype=np.float32)
        b = np.asarray(xB_before[ix, iy, iz], dtype=np.float32)
        a = np.asarray(xB_after[ix, iy, iz], dtype=np.float32)
        hp = h_phi(p)
        xtb = (1.0 - hp) * b + hp
        xta = (1.0 - hp) * a + hp
        wmax = np.asarray(W_box_max[ix, iy, iz], dtype=np.float32)
        lw = np.asarray(local_comp_weight[ix, iy, iz], dtype=np.float32)
        q = points - nucleus.center_nm[None, :]
        qx = q[:, 0][:, None, None].astype(np.float32)
        qy = q[:, 1][None, :, None].astype(np.float32)
        qz = q[:, 2][None, None, :].astype(np.float32)
        # Along a Cartesian line only one q component varies, so compute W_box directly from margins.
        h_axis = half_box[axis_idx]
        w_this = np.where(np.abs(s_vals) <= h_axis, 1.0, 0.0)
        for i, s in enumerate(s_vals):
            rows.append(
                {
                    "nucleus_name": nucleus.name,
                    "axis": axis,
                    "coordinate_relative_nm": float(s),
                    "coordinate_global_nm": float(points[i, axis_idx]),
                    "phi": float(p[i]),
                    "xB_before": float(b[i]),
                    "xB_after": float(a[i]),
                    "delta_xB": float(a[i] - b[i]),
                    "xBtot_before": float(xtb[i]),
                    "xBtot_after": float(xta[i]),
                    "delta_xBtot": float(xta[i] - xtb[i]),
                    "W_box_for_this_nucleus": float(w_this[i]),
                    "W_box_max": float(wmax[i]),
                    "local_comp_weight": float(lw[i]),
                    "xB_edge_i": nucleus.xB_edge_i,
                    "source_box_boundary_position_nm": float(h_axis),
                    "local_comp_shell_inner_position_nm": float(h_axis + local_comp_inner_nm),
                    "local_comp_shell_outer_position_nm": float(h_axis + local_comp_outer_nm),
                }
            )
    csv_path = out_dir / f"per_nucleus_{nucleus.name}_profiles.csv"
    png_path = out_dir / f"per_nucleus_{nucleus.name}_profiles.png"
    write_summary_csv(csv_path, rows)
    fig, axs = plt.subplots(3, 4, figsize=(16, 10), constrained_layout=True)
    for ai, axis in enumerate(("x", "y", "z")):
        sub = [r for r in rows if r["axis"] == axis]
        s = np.array([r["coordinate_relative_nm"] for r in sub])
        p = np.array([r["phi"] for r in sub])
        b = np.array([r["xB_before"] for r in sub])
        a = np.array([r["xB_after"] for r in sub])
        dx = np.array([r["delta_xB"] for r in sub])
        xtb = np.array([r["xBtot_before"] for r in sub])
        xta = np.array([r["xBtot_after"] for r in sub])
        w = np.array([r["W_box_max"] for r in sub])
        lw = np.array([r["local_comp_weight"] for r in sub])
        h_axis = half_box[ai]
        markers = [h_axis, -h_axis, h_axis + local_comp_outer_nm, -(h_axis + local_comp_outer_nm)]
        if local_comp_inner_nm > 0:
            markers += [h_axis + local_comp_inner_nm, -(h_axis + local_comp_inner_nm)]
        axs[ai, 0].plot(s, p, label="phi")
        axs[ai, 0].plot(s, w, label="W_box_max")
        axs[ai, 1].plot(s, b, label="xB before")
        axs[ai, 1].plot(s, a, label="xB after")
        axs[ai, 1].axhline(nucleus.xB_edge_i, color="0.4", ls=":", lw=0.8, label="xB_edge")
        axs[ai, 2].plot(s, dx, label="delta xB")
        axs[ai, 2].plot(s, lw, label="local weight")
        axs[ai, 3].plot(s, xtb, label="xBtot before")
        axs[ai, 3].plot(s, xta, label="xBtot after")
        for ax in axs[ai]:
            for marker in markers:
                ax.axvline(marker, color="0.5", ls="--", lw=0.6)
            ax.grid(alpha=0.25)
            ax.set_title(f"{nucleus.name} {axis}-axis")
    for ax in axs[0]:
        ax.legend(fontsize=7, frameon=False)
    fig.savefig(png_path, dpi=dpi)
    plt.close(fig)


def write_histogram_summary(
    out_csv: Path,
    out_png: Path,
    *,
    phi_total: np.memmap,
    xB_before: np.memmap,
    xB_after: np.memmap,
    W_box_max: np.memmap,
    local_comp_weight: np.memmap,
    W_comp_threshold: float,
    phi_matrix_threshold: float,
    max_samples: int,
    dpi: int,
) -> None:
    phi = np.asarray(phi_total, dtype=np.float32).ravel()
    b = np.asarray(xB_before, dtype=np.float32).ravel()
    a = np.asarray(xB_after, dtype=np.float32).ravel()
    w = np.asarray(W_box_max, dtype=np.float32).ravel()
    lw = np.asarray(local_comp_weight, dtype=np.float32).ravel()
    hp = h_phi(phi)
    xtb = (1.0 - hp) * b + hp
    xta = (1.0 - hp) * a + hp
    dx = a - b
    dxt = xta - xtb
    regions = {
        "global": np.ones_like(phi, dtype=bool),
        "inside_source_boxes": w > W_comp_threshold,
        "local_comp_shell": lw > 1.0e-6,
        "far_outside_shell": (w < W_comp_threshold) & (lw < 1.0e-6) & (phi < phi_matrix_threshold),
    }
    rows: list[dict[str, Any]] = []
    for name, mask in regions.items():
        count = int(np.count_nonzero(mask))
        row: dict[str, Any] = {"region": name, "count": count, "fraction": count / phi.size}
        for label, arr in {
            "xB_before": b,
            "xB_after": a,
            "delta_xB": dx,
            "xBtot_before": xtb,
            "xBtot_after": xta,
            "delta_xBtot": dxt,
        }.items():
            vals = arr[mask]
            row[f"{label}_min"] = float(np.min(vals)) if vals.size else float("nan")
            row[f"{label}_max"] = float(np.max(vals)) if vals.size else float("nan")
            row[f"{label}_mean"] = float(np.mean(vals)) if vals.size else float("nan")
            row[f"{label}_std"] = float(np.std(vals)) if vals.size else float("nan")
        rows.append(row)
    write_summary_csv(out_csv, rows)
    rng = np.random.default_rng(1234)
    idx = np.arange(phi.size)
    if phi.size > max_samples:
        idx = rng.choice(idx, size=max_samples, replace=False)
    fig, axs = plt.subplots(2, 3, figsize=(14, 8), constrained_layout=True)
    plot_items = [
        ("xB_before", b),
        ("xB_after", a),
        ("delta_xB", dx),
        ("xBtot_before", xtb),
        ("xBtot_after", xta),
        ("delta_xBtot", dxt),
    ]
    for ax, (label, arr) in zip(axs.ravel(), plot_items):
        ax.hist(arr[idx], bins=160, histtype="step")
        ax.set_title(label)
        ax.grid(alpha=0.25)
    fig.savefig(out_png, dpi=dpi)
    plt.close(fig)
def compensate_far_field(
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
) -> tuple[list[dict[str, float]], float, list[str]]:
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
    return history, correction_total, warnings


def compensate_closed_box(
    phi_total: np.memmap,
    xB: np.memmap,
    *,
    xBtot_target: float,
    xB_min: float,
    xB_max: float,
    iters: int,
    tol: float,
    chunk_z: int,
) -> tuple[list[dict[str, float]], float, list[str]]:
    warnings = [
        "closed-box mode applies a global matrix-composition shift and may destroy near-nucleus depletion profiles; use only as a control."
    ]
    nx, ny, nz = phi_total.shape
    n = nx * ny * nz
    correction_total = 0.0
    history: list[dict[str, float]] = []
    for iteration in range(iters):
        mean_xbtot, _mean_xb, _mean_phi, _mean_h = compute_mean_xbtot(phi_total, xB, chunk_z=chunk_z)
        mass_error = xBtot_target - mean_xbtot
        denom_sum = 0.0
        for z0 in range(0, nz, chunk_z):
            z1 = min(nz, z0 + chunk_z)
            phi = np.asarray(phi_total[:, :, z0:z1], dtype=np.float32)
            denom_sum += float(np.sum(1.0 - h_phi(phi), dtype=np.float64))
        denom = denom_sum / n
        if denom <= 1.0e-20:
            warnings.append("closed-box denominator is too small; cannot compensate mass.")
            break
        c_global = mass_error / denom
        correction_total += c_global
        lower_clip = 0
        upper_clip = 0
        for z0 in range(0, nz, chunk_z):
            z1 = min(nz, z0 + chunk_z)
            xb = np.array(xB[:, :, z0:z1], dtype=np.float32, copy=True)
            xb += np.float32(c_global)
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
            "C_global": float(c_global),
            "C_global_total": float(correction_total),
            "lower_clipped_fraction": float(lower_clip / n),
            "upper_clipped_fraction": float(upper_clip / n),
            "mean_xBtot_after": float(final_mean),
            "mass_error_after": float(final_error),
        }
        history.append(rec)
        if rec["lower_clipped_fraction"] > 1.0e-4 or rec["upper_clipped_fraction"] > 1.0e-4:
            warnings.append(
                f"closed-box clipping fraction exceeded 1e-4 at iter {iteration}: "
                f"lower={rec['lower_clipped_fraction']:.3e}, upper={rec['upper_clipped_fraction']:.3e}"
            )
        if abs(final_error) <= tol:
            break
    return history, correction_total, warnings


def compensate_local_shell(
    phi_total: np.memmap,
    xB: np.memmap,
    local_comp_weight: np.memmap,
    *,
    xBtot_target: float,
    xB_min: float,
    xB_max: float,
    iters: int,
    tol: float,
    chunk_z: int,
) -> tuple[list[dict[str, float]], float, list[str]]:
    warnings: list[str] = []
    nx, ny, nz = phi_total.shape
    n = nx * ny * nz
    correction_total = 0.0
    history: list[dict[str, float]] = []
    for iteration in range(iters):
        mean_xbtot, _mean_xb, _mean_phi, _mean_h = compute_mean_xbtot(phi_total, xB, chunk_z=chunk_z)
        mass_error = xBtot_target - mean_xbtot
        denom_sum = 0.0
        shell_count = 0
        for z0 in range(0, nz, chunk_z):
            z1 = min(nz, z0 + chunk_z)
            phi = np.asarray(phi_total[:, :, z0:z1], dtype=np.float32)
            weight = np.asarray(local_comp_weight[:, :, z0:z1], dtype=np.float32)
            shell_mask = weight > 1.0e-12
            if np.any(shell_mask):
                hp = h_phi(phi)
                denom_sum += float(np.sum((1.0 - hp) * weight, dtype=np.float64))
                shell_count += int(np.count_nonzero(shell_mask))
        denom = denom_sum / n
        if denom <= 1.0e-20:
            warnings.append("local compensation shell is empty or denom too small; cannot compensate mass locally.")
            break
        c_local = mass_error / denom
        correction_total += c_local
        lower_clip = 0
        upper_clip = 0
        for z0 in range(0, nz, chunk_z):
            z1 = min(nz, z0 + chunk_z)
            weight = np.asarray(local_comp_weight[:, :, z0:z1], dtype=np.float32)
            xb = np.array(xB[:, :, z0:z1], dtype=np.float32, copy=True)
            xb += np.float32(c_local) * weight
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
            "C_local": float(c_local),
            "C_local_total": float(correction_total),
            "local_shell_fraction": float(shell_count / n),
            "lower_clipped_fraction": float(lower_clip / n),
            "upper_clipped_fraction": float(upper_clip / n),
            "mean_xBtot_after": float(final_mean),
            "mass_error_after": float(final_error),
        }
        history.append(rec)
        if abs(c_local) > 0.01:
            warnings.append(f"strong warning: |C_local|={abs(c_local):.4g} > 0.01; local compensation is physically questionable.")
        elif abs(c_local) > 0.003:
            warnings.append(f"warning: |C_local|={abs(c_local):.4g} > 0.003; local compensation is large.")
        if rec["lower_clipped_fraction"] > 1.0e-4 or rec["upper_clipped_fraction"] > 1.0e-4:
            warnings.append(
                f"local-shell clipping fraction exceeded 1e-4 at iter {iteration}: "
                f"lower={rec['lower_clipped_fraction']:.3e}, upper={rec['upper_clipped_fraction']:.3e}"
            )
        if abs(final_error) <= tol:
            break
    return history, correction_total, warnings


def write_center_profiles(
    out_csv: Path,
    out_png: Path,
    phi_total: np.memmap,
    xB: np.memmap,
    *,
    nuclei: list[RescaledNucleus],
    target_dx_nm: float,
    origin_nm: np.ndarray,
    xB_far: float,
) -> None:
    nx, ny, nz = phi_total.shape
    rows: list[dict[str, Any]] = []

    def add_axis(scope: str, center_nm: np.ndarray, axis: str) -> None:
        cx = int(np.clip(math.floor((center_nm[0] - origin_nm[0]) / target_dx_nm), 0, nx - 1))
        cy = int(np.clip(math.floor((center_nm[1] - origin_nm[1]) / target_dx_nm), 0, ny - 1))
        cz = int(np.clip(math.floor((center_nm[2] - origin_nm[2]) / target_dx_nm), 0, nz - 1))
        if axis == "x":
            coords = origin_nm[0] + (np.arange(nx) + 0.5) * target_dx_nm - center_nm[0]
            p = np.asarray(phi_total[:, cy, cz])
            xb = np.asarray(xB[:, cy, cz])
        elif axis == "y":
            coords = origin_nm[1] + (np.arange(ny) + 0.5) * target_dx_nm - center_nm[1]
            p = np.asarray(phi_total[cx, :, cz])
            xb = np.asarray(xB[cx, :, cz])
        else:
            coords = origin_nm[2] + (np.arange(nz) + 0.5) * target_dx_nm - center_nm[2]
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
                    "coordinate_nm_from_center": float(coords[i]),
                    "phi": float(p[i]),
                    "xB": float(xb[i]),
                    "xBtot": float(xt[i]),
                }
            )

    box_center = origin_nm + np.array([nx, ny, nz], dtype=np.float64) * target_dx_nm * 0.5
    for axis in ("x", "y", "z"):
        add_axis("target_box_center", box_center, axis)
    for nucleus in nuclei:
        for axis in ("x", "y", "z"):
            add_axis(nucleus.name, nucleus.center_nm, axis)

    write_summary_csv(out_csv, rows)
    fig, axs = plt.subplots(3, 2, figsize=(13, 11), constrained_layout=True)
    for scope in ["target_box_center"] + [n.name for n in nuclei]:
        if scope != "target_box_center" and len(nuclei) > 8:
            continue
        sub = [row for row in rows if row["scope"] == scope]
        for ai, axis in enumerate(("x", "y", "z")):
            axis_rows = [row for row in sub if row["axis"] == axis]
            coord = np.array([row["coordinate_nm_from_center"] for row in axis_rows], dtype=float)
            phi = np.array([row["phi"] for row in axis_rows], dtype=float)
            xb = np.array([row["xB"] for row in axis_rows], dtype=float)
            axs[ai, 0].plot(coord, phi, lw=1.2, label=scope)
            axs[ai, 1].plot(coord, xb, lw=1.2, label=scope)
            axs[ai, 1].axhline(xB_far, color="0.4", lw=0.8, ls=":")
            axs[ai, 0].set_title(f"{axis}-axis phi")
            axs[ai, 1].set_title(f"{axis}-axis xB")
            axs[ai, 0].grid(alpha=0.25)
            axs[ai, 1].grid(alpha=0.25)
    axs[0, 0].legend(fontsize=7, frameon=False)
    axs[0, 1].legend(fontsize=7, frameon=False)
    fig.savefig(out_png, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Physically rescaled multi-nucleus embedding: source 400^3 dx=0.1 nm remains a 40 nm object in target coordinates."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    parser.add_argument("--chunk-z", type=int, default=None)
    parser.add_argument("--dtype", choices=("float32", "float64"), default=None)
    parser.add_argument("--write-vtk", action="store_true", default=None)
    parser.add_argument("--write-plots", action="store_true", default=None)
    parser.add_argument("--write-comparison-plots", action="store_true", default=None)
    parser.add_argument("--plot-dpi", type=int, default=None)
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

    nx, ny, nz = _target_shape(global_cfg)
    target_dx_nm = float(global_cfg.get("target_dx_nm", global_cfg.get("dx_nm", 1.0)))
    target_interface_width_nm = (
        float(global_cfg["target_interface_width_nm"])
        if global_cfg.get("target_interface_width_nm") is not None
        else None
    )
    source_default_dx_nm = float(global_cfg.get("source_default_dx_nm", 0.1))
    source_default_interface_width_nm = float(global_cfg.get("source_default_interface_width_nm", 0.6))
    origin_nm = np.array(global_cfg.get("origin_nm", [0.0, 0.0, 0.0]), dtype=np.float64)
    xB_background_constant = float(global_cfg.get("xB_background_constant", global_cfg.get("xB_far", 0.030)))
    xB_far = float(global_cfg.get("xB_far", xB_background_constant))
    xB0_nominal = float(global_cfg.get("xB0_target_nominal", xB_far))
    xBtot_target_mode = str(global_cfg.get("xBtot_target_mode", "nominal"))
    if xBtot_target_mode == "manual":
        xBtot_target = float(global_cfg["xBtot_target_manual"])
    elif xBtot_target_mode == "nominal":
        xBtot_target = xB0_nominal
    else:
        raise ValueError(f"Unsupported xBtot_target_mode: {xBtot_target_mode}")
    xB_min = float(global_cfg.get("xB_min", 1.0e-8))
    xB_max = float(global_cfg.get("xB_max", 0.035))
    mass_mode = str(global_cfg.get("mass_mode", "far-field-compensate"))
    W_comp_threshold = float(global_cfg.get("W_comp_threshold", 1.0e-3))
    phi_matrix_threshold = float(global_cfg.get("phi_matrix_threshold", 0.05))
    mass_iters = int(global_cfg.get("mass_correction_iters", 10))
    mass_tol = float(global_cfg.get("mass_tol", 1.0e-9))
    write_vtk = bool(global_cfg.get("write_vtk", False)) if args.write_vtk is None else bool(args.write_vtk)
    write_plots = bool(global_cfg.get("write_plots", True)) if args.write_plots is None else bool(args.write_plots)
    write_comparison_plots = (
        bool(global_cfg.get("write_comparison_plots", write_plots))
        if args.write_comparison_plots is None
        else bool(args.write_comparison_plots)
    )
    plot_dpi = int(args.plot_dpi if args.plot_dpi is not None else global_cfg.get("plot_dpi", 180))
    plot_slices = list(global_cfg.get("plot_slices", ["mid_x", "mid_y", "mid_z"]))
    plot_nucleus_local_slices = bool(global_cfg.get("plot_nucleus_local_slices", True))
    plot_center_axis_profiles = bool(global_cfg.get("plot_center_axis_profiles", True))
    plot_per_nucleus_profiles = bool(global_cfg.get("plot_per_nucleus_profiles", True))
    plot_histograms = bool(global_cfg.get("plot_histograms", True))
    histogram_max_samples = int(global_cfg.get("histogram_max_samples", 5_000_000))
    local_comp_inner_nm = float(global_cfg.get("local_comp_inner_nm", global_cfg.get("local_shell_inner_nm", 0.0)))
    local_comp_outer_nm = float(global_cfg.get("local_comp_outer_nm", global_cfg.get("local_shell_outer_nm", 10.0)))
    dtype = np.dtype(args.dtype or global_cfg.get("dtype", "float32"))
    xB_combine_mode = args.xB_combine_mode or str(global_cfg.get("xB_combine_mode", "min"))
    phi_combine_mode = str(global_cfg.get("phi_combine_mode", "max"))
    if phi_combine_mode != "max":
        raise ValueError("Only phi_combine_mode=max is implemented.")
    if xB_combine_mode not in {"min", "weighted-average"}:
        raise ValueError(f"Unsupported xB_combine_mode: {xB_combine_mode}")
    chunk_z = args.chunk_z or int(global_cfg.get("chunk_z") or (32 if nx * ny * nz >= 256**3 else nz))
    chunk_z = max(1, min(chunk_z, nz))

    nuclei = [
        nuc
        for idx, raw in enumerate(cfg.get("nuclei", []))
        if (
            nuc := load_nucleus(
                raw,
                idx,
                config_dir=config_dir,
                global_cfg=global_cfg,
                target_dx_nm=target_dx_nm,
                target_interface_width_nm=target_interface_width_nm,
                xB_far=xB_far,
            )
        )
        is not None
    ]
    if not nuclei:
        raise ValueError("No enabled nuclei in config.")

    shape = (nx, ny, nz)
    phi_total = memmap_array(work_dir / "phi_total.dat", shape, dtype, 0.0)
    xB = memmap_array(work_dir / "xB.dat", shape, dtype, 0.0)
    background_info, background_warnings = initialize_background(
        xB,
        global_cfg=global_cfg,
        config_dir=config_dir,
        target_shape=shape,
        target_dx_nm=target_dx_nm,
        constant_default=xB_background_constant,
        chunk_z=chunk_z,
    )
    edge_warnings = resolve_nucleus_edges(
        nuclei,
        xB,
        global_cfg=global_cfg,
        background_info=background_info,
        target_dx_nm=target_dx_nm,
        origin_nm=origin_nm,
        xB_min=xB_min,
        xB_max=xB_max,
        chunk_z=chunk_z,
    )
    W_box_max = memmap_array(work_dir / "W_box_max.dat", shape, dtype, 0.0)
    local_comp_weight = memmap_array(work_dir / "local_comp_weight.dat", shape, dtype, 0.0)
    active_phi_count = memmap_array(work_dir / "active_phi_count.dat", shape, np.uint16, 0)
    active_source_box_count = memmap_array(work_dir / "active_source_box_count.dat", shape, np.uint16, 0)
    if xB_combine_mode == "weighted-average":
        weight_sum = memmap_array(work_dir / "weight_sum.dat", shape, dtype, 1.0)
        weighted_sum = memmap_array(work_dir / "weighted_sum.dat", shape, dtype, 0.0)
        for z0 in range(0, nz, chunk_z):
            z1 = min(nz, z0 + chunk_z)
            weighted_sum[:, :, z0:z1] = np.asarray(xB[:, :, z0:z1], dtype=dtype)
    else:
        weight_sum = None
        weighted_sum = None

    x_coords = (origin_nm[0] + (np.arange(nx, dtype=np.float32) + np.float32(0.5)) * np.float32(target_dx_nm))
    y_coords = (origin_nm[1] + (np.arange(ny, dtype=np.float32) + np.float32(0.5)) * np.float32(target_dx_nm))

    print(
        f"[info] rescaled embedding {len(nuclei)} nuclei into target {nx}x{ny}x{nz}, "
        f"target_dx_nm={target_dx_nm:g}, chunk_z={chunk_z}"
    )
    for nucleus in nuclei:
        print(
            f"[info] {nucleus.name}: source_box_nm={nucleus.source_box_size_nm.tolist()}, "
            f"target_cells={nucleus.source_box_size_target_cells.tolist()}, "
            f"lambda_source_nm={nucleus.source_interface_width_nm:g}, "
            f"lambda_target_nm={nucleus.target_interface_width_nm:g}, "
            f"lambda_scale={nucleus.scale_factor_interface_width:g}, "
            f"R_target_nm~{float(np.mean(nucleus.semiaxes_target_nm)):.3g}, "
            f"xB_edge={nucleus.xB_edge_i:.6g} ({nucleus.xB_edge_mode})"
        )

    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        z_coords = origin_nm[2] + (np.arange(z0, z1, dtype=np.float32) + np.float32(0.5)) * np.float32(target_dx_nm)
        phi_chunk = np.array(phi_total[:, :, z0:z1], dtype=np.float32, copy=True)
        xb_chunk = np.array(xB[:, :, z0:z1], dtype=np.float32, copy=True)
        wmax_chunk = np.array(W_box_max[:, :, z0:z1], dtype=np.float32, copy=True)
        local_comp_chunk = np.array(local_comp_weight[:, :, z0:z1], dtype=np.float32, copy=True)
        active_phi_chunk = np.array(active_phi_count[:, :, z0:z1], dtype=np.uint16, copy=True)
        active_source_chunk = np.array(active_source_box_count[:, :, z0:z1], dtype=np.uint16, copy=True)
        if xB_combine_mode == "weighted-average":
            wsum_chunk = np.array(weight_sum[:, :, z0:z1], dtype=np.float32, copy=True)  # type: ignore[index]
            xsum_chunk = np.array(weighted_sum[:, :, z0:z1], dtype=np.float32, copy=True)  # type: ignore[index]

        for nucleus in nuclei:
            d_target_nm, family_idx, _rho, q = chunk_geometry_physical(
                nucleus=nucleus,
                x_nm=x_coords,
                y_nm=y_coords,
                z_nm=z_coords,
            )
            phi_i, xB_local = rebuild_rescaled_profile_chunk(nucleus, d_target_nm, family_idx)
            W_box = source_box_window_physical(nucleus, *q)
            W_local_shell = source_box_shell_weight_physical(
                nucleus,
                *q,
                inner_nm=local_comp_inner_nm,
                outer_nm=local_comp_outer_nm,
            )
            xB_edge = np.float32(nucleus.xB_edge_i)
            xB_candidate = (xB_edge + W_box * (xB_local - xB_edge)).astype(np.float32)
            update_mask = W_box > 0.0

            if xB_combine_mode == "min":
                if np.any(update_mask):
                    xb_chunk[update_mask] = np.minimum(xb_chunk[update_mask], xB_candidate[update_mask])
            else:
                wsum_chunk += W_box
                xsum_chunk += W_box * xB_candidate
            phi_chunk = np.maximum(phi_chunk, phi_i)
            wmax_chunk = np.maximum(wmax_chunk, W_box)
            local_comp_chunk = np.maximum(local_comp_chunk, W_local_shell)
            active_phi_chunk += (phi_i > 0.05).astype(np.uint16)
            active_source_chunk += (W_box > W_comp_threshold).astype(np.uint16)

            nucleus.accum["W_box_nonzero_points"] += float(np.count_nonzero(W_box > 1.0e-6))
            nucleus.accum["phi_gt_0p5_points"] += float(np.count_nonzero(phi_i > 0.5))
            nucleus.accum["phi_gt_0p05_points"] += float(np.count_nonzero(phi_i > 0.05))
            nucleus.accum["max_phi_contribution"] = max(nucleus.accum["max_phi_contribution"], float(np.max(phi_i)))
            near_mask = (W_box > 0.999) & (phi_i < 0.05)
            interface_mask = (W_box > 0.999) & (phi_i > 0.1) & (phi_i < 0.9)
            boundary_mask = (W_box > W_comp_threshold) & (W_box < 0.999)
            transition_mask = (W_box > 0.0) & (W_box < 0.999)
            if np.any(near_mask):
                nucleus.accum["near_matrix_xB_sum"] += float(np.sum(xB_candidate[near_mask], dtype=np.float64))
                nucleus.accum["near_matrix_xB_count"] += float(np.count_nonzero(near_mask))
            if np.any(interface_mask):
                nucleus.accum["interface_xB_sum"] += float(np.sum(xB_candidate[interface_mask], dtype=np.float64))
                nucleus.accum["interface_xB_count"] += float(np.count_nonzero(interface_mask))
            if np.any(boundary_mask):
                vals = xB_candidate[boundary_mask]
                nucleus.accum["boundary_xB_before_sum"] += float(np.sum(vals, dtype=np.float64))
                nucleus.accum["boundary_xB_before_count"] += float(vals.size)
                nucleus.accum["boundary_xB_before_min"] = min(nucleus.accum["boundary_xB_before_min"], float(np.min(vals)))
                nucleus.accum["boundary_xB_before_max"] = max(nucleus.accum["boundary_xB_before_max"], float(np.max(vals)))
            if np.any(transition_mask):
                vals = xB_candidate[transition_mask]
                nucleus.accum["transition_xB_min"] = min(nucleus.accum["transition_xB_min"], float(np.min(vals)))
                nucleus.accum["transition_xB_max"] = max(nucleus.accum["transition_xB_max"], float(np.max(vals)))

        if xB_combine_mode == "weighted-average":
            xb_chunk = xsum_chunk / np.maximum(wsum_chunk, 1.0e-12)
            weight_sum[:, :, z0:z1] = wsum_chunk  # type: ignore[index]
            weighted_sum[:, :, z0:z1] = xsum_chunk  # type: ignore[index]
        np.clip(xb_chunk, xB_min, xB_max, out=xb_chunk)
        phi_total[:, :, z0:z1] = phi_chunk.astype(dtype, copy=False)
        xB[:, :, z0:z1] = xb_chunk.astype(dtype, copy=False)
        W_box_max[:, :, z0:z1] = wmax_chunk.astype(dtype, copy=False)
        local_comp_weight[:, :, z0:z1] = local_comp_chunk.astype(dtype, copy=False)
        active_phi_count[:, :, z0:z1] = active_phi_chunk
        active_source_box_count[:, :, z0:z1] = active_source_chunk

    phi_total.flush()
    xB.flush()
    W_box_max.flush()
    local_comp_weight.flush()
    active_phi_count.flush()
    active_source_box_count.flush()

    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        wmax = np.asarray(W_box_max[:, :, z0:z1], dtype=np.float32)
        lw = np.array(local_comp_weight[:, :, z0:z1], dtype=np.float32, copy=True)
        lw[wmax >= W_comp_threshold] = 0.0
        local_comp_weight[:, :, z0:z1] = lw.astype(dtype, copy=False)
    local_comp_weight.flush()

    xB_before_compensation = memmap_array(work_dir / "xB_before_compensation.dat", shape, dtype, 0.0)
    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        xB_before_compensation[:, :, z0:z1] = np.asarray(xB[:, :, z0:z1], dtype=dtype)
    xB_before_compensation.flush()

    stats_before = collect_stats(
        phi_total,
        xB_before_compensation,
        W_box_max,
        active_phi_count,
        active_source_box_count,
        xBtot_target=xBtot_target,
        xB_min=xB_min,
        xB_max=xB_max,
        W_comp_threshold=W_comp_threshold,
        phi_matrix_threshold=phi_matrix_threshold,
        chunk_z=chunk_z,
    )
    recompute_boundary_xB_stats(
        nuclei,
        xB_before_compensation,
        phase="before",
        target_dx_nm=target_dx_nm,
        origin_nm=origin_nm,
        W_comp_threshold=W_comp_threshold,
        chunk_z=chunk_z,
    )
    warnings: list[str] = [*background_warnings, *edge_warnings]
    if stats_before["overlap_fraction_phi_i_gt_0p05"] > 0.0:
        warnings.append(f"phi overlap detected: {stats_before['overlap_fraction_phi_i_gt_0p05']:.3e}")
    if stats_before["overlap_fraction_source_box_gt_threshold"] > 0.0:
        warnings.append(f"source-box influence overlap detected: {stats_before['overlap_fraction_source_box_gt_threshold']:.3e}")

    correction_history: list[dict[str, float]] = []
    correction_total = 0.0
    if mass_mode == "far-field-compensate":
        correction_history, correction_total, mass_warnings = compensate_far_field(
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
    elif mass_mode == "closed-box":
        correction_history, correction_total, mass_warnings = compensate_closed_box(
            phi_total,
            xB,
            xBtot_target=xBtot_target,
            xB_min=xB_min,
            xB_max=xB_max,
            iters=mass_iters,
            tol=mass_tol,
            chunk_z=chunk_z,
        )
        warnings.extend(mass_warnings)
    elif mass_mode == "local-shell-compensate":
        correction_history, correction_total, mass_warnings = compensate_local_shell(
            phi_total,
            xB,
            local_comp_weight,
            xBtot_target=xBtot_target,
            xB_min=xB_min,
            xB_max=xB_max,
            iters=mass_iters,
            tol=mass_tol,
            chunk_z=chunk_z,
        )
        warnings.extend(mass_warnings)
    elif mass_mode == "embedding-report":
        pass
    else:
        raise ValueError(
            f"Unsupported mass_mode: {mass_mode}. Use embedding-report, far-field-compensate, local-shell-compensate, or closed-box."
        )

    stats_after = collect_stats(
        phi_total,
        xB,
        W_box_max,
        active_phi_count,
        active_source_box_count,
        xBtot_target=xBtot_target,
        xB_min=xB_min,
        xB_max=xB_max,
        W_comp_threshold=W_comp_threshold,
        phi_matrix_threshold=phi_matrix_threshold,
        chunk_z=chunk_z,
    )
    delta_diagnostics = compute_delta_diagnostics(
        phi_total,
        xB_before_compensation,
        xB,
        W_box_max,
        local_comp_weight,
        W_comp_threshold=W_comp_threshold,
        phi_matrix_threshold=phi_matrix_threshold,
        chunk_z=chunk_z,
    )
    if abs(stats_after["mass_error_vs_target"]) > mass_tol and mass_mode in {"far-field-compensate", "local-shell-compensate"}:
        warnings.append(f"final mass error {stats_after['mass_error_vs_target']:.6e} exceeds mass_tol={mass_tol:.3e}")
    if delta_diagnostics["max_abs_delta_xB_inside_source_boxes"] > 1.0e-6:
        warnings.append(
            f"max_abs_delta_xB_inside_source_boxes={delta_diagnostics['max_abs_delta_xB_inside_source_boxes']:.3e} > 1e-6"
        )
    if delta_diagnostics["max_abs_delta_xB_far_outside_shell"] > 1.0e-6 and mass_mode == "local-shell-compensate":
        warnings.append(
            f"max_abs_delta_xB_far_outside_shell={delta_diagnostics['max_abs_delta_xB_far_outside_shell']:.3e} > 1e-6"
        )
    frac_shell = delta_diagnostics["fraction_of_delta_mass_in_local_comp_shell"]
    if mass_mode == "local-shell-compensate" and math.isfinite(frac_shell) and abs(frac_shell) < 0.95:
        warnings.append(f"fraction_of_delta_mass_in_local_comp_shell={frac_shell:.6g} < 0.95")

    recompute_boundary_xB_stats(
        nuclei,
        xB,
        phase="after",
        target_dx_nm=target_dx_nm,
        origin_nm=origin_nm,
        W_comp_threshold=W_comp_threshold,
        chunk_z=chunk_z,
    )

    total_points = float(nx * ny * nz)
    nuclei_rows: list[dict[str, Any]] = []
    for nucleus in nuclei:
        near_count = nucleus.accum["near_matrix_xB_count"]
        interface_count = nucleus.accum["interface_xB_count"]
        boundary_before_count = nucleus.accum["boundary_xB_before_count"]
        boundary_after_count = nucleus.accum["boundary_xB_after_count"]
        boundary_before_mean = (
            nucleus.accum["boundary_xB_before_sum"] / boundary_before_count if boundary_before_count else float("nan")
        )
        boundary_after_mean = (
            nucleus.accum["boundary_xB_after_sum"] / boundary_after_count if boundary_after_count else float("nan")
        )
        if (
            math.isfinite(boundary_before_mean)
            and math.isfinite(boundary_after_mean)
            and abs(boundary_after_mean - boundary_before_mean) > 1.0e-5
            and mass_mode == "far-field-compensate"
        ):
            warnings.append(
                f"nucleus {nucleus.name}: protected boundary xB changed by "
                f"{boundary_after_mean - boundary_before_mean:.3e}; check W_comp_threshold/far_mask."
            )
        source_radius_est = (
            (3.0 * nucleus.accum["phi_gt_0p5_points"] * target_dx_nm**3 / (4.0 * math.pi)) ** (1.0 / 3.0)
            if nucleus.accum["phi_gt_0p5_points"] > 0
            else float("nan")
        )
        nuclei_rows.append(
            {
                "name": nucleus.name,
                "center_nm": " ".join(f"{v:.6g}" for v in nucleus.center_nm),
                "source_dyn_dir": str(nucleus.source_dyn_dir) if nucleus.source_dyn_dir else "",
                "profile_dir": str(nucleus.profile_dir),
                "source_grid_N": " ".join(str(v) for v in nucleus.source_grid_n),
                "source_dx_nm": nucleus.source_dx_nm,
                "target_dx_nm": target_dx_nm,
                "source_interface_width_nm": nucleus.source_interface_width_nm,
                "target_interface_width_nm": nucleus.target_interface_width_nm,
                "effective_interface_width_target_cells": nucleus.target_interface_width_nm / target_dx_nm,
                "source_box_size_nm": " ".join(f"{v:.6g}" for v in nucleus.source_box_size_nm),
                "source_box_size_in_target_cells": " ".join(f"{v:.6g}" for v in nucleus.source_box_size_target_cells),
                "scale_factor_geometry": nucleus.scale_factor_geometry,
                "scale_factor_interface_width": nucleus.scale_factor_interface_width,
                "scale_factor_xB_profile_width": nucleus.scale_factor_xB_profile_width,
                "scale_source_box_with_geometry": nucleus.scale_source_box_with_geometry,
                "semiaxes_source_nm": " ".join(f"{v:.6g}" for v in nucleus.semiaxes_source_nm),
                "semiaxes_target_nm": " ".join(f"{v:.6g}" for v in nucleus.semiaxes_target_nm),
                "target_radius_mean_grid_cells": float(np.mean(nucleus.semiaxes_target_nm) / target_dx_nm),
                "target_equiv_radius_from_embedded_phi_nm": source_radius_est,
                "xB_matrix_near": nucleus.xB_matrix_near,
                "xB_matrix_reference": nucleus.xB_matrix_reference,
                "xB_edge_mode_used": nucleus.xB_edge_mode,
                "xB_edge_value_config": nucleus.xB_edge_value_config if nucleus.xB_edge_value_config is not None else "",
                "xB_edge_i_used": nucleus.xB_edge_i,
                "xB_edge_sample_stat": nucleus.xB_edge_sample_stat,
                "xB_edge_sample_count": nucleus.xB_edge_sample_count,
                "xB_edge_sample_mean": nucleus.xB_edge_sample_mean,
                "xB_edge_sample_median": nucleus.xB_edge_sample_median,
                "local_matrix_xB_estimate": nucleus.accum["near_matrix_xB_sum"] / near_count if near_count else float("nan"),
                "local_interface_xB_estimate": nucleus.accum["interface_xB_sum"] / interface_count if interface_count else float("nan"),
                "boundary_transition_xB_mean_before_compensation": boundary_before_mean,
                "boundary_transition_xB_mean_after_compensation": boundary_after_mean,
                "boundary_transition_xB_delta_mean": (
                    boundary_after_mean - boundary_before_mean
                    if math.isfinite(boundary_before_mean) and math.isfinite(boundary_after_mean)
                    else float("nan")
                ),
                "boundary_transition_xB_min_before": nucleus.accum["boundary_xB_before_min"],
                "boundary_transition_xB_max_before": nucleus.accum["boundary_xB_before_max"],
                "boundary_transition_xB_min_after": nucleus.accum["boundary_xB_after_min"],
                "boundary_transition_xB_max_after": nucleus.accum["boundary_xB_after_max"],
                "transition_xB_min_within_W_0_1": nucleus.accum["transition_xB_min"],
                "transition_xB_max_within_W_0_1": nucleus.accum["transition_xB_max"],
                "source_actual_mean_xBtot": nucleus.source_stats.get("source_xBtot_mean", ""),
                "W_box_nonzero_fraction": nucleus.accum["W_box_nonzero_points"] / total_points,
                "max_phi_contribution": nucleus.accum["max_phi_contribution"],
                "profile_file": nucleus.profile_info.get("profile_file", ""),
            }
        )
    write_summary_csv(out_dir / "nuclei_summary.csv", nuclei_rows)

    diagnostics = {
        "Global": {
            "target_N": [nx, ny, nz],
            "target_dx_nm": target_dx_nm,
            "target_interface_width_nm": target_interface_width_nm,
            "source_default_dx_nm": source_default_dx_nm,
            "source_default_interface_width_nm": source_default_interface_width_nm,
            "effective_interface_width_target_cells": (
                target_interface_width_nm / target_dx_nm if target_interface_width_nm is not None else None
            ),
            "origin_nm": origin_nm.tolist(),
            "target_box_size_nm": [nx * target_dx_nm, ny * target_dx_nm, nz * target_dx_nm],
            "number_of_nuclei": len(nuclei),
            "xB_far_requested": xB_far,
            "xB0_target_nominal": xB0_nominal,
            "xBtot_target_mode": xBtot_target_mode,
            "xBtot_target_used": xBtot_target,
            "mass_mode": mass_mode,
            "xB_combine_mode": xB_combine_mode,
            "phi_combine_mode": phi_combine_mode,
            "dtype": str(dtype),
            "chunk_z": chunk_z,
            "local_comp_inner_nm": local_comp_inner_nm,
            "local_comp_outer_nm": local_comp_outer_nm,
            "write_comparison_plots": write_comparison_plots,
        },
        "Background": {
            **background_info,
            "xB_edge_mode_default": str(global_cfg.get("xB_edge_mode_default", "global-constant")),
            "xB_edge_shell_inner_margin_nm": float(global_cfg.get("xB_edge_shell_inner_margin_nm", 0.0)),
            "xB_edge_shell_outer_margin_nm": float(global_cfg.get("xB_edge_shell_outer_margin_nm", 3.0)),
            "edge_sample_stat": str(global_cfg.get("edge_sample_stat", "mean")),
        },
        "Before compensation": stats_before,
        "After compensation": {
            **stats_after,
            "final_mass_error": stats_after["mass_error_vs_target"],
            "C_far_per_iteration": [r["C_far"] for r in correction_history if "C_far" in r],
            "C_local_per_iteration": [r["C_local"] for r in correction_history if "C_local" in r],
            "C_global_per_iteration": [r["C_global"] for r in correction_history if "C_global" in r],
            "final_C_far_total": correction_total if mass_mode == "far-field-compensate" else 0.0,
            "final_C_local_total": correction_total if mass_mode == "local-shell-compensate" else 0.0,
            "final_C_global_total": correction_total if mass_mode == "closed-box" else 0.0,
            "xB_far_after_mass_compensation": xB_far + correction_total if mass_mode == "far-field-compensate" else xB_far,
            "mass_converged": abs(stats_after["mass_error_vs_target"]) <= mass_tol,
            "max_abs_phi_after_minus_before": 0.0,
        },
        "Delta": delta_diagnostics,
        "Mass redistribution": {
            "total_delta_mass_global": delta_diagnostics["total_delta_mass_global"],
            "delta_mass_inside_source_boxes": delta_diagnostics["delta_mass_inside_source_boxes"],
            "delta_mass_local_comp_shell": delta_diagnostics["delta_mass_local_comp_shell"],
            "delta_mass_far_outside_shell": delta_diagnostics["delta_mass_far_outside_shell"],
            "fraction_of_delta_mass_in_local_comp_shell": delta_diagnostics["fraction_of_delta_mass_in_local_comp_shell"],
        },
        "Source target scale": nuclei_rows,
        "Overlap": {
            "active_phi_count_max": stats_after["active_phi_count_max"],
            "active_source_box_count_max": stats_after["active_source_box_count_max"],
            "overlap_fraction_phi_i_gt_0p05": stats_after["overlap_fraction_phi_i_gt_0p05"],
            "overlap_fraction_source_box_gt_threshold": stats_after["overlap_fraction_source_box_gt_threshold"],
        },
        "Mass correction history": correction_history,
        "Warnings": warnings,
    }
    (out_dir / "diagnostics.json").write_text(json.dumps(diagnostics, indent=2), encoding="utf-8")
    write_diagnostics_txt(out_dir / "diagnostics.txt", diagnostics)

    write_before_after_fields(
        out_dir,
        write_vtk=write_vtk,
        phi_total=phi_total,
        xB_before=xB_before_compensation,
        xB_after=xB,
        W_box_max=W_box_max,
        local_comp_weight=local_comp_weight,
        active_source_box_count=active_source_box_count,
        active_phi_count=active_phi_count,
        target_dx_nm=target_dx_nm,
        origin_nm=origin_nm,
        chunk_z=chunk_z,
    )

    if write_vtk:
        write_vtk_scalar_chunked(
            out_dir / "far_mask.vtk",
            shape,
            "far_mask",
            target_dx_nm,
            origin_nm,
            chunk_z,
            lambda z0, z1: (
                (np.asarray(phi_total[:, :, z0:z1]) < phi_matrix_threshold)
                & (np.asarray(W_box_max[:, :, z0:z1]) < W_comp_threshold)
            ).astype(np.float32),
        )

    if write_comparison_plots:
        for plane in plot_slices:
            if plane not in {"mid_x", "mid_y", "mid_z"}:
                warnings.append(f"unknown plot slice {plane}; skipped")
                continue
            out_name = out_dir / f"global_slice_{plane}_comparison.png"
            plot_global_slice_comparison(
                out_name,
                plane=plane,
                phi_total=phi_total,
                xB_before=xB_before_compensation,
                xB_after=xB,
                W_box_max=W_box_max,
                local_comp_weight=local_comp_weight,
                active_source_box_count=active_source_box_count,
                nuclei=nuclei,
                target_dx_nm=target_dx_nm,
                origin_nm=origin_nm,
                local_comp_outer_nm=local_comp_outer_nm,
                dpi=plot_dpi,
            )
        if (out_dir / "global_slice_mid_z_comparison.png").exists():
            (out_dir / "global_compensation_slices.png").write_bytes((out_dir / "global_slice_mid_z_comparison.png").read_bytes())
        if plot_center_axis_profiles:
            write_center_axis_before_after(
                out_dir / "center_axis_before_after_profiles.csv",
                out_dir / "center_axis_before_after_profiles.png",
                phi_total=phi_total,
                xB_before=xB_before_compensation,
                xB_after=xB,
                W_box_max=W_box_max,
                local_comp_weight=local_comp_weight,
                active_source_box_count=active_source_box_count,
                target_dx_nm=target_dx_nm,
                origin_nm=origin_nm,
                xB_background_constant=float(background_info.get("xB_background_constant", xB_far)),
                dpi=plot_dpi,
            )
        if plot_per_nucleus_profiles:
            for nucleus in nuclei:
                write_per_nucleus_profiles(
                    out_dir,
                    nucleus=nucleus,
                    phi_total=phi_total,
                    xB_before=xB_before_compensation,
                    xB_after=xB,
                    W_box_max=W_box_max,
                    local_comp_weight=local_comp_weight,
                    target_dx_nm=target_dx_nm,
                    origin_nm=origin_nm,
                    local_comp_inner_nm=local_comp_inner_nm,
                    local_comp_outer_nm=local_comp_outer_nm,
                    dpi=plot_dpi,
                )
        if plot_histograms:
            write_histogram_summary(
                out_dir / "compensation_histograms.csv",
                out_dir / "compensation_histograms.png",
                phi_total=phi_total,
                xB_before=xB_before_compensation,
                xB_after=xB,
                W_box_max=W_box_max,
                local_comp_weight=local_comp_weight,
                W_comp_threshold=W_comp_threshold,
                phi_matrix_threshold=phi_matrix_threshold,
                max_samples=histogram_max_samples,
                dpi=plot_dpi,
            )

    print(f"out_dir={out_dir}")
    print(f"target={nx}x{ny}x{nz}, target_dx_nm={target_dx_nm:g}")
    print(f"nuclei={len(nuclei)}")
    print(f"mean_xBtot_before={stats_before['mean_xBtot']:.10e}")
    print(f"mean_xBtot_after={stats_after['mean_xBtot']:.10e}")
    print(f"mass_error_after={stats_after['mass_error_vs_target']:.10e}")
    correction_label = (
        "C_far_total"
        if mass_mode == "far-field-compensate"
        else "C_local_total"
        if mass_mode == "local-shell-compensate"
        else "C_global_total"
        if mass_mode == "closed-box"
        else "C_total"
    )
    print(f"{correction_label}={correction_total:.10e}")
    print(f"far_mask_fraction={stats_after['far_mask_fraction']:.6e}")
    print(f"warnings={len(warnings)}")
    for warning in warnings:
        print(f"[warn] {warning}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
