#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from collections import deque
from pathlib import Path
import sys
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.generate_continue_dynamic_geometry_summaries import _read_legacy_scalar_vtk


def h_phi(phi: np.ndarray) -> np.ndarray:
    p = np.clip(phi, 0.0, 1.0)
    return 6.0 * p**5 - 15.0 * p**4 + 10.0 * p**3


def write_vtk_scalar(path: Path, data: np.ndarray, name: str, spacing: tuple[float, float, float]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny, nz = data.shape
    with path.open("w", encoding="ascii") as f:
        f.write("# vtk DataFile Version 3.0\n")
        f.write(f"{name}\n")
        f.write("ASCII\n")
        f.write("DATASET STRUCTURED_POINTS\n")
        f.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        f.write("ORIGIN 0 0 0\n")
        f.write(f"SPACING {spacing[0]:.12g} {spacing[1]:.12g} {spacing[2]:.12g}\n")
        f.write(f"POINT_DATA {nx * ny * nz}\n")
        f.write(f"SCALARS {name} double 1\n")
        f.write("LOOKUP_TABLE default\n")
        flat = data.ravel(order="C")
        for start in range(0, flat.size, 6):
            f.write(" ".join(f"{v:.10e}" for v in flat[start:start + 6]))
            f.write("\n")


def trilinear_resample(
    field: np.ndarray,
    src_spacing: tuple[float, float, float],
    dst_shape: tuple[int, int, int],
    dst_spacing: tuple[float, float, float],
    src_center_nm: np.ndarray,
    dst_center_nm: np.ndarray,
) -> np.ndarray:
    nx, ny, nz = dst_shape
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    points = np.stack(
        [
            (ii + 0.5) * dst_spacing[0] - dst_center_nm[0] + src_center_nm[0],
            (jj + 0.5) * dst_spacing[1] - dst_center_nm[1] + src_center_nm[1],
            (kk + 0.5) * dst_spacing[2] - dst_center_nm[2] + src_center_nm[2],
        ],
        axis=-1,
    )

    sx, sy, sz = src_spacing
    gx = np.clip(points[..., 0] / sx - 0.5, 0.0, field.shape[0] - 1.000001)
    gy = np.clip(points[..., 1] / sy - 0.5, 0.0, field.shape[1] - 1.000001)
    gz = np.clip(points[..., 2] / sz - 0.5, 0.0, field.shape[2] - 1.000001)

    x0 = np.floor(gx).astype(int)
    y0 = np.floor(gy).astype(int)
    z0 = np.floor(gz).astype(int)
    x1 = np.clip(x0 + 1, 0, field.shape[0] - 1)
    y1 = np.clip(y0 + 1, 0, field.shape[1] - 1)
    z1 = np.clip(z0 + 1, 0, field.shape[2] - 1)

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


def centroid_from_phi(phi: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    w = h_phi(phi)
    total = float(np.sum(w))
    if total <= 1.0e-14:
        return 0.5 * np.array(phi.shape, dtype=float) * np.array(spacing, dtype=float)
    coords = [
        (np.arange(phi.shape[axis], dtype=float) + 0.5) * spacing[axis]
        for axis in range(3)
    ]
    wx = np.sum(w, axis=(1, 2))
    wy = np.sum(w, axis=(0, 2))
    wz = np.sum(w, axis=(0, 1))
    return np.array(
        [
            float(np.dot(coords[0], wx) / total),
            float(np.dot(coords[1], wy) / total),
            float(np.dot(coords[2], wz) / total),
        ],
        dtype=float,
    )


def smooth_sphere_field(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    center_nm: np.ndarray,
    radius_nm: float,
    interface_width_nm: float,
) -> np.ndarray:
    nx, ny, nz = shape
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    x = (ii + 0.5) * spacing[0] - center_nm[0]
    y = (jj + 0.5) * spacing[1] - center_nm[1]
    z = (kk + 0.5) * spacing[2] - center_nm[2]
    r = np.sqrt(x * x + y * y + z * z)
    width = max(interface_width_nm, 1.0e-12)
    return np.clip(0.5 * (1.0 - np.tanh((r - radius_nm) / width)), 0.0, 1.0)


def target_h_volume(r_eff_nm: float) -> float:
    return 4.0 * math.pi * r_eff_nm**3 / 3.0


def reconstruct_exact_volume_phi(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    center_nm: np.ndarray,
    r_eff_nm: float,
    interface_width_nm: float,
) -> tuple[np.ndarray, float]:
    voxel = spacing[0] * spacing[1] * spacing[2]
    target = target_h_volume(r_eff_nm)
    lo = max(0.0, r_eff_nm - 4.0 * interface_width_nm - max(spacing))
    hi = r_eff_nm + 4.0 * interface_width_nm + max(spacing)
    best_phi = smooth_sphere_field(shape, spacing, center_nm, r_eff_nm, interface_width_nm)
    best_radius = r_eff_nm
    best_err = abs(float(np.sum(h_phi(best_phi)) * voxel) - target)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        phi = smooth_sphere_field(shape, spacing, center_nm, mid, interface_width_nm)
        vol = float(np.sum(h_phi(phi)) * voxel)
        err = abs(vol - target)
        if err < best_err:
            best_phi = phi
            best_radius = mid
            best_err = err
        if vol < target:
            lo = mid
        else:
            hi = mid
    return best_phi, best_radius


def reconstruct_diffuse_kernel_exact_volume(
    shape: tuple[int, int, int],
    spacing: tuple[float, float, float],
    center_nm: np.ndarray,
    r_eff_nm: float,
    min_resolvable_radius_nm: float,
    interface_width_nm: float,
) -> tuple[np.ndarray, float, float]:
    nx, ny, nz = shape
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    x = (ii + 0.5) * spacing[0] - center_nm[0]
    y = (jj + 0.5) * spacing[1] - center_nm[1]
    z = (kk + 0.5) * spacing[2] - center_nm[2]
    r2 = x * x + y * y + z * z
    sigma_nm = max(0.5 * min_resolvable_radius_nm, interface_width_nm, min(spacing))
    kernel = np.exp(-0.5 * r2 / max(sigma_nm * sigma_nm, 1.0e-30))
    voxel = float(np.prod(spacing))
    target = target_h_volume(r_eff_nm)
    lo = 0.0
    hi = 1.0
    best_amp = 1.0
    best_phi = kernel
    best_err = abs(float(np.sum(h_phi(kernel)) * voxel) - target)
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        phi = np.clip(mid * kernel, 0.0, 1.0)
        vol = float(np.sum(h_phi(phi)) * voxel)
        err = abs(vol - target)
        if err < best_err:
            best_err = err
            best_amp = mid
            best_phi = phi
        if vol < target:
            lo = mid
        else:
            hi = mid
    return best_phi, sigma_nm, best_amp


def largest_component_count(mask: np.ndarray) -> tuple[int, int]:
    visited = np.zeros(mask.shape, dtype=np.uint8)
    neighbors = [(1, 0, 0), (-1, 0, 0), (0, 1, 0), (0, -1, 0), (0, 0, 1), (0, 0, -1)]
    ncomp = 0
    largest = 0
    nx, ny, nz = mask.shape
    for seed in np.argwhere(mask):
        sx, sy, sz = [int(v) for v in seed]
        if visited[sx, sy, sz]:
            continue
        ncomp += 1
        q: deque[tuple[int, int, int]] = deque([(sx, sy, sz)])
        visited[sx, sy, sz] = 1
        count = 0
        while q:
            i, j, k = q.popleft()
            count += 1
            for di, dj, dk in neighbors:
                ni, nj, nk = i + di, j + dj, k + dk
                if ni < 0 or nj < 0 or nk < 0 or ni >= nx or nj >= ny or nk >= nz:
                    continue
                if visited[ni, nj, nk] or not mask[ni, nj, nk]:
                    continue
                visited[ni, nj, nk] = 1
                q.append((ni, nj, nk))
        largest = max(largest, count)
    return ncomp, largest


def compensate_xb_mass(
    xB: np.ndarray,
    phi: np.ndarray,
    total_before: float,
    xB_min: float,
    xB_max: float,
    shell_inner_phi: float,
    shell_outer_phi: float,
    max_iter: int = 16,
) -> tuple[np.ndarray, dict[str, float]]:
    out = np.clip(np.array(xB, dtype=float, copy=True), xB_min, xB_max)
    shell = (phi <= shell_inner_phi) & (phi >= shell_outer_phi)
    if not np.any(shell):
        shell = phi < shell_inner_phi
    weights = (1.0 - h_phi(phi)) * shell.astype(float)
    active = weights > 1.0e-14
    for _ in range(max_iter):
        err = total_before - float(np.sum(out))
        denom = float(np.sum(weights[active]))
        if abs(err) <= 1.0e-10 * max(abs(total_before), 1.0):
            break
        if denom <= 1.0e-30:
            break
        out[active] = np.clip(out[active] + err * weights[active] / denom, xB_min, xB_max)
    total_after = float(np.sum(out))
    return out, {
        "total_xB_before": float(total_before),
        "total_xB_after": total_after,
        "total_xB_error": float(total_after - total_before),
        "compensation_active_fraction": float(np.mean(active)),
    }


def gradient_max(phi: np.ndarray, spacing: tuple[float, float, float]) -> float:
    grads = np.gradient(phi, *spacing, edge_order=1)
    mag = np.sqrt(grads[0] ** 2 + grads[1] ** 2 + grads[2] ** 2)
    return float(np.max(mag))


def build_nucleus(args: argparse.Namespace) -> dict[str, Any]:
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    phi_src, _, src_spacing = _read_legacy_scalar_vtk(args.phi_vtk.expanduser().resolve())
    if args.xb_vtk:
        xB_src, _, _ = _read_legacy_scalar_vtk(args.xb_vtk.expanduser().resolve())
    else:
        xB_src = np.full(phi_src.shape, args.matrix_xb, dtype=float)

    dst_spacing = (args.dx_nm, args.dy_nm, args.dz_nm)
    if args.nx and args.ny and args.nz:
        dst_shape = (args.nx, args.ny, args.nz)
    else:
        pad = max(args.padding_nm, 4.0 * args.interface_width_nm, 3.0 * max(dst_spacing))
        extent = 2.0 * (max(args.r_eff_nm, args.min_resolvable_radius_dx * args.dx_nm) + pad)
        n = max(8, int(math.ceil(extent / args.dx_nm)))
        dst_shape = (n, n, n)

    dst_center = 0.5 * np.array(dst_shape, dtype=float) * np.array(dst_spacing, dtype=float)
    src_center = centroid_from_phi(phi_src, src_spacing)
    r_over_dx = args.r_eff_nm / min(dst_spacing)
    scale_mode = "direct_resample" if r_over_dx >= args.min_resolvable_radius_dx else "diffuse_reconstruction"

    xB_base = trilinear_resample(xB_src, src_spacing, dst_shape, dst_spacing, src_center, dst_center)
    total_xB_before = float(np.sum(xB_base))

    if scale_mode == "direct_resample":
        phi_candidate = trilinear_resample(phi_src, src_spacing, dst_shape, dst_spacing, src_center, dst_center)
        phi_candidate = np.clip(phi_candidate, 0.0, 1.0)
        vol = float(np.sum(h_phi(phi_candidate)) * np.prod(dst_spacing))
        if vol <= 1.0e-30 or abs(vol - target_h_volume(args.r_eff_nm)) / target_h_volume(args.r_eff_nm) > args.volume_tol_rel:
            phi_init, effective_radius = reconstruct_exact_volume_phi(
                dst_shape, dst_spacing, dst_center, args.r_eff_nm, args.interface_width_nm
            )
            scale_mode = "direct_resample_volume_rebuilt"
        else:
            phi_init = phi_candidate
            effective_radius = args.r_eff_nm
    else:
        min_radius = args.min_resolvable_radius_dx * min(dst_spacing)
        phi_init, diffuse_sigma_nm, diffuse_amplitude = reconstruct_diffuse_kernel_exact_volume(
            dst_shape,
            dst_spacing,
            dst_center,
            args.r_eff_nm,
            min_radius,
            max(args.interface_width_nm, min(dst_spacing)),
        )
        effective_radius = args.r_eff_nm
        scale_mode = "diffuse_reconstruction_min_radius_projection"

    xB_init, mass_meta = compensate_xb_mass(
        xB_base,
        phi_init,
        total_xB_before,
        args.xb_min,
        args.xb_max,
        args.shell_inner_phi,
        args.shell_outer_phi,
    )

    voxel_volume = float(np.prod(dst_spacing))
    h_volume = float(np.sum(h_phi(phi_init)) * voxel_volume)
    actual_component_threshold = min(args.component_threshold, 0.5 * float(np.max(phi_init)))
    components, largest = largest_component_count(phi_init > actual_component_threshold)
    phi_grad_max = gradient_max(phi_init, dst_spacing)
    volume_error = h_volume - target_h_volume(args.r_eff_nm)
    rel_volume_error = volume_error / max(target_h_volume(args.r_eff_nm), 1.0e-30)
    smooth_limit = 1.5 / max(args.interface_width_nm, min(dst_spacing))
    insertion_ready = (
        np.isfinite(phi_init).all()
        and np.isfinite(xB_init).all()
        and float(np.min(phi_init)) >= -1.0e-10
        and float(np.max(phi_init)) <= 1.0 + 1.0e-10
        and components == 1
        and abs(mass_meta["total_xB_error"]) <= args.mass_tol_abs
        and abs(rel_volume_error) <= args.volume_tol_rel
        and phi_grad_max <= smooth_limit
    )

    np.save(output_dir / "phi_init.npy", phi_init.astype(np.float64))
    np.save(output_dir / "xB_init.npy", xB_init.astype(np.float64))
    write_vtk_scalar(output_dir / "phi_init.vtk", phi_init, "phi", dst_spacing)
    write_vtk_scalar(output_dir / "xB_init.vtk", xB_init, "xB", dst_spacing)

    metadata: dict[str, Any] = {
        "schema_version": 1,
        "generator": "nucleus_generator/nucleus_resampler.py",
        "source_phi_vtk": str(args.phi_vtk),
        "source_xB_vtk": str(args.xb_vtk) if args.xb_vtk else "",
        "nx": dst_shape[0],
        "ny": dst_shape[1],
        "nz": dst_shape[2],
        "dx_nm": dst_spacing[0],
        "dy_nm": dst_spacing[1],
        "dz_nm": dst_spacing[2],
        "r_eff_nm": args.r_eff_nm,
        "r_eff_over_dx": r_over_dx,
        "effective_reconstruction_radius_nm": effective_radius,
        "diffuse_sigma_nm": float(diffuse_sigma_nm) if "diffuse_sigma_nm" in locals() else None,
        "diffuse_amplitude": float(diffuse_amplitude) if "diffuse_amplitude" in locals() else None,
        "minimum_resolvable_radius_nm": args.min_resolvable_radius_dx * min(dst_spacing),
        "interface_width_nm": args.interface_width_nm,
        "center_nm": [float(v) for v in dst_center],
        "semiaxes_nm": [args.r_eff_nm, args.r_eff_nm, args.r_eff_nm],
        "scale_mode": scale_mode,
        "scale_bridge_rule": "r_eff/dx < min_resolvable_radius_dx uses diffuse reconstruction; otherwise direct resampling is attempted",
        "target_volume_nm3": target_h_volume(args.r_eff_nm),
        "h_volume_nm3": h_volume,
        "h_volume_error_nm3": volume_error,
        "h_volume_relative_error": rel_volume_error,
        "phi_min": float(np.min(phi_init)),
        "phi_max": float(np.max(phi_init)),
        "phi_grad_max": phi_grad_max,
        "phi_grad_limit": smooth_limit,
        "connected_components_phi_gt_threshold": components,
        "largest_component_voxels": largest,
        "component_threshold": actual_component_threshold,
        "requested_component_threshold": args.component_threshold,
        "mass_conservation": mass_meta,
        "xB_min": float(np.min(xB_init)),
        "xB_max": float(np.max(xB_init)),
        "xB_mean": float(np.mean(xB_init)),
        "insertion_ready": bool(insertion_ready),
        "readiness_criteria": {
            "finite_fields": bool(np.isfinite(phi_init).all() and np.isfinite(xB_init).all()),
            "phi_in_unit_interval": bool(float(np.min(phi_init)) >= -1.0e-10 and float(np.max(phi_init)) <= 1.0 + 1.0e-10),
            "single_connected_component": bool(components == 1),
            "mass_error_within_tolerance": bool(abs(mass_meta["total_xB_error"]) <= args.mass_tol_abs),
            "volume_error_within_tolerance": bool(abs(rel_volume_error) <= args.volume_tol_rel),
            "smoothness_within_limit": bool(phi_grad_max <= smooth_limit),
        },
        "outputs": {
            "phi_npy": "phi_init.npy",
            "xB_npy": "xB_init.npy",
            "phi_vtk": "phi_init.vtk",
            "xB_vtk": "xB_init.vtk",
            "metadata": "nucleus_metadata.json",
        },
    }
    (output_dir / "nucleus_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps({
        "output_dir": str(output_dir),
        "scale_mode": scale_mode,
        "r_eff_over_dx": r_over_dx,
        "insertion_ready": insertion_ready,
        "mass_error": mass_meta["total_xB_error"],
        "h_volume_relative_error": rel_volume_error,
    }, indent=2))
    return metadata


def main() -> int:
    parser = argparse.ArgumentParser(description="Build a CUDA-ready nucleus initial-condition object from minimized/dynamic nucleus fields.")
    parser.add_argument("--phi-vtk", type=Path, required=True)
    parser.add_argument("--xb-vtk", type=Path, default=None)
    parser.add_argument("--r-eff-nm", type=float, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--dx-nm", type=float, required=True)
    parser.add_argument("--dy-nm", type=float, default=None)
    parser.add_argument("--dz-nm", type=float, default=None)
    parser.add_argument("--nx", type=int, default=0)
    parser.add_argument("--ny", type=int, default=0)
    parser.add_argument("--nz", type=int, default=0)
    parser.add_argument("--interface-width-nm", type=float, default=0.3)
    parser.add_argument("--matrix-xb", type=float, default=0.03)
    parser.add_argument("--xb-min", type=float, default=1.0e-9)
    parser.add_argument("--xb-max", type=float, default=0.999999)
    parser.add_argument("--padding-nm", type=float, default=2.0)
    parser.add_argument("--min-resolvable-radius-dx", type=float, default=3.0)
    parser.add_argument("--component-threshold", type=float, default=0.5)
    parser.add_argument("--shell-inner-phi", type=float, default=0.45)
    parser.add_argument("--shell-outer-phi", type=float, default=1.0e-4)
    parser.add_argument("--mass-tol-abs", type=float, default=1.0e-8)
    parser.add_argument("--volume-tol-rel", type=float, default=5.0e-3)
    parser.add_argument("--volume-rebuild-tol", type=float, default=5.0e-2)
    args = parser.parse_args()
    if args.dy_nm is None:
        args.dy_nm = args.dx_nm
    if args.dz_nm is None:
        args.dz_nm = args.dx_nm
    if args.r_eff_nm <= 0.0:
        raise ValueError("--r-eff-nm must be positive")
    if args.dx_nm <= 0.0 or args.dy_nm <= 0.0 or args.dz_nm <= 0.0:
        raise ValueError("grid spacing must be positive")
    if any(v < 0 for v in (args.nx, args.ny, args.nz)):
        raise ValueError("grid dimensions must be non-negative")
    if any(v > 0 for v in (args.nx, args.ny, args.nz)) and not all(v > 0 for v in (args.nx, args.ny, args.nz)):
        raise ValueError("--nx/--ny/--nz must be provided together")
    build_nucleus(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
