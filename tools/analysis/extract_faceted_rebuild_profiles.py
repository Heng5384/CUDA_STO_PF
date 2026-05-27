#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.analyze_cnt_peak_table import parse_summary_file
from tools.analysis.generate_continue_dynamic_geometry_summaries import _read_legacy_scalar_vtk


def _resolve_xb_vtk(phi_vtk: Path) -> Path | None:
    name = phi_vtk.name
    for old, new in (("phi_final_", "xB_final_"), ("phi_", "xB_"), ("phi_init_", "xB_init_")):
        if name.startswith(old):
            cand = phi_vtk.with_name(new + name[len(old):])
            if cand.exists():
                return cand
    for cand in sorted(phi_vtk.parent.glob("xB*.vtk")):
        return cand
    return None


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


def _largest_component(mask: np.ndarray) -> np.ndarray:
    from collections import deque

    nx, ny, nz = mask.shape
    visited = np.zeros(mask.shape, dtype=np.uint8)
    best: list[tuple[int, int, int]] = []
    neighbors = [(dx, dy, dz) for dx in (-1, 0, 1) for dy in (-1, 0, 1) for dz in (-1, 0, 1) if (dx, dy, dz) != (0, 0, 0)]

    for sx, sy, sz in np.argwhere(mask):
        if visited[sx, sy, sz]:
            continue
        q: deque[tuple[int, int, int]] = deque([(int(sx), int(sy), int(sz))])
        visited[sx, sy, sz] = 1
        comp: list[tuple[int, int, int]] = []
        while q:
            x, y, z = q.popleft()
            comp.append((x, y, z))
            for dx, dy, dz in neighbors:
                xn, yn, zn = x + dx, y + dy, z + dz
                if xn < 0 or yn < 0 or zn < 0 or xn >= nx or yn >= ny or zn >= nz:
                    continue
                if visited[xn, yn, zn] or not mask[xn, yn, zn]:
                    continue
                visited[xn, yn, zn] = 1
                q.append((xn, yn, zn))
        if len(comp) > len(best):
            best = comp

    out = np.zeros(mask.shape, dtype=bool)
    if best:
        idx = np.array(best, dtype=int)
        out[idx[:, 0], idx[:, 1], idx[:, 2]] = True
    return out


def _extract_boundary_points_and_normals(phi: np.ndarray, spacing: tuple[float, float, float], threshold: float) -> tuple[np.ndarray, np.ndarray]:
    mask = _largest_component(phi > threshold)
    selected_local = np.argwhere(mask)
    if selected_local.size == 0:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)

    gx, gy, gz = np.gradient(phi, *spacing, edge_order=1)
    nx, ny, nz = mask.shape
    neighbors6 = [(-1, 0, 0), (1, 0, 0), (0, -1, 0), (0, 1, 0), (0, 0, -1), (0, 0, 1)]
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []

    for lx, ly, lz in selected_local:
        is_boundary = False
        for ddx, ddy, ddz in neighbors6:
            xn, yn, zn = lx + ddx, ly + ddy, lz + ddz
            if xn < 0 or yn < 0 or zn < 0 or xn >= nx or yn >= ny or zn >= nz or not mask[xn, yn, zn]:
                is_boundary = True
                break
        if not is_boundary:
            continue
        normal = np.array([-gx[lx, ly, lz], -gy[lx, ly, lz], -gz[lx, ly, lz]], dtype=float)
        nn = np.linalg.norm(normal)
        if nn < 1.0e-12:
            continue
        normal /= nn
        points.append(np.array([lx * spacing[0], ly * spacing[1], lz * spacing[2]], dtype=float))
        normals.append(normal)

    if not points:
        return np.zeros((0, 3), dtype=float), np.zeros((0, 3), dtype=float)
    return np.array(points, dtype=float), np.array(normals, dtype=float)


def _octant_directions() -> tuple[list[str], np.ndarray]:
    labels: list[str] = []
    vecs: list[np.ndarray] = []
    for sx in (-1, 1):
        for sy in (-1, 1):
            for sz in (-1, 1):
                labels.append(f"{sx:+d}{sy:+d}{sz:+d}")
                v = np.array([sx, sy, sz], dtype=float)
                v /= np.linalg.norm(v)
                vecs.append(v)
    return labels, np.array(vecs, dtype=float)


def _save_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _classify_region_from_planes(points_local: np.ndarray, family_dirs: np.ndarray, plane_tol_nm: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    supports = points_local @ family_dirs.T
    h = supports.max(axis=0)
    gaps = h[None, :] - supports
    near = gaps <= plane_tol_nm
    near_count = np.sum(near, axis=1)
    region = np.full(points_local.shape[0], "face", dtype=object)
    region[near_count == 2] = "edge"
    region[near_count >= 3] = "corner"
    family_assign = np.argmin(gaps, axis=1)
    return region, family_assign, gaps


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract 8-family faceted phi/xB rebuild profiles from one dynamic nucleus.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--phi-vtk", type=Path, default=None)
    parser.add_argument("--xb-vtk", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--inside-range-nm", type=float, default=2.0)
    parser.add_argument("--outside-range-nm", type=float, default=2.0)
    parser.add_argument("--step-nm", type=float, default=0.05)
    parser.add_argument("--max-samples-per-family", type=int, default=1500)
    parser.add_argument("--family-cone-deg", type=float, default=28.0)
    parser.add_argument("--plane-tol-nm", type=float, default=0.35, help="distance to supporting planes for face/edge/corner classification")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    summary_path = args.summary.expanduser().resolve()
    info = parse_summary_file(summary_path)
    phi_vtk = args.phi_vtk.expanduser().resolve() if args.phi_vtk else Path(str(info["phi_vtk_file"])).expanduser().resolve()
    xb_vtk = args.xb_vtk.expanduser().resolve() if args.xb_vtk else _resolve_xb_vtk(phi_vtk)
    if xb_vtk is None or not xb_vtk.exists():
        raise FileNotFoundError(f"Cannot locate xB vtk for {phi_vtk}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (summary_path.parent / "_tmp_faceted_rebuild_profiles")
    output_dir.mkdir(parents=True, exist_ok=True)

    phi, _, spacing_vtk = _read_legacy_scalar_vtk(phi_vtk)
    xb, _, _ = _read_legacy_scalar_vtk(xb_vtk)
    spacing = (
        float(info.get("spacing_x", spacing_vtk[0])),
        float(info.get("spacing_y", spacing_vtk[1])),
        float(info.get("spacing_z", spacing_vtk[2])),
    )

    boundary_points, boundary_normals = _extract_boundary_points_and_normals(phi, spacing, args.threshold)
    if boundary_points.shape[0] == 0:
        raise RuntimeError("No boundary points found.")

    axes = np.array([
        [float(info["long_axis_x"]), float(info["long_axis_y"]), float(info["long_axis_z"])],
        [float(info["mid_axis_x"]), float(info["mid_axis_y"]), float(info["mid_axis_z"])],
        [float(info["short_axis_x"]), float(info["short_axis_y"]), float(info["short_axis_z"])],
    ], dtype=float)
    normals_local = boundary_normals @ axes.T
    center = np.array([float(info["center_x"]), float(info["center_y"]), float(info["center_z"])], dtype=float)
    points_local = (boundary_points - center[None, :]) @ axes.T

    family_labels, family_dirs = _octant_directions()
    dots = normals_local @ family_dirs.T
    region_labels, family_assign, plane_gaps = _classify_region_from_planes(points_local, family_dirs, args.plane_tol_nm)
    family_best_dot = np.max(dots, axis=1)
    cone_cos = math.cos(math.radians(args.family_cone_deg))

    rng = np.random.default_rng(args.seed)
    u_grid = np.arange(-args.inside_range_nm, args.outside_range_nm + 0.5 * args.step_nm, args.step_nm)

    family_rows: list[dict[str, object]] = []
    meta_rows: list[dict[str, object]] = []
    profile_rows: list[dict[str, object]] = []
    raw_rows: list[dict[str, object]] = []

    fig, axes_plot = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    cmap = plt.get_cmap("tab10")
    region_order = ["face", "edge", "corner"]

    for region_idx, region_name in enumerate(region_order):
        for family_idx, label in enumerate(family_labels):
            sel = np.where((family_assign == family_idx) & (family_best_dot >= cone_cos) & (region_labels == region_name))[0]
            if sel.size == 0:
                continue
            if sel.size > args.max_samples_per_family:
                sel = rng.choice(sel, size=args.max_samples_per_family, replace=False)

            pts = boundary_points[sel]
            nrm = boundary_normals[sel]
            local = normals_local[sel]
            mean_local = np.mean(local, axis=0)
            mean_local /= np.linalg.norm(mean_local)
            mean_world = axes.T @ mean_local
            mean_world /= np.linalg.norm(mean_world)

            sample_phi = np.empty((sel.size, u_grid.size), dtype=float)
            sample_xb = np.empty((sel.size, u_grid.size), dtype=float)

            for i in range(sel.size):
                points = pts[i][None, :] + u_grid[:, None] * nrm[i][None, :]
                sample_phi[i] = _trilinear_sample(phi, spacing, points)
                sample_xb[i] = _trilinear_sample(xb, spacing, points)

            phi_mean = np.mean(sample_phi, axis=0)
            phi_std = np.std(sample_phi, axis=0)
            xb_mean = np.mean(sample_xb, axis=0)
            xb_std = np.std(sample_xb, axis=0)

            color = cmap(family_idx % 10)
            axes_plot[region_idx, 0].plot(u_grid, phi_mean, lw=1.6, color=color, label=label)
            axes_plot[region_idx, 1].plot(u_grid, xb_mean, lw=1.6, color=color, label=label)

            meta_rows.append(
                {
                    "family": label,
                    "region": region_name,
                    "count": int(sel.size),
                    "fraction_of_boundary": float(sel.size / boundary_points.shape[0]),
                    "mean_dir_long": float(mean_local[0]),
                    "mean_dir_mid": float(mean_local[1]),
                    "mean_dir_short": float(mean_local[2]),
                    "mean_world_x": float(mean_world[0]),
                    "mean_world_y": float(mean_world[1]),
                    "mean_world_z": float(mean_world[2]),
                    "mean_dot_to_family_dir": float(np.mean(family_best_dot[sel])),
                    "mean_plane_gap_nm": float(np.mean(plane_gaps[sel, family_idx])),
                }
            )
            for u, pm, ps, xm, xs in zip(u_grid, phi_mean, phi_std, xb_mean, xb_std):
                profile_rows.append(
                    {
                        "family": label,
                        "region": region_name,
                        "u_nm": float(u),
                        "phi_mean": float(pm),
                        "phi_std": float(ps),
                        "xB_mean": float(xm),
                        "xB_std": float(xs),
                        "sample_count": int(sel.size),
                    }
                )
            sample_take = min(40, sel.size)
            for i in range(sample_take):
                for u, ph, xb_val in zip(u_grid, sample_phi[i], sample_xb[i]):
                    raw_rows.append(
                        {
                            "family": label,
                            "region": region_name,
                            "sample_id": int(i),
                            "u_nm": float(u),
                            "phi": float(ph),
                            "xB": float(xb_val),
                        }
                    )

    for region_idx, region_name in enumerate(region_order):
        axes_plot[region_idx, 0].set_title(f"{region_name} profiles: phi")
        axes_plot[region_idx, 1].set_title(f"{region_name} profiles: xB")
        axes_plot[region_idx, 0].set_ylabel("phi")
        axes_plot[region_idx, 1].set_ylabel("xB")
        for col in range(2):
            axes_plot[region_idx, col].grid(True, alpha=0.25)
            axes_plot[region_idx, col].legend(frameon=False, ncol=2, fontsize=8)
    axes_plot[-1, 0].set_xlabel("u along local outward normal (nm)")
    axes_plot[-1, 1].set_xlabel("u along local outward normal (nm)")

    plot_png = output_dir / "faceted_rebuild_profiles.png"
    fig.suptitle(f"8-family + face/edge/corner rebuild profiles: {summary_path.parent.name}", fontsize=13)
    fig.savefig(plot_png, dpi=180)
    plt.close(fig)

    meta_json = output_dir / "faceted_family_metadata.json"
    meta_json.write_text(
        json.dumps(
            {
                "summary_path": str(summary_path),
                "phi_vtk": str(phi_vtk),
                "xb_vtk": str(xb_vtk),
                "boundary_count": int(boundary_points.shape[0]),
                "family_cone_deg": args.family_cone_deg,
                "plane_tol_nm": args.plane_tol_nm,
                "inside_range_nm": args.inside_range_nm,
                "outside_range_nm": args.outside_range_nm,
                "step_nm": args.step_nm,
                "families": meta_rows,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    _save_csv(output_dir / "faceted_family_profiles.csv", ["family", "region", "u_nm", "phi_mean", "phi_std", "xB_mean", "xB_std", "sample_count"], profile_rows)
    _save_csv(output_dir / "faceted_family_raw_samples.csv", ["family", "region", "sample_id", "u_nm", "phi", "xB"], raw_rows)
    _save_csv(output_dir / "faceted_family_metadata.csv", ["family", "region", "count", "fraction_of_boundary", "mean_dir_long", "mean_dir_mid", "mean_dir_short", "mean_world_x", "mean_world_y", "mean_world_z", "mean_dot_to_family_dir", "mean_plane_gap_nm"], meta_rows)

    print(f"summary={summary_path}")
    print(f"phi_vtk={phi_vtk}")
    print(f"xb_vtk={xb_vtk}")
    print(f"plot={plot_png}")
    print(f"metadata_json={meta_json}")
    print(f"profiles_csv={output_dir / 'faceted_family_profiles.csv'}")
    print(f"raw_samples_csv={output_dir / 'faceted_family_raw_samples.csv'}")
    print(f"metadata_csv={output_dir / 'faceted_family_metadata.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
