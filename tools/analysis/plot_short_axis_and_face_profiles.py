#!/usr/bin/env python3
from __future__ import annotations

import argparse
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


def _find_nearest_crossing(s: np.ndarray, phi: np.ndarray, threshold: float) -> float | None:
    candidates: list[float] = []
    for i in range(len(s) - 1):
        pi = phi[i]
        pj = phi[i + 1]
        if pi == threshold:
            candidates.append(float(s[i]))
            continue
        if (pi - threshold) * (pj - threshold) <= 0.0 and pi != pj:
            t = (threshold - pi) / (pj - pi)
            candidates.append(float(s[i] + t * (s[i + 1] - s[i])))
    if not candidates:
        return None
    return min(candidates, key=abs)


def _interp_1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x, y, left=np.nan, right=np.nan)


def _vector_from_info(info: dict[str, object], prefix: str) -> np.ndarray:
    return np.array([
        float(info[f"{prefix}_x"]),
        float(info[f"{prefix}_y"]),
        float(info[f"{prefix}_z"]),
    ], dtype=float)


def main() -> int:
    parser = argparse.ArgumentParser(description="Plot centered short-axis full cut and +/- short-face local profiles.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--phi-vtk", type=Path, default=None)
    parser.add_argument("--xb-vtk", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--center-half-range-nm", type=float, default=10.0)
    parser.add_argument("--face-inside-range-nm", type=float, default=2.5)
    parser.add_argument("--face-outside-range-nm", type=float, default=2.5)
    parser.add_argument("--step-nm", type=float, default=0.05)
    args = parser.parse_args()

    summary_path = args.summary.expanduser().resolve()
    info = parse_summary_file(summary_path)
    phi_vtk = args.phi_vtk.expanduser().resolve() if args.phi_vtk else Path(str(info["phi_vtk_file"])).expanduser().resolve()
    xb_vtk = args.xb_vtk.expanduser().resolve() if args.xb_vtk else _resolve_xb_vtk(phi_vtk)
    if xb_vtk is None or not xb_vtk.exists():
        raise FileNotFoundError(f"Cannot locate xB vtk for {phi_vtk}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (summary_path.parent / "_tmp_short_views")
    output_dir.mkdir(parents=True, exist_ok=True)

    phi, _, spacing_vtk = _read_legacy_scalar_vtk(phi_vtk)
    xb, _, _ = _read_legacy_scalar_vtk(xb_vtk)
    spacing = (
        float(info.get("spacing_x", spacing_vtk[0])),
        float(info.get("spacing_y", spacing_vtk[1])),
        float(info.get("spacing_z", spacing_vtk[2])),
    )

    center = np.array([float(info["center_x"]), float(info["center_y"]), float(info["center_z"])], dtype=float)
    short_axis = _vector_from_info(info, "short_axis")
    short_axis /= np.linalg.norm(short_axis)

    s_center = np.arange(-args.center_half_range_nm, args.center_half_range_nm + 0.5 * args.step_nm, args.step_nm)
    points_center = center[None, :] + s_center[:, None] * short_axis[None, :]
    phi_center = _trilinear_sample(phi, spacing, points_center)
    xb_center = _trilinear_sample(xb, spacing, points_center)

    fig_center, axs_center = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    axs_center[0].plot(s_center, phi_center, color="#2c3e50", lw=2.0)
    axs_center[1].plot(s_center, xb_center, color="#2c3e50", lw=2.0)
    axs_center[0].set_title("Centered short-axis full cut: phi")
    axs_center[1].set_title("Centered short-axis full cut: xB")
    axs_center[0].set_xlabel("s along short axis (nm)")
    axs_center[1].set_xlabel("s along short axis (nm)")
    axs_center[0].set_ylabel("phi")
    axs_center[1].set_ylabel("xB")
    for ax in axs_center:
        ax.grid(True, alpha=0.25)
    fig_center.suptitle(f"Centered short-axis full cut: {summary_path.parent.name}", fontsize=13)
    centered_png = output_dir / "centered_short_axis_full_cut.png"
    fig_center.savefig(centered_png, dpi=180)
    plt.close(fig_center)

    face_names = ["+short", "-short"]
    face_points = {
        "+short": _vector_from_info(info, "short_face_point"),
        "-short": 2.0 * center - _vector_from_info(info, "short_face_point"),
    }
    face_normals = {
        "+short": _vector_from_info(info, "short_face_normal"),
        "-short": -_vector_from_info(info, "short_face_normal"),
    }

    u_grid = np.arange(-args.face_inside_range_nm, args.face_outside_range_nm + 0.5 * args.step_nm, args.step_nm)
    xi_grid = np.arange(-args.face_inside_range_nm, args.face_outside_range_nm + 0.5 * args.step_nm, args.step_nm)

    fig_face, axs_face = plt.subplots(1, 2, figsize=(12, 4.6), constrained_layout=True)
    colors = {"+short": "#c0392b", "-short": "#2980b9"}
    rows: list[dict[str, object]] = []
    for face_name in face_names:
        p0 = face_points[face_name]
        n0 = face_normals[face_name]
        n0 = n0 / np.linalg.norm(n0)
        points = p0[None, :] + u_grid[:, None] * n0[None, :]
        phi_line = _trilinear_sample(phi, spacing, points)
        xb_line = _trilinear_sample(xb, spacing, points)
        cross = _find_nearest_crossing(u_grid, phi_line, args.threshold)
        if cross is None:
            phi_aligned = np.full_like(xi_grid, np.nan)
            xb_aligned = np.full_like(xi_grid, np.nan)
        else:
            xi_line = u_grid - cross
            phi_aligned = _interp_1d(xi_line, phi_line, xi_grid)
            xb_aligned = _interp_1d(xi_line, xb_line, xi_grid)

        axs_face[0].plot(xi_grid, phi_aligned, color=colors[face_name], lw=2.0, label=face_name)
        axs_face[1].plot(xi_grid, xb_aligned, color=colors[face_name], lw=2.0, label=face_name)

        for xi, ph, xb_val in zip(xi_grid, phi_aligned, xb_aligned):
            rows.append({"face": face_name, "xi_nm": float(xi), "phi": float(ph), "xB": float(xb_val)})

    axs_face[0].set_title("+short vs -short local profiles: phi")
    axs_face[1].set_title("+short vs -short local profiles: xB")
    axs_face[0].set_xlabel("xi from local interface (nm)")
    axs_face[1].set_xlabel("xi from local interface (nm)")
    axs_face[0].set_ylabel("phi")
    axs_face[1].set_ylabel("xB")
    for ax in axs_face:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    fig_face.suptitle(f"+short and -short local profiles: {summary_path.parent.name}", fontsize=13)
    face_png = output_dir / "short_face_local_profiles.png"
    fig_face.savefig(face_png, dpi=180)
    plt.close(fig_face)

    csv_path = output_dir / "short_face_local_profiles.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        import csv
        writer = csv.DictWriter(f, fieldnames=["face", "xi_nm", "phi", "xB"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"summary={summary_path}")
    print(f"phi_vtk={phi_vtk}")
    print(f"xb_vtk={xb_vtk}")
    print(f"centered_plot={centered_png}")
    print(f"face_plot={face_png}")
    print(f"face_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
