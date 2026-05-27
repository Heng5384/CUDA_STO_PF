#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
    if name.startswith("phi_"):
        cand = phi_vtk.with_name("xB_" + name[len("phi_"):])
        if cand.exists():
            return cand
    if name.startswith("phi_final_"):
        cand = phi_vtk.with_name("xB_final_" + name[len("phi_final_"):])
        if cand.exists():
            return cand
    if name.startswith("phi_init_"):
        cand = phi_vtk.with_name("xB_init_" + name[len("phi_init_"):])
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


def _find_crossing(s: np.ndarray, phi: np.ndarray, threshold: float, positive_side: bool) -> float | None:
    center_idx = int(np.argmin(np.abs(s)))
    if positive_side:
        indices = range(center_idx, len(s) - 1)
    else:
        indices = range(center_idx, 0, -1)

    for i in indices:
        j = i + 1 if positive_side else i - 1
        pi = phi[i]
        pj = phi[j]
        if (pi - threshold) == 0.0:
            return float(s[i])
        if (pi - threshold) * (pj - threshold) <= 0.0 and pi != pj:
            t = (threshold - pi) / (pj - pi)
            return float(s[i] + t * (s[j] - s[i]))
    return None


def _interp_1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x, y, left=np.nan, right=np.nan)


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


def _build_aligned_profile(
    s: np.ndarray,
    phi: np.ndarray,
    xb: np.ndarray,
    threshold: float,
    xi_grid: np.ndarray,
) -> dict[str, np.ndarray | float | None]:
    s_pos = _find_crossing(s, phi, threshold, positive_side=True)
    s_neg = _find_crossing(s, phi, threshold, positive_side=False)

    phi_parts: list[np.ndarray] = []
    xb_parts: list[np.ndarray] = []

    if s_pos is not None:
        xi = s - s_pos
        phi_parts.append(_interp_1d(xi, phi, xi_grid))
        xb_parts.append(_interp_1d(xi, xb, xi_grid))

    if s_neg is not None:
        xi = s_neg - s
        phi_parts.append(_interp_1d(xi[::-1], phi[::-1], xi_grid))
        xb_parts.append(_interp_1d(xi[::-1], xb[::-1], xi_grid))

    phi_avg = np.nanmean(np.vstack(phi_parts), axis=0) if phi_parts else np.full_like(xi_grid, np.nan)
    xb_avg = np.nanmean(np.vstack(xb_parts), axis=0) if xb_parts else np.full_like(xi_grid, np.nan)

    return {
        "phi_avg": phi_avg,
        "xb_avg": xb_avg,
        "s_cross_pos_nm": s_pos,
        "s_cross_neg_nm": s_neg,
    }


def _interface_width_nm(xi: np.ndarray, phi: np.ndarray, lo: float = 0.1, hi: float = 0.9) -> float | None:
    valid = np.isfinite(phi)
    if valid.sum() < 4:
        return None
    x = xi[valid]
    y = phi[valid]
    order = np.argsort(y)[::-1]
    y_desc = y[order]
    x_desc = x[order]
    if np.nanmin(y_desc) > lo or np.nanmax(y_desc) < hi:
        return None
    x_hi = float(np.interp(hi, y_desc[::-1], x_desc[::-1]))
    x_lo = float(np.interp(lo, y_desc[::-1], x_desc[::-1]))
    return abs(x_lo - x_hi)


def _save_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _as_float(value: object, default: float | None = None) -> float | None:
    if value in (None, "", "None"):
        return default
    return float(value)


def _vector_from_info(info: dict[str, object], prefix: str) -> np.ndarray | None:
    x = _as_float(info.get(f"{prefix}_x"))
    y = _as_float(info.get(f"{prefix}_y"))
    z = _as_float(info.get(f"{prefix}_z"))
    if x is None or y is None or z is None:
        return None
    return np.array([x, y, z], dtype=float)


def _auto_axis_half_range_nm(info: dict[str, object], interface_window_nm: float) -> float:
    lengths = [_as_float(info.get(key)) for key in ("L1_long", "L2_mid", "L3_short")]
    lengths = [x for x in lengths if x is not None]
    if lengths:
        return 0.55 * max(lengths) + 2.0 * interface_window_nm
    bbox = [_as_float(info.get(key)) for key in ("bbox_x", "bbox_y", "bbox_z")]
    bbox = [x for x in bbox if x is not None]
    if bbox:
        return 0.55 * max(bbox) + 2.0 * interface_window_nm
    return 6.0


def _face_inner_range_nm(axis_length_nm: float | None, interface_window_nm: float) -> float:
    if axis_length_nm is None:
        return 4.0 * interface_window_nm
    return 1.05 * axis_length_nm


def _extract_profile(
    field_phi: np.ndarray,
    field_xb: np.ndarray,
    spacing: tuple[float, float, float],
    origin: np.ndarray,
    direction: np.ndarray,
    s_grid: np.ndarray,
    threshold: float,
    xi_grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray | float | None]]:
    direction = direction / np.linalg.norm(direction)
    points = origin[None, :] + s_grid[:, None] * direction[None, :]
    phi_line = _trilinear_sample(field_phi, spacing, points)
    xb_line = _trilinear_sample(field_xb, spacing, points)
    aligned = _build_aligned_profile(s_grid, phi_line, xb_line, threshold, xi_grid)
    return phi_line, xb_line, aligned


def main() -> int:
    parser = argparse.ArgumentParser(description="Extract exploratory directional phi/xB interface profiles for one nucleus.")
    parser.add_argument("--summary", type=Path, required=True, help="summary.txt for one nucleus")
    parser.add_argument("--phi-vtk", type=Path, default=None, help="optional explicit phi vtk path")
    parser.add_argument("--xb-vtk", type=Path, default=None, help="optional explicit xB vtk path")
    parser.add_argument("--output-dir", type=Path, default=None, help="directory for csv/png/json outputs")
    parser.add_argument("--half-range-nm", type=float, default=None, help="half window for centered directional cuts; default is geometry-based")
    parser.add_argument("--step-nm", type=float, default=0.02, help="sampling step along each axis")
    parser.add_argument("--interface-window-nm", type=float, default=1.5, help="window around phi=0.5 interface for aligned comparison")
    parser.add_argument("--threshold", type=float, default=0.5, help="interface threshold for alignment")
    args = parser.parse_args()

    summary_path = args.summary.expanduser().resolve()
    info = parse_summary_file(summary_path)
    phi_vtk = args.phi_vtk.expanduser().resolve() if args.phi_vtk else Path(str(info["phi_vtk_file"])).expanduser().resolve()
    xb_vtk = args.xb_vtk.expanduser().resolve() if args.xb_vtk else _resolve_xb_vtk(phi_vtk)
    if xb_vtk is None or not xb_vtk.exists():
        raise FileNotFoundError(f"Cannot locate xB vtk for {phi_vtk}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (summary_path.parent / "_tmp_directional_profiles")
    output_dir.mkdir(parents=True, exist_ok=True)

    phi, _, spacing_vtk = _read_legacy_scalar_vtk(phi_vtk)
    xb, _, _ = _read_legacy_scalar_vtk(xb_vtk)
    spacing = (
        float(info.get("spacing_x", spacing_vtk[0])),
        float(info.get("spacing_y", spacing_vtk[1])),
        float(info.get("spacing_z", spacing_vtk[2])),
    )

    center = np.array([float(info["center_x"]), float(info["center_y"]), float(info["center_z"])], dtype=float)
    axes = {
        "long": np.array([float(info["long_axis_x"]), float(info["long_axis_y"]), float(info["long_axis_z"])], dtype=float),
        "mid": np.array([float(info["mid_axis_x"]), float(info["mid_axis_y"]), float(info["mid_axis_z"])], dtype=float),
        "short": np.array([float(info["short_axis_x"]), float(info["short_axis_y"]), float(info["short_axis_z"])], dtype=float),
    }
    axis_lengths = {
        "long": _as_float(info.get("L1_long")),
        "mid": _as_float(info.get("L2_mid")),
        "short": _as_float(info.get("L3_short")),
    }

    axis_half_range_nm = args.half_range_nm if args.half_range_nm is not None else _auto_axis_half_range_nm(info, args.interface_window_nm)
    s_grid = np.arange(-axis_half_range_nm, axis_half_range_nm + 0.5 * args.step_nm, args.step_nm)
    xi_grid = np.arange(-args.interface_window_nm, args.interface_window_nm + 0.5 * args.step_nm, args.step_nm)

    raw_rows: list[dict[str, object]] = []
    aligned_rows: list[dict[str, object]] = []
    metrics: list[dict[str, object]] = []
    face_raw_rows: list[dict[str, object]] = []
    face_aligned_rows: list[dict[str, object]] = []
    face_metrics: list[dict[str, object]] = []

    fig, axes_plot = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)
    fig_face, axes_face = plt.subplots(2, 2, figsize=(12, 9), constrained_layout=True)

    colors = {"long": "#c0392b", "mid": "#2980b9", "short": "#27ae60"}
    axis_crosses: list[float] = []
    for axis_name, direction in axes.items():
        phi_line, xb_line, aligned = _extract_profile(phi, xb, spacing, center, direction, s_grid, args.threshold, xi_grid)
        phi_avg = np.asarray(aligned["phi_avg"])
        xb_avg = np.asarray(aligned["xb_avg"])
        width = _interface_width_nm(xi_grid, phi_avg)
        if aligned["s_cross_pos_nm"] is not None:
            axis_crosses.append(abs(float(aligned["s_cross_pos_nm"])))
        if aligned["s_cross_neg_nm"] is not None:
            axis_crosses.append(abs(float(aligned["s_cross_neg_nm"])))

        metrics.append(
            {
                "axis": axis_name,
                "s_cross_pos_nm": aligned["s_cross_pos_nm"],
                "s_cross_neg_nm": aligned["s_cross_neg_nm"],
                "phi_width_10_90_nm": width,
                "xB_inside_at_xi_minus0p5": float(np.interp(-0.5, xi_grid, xb_avg)) if np.isfinite(xb_avg).any() else None,
                "xB_outside_at_xi_plus0p5": float(np.interp(0.5, xi_grid, xb_avg)) if np.isfinite(xb_avg).any() else None,
            }
        )

        for s, ph, xb_val in zip(s_grid, phi_line, xb_line):
            raw_rows.append({"axis": axis_name, "s_nm": float(s), "phi": float(ph), "xB": float(xb_val)})
        for xi, ph, xb_val in zip(xi_grid, phi_avg, xb_avg):
            aligned_rows.append({"axis": axis_name, "xi_nm": float(xi), "phi_avg": float(ph), "xB_avg": float(xb_val)})

        color = colors[axis_name]
        axes_plot[0, 0].plot(s_grid, phi_line, color=color, label=axis_name, lw=1.6)
        axes_plot[0, 1].plot(s_grid, xb_line, color=color, label=axis_name, lw=1.6)
        axes_plot[1, 0].plot(xi_grid, phi_avg, color=color, label=axis_name, lw=1.8)
        axes_plot[1, 1].plot(xi_grid, xb_avg, color=color, label=axis_name, lw=1.8)

    raw_limit_nm = max(axis_crosses) + 2.0 * args.interface_window_nm if axis_crosses else axis_half_range_nm
    axes_plot[0, 0].set_xlim(-raw_limit_nm, raw_limit_nm)
    axes_plot[0, 1].set_xlim(-raw_limit_nm, raw_limit_nm)
    axes_plot[0, 0].set_title("Raw line cuts: phi")
    axes_plot[0, 1].set_title("Raw line cuts: xB")
    axes_plot[1, 0].set_title("Aligned interface profiles: phi")
    axes_plot[1, 1].set_title("Aligned interface profiles: xB")
    for ax in axes_plot.ravel():
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    axes_plot[0, 0].set_xlabel("s (nm)")
    axes_plot[0, 0].set_ylabel("phi")
    axes_plot[0, 1].set_xlabel("s (nm)")
    axes_plot[0, 1].set_ylabel("xB")
    axes_plot[1, 0].set_xlabel("xi from phi=0.5 interface (nm)")
    axes_plot[1, 0].set_ylabel("phi")
    axes_plot[1, 1].set_xlabel("xi from phi=0.5 interface (nm)")
    axes_plot[1, 1].set_ylabel("xB")

    title = summary_path.parent.name
    fig.suptitle(f"Centered principal-axis profiles: {title}", fontsize=13)

    face_specs: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for axis_name in ("long", "mid", "short"):
        face_point = _vector_from_info(info, f"{axis_name}_face_point")
        face_normal = _vector_from_info(info, f"{axis_name}_face_normal")
        if face_point is None or face_normal is None:
            continue
        face_specs.append((axis_name, face_point, face_normal, axes[axis_name]))

    for axis_name, face_point, face_normal, axis_dir in face_specs:
        face_normal = face_normal / np.linalg.norm(face_normal)
        if np.dot(face_normal, axis_dir) < 0.0:
            face_normal = -face_normal
        inner_range_nm = _face_inner_range_nm(axis_lengths[axis_name], args.interface_window_nm)
        outer_range_nm = 2.5 * args.interface_window_nm
        face_s_grid = np.arange(-inner_range_nm, outer_range_nm + 0.5 * args.step_nm, args.step_nm)
        phi_line, xb_line, _ = _extract_profile(phi, xb, spacing, face_point, face_normal, face_s_grid, args.threshold, xi_grid)
        face_cross = _find_nearest_crossing(face_s_grid, phi_line, args.threshold)
        if face_cross is not None:
            face_xi = face_s_grid - face_cross
            phi_avg = _interp_1d(face_xi, phi_line, xi_grid)
            xb_avg = _interp_1d(face_xi, xb_line, xi_grid)
        else:
            phi_avg = np.full_like(xi_grid, np.nan)
            xb_avg = np.full_like(xi_grid, np.nan)
        width = _interface_width_nm(xi_grid, phi_avg)

        face_metrics.append(
            {
                "face": f"+{axis_name}",
                "s_cross_pos_nm": face_cross if face_cross is not None and face_cross >= 0.0 else None,
                "s_cross_neg_nm": face_cross if face_cross is not None and face_cross < 0.0 else None,
                "phi_width_10_90_nm": width,
                "xB_inside_at_xi_minus0p5": float(np.interp(-0.5, xi_grid, xb_avg)) if np.isfinite(xb_avg).any() else None,
                "xB_outside_at_xi_plus0p5": float(np.interp(0.5, xi_grid, xb_avg)) if np.isfinite(xb_avg).any() else None,
            }
        )

        for s, ph, xb_val in zip(face_s_grid, phi_line, xb_line):
            face_raw_rows.append({"face": f"+{axis_name}", "u_nm": float(s), "phi": float(ph), "xB": float(xb_val)})
        for xi, ph, xb_val in zip(xi_grid, phi_avg, xb_avg):
            face_aligned_rows.append({"face": f"+{axis_name}", "xi_nm": float(xi), "phi_avg": float(ph), "xB_avg": float(xb_val)})

        color = colors[axis_name]
        axes_face[0, 0].plot(face_s_grid, phi_line, color=color, label=f"+{axis_name}", lw=1.6)
        axes_face[0, 1].plot(face_s_grid, xb_line, color=color, label=f"+{axis_name}", lw=1.6)
        axes_face[1, 0].plot(xi_grid, phi_avg, color=color, label=f"+{axis_name}", lw=1.8)
        axes_face[1, 1].plot(xi_grid, xb_avg, color=color, label=f"+{axis_name}", lw=1.8)

    axes_face[0, 0].set_title("Raw face-normal cuts: phi")
    axes_face[0, 1].set_title("Raw face-normal cuts: xB")
    axes_face[1, 0].set_title("Aligned face-normal profiles: phi")
    axes_face[1, 1].set_title("Aligned face-normal profiles: xB")
    for ax in axes_face.ravel():
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    axes_face[0, 0].set_xlabel("u from representative +face point (nm)")
    axes_face[0, 0].set_ylabel("phi")
    axes_face[0, 1].set_xlabel("u from representative +face point (nm)")
    axes_face[0, 1].set_ylabel("xB")
    axes_face[1, 0].set_xlabel("xi from phi=0.5 interface (nm)")
    axes_face[1, 0].set_ylabel("phi")
    axes_face[1, 1].set_xlabel("xi from phi=0.5 interface (nm)")
    axes_face[1, 1].set_ylabel("xB")
    fig_face.suptitle(f"Representative +face normal profiles: {title}", fontsize=13)

    raw_csv = output_dir / "raw_directional_profiles.csv"
    aligned_csv = output_dir / "aligned_interface_profiles.csv"
    metrics_json = output_dir / "directional_profile_metrics.json"
    plot_png = output_dir / "directional_interface_profiles.png"
    face_raw_csv = output_dir / "raw_face_profiles.csv"
    face_aligned_csv = output_dir / "aligned_face_profiles.csv"
    face_metrics_json = output_dir / "face_profile_metrics.json"
    face_plot_png = output_dir / "face_directional_interface_profiles.png"

    _save_csv(raw_csv, ["axis", "s_nm", "phi", "xB"], raw_rows)
    _save_csv(aligned_csv, ["axis", "xi_nm", "phi_avg", "xB_avg"], aligned_rows)
    _save_csv(face_raw_csv, ["face", "u_nm", "phi", "xB"], face_raw_rows)
    _save_csv(face_aligned_csv, ["face", "xi_nm", "phi_avg", "xB_avg"], face_aligned_rows)
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    face_metrics_json.write_text(json.dumps(face_metrics, indent=2), encoding="utf-8")
    fig.savefig(plot_png, dpi=180)
    fig_face.savefig(face_plot_png, dpi=180)
    plt.close(fig)
    plt.close(fig_face)

    print(f"summary={summary_path}")
    print(f"phi_vtk={phi_vtk}")
    print(f"xb_vtk={xb_vtk}")
    print(f"plot={plot_png}")
    print(f"face_plot={face_plot_png}")
    print(f"raw_csv={raw_csv}")
    print(f"aligned_csv={aligned_csv}")
    print(f"metrics_json={metrics_json}")
    print(f"face_raw_csv={face_raw_csv}")
    print(f"face_aligned_csv={face_aligned_csv}")
    print(f"face_metrics_json={face_metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
