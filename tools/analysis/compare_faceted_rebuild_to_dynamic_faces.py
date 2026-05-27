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


def _vector_from_info(info: dict[str, object], prefix: str) -> np.ndarray:
    return np.array([
        float(info[f"{prefix}_x"]),
        float(info[f"{prefix}_y"]),
        float(info[f"{prefix}_z"]),
    ], dtype=float)


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


def _load_family_profiles(path: Path) -> tuple[np.ndarray, dict[tuple[str, str], dict[str, np.ndarray]]]:
    rows = list(csv.DictReader(path.open()))
    keys = sorted(set((r["family"], r.get("region", "all")) for r in rows))
    xi = np.array(sorted(set(float(r["u_nm"]) for r in rows)), dtype=float)
    out: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for key in keys:
        fam, region = key
        pts = [r for r in rows if r["family"] == fam and r.get("region", "all") == region]
        pts.sort(key=lambda r: float(r["u_nm"]))
        out[key] = {
            "phi_mean": np.array([float(r["phi_mean"]) for r in pts], dtype=float),
            "xB_mean": np.array([float(r["xB_mean"]) for r in pts], dtype=float),
            "phi_std": np.array([float(r["phi_std"]) for r in pts], dtype=float),
            "xB_std": np.array([float(r["xB_std"]) for r in pts], dtype=float),
        }
    return xi, out


def _save_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def _rmse(a: np.ndarray, b: np.ndarray) -> float | None:
    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        return None
    return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare faceted reconstructed profiles against actual dynamic representative face profiles.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--phi-vtk", type=Path, default=None)
    parser.add_argument("--xb-vtk", type=Path, default=None)
    parser.add_argument("--family-profiles-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--inside-range-nm", type=float, default=2.5)
    parser.add_argument("--outside-range-nm", type=float, default=2.5)
    parser.add_argument("--step-nm", type=float, default=0.05)
    args = parser.parse_args()

    summary_path = args.summary.expanduser().resolve()
    info = parse_summary_file(summary_path)
    phi_vtk = args.phi_vtk.expanduser().resolve() if args.phi_vtk else Path(str(info["phi_vtk_file"])).expanduser().resolve()
    xb_vtk = args.xb_vtk.expanduser().resolve() if args.xb_vtk else _resolve_xb_vtk(phi_vtk)
    if xb_vtk is None or not xb_vtk.exists():
        raise FileNotFoundError(f"Cannot locate xB vtk for {phi_vtk}")
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (summary_path.parent / "_tmp_rebuild_compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    phi, _, spacing_vtk = _read_legacy_scalar_vtk(phi_vtk)
    xb, _, _ = _read_legacy_scalar_vtk(xb_vtk)
    spacing = (
        float(info.get("spacing_x", spacing_vtk[0])),
        float(info.get("spacing_y", spacing_vtk[1])),
        float(info.get("spacing_z", spacing_vtk[2])),
    )

    axes = np.array([
        [float(info["long_axis_x"]), float(info["long_axis_y"]), float(info["long_axis_z"])],
        [float(info["mid_axis_x"]), float(info["mid_axis_y"]), float(info["mid_axis_z"])],
        [float(info["short_axis_x"]), float(info["short_axis_y"]), float(info["short_axis_z"])],
    ], dtype=float)
    family_labels, family_dirs = _octant_directions()
    xi_model, family_profiles = _load_family_profiles(args.family_profiles_csv.expanduser().resolve())
    xi_actual = np.arange(-args.inside_range_nm, args.outside_range_nm + 0.5 * args.step_nm, args.step_nm)

    face_names = ["long", "mid", "short"]
    compare_rows: list[dict[str, object]] = []
    fig, axes_plot = plt.subplots(3, 2, figsize=(12, 12), constrained_layout=True)
    colors = {"actual": "#2c3e50", "recon": "#d35400"}

    for row_idx, face_name in enumerate(face_names):
        face_point = _vector_from_info(info, f"{face_name}_face_point")
        face_normal = _vector_from_info(info, f"{face_name}_face_normal")
        face_normal = face_normal / np.linalg.norm(face_normal)
        local_dir = axes @ face_normal
        fam_idx = int(np.argmax(family_dirs @ local_dir))
        family_label = family_labels[fam_idx]

        s_grid = np.arange(-args.inside_range_nm, args.outside_range_nm + 0.5 * args.step_nm, args.step_nm)
        points = face_point[None, :] + s_grid[:, None] * face_normal[None, :]
        phi_line = _trilinear_sample(phi, spacing, points)
        xb_line = _trilinear_sample(xb, spacing, points)
        cross = _find_nearest_crossing(s_grid, phi_line, args.threshold)
        if cross is None:
            phi_actual = np.full_like(xi_actual, np.nan)
            xb_actual = np.full_like(xi_actual, np.nan)
        else:
            xi_line = s_grid - cross
            phi_actual = _interp_1d(xi_line, phi_line, xi_actual)
            xb_actual = _interp_1d(xi_line, xb_line, xi_actual)

        face_key = (family_label, "face") if (family_label, "face") in family_profiles else (family_label, "all")
        edge_key = (family_label, "edge")
        phi_face = _interp_1d(xi_model, family_profiles[face_key]["phi_mean"], xi_actual)
        xb_face = _interp_1d(xi_model, family_profiles[face_key]["xB_mean"], xi_actual)

        best_alpha = 0.0
        best_loss = float("inf")
        phi_recon = phi_face.copy()
        xb_recon = xb_face.copy()
        recon_region = face_key[1]
        if edge_key in family_profiles:
            phi_edge = _interp_1d(xi_model, family_profiles[edge_key]["phi_mean"], xi_actual)
            xb_edge = _interp_1d(xi_model, family_profiles[edge_key]["xB_mean"], xi_actual)
            for alpha in np.linspace(0.0, 1.0, 51):
                phi_trial = (1.0 - alpha) * phi_face + alpha * phi_edge
                xb_trial = (1.0 - alpha) * xb_face + alpha * xb_edge
                phi_err = _rmse(phi_actual, phi_trial) or 0.0
                xb_err = _rmse(xb_actual, xb_trial) or 0.0
                loss = (phi_err / 0.02) ** 2 + (xb_err / 2.0e-4) ** 2
                if loss < best_loss:
                    best_loss = loss
                    best_alpha = float(alpha)
                    phi_recon = phi_trial
                    xb_recon = xb_trial
            recon_region = "face+edge"

        phi_rmse = _rmse(phi_actual, phi_recon)
        xb_rmse = _rmse(xb_actual, xb_recon)

        compare_rows.append(
            {
                "face": f"+{face_name}",
                "family": family_label,
                "region": recon_region,
                "edge_alpha_fit": best_alpha,
                "phi_rmse": phi_rmse,
                "xB_rmse": xb_rmse,
                "face_normal_long": float(local_dir[0]),
                "face_normal_mid": float(local_dir[1]),
                "face_normal_short": float(local_dir[2]),
            }
        )

        ax_phi = axes_plot[row_idx, 0]
        ax_xb = axes_plot[row_idx, 1]
        ax_phi.plot(xi_actual, phi_actual, color=colors["actual"], lw=2.0, label=f"actual +{face_name}")
        ax_phi.plot(xi_actual, phi_recon, color=colors["recon"], lw=1.8, ls="--", label=f"recon {family_label}:{recon_region}")
        ax_xb.plot(xi_actual, xb_actual, color=colors["actual"], lw=2.0, label=f"actual +{face_name}")
        ax_xb.plot(xi_actual, xb_recon, color=colors["recon"], lw=1.8, ls="--", label=f"recon {family_label}:{recon_region}")
        ax_phi.set_title(f"+{face_name} phi")
        ax_xb.set_title(f"+{face_name} xB")
        ax_phi.grid(True, alpha=0.25)
        ax_xb.grid(True, alpha=0.25)
        ax_phi.legend(frameon=False, fontsize=8)
        ax_xb.legend(frameon=False, fontsize=8)
        ax_phi.set_ylabel("phi")
        ax_xb.set_ylabel("xB")
        ax_phi.text(0.02, 0.06, f"RMSE={phi_rmse:.4e}" if phi_rmse is not None else "RMSE=NA", transform=ax_phi.transAxes, fontsize=9)
        ax_xb.text(0.02, 0.06, f"RMSE={xb_rmse:.4e}, alpha={best_alpha:.2f}" if xb_rmse is not None else "RMSE=NA", transform=ax_xb.transAxes, fontsize=9)

    for ax in axes_plot[-1, :]:
        ax.set_xlabel("xi from local interface (nm)")

    fig.suptitle(f"Dynamic face profiles: actual vs faceted reconstruction ({summary_path.parent.name})", fontsize=13)
    plot_png = output_dir / "dynamic_face_reconstruction_compare.png"
    fig.savefig(plot_png, dpi=180)
    plt.close(fig)

    metrics_json = output_dir / "dynamic_face_reconstruction_compare_metrics.json"
    metrics_json.write_text(json.dumps(compare_rows, indent=2), encoding="utf-8")
    _save_csv(output_dir / "dynamic_face_reconstruction_compare_metrics.csv", ["face", "family", "region", "edge_alpha_fit", "phi_rmse", "xB_rmse", "face_normal_long", "face_normal_mid", "face_normal_short"], compare_rows)

    print(f"summary={summary_path}")
    print(f"phi_vtk={phi_vtk}")
    print(f"xb_vtk={xb_vtk}")
    print(f"plot={plot_png}")
    print(f"metrics_json={metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
