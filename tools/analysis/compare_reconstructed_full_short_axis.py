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


def _interp_1d(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x, y, left=np.nan, right=np.nan)


def _interp_1d_constant(x: np.ndarray, y: np.ndarray, x_new: np.ndarray) -> np.ndarray:
    return np.interp(x_new, x, y, left=float(y[0]), right=float(y[-1]))


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


def _find_nearest_crossing(s: np.ndarray, phi: np.ndarray, threshold: float) -> float | None:
    vals = []
    for i in range(len(s) - 1):
        pi = phi[i]
        pj = phi[i + 1]
        if pi == threshold:
            vals.append(float(s[i]))
            continue
        if (pi - threshold) * (pj - threshold) <= 0.0 and pi != pj:
            t = (threshold - pi) / (pj - pi)
            vals.append(float(s[i] + t * (s[i + 1] - s[i])))
    if not vals:
        return None
    return min(vals, key=abs)


def _vector_from_info(info: dict[str, object], prefix: str) -> np.ndarray:
    return np.array([
        float(info[f"{prefix}_x"]),
        float(info[f"{prefix}_y"]),
        float(info[f"{prefix}_z"]),
    ], dtype=float)


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
        }
    return xi, out


def _rmse(a: np.ndarray, b: np.ndarray) -> float | None:
    valid = np.isfinite(a) & np.isfinite(b)
    if not valid.any():
        return None
    return float(np.sqrt(np.mean((a[valid] - b[valid]) ** 2)))


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare full centered short-axis actual cut against reconstructed cut from +/- short face models.")
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--phi-vtk", type=Path, default=None)
    parser.add_argument("--xb-vtk", type=Path, default=None)
    parser.add_argument("--family-profiles-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--center-half-range-nm", type=float, default=None)
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
    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (summary_path.parent / "_tmp_full_short_compare")
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
    axes = np.array([
        [float(info["long_axis_x"]), float(info["long_axis_y"]), float(info["long_axis_z"])],
        [float(info["mid_axis_x"]), float(info["mid_axis_y"]), float(info["mid_axis_z"])],
        [float(info["short_axis_x"]), float(info["short_axis_y"]), float(info["short_axis_z"])],
    ], dtype=float)
    family_labels, family_dirs = _octant_directions()
    xi_model, family_profiles = _load_family_profiles(args.family_profiles_csv.expanduser().resolve())

    nx, ny, nz = phi.shape
    box_half_range_nm = 0.5 * min(nx * spacing[0], ny * spacing[1], nz * spacing[2])
    center_half_range_nm = args.center_half_range_nm if args.center_half_range_nm is not None else box_half_range_nm
    s_center = np.arange(-center_half_range_nm, center_half_range_nm + 0.5 * args.step_nm, args.step_nm)
    center_points = center[None, :] + s_center[:, None] * short_axis[None, :]
    phi_actual = _trilinear_sample(phi, spacing, center_points)
    xb_actual = _trilinear_sample(xb, spacing, center_points)
    s_neg = _find_crossing(s_center, phi_actual, args.threshold, positive_side=False)
    s_pos = _find_crossing(s_center, phi_actual, args.threshold, positive_side=True)
    if s_neg is None or s_pos is None:
        raise RuntimeError("Could not locate both short-axis interfaces from centered cut.")

    face_models: dict[str, dict[str, object]] = {}
    for face_name, sign in (("+short", +1.0), ("-short", -1.0)):
        point = _vector_from_info(info, "short_face_point")
        normal = _vector_from_info(info, "short_face_normal")
        if sign < 0.0:
            point = 2.0 * center - point
            normal = -normal
        normal /= np.linalg.norm(normal)
        local_dir = axes @ normal
        fam_idx = int(np.argmax(family_dirs @ local_dir))
        family_label = family_labels[fam_idx]

        u_grid = np.arange(-args.face_inside_range_nm, args.face_outside_range_nm + 0.5 * args.step_nm, args.step_nm)
        points = point[None, :] + u_grid[:, None] * normal[None, :]
        phi_line = _trilinear_sample(phi, spacing, points)
        xb_line = _trilinear_sample(xb, spacing, points)
        cross = _find_nearest_crossing(u_grid, phi_line, args.threshold)
        if cross is None:
            raise RuntimeError(f"Could not align {face_name} local profile.")
        xi_local = u_grid - cross
        phi_local_actual = _interp_1d_constant(xi_local, phi_line, xi_model)
        xb_local_actual = _interp_1d_constant(xi_local, xb_line, xi_model)

        face_key = (family_label, "face") if (family_label, "face") in family_profiles else (family_label, "all")
        edge_key = (family_label, "edge")
        phi_face = family_profiles[face_key]["phi_mean"]
        xb_face = family_profiles[face_key]["xB_mean"]
        best_alpha = 0.0
        best_phi = phi_face.copy()
        best_xb = xb_face.copy()
        best_loss = float("inf")
        if edge_key in family_profiles:
            phi_edge = family_profiles[edge_key]["phi_mean"]
            xb_edge = family_profiles[edge_key]["xB_mean"]
            for alpha in np.linspace(0.0, 1.0, 51):
                phi_trial = (1.0 - alpha) * phi_face + alpha * phi_edge
                xb_trial = (1.0 - alpha) * xb_face + alpha * xb_edge
                phi_err = _rmse(phi_local_actual, phi_trial) or 0.0
                xb_err = _rmse(xb_local_actual, xb_trial) or 0.0
                loss = (phi_err / 0.02) ** 2 + (xb_err / 2.0e-4) ** 2
                if loss < best_loss:
                    best_loss = loss
                    best_alpha = float(alpha)
                    best_phi = phi_trial
                    best_xb = xb_trial

        face_models[face_name] = {
            "family": family_label,
            "alpha": best_alpha,
            "phi_model": best_phi,
            "xb_model": best_xb,
        }

    xi_plus = s_center - s_pos
    xi_minus = s_neg - s_center
    phi_plus = _interp_1d_constant(xi_model, np.asarray(face_models["+short"]["phi_model"]), xi_plus)
    xb_plus = _interp_1d_constant(xi_model, np.asarray(face_models["+short"]["xb_model"]), xi_plus)
    phi_minus = _interp_1d_constant(xi_model, np.asarray(face_models["-short"]["phi_model"]), xi_minus)
    xb_minus = _interp_1d_constant(xi_model, np.asarray(face_models["-short"]["xb_model"]), xi_minus)

    phi_recon = np.where(s_center >= 0.0, phi_plus, phi_minus)
    xb_recon = np.where(s_center >= 0.0, xb_plus, xb_minus)

    phi_rmse = _rmse(phi_actual, phi_recon)
    xb_rmse = _rmse(xb_actual, xb_recon)

    fig, axs = plt.subplots(1, 2, figsize=(12, 4.8), constrained_layout=True)
    axs[0].plot(s_center, phi_actual, color="#2c3e50", lw=2.0, label="actual")
    axs[0].plot(s_center, phi_recon, color="#d35400", lw=1.8, ls="--", label="reconstructed")
    axs[1].plot(s_center, xb_actual, color="#2c3e50", lw=2.0, label="actual")
    axs[1].plot(s_center, xb_recon, color="#d35400", lw=1.8, ls="--", label="reconstructed")
    axs[0].set_title("Centered short-axis full cut: phi")
    axs[1].set_title("Centered short-axis full cut: xB")
    axs[0].set_xlabel("s along short axis (nm)")
    axs[1].set_xlabel("s along short axis (nm)")
    axs[0].set_ylabel("phi")
    axs[1].set_ylabel("xB")
    for ax in axs:
        ax.grid(True, alpha=0.25)
        ax.legend(frameon=False)
    axs[0].text(0.02, 0.06, f"RMSE={phi_rmse:.4e}" if phi_rmse is not None else "RMSE=NA", transform=axs[0].transAxes, fontsize=9)
    axs[1].text(
        0.02,
        0.06,
        (
            f"RMSE={xb_rmse:.4e}\n"
            f"+alpha={face_models['+short']['alpha']:.2f}, -alpha={face_models['-short']['alpha']:.2f}"
        ) if xb_rmse is not None else "RMSE=NA",
        transform=axs[1].transAxes,
        fontsize=9,
    )
    fig.suptitle(f"Centered short-axis: actual vs reconstructed ({summary_path.parent.name})", fontsize=13)
    plot_png = output_dir / "centered_short_axis_actual_vs_reconstructed.png"
    fig.savefig(plot_png, dpi=180)
    plt.close(fig)

    metrics = {
        "summary_path": str(summary_path),
        "phi_vtk": str(phi_vtk),
        "xb_vtk": str(xb_vtk),
        "center_half_range_nm": float(center_half_range_nm),
        "box_half_range_nm": float(box_half_range_nm),
        "s_neg_nm": float(s_neg),
        "s_pos_nm": float(s_pos),
        "phi_rmse": phi_rmse,
        "xB_rmse": xb_rmse,
        "plus_family": face_models["+short"]["family"],
        "minus_family": face_models["-short"]["family"],
        "plus_edge_alpha_fit": face_models["+short"]["alpha"],
        "minus_edge_alpha_fit": face_models["-short"]["alpha"],
    }
    metrics_json = output_dir / "centered_short_axis_actual_vs_reconstructed_metrics.json"
    metrics_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    csv_path = output_dir / "centered_short_axis_actual_vs_reconstructed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["s_nm", "phi_actual", "phi_reconstructed", "xB_actual", "xB_reconstructed"],
        )
        writer.writeheader()
        for s_val, pa, pr, xa, xr in zip(s_center, phi_actual, phi_recon, xb_actual, xb_recon):
            writer.writerow({
                "s_nm": float(s_val),
                "phi_actual": float(pa),
                "phi_reconstructed": float(pr),
                "xB_actual": float(xa),
                "xB_reconstructed": float(xr),
            })

    print(f"summary={summary_path}")
    print(f"phi_vtk={phi_vtk}")
    print(f"xb_vtk={xb_vtk}")
    print(f"plot={plot_png}")
    print(f"metrics_json={metrics_json}")
    print(f"profile_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
