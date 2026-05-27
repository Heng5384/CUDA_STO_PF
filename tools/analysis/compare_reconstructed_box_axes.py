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
from tools.analysis.compare_reconstructed_full_short_axis import (
    _find_crossing,
    _find_nearest_crossing,
    _interp_1d,
    _interp_1d_constant,
    _load_family_profiles,
    _octant_directions,
    _resolve_xb_vtk,
    _rmse,
    _trilinear_sample,
)
from tools.analysis.generate_continue_dynamic_geometry_summaries import _read_legacy_scalar_vtk


def _vector_from_info(info: dict[str, object], prefix: str) -> np.ndarray:
    return np.array([
        float(info[f"{prefix}_x"]),
        float(info[f"{prefix}_y"]),
        float(info[f"{prefix}_z"]),
    ], dtype=float)


def _choose_model(
    *,
    family_label: str,
    family_profiles: dict[tuple[str, str], dict[str, np.ndarray]],
    xi_model: np.ndarray,
    xi_actual: np.ndarray,
    phi_actual: np.ndarray,
    xb_actual: np.ndarray,
) -> dict[str, object]:
    face_key = (family_label, "face") if (family_label, "face") in family_profiles else (family_label, "all")
    edge_key = (family_label, "edge")
    phi_face = family_profiles[face_key]["phi_mean"]
    xb_face = family_profiles[face_key]["xB_mean"]

    phi_line = _interp_1d_constant(xi_actual, phi_actual, xi_model)
    xb_line = _interp_1d_constant(xi_actual, xb_actual, xi_model)

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
            phi_err = _rmse(phi_line, phi_trial) or 0.0
            xb_err = _rmse(xb_line, xb_trial) or 0.0
            loss = (phi_err / 0.02) ** 2 + (xb_err / 2.0e-4) ** 2
            if loss < best_loss:
                best_loss = loss
                best_alpha = float(alpha)
                best_phi = phi_trial
                best_xb = xb_trial

    return {
        "family": family_label,
        "alpha": best_alpha,
        "phi_model": best_phi,
        "xb_model": best_xb,
    }


def _sample_axis(
    *,
    field: np.ndarray,
    spacing: tuple[float, float, float],
    center: np.ndarray,
    axis_xyz: np.ndarray,
    s_grid: np.ndarray,
) -> np.ndarray:
    points = center[None, :] + s_grid[:, None] * axis_xyz[None, :]
    return _trilinear_sample(field, spacing, points)


def _unique_mean_profile(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(x)
    x_sorted = x[order]
    y_sorted = y[order]
    unique_x, inverse = np.unique(x_sorted, return_inverse=True)
    y_sum = np.zeros_like(unique_x, dtype=float)
    counts = np.zeros_like(unique_x, dtype=float)
    np.add.at(y_sum, inverse, y_sorted)
    np.add.at(counts, inverse, 1.0)
    return unique_x, y_sum / np.maximum(counts, 1.0)


def _mean_delta_profile_lookup(
    *,
    family_profiles: dict[tuple[str, str], dict[str, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    phi_profiles: list[np.ndarray] = []
    delta_profiles: list[np.ndarray] = []
    families = sorted({family for family, _region in family_profiles})
    for family in families:
        key = (family, "face") if (family, "face") in family_profiles else (family, "all")
        if key not in family_profiles:
            continue
        phi_model = np.asarray(family_profiles[key]["phi_mean"], dtype=float)
        xb_model = np.asarray(family_profiles[key]["xB_mean"], dtype=float)
        phi_profiles.append(phi_model)
        delta_profiles.append(xb_model - float(xb_model[-1]))
    if not phi_profiles:
        return np.array([0.0, 1.0], dtype=float), np.array([0.0, 0.0], dtype=float)

    phi_profile = np.mean(np.vstack(phi_profiles), axis=0)
    delta_profile = np.mean(np.vstack(delta_profiles), axis=0)
    return _unique_mean_profile(phi_profile, delta_profile)


def _estimate_global_mass_shift_from_phi(
    *,
    phi: np.ndarray,
    xb: np.ndarray,
    xb_matrix_far: float,
    family_profiles: dict[tuple[str, str], dict[str, np.ndarray]],
) -> tuple[float, float, float]:
    phi_lookup, delta_lookup = _mean_delta_profile_lookup(family_profiles=family_profiles)
    flat_phi = phi.ravel()
    total = 0.0
    chunk = 4_000_000
    for start in range(0, flat_phi.size, chunk):
        vals = np.interp(flat_phi[start:start + chunk], phi_lookup, delta_lookup)
        total += float(np.sum(xb_matrix_far + vals))
    recon_mean_before_shift = total / float(flat_phi.size)
    actual_mean = float(np.mean(xb))
    return actual_mean - recon_mean_before_shift, actual_mean, recon_mean_before_shift


def _estimate_principal_semiaxes(
    *,
    phi: np.ndarray,
    spacing: tuple[float, float, float],
    center: np.ndarray,
    axes_local: np.ndarray,
    threshold: float,
) -> np.ndarray:
    nx, ny, nz = phi.shape
    half_range = 0.5 * min(nx * spacing[0], ny * spacing[1], nz * spacing[2])
    s_grid = np.arange(-half_range, half_range + 0.05, 0.05)
    semiaxes: list[float] = []
    for axis_xyz in axes_local:
        axis_xyz = axis_xyz / np.linalg.norm(axis_xyz)
        vals = _sample_axis(field=phi, spacing=spacing, center=center, axis_xyz=axis_xyz, s_grid=s_grid)
        s_neg = _find_crossing(s_grid, vals, threshold, positive_side=False)
        s_pos = _find_crossing(s_grid, vals, threshold, positive_side=True)
        if s_neg is None or s_pos is None:
            semiaxes.append(1.0)
        else:
            semiaxes.append(0.5 * (abs(float(s_neg)) + abs(float(s_pos))))
    return np.maximum(np.array(semiaxes, dtype=float), 1.0e-6)


def _ellipsoid_rho_at(points_xyz: np.ndarray, center: np.ndarray, axes_local: np.ndarray, semiaxes: np.ndarray) -> np.ndarray:
    rel = points_xyz - center[None, :]
    local = rel @ axes_local.T
    return np.sqrt(np.sum((local / semiaxes[None, :]) ** 2, axis=1))


def _fit_ellipsoid_binned_background(
    *,
    xb: np.ndarray,
    phi: np.ndarray,
    spacing: tuple[float, float, float],
    center: np.ndarray,
    axes_local: np.ndarray,
    semiaxes: np.ndarray,
    matrix_phi_max: float,
    n_bins: int = 240,
    max_samples: int = 1_000_000,
) -> tuple[np.ndarray, np.ndarray, float]:
    idx = np.argwhere(phi <= matrix_phi_max)
    if idx.shape[0] < 16:
        idx = np.argwhere(np.ones_like(phi, dtype=bool))
    if idx.shape[0] > max_samples:
        rng = np.random.default_rng(12345)
        idx = idx[rng.choice(idx.shape[0], size=max_samples, replace=False)]

    coords = idx.astype(float)
    coords[:, 0] *= spacing[0]
    coords[:, 1] *= spacing[1]
    coords[:, 2] *= spacing[2]
    rho = _ellipsoid_rho_at(coords, center, axes_local, semiaxes)
    values = xb[idx[:, 0], idx[:, 1], idx[:, 2]]

    rho_min = max(1.0, float(np.nanmin(rho)))
    rho_max = float(np.nanmax(rho))
    edges = np.linspace(rho_min, rho_max, n_bins + 1)
    bin_idx = np.clip(np.searchsorted(edges, rho, side="right") - 1, 0, n_bins - 1)
    sums = np.zeros(n_bins, dtype=float)
    counts = np.zeros(n_bins, dtype=float)
    np.add.at(sums, bin_idx, values)
    np.add.at(counts, bin_idx, 1.0)
    centers = 0.5 * (edges[:-1] + edges[1:])
    means = np.full(n_bins, np.nan, dtype=float)
    valid = counts > 0
    means[valid] = sums[valid] / counts[valid]
    if valid.sum() == 0:
        means[:] = float(np.mean(values))
    elif valid.sum() == 1:
        means[:] = means[valid][0]
    else:
        means = np.interp(centers, centers[valid], means[valid], left=means[valid][0], right=means[valid][-1])

    pred = np.interp(rho, centers, means, left=means[0], right=means[-1])
    rmse = float(np.sqrt(np.mean((values - pred) ** 2)))
    return centers, means, rmse


def _ellipsoid_background_at(
    points_xyz: np.ndarray,
    *,
    center: np.ndarray,
    axes_local: np.ndarray,
    semiaxes: np.ndarray,
    rho_bins: np.ndarray,
    xb_bins: np.ndarray,
) -> np.ndarray:
    rho = _ellipsoid_rho_at(points_xyz, center, axes_local, semiaxes)
    return np.interp(rho, rho_bins, xb_bins, left=float(xb_bins[0]), right=float(xb_bins[-1]))


def _estimate_global_mass_shift_general(
    *,
    phi: np.ndarray,
    xb: np.ndarray,
    spacing: tuple[float, float, float],
    family_profiles: dict[tuple[str, str], dict[str, np.ndarray]],
    background_at,
) -> tuple[float, float, float]:
    phi_lookup, delta_lookup = _mean_delta_profile_lookup(family_profiles=family_profiles)
    flat_phi = phi.ravel()
    total = 0.0
    chunk = 2_000_000
    nx, ny, nz = phi.shape
    for start in range(0, flat_phi.size, chunk):
        stop = min(start + chunk, flat_phi.size)
        flat_idx = np.arange(start, stop, dtype=np.int64)
        ix, iy, iz = np.unravel_index(flat_idx, (nx, ny, nz))
        points = np.column_stack([ix * spacing[0], iy * spacing[1], iz * spacing[2]]).astype(float)
        delta = np.interp(flat_phi[start:stop], phi_lookup, delta_lookup)
        total += float(np.sum(background_at(points) + delta))
    recon_mean_before_shift = total / float(flat_phi.size)
    actual_mean = float(np.mean(xb))
    return actual_mean - recon_mean_before_shift, actual_mean, recon_mean_before_shift


def _fit_linear_background(
    *,
    xb: np.ndarray,
    phi: np.ndarray,
    spacing: tuple[float, float, float],
    matrix_phi_max: float,
    max_samples: int = 500_000,
) -> tuple[np.ndarray, float]:
    mask = phi <= matrix_phi_max
    idx = np.argwhere(mask)
    if idx.shape[0] < 4:
        idx = np.argwhere(np.ones_like(phi, dtype=bool))
    if idx.shape[0] > max_samples:
        rng = np.random.default_rng(12345)
        idx = idx[rng.choice(idx.shape[0], size=max_samples, replace=False)]
    coords = idx.astype(float)
    coords[:, 0] *= spacing[0]
    coords[:, 1] *= spacing[1]
    coords[:, 2] *= spacing[2]
    values = xb[idx[:, 0], idx[:, 1], idx[:, 2]]
    design = np.column_stack([np.ones(idx.shape[0]), coords])
    coeff, *_ = np.linalg.lstsq(design, values, rcond=None)
    residual = values - design @ coeff
    return coeff.astype(float), float(np.sqrt(np.mean(residual ** 2)))


def _linear_background_at(coeff: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    return coeff[0] + points_xyz @ coeff[1:4]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Compare whole-box x/y/z center-axis dynamic profiles against reconstructed phi/xB profiles."
    )
    parser.add_argument("--summary", type=Path, required=True)
    parser.add_argument("--phi-vtk", type=Path, default=None)
    parser.add_argument("--xb-vtk", type=Path, default=None)
    parser.add_argument("--family-profiles-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--threshold", type=float, default=0.5)
    parser.add_argument("--line-step-nm", type=float, default=0.05)
    parser.add_argument("--local-half-range-nm", type=float, default=2.5)
    parser.add_argument("--matrix-phi-max", type=float, default=1.0e-3)
    parser.add_argument("--background-mode", choices=("constant", "linear", "ellipsoid-binned"), default="ellipsoid-binned")
    parser.add_argument(
        "--mass-correction-mode",
        choices=("global-phi", "line", "none"),
        default="global-phi",
        help="global-phi uses one full-volume shift estimated from phi; line shifts each plotted axis independently.",
    )
    args = parser.parse_args()

    summary_path = args.summary.expanduser().resolve()
    info = parse_summary_file(summary_path)
    phi_vtk = args.phi_vtk.expanduser().resolve() if args.phi_vtk else Path(str(info["phi_vtk_file"])).expanduser().resolve()
    xb_vtk = args.xb_vtk.expanduser().resolve() if args.xb_vtk else _resolve_xb_vtk(phi_vtk)
    if xb_vtk is None or not xb_vtk.exists():
        raise FileNotFoundError(f"Cannot locate xB vtk for {phi_vtk}")

    output_dir = args.output_dir.expanduser().resolve() if args.output_dir else (summary_path.parent / "_tmp_box_axes_compare")
    output_dir.mkdir(parents=True, exist_ok=True)

    phi, _, spacing_vtk = _read_legacy_scalar_vtk(phi_vtk)
    xb, _, _ = _read_legacy_scalar_vtk(xb_vtk)
    spacing = (
        float(info.get("spacing_x", spacing_vtk[0])),
        float(info.get("spacing_y", spacing_vtk[1])),
        float(info.get("spacing_z", spacing_vtk[2])),
    )
    nx, ny, nz = phi.shape
    box_size = np.array([nx * spacing[0], ny * spacing[1], nz * spacing[2]], dtype=float)
    center = np.array([float(info["center_x"]), float(info["center_y"]), float(info["center_z"])], dtype=float)
    axes_local = np.array([
        [float(info["long_axis_x"]), float(info["long_axis_y"]), float(info["long_axis_z"])],
        [float(info["mid_axis_x"]), float(info["mid_axis_y"]), float(info["mid_axis_z"])],
        [float(info["short_axis_x"]), float(info["short_axis_y"]), float(info["short_axis_z"])],
    ], dtype=float)

    matrix_mask = phi <= args.matrix_phi_max
    if matrix_mask.any():
        xb_matrix_far = float(np.mean(xb[matrix_mask]))
    else:
        edge_vals = np.concatenate([
            xb[0, :, :].ravel(), xb[-1, :, :].ravel(),
            xb[:, 0, :].ravel(), xb[:, -1, :].ravel(),
            xb[:, :, 0].ravel(), xb[:, :, -1].ravel(),
        ])
        xb_matrix_far = float(np.mean(edge_vals))
    background_coeff = np.array([xb_matrix_far, 0.0, 0.0, 0.0], dtype=float)
    background_fit_rmse = 0.0
    semiaxes = _estimate_principal_semiaxes(
        phi=phi,
        spacing=spacing,
        center=center,
        axes_local=axes_local,
        threshold=args.threshold,
    )
    rho_bins = np.array([], dtype=float)
    xb_background_bins = np.array([], dtype=float)
    if args.background_mode == "linear":
        background_coeff, background_fit_rmse = _fit_linear_background(
            xb=xb,
            phi=phi,
            spacing=spacing,
            matrix_phi_max=args.matrix_phi_max,
        )
    elif args.background_mode == "ellipsoid-binned":
        rho_bins, xb_background_bins, background_fit_rmse = _fit_ellipsoid_binned_background(
            xb=xb,
            phi=phi,
            spacing=spacing,
            center=center,
            axes_local=axes_local,
            semiaxes=semiaxes,
            matrix_phi_max=args.matrix_phi_max,
        )

    family_labels, family_dirs = _octant_directions()
    xi_model, family_profiles = _load_family_profiles(args.family_profiles_csv.expanduser().resolve())

    def background_at(points_xyz: np.ndarray) -> np.ndarray:
        if args.background_mode == "linear":
            return _linear_background_at(background_coeff, points_xyz)
        if args.background_mode == "ellipsoid-binned":
            return _ellipsoid_background_at(
                points_xyz,
                center=center,
                axes_local=axes_local,
                semiaxes=semiaxes,
                rho_bins=rho_bins,
                xb_bins=xb_background_bins,
            )
        return np.full(points_xyz.shape[0], xb_matrix_far, dtype=float)

    global_mass_shift, actual_global_mean, recon_global_mean_before_shift = _estimate_global_mass_shift_general(
        phi=phi,
        xb=xb,
        spacing=spacing,
        family_profiles=family_profiles,
        background_at=background_at,
    )

    axis_defs = [
        ("x", np.array([1.0, 0.0, 0.0]), 0),
        ("y", np.array([0.0, 1.0, 0.0]), 1),
        ("z", np.array([0.0, 0.0, 1.0]), 2),
    ]
    metrics: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []

    fig, axs = plt.subplots(3, 2, figsize=(13.5, 11.5), constrained_layout=True)
    for row_idx, (axis_name, axis_xyz, dim_idx) in enumerate(axis_defs):
        s_min = -center[dim_idx]
        s_max = box_size[dim_idx] - center[dim_idx]
        s_grid = np.arange(s_min, s_max + 0.5 * args.line_step_nm, args.line_step_nm)
        points_axis = center[None, :] + s_grid[:, None] * axis_xyz[None, :]
        phi_actual = _trilinear_sample(phi, spacing, points_axis)
        xb_actual = _trilinear_sample(xb, spacing, points_axis)

        s_neg = _find_crossing(s_grid, phi_actual, args.threshold, positive_side=False)
        s_pos = _find_crossing(s_grid, phi_actual, args.threshold, positive_side=True)
        if s_neg is None or s_pos is None:
            raise RuntimeError(f"Could not locate both interfaces along {axis_name} axis.")

        side_models: dict[str, dict[str, object]] = {}
        for side_name, normal_xyz, s_cross in (
            ("plus", axis_xyz, s_pos),
            ("minus", -axis_xyz, s_neg),
        ):
            local_dir = axes_local @ normal_xyz
            fam_idx = int(np.argmax(family_dirs @ local_dir))
            family_label = family_labels[fam_idx]
            u_grid = np.arange(-args.local_half_range_nm, args.local_half_range_nm + 0.5 * args.line_step_nm, args.line_step_nm)
            if side_name == "plus":
                s_local = s_cross + u_grid
                xi_local_raw = s_local - s_cross
            else:
                s_local = s_cross - u_grid
                xi_local_raw = s_cross - s_local
            points = center[None, :] + s_local[:, None] * axis_xyz[None, :]
            phi_local = _trilinear_sample(phi, spacing, points)
            xb_local = _trilinear_sample(xb, spacing, points)
            cross_local = _find_nearest_crossing(xi_local_raw, phi_local, args.threshold)
            if cross_local is None:
                xi_actual = xi_local_raw
            else:
                xi_actual = xi_local_raw - cross_local
            side_models[side_name] = _choose_model(
                family_label=family_label,
                family_profiles=family_profiles,
                xi_model=xi_model,
                xi_actual=xi_actual,
                phi_actual=phi_local,
                xb_actual=xb_local,
            )

        xi_plus = s_grid - s_pos
        xi_minus = s_neg - s_grid
        phi_plus = _interp_1d_constant(xi_model, np.asarray(side_models["plus"]["phi_model"]), xi_plus)
        phi_minus = _interp_1d_constant(xi_model, np.asarray(side_models["minus"]["phi_model"]), xi_minus)
        phi_recon = np.where(s_grid >= 0.0, phi_plus, phi_minus)

        xb_plus_model = np.asarray(side_models["plus"]["xb_model"])
        xb_minus_model = np.asarray(side_models["minus"]["xb_model"])
        xb_plus_delta = xb_plus_model - float(xb_plus_model[-1])
        xb_minus_delta = xb_minus_model - float(xb_minus_model[-1])
        xb_background_axis = background_at(points_axis)
        xb_plus = xb_background_axis + _interp_1d_constant(xi_model, xb_plus_delta, xi_plus)
        xb_minus = xb_background_axis + _interp_1d_constant(xi_model, xb_minus_delta, xi_minus)
        xb_recon_uncorrected = np.where(s_grid >= 0.0, xb_plus, xb_minus)
        mass_shift = 0.0
        xb_recon = xb_recon_uncorrected
        if args.mass_correction_mode == "global-phi":
            mass_shift = global_mass_shift
            xb_recon = xb_recon_uncorrected + mass_shift
        elif args.mass_correction_mode == "line":
            mass_shift = float(np.mean(xb_actual) - np.mean(xb_recon_uncorrected))
            xb_recon = xb_recon_uncorrected + mass_shift

        phi_rmse = _rmse(phi_actual, phi_recon)
        xb_rmse = _rmse(xb_actual, xb_recon)
        metrics.append({
            "axis": axis_name,
            "s_min_nm": float(s_min),
            "s_max_nm": float(s_max),
            "s_neg_nm": float(s_neg),
            "s_pos_nm": float(s_pos),
            "phi_rmse": phi_rmse,
            "xB_rmse": xb_rmse,
            "xB_matrix_far": xb_matrix_far,
            "xB_background_mode": args.background_mode,
            "xB_mass_shift": mass_shift,
            "plus_family": side_models["plus"]["family"],
            "minus_family": side_models["minus"]["family"],
            "plus_edge_alpha_fit": side_models["plus"]["alpha"],
            "minus_edge_alpha_fit": side_models["minus"]["alpha"],
        })

        ax_phi = axs[row_idx, 0]
        ax_xb = axs[row_idx, 1]
        ax_phi.plot(s_grid, phi_actual, color="#2c3e50", lw=1.8, label="actual")
        ax_phi.plot(s_grid, phi_recon, color="#d35400", lw=1.6, ls="--", label="reconstructed")
        ax_xb.plot(s_grid, xb_actual, color="#2c3e50", lw=1.8, label="actual")
        ax_xb.plot(s_grid, xb_recon, color="#d35400", lw=1.6, ls="--", label="reconstructed")
        ax_phi.set_title(f"{axis_name}-axis phi")
        ax_xb.set_title(f"{axis_name}-axis xB")
        ax_phi.set_ylabel("phi")
        ax_xb.set_ylabel("xB")
        ax_phi.text(0.02, 0.08, f"RMSE={phi_rmse:.4e}", transform=ax_phi.transAxes, fontsize=8)
        ax_xb.text(0.02, 0.08, f"RMSE={xb_rmse:.4e}\nshift={mass_shift:.2e}", transform=ax_xb.transAxes, fontsize=8)
        for ax in (ax_phi, ax_xb):
            ax.set_xlabel(f"s along {axis_name} axis (nm)")
            ax.grid(True, alpha=0.25)
            ax.legend(frameon=False, fontsize=8)

        for s_val, pa, pr, xa, xr, xu in zip(s_grid, phi_actual, phi_recon, xb_actual, xb_recon, xb_recon_uncorrected):
            rows.append({
                "axis": axis_name,
                "s_nm": float(s_val),
                "phi_actual": float(pa),
                "phi_reconstructed": float(pr),
                "xB_actual": float(xa),
                "xB_reconstructed": float(xr),
                "xB_reconstructed_before_mass_shift": float(xu),
            })

    fig.suptitle(f"Whole-box center-axis actual vs reconstructed: {summary_path.parent.name}", fontsize=14)
    plot_png = output_dir / "box_axes_actual_vs_reconstructed.png"
    fig.savefig(plot_png, dpi=180)
    plt.close(fig)

    csv_path = output_dir / "box_axes_actual_vs_reconstructed.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "axis",
                "s_nm",
                "phi_actual",
                "phi_reconstructed",
                "xB_actual",
                "xB_reconstructed",
                "xB_reconstructed_before_mass_shift",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    metrics_json = output_dir / "box_axes_actual_vs_reconstructed_metrics.json"
    metrics_json.write_text(json.dumps({
        "summary_path": str(summary_path),
        "phi_vtk": str(phi_vtk),
        "xb_vtk": str(xb_vtk),
        "family_profiles_csv": str(args.family_profiles_csv.expanduser().resolve()),
        "box_size_nm": [float(v) for v in box_size],
        "center_nm": [float(v) for v in center],
        "xB_matrix_far": xb_matrix_far,
        "xB_background_mode": args.background_mode,
        "xB_background_coeff_a0_ax_ay_az": [float(v) for v in background_coeff],
        "ellipsoid_semiaxes_long_mid_short_nm": [float(v) for v in semiaxes],
        "ellipsoid_background_rho_min_max": (
            [float(rho_bins[0]), float(rho_bins[-1])] if rho_bins.size else None
        ),
        "xB_background_fit_rmse": background_fit_rmse,
        "actual_global_xB_mean": actual_global_mean,
        "reconstructed_global_xB_mean_before_shift": recon_global_mean_before_shift,
        "global_xB_mass_shift": global_mass_shift,
        "mass_correction_mode": args.mass_correction_mode,
        "axes": metrics,
    }, indent=2), encoding="utf-8")

    print(f"summary={summary_path}")
    print(f"phi_vtk={phi_vtk}")
    print(f"xb_vtk={xb_vtk}")
    print(f"plot={plot_png}")
    print(f"profile_csv={csv_path}")
    print(f"metrics_json={metrics_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
