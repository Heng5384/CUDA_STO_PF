#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
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
    compute_mean_xbtot,
    h_phi,
    memmap_array,
    resolve_path,
    write_diagnostics_txt,
    write_summary_csv,
)
from tools.analysis.embed_multiple_reconstructed_nuclei_rescaled import (
    RescaledNucleus,
    chunk_geometry_physical,
    chunk_source_box_margin,
    collect_stats,
    compensate_local_shell,
    compute_delta_diagnostics,
    load_nucleus,
    rebuild_rescaled_profile_chunk,
    source_box_shell_weight_physical,
    source_box_window_physical,
    write_vtk_scalar,
)


def _shape_from_global(global_cfg: dict[str, Any]) -> tuple[int, int, int]:
    if "target_N" in global_cfg:
        vals = global_cfg["target_N"]
        return int(vals[0]), int(vals[1]), int(vals[2])
    return int(global_cfg["Nx"]), int(global_cfg["Ny"]), int(global_cfg["Nz"])


def _axis_coords(n: int, dx_nm: float, origin_nm: float) -> np.ndarray:
    return origin_nm + (np.arange(n, dtype=np.float32) + np.float32(0.5)) * np.float32(dx_nm)


def _zeros_like_memmap(path: Path, shape: tuple[int, int, int], dtype: np.dtype) -> np.memmap:
    return memmap_array(path, shape, dtype, 0.0)


def _copy_memmap(src: np.memmap, dst: np.memmap, chunk_z: int) -> None:
    nz = src.shape[2]
    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        dst[:, :, z0:z1] = np.asarray(src[:, :, z0:z1], dtype=dst.dtype)
    dst.flush()


def _write_xbtot_vtk(path: Path, phi: np.memmap, xb: np.memmap, dx_nm: float, origin_nm: np.ndarray, chunk_z: int) -> None:
    shape = phi.shape
    tmp = np.empty(shape, dtype=np.float32)
    nz = shape[2]
    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        p = np.asarray(phi[:, :, z0:z1], dtype=np.float32)
        x = np.asarray(xb[:, :, z0:z1], dtype=np.float32)
        tmp[:, :, z0:z1] = ((1.0 - h_phi(p)) * x + h_phi(p)).astype(np.float32)
    write_vtk_scalar(path, tmp, path.stem, dx_nm, origin_nm)


def _sample_current_background_shell(
    xB: np.memmap,
    phi: np.memmap,
    existing_wbox: np.memmap,
    nucleus: RescaledNucleus,
    *,
    target_dx_nm: float,
    origin_nm: np.ndarray,
    inner_nm: float,
    outer_nm: float,
    stat: str,
    W_comp_threshold: float,
    phi_matrix_threshold: float,
    chunk_z: int,
) -> dict[str, float]:
    nx, ny, nz = xB.shape
    xs = _axis_coords(nx, target_dx_nm, float(origin_nm[0]))
    ys = _axis_coords(ny, target_dx_nm, float(origin_nm[1]))
    values: list[np.ndarray] = []
    overlap_count = 0
    sample_count_total = 0
    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        zs = _axis_coords(z1 - z0, target_dx_nm, float(origin_nm[2]) + z0 * target_dx_nm)
        qx = xs[:, None, None] - nucleus.center_nm[0]
        qy = ys[None, :, None] - nucleus.center_nm[1]
        qz = zs[None, None, :] - nucleus.center_nm[2]
        m = chunk_source_box_margin(nucleus, qx, qy, qz)
        outside_dist = -m
        shell = (outside_dist >= inner_nm) & (outside_dist <= outer_nm)
        if not np.any(shell):
            continue
        protected = (
            (np.asarray(existing_wbox[:, :, z0:z1], dtype=np.float32) > W_comp_threshold)
            | (np.asarray(phi[:, :, z0:z1], dtype=np.float32) > phi_matrix_threshold)
        )
        overlap_count += int(np.count_nonzero(shell & protected))
        mask = shell & ~protected
        sample_count_total += int(np.count_nonzero(shell))
        if np.any(mask):
            values.append(np.asarray(xB[:, :, z0:z1], dtype=np.float32)[mask])
    if not values:
        raise RuntimeError(
            f"event {nucleus.name}: no valid points for sample-current-background-shell "
            f"(all shell points may overlap protected regions)."
        )
    data = np.concatenate(values)
    mean = float(np.mean(data, dtype=np.float64))
    median = float(np.median(data))
    selected = median if stat == "median" else mean
    return {
        "xB_edge_i": selected,
        "sample_count": float(data.size),
        "raw_shell_count": float(sample_count_total),
        "protected_overlap_count": float(overlap_count),
        "sample_mean": mean,
        "sample_median": median,
        "sample_min": float(np.min(data)),
        "sample_max": float(np.max(data)),
    }


def _check_event_geometry(
    nucleus: RescaledNucleus,
    *,
    box_size_nm: np.ndarray,
    local_comp_outer_nm: float,
    reject_if_outside: bool,
) -> list[str]:
    warnings: list[str] = []
    half = 0.5 * nucleus.source_box_size_nm + local_comp_outer_nm
    lo = nucleus.center_nm - half
    hi = nucleus.center_nm + half
    if np.any(nucleus.center_nm < 0.0) or np.any(nucleus.center_nm > box_size_nm):
        raise ValueError(f"nucleus {nucleus.name}: center outside target box: {nucleus.center_nm.tolist()}")
    if np.any(lo < 0.0) or np.any(hi > box_size_nm):
        msg = (
            f"nucleus {nucleus.name}: source box + local shell crosses target boundary; "
            f"lo={lo.tolist()}, hi={hi.tolist()}, box={box_size_nm.tolist()}"
        )
        if reject_if_outside:
            raise ValueError(msg)
        warnings.append(msg)
    return warnings


def _append_axis_profile(
    rows: list[dict[str, Any]],
    *,
    event_name: str,
    step: int,
    phi: np.memmap,
    xB_before: np.memmap,
    xB_after: np.memmap,
    W_new: np.memmap,
    local_weight: np.memmap,
    target_dx_nm: float,
    origin_nm: np.ndarray,
) -> None:
    nx, ny, nz = phi.shape
    cx, cy, cz = nx // 2, ny // 2, nz // 2
    specs = [
        ("x", [(i, cy, cz) for i in range(nx)], origin_nm[0]),
        ("y", [(cx, j, cz) for j in range(ny)], origin_nm[1]),
        ("z", [(cx, cy, k) for k in range(nz)], origin_nm[2]),
    ]
    for axis, idxs, origin in specs:
        for n, idx in enumerate(idxs):
            i, j, k = idx
            p = float(phi[i, j, k])
            xb0 = float(xB_before[i, j, k])
            xb1 = float(xB_after[i, j, k])
            hp = float(h_phi(np.array([p], dtype=np.float32))[0])
            rows.append(
                {
                    "event": event_name,
                    "step": step,
                    "axis": axis,
                    "index": n,
                    "coord_nm": float(origin + (n + 0.5) * target_dx_nm),
                    "phi": p,
                    "xB_before": xb0,
                    "xB_after": xb1,
                    "delta_xB": xb1 - xb0,
                    "xBtot_before": (1.0 - hp) * xb0 + hp,
                    "xBtot_after": (1.0 - hp) * xb1 + hp,
                    "W_new": float(W_new[i, j, k]),
                    "local_comp_weight": float(local_weight[i, j, k]),
                }
            )


def _write_event_plot(csv_path: Path, out_png: Path, title: str) -> None:
    rows = list(csv.DictReader(csv_path.open()))
    if not rows:
        return
    fig, axs = plt.subplots(3, 3, figsize=(13, 10), sharex=False)
    for ai, axis in enumerate(("x", "y", "z")):
        rr = [r for r in rows if r["axis"] == axis]
        coord = np.array([float(r["coord_nm"]) for r in rr])
        phi = np.array([float(r["phi"]) for r in rr])
        xb0 = np.array([float(r["xB_before"]) for r in rr])
        xb1 = np.array([float(r["xB_after"]) for r in rr])
        dx = np.array([float(r["delta_xB"]) for r in rr])
        w = np.array([float(r["local_comp_weight"]) for r in rr])
        axs[ai, 0].plot(coord, phi, label="phi")
        axs[ai, 0].plot(coord, np.array([float(r["W_new"]) for r in rr]), label="W_new", ls="--")
        axs[ai, 0].set_ylabel(axis)
        axs[ai, 0].legend(fontsize=8)
        axs[ai, 1].plot(coord, xb0, label="xB before")
        axs[ai, 1].plot(coord, xb1, label="xB after", ls="--")
        axs[ai, 1].legend(fontsize=8)
        axs[ai, 2].plot(coord, dx, label="delta xB")
        axs[ai, 2].plot(coord, w * (np.max(np.abs(dx)) if np.max(np.abs(dx)) > 0 else 1.0), label="local weight scaled", ls=":")
        axs[ai, 2].legend(fontsize=8)
    fig.suptitle(title)
    for ax in axs[-1, :]:
        ax.set_xlabel("coordinate nm")
    fig.tight_layout()
    fig.savefig(out_png, dpi=160)
    plt.close(fig)


def _build_nuclei(
    raws: list[dict[str, Any]],
    *,
    config_dir: Path,
    global_cfg: dict[str, Any],
    target_dx_nm: float,
    target_interface_width_nm: float,
    xB_far: float,
) -> list[RescaledNucleus]:
    nuclei: list[RescaledNucleus] = []
    for idx, raw in enumerate(raws):
        nuc = load_nucleus(
            raw,
            idx,
            config_dir=config_dir,
            global_cfg=global_cfg,
            target_dx_nm=target_dx_nm,
            target_interface_width_nm=target_interface_width_nm,
            xB_far=xB_far,
        )
        if nuc is not None:
            nuclei.append(nuc)
    return nuclei


def insert_events_at_step(
    *,
    step: int,
    event_raws: list[dict[str, Any]],
    cfg: dict[str, Any],
    config_dir: Path,
    out_dir: Path,
    phi: np.memmap,
    xB: np.memmap,
    W_existing: np.memmap,
    target_dx_nm: float,
    target_interface_width_nm: float,
    origin_nm: np.ndarray,
    chunk_z: int,
    dtype: np.dtype,
) -> dict[str, Any]:
    global_cfg = cfg["global"]
    shape = phi.shape
    nx, ny, nz = shape
    npts = nx * ny * nz
    W_comp_threshold = float(global_cfg.get("W_comp_threshold", 1.0e-3))
    phi_matrix_threshold = float(global_cfg.get("phi_matrix_threshold", 0.05))
    xB_min = float(global_cfg.get("xB_min", 1e-8))
    xB_max = float(global_cfg.get("xB_max", 0.035))
    xB_far = float(global_cfg.get("xB_background_constant", global_cfg.get("xB_far", 0.030)))
    mass_iters = int(global_cfg.get("mass_correction_iters", 10))
    mass_tol = float(global_cfg.get("mass_tol", 1.0e-9))
    edge_stat_default = str(global_cfg.get("edge_sample_stat", "median"))
    box_size_nm = np.array([nx, ny, nz], dtype=np.float64) * target_dx_nm
    reject_if_outside = bool(global_cfg.get("reject_if_outside", True))
    reject_if_overlap = bool(global_cfg.get("reject_if_overlap", False))
    max_overlap_fraction = float(global_cfg.get("max_new_existing_overlap_fraction", 1.0e-4))

    event_dir = out_dir / f"event_step{step:06d}_{'_'.join(str(r.get('name', i)) for i, r in enumerate(event_raws))}"
    event_dir.mkdir(parents=True, exist_ok=True)
    work = event_dir / "_work_arrays"
    work.mkdir(parents=True, exist_ok=True)
    xB_before = memmap_array(work / "xB_before.dat", shape, dtype, 0.0)
    _copy_memmap(xB, xB_before, chunk_z)

    mean_before, mean_xb_before, mean_phi_before, mean_h_before = compute_mean_xbtot(phi, xB, chunk_z=chunk_z)
    nuclei = _build_nuclei(
        event_raws,
        config_dir=config_dir,
        global_cfg=global_cfg,
        target_dx_nm=target_dx_nm,
        target_interface_width_nm=target_interface_width_nm,
        xB_far=xB_far,
    )
    warnings: list[str] = []
    edge_rows: list[dict[str, Any]] = []
    local_inner = float(global_cfg.get("local_comp_inner_nm", 0.0))
    local_outer = float(global_cfg.get("local_comp_outer_nm", 20.0))
    xs = _axis_coords(nx, target_dx_nm, float(origin_nm[0]))
    ys = _axis_coords(ny, target_dx_nm, float(origin_nm[1]))

    for raw, nucleus in zip(event_raws, nuclei):
        warnings.extend(
            _check_event_geometry(
                nucleus,
                box_size_nm=box_size_nm,
                local_comp_outer_nm=float(raw.get("local_comp_outer_nm", local_outer)),
                reject_if_outside=reject_if_outside,
            )
        )
        mode = str(raw.get("xB_edge_mode", nucleus.xB_edge_mode or global_cfg.get("xB_edge_mode_default", "global-constant")))
        stat = str(raw.get("edge_sample_stat", edge_stat_default))
        if mode == "sample-current-background-shell":
            sample = _sample_current_background_shell(
                xB,
                phi,
                W_existing,
                nucleus,
                target_dx_nm=target_dx_nm,
                origin_nm=origin_nm,
                inner_nm=float(raw.get("xB_edge_shell_inner_margin_nm", global_cfg.get("xB_edge_shell_inner_margin_nm", 0.0))),
                outer_nm=float(raw.get("xB_edge_shell_outer_margin_nm", global_cfg.get("xB_edge_shell_outer_margin_nm", 3.0))),
                stat=stat,
                W_comp_threshold=W_comp_threshold,
                phi_matrix_threshold=phi_matrix_threshold,
                chunk_z=chunk_z,
            )
            nucleus.xB_edge_i = sample["xB_edge_i"]
            nucleus.xB_edge_mode = mode
            nucleus.xB_edge_sample_stat = stat
            nucleus.xB_edge_sample_count = int(sample["sample_count"])
            nucleus.xB_edge_sample_mean = sample["sample_mean"]
            nucleus.xB_edge_sample_median = sample["sample_median"]
            edge_rows.append({"name": nucleus.name, "mode": mode, **sample})
        elif mode == "global-constant":
            nucleus.xB_edge_i = xB_far
            nucleus.xB_edge_mode = mode
            edge_rows.append({"name": nucleus.name, "mode": mode, "xB_edge_i": xB_far, "sample_count": 0.0})
        elif mode == "per-nucleus":
            if nucleus.xB_edge_value_config is None:
                raise ValueError(f"{nucleus.name}: per-nucleus edge mode requires xB_edge_value")
            nucleus.xB_edge_i = float(nucleus.xB_edge_value_config)
            nucleus.xB_edge_mode = mode
            edge_rows.append({"name": nucleus.name, "mode": mode, "xB_edge_i": nucleus.xB_edge_i, "sample_count": 0.0})
        else:
            raise ValueError(f"Unsupported scheduled event xB_edge_mode={mode}")

    W_new = memmap_array(work / "W_new.dat", shape, dtype, 0.0)
    local_weight = memmap_array(work / "local_comp_weight.dat", shape, dtype, 0.0)
    active_new = memmap_array(work / "active_new.dat", shape, np.uint16, 0)
    new_existing_overlap = 0
    new_points = 0

    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        zs = _axis_coords(z1 - z0, target_dx_nm, float(origin_nm[2]) + z0 * target_dx_nm)
        phi_chunk = np.array(phi[:, :, z0:z1], dtype=np.float32, copy=True)
        xb_chunk = np.array(xB[:, :, z0:z1], dtype=np.float32, copy=True)
        w_existing = np.asarray(W_existing[:, :, z0:z1], dtype=np.float32)
        w_new_chunk = np.zeros_like(phi_chunk, dtype=np.float32)
        local_chunk = np.zeros_like(phi_chunk, dtype=np.float32)
        active_chunk = np.zeros(phi_chunk.shape, dtype=np.uint16)
        for raw, nucleus in zip(event_raws, nuclei):
            d_target_nm, family_idx, _rho, q = chunk_geometry_physical(nucleus=nucleus, x_nm=xs, y_nm=ys, z_nm=zs)
            phi_i, xB_local = rebuild_rescaled_profile_chunk(nucleus, d_target_nm, family_idx)
            W_box = source_box_window_physical(nucleus, *q)
            shell = source_box_shell_weight_physical(
                nucleus,
                *q,
                inner_nm=float(raw.get("local_comp_inner_nm", local_inner)),
                outer_nm=float(raw.get("local_comp_outer_nm", local_outer)),
            )
            update_mask = W_box > 0.0
            if np.any(update_mask):
                xB_edge = np.float32(nucleus.xB_edge_i)
                xB_candidate = (xB_edge + W_box * (xB_local - xB_edge)).astype(np.float32)
                xb_chunk[update_mask] = np.minimum(xb_chunk[update_mask], xB_candidate[update_mask])
            phi_chunk = np.maximum(phi_chunk, phi_i)
            w_new_chunk = np.maximum(w_new_chunk, W_box)
            local_chunk = np.maximum(local_chunk, shell)
            active_chunk += (W_box > W_comp_threshold).astype(np.uint16)
        overlap_mask = (w_new_chunk > W_comp_threshold) & (w_existing > W_comp_threshold)
        new_existing_overlap += int(np.count_nonzero(overlap_mask))
        new_points += int(np.count_nonzero(w_new_chunk > W_comp_threshold))
        # Protect old/new source boxes and precipitate/interface from local compensation.
        local_chunk[(w_existing > W_comp_threshold) | (w_new_chunk > W_comp_threshold) | (phi_chunk > phi_matrix_threshold)] = 0.0
        np.clip(xb_chunk, xB_min, xB_max, out=xb_chunk)
        phi[:, :, z0:z1] = phi_chunk.astype(dtype, copy=False)
        xB[:, :, z0:z1] = xb_chunk.astype(dtype, copy=False)
        W_new[:, :, z0:z1] = w_new_chunk.astype(dtype, copy=False)
        local_weight[:, :, z0:z1] = local_chunk.astype(dtype, copy=False)
        active_new[:, :, z0:z1] = active_chunk
    phi.flush()
    xB.flush()
    W_new.flush()
    local_weight.flush()
    active_new.flush()

    overlap_fraction_of_new = new_existing_overlap / max(new_points, 1)
    if overlap_fraction_of_new > 0.0:
        msg = f"new source-box overlap with existing protected source boxes: {overlap_fraction_of_new:.3e} of new protected points"
        if reject_if_overlap and overlap_fraction_of_new > max_overlap_fraction:
            raise RuntimeError(msg)
        warnings.append(msg)

    mean_after_embed, _mean_xb_after_embed, _mean_phi_after_embed, _mean_h_after_embed = compute_mean_xbtot(phi, xB, chunk_z=chunk_z)
    history, correction_total, mass_warnings = compensate_local_shell(
        phi,
        xB,
        local_weight,
        xBtot_target=mean_before,
        xB_min=xB_min,
        xB_max=xB_max,
        iters=mass_iters,
        tol=mass_tol,
        chunk_z=chunk_z,
    )
    warnings.extend(mass_warnings)
    mean_after, mean_xb_after, mean_phi_after, mean_h_after = compute_mean_xbtot(phi, xB, chunk_z=chunk_z)

    # Commit new source boxes only after the event successfully compensated.
    for z0 in range(0, nz, chunk_z):
        z1 = min(nz, z0 + chunk_z)
        W_existing[:, :, z0:z1] = np.maximum(
            np.asarray(W_existing[:, :, z0:z1], dtype=np.float32),
            np.asarray(W_new[:, :, z0:z1], dtype=np.float32),
        ).astype(dtype, copy=False)
    W_existing.flush()

    xB_after = xB
    delta = compute_delta_diagnostics(
        phi,
        xB_before,
        xB_after,
        W_new,
        local_weight,
        W_comp_threshold=W_comp_threshold,
        phi_matrix_threshold=phi_matrix_threshold,
        chunk_z=chunk_z,
    )
    event_stats = {
        "event_step": step,
        "event_names": [str(r.get("name", "")) for r in event_raws],
        "n_nuclei": len(nuclei),
        "mean_xBtot_before_event": mean_before,
        "mean_xB_before_event": mean_xb_before,
        "mean_phi_before_event": mean_phi_before,
        "mean_hphi_before_event": mean_h_before,
        "mean_xBtot_after_embedding_before_compensation": mean_after_embed,
        "mass_jump_before_compensation": mean_after_embed - mean_before,
        "mean_xBtot_after_compensation": mean_after,
        "mean_xB_after_compensation": mean_xb_after,
        "mean_phi_after_compensation": mean_phi_after,
        "mean_hphi_after_compensation": mean_h_after,
        "mass_jump_after_compensation": mean_after - mean_before,
        "event_mass_error": mean_after - mean_before,
        "relative_event_mass_error": (mean_after - mean_before) / mean_before if abs(mean_before) > 0 else math.nan,
        "C_local_total": correction_total,
        "new_existing_overlap_fraction_of_new": overlap_fraction_of_new,
        "correction_history": history,
        "edge_samples": edge_rows,
        "delta_diagnostics": delta,
        "warnings": warnings,
    }
    (event_dir / "event_diagnostics.json").write_text(json.dumps(event_stats, indent=2), encoding="utf-8")
    write_diagnostics_txt(event_dir / "event_diagnostics.txt", event_stats)

    profile_rows: list[dict[str, Any]] = []
    _append_axis_profile(
        profile_rows,
        event_name="+".join(event_stats["event_names"]),
        step=step,
        phi=phi,
        xB_before=xB_before,
        xB_after=xB_after,
        W_new=W_new,
        local_weight=local_weight,
        target_dx_nm=target_dx_nm,
        origin_nm=origin_nm,
    )
    write_summary_csv(event_dir / "event_before_after_profiles.csv", profile_rows)
    _write_event_plot(
        event_dir / "event_before_after_profiles.csv",
        event_dir / "event_before_after_profiles.png",
        f"scheduled insertion step {step}",
    )

    if bool(global_cfg.get("write_intermediate_vtk", False)):
        write_vtk_scalar(event_dir / "phi_after_event.vtk", phi, "phi_after_event", target_dx_nm, origin_nm)
        write_vtk_scalar(event_dir / "xB_before_event.vtk", xB_before, "xB_before_event", target_dx_nm, origin_nm)
        write_vtk_scalar(event_dir / "xB_after_event.vtk", xB_after, "xB_after_event", target_dx_nm, origin_nm)
        write_vtk_scalar(event_dir / "W_box_new_event.vtk", W_new, "W_box_new_event", target_dx_nm, origin_nm)
        write_vtk_scalar(event_dir / "local_comp_weight_event.vtk", local_weight, "local_comp_weight_event", target_dx_nm, origin_nm)
        _write_xbtot_vtk(event_dir / "xBtot_after_event.vtk", phi, xB_after, target_dx_nm, origin_nm, chunk_z)

    return event_stats


def run_insertions_only(config_path: Path, out_dir: Path | None = None) -> dict[str, Any]:
    config_path = config_path.expanduser().resolve()
    config_dir = config_path.parent
    cfg = json.loads(config_path.read_text(encoding="utf-8"))
    global_cfg = cfg.setdefault("global", {})
    if out_dir is None:
        out_dir = resolve_path(global_cfg.get("out_dir"), config_dir=config_dir)
    if out_dir is None:
        out_dir = Path("Results/scheduled_nucleation_insertions_only").resolve()
    out_dir = out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    work = out_dir / "_work_arrays"
    work.mkdir(parents=True, exist_ok=True)

    shape = _shape_from_global(global_cfg)
    dtype = np.dtype(global_cfg.get("dtype", "float32"))
    chunk_z = int(global_cfg.get("chunk_z") or (32 if shape[0] * shape[1] * shape[2] >= 256**3 else shape[2]))
    target_dx_nm = float(global_cfg.get("target_dx_nm", global_cfg.get("dx_nm", 1.0)))
    target_interface_width_nm = float(global_cfg.get("target_interface_width_nm", 4.0))
    origin_nm = np.array(global_cfg.get("origin_nm", [0.0, 0.0, 0.0]), dtype=np.float64)
    initial_phi = float(global_cfg.get("initial_phi", 0.0))
    initial_xB = float(global_cfg.get("initial_xB", global_cfg.get("xB_background_constant", 0.030)))

    phi = memmap_array(work / "phi_current.dat", shape, dtype, initial_phi)
    xB = memmap_array(work / "xB_current.dat", shape, dtype, initial_xB)
    W_existing = memmap_array(work / "W_existing.dat", shape, dtype, 0.0)

    events_by_step: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for raw in cfg.get("nucleation_events", []):
        events_by_step[int(raw["step"])].append(raw)
    if not events_by_step:
        raise ValueError("config has no nucleation_events")

    initial_mean, initial_mean_xb, initial_mean_phi, initial_mean_h = compute_mean_xbtot(phi, xB, chunk_z=chunk_z)
    event_summaries: list[dict[str, Any]] = []
    for step in sorted(events_by_step):
        event_summaries.append(
            insert_events_at_step(
                step=step,
                event_raws=events_by_step[step],
                cfg=cfg,
                config_dir=config_dir,
                out_dir=out_dir,
                phi=phi,
                xB=xB,
                W_existing=W_existing,
                target_dx_nm=target_dx_nm,
                target_interface_width_nm=target_interface_width_nm,
                origin_nm=origin_nm,
                chunk_z=chunk_z,
                dtype=dtype,
            )
        )

    final_mean, final_mean_xb, final_mean_phi, final_mean_h = compute_mean_xbtot(phi, xB, chunk_z=chunk_z)
    stats_final = collect_stats(
        phi,
        xB,
        W_existing,
        memmap_array(work / "dummy_active_phi.dat", shape, np.uint16, 0),
        memmap_array(work / "dummy_active_source.dat", shape, np.uint16, 0),
        xBtot_target=initial_mean,
        xB_min=float(global_cfg.get("xB_min", 1e-8)),
        xB_max=float(global_cfg.get("xB_max", 0.035)),
        W_comp_threshold=float(global_cfg.get("W_comp_threshold", 1.0e-3)),
        phi_matrix_threshold=float(global_cfg.get("phi_matrix_threshold", 0.05)),
        chunk_z=chunk_z,
    )
    summary = {
        "mode": "insertions_only",
        "config": str(config_path),
        "out_dir": str(out_dir),
        "target_N": list(shape),
        "target_dx_nm": target_dx_nm,
        "target_interface_width_nm": target_interface_width_nm,
        "initial_mean_xBtot": initial_mean,
        "initial_mean_xB": initial_mean_xb,
        "initial_mean_phi": initial_mean_phi,
        "initial_mean_hphi": initial_mean_h,
        "final_mean_xBtot": final_mean,
        "final_mean_xB": final_mean_xb,
        "final_mean_phi": final_mean_phi,
        "final_mean_hphi": final_mean_h,
        "final_mass_change": final_mean - initial_mean,
        "final_relative_mass_change": (final_mean - initial_mean) / initial_mean if abs(initial_mean) > 0 else math.nan,
        "final_stats": stats_final,
        "events": event_summaries,
    }
    (out_dir / "scheduled_nucleation_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    rows = []
    for ev in event_summaries:
        rows.append(
            {
                "step": ev["event_step"],
                "event_names": "+".join(ev["event_names"]),
                "n_nuclei": ev["n_nuclei"],
                "mean_xBtot_before": ev["mean_xBtot_before_event"],
                "mean_xBtot_after_embedding": ev["mean_xBtot_after_embedding_before_compensation"],
                "mean_xBtot_after_comp": ev["mean_xBtot_after_compensation"],
                "mass_jump_before_comp": ev["mass_jump_before_compensation"],
                "mass_jump_after_comp": ev["mass_jump_after_compensation"],
                "relative_event_mass_error": ev["relative_event_mass_error"],
                "C_local_total": ev["C_local_total"],
                "new_existing_overlap_fraction_of_new": ev["new_existing_overlap_fraction_of_new"],
                "warnings_count": len(ev["warnings"]),
            }
        )
    write_summary_csv(out_dir / "scheduled_nucleation_summary.csv", rows)
    write_vtk_scalar(out_dir / "final_phi.vtk", phi, "final_phi", target_dx_nm, origin_nm)
    write_vtk_scalar(out_dir / "final_xB.vtk", xB, "final_xB", target_dx_nm, origin_nm)
    _write_xbtot_vtk(out_dir / "final_xBtot.vtk", phi, xB, target_dx_nm, origin_nm, chunk_z)
    print(f"[ok] wrote {out_dir / 'scheduled_nucleation_summary.json'}")
    print(f"[summary] events={len(event_summaries)}, final_mean_xBtot={final_mean:.10e}, rel_change={summary['final_relative_mass_change']:.3e}")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Pure-Python scheduled nucleation insertion dry-run. It validates repeated event insertion and local-shell xBtot compensation without running PF dynamics."
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    run_insertions_only(args.config, args.out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
