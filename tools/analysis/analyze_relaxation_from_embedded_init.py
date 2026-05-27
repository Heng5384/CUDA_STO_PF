#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_csv(path: Path) -> dict[str, np.ndarray]:
    with path.open("r", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        raise ValueError(f"No rows in {path}")
    out: dict[str, list[float]] = {k: [] for k in rows[0].keys()}
    for row in rows:
        for k, v in row.items():
            out[k].append(float(v))
    return {k: np.asarray(v, dtype=float) for k, v in out.items()}


def h_phi(phi: np.ndarray) -> np.ndarray:
    p = np.clip(phi, 0.0, 1.0)
    return 6.0 * p**5 - 15.0 * p**4 + 10.0 * p**3


def read_cuda_vtk(path: Path) -> tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]:
    dims: tuple[int, int, int] | None = None
    spacing = (1.0, 1.0, 1.0)
    with path.open("rb") as f:
        while True:
            line = f.readline()
            if not line:
                raise ValueError(f"Unexpected EOF before LOOKUP_TABLE in {path}")
            text = line.decode("ascii", errors="replace").strip()
            if text.startswith("DIMENSIONS"):
                _, sx, sy, sz = text.split()
                dims = (int(sx), int(sy), int(sz))
            elif text.startswith("SPACING") or text.startswith("ASPECT_RATIO"):
                _, sx, sy, sz = text.split()
                spacing = (float(sx), float(sy), float(sz))
            elif text.startswith("LOOKUP_TABLE"):
                break
        values = np.fromstring(f.read().decode("ascii", errors="replace"), sep=" ", dtype=np.float64)
    if dims is None:
        raise ValueError(f"Missing DIMENSIONS in {path}")
    nx, ny, nz = dims
    if values.size != nx * ny * nz:
        raise ValueError(f"VTK scalar count mismatch in {path}: got {values.size}, expected {nx*ny*nz}")
    # main_cuda writes values with loops k, j, i; convert back to idx=i*(Ny*Nz)+j*Nz+k.
    return values.reshape((nz, ny, nx), order="C").transpose(2, 1, 0), dims, spacing


def find_step_vtk(run_dir: Path, prefix: str, step: int | None) -> Path | None:
    if step is not None:
        p = run_dir / f"{prefix}_{step}.vtk"
        return p if p.exists() else None
    best: tuple[int, Path] | None = None
    pat = re.compile(rf"^{re.escape(prefix)}_(\d+)\.vtk$")
    for p in run_dir.glob(f"{prefix}_*.vtk"):
        m = pat.match(p.name)
        if not m:
            continue
        s = int(m.group(1))
        if best is None or s > best[0]:
            best = (s, p)
    return best[1] if best else None


def plot_diagnostics(data: dict[str, np.ndarray], out: Path, title: str) -> None:
    step = data["step"]
    fig, axs = plt.subplots(4, 1, figsize=(9, 11), constrained_layout=True)
    axs[0].plot(step, data["mean_xBtot"], label="mean_xBtot")
    axs[0].set_ylabel("mean xBtot")
    axs[0].legend()
    axs[1].plot(step, data["relative_drift_vs_initial"], label="relative drift")
    axs[1].axhline(1e-4, color="0.6", linestyle="--")
    axs[1].axhline(-1e-4, color="0.6", linestyle="--")
    axs[1].set_ylabel("drift")
    axs[1].legend()
    axs[2].plot(step, data["xB_min"], label="xB_min")
    axs[2].plot(step, data["xB_max"], label="xB_max")
    axs[2].set_ylabel("xB range")
    axs[2].legend()
    axs[3].plot(step, data["mean_phi"], label="mean_phi")
    axs[3].plot(step, data["mean_hphi"], label="mean_hphi")
    axs[3].set_xlabel("step")
    axs[3].set_ylabel("phi/hphi")
    axs[3].legend()
    fig.suptitle(title)
    fig.savefig(out, dpi=180)
    plt.close(fig)


def plot_before_after(run_dir: Path, out: Path, final_step: int | None, init_meta: dict[str, Any]) -> dict[str, Any]:
    phi0 = run_dir / "phi_init.vtk"
    xb0 = run_dir / "xB_0.vtk"
    xbt0 = run_dir / "xBtot_0.vtk"
    phif = find_step_vtk(run_dir, "phi", final_step)
    xbf = find_step_vtk(run_dir, "xB", final_step)
    xbtf = find_step_vtk(run_dir, "xBtot", final_step)
    paths = [phi0, xb0, xbt0, phif, xbf, xbtf]
    if any(p is None or not Path(p).exists() for p in paths):
        return {"vtk_comparison": "skipped", "reason": "missing initial/final VTK files"}

    phi_i, dims, _ = read_cuda_vtk(phi0)
    xb_i, _, _ = read_cuda_vtk(xb0)
    xbt_i, _, _ = read_cuda_vtk(xbt0)
    phi_f, _, _ = read_cuda_vtk(phif)  # type: ignore[arg-type]
    xb_f, _, _ = read_cuda_vtk(xbf)  # type: ignore[arg-type]
    xbt_f, _, _ = read_cuda_vtk(xbtf)  # type: ignore[arg-type]
    mid = tuple(d // 2 for d in dims)
    fig, axs = plt.subplots(3, 3, figsize=(12, 12), constrained_layout=True)
    fields = [
        ("phi", phi_i[:, :, mid[2]], phi_f[:, :, mid[2]], phi_f[:, :, mid[2]] - phi_i[:, :, mid[2]]),
        ("xB", xb_i[:, :, mid[2]], xb_f[:, :, mid[2]], xb_f[:, :, mid[2]] - xb_i[:, :, mid[2]]),
        ("xBtot", xbt_i[:, :, mid[2]], xbt_f[:, :, mid[2]], xbt_f[:, :, mid[2]] - xbt_i[:, :, mid[2]]),
    ]
    for r, (name, before, after, delta) in enumerate(fields):
        vmin = float(min(np.min(before), np.min(after)))
        vmax = float(max(np.max(before), np.max(after)))
        dmax = float(max(np.max(np.abs(delta)), 1e-12))
        for c, (arr, title, lo, hi) in enumerate(
            [(before, "before", vmin, vmax), (after, "after", vmin, vmax), (delta, "delta", -dmax, dmax)]
        ):
            im = axs[r, c].imshow(arr.T, origin="lower", vmin=lo, vmax=hi)
            axs[r, c].set_title(f"{name} {title}")
            fig.colorbar(im, ax=axs[r, c], shrink=0.75)
    fig.suptitle(
        f"Relaxation before/after mid-z, dx={init_meta.get('dx_nm')} nm, "
        f"lambda={init_meta.get('interface_width_nm')} nm, dt={init_meta.get('dt_recommended')}"
    )
    fig.savefig(out, dpi=180)
    plt.close(fig)
    return {
        "vtk_comparison": "written",
        "dims": list(dims),
        "final_phi_vtk": str(phif),
        "final_xB_vtk": str(xbf),
        "final_xBtot_vtk": str(xbtf),
        "delta_phi_max_abs": float(np.max(np.abs(phi_f - phi_i))),
        "delta_xB_max_abs": float(np.max(np.abs(xb_f - xb_i))),
        "delta_xBtot_max_abs": float(np.max(np.abs(xbt_f - xbt_i))),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze PF relaxation started from Python embedded raw fields.")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--init-diagnostics", type=Path, default=None)
    parser.add_argument("--init-meta", type=Path, required=True)
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--final-step", type=int, default=None)
    args = parser.parse_args()

    run_dir = args.run_dir.expanduser().resolve()
    out_json = run_dir / "relaxation_summary.json"
    out_txt = run_dir / "relaxation_summary.txt"
    init_meta = json.loads(args.init_meta.expanduser().resolve().read_text(encoding="utf-8"))
    init_diag = (
        json.loads(args.init_diagnostics.expanduser().resolve().read_text(encoding="utf-8"))
        if args.init_diagnostics
        else None
    )
    csv_path = args.csv.expanduser().resolve() if args.csv else run_dir / "relaxation_diagnostics.csv"
    data = read_csv(csv_path)
    plot_diagnostics(
        data,
        run_dir / "relaxation_diagnostics.png",
        f"Embedded-init relaxation: dx={init_meta.get('dx_nm')} nm, lambda={init_meta.get('interface_width_nm')} nm",
    )
    vtk_info = plot_before_after(run_dir, run_dir / "relaxation_before_after_slices.png", args.final_step, init_meta)
    final_idx = -1
    summary = {
        "run_dir": str(run_dir),
        "csv": str(csv_path),
        "dx_target_nm": init_meta.get("dx_nm"),
        "interface_width_target_nm": init_meta.get("interface_width_nm"),
        "dt_recommended": init_meta.get("dt_recommended"),
        "steps_recorded": int(data["step"].size),
        "initial_mean_xBtot": float(data["initial_mean_xBtot"][0]),
        "final_step": int(data["step"][final_idx]),
        "final_mean_xBtot": float(data["mean_xBtot"][final_idx]),
        "final_relative_drift": float(data["relative_drift_vs_initial"][final_idx]),
        "max_abs_relative_drift": float(np.max(np.abs(data["relative_drift_vs_initial"]))),
        "xB_min_final": float(data["xB_min"][final_idx]),
        "xB_max_final": float(data["xB_max"][final_idx]),
        "phi_min_final": float(data["phi_min"][final_idx]),
        "phi_max_final": float(data["phi_max"][final_idx]),
        "init_diagnostics": str(args.init_diagnostics) if args.init_diagnostics else None,
        "init_mass_mean_xBtot": init_meta.get("mean_xBtot"),
        "embedded_init_diagnostics_present": init_diag is not None,
        **vtk_info,
    }
    warnings: list[str] = []
    if summary["xB_min_final"] <= 0.0:
        warnings.append("xB_min_final <= 0")
    if summary["xB_max_final"] > float(init_meta.get("xB_max_safe", 0.035)) + 1e-12:
        warnings.append("xB_max_final exceeds xB_max_safe")
    if summary["max_abs_relative_drift"] > 1.0e-3:
        warnings.append("mean_xBtot relative drift exceeded 1e-3")
    elif summary["max_abs_relative_drift"] > 1.0e-4:
        warnings.append("mean_xBtot relative drift exceeded 1e-4")
    summary["warnings"] = warnings
    out_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    lines = [f"{k}: {v}" for k, v in summary.items() if k != "warnings"]
    lines.append("warnings: " + (", ".join(warnings) if warnings else "none"))
    out_txt.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[ok] wrote {out_json}")
    print(f"[ok] wrote {run_dir / 'relaxation_diagnostics.png'}")
    print(f"[summary] final drift={summary['final_relative_drift']:.3e}, xB=[{summary['xB_min_final']:.6g}, {summary['xB_max_final']:.6g}]")


if __name__ == "__main__":
    main()
