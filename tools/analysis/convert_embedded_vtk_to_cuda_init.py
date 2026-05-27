#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def h_phi(phi: np.ndarray) -> np.ndarray:
    p = np.clip(phi, 0.0, 1.0)
    return 6.0 * p**5 - 15.0 * p**4 + 10.0 * p**3


def read_ascii_structured_points(path: Path) -> tuple[np.ndarray, tuple[int, int, int], tuple[float, float, float]]:
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
        payload = f.read().decode("ascii", errors="replace")
    if dims is None:
        raise ValueError(f"Missing DIMENSIONS in {path}")
    values = np.fromstring(payload, sep=" ", dtype=np.float64)
    expected = dims[0] * dims[1] * dims[2]
    if values.size != expected:
        raise ValueError(f"Scalar count mismatch in {path}: got {values.size}, expected {expected}")
    # This converter is for Python-generated embedded VTK, which is written in C order:
    # idx = i*(Ny*Nz) + j*Nz + k, matching main_cuda raw field layout.
    return values.reshape(dims, order="C"), dims, spacing


def stats(arr: np.ndarray) -> dict[str, float]:
    return {
        "min": float(np.min(arr)),
        "max": float(np.max(arr)),
        "mean": float(np.mean(arr, dtype=np.float64)),
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert Python embedded phi/xB ASCII VTK fields to raw arrays for main_cuda --init-mode raw_fields."
    )
    parser.add_argument("--phi-vtk", type=Path, required=True)
    parser.add_argument("--xB-vtk", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--dx-nm", type=float, required=True)
    parser.add_argument("--interface-width-nm", type=float, required=True)
    parser.add_argument("--dt-recommended", type=float, default=0.025)
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--order", choices=("C",), default="C")
    parser.add_argument("--xBtot-target", type=float, default=None)
    parser.add_argument("--xB-max-safe", type=float, default=0.035)
    args = parser.parse_args()

    phi, dims, phi_spacing = read_ascii_structured_points(args.phi_vtk.expanduser().resolve())
    xb, xb_dims, xb_spacing = read_ascii_structured_points(args.xB_vtk.expanduser().resolve())
    if xb_dims != dims:
        raise ValueError(f"phi/xB dimensions differ: phi={dims}, xB={xb_dims}")
    if any(abs(a - b) > 1.0e-9 for a, b in zip(phi_spacing, xb_spacing)):
        raise ValueError(f"phi/xB VTK spacing differs: phi={phi_spacing}, xB={xb_spacing}")
    if any(abs(float(s) - args.dx_nm) > 1.0e-6 for s in phi_spacing):
        print(
            f"[warn] VTK spacing {phi_spacing} differs from --dx-nm={args.dx_nm:g}; "
            "init_meta.json will use --dx-nm for PF validation."
        )

    if not np.all(np.isfinite(phi)) or not np.all(np.isfinite(xb)):
        raise ValueError("phi/xB contains NaN or Inf")

    h = h_phi(phi)
    xbtot = (1.0 - h) * xb + h
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    np_dtype = np.float32 if args.dtype == "float32" else np.float64
    phi_raw = out_dir / "phi_init.raw"
    xb_raw = out_dir / "xB_init.raw"
    phi.astype(np_dtype, copy=False).ravel(order="C").tofile(phi_raw)
    xb.astype(np_dtype, copy=False).ravel(order="C").tofile(xb_raw)

    meta = {
        "Nx": dims[0],
        "Ny": dims[1],
        "Nz": dims[2],
        "dx_nm": float(args.dx_nm),
        "vtk_spacing_nm": list(phi_spacing),
        "interface_width_nm": float(args.interface_width_nm),
        "dt_recommended": float(args.dt_recommended),
        "dtype": args.dtype,
        "order": args.order,
        "phi_raw": str(phi_raw),
        "xB_raw": str(xb_raw),
        "phi": stats(phi),
        "xB": stats(xb),
        "mean_hphi": float(np.mean(h, dtype=np.float64)),
        "mean_xBtot": float(np.mean(xbtot, dtype=np.float64)),
        "xBtot_target": args.xBtot_target,
        "xB_max_safe": float(args.xB_max_safe),
        "xBtot_formula": "xBtot = (1 - h(phi)) * xB + h(phi), h=6phi^5-15phi^4+10phi^3",
        "layout_note": "raw files are C order, idx=i*(Ny*Nz)+j*Nz+k, matching main_cuda field layout",
    }
    (out_dir / "init_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"[ok] wrote {phi_raw}")
    print(f"[ok] wrote {xb_raw}")
    print(f"[ok] wrote {out_dir / 'init_meta.json'}")
    print(
        f"[summary] dims={dims}, dx_nm={args.dx_nm:g}, interface_width_nm={args.interface_width_nm:g}, "
        f"mean_xBtot={meta['mean_xBtot']:.8e}, xB_range=[{meta['xB']['min']:.6g}, {meta['xB']['max']:.6g}]"
    )


if __name__ == "__main__":
    main()
