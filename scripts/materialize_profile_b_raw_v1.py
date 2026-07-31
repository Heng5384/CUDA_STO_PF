#!/usr/bin/env python3
"""Materialize the minimized profile-B VTK pair as raw fields.

The raw files are an isolated audit fixture only.  They are not production
parameters and are accepted by the uncommitted zero-mode fresh-profile
extension, not by the clean qualification binary.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def read_vtk(path: Path) -> tuple[np.ndarray, tuple[int, int, int]]:
    with path.open("r", encoding="ascii") as f:
        header = [next(f) for _ in range(10)]
    dims = tuple(int(x) for x in header[4].split()[1:4])
    if header[2].strip() != "ASCII" or not header[8].startswith("SCALARS "):
        raise ValueError(f"unsupported VTK header: {path}")
    arr = np.loadtxt(path, skiprows=10, dtype=np.float64)
    if arr.size != int(np.prod(dims)):
        raise ValueError(f"wrong scalar count in {path}")
    # write_vtk_cuda writes k,j,i loops (x/first coordinate fastest in the
    # scalar stream).  Fortran-order reshape restores runtime i,j,k before
    # writing the C-order raw field expected by --init-mode raw_fields.
    return arr.reshape(dims, order="F"), dims


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--phi-vtk", type=Path, required=True)
    ap.add_argument("--xb-vtk", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    phi, dims = read_vtk(args.phi_vtk)
    xb, dims_xb = read_vtk(args.xb_vtk)
    if dims != dims_xb:
        raise ValueError("phi/xB grids differ")
    args.out.mkdir(parents=True, exist_ok=True)
    phi.astype("<f8").tofile(args.out / "phi_init.raw")
    xb.astype("<f8").tofile(args.out / "xB_init.raw")
    meta = {
        "Nx": dims[0], "Ny": dims[1], "Nz": dims[2],
        "dx_nm": 1.0, "interface_width_nm": 4.0,
        "dtype": "float64", "order": "C",
        "source_phi_vtk": str(args.phi_vtk),
        "source_xB_vtk": str(args.xb_vtk),
        "mean_phi": float(np.mean(phi)),
        "mean_xB": float(np.mean(xb)),
        "mean_h_phi": float(np.mean(phi**3 * (6*phi**2 - 15*phi + 10))),
    }
    (args.out / "init_meta.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(meta, sort_keys=True))


if __name__ == "__main__":
    main()
