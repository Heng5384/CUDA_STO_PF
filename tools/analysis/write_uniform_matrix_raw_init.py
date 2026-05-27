#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Write a uniform matrix-only raw init bundle for main_cuda --init-mode raw_fields."
    )
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--nx", type=int, required=True)
    parser.add_argument("--ny", type=int, required=True)
    parser.add_argument("--nz", type=int, required=True)
    parser.add_argument("--dx-nm", type=float, required=True)
    parser.add_argument("--interface-width-nm", type=float, required=True)
    parser.add_argument("--xb-out", type=float, required=True)
    parser.add_argument("--dtype", default="float32", choices=("float32",))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = args.out_dir.expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    shape = (args.nx, args.ny, args.nz)
    phi = np.zeros(shape, dtype=np.float32)
    xb = np.full(shape, float(args.xb_out), dtype=np.float32)

    phi_path = out_dir / "phi_init.raw"
    xb_path = out_dir / "xB_init.raw"
    meta_path = out_dir / "init_meta.json"

    phi.tofile(phi_path)
    xb.tofile(xb_path)

    payload = {
        "Nx": args.nx,
        "Ny": args.ny,
        "Nz": args.nz,
        "dx_nm": float(args.dx_nm),
        "interface_width_nm": float(args.interface_width_nm),
        "dtype": args.dtype,
        "order": "C",
        "dt_recommended": None,
        "mean_xBtot": float(args.xb_out),
        "xB_max_safe": float(args.xb_out),
        "reference_type": "matrix_only_same_strain",
        "mean_phi": 0.0,
        "mean_h": 0.0,
        "voxel_count": 0,
    }
    meta_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(out_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
