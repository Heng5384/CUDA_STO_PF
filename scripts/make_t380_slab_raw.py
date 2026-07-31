#!/usr/bin/env python3
import json
import argparse
from pathlib import Path
import numpy as np

ap = argparse.ArgumentParser()
ap.add_argument("--out", type=Path, default=Path(__file__).resolve().parent / "t380_slab_raw_128")
ap.add_argument("--xB", type=float, default=0.004664951821454195)
ap.add_argument("--n", type=int, default=128)
args = ap.parse_args()
out = args.out
out.mkdir(parents=True, exist_ok=True)
n = args.n
dx_nm = 1.0
lambda_nm = 4.0
w = lambda_nm / 2.0
L = n * dx_nm
x = (np.arange(n, dtype=np.float64) + 0.5) * dx_nm
r = np.abs(x - L / 2.0)
phi_1d = 0.5 * (1.0 + np.tanh((16.0 - r) / w))
phi = np.broadcast_to(phi_1d[:, None, None], (n, n, n)).copy()
xb = np.full((n, n, n), args.xB, dtype=np.float64)
h = phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)
mean_xBtot = float(np.mean((1.0 - h) * xb + h))
phi.astype("<f8").ravel(order="C").tofile(out / "phi.raw.f64")
xb.astype("<f8").ravel(order="C").tofile(out / "xB.raw.f64")
(out / "raw_init_meta.json").write_text(json.dumps({
    "Nx": n, "Ny": n, "Nz": n, "dx_nm": dx_nm,
    "interface_width_nm": lambda_nm, "dt_recommended": 0.0001,
    "mean_xBtot": mean_xBtot, "xB_max_safe": 0.95,
    "dtype": "float64", "order": "C"
}, indent=2) + "\n")
print(json.dumps({"out": str(out), "xB": args.xB, "mean_xBtot": mean_xBtot,
                  "phi_min": float(phi.min()), "phi_max": float(phi.max())}))
