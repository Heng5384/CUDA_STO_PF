#!/usr/bin/env python3
"""Summarize the isolated profile A/B/C extension runs without copying VTK."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np


def read_vtk(path: Path) -> tuple[np.ndarray, tuple[int, int, int]]:
    with path.open("r", encoding="ascii") as f:
        h = [next(f) for _ in range(10)]
    dims = tuple(int(x) for x in h[4].split()[1:4])
    a = np.loadtxt(path, skiprows=10, dtype=np.float64)
    if a.size != int(np.prod(dims)):
        raise ValueError(f"{path}: scalar count mismatch")
    # write_vtk_cuda serializes k,j,i; restore runtime i,j,k coordinates.
    return a.reshape(dims, order="F"), dims


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a-phi", type=Path, required=True)
    ap.add_argument("--a-xb", type=Path, required=True)
    ap.add_argument("--a-xbtot", type=Path, required=True)
    ap.add_argument("--b-phi", type=Path, required=True)
    ap.add_argument("--b-xb", type=Path, required=True)
    ap.add_argument("--b-xbtot", type=Path, required=True)
    ap.add_argument("--c-phi", type=Path, required=True)
    ap.add_argument("--c-xb", type=Path, required=True)
    ap.add_argument("--c-xbtot", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    paths = {"A": (args.a_phi, args.a_xb, args.a_xbtot),
             "B": (args.b_phi, args.b_xb, args.b_xbtot),
             "C": (args.c_phi, args.c_xb, args.c_xbtot)}
    fields = {}
    dims = None
    for name, (p, x, xt) in paths.items():
        phi, d = read_vtk(p); xb, dx = read_vtk(x); xbt, dxt = read_vtk(xt)
        if not (d == dx == dxt):
            raise ValueError(f"{name}: grid mismatch")
        if dims is None: dims = d
        if d != dims: raise ValueError("A/B/C grid mismatch")
        fields[name] = (phi, xb, xbt)
    result = {"grid": list(dims), "cases": {}, "differences": {}}
    for name, (phi, xb, xbt) in fields.items():
        hp = h(phi)
        result["cases"][name] = {
            "mean_phi": float(np.mean(phi)), "mean_h_phi": float(np.mean(hp)),
            "mean_xB_matrix": float(np.mean(xb)), "mean_xBtot": float(np.mean(xbt)),
            "phi_min": float(np.min(phi)), "phi_max": float(np.max(phi)),
            "xB_min": float(np.min(xb)), "xB_max": float(np.max(xb)),
        }
    for lhs, rhs in (("A", "B"), ("B", "C"), ("A", "C")):
        p0, x0, _ = fields[lhs]; p1, x1, _ = fields[rhs]
        result["differences"][f"{lhs}_vs_{rhs}"] = {
            "mean_abs_phi": float(np.mean(np.abs(p1-p0))),
            "max_abs_phi": float(np.max(np.abs(p1-p0))),
            "phi_L1_normalized_by_mean_h_A": float(np.mean(np.abs(p1-p0)) / max(np.mean(h(p0)), 1e-300)),
            "mean_abs_xB": float(np.mean(np.abs(x1-x0))),
            "max_abs_xB": float(np.max(np.abs(x1-x0))),
        }
    result["provenance"] = {
        "profile_runtime": "uncommitted_zero_mode_fresh_raw_extension",
        "clean_branch_commit": "6b69895af2d1b86b99c57c5479ff767349c61efe",
        "note": "A/B/C are valid extension evidence; not evidence from the clean committed binary.",
    }
    args.out.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
