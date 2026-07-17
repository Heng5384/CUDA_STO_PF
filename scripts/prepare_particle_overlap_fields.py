#!/usr/bin/env python3
"""Prepare deterministic multiparticle Ctot fields for a PF/oracle overlap test."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def particle(text: str) -> tuple[float, float, float, float]:
    values = tuple(float(x) for x in text.split(","))
    if len(values) != 4 or values[3] <= 0.0:
        raise argparse.ArgumentTypeError("particle must be x,y,z,r with r>0")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--template-meta", type=Path, required=True)
    parser.add_argument("--size", type=int, default=64)
    parser.add_argument("--dx-nm", type=float, default=1.0)
    parser.add_argument("--interface-half-width-nm", type=float, default=2.0)
    parser.add_argument("--matrix-xB", type=float, default=0.012)
    parser.add_argument("--particle", type=particle, action="append", required=True)
    args = parser.parse_args()

    n = args.size
    axes = np.indices((n, n, n), dtype=np.float64)
    phi = np.zeros((n, n, n), dtype=np.float64)
    length = n * args.dx_nm
    for cx, cy, cz, radius in args.particle:
        distance2 = np.zeros_like(phi)
        for axis, center in zip(axes, (cx, cy, cz)):
            delta = (axis * args.dx_nm - center + 0.5 * length) % length - 0.5 * length
            distance2 += delta * delta
        profile = 0.5 * (1.0 - np.tanh(
            (np.sqrt(distance2) - radius) / args.interface_half_width_nm))
        phi = np.maximum(phi, profile)
    hp = h(phi)
    xalpha = np.full_like(phi, args.matrix_xB)
    ctot = hp + (1.0 - hp) * xalpha

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    phi.tofile(out / "phi_init.raw")
    xalpha.tofile(out / "xB_init.raw")
    ctot.tofile(out / "Ctot_init.raw")

    text = args.template_meta.read_text(encoding="utf-8")
    metadata = json.loads(text.replace(": nan", ": null"))
    metadata.update({
        "Nx": n, "Ny": n, "Nz": n, "dx_nm": args.dx_nm,
        "interface_width_nm": 2.0 * args.interface_half_width_nm,
        "bdf2_history_valid": 0, "variable_controller_valid": 0,
        "step": 0, "time_code": 0.0,
        "reference_type": "multiparticle_pf_oracle_overlap_v1",
    })
    metadata.pop("bdf2_Ctot_nm1_file", None)
    metadata.pop("bdf2_phi_nm1_file", None)
    (out / "init_meta.json").write_text(json.dumps(metadata, indent=2) + "\n")
    metrics = {
        "particles_requested": [list(x) for x in args.particle],
        "matrix_xB": args.matrix_xB,
        "h_volume_cells": float(np.sum(hp)),
        "Ctot_inventory": float(np.sum(ctot)),
        "storage_error_Linf": float(np.max(np.abs(ctot - hp - (1.0 - hp) * xalpha))),
        "phi_min": float(np.min(phi)), "phi_max": float(np.max(phi)),
    }
    (out / "initial_metrics.json").write_text(json.dumps(metrics, indent=2) + "\n")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
