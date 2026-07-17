#!/usr/bin/env python3
"""Create a periodic moving planar Ctot benchmark without changing physics."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--size", type=int, default=400)
    parser.add_argument("--dx-nm", type=float, default=1.0)
    parser.add_argument("--interface-half-width-nm", type=float, default=2.0)
    parser.add_argument("--slab-half-width-nm", type=float, default=100.0)
    parser.add_argument("--matrix-xB", type=float, default=0.03)
    parser.add_argument("--template-meta", type=Path, required=True)
    args = parser.parse_args()
    if args.size <= 0 or args.size % 2:
        raise ValueError("size must be a positive even integer")

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    coordinate = (np.arange(args.size, dtype=np.float64) - args.size // 2) * args.dx_nm
    distance = np.abs(coordinate)
    phi_line = 0.5 * (1.0 - np.tanh(
        (distance - args.slab_half_width_nm) / args.interface_half_width_nm))
    h_line = h(phi_line)
    ctot_line = h_line + (1.0 - h_line) * args.matrix_xB

    shape = (args.size, args.size, args.size)
    paths = {
        "phi": out / "phi_init.raw",
        "xB_alpha": out / "xB_init.raw",
        "Ctot": out / "Ctot_init.raw",
    }
    for name, line in (
        ("phi", phi_line),
        ("xB_alpha", np.full(args.size, args.matrix_xB)),
        ("Ctot", ctot_line),
    ):
        field = np.memmap(paths[name], dtype=np.float64, mode="w+", shape=shape)
        field[:] = line[:, None, None]
        field.flush()
        del field

    template_text = args.template_meta.read_text(encoding="utf-8")
    metadata = json.loads(template_text.replace(": nan", ": null"))
    metadata.update({
        "schema": "ctot_checkpoint_v1",
        "Nx": args.size,
        "Ny": args.size,
        "Nz": args.size,
        "dx_nm": args.dx_nm,
        "interface_width_nm": 2.0 * args.interface_half_width_nm,
        "dtype": "float64",
        "order": "C",
        "authoritative_state": "Ctot",
        "bdf2_history_valid": 0,
        "variable_controller_valid": 0,
        "step": 0,
        "time_code": 0.0,
        "migrated_from_legacy_restart": False,
        "reference_type": "periodic_planar_moving_interface_400cube_throughput_v1",
    })
    for key in ("bdf2_Ctot_nm1_file", "bdf2_phi_nm1_file"):
        metadata.pop(key, None)
    (out / "init_meta.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    metrics = {
        "grid": list(shape),
        "dx_nm": args.dx_nm,
        "matrix_xB": args.matrix_xB,
        "slab_half_width_nm": args.slab_half_width_nm,
        "interface_half_width_nm": args.interface_half_width_nm,
        "h_volume_cells": float(np.sum(h_line) * args.size * args.size),
        "mean_Ctot": float(np.mean(ctot_line)),
        "hashes": {name: sha256(path) for name, path in paths.items()},
        "role": "performance_state_not_new_physics",
    }
    (out / "profile_metrics.json").write_text(
        json.dumps(metrics, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
