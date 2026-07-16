#!/usr/bin/env python3
"""Measure reduction-order sensitivity of frozen FP32 mechanics fields."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_case(spec: str) -> tuple[str, Path]:
    label, separator, path = spec.partition("=")
    if not separator:
        raise argparse.ArgumentTypeError("case must be LABEL=DUMP_DIR")
    return label, Path(path)


def ordered_sum(values: np.ndarray, order: str) -> float:
    values = values.astype(np.float32, copy=False).ravel()
    if order == "numpy_f64":
        return float(np.sum(values.astype(np.float64), dtype=np.float64))
    if order == "numpy_f32_forward":
        return float(np.sum(values, dtype=np.float32))
    if order == "numpy_f32_reverse":
        return float(np.sum(values[::-1], dtype=np.float32))
    if order == "numpy_f32_ascending":
        return float(np.sum(np.sort(values), dtype=np.float32))
    if order == "numpy_f32_descending":
        return float(np.sum(np.sort(values)[::-1], dtype=np.float32))
    if order.startswith("chunked_"):
        chunk_size = int(order.split("_", 1)[1])
        partials = [
            np.sum(values[start : start + chunk_size], dtype=np.float32)
            for start in range(0, values.size, chunk_size)
        ]
        return float(np.sum(np.asarray(partials, dtype=np.float64), dtype=np.float64))
    raise ValueError(order)


def analyze(label: str, dump_dir: Path) -> list[dict[str, object]]:
    meta = json.loads((dump_dir / "mechanics_fp32_diagnostics.json").read_text())
    shape = (int(meta["Nx"]), int(meta["Ny"]), int(meta["Nz"]))
    kshape = (shape[0], shape[1], shape[2] // 2 + 1)
    force_components = []
    for axis in "xyz":
        force_k = np.fromfile(
            dump_dir / f"mechanics_force_{axis}_k_f32.raw", dtype=np.complex64
        ).reshape(kshape)
        force_components.append(
            np.fft.irfftn(force_k.astype(np.complex128), s=shape, axes=(0, 1, 2)).real
        )
    force_sq = sum(component * component for component in force_components)

    stress_components = []
    for name in ("xx", "yy", "zz", "xy", "xz", "yz"):
        stress_components.append(
            np.fromfile(dump_dir / f"mechanics_s{name}_f32.raw", dtype=np.float32).reshape(shape)
        )
    stress_sq = (
        stress_components[0].astype(np.float64) ** 2
        + stress_components[1].astype(np.float64) ** 2
        + stress_components[2].astype(np.float64) ** 2
        + 2.0
        * (
            stress_components[3].astype(np.float64) ** 2
            + stress_components[4].astype(np.float64) ** 2
            + stress_components[5].astype(np.float64) ** 2
        )
    )
    orders = (
        "numpy_f64",
        "numpy_f32_forward",
        "numpy_f32_reverse",
        "numpy_f32_ascending",
        "numpy_f32_descending",
        "chunked_32",
        "chunked_256",
        "chunked_1024",
    )
    rows = []
    for order in orders:
        force_sum = ordered_sum(force_sq, order)
        stress_sum = ordered_sum(stress_sq, order)
        force_l2 = math.sqrt(max(force_sum, 0.0) / force_sq.size)
        stress_l2 = math.sqrt(max(stress_sum, 0.0) / stress_sq.size)
        eta_l2 = force_l2 / max(float(meta["kmax_dealiased"]) * stress_l2, np.finfo(float).tiny)
        rows.append(
            {
                "case": label,
                "order": order,
                "force_l2": force_l2,
                "stress_l2": stress_l2,
                "eta_l2": eta_l2,
                "eta_linf_order_independent": float(meta["eta_Linf"]),
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", action="append", type=parse_case, required=True)
    parser.add_argument("--csv", type=Path, required=True)
    parser.add_argument("--summary", type=Path, required=True)
    args = parser.parse_args()
    rows = []
    for label, directory in args.case:
        rows.extend(analyze(label, directory))
    args.csv.parent.mkdir(parents=True, exist_ok=True)
    with args.csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summaries = []
    for label, _ in args.case:
        values = [float(row["eta_l2"]) for row in rows if row["case"] == label]
        summaries.append(
            {
                "case": label,
                "eta_l2_min": min(values),
                "eta_l2_max": max(values),
                "eta_l2_relative_spread": (max(values) - min(values)) / np.mean(values),
            }
        )
    args.summary.write_text(
        json.dumps({"schema": "mechanics_reduction_order_v1", "cases": summaries}, indent=2) + "\n"
    )
    print(json.dumps(summaries, indent=2))


if __name__ == "__main__":
    main()
