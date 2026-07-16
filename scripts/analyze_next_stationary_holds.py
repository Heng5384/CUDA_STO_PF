#!/usr/bin/env python3
"""Measure stationary-cylinder drift from authoritative runtime checkpoints."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import re

import numpy as np


def h_stable(phi: np.ndarray) -> np.ndarray:
    direct = phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)
    reflected = (1.0 - phi)**3 * (
        6.0 * (1.0 - phi)**2 - 15.0 * (1.0 - phi) + 10.0
    )
    return np.where(phi > 0.5, 1.0 - reflected, direct)


def checkpoint(case: Path, step: int, field: str) -> Path:
    matches = list(case.rglob(f"ctot_checkpoint_step{step:06d}_{field}.raw"))
    if len(matches) != 1:
        raise RuntimeError(
            f"expected one checkpoint for step={step} field={field}: {matches}"
        )
    return matches[0]


def load(path: Path, shape: tuple[int, int, int]) -> np.ndarray:
    values = np.fromfile(path, dtype=np.float64)
    if values.size != math.prod(shape):
        raise ValueError(f"field size mismatch: {path}")
    return values.reshape(shape)


def radii(phi: np.ndarray, dx_nm: float) -> tuple[float, float]:
    plane = phi.mean(axis=1)
    h_radius = math.sqrt(float(np.sum(h_stable(plane))) * dx_nm**2 / math.pi)
    center = plane.shape[0] // 2
    line = plane[center:, center]
    crossing = np.flatnonzero((line[:-1] >= 0.5) & (line[1:] < 0.5))
    if not crossing.size:
        return h_radius, math.nan
    index = int(crossing[0])
    fraction = (0.5 - line[index]) / (line[index + 1] - line[index])
    return h_radius, (index + float(fraction)) * dx_nm


def energy_endpoints(case: Path) -> tuple[float, dict[int, float], float]:
    paths = list(case.rglob("ctot_energy_work.csv"))
    if len(paths) != 1:
        raise RuntimeError(f"missing energy ledger: {case}")
    with paths[0].open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    accepted = [row for row in rows if row["accepted"] == "1"]
    initial = float(accepted[0]["F_before"])
    endpoints = {
        int(row["physical_step_id"]): float(row["F_final"])
        for row in accepted
    }
    max_balance = max(abs(float(row["energy_balance_residual"]))
                      for row in accepted)
    return initial, endpoints, max_balance


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--oracle-root", type=Path, required=True)
    parser.add_argument("--run-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for ratio in (5, 8, 10, 15, 20):
        oracle = args.oracle_root / f"r{ratio}"
        run = args.run_root / f"r{ratio}"
        manifest = json.loads((oracle / "benchmark_manifest.json").read_text())
        shape = tuple(int(value) for value in manifest["grid"])
        dx_nm = float(manifest["dx_nm"])
        initial_phi = load(oracle / "phi_init.raw", shape)
        initial_c = load(oracle / "Ctot_init.raw", shape)
        r_h0, r_half0 = radii(initial_phi, dx_nm)
        mass0 = float(np.sum(initial_c, dtype=np.float64))
        energy0, energies, max_balance = energy_endpoints(run)
        log = (run / "run.log").read_text(errors="replace")
        accepts = len(re.findall(r"CTOT_MIMETIC_BE_ACCEPT", log))
        retries = len(re.findall(r"CTOT_COUPLED_STEP_RETRY", log))
        rejects = len(re.findall(r"CTOT_COUPLED_STEP_REJECT", log))
        for step in (100, 500, 1000):
            phi = load(checkpoint(run, step, "phi"), shape)
            ctot = load(checkpoint(run, step, "Ctot"), shape)
            r_h, r_half = radii(phi, dx_nm)
            mass_error = abs(float(np.sum(ctot, dtype=np.float64)) - mass0) / abs(mass0)
            rows.append({
                "R_over_lambda": ratio,
                "grid": "x".join(str(value) for value in shape),
                "hold_steps": step,
                "dt_code": manifest["dt_code"],
                "accepted_steps_total": accepts,
                "retries_total": retries,
                "rejects_total": rejects,
                "mass_error_rel": mass_error,
                "phi_half_radius_initial_nm": r_half0,
                "phi_half_radius_nm": r_half,
                "phi_half_radius_drift_nm": r_half - r_half0,
                "h_volume_radius_initial_nm": r_h0,
                "h_volume_radius_nm": r_h,
                "h_volume_radius_drift_nm": r_h - r_h0,
                "F_initial": energy0,
                "F_final": energies[step],
                "energy_drift": energies[step] - energy0,
                "max_energy_balance_residual": max_balance,
            })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"stationary_hold_rows={len(rows)}")
    print(f"output={args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
