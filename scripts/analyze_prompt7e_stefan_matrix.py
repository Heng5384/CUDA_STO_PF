#!/usr/bin/env python3
"""Analyze equal-time Prompt-7e one-sided Stefan benchmark cases."""

from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def crossings(phi: np.ndarray, dx: float) -> list[float]:
    out: list[float] = []
    for i, left in enumerate(phi):
        right = phi[(i + 1) % phi.size]
        if (left - 0.5) * (right - 0.5) < 0.0:
            out.append((i + (0.5 - left) / (right - left)) * dx)
    if len(out) != 2:
        raise RuntimeError(f"expected two crossings, got {out}")
    return out


def one(pattern: str) -> Path:
    matches = [Path(p) for p in glob.glob(pattern, recursive=True)]
    if len(matches) != 1:
        raise RuntimeError(f"expected one file for {pattern}, got {matches}")
    return matches[0]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for case in sorted(path for path in args.matrix_root.iterdir() if path.is_dir()):
        manifest = json.loads((case / "input/benchmark_manifest.json").read_text())
        grid = tuple(manifest["grid"])
        dx = float(manifest["dx_nm"])
        phi0 = np.fromfile(case / "input/phi_init.raw", dtype=np.float64)
        C0 = np.fromfile(case / "input/Ctot_init.raw", dtype=np.float64)
        phi1 = np.fromfile(one(str(case / "run/**/ctot_checkpoint_step*_phi.raw")),
                           dtype=np.float64)
        C1 = np.fromfile(one(str(case / "run/**/ctot_checkpoint_step*_Ctot.raw")),
                         dtype=np.float64)
        x1 = np.fromfile(one(str(case / "run/**/ctot_checkpoint_step*_xB_alpha.raw")),
                         dtype=np.float64)
        area = grid[1] * grid[2]
        p0 = phi0.reshape(grid).mean(axis=(1, 2))
        p1 = phi1.reshape(grid).mean(axis=(1, 2))
        xline = x1.reshape(grid).mean(axis=(1, 2))
        c0 = crossings(p0, dx)
        c1 = crossings(p1, dx)
        displacement = 0.5 * ((c1[1] - c1[0]) - (c0[1] - c0[0]))
        params_text = (case / "input/benchmark.params").read_text()
        values = {}
        for raw in params_text.splitlines():
            line = raw.split("#", 1)[0].strip()
            if "=" in line:
                key, value = line.split("=", 1)
                values[key.strip()] = value.strip()
        t0 = float(values["t_real_unit_s"])
        final_code_time = float(manifest["final_code_time"])
        elapsed_s = final_code_time * t0
        velocity = displacement / elapsed_s
        D_nm2_s = (float(manifest["sharp_diffusion_length_nm"]) ** 2 /
                   float(manifest["sharp_start_time_s"]))
        eta = float(manifest["sharp_similarity_parameter"])
        t_start = float(manifest["sharp_start_time_s"])
        sharp_displacement = 2.0 * eta * math.sqrt(D_nm2_s) * (
            math.sqrt(t_start + elapsed_s) - math.sqrt(t_start))
        sharp_velocity = sharp_displacement / elapsed_s
        temperature_K = float(values["temperature_C"]) + 273.15
        mu_ref = float(values["mu_reference_scale"])
        mu = np.array([
            (unit.mu_Ag2Te(temperature_K, min(max(x, 1.0e-12), 1.0 - 1.0e-12)) -
             unit.mu_PbTe(temperature_K, min(max(x, 1.0e-12), 1.0 - 1.0e-12))) /
            mu_ref for x in xline
        ])
        right_idx = int(round(c1[1] / dx)) % grid[0]
        window = np.array([(right_idx + offset) % grid[0]
                           for offset in range(-12, 13)])
        beta_idx = int(window[np.argmin(np.abs(p1[window] - 0.9))])
        matrix_idx = int(window[np.argmin(np.abs(p1[window] - 0.1))])
        delta_h = float((h(phi1) - h(phi0)).sum())
        delta_q = float(((C1 - h(phi1)) - (C0 - h(phi0))).sum())
        log = (case / "run/run.log").read_text()
        accepts = re.findall(r"CTOT_FV_BE_ACCEPT[^\n]+", log)
        retries = len(re.findall(r"CTOT_COUPLED_STEP_RETRY", log))
        mass_values = [float(value) for value in
                       re.findall(r"CTOT_FV_BE_ACCEPT[^\n]*mass_error=([^ ]+)", log)]
        rows.append({
            "case": case.name,
            "interface_resolution": manifest["interface_resolution"],
            "dx_nm": dx,
            "L_phi_factor": manifest["L_phi_factor"],
            "L_phi_code": manifest["L_phi_code"],
            "requested_dt": manifest["dt_code"],
            "accepted_dt_min": manifest["dt_code"],
            "steps": manifest["nsteps"],
            "elapsed_s": elapsed_s,
            "phi_half_displacement_nm": displacement,
            "phi_half_velocity_nm_s": velocity,
            "sharp_velocity_nm_s": sharp_velocity,
            "velocity_ratio": velocity / sharp_velocity,
            "velocity_error_rel": abs(velocity - sharp_velocity) /
                                  abs(sharp_velocity),
            "chemical_potential_matrix_phi0p1": mu[matrix_idx],
            "chemical_potential_beta_phi0p9": mu[beta_idx],
            "chemical_potential_jump": mu[matrix_idx] - mu[beta_idx],
            "matrix_side_xB_phi0p1": xline[matrix_idx],
            "beta_side_xB_phi0p9_context": xline[beta_idx],
            "beta_side_flux": 0.0,
            "delta_h": delta_h,
            "delta_q": delta_q,
            "stefan_storage_residual_rel": abs(delta_h + delta_q) /
                                           max(abs(delta_h), abs(delta_q), 1.0),
            "mass_error_rel": abs(float((C1 - C0).sum())) /
                              max(abs(float(C0.sum())), 1.0),
            "runtime_mass_error_max": max(map(abs, mass_values), default=math.nan),
            "accepted_steps": len(accepts),
            "retry_count": retries,
            "accepted": len(accepts) == int(manifest["nsteps"]) and retries == 0,
        })
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"stefan_cases={len(rows)}")
    print(f"stefan_failed={sum(not row['accepted'] for row in rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
