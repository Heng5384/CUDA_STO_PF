#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import glob
import json
import math
import re
from pathlib import Path

import numpy as np


def h(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def params(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            out[key.strip()] = value.strip()
    return out


def one(path_pattern: str) -> Path:
    matches = sorted(Path(p) for p in glob.glob(path_pattern, recursive=True))
    if len(matches) != 1:
        raise RuntimeError(f"expected one match for {path_pattern}, got {matches}")
    return matches[0]


def crossing_positions(phi_1d: np.ndarray, dx_nm: float) -> list[float]:
    result: list[float] = []
    for idx, left in enumerate(phi_1d):
        right = phi_1d[(idx + 1) % len(phi_1d)]
        if (left - 0.5) * (right - 0.5) < 0.0:
            fraction = (0.5 - left) / (right - left)
            result.append((idx + fraction) * dx_nm)
    if len(result) != 2:
        raise RuntimeError(f"expected two planar phi=0.5 crossings, got {result}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    rows: list[dict[str, object]] = []
    for case_dir in sorted(p for p in args.matrix_root.iterdir() if p.is_dir()):
        input_dir = case_dir / "input"
        run_dir = case_dir / "run"
        if not input_dir.is_dir() or not run_dir.is_dir():
            continue
        manifest = json.loads((input_dir / "benchmark_manifest.json").read_text())
        values = params(input_dir / "benchmark.params")
        phi0 = np.fromfile(input_dir / "phi_init.raw", dtype=np.float64)
        C0 = np.fromfile(input_dir / "Ctot_init.raw", dtype=np.float64)
        phi1 = np.fromfile(
            one(str(run_dir / "**/ctot_checkpoint_step000001_phi.raw")),
            dtype=np.float64,
        )
        C1 = np.fromfile(
            one(str(run_dir / "**/ctot_checkpoint_step000001_Ctot.raw")),
            dtype=np.float64,
        )
        x1 = np.fromfile(
            one(str(run_dir / "**/ctot_checkpoint_step000001_xB_alpha.raw")),
            dtype=np.float64,
        )
        ny, nz = manifest["grid"][1:]
        area_cells = ny * nz
        dx_nm = float(manifest["dx_nm"])
        dt_code = float(manifest["dt_code"])
        t0_s = float(values["t_real_unit_s"])
        dt_s = dt_code * t0_s
        h0, h1 = h(phi0), h(phi1)
        phi0_1d = phi0.reshape(manifest["grid"]).mean(axis=(1, 2))
        phi1_1d = phi1.reshape(manifest["grid"]).mean(axis=(1, 2))
        crossing0 = crossing_positions(phi0_1d, dx_nm)
        crossing1 = crossing_positions(phi1_1d, dx_nm)
        thickness0_nm = float(h0.sum() / area_cells * dx_nm)
        thickness1_nm = float(h1.sum() / area_cells * dx_nm)
        displacement_nm = 0.5 * (thickness1_nm - thickness0_nm)
        velocity_nm_s = displacement_nm / dt_s
        crossing_displacement_nm = 0.5 * (
            (crossing1[1] - crossing1[0]) -
            (crossing0[1] - crossing0[0])
        )
        crossing_velocity_nm_s = crossing_displacement_nm / dt_s
        start_s = float(manifest["sharp_start_time_s"])
        diffusion_length_nm = float(manifest["sharp_diffusion_length_nm"])
        D_nm2_s = diffusion_length_nm**2 / start_s
        similarity = float(manifest["sharp_similarity_parameter"])
        sharp_displacement_nm = 2.0 * similarity * math.sqrt(D_nm2_s) * (
            math.sqrt(start_s + dt_s) - math.sqrt(start_s)
        )
        sharp_velocity_nm_s = sharp_displacement_nm / dt_s
        q0, q1 = C0 - h0, C1 - h1
        delta_h = float((h1 - h0).sum())
        delta_q = float((q1 - q0).sum())
        delta_C = float((C1 - C0).sum())
        stefan_scale = max(abs(delta_h), abs(delta_q), 1.0e-300)
        interface_index = int(np.argmin(np.abs(phi1 - 0.5)))
        mu_scale = float(values["mu_reference_scale"])
        # The scalar backend values are embedded in the runtime params as the
        # beta reaction reference; report the runtime interface composition and
        # leave the exact reaction driving to the source-backed report parser.
        log = (run_dir / "run.log").read_text()
        accept = re.findall(r"CTOT_FV_BE_ACCEPT[^\n]+", log)
        outer = re.findall(r"outer_iters=(\d+)", accept[-1] if accept else "")
        nonlinear = re.findall(r"nonlinear_iters=(\d+)", accept[-1] if accept else "")
        mass = re.findall(r"mass_error=([^ ]+)", accept[-1] if accept else "")
        rows.append(
            {
                "case": case_dir.name,
                "dx_nm": dx_nm,
                "interface_resolution": manifest["interface_resolution"],
                "L_phi_factor": manifest["L_phi_factor"],
                "L_phi_code": manifest["L_phi_code"],
                "dt_code": dt_code,
                "dt_s": dt_s,
                "h_thickness_initial_nm": thickness0_nm,
                "h_thickness_final_nm": thickness1_nm,
                "interface_displacement_nm": displacement_nm,
                "interface_velocity_nm_s": velocity_nm_s,
                "phi_half_crossing_displacement_nm": crossing_displacement_nm,
                "phi_half_crossing_velocity_nm_s": crossing_velocity_nm_s,
                "sharp_interface_displacement_nm": sharp_displacement_nm,
                "sharp_interface_velocity_nm_s": sharp_velocity_nm_s,
                "velocity_ratio_pf_over_sharp": velocity_nm_s / sharp_velocity_nm_s,
                "velocity_relative_error": abs(velocity_nm_s - sharp_velocity_nm_s)
                / abs(sharp_velocity_nm_s),
                "crossing_velocity_ratio_pf_over_sharp":
                    crossing_velocity_nm_s / sharp_velocity_nm_s,
                "crossing_velocity_relative_error":
                    abs(crossing_velocity_nm_s - sharp_velocity_nm_s) /
                    abs(sharp_velocity_nm_s),
                "delta_h_cell_units": delta_h,
                "delta_q_cell_units": delta_q,
                "delta_C_cell_units": delta_C,
                "stefan_ledger_residual_rel": abs(delta_h + delta_q) / stefan_scale,
                "total_mass_error_rel": abs(delta_C) / max(abs(float(C0.sum())), 1.0),
                "matrix_xB_at_phi_half": float(x1[interface_index]),
                "matrix_xB_min": float(x1.min()),
                "matrix_xB_max": float(x1.max()),
                "phi_min": float(phi1.min()),
                "phi_max": float(phi1.max()),
                "beta_side_flux": 0.0,
                "beta_side_flux_basis": "exact_closed_face_one_sided_mobility",
                "mu_reference_scale": mu_scale,
                "nonlinear_iters_final_outer": int(nonlinear[-1]) if nonlinear else -1,
                "outer_iters": int(outer[-1]) if outer else -1,
                "runtime_mass_error": float(mass[-1]) if mass else math.nan,
                "accepted": bool(accept),
            }
        )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
