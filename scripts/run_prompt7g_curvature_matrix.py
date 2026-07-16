#!/usr/bin/env python3
"""Prepare and run the low-dimensional Prompt-7g curvature matrix."""

from __future__ import annotations

import argparse
import json
import math
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402


def parse_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def solve_gibbs_thomson_x(T: float, radius_nm: float, gamma: float,
                          Vm: float, scale: float) -> float:
    x_eq = unit.xAg2Te_eq_from_T(T)
    mu_eq = unit.mu_Ag2Te(T, x_eq) - unit.mu_PbTe(T, x_eq)
    target = mu_eq + gamma * Vm / (radius_nm * 1.0e-9)
    lo, hi = x_eq, 0.15
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        mu = unit.mu_Ag2Te(T, mid) - unit.mu_PbTe(T, mid)
        if mu < target:
            lo = mid
        else:
            hi = mid
    value = 0.5 * (lo + hi)
    residual = ((unit.mu_Ag2Te(T, value) - unit.mu_PbTe(T, value) - target) /
                scale)
    if abs(residual) > 1.0e-13:
        raise RuntimeError(f"Gibbs-Thomson root residual {residual}")
    return value


def execute(command: list[str], cwd: Path, log: Path) -> None:
    with log.open("w") as handle:
        completed = subprocess.run(command, cwd=cwd, stdout=handle,
                                   stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise RuntimeError(f"command failed rc={completed.returncode}: {command}; {log}")


def prepare(prepare_script: Path, base_params: Path, out: Path,
            points: float, radius: float, domain: float, matrix_x: float,
            dt: float, steps: int, correction: int,
            phi_source: Path | None = None, interface_x: float | None = None,
            start_time: float | None = None) -> None:
    command = [
        sys.executable, str(prepare_script), "--base-params", str(base_params),
        "--out-dir", str(out), "--interface-points", str(points),
        "--mode", "ctot_mimetic_be", "--domain-nm", str(domain),
        "--radius-nm", str(radius), "--matrix-xB", str(matrix_x),
        "--dt", str(dt), "--nsteps", str(steps),
        "--finite-interface-correction", str(correction),
    ]
    if phi_source is not None:
        command += ["--phi-raw-source", str(phi_source)]
    if interface_x is not None and start_time is not None:
        command += ["--radial-diffusion-interface-xB", str(interface_x),
                    "--radial-diffusion-start-time-code", str(start_time)]
    subprocess.run(command, cwd=ROOT, check=True)


def run_runtime(binary: Path, case: Path, steps: int, dt: float) -> None:
    manifest = json.loads((case / "input" / "benchmark_manifest.json").read_text())
    nx, ny, nz = manifest["grid"]
    command = [str(binary), str(nx), str(ny), str(nz), str(dt), str(steps),
               str(max(1, min(steps, 50))), "1", "0", "--pf-param-file",
               "../input/benchmark.params", "--init-mode", "raw_fields",
               "--init-phi-raw", "../input/phi_init.raw", "--init-xB-raw",
               "../input/xB_init.raw", "--init-Ctot-raw",
               "../input/Ctot_init.raw", "--init-meta", "../input/init_meta.json"]
    run_dir = case / "run"
    run_dir.mkdir(parents=True, exist_ok=True)
    execute(command, run_dir, case / "run.log")


def final_phi(case: Path) -> Path:
    candidates = list(case.rglob("ctot_checkpoint_step*_phi.raw"))
    if not candidates:
        raise RuntimeError(f"no relaxed phi checkpoint in {case}")
    return max(candidates, key=lambda path: path.name)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--pre-relax-steps", type=int, default=300)
    parser.add_argument("--moving-steps", type=int, default=200)
    parser.add_argument("--dt", type=float, default=6.25e-6)
    args = parser.parse_args()
    values = parse_params(args.base_params)
    T = float(values["temperature_C"]) + 273.15
    gamma = float(values["gamma_Jm2"])
    scale = float(values["mu_reference_scale"])
    lambda_nm = float(values["lambda_sm_m"]) * 1.0e9
    far_x = 0.0078305391025
    prepare_script = ROOT / "scripts" / "prepare_prompt7f_circular_benchmark.py"
    root = args.output_root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    manifest_rows: list[dict[str, object]] = []
    for ratio in (5.0, 8.0, 10.0, 15.0, 20.0):
        radius = ratio * lambda_nm
        domain = max(16.0, 2.0 * (radius + 4.0 * lambda_nm))
        x_gt = solve_gibbs_thomson_x(
            T, radius, gamma, unit.USER_PHYSICAL_INPUTS.Vm_compound, scale)
        label = f"Rlambda{ratio:g}"
        equilibrium = root / f"{label}_equilibrium"
        prepare(prepare_script, args.base_params, equilibrium / "input", 12.0,
                radius, domain, x_gt, args.dt, args.pre_relax_steps, 0)
        run_runtime(args.binary.resolve(), equilibrium, args.pre_relax_steps, args.dt)
        phi_relaxed = final_phi(equilibrium)
        for correction in (0, 1):
            moving = root / f"{label}_moving_corr{correction}"
            prepare(prepare_script, args.base_params, moving / "input", 12.0,
                    radius, domain, far_x, args.dt, args.moving_steps, correction,
                    phi_relaxed, x_gt, 4.0 / 9.0)
            run_runtime(args.binary.resolve(), moving, args.moving_steps, args.dt)
        manifest_rows.append({
            "R_over_lambda": ratio, "radius_nm": radius,
            "domain_nm": domain, "xB_GT": x_gt,
            "pre_relax_steps": args.pre_relax_steps,
            "moving_steps": args.moving_steps, "dt": args.dt,
            "provenance": "prompt7g_mimetic_curvature_matrix_no_fit",
        })
    (root / "curvature_run_manifest.json").write_text(
        json.dumps(manifest_rows, indent=2) + "\n")
    print(f"curvature_cases_completed={len(manifest_rows) * 3}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
