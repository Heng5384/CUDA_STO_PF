#!/usr/bin/env python3
"""Host reference and static-contract tests for elastic target profiles."""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
MATERIALIZER = ROOT / "scripts/materialize_pf_elastic_target_profile_v1.py"
SOURCE = ROOT / "main_cuda.cu"
HEADER = ROOT / "pf_params.h"


def h_of_phi(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def solve_logit_shift(
    y_star: np.ndarray,
    phi: np.ndarray,
    target: float,
    v_b: float,
    tolerance: float = 1.0e-13,
) -> tuple[float, np.ndarray]:
    alpha = 1.0 - h_of_phi(phi)

    def evaluate(shift: float) -> tuple[float, float, np.ndarray]:
        y = y_star + shift
        x = np.where(
            y >= 0.0,
            1.0 / (1.0 + np.exp(-y)),
            np.exp(y) / (1.0 + np.exp(y)),
        )
        residual = float(np.sum(alpha * x + v_b * (1.0 - alpha)) - target)
        derivative = float(np.sum(alpha * x * (1.0 - x)))
        return residual, derivative, x

    shift = 0.0
    for _ in range(64):
        residual, derivative, x = evaluate(shift)
        if abs(residual) <= tolerance * max(abs(target), 1.0):
            return shift, x
        if derivative <= 0.0:
            raise RuntimeError("infeasible reference constraint")
        shift -= residual / derivative
    raise RuntimeError("reference constraint did not converge")


def write_vtk(path: Path, field: str, array: np.ndarray) -> None:
    nx, ny, nz = array.shape
    with path.open("w", encoding="ascii") as handle:
        handle.write("# vtk DataFile Version 3.0\n")
        handle.write("synthetic target profile\n")
        handle.write("ASCII\n")
        handle.write("DATASET STRUCTURED_POINTS\n")
        handle.write(f"DIMENSIONS {nx} {ny} {nz}\n")
        handle.write("ORIGIN 0 0 0\n")
        handle.write("SPACING 1 1 1\n")
        handle.write(f"POINT_DATA {array.size}\n")
        handle.write(f"SCALARS {field} double 1\n")
        handle.write("LOOKUP_TABLE default\n")
        for value in array.ravel(order="F"):
            handle.write(f"{value:.17e}\n")


def main() -> int:
    checks: list[tuple[str, bool, str]] = []
    source = SOURCE.read_text(encoding="utf-8")
    header = HEADER.read_text(encoding="utf-8")
    static_needles = (
        "MINIMIZE_CONSERVED_MASS_CONSTRAINT_V1",
        "run_pf_conserved_y_zero_mode_host(",
        "gpu_reduce_sum_xBtot(",
        "minimize_mass_constraint_enabled",
        "minimize_mass_constraint_trace.csv",
        "minimize_min_pseudo_time",
        "--minimize-min-pseudo-time",
    )
    for needle in static_needles:
        checks.append((f"static:{needle}", needle in source or needle in header, ""))

    rng = np.random.default_rng(20260730)
    phi0 = rng.uniform(0.0, 1.0, size=(12, 10, 8))
    xb0 = rng.uniform(0.004, 0.012, size=phi0.shape)
    y0 = np.log(xb0 / (1.0 - xb0))
    target = float(np.sum((1.0 - h_of_phi(phi0)) * xb0 + h_of_phi(phi0)))
    phi1 = np.clip(phi0 + rng.normal(0.0, 0.02, size=phi0.shape), 0.0, 1.0)
    shift, xb1 = solve_logit_shift(y0, phi1, target, 1.0)
    final = float(np.sum((1.0 - h_of_phi(phi1)) * xb1 + h_of_phi(phi1)))
    relative = abs(final - target) / max(abs(target), 1.0)
    checks.append(("reference:exact_mass", relative <= 1.0e-13, f"{relative:.3e}"))
    checks.append(("reference:finite_shift", math.isfinite(shift), f"{shift:.3e}"))

    with tempfile.TemporaryDirectory(prefix="pf_elastic_profile_test_") as raw_tmp:
        tmp = Path(raw_tmp)
        n = 20
        coords = np.arange(n, dtype=float)
        delta = (coords - n / 2.0 + n / 2.0) % n - n / 2.0
        rr = np.sqrt(
            delta[:, None, None] ** 2
            + delta[None, :, None] ** 2
            + delta[None, None, :] ** 2
        )
        phi = 0.5 * (1.0 + np.tanh((4.0 - rr) / 2.0))
        xb = np.full_like(phi, 0.01)
        h = h_of_phi(phi)
        mass = float(np.sum((1.0 - h) * xb + h))
        radius = (3.0 * float(np.sum(h)) / (4.0 * math.pi)) ** (1.0 / 3.0)
        phi_vtk = tmp / "phi.vtk"
        xb_vtk = tmp / "xB.vtk"
        trace = tmp / "minimize_mass_constraint_trace.csv"
        run_log = tmp / "run.log"
        params = tmp / "pf.params"
        output = tmp / "profile"
        write_vtk(phi_vtk, "phi", phi)
        write_vtk(xb_vtk, "xB", xb)
        with trace.open("w", encoding="utf-8", newline="") as handle:
            columns = [
                "iter",
                "target_mass_code",
                "current_mass_code",
                "lambda",
                "residual_code",
                "residual_relative",
                "derivative_code",
                "iterations",
                "newton_steps",
                "bisection_steps",
                "Y_star_min",
                "Y_star_max",
                "lambda_lower",
                "lambda_upper",
                "accepted_constraint_steps",
            ]
            writer = csv.DictWriter(handle, fieldnames=columns)
            writer.writeheader()
            writer.writerow(
                {
                    "iter": 1,
                    "target_mass_code": f"{mass:.17e}",
                    "current_mass_code": f"{mass:.17e}",
                    "lambda": 0.0,
                    "residual_code": 0.0,
                    "residual_relative": 0.0,
                    "derivative_code": 1.0,
                    "iterations": 1,
                    "newton_steps": 1,
                    "bisection_steps": 0,
                    "Y_star_min": -5.0,
                    "Y_star_max": -4.0,
                    "lambda_lower": -1.0,
                    "lambda_upper": 1.0,
                    "accepted_constraint_steps": 1,
                }
            )
        run_log.write_text(
            "MINIMIZE_TARGET_PROFILE_CONVERGENCE_FINAL_AUDIT "
            "status=PASS mode=ELASTIC_CONSERVED_TARGET_PROFILE_V1 "
            "converged=true converged_iter=100 max_iter=1000 "
            "trigger=condA consecutive_required=10 "
            "rms_res=1e-8 rms_dphi=1e-8 rms_dY=1e-8 "
            "energy_diff_rel=1e-10 vol_err_rel=0 mass_err_rel=0\n"
            "MINIMIZE_MASS_CONSTRAINT_FINAL_AUDIT "
            "status=PASS mode=MINIMIZE_CONSERVED_MASS_CONSTRAINT_V1 "
            f"target_mass_code={mass:.17e} final_mass_code={mass:.17e}\n",
            encoding="utf-8",
        )
        params.write_text(
            "elastic_enabled=1\n"
            "enable_gp_assisted_beta_nucleation=0\n"
            "enable_gp_runtime_library_nucleation=0\n"
            "gp_initial_population_enabled=0\n"
            "v_B=1\n",
            encoding="utf-8",
        )
        command = [
            sys.executable,
            str(MATERIALIZER),
            "--phi-vtk",
            str(phi_vtk),
            "--xb-vtk",
            str(xb_vtk),
            "--constraint-trace",
            str(trace),
            "--run-log",
            str(run_log),
            "--param-file",
            str(params),
            "--out",
            str(output),
            "--target-radius-nm",
            f"{radius:.17g}",
            "--source-commit",
            "synthetic",
            "--source-tree-sha256",
            "1" * 64,
            "--binary-sha256",
            "0" * 64,
        ]
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
        checks.append(
            (
                "integration:materializer",
                completed.returncode == 0,
                completed.stderr.strip() or completed.stdout.strip(),
            )
        )
        if completed.returncode == 0:
            manifest = json.loads(
                (output / "profile_manifest.json").read_text(encoding="utf-8")
            )
            checks.append(
                (
                    "integration:canonical_formula",
                    manifest["constraint_contract"]["mass_error_relative"] <= 1.0e-12,
                    f"{manifest['constraint_contract']['mass_error_relative']:.3e}",
                )
            )
            checks.append(
                (
                    "integration:elastic_provenance",
                    int(float(manifest["parameter_values"]["elastic_enabled"])) == 1,
                    "",
                )
            )
            init_meta = json.loads(
                (output / "init_meta.json").read_text(encoding="utf-8")
            )
            checks.append(
                (
                    "integration:runtime_raw_load_contract",
                    init_meta["dtype"] == "float64"
                    and init_meta["order"] == "C"
                    and init_meta["phi_sha256"]
                    == manifest["fields"]["phi"]["sha256"]
                    and init_meta["xB_sha256"]
                    == manifest["fields"]["xB_alpha"]["sha256"],
                    "",
                )
            )

    failed = [row for row in checks if not row[1]]
    payload = {
        "status": "PASS" if not failed else "FAIL",
        "test_count": len(checks),
        "failed_count": len(failed),
        "checks": [
            {"test": name, "status": "PASS" if passed else "FAIL", "detail": detail}
            for name, passed, detail in checks
        ],
    }
    print(json.dumps(payload, indent=2))
    return 0 if not failed else 1


if __name__ == "__main__":
    raise SystemExit(main())
