#!/usr/bin/env python3
"""Run the Research2 source-free L_phi plateau on one workstation GPU."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import time


ROOT = Path(__file__).resolve().parents[1]
T400_LPHI_REF_CODE = 5.34408846543238347
T400_LPHI_REF_PHYS = 1.71868984663545148e-9


def token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def floats(value: str) -> list[float]:
    return [float(item) for item in value.split(",") if item.strip()]


def parse_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def choose_dt(factor: float, dx_nm: float) -> float:
    # Existing finite-interface-OFF evidence is stable at 2.5e-5 through 32x
    # and 1.25e-5 at 64x for dx=0.1 nm. Scale only for phase stiffness.
    dx_scale = (dx_nm / 0.1) ** 2
    return min(2.5e-5 * dx_scale,
               8.0e-4 * dx_scale / max(factor, 1.0))


def run_case(args: argparse.Namespace, factor: float) -> dict[str, object]:
    case_name = f"T{int(round(args.temperature_C))}_L{token(factor)}"
    case_dir = args.matrix_root / case_name
    input_dir = case_dir / "input"
    run_dir = case_dir / "run"
    status_path = case_dir / "status.json"
    if args.resume and status_path.is_file():
        prior = json.loads(status_path.read_text())
        if int(prior.get("returncode", 1)) == 0:
            return prior
    if case_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing existing case without --resume: {case_dir}")
    input_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)

    base = parse_params(args.base_params)
    base_temperature = float(base["temperature_C"])
    if abs(base_temperature - args.temperature_C) > 1.0e-12:
        raise RuntimeError(
            f"base/runtime temperature mismatch: {base_temperature} vs "
            f"{args.temperature_C}"
        )
    lphi_ref_code = float(base["L_phi_code_value"])
    lphi_ref_phys = float(base["L_phi_physical_value"])
    t0_s = (args.t_real_unit_s if args.t_real_unit_s is not None
            else float(base["t_real_unit"]))
    dt = args.dt if args.dt is not None else choose_dt(factor, args.dx_nm)
    final_code_time = args.observation_time_s / t0_s
    nsteps = max(1, int(math.ceil(final_code_time / dt)))
    dt = final_code_time / nsteps
    checkpoint_every = nsteps

    prepare = [
        sys.executable,
        str(ROOT / "scripts/prepare_one_sided_planar_benchmark.py"),
        "--base-params", str(args.base_params),
        "--out-dir", str(input_dir),
        "--dx-nm", str(args.dx_nm),
        "--domain-nm", str(args.domain_nm),
        "--ny", str(args.ny),
        "--nz", str(args.nz),
        "--matrix-xB", str(args.matrix_xB),
        "--sharp-start-time-s", str(args.sharp_start_time_s),
        "--lphi-factor", str(factor),
        "--dt", str(dt),
        "--nsteps", str(nsteps),
        "--mode", "ctot_mimetic_be",
        "--outer-max-iter", str(args.outer_max_iter),
    ]
    subprocess.run(prepare, cwd=ROOT, check=True, capture_output=True, text=True)
    manifest_path = input_dir / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["finite_interface_antitrapping_enabled"]:
        raise RuntimeError("Research2 plateau requires finite-interface mode OFF")
    if abs(float(manifest["L_phi_code"]) - lphi_ref_code * factor) > 1.0e-11:
        raise RuntimeError("L_phi reference mismatch")
    nx, ny, nz = map(int, manifest["grid"])

    command = [
        str(args.binary), str(nx), str(ny), str(nz), str(dt), str(nsteps),
        str(checkpoint_every), str(checkpoint_every), "0",
        "--mode=dynamics",
        "--pf-param-file", str(input_dir / "benchmark.params"),
        "--temperature-C", str(args.temperature_C),
        "--init-mode", "raw_fields",
        "--init-phi-raw", str(input_dir / "phi_init.raw"),
        "--init-xB-raw", str(input_dir / "xB_init.raw"),
        "--init-Ctot-raw", str(input_dir / "Ctot_init.raw"),
        "--init-meta", str(input_dir / "init_meta.json"),
    ]
    env = os.environ.copy()
    env["CUDA_STO_RESULTS_ROOT"] = str(run_dir / "results")
    started = time.monotonic()
    with (run_dir / "run.log").open("w") as log:
        completed = subprocess.run(
            command, cwd=run_dir, env=env, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    wall_s = time.monotonic() - started
    log_text = (run_dir / "run.log").read_text(errors="replace")
    accepts = re.findall(r"CTOT_MIMETIC_BE_ACCEPT[^\n]+", log_text)
    retries = len(re.findall(r"CTOT_COUPLED_STEP_RETRY", log_text))
    rejects = len(re.findall(r"CTOT_MIMETIC_BE_REJECT", log_text))
    result: dict[str, object] = {
        "case": case_name,
        "temperature_C": args.temperature_C,
        "L_phi_factor": factor,
        "L_phi_reference_code": lphi_ref_code,
        "L_phi_reference_physical": lphi_ref_phys,
        "L_phi_code": lphi_ref_code * factor,
        "dx_nm": args.dx_nm,
        "interface_resolution": manifest["interface_resolution"],
        "dt_code": dt,
        "nsteps": nsteps,
        "observation_time_s": dt * nsteps * t0_s,
        "accepted_steps": len(accepts),
        "retry_count": retries,
        "reject_count": rejects,
        "returncode": completed.returncode,
        "wall_time_s": wall_s,
        "finite_interface_mode": "off",
        "elasticity": "off",
        "GP_S3": "off",
        "automatic_dt_growth": "off",
        "binary_sha256": sha256(args.binary),
        "params_sha256": sha256(input_dir / "benchmark.params"),
        "command": command,
        "provenance": "research2_source_free_fast_interface_plateau_not_mobility_fit",
    }
    status_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--temperature-C", type=float, default=400.0)
    parser.add_argument("--factors", default="0.3,1,3,10,30,100")
    parser.add_argument("--dx-nm", type=float, default=0.1)
    parser.add_argument("--domain-nm", type=float, default=19.2)
    parser.add_argument("--ny", type=int, default=2)
    parser.add_argument("--nz", type=int, default=2)
    parser.add_argument("--matrix-xB", type=float, default=0.05)
    parser.add_argument("--sharp-start-time-s", type=float, default=0.1)
    parser.add_argument("--observation-time-s", type=float,
                        default=2.0821852621180848e-4)
    parser.add_argument("--t-real-unit-s", type=float)
    parser.add_argument("--dt", type=float)
    parser.add_argument("--outer-max-iter", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.base_params = args.base_params.resolve()
    args.matrix_root = args.matrix_root.resolve()
    if not args.binary.is_file() or not args.base_params.is_file():
        raise FileNotFoundError("binary or base params missing")
    args.matrix_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for factor in floats(args.factors):
        row = run_case(args, factor)
        rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)
    status_csv = args.matrix_root / "research2_plateau_run_status.csv"
    fields = [key for key in rows[0] if key != "command"]
    with status_csv.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    failed = sum(int(row["returncode"]) != 0 or
                 int(row["accepted_steps"]) != int(row["nsteps"])
                 for row in rows)
    print(f"research2_plateau_cases={len(rows)}")
    print(f"research2_plateau_failures={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
