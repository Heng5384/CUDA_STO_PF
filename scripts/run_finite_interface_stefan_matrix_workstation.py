#!/usr/bin/env python3
"""Run the low-cost planar finite-interface matrix on one workstation GPU."""

from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
from pathlib import Path
import re
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def token(value: float) -> str:
    return f"{value:g}".replace("-", "m").replace(".", "p")


def run_case(
    binary: Path,
    base_params: Path,
    matrix_root: Path,
    dx_nm: float,
    factor: float,
    dt: float,
    nsteps: int,
    checkpoint_every: int,
    outer_max_iter: int,
    antitrapping: bool,
    resume: bool,
) -> dict[str, object]:
    name = f"dx{token(dx_nm)}_L{token(factor)}_at{int(antitrapping)}"
    case = matrix_root / name
    input_dir = case / "input"
    run_dir = case / "run"
    status_path = case / "status.json"
    if status_path.is_file() and resume:
        prior = json.loads(status_path.read_text())
        if prior.get("returncode") == 0:
            return prior
    if case.exists() and not resume:
        raise FileExistsError(f"case already exists: {case}")
    input_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    prepare = [
        sys.executable,
        str(ROOT / "scripts/prepare_one_sided_planar_benchmark.py"),
        "--base-params", str(base_params),
        "--out-dir", str(input_dir),
        "--dx-nm", str(dx_nm),
        "--domain-nm", "19.2",
        "--matrix-xB", "0.05",
        "--sharp-start-time-s", "0.1",
        "--lphi-factor", str(factor),
        "--dt", str(dt),
        "--nsteps", str(nsteps),
        "--mode", "ctot_fv_be",
        "--outer-max-iter", str(outer_max_iter),
    ]
    if antitrapping:
        prepare.append("--finite-interface-antitrapping")
    subprocess.run(prepare, cwd=ROOT, check=True, capture_output=True, text=True)
    manifest = json.loads((input_dir / "benchmark_manifest.json").read_text())
    nx, ny, nz = manifest["grid"]
    command = [
        str(binary), str(nx), str(ny), str(nz), str(dt), str(nsteps),
        str(checkpoint_every), str(checkpoint_every), "0",
        "--mode=dynamics",
        "--pf-param-file", str(input_dir / "benchmark.params"),
        "--temperature-C", "400",
        "--init-mode", "raw_fields",
        "--init-phi-raw", str(input_dir / "phi_init.raw"),
        "--init-xB-raw", str(input_dir / "xB_init.raw"),
        "--init-Ctot-raw", str(input_dir / "Ctot_init.raw"),
        "--init-meta", str(input_dir / "init_meta.json"),
    ]
    environment = os.environ.copy()
    environment["CUDA_STO_RESULTS_ROOT"] = str(run_dir / "results")
    with (run_dir / "run.log").open("w") as log:
        completed = subprocess.run(
            command, cwd=run_dir, env=environment,
            stdout=log, stderr=subprocess.STDOUT, check=False,
        )
    log_text = (run_dir / "run.log").read_text()
    result = {
        "case": name,
        "dx_nm": dx_nm,
        "interface_resolution": manifest["interface_resolution"],
        "L_phi_factor": factor,
        "antitrapping": antitrapping,
        "dt": dt,
        "nsteps": nsteps,
        "checkpoint_every": checkpoint_every,
        "outer_max_iter": outer_max_iter,
        "returncode": completed.returncode,
        "accepted_steps": len(re.findall(
            r"CTOT_(?:FV|SPECTRAL)_BE_ACCEPT[^\n]+", log_text)),
        "retry_count": len(re.findall(r"CTOT_COUPLED_STEP_RETRY", log_text)),
        "binary_sha256": sha256(binary),
        "base_params_sha256": sha256(base_params),
        "generated_params_sha256": sha256(input_dir / "benchmark.params"),
        "command": command,
        "provenance": "finite_interface_asymptotic_audit_not_solver_compensation",
    }
    status_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def csv_floats(text: str) -> list[float]:
    return [float(item) for item in text.split(",") if item.strip()]


def require_unscaled_base_params(path: Path) -> None:
    manifest_path = path.parent / "benchmark_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            "base params require a sibling benchmark_manifest.json proving "
            "L_phi_factor=1"
        )
    manifest = json.loads(manifest_path.read_text())
    factor = float(manifest.get("L_phi_factor", float("nan")))
    if factor != 1.0:
        raise ValueError(
            f"refusing already-scaled base params: L_phi_factor={factor}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--dx-nm", default="0.1")
    parser.add_argument("--lphi-factors", default="16,32,64")
    parser.add_argument("--dt", type=float, default=1.25e-5)
    parser.add_argument("--nsteps", type=int, default=4000)
    parser.add_argument("--checkpoint-every", type=int, default=500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--outer-max-iter", type=int, default=16)
    parser.add_argument("--finite-interface-antitrapping", action="store_true")
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.checkpoint_every < 1 or args.outer_max_iter < 1:
        raise ValueError("workers and checkpoint interval must be positive")
    require_unscaled_base_params(args.base_params.resolve())
    args.matrix_root.mkdir(parents=True, exist_ok=True)
    jobs = [
        (dx, factor)
        for dx in csv_floats(args.dx_nm)
        for factor in csv_floats(args.lphi_factors)
    ]
    results: list[dict[str, object]] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(
            run_case, args.binary.resolve(), args.base_params.resolve(),
            args.matrix_root.resolve(), dx, factor, args.dt, args.nsteps,
            args.checkpoint_every, args.outer_max_iter,
            args.finite_interface_antitrapping,
            args.resume,
        ) for dx, factor in jobs]
        for future in concurrent.futures.as_completed(futures):
            result = future.result()
            results.append(result)
            print(json.dumps(result, sort_keys=True), flush=True)
    failures = sum(int(result["returncode"]) != 0 for result in results)
    print(f"matrix_cases={len(results)}")
    print(f"matrix_failures={failures}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
