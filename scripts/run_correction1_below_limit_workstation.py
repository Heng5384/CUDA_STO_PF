#!/usr/bin/env python3
"""Execute the low-cost T400 Ji--Chen below-limit correction matrix."""

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
sys.path.insert(0, str(ROOT))
from scripts.correction1_ji_chen_mapping import (  # noqa: E402
    corrected_limit,
    load_inputs,
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def parse_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            values[key.strip()] = value.strip()
    return values


def write_corrected_base(
    source: Path, target: Path, lphi_code: float, lphi_physical: float
) -> None:
    text = source.read_text()
    if not text.endswith("\n"):
        text += "\n"
    text += "\n# Correction1 strict Ji-Chen S2 outer-domain reference\n"
    text += "zeta0_phi=1.00000000000000000e+00\n"
    text += f"L_phi={lphi_code:.17e}\n"
    text += f"L_phi_code_value={lphi_code:.17e}\n"
    text += f"L_phi_physical_value={lphi_physical:.17e}\n"
    text += "L_phi_calibration_mode=one_sided_diffusion_controlled\n"
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(text)


def run_case(
    case: dict[str, object], args: argparse.Namespace, corrected_base: Path,
    limit: dict[str, float],
) -> dict[str, object]:
    case_id = str(case["case_id"])
    case_dir = args.matrix_root / case_id
    status_path = case_dir / "status.json"
    if args.resume and status_path.is_file():
        prior = json.loads(status_path.read_text())
        if int(prior.get("returncode", 1)) == 0:
            return prior
    if case_dir.exists() and not args.resume:
        raise FileExistsError(f"refusing existing case without --resume: {case_dir}")
    input_dir = case_dir / "input"
    run_dir = case_dir / "run"
    input_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    ratio = float(case["ratio_to_L_phi_diff"])
    if not 0.0 < ratio < 1.0:
        raise ValueError(f"physical scan ratio must be strictly below one: {ratio}")
    dx_nm = float(case["dx_nm"])
    base_dt = 2.5e-5 * (dx_nm / 0.1) ** 2
    requested_dt = base_dt * float(case["dt_scale"])
    t0_s = limit["time_scale_s"]
    final_code_time = float(case["observation_time_s"]) / t0_s
    nsteps = max(1, int(math.ceil(final_code_time / requested_dt)))
    dt = final_code_time / nsteps
    prepare = [
        sys.executable,
        str(ROOT / "scripts/prepare_one_sided_planar_benchmark.py"),
        "--base-params", str(corrected_base),
        "--out-dir", str(input_dir),
        "--dx-nm", str(dx_nm),
        "--domain-nm", str(args.domain_nm),
        "--ny", "2", "--nz", "2",
        "--matrix-xB", str(case["matrix_xB"]),
        "--sharp-start-time-s", str(args.sharp_start_time_s),
        "--lphi-factor", str(ratio),
        "--dt", str(dt), "--nsteps", str(nsteps),
        "--mode", "ctot_mimetic_be",
        "--outer-max-iter", str(args.outer_max_iter),
    ]
    subprocess.run(prepare, cwd=ROOT, check=True, capture_output=True, text=True)
    manifest_path = input_dir / "benchmark_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest["finite_interface_antitrapping_enabled"]:
        raise RuntimeError("finite-interface correction must remain OFF")
    expected_code = ratio * limit["L_phi_diff_code"]
    if abs(float(manifest["L_phi_code"]) - expected_code) > 2.0e-12:
        raise RuntimeError("prepared L_phi does not match strict below-limit value")
    nx, ny, nz = map(int, manifest["grid"])
    command = [
        str(args.binary), str(nx), str(ny), str(nz), str(dt), str(nsteps),
        str(nsteps), str(nsteps), "0", "--mode=dynamics",
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
    started_wall = time.time()
    started_mono = time.monotonic()
    with (run_dir / "run.log").open("w") as log:
        completed = subprocess.run(
            command, cwd=run_dir, env=environment, stdout=log,
            stderr=subprocess.STDOUT, check=False,
        )
    wall = time.monotonic() - started_mono
    log_text = (run_dir / "run.log").read_text(errors="replace")
    accepts = re.findall(r"CTOT_MIMETIC_BE_ACCEPT[^\n]+", log_text)
    result: dict[str, object] = {
        "case": case_id,
        "purpose": case["purpose"],
        "temperature_C": 400.0,
        "L_phi_factor": ratio,
        "ratio_to_L_phi_diff": ratio,
        "L_phi_reference_code": limit["L_phi_diff_code"],
        "L_phi_reference_physical": limit["L_phi_diff_physical_m3_J_s"],
        "L_phi_code": expected_code,
        "L_phi_physical": ratio * limit["L_phi_diff_physical_m3_J_s"],
        "dx_nm": dx_nm,
        "interface_resolution": manifest["interface_resolution"],
        "dt_scale": case["dt_scale"],
        "dt_code": dt,
        "nsteps": nsteps,
        "observation_time_s": float(case["observation_time_s"]),
        "accepted_steps": len(accepts),
        "retry_count": len(re.findall(r"CTOT_COUPLED_STEP_RETRY", log_text)),
        "reject_count": len(re.findall(r"CTOT_MIMETIC_BE_REJECT", log_text)),
        "returncode": completed.returncode,
        "started_unix_s": started_wall,
        "wall_time_s": wall,
        "finite_interface_mode": "off",
        "elasticity": "off",
        "GP_S3": "off",
        "automatic_dt_growth": "off",
        "binary_sha256": sha256(args.binary),
        "params_sha256": sha256(input_dir / "benchmark.params"),
        "command": command,
        "provenance": "correction1_strict_ji_chen_below_limit_not_growth_tuning",
    }
    status_path.write_text(json.dumps(result, indent=2) + "\n")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, required=True)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--physical-inputs", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--matrix-root", type=Path, required=True)
    parser.add_argument("--domain-nm", type=float, default=19.2)
    parser.add_argument("--sharp-start-time-s", type=float, default=0.1)
    parser.add_argument("--outer-max-iter", type=int, default=16)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()
    args.binary = args.binary.resolve()
    args.base_params = args.base_params.resolve()
    args.physical_inputs = args.physical_inputs.resolve()
    args.manifest = args.manifest.resolve()
    args.matrix_root = args.matrix_root.resolve()
    if not args.binary.is_file() or not args.base_params.is_file():
        raise FileNotFoundError("binary or base params missing")
    manifest = json.loads(args.manifest.read_text())
    inputs = load_inputs(args.physical_inputs)
    limit = corrected_limit(inputs, 400.0)
    corrected_base = args.matrix_root / "shared/corrected_T400_base.params"
    write_corrected_base(
        args.base_params, corrected_base, limit["L_phi_diff_code"],
        limit["L_phi_diff_physical_m3_J_s"],
    )
    rows = []
    for case in manifest["cases"]:
        rows.append(run_case(case, args, corrected_base, limit))
        print(json.dumps(rows[-1], sort_keys=True), flush=True)
    fields = [key for key in rows[0] if key != "command"]
    with (args.matrix_root / "correction1_run_status.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows({key: row[key] for key in fields} for row in rows)
    failed = sum(
        int(row["returncode"]) != 0
        or int(row["accepted_steps"]) != int(row["nsteps"])
        for row in rows
    )
    print(f"below_limit_scan_cases={len(rows)}")
    print(f"below_limit_scan_failures={failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
