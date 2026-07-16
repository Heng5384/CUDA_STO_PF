#!/usr/bin/env python3
"""Run the default-off small-grid FP32 mechanics evidence matrix.

This runner changes only diagnostic controls, Green iteration budget, case tag,
and the explicitly requested dt-repeat probes.  It does not alter mechanics or
PF physics.  A nonzero main_cuda exit is expected while the legacy absolute
mechanical gate is still authoritative; the first mechanics state is dumped
before that rejection.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
SAFETY_FACTOR_FIXED_BEFORE_RUNS = 4.0


def read_params(path: Path) -> dict[str, str]:
    values: dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#"):
            continue
        key, value = line.split("=", 1)
        values[key.strip()] = value.strip()
    return values


def write_params(path: Path, values: dict[str, str]) -> None:
    lines = [
        "# FP32 mechanics floor evidence; physics copied from coarse4 baseline.",
        "# safety_factor=4.0 was frozen before observing this matrix.",
    ]
    lines.extend(f"{key}={values[key]}" for key in sorted(values))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def case_matrix() -> list[tuple[str, int, float]]:
    cases = [(f"iter{iteration}", iteration, 1.0e-6)
             for iteration in (20, 40, 80, 120, 200, 400)]
    cases.extend((f"iter40_repeat{repeat}", 40, 1.0e-6)
                 for repeat in (2, 3))
    cases.extend([
        ("iter40_dt_half", 40, 0.5e-6),
        ("iter40_dt_double", 40, 2.0e-6),
    ])
    return cases


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--binary", type=Path, default=ROOT / "main_cuda")
    parser.add_argument(
        "--baseline", type=Path,
        default=ROOT / "tmp/coarse4_stage1_smoke/finite_lphi_r0p90_elastic1.params",
    )
    parser.add_argument(
        "--state-root", type=Path,
        default=ROOT / "tmp/coarse4_stage1_smoke",
    )
    parser.add_argument(
        "--output-root", type=Path,
        default=ROOT / "tmp/mechanics_fp32_floor_matrix_v2",
    )
    parser.add_argument("--only", action="append", default=[])
    parser.add_argument("--diagnostic-solve-index", type=int, default=1)
    args = parser.parse_args()
    if args.diagnostic_solve_index < 1:
        raise ValueError("diagnostic solve index must be positive")

    binary = args.binary.resolve()
    baseline = args.baseline.resolve()
    state_root = args.state_root.resolve()
    output = args.output_root.resolve()
    inputs = output / "inputs"
    logs = output / "logs"
    results = output / "results"
    evidence = output / "evidence"
    for directory in (inputs, logs, results, evidence):
        directory.mkdir(parents=True, exist_ok=True)

    required = {
        "phi": state_root / "phi_init.raw",
        "xB": state_root / "xB_init.raw",
        "Ctot": state_root / "Ctot_init.raw",
        "meta": state_root / "init_meta.json",
    }
    for path in (binary, baseline, *required.values()):
        if not path.is_file():
            raise FileNotFoundError(path)
    state_meta = json.loads(required["meta"].read_text(encoding="utf-8"))
    grid = (int(state_meta["Nx"]), int(state_meta["Ny"]), int(state_meta["Nz"]))

    baseline_values = read_params(baseline)
    manifest: dict[str, object] = {
        "schema": "mechanics_fp32_floor_matrix_v1",
        "safety_factor_fixed_before_runs": SAFETY_FACTOR_FIXED_BEFORE_RUNS,
        "binary_sha256": sha256(binary),
        "baseline_params_sha256": sha256(baseline),
        "state_sha256": {key: sha256(path) for key, path in required.items()},
        "grid": list(grid),
        "diagnostic_solve_index": args.diagnostic_solve_index,
        "cases": [],
    }
    rows: list[dict[str, object]] = []
    selected = set(args.only)
    for case_id, iterations, dt in case_matrix():
        if selected and case_id not in selected:
            continue
        values = dict(baseline_values)
        values.update({
            # Preserve the historical budget semantics exactly.  The runtime
            # reports Green_iterations=elastic_iter_max-1.
            "elastic_iter_max": str(iterations),
            "dt": f"{dt:.17e}",
            "init_case_tag": f"mechanics_fp32_{case_id}",
            "mechanics_fp32_diagnostics_enabled": "1",
            "mechanics_fp32_dump_first_solve_fields": "1",
            "mechanics_fp32_diagnostic_solve_index":
                str(args.diagnostic_solve_index),
        })
        params = inputs / f"{case_id}.params"
        write_params(params, values)
        command = [
            str(binary), *(str(value) for value in grid), f"{dt:.17e}",
            "1", "1", "1", "1",
            "--mode", "dynamics",
            "--pf-param-file", str(params),
            "--temperature-C", "400",
            "--init-mode", "raw_fields",
            "--init-phi-raw", str(required["phi"]),
            "--init-xB-raw", str(required["xB"]),
            "--init-Ctot-raw", str(required["Ctot"]),
            "--init-meta", str(required["meta"]),
            "--init-case-tag", f"mechanics_fp32_{case_id}",
        ]
        env = dict(os.environ)
        env["CUDA_STO_RESULTS_ROOT"] = str(results)
        log_path = logs / f"{case_id}.log"
        with log_path.open("w", encoding="utf-8") as log:
            completed = subprocess.run(
                command, cwd=ROOT, env=env, stdout=log,
                stderr=subprocess.STDOUT, check=False,
            )
        matches = list(results.glob(
            f"*/mechanics_fp32_{case_id}/mechanics_fp32_diagnostics.json"))
        if len(matches) != 1:
            raise RuntimeError(
                f"{case_id}: expected one diagnostic JSON, found {len(matches)}"
            )
        source_dir = matches[0].parent
        target_dir = evidence / case_id
        if target_dir.exists():
            raise FileExistsError(
                f"refusing to replace existing evidence directory: {target_dir}"
            )
        shutil.copytree(source_dir, target_dir)
        diag = json.loads((target_dir / "mechanics_fp32_diagnostics.json").read_text())
        row = {
            "case_id": case_id,
            "elastic_iter_max": iterations,
            "green_iterations_reported": iterations - 1,
            "dt": dt,
            "exit_code": completed.returncode,
            "params_sha256": sha256(params),
            "legacy_spectral_Linf": diag["legacy_spectral_Linf"],
            "force_Linf": diag["force_Linf"],
            "stress_Linf": diag["stress_Linf"],
            "eta_Linf": diag["eta_Linf"],
            "elastic_energy_mean": diag["elastic_energy_mean"],
            "elastic_phase_driving_L2": diag["elastic_phase_driving_L2"],
            "evidence_dir": str(target_dir.relative_to(ROOT)),
            "log": str(log_path.relative_to(ROOT)),
        }
        rows.append(row)
        manifest["cases"].append(row)
        print(f"{case_id}: exit={completed.returncode} eta={diag['eta_Linf']:.9e}")

    with (output / "floor_matrix.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
