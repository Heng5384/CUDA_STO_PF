#!/usr/bin/env python3
"""Compare the JSON-driven KWN contract with the generated C++ PF probe.

The probe is compiled into a temporary directory.  Thus this gate detects a
stale or non-host-compilable generated header without leaving a binary in the
repository, and it never attempts a CUDA or production PF run.
"""

from __future__ import annotations

import csv
import json
import math
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from kwn_mvp.contract import load_validation_contract  # noqa: E402


CONTRACT_PATH = ROOT / "contracts" / "pf_kwn_validation_contract_v1.json"
OUTPUT_DIR = ROOT / "outputs" / "kwn_pf_state_closure_v1"
OUTPUT_CSV = OUTPUT_DIR / "thermo_cross_language.csv"
SUMMARY_JSON = OUTPUT_DIR / "thermo_cross_language_summary.json"

X_VALUES = (
    1.0e-6,
    3.0e-6,
    1.0e-5,
    3.0e-5,
    1.0e-4,
    3.0e-4,
    1.0e-3,
    2.0e-3,
    4.0e-3,
    0.004649261005504824,
    0.005,
    0.008,
    0.01,
    0.02,
    0.035,
    0.05,
    0.08,
    0.09,
    0.1,
    0.2,
)
RADII_NM = (2.0, 5.0, 10.0, 20.0, 50.0)
RELATIVE_TOLERANCE = 1.0e-10
SOLVUS_ABSOLUTE_TOLERANCE = 1.0e-8


def _csv_values(values: Iterable[float]) -> str:
    return ",".join(format(value, ".17g") for value in values)


def _relative_error(left: float, right: float) -> float:
    return abs(left - right) / max(abs(left), abs(right), 1.0)


def _compile_and_run_probe(contract_hash: str, temperature_k: float) -> list[dict[str, str]]:
    """Compile the generated C++ view and return its full CSV rows."""

    generator = ROOT / "tools" / "generate_pf_contract_header.py"
    subprocess.run(
        [sys.executable, str(generator), "--check"],
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    with tempfile.TemporaryDirectory(prefix="pf_kwn_thermo_probe_") as temporary:
        executable = Path(temporary) / "pf_thermo_probe"
        subprocess.run(
            [
                "c++",
                "-O2",
                "-std=c++14",
                "-Wall",
                "-Wextra",
                "-pedantic",
                "-I.",
                "-o",
                str(executable),
                "tools/pf_thermo_probe.cpp",
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
        result = subprocess.run(
            [
                str(executable),
                "--temperature-k",
                format(temperature_k, ".17g"),
                "--x-values",
                _csv_values(X_VALUES),
                "--radii-nm",
                _csv_values(RADII_NM),
                "--expected-contract-hash",
                contract_hash,
            ],
            cwd=ROOT,
            check=True,
            text=True,
            capture_output=True,
        )
    return list(csv.DictReader(result.stdout.splitlines()))


def main() -> int:
    contract = load_validation_contract(CONTRACT_PATH)
    rows = _compile_and_run_probe(contract.sha256, contract.temperature_k)
    expected_rows = len(X_VALUES) * len(RADII_NM)
    if len(rows) != expected_rows:
        raise RuntimeError(f"probe row count mismatch: expected={expected_rows} actual={len(rows)}")

    output_rows: list[dict[str, object]] = []
    maximum_relative_error = 0.0
    maximum_solvus_absolute_error = 0.0
    convex_enabled = int(contract.value("thermodynamics.convex_extrapolation")["enabled"])
    fields = (
        ("G_alpha_J_mol", contract.g_alpha_j_mol),
        ("mu_A_J_mol", contract.chemical_potential_a_j_mol),
        ("mu_B_J_mol", contract.chemical_potential_b_j_mol),
        ("dGdx_J_mol", contract.mu_alpha_j_mol),
        ("d2Gdx2_J_mol", contract.d2g_alpha_dx2_j_mol),
        ("driving_force_beta_J_mol", contract.beta_driving_force_j_mol),
    )
    for row in rows:
        if row["contract_hash"] != contract.sha256:
            raise RuntimeError("generated C++ probe emitted a different contract hash")
        if int(row["convex_extrapolation_enabled"]) != convex_enabled:
            raise RuntimeError("PF wrapper convex-extrapolation switch differs from contract")
        temperature = float(row["temperature_K"])
        x_b = float(row["xB"])
        radius_m = float(row["radius_m"])
        comparison: dict[str, object] = dict(row)
        row_max = 0.0
        for name, evaluator in fields:
            python_value = evaluator(temperature, x_b)
            cxx_value = float(row[name])
            error = _relative_error(python_value, cxx_value)
            comparison[f"python_{name}"] = python_value
            comparison[f"relative_error_{name}"] = error
            row_max = max(row_max, error)
        python_d = contract.matrix_diffusivity_m2_s(temperature)
        cxx_d = float(row["D_alpha_m2_s"])
        d_error = _relative_error(python_d, cxx_d)
        comparison["python_D_alpha_m2_s"] = python_d
        comparison["relative_error_D_alpha_m2_s"] = d_error
        row_max = max(row_max, d_error)
        python_solvus = contract.planar_solvus_xb(temperature)
        solvus_error = abs(python_solvus - float(row["planar_solvus_xB"]))
        comparison["python_planar_solvus_xB"] = python_solvus
        comparison["absolute_error_planar_solvus_xB"] = solvus_error
        maximum_solvus_absolute_error = max(maximum_solvus_absolute_error, solvus_error)
        python_curvature = contract.curvature_equilibrium_xb(temperature, radius_m)
        curvature_error = _relative_error(
            python_curvature, float(row["curvature_equilibrium_xB"])
        )
        comparison["python_curvature_equilibrium_xB"] = python_curvature
        comparison["relative_error_curvature_equilibrium_xB"] = curvature_error
        row_max = max(row_max, curvature_error)
        for volume_name, expected in (
            ("Vm_alpha_m3_mol", contract.vm_alpha_m3_mol),
            ("Vm_beta_m3_mol", contract.vm_beta_m3_mol),
        ):
            volume_error = _relative_error(expected, float(row[volume_name]))
            comparison[f"relative_error_{volume_name}"] = volume_error
            row_max = max(row_max, volume_error)
        comparison["maximum_relative_error"] = row_max
        comparison["row_status"] = "PASS" if row_max <= RELATIVE_TOLERANCE and solvus_error <= SOLVUS_ABSOLUTE_TOLERANCE else "FAIL"
        maximum_relative_error = max(maximum_relative_error, row_max)
        output_rows.append(comparison)

    solvus = contract.planar_solvus_xb(contract.temperature_k)
    dissolution_drive = contract.beta_driving_force_j_mol(contract.temperature_k, solvus * 0.5)
    growth_drive = contract.beta_driving_force_j_mol(contract.temperature_k, solvus * 2.0)
    direction_pass = dissolution_drive < 0.0 < growth_drive
    status = (
        "PASS_THERMO_VALIDATION_HASH_BIND"
        if (
            maximum_relative_error <= RELATIVE_TOLERANCE
            and maximum_solvus_absolute_error <= SOLVUS_ABSOLUTE_TOLERANCE
            and direction_pass
            and all(row["row_status"] == "PASS" for row in output_rows)
        )
        else "FAIL_THERMO_DIRECTION"
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUTPUT_CSV.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(output_rows[0]),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(output_rows)
    summary = {
        "status": status,
        "contract_hash": contract.sha256,
        "temperature_K": contract.temperature_k,
        "rows": len(output_rows),
        "maximum_relative_error": maximum_relative_error,
        "maximum_solvus_absolute_error": maximum_solvus_absolute_error,
        "relative_tolerance": RELATIVE_TOLERANCE,
        "solvus_absolute_tolerance": SOLVUS_ABSOLUTE_TOLERANCE,
        "dissolution_driving_force_J_mol": dissolution_drive,
        "growth_driving_force_J_mol": growth_drive,
        "direction_pass": direction_pass,
        "convex_extrapolation_enabled": bool(convex_enabled),
        "xB_grid_max": max(X_VALUES),
        "probe": "tools/pf_thermo_probe.cpp compiled host-only through thermo_utils.h and generated/pf_kwn_validation_contract_v1.h",
    }
    SUMMARY_JSON.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if status == "PASS_THERMO_VALIDATION_HASH_BIND" else 2


if __name__ == "__main__":
    raise SystemExit(main())
