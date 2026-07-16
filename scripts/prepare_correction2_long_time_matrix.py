#!/usr/bin/env python3
"""Prepare matched Fo=1/3/10/30 and representative dt/2 cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402
from scripts.prepare_correction2_matched_planar_state import prepare_family  # noqa: E402


def prepare_case(
    state_dir: Path, out: Path, target_fo: float, dt_code: float,
    preage_ratio: float, matrix_x: float,
) -> dict[str, object]:
    command = [
        sys.executable, str(ROOT / "scripts/prepare_correction2_runtime_case.py"),
        "--state-dir", str(state_dir),
        "--base-params", str(ROOT / "tmp/correction1_Lphi_below_limit/T400_ratio_0p9_dx0p1_dt1/input/benchmark.params"),
        "--selected-mode", str(ROOT / "examples/correction1_selected_diffusion_limit_mode.json"),
        "--out-dir", str(out),
        "--lphi-ratio", "0.9",
        "--target-Fo", str(target_fo),
        "--dt-code", str(dt_code),
        "--study", "matched_long_time",
        "--drive-amplitude", "0.25",
    ]
    subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
    manifest_path = out / "runtime_manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["preage_diffusion_width_over_lambda"] = preage_ratio
    manifest["matrix_xB"] = matrix_x
    manifest["dt_refinement"] = dt_code < 7.5e-4
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=ROOT / "tmp/correction2_long_time"
    )
    args = parser.parse_args()
    root = args.root.resolve()
    state_root = root / "states"
    case_root = root / "cases"
    x_eq = unit.xAg2Te_eq_from_T(673.15)
    matrix_x = x_eq + 0.25 * (0.05 - x_eq)
    family = prepare_family(
        state_root, [4.0, 6.0, 8.0], domain_nm=19.2, dx_nm=0.075,
        lambda_nm=0.6, sharp_cells=800, matrix_x=matrix_x,
    )
    cases: list[dict[str, object]] = []
    case_root.mkdir(parents=True, exist_ok=True)
    for state in family["cases"]:
        preage = float(state["requested_diffusion_width_over_lambda"])
        state_dir = (ROOT / state["phi_raw"]).parent
        for target in (1.0, 3.0, 10.0, 30.0):
            case_id = f"long_ell{preage:g}_Fo{target:g}_dt1e3"
            manifest = prepare_case(
                state_dir, case_root / case_id, target, 1.0e-3, preage, matrix_x
            )
            cases.append({"case": case_id, **manifest})
    state = family["cases"][0]
    state_dir = (ROOT / state["phi_raw"]).parent
    for target in (10.0, 30.0):
        case_id = f"long_ell4_Fo{target:g}_dt5e4"
        manifest = prepare_case(
            state_dir, case_root / case_id, target, 5.0e-4, 4.0, matrix_x
        )
        cases.append({"case": case_id, **manifest})
    payload = {
        "schema": "correction2_long_time_matrix_v1",
        "temperature_C": 400.0,
        "drive_amplitude": 0.25,
        "matrix_xB": matrix_x,
        "Lphi_ratio": 0.9,
        "preage_diffusion_width_over_lambda": [4.0, 6.0, 8.0],
        "incremental_Fo": [1.0, 3.0, 10.0, 30.0],
        "cases": cases,
        "finite_interface_mode": "off",
        "GP_S3_enabled": False,
    }
    root.mkdir(parents=True, exist_ok=True)
    (root / "matrix_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"long_time_cases={len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
