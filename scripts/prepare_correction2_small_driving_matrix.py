#!/usr/bin/env python3
"""Prepare the strict below-limit small-driving matrix for Correction Flow 2."""

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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--root", type=Path, default=ROOT / "tmp/correction2_small_driving"
    )
    parser.add_argument("--dt-code", type=float, default=1.0e-3)
    parser.add_argument("--target-Fo", type=float, default=3.0)
    args = parser.parse_args()
    root = args.root.resolve()
    state_root = root / "states"
    case_root = root / "cases"
    state_root.mkdir(parents=True, exist_ok=True)
    case_root.mkdir(parents=True, exist_ok=True)
    x_eq = unit.xAg2Te_eq_from_T(673.15)
    full_x = 0.05
    amplitudes = (1.0, 0.5, 0.25)
    ratios = (0.90, 0.95, 0.98, 0.99)
    cases: list[dict[str, object]] = []
    for amplitude in amplitudes:
        matrix_x = x_eq + amplitude * (full_x - x_eq)
        amplitude_tag = str(amplitude).replace(".", "p")
        family_root = state_root / f"A{amplitude_tag}"
        family = prepare_family(
            family_root, [4.0], domain_nm=19.2, dx_nm=0.075,
            lambda_nm=0.6, sharp_cells=800, matrix_x=matrix_x,
        )
        state_path = ROOT / family["cases"][0]["phi_raw"]
        state_dir = state_path.parent
        for ratio in ratios:
            ratio_tag = str(ratio).replace(".", "p")
            case_id = f"small_A{amplitude_tag}_r{ratio_tag}_Fo{args.target_Fo:g}_dt1e3"
            out = case_root / case_id
            command = [
                sys.executable, str(ROOT / "scripts/prepare_correction2_runtime_case.py"),
                "--state-dir", str(state_dir),
                "--base-params", str(ROOT / "tmp/correction1_Lphi_below_limit/T400_ratio_0p9_dx0p1_dt1/input/benchmark.params"),
                "--selected-mode", str(ROOT / "examples/correction1_selected_diffusion_limit_mode.json"),
                "--out-dir", str(out),
                "--lphi-ratio", str(ratio),
                "--target-Fo", str(args.target_Fo),
                "--dt-code", str(args.dt_code),
                "--study", "small_driving_linearity",
                "--drive-amplitude", str(amplitude),
            ]
            subprocess.run(command, cwd=ROOT, check=True, capture_output=True, text=True)
            manifest = json.loads((out / "runtime_manifest.json").read_text())
            cases.append({"case": case_id, **manifest})
    payload = {
        "schema": "correction2_small_driving_matrix_v1",
        "temperature_C": 400.0,
        "xB_eq": x_eq,
        "full_driving_matrix_xB": full_x,
        "driving_amplitudes": amplitudes,
        "Lphi_ratios": ratios,
        "target_incremental_Fo": args.target_Fo,
        "cases": cases,
        "finite_interface_mode": "off",
        "GP_S3_enabled": False,
    }
    (root / "matrix_manifest.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(f"small_driving_cases={len(cases)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
