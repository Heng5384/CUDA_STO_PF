#!/usr/bin/env python3
"""Prepare one matched Correction-2 PF runtime case without changing physics."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import shutil
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402


def parse_params(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-dir", type=Path, required=True)
    parser.add_argument("--base-params", type=Path, required=True)
    parser.add_argument("--selected-mode", type=Path, required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--lphi-ratio", type=float, required=True)
    parser.add_argument("--target-Fo", type=float, required=True)
    parser.add_argument("--dt-code", type=float, default=1.0e-3)
    parser.add_argument("--outer-max-iter", type=int, default=20)
    parser.add_argument("--study", default="unspecified")
    parser.add_argument("--drive-amplitude", type=float, default=math.nan)
    args = parser.parse_args()
    if not (0.0 < args.lphi_ratio < 1.0):
        raise ValueError("Lphi ratio must remain strictly below one")
    if not (args.target_Fo > 0.0 and args.dt_code > 0.0):
        raise ValueError("target Fo and dt must be positive")

    state_meta = json.loads((args.state_dir / "matched_state_meta.json").read_text())
    values = parse_params(args.base_params)
    selected = json.loads(args.selected_mode.read_text())
    selected_ratio = float(selected["ratio_to_L_phi_diff"])
    lphi_diff_code = float(selected["L_phi_research_diff_code"]) / selected_ratio
    lphi_diff_phys = (
        float(selected["L_phi_research_diff_physical_m3_J_s"]) / selected_ratio
    )
    temperature_k = float(state_meta["temperature_C"]) + 273.15
    diffusivity_m2_s = unit.D_Ag_in_PbTe_m2_per_s(temperature_k)
    diffusivity_nm2_s = diffusivity_m2_s * 1.0e18
    lambda_nm = float(state_meta["lambda_nm"])
    elapsed_s = args.target_Fo * lambda_nm**2 / diffusivity_nm2_s
    t0_s = float(values["t_real_unit_s"])
    final_code_time = elapsed_s / t0_s
    nsteps = max(1, int(math.ceil(final_code_time / args.dt_code)))
    dt_code = final_code_time / nsteps
    dx_nm = float(state_meta["dx_nm"])
    d_code = float(values["D_alpha"])
    dx_reference_nm = math.sqrt(diffusivity_m2_s * t0_s / d_code) * 1.0e9
    dx_code = dx_nm / dx_reference_nm

    out = args.out_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    for name in ("phi_init.raw", "xB_init.raw", "Ctot_init.raw", "sharp_state.npz"):
        shutil.copy2(args.state_dir / name, out / name)
    nx = int(state_meta["grid_nx"])
    raw_shape = (nx, 2, 2)
    ctot = np.fromfile(out / "Ctot_init.raw", dtype=np.float64).reshape(raw_shape)
    init_meta = {
        "Nx": nx, "Ny": 2, "Nz": 2,
        "dx_nm": dx_nm,
        "interface_width_nm": lambda_nm,
        "dtype": "float64", "order": "C",
        "dt_recommended": dt_code,
        "mean_xBtot": float(ctot.mean()),
        "xB_max_safe": float(state_meta["xB_max"]),
        "reference_type": "correction2_matched_preaged_planar",
        "source_state": state_meta["case"],
    }
    (out / "init_meta.json").write_text(json.dumps(init_meta, indent=2) + "\n")

    text = args.base_params.read_text()
    if not text.endswith("\n"):
        text += "\n"
    overlay: dict[str, object] = {
        "dx": dx_code, "dy": dx_code, "dz": dx_code,
        "dt": dt_code,
        "ic_phi_iface_w": lambda_nm / (2.0 * dx_nm),
        "composition_evolution_mode": "ctot_mimetic_be",
        "L_phi": args.lphi_ratio * lphi_diff_code,
        "L_phi_code_value": args.lphi_ratio * lphi_diff_code,
        "L_phi_physical_value": args.lphi_ratio * lphi_diff_phys,
        "L_phi_calibration_mode": "one_sided_diffusion_controlled",
        "ctot_nonlinear_max_iter": 500,
        "ctot_step_max_retries": 8,
        "ctot_elastic_validation_enabled": 0,
        "elastic_enabled": 0,
        "ctot_automatic_dt_growth": 0,
        "ctot_outer_max_iter": args.outer_max_iter,
        "ctot_finite_interface_antitrapping_enabled": 0,
        "D_beta": 0.0,
    }
    text += "\n# Correction2 matched-state runtime overlay\n"
    for key, value in overlay.items():
        if isinstance(value, str):
            text += f"{key}={value}\n"
        elif isinstance(value, int):
            text += f"{key}={value}\n"
        else:
            text += f"{key}={float(value):.17e}\n"
    (out / "runtime.params").write_text(text)
    manifest = {
        "schema": "correction2_matched_runtime_case_v1",
        "state_case": state_meta["case"],
        "study": args.study,
        "drive_amplitude": args.drive_amplitude,
        "grid": [nx, 2, 2],
        "dx_nm": dx_nm,
        "dx_code": dx_code,
        "lambda_nm": lambda_nm,
        "lambda_over_dx": lambda_nm / dx_nm,
        "preage_s": state_meta["preage_s"],
        "preage_Fo": state_meta["Fo_lambda_preage"],
        "matrix_xB": state_meta["matrix_xB"],
        "L_phi_ratio": args.lphi_ratio,
        "L_phi_code": args.lphi_ratio * lphi_diff_code,
        "target_incremental_Fo": args.target_Fo,
        "elapsed_s": elapsed_s,
        "final_code_time": final_code_time,
        "dt_code": dt_code,
        "nsteps": nsteps,
        "finite_interface_mode": "off",
        "elasticity": "off",
        "GP_S3": "off",
        "D_beta": 0.0,
        "provenance": "correction2_common_sharp_pf_matched_state",
    }
    (out / "runtime_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(out)
    print(f"nsteps={nsteps}")
    print(f"dt_code={dt_code:.17e}")
    print(f"elapsed_s={elapsed_s:.17e}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
