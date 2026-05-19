#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.analysis.compute_schur_rc_predictions import (
    DEFAULT_MODES,
    DEFAULT_STRAINS,
    build_cases,
    calc_elastic_energy_schur,
    get_chemical_driving_force_Jm3,
    get_principal_eigenstrain,
    load_inputs,
)
from tools.analysis.workflow_utils import (
    build_cnt_prefix,
    case_output_rel,
    ensure_workflow_subdirs,
    reference_case_output_rel,
    workflow_name,
)


def _parse_csv_floats(text: str) -> list[float]:
    return [float(x.strip()) for x in text.split(",") if x.strip()]


def _parse_csv_strings(text: str) -> list[str]:
    return [x.strip() for x in text.split(",") if x.strip()]


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _scan_radii(rc_nm: float, window_nm: float, step_nm: float) -> list[float]:
    count = int(round(2.0 * window_nm / step_nm)) + 1
    return [round(rc_nm - window_nm + i * step_nm, 6) for i in range(count)]


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a standardized CNT workflow bundle: physical inputs, Schur prediction table, and a single CNT scan guide CSV."
    )
    parser.add_argument("--temp-c", type=float, required=True, help="temperature in Celsius")
    parser.add_argument("--xb-out", type=float, required=True, help="far-field matrix xB")
    parser.add_argument("--base-json", default="physical_inputs.example.json", help="base physical input json")
    parser.add_argument("--workflow-root", default="Results/workflows", help="workflow bundle root")
    parser.add_argument("--workflow-name", default=None, help="optional explicit workflow directory name")
    parser.add_argument("--lambda-sm-nm", type=float, default=None, help="optional override for lambda_sm in nm")
    parser.add_argument("--pf-dx-nm", type=float, default=None, help="optional override for pf_dx in nm")
    parser.add_argument("--phys-dx-ref-nm", type=float, default=None, help="optional override for phys_dx_ref in nm")
    parser.add_argument("--grid", default="400,400,400", help="NX,NY,NZ")
    parser.add_argument("--dt", type=float, default=0.1, help="simulation dt for CNT scan")
    parser.add_argument("--steps", type=int, default=30000, help="CNT scan steps per point")
    parser.add_argument("--out-every", type=int, default=250, help="VTK output interval")
    parser.add_argument("--csv-out-every", type=int, default=10, help="CSV output interval")
    parser.add_argument("--elastic", type=int, default=1, help="elastic flag")
    parser.add_argument("--window-nm", type=float, default=0.5, help="radius scan half-window around Schur prediction")
    parser.add_argument("--step-nm", type=float, default=0.1, help="radius scan step")
    parser.add_argument("--strains", default="0,0.0025,0.005,0.0075,0.01", help="strain magnitudes")
    parser.add_argument("--modes", default="exx,eyy,exx_eyy", help="load modes")
    args = parser.parse_args()

    nx, ny, nz = [int(x.strip()) for x in args.grid.split(",")]
    base_json = Path(args.base_json).expanduser().resolve()
    workflow_root = Path(args.workflow_root).expanduser().resolve()

    raw_inputs = load_inputs(base_json)
    raw_inputs["temperature_C"] = args.temp_c
    if args.lambda_sm_nm is not None:
        raw_inputs["lambda_sm"] = args.lambda_sm_nm * 1.0e-9
    if args.pf_dx_nm is not None:
        raw_inputs["pf_dx"] = args.pf_dx_nm * 1.0e-9
    if args.phys_dx_ref_nm is not None:
        raw_inputs["phys_dx_ref"] = args.phys_dx_ref_nm * 1.0e-9

    wf_name = args.workflow_name or workflow_name(args.temp_c, args.xb_out)
    workflow_dir = workflow_root / wf_name
    subdirs = ensure_workflow_subdirs(workflow_dir)
    workflow_dir_rel = workflow_dir.relative_to(REPO_ROOT)
    raw_results_root_rel = subdirs["raw"].relative_to(REPO_ROOT)

    physical_json = subdirs["input"] / "physical_inputs.json"
    _write_json(physical_json, raw_inputs)

    T_K = float(raw_inputs["temperature_C"]) + 273.15
    gamma = float(raw_inputs["gamma"])
    Vm_compound = float(raw_inputs["Vm_compound"])
    pf_dx_nm = float(raw_inputs.get("pf_dx", raw_inputs.get("dx"))) * 1.0e9
    lambda_sm_nm = float(raw_inputs["lambda_sm"]) * 1.0e9
    e0 = get_principal_eigenstrain(raw_inputs)
    C_precip = np.array(raw_inputs["C_tensor_Ag2Te_GPa"], dtype=float)
    DG_chem, x_eq = get_chemical_driving_force_Jm3(T_K, args.xb_out, Vm_compound)

    strains = _parse_csv_floats(args.strains)
    modes = _parse_csv_strings(args.modes)
    cases = build_cases(strains if strains else list(DEFAULT_STRAINS), modes if modes else list(DEFAULT_MODES))
    prefix = build_cnt_prefix(args.temp_c, args.xb_out)

    prediction_rows: list[dict[str, object]] = []
    guide_rows: list[dict[str, object]] = []

    for case in cases:
        e_ext = np.array(case.e_ext, dtype=float)
        E_el, e_r_star, cond = calc_elastic_energy_schur(
            C6_GPa=C_precip,
            e0=e0,
            e_ext=e_ext,
            fixed_idx=case.fixed_idx,
            free_idx=case.free_idx,
        )
        DG_net = DG_chem - E_el
        rc_nm = math.inf if DG_net <= 0.0 else 2.0 * gamma / DG_net * 1.0e9
        base_case_tag = f"{prefix}_{case.mode}_{case.strain_tag}"
        prediction_rows.append(
            {
                "workflow_name": wf_name,
                "base_case_tag": base_case_tag,
                "mode": case.mode,
                "strain": case.strain,
                "E0_xx": e_ext[0],
                "E0_yy": e_ext[1],
                "E0_zz": e_ext[2],
                "E0_yz": e_ext[3],
                "E0_xz": e_ext[4],
                "E0_xy": e_ext[5],
                "T_C": args.temp_c,
                "xB_out": args.xb_out,
                "xB_eq": x_eq,
                "dx_nm": pf_dx_nm,
                "rc_schur_nm": rc_nm,
                "window_nm": args.window_nm,
                "step_nm": args.step_nm,
                "DG_chem_Jm3": DG_chem,
                "E_el_Jm3": E_el,
                "DG_net_Jm3": DG_net,
                "Crr_condition_number": cond,
                "e_r_star_0": e_r_star[0] if len(e_r_star) > 0 else math.nan,
                "e_r_star_1": e_r_star[1] if len(e_r_star) > 1 else math.nan,
                "e_r_star_2": e_r_star[2] if len(e_r_star) > 2 else math.nan,
                "e_r_star_3": e_r_star[3] if len(e_r_star) > 3 else math.nan,
                "e_r_star_4": e_r_star[4] if len(e_r_star) > 4 else math.nan,
            }
        )

        ref_path_info = reference_case_output_rel(
            results_root_rel=raw_results_root_rel,
            base_case_tag=base_case_tag,
            elastic=args.elastic,
            temp_c=args.temp_c,
            nx=nx,
            ny=ny,
            nz=nz,
            dt=args.dt,
            nsteps=1,
            xb_out=args.xb_out,
        )
        ref_case_tag = Path(ref_path_info["case_dir_rel"]).name
        guide_rows.append(
            {
                "workflow_name": wf_name,
                "workflow_dir_rel": str(workflow_dir_rel),
                "raw_results_root_rel": str(raw_results_root_rel),
                "physical_input_json_rel": str(physical_json.relative_to(REPO_ROOT)),
                "base_case_tag": base_case_tag,
                "case_tag": ref_case_tag,
                "row_type": "reference",
                "reference_type": "matrix_only_same_strain",
                "reference_confidence": "high",
                "mode": case.mode,
                "strain": case.strain,
                "E0_xx": e_ext[0],
                "E0_yy": e_ext[1],
                "E0_zz": e_ext[2],
                "E0_yz": e_ext[3],
                "E0_xz": e_ext[4],
                "E0_xy": e_ext[5],
                "T_C": args.temp_c,
                "xB_out": args.xb_out,
                "xB_eq": x_eq,
                "nx": nx,
                "ny": ny,
                "nz": nz,
                "dt": args.dt,
                "nsteps": args.steps,
                "out_every": args.out_every,
                "csv_out_every": args.csv_out_every,
                "elastic": args.elastic,
                "window_nm": args.window_nm,
                "step_nm": args.step_nm,
                "radius_index": -1,
                "radius_nm": 0.0,
                "rc_schur_nm": rc_nm,
                "pf_dx_nm": pf_dx_nm,
                "lambda_sm_nm": lambda_sm_nm,
                "output_root_rel": ref_path_info["output_root_rel"],
                "case_dir_rel": ref_path_info["case_dir_rel"],
                "summary_rel": ref_path_info["summary_rel"],
                "energy_csv_rel": ref_path_info["energy_csv_rel"],
                "phi_final_rel": ref_path_info["phi_final_rel"],
                "xb_final_rel": ref_path_info["xb_final_rel"],
                "pf_input_rel": ref_path_info["pf_input_rel"],
                "reference_energy_rel": ref_path_info["reference_energy_rel"],
                "raw_init_dir_rel": ref_path_info["raw_init_dir_rel"],
            }
        )

        for radius_index, radius_nm in enumerate(_scan_radii(rc_nm, args.window_nm, args.step_nm)):
            path_info = case_output_rel(
                results_root_rel=raw_results_root_rel,
                base_case_tag=base_case_tag,
                elastic=args.elastic,
                temp_c=args.temp_c,
                nx=nx,
                ny=ny,
                nz=nz,
                dt=args.dt,
                nsteps=args.steps,
                radius_nm=radius_nm,
                xb_out=args.xb_out,
            )
            case_tag = Path(path_info["case_dir_rel"]).name
            guide_rows.append(
                {
                    "workflow_name": wf_name,
                    "workflow_dir_rel": str(workflow_dir_rel),
                    "raw_results_root_rel": str(raw_results_root_rel),
                    "physical_input_json_rel": str(physical_json.relative_to(REPO_ROOT)),
                    "base_case_tag": base_case_tag,
                    "case_tag": case_tag,
                    "row_type": "scan_point",
                    "reference_type": "",
                    "reference_confidence": "",
                    "mode": case.mode,
                    "strain": case.strain,
                    "E0_xx": e_ext[0],
                    "E0_yy": e_ext[1],
                    "E0_zz": e_ext[2],
                    "E0_yz": e_ext[3],
                    "E0_xz": e_ext[4],
                    "E0_xy": e_ext[5],
                    "T_C": args.temp_c,
                    "xB_out": args.xb_out,
                    "xB_eq": x_eq,
                    "nx": nx,
                    "ny": ny,
                    "nz": nz,
                    "dt": args.dt,
                    "nsteps": args.steps,
                    "out_every": args.out_every,
                    "csv_out_every": args.csv_out_every,
                    "elastic": args.elastic,
                    "window_nm": args.window_nm,
                    "step_nm": args.step_nm,
                    "radius_index": radius_index,
                    "radius_nm": radius_nm,
                    "rc_schur_nm": rc_nm,
                    "pf_dx_nm": pf_dx_nm,
                    "lambda_sm_nm": lambda_sm_nm,
                    "output_root_rel": path_info["output_root_rel"],
                    "case_dir_rel": path_info["case_dir_rel"],
                    "summary_rel": path_info["summary_rel"],
                    "energy_csv_rel": path_info["energy_csv_rel"],
                    "phi_final_rel": path_info["phi_final_rel"],
                    "xb_final_rel": path_info["xb_final_rel"],
                    "pf_input_rel": path_info["pf_input_rel"],
                }
            )

    prediction_csv = subdirs["input"] / "schur_rc_predictions.csv"
    with prediction_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(prediction_rows[0].keys()))
        writer.writeheader()
        writer.writerows(prediction_rows)

    guide_csv = subdirs["input"] / "guide_cnt_scan.csv"
    with guide_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(guide_rows[0].keys()))
        writer.writeheader()
        writer.writerows(guide_rows)

    metadata = {
        "workflow_name": wf_name,
        "workflow_dir": str(workflow_dir),
        "physical_input_json": str(physical_json),
        "prediction_csv": str(prediction_csv),
        "guide_cnt_scan_csv": str(guide_csv),
        "temperature_C": args.temp_c,
        "xB_out": args.xb_out,
        "grid": [nx, ny, nz],
        "dt": args.dt,
        "steps": args.steps,
        "out_every": args.out_every,
        "csv_out_every": args.csv_out_every,
        "elastic": args.elastic,
        "window_nm": args.window_nm,
        "step_nm": args.step_nm,
        "modes": modes,
        "strains": strains,
    }
    _write_json(subdirs["input"] / "workflow_meta.json", metadata)

    print(f"workflow_dir={workflow_dir}")
    print(f"physical_json={physical_json}")
    print(f"prediction_csv={prediction_csv}")
    print(f"guide_cnt_scan_csv={guide_csv}")
    print(f"prediction_rows={len(prediction_rows)}")
    print(f"guide_rows={len(guide_rows)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
