#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import fsolve


R_GAS = 8.314462618


@dataclass(frozen=True)
class LoadCase:
    label: str
    mode: str
    strain: float
    strain_tag: str
    fixed_idx: tuple[int, ...]
    free_idx: tuple[int, ...]
    e_ext: tuple[float, float, float, float, float, float]


DEFAULT_STRAINS: tuple[float, ...] = (0.0, 0.0025, 0.0050, 0.0075, 0.0100)
DEFAULT_MODES: tuple[str, ...] = ("exx", "eyy", "exx_eyy")


def format_strain_tag(strain: float) -> str:
    if abs(strain) < 1e-15:
        return "s000"
    magnitude = f"{abs(strain):.4f}".rstrip("0").rstrip(".").replace(".", "p")
    prefix = "s" if strain > 0 else "sm"
    return f"{prefix}{magnitude}"


def short_case_tag(strain: float) -> str:
    if abs(strain) < 1e-15:
        return "s0"
    return format_strain_tag(strain)


def format_label(mode: str, strain: float) -> str:
    if abs(strain) < 1e-15:
        return "s=0.0"
    s_txt = f"{strain:+.4f}".rstrip("0").rstrip(".")
    if mode == "exx_eyy":
        return f"biaxial s={s_txt}"
    if mode == "eyy":
        return f"eyy s={s_txt}"
    return f"s={s_txt}"


def build_load_case(mode: str, strain: float) -> LoadCase:
    strain_tag = format_strain_tag(strain)
    if mode == "exx":
        return LoadCase(
            label=format_label(mode, strain),
            mode=mode,
            strain=strain,
            strain_tag=strain_tag,
            fixed_idx=(0,),
            free_idx=(1, 2, 3, 4, 5),
            e_ext=(strain, 0.0, 0.0, 0.0, 0.0, 0.0),
        )
    if mode == "eyy":
        return LoadCase(
            label=format_label(mode, strain),
            mode=mode,
            strain=strain,
            strain_tag=strain_tag,
            fixed_idx=(1,),
            free_idx=(0, 2, 3, 4, 5),
            e_ext=(0.0, strain, 0.0, 0.0, 0.0, 0.0),
        )
    if mode == "exx_eyy":
        return LoadCase(
            label=format_label(mode, strain),
            mode=mode,
            strain=strain,
            strain_tag=strain_tag,
            fixed_idx=(0, 1),
            free_idx=(2, 3, 4, 5),
            e_ext=(strain, strain, 0.0, 0.0, 0.0, 0.0),
        )
    raise ValueError(f"unsupported mode: {mode}")


def build_cases(strains: list[float], modes: list[str]) -> list[LoadCase]:
    cases: list[LoadCase] = []
    for mode in modes:
        for base_strain in strains:
            if abs(base_strain) < 1e-15:
                if mode == "exx":
                    cases.append(build_load_case(mode, 0.0))
                continue
            cases.append(build_load_case(mode, abs(base_strain)))
            cases.append(build_load_case(mode, -abs(base_strain)))
    return cases


def load_inputs(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def get_principal_eigenstrain(inputs: dict) -> np.ndarray:
    if "eigenstrain_tensor" in inputs:
        tensor = np.array(inputs["eigenstrain_tensor"], dtype=float)
        return np.array(
            [tensor[0][0], tensor[1][1], tensor[2][2], tensor[1][2], tensor[0][2], tensor[0][1]],
            dtype=float,
        )
    if "eigenstrain_principal" in inputs:
        vals = inputs["eigenstrain_principal"]
        return np.array([vals[0], vals[1], vals[2], 0.0, 0.0, 0.0], dtype=float)
    raise ValueError("input json 缺少 eigenstrain_principal 或 eigenstrain_tensor")


def L_pseudobinary(T: float) -> float:
    return 41212.9 - 18.05 * T


def mu_Ag2Te_matrix(T: float, x: float) -> float:
    return R_GAS * T * np.log(x) + L_pseudobinary(T) * (1.0 - x) ** 2


def get_chemical_driving_force_Jm3(T_K: float, x0: float, Vm_compound: float) -> tuple[float, float]:
    def eq_func(x_in):
        x = x_in.item() if hasattr(x_in, "item") else x_in
        if x <= 1e-10:
            return 1e5
        return R_GAS * T_K * math.log(x) + L_pseudobinary(T_K) * (1.0 - x) ** 2

    guess = math.exp(-L_pseudobinary(T_K) / (R_GAS * T_K))
    x_eq = fsolve(eq_func, guess).item()
    d_mu = mu_Ag2Te_matrix(T_K, x0) - mu_Ag2Te_matrix(T_K, x_eq)
    return d_mu / Vm_compound, x_eq


def calc_elastic_energy_schur(
    C6_GPa: np.ndarray,
    e0: np.ndarray,
    e_ext: np.ndarray,
    fixed_idx: Iterable[int],
    free_idx: Iterable[int],
) -> tuple[float, np.ndarray, float]:
    fixed_idx = tuple(fixed_idx)
    free_idx = tuple(free_idx)
    C = np.array(C6_GPa, dtype=float) * 1.0e9

    e_f = e_ext[list(fixed_idx)]
    e0_f = e0[list(fixed_idx)]
    e0_r = e0[list(free_idx)]

    C_ff = C[np.ix_(fixed_idx, fixed_idx)]
    C_fr = C[np.ix_(fixed_idx, free_idx)]
    C_rf = C[np.ix_(free_idx, fixed_idx)]
    C_rr = C[np.ix_(free_idx, free_idx)]

    cond = float(np.linalg.cond(C_rr))
    if not np.isfinite(cond):
        raise np.linalg.LinAlgError("C_rr 条件数非有限，无法做 Schur complement")
    if cond > 1.0e12:
        raise np.linalg.LinAlgError(f"C_rr 条件数过大: {cond:.3e}")

    C_rr_inv = np.linalg.inv(C_rr)
    schur = C_ff - C_fr @ C_rr_inv @ C_rf
    delta_f = e_f - e0_f
    e_r_star = e0_r - C_rr_inv @ C_rf @ delta_f
    E_min = float(0.5 * delta_f.T @ schur @ delta_f)
    return E_min, e_r_star, cond


def case_tag(prefix: str, strain_tag: str) -> str:
    return f"{prefix}_{strain_tag}"


def main() -> None:
    parser = argparse.ArgumentParser(
        description="用 strictref 语义计算 exx / eyy / exx+eyy 载荷下的 Schur 预测临界半径，并输出 case csv。"
    )
    parser.add_argument(
        "--input-json",
        default="physical_inputs.example.json",
        help="physical input json 路径",
    )
    parser.add_argument(
        "--output-dir",
        default="Results_scan/constraint_cnt_strictref_test_T400_x0_0p03",
        help="输出目录",
    )
    parser.add_argument(
        "--xB-out",
        type=float,
        default=0.03,
        help="matrix far-field xB (default: 0.03)",
    )
    parser.add_argument(
        "--window-nm",
        type=float,
        default=0.5,
        help="constraint sweep 半径窗口半宽 (nm)",
    )
    parser.add_argument(
        "--step-nm",
        type=float,
        default=0.1,
        help="constraint sweep 半径步长 (nm)",
    )
    parser.add_argument(
        "--prefix",
        default="cntcon_T400_xB0p030_strictref",
        help="base case tag prefix",
    )
    parser.add_argument(
        "--strains",
        default="0,0.0025,0.005,0.0075,0.01",
        help="strain magnitudes list, comma separated; zero only generates exx s=0",
    )
    parser.add_argument(
        "--modes",
        default="exx,eyy,exx_eyy",
        help="modes to generate, comma separated",
    )
    args = parser.parse_args()

    input_path = Path(args.input_json)
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    inputs = load_inputs(input_path)
    gamma = float(inputs["gamma"])
    T_C = float(inputs["temperature_C"])
    T_K = T_C + 273.15
    Vm_compound = float(inputs["Vm_compound"])
    pf_dx_nm = float(inputs.get("pf_dx", inputs.get("dx"))) * 1.0e9
    e0 = get_principal_eigenstrain(inputs)
    C_precip = np.array(inputs["C_tensor_Ag2Te_GPa"], dtype=float)

    DG_chem, x_eq = get_chemical_driving_force_Jm3(T_K, args.xB_out, Vm_compound)

    strains = [float(x.strip()) for x in args.strains.split(",") if x.strip()]
    modes = [x.strip() for x in args.modes.split(",") if x.strip()]
    cases = build_cases(strains, modes)

    records = []
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
        if DG_net <= 0.0:
            rc_nm = math.inf
        else:
            rc_nm = 2.0 * gamma / DG_net * 1.0e9

        base_case_tag = case_tag(f"{args.prefix}_{case.mode}", case.strain_tag)
        row = {
            "label": case.label,
            "base_case_tag": base_case_tag,
            "mode": case.mode,
            "strain": case.strain,
            "T_C": T_C,
            "xB_out": args.xB_out,
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
            "E0_xx": e_ext[0],
            "E0_yy": e_ext[1],
            "E0_zz": e_ext[2],
            "E0_yz": e_ext[3],
            "E0_xz": e_ext[4],
            "E0_xy": e_ext[5],
        }
        records.append(row)

        mode_file_tag = case.mode.replace("+", "_").replace("-", "m")
        short_tag = short_case_tag(case.strain)
        case_csv_path = out_dir / f"case_{mode_file_tag}_{short_tag}_L5.csv"
        with case_csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=[
                    "base_case_tag",
                    "mode",
                    "strain",
                    "T_C",
                    "xB_out",
                    "dx_nm",
                    "rc_schur_nm",
                    "window_nm",
                    "step_nm",
                    "E0_xx",
                    "E0_yy",
                    "E0_zz",
                    "E0_yz",
                    "E0_xz",
                    "E0_xy",
                ],
            )
            writer.writeheader()
            writer.writerow(
                {
                    "base_case_tag": base_case_tag,
                    "mode": case.mode,
                    "strain": case.strain,
                    "T_C": T_C,
                    "xB_out": args.xB_out,
                    "dx_nm": pf_dx_nm,
                    "rc_schur_nm": rc_nm,
                    "window_nm": args.window_nm,
                    "step_nm": args.step_nm,
                    "E0_xx": e_ext[0],
                    "E0_yy": e_ext[1],
                    "E0_zz": e_ext[2],
                    "E0_yz": e_ext[3],
                    "E0_xz": e_ext[4],
                    "E0_xy": e_ext[5],
                }
            )

    summary_csv = out_dir / "schur_rc_predictions.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(records[0].keys()))
        writer.writeheader()
        writer.writerows(records)

    print(f"input_json={input_path}")
    print(f"output_dir={out_dir}")
    print(f"summary_csv={summary_csv}")
    for rec in records:
        print(
            "{label}: rc_schur_nm={rc:.6f}, DG_chem={dg:.6e}, E_el={eel:.6e}, DG_net={net:.6e}".format(
                label=rec["label"],
                rc=rec["rc_schur_nm"],
                dg=rec["DG_chem_Jm3"],
                eel=rec["E_el_Jm3"],
                net=rec["DG_net_Jm3"],
            )
        )


if __name__ == "__main__":
    main()
