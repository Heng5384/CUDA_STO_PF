#!/usr/bin/env python3
import argparse
import csv
import json
import math
from pathlib import Path


def xag_to_xb(x_ag: float) -> float:
    return 2.0 * x_ag / (2.0 - x_ag)


def xb_to_xag(x_b: float) -> float:
    return 2.0 * x_b / (2.0 + x_b)


def l_alpha(T_K: float, p: dict) -> float:
    return p["L_alpha0_J_mol"] + p["L_alpha1_J_mol_K"] * T_K


def f_regular(x_b: float, T_K: float, p: dict) -> float:
    x = min(max(x_b, 1.0e-300), 1.0 - 1.0e-12)
    return 8.314462618 * T_K * math.log(x) + l_alpha(T_K, p) * (1.0 - x) ** 2


def solve_xb_eq(T_K: float, p: dict) -> float:
    lo = 1.0e-12
    hi = 0.49
    flo = f_regular(lo, T_K, p)
    fhi = f_regular(hi, T_K, p)
    if flo > 0.0:
        return lo
    if fhi < 0.0:
        hi = 0.999999
        fhi = f_regular(hi, T_K, p)
        if fhi < 0.0:
            return xag_to_xb(p["xAg_default"])
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = f_regular(mid, T_K, p)
        if fm > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def delta_gv(x_ag: float, T_K: float, p: dict) -> tuple[float, float]:
    x_b = xag_to_xb(x_ag)
    x_b_eq = solve_xb_eq(T_K, p)
    x_ag_eq = xb_to_xag(x_b_eq)
    if not (x_ag > x_ag_eq + p["xeq_guard_xAg_atomic_fraction"]):
        return 0.0, x_ag_eq
    dmu = max(f_regular(x_b, T_K, p) - f_regular(x_b_eq, T_K, p), 0.0)
    V_m = 6.02214076e23 * p["a_PbTe_m"] ** 3 / 4.0
    return dmu / V_m, x_ag_eq


def d_ag(T_K: float, p: dict) -> float:
    return p["D0_m2_s"] * math.exp(-p["Q_J_mol"] / (8.314462618 * T_K))


def j_gp(x_ag: float, T_K: float, p: dict) -> tuple[float, float, float]:
    dg, x_ag_eq = delta_gv(x_ag, T_K, p)
    if dg <= 0.0:
        return 0.0, x_ag_eq, d_ag(T_K, p)
    expo = -p["B_eff_J3_m_minus6"] / (1.380649e-23 * T_K * dg * dg)
    if expo < -700.0:
        return 0.0, x_ag_eq, d_ag(T_K, p)
    return p["A_m_minus5"] * d_ag(T_K, p) * math.exp(expo), x_ag_eq, d_ag(T_K, p)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", required=True)
    ap.add_argument("--csv-out", required=True)
    args = ap.parse_args()

    payload = json.loads(Path(args.json).read_text())
    p = payload["parameters"]
    geom = payload["validation_runtime_geometry"]
    rows = []
    for vp in payload["validation_points"]:
        T_C = vp["T_C"]
        T_K = vp["T_K"]
        x_ag = vp["xAg"]
        J, x_ag_eq, D = j_gp(x_ag, T_K, p)
        expected = J * geom["box_volume_m3"] * geom["window_s"]
        rows.append({
            "T_C": T_C,
            "T_K": T_K,
            "xAg": x_ag,
            "xAg_eq_calc": x_ag_eq,
            "xAg_eq_ref": vp["xAg_eq"],
            "D_Ag_calc": D,
            "D_Ag_ref": vp["D_Ag_m2_s"],
            "J_GP_calc": J,
            "J_GP_ref": vp["J_GP_m3_s"],
            "expected_births_calc": expected,
            "expected_births_ref": vp["expected_births_128cubed_74s"],
            "J_rel_err": 0.0 if vp["J_GP_m3_s"] == 0.0 else abs(J - vp["J_GP_m3_s"]) / vp["J_GP_m3_s"],
            "birth_rel_err": 0.0 if vp["expected_births_128cubed_74s"] == 0.0 else abs(expected - vp["expected_births_128cubed_74s"]) / vp["expected_births_128cubed_74s"],
        })

    out_path = Path(args.csv_out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
