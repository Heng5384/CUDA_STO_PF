#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reports" / "gp_assisted_beta_mass_ledger"
KB = 1.380649e-23
R = 8.31446261815324


def GHSER_Pb(T: float) -> float:
    if T < 600.61:
        return -10544.679 + 183.372894 * T - 35.6687 * T * math.log(T) + 0.01583435 * T * T - 5.240417e-6 * T**3 + 155015.0 / T
    return -13067.662 + 211.206579 * T - 38.5844296 * T * math.log(T) + 0.018531982 * T * T - 5.764227e-6 * T**3 + 155015.0 / T


def GHSER_Ag(T: float) -> float:
    if T < 1234.93:
        return -7209.512 + 118.202013 * T - 23.8463314 * T * math.log(T) - 0.001790585 * T * T - 3.98587e-7 * T**3 - 12011.0 / T
    return -15095.252 + 190.266404 * T - 33.472 * T * math.log(T) + 1.411773e29 * T**-9


def GHSER_Te(T: float) -> float:
    if T < 722.66:
        return -10544.679 + 183.372894 * T - 35.6687 * T * math.log(T) + 0.01583435 * T * T - 5.240417e-6 * T**3 + 155015.0 / T
    return 9160.595 - 129.265373 * T + 13.004 * T * math.log(T) - 0.0362361 * T * T + 5.006367e-6 * T**3 - 1.28681e30 * T**-9


def G_Ag2Te_Solid(T: float) -> float:
    base_per_atom = -10128.93 - 12.645115 * T
    return 3.0 * (base_per_atom + (2.0 / 3.0) * GHSER_Ag(T) + (1.0 / 3.0) * GHSER_Te(T))


def L_param(T: float) -> float:
    return 41212.9 - 18.05 * T


def mu_Ag2Te_alpha(T: float, xB: float) -> float:
    x = min(max(xB, 1e-12), 1.0 - 1e-12)
    return G_Ag2Te_Solid(T) + R * T * math.log(x) + L_param(T) * (1.0 - x) ** 2


def barrier(T: float, xB: float, gamma: float, vm: float) -> tuple[float, float]:
    dg = mu_Ag2Te_alpha(T, xB) - G_Ag2Te_Solid(T)
    dgv = dg / vm
    dG = 16.0 * math.pi * gamma**3 / (3.0 * dgv**2)
    rstar = 2.0 * gamma / dgv
    return dG / (KB * T), rstar * 1e9


def write_template(out_dir: Path, S: float, xB: float) -> list[dict[str, object]]:
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for tc in (380.0, 400.0, 450.0):
        T = tc + 273.15
        homo, rstar_nm = barrier(T, xB, gamma=0.168, vm=4.1009e-5)
        rows.append({
            "T_C": tc,
            "T_K": T,
            "S_used": S,
            "fit_or_prediction": "fit" if tc == 380.0 else "prediction",
            "xB_alpha": xB,
            "DeltaG_homo_kBT": homo,
            "DeltaG_GP_kBT": S * homo,
            "critical_radius_nm": rstar_nm,
            "D_beta_or_D_Ag": "placeholder: fill independently measured/Arrhenius diffusivity",
            "prefactor": "placeholder: fit only at 380C",
            "predicted_event_rate": "placeholder",
            "predicted_beta_number_density": "placeholder",
            "experimental_beta_number_density": "placeholder: unavailable in repo scan",
            "notes": "S locked after 380C fit; placeholders are explicitly marked",
        })
    path = out_dir / "locked_S_prediction_template.csv"
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate GP-assisted locked-S cross-temperature prediction template.")
    parser.add_argument("--out-dir", type=Path, default=OUT_DIR)
    parser.add_argument("--S", type=float, default=0.05)
    parser.add_argument("--xB-alpha", type=float, default=0.03)
    args = parser.parse_args()
    rows = write_template(args.out_dir.resolve(), args.S, args.xB_alpha)
    ok = rows[0]["fit_or_prediction"] == "fit" and all(r["fit_or_prediction"] == "prediction" for r in rows[1:])
    print(f"locked_S_prediction_template_generated={ok}")
    print(f"output_csv={args.out_dir.resolve() / 'locked_S_prediction_template.csv'}")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
