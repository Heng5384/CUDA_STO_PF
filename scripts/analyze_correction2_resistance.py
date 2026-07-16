#!/usr/bin/env python3
"""Diagnostic resistance decomposition for Correction Flow 2."""

from __future__ import annotations

import csv
from dataclasses import replace
import json
import math
from pathlib import Path
import sys

import numpy as np
from scipy.stats import spearmanr


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.correction1_ji_chen_mapping import (  # noqa: E402
    corrected_limit,
    inverse_interface_resistance,
)
from scripts.prepare_correction2_physical_lambda_matrix import (  # noqa: E402
    parse_physical_inputs,
)


REPORT_ROOT = ROOT / "reports/pf_ctot_production_candidate"


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError("refusing empty resistance metrics")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def finite_phase_resistance(
    lambda_nm: float, ratio: float, physical_inputs_path: Path,
) -> tuple[float, float, float]:
    inputs = parse_physical_inputs(physical_inputs_path)
    current = replace(
        inputs, temperature_C=400.0, lambda_sm=lambda_nm * 1.0e-9,
        L_phi_calibration_mode="one_sided_diffusion_controlled",
    )
    limit = corrected_limit(current, 400.0)
    lphi = ratio * limit["L_phi_diff_physical_m3_J_s"]
    phase = inverse_interface_resistance(lphi, limit)
    jc_diff = (
        limit["zeta0"] * limit["zeta_J_mol"] * limit["lambda_JC_m"]
        / (2.0 * limit["D_alpha_m2_s"])
    )
    return phase, jc_diff, lphi


def enrich(source: dict[str, str], family: str,
           physical_inputs_path: Path) -> dict[str, object]:
    lambda_nm = float(source.get("lambda_nm", 0.6))
    ratio = float(source["L_phi_ratio"])
    phase_r, jc_diff_r, lphi = finite_phase_resistance(
        lambda_nm, ratio, physical_inputs_path
    )
    mu_ref = float(source.get("mu_reference_J_mol", 137790.24))
    far_drive_hat = float(source["far_field_reaction_drive_hat"])
    jump_hat = float(source["interface_muB_jump_hat"])
    drive = abs(far_drive_hat * mu_ref)
    jump = abs(jump_hat * mu_ref)
    v_pf = abs(float(source["PF_velocity_h_nm_s"])) * 1.0e-9
    v_sharp = abs(float(source["sharp_velocity_nm_s"])) * 1.0e-9
    r_pf = drive / max(v_pf, 1.0e-300)
    r_matrix = drive / max(v_sharp, 1.0e-300)
    residual = r_pf - r_matrix - phase_r
    storage = abs(float(source["stefan_storage_residual_rel"]))
    return {
        "case": source["case"],
        "family": family,
        "drive_amplitude": source.get("drive_amplitude", "nan"),
        "incremental_Fo": source["target_incremental_Fo"],
        "lambda_nm": lambda_nm,
        "L_phi_ratio": ratio,
        "PF_velocity_nm_s": source["PF_velocity_h_nm_s"],
        "sharp_velocity_nm_s": source["sharp_velocity_nm_s"],
        "PF_sharp_error_rel": source["PF_sharp_velocity_error_rel"],
        "effective_resistance_ratio_sharp_over_PF": v_sharp / max(v_pf, 1.0e-300),
        "extra_effective_resistance_ratio": v_sharp / max(v_pf, 1.0e-300) - 1.0,
        "driving_proxy_J_mol": drive,
        "PF_total_resistance_proxy_J_s_mol_m": r_pf,
        "sharp_matrix_resistance_proxy_J_s_mol_m": r_matrix,
        "finite_Lphi_inverse_MI_J_s_mol_m": phase_r,
        "JiChen_diffusive_term_J_s_mol_m": jc_diff_r,
        "finite_phase_to_JiChen_diffusive_ratio": phase_r / jc_diff_r,
        "residual_diffuse_resistance_proxy_J_s_mol_m": residual,
        "residual_diffuse_fraction_of_PF_total": residual / r_pf,
        "chemical_potential_jump_J_mol": jump,
        "chemical_jump_fraction_of_far_drive": jump / max(drive, 1.0e-300),
        "chemical_jump_resistance_proxy_J_s_mol_m": jump / max(v_pf, 1.0e-300),
        "storage_conversion_residual_rel": storage,
        "storage_conversion_closed_to_1e10": storage <= 1.0e-10,
        "L_phi_physical_m3_J_s": lphi,
        "numerical_hard_gates_pass": source["numerical_hard_gates_pass"],
        "interpretation": "diagnostic_not_constitutive_fit",
    }


def correlation(rows: list[dict[str, object]], xkey: str,
                ykey: str = "PF_sharp_error_rel") -> float:
    x = [float(row[xkey]) for row in rows]
    y = [float(row[ykey]) for row in rows]
    if len(x) < 3 or len(set(x)) < 2 or len(set(y)) < 2:
        return math.nan
    return float(spearmanr(x, y).statistic)


def main() -> int:
    physical_inputs = ROOT / "physical_inputs.example.json"
    collections = [
        ("small_driving", REPORT_ROOT / "correction2_small_driving_metrics.csv"),
        ("matched_long_time", REPORT_ROOT / "correction2_matched_long_time_metrics.csv"),
        ("physical_lambda", REPORT_ROOT / "correction2_physical_lambda_metrics.csv"),
    ]
    rows: list[dict[str, object]] = []
    for family, path in collections:
        for source in read_csv(path):
            if source["numerical_hard_gates_pass"] == "True":
                rows.append(enrich(source, family, physical_inputs))
    write_csv(REPORT_ROOT / "correction2_resistance_metrics.csv", rows)

    early = [row for row in rows if (
        row["family"] == "matched_long_time"
        and abs(float(row["drive_amplitude"])-0.25) < 1.0e-12
        and abs(float(row["L_phi_ratio"])-0.9) < 1.0e-12
        and float(row["incremental_Fo"]) <= 10.0
        and "ell4" in str(row["case"])
        and float(row.get("incremental_Fo", 0)) > 0
    )]
    driving = [row for row in rows if (
        row["family"] == "small_driving"
        and abs(float(row["L_phi_ratio"])-0.9) < 1.0e-12
    )]
    lphi = [row for row in rows if (
        row["family"] == "small_driving"
        and abs(float(row["drive_amplitude"])-0.25) < 1.0e-12
    )]
    lambdas = [row for row in rows if row["family"] == "physical_lambda"]
    storage_max = max(float(row["storage_conversion_residual_rel"]) for row in rows)
    lambda_rho = correlation(lambdas, "lambda_nm")
    early_rho = correlation(early, "incremental_Fo")
    drive_rho = correlation(driving, "drive_amplitude")
    lphi_rho = correlation(lphi, "L_phi_ratio")
    all_storage_closed = all(bool(row["storage_conversion_closed_to_1e10"]) for row in rows)
    lambda_errors = {
        float(row["lambda_nm"]): float(row["PF_sharp_error_rel"])
        for row in lambdas
    }
    lambda_trend = (
        len(lambda_errors) == 3
        and lambda_errors[0.3] <= lambda_errors[0.45] <= lambda_errors[0.6]
    )
    primary = (
        "FINITE_DIFFUSE_INTERFACE_PHASE_RESPONSE_RESISTANCE"
        if lambda_trend and all_storage_closed
        else "MIXED_OR_UNRESOLVED_RESISTANCE"
    )
    REPORT_ROOT.joinpath("correction2_resistance_decomposition.md").write_text(f"""# Correction 2 Resistance Decomposition

This is a diagnostic decomposition, not a constitutive fit.  For each row the
same far-field chemical-driving proxy is divided by the PF and sharp average
velocities.  The sharp value defines the finite-box matrix-diffusion proxy.
The finite-phase term is independently recovered from Ji--Chen S2:

```text
1/M_I = 2/(3 c Lphi lambda) - zeta0*zeta*lambda/(2 Dalpha).
```

The remaining resistance is `R_PF - R_sharp - 1/M_I`.  The chemical-potential
jump is reported separately.  Storage conversion is assessed from the exact
`Delta h + Delta q_alpha`/Stefan residual and is never fitted.

## Correlation diagnostics

| diagnostic | Spearman rho(error) | interpretation |
|---|---:|---|
| incremental Fo, matched ell4 Fo<=10 | {early_rho:.6f} | early-transient dependence |
| driving amplitude, ratio 0.90 | {drive_rho:.6f} | weak nonlinearity check |
| physical lambda, fixed outer state | {lambda_rho:.6f} | finite-width dependence |
| Lphi ratio, A=0.25 | {lphi_rho:.6f} | phase-mobility plateau |

Maximum storage-conversion residual is `{storage_max:.6e}` and all rows are
closed to `1e-10`: `{all_storage_closed}`.  Thus no measurable mass/storage
lag is available to explain an O(10%) velocity error.  The observed physical-
lambda trend is monotone toward the sharp result: `{lambda_trend}`.

`primary_mismatch_classification={primary}`

`residual_diffuse_interface_resistance_status={'IDENTIFIED' if lambda_trend else 'UNRESOLVED'}`
""")
    print(f"resistance_rows={len(rows)}")
    print(f"primary_mismatch_classification={primary}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
