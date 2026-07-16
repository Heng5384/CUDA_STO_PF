#!/usr/bin/env python3
"""Reclassify Correction 1 and audit its asymptotic/initial-state evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
import sys

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import Unit_Psedobinary as unit  # noqa: E402
from scripts.correction1_planar_sharp_oracle import PlanarSharpOracle  # noqa: E402


REPORT_ROOT = ROOT / "reports/pf_ctot_production_candidate"


def h_switch(phi: np.ndarray) -> np.ndarray:
    return phi**3 * (6.0 * phi**2 - 15.0 * phi + 10.0)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_params(path: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.split("#", 1)[0].strip()
        if "=" in line:
            key, value = line.split("=", 1)
            result[key.strip()] = value.strip()
    return result


def case_directories(matrix_root: Path) -> list[Path]:
    return sorted(
        path for path in matrix_root.iterdir()
        if path.is_dir()
        and (path / "input/benchmark_manifest.json").is_file()
        and (path / "status.json").is_file()
    )


def window_bounds(window: str) -> tuple[float, float]:
    return {
        "early": (0.0, 0.25),
        "middle": (0.25, 0.75),
        "late": (0.75, 1.0),
        "full": (0.0, 1.0),
    }[window]


def thermodynamic_drive(
    temperature_k: float, x_eq: float, x_inf: float, mu_ref: float
) -> dict[str, float]:
    mu_b_eq = unit.mu_Ag2Te(temperature_k, x_eq)
    mu_b_inf = unit.mu_Ag2Te(temperature_k, x_inf)
    g_second_eq = unit.g_alpha_second_unified(temperature_k, x_eq)
    mu_b_slope_eq = (1.0 - x_eq) * g_second_eq
    exact = mu_b_inf - mu_b_eq
    linear = mu_b_slope_eq * (x_inf - x_eq)
    return {
        "reaction_affinity_J_mol": exact,
        "reaction_affinity_over_mu_ref": exact / mu_ref,
        "linearized_affinity_J_mol": linear,
        "linearized_affinity_over_mu_ref": linear / mu_ref,
        "thermodynamic_factor_gpp_J_mol": g_second_eq,
        "supersaturation_ratio": (x_inf - x_eq) / (1.0 - x_eq),
    }


def initial_condition_metrics(case_dir: Path) -> dict[str, object]:
    manifest = json.loads((case_dir / "input/benchmark_manifest.json").read_text())
    nx, ny, nz = map(int, manifest["grid"])
    shape = (nx, ny, nz)
    dx_nm = float(manifest["dx_nm"])
    domain_nm = float(manifest["domain_nm"])
    phi = np.fromfile(case_dir / "input/phi_init.raw", dtype=np.float64).reshape(shape)
    x_b = np.fromfile(case_dir / "input/xB_init.raw", dtype=np.float64).reshape(shape)
    ctot = np.fromfile(case_dir / "input/Ctot_init.raw", dtype=np.float64).reshape(shape)
    p = phi.mean(axis=(1, 2))
    x_line = x_b.mean(axis=(1, 2))
    h_line = h_switch(p)
    coordinates = np.arange(nx, dtype=np.float64) * dx_nm
    left = 0.35 * domain_nm
    right = 0.65 * domain_nm
    distance = np.where(
        coordinates < left,
        left - coordinates,
        np.where(coordinates > right, coordinates - right, np.nan),
    )
    matrix = np.isfinite(distance)

    oracle = PlanarSharpOracle(
        temperature_c=400.0,
        domain_nm=domain_nm,
        matrix_x=float(manifest["matrix_xB"]),
        start_time_s=float(manifest["sharp_start_time_s"]),
        cells=4000,
        pf_dx_nm=dx_nm,
    )
    sharp_state = oracle.initial_state("matched_growth")
    sharp_x = oracle.concentration(sharp_state)
    sharp_distance = (
        np.arange(oracle.cells, dtype=np.float64) + 0.5
    ) * oracle.dy * (oracle.outer_nm - oracle.radius_initial_nm)
    sharp_on_pf = np.interp(
        distance[matrix], sharp_distance, sharp_x,
        left=oracle.x_eq, right=float(sharp_x[-1]),
    )
    profile_delta = x_line[matrix] - sharp_on_pf
    pf_half_inventory = float(ctot.sum() * dx_nm / (2.0 * ny * nz))
    sharp_half_inventory = oracle.inventory(sharp_state)
    raw_similarity = PlanarSharpOracle(
        temperature_c=400.0,
        domain_nm=domain_nm,
        matrix_x=float(manifest["matrix_xB"]),
        start_time_s=float(manifest["sharp_start_time_s"]),
        cells=4000,
        pf_dx_nm=dx_nm,
    )
    sharp_uncorrected = raw_similarity.initial_state("matched_growth")
    # The correction can be recovered exactly against the common analytic
    # similarity profile because initial_state applies it linearly in distance.
    from scripts.correction1_planar_sharp_oracle import similarity_profile
    uncorrected_x = similarity_profile(
        sharp_distance, 400.0, float(manifest["matrix_xB"]),
        float(manifest["sharp_start_time_s"]),
    )
    correction = sharp_x - uncorrected_x
    matrix_weight = 1.0 - h_line[matrix]
    x_inf = float(manifest["matrix_xB"])
    x_eq = float(manifest["xB_eq"])
    available = float(np.sum(matrix_weight * (x_inf - x_eq)))
    depleted = float(np.sum(matrix_weight * (x_inf - x_line[matrix])))
    return {
        "case": case_dir.name,
        "pf_h_volume_fraction": float(h_switch(phi).mean()),
        "pf_half_inventory_nm": pf_half_inventory,
        "sharp_half_inventory_nm": sharp_half_inventory,
        "inventory_difference_abs_nm": abs(pf_half_inventory - sharp_half_inventory),
        "inventory_difference_rel": abs(pf_half_inventory - sharp_half_inventory)
        / max(abs(pf_half_inventory), 1.0),
        "matrix_profile_L2": float(np.sqrt(np.mean(profile_delta**2))),
        "matrix_profile_Linf": float(np.max(np.abs(profile_delta))),
        "sharp_inventory_correction_min": float(np.min(correction)),
        "sharp_inventory_correction_max": float(np.max(correction)),
        "sharp_inventory_correction_L2": float(np.sqrt(np.mean(correction**2))),
        "pf_matrix_x_interface_nearest": float(x_line[matrix][np.argmin(distance[matrix])]),
        "sharp_matrix_x_interface_continuous": oracle.x_eq,
        "pf_matrix_x_outer": float(x_line[matrix][np.argmax(distance[matrix])]),
        "sharp_matrix_x_outer_cell": float(sharp_x[-1]),
        "finite_box_depletion_fraction": depleted / max(available, 1.0e-300),
        "same_equimolar_h_surface": True,
        "same_total_inventory": abs(pf_half_inventory - sharp_half_inventory) <= 1.0e-12,
        "same_matrix_profile_pointwise": bool(np.max(np.abs(profile_delta)) <= 1.0e-12),
        "same_continuous_interface_composition": True,
        "same_outer_no_flux_symmetry_ensemble": True,
        "same_analytic_preage_family_before_inventory_correction": True,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--matrix-root", type=Path,
        default=ROOT / "tmp/correction1_Lphi_below_limit",
    )
    parser.add_argument(
        "--metrics", type=Path,
        default=REPORT_ROOT / "correction1_below_limit_metrics.csv",
    )
    parser.add_argument("--report-root", type=Path, default=REPORT_ROOT)
    args = parser.parse_args()
    rows = read_csv(args.metrics)
    by_case = {row["case"]: row for row in rows}
    cases = case_directories(args.matrix_root)
    if {path.name for path in cases} != set(by_case):
        raise RuntimeError("frozen Correction 1 cases and metrics do not match")

    sharp_ref = json.loads(
        (args.report_root / "correction1_diffusion_sharp_reference.json").read_text()
    )
    sharp_velocity = float(sharp_ref["velocity_average_nm_s"])
    selected = by_case["T400_ratio_0p9_dx0p1_dt1"]
    selected_velocity = float(selected["full_window_velocity_nm_s"])
    velocity_ratio = selected_velocity / sharp_velocity
    resistance_ratio = sharp_velocity / selected_velocity
    status_report = f"""# Correction 2 Status Reclassification

Correction 1 is preserved verbatim, but its scientific classification is
narrowed to what the frozen evidence actually establishes.

| diagnostic | value |
|---|---:|
| selected PF velocity (nm/s) | {selected_velocity:.17e} |
| independent sharp velocity (nm/s) | {sharp_velocity:.17e} |
| `V_PF/V_sharp` | {velocity_ratio:.17e} |
| diagnostic effective-resistance ratio `V_sharp/V_PF` | {resistance_ratio:.17e} |
| diagnostic extra-resistance ratio `V_sharp/V_PF - 1` | {resistance_ratio - 1.0:.17e} |
| relative velocity mismatch | {abs(selected_velocity-sharp_velocity)/abs(sharp_velocity):.8f} |

`PHASE_MOBILITY_PLATEAU=PASS`: additional below-limit `L_phi` produces weak
changes in the frozen short run.

`QUANTITATIVE_DIFFUSION_LIMIT=NOT_ESTABLISHED`: the old fit window is an
early transient and the PF/sharp initial matrix profiles are not pointwise
identical.  The resistance ratios above are diagnostics only; they are not a
unique constitutive interface resistance.
"""
    (args.report_root / "correction2_status_reclassification.md").write_text(
        status_report
    )

    dimensionless_rows: list[dict[str, object]] = []
    for case_dir in cases:
        manifest = json.loads((case_dir / "input/benchmark_manifest.json").read_text())
        params = parse_params(case_dir / "input/benchmark.params")
        metric = by_case[case_dir.name]
        temperature_k = float(metric["temperature_C"]) + 273.15
        diffusivity_m2_s = unit.D_Ag_in_PbTe_m2_per_s(temperature_k)
        diffusivity_nm2_s = diffusivity_m2_s * 1.0e18
        lambda_nm = float(manifest["lambda_nm"])
        elapsed = float(metric["elapsed_s"])
        preage = float(manifest["sharp_start_time_s"])
        velocity = float(metric["full_window_velocity_nm_s"])
        dx_nm = float(metric["dx_nm"])
        matrix_width_nm = 0.35 * float(manifest["domain_nm"])
        drive = thermodynamic_drive(
            temperature_k, float(manifest["xB_eq"]),
            float(manifest["matrix_xB"]), float(params["mu_reference_scale"]),
        )
        initial = initial_condition_metrics(case_dir)
        for window in ("early", "middle", "late", "full"):
            a, b = window_bounds(window)
            t_start = a * elapsed
            t_end = b * elapsed
            duration = t_end - t_start
            is_full = window == "full"
            displacement = velocity * duration if is_full else math.nan
            dimensionless_rows.append({
                "case": case_dir.name,
                "window": window,
                "measurement_status": (
                    "MEASURED_FROM_INITIAL_FINAL_ACCEPTED_FIELDS" if is_full
                    else "UNAVAILABLE_NO_INTERMEDIATE_ACCEPTED_FIELD_CHECKPOINT"
                ),
                "t_start_s": t_start,
                "t_end_s": t_end,
                "duration_s": duration,
                "preage_s": preage,
                "D_alpha_m2_s": diffusivity_m2_s,
                "lambda_nm": lambda_nm,
                "dx_nm": dx_nm,
                "Fo_lambda_increment_start": diffusivity_nm2_s * t_start / lambda_nm**2,
                "Fo_lambda_increment_end": diffusivity_nm2_s * t_end / lambda_nm**2,
                "Fo_lambda_window_duration": diffusivity_nm2_s * duration / lambda_nm**2,
                "Fo_lambda_total_age_start": diffusivity_nm2_s * (preage+t_start) / lambda_nm**2,
                "Fo_lambda_total_age_end": diffusivity_nm2_s * (preage+t_end) / lambda_nm**2,
                "diffusion_length_increment_nm": math.sqrt(diffusivity_nm2_s * duration),
                "diffusion_length_increment_over_lambda": math.sqrt(diffusivity_nm2_s * duration) / lambda_nm,
                "diffusion_length_total_age_nm": math.sqrt(diffusivity_nm2_s * (preage+t_end)),
                "diffusion_length_total_age_over_lambda": math.sqrt(diffusivity_nm2_s * (preage+t_end)) / lambda_nm,
                "PF_velocity_nm_s": velocity if is_full else math.nan,
                "PF_Pe_lambda": abs(velocity) * lambda_nm / diffusivity_nm2_s if is_full else math.nan,
                "sharp_velocity_nm_s": sharp_velocity if is_full else math.nan,
                "sharp_Pe_lambda": abs(sharp_velocity) * lambda_nm / diffusivity_nm2_s if is_full else math.nan,
                "PF_displacement_nm": displacement,
                "PF_displacement_over_dx": displacement / dx_nm if is_full else math.nan,
                "PF_displacement_over_lambda": displacement / lambda_nm if is_full else math.nan,
                "finite_box_diffusion_length_fraction": math.sqrt(diffusivity_nm2_s * (preage+t_end)) / matrix_width_nm,
                "finite_box_depletion_fraction": initial["finite_box_depletion_fraction"],
                **drive,
                "Fo10_operational_candidate": bool(
                    diffusivity_nm2_s * (preage+t_end) / lambda_nm**2 >= 10.0
                ),
                "quantitative_fit_eligible": False,
                "ineligibility_reason": (
                    "no intermediate accepted-state field" if not is_full else
                    "Fo_total_age_below_10_and_initial_profiles_not_pointwise_matched"
                ),
            })
    write_csv(args.report_root / "correction2_dimensionless_windows.csv", dimensionless_rows)

    full_rows = [row for row in dimensionless_rows if row["window"] == "full"]
    table = "\n".join(
        f"| {row['case']} | {float(row['Fo_lambda_window_duration']):.6e} | "
        f"{float(row['Fo_lambda_total_age_end']):.6f} | "
        f"{float(row['PF_Pe_lambda']):.6e} | "
        f"{float(row['PF_displacement_over_dx']):.6e} | "
        f"{float(row['finite_box_depletion_fraction']):.6f} | no |"
        for row in full_rows
    )
    (args.report_root / "correction2_asymptotic_window_audit.md").write_text(f"""# Correction 2 Asymptotic Window Audit

The frozen runs contain accepted fields only at the initial and final states;
therefore early/middle/late window velocities cannot be reconstructed.  Those
rows are retained explicitly as unavailable in
`correction2_dimensionless_windows.csv`, rather than inferred from nonlinear
iteration work buffers.

| case | incremental Fo | total-age Fo | PF Pe | displacement/dx | finite-box depletion | eligible |
|---|---:|---:|---:|---:|---:|---|
{table}

The current observation duration gives `Fo_lambda` of approximately
`{float(full_rows[0]['Fo_lambda_window_duration']):.6e}`.  Even including the
0.1 s analytic pre-age, the total-age value is only
`{float(full_rows[0]['Fo_lambda_total_age_end']):.6f}`, below the operational
`Fo_lambda >= 10` quasi-steady gate.  `Pe_lambda` is small, but the interface
moves only O(1e-5) grid cells.  Small Pe does not repair the missing asymptotic
time window.

`current_window_classification=UNRESOLVED_EARLY_DIFFUSION_TRANSIENT`
""")

    init_rows = [initial_condition_metrics(path) for path in cases]
    representative = next(row for row in init_rows if row["case"] == "T400_ratio_0p9_dx0p1_dt1")
    (args.report_root / "correction2_initial_condition_metrics.csv").parent.mkdir(
        parents=True, exist_ok=True
    )
    write_csv(args.report_root / "correction2_initial_condition_metrics.csv", init_rows)
    (args.report_root / "correction2_initial_condition_audit.md").write_text(f"""# Correction 2 Initial-Condition Audit

The PF and sharp initializers share the same nominal planar surface, total
inventory, continuous interface equilibrium composition, no-flux/symmetry
finite-box ensemble, and analytic 0.1 s similarity-profile family.  They do
**not** share the same pointwise matrix profile.

The sharp oracle adds a distance-weighted far-field correction solely to make
its sharp beta-plus-matrix inventory equal the diffuse PF cell sum.  The PF
initializer does not contain that correction.  For the selected case:

| quantity | value |
|---|---:|
| PF half-box inventory (xB nm) | {float(representative['pf_half_inventory_nm']):.17e} |
| sharp half-box inventory (xB nm) | {float(representative['sharp_half_inventory_nm']):.17e} |
| relative inventory difference | {float(representative['inventory_difference_rel']):.3e} |
| matrix profile L2 difference | {float(representative['matrix_profile_L2']):.17e} |
| matrix profile Linf difference | {float(representative['matrix_profile_Linf']):.17e} |
| sharp correction maximum | {float(representative['sharp_inventory_correction_max']):.17e} |
| PF nearest matrix-side xB | {float(representative['pf_matrix_x_interface_nearest']):.17e} |
| sharp continuous interface xB | {float(representative['sharp_matrix_x_interface_continuous']):.17e} |

Thus exact global inventory matching was achieved by changing the sharp outer
profile, not by constructing one common state and mapping it to both
representations.  The old early-time velocity comparison is consequently not
a matched-initial-condition validation.

`initial_condition_matching_status=GLOBAL_INVENTORY_ONLY_NOT_POINTWISE_MATCHED`
""")
    print("correction2_baseline_reclassification=complete")
    print(f"V_PF_over_V_sharp={velocity_ratio:.17e}")
    print(f"Fo_lambda_current_window={float(full_rows[0]['Fo_lambda_window_duration']):.17e}")
    print(f"Fo_lambda_total_age={float(full_rows[0]['Fo_lambda_total_age_end']):.17e}")
    print("quantitative_diffusion_limit_status=NOT_ESTABLISHED")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
