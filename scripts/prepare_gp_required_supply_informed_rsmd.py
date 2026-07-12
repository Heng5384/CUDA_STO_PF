#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PARAM_DIR = ROOT / "params/gp_required_supply_informed_rsmd"
REPORT_DIR = ROOT / "reports/gp_required_supply_informed_rsmd"
BASE_DIR = ROOT / "params/beta_capacity_gated_handoff"

XB_FAR = 0.0078305391025
PROVENANCE = "scenario_bracket_not_calibrated"
KERNEL_RADIUS_DX = 1.5
N_STEPS = 1040

SEEDS = {
    380: {
        "seed_id": "nlib_00006",
        "R_eff_h_nm": 5.358726,
        "xBcrit_beta": 0.011191599269189258,
        "base_params": "T380_S05_staged_handoff_diag_20.params",
        "R_exchange_nm": [4.0, 8.0, 12.0],
    },
    400: {
        "seed_id": "nlib_dc_T400_xB003",
        "R_eff_h_nm": 4.735881,
        "xBcrit_beta": 0.016708547037,
        "base_params": "T400_S05_staged_handoff_diag_20.params",
        "R_exchange_nm": [8.0, 12.0, 16.0],
    },
}

SCENARIOS = [
    ("null", 0.0, None, "after-quench far-field control"),
    ("marginal", 1.0, None, "ceiling equals beta-side xBcrit"),
    ("nominal", 1.2, None, "20 percent margin above beta-side xBcrit gap"),
    ("strong", 1.5, None, "50 percent margin above beta-side xBcrit gap"),
    ("upper_guard", None, 0.024, "guard ceiling below xB_tot"),
    ("counterfactual_only", None, 0.030, "counterfactual_not_physical"),
]
RUN_SCENARIOS = {"null", "marginal", "nominal", "strong", "upper_guard"}
CHI_REL = [0.3, 1.0, 3.0]


def tag_float(v: float) -> str:
    return f"{v:.12g}".replace(".", "p").replace("-", "m")


def ceiling_for(xbcrit: float, s_margin: float | None, explicit: float | None) -> float:
    if explicit is not None:
        return explicit
    assert s_margin is not None
    return XB_FAR + s_margin * (xbcrit - XB_FAR)


def parse_params(path: Path) -> tuple[list[str], dict[str, str], list[str]]:
    order: list[str] = []
    vals: dict[str, str] = {}
    raw = path.read_text().splitlines()
    for line in raw:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, value = stripped.split("=", 1)
        key = key.strip()
        vals[key] = value.strip()
        if key not in order:
            order.append(key)
    return order, vals, raw


def write_params(path: Path, order: list[str], vals: dict[str, str], header: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [header.rstrip(), ""]
    for key in order:
        if key in vals:
            lines.append(f"{key}={vals[key]}")
    extras = [key for key in vals if key not in order]
    if extras:
        lines.append("")
        lines.append("# required-supply-informed GP supply scenario overrides")
        for key in sorted(extras):
            lines.append(f"{key}={vals[key]}")
    path.write_text("\n".join(lines) + "\n")


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def make_ceiling_table() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for T, seed in SEEDS.items():
        for scenario_id, s_margin, explicit, note in SCENARIOS:
            xbc = ceiling_for(seed["xBcrit_beta"], s_margin, explicit)
            rows.append({
                "T_C": T,
                "seed_id": seed["seed_id"],
                "R_eff_h_nm": seed["R_eff_h_nm"],
                "xB_far": XB_FAR,
                "xBcrit_beta": seed["xBcrit_beta"],
                "scenario_id": scenario_id,
                "s_margin": "" if s_margin is None else s_margin,
                "xB_ceiling_eff": xbc,
                "provenance": PROVENANCE,
                "physical_claim_allowed": (
                    "false" if scenario_id == "counterfactual_only"
                    else "scenario_only_no_gp_thermodynamic_claim"
                ),
                "notes": note,
            })
    return rows


def make_case(T: int, scenario: dict[str, object], R_exchange: float, chi: float) -> tuple[Path, str]:
    seed = SEEDS[T]
    order, vals, _ = parse_params(BASE_DIR / str(seed["base_params"]))
    xB_target = float(scenario["xB_ceiling_eff"])
    scenario_id = str(scenario["scenario_id"])
    name = (
        f"T{T}_gp_supply_{scenario_id}_xB{tag_float(xB_target)}_"
        f"R{tag_float(R_exchange)}_chi{tag_float(chi)}_k{tag_float(KERNEL_RADIUS_DX)}_1000post"
    )
    overrides = {
        "temperature_C": f"{float(T):.16e}",
        "ic_23d_xB_out": f"{XB_FAR:.16e}",
        "ic_xB_eq_matrix": f"{XB_FAR:.16e}",
        "gp_initial_xB_tot": "0.03",
        "gp_literature_model_enabled": "0",
        "gp_birth_model": "prescribed_sites",
        "gp_site_mode": "single",
        "gp_n_sites": "1",
        "gp_debug_site_ix": "64",
        "gp_debug_site_iy": "64",
        "gp_debug_site_iz": "64",
        "gp_site_B_mass_equiv": "1.0e-300",
        "gp_site_S_factor": "1.0",
        "gp_literature_xAg_mode": "from_current_mean_xB_alpha",
        "gp_growth_enabled": "0",
        "gp_radius_evolution_enabled": "0",
        "gp_inventory_growth_enabled": "0",
        "enable_legacy_gp_storage_coupling": "0",
        "resolved_handoff_xB_write_mode": "preserve_profile_xB_alpha_in_support",
        "scheduled_nuc_source_lambda_nm": "0.6",
        "scheduled_nuc_target_lambda_nm": "0.6",
        "scheduled_nuc_scale_interface_width": "1.0",
        "scheduled_nuc_scale_xB_profile_width": "1.0",
        "beta_staged_accumulation_enabled": "1",
        "beta_staged_accumulation_interval_steps": "10",
        "beta_staged_accumulation_GP_capture_radius_nm": "12",
        "beta_staged_accumulation_matrix_draw_radius_nm": "64",
        "beta_staged_accumulation_max_fraction_per_step": "0.1",
        "beta_staged_accumulation_max_inventory_per_step": "1.0e300",
        "beta_staged_insert_when_target_reached": "1",
        "beta_staged_debug_accelerated_accumulation": "1",
        "beta_staged_debug_accumulation_rate_multiplier": "1000000",
        "beta_staged_debug_stop_after_resolved_insert": "0",
        "diagnostic_rsmd_enabled": "1",
        "diagnostic_rsmd_T_only": str(T),
        "diagnostic_rsmd_xB_halo_target": f"{xB_target:.16e}",
        "diagnostic_rsmd_R_exchange_nm": f"{R_exchange:.16e}",
        "diagnostic_rsmd_chi_rel": f"{chi:.16e}",
        "diagnostic_rsmd_kernel_radius_dx": f"{KERNEL_RADIUS_DX:.16e}",
        "diagnostic_rsmd_h_src_max": "0.1",
        "diagnostic_rsmd_f_max_per_step": "0.05",
        "diagnostic_rsmd_release_window_steps": str(N_STEPS),
        "diagnostic_rsmd_seed_R_eff_h_nm": f"{float(seed['R_eff_h_nm']):.16e}",
        "diagnostic_rsmd_provenance": PROVENANCE,
        "y_update_mass_projection_enabled": "1",
        "y_update_mass_projection_report_enabled": "1",
    }
    vals.update(overrides)
    header = f"""# Required-supply-informed GP supply scenario RSMD parameter file
# Base: {seed['base_params']}
# T_C={T}
# seed_id={seed['seed_id']}
# scenario_id={scenario_id}
# xB_ceiling_eff={xB_target:.16e}
# provenance={PROVENANCE}
# This is a scenario release ceiling, not calibrated GP thermodynamics or GP solvus.
"""
    out = PARAM_DIR / f"{name}.params"
    write_params(out, order, vals, header)
    return out, name


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--include-counterfactual-runs", action="store_true",
                        help="Also put xB=0.03 counterfactual rows in the run matrix.")
    args = parser.parse_args()

    PARAM_DIR.mkdir(parents=True, exist_ok=True)
    REPORT_DIR.mkdir(parents=True, exist_ok=True)

    ceiling_rows = make_ceiling_table()
    ceiling_fields = [
        "T_C", "seed_id", "R_eff_h_nm", "xB_far", "xBcrit_beta",
        "scenario_id", "s_margin", "xB_ceiling_eff", "provenance",
        "physical_claim_allowed", "notes",
    ]
    write_csv(REPORT_DIR / "gp_required_supply_informed_ceiling_table.csv",
              ceiling_rows, ceiling_fields)

    matrix_rows: list[dict[str, object]] = []
    scenarios_by_T = {(int(r["T_C"]), str(r["scenario_id"])): r for r in ceiling_rows}
    for T, seed in SEEDS.items():
        for scenario_id, _, _, _ in SCENARIOS:
            if scenario_id not in RUN_SCENARIOS and not args.include_counterfactual_runs:
                continue
            scenario = scenarios_by_T[(T, scenario_id)]
            for R_exchange in seed["R_exchange_nm"]:
                for chi in CHI_REL:
                    param_path, case = make_case(T, scenario, float(R_exchange), chi)
                    counterfactual = scenario_id == "counterfactual_only"
                    matrix_rows.append({
                        "case": case,
                        "T_C": T,
                        "seed_id": seed["seed_id"],
                        "R_eff_h_nm": seed["R_eff_h_nm"],
                        "xB_far": XB_FAR,
                        "xBcrit_beta": seed["xBcrit_beta"],
                        "scenario_id": scenario_id,
                        "s_margin": scenario["s_margin"],
                        "xB_ceiling_eff": scenario["xB_ceiling_eff"],
                        "R_exchange_nm": R_exchange,
                        "chi_rel": chi,
                        "kernel_radius_dx": KERNEL_RADIUS_DX,
                        "nsteps": N_STEPS,
                        "param_file": str(param_path.relative_to(ROOT)),
                        "run_selected": 1,
                        "provenance": PROVENANCE,
                        "physical_claim_allowed": scenario["physical_claim_allowed"],
                        "counterfactual_only": int(counterfactual),
                        "notes": scenario["notes"],
                    })
    matrix_fields = [
        "case", "T_C", "seed_id", "R_eff_h_nm", "xB_far", "xBcrit_beta",
        "scenario_id", "s_margin", "xB_ceiling_eff", "R_exchange_nm",
        "chi_rel", "kernel_radius_dx", "nsteps", "param_file",
        "run_selected", "provenance", "physical_claim_allowed",
        "counterfactual_only", "notes",
    ]
    write_csv(REPORT_DIR / "gp_supply_scenario_run_matrix.csv",
              matrix_rows, matrix_fields)
    write_csv(PARAM_DIR / "manifest.csv", matrix_rows, matrix_fields)
    print(f"wrote ceiling rows={len(ceiling_rows)} run matrix rows={len(matrix_rows)}")
    print(f"ceiling_table={REPORT_DIR / 'gp_required_supply_informed_ceiling_table.csv'}")
    print(f"run_matrix={REPORT_DIR / 'gp_supply_scenario_run_matrix.csv'}")


if __name__ == "__main__":
    main()
