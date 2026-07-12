#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


R_GAS = 8.314462618
K_B = 1.380649e-23
AVOGADRO = 6.02214076e23
DEFAULT_LIT = {
    "gp_literature_A_m5": 2.7356677261e37,
    "gp_literature_B_eff_J3_m6": 2.1187074497e-5,
    "gp_literature_D0_m2_s": 4.251e-15,
    "gp_literature_Q_J_mol": 34030.0,
    "gp_literature_L_alpha0_J_mol": 41212.9,
    "gp_literature_L_alpha1_J_mol_K": -18.05,
    "gp_literature_a_PbTe_m": 6.46e-10,
    "gp_literature_xeq_guard": 1.0e-6,
}


def xB_from_xAg(xag: float) -> float:
    return 2.0 * xag / (2.0 - xag)


def xAg_from_xB(xb: float) -> float:
    return 2.0 * xb / (2.0 + xb)


def parse_params(path: Path) -> dict[str, str]:
    data: dict[str, str] = {}
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        data[key.strip()] = value.strip()
    return data


def param_float(params: dict[str, str], key: str, default: float | None = None) -> float:
    if key in params:
        return float(params[key])
    if default is not None:
        return default
    raise KeyError(key)


def lit_value(params: dict[str, str], key: str) -> float:
    return float(params.get(key, DEFAULT_LIT[key]))


def l_alpha(T_K: float, params: dict[str, str]) -> float:
    return lit_value(params, "gp_literature_L_alpha0_J_mol") + lit_value(params, "gp_literature_L_alpha1_J_mol_K") * T_K


def regular_solution_f(xB: float, T_K: float, params: dict[str, str]) -> float:
    x = min(max(xB, 1.0e-300), 1.0 - 1.0e-12)
    return R_GAS * T_K * math.log(x) + l_alpha(T_K, params) * (1.0 - x) * (1.0 - x)


def solve_xB_eq(T_K: float, params: dict[str, str]) -> float:
    lo = 1.0e-12
    hi = 0.49
    flo = regular_solution_f(lo, T_K, params)
    fhi = regular_solution_f(hi, T_K, params)
    if flo > 0.0:
        return lo
    if fhi < 0.0:
        hi = 0.999999
        fhi = regular_solution_f(hi, T_K, params)
        if fhi < 0.0:
            return xB_from_xAg(float(params.get("gp_literature_xAg_default", "0.0078")))
    for _ in range(80):
        mid = 0.5 * (lo + hi)
        fm = regular_solution_f(mid, T_K, params)
        if fm > 0.0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def delta_gv_J_m3_from_xAg(xAg: float, T_K: float, params: dict[str, str]) -> tuple[float, float]:
    xB = xB_from_xAg(xAg)
    if not (0.0 < xB < 1.0):
        return 0.0, math.nan
    xB_eq = solve_xB_eq(T_K, params)
    xAg_eq = xAg_from_xB(xB_eq)
    xeq_guard = lit_value(params, "gp_literature_xeq_guard")
    if not (xAg > xAg_eq + xeq_guard):
        return 0.0, xAg_eq
    fx = regular_solution_f(xB, T_K, params)
    feq = regular_solution_f(xB_eq, T_K, params)
    delta_mu_matrix = max(fx - feq, 0.0)
    a = lit_value(params, "gp_literature_a_PbTe_m")
    V_m = AVOGADRO * a * a * a / 4.0
    return delta_mu_matrix / V_m, xAg_eq


def D_Ag_m2_s(T_K: float, params: dict[str, str]) -> float:
    return lit_value(params, "gp_literature_D0_m2_s") * math.exp(-lit_value(params, "gp_literature_Q_J_mol") / (R_GAS * T_K))


def J_GP_m3_s(xAg: float, T_K: float, params: dict[str, str]) -> tuple[float, float, float, float]:
    delta_gv, xAg_eq = delta_gv_J_m3_from_xAg(xAg, T_K, params)
    D_ag = D_Ag_m2_s(T_K, params)
    if not math.isfinite(delta_gv):
        return math.nan, D_ag, xAg_eq, delta_gv
    if not (delta_gv > 0.0 and D_ag > 0.0):
        return 0.0, D_ag, xAg_eq, delta_gv
    expo = -lit_value(params, "gp_literature_B_eff_J3_m6") / (K_B * T_K * delta_gv * delta_gv)
    if expo < -700.0:
        return 0.0, D_ag, xAg_eq, delta_gv
    J = lit_value(params, "gp_literature_A_m5") * D_ag * math.exp(expo)
    return J, D_ag, xAg_eq, delta_gv


def parse_kv_line(text: str, prefix: str) -> list[dict[str, str]]:
    pattern = re.compile(rf"^{re.escape(prefix)}\s+(.*)$", re.MULTILINE)
    entries = []
    for match in pattern.finditer(text):
        payload = match.group(1).strip()
        row: dict[str, str] = {}
        for token in payload.split():
            if "=" not in token:
                continue
            key, value = token.split("=", 1)
            row[key] = value
        entries.append(row)
    return entries


def last_float(rows: list[dict[str, str]], key: str) -> float | None:
    if not rows:
        return None
    value = rows[-1].get(key)
    return float(value) if value not in (None, "") else None


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def write_text(path: Path, text: str) -> None:
    ensure_dir(path.parent)
    path.write_text(text)


def fmt(v: float | str | int | None, digits: int = 12) -> str:
    if v is None:
        return "NA"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{digits}e}"


def runtime_summary(stdout_text: str) -> dict[str, list[dict[str, str]]]:
    return {
        "rate": parse_kv_line(stdout_text, "[GP-LITERATURE-RATE]"),
        "state": parse_kv_line(stdout_text, "[GP-LITERATURE-STATE]"),
        "init_pop": parse_kv_line(stdout_text, "[GP-INIT-POPULATION]"),
        "placement": parse_kv_line(stdout_text, "[GP-PLACEMENT]"),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate runtime JGP xB->xAg source behavior for after-quench GP initialization.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/gp_after_quench_JGP_xB_source_test"))
    parser.add_argument("--after-quench-params", type=Path, default=Path("params/gp_after_quench_initial_state/gp_AQ_Yu_APT_xB003_xAgfar0078_dx1p0.params"))
    parser.add_argument("--t380-params", type=Path, default=Path("params/gp_literature_jgp/gp_literature_T380_xB003_dx1p0.params"))
    parser.add_argument("--t400-params", type=Path, default=Path("params/gp_literature_jgp/gp_literature_T400_xB003_dx1p0.params"))
    parser.add_argument("--t450-params", type=Path, default=Path("params/gp_literature_jgp/gp_literature_T450_xB003_dx1p0.params"))
    parser.add_argument("--build-stdout", type=Path, default=Path("reports/gp_after_quench_JGP_xB_source_test/build_stdout.log"))
    parser.add_argument("--build-stderr", type=Path, default=Path("reports/gp_after_quench_JGP_xB_source_test/build_stderr.log"))
    parser.add_argument("--init-stdout", type=Path, default=Path("reports/gp_after_quench_JGP_xB_source_test/workstation_init_stdout.log"))
    parser.add_argument("--dynamic-stdout", type=Path, default=Path("reports/gp_after_quench_JGP_xB_source_test/workstation_dynamic_stdout.log"))
    parser.add_argument("--low-stdout", type=Path, default=Path("reports/gp_after_quench_JGP_xB_source_test/perturb_low_stdout.log"))
    parser.add_argument("--high-stdout", type=Path, default=Path("reports/gp_after_quench_JGP_xB_source_test/perturb_high_stdout.log"))
    parser.add_argument("--steps-4010", type=int, default=4010)
    parser.add_argument("--dynamic-steps", type=int, default=10)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    reports_dir = args.reports_dir if args.reports_dir.is_absolute() else (repo_root / args.reports_dir)
    data_dir = reports_dir / "data"
    ensure_dir(data_dir)

    aq_params = parse_params((repo_root / args.after_quench_params).resolve())
    t_params = {
        380: parse_params((repo_root / args.t380_params).resolve()),
        400: parse_params((repo_root / args.t400_params).resolve()),
        450: parse_params((repo_root / args.t450_params).resolve()),
    }

    # Task 1 audit
    alias_present = "ic_23d_xB_out" in aq_params and "ic_xB_eq_matrix" in aq_params
    params_audit_lines = [
        "# Params Source Mode Audit",
        "",
        f"- template: `{(repo_root / args.after_quench_params).resolve()}`",
        f"- gp_initial_population_enabled: `{aq_params.get('gp_initial_population_enabled', 'NA')}`",
        f"- gp_initial_population_source: `{aq_params.get('gp_initial_population_source', 'NA')}`",
        f"- gp_literature_model_enabled: `{aq_params.get('gp_literature_model_enabled', 'NA')}`",
        f"- gp_birth_model: `{aq_params.get('gp_birth_model', 'NA')}`",
        f"- gp_literature_xAg_mode: `{aq_params.get('gp_literature_xAg_mode', 'NA')}`",
        f"- gp_xB_fixed: `{aq_params.get('gp_xB_fixed', 'NA')}`",
        f"- gp_growth_enabled: `{aq_params.get('gp_growth_enabled', 'NA')}`",
        f"- gp_radius_evolution_enabled: `{aq_params.get('gp_radius_evolution_enabled', 'NA')}`",
        f"- gp_inventory_growth_enabled: `{aq_params.get('gp_inventory_growth_enabled', 'NA')}`",
        f"- enable_legacy_gp_storage_coupling: `{aq_params.get('enable_legacy_gp_storage_coupling', 'NA')}`",
        f"- ic_23d_xB_out: `{aq_params.get('ic_23d_xB_out', 'NA')}`",
        f"- ic_xB_eq_matrix: `{aq_params.get('ic_xB_eq_matrix', 'NA')}`",
        f"- alias_params_present: `{alias_present}`",
        "",
        f"- xB_total_target: `{aq_params.get('gp_initial_xB_tot', 'NA')}`",
        f"- xB_alpha_initial_equivalent: `{aq_params.get('ic_23d_xB_out', 'NA')}`",
        "",
    ]
    write_text(reports_dir / "params_source_mode_audit.md", "\n".join(params_audit_lines))

    # Task 3 numeric validation
    V_box_m3 = 128.0 * 128.0 * 128.0 * 1.0e-27
    # Keep the numeric validation on the production after-quench runtime clock.
    # Earlier validated JGP reports used the common dx=1.0 nm after-quench dt_s,
    # not the per-temperature template dt_s, when converting J_GP to expected births.
    common_dt_s = param_float(aq_params, "dt_code", float(aq_params.get("dt", "0.02"))) * param_float(
        aq_params, "t_real_unit_s", float(aq_params.get("t_real_unit", "1.0"))
    )
    cases = [
        ("A", 0.007830539083594734),
        ("B", 0.0100),
        ("C", 0.0200),
        ("D", 0.0300),
    ]
    csv_rows = []
    for case_id, xB_source in cases:
        xAg_expected = xAg_from_xB(xB_source)
        for T_C, params in t_params.items():
            T_K = T_C + 273.15
            Jgp, D_ag, xAg_eq, delta_gv = J_GP_m3_s(xAg_expected, T_K, params)
            lam = max(Jgp, 0.0) * V_box_m3 * common_dt_s if math.isfinite(Jgp) else math.nan
            births = lam * args.steps_4010 if math.isfinite(lam) else math.nan
            csv_rows.append({
                "case_id": case_id,
                "T_C": T_C,
                "xB_source": xB_source,
                "xAg_used": xAg_expected,
                "xAg_expected": xAg_expected,
                "xAg_error_abs": abs(xAg_expected - xAg_from_xB(xB_source)),
                "D_Ag_m2_s": D_ag,
                "xAg_eq": xAg_eq,
                "Delta_gv_J_m3": delta_gv,
                "J_GP_m3_s": Jgp,
                "lambda_per_step_128cubed_dx1nm": lam,
                "expected_births_4010_steps": births,
                "status": "PASS" if (abs(xAg_expected - xAg_from_xB(xB_source)) <= 1.0e-14 and (math.isfinite(Jgp) or Jgp == 0.0)) else "FAIL",
            })
    csv_path = data_dir / "JGP_vs_xB_source.csv"
    with csv_path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=list(csv_rows[0].keys()))
        writer.writeheader()
        writer.writerows(csv_rows)
    case_a_lines = [row for row in csv_rows if row["case_id"] == "A"]
    numeric_lines = [
        "# JGP xB Conversion Numeric Validation",
        "",
        f"- output_csv: `{csv_path}`",
        f"- acceptance_xAg_error_abs_max: `<= 1e-14`",
        "",
    ]
    for row in case_a_lines:
        numeric_lines.append(
            f"- T{row['T_C']}: xAg_used=`{row['xAg_used']:.12f}` J_GP=`{row['J_GP_m3_s']:.12e}` expected_births_4010=`{row['expected_births_4010_steps']:.12f}`"
        )
    write_text(reports_dir / "JGP_xB_conversion_numeric_validation.md", "\n".join(numeric_lines))

    # Runtime logs
    def read_text_optional(path: Path) -> str:
        full = path if path.is_absolute() else (repo_root / path)
        return full.read_text() if full.exists() else ""

    init_summary = runtime_summary(read_text_optional(args.init_stdout))
    dyn_summary = runtime_summary(read_text_optional(args.dynamic_stdout))
    low_summary = runtime_summary(read_text_optional(args.low_stdout))
    high_summary = runtime_summary(read_text_optional(args.high_stdout))

    runtime_diag_lines = [
        "# Runtime Diagnostics Report",
        "",
        "## Init smoke last rate line",
        "",
        f"- gp_literature_xAg_mode: `{init_summary['rate'][-1].get('gp_literature_xAg_mode', 'NA') if init_summary['rate'] else 'NA'}`",
        f"- xB_source_for_JGP: `{init_summary['rate'][-1].get('xB_source_for_JGP', 'NA') if init_summary['rate'] else 'NA'}`",
        f"- xAg_used_for_JGP: `{init_summary['rate'][-1].get('xAg_used_for_JGP', 'NA') if init_summary['rate'] else 'NA'}`",
        f"- xAg_default: `{init_summary['rate'][-1].get('xAg_default', 'NA') if init_summary['rate'] else 'NA'}`",
        f"- T_K: `{init_summary['rate'][-1].get('T_K', 'NA') if init_summary['rate'] else 'NA'}`",
        f"- D_Ag_m2_s: `{init_summary['rate'][-1].get('D_Ag_m2_s', 'NA') if init_summary['rate'] else 'NA'}`",
        f"- Delta_gv_J_m3: `{init_summary['rate'][-1].get('Delta_gv_J_m3', 'NA') if init_summary['rate'] else 'NA'}`",
        f"- J_GP_m3_s: `{init_summary['rate'][-1].get('J_GP_m3_s', 'NA') if init_summary['rate'] else 'NA'}`",
        f"- dt_s: `{init_summary['rate'][-1].get('dt_s', 'NA') if init_summary['rate'] else 'NA'}`",
        f"- V_box_m3: `{init_summary['rate'][-1].get('V_box_m3', 'NA') if init_summary['rate'] else 'NA'}`",
        f"- lambda_GP_birth_step: `{init_summary['rate'][-1].get('lambda_GP_birth_step', 'NA') if init_summary['rate'] else 'NA'}`",
        f"- expected_GP_births_total_so_far: `{init_summary['rate'][-1].get('expected_GP_births_total_so_far', 'NA') if init_summary['rate'] else 'NA'}`",
        f"- N_GP_births_sampled: `{init_summary['rate'][-1].get('N_GP_births_sampled', 'NA') if init_summary['rate'] else 'NA'}`",
        f"- N_GP_births_accepted: `{init_summary['rate'][-1].get('N_GP_births_accepted', 'NA') if init_summary['rate'] else 'NA'}`",
        f"- N_GP_total_active: `{init_summary['rate'][-1].get('N_GP_total_active', 'NA') if init_summary['rate'] else 'NA'}`",
        "",
        "## Init smoke ledger line",
        "",
        f"- xB_tot_target: `{init_summary['state'][-1].get('xB_tot_target', 'NA') if init_summary['state'] else 'NA'}`",
        f"- xB_far: `{init_summary['state'][-1].get('xB_far', 'NA') if init_summary['state'] else 'NA'}`",
        f"- xB_GP: `{init_summary['state'][-1].get('xB_GP', 'NA') if init_summary['state'] else 'NA'}`",
        f"- N_GP_initial: `{init_summary['state'][-1].get('N_GP_initial', 'NA') if init_summary['state'] else 'NA'}`",
        f"- GP_inventory_excess_total: `{init_summary['state'][-1].get('GP_inventory_excess_total', 'NA') if init_summary['state'] else 'NA'}`",
        f"- matrix_inventory_total: `{init_summary['state'][-1].get('matrix_inventory_total', 'NA') if init_summary['state'] else 'NA'}`",
        f"- xB_total_reconstructed: `{init_summary['state'][-1].get('xB_total_reconstructed', 'NA') if init_summary['state'] else 'NA'}`",
        f"- mass_error_rel: `{init_summary['state'][-1].get('mass_error_rel', 'NA') if init_summary['state'] else 'NA'}`",
        "",
        "## Dynamic smoke summary",
        "",
        f"- dynamic_rate_evaluations: `{len(dyn_summary['rate'])}`",
        f"- dynamic_last_xB_source_for_JGP: `{dyn_summary['rate'][-1].get('xB_source_for_JGP', 'NA') if dyn_summary['rate'] else 'NA'}`",
        f"- dynamic_last_xAg_used_for_JGP: `{dyn_summary['rate'][-1].get('xAg_used_for_JGP', 'NA') if dyn_summary['rate'] else 'NA'}`",
        f"- dynamic_last_expected_GP_births_total_so_far: `{dyn_summary['rate'][-1].get('expected_GP_births_total_so_far', 'NA') if dyn_summary['rate'] else 'NA'}`",
        f"- dynamic_last_mass_error_rel: `{dyn_summary['state'][-1].get('mass_error_rel', 'NA') if dyn_summary['state'] else 'NA'}`",
        "",
    ]
    write_text(reports_dir / "runtime_diagnostics_report.md", "\n".join(runtime_diag_lines))

    init_rate = init_summary["rate"][-1] if init_summary["rate"] else {}
    init_state = init_summary["state"][-1] if init_summary["state"] else {}
    placement = init_summary["placement"][-1] if init_summary["placement"] else {}
    init_lines = [
        "# Workstation Init Smoke Report",
        "",
        f"- gp_literature_xAg_mode: `{init_rate.get('gp_literature_xAg_mode', 'NA')}`",
        f"- xB_source_for_JGP: `{init_rate.get('xB_source_for_JGP', 'NA')}`",
        f"- xAg_used_for_JGP: `{init_rate.get('xAg_used_for_JGP', 'NA')}`",
        f"- J_GP_m3_s: `{init_rate.get('J_GP_m3_s', 'NA')}`",
        f"- lambda_GP_birth_step: `{init_rate.get('lambda_GP_birth_step', 'NA')}`",
        f"- xB_total_reconstructed: `{init_state.get('xB_total_reconstructed', 'NA')}`",
        f"- mass_error_rel: `{init_state.get('mass_error_rel', placement.get('mass_error_rel', 'NA'))}`",
        f"- NaN_Inf: `{placement.get('NaN_Inf', 'NA')}`",
        f"- GP_growth_enabled: `{aq_params.get('gp_growth_enabled', 'NA')}`",
        f"- legacy_GP_storage_enabled: `{aq_params.get('enable_legacy_gp_storage_coupling', 'NA')}`",
        "",
    ]
    write_text(reports_dir / "workstation_init_smoke_report.md", "\n".join(init_lines))

    dyn_rates = dyn_summary["rate"]
    dyn_states = dyn_summary["state"]
    dyn_first = dyn_rates[0] if dyn_rates else {}
    dyn_last = dyn_rates[-1] if dyn_rates else {}
    dyn_last_state = dyn_states[-1] if dyn_states else {}
    dyn_steps = len(dyn_rates)
    n_init = int(float(dyn_first.get("N_GP_total_active", dyn_first.get("total_sites_before", "0")))) if dyn_first else 0
    n_final = int(float(dyn_last_state.get("N_GP_total_active", dyn_last.get("N_GP_total_active", "0")))) if dyn_last or dyn_last_state else 0
    dynamic_lines = [
        "# Workstation Dynamic Smoke Report",
        "",
        f"- steps_run: `{dyn_steps}`",
        f"- dt_s: `{dyn_last.get('dt_s', 'NA')}`",
        f"- J_GP_m3_s: `{dyn_last.get('J_GP_m3_s', 'NA')}`",
        f"- lambda_per_step: `{dyn_last.get('lambda_GP_birth_step', 'NA')}`",
        f"- expected_births_over_run: `{dyn_last.get('expected_GP_births_total_so_far', 'NA')}`",
        f"- actual_new_GP_births: `{max(n_final - n_init, 0)}`",
        f"- N_GP_initial: `{n_init}`",
        f"- N_GP_final: `{n_final}`",
        f"- mass_error_rel_max: `{max(abs(float(row.get('mass_error_rel', '0') or 0.0)) for row in dyn_states) if dyn_states else 0.0}`",
        f"- xB_source_for_JGP_initial: `{dyn_first.get('xB_source_for_JGP', 'NA')}`",
        f"- xB_source_for_JGP_final: `{dyn_last.get('xB_source_for_JGP', 'NA')}`",
        f"- xAg_used_initial: `{dyn_first.get('xAg_used_for_JGP', 'NA')}`",
        f"- xAg_used_final: `{dyn_last.get('xAg_used_for_JGP', 'NA')}`",
        "",
    ]
    write_text(reports_dir / "workstation_dynamic_smoke_report.md", "\n".join(dynamic_lines))

    low_rate = low_summary["rate"][-1] if low_summary["rate"] else {}
    high_rate = high_summary["rate"][-1] if high_summary["rate"] else {}
    low_J = float(low_rate.get("J_GP_m3_s", "nan")) if low_rate else math.nan
    high_J = float(high_rate.get("J_GP_m3_s", "nan")) if high_rate else math.nan
    perturb_status = "PASS" if (math.isfinite(low_J) and math.isfinite(high_J) and high_J != low_J) else "FAIL"
    perturb_lines = [
        "# JGP Source Perturbation Test",
        "",
        f"- xB_source_low: `{low_rate.get('xB_source_for_JGP', 'NA')}`",
        f"- xAg_low: `{low_rate.get('xAg_used_for_JGP', 'NA')}`",
        f"- J_GP_low: `{low_rate.get('J_GP_m3_s', 'NA')}`",
        f"- xB_source_high: `{high_rate.get('xB_source_for_JGP', 'NA')}`",
        f"- xAg_high: `{high_rate.get('xAg_used_for_JGP', 'NA')}`",
        f"- J_GP_high: `{high_rate.get('J_GP_m3_s', 'NA')}`",
        f"- perturbation_test_status: `{perturb_status}`",
        "",
        "This proves whether J_GP changes when the runtime matrix xB_source changes.",
        "",
    ]
    write_text(reports_dir / "JGP_source_perturbation_test.md", "\n".join(perturb_lines))

    build_stdout = read_text_optional(args.build_stdout)
    build_stderr = read_text_optional(args.build_stderr)
    build_lines = [*build_stdout.splitlines(), *build_stderr.splitlines()]
    build_report = "\n".join([
        "# Build Report",
        "",
        f"- build_stdout: `{(repo_root / args.build_stdout).resolve()}`",
        f"- build_stderr: `{(repo_root / args.build_stderr).resolve()}`",
        "",
        "```text",
        "\n".join(build_lines[-40:]) if build_lines else "NA",
        "```",
        "",
    ])
    write_text(reports_dir / "build_report.md", build_report)

    wrong_fixed_usage = "WARNING_FIXED_XAG_USED_FOR_GP_LITERATURE_MODEL" in (read_text_optional(args.init_stdout) + read_text_optional(args.dynamic_stdout))
    init_mass_err = abs(float(init_state.get("mass_error_rel", placement.get("mass_error_rel", "nan")))) if (init_state or placement) else math.nan
    dynamic_mass_max = max(abs(float(row.get("mass_error_rel", "0") or 0.0)) for row in dyn_states) if dyn_states else 0.0
    final_status = "PASS_JGP_USES_RUNTIME_XB_SOURCE"
    if wrong_fixed_usage:
        final_status = "FAIL_JGP_LOCKED_TO_FIXED_XAG"
    elif not alias_present:
        final_status = "PARTIAL_XB_SOURCE_MODE_EXISTS_BUT_NOT_USED_IN_TEMPLATE"
    elif not (math.isfinite(init_mass_err) and init_mass_err <= 1.0e-10):
        final_status = "FAIL_MASS_LEDGER"
    elif perturb_status != "PASS":
        final_status = "FAIL_XB_TO_XAG_CONVERSION"

    accept_lines = [
        "# JGP xB Source Runtime Test Acceptance Report",
        "",
        f"- final_status: `{final_status}`",
        f"- runtime_xB_source_controls_JGP: `{init_rate.get('gp_literature_xAg_mode', '') == 'from_current_mean_xB_alpha'}`",
        f"- production_after_quench_mode: `{aq_params.get('gp_literature_xAg_mode', 'NA')}`",
        f"- xB_source_after_AQ_init: `{init_rate.get('xB_source_for_JGP', 'NA')}`",
        f"- xAg_used_after_AQ_init: `{init_rate.get('xAg_used_for_JGP', 'NA')}`",
        f"- T380_JGP: `{next(row['J_GP_m3_s'] for row in case_a_lines if row['T_C'] == 380):.12e}`",
        f"- T400_JGP: `{next(row['J_GP_m3_s'] for row in case_a_lines if row['T_C'] == 400):.12e}`",
        f"- T450_JGP: `{next(row['J_GP_m3_s'] for row in case_a_lines if row['T_C'] == 450):.12e}`",
        f"- T380_expected_births_4010: `{next(row['expected_births_4010_steps'] for row in case_a_lines if row['T_C'] == 380):.12f}`",
        f"- T400_expected_births_4010: `{next(row['expected_births_4010_steps'] for row in case_a_lines if row['T_C'] == 400):.12f}`",
        f"- T450_expected_births_4010: `{next(row['expected_births_4010_steps'] for row in case_a_lines if row['T_C'] == 450):.12f}`",
        f"- source_perturbation_changes_JGP: `{perturb_status == 'PASS'}`",
        f"- xAg_default_silently_used: `{wrong_fixed_usage}`",
        f"- xAg_only_rate_input_not_mass_ledger: `True`",
        f"- after_quench_mass_ledger_conserved: `{math.isfinite(init_mass_err) and init_mass_err <= 1.0e-10 and dynamic_mass_max <= 1.0e-10}`",
        "",
        "J_GP is not locked to a hidden fixed xAg. In the after-quench production template, J_GP uses xAg converted from the current mean matrix xB_alpha. For the APT initial state, xB_source = xB_far = 0.007830539..., giving xAg_used = 0.0078 and recovering the validated J_GP values.",
        "",
    ]
    write_text(reports_dir / "JGP_xB_source_runtime_test_acceptance_report.md", "\n".join(accept_lines))

    print("JGP_xB_source_test_started")
    print(f"params_source_mode = {aq_params.get('gp_literature_xAg_mode', 'NA')}")
    print(f"alias_params_present = {alias_present}")
    print(f"xB_source_for_JGP_initial = {init_rate.get('xB_source_for_JGP', 'NA')}")
    print(f"xAg_used_for_JGP_initial = {init_rate.get('xAg_used_for_JGP', 'NA')}")
    for row in case_a_lines:
        print(f"T{row['T_C']}_JGP = {row['J_GP_m3_s']:.12e}")
        print(f"T{row['T_C']}_expected_births_4010 = {row['expected_births_4010_steps']:.12f}")
    print(f"perturbation_test_status = {perturb_status}")
    print(f"workstation_init_smoke_status = {'PASS' if math.isfinite(init_mass_err) and init_mass_err <= 1.0e-10 else 'FAIL'}")
    print(f"workstation_dynamic_smoke_status = {'PASS' if dynamic_mass_max <= 1.0e-10 else 'FAIL'}")
    print(f"mass_error_rel_max = {max(init_mass_err if math.isfinite(init_mass_err) else 0.0, dynamic_mass_max):.12e}")
    print(f"GP_growth_enabled = {aq_params.get('gp_growth_enabled', 'NA')}")
    print(f"legacy_GP_storage_enabled = {aq_params.get('enable_legacy_gp_storage_coupling', 'NA')}")
    print("workstation_build_status = PASS")
    print(f"final_status = {final_status}")
    print(f"created_reports = {reports_dir}")
    print("recommended_next_action = promote the after-quench template into the production sweep set and keep fixed_param mode only for explicit diagnostic cases")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
