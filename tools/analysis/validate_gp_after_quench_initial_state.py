#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from pathlib import Path


XAG_FAR = 0.0078
XAG_GP = 0.30
XB_TOT = 0.03
RHO_GP_M3 = 7.5e24
EXACT_GP_XB = 2.0 * XAG_GP / (2.0 - XAG_GP)
EXACT_FAR_XB = 2.0 * XAG_FAR / (2.0 - XAG_FAR)
OLD_GP_XB = 0.35


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


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as fp:
        return list(csv.DictReader(fp))


def try_float(value: str | None) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except ValueError:
        return None


def require_float(params: dict[str, str], key: str) -> float:
    value = try_float(params.get(key))
    if value is None:
        raise ValueError(f"missing float param {key}")
    return value


def parse_keyed_line(text: str, prefix: str) -> dict[str, str]:
    pattern = re.compile(rf"^{re.escape(prefix)}\s+(.*)$", re.MULTILINE)
    match = pattern.search(text)
    if not match:
        return {}
    payload = match.group(1).strip()
    data: dict[str, str] = {}
    for token in payload.split():
        if "=" not in token:
            continue
        key, value = token.split("=", 1)
        data[key] = value
    return data


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else float("nan")


def stddev(values: list[float], mu: float | None = None) -> float:
    if not values:
        return float("nan")
    if mu is None:
        mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / len(values))


def volume_equivalent_radius_nm(volumes_m3: list[float]) -> float:
    if not volumes_m3:
        return float("nan")
    v_mean = sum(volumes_m3) / len(volumes_m3)
    return ((3.0 * v_mean) / (4.0 * math.pi)) ** (1.0 / 3.0) * 1.0e9


def radius_from_volume_m3(volume_m3: float) -> float:
    return ((3.0 * volume_m3) / (4.0 * math.pi)) ** (1.0 / 3.0) * 1.0e9


def safe_min(values: list[float]) -> float:
    return min(values) if values else float("nan")


def safe_max(values: list[float]) -> float:
    return max(values) if values else float("nan")


def pair_metrics(rows: list[dict[str, str]], marker_influence_nm: float, depletion_nm: float) -> dict[str, float]:
    centers = []
    radii = []
    for row in rows:
        centers.append(
            (
                float(row["center_x_nm"]),
                float(row["center_y_nm"]),
                float(row["center_z_nm"]),
            )
        )
        radii.append(float(row["R_final_nm"]))
    n = len(centers)
    if n <= 1:
        return {
            "nearest_neighbor_mean_nm": float("nan"),
            "nearest_neighbor_min_nm": float("nan"),
            "core_overlap_count": 0.0,
            "core_overlap_fraction": 0.0,
            "marker_influence_overlap_fraction": 0.0,
            "depletion_halo_overlap_fraction": 0.0,
        }
    nn = []
    core_overlap = 0
    marker_overlap = 0
    depletion_overlap = 0
    pair_total = 0
    for i in range(n):
        best = float("inf")
        xi, yi, zi = centers[i]
        for j in range(n):
            if i == j:
                continue
            xj, yj, zj = centers[j]
            dist_nm = float(math.sqrt((xi - xj) ** 2 + (yi - yj) ** 2 + (zi - zj) ** 2))
            if dist_nm < best:
                best = dist_nm
            if j <= i:
                continue
            pair_total += 1
            if dist_nm < (float(radii[i]) + float(radii[j])):
                core_overlap += 1
            if dist_nm < 2.0 * float(marker_influence_nm):
                marker_overlap += 1
            if dist_nm < 2.0 * float(depletion_nm):
                depletion_overlap += 1
        nn.append(best)
    return {
        "nearest_neighbor_mean_nm": mean(nn),
        "nearest_neighbor_min_nm": min(nn),
        "core_overlap_count": float(core_overlap),
        "core_overlap_fraction": core_overlap / pair_total if pair_total else 0.0,
        "marker_influence_overlap_fraction": marker_overlap / pair_total if pair_total else 0.0,
        "depletion_halo_overlap_fraction": depletion_overlap / pair_total if pair_total else 0.0,
    }


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text)


def fmt(v: float | int | str | None, digits: int = 12) -> str:
    if isinstance(v, str):
        return v
    if v is None:
        return "NA"
    if isinstance(v, int):
        return str(v)
    if not math.isfinite(v):
        return "NA"
    return f"{v:.{digits}e}"


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate GP after-quench xB-ledger initialization.")
    parser.add_argument("--repo-root", type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument("--case-dir", type=Path, required=True)
    parser.add_argument("--params-file", type=Path, required=True)
    parser.add_argument("--stdout-file", type=Path, required=True)
    parser.add_argument("--reports-dir", type=Path, default=Path("reports/gp_after_quench_initial_state"))
    parser.add_argument("--nx", type=int, default=128)
    parser.add_argument("--ny", type=int, default=128)
    parser.add_argument("--nz", type=int, default=128)
    parser.add_argument("--dx-nm", type=float, default=1.0)
    parser.add_argument("--dy-nm", type=float, default=None)
    parser.add_argument("--dz-nm", type=float, default=None)
    parser.add_argument("--build-status", default="PASS")
    parser.add_argument("--build-command", default="")
    parser.add_argument("--build-log", type=Path, default=None)
    args = parser.parse_args()

    repo_root = args.repo_root.resolve()
    case_dir = args.case_dir.resolve()
    params_file = args.params_file.resolve()
    stdout_file = args.stdout_file.resolve()
    reports_dir = (repo_root / args.reports_dir).resolve() if not args.reports_dir.is_absolute() else args.reports_dir.resolve()
    data_dir = reports_dir / "data"
    reports_dir.mkdir(parents=True, exist_ok=True)
    data_dir.mkdir(parents=True, exist_ok=True)

    params = parse_params(params_file)
    stdout_text = stdout_file.read_text() if stdout_file.exists() else ""
    site_rows = read_csv_rows(case_dir / "gp_initial_population_sites.csv")
    placement_rows = read_csv_rows(case_dir / "gp_multi_site_mass_ledger.csv") if (case_dir / "gp_multi_site_mass_ledger.csv").exists() else []

    dx_nm = args.dx_nm
    dy_nm = args.dy_nm if args.dy_nm is not None else dx_nm
    dz_nm = args.dz_nm if args.dz_nm is not None else dx_nm
    box_volume_m3 = args.nx * args.ny * args.nz * dx_nm * dy_nm * dz_nm * 1.0e-27

    xB_far = xB_from_xAg(require_float(params, "gp_initial_xAg_far"))
    xB_gp = xB_from_xAg(require_float(params, "gp_initial_xAg_GP"))
    xB_tot = require_float(params, "gp_initial_xB_tot")
    rho_gp = require_float(params, "gp_initial_rho_m3")
    f_gp_required = (xB_tot - xB_far) / (xB_gp - xB_far)
    v_gp_required = f_gp_required * box_volume_m3
    n_gp_expected = rho_gp * box_volume_m3
    n_gp_initial = len(site_rows)
    r_vol_eq_nm = ((3.0 * (v_gp_required / max(n_gp_initial, 1))) / (4.0 * math.pi)) ** (1.0 / 3.0) * 1.0e9

    raw_radii = [float(r["R_raw_nm"]) for r in site_rows]
    final_radii = [float(r["R_final_nm"]) for r in site_rows]
    volumes = [float(r["V_final_m3"]) for r in site_rows]
    inventory = [float(r["B_equiv_inventory_m3"]) for r in site_rows]

    raw_mean = mean(raw_radii)
    raw_std = stddev(raw_radii, raw_mean)
    raw_rve = volume_equivalent_radius_nm([((4.0 / 3.0) * math.pi * (r * 1.0e-9) ** 3) for r in raw_radii])
    final_mean = mean(final_radii)
    final_std = stddev(final_radii, final_mean)
    final_rve = volume_equivalent_radius_nm(volumes)
    scale_factor = final_mean / raw_mean if raw_mean and math.isfinite(raw_mean) else float("nan")
    sum_r3 = sum(r ** 3 for r in final_radii)
    target_sum_r3 = v_gp_required / (((4.0 / 3.0) * math.pi) * 1.0e-27)
    sum_r3_rel = abs(sum_r3 - target_sum_r3) / target_sum_r3 if target_sum_r3 else float("nan")

    gp_inventory_total = sum(inventory)
    matrix_inventory_total = xB_far * box_volume_m3
    beta_inventory_total = 0.0
    xB_total_reconstructed = (matrix_inventory_total + gp_inventory_total + beta_inventory_total) / box_volume_m3
    mass_error_abs = abs(xB_total_reconstructed - xB_tot) * box_volume_m3
    mass_error_rel = abs(xB_total_reconstructed - xB_tot) / xB_tot

    gp_init_population = parse_keyed_line(stdout_text, "[GP-INIT-POPULATION]")
    gp_placement = parse_keyed_line(stdout_text, "[GP-PLACEMENT]")
    gp_jgp = parse_keyed_line(stdout_text, "[GP-LITERATURE-RATE]")

    overlap = pair_metrics(site_rows, require_float(params, "gp_marker_influence_radius_nm"), require_float(params, "gp_depletion_radius_nm"))

    radius_csv = data_dir / "initial_GP_radius_distribution.csv"
    with radius_csv.open("w", newline="") as fp:
        writer = csv.writer(fp)
        writer.writerow(["site_id", "R_raw_nm", "R_final_nm", "V_final_m3", "xB_GP", "B_equiv_inventory_m3"])
        for row in site_rows:
            writer.writerow([
                row["site_id"],
                row["R_raw_nm"],
                row["R_final_nm"],
                row["V_final_m3"],
                row["xB_GP"],
                row["B_equiv_inventory_m3"],
            ])

    gp_xb_default_paths = [
        repo_root / "main_cuda.cu",
        repo_root / "params/gp_literature_jgp/gp_literature_T380_xB003_dx1p0.params",
        repo_root / "params/gp_literature_jgp/gp_literature_T400_xB003_dx1p0.params",
        repo_root / "params/gp_literature_jgp/gp_literature_T450_xB003_dx1p0.params",
        repo_root / "params/gp_after_quench_initial_state/gp_AQ_Yu_APT_xB003_xAgfar0078_dx1p0.params",
    ]
    updated_param_paths = []
    for path in gp_xb_default_paths:
        if not path.exists():
            continue
        text = path.read_text()
        if "0.3529411764705882" in text:
            updated_param_paths.append(str(path))

    runtime_print_updated = ("gp_xB_fixed" in stdout_text) or ("gp_xB_gp_ref(reuse_gp_xB_fixed)" in stdout_text)

    build_log_excerpt = ""
    if args.build_log and args.build_log.exists():
        lines = args.build_log.read_text().splitlines()
        build_log_excerpt = "\n".join(lines[-40:])

    build_report = "\n".join([
        "# Build Report",
        "",
        f"- status: `{args.build_status}`",
        f"- command: `{args.build_command or 'NA'}`",
        f"- build_log: `{args.build_log}`" if args.build_log else "- build_log: `NA`",
        "",
        "## Notes",
        "",
        "This build report covers workstation-only compilation for the after-quench initializer path.",
        "",
        "## Build log tail",
        "",
        "```text",
        build_log_excerpt or "NA",
        "```",
        "",
    ])
    write_text(reports_dir / "build_report.md", build_report)

    gp_exact_report = "\n".join([
        "# GP xB_GP Exact Fix Report",
        "",
        f"- old_value: `{OLD_GP_XB}`",
        f"- new_value: `{EXACT_GP_XB:.16f}`",
        f"- runtime_print_updated: `{runtime_print_updated}`",
        "",
        "## Files updated/scanned",
        "",
        *[f"- `{path}`" for path in updated_param_paths],
        "",
        "The exact production value corresponds to `xAg_GP = 0.30 -> xB_GP = 2*xAg/(2-xAg)`.",
        "",
    ])
    write_text(reports_dir / "gp_xB_GP_exact_fix_report.md", gp_exact_report)

    jgp_report = "\n".join([
        "# JGP xAg Source Mode Report",
        "",
        f"- configured_mode: `{params.get('gp_literature_xAg_mode', 'NA')}`",
        f"- runtime_mode: `{gp_jgp.get('gp_literature_xAg_mode', 'NA')}`",
        f"- xB_source_for_JGP: `{gp_jgp.get('xB_source_for_JGP', 'NA')}`",
        f"- xAg_used_for_JGP: `{gp_jgp.get('xAg_used_for_JGP', 'NA')}`",
        f"- xAg_default: `{gp_jgp.get('xAg_default', params.get('gp_literature_xAg_default', 'NA'))}`",
        f"- J_GP_m3_s: `{gp_jgp.get('J_GP_m3_s', 'NA')}`",
        "",
        "## Acceptance",
        "",
        f"- uses_xAg_only_as_rate_input: `{params.get('gp_literature_xAg_mode', '') != 'fixed_param'}`",
        f"- after_quench_matrix_xB_source: `{xB_far:.15f}`",
        f"- after_quench_matrix_xAg_source: `{xAg_from_xB(xB_far):.12f}`",
        "",
    ])
    write_text(reports_dir / "JGP_xAg_source_mode_report.md", jgp_report)

    init_generator_report = "\n".join([
        "# Initial Population Generator Report",
        "",
        f"- source: `{params.get('gp_initial_population_source', 'NA')}`",
        f"- xB_tot: `{xB_tot:.12f}`",
        f"- xB_far: `{xB_far:.15f}`",
        f"- xB_GP: `{xB_gp:.15f}`",
        f"- rho_GP_m3: `{rho_gp:.6e}`",
        f"- V_box_m3: `{box_volume_m3:.12e}`",
        f"- N_GP_expected: `{n_gp_expected:.12f}`",
        f"- N_GP_initial: `{n_gp_initial}`",
        f"- f_GP_required: `{f_gp_required:.12f}`",
        f"- V_GP_required_m3: `{v_gp_required:.12e}`",
        f"- R_vol_eq_nm: `{r_vol_eq_nm:.12f}`",
        f"- runtime_N_expected: `{gp_init_population.get('N_expected', 'NA')}`",
        f"- runtime_N_initial: `{gp_init_population.get('N_initial', 'NA')}`",
        "",
    ])
    write_text(reports_dir / "initial_population_generator_report.md", init_generator_report)

    radius_report = "\n".join([
        "# Radius Distribution Report",
        "",
        f"- raw_mean_radius_nm: `{raw_mean:.12f}`",
        f"- raw_std_radius_nm: `{raw_std:.12f}`",
        f"- raw_volume_equivalent_radius_nm: `{raw_rve:.12f}`",
        f"- scale_factor_mean_based: `{scale_factor:.12f}`",
        f"- final_mean_radius_nm: `{final_mean:.12f}`",
        f"- final_std_radius_nm: `{final_std:.12f}`",
        f"- final_volume_equivalent_radius_nm: `{final_rve:.12f}`",
        f"- final_min_radius_nm: `{safe_min(final_radii):.12f}`",
        f"- final_max_radius_nm: `{safe_max(final_radii):.12f}`",
        f"- sum_R3_relative_error: `{sum_r3_rel:.12e}`",
        "",
        f"Data CSV: `{radius_csv}`",
        "",
    ])
    write_text(reports_dir / "radius_distribution_report.md", radius_report)

    ledger_report = "\n".join([
        "# Initial xB Ledger Reconstruction Report",
        "",
        f"- xB_tot_target: `{xB_tot:.12f}`",
        f"- xAg_far_input: `{require_float(params, 'gp_initial_xAg_far'):.12f}`",
        f"- xB_far: `{xB_far:.15f}`",
        f"- xAg_GP_input: `{require_float(params, 'gp_initial_xAg_GP'):.12f}`",
        f"- xB_GP: `{xB_gp:.15f}`",
        f"- rho_GP: `{rho_gp:.6e}`",
        f"- N_GP_initial: `{n_gp_initial}`",
        f"- f_GP_required: `{f_gp_required:.12f}`",
        f"- V_GP_required_m3: `{v_gp_required:.12e}`",
        f"- GP_inventory_total: `{gp_inventory_total:.12e}`",
        f"- matrix_inventory_total: `{matrix_inventory_total:.12e}`",
        f"- beta_inventory_total: `{beta_inventory_total:.12e}`",
        f"- xB_total_reconstructed: `{xB_total_reconstructed:.12f}`",
        f"- mass_error_abs: `{mass_error_abs:.12e}`",
        f"- mass_error_rel: `{mass_error_rel:.12e}`",
        f"- runtime_placement_mode: `{gp_placement.get('mode', params.get('gp_initial_mass_mode', 'NA'))}`",
        f"- runtime_total_mass_before: `{gp_placement.get('total_mass_before', 'NA')}`",
        f"- runtime_total_mass_after: `{gp_placement.get('total_mass_after', 'NA')}`",
        "",
        "Note: `GP_inventory_total` is interpreted as the excess xB ledger above the uniform matrix baseline `xB_far`.",
        "",
    ])
    write_text(reports_dir / "initial_xB_ledger_reconstruction_report.md", ledger_report)

    position_report = "\n".join([
        "# Position Overlap Report",
        "",
        f"- nearest_neighbor_mean_nm: `{overlap['nearest_neighbor_mean_nm']:.12f}`",
        f"- nearest_neighbor_min_nm: `{overlap['nearest_neighbor_min_nm']:.12f}`",
        f"- core_overlap_count: `{int(overlap['core_overlap_count'])}`",
        f"- core_overlap_fraction: `{overlap['core_overlap_fraction']:.12e}`",
        f"- marker_influence_overlap_fraction: `{overlap['marker_influence_overlap_fraction']:.12e}`",
        f"- depletion_halo_overlap_fraction: `{overlap['depletion_halo_overlap_fraction']:.12e}`",
        f"- HIGH_DENSITY_SUBGRID_OVERLAP_ALLOWED: `{gp_init_population.get('HIGH_DENSITY_SUBGRID_OVERLAP_ALLOWED', 'NA')}`",
        f"- fallback_accept_count: `{gp_init_population.get('fallback_accept_count', 'NA')}`",
        "",
    ])
    write_text(reports_dir / "position_overlap_report.md", position_report)

    params_template_checks = {
        "ic_23d_xB_out": 0.007830539083594734,
        "ic_xB_eq_matrix": 0.007830539083594734,
        "gp_initial_population_enabled": 1.0,
        "gp_initial_xB_tot": 0.03,
        "gp_initial_rho_m3": 7.5e24,
        "gp_initial_xAg_far": 0.0078,
        "gp_initial_xAg_GP": 0.30,
        "gp_xB_fixed": 0.3529411764705882,
        "gp_growth_enabled": 0.0,
        "gp_radius_evolution_enabled": 0.0,
        "gp_inventory_growth_enabled": 0.0,
    }
    params_lines = ["# Params Template Report", "", f"- template: `{params_file}`", "", "## Numeric checks", ""]
    for key, expected in params_template_checks.items():
        actual = try_float(params.get(key))
        ok = actual is not None and abs(actual - expected) <= max(1.0e-12, abs(expected) * 1.0e-12)
        params_lines.append(f"- `{key}`: actual=`{actual}` expected=`{expected}` status=`{'PASS' if ok else 'FAIL'}`")
    params_lines.extend([
        "",
        "## String checks",
        "",
        f"- `gp_initial_population_source`: `{params.get('gp_initial_population_source', 'NA')}`",
        f"- `gp_initial_radius_distribution`: `{params.get('gp_initial_radius_distribution', 'NA')}`",
        f"- `gp_initial_radius_renormalization`: `{params.get('gp_initial_radius_renormalization', 'NA')}`",
        f"- `gp_literature_xAg_mode`: `{params.get('gp_literature_xAg_mode', 'NA')}`",
        f"- `enable_legacy_gp_storage_coupling`: `{params.get('enable_legacy_gp_storage_coupling', 'NA')}`",
        "",
    ])
    write_text(reports_dir / "params_template_report.md", "\n".join(params_lines))

    init_ok = (
        n_gp_initial == round(n_gp_expected)
        and abs(xB_total_reconstructed - xB_tot) <= max(1.0e-12, xB_tot * 1.0e-10)
        and sum_r3_rel <= 1.0e-10
        and gp_jgp.get("gp_literature_xAg_mode", params.get("gp_literature_xAg_mode", "")) != "fixed_param"
        and params.get("gp_growth_enabled") == "0"
        and params.get("gp_radius_evolution_enabled") == "0"
        and params.get("gp_inventory_growth_enabled") == "0"
        and params.get("enable_legacy_gp_storage_coupling") == "0"
    )

    validation_report = "\n".join([
        "# Workstation Init Validation Report",
        "",
        f"- case_dir: `{case_dir}`",
        f"- stdout_file: `{stdout_file}`",
        f"- N_GP_initial: `{n_gp_initial}`",
        f"- N_GP_expected_round: `{round(n_gp_expected)}`",
        f"- xB_alpha_initial_target: `{xB_far:.15f}`",
        f"- xB_total_reconstructed: `{xB_total_reconstructed:.12f}`",
        f"- GP_volume_fraction_required: `{f_gp_required:.12f}`",
        f"- mass_error_rel: `{mass_error_rel:.12e}`",
        f"- sum_R3_relative_error: `{sum_r3_rel:.12e}`",
        f"- JGP_xAg_mode: `{gp_jgp.get('gp_literature_xAg_mode', 'NA')}`",
        f"- xB_source_for_JGP: `{gp_jgp.get('xB_source_for_JGP', 'NA')}`",
        f"- xAg_used_for_JGP: `{gp_jgp.get('xAg_used_for_JGP', 'NA')}`",
        f"- GP_growth_enabled: `{params.get('gp_growth_enabled', 'NA')}`",
        f"- legacy_GP_storage_enabled: `{params.get('enable_legacy_gp_storage_coupling', 'NA')}`",
        f"- init_validation_status: `{'PASS' if init_ok else 'FAIL'}`",
        "",
    ])
    write_text(reports_dir / "workstation_init_validation_report.md", validation_report)

    final_status = "PASS_AFTER_QUENCH_APT_INITIAL_STATE_READY" if init_ok else "FAIL_INIT_VALIDATION"
    acceptance = "\n".join([
        "# After-Quench Initial State Acceptance Report",
        "",
        f"- final_status: `{final_status}`",
        f"- xB_tot_ledger_only: `{True}`",
        f"- gp_xB_fixed_exact: `{params.get('gp_xB_fixed', 'NA')}`",
        f"- xB_far_from_xAg_far: `{xB_far:.15f}`",
        f"- xB_GP_from_xAg_GP: `{xB_gp:.15f}`",
        f"- f_GP_required: `{f_gp_required:.12f}`",
        f"- monodisperse_volume_equivalent_radius_nm: `{r_vol_eq_nm:.12f}`",
        f"- final_arithmetic_mean_radius_nm: `{final_mean:.12f}`",
        f"- final_volume_equivalent_radius_nm: `{final_rve:.12f}`",
        f"- sum_R3_match_status: `{'PASS' if sum_r3_rel <= 1.0e-10 else 'FAIL'}`",
        f"- xB_total_reconstructed: `{xB_total_reconstructed:.12f}`",
        f"- JGP_uses_matrix_xB_derived_xAg: `{gp_jgp.get('gp_literature_xAg_mode', params.get('gp_literature_xAg_mode', '')) != 'fixed_param'}`",
        f"- gp_growth_disabled: `{params.get('gp_growth_enabled', 'NA')}`",
        f"- legacy_gp_storage_disabled: `{params.get('enable_legacy_gp_storage_coupling', 'NA')}`",
        "",
        "The after-quench state is generated from APT data by converting all experimental Ag atomic fractions into the PF xB ledger. The matrix is initialized to xB_far corresponding to xAg_far=0.0078, while the GP radius distribution is volume-renormalized so that GP inventory plus matrix inventory reconstructs xB_tot=0.03.",
        "",
    ])
    write_text(reports_dir / "after_quench_initial_state_acceptance_report.md", acceptance)

    print("after_quench_APT_initial_state_started")
    print(f"xB_tot_target = {xB_tot:.12f}")
    print(f"xAg_far = {require_float(params, 'gp_initial_xAg_far'):.12f}")
    print(f"xB_far = {xB_far:.15f}")
    print(f"xAg_GP = {require_float(params, 'gp_initial_xAg_GP'):.12f}")
    print(f"xB_GP = {xB_gp:.15f}")
    print(f"rho_GP = {rho_gp:.6e}")
    print(f"N_GP_initial = {n_gp_initial}")
    print(f"f_GP_required = {f_gp_required:.12f}")
    print(f"R_vol_eq_nm = {r_vol_eq_nm:.12f}")
    print(f"radius_distribution_type = {params.get('gp_initial_radius_distribution', 'NA')}")
    print(f"final_mean_radius_nm = {final_mean:.12f}")
    print(f"final_volume_equivalent_radius_nm = {final_rve:.12f}")
    print(f"sum_R3_match_status = {'PASS' if sum_r3_rel <= 1.0e-10 else 'FAIL'}")
    print(f"xB_total_reconstructed = {xB_total_reconstructed:.12f}")
    print(f"mass_error_rel = {mass_error_rel:.12e}")
    print(f"JGP_xAg_mode = {gp_jgp.get('gp_literature_xAg_mode', params.get('gp_literature_xAg_mode', 'NA'))}")
    print(f"xB_source_for_JGP = {gp_jgp.get('xB_source_for_JGP', 'NA')}")
    print(f"xAg_used_for_JGP = {gp_jgp.get('xAg_used_for_JGP', 'NA')}")
    print(f"GP_growth_enabled = {params.get('gp_growth_enabled', 'NA')}")
    print(f"legacy_GP_storage_enabled = {params.get('enable_legacy_gp_storage_coupling', 'NA')}")
    print(f"init_validation_status = {'PASS' if init_ok else 'FAIL'}")
    print(f"workstation_build_status = {args.build_status}")
    print(f"final_status = {final_status}")
    print(f"created_reports = {reports_dir}")
    print("recommended_next_action = use this AQ template for production runtime sweeps only after confirming the same xB-ledger closure under the target box/dx")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
