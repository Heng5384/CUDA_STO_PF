#!/usr/bin/env python3
from __future__ import annotations

import csv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "params/beta_capacity_gated_handoff/T380_S05_staged_handoff_diag_20.params"
OUT = ROOT / "params/diagnostic_rsmd_T380_refined_required_supply_map"

XB_FAR = 0.0078305391025
XB_CRIT = 0.011191599269189258
R_EFF = 5.358726490447833


def tag_float(v: float) -> str:
    return f"{v:.12g}".replace(".", "p").replace("-", "m")


def selected_for_stage(xb: float, r_exchange: float, chi: float, stage: str) -> tuple[int, str]:
    coarse_xb = {0.0095, 0.0100, 0.0105, XB_CRIT, 0.0120}
    coarse_r = {6.0, 8.0, 10.0}
    if any(abs(xb - v) < 1e-14 for v in coarse_xb) and r_exchange in coarse_r and abs(chi - 1.0) < 1e-14:
        return 1, "coarse_to_refine_stage1"
    if stage == "expansion":
        high_supply = (
            (abs(xb - 0.0120) < 1e-14 and r_exchange in {10.0, 12.0} and chi in {3.0})
            or (abs(xb - 0.0120) < 1e-14 and r_exchange == 12.0 and abs(chi - 1.0) < 1e-14)
            or (abs(xb - 0.0130) < 1e-14 and r_exchange in {10.0, 12.0} and chi in {1.0, 3.0})
        )
        if high_supply:
            return 1, "high_supply_expansion_stage2"
        return 0, "skipped_expansion_compute_budget"
    return 1, "full_grid_requested"


def write_case(name: str, enabled: int, target: float, r_exchange: float,
               chi: float, kernel_dx: float, nsteps: int) -> Path:
    text = BASE.read_text()
    overrides = f"""

# refined diagnostic RSMD required-supply map overrides
resolved_handoff_xB_write_mode=preserve_profile_xB_alpha_in_support
scheduled_nuc_source_lambda_nm=0.6
scheduled_nuc_target_lambda_nm=0.6
scheduled_nuc_scale_interface_width=1.0
scheduled_nuc_scale_xB_profile_width=1.0
diagnostic_rsmd_enabled={enabled}
diagnostic_rsmd_T_only=380
diagnostic_rsmd_xB_halo_target={target:.15g}
diagnostic_rsmd_R_exchange_nm={r_exchange:.15g}
diagnostic_rsmd_chi_rel={chi:.15g}
diagnostic_rsmd_kernel_radius_dx={kernel_dx:.15g}
diagnostic_rsmd_h_src_max=0.1
diagnostic_rsmd_f_max_per_step=0.05
diagnostic_rsmd_release_window_steps={nsteps}
diagnostic_rsmd_seed_R_eff_h_nm={R_EFF:.15g}
diagnostic_rsmd_provenance=required_supply_diagnostic
y_update_mass_projection_enabled=1
y_update_mass_projection_report_enabled=1
"""
    path = OUT / f"{name}.params"
    path.write_text(text.rstrip() + overrides)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    stage = os.environ.get("DIAGNOSTIC_RSMD_T380_STAGE", "coarse").strip().lower()
    if stage not in {"coarse", "expansion", "full"}:
        raise SystemExit(f"unsupported DIAGNOSTIC_RSMD_T380_STAGE={stage!r}")
    kernel = 1.5
    # The handoff in this controlled path occurs near step 40, so 1040 total
    # steps gives approximately 1000 post-handoff steps.
    nsteps = 1040
    cases = []

    baseline = write_case("T380_refined_rsmd_disabled_baseline_1000post", 0, XB_FAR, 8.0, 1.0, kernel, nsteps)
    cases.append({
        "case": baseline.stem,
        "param_file": str(baseline.relative_to(ROOT)),
        "diagnostic_rsmd_enabled": 0,
        "xB_halo_target": XB_FAR,
        "R_exchange_nm": 8.0,
        "chi_rel": 1.0,
        "kernel_radius_dx": kernel,
        "nsteps": nsteps,
        "run_selected": 1,
        "selection_reason": "no_source_control_1000post",
    })

    xbs = [0.0085, 0.0090, 0.0095, 0.0100, 0.0105, XB_CRIT, 0.0120, 0.0130]
    radii = [4.0, 6.0, 8.0, 10.0, 12.0]
    chis = [0.3, 1.0, 3.0]
    for target in xbs:
        for r_exchange in radii:
            for chi in chis:
                run_selected, reason = selected_for_stage(target, r_exchange, chi, stage)
                name = (
                    f"T380_refined_rsmd_xB{tag_float(target)}_R{tag_float(r_exchange)}_"
                    f"chi{tag_float(chi)}_k{tag_float(kernel)}_1000post"
                )
                path = write_case(name, 1, target, r_exchange, chi, kernel, nsteps)
                cases.append({
                    "case": name,
                    "param_file": str(path.relative_to(ROOT)),
                    "diagnostic_rsmd_enabled": 1,
                    "xB_halo_target": target,
                    "R_exchange_nm": r_exchange,
                    "chi_rel": chi,
                    "kernel_radius_dx": kernel,
                    "nsteps": nsteps,
                    "run_selected": run_selected,
                    "selection_reason": reason,
                })

    manifest = OUT / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        fieldnames = list(cases[0].keys())
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(cases)
    selected = sum(1 for row in cases if row["run_selected"])
    print(f"wrote {len(cases)} refined diagnostic T380 cases to {OUT}; selected={selected}")


if __name__ == "__main__":
    main()
