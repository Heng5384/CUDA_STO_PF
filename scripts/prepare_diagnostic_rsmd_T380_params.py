#!/usr/bin/env python3
from pathlib import Path
import csv


ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT / "params/beta_capacity_gated_handoff/T380_S05_staged_handoff_diag_20.params"
OUT = ROOT / "params/diagnostic_rsmd_source_engine_T380"

XB_FAR = 0.0078305391025
XB_CRIT = 0.011191599269189258
R_EFF = 5.358726490447833


def tag_float(v: float) -> str:
    return f"{v:.12g}".replace(".", "p").replace("-", "m")


def write_case(name: str, enabled: int, target: float, r_exchange: float,
               chi: float, kernel_dx: float, nsteps: int) -> Path:
    text = BASE.read_text()
    overrides = f"""

# diagnostic RSMD source engine overrides
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
    cases = []
    nsteps = 120
    baseline = write_case("T380_rsmd_disabled_baseline", 0, XB_FAR, 8.0, 1.0, 1.5, nsteps)
    cases.append({
        "case": "T380_rsmd_disabled_baseline",
        "param_file": str(baseline.relative_to(ROOT)),
        "diagnostic_rsmd_enabled": 0,
        "xB_halo_target": XB_FAR,
        "R_exchange_nm": 8.0,
        "chi_rel": 1.0,
        "kernel_radius_dx": 1.5,
        "nsteps": nsteps,
    })
    for target in [0.010, XB_CRIT, 0.013, 0.016]:
        for r_exchange in [4.0, 8.0, 12.0]:
            chi = 1.0
            kernel = 1.5
            name = (
                f"T380_rsmd_xB{tag_float(target)}_R{tag_float(r_exchange)}_"
                f"chi{tag_float(chi)}_k{tag_float(kernel)}"
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
            })

    manifest = OUT / "manifest.csv"
    with manifest.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(cases[0].keys()))
        writer.writeheader()
        writer.writerows(cases)
    print(f"wrote {len(cases)} diagnostic RSMD T380 cases to {OUT}")


if __name__ == "__main__":
    main()
