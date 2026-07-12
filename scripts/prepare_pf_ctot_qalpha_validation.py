#!/usr/bin/env python3
"""Prepare PF-only conservative composition validation parameters."""

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "params/pf_only_x_q_validation/T400_PFX_Ds5_eps0p10_dt0p0005.params"
OUT = ROOT / "params/pf_ctot_qalpha_validation"


def replace(text: str, key: str, value: str) -> str:
    prefix = key + "="
    lines = [line for line in text.splitlines() if not line.startswith(prefix)]
    lines.append(prefix + value)
    return "\n".join(lines) + "\n"


def make(mode: str, strategy: str, dt: str, nsteps: int, temperature: int = 400) -> Path:
    text = SOURCE.read_text()
    overrides = {
        "temperature_C": str(temperature),
        "dt": dt,
        "dt_code": dt,
        "nsteps": str(nsteps),
        "out_every": str(nsteps),
        "csv_out_every": "1",
        "pf_composition_mode": mode,
        "pf_conservative_flux_strategy": strategy,
        "pf_conservative_bound_tol": "1e-12",
        "pf_conservative_mass_tol": "1e-10",
        "pf_conservative_beta_support_eps": "1e-10",
        "pf_conservative_max_subcycles": "64",
        "pf_conservative_one_step_replay": "1",
        "pf_y_update_mode": "lagged_rhs",
        "y_update_mass_projection_enabled": "0",
        "y_update_mass_projection_report_enabled": "0",
        "diagnostic_rsmd_enabled": "0",
        "enable_gp_assisted_beta_nucleation": "0",
        "enable_gp_runtime_library_nucleation": "0",
        "enable_runtime_nucleus_library": "0",
        "enable_dynamic_continue_bridge": "0",
        "gp_stochastic_enabled": "0",
        "gp_literature_model_enabled": "0",
        "gp_initial_population_enabled": "0",
        "gp_growth_enabled": "0",
        "gp_radius_evolution_enabled": "0",
        "gp_inventory_growth_enabled": "0",
        "gp_nuc_enabled": "0",
        "gp_to_beta_enabled": "0",
        "beta_debug_force_single_event": "0",
        "beta_staged_conversion_enabled": "0",
        "beta_staged_accumulation_enabled": "0",
        "scheduled_nuc_enabled": "0",
        "enable_legacy_gp_storage_coupling": "0",
        "gp_barrier_only_mode": "1",
        "pf_baseline_control_mode": "full",
        "dynamics_mass_diag_enabled": "1",
        "dynamics_mass_diag_interval": "1",
        "phi_eta_rhs_attribution_diag_enabled": "0",
    }
    for key, value in overrides.items():
        text = replace(text, key, value)
    tag = f"T{temperature}_{mode}_{strategy}_dt{dt.replace('.', 'p')}_n{nsteps}"
    path = OUT / f"{tag}.params"
    path.write_text(text)
    return path


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    rows = []
    for mode in ("ctot_conservative_split", "qalpha_conservative_local_transaction"):
        for strategy in ("pairwise_limited", "pairwise_backward_euler"):
            for dt, nsteps in (("0.002", 250), ("0.001", 500),
                               ("0.0005", 1000), ("0.00025", 2000)):
                path = make(mode, strategy, dt, nsteps)
                rows.append((path.stem, path.relative_to(ROOT), mode, strategy, dt, nsteps))
    manifest = OUT / "case_manifest.csv"
    manifest.write_text(
        "case,param_file,mode,strategy,dt,nsteps\n" +
        "".join(",".join(map(str, row)) + "\n" for row in rows)
    )
    print(manifest)


if __name__ == "__main__":
    main()
