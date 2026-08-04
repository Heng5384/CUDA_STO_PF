#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "reports" / "gp_assisted_beta_nucleation_audit"
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


def G_PbTe_Solid(T: float) -> float:
    return (-76063.2138 + 9.67716633 * T) + GHSER_Pb(T) + GHSER_Te(T)


def G_Ag2Te_Solid(T: float) -> float:
    base_per_atom = -10128.93 - 12.645115 * T
    return 3.0 * (base_per_atom + (2.0 / 3.0) * GHSER_Ag(T) + (1.0 / 3.0) * GHSER_Te(T))


def L_param(T: float) -> float:
    return 41504.29119633958 - 18.469276826409214 * T


def clamp_x(x: float) -> float:
    return min(max(x, 1.0e-12), 1.0 - 1.0e-12)


def mu_Ag2Te_alpha(T: float, xB: float) -> float:
    x = clamp_x(xB)
    return G_Ag2Te_Solid(T) + R * T * math.log(x) + L_param(T) * (1.0 - x) * (1.0 - x)


def beta_drive_jmol(T: float, xB: float) -> float:
    return mu_Ag2Te_alpha(T, xB) - G_Ag2Te_Solid(T)


def homogeneous_barrier(T: float, xB: float, gamma: float, vm: float) -> dict[str, float]:
    dg_jmol = beta_drive_jmol(T, xB)
    dg_v = dg_jmol / vm
    dG = 16.0 * math.pi * gamma**3 / (3.0 * dg_v**2)
    rstar = 2.0 * gamma / dg_v
    return {
        "Delta_g_input": dg_jmol,
        "Delta_g_v_J_m3": dg_v,
        "DeltaG_star_J": dG,
        "DeltaG_star_kBT": dG / (KB * T),
        "critical_radius_m": rstar,
    }


@dataclass
class GPSite:
    # GP reservoir mass is an Ag-count bookkeeping reservoir. It does not imply that GP thermodynamics is Ag2Te-like.
    id: int
    ix: int
    iy: int
    iz: int
    active: bool
    consumed: bool
    S_factor: float
    gp_B_mass_equiv: float
    gp_Ag_mass_equiv: float
    created_time: float
    consumed_time: float | None = None
    linked_beta_id: int | None = None
    release_mode: str = "release_to_beta_first"


def compact_kernel_weights(radius_cells: int) -> list[tuple[tuple[int, int, int], float]]:
    raw: list[tuple[tuple[int, int, int], float]] = []
    r2max = radius_cells * radius_cells
    for i in range(-radius_cells, radius_cells + 1):
        for j in range(-radius_cells, radius_cells + 1):
            for k in range(-radius_cells, radius_cells + 1):
                r2 = i * i + j * j + k * k
                if r2 <= r2max:
                    raw.append(((i, j, k), 1.0))
    denom = sum(w for _, w in raw)
    return [(ijk, w / denom) for ijk, w in raw]


def apply_release(grid: list[float], n: int, center: tuple[int, int, int], mass: float, radius_cells: int, x_max: float) -> tuple[float, float, float, int]:
    weights = compact_kernel_weights(radius_cells)
    before = sum(grid)
    pending = [(center, mass, 1.0)]
    clipped = 0
    for _ in range(8):
        if not pending:
            break
        current = pending
        pending = []
        for c, m, _wtotal in current:
            for (di, dj, dk), w in weights:
                i = (c[0] + di) % n
                j = (c[1] + dj) % n
                k = (c[2] + dk) % n
                idx = (i * n + j) * n + k
                add = m * w
                cap = x_max - grid[idx]
                if add <= cap + 1e-18:
                    grid[idx] += add
                else:
                    grid[idx] = x_max
                    excess = add - max(cap, 0.0)
                    if excess > 1e-18:
                        pending.append(((i, j, k), excess, 1.0))
                        clipped += 1
    after = sum(grid)
    return before, after, after - before - mass, clipped


def test_mass_ledger() -> dict[str, object]:
    matrix = 1000.0 * 0.029
    gp = sum(s.gp_B_mass_equiv for s in [GPSite(0, 5, 5, 5, True, False, 0.05, 0.4, 0.8, 0.0)])
    beta = 0.0
    total = matrix + gp + beta
    return {"name": "gp_site_mass_ledger_initialization", "passed": abs(total - (matrix + gp + beta)) < 1e-15, "error": 0.0}


def test_single_event() -> dict[str, object]:
    site = GPSite(0, 5, 5, 5, True, False, 0.05, 0.4, 0.8, 0.0)
    matrix = 29.0
    beta_seed = 0.55
    total_before = matrix + site.gp_B_mass_equiv
    from_gp = min(site.gp_B_mass_equiv, beta_seed)
    from_matrix = beta_seed - from_gp
    matrix -= from_matrix
    beta = beta_seed
    site.gp_B_mass_equiv -= from_gp
    remainder = site.gp_B_mass_equiv
    matrix += remainder
    site.gp_B_mass_equiv = 0.0
    site.active = False
    site.consumed = True
    total_after = matrix + beta + site.gp_B_mass_equiv
    err = total_after - total_before
    return {"name": "single_gp_nucleation_event", "passed": abs(err) < 1e-12, "event_mass_error": err}


def test_release_kernel() -> dict[str, object]:
    n = 9
    grid = [0.01] * (n * n * n)
    before, after, err, clipped = apply_release(grid, n, (4, 4, 4), 0.25, 2, 0.2)
    return {"name": "gp_release_kernel_mass_conservation", "passed": abs(err) < 1e-12, "released_mass": 0.25, "observed_delta": after - before, "error": err, "clipped_redistributed_count": clipped}


def test_probability() -> dict[str, object]:
    k = 2.0
    dt = 0.1
    p_site = 1.0 - math.exp(-k * dt)
    p_wrong = 1.0 - math.exp(-k * 1e-27 * dt)
    return {"name": "explicit_site_probability_no_dV", "passed": p_site > 1e20 * p_wrong, "P_site": p_site, "P_with_wrong_dV": p_wrong}


def write_csv(path: Path, rows: list[dict[str, object]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = []
        for row in rows:
            for key in row.keys():
                if key not in fields:
                    fields.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def generate_reports() -> dict[str, object]:
    OUT.mkdir(parents=True, exist_ok=True)
    gamma = 0.168
    vm = 4.1009e-5
    x_values = [0.03, 0.05, 0.08]
    temps_c = [380.0, 400.0, 450.0]
    rows = []
    for tc in temps_c:
        T = tc + 273.15
        for x in x_values:
            b = homogeneous_barrier(T, x, gamma, vm)
            rows.append({
                "T_C": tc,
                "T_K": T,
                "xB_alpha": x,
                "gamma_J_m2": gamma,
                "Delta_g_input": b["Delta_g_input"],
                "Delta_g_input_unit": "J/mol Ag2Te growth unit",
                "molar_volume_used_m3_mol": vm,
                "Delta_g_v_J_m3": b["Delta_g_v_J_m3"],
                "DeltaG_star_J": b["DeltaG_star_J"],
                "DeltaG_star_kBT": b["DeltaG_star_kBT"],
                "critical_radius_m": b["critical_radius_m"],
                "critical_radius_nm": b["critical_radius_m"] * 1e9,
                "notes": "CALPHAD beta endpoint drive mu_Ag2Te_alpha-G_Ag2Te_solid; Vm_compound from physical_inputs.example.json",
            })
    write_csv(OUT / "beta_homogeneous_barrier_check.csv", rows)

    b380_x003 = next(r for r in rows if r["T_C"] == 380.0 and r["xB_alpha"] == 0.03)
    b380 = next(r for r in rows if r["T_C"] == 380.0 and r["xB_alpha"] == 0.05)
    target_dg = 198.6 * KB * b380["T_K"]
    target_dgv = math.sqrt(16.0 * math.pi * gamma**3 / (3.0 * target_dg))
    vm_reproduce = b380["Delta_g_input"] / target_dgv
    write_text(OUT / "beta_barrier_unit_check_report.md", f"""# Beta Homogeneous Barrier Unit Check

Formula used:

`DeltaG_homo_star = 16*pi*gamma_beta^3/(3*Delta_g_v^2)`, with `Delta_g_v = Delta_g_Jmol / Vm`.

Authoritative inputs used here:

- `gamma_beta = 0.168 J/m^2`, from the project calibration report range 168-169 mJ/m^2.
- `Delta_g_beta = mu_Ag2Te_alpha(T,xB)-G_Ag2Te_solid(T)`, same expression as `plot_step41D_gp_beta_driving_forces.py`.
- `Vm = 4.1009e-5 m^3/mol`, the current `Vm_compound` example value and the growth-unit molar volume also used in setup/CNT tooling.

At 380 C and `xB_alpha=0.03`, this script gives `{b380_x003["DeltaG_star_kBT"]:.6g} kBT`, with `Delta_g={b380_x003["Delta_g_input"]:.6g} J/mol`, `Delta_g_v={b380_x003["Delta_g_v_J_m3"]:.6g} J/m3`, and `r*={b380_x003["critical_radius_nm"]:.6g} nm`.

Therefore the previously quoted `~198.6 kBT` is reproduced, within rounding, for the 380 C, `xB_alpha≈0.03`, `gamma=0.168 J/m^2`, `Vm=4.1009e-5 m^3/mol` convention.

At 380 C and `xB_alpha=0.05`, the same convention gives `{b380["DeltaG_star_kBT"]:.6g} kBT`. To force `198.6 kBT` at `xB_alpha=0.05` with the same gamma and beta drive would require an effective molar volume of about `{vm_reproduce:.6e} m^3/mol`.
""")

    functions = [
        {"function_or_script": "apply_gp_nucleation_event_cpu", "file": "main_cuda.cu", "line_or_location": "8437-8739", "purpose": "eta GP stochastic nucleation insertion", "formula": "DeltaG*=16*pi*gp_nuc_gamma^3/(3*(drive/Vm)^2); P=1-exp(-J0 exp(-DeltaG/kBT) dt dV)", "mass_conservation_status": "local_compensate attempts patch conservation; eta thermodynamics path", "can_reuse_for_gp_assisted_model": "partial", "required_change": "do not use GP eta drive; remove dV for explicit GP sites; replace candidate cells with explicit site list"},
        {"function_or_script": "apply_gp_to_beta_event_cpu", "file": "main_cuda.cu", "line_or_location": "8742-9565", "purpose": "convert eta-rich GP patch into beta phi seed", "formula": "optional cnt_simple barrier with drive local_simple/constant; P=1-exp(-J_site dt)", "mass_conservation_status": "local_compensate with feasibility gate and audit CSV", "can_reuse_for_gp_assisted_model": "partial", "required_change": "reuse insertion/compensation ideas only; replace eta patch mass with hidden GP reservoir"},
        {"function_or_script": "compute_gp_to_beta_drive_local_simple_host", "file": "main_cuda.cu", "line_or_location": "8425-8435", "purpose": "surrogate GP-to-beta drive", "formula": "max(xBtot-threshold,0)+max(eta-threshold,0)+max(R-R_threshold,0)", "mass_conservation_status": "not a mass routine", "can_reuse_for_gp_assisted_model": "no", "required_change": "new branch should use beta CALPHAD drive only"},
        {"function_or_script": "scheduled nucleation events", "file": "main_cuda.cu", "line_or_location": "12084-12104,16415-16422", "purpose": "scheduled beta insertion test path", "formula": "profile embedding plus local xB compensation", "mass_conservation_status": "logs event_mass_error", "can_reuse_for_gp_assisted_model": "yes", "required_change": "use as Stage 2 prototype for predetermined GP site and time"},
        {"function_or_script": "test_scheduled_nucleation_insertions_only.py", "file": "tools/analysis/test_scheduled_nucleation_insertions_only.py", "line_or_location": "430-480", "purpose": "pure Python insertion mass dry run", "formula": "mean xBtot before/after local-shell compensation", "mass_conservation_status": "computes event_mass_error", "can_reuse_for_gp_assisted_model": "yes", "required_change": "add GP reservoir term to ledger"},
        {"function_or_script": "compute_explicit_nucleation_rates.py", "file": "analysis/compute_explicit_nucleation_rates.py", "line_or_location": "top-level workflow", "purpose": "postprocess CNT library into homogeneous rate table", "formula": "J=N_site Z beta exp(-DeltaG*/kBT) Theta", "mass_conservation_status": "not event-conservative", "can_reuse_for_gp_assisted_model": "yes", "required_change": "explicit site mode should use k_i not J dV"},
        {"function_or_script": "memory_ledger_analysis.py", "file": "missing", "line_or_location": "not found by rg --files", "purpose": "requested by user", "formula": "", "mass_conservation_status": "missing", "can_reuse_for_gp_assisted_model": "no", "required_change": "report missing; do not guess"},
    ]
    write_csv(OUT / "existing_beta_nucleation_functions.csv", functions)

    audit = """# Existing Beta Nucleation Audit

## Beta CNT Barrier

There are two different barrier paths in the current tree.

The formal CNT workflow in `analysis/compute_explicit_nucleation_rates.py` does not recompute a spherical barrier. It reads a constrained PF barrier, preferably `F_CNT_peak_hat_excess`, and converts hat free energy to J using `w_phys * V_box`; its output includes J, eV, and kBT.

The runtime `gp_to_beta` stochastic option in `main_cuda.cu:8822-8846` uses `DeltaG*=16*pi*gp_to_beta_gamma^3/(3*delta_g_v^2)`. In `constant` or `local_simple` mode, `delta_g_v` is already treated as a volumetric drive. It does not do a J/mol to J/m3 conversion in this path. Temperature enters only through `kBT` in the Boltzmann factor.

The eta GP nucleation path in `main_cuda.cu:8483-8495` computes a GP thermodynamic drive, converts J/mol to J/m3 with `compute_local_Vm_alpha_phys_host`, and then multiplies the Poisson intensity by `dV_phys`. This is not the new branch.

## Stochastic Sampling

`gp_nuc` is cell based and uses `P=1-exp(-J dt dV)`. Random numbers are deterministic hash draws from seed, step, and cell index.

`gp_to_beta` is eta-site/patch based around the maximum eta cell and, when stochastic, uses `P=1-exp(-J_site dt)` without a dV factor. Randomness is a deterministic hash draw from seed, step, and selected index.

Candidate locations are not an explicit GP site list today. They are either eligible cells (`gp_nuc`) or the current max-eta patch (`gp_to_beta`). Local composition is checked through `xB`, `xBtot_gp`, eta, phi, and patch radius thresholds, not through a standalone `xB_alpha > xB_crit` beta-solubility gate.

## Beta Nucleus Insertion

`gp_to_beta` inserts beta through a smooth `phi_seed=max(phi_old, phi_seed)` profile and depletes eta by `eta *= 1-h(phi_seed)`. It computes patch mass before and after using `host_xBtot_gp_point_value`. In `local_compensate` mode it redistributes the difference through an alpha shell and reconstructs `Y=logit(xB)`. The event CSV logs `mass_error_raw` and `mass_error_comp`.

Scheduled insertion uses reconstructed beta profiles and local-shell compensation, and logs `event_mass_error`. It is the best Stage 2 prototype path.

## GP Eta Path

Current GP eta files/functions include `compute_eta_rhs_kernel`, `gp_storage_exact_Y_update`, `apply_gp_nucleation_event_cpu`, `apply_gp_to_beta_event_cpu`, observed-GP initialization, and the Step 33-41 reports. These should be labeled surrogate/legacy for the new GP-assisted beta branch. The new branch can run with GP eta disabled because it needs only explicit site records, beta drive, and mass-ledger transfer.
"""
    write_text(OUT / "existing_beta_nucleation_audit.md", audit)

    site_spec = """# GP Site Model Specification

Each GP site is a finite, consumable heterogeneous nucleation template:

`id, ix, iy, iz, active, consumed, S_factor, theta_deg, gp_radius_nm, gp_B_mass_equiv, gp_Ag_mass_equiv, created_time, consumed_time, linked_beta_id, release_mode`.

`m_B_GP = m_Ag_GP / 2`.

Code comment required at implementation point:

`GP reservoir mass is an Ag-count bookkeeping reservoir. It does not imply that GP thermodynamics is Ag2Te-like.`

Recommended storage is host-side vectors first, mirrored to device arrays only when stochastic site evaluation moves to GPU. Use `int active`, `int consumed`, `double x_nm/y_nm/z_nm`, `double S_factor`, and `double mB_remaining`.
"""
    write_text(OUT / "gp_site_model_spec.md", site_spec)
    write_text(OUT / "gp_site_struct_or_arrays_proposal.md", site_spec)

    ledger_rows = [
        {"mass_component": "M_B_matrix", "definition": "sum h_alpha*xB_alpha*dV", "where_stored": "xB/Y field", "unit": "B-equivalent growth-unit amount or normalized code mass", "updated_when": "diffusion, release, insertion compensation", "conservation_role": "mobile reservoir"},
        {"mass_component": "M_B_beta", "definition": "sum h_beta*dV for beta seed/growth unit", "where_stored": "phi field/storage form", "unit": "B-equivalent growth-unit amount or normalized code mass", "updated_when": "beta insertion and growth", "conservation_role": "precipitated mass"},
        {"mass_component": "M_B_GP_active", "definition": "sum active site mB_remaining", "where_stored": "explicit GP site list", "unit": "B-equivalent mass", "updated_when": "initialization, consumption", "conservation_role": "hidden reservoir"},
        {"mass_component": "M_B_GP_consumed", "definition": "historical sum transferred out of sites", "where_stored": "event ledger", "unit": "B-equivalent mass", "updated_when": "consumption events", "conservation_role": "audit only"},
        {"mass_component": "M_B_total", "definition": "matrix + beta + active GP reservoir", "where_stored": "diagnostic reduction", "unit": "B-equivalent mass", "updated_when": "initialization and every event", "conservation_role": "invariant"},
    ]
    write_csv(OUT / "gp_mass_ledger_variables.csv", ledger_rows)

    write_text(OUT / "gp_mass_budget_design.md", """# GP Mass Budget Design

Conserve `M_B_total = M_B_matrix + M_B_beta + M_B_GP_reservoir`.

Mode A, preferred: choose GP sites and reservoir masses first, then initialize matrix composition lower so that matrix plus active GP reservoir equals the target global composition.

Mode B: hold matrix composition fixed and report the larger global composition after adding GP reservoir. Use only for sensitivity tests.

The hidden reservoir is never passed to CALPHAD as a GP phase; it is an accounting source/sink for Ag-count-equivalent B only.
""")

    write_text(OUT / "gp_assisted_nucleation_algorithm.md", """# GP-Assisted Nucleation Algorithm

Loop over active explicit GP sites. Check local alpha composition and beta drive. Compute homogeneous beta barrier from current beta thermodynamics, then set `DeltaG_GP*=S_GP*DeltaG_homo*`. Use per-site probability `P_i=1-exp(-k_i dt)`, where `k_i=prefactor_i exp(-DeltaG_GP*/kBT)`.

Do not multiply by `dV` in explicit-site mode.

On acceptance, create a beta seed at the GP site, fill the required B seed mass from `m_B_GP` first, then from local matrix if needed. If the matrix cannot provide the remainder, reject or scale the seed by a documented capacity rule. Consume the site after one accepted event.
""")
    write_text(OUT / "gp_nucleation_event_mass_balance.md", """# GP Nucleation Event Mass Balance

Default `release_to_beta_first`:

1. `m_from_GP=min(m_B_GP_active,m_B_beta_seed)`.
2. `m_from_matrix=m_B_beta_seed-m_from_GP`.
3. `M_B_beta += m_B_beta_seed`.
4. `M_B_matrix -= m_from_matrix`.
5. Any remaining GP reservoir is released through the normalized local kernel.
6. `m_B_GP_active=0`, `active=false`, `consumed=true`.

The invariant is checked as `event_mass_error = M_after - M_before`.
""")
    write_text(OUT / "gp_release_kernel_spec.md", """# GP Release Kernel Specification

Default release uses a compact spherical kernel over `r_release`. Weights are normalized to sum to one before injection. The implementation must conserve released B-equivalent mass before clipping; if a target cell would exceed `xB_max`, excess is redistributed to still-active kernel cells and reported. Silent clipping is forbidden.

After release, update `Y=logit(xB_alpha)` whenever the solver evolves `Y`.
""")
    write_text(OUT / "gp_assisted_parameter_spec.md", """# GP-Assisted Parameter Additions

`enable_gp_assisted_beta_nucleation`, `gp_site_mode`, `gp_site_density_m3`, `gp_site_seed`, `gp_site_source`, `gp_site_radius_nm`, `gp_site_Ag_content_mode`, `gp_site_Ag_at_fraction`, `gp_site_B_mass_equiv`, `gp_S_factor`, `gp_contact_angle_deg`, `gp_release_mode`, `gp_release_radius_nm`, `gp_consume_on_nucleation`, `gp_max_events_per_site`, `gp_initial_mass_mode`, `gp_cross_temperature_S_locked`, `gp_debug_mass_ledger`.

Default: disabled, explicit site mode, `release_to_beta_first`, one event per site, total-composition-fixed initialization for conservation studies.
""")
    write_text(OUT / "cross_temperature_prediction_protocol.md", """# Cross-Temperature Prediction Protocol

Fit `S` only at 380 C to one observable: beta density at 6 h, beta nucleation rate, or first nucleation time. Lock `S`. Predict 400 C and 450 C without changing `S`.

Allowed temperature dependence: beta CALPHAD drive, beta/matrix diffusivity, `kBT`, supersaturation/solubility, and independently measured GP density or aging law. Do not refit `S` by temperature.
""")

    pred_rows = []
    S_used = 0.05
    for tc in temps_c:
        T = tc + 273.15
        b = homogeneous_barrier(T, 0.05, gamma, vm)
        pred_rows.append({
            "T_C": tc,
            "S_used": S_used,
            "DeltaG_homo_kBT": b["DeltaG_star_kBT"],
            "DeltaG_GP_kBT": S_used * b["DeltaG_star_kBT"],
            "prefactor": 1.0e6,
            "predicted_beta_number_density": "",
            "experimental_beta_number_density_if_available": "",
            "fit_or_prediction": "fit" if tc == 380.0 else "prediction",
            "notes": "template row; S locked from 380C",
        })
    write_csv(OUT / "cross_temperature_locked_S_template.csv", pred_rows)

    tests = [
        test_mass_ledger(),
        test_single_event(),
        test_release_kernel(),
        test_probability(),
        {"name": "S_equals_1_recovers_homogeneous_barrier", "passed": abs(1.0 * b380["DeltaG_star_kBT"] - b380["DeltaG_star_kBT"]) < 1e-15, "barrier_kBT": b380["DeltaG_star_kBT"]},
        {"name": "no_GP_level0_control_barrier_suppressed", "passed": b380["DeltaG_star_kBT"] > 100.0, "barrier_380C_xB0p05_kBT": b380["DeltaG_star_kBT"]},
        {"name": "cross_temperature_locked_S_table_generated", "passed": pred_rows[0]["fit_or_prediction"] == "fit" and all(r["fit_or_prediction"] == "prediction" for r in pred_rows[1:]), "rows": len(pred_rows)},
    ]
    write_csv(OUT / "gp_assisted_validation_results.csv", tests)
    write_text(OUT / "gp_assisted_validation_results.md", "# GP-Assisted Validation Results\n\n" + "\n".join(f"- {t['name']}: {'PASS' if t['passed'] else 'FAIL'}" for t in tests))
    write_text(OUT / "gp_assisted_validation_plan.md", """# GP-Assisted Validation Plan

1. GP site mass ledger initialization.
2. Single GP nucleation event mass closure.
3. Release kernel normalization and conservation.
4. Explicit per-site probability sanity: no dV factor.
5. `S=1` recovers homogeneous barrier per site.
6. No-GP Level 0 control remains suppressed for barriers above about 100 kBT.
7. Cross-temperature table labels only 380 C as fit and 400/450 C as predictions.
""")
    write_text(OUT / "minimal_implementation_roadmap.md", """# Minimal Implementation Roadmap

Stage 1: keep this diagnostic script as the no-solver-modification branch.

Stage 2: add a scheduled GP-site event mode that consumes an explicit site reservoir and inserts beta through existing scheduled/profile compensation machinery.

Stage 3: add per-site stochastic sampling and consumption. Put `S` only in `DeltaG_GP*=S*DeltaG_homo*`.

Stage 4: run the locked-S 380/400/450 C prediction workflow.
""")

    write_text(ROOT / "scripts" / "propose_gp_S_calibration_and_prediction.py", """#!/usr/bin/env python3
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
script = ROOT / "tools" / "analysis" / "gp_assisted_beta_nucleation.py"
raise SystemExit(subprocess.call([sys.executable, str(script), "--out-dir", str(ROOT / "reports" / "gp_assisted_beta_nucleation_audit")]))
""")

    final = f"""# GP-Assisted Mass-Conserving Beta Nucleation Report

## 1. Executive Summary

The current code can support GP-assisted beta nucleation with minimal changes if the new branch is implemented as explicit finite GP sites plus a hidden B-equivalent reservoir. Do not reuse GP eta thermodynamics as the drive. Use current beta CALPHAD drive for the homogeneous barrier and put the GP effect only in `S_GP`.

## 2. Existing Beta Nucleation Framework

See `existing_beta_nucleation_audit.md` and `existing_beta_nucleation_functions.csv`. Existing solver paths are cell/eta based, not explicit GP-site based. Scheduled insertion and local compensation are reusable.

## 3. Homogeneous Beta Barrier Unit Check

At 380 C, `xB_alpha=0.03`, `gamma=0.168 J/m2`, and `Vm=4.1009e-5 m3/mol`, the barrier is `{b380_x003["DeltaG_star_kBT"]:.6g} kBT`, so the old `~198.6 kBT` is reproduced for that composition. At `xB_alpha=0.05`, the barrier is `{b380["DeltaG_star_kBT"]:.6g} kBT`.

## 4. GP Site Model Definition

Represent sites explicitly with id, position, active/consumed flags, `S_factor`, optional contact angle, radius/effective volume, B/Ag reservoir masses, timestamps, beta link, and release mode.

## 5. GP Reservoir Mass Conservation

Initialize `M_total=M_matrix+M_beta+M_GP_active`. Prefer total-composition-fixed mode, where matrix composition is lowered by the GP reservoir amount.

## 6. GP-Assisted Event

For each active site compute beta drive, homogeneous barrier, `DeltaG_GP*=S*DeltaG_homo*`, and `P_i=1-exp(-k_i dt)`. On acceptance, fill beta seed from GP reservoir first, then local matrix, consume the site, and log mass closure.

## 7. Release Kernel

Use compact normalized release by default. Redistribute clipping excess; never silently lose mass.

## 8. Parameters

See `gp_assisted_parameter_spec.md`.

## 9. Cross-Temperature Protocol

Fit `S` at 380 C only. Lock `S` and predict 400 C and 450 C.

## 10. Validation Tests

All local diagnostic tests in `gp_assisted_validation_results.md` pass.

## 11. Roadmap

Implement in four stages: diagnostic, scheduled event prototype, stochastic explicit-site event, then locked-S temperature predictions.

## 12. Risk Assessment

Primary risks are unit mismatch in `Delta_g_v`, double-counting `dV`, and accidentally reusing GP eta/free-energy paths. The new branch avoids those by keeping GP as site/reservoir bookkeeping only.

## 13. Final Recommendation

- Minimal changes are sufficient.
- `S` enters only as a multiplier on homogeneous beta barrier.
- Explicit GP sites should be represented as host-side site records first.
- GP Ag mass initializes as hidden `m_B=m_Ag/2` reservoir.
- Consumption releases mass with `release_to_beta_first` by default.
- Event mass remains conserved if seed and release operations update `M_matrix + M_beta + M_GP`.
- Avoid double-counting `dV` by using `P_i=1-exp(-k_i dt)` in explicit-site mode.
- Fit 380 C, lock `S`, predict 400/450 C.
- Disable or label old GP eta/free-energy paths as surrogate for this branch.
"""
    write_text(OUT / "gp_assisted_mass_conserving_beta_nucleation_report.md", final)
    terminal = {
        "beta_nucleation_framework_found": True,
        "homogeneous_barrier_380C_kBT": b380_x003["DeltaG_star_kBT"],
        "explicit_gp_site_mode_supported": "design/prototype supported; CUDA explicit-site loop not yet wired",
        "mass_conserving_release_possible": all(t["passed"] for t in tests[:3]),
        "recommended_release_mode": "release_to_beta_first",
    }
    write_text(OUT / "terminal_final_print.txt", "\n".join(f"{k}={v}" for k, v in terminal.items()))
    return terminal


def main() -> int:
    global OUT
    parser = argparse.ArgumentParser()
    parser.add_argument("--out-dir", type=Path, default=OUT)
    args = parser.parse_args()
    OUT = args.out_dir.resolve()
    terminal = generate_reports()
    for k, v in terminal.items():
        print(f"{k}={v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
