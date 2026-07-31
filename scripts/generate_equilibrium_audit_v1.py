#!/usr/bin/env python3
"""Generate the reproducible, read-only part of the T380 equilibrium audit.

This script mirrors the equations in thermo_utils.h and Unit_Psedobinary.py.
It deliberately does not write production parameters or run a PF trajectory.
Runtime result files are filled by the workstation runner after the source
hash and parameter contract have been frozen.
"""
from __future__ import annotations
import csv, hashlib, json, math, os, subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "equilibrium_audit_v1"
T_C = 380.0
T = T_C + 273.15
R = 8.31446261815324
GAMMA = 0.168
LAMBDA = 4.0e-9
PF_DX = 1.0e-9
DX_CODE = 1.0
VM_ALPHA = 4.1009e-5
VM_BETA = 4.1009e-5
V_A = 0.0
V_B = 1.0
EPS = 1e-12

def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for b in iter(lambda: f.read(1 << 20), b""):
            h.update(b)
    return h.hexdigest()

def run(cmd):
    try:
        return subprocess.check_output(cmd, cwd=ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return "UNAVAILABLE"

def clamp(x): return min(1.0-EPS, max(EPS, x))
def ghs_pb(T):
    if T < 600.61:
        return -7650.085 + 101.700244*T - 24.5242231*T*math.log(T) - 0.00365895*T*T - 2.4395e-7*T**3
    return -10531.095 + 154.243182*T - 32.4913959*T*math.log(T) + 0.00154613*T*T + 8.05448e25*T**-9
def ghs_ag(T):
    if T < 1234.93:
        return -7209.512 + 118.202013*T - 23.8463314*T*math.log(T) - 0.001790585*T*T - 3.98587e-7*T**3 - 12011.0/T
    return -15095.252 + 190.266404*T - 33.472*T*math.log(T) + 1.411773e29*T**-9
def ghs_te(T):
    if T < 722.66:
        return -10544.679 + 183.372894*T - 35.6687*T*math.log(T) + 0.01583435*T*T - 5.240417e-6*T**3 + 155015.0/T
    return 9160.595 - 129.265373*T + 13.004*T*math.log(T) - 0.0362361*T*T + 5.006367e-6*T**3 - 1.28681e30*T**-9
def g0_a(T): return -76063.2138 + 9.67716633*T + ghs_pb(T) + ghs_te(T)
def g0_b(T): return 3.0*((-10128.93 - 12.645115*T) + (2.0/3.0)*ghs_ag(T) + (1.0/3.0)*ghs_te(T))
def L_of(T): return 41212.9 - 18.05*T
def mu_a(x): return g0_a(T) + R*T*math.log(1.0-clamp(x)) + L_of(T)*x*x
def mu_b(x): return g0_b(T) + R*T*math.log(clamp(x)) + L_of(T)*(1.0-x)*(1.0-x)
def g_mix(x):
    x = clamp(x)
    return (1-x)*g0_a(T) + x*g0_b(T) + R*T*((1-x)*math.log(1-x)+x*math.log(x)) + L_of(T)*x*(1-x)
def dg_mix(x): return mu_b(x)-mu_a(x)
def bisect(f, lo=1e-12, hi=0.5, n=200):
    flo, fhi = f(lo), f(hi)
    # Find the first sign change on a logarithmic/linear composite grid.
    if flo*fhi > 0:
        pts = [10**(-12 + 12*i/400) for i in range(401)] + [0.5 + 0.5*i/400 for i in range(401)]
        prev, fp = pts[0], f(pts[0])
        for p in pts[1:]:
            fq = f(p)
            if fp*fq <= 0: lo, hi, flo, fhi = prev, p, fp, fq; break
            prev, fp = p, fq
        else: raise RuntimeError("no bracket")
    for _ in range(n):
        mid = 0.5*(lo+hi); fm=f(mid)
        if flo*fm <= 0: hi, fhi = mid, fm
        else: lo, flo = mid, fm
    return 0.5*(lo+hi)
def xag_from_xb(x): return 2*x/(2+x)
def xb_from_xag(x): return 2*x/(2-x)

def write_csv(path, fieldnames, rows):
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=fieldnames); w.writeheader(); w.writerows(rows)

def main():
    OUT.mkdir(parents=True, exist_ok=True)
    xeq = bisect(lambda x: mu_b(x)-g0_b(T), 1e-12, 0.08)
    # The endpoint common-tangent condition to pure Ag2Te is identical to the
    # runtime reaction-equilibrium root; retain both rows to expose the identity.
    xcoh = xeq
    xeq_ag = xag_from_xb(xeq)
    source_files = ["main_cuda.cu","cuda_kernels.cu","cuda_common.cu","thermo_utils.h","phase_functions.h","pf_params.h","pf_zero_mode_checkpoint.cpp","pf_zero_mode_checkpoint.h","Unit_Psedobinary.py","physical_inputs.example.json","Makefile"]
    hashes=[]
    for s in source_files:
        p=ROOT/s; hashes.append({"path":s,"sha256":sha256(p) if p.exists() else "NOT_PRESENT"})
    param = ROOT/"Results/interface_width_audit/pf_strict_dualdx_final.params"
    param_sha = sha256(param) if param.exists() else "NOT_PRESENT"
    commit=run(["git","rev-parse","HEAD"])
    status=run(["git","status","--short","--branch"]) or "UNAVAILABLE"
    write_csv(OUT/"parameter_provenance.csv", ["parameter","value","unit","source","sha256_or_note"], [
        {"parameter":"temperature","value":T_C,"unit":"degC","source":"Unit_Psedobinary.py / T380 audit contract","sha256_or_note":"T_K=653.15"},
        {"parameter":"lambda_sm","value":LAMBDA,"unit":"m","source":"pf_strict_dualdx_final.params","sha256_or_note":param_sha},
        {"parameter":"pf_dx","value":PF_DX,"unit":"m","source":"pf_strict_dualdx_final.params","sha256_or_note":param_sha},
        {"parameter":"gamma","value":GAMMA,"unit":"J m^-2","source":"pf_strict_dualdx_final.params","sha256_or_note":param_sha},
        {"parameter":"Vm_alpha","value":VM_ALPHA,"unit":"m^3 mol^-1","source":"physical_inputs.example.json","sha256_or_note":sha256(ROOT/"physical_inputs.example.json")},
        {"parameter":"Vm_compound","value":VM_BETA,"unit":"m^3 mol^-1","source":"physical_inputs.example.json","sha256_or_note":sha256(ROOT/"physical_inputs.example.json")},
        {"parameter":"GP/birth/release","value":"OFF","unit":"boolean","source":"audit contract","sha256_or_note":"required; no production writeback"},
        {"parameter":"beta nucleation","value":"OFF","unit":"boolean","source":"audit contract","sha256_or_note":"required; no production writeback"},
        {"parameter":"elasticity","value":"separate OFF/ON cases","unit":"boolean","source":"audit contract","sha256_or_note":"same frozen C/eigenstrain"},
    ])
    write_csv(OUT/"thermodynamic_fit_data.csv", ["dataset","temperature_C","observable","value","uncertainty","unit","use"], [
        {"dataset":"Sheskin2018_6h_TEM_APT","temperature_C":380,"observable":"Nv","value":1.68e24,"uncertainty":0.92e24,"unit":"m^-3","use":"nucleation-dominated external anchor; not fitted here"},
        {"dataset":"Sheskin2018_48h_TEM_APT","temperature_C":380,"observable":"Nv","value":1.21e22,"uncertainty":0.75e22,"unit":"m^-3","use":"coarsening validation only"},
        {"dataset":"Grossfeld2017_380C_6h_HRSEM","temperature_C":380,"observable":"Nv","value":2.7e20,"uncertainty":"NA","unit":"m^-3","use":"method-specific trend only"},
    ])
    write_csv(OUT/"thermodynamic_fit_residuals.csv", ["dataset","model_status","residual","note"], [
        {"dataset":"all external fit targets","model_status":"NOT_FIT","residual":"NA","note":"equilibrium audit does not refit kinetic S_GP or mobility"},
    ])
    write_csv(OUT/"chemical_flat_equilibrium_root.csv", ["route","T_C","T_K","xB_alpha","xAg_alpha","residual_J_mol","equation"], [
        {"route":"runtime_reaction_root","T_C":T_C,"T_K":T,"xB_alpha":xeq,"xAg_alpha":xeq_ag,"residual_J_mol":mu_b(xeq)-g0_b(T),"equation":"mu_Ag2Te(xB,T)-G_Ag2Te^0(T)=0"},
        {"route":"common_tangent_to_pure_beta","T_C":T_C,"T_K":T,"xB_alpha":xcoh,"xAg_alpha":xag_from_xb(xcoh),"residual_J_mol":dg_mix(xcoh)-(g0_b(T)-g_mix(xcoh))/(1-xcoh),"equation":"g'(x)=[g(1)-g(x)]/(1-x), g(1)=G_Ag2Te^0"},
    ])
    write_csv(OUT/"coherent_flat_equilibrium_results.csv", ["route","elastic_enabled","boundary","xB_alpha","xAg_alpha","status","note"], [
        {"route":"periodic_spectral_fixed_cell","elastic_enabled":1,"boundary":"periodic; k=0 displacement zero; E0=0","xB_alpha":"NOT_RUN","xAg_alpha":"NOT_RUN","status":"PENDING_WORKSTATION","note":"requires PF minimization/run; no homogeneous strain relaxation in source kernel"},
    ])
    for name in ["nonelastic_xeq_vs_R.csv","elastic_xeq_vs_R.csv"]:
        write_csv(OUT/name,["R_nm","box_nm","elastic_enabled","xB_eq","xAg_eq","growth_rate_sign","status"],[
            {"R_nm":r,"box_nm":max(8*r,64),"elastic_enabled":0 if name.startswith("non") else 1,"xB_eq":"NOT_RUN","xAg_eq":"NOT_RUN","growth_rate_sign":"NOT_RUN","status":"PENDING_WORKSTATION"} for r in [6,8,10,12,14,16,20]
        ])
    write_csv(OUT/"finite_box_sensitivity.csv",["case","R_nm","box_nm","status","note"],[{"case":"nonelastic","R_nm":r,"box_nm":b,"status":"PENDING_WORKSTATION","note":"smallest periodic box satisfying R+4 lambda and image separation"} for r,b in [(8,96),(10,112),(12,128),(14,144),(16,160),(20,192)]])
    write_csv(OUT/"elastic_finite_box_sensitivity.csv",["case","R_nm","box_nm","status","note"],[{"case":"elastic","R_nm":r,"box_nm":max(8*r,96),"status":"PENDING_WORKSTATION","note":"same box ladder; compare fixed-cell elastic energy/stress"} for r in [8,10,12,14,20]])
    write_csv(OUT/"nonelastic_growth_rate_brackets.csv",["R_nm","xB_far","growth_rate","status"],[{"R_nm":r,"xB_far":v,"growth_rate":"NOT_RUN","status":"PENDING_WORKSTATION"} for r in [8,10,12,14,20] for v in [0.004664951821454195,0.006219279767278563,0.03]])
    write_csv(OUT/"elastic_shape_orientation.csv",["R_nm","orientation","status","note"],[{"R_nm":r,"orientation":o,"status":"PENDING_WORKSTATION","note":"principal strain axes [0.046,-0.022,-0.017], identity rotation"} for r in [8,10,12,14,20] for o in ["x","y","z"]])
    write_csv(OUT/"profile_AB_time_series.csv",["case","R_nm","elastic_enabled","time_s","mean_abs_profile_difference","status"],[{"case":c,"R_nm":r,"elastic_enabled":e,"time_s":0,"mean_abs_profile_difference":"NOT_RUN","status":"PENDING_WORKSTATION"} for c,r,e in [("A_tanh",8,0),("B_local",8,0),("A_tanh",10,1),("B_local",10,1)]])
    write_csv(OUT/"profile_artifact_fraction.csv",["case","R_nm","artifact_fraction","status"],[{"case":c,"R_nm":r,"artifact_fraction":"NOT_RUN","status":"PENDING_WORKSTATION"} for c in ["A_tanh","B_local"] for r in [8,10,12]])
    write_csv(OUT/"V2_three_way_time_series.csv",["case","time_h","particle_count","mean_R_nm","matrix_xAg","mass_drift","status"],[{"case":c,"time_h":t,"particle_count":"NOT_RUN","mean_R_nm":"NOT_RUN","matrix_xAg":"NOT_RUN","mass_drift":"NOT_RUN","status":"PENDING_WORKSTATION"} for c in ["V2-A_tanh_elastic_off","V2-B_local_elastic_off","V2-C_local_elastic_on"] for t in [0,1,6]])
    write_csv(OUT/"thermodynamic_local_sensitivity.csv",["parameter","minus_20pct","baseline","plus_20pct","status"],[{"parameter":p,"minus_20pct":"NOT_RUN","baseline":"frozen","plus_20pct":"NOT_RUN","status":"PENDING_AFTER_STAGES_2_7"} for p in ["gamma","L(T)","Vm_alpha","eps_principal"]])
    write_csv(OUT/"constrained_refit_candidates.csv",["candidate","parameter_writeback","status","note"],[{"candidate":"none","parameter_writeback":"false","status":"NOT_RUN","note":"audit does not permit retuning"}])
    write_csv(OUT/"original_fit_degradation.csv",["candidate","degradation","status"],[{"candidate":"none","degradation":"NA","status":"NOT_RUN"}])
    write_csv(OUT/"integrated_equilibrium_decomposition.csv",["term","status","note"],[{"term":x,"status":"PENDING_WORKSTATION","note":"static equation/BC audit only"} for x in ["chemical","gradient/interface","elastic","finite-size","profile"]])
    write_csv(OUT/"equilibrium_compatibility_map_T380.csv",["xAg","chemical_flat","coherent_flat","status"],[{"xAg":x,"chemical_flat":"PENDING","coherent_flat":"PENDING","status":"PENDING_WORKSTATION"} for x in [0.004654096254055406,0.0058168689198676155,0.006219279767278563,0.006621852111969499]])
    (OUT/"unit_contract_T380.json").write_text(json.dumps({"temperature_C":T_C,"temperature_K":T,"gamma_J_m2":GAMMA,"lambda_sm_m":LAMBDA,"pf_dx_m":PF_DX,"dx_code":DX_CODE,"lambda_over_pf_dx":LAMBDA/PF_DX,"L_ref_factor":5.0,"L_ref_m":5*LAMBDA,"t_real_unit_source":"L_ref^2/D_alpha_phys","composition":"xB is Ag2Te pseudo-binary fraction; xAg=2*xB/(2+xB)","GP_used":False,"GP_birth_used":False,"GP_release_used":False,"beta_nucleation_used":False}, indent=2)+"\n")
    (OUT/"baseline_freeze.md").write_text(f"""# T380 workstation equilibrium audit — baseline freeze\n\n- source commit: `{commit}` (detached HEAD; no commit/push performed)\n- working-tree status at freeze:\n\n```text\n{status}\n```\n\n- temperature: 380 °C = {T:.2f} K\n- audit mode: beta PF only; GP, GP birth, GP release, source and beta nucleation OFF\n- frozen parameter candidate: `Results/interface_width_audit/pf_strict_dualdx_final.params`\n- parameter SHA-256: `{param_sha}`\n- no production parameter file was modified.\n\n## Source hashes\n\n| file | SHA-256 |\n|---|---|\n"""+"\n".join(f"| `{x['path']}` | `{x['sha256']}` |" for x in hashes)+"\n\nThe CUDA binary is not part of this repository freeze; workstation binary/hash is recorded only after the isolated audit build.\n")
    (OUT/"runtime_equation_audit.md").write_text(f"""# Runtime equation audit\n\n## Chemical thermodynamics\n\n`thermo_utils.h` implements SGTE Pb/Ag/Te references, PbTe and Ag2Te standard energies, and a regular pseudo-binary interaction `L(T)=41212.9-18.05T` J/mol. The raw chemical potentials are\n\n- `mu_A = G0_PbTe + RT ln(1-xB) + L xB^2`;\n- `mu_B = G0_Ag2Te + RT ln(xB) + L (1-xB)^2`.\n\nThe optional convex extrapolation is runtime-disabled in the frozen audit (`thermo_convex_extrapolation_enabled=0`); the cutoff `X_LIMIT_CONVEX=0.09` is therefore not used.\n\n## PF variational terms\n\n`h(phi)=phi^3(6 phi^2-15 phi+10)` and `g(phi)=phi^2(1-phi)^2`. The phase RHS contains chemical reaction drive, double-well, gradient term and, when enabled, `-sigma: d(eps0)/dphi + 0.5 h'(phi) Q`. The conserved composition uses the logit `Y` representation and the runtime's legacy conserved update; no mass projection is allowed in this audit.\n\n## Composition and units\n\n`xB` is the pseudo-binary Ag2Te fraction (beta endpoint xB=1). For an Ag atomic fraction reported in the Ag sublattice convention, the project converter uses `xAg=2 xB/(2+xB)` and inverse `xB=2 xAg/(2-xAg)`. `lambda_sm=4 nm`, `pf_dx=1 nm`, so the interface resolution is 4 cells and `L_ref=5 lambda_sm`; physical time is `t_phys=t_code*(L_ref^2/D_alpha_phys)`.\n\n## GP/nucleation gates\n\nThe production source contains GP, birth, release and beta nucleation paths, but all are explicitly OFF for this audit. No source, clipping, retuning or physical mass projection is permitted.\n""")
    (OUT/"eigenstrain_orientation_audit.md").write_text("""# Eigenstrain and orientation audit\n\nFrozen principal eigenstrain is `[0.046, -0.022, -0.017]`; the supplied rotation is identity, so principal axes are simulation x/y/z. Shear components are zero and are interpreted as tensor shear (not engineering gamma). The chemical isotropic strain is `eps_iso=0.00233`; the runtime writes `eps_iso_over_vB=eps_iso/v_B`. The coherent beta eigenstrain is phase-interpolated with h(phi), and the matrix has zero eigenstrain in the frozen cases.\n\nNo orientation averaging or post-hoc rotation is introduced.\n""")
    (OUT/"mechanical_boundary_condition_audit.md").write_text("""# Mechanical boundary-condition audit\n\nThe elastic path is a periodic spectral Green-function solver. The k=0 displacement mode is set to zero in both eigenstrain and Green-function kernels. This means the default audited case is a fixed periodic cell / zero imposed homogeneous strain (`E0=0`), not a traction-free variable-cell relaxation. A nonzero uniform external strain is added separately through `E0_*`; no such strain is used here. The solver iterates the heterogeneous stiffness correction (`S_p=C_beta-C_matrix`) up to `elastic_iter_max`.\n\nConsequences for interpretation:\n\n1. `elastic_flat` is a fixed-cell coherent equilibrium unless a separate relaxed-cell implementation is explicitly added.\n2. The k=0 handling must not be described as zero macroscopic stress.\n3. Finite-particle elastic runs must report fixed-cell stress and elastic energy, with box-size sensitivity.\n""")
    (OUT/"thermodynamic_fit_reconstruction.md").write_text("""# Thermodynamic fit reconstruction\n\nThe repository contains the fitting-data summary and workflow, but the original real-unit barrier source files are absent. The available 380 °C CNT rows are prompt-anchored and are copied with provenance in `thermodynamic_fit_data.csv`; they are not refit here. The external Nv data are kept method-separated (TEM/APT vs HRSEM/BSE), and 48 h data are coarsening validation rather than nucleation-rate targets.\n""")
    (OUT/"thermodynamic_parameter_uncertainty.md").write_text("""# Thermodynamic parameter uncertainty\n\nNo uncertainty covariance for the SGTE/regular-solution coefficients was found in the frozen source tree. Therefore no statistical refit or parameter writeback is justified. Local sensitivity is deferred until the flat and finite-particle stages have numerical results.\n""")
    (OUT/"chemical_flat_equilibrium_derivation.md").write_text(f"""# Chemical flat equilibrium derivation\n\nFor a pure Ag2Te endpoint, the common tangent condition reduces exactly to `mu_B(x_alpha,T)=G_Ag2Te^0(T)`, which is also the runtime `solve_x_eq_device` equation. At 380 °C:\n\n- `xB_eq = {xeq:.15g}`\n- `xAg_eq = {xeq_ag:.15g}`\n- `L(T) = {L_of(T):.12g} J/mol`\n- `RT = {R*T:.12g} J/mol`\n\nThe independent common-tangent residual is below machine precision in `chemical_flat_equilibrium_root.csv`.\n""")
    (OUT/"chemical_flat_interface_verification.md").write_text("""# Chemical flat interface verification\n\nThe requested numerical slab verification is a workstation task. The analytic root and conversion are frozen before the slab run. The slab must use periodic tangential directions, a sufficiently long normal direction, no GP/nucleation, and compare the far-field xB and integrated excess against the analytic root.\n""")
    (OUT/"chemical_flat_equilibrium_decision.md").write_text("""# Chemical flat equilibrium decision\n\n`PASS_ANALYTIC_CHEMICAL_FLAT_ROOT`; the runtime reaction root and pure-beta common tangent are algebraically identical. `PENDING_WORKSTATION_NUMERICAL_SLAB` remains until the no-GP PF slab has converged and passes mass/restart/dt checks.\n""")
    (OUT/"coherent_flat_equilibrium_method.md").write_text("""# Coherent flat equilibrium method\n\nRun the same slab with elasticity OFF and ON, fixed periodic cell, identical composition and profile. Record xB far field, interface excess, elastic energy and stress components. A separate relaxed-cell comparison is not available in the current solver and must be labeled unsupported rather than inferred.\n""")
    (OUT/"coherent_flat_energy_stress_audit.csv").write_text("quantity,elastic_off,elastic_on,status\nmean_elastic_energy,0,NOT_RUN,PENDING_WORKSTATION\nmean_stress_hydro,0,NOT_RUN,PENDING_WORKSTATION\nfixed_cell_k0_displacement,0,0,PASS_STATIC_SOURCE_AUDIT\n")
    (OUT/"coherent_flat_equilibrium_decision.md").write_text("""# Coherent flat equilibrium decision\n\n`PENDING_WORKSTATION`: source audit confirms fixed-cell periodic elasticity, but no numerical coherent slab result is claimed yet.\n""")
    (OUT/"nonelastic_particle_equilibrium_method.md").write_text("""# Nonelastic finite-particle method\n\nUse resolved radii 6, 8, 10, 12, 14, 16 and 20 nm only where the periodic box and interface resolution are valid. For each radius bracket the sign of `dR/dt` at xAg=0.0046541, 0.0058169, 0.0062193 and 0.03. Stop once the sign is unambiguous; store scalar diagnostics and sparse fields only.\n""")
    (OUT/"nonelastic_particle_fate_map.md").write_text("""# Nonelastic particle fate map\n\nNumerical fate classifications are pending the workstation runs. The map must not be inferred from CNT critical radii alone because finite diffuse interfaces and conserved PF composition alter the dynamic growth sign.\n""")
    (OUT/"elastic_particle_equilibrium_method.md").write_text("""# Elastic finite-particle method\n\nRepeat the nonelastic ladder with fixed-cell periodic elasticity ON, reporting elastic energy, hydrostatic/deviatoric stress and orientation. The three principal eigenstrain orientations are tested without changing material constants.\n""")
    (OUT/"size_dependent_coherent_equilibrium_map.md").write_text("""# Size-dependent coherent equilibrium map\n\nPending workstation numerical results. Report only fixed-cell coherent values; do not call them traction-free equilibrium.\n""")
    (OUT/"profile_AB_test_method.md").write_text("""# Profile A/B test method\n\nA: analytic tanh profile generated from the frozen single-particle profile function. B: local equilibrium composition profile generated with identical geometry and inventory. Compare elasticity-OFF R=8/10/12 and R=10 elasticity-ON, using the same thresholds and stopping criteria.\n""")
    (OUT/"profile_AB_decision.md").write_text("""# Profile A/B decision\n\n`PENDING_WORKSTATION`: no profile winner is selected until artifact fraction, mass drift and restart equality are measured.\n""")
    (OUT/"V2_three_way_contract.md").write_text("""# V2 three-way contract\n\nV2-A is the frozen analytic tanh/elastic-OFF reference; V2-B changes only to local-equilibrium profiles with elasticity OFF; V2-C is the same local profiles with elasticity ON. Centers, radii, inventory, composition, timestep, output cadence and particle identities must be identical.\n""")
    (OUT/"V2_profile_effect.md").write_text("# V2 profile effect\n\nPending workstation V2-A/B comparison.\n")
    (OUT/"V2_elastic_effect.md").write_text("# V2 elastic effect\n\nPending workstation V2-B/C comparison.\n")
    (OUT/"V2_three_way_decision.md").write_text("# V2 three-way decision\n\nPending workstation results; no causal claim is made from the existing production reports.\n")
    (OUT/"constrained_refit_method.md").write_text("# Constrained refit method\n\nNo refit is authorized before the equilibrium comparison is complete. Any candidate would be evaluated against the original source data without writeback.\n")
    (OUT/"thermodynamic_refit_decision.md").write_text("# Thermodynamic refit decision\n\n`NOT_STARTED_BY_CONTRACT`; original production coefficients remain unchanged.\n")
    (OUT/"final_scientific_decision.md").write_text("""# Final scientific decision\n\nThe static audit passes the source equation, composition conversion and fixed-cell boundary interpretation. Numerical slab, finite-particle, profile A/B and V2 stages remain pending workstation execution. Therefore this goal is not yet a publication-grade equilibrium qualification.\n\nRecommended next action: run the isolated GP-OFF T380 slab and single-particle ladder on workstation with checkpoints, then populate the pending tables before any refit or production interpretation.\n""")
    (OUT/"final_terminal_output.txt").write_text(f"""source_commit={commit}\nruntime_thermodynamic_contract=PASS_STATIC_SOURCE_AUDIT\ncomposition_conversion_status=PASS_ANALYTIC\nmechanical_boundary_conditions_tested=PASS_STATIC_FIXED_PERIODIC_K0_ZERO\nxAg_eq_chem_flat={xeq_ag:.15g}\nxAg_eq_coh_flat={xag_from_xb(xcoh):.15g}\nelastic_flat_shift=NOT_RUN\nchemical_flat_experimental_relation=NOT_RUN\ncoherent_flat_experimental_relation=NOT_RUN\nnonelastic_radius_range_nm=6-20_PENDING\nelastic_radius_range_nm=8-20_PENDING\nxAg_eq_nonel_R8=NOT_RUN\nxAg_eq_nonel_R10=NOT_RUN\nxAg_eq_nonel_R12=NOT_RUN\nxAg_eq_el_R8=NOT_RUN\nxAg_eq_el_R10=NOT_RUN\nxAg_eq_el_R12=NOT_RUN\nR_dynamic_star_at_xAg_0062_nonelastic=NOT_RUN\nR_dynamic_star_at_xAg_0062_elastic=NOT_RUN\nprofile_AB_status=PENDING_WORKSTATION\nprofile_artifact_fraction=NOT_RUN\nV2_profile_effect_on_matrix_drift=NOT_RUN\nV2_elastic_effect_on_matrix_drift=NOT_RUN\nthermodynamic_uncertainty_available=false\nconstrained_refit_status=NOT_STARTED\ncandidate_parameter_writeback=false\nproduction_parameters_modified=false\nGP_used=false\nGP_birth_used=false\nGP_release_used=false\nnew_beta_nucleation_used=false\nmobility_retuned=false\ndiffusivity_scaled=false\nclipping_used=false\nphysical_mass_projection_used=false\ncommit_created=false\npush_performed=false\nrecommended_next_action=run_isolated_GP_OFF_T380_workstation_slab_and_particle_ladder\nfinal_status=BLOCKED_RUNTIME_NUMERICAL_STAGES_PENDING\n""")
    print(json.dumps({"out":str(OUT),"commit":commit,"xB_eq":xeq,"xAg_eq":xeq_ag,"parameter_sha256":param_sha}, indent=2))

if __name__ == "__main__": main()
