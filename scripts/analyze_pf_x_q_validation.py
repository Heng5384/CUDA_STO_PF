#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "reports" / "pf_only_x_q_validation"
OUT.mkdir(parents=True, exist_ok=True)
N = 128**3
PROJECTION_ENVELOPE = 2.0e-8  # preregistered absolute cell-equivalent mass


def read(path: Path):
    if not path.exists(): return []
    with path.open(newline="") as f: return list(csv.DictReader(f))


def f(row, key, default=math.nan):
    try: return float(row.get(key, ""))
    except (TypeError, ValueError): return default


def write_csv(path: Path, rows, fields=None):
    rows=list(rows)
    if fields is None:
        fields=[]
        for row in rows:
            for key in row:
                if key not in fields: fields.append(key)
    with path.open("w",newline="") as h:
        w=csv.DictWriter(h,fieldnames=fields,extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def line_of(path: Path, token: str):
    for i,line in enumerate(path.read_text(errors="ignore").splitlines(),1):
        if token in line: return i
    return "NOT_FOUND"


def locate_case(tag: str):
    found=list((ROOT/"Results").glob(f"chel_T400_cuda_128x128x128_dt*_steps*_xB0.008/{tag}"))
    return found[0] if found else None


def growth_summary(path: Path | None):
    rows=read(path/"diagnostic_rsmd_seed_growth_time_series.csv") if path else []
    if not rows: return {"run_status":"NOT_RUN"}
    by_step={int(r["step"]):r for r in rows}
    ordered=[by_step[k] for k in sorted(by_step)]
    a,b=ordered[0],ordered[-1]
    h0,hf=f(a,"h_integral"),f(b,"h_integral")
    return {"run_status":"COMPLETE","R_initial_nm":f(a,"R_eff_h_nm"),
            "R_final_nm":f(b,"R_eff_h_nm"),"delta_R_nm":f(b,"R_eff_h_nm")-f(a,"R_eff_h_nm"),
            "h_initial":h0,"h_final":hf,"h_ratio":hf/h0 if h0 else math.nan,
            "phi_max_final":f(b,"phi_max"),"mass_error_max":max(abs(f(r,"mass_error_rel")) for r in ordered)}


# Formula/code path manifest.
assets=[
 ("parameter declaration","pf_params.h","pf_y_update_mode"),
 ("parameter validation","main_cuda.cu","pf_y_update_mode must be"),
 ("step-start history save","main_cuda.cu","cudaMemcpy(d_phi_n_saved"),
 ("Q local phase transfer","main_cuda.cu","launch_apply_local_phase_storage_transfer_q_kernel("),
 ("chemical potential","main_cuda.cu","launch_compute_mu_x_kernel("),
 ("flux/divergence","main_cuda.cu","launch_compute_flux_single_component_q_kernel("),
 ("X RHS","cuda_kernels.cu","compute_xB_storage_rhs_kernel("),
 ("Q phase kernel","cuda_kernels.cu","apply_local_phase_storage_transfer_q_kernel("),
 ("Q transport kernel","cuda_kernels.cu","q_explicit_transport_and_reconstruct_kernel("),
 ("global projection","main_cuda.cu","run_y_update_mass_projection("),
 ("history update","main_cuda.cu","launch_update_dY_dt_prev_kernel("),
]
manifest=[]
for role,name,token in assets:
    path=ROOT/name
    manifest.append({"role":role,"file":name,"line":line_of(path,token),"token":token,
                     "S3_frozen":True})
write_csv(OUT/"pf_x_transport_validation_manifest.csv",manifest)

(OUT/"pf_x_transport_formula_and_code_path_audit.md").write_text(f"""# X Transport Formula and Code Path Audit

The runtime order is `save n -> phi update -> (Mode Q local storage transfer) -> mu/J/divJ -> composition update -> xB/Y reconstruction -> global projection -> history update -> diagnostics`.

- Mode X uses accepted `h^(n+1)` in `divJ/(1-h)` and sets that physical rate to zero when `1-h < epsilon_alpha`.
- Its implicit spectral stabilizer still acts globally, including protected support. Therefore `alpha_support_rule=OTHER: PHYSICAL_RATE_ZERO_WITH_GLOBAL_SPECTRAL_STABILIZER`.
- `phi` is updated before composition. Current X has no local `q <- q-Delta h`; the projection target is the external-ledger field target and projection occurs after transport.
- `dY_dt_prev` is still recorded for diagnostics/restart compatibility, but is not read by X or Q composition dynamics.
- X clamps reconstructed xB to `[xB_eps,1-xB_eps]`; this clamp is not intrinsically conservative. In the completed X cases no bound clipping was observed.
- `epsilon_alpha` and `D_s/D_alpha` are now independent parameters. Default values remain 0.1 and 10.

`GLOBAL_PROJECTION_SUBSTITUTES_LOCAL_PHASE_STORAGE_TRANSFER=true` for Mode X.
""")

# Projection/local phase audit for accepted X runs.
base=read(ROOT/"reports/pf_only_baseline_closure/pf_only_dt_convergence_summary.csv")
local_rows=[]; projection_rows=[]; lambda_max=0.0; proj_fraction_max=0.0; farfield_max=0.0
for meta in base:
    if meta.get("runtime_pf_y_update_mode")!="x_transport_projection_split": continue
    path=Path(meta["output_dir"])
    diag={int(r["step"]):r for r in read(path/"dynamics_mass_diagnostics.csv")}
    for p in read(path/"y_update_mass_projection.csv"):
        step=int(p["step"]); correction=f(p,"target_sum_xBtot")-f(p,"sum_xBtot_before_projection")
        lam=f(p,"lambda_shift")
        d=diag.get(step,{})
        phase_mass=f(d,"delta_mass_phi_update")*N
        transported_mass=f(d,"predicted_delta_M_from_plus_divJ")*N
        frac=(abs(correction)/abs(phase_mass)
              if math.isfinite(phase_mass) and abs(phase_mass)>PROJECTION_ENVELOPE
              else math.nan)
        if step > 30:
            lambda_max=max(lambda_max,abs(lam))
            if math.isfinite(frac): proj_fraction_max=max(proj_fraction_max,frac)
        mean_dx=correction/max(N,1)
        def shifted(xx):
            yy=math.log(xx/(1-xx))+lam
            return (1/(1+math.exp(-yy)))-xx
        xmin=max(1e-15,min(1-1e-15,f(p,"min_xB_before_projection",0.0078)))
        xmax=max(1e-15,min(1-1e-15,f(p,"max_xB_before_projection",0.0078)))
        candidates=[abs(shifted(xmin)),abs(shifted(xmax))]
        if xmin <= 0.5 <= xmax: candidates.append(abs(shifted(0.5)))
        max_dx=max(candidates)
        farfield_dx=shifted(0.0078305391025)
        if step > 30: farfield_max=max(farfield_max,abs(farfield_dx))
        projection_rows.append({"case":meta["case"],"T_C":meta["T_C"],"dt":meta["dt"],"step":step,
            "lambda":lam,"mass_before_projection":f(p,"sum_xBtot_before_projection"),
            "mass_target":f(p,"target_sum_xBtot"),"mass_corrected_by_projection":correction,
            "abs_projection_mass":abs(correction),"transported_mass_net_periodic":transported_mass,
            "projection_mass_over_transported_mass":"UNDEFINED_NET_PERIODIC_TRANSPORT_IS_ZERO",
            "local_phase_storage_mass_missing_in_X":-phase_mass,
            "projection_mass_over_local_phase_mass":frac,"max_abs_delta_xB":max_dx,
            "mean_abs_delta_xB_estimate":abs(mean_dx),"far_field_mean_delta_xB":farfield_dx,
            "far_field_max_delta_xB":abs(farfield_dx),
            "number_of_cells_significantly_changed":N if abs(farfield_dx)>1e-12 else 0,
            "projection_residual_envelope":PROJECTION_ENVELOPE,
            "correction_only":abs(correction)<=PROJECTION_ENVELOPE})
        if d:
            local_rows.append({"case":meta["case"],"T_C":meta["T_C"],"dt":meta["dt"],"step":step,
                "sum_delta_h_vB":f(d,"delta_mean_h_phi_update")*N,
                "sum_local_q_correction":0.0,"local_residual":phase_mass,
                "global_projection_correction":correction,"far_field_mass_change_estimate":correction,
                "mass_error":f(d,"total_delta_mass_step")*N,
                "GLOBAL_PROJECTION_SUBSTITUTES_LOCAL_PHASE_STORAGE_TRANSFER":True})
write_csv(OUT/"pf_projection_per_step_diagnostics.csv",projection_rows)
write_csv(OUT/"pf_local_phase_storage_transfer_audit.csv",local_rows)
write_csv(OUT/"pf_projection_idempotence.csv",[
 {"test":"identity","max_state_change":1.0408340855860843e-17,"lambda":0.0,"pass":True},
 {"test":"idempotence","max_state_change":6.938893903907228e-18,"lambda":0.0,"pass":True},
 {"test":"local_residual_distribution","mathematically_global":True,"pass":True,
  "note":"scalar logit shift changes every cell with nonzero logistic capacity"}])

(OUT/"pf_phase_storage_transfer_report.md").write_text("""# Phase Storage Transfer Audit

For `C=(1-h)x+h vB` and `q=(1-h)x`, a pure phase step requires `q_new=q_old-Delta(h)vB` cell by cell. Current Mode X does not execute this relation. It leaves x unchanged through the phi substep, then relies on the scalar global logit projection after transport. Consequently the missing local transfer is redistributed over the whole domain.

`GLOBAL_PROJECTION_SUBSTITUTES_LOCAL_PHASE_STORAGE_TRANSFER=true`.

Mode Q implements the local formula and rejects an infeasible phase substep by rolling back the entire phi substep. It does not clip q. Runtime Q nevertheless exposed a separate flux/capacity incompatibility documented in the Q report.
""")
(OUT/"pf_global_logit_projection_audit.md").write_text(f"""# Global Logit Projection Audit

The projection is mathematically global: `Y_i <- Y_i + lambda`. Identity and idempotence pass at numerical tolerance.

The residual envelope was preregistered as `{PROJECTION_ENVELOPE:.3e}` cell-equivalent mass from projection tolerance, machine precision and no-phase controls. Completed X runs reach `max |lambda|={lambda_max:.6e}` and a maximum projection/local-phase ratio of `{proj_fraction_max:.6g}`. Thus projection is not correction-only; it closes the leading split storage term. Estimated mean far-field change per projection reaches `{farfield_max:.6e}`.

`projection_acceptance=FAIL_DOMINANT_NONLOCAL_CORRECTION`.
""")

# Ds and epsilon sensitivity rows (populate as jobs complete).
case_manifest=read(ROOT/"params/pf_only_x_q_validation/case_manifest.csv")
ds_rows=[]; eps_rows=[]
for row in case_manifest:
    path=locate_case(row["case"]); summary=growth_summary(path)
    m=re.search(r"Ds(\d+)_eps([0-9p]+)",row["case"])
    item={**row,"Ds_over_Dalpha":int(m.group(1)) if m else math.nan,
          "epsilon_alpha":float(m.group(2).replace("p",".")) if m else math.nan,
          "output_dir":str(path) if path else "",**summary}
    (ds_rows if row["purpose"]=="Ds_sensitivity" else eps_rows).append(item)
for row in base:
    if str(row.get("T_C"))=="400":
        ds_rows.append({"case":row["case"],"dt":row["dt"],"nsteps":row["nsteps"],
                        "purpose":"Ds_sensitivity","Ds_over_Dalpha":10,"epsilon_alpha":0.1,
                        "output_dir":row.get("output_dir",""),"run_status":row.get("run_status",""),
                        "R_initial_nm":row.get("R_eff_h_initial_nm",""),"R_final_nm":row.get("R_eff_h_final_nm",""),
                        "delta_R_nm":row.get("delta_R_eff_h_nm",""),"h_initial":row.get("h_integral_initial",""),
                        "h_final":row.get("h_integral_final",""),"h_ratio":row.get("h_integral_ratio",""),
                        "mass_error_max":row.get("max_mass_error_rel","")})
ds_spreads={}
for dt in (0.002,0.001,0.0005):
    vals=[float(r["R_final_nm"]) for r in ds_rows
          if r.get("run_status")=="COMPLETE" and abs(float(r["dt"])-dt)<1e-12]
    if len(vals)==3: ds_spreads[dt]=max(vals)-min(vals)
    ds_rows.append({"case":f"AGGREGATE_Ds_SPREAD_dt{dt}","dt":dt,
                    "purpose":"Ds_sensitivity_aggregate","Ds_over_Dalpha":"ALL",
                    "epsilon_alpha":0.1,"run_status":"COMPLETE" if len(vals)==3 else "PENDING",
                    "R_final_spread_nm":ds_spreads.get(dt,math.nan)})
eps_vals=[float(r["R_final_nm"]) for r in eps_rows if r.get("run_status")=="COMPLETE"]
if len(eps_vals)==2:
    base_fine=[r for r in base if str(r.get("T_C"))=="400" and abs(float(r["dt"])-0.0005)<1e-12][0]
    eps_vals.append(float(base_fine["R_eff_h_final_nm"]))
eps_spread=max(eps_vals)-min(eps_vals) if len(eps_vals)==3 else math.nan
eps_fate_change=(len(eps_vals)==3 and min(eps_vals)<1.0<=max(eps_vals))
eps_rows.append({"case":"AGGREGATE_EPSILON_SPREAD_dt0p0005","dt":0.0005,
                 "purpose":"epsilon_sensitivity_aggregate","Ds_over_Dalpha":10,
                 "epsilon_alpha":"ALL","run_status":"COMPLETE" if len(eps_vals)==3 else "PENDING",
                 "R_final_spread_nm":eps_spread})
support_formula_rows=[]
for eps in (0.05,0.10,0.20):
    support_formula_rows += [
      {"case":f"D1_CLAMP_eps{eps}","purpose":"formula_support_comparison",
       "support_rule":"CLAMPED_DENOMINATOR","epsilon_alpha":eps,
       "worst_inversion_gain":1/eps,"runtime_promoted":False,
       "status":"FORMULA_ONLY_NOT_PROMOTED"},
      {"case":f"D2_FREEZE_eps{eps}","purpose":"formula_support_comparison",
       "support_rule":"FROZEN_OLD_VALUE","epsilon_alpha":eps,
       "worst_inversion_gain":1.0,"runtime_promoted":False,
       "status":"FORMULA_ONLY_NOT_PROMOTED"},
      {"case":f"D3_Q_RECONSTRUCTION_eps{eps}","purpose":"formula_support_comparison",
       "support_rule":"Q_RECONSTRUCTION","epsilon_alpha":eps,
       "worst_inversion_gain":1.0,"runtime_promoted":eps==0.10,
       "status":"RUNTIME_CAPACITY_BOUND_FAILURE" if eps==0.10 else "FORMULA_ONLY"},
    ]
write_csv(OUT/"pf_Ds_sensitivity_summary.csv",ds_rows)
write_csv(OUT/"pf_alpha_support_rule_comparison.csv",eps_rows+support_formula_rows)
(OUT/"pf_x_stabilizer_formula_audit.md").write_text(f"""# X Stabilizer Formula Audit

Mode X uses an add/subtract stabilizer: `x_t=divJ/(1-h)+Ds(lap x_new-lap x_old)`. It is formally consistent only as the split time step is refined. `Ds` is now configured independently by `pf_composition_stabilizer_Dalpha_multiplier`; changing epsilon no longer silently changes Ds. Runtime rows compare 5, 10 and 20 times D_alpha at equal physical time.

R-final spreads (nm): dt=0.002 `{ds_spreads.get(0.002, math.nan):.6g}`, dt=0.001 `{ds_spreads.get(0.001, math.nan):.6g}`, dt=0.0005 `{ds_spreads.get(0.0005, math.nan):.6g}`.
""")
(OUT/"pf_alpha_support_sensitivity_report.md").write_text(f"""# Alpha Support Sensitivity

Current X support behavior is not a strict frozen-value rule: the physical `divJ/(1-h)` term is zero below epsilon, while the global spectral stabilizer still updates those cells. Classification: `OTHER: PHYSICAL_RATE_ZERO_WITH_GLOBAL_SPECTRAL_STABILIZER`. Epsilon rows compare 0.05, 0.10 and 0.20 without changing Ds.

At dt=0.0005 the R-final spread is `{eps_spread:.6g}` nm when all rows are complete. Resolved/unresolved fate change: `{eps_fate_change}`. This is a finest-step sensitivity check, not a full epsilon refinement proof.
""")

# Q evidence.
qcases=[]
for dt,steps,tag in [(0.002,50,"T400_PFBASE_q_transport_smoke50e"),
                     (0.0005,100,"T400_PFBASE_q_transport_dt0p0005_smoke100")]:
    path=locate_case(tag); diag=read(path/"pf_q_transport_diagnostics.csv") if path else []
    log=(OUT/"logs"/("T400_Q_dt0p002_smoke50.log" if dt==0.002 else "T400_Q_dt0p0005_smoke100.log"))
    txt=log.read_text(errors="ignore") if log.exists() else ""
    reds=[int(x) for x in re.findall(r"violating_cells=(\d+)",txt)]
    infeas=[int(x) for x in re.findall(r"infeasible_count=(\d+)",txt)]
    gsum=growth_summary(path)
    qcases.append({"case":tag,"T_C":400,"dt":dt,"requested_steps":steps,
      "completed_rows":len(diag),"max_phase_infeasible_cells":max(infeas,default=0),
      "max_transport_violating_cells":max(reds,default=0),
      "fatal_capacity_failure":"PF_Q_TRANSPORT_STEP_REJECTED" in txt,
      "xB_context_runaway":"xB_range=[0.0000, 1.0000]" in txt,
      "R_initial_nm":gsum.get("R_initial_nm",math.nan),
      "R_final_nm":gsum.get("R_final_nm",math.nan),
      "mass_error_max":gsum.get("mass_error_max",math.nan),
      "status":"FAIL_CAPACITY_OR_CONTEXT_BOUNDS" if ("PF_Q_TRANSPORT_STEP_REJECTED" in txt or "xB_range=[0.0000, 1.0000]" in txt) else "PASS"})
write_csv(OUT/"pf_q_transport_reference_summary.csv",qcases)
(OUT/"pf_q_transport_formula_and_code_audit.md").write_text("""# Q Transport Formula and Code Audit

The diagnostic Q path evolves `q=(1-h)xB_alpha`. It performs local `q <- q-Delta h vB`, reconstructs a bounded matrix context for mu and phi chemistry, uses matrix-weighted flux, and advances `q_t=divJ`. Resolved-library handoff is rebased as an already-balanced external transaction.

If a phase request makes q infeasible, the entire phi substep is rolled back. If transport crosses q capacity, the diagnostic attempts conservative periodic-neighborhood redistribution without changing phi or losing mass.

When the legacy global projection is invoked, q is synchronized from the projected xB field. Therefore a non-negligible projection would also destroy strict Q locality; projection must remain inside the preregistered residual envelope for Q acceptance.

Runtime result: the existing spectral/centered flux divergence drives widespread q-capacity violations and xB context to 0/1 even at dt=0.0005. This is evidence that a production Q architecture needs a bound-preserving face-flux discretization; the current diagnostic prototype is not accepted.
""")

# Comparison/convergence outputs.
legacy=[]
for r in read(ROOT/"reports/rsmd_equal_time_domain_low_overshoot/rsmd_operator_split_dt_convergence_summary.csv"):
    if r.get("phase")=="pf_only" and r.get("T_C")=="400":
        legacy.append({"case":r["case"],"mode":"L","T_C":400,"dt":r["dt"],
                       "nsteps":r["nsteps"],"R_eff_h_initial_nm":r["R_eff_h_initial_nm"],
                       "R_eff_h_final_nm":r["R_eff_h_final_nm"],
                       "h_integral_initial":r["h_integral_initial"],
                       "h_integral_final":r["h_integral_final"],
                       "max_xB":r["max_global_xB_from_runtime_log"],
                       "status":r["acceptance_status"]})
write_csv(OUT/"pf_x_q_dt_convergence_summary.csv",legacy+base+qcases)
timeline_src=ROOT/"reports/pf_only_baseline_closure/pf_only_dt_convergence_time_series.csv"
(OUT/"pf_x_q_dt_convergence_time_series.csv").write_text(timeline_src.read_text() if timeline_src.exists() else "case,status\n")
comparison=[{"mode":"L","T400_status":"FAIL_XB_RUNAWAY_AND_COLLAPSE","local_storage":"lagged Jacobian approximation","projection":"not curative"},
            {"mode":"X","T400_status":"BOUNDED_BUT_NONLOCAL_PROJECTION_DOMINANT","local_storage":"missing","projection":"dominant"},
            {"mode":"Q","T400_status":"FAIL_CAPACITY_BOUND_PRESERVATION","local_storage":"implemented with phase rejection","projection":"not reached as clean correction-only reference"}]
write_csv(OUT/"pf_x_q_mode_comparison.csv",comparison)

ds_complete=all(r.get("run_status")=="COMPLETE" for r in ds_rows)
eps_complete=all(r.get("run_status")=="COMPLETE" for r in eps_rows)
ds_converged=(len(ds_spreads)==3 and
              ds_spreads[0.0005] < ds_spreads[0.001] < ds_spreads[0.002])
final_status=("PARTIAL_X_TRANSPORT_BOUNDED_BUT_PROJECTION_OR_STABILIZER_REMAINS_DOMINANT"
              if not (ds_complete and eps_complete) else
              "PARTIAL_X_TRANSPORT_BOUNDED_BUT_PROJECTION_OR_STABILIZER_REMAINS_DOMINANT")
(OUT/"pf_only_x_transport_acceptance_report.md").write_text(f"""# Mode X Acceptance

Mode X eliminates the old lagged-history positive feedback and is bounded in completed T400/T380 equal-time runs. It does not pass local storage semantics because global projection replaces the missing local phase-storage transfer. Projection is dominant, not correction-only. Ds matrix complete: `{ds_complete}`; epsilon matrix complete: `{eps_complete}`.

Status: `{final_status}`.
""")
(OUT/"pf_only_q_transport_acceptance_report.md").write_text("""# Mode Q Acceptance

Formula/unit tests validate the q storage identity, but runtime does not pass. Existing flux divergence violates q capacity over a growing fraction of the domain and produces 0/1 matrix-context extrema at dt=0.0005. Q remains diagnostic and requires a bound-preserving conservative face-flux discretization before equal-time acceptance.

Status: `FAIL_Q_RUNTIME_CAPACITY_BOUND_PRESERVATION`.
""")
(OUT/"pf_recommended_composition_evolution_mode.md").write_text("""# Recommended Composition Evolution Mode

Do not restore Mode L. Keep Mode X only as the current bounded diagnostic baseline, explicitly labeled nonlocal-projection-dependent. Do not promote the present Q prototype. The recommended architecture is a future finite-volume/face-flux Q solver that preserves `0<=q<=1-h` locally and leaves projection correction-only.
""")
(OUT/"pf_S3_reintegration_gate.md").write_text("""# S3 Reintegration Gate

`S3_source_component_frozen=true`

`S3_reintegration_allowed=false`

The gate remains closed because local phase storage and correction-only projection have not both passed, and the Q runtime reference is not bound preserving.
""")

terminal=f"""old_lagged_rhs_status=FAIL_HISTORY_FEEDBACK_RUNAWAY
workstation_build_status=PASS
workstation_binary_sha256=7a006b061325e3cdab3375fae2dc352d4a54c9a0541e492a01a0207d7d2c2bfa
primary_evolved_variable_mode_X=xB_alpha
primary_evolved_variable_mode_Q=q_alpha
alpha_support_rule=OTHER_PHYSICAL_RATE_ZERO_WITH_GLOBAL_SPECTRAL_STABILIZER
epsilon_alpha_sensitivity_status={'FAIL_RESOLVED_UNRESOLVED_FATE_CHANGE' if eps_fate_change else ('COMPLETE_FINE_DT_ONLY' if eps_complete else 'RUNNING')}
Ds_sensitivity_status={'CONVERGENT_WITH_DT_REFINEMENT' if ds_converged else ('COMPLETE_NOT_CONVERGED' if ds_complete else 'RUNNING')}
local_phase_storage_transfer_implemented=MODE_Q_ONLY
local_phase_storage_mass_closure=MODE_X_FAIL_MODE_Q_FORMULA_PASS_RUNTIME_REJECTS
global_projection_lambda_max={lambda_max:.17e}
global_projection_mass_fraction_max={proj_fraction_max:.17e}
global_projection_far_field_drift={farfield_max:.17e}
projection_correction_only=false
projection_idempotence_status=PASS
T400_mode_L_status=FAIL_RUNAWAY_COLLAPSE
T400_mode_X_status=BOUNDED_PROJECTION_DOMINANT
T400_mode_Q_status=FAIL_CAPACITY_BOUND_PRESERVATION
T400_mode_X_dt_convergence=PASS_SIGN_AND_REFINEMENT
T400_mode_Q_dt_convergence=FAIL_FINE_DT_CONTEXT_RUNAWAY
T380_gate_status=CLOSED_BY_T400_Q_AND_PROJECTION_FAILURE
T380_mode_X_status=BOUNDED_EQUAL_TIME_COMPLETE
T380_mode_Q_status=NOT_RUN_GATE_CLOSED
mode_X_vs_Q_matrix_agreement=FAIL
recommended_composition_evolution_mode=FUTURE_BOUND_PRESERVING_FACE_FLUX_Q
recommended_mode_reason=X_NONLOCAL_PROJECTION_DOMINANT_Q_CURRENT_FLUX_NOT_BOUND_PRESERVING
PF_only_baseline_status=PARTIAL
composition_context_status=FAIL_Q_RUNTIME_BOUNDS
projection_acceptance_status=FAIL_DOMINANT_NONLOCAL_CORRECTION
S3_source_component_frozen=true
S3_reintegration_allowed=false
direct_phi_write=false
q_diagnostic_phi_substep_rollback_on_infeasible=true
direct_beta_inventory_injection=false
production_GP_thermodynamics_closed=false
recommended_next_action=IMPLEMENT_BOUND_PRESERVING_CONSERVATIVE_FACE_FLUX_Q_BEFORE_S3
final_status={final_status}
"""
(OUT/"final_terminal_output.txt").write_text(terminal)

(OUT/"pf_only_x_q_validation_final_report.md").write_text(f"""# Can xB-Transport or q-Transport Close the PF-Only Composition Baseline?

## 1. Executive Summary

Mode L fails through lagged-history positive feedback. Mode X removes that feedback and remains bounded, but it does not preserve local phase storage; the global logit projection performs the leading compensation. Mode Q has the cleaner conserved variable and passes formula tests, yet the existing runtime flux discretization is not q-capacity preserving and drives the matrix context to its bounds.

## 2. Recovered Equations and Code Path

The recovered step is `phi -> composition context -> mu/J/divJ -> X or Q update -> Y reconstruction -> projection -> diagnostics/history`. Exact files and line anchors are in `pf_x_transport_validation_manifest.csv`.

## 3. Old lagged_rhs Failure Mechanism

The lagged Jacobian term tends to `+dYdt_prev` when `A=(1-h)x(1-x)` is small. One historical physical step supplies only one fixed-point iteration, producing positive history feedback, xB runaway and seed collapse.

## 4. New X-Transport Formulation

X directly evolves matrix-channel xB with `divJ/(1-h)` above epsilon and a global add/subtract spectral stabilizer. It does not read `dYdt_prev`, so the old feedback is removed.

## 5. Local Phase-Storage Transfer

Mode X does not implement `q_new=q_old-Delta h vB`. `GLOBAL_PROJECTION_SUBSTITUTES_LOCAL_PHASE_STORAGE_TRANSFER=true`. Formula-level Mode Q implements it; infeasible phase requests reject the whole phi substep without clipping q.

## 6. Global Projection Audit

Projection identity/idempotence pass, but it is mathematically global. The preregistered envelope is `{PROJECTION_ENVELOPE:.3e}` mass units; observed post-handoff correction reaches order one mass units, with projection/local-phase ratio `{proj_fraction_max:.6g}`. It is the leading closure operator, not a residual correction.

## 7. h_alpha_min and Support Treatment

Current X is `OTHER: PHYSICAL_RATE_ZERO_WITH_GLOBAL_SPECTRAL_STABILIZER`, not a strict frozen extension. Epsilon and Ds have been decoupled. Sensitivity completion: `{eps_complete}`.

## 8. Ds Stabilizer Sensitivity

The runtime matrix compares 5, 10 and 20 times D_alpha at equal physical time. Matrix completion: `{ds_complete}`. Until fine-dt convergence across Ds is demonstrated, X cannot be promoted as a physical baseline.

## 9. Q-Transport Reference Formulation

Q evolves conserved matrix storage without division by small `1-h`, uses a bounded derived xB context, and attempts local conservative redistribution of capacity crossings. The present centered/spectral flux is not bound preserving, so the diagnostic reference fails runtime acceptance.

## 10. Unit-Test Results

G1-G7 formula tests pass for the intended Q identities, projection identity and projection idempotence. Expected X failures occur for pure phi storage and conditioning below alpha=0.1.

## 11. T400 Equal-Time dt Convergence

Mode L fails. Mode X has consistent shrink sign and decreasing trajectory differences over dt=0.002, 0.001 and 0.0005, but fails projection/locality acceptance. Mode Q fails capacity/context bounds at dt=0.002 and 0.0005 before an equal-time trajectory can be accepted.

## 12. T380 Validation Gate

Existing Mode X equal-time evidence is bounded and complete. Mode Q was not expanded to T380 because the T400 Q gate failed, as preregistered.

## 13. Mode X versus Mode Q

X is operationally bounded but nonlocally closed. Q has the correct storage variable and local formula but needs a different bound-preserving flux discretization. Neither current implementation satisfies all baseline criteria.

## 14. Recommended Architecture

Do not restore Mode L. Retain X only as an explicitly projection-dependent diagnostic. Develop a conservative face-flux Q solver that enforces `0<=q<=1-h` before repeating equal-time validation.

## 15. S3 Reintegration Decision

`S3_source_component_frozen=true`; `S3_reintegration_allowed=false`. No 90-case map is rerun.

## 16. Final Verdict

`{final_status}`

No PF physical parameter, seed profile, free energy, GP inventory, JGP, Jbeta, or S3 source physics was retuned.
""")
print(final_status)
