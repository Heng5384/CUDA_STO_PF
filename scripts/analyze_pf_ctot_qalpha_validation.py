#!/usr/bin/env python3
"""Assemble equal-time conservative PF validation summaries."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
REPORT = ROOT / "reports/pf_ctot_qalpha_validation"
R_GAS = 8.31446261815324


def read_vtk(path: Path):
    with path.open() as f:
        lines = [next(f) for _ in range(10)]
        dims = tuple(map(int, lines[4].split()[1:4]))
        data = np.fromstring(f.read(), sep=" ", dtype=float)
    return data.reshape(dims)


def h(phi):
    p = np.clip(phi, 0.0, 1.0)
    return p**3*(6*p**2-15*p+10)


def gpure(phi):
    p = np.clip(phi, 0.0, 1.0)
    return p*p*(1-p)*(1-p)


def g_ref_pb(T):
    if T < 600.61:
        pb = -7650.085 + 101.700244*T - 24.5242231*T*math.log(T) - 0.00365895*T*T - 2.4395e-7*T**3
    else:
        pb = -10531.095 + 154.243182*T - 32.4913959*T*math.log(T) + 0.00154613*T*T + 8.05448e25*T**-9
    if T < 722.66:
        te = -10544.679 + 183.372894*T - 35.6687*T*math.log(T) + 0.01583435*T*T - 5.240417e-6*T**3 + 155015/T
    else:
        te = 9160.595 - 129.265373*T + 13.004*T*math.log(T) - 0.0362361*T*T + 5.006367e-6*T**3 - 1.28681e30*T**-9
    return -76063.2138 + 9.67716633*T + pb + te


def g_ref_ag2te(T):
    if T < 1234.93:
        ag = -7209.512 + 118.202013*T - 23.8463314*T*math.log(T) - 0.001790585*T*T - 3.98587e-7*T**3 - 12011/T
    else:
        ag = -15095.252 + 190.266404*T - 33.472*T*math.log(T) + 1.411773e29*T**-9
    if T < 722.66:
        te = -10544.679 + 183.372894*T - 35.6687*T*math.log(T) + 0.01583435*T*T - 5.240417e-6*T**3 + 155015/T
    else:
        te = 9160.595 - 129.265373*T + 13.004*T*math.log(T) - 0.0362361*T*T + 5.006367e-6*T**3 - 1.28681e30*T**-9
    return 3*(-10128.93 - 12.645115*T + (2/3)*ag + (1/3)*te)


def xeq(T):
    L = 41504.29119633958 - 18.469276826409214*T
    x = max(math.exp(-L/(R_GAS*T)), 1e-9)
    for _ in range(20):
        f = R_GAS*T*math.log(x)+L*(1-x)**2
        df = R_GAS*T/x-2*L*(1-x)
        x = min(max(x-f/df, 1e-10), 0.999)
    return x


def energy_proxy(phi, xb, T, scale, W=1.0, kappa=0.045):
    x = np.clip(xb, 1e-12, 1-1e-12)
    L = 41504.29119633958-18.469276826409214*T
    ga = ((1-x)*g_ref_pb(T)+x*g_ref_ag2te(T)+
          R_GAS*T*((1-x)*np.log(1-x)+x*np.log(x))+L*x*(1-x))/scale
    xe = xeq(T)
    mu0 = (g_ref_ag2te(T)+R_GAS*T*math.log(xe)+L*(1-xe)**2)/scale
    hh = h(phi)
    chemical = float(np.sum((1-hh)*ga+hh*mu0))
    grad = 0.0
    for axis in range(3):
        d = 0.5*(np.roll(phi, -1, axis=axis)-np.roll(phi, 1, axis=axis))
        grad += float(np.sum(d*d))
    return chemical + 0.5*kappa*grad + W*float(np.sum(gpure(phi)))


def find_case(T, mode, dtag, n):
    pattern = f"Results/chel_T{T}_cuda_128x128x128_dt*_steps{n}_xB0.008/T{T}_{mode}_F2_dt{dtag}_equalT"
    found = list(ROOT.glob(pattern))
    return found[0] if found else None


def summarize_temperature(T, specs):
    summary, series = [], []
    for mode, dtag, dt, n in specs:
        case = find_case(T, mode, dtag, n)
        if not case:
            continue
        diag = list(csv.DictReader((case/"pf_conservative_step_diagnostics.csv").open()))
        vf_file = next(case.glob("vf_precip_vs_time_*.csv"))
        vf = list(csv.DictReader(vf_file.open()))
        relax = list(csv.DictReader((case/"relaxation_diagnostics.csv").open()))
        phi0, xb0 = read_vtk(case/"phi_init.vtk"), read_vtk(case/"xB_0.vtk")
        phif, xbf = read_vtk(case/f"phi_{n}.vtk"), read_vtk(case/f"xB_{n}.vtk")
        hh0, hhf = h(phi0), h(phif)
        mass_res = max(abs(float(r["mass_after_phase"])-float(r["mass_before_transport"])) for r in diag)
        mass_scale = max(1.0, abs(float(diag[0]["mass_before_transport"])))
        e0 = energy_proxy(phi0, xb0, T+273.15, 1.3779024e5)
        ef = energy_proxy(phif, xbf, T+273.15, 1.3779024e5)
        row = {
            "case": case.name, "T_C": T, "mode": mode, "strategy": "pairwise_limited",
            "dt": dt, "nsteps": n, "physical_time_code": dt*n,
            "R_eff_h_initial_nm": (3*hh0.sum()/(4*math.pi))**(1/3),
            "R_eff_h_final_nm": (3*hhf.sum()/(4*math.pi))**(1/3),
            "h_integral_initial": hh0.sum(), "h_integral_final": hhf.sum(),
            "support_phi_gt_0p5_final": np.count_nonzero(phif>0.5),
            "min_C_reconstructed": np.min((1-hhf)*xbf+hhf),
            "max_C_reconstructed": np.max((1-hhf)*xbf+hhf),
            "min_q_reconstructed": np.min((1-hhf)*xbf),
            "max_q_reconstructed": np.max((1-hhf)*xbf),
            "max_authoritative_mass_error_rel": mass_res/mass_scale,
            "bound_violation_count": max(float(r["bound_violation_count"]) for r in diag),
            "limited_face_count_max": max(float(r["limited_face_count"]) for r in diag),
            "phase_constraint_count_max": max(float(r["phase_constraint_count"]) for r in diag),
            "energy_proxy_initial": e0, "energy_proxy_final": ef,
            "energy_proxy_delta": ef-e0,
            "energy_scope": "chemical+barrier+gradient; elastic excluded",
            "projection_used": False,
            "status": "PASS" if mass_res/mass_scale <= 1e-10 and max(float(r["bound_violation_count"]) for r in diag)==0 else "FAIL",
        }
        summary.append(row)
        for r in vf:
            series.append({"case": case.name, "T_C": T, "mode": mode, "dt": dt,
                           "step": r["step"], "t_code": r["t_code"],
                           "t_real_s": r["t_real_s"], "vf_precip": r["vf_precip"],
                           "R_avg_internal": r["R_avg"]})
    return summary, series


def write_csv(path, rows):
    if not rows:
        path.write_text(""); return
    with path.open("w", newline="") as f:
        w=csv.DictWriter(f, fieldnames=rows[0].keys()); w.writeheader(); w.writerows(rows)


def main():
    specs=[]
    for mode in ("ctot_conservative_split", "qalpha_conservative_local_transaction"):
        specs += [(mode,"0p002",0.002,250),(mode,"0p001",0.001,500),(mode,"0p0005",0.0005,1000)]
    s400,t400=summarize_temperature(400,specs)
    s380,t380=summarize_temperature(380,specs)
    write_csv(REPORT/"pf_T400_ctot_qalpha_dt_summary.csv",s400)
    write_csv(REPORT/"pf_T400_ctot_qalpha_time_series.csv",t400)
    write_csv(REPORT/"pf_T380_ctot_qalpha_dt_summary.csv",s380)
    write_csv(REPORT/"pf_T380_ctot_qalpha_time_series.csv",t380)
    print(json.dumps({"T400_cases":len(s400),"T380_cases":len(s380)}))


if __name__ == "__main__": main()
