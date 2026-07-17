#!/usr/bin/env python3
"""Apply the registered long-window gates to variable-step BDF2."""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np


DT_RE = re.compile(
    r"CTOT_VARIABLE_BDF2_CONTROLLER_COMMIT step=(\d+).*accepted_dt=([^ ]+)")
RETRY_RE = re.compile(r"CTOT_COUPLED_STEP_RETRY physical_step_id=(\d+)")
SUMMARY_RE = re.compile(
    r"macro_steps=(\d+).*internal_trial_rejects=(\d+).*retry_fraction=([^ ]+).*"
    r"fallback_macros=(\d+).*fallback_fraction=([^ ]+).*"
    r"reject_trial_wall_fraction=([^ ]+)")


def h(phi): return phi**3 * (6*phi**2 - 15*phi + 10)


def only(root: Path, pattern: str) -> Path:
    matches = list(root.rglob(pattern))
    if len(matches) != 1: raise RuntimeError(f"{root}: {pattern}: {matches}")
    return matches[0]


def fields(root: Path) -> dict:
    def raw(stem): return np.fromfile(only(root, f"ctot_checkpoint*_{stem}.raw"), dtype=np.float64)
    meta = json.loads(only(root, "ctot_checkpoint*_meta.json").read_text().replace(": nan", ": null"))
    return {"C": raw("Ctot"), "phi": raw("phi"), "x": raw("xB_alpha"), "meta": meta}


def crossings(phi, level=0.5):
    result=[]
    for i in range(phi.size):
        j=(i+1)%phi.size; a=phi[i]-level; b=phi[j]-level
        if a*b<0: result.append((i-a/(b-a))%phi.size)
    return sorted(result)


def crossing_error(a,b,n=512):
    if len(a)!=len(b): return math.inf
    return max((min(abs(x-y),n-abs(x-y)) for x,y in zip(a,b)), default=0.0)


def main() -> int:
    p=argparse.ArgumentParser(); p.add_argument("--run-root",type=Path,required=True)
    p.add_argument("--initial-root",type=Path,required=True); p.add_argument("--report-root",type=Path,required=True)
    a=p.parse_args(); a.report_root.mkdir(parents=True,exist_ok=True)
    case={"variable":fields(a.run_root/"variable_long_8000"),
          "fixed":fields(a.run_root/"fixed_G9_dt4_long"),
          "strict":fields(a.run_root/"strict_G12_dt32_long")}
    init={"C":np.fromfile(a.initial_root/"ctot_checkpoint_step005482_Ctot.raw",dtype=np.float64),
          "phi":np.fromfile(a.initial_root/"ctot_checkpoint_step005482_phi.raw",dtype=np.float64)}
    ref=case["strict"]; hi=h(init["phi"]); hr=h(ref["phi"]); qr=ref["C"]-hr
    transfer_ref=float(np.sum(hr)-np.sum(hi)); far_mask=hr<1e-3
    rows=[]
    for name,c in case.items():
        hc=h(c["phi"]); q=c["C"]-hc; transfer=float(np.sum(hc)-np.sum(hi))
        rows.append({"case":name,
          "time_code":c["meta"]["bdf2_time_code"],
          "cumulative_transfer_error":abs(transfer-transfer_ref)/max(abs(transfer_ref),1e-300),
          "beta_amount_error":abs(np.sum(hc)-np.sum(hr))/max(abs(np.sum(hr)),1e-300),
          "interface_position_error_dx":crossing_error(crossings(c["phi"]),crossings(ref["phi"])),
          "capacity_weighted_matrix_profile_error":np.linalg.norm(q-qr)/max(np.linalg.norm(qr),1e-300),
          "far_field_error":abs(np.mean(c["x"][far_mask])-np.mean(ref["x"][far_mask]))/max(abs(np.mean(ref["x"][far_mask])),1e-300),
          "growth_direction_same":math.copysign(1,transfer)==math.copysign(1,transfer_ref),
          "h_volume":float(np.sum(hc)),"transfer":transfer})
    with (a.report_root/"variable_step_long_accuracy_metrics.csv").open("w",newline="") as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    v=rows[0]; log=(a.run_root/"variable_long_8000/run.log").read_text(errors="replace")
    sm=SUMMARY_RE.findall(log)[-1]
    dt_pairs=[(int(step),float(value)) for step,value in DT_RE.findall(log)]
    dts=np.array([value for _,value in dt_pairs])
    retry_steps={int(step) for step in RETRY_RE.findall(log)}
    dt_total=float(np.sum(dts))
    with (a.report_root/"variable_step_dt_distribution.csv").open("w",newline="") as f:
        fieldnames=["step","accepted_dt_code","accepted_dt_physical_s",
                    "physical_time_fraction","retry_associated"]
        w=csv.DictWriter(f,fieldnames=fieldnames); w.writeheader()
        for step,value in dt_pairs:
            w.writerow({"step":step,"accepted_dt_code":value,
                        "accepted_dt_physical_s":value*41.1295854245547687,
                        "physical_time_fraction":value/dt_total,
                        "retry_associated":int(step in retry_steps)})
    gates={"transfer":v["cumulative_transfer_error"]<=.03,"beta":v["beta_amount_error"]<=.03,
      "interface":v["interface_position_error_dx"]<=.5,"profile":v["capacity_weighted_matrix_profile_error"]<=.05,
      "far":v["far_field_error"]<=.02,"direction":v["growth_direction_same"],
      "retry":float(sm[2])<=.01,"fallback":float(sm[4])<=.01,"overhead":float(sm[5])<=.05}
    qoi_keys=("transfer","beta","interface","profile","far","direction")
    runtime_keys=("retry","fallback","overhead")
    qoi_pass=all(gates[key] for key in qoi_keys)
    runtime_pass=all(gates[key] for key in runtime_keys)
    if qoi_pass and runtime_pass:
        status="PASS_VARIABLE_STEP_LONG_WINDOW_ACCURACY"
    elif qoi_pass:
        status="FAIL_VARIABLE_STEP_LONG_WINDOW_PRODUCTION_RETRY_GATE"
    else:
        status="FAIL_VARIABLE_STEP_LONG_WINDOW_ACCURACY"
    bands={}
    for value in dts:
        key=float(value)
        bands[key]=bands.get(key,0)+1
    ranked_bands=sorted(bands.items(), key=lambda item:(-item[1],item[0]))
    top_band_text=", ".join(
        f"{value:.9e}: {count} steps ({count/len(dts):.3%} of steps, "
        f"{value*count/dt_total:.3%} of accepted code time)"
        for value,count in ranked_bands[:5])
    if len(ranked_bands)>5:
        other_count=sum(count for _,count in ranked_bands[5:])
        other_time=sum(value*count for value,count in ranked_bands[5:])
        top_band_text += (f", other {len(ranked_bands)-5} exact values: "
                          f"{other_count} steps ({other_count/len(dts):.3%}, "
                          f"{other_time/dt_total:.3%} of accepted code time)")
    changes=np.diff(dts)
    nonzero=np.sign(changes[np.abs(changes)>0.0])
    reversals=int(np.sum(nonzero[1:]*nonzero[:-1]<0)) if nonzero.size>1 else 0
    reductions=int(np.sum(changes<0.0))
    increases=int(np.sum(changes>0.0))
    report=f"""# Variable-step long-window accuracy

The variable trajectory ran 8000 accepted macros and was compared at matched
physical time with G9/dt4 and G12/dt32 trajectories from the same accepted history.

| Metric | Error | Gate | Pass |
|---|---:|---:|---|
| cumulative transfer | {v['cumulative_transfer_error']:.6e} | 3% | {gates['transfer']} |
| beta amount | {v['beta_amount_error']:.6e} | 3% | {gates['beta']} |
| interface trajectory | {v['interface_position_error_dx']:.6e} dx | 0.5 dx | {gates['interface']} |
| capacity-weighted matrix profile | {v['capacity_weighted_matrix_profile_error']:.6e} | 5% | {gates['profile']} |
| far field | {v['far_field_error']:.6e} | 2% | {gates['far']} |

Retry/fallback/overhead are {float(sm[2]):.6%}, {float(sm[4]):.6%}, and
{float(sm[5]):.6%}. Accepted dt mean/p50/p90/p99 are {np.mean(dts):.9e},
{np.quantile(dts,.5):.9e}, {np.quantile(dts,.9):.9e}, and {np.quantile(dts,.99):.9e}.
The dominant accepted dt bands are: {top_band_text}.
There are `{increases}` accepted-dt increases, `{reductions}` reductions, and
`{reversals}` direction reversals. `{len(retry_steps)}` accepted macro indices
are associated with at least one rejected trial.

Status: `{status}`.
"""
    (a.report_root/"variable_step_long_accuracy.md").write_text(report)
    print(status); return 0 if all(gates.values()) else 2


if __name__=="__main__": raise SystemExit(main())
